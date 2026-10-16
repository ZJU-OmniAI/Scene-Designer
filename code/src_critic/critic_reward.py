"""Scene-critic GRPO reward CLI — fixed to the v10 LoRA.

Give it K rollout JSON files (one per candidate scene in a GRPO group);
it loads the v10 critic, ranks the scenes BEST→WORST, and returns a GRPO
advantage per rollout (zero-mean, ~unit-variance within the group).

Each rollout JSON must point at that rollout's rendered top-down PNG. Any
ONE of these keys is accepted (checked in order):
    "image", "image_path", "render", "topdown"
Optionally an "id" key labels the rollout in the output (else the JSON
filename stem is used).

PHYSICAL-LEGALITY GATE (on by default): if a rollout JSON also has a
    "scene" / "scene_dir"  key  →  path to that rollout's raw-style scene
directory (floor_plan.json + furniture_*.json), the geometry is checked for
physical violations (out-of-room / overlap / blocked door). Any violation ⇒
that rollout is physically ILLEGAL and gets the MAXIMUM negative reward;
legal rollouts go through the normal critic ranking. Disable with
--no-legality. Rollouts without a scene path are treated as "unchecked"
(legal).

Examples:
    # K=4 rollouts, default Borda reward, auto-pick a free GPU
    python -m src_critic.critic_reward r0.json r1.json r2.json r3.json

    # explicit image paths instead of JSON files
    python -m src_critic.critic_reward --images a.png b.png c.png

    # stepwise (Borda + PL confidence), specific GPU
    python -m src_critic.critic_reward *.json --method stepwise --gpu 6

Output (stdout, JSON):
    {
      "ranking":   ["r1","r3","r0","r2"],          # best→worst, by id
      "advantage": {"r0": -0.45, "r1": 1.34, ...}, # GRPO reward per rollout
      "raw_score": {"r0": 0.33, ...},              # pre-normalization Borda
      "method": "borda", "K": 4,
      "reasoning": "..."                            # critic CoT (if --show-reasoning)
    }
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO_ROOT / "model" / "Qwen3-VL-8B-Instruct"
DEFAULT_ADAPTER = REPO_ROOT / "datasets/critic_pairs/runs/scene_critic_lora_v10/final"
_IMG_KEYS = ("image", "image_path", "render", "topdown")
_SCENE_KEYS = ("scene", "scene_dir", "scene_path")


def _resolve(p_dir: Path, val: str) -> str:
    v = Path(val)
    if not v.is_absolute() and not v.exists():
        v = p_dir / val
    return str(v)


def _from_json(p: Path) -> tuple[str, str, str | None]:
    """Return (rollout_id, image_path, scene_path|None) from a rollout JSON."""
    obj = json.loads(p.read_text())
    img = next((obj[k] for k in _IMG_KEYS if obj.get(k)), None)
    if img is None:
        raise SystemExit(f"{p}: no image key found (one of {_IMG_KEYS})")
    rid = str(obj.get("id") or p.stem)
    scene = next((obj[k] for k in _SCENE_KEYS if obj.get(k)), None)
    scene_path = _resolve(p.parent, scene) if scene else None
    return rid, _resolve(p.parent, img), scene_path


def _pick_free_gpu() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True)
        best, best_free = "0", -1
        for line in out.strip().splitlines():
            idx, used, total = (x.strip() for x in line.split(","))
            free = int(total) - int(used)
            if free > best_free:
                best_free, best = free, idx
        return best
    except Exception:
        return "0"


def main() -> int:
    ap = argparse.ArgumentParser(description="Scene-critic GRPO reward (v10).")
    ap.add_argument("rollouts", nargs="*", type=Path,
                    help="K rollout JSON files (each pointing at a render).")
    ap.add_argument("--images", nargs="*", type=Path, default=None,
                    help="Alternatively, K image paths directly (skip JSON).")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER,
                    help="LoRA adapter dir (default: fixed to v10).")
    ap.add_argument("--method", choices=["borda", "stepwise"], default="borda")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gpu", type=str, default=None,
                    help="CUDA device index; default auto-picks the freest GPU.")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--show-reasoning", action="store_true")
    ap.add_argument("--no-legality", action="store_true",
                    help="Disable the geometric physical-violation gate.")
    ap.add_argument("--illegal-floor", type=float, default=-1.0,
                    help="Raw score for physically-illegal rollouts (below "
                         "Borda's [0,1] range → most-negative advantage).")
    args = ap.parse_args()

    # Resolve the K inputs → ordered (id, image_path, scene_path)
    if args.images:
        items = [(p.stem, str(p), None) for p in args.images]
    elif args.rollouts:
        items = [_from_json(p) for p in args.rollouts]
    else:
        ap.error("provide K rollout JSON files or --images")
    ids = [i for i, _, _ in items]
    image_paths = [p for _, p, _ in items]
    scene_paths = [s for _, _, s in items]
    if len(set(ids)) != len(ids):
        ids = [f"{i}#{n}" for n, i in enumerate(ids)]  # disambiguate dup ids
    for ip in image_paths:
        if not Path(ip).exists():
            raise SystemExit(f"image not found: {ip}")

    # --- physical-legality gate (geometry only, no model) ---
    legal_flags, violations = [True] * len(ids), [[] for _ in ids]
    if not args.no_legality and any(s for s in scene_paths):
        from .grpo_advantage import legality_flags_from_scenes
        legal_flags, violations = legality_flags_from_scenes(scene_paths)
        for rid, lf, vs in zip(ids, legal_flags, violations):
            if not lf:
                print(f"[critic_reward] ILLEGAL {rid}: {vs}", file=sys.stderr)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu or _pick_free_gpu()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    n_illegal = sum(1 for f in legal_flags if not f)
    # Skip loading the 8B model entirely if every rollout is illegal.
    if n_illegal == len(ids):
        from .grpo_advantage import advantage_from_scores
        floor = {chr(ord("A") + i): args.illegal_floor for i in range(len(ids))}
        res = advantage_from_scores(floor)
    else:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from peft import PeftModel
        from .grpo_advantage import (
            advantage_from_critic, advantage_from_critic_with_legality)

        print(f"[critic_reward] base={args.base} adapter={args.adapter} "
              f"gpu={os.environ['CUDA_VISIBLE_DEVICES']} K={len(items)} "
              f"illegal={n_illegal}", file=sys.stderr)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.base), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(str(args.base), trust_remote_code=True)
        if args.adapter and Path(args.adapter).exists():
            model = PeftModel.from_pretrained(model, str(args.adapter))
        model.eval()

        common = dict(method=args.method, alpha=args.alpha,
                      max_new_tokens=args.max_new_tokens, return_raw=args.show_reasoning)
        if args.no_legality or n_illegal == 0:
            res = advantage_from_critic(model, processor, image_paths, **common)
        else:
            res = advantage_from_critic_with_legality(
                model, processor, image_paths, legal_flags,
                illegal_floor=args.illegal_floor, **common)

    # label (A,B,C,..) → rollout id
    lab2id = {chr(ord("A") + i): ids[i] for i in range(len(ids))}
    viol_by_id = {ids[i]: violations[i] for i in range(len(ids))}
    legal_by_id = {ids[i]: bool(legal_flags[i]) for i in range(len(ids))}
    out = {
        "ranking":   [lab2id[l] for l in res.ranking],
        "advantage": {lab2id[l]: round(v, 4) for l, v in res.advantage.items()},
        "raw_score": {lab2id[l]: round(v, 4) for l, v in res.raw_score.items()},
        "legal":     legal_by_id,
        "violations": {k: v for k, v in viol_by_id.items() if v},
        "method": res.method, "K": res.K,
    }
    if args.show_reasoning:
        out["reasoning"] = res.raw_text
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
