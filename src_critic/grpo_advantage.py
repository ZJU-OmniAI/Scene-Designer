"""Convert critic ranking output → per-rollout GRPO advantage.

The critic emits `<reasoning>...</reasoning><answer>BACD</answer>` over K=2-4
candidates. This module turns that ordinal output into a scalar advantage A_i
suitable for GRPO's `r·ratio·A` term: zero-mean, ~unit variance, bounded.

Two scoring methods are provided:

* `borda`     — pure ordinal rank, K-1 linear positions. Cheap, well-calibrated
                to the critic's training distribution. **Default**.
* `stepwise`  — Borda + α · stepwise restricted-softmax confidence (Plackett-
                Luce on single-letter token positions). Adds margin signal but
                requires one extra teacher-forced forward.
                NOTE: Qwen's BPE merges consecutive letter tokens (`>A`, `AC`,
                `BD`, …), so the model's natural mass at the post-`<answer>`
                position is on merged tokens, not single letters. Reading the
                single-letter restricted softmax is mildly out-of-distribution.
                We mitigate via `clip_at_uniform=True` (default): the
                confidence term is clipped at 0, so stepwise can only *boost*
                the Borda signal — it never demotes the natural top pick.

Both end with `(s_i - mean) / (std + ε)` GRPO normalization within the rollout
group.

Quick API:

    # Path A — already have the parsed ranking (e.g. from API critic):
    res = advantage_from_ranking(["B","A","C","D"])

    # Path B — call the critic end-to-end (Qwen3-VL + LoRA):
    res = advantage_from_critic(
        model, processor, image_paths, user_text,
        method="stepwise", alpha=0.5,
    )
    # res.advantage = {"A": -0.1, "B": +1.47, "C": -0.14, "D": -1.24}
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


_ANS_RE = re.compile(r"<answer>\s*([A-Da-d \t,>=]+)\s*</answer>", re.DOTALL)
_LABELS = "ABCD"


# ---------------------------------------------------------------------------
# Pure-Python core (no torch import unless you call advantage_from_critic)
# ---------------------------------------------------------------------------

def parse_ranking(text: str, K: int) -> list[str] | None:
    """Extract a strict best→worst letter chain from a critic response.

    Accepts `<answer>BACD</answer>`, `<answer>B > A > C > D</answer>`, etc.
    Returns None if no valid permutation of {A..A+K-1} is found.
    """
    m = _ANS_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    if re.search(r"[=]|\bequal\b|\btie\b", raw, flags=re.IGNORECASE):
        return None
    letters = [c.upper() for c in raw if c.upper() in _LABELS]
    expected = set(_LABELS[:K])
    if len(letters) != K or set(letters) != expected:
        return None
    return letters


def borda_score(ranking: list[str], K: int | None = None) -> dict[str, float]:
    """Normalized Borda count ∈ [0, 1].  rank 0 (best) → 1.0,  rank K-1 → 0.0."""
    K = K or len(ranking)
    denom = max(K - 1, 1)
    return {lab: (K - 1 - i) / denom for i, lab in enumerate(ranking)}


def normalize_advantage(scores: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    """GRPO-style: subtract mean, divide by std. Zero-input → zero-output."""
    if not scores:
        return {}
    vals = list(scores.values())
    K = len(vals)
    mean = sum(vals) / K
    var = sum((v - mean) ** 2 for v in vals) / K
    std = var ** 0.5 + eps
    return {k: (v - mean) / std for k, v in scores.items()}


@dataclass
class AdvantageResult:
    ranking: list[str]                 # best→worst, e.g. ['B','A','C','D']
    advantage: dict[str, float]        # label → A_i for GRPO
    raw_score: dict[str, float]        # label → pre-normalization score
    stepwise_logp: dict[str, float] | None  # label → log P(picked at its step | remaining)
    method: str                        # 'borda' | 'stepwise' | '*+parse_fail'
    raw_text: str                      # critic decoded text (kept iff return_raw)
    K: int

    def to_dict(self) -> dict:
        return asdict(self)


def advantage_from_ranking(
    ranking: list[str],
    K: int | None = None,
) -> AdvantageResult:
    """Cheap path: caller already has the ranking string. Returns Borda advantage."""
    K = K or len(ranking)
    raw = borda_score(ranking, K)
    adv = normalize_advantage(raw)
    return AdvantageResult(
        ranking=list(ranking), advantage=adv, raw_score=raw,
        stepwise_logp=None, method="borda", raw_text="", K=K,
    )


def advantage_from_scores(
    scores: dict[str, float],
    ranking: list[str] | None = None,
) -> AdvantageResult:
    """When the caller already has per-option scalar scores (e.g., severity,
    PL utilities, top-softmax probabilities), just standardize them."""
    adv = normalize_advantage(scores)
    if ranking is None:
        ranking = sorted(scores, key=scores.get, reverse=True)
    return AdvantageResult(
        ranking=list(ranking), advantage=adv, raw_score=dict(scores),
        stepwise_logp=None, method="external", raw_text="", K=len(scores),
    )


# ---------------------------------------------------------------------------
# VL-coupled path (only imports torch/transformers when called)
# ---------------------------------------------------------------------------

DEFAULT_USER_TEXT = (
    "You see {n} top-down photoreal renderings of the same room labelled "
    "SCENE A, SCENE B{rest}. Geometry, doors and windows are identical; only "
    "furniture differs.\n\nRank the scenes from BEST to WORST by overall "
    "arrangement quality. First reason briefly inside `<reasoning>...</reasoning>`, "
    "then output a single permutation of the labels inside "
    "`<answer>...</answer>` (no separators, no ties)."
)


def build_default_user_text(K: int) -> str:
    labels = _LABELS[:K]
    rest = (", SCENE " + ", SCENE ".join(labels[2:])) if K > 2 else ""
    return DEFAULT_USER_TEXT.format(n=K, rest=rest)


def _build_chat_inputs(processor, image_paths: list[str], user_text: str):
    """Mirror eval_qwen3vl.py prompt structure."""
    from PIL import Image
    content = [{"type": "image", "image": str(p)} for p in image_paths]
    content.append({"type": "text", "text": user_text})
    msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    images = [Image.open(p).convert("RGB") for p in image_paths]
    return text, images


def _letter_token_ids(tokenizer, labels: list[str]) -> dict[str, int]:
    out = {}
    for c in labels:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Letter '{c}' is not a single token in this tokenizer "
                f"(got {ids}); stepwise scoring needs single-letter tokens.")
        out[c] = ids[0]
    return out


def _stepwise_logprob(
    model, processor, inputs, gen_ids, ranking: list[str], labels: list[str],
) -> dict[str, float]:
    """Teacher-forced forward to read restricted log-softmax at each answer step.

    Construction: take the critic's own (reasoning + `<answer>`) prefix verbatim,
    then append `ranking[0], ranking[1], …` as SINGLE-LETTER tokens. Read the
    logits at the position predicting each letter, restrict to remaining letter
    tokens, log-softmax. This is the Plackett-Luce conditional at step k.
    """
    import torch
    tok = processor.tokenizer
    letter_ids = _letter_token_ids(tok, labels)

    # Decode the critic's generation to find the `<answer>` boundary in text.
    gen_text = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
    open_idx = gen_text.find("<answer>")
    if open_idx < 0:
        return {lab: 0.0 for lab in labels}
    prefix_text = gen_text[: open_idx + len("<answer>")]

    prompt_ids = inputs["input_ids"][0]
    device = prompt_ids.device
    dtype = prompt_ids.dtype

    # Re-tokenize the (reasoning + <answer>) prefix WITHOUT special tokens.
    prefix_ids = torch.tensor(
        tok.encode(prefix_text, add_special_tokens=False),
        device=device, dtype=dtype,
    )
    rank_ids = torch.tensor(
        [letter_ids[c] for c in ranking],
        device=device, dtype=dtype,
    )
    full_ids = torch.cat([prompt_ids, prefix_ids, rank_ids]).unsqueeze(0)

    # Build a fresh attention_mask matching the new length; reuse pixel inputs.
    attn = torch.ones_like(full_ids, dtype=inputs["attention_mask"].dtype)
    fwd = {"input_ids": full_ids, "attention_mask": attn}
    for k in ("pixel_values", "image_grid_thw"):
        if k in inputs:
            fwd[k] = inputs[k]
    # M-RoPE needs mm_token_type_ids; appended tokens are all text (type=0).
    if "mm_token_type_ids" in inputs:
        orig_mm = inputs["mm_token_type_ids"][0]
        pad_len = full_ids.shape[1] - orig_mm.shape[0]
        pad = torch.zeros(pad_len, dtype=orig_mm.dtype, device=orig_mm.device)
        fwd["mm_token_type_ids"] = torch.cat([orig_mm, pad]).unsqueeze(0)

    with torch.no_grad():
        out = model(**fwd)
    logits = out.logits[0]                         # [L, vocab]

    base = prompt_ids.shape[0] + prefix_ids.shape[0]
    K = len(ranking)
    remaining_ids = [letter_ids[c] for c in labels]
    remaining_lab = list(labels)
    result: dict[str, float] = {}
    for k, picked in enumerate(ranking):
        pos = base + k - 1                         # logits @ pos predict token @ pos+1
        sub = logits[pos, remaining_ids].float()
        log_sm = torch.log_softmax(sub, dim=0)
        idx = remaining_lab.index(picked)
        result[picked] = float(log_sm[idx].item())
        remaining_ids.pop(idx)
        remaining_lab.pop(idx)
    # Degenerate last step (|R|=1 → log_softmax = 0) is correctly handled.
    return result


def advantage_from_critic(
    model,
    processor,
    image_paths: list[str | Path],
    user_text: str | None = None,
    *,
    labels: list[str] | None = None,
    method: str = "borda",
    alpha: float = 0.5,
    clip_at_uniform: bool = True,
    max_new_tokens: int = 2048,
    return_raw: bool = False,
) -> AdvantageResult:
    """End-to-end critic call → GRPO advantage for a single rollout group.

    Args:
        model:        a loaded Qwen3-VL(+LoRA) model in eval mode.
        processor:    matching AutoProcessor.
        image_paths:  K rollout result images.
        user_text:    custom prompt; if None, a default ranking prompt is used.
        method:       'borda' (default, fastest) or 'stepwise' (PL D.1).
        alpha:        weight on confidence term when method='stepwise'.

    Returns AdvantageResult; if the critic fails to emit a valid <answer>, the
    advantage dict is all zeros with method suffix '+parse_fail'.
    """
    import torch
    K = len(image_paths)
    labels = list(labels) if labels else list(_LABELS[:K])
    if user_text is None:
        user_text = build_default_user_text(K)

    text, images = _build_chat_inputs(processor, image_paths, user_text)
    inputs = processor(
        text=[text], images=images, return_tensors="pt", padding=True,
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1,
        )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    gen_text = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]

    ranking = parse_ranking(gen_text, K)
    if ranking is None:
        return AdvantageResult(
            ranking=[], advantage={lab: 0.0 for lab in labels},
            raw_score={lab: 0.0 for lab in labels},
            stepwise_logp=None,
            method=f"{method}+parse_fail",
            raw_text=gen_text, K=K,
        )

    raw = borda_score(ranking, K)
    stepwise_logp = None

    if method == "stepwise":
        stepwise_logp = _stepwise_logprob(
            model, processor, inputs, gen_ids, ranking, labels,
        )
        # confidence_k = z_k - log(1/|R_k|);  >= 0 iff model is above uniform.
        # Two safety guards prevent BPE-bias at step 0 from breaking Borda
        # monotonicity (rank-i adv must stay >= rank-(i+1) adv):
        #   1) clip_at_uniform: floor at 0   → can only *boost*, never demote
        #   2) monotone propagation         → earlier steps inherit at least
        #      the confidence of later steps, so rank-k boost >= rank-(k+1)
        confs = []
        for k, lab in enumerate(ranking):
            R = K - k
            c = stepwise_logp[lab] - math.log(1.0 / R) if R > 1 else 0.0
            if clip_at_uniform:
                c = max(c, 0.0)
            confs.append(c)
        # Right-to-left running max: ensures conf[k] >= conf[k+1] so the
        # boost a higher-ranked option gets is never below a lower-ranked one.
        for k in range(K - 2, -1, -1):
            confs[k] = max(confs[k], confs[k + 1])
        for k, lab in enumerate(ranking):
            raw[lab] = raw[lab] + alpha * confs[k]
    elif method != "borda":
        raise ValueError(f"unknown method: {method!r}")

    adv = normalize_advantage(raw)
    return AdvantageResult(
        ranking=ranking, advantage=adv, raw_score=raw,
        stepwise_logp=stepwise_logp, method=method,
        raw_text=gen_text if return_raw else "", K=K,
    )


# ---------------------------------------------------------------------------
# Physical-legality gate (geometry-based hard negative)
# ---------------------------------------------------------------------------

ILLEGAL_FLOOR = -1.0   # raw score for a physically-illegal rollout (below
                       # Borda's [0,1] range, so it standardizes to the most
                       # negative advantage in the group).


def legality_flags_from_scenes(scene_paths, **check_kw):
    """Geometry-only physical-legality check for K rollouts.

    `scene_paths[i]` is the path to rollout i's scene — a raw-style scene dir
    (containing floor_plan.json + furniture_*.json). Returns
    `(legal_flags, violations)` where legal_flags[i] is True iff rollout i has
    NO physical violation, and violations[i] is the list of
    (kind, object_id) problems (empty if legal). A None/unloadable path is
    treated as legal=True (can't check) with violation `[("unchecked","")]`.

    Pure geometry — no torch. Lazily imports perturb/scene_io.
    """
    from pathlib import Path as _P
    from .perturb import physical_violations
    from .scene_io import load_scene
    legal, viol = [], []
    for p in scene_paths:
        if p is None:
            legal.append(True); viol.append([("unchecked", "")]); continue
        try:
            scene = load_scene(_P(p))
            v = physical_violations(scene, **check_kw)
            legal.append(len(v) == 0); viol.append(v)
        except Exception as e:
            legal.append(True); viol.append([("unchecked", repr(e))])
    return legal, viol


def advantage_from_critic_with_legality(
    model,
    processor,
    image_paths: list[str | Path],
    legal_flags: list[bool],
    user_text: str | None = None,
    *,
    labels: list[str] | None = None,
    method: str = "borda",
    alpha: float = 0.5,
    illegal_floor: float = ILLEGAL_FLOOR,
    **gen_kw,
) -> AdvantageResult:
    """GRPO advantage with a HARD physical-legality gate in front of the critic.

    `legal_flags[i]` is True iff rollout i passed the geometric
    physical-violation check (see `perturb.physical_violations`).

    Rule:
      * physically ILLEGAL rollouts get a fixed floor raw-score
        (`illegal_floor`, below the Borda range) and are NEVER shown to the
        critic → after group standardization they receive the most-negative
        advantage in the group ("maximum negative reward").
      * the LEGAL rollouts go through the existing critic-ranking logic
        unchanged (Borda / stepwise on the legal subset).
      * all K are standardized together, so legality dominates: any legal
        rollout outranks any illegal one.

    Degenerate groups: if every rollout is legal this is identical to
    `advantage_from_critic`; if exactly one is legal it gets the top score
    without a critic call; if none are legal all advantages are 0 (no signal).
    """
    K = len(image_paths)
    labels = list(labels) if labels else list(_LABELS[:K])
    legal_idx = [i for i in range(K) if legal_flags[i]]

    # Fast path: everything legal → unchanged existing logic.
    if len(legal_idx) == K:
        return advantage_from_critic(
            model, processor, image_paths, user_text,
            labels=labels, method=method, alpha=alpha, **gen_kw)

    raw = {labels[i]: float(illegal_floor) for i in range(K)}
    ranking: list[str] = []
    method_used = f"{method}+legality"
    sub = None

    if len(legal_idx) == 1:
        # single legal rollout: it's the unique best, no critic needed.
        raw[labels[legal_idx[0]]] = 1.0
        ranking = [labels[legal_idx[0]]]
    elif len(legal_idx) >= 2:
        sub_paths = [image_paths[i] for i in legal_idx]
        sub_labels = [labels[i] for i in legal_idx]
        sub = advantage_from_critic(
            model, processor, sub_paths, user_text,
            labels=list(_LABELS[:len(sub_paths)]), method=method, alpha=alpha, **gen_kw)
        positional_to_original = dict(zip(_LABELS[:len(sub_paths)], sub_labels))
        for lab, v in sub.raw_score.items():
            raw[positional_to_original[lab]] = v
        ranking = [positional_to_original[lab] for lab in sub.ranking]
        method_used = f"{sub.method}+legality"
    # illegal rollouts trail the ranking (worst), in label order
    ranking = ranking + [labels[i] for i in range(K) if not legal_flags[i]]

    adv = normalize_advantage(raw)
    return AdvantageResult(
        ranking=ranking, advantage=adv, raw_score=raw,
        stepwise_logp=(sub.stepwise_logp if sub else None),
        method=method_used,
        raw_text=(sub.raw_text if sub else ""), K=K,
    )


# ---------------------------------------------------------------------------
# CLI: self-test + real-critic demo
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Pure-Python tests that don't need a model."""
    cases = [
        (["B", "A", "C", "D"], "K=4 typical"),
        (["A", "B"],           "K=2"),
        (["D", "C", "B", "A"], "K=4 reversed"),
        (["A", "B", "C"],      "K=3"),
    ]
    print("=== advantage_from_ranking (Borda) self-test ===")
    for rk, tag in cases:
        r = advantage_from_ranking(rk)
        adv = {k: round(v, 3) for k, v in r.advantage.items()}
        print(f"  {tag:18s}  rank={rk}  advantage={adv}")
    # Verify GRPO invariants
    r = advantage_from_ranking(["B", "A", "C", "D"])
    vals = list(r.advantage.values())
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    assert abs(m) < 1e-6,        f"mean should be 0, got {m}"
    assert abs(v ** 0.5 - 1) < 1e-3, f"std should be ~1, got {v ** 0.5}"
    # Worked example from doc
    print()
    print("=== parse_ranking ===")
    for raw in [
        "<reasoning>...</reasoning><answer>BACD</answer>",
        "<answer>B > A > C > D</answer>",
        "<answer>scene_a, scene_b</answer>",       # garbage
        "no answer here",                          # missing
    ]:
        print(f"  {raw[:50]!r:55s} → {parse_ranking(raw, 4)}")
    print()
    print("ok")
    return 0


def _real_demo(args) -> int:
    """Run the full pipeline on a few records from a sft_test*.jsonl file."""
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    print(f"[demo] loading base model from {args.model}", file=sys.stderr)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    if args.adapter:
        print(f"[demo] loading LoRA from {args.adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    records = [json.loads(l) for l in args.test_jsonl.read_text().splitlines() if l.strip()]
    records = records[: args.limit]
    print(f"[demo] {len(records)} records, method={args.method}", file=sys.stderr)

    for i, r in enumerate(records):
        # Extract image paths and the user text from the test record.
        user_content = r["messages"][0]["content"]
        image_paths = [c["path"] for c in user_content if c.get("type") == "image"]
        text_parts = [c["text"] for c in user_content if c.get("type") == "text"]
        user_text = text_parts[-1] if text_parts else None
        K = r.get("n", len(image_paths))

        res = advantage_from_critic(
            model, processor, image_paths, user_text,
            method=args.method, alpha=args.alpha,
            return_raw=True,
        )
        gt = r["ground_truth"]["ranking_best_to_worst"]
        print(f"\n--- [{i+1}/{len(records)}] id={r['id']} K={K} ---")
        print(f"  gt        : {gt}")
        print(f"  predicted : {res.ranking}  (method={res.method})")
        print(f"  raw_score : { {k: round(v,3) for k,v in res.raw_score.items()} }")
        print(f"  advantage : { {k: round(v,3) for k,v in res.advantage.items()} }")
        if res.stepwise_logp is not None:
            print(f"  stepwise z: { {k: round(v,3) for k,v in res.stepwise_logp.items()} }")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="Run pure-Python tests, no model needed.")
    ap.add_argument("--model", type=Path, help="Base model dir for demo.")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir.")
    ap.add_argument("--test-jsonl", type=Path, help="Pick records from this file for demo.")
    ap.add_argument("--limit", type=int, default=3, help="How many demo records.")
    ap.add_argument("--method", choices=["borda", "stepwise"], default="stepwise")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Confidence weight (stepwise method only).")
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()

    if args.selftest or not args.model:
        return _selftest()
    if not args.test_jsonl:
        print("--test-jsonl required for demo mode", file=sys.stderr)
        return 2
    return _real_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
