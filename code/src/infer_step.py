"""Sanity-check inference, walked in trajectory order.

Walks every step in order:
    floor_plan_0..N → furniture_0..N → wall_0..N → ceiling_0..N
For each step, feeds the model the SAME input the trainer used
(system + task + memory + image) and lets it generate the assistant turn.
Then prints model output next to ground truth + a per-tool-call match
summary (replay-style: compare name + arguments 1:1).

Note: this is INDEPENDENT-step inference — at step k the memory and image
come from the dataset (built from ground-truth tool_results of steps 0..k-1),
not from the model's own previous predictions. So order only affects the
print order, not what gets fed to the model.

Usage:
  # walk all 42 steps in trajectory order
  python src/infer_step.py

  # only the first 5 (good first smoke test)
  python src/infer_step.py --limit 5

  # whole furniture stage only
  python src/infer_step.py --stage furniture

  # one specific step
  python src/infer_step.py --stage furniture --step-idx 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_dataset import StepSample, _build_messages, discover_step_samples  # noqa: E402


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = REPO_ROOT_DEFAULT / "model" / "OpenSceneAgent"
SFT_ROOT_DEFAULT = REPO_ROOT_DEFAULT / "datasets" / "sft"


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def parse_assistant_text(text: str) -> dict:
    """Extract <think> and every <tool_call> JSON block from a model emit."""
    think_m = _THINK_RE.search(text)
    think = think_m.group(1).strip() if think_m else None

    tool_calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        body = m.group(1).strip()
        try:
            tc = json.loads(body)
            tool_calls.append({
                "name": tc.get("name"),
                "arguments": tc.get("arguments") or {},
            })
        except json.JSONDecodeError:
            tool_calls.append({"_raw": body, "_parse_error": True})
    return {"think": think, "tool_calls": tool_calls}


def compare_tool_calls(predicted: list[dict], gt: list[dict]) -> dict:
    """Per-position match: same name AND same arguments."""
    n_match_name = 0
    n_match_args = 0
    for i in range(min(len(predicted), len(gt))):
        p, g = predicted[i], gt[i]
        if p.get("name") == g.get("name"):
            n_match_name += 1
            if p.get("arguments") == g.get("arguments"):
                n_match_args += 1
    return {
        "n_predicted": len(predicted),
        "n_gt": len(gt),
        "n_match_name": n_match_name,
        "n_match_args": n_match_args,
    }


_STAGE_ORDER = {"floor_plan": 0, "furniture": 1, "wall": 2, "ceiling": 3}


def pick_samples(samples: list[StepSample], args) -> list[StepSample]:
    """Walk samples in trajectory order: floor_plan → furniture → wall → ceiling,
    each stage in step_idx order. Optional --stage / --step-idx / --limit filters."""
    if args.stage is not None and args.step_idx is not None:
        hit = [s for s in samples if s.stage == args.stage and s.step_idx == args.step_idx]
        if not hit:
            sys.exit(f"no sample with stage={args.stage} step={args.step_idx}")
        return hit
    pool = samples if args.stage is None else [s for s in samples if s.stage == args.stage]
    pool = sorted(pool, key=lambda s: (_STAGE_ORDER.get(s.stage, 99), s.step_idx))
    if args.limit > 0:
        pool = pool[: args.limit]
    return pool


def run_one(model, processor, sample: StepSample, max_new_tokens: int) -> str:
    """Build the prompt the same way training did, generate, return only the new tokens (decoded)."""
    _, prompt_msgs = _build_messages(sample)
    prompt_text = processor.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    from src.scene_observation import load_message_images
    images = load_message_images(prompt_msgs)
    inputs = processor(text=[prompt_text], images=images, return_tensors="pt").to(model.device)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[0, prompt_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=False)


def _print_block(title: str, body: str) -> None:
    print(f"\n--- {title} ---")
    print(body if body else "(empty)")


def _print_input(sample: StepSample, *, show_system: bool = False) -> None:
    """Show exactly what the model receives at this step: image + task + memory
    (+ optional system prompt body)."""
    img = str(sample.image_path) if sample.image_path else "(no image — floor_plan stage / first step)"
    _print_block("INPUT image", img)
    _print_block("INPUT user-text (## Task / ## Memory)",
                 f"## Task\n{sample.task_context}\n\n## Memory\n{sample.memory_str}")
    sys_body = sample.system or ""
    if show_system:
        _print_block("INPUT system prompt (full)", sys_body)
    else:
        sys_preview = sys_body[:200].replace("\n", " ")
        _print_block("INPUT system prompt", f"({len(sys_body)} chars; first 200) {sys_preview}…")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=str(MODEL_DEFAULT))
    ap.add_argument("--sft-root", default=str(SFT_ROOT_DEFAULT))
    ap.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    ap.add_argument("--limit", type=int, default=0,
                    help="truncate to the first N steps in trajectory order. "
                         "default 0 = run all 42 steps.")
    ap.add_argument("--stage", choices=["floor_plan", "furniture", "wall", "ceiling"],
                    help="only run this one stage")
    ap.add_argument("--step-idx", type=int, help="(with --stage) pick this single step")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-pixels", type=int, default=65536)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--show-raw", action="store_true", default=True,
                    help="print the raw decoded model output (default: on)")
    ap.add_argument("--no-show-raw", dest="show_raw", action="store_false")
    ap.add_argument("--show-system", action="store_true", default=False,
                    help="dump the full system prompt body (default: only first 200 chars)")
    args = ap.parse_args()

    print(f"[load] processor + model from {args.model_dir}")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    from src.scene_observation import configure_image_budget
    configure_image_budget(processor, args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    model.config.use_cache = True  # KV cache for fast generation

    samples = discover_step_samples(Path(args.sft_root), Path(args.repo_root))
    print(f"[load] {len(samples)} step samples available in dataset")

    chosen = pick_samples(samples, args)
    print(f"\nrunning {len(chosen)} sample(s)\n")

    totals = {"n_match_name": 0, "n_match_args": 0, "n_gt": 0, "n_predicted": 0}
    for i, s in enumerate(chosen):
        print("=" * 80)
        print(f"[{i + 1}/{len(chosen)}] stage={s.stage} step={s.step_idx} "
              f"src={s.source_file} image={'yes' if s.image_path else 'no'}")
        print("=" * 80)

        gt_calls = [{"name": tc["name"], "arguments": tc.get("arguments") or {}}
                    for tc in (s.tool_calls or [])]

        # Show the model's input BEFORE generation so you can read the prompt
        # next to the prediction.
        _print_input(s, show_system=args.show_system)

        generated = run_one(model, processor, s, args.max_new_tokens)
        parsed = parse_assistant_text(generated)

        if args.show_raw:
            _print_block("RAW MODEL OUTPUT (verbatim)",
                         "↓↓↓\n" + generated + "\n↑↑↑")
        _print_block("GENERATED <think>", parsed["think"] or "(no <think> block found)")
        _print_block(
            "GENERATED tool_calls",
            "\n".join(
                f"  {tc.get('name')}({json.dumps(tc.get('arguments'), ensure_ascii=False)})"
                if "_raw" not in tc else f"  [parse-error] {tc['_raw'][:120]}"
                for tc in parsed["tool_calls"]
            ) if parsed["tool_calls"] else "(none)",
        )

        gt_think_short = s.assistant_content
        if len(gt_think_short) > 400:
            gt_think_short = gt_think_short[:400] + "  ...(truncated)"
        _print_block("GROUND TRUTH <think>", gt_think_short)
        _print_block(
            "GROUND TRUTH tool_calls",
            "\n".join(
                f"  {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})"
                for tc in gt_calls
            ) if gt_calls else "(none — summary turn)",
        )

        diff = compare_tool_calls(parsed["tool_calls"], gt_calls)
        for k in totals:
            totals[k] += diff[k]
        print(f"\n--- MATCH: name {diff['n_match_name']}/{diff['n_gt']} "
              f"| args {diff['n_match_args']}/{diff['n_gt']} "
              f"| predicted {diff['n_predicted']} calls\n")

    if len(chosen) > 1:
        print("=" * 80)
        print(f"OVERALL: name {totals['n_match_name']}/{totals['n_gt']}  "
              f"args {totals['n_match_args']}/{totals['n_gt']}  "
              f"(predicted {totals['n_predicted']}; gt {totals['n_gt']})")
        print("=" * 80)


if __name__ == "__main__":
    main()
