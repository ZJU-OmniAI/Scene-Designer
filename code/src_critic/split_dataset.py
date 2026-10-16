"""Split a critic-pairs `sft.jsonl` into train + 3-subset test partitions.

Strategy: stratified-by-scene. We split SCENES (not records) so a held-
out test scene is genuinely unseen by the model. Random-shuffle scenes
with a fixed seed, take the last `--test-frac` fraction as test, the
rest as train. Within the test split, write three subset files:

  * `sft_test_n2.jsonl`         — every record with n == 2
  * `sft_test_cumulative.jsonl` — n in {3, 4} AND mode == "cumulative"
  * `sft_test_sibling.jsonl`    — n in {3, 4} AND mode == "sibling"

Each subset gets its own evaluation rule (see `eval_qwen3vl.py`):
exact-match for n2 / cumulative; tier-flexibility-aware match for
sibling (because sibling ground-truth answers are sometimes Gemini's
verdict within tier-equivalent positions, so an exact-match against the
stored answer would unfairly penalize an alternative-but-equivalent
ranking).

Usage:
    python -m src_critic.split_dataset \\
        --in datasets/critic_pairs/sft.jsonl \\
        --out-train datasets/critic_pairs/sft_train.jsonl \\
        --test-out-dir datasets/critic_pairs \\
        --test-frac 0.10 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _subset_for(record: dict) -> str:
    """Returns one of "n2", "cumulative", "sibling"."""
    n = record.get("n", 2)
    if n == 2:
        return "n2"
    mode = record.get("ground_truth", {}).get("ranking_mode")
    if mode == "cumulative":
        return "cumulative"
    if mode == "sibling":
        return "sibling"
    return "other"  # shouldn't normally hit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out-train", type=Path, required=True)
    ap.add_argument("--test-out-dir", type=Path, required=True,
                    help="Directory for test subset files: "
                         "sft_test_n2.jsonl / sft_test_cumulative.jsonl / "
                         "sft_test_sibling.jsonl. Also writes a combined "
                         "sft_test.jsonl for backwards compatibility.")
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = [json.loads(line) for line in args.inp.read_text().splitlines()
               if line.strip()]
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        scene = Path(r["scene_dir"]).name
        by_scene[scene].append(r)
    scene_names = sorted(by_scene.keys())
    rng = random.Random(args.seed)
    rng.shuffle(scene_names)
    n_test = max(1, int(round(len(scene_names) * args.test_frac)))
    test_scenes = set(scene_names[-n_test:])
    train_scenes = set(scene_names[:-n_test])

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.test_out_dir.mkdir(parents=True, exist_ok=True)

    test_paths = {
        "n2":         args.test_out_dir / "sft_test_n2.jsonl",
        "cumulative": args.test_out_dir / "sft_test_cumulative.jsonl",
        "sibling":    args.test_out_dir / "sft_test_sibling.jsonl",
        "other":      args.test_out_dir / "sft_test_other.jsonl",
    }
    test_combined = args.test_out_dir / "sft_test.jsonl"

    train_fh = args.out_train.open("w", encoding="utf-8")
    sub_fhs = {k: p.open("w", encoding="utf-8") for k, p in test_paths.items()}
    combined_fh = test_combined.open("w", encoding="utf-8")

    counts = defaultdict(int)
    n_tr = 0
    try:
        for r in records:
            scene = Path(r["scene_dir"]).name
            line = json.dumps(r, ensure_ascii=False)
            if scene in test_scenes:
                subset = _subset_for(r)
                sub_fhs[subset].write(line + "\n")
                combined_fh.write(line + "\n")
                counts[subset] += 1
            else:
                train_fh.write(line + "\n")
                n_tr += 1
    finally:
        train_fh.close()
        combined_fh.close()
        for fh in sub_fhs.values():
            fh.close()
        # Drop empty `other` shard
        if counts.get("other", 0) == 0:
            test_paths["other"].unlink(missing_ok=True)

    print(f"scenes: {len(train_scenes)} train, {len(test_scenes)} test")
    print(f"train records: {n_tr}")
    print(f"test records by subset:")
    for k in ("n2", "cumulative", "sibling", "other"):
        if counts.get(k, 0):
            print(f"  {k}: {counts[k]}  -> {test_paths[k].name}")
    print(f"combined test: sft_test.jsonl ({sum(counts.values())} records)")
    print(f"test scenes: {sorted(test_scenes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
