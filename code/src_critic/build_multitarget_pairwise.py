"""Convert each multitarget 4-way record into 3 pairwise (n=2) records.

Each multitarget record has 4 scenes: 1 anchor (severity 0.0) + 3 perturbed
scenes with severities like (0.2, 0.4, 0.7). The anchor is excluded; from
the remaining 3 perturbed scenes we generate C(3,2) = 3 pairs. Each pair
is a clean "is removing X worse than removing Y?" comparison — the lower
severity scene wins. New labels A/B are assigned randomly to defeat any
positional shortcut.

Output records have format compatible with
`generate_image_grounded_reason.py` — i.e. only the user message is
populated; the assistant message can be filled in by a follow-up IG run.

Usage:
    python -m src_critic.build_multitarget_pairwise \\
        --inp datasets/critic_pairs/image_grounded_reason/multitarget_636_ig.jsonl \\
        --out datasets/critic_pairs/multitarget_pairwise.jsonl \\
        --filter-judgments datasets/critic_pairs/judgments/multitarget_judge.jsonl \\
        --seed 20260515
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path


_USER_TEXT = (
    "You are a scene-quality judge. 2 top-down photoreal renderings of "
    "the same room are shown as SCENE A, SCENE B. They share "
    "the same annotations: a 5x5 grid of red coordinate dots labelled "
    "`(x, y)`; blue object_id labels (e.g. `sofa_0`); blue wall labels; "
    "green door labels; orange window labels; and a black arrow on every "
    "furniture rectangle.\n\n"
    "Arrow convention — the arrow tail is the object's geometric centre, "
    "the HEAD points to the FRONT of the piece. FRONT is the side with the "
    "functional / interactive surface (cabinet doors, drawer fronts, open "
    "shelves, seating cushions, screen, foot of a bed, etc.) — the side a "
    "user / room should face. BACK is the opposite side and should be "
    "turned toward a wall. Apply this concept generally even for furniture "
    "types not listed.\n\n"
    "These scenes are RELATED: they share identical room geometry, walls, "
    "doors and windows; they differ only in furniture placement, "
    "orientation, or presence. One of them is the unperturbed original and "
    "the others have one or more layout faults (a piece moved, rotated, "
    "swapped or removed). Rank them from BEST to WORST layout.\n\n"
    "Respond with <reasoning>...</reasoning> followed by EXACTLY one "
    "tag of the form `<answer>...</answer>` containing the scene "
    "letters concatenated in best→worst order with NO separators "
    "(e.g. `<answer>BA</answer>` for size 2 with B better than A; "
    "`<answer>BCA</answer>` for size 3; `<answer>BCDA</answer>` for "
    "size 4). Use ONLY the letters from {A, B, C, D} as appropriate "
    "for the size. Do NOT prefix with `SCENE`, do NOT use `>`, `=`, "
    "`tie`, `equal`, commas, or any other wording inside the answer "
    "tag — bare concatenated letters only."
)


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _extract_image_path(rec: dict, label: str) -> str | None:
    """Return image path for a given label using messages[0]."""
    msgs = rec.get("messages") or []
    if not msgs:
        return None
    content = msgs[0].get("content") or []
    # Walk content: text+image pairs in label order
    labels = list("ABCD"[: rec.get("n", 4)])
    img_entries = [c for c in content if c.get("type") == "image"]
    if len(img_entries) != len(labels):
        return None
    for lab, c in zip(labels, img_entries):
        if lab == label:
            return c.get("path") or c.get("image")
    return None


def _build_pair_record(parent: dict, lab1: str, lab2: str,
                       rng: random.Random) -> dict | None:
    gt = parent.get("ground_truth") or {}
    severity = gt.get("severity_per_label") or {}
    if lab1 not in severity or lab2 not in severity:
        return None
    sev1, sev2 = severity[lab1], severity[lab2]
    if sev1 == sev2:
        # tied severity — skip (no clean ranking signal)
        return None
    better_old, worse_old = (lab1, lab2) if sev1 < sev2 else (lab2, lab1)
    # Random label assignment
    flip = rng.random() < 0.5
    new_for_better = "B" if flip else "A"
    new_for_worse = "A" if flip else "B"
    img_better = _extract_image_path(parent, better_old)
    img_worse = _extract_image_path(parent, worse_old)
    if not img_better or not img_worse:
        return None
    # Build content in NEW label order A, B
    # (regardless of which is better)
    new_label_to_old = {"A": worse_old if flip else better_old,
                        "B": better_old if flip else worse_old}
    new_label_to_img = {"A": img_worse if flip else img_better,
                        "B": img_better if flip else img_worse}
    new_ranking = [new_for_better, new_for_worse]  # best → worst

    # Build user message content
    new_content: list[dict] = []
    for lab in ("A", "B"):
        new_content.append({
            "type": "image",
            "path": new_label_to_img[lab],
        })
    new_content.append({"type": "text", "text": _USER_TEXT})

    parent_id = parent.get("id", "")
    new_id = f"{parent_id}__pair_{better_old}{worse_old}_n2_mtp"
    out = {
        "id": new_id,
        "format": "ranking_n2_multitarget_pair",
        "scene_dir": parent.get("scene_dir"),
        "n": 2,
        "images": {lab: new_label_to_img[lab] for lab in ("A", "B")},
        "judge_images": parent.get("judge_images"),
        "ground_truth": {
            "ranking_best_to_worst": new_ranking,
            "chain_index_at_label": {new_for_better: 0, new_for_worse: 1},
            "ranking_mode": "multitarget_pairwise",
            "tie_sets_chain_indices": [],
            "perturbations_by_label": {
                lab: gt.get("perturbations_by_label", {}).get(
                    new_label_to_old[lab], [])
                for lab in ("A", "B")
            },
            "severity_per_label": {
                lab: severity[new_label_to_old[lab]] for lab in ("A", "B")
            },
            "source_parent_id": parent_id,
            "source_labels": {"A": new_label_to_old["A"],
                              "B": new_label_to_old["B"]},
        },
        "messages": [{"role": "user", "content": new_content}],
        "_label_perm": {"A": new_label_to_old["A"],
                        "B": new_label_to_old["B"]},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True,
                    help="multitarget_636_ig.jsonl (or filtered variant)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--filter-judgments", type=Path, default=None,
                    help="Optional path to judge_multitarget_quality output. "
                         "Records with ok=false are dropped.")
    ap.add_argument("--seed", type=int, default=20260515)
    args = ap.parse_args()

    records = _load_jsonl(args.inp)
    n_in = len(records)
    print(f"[load] {n_in} parent records", file=sys.stderr)

    drop_ids: set[str] = set()
    if args.filter_judgments and args.filter_judgments.exists():
        judgments = _load_jsonl(args.filter_judgments)
        # Filter: drop records explicitly flagged ok=false by GPT-5.2.
        # (Records with ok=true but minor kt-disagreement remain — GPT
        # accepted them as reasonable alternatives.)
        for j in judgments:
            if not j.get("ok"):
                drop_ids.add(j["id"])
        print(f"[filter] dropping {len(drop_ids)} records flagged ok=false",
              file=sys.stderr)
        records = [r for r in records if r["id"] not in drop_ids]
        print(f"[filter] {len(records)} records survive", file=sys.stderr)

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out.open("w", encoding="utf-8")

    n_emit = 0
    n_skip = 0
    for parent in records:
        gt = parent.get("ground_truth") or {}
        severity = gt.get("severity_per_label") or {}
        if not severity:
            n_skip += 1
            continue
        # Identify perturbed labels (severity > 0)
        perturbed = [lab for lab, s in severity.items() if s > 0.0]
        if len(perturbed) < 2:
            n_skip += 1
            continue
        # All C(k,2) pairs among perturbed scenes
        for lab1, lab2 in combinations(sorted(perturbed), 2):
            rec = _build_pair_record(parent, lab1, lab2, rng)
            if rec is None:
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_emit += 1
    fout.close()
    print(f"\n[done] wrote {n_emit} pairwise records, skipped {n_skip} parents → {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
