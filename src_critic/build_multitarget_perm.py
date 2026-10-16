"""Generate label-permutation augmentation for multitarget 4-way records.

For each surviving (mt-judge ok=true) parent, sample k non-identity
permutations of S_n (default 4 from S4=24). For each permutation, build a
new record with:
- images re-ordered so that the new slot L holds the image originally at
  position pi^{-1}(L);
- assistant reasoning relabelled (SCENE A → SCENE pi(A), ...);
- ground_truth.ranking_best_to_worst, severity_per_label,
  perturbations_by_label, chain_index_at_label all relabelled;
- <answer>XXXX</answer> chain relabelled;
- _label_perm updated to compose with parent's perm.

The text-only relabelling uses uppercase boundary tokens (SCENE_X and
isolated single letters in the answer tag), so it's safe.

Usage:
  python -m src_critic.build_multitarget_perm \\
    --inp multitarget_636_ig.jsonl \\
    --judge multitarget_judge.jsonl \\
    --k 4 --seed 20260516 \\
    --out multitarget_perm_aug.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from itertools import permutations
from pathlib import Path


_ANS_RE = re.compile(r"<answer>\s*([A-D]+)\s*</answer>", re.IGNORECASE)


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _relabel_text(txt: str, mapping: dict[str, str]) -> str:
    """Replace SCENE X and isolated answer-letter chains using a 2-pass
    token swap to avoid aliasing (A→B and B→A done correctly)."""
    if not txt:
        return txt
    # Use unique sentinels
    tmp = txt
    sentinels = {old: f"__§{i}§__" for i, old in enumerate(mapping)}
    for old, sentinel in sentinels.items():
        tmp = re.sub(rf"\bSCENE\s+{old}\b", f"SCENE {sentinel}",
                     tmp, flags=re.IGNORECASE)
    for old, sentinel in sentinels.items():
        new = mapping[old]
        tmp = tmp.replace(f"SCENE {sentinel}", f"SCENE {new}")
    # Now relabel the answer chain.
    def _ans_sub(m: re.Match) -> str:
        chain = m.group(1).upper()
        new_chain = "".join(mapping.get(c, c) for c in chain)
        return f"<answer>{new_chain}</answer>"
    tmp = _ANS_RE.sub(_ans_sub, tmp)
    return tmp


def _permute_record(parent: dict, perm: tuple[str, ...],
                    labels: list[str]) -> dict | None:
    """perm[i] is the OLD label whose image moves to NEW slot labels[i].
    Equivalently mapping new->old. The relabelling for text/ranking is
    OLD→NEW, which is the inverse of perm (i.e., pi^{-1}).
    """
    n = len(labels)
    new_to_old = dict(zip(labels, perm))
    old_to_new = {v: k for k, v in new_to_old.items()}
    msgs = parent.get("messages") or []
    if len(msgs) < 2:
        return None
    user_content = msgs[0].get("content") or []
    asst_content = msgs[1].get("content")

    # Split user content into image entries + final text.
    img_entries = [c for c in user_content if c.get("type") == "image"]
    text_entries = [c for c in user_content if c.get("type") == "text"]
    if len(img_entries) != n:
        return None
    # New image order = images at old positions perm[0], perm[1], ...
    old_label_to_img = dict(zip(labels, img_entries))
    new_imgs = [old_label_to_img[new_to_old[lab]] for lab in labels]
    new_user = list(new_imgs) + text_entries  # text unchanged (mentions A/B/C/D generically)

    # Relabel assistant content (string or list).
    if isinstance(asst_content, str):
        new_asst = _relabel_text(asst_content, old_to_new)
    elif isinstance(asst_content, list):
        new_asst = []
        for c in asst_content:
            if c.get("type") == "text":
                cc = dict(c)
                cc["text"] = _relabel_text(c.get("text", ""), old_to_new)
                new_asst.append(cc)
            else:
                new_asst.append(c)
    else:
        return None

    gt = parent.get("ground_truth") or {}
    old_chain = gt.get("ranking_best_to_worst") or []
    new_chain = [old_to_new[x] for x in old_chain]
    new_chain_index = {old_to_new[lab]: i for lab, i in
                       (gt.get("chain_index_at_label") or {}).items()}
    new_sev = {old_to_new[lab]: v for lab, v in
               (gt.get("severity_per_label") or {}).items()}
    new_perts = {old_to_new[lab]: v for lab, v in
                 (gt.get("perturbations_by_label") or {}).items()}

    old_label_perm_raw = parent.get("_label_perm")
    if isinstance(old_label_perm_raw, dict):
        old_label_perm = old_label_perm_raw
    elif isinstance(old_label_perm_raw, str) and len(old_label_perm_raw) == n:
        # Compact form like "ACBD": labels[i] maps to old_label_perm_raw[i]
        old_label_perm = dict(zip(labels, list(old_label_perm_raw)))
    else:
        old_label_perm = {lab: lab for lab in labels}
    # Compose: new_label -> old_label_perm[old_label_at_new_slot]
    new_label_perm = {lab: old_label_perm.get(new_to_old[lab], new_to_old[lab])
                      for lab in labels}

    perm_tag = "".join(perm)  # e.g., "BCAD"
    parent_id = parent.get("id", "")
    new_id = f"{parent_id}__perm_{perm_tag}"

    out = {
        "id": new_id,
        "format": parent.get("format"),
        "scene_dir": parent.get("scene_dir"),
        "n": n,
        "images": {lab: new_imgs[i].get("path") for i, lab in enumerate(labels)},
        "judge_images": parent.get("judge_images"),
        "ground_truth": {
            "ranking_best_to_worst": new_chain,
            "chain_index_at_label": new_chain_index,
            "ranking_mode": gt.get("ranking_mode"),
            "tie_sets_chain_indices": gt.get("tie_sets_chain_indices", []),
            "perturbations": gt.get("perturbations"),
            "perturbations_by_label": new_perts,
            "severity_per_label": new_sev,
            "source_parent_id": parent_id,
            "perm_new_to_old": new_to_old,
        },
        "messages": [
            {"role": "user", "content": new_user},
            {"role": "assistant", "content": new_asst},
        ],
        "_label_perm": new_label_perm,
        "_reasoning_method": parent.get("_reasoning_method"),
        "_aug_source": "multitarget_perm",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--judge", type=Path, default=None,
                    help="Optional judge file; drop parents with ok=false.")
    ap.add_argument("--k", type=int, default=4,
                    help="Number of non-identity perms per parent")
    ap.add_argument("--seed", type=int, default=20260516)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parents = _load_jsonl(args.inp)
    print(f"[load] {len(parents)} parents", file=sys.stderr)
    drop: set[str] = set()
    if args.judge and args.judge.exists():
        for j in _load_jsonl(args.judge):
            if not j.get("ok"):
                drop.add(j["id"])
        print(f"[judge] dropping {len(drop)} ok=false parents", file=sys.stderr)
    parents = [p for p in parents if p["id"] not in drop]
    print(f"[load] {len(parents)} surviving parents", file=sys.stderr)

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out.open("w", encoding="utf-8")
    n_emit = 0
    n_skip = 0
    for parent in parents:
        n = int(parent.get("n", 4))
        labels = list("ABCD"[:n])
        all_perms = [p for p in permutations(labels) if list(p) != labels]
        rng.shuffle(all_perms)
        chosen = all_perms[: args.k]
        for perm in chosen:
            rec = _permute_record(parent, perm, labels)
            if rec is None:
                n_skip += 1
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_emit += 1
    fout.close()
    print(f"\n[done] wrote {n_emit} perm records (skipped={n_skip}) → {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
