"""Permute the A/B/C/D labels of n4 records so the GT answer
distribution is uniform across all 24 4-letter permutations.

Why this matters: the n4 generators always emit records as
  anchor → light → medium → worst   labeled   A → B → C → D
so the model can shortcut to "always output ABCD". This script
randomly remaps labels per record, rewriting:
  * messages[0].content image order
  * ground_truth.ranking_best_to_worst
  * ground_truth.chain_index_at_label
  * ground_truth.perturbations_by_label
  * ground_truth.severity_per_label (if present)
  * images / judge_images dict keys
  * gemini.ranking (legacy)
  * messages[1].content (assistant reasoning + <answer> tag): every
    "Scene X" / "SCENE X" mention is renamed.

The permutation is DETERMINISTIC from the record id so re-applying is
idempotent (same input → same output).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def _perm_for_record(record_id: str, n: int,
                     force_uniform_seed: int | None = None) -> list[str]:
    """Pseudo-random permutation of ['A','B',...] of length `n`,
    deterministic on record id. Re-running the script on the same
    record yields the same permutation.

    For n=2 we permute only {A,B}. For n=4 we permute {A,B,C,D}."""
    if n not in (2, 4):
        raise ValueError(f"only n=2 or n=4 supported, got {n}")
    h = hashlib.sha256(record_id.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")
    if force_uniform_seed is not None:
        seed ^= force_uniform_seed
    import random
    rng = random.Random(seed)
    perm = list("ABCD"[:n])
    rng.shuffle(perm)
    return perm


_SCENE_LABEL_RE = re.compile(r"(?i)\bSCENE\s+([ABCD])\b")


def _rewrite_scene_mentions(text: str, label_map: dict[str, str]) -> str:
    """Two-step substitution to avoid collisions when labels swap.
    First map each old label to a sentinel, then sentinels to new labels.
    Only handles labels present in label_map (i.e., for n=2 records,
    only A and B are touched)."""
    if not text:
        return text
    sentinel_all = {"A": "\x01", "B": "\x02", "C": "\x03", "D": "\x04"}
    sentinel = {old: sentinel_all[old] for old in label_map}
    def step1(m):
        kw = m.group(0).split()[0]  # "Scene" or "SCENE"
        old = m.group(1).upper()
        if old not in sentinel:
            return m.group(0)  # leave untouched (e.g., n=2 record mentioning Scene C — unlikely)
        return f"{kw} {sentinel[old]}"
    out = _SCENE_LABEL_RE.sub(step1, text)
    for old, snt in sentinel.items():
        new = label_map[old]
        out = out.replace(snt, new)
    return out


_ANSWER_RE = re.compile(r"<answer>([A-D]+)</answer>", re.IGNORECASE)


def _rewrite_answer(text: str, label_map: dict[str, str]) -> str:
    def repl(m):
        old_answer = m.group(1).upper()
        new_answer = "".join(label_map[c] for c in old_answer)
        return f"<answer>{new_answer}</answer>"
    return _ANSWER_RE.sub(repl, text)


def permute_record(rec: dict, *, force_uniform_seed: int | None = None) -> dict:
    """Apply a deterministic label permutation to one record (n2 or n4).
    Returns a NEW record dict (does not mutate input)."""
    n = int(rec.get("n", 2))
    perm = _perm_for_record(rec["id"], n, force_uniform_seed=force_uniform_seed)
    base_labels = list("ABCD"[:n])
    # perm[i] is the new label for the image originally at position i
    # (old label = base_labels[i]).
    old_to_new = {base_labels[i]: perm[i] for i in range(n)}
    new_to_old = {v: k for k, v in old_to_new.items()}

    out = json.loads(json.dumps(rec))  # deep copy

    # 1) messages[0].content: n image entries followed by 1 text entry.
    content = out["messages"][0]["content"]
    if isinstance(content, list) and len(content) >= n + 1:
        imgs = content[:n]
        rest = content[n:]
        new_imgs = []
        for i in range(n):
            new_label = base_labels[i]
            old_label = new_to_old[new_label]
            old_idx = ord(old_label) - ord("A")
            new_imgs.append(imgs[old_idx])
        out["messages"][0]["content"] = new_imgs + rest

    # 2) ground_truth fields
    gt = out["ground_truth"]
    if isinstance(gt.get("ranking_best_to_worst"), list):
        gt["ranking_best_to_worst"] = [old_to_new[lab]
                                       for lab in gt["ranking_best_to_worst"]]
    if "chain_index_at_label" in gt and isinstance(gt["chain_index_at_label"], dict):
        gt["chain_index_at_label"] = {old_to_new[lab]: idx
                                      for lab, idx in gt["chain_index_at_label"].items()}
    if "perturbations_by_label" in gt and isinstance(gt["perturbations_by_label"], dict):
        gt["perturbations_by_label"] = {old_to_new[lab]: perts
                                        for lab, perts in gt["perturbations_by_label"].items()}
    if "severity_per_label" in gt and isinstance(gt["severity_per_label"], dict):
        gt["severity_per_label"] = {old_to_new[lab]: v
                                    for lab, v in gt["severity_per_label"].items()}

    # 3) images / judge_images dicts (legacy / for eval)
    for key in ("images", "judge_images"):
        if key in out and isinstance(out[key], dict):
            out[key] = {old_to_new[lab]: path for lab, path in out[key].items()}

    # 4) gemini.ranking (legacy)
    if ("gemini" in out and isinstance(out["gemini"], dict)
            and isinstance(out["gemini"].get("ranking"), list)):
        out["gemini"]["ranking"] = [old_to_new[lab] for lab in out["gemini"]["ranking"]]

    # 5) Assistant message: rewrite "Scene X" and <answer>...</answer>
    if len(out["messages"]) >= 2 and out["messages"][1].get("role") == "assistant":
        asst = out["messages"][1].get("content") or ""
        asst = _rewrite_scene_mentions(asst, old_to_new)
        asst = _rewrite_answer(asst, old_to_new)
        out["messages"][1]["content"] = asst

    # Stash the perm for auditability.
    out["_label_perm"] = "".join(perm)
    return out


def permute_file(inp: Path, outp: Path,
                 force_uniform_seed: int | None = None) -> tuple[int, dict[str, int]]:
    """Permute every record in `inp`, write to `outp`. Returns
    (n_records, distribution_of_answer_strings)."""
    n = 0
    from collections import Counter
    cnt = Counter()
    with inp.open() as f, outp.open("w") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            new_r = permute_record(r, force_uniform_seed=force_uniform_seed)
            g.write(json.dumps(new_r, ensure_ascii=False) + "\n")
            n += 1
            cnt["".join(new_r["ground_truth"]["ranking_best_to_worst"])] += 1
    return n, cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-xor", type=int, default=None,
                    help="Optional XOR onto the record-id hash so you "
                         "can produce a different permutation set.")
    args = ap.parse_args()
    n, cnt = permute_file(args.inp, args.out, force_uniform_seed=args.seed_xor)
    print(f"wrote {n} records to {args.out}", file=sys.stderr)
    print(f"answer distribution (top 12):", file=sys.stderr)
    for ans, c in sorted(cnt.items(), key=lambda x: -x[1])[:12]:
        print(f"  {ans}  {c}  ({c/n*100:.1f}%)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
