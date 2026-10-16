"""One-off fix: after `permute_n4_labels.py` had rewritten "Scene X"
mentions, the reasoning's bare-letter ranking conclusions like
"A > B > C > D" or "A > B" were left UN-permuted because the original
regex only matched the "Scene X" form. This produces records whose
<answer> tag (rewritten correctly) and reasoning conclusion text
(stale) disagree — corrupting the training signal.

This script reads each permuted file, recovers the per-record
permutation from the `_label_perm` field that the permute script
stashed, and rewrites any remaining bare-letter sequence that
references the old A/B/C/D labels in the reasoning.

Patterns handled (within messages[1].content text):
  * `A > B`, `A > B > C`, `A > B > C > D` (any length 2..4)
  * `A, B, C, D` lists (length 2..4)
  * Bare `[A-D]` tokens at word boundary that aren't part of a longer
    word (e.g. "is A", "ranks A", but NOT the article "a") — limited to
    cases where there's at least one adjacent letter-token within a
    short window (so accidental article matches are skipped).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# `A > B` chains of length 2..4. Captures all 4 groups even if only 2-3 letters present.
_GT_CHAIN_RE = re.compile(
    r"\b([A-D])\s*>\s*([A-D])(?:\s*>\s*([A-D]))?(?:\s*>\s*([A-D]))?\b"
)

# `A, B, C, D` style lists (with optional "and" / "then" / "followed by")
_COMMA_CHAIN_RE = re.compile(
    r"\b([A-D])\s*[,]\s*(?:then\s+|followed by\s+|and\s+)?([A-D])"
    r"(?:\s*[,]\s*(?:then\s+|followed by\s+|and\s+)?([A-D]))?"
    r"(?:\s*[,]\s*(?:then\s+|followed by\s+|and\s+)?([A-D]))?\b"
)


def _remap_chain_match(m: re.Match, mapping: dict[str, str]) -> str:
    """Reconstruct the matched chain with letters remapped."""
    raw = m.group(0)
    # Replace each captured letter with its mapped letter in the raw string.
    # We replace by position to preserve the original separators / spacing.
    # The capture groups appear in `m.group(1..k)` order; their positions
    # in `raw` are the spans.
    out = list(raw)
    for gi in range(1, m.lastindex + 1 if m.lastindex else 1):
        if m.group(gi) is None:
            continue
        start, end = m.span(gi)
        # `start`/`end` are absolute offsets in the original string —
        # translate to offsets within `raw`:
        raw_start = start - m.start()
        raw_end = end - m.start()
        old = raw[raw_start:raw_end]
        new = mapping.get(old, old)
        out[raw_start:raw_end] = list(new)
    return "".join(out)


def fix_reasoning(text: str, mapping: dict[str, str]) -> str:
    """Brute-force remap: every isolated uppercase A/B/C/D character in the
    reasoning body gets mapped to its new label. To avoid double-mapping
    already-correct content, we first mask out:
      * `Scene X` references (already permuted by permute_n4_labels.py)
      * `<answer>...</answer>` tag (already permuted)

    Empirically (sampled 100 records), ALL isolated A/B/C/D in the
    polished reasoning are ranking-letter references; the article "A"
    does not appear because the polish prompt produces formal interior-
    design prose ("the sofa", "a sofa" → "the sofa")."""
    if not text:
        return text
    # Mask Scene-prefixed labels (already correctly mapped). Placeholder
    # uses ONLY non-A/B/C/D characters so the bare-letter remap below
    # doesn't touch the masked region.
    scene_holders: list[str] = []
    def cap_scene(m):
        scene_holders.append(m.group(0))
        return f"\x01{len(scene_holders) - 1}\x02"
    masked = re.sub(r"(?i)Scene\s+[A-D]\b", cap_scene, text)
    # Mask <answer>...</answer> tag (already correctly mapped).
    answer_holders: list[str] = []
    def cap_ans(m):
        answer_holders.append(m.group(0))
        return f"\x03{len(answer_holders) - 1}\x04"
    masked = re.sub(r"(?i)<answer>[A-D]+</answer>", cap_ans, masked)
    # Now remap every isolated A/B/C/D.
    masked = re.sub(
        r"(?<![A-Za-z])([A-D])(?![a-z])",
        lambda m: mapping.get(m.group(1), m.group(1)),
        masked,
    )
    # Restore masked content.
    for i, sm in enumerate(scene_holders):
        masked = masked.replace(f"\x01{i}\x02", sm)
    for i, an in enumerate(answer_holders):
        masked = masked.replace(f"\x03{i}\x04", an)
    return masked


def _label_perm_to_mapping(perm: str) -> dict[str, str]:
    """Convert a permutation string like "ACBD" into old→new mapping
    {A: A, B: C, C: B, D: D}. Works for n=2 ("BA" → {A:B, B:A}) too."""
    base = "ABCD"[: len(perm)]
    return {base[i]: perm[i] for i in range(len(perm))}


def fix_record(rec: dict) -> dict:
    perm = rec.get("_label_perm")
    if not perm:
        return rec
    mapping = _label_perm_to_mapping(perm)
    msgs = rec.get("messages", [])
    if len(msgs) >= 2 and msgs[1].get("role") == "assistant":
        text = msgs[1].get("content") or ""
        msgs[1]["content"] = fix_reasoning(text, mapping)
    return rec


def fix_file(inp: Path, outp: Path) -> tuple[int, int]:
    n_in = 0; n_changed = 0
    with inp.open() as f, outp.open("w") as g:
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            r = json.loads(line)
            old_text = ""
            if len(r.get("messages", [])) >= 2 and r["messages"][1].get("role") == "assistant":
                old_text = r["messages"][1].get("content") or ""
            r = fix_record(r)
            new_text = ""
            if len(r.get("messages", [])) >= 2 and r["messages"][1].get("role") == "assistant":
                new_text = r["messages"][1].get("content") or ""
            if new_text != old_text:
                n_changed += 1
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    return n_in, n_changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    n_in, n_changed = fix_file(args.inp, args.out)
    print(f"{args.inp.name} → {args.out.name}: {n_in} records, "
          f"{n_changed} reasoning texts modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
