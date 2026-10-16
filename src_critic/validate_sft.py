"""Pre-flight validator for SFT jsonl files.

Run BEFORE splitting / training / eval to catch:

  * structural corruption (missing keys, wrong shapes, bad msg layout)
  * image paths that no longer exist on disk
  * answer-vs-ground-truth mismatch in the assistant completion
  * privileged info that leaked through polish:
        - explicit numeric degrees ("180 degrees", "45°"),
        - coordinate parentheses like "(2, 3)"
        - cross-scene action descriptions ("was moved from", "by rotating")
  * truly dynamic verb phrases (auxiliary + past-participle of a banned
    verb, or "by + Verbing" with a banned verb).

Static uses like "the sofa is placed against the wall" or "an awkwardly
rotated bookcase" are NOT flagged — those describe the visible state of
a single scene and are fine.

Hard failures (schema / images / answer mismatch) get split into a
quarantine file; the cleaned input is rewritten in-place with a `.bak`
backup. Soft warnings (privileged info / dynamic phrases) are reported
but do NOT remove records — they just surface what should be re-polished.

Usage:
    python -m src_critic.validate_sft datasets/critic_pairs/sft.jsonl
    python -m src_critic.validate_sft datasets/critic_pairs/sft.jsonl --strict
        # in --strict, soft warnings also quarantine the record.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


_REQUIRED_TOP = {"id", "n", "format", "images", "judge_images",
                 "ground_truth", "messages"}
_REQUIRED_GT = {"ranking_best_to_worst", "ranking_mode", "perturbations",
                "chain_index_at_label", "perturbations_by_label"}
_REASONING_RE = re.compile(r"<reasoning>(.+?)</reasoning>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>([^<]+)</answer>", re.IGNORECASE)


# --- soft-warn patterns (privileged info / dynamic action) ----------------

# Numeric degree mentions — privileged perturbation info, not visible from
# a top-down render alone.
_PAT_DEGREES = re.compile(r"\b\d{1,3}\s*(?:°|degree[s]?)\b", re.IGNORECASE)

# Coordinate parentheses leaked into prose — `(2, 3)` style.
_PAT_COORDS = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")

# Cross-scene comparison phrases that reveal the scenes are perturbations
# of each other.
_PAT_COMPARISON = re.compile(
    r"\b(?:compared\s+to\s+scene|relative\s+to\s+scene|"
    r"differs\s+from\s+scene|the\s+difference\s+(?:is|between))\b",
    re.IGNORECASE,
)

# Auxiliary + past-participle of a dynamic verb. This catches
# "was moved", "has been rotated", "got swapped", "is now placed at"
# (the "is now" form is genuinely dynamic / change-implying).
_DYN_VERBS = (
    "moved", "shifted", "relocated", "rotated", "swapped",
    "removed", "rearranged",
)
_PAT_AUX_DYN = re.compile(
    r"\b(?:was|were|has\s+been|have\s+been|got|is\s+now|are\s+now)\s+"
    r"(?:" + "|".join(_DYN_VERBS) + r")\b",
    re.IGNORECASE,
)

# "by + Verbing" forms describe an action performed on the scene.
_PAT_BY_GERUND = re.compile(
    r"\bby\s+(?:moving|shifting|relocating|rotating|swapping|"
    r"removing|placing|pushing|turning|rearranging)\b",
    re.IGNORECASE,
)


def _extract_reasoning(asst_content) -> str:
    if isinstance(asst_content, list):
        text = " ".join(b.get("text", "") for b in asst_content
                        if b.get("type") == "text")
    else:
        text = asst_content or ""
    m = _REASONING_RE.search(text)
    return m.group(1) if m else ""


def _extract_answer(asst_content) -> list[str]:
    """Extract the ranking from the assistant message's `<answer>…</answer>`.

    Tolerant to all the formats the SFT trainer / eval will see in practice:
    * the new bare-letter concatenated form `BCDA` (current SFT format),
    * `>`-separated form `B > C > D > A` (legacy format / other models),
    * verbose `SCENE A > SCENE B` (gpt-5.2 default), and
    * HTML-escaped `&gt;`.
    Stays in sync with `eval_qwen3vl._parse_answer` — same regex + post-
    processing — so a record that the eval pipeline can score is also a
    record that the validator considers well-formed.
    """
    if isinstance(asst_content, list):
        text = " ".join(b.get("text", "") for b in asst_content
                        if b.get("type") == "text")
    else:
        text = asst_content or ""
    m = _ANSWER_RE.search(text)
    if not m:
        return []
    raw = m.group(1)
    if re.search(r"[=]|\bequal\b|\btie[ds]?\b|\band\b", raw, flags=re.IGNORECASE):
        return []
    return [c for c in raw.upper() if c in "ABCD"]


def _check_schema(rec: dict) -> list[str]:
    errs = []
    miss = _REQUIRED_TOP - rec.keys()
    if miss:
        errs.append(f"top_missing:{sorted(miss)}")
        return errs
    n = rec.get("n")
    if n not in (2, 3, 4):
        errs.append(f"bad_n:{n}")
    miss_gt = _REQUIRED_GT - rec["ground_truth"].keys()
    if miss_gt:
        errs.append(f"gt_missing:{sorted(miss_gt)}")
    expected_labels = ["A", "B", "C", "D"][:n] if isinstance(n, int) else []
    rk = rec["ground_truth"].get("ranking_best_to_worst", [])
    if sorted(rk) != sorted(expected_labels):
        errs.append(f"bad_ranking_labels:{rk}")
    msgs = rec.get("messages", [])
    if len(msgs) != 2:
        errs.append(f"bad_msgs_len:{len(msgs)}")
    elif msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
        errs.append("bad_msg_roles")
    return errs


def _check_images(rec: dict) -> list[str]:
    errs = []
    for k in ("images", "judge_images"):
        for label, path in rec.get(k, {}).items():
            if not Path(path).exists():
                errs.append(f"missing_{k}:{label}")
    return errs


def _check_answer(rec: dict) -> list[str]:
    if len(rec.get("messages", [])) < 2:
        return ["no_assistant_msg"]
    pred = _extract_answer(rec["messages"][1].get("content"))
    gt = rec["ground_truth"].get("ranking_best_to_worst", [])
    if not pred:
        return ["no_answer_tag"]
    if pred != gt:
        return [f"answer_vs_gt_mismatch:{pred}_vs_{gt}"]
    return []


def _check_soft(rec: dict) -> list[tuple[str, str]]:
    """Return list of (warn_code, matched_substring) tuples."""
    if len(rec.get("messages", [])) < 2:
        return []
    reasoning = _extract_reasoning(rec["messages"][1].get("content"))
    if not reasoning:
        return [("empty_reasoning", "")]
    warns: list[tuple[str, str]] = []
    for code, pat in (
        ("privileged:degrees",  _PAT_DEGREES),
        ("privileged:coords",   _PAT_COORDS),
        ("dynamic:comparison",  _PAT_COMPARISON),
        ("dynamic:aux_verb",    _PAT_AUX_DYN),
        ("dynamic:by_gerund",   _PAT_BY_GERUND),
    ):
        m = pat.search(reasoning)
        if m:
            start = max(0, m.start() - 60)
            end = min(len(reasoning), m.end() + 60)
            ctx = reasoning[start:end].replace("\n", " ")
            warns.append((code, f"…{ctx}…"))
    return warns


def _sample_records(records: list[dict], k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if k >= len(records):
        return list(records)
    return rng.sample(records, k)


def _print_sample(rec: dict) -> None:
    print(f"  id          = {rec.get('id')}")
    print(f"  n           = {rec.get('n')}")
    gt = rec.get("ground_truth", {})
    print(f"  mode        = {gt.get('ranking_mode')}")
    print(f"  ranking_gt  = {gt.get('ranking_best_to_worst')}")
    perts = gt.get("perturbations", [])
    print(f"  perturbs    = {[p.get('kind') for p in perts]}")
    for k in ("images", "judge_images"):
        img_dict = rec.get(k, {})
        for label, path in img_dict.items():
            p = Path(path)
            if p.exists():
                size_kb = p.stat().st_size // 1024
                print(f"  {k}.{label} = {path}  ({size_kb} KB) OK")
            else:
                print(f"  {k}.{label} = {path}  MISSING")
    msgs = rec.get("messages", [])
    if len(msgs) >= 2:
        reasoning = _extract_reasoning(msgs[1].get("content"))
        answer = _extract_answer(msgs[1].get("content"))
        print(f"  answer      = {answer}")
        print(f"  reasoning   = {reasoning}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--quarantine", type=Path, default=None,
                    help="Where to write hard-failed records. Defaults to "
                         "<path>.quarantine.jsonl")
    ap.add_argument("--strict", action="store_true",
                    help="Treat soft warnings as hard failures.")
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite <path> in-place with only the passing "
                         "records (a `.bak` is kept). Without --apply we "
                         "only report.")
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, print full content (id, image paths, "
                         "answer, reasoning) for N randomly-picked records "
                         "to manually eyeball quality.")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="Seed for the --sample random pick.")
    args = ap.parse_args()
    if not args.path.exists():
        print(f"[validate] missing {args.path}", file=sys.stderr)
        return 2

    quarantine = args.quarantine or args.path.with_suffix(args.path.suffix + ".quarantine.jsonl")

    total = 0
    hard_fail = 0
    soft_warn = 0
    err_counter: Counter = Counter()
    warn_counter: Counter = Counter()
    keep: list[str] = []
    drop: list[tuple[str, list[str]]] = []
    soft_examples: dict = defaultdict(list)
    all_records: list[dict] = []

    with args.path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                hard_fail += 1
                err_counter["parse_error"] += 1
                drop.append((line, [f"parse_error:{e!r}"]))
                continue
            all_records.append(rec)
            errs = _check_schema(rec)
            if not errs:
                errs = _check_images(rec) + _check_answer(rec)
            warns = _check_soft(rec)
            warn_codes = [w[0] for w in warns]
            if errs or (args.strict and warns):
                hard_fail += 1
                drop.append((line, errs + (warn_codes if args.strict else [])))
                for e in errs + (warn_codes if args.strict else []):
                    err_counter[e.split(":")[0]] += 1
            else:
                keep.append(line)
            if warns:
                soft_warn += 1
                for code, ctx in warns:
                    warn_counter[code] += 1
                    if len(soft_examples[code]) < 3:
                        soft_examples[code].append(f"id={rec.get('id')}: {ctx}")

    print(f"[validate] {args.path}")
    print(f"  total      = {total}")
    print(f"  pass       = {total - hard_fail}")
    print(f"  hard_fail  = {hard_fail}")
    if err_counter:
        print(f"  errors     = {dict(err_counter)}")
    print(f"  soft_warns = {soft_warn}  (records with at least one warning)")
    if warn_counter:
        print(f"  warn types = {dict(warn_counter)}")
        for w, ex in soft_examples.items():
            print(f"  -- {w} examples ({len(ex)} of {warn_counter[w]}):")
            for s in ex:
                print(f"      {s}")

    if args.sample > 0 and all_records:
        print(f"\n=== content sample ({args.sample} of {len(all_records)} records, seed={args.sample_seed}) ===\n")
        for rec in _sample_records(all_records, args.sample, args.sample_seed):
            _print_sample(rec)

    if args.apply and drop:
        bak = args.path.with_suffix(args.path.suffix + ".bak")
        bak.write_text(args.path.read_text())
        args.path.write_text("".join(keep))
        with quarantine.open("w") as fh:
            for line, errs in drop:
                rec = {"line": line.rstrip("\n"), "errors": errs}
                fh.write(json.dumps(rec) + "\n")
        print(f"  [apply] kept {len(keep)} → {args.path}")
        print(f"  [apply] quarantined {len(drop)} → {quarantine}")
        print(f"  [apply] backup → {bak}")

    return 0 if hard_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
