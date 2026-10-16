"""Generate assistant reasoning + answer messages for n4 records.

Pipeline per record:
  1. Build a rule-based STATIC draft from `perturbations_by_label` + GT
     ranking. For psw (parameter sweep) we use per-kind 4-bucket templates
     keyed on severity-rank; for cum (cumulative) we list the perturbations
     that each scene exhibits.
  2. Send the draft to Gemini-2.5-flash via `polish_ranking_reasoning`
     (re-used from src_critic.judge) — that function rewrites into a
     concise 80-160 word paragraph that strips action-verbs, comparative
     phrases, and numeric values.
  3. Wrap polished text in `<reasoning>...</reasoning>\\n<answer>BACD</answer>`
     and replace messages[1] (or append) in the SFT record.

Parallelized via threads (network-bound). Reads either sft_n4_train or
sft_n4_test, writes polished file alongside.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from .judge import make_client, polish_ranking_reasoning


# ---------- per-kind static templates for psw ----------

_PSW_TEMPLATES = {
    "rotate_45_offset": {
        "anchor":   "the {obj} faces its natural orientation, in line with the surrounding room geometry",
        "slight":   "the {obj} is angled slightly away from its expected orientation",
        "moderate": "the {obj} sits at a noticeably awkward angle, off-axis from the rest of the room",
        "severe":   "the {obj} faces sideways, sharply out of alignment with the room layout",
    },
    "move_off_wall": {
        "anchor":   "the {obj} sits flush against the wall as intended",
        "slight":   "the {obj} is pulled slightly away from the wall, leaving a small gap behind it",
        "moderate": "the {obj} stands clearly off the wall with a wide gap behind it",
        "severe":   "the {obj} sits far from any wall, floating in the middle of the room",
    },
    "move_far_from_focal": {
        "anchor":   "the {obj} sits within its functional grouping, anchoring the activity zone",
        "slight":   "the {obj} sits slightly removed from its functional grouping",
        "moderate": "the {obj} is set well apart from its functional group, weakening the arrangement",
        "severe":   "the {obj} is isolated far from its functional group, leaving the zone disjointed",
    },
    "move_to_corner": {
        "anchor":   "the {obj} occupies a central, functional position in the room",
        "slight":   "the {obj} drifts toward the room corner, slightly off-center",
        "moderate": "the {obj} sits well off to one corner, away from the main usable area",
        "severe":   "the {obj} is pushed into the room corner, far from any functional zone",
    },
}


# ---------- per-kind STATIC phrasing for cum (one perturbation kind) ----------
#
# For cum we list, per scene, the static state that each applied
# perturbation produces. A scene with K perturbations gets K clauses
# joined with semicolons. Anchor scene (0 perts) gets a generic positive
# clause.
_CUM_STATIC = {
    "move_to_corner":      "the {obj} sits in a far corner of the room, removed from the main usable area",
    "move_blocking_door":  "the {obj} sits in the doorway clearance band, obstructing passage",
    "move_off_wall":       "the {obj} is set apart from any wall, floating away from the room perimeter",
    "move_far_from_focal": "the {obj} is isolated far from its functional grouping",
    "rotate_180":          "the {obj} faces backward, with its front side turned away from the room",
    "rotate_45_offset":    "the {obj} sits at a sharp off-axis angle, out of alignment with the room",
    "face_wall":           "the {obj} faces directly into a nearby wall",
    "swap_two":            "the {obj} occupies a position better suited to a different piece",
    "remove_object":       "the {obj} is absent from the scene",
}

# Fallback generic clause when an unrecognized kind appears.
_GENERIC_CLAUSE = "the {obj} is placed in an awkward, suboptimal position"


def _obj_pretty(obj_id: str) -> str:
    """Strip trailing `_0` / `_1` index and convert underscores to spaces."""
    parts = obj_id.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts) if parts else obj_id


def _yaw_diff_abs(a: float, b: float) -> float:
    """Smallest absolute angular difference, in radians, in [0, π]."""
    d = abs(a - b) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def _psw_severity(p) -> float:
    """Severity scalar for psw: yaw-delta for rotate kinds, euclidean
    position delta for move kinds."""
    kind = p["kind"]
    np_ = p.get("new_pose") or [0.0, 0.0, 0.0]
    op_ = p.get("original_pose") or [0.0, 0.0, 0.0]
    if kind == "rotate_45_offset":
        return _yaw_diff_abs(float(np_[2]), float(op_[2]))
    dx = float(np_[0]) - float(op_[0])
    dy = float(np_[1]) - float(op_[1])
    return math.sqrt(dx * dx + dy * dy)


def _psw_draft(rec: dict) -> str:
    """Build a rule-based static draft for a psw record. Ranking is
    deterministic A > B > C > D by construction."""
    gt = rec["ground_truth"]
    perts_by_label = gt["perturbations_by_label"]
    ranking = gt["ranking_best_to_worst"]
    # First non-empty label's perturbation tells us kind + target.
    a_perts = perts_by_label.get("A", [])
    sample_p = None
    for lab in ("B", "C", "D", "A"):
        ps = perts_by_label.get(lab) or []
        if ps:
            sample_p = ps[0]
            break
    if sample_p is None:
        return ""
    kind = sample_p["kind"]
    obj_pretty = _obj_pretty(sample_p["object_id"])

    # Severity per label. A has no perturbation → severity 0 (anchor).
    sev = {}
    for lab in ("A", "B", "C", "D"):
        ps = perts_by_label.get(lab) or []
        sev[lab] = 0.0 if not ps else _psw_severity(ps[0])

    # Bucket: anchor (sev==0), then ascending → slight / moderate / severe.
    # Tie-break on severity uses the GT ranking order so the static
    # description aligns with the labelled ranking.
    non_anchor = sorted(
        [(lab, sev[lab]) for lab in ("A", "B", "C", "D") if sev[lab] > 0],
        key=lambda x: x[1],
    )
    qual = {}
    for lab in ("A", "B", "C", "D"):
        if sev[lab] == 0:
            qual[lab] = "anchor"
    if len(non_anchor) >= 1:
        qual[non_anchor[0][0]] = "slight"
    if len(non_anchor) >= 2:
        qual[non_anchor[1][0]] = "moderate"
    if len(non_anchor) >= 3:
        qual[non_anchor[2][0]] = "severe"
    # Edge: if all 4 have sev>0 (shouldn't happen — A is always anchor),
    # the first becomes 'slight' and so on without an explicit anchor.

    tmpl = _PSW_TEMPLATES.get(kind)
    lines = []
    for lab in ("A", "B", "C", "D"):
        q = qual.get(lab, "moderate")
        if tmpl is None:
            line = f"Scene {lab}: " + _GENERIC_CLAUSE.format(obj=obj_pretty)
        else:
            line = f"Scene {lab}: " + tmpl[q].format(obj=obj_pretty)
        lines.append(line)
    lines.append("Ranking from best to worst: " + " > ".join(ranking) + ".")
    return "\n".join(lines)


def _cum_draft(rec: dict) -> str:
    """Build a rule-based static draft for a cum record. Each scene's
    state is the union of static-clauses for its perturbations; the
    anchor scene gets a generic positive line."""
    gt = rec["ground_truth"]
    perts_by_label = gt["perturbations_by_label"]
    ranking = gt["ranking_best_to_worst"]
    lines = []
    for lab in ("A", "B", "C", "D"):
        perts = perts_by_label.get(lab) or []
        if not perts:
            lines.append(f"Scene {lab}: the layout reads as a coherent, "
                         f"functional arrangement with each piece in a "
                         f"sensible position")
            continue
        clauses = []
        for p in perts:
            kind = p["kind"]
            obj_pretty = _obj_pretty(p["object_id"])
            tmpl = _CUM_STATIC.get(kind, _GENERIC_CLAUSE)
            clauses.append(tmpl.format(obj=obj_pretty))
        lines.append(f"Scene {lab}: " + "; ".join(clauses))
    lines.append("Ranking from best to worst: " + " > ".join(ranking) + ".")
    return "\n".join(lines)


def _perturbation_summary_for_polish(rec: dict) -> str:
    """Compact reference summary the polish prompt is fed (FOR REFERENCE
    ONLY — not echoed in output). One bullet per (scene, kind, target).
    Mirrors the style used by n2 polish calls."""
    perts_by_label = rec["ground_truth"]["perturbations_by_label"]
    bullets = []
    for lab in ("A", "B", "C", "D"):
        perts = perts_by_label.get(lab) or []
        if not perts:
            bullets.append(f"- Scene {lab}: original (no perturbation)")
            continue
        for p in perts:
            kind = p["kind"]
            obj = p["object_id"]
            bullets.append(f"- Scene {lab}: {kind} on {obj}")
    return "\n".join(bullets)


def _build_assistant_msg(polished: str, ranking: list[str]) -> str:
    answer = "".join(ranking)
    return f"<reasoning>\n{polished}\n</reasoning>\n<answer>{answer}</answer>"


def _process_one(rec: dict, client) -> dict | None:
    """Return the patched record with `messages[1]` set to the polished
    assistant message. Returns None on failure (caller skips/keeps raw)."""
    is_psw = rec["id"].endswith("_n4_psw")
    is_cum = rec["id"].endswith("_n4_cum")
    is_multikind = rec["id"].endswith("_n4_multikind")
    is_multitarget = rec["id"].endswith("_n4_multitarget")
    if not (is_psw or is_cum or is_multikind or is_multitarget):
        return None
    # multikind/multitarget records mirror cum's structure (each scene
    # has a list of applied perturbations; anchor is empty), so they
    # reuse the cum-style draft builder.
    draft = _psw_draft(rec) if is_psw else _cum_draft(rec)
    if not draft:
        return None
    ranking = rec["ground_truth"]["ranking_best_to_worst"]
    summary = _perturbation_summary_for_polish(rec)
    polished = polish_ranking_reasoning(
        draft, ranking=ranking, perturbation_summary=summary,
        client=client, record_id=rec["id"], max_attempts=4,
        # cum records list up to 3 perturbations × 4 scenes, so the
        # natural-language version needs headroom above the polish
        # function's default (Gemini also spends ~500-1500 tokens on
        # invisible internal reasoning).
        max_tokens=4000,
    )
    if not polished or polished.strip() == draft.strip():
        # Polish failed silently (returned raw). We still emit something
        # — but it will read awkwardly. Better than dropping the record.
        polished = draft
    asst = {"role": "assistant", "content": _build_assistant_msg(polished, ranking)}
    msgs = list(rec.get("messages", []))
    if len(msgs) >= 2 and msgs[1].get("role") == "assistant":
        msgs[1] = asst
    elif len(msgs) >= 1:
        msgs.append(asst)
    else:
        return None
    out = dict(rec)
    out["messages"] = msgs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True,
                    help="Input n4 jsonl (no assistant messages).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output jsonl with polished reasoning.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-records", type=int, default=0,
                    help="0 = all. For smoke-test runs.")
    args = ap.parse_args()

    records = []
    with args.inp.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.max_records:
        records = records[: args.max_records]
    print(f"loaded {len(records)} records from {args.inp}", file=sys.stderr)

    client = make_client()
    write_lock = Lock()
    fh = args.out.open("w", encoding="utf-8")
    done = 0
    failed = 0
    t0 = time.time()

    def worker(rec):
        try:
            return _process_one(rec, client)
        except Exception as e:
            print(f"[err] {rec.get('id')}: {e!r}", file=sys.stderr)
            traceback.print_exc()
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, r) for r in records]
        for fut in as_completed(futs):
            out_rec = fut.result()
            with write_lock:
                if out_rec is None:
                    failed += 1
                else:
                    fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    fh.flush()
                done += 1
                if done % 25 == 0 or done == len(records):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1)
                    print(f"[{done}/{len(records)}] "
                          f"failed={failed} elapsed={elapsed:.0f}s "
                          f"rate={rate:.1f}/s", file=sys.stderr)
    fh.close()
    print(f"\nwrote {args.out} ({done - failed} ok, {failed} failed)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
