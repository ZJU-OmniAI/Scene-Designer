"""Per-(kind, object) severity scalar in [0, 1], built from the
LLM-scored object properties (`_obj_properties.json`).

The scalar is then used by `build_n4_multikind.py` (Type 1: same target,
different kinds) and `build_n4_multitarget.py` (Type 2: same kind,
different targets) to pick 4 candidates whose severity scores are
well-separated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Kinds for which the (kind, object) severity is essentially constant
# across objects (the kind itself drives the score, with mild modulation
# by object size / focal role).
_KIND_BASE_SCORE = {
    # Catastrophic — strong baseline, modulated by importance.
    "remove_object":       0.0,   # severity = importance(obj); pure object-driven
    # Functional violation — high baseline, modulated by object size.
    "move_blocking_door":  0.55,
    # Directional kinds — meaningful only on directional objects.
    "face_wall":           0.0,
    "rotate_180":          0.0,
    "rotate_45_offset":    0.0,
    # Layout/positional kinds — modulated by importance.
    "move_to_corner":      0.20,
    "move_off_wall":       0.10,
    "move_far_from_focal": 0.20,
}


def severity(kind: str, obj_properties: dict[str, float]) -> float:
    """Map (kind, object_properties) -> severity scalar in [0, 1].

    `obj_properties` is the LLM-scored dict for ONE object:
      `{"importance": float, "directionality": float, "rationale": ...}`

    Larger return value = larger expected visual / functional damage.
    Formulas are kind-specific so the same scalar means "this damage is
    discriminable from another scalar by ~0.2".
    """
    imp = float(obj_properties.get("importance", 0.0))
    dirn = float(obj_properties.get("directionality", 0.0))

    if kind == "remove_object":
        # Pure object-importance — removing a sofa is much worse than
        # removing a side table.
        return min(1.0, imp)

    if kind == "move_blocking_door":
        # Roughly uniform across objects, with a small boost for
        # important / large pieces. Anything in the doorway is bad.
        return min(1.0, _KIND_BASE_SCORE[kind] + 0.35 * imp)

    if kind in ("face_wall", "rotate_180"):
        # Only meaningful on directional objects; a circular side_table
        # rotated 180° is invisible. Weight by directionality AND
        # importance (rotating a tiny chair is less impactful than
        # rotating a sofa).
        return min(1.0, 0.4 * dirn + 0.5 * dirn * imp)

    if kind == "rotate_45_offset":
        # Subtle rotation — weight lower than face_wall / rotate_180.
        return min(1.0, 0.2 * dirn + 0.3 * dirn * imp)

    if kind == "move_to_corner":
        # Moving an important piece to a corner is bad; moving a
        # decorative lamp is barely noticed.
        return min(1.0, _KIND_BASE_SCORE[kind] + 0.6 * imp)

    if kind == "move_off_wall":
        # Wall pieces matter; floating lamps don't. Use importance as
        # proxy for "this piece belonged against a wall".
        return min(1.0, _KIND_BASE_SCORE[kind] + 0.6 * imp)

    if kind == "move_far_from_focal":
        # Moving the focal piece itself away from its group is
        # catastrophic; moving an auxiliary piece is mild.
        return min(1.0, _KIND_BASE_SCORE[kind] + 0.65 * imp)

    # Unknown kind — return a low default so it's filtered out by gap.
    return 0.0


def object_severity_for_kind(kind: str, obj_properties: dict[str, float]) -> float:
    """For Type-2 (same-kind, multi-target) selection: use ONLY the
    object's own attributes, full [0,1] range. The kind-base offset that
    `severity()` adds is a constant across the 4 candidates and doesn't
    help discriminate — it only narrows the available range.

    Mapping (one OBJECT-level scalar per kind):
      * remove_object / move_* (positional + blocking_door): `importance`
        — moving / removing an important piece is the variant that hurts;
        moving a decorative lamp is barely noticed.
      * rotate kinds (face_wall, rotate_180, rotate_45_offset):
        `directionality × importance` — only directional pieces are
        affected by yaw flips; weight by importance so a swivel-back
        sofa outranks a swivel-back side chair.
    """
    imp = float(obj_properties.get("importance", 0.0))
    dirn = float(obj_properties.get("directionality", 0.0))
    if kind in ("remove_object", "move_blocking_door",
                "move_to_corner", "move_off_wall", "move_far_from_focal"):
        return min(1.0, imp)
    if kind in ("face_wall", "rotate_180", "rotate_45_offset"):
        return min(1.0, dirn * imp)
    return 0.0


def load_properties(scene_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the cached LLM properties for a scene. Returns {} if missing
    (caller decides how to handle)."""
    p = scene_dir / "_obj_properties.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def severity_matrix(scene_dir: Path, *,
                    obj_ids: list[str] | None = None,
                    kinds: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Build a {obj_id: {kind: severity}} matrix using the cached
    properties. Returns {} if properties missing."""
    props = load_properties(scene_dir)
    if not props:
        return {}
    if obj_ids is None:
        obj_ids = list(props.keys())
    if kinds is None:
        kinds = list(_KIND_BASE_SCORE.keys())
    out = {}
    for obj_id in obj_ids:
        if obj_id not in props:
            continue
        row = {}
        for kind in kinds:
            row[kind] = severity(kind, props[obj_id])
        out[obj_id] = row
    return out


def pick_4_max_gap(scores: dict[str, float], *,
                   min_gap: float = 0.15) -> list[str] | None:
    """Given a {key: score} dict, pick 4 keys whose scores are most
    spread out AND every adjacent pair has gap ≥ `min_gap`. Returns
    None if no valid 4-tuple exists.

    Strategy: rank by score, then greedy from extremes:
      * pick min and max
      * pick the two intermediate keys whose scores are closest to
        the 1/3 and 2/3 points of the [min, max] range while still
        satisfying min_gap to chosen anchors.
    """
    if len(scores) < 4:
        return None
    # Round to 3 decimals to dodge float-precision edge cases where
    # 0.85 + 0.0 looks like 0.8500000000000001 and gap comparisons
    # silently fail (e.g. 0.14999... < 0.15).
    sorted_items = sorted(((k, round(v, 3)) for k, v in scores.items()),
                          key=lambda x: x[1])
    keys = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    if vals[-1] - vals[0] < min_gap * 3.0 - 1e-9:
        # Even the full span isn't enough to fit 3 gaps.
        return None
    # Try every choice of 4 indices that includes endpoints.
    # n is small (≤ ~15 furniture in a scene) so O(C(n,2)) is fine.
    best = None
    n = len(vals)
    eps = 1e-9
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            chosen = [vals[0], vals[i], vals[j], vals[-1]]
            gaps = [chosen[k + 1] - chosen[k] for k in range(3)]
            if min(gaps) < min_gap - eps:
                continue
            # Score = minimum gap (we want to maximize the bottleneck gap).
            score = min(gaps)
            if best is None or score > best[0]:
                best = (score, [keys[0], keys[i], keys[j], keys[-1]])
    if best is None:
        return None
    return best[1]


def enumerate_3_tuples_above_anchor(
    scores: dict[str, float], *,
    min_gap: float = 0.15,
    max_tuples: int = 3,
    min_distinct: int = 2,
) -> list[list[str]]:
    """Return up to `max_tuples` 3-tuples of keys whose severities
    are monotone-increasing with adjacent gap ≥ `min_gap` AND the
    smallest severity ≥ `min_gap` (i.e. enough gap above the implicit
    anchor severity 0). Sorted by bottleneck gap descending. Each
    successive tuple must differ from all previously picked by
    ≥ `min_distinct` elements.

    This is the right primitive for Type-2 (multitarget) records:
    A is the anchor scene (no perturbation, severity = 0) and we
    need 3 distinct targets at increasing severities for B/C/D.
    The previous `enumerate_4_tuples` returned 4 objects and the
    generator silently discarded the lowest-severity one — that
    discarded "object 0" never appeared in the rendered scenes, so
    different choices of object 0 produced duplicate records sharing
    the same B/C/D triple. Returning 3-tuples directly avoids that."""
    from itertools import combinations
    if len(scores) < 3:
        return []
    sorted_items = sorted(((k, round(v, 3)) for k, v in scores.items()),
                          key=lambda x: x[1])
    keys = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    eps = 1e-9
    if vals[-1] < min_gap * 3 - eps:
        # Even the largest severity can't fit anchor (0) + 3 gaps.
        return []
    n = len(vals)
    candidates: list[tuple[float, list[str]]] = []
    for combo in combinations(range(n), 3):
        chosen = [vals[i] for i in combo]
        gaps = [chosen[0]] + [chosen[k + 1] - chosen[k] for k in range(2)]
        bottleneck = min(gaps)
        if bottleneck < min_gap - eps:
            continue
        candidates.append((bottleneck, [keys[i] for i in combo]))
    candidates.sort(key=lambda x: -x[0])
    picked: list[list[str]] = []
    for _, tup in candidates:
        if len(picked) >= max_tuples:
            break
        too_similar = False
        for prev in picked:
            overlap = len(set(tup) & set(prev))
            if 3 - overlap < min_distinct:
                too_similar = True
                break
        if too_similar:
            continue
        picked.append(tup)
    return picked


def enumerate_4_tuples(scores: dict[str, float], *,
                       min_gap: float = 0.15,
                       max_tuples: int = 3,
                       min_distinct: int = 2) -> list[list[str]]:
    """Return up to `max_tuples` 4-tuples of keys whose severities are
    monotone with adjacent gap ≥ `min_gap`, sorted by bottleneck gap
    descending. Each successive tuple must differ from all previously
    picked by ≥ `min_distinct` elements (so we get diverse coverage
    of the available 4-spread combinations, not near-duplicates).

    Used by `generate_multitarget` to extract multiple records per
    (scene, kind) when the scene has many importance-spread objects."""
    from itertools import combinations
    if len(scores) < 4:
        return []
    sorted_items = sorted(((k, round(v, 3)) for k, v in scores.items()),
                          key=lambda x: x[1])
    keys = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    eps = 1e-9
    if vals[-1] - vals[0] < 3 * min_gap - eps:
        return []
    n = len(vals)
    candidates: list[tuple[float, list[str]]] = []
    for combo in combinations(range(n), 4):
        chosen = [vals[i] for i in combo]
        gaps = [chosen[k + 1] - chosen[k] for k in range(3)]
        bottleneck = min(gaps)
        if bottleneck < min_gap - eps:
            continue
        candidates.append((bottleneck, [keys[i] for i in combo]))
    candidates.sort(key=lambda x: -x[0])
    picked: list[list[str]] = []
    for _, tup in candidates:
        if len(picked) >= max_tuples:
            break
        too_similar = False
        for prev in picked:
            overlap = len(set(tup) & set(prev))
            if 4 - overlap < min_distinct:
                too_similar = True
                break
        if too_similar:
            continue
        picked.append(tup)
    return picked
