"""Generate a 'bad' (or 'less good') variant of a scene by perturbing one object.

Each perturbation is one of three classes:

  * **hard_collision**  — the new pose physically violates the room
    (out-of-bounds, overlap, blocks a door). The original is unambiguously
    better.
  * **directional_bad** — the new pose preserves physical validity but breaks
    the object's functional orientation (e.g. front faces a wall). Original
    better.
  * **layout_soft**     — the new pose is also physically valid but is a less
    coherent layout (storage off-wall, sofa pulled away from coffee table,
    object removed entirely). The original is *probably* better; we still
    mark `original` as the GT but label these pairs as "soft".

The class label is recorded in `Perturbation.gt_strength` so SFT consumers
can up-/down-weight the supervision signal.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

from .scene_io import Furniture, Scene


# Furniture whose 'front' (yaw=0 → +y) direction is functionally meaningful
# enough that we PERMIT rotation perturbations on it. This is now an
# OPT-OUT list: any piece NOT matching `NON_ROTATABLE_TOKENS` (see below)
# is considered rotatable. Conversely, `DIRECTIONAL_TOKENS` is kept for
# code that still uses it as an include-check, populated with the same
# items as before; new generators should prefer the NON_ROTATABLE check.
DIRECTIONAL_TOKENS = (
    # seating / sleeping
    "sofa", "couch", "armchair", "chair", "stool", "bench", "bed",
    # display & media
    "tv", "media", "console", "monitor", "speaker",
    # work surfaces
    "desk", "vanity", "workbench",
    # storage with door/drawer
    "nightstand", "dresser", "wardrobe", "bookshelf", "bookcase",
    "cabinet", "credenza", "shelving", "shelf", "sideboard", "rack",
    # kitchen / appliances (control face / door clearly oriented)
    "fridge", "refrigerator", "freezer", "stove", "oven", "range",
    "microwave", "dishwasher", "washer", "washing", "dryer", "sink",
    "island",
    # bathroom fixtures (front faces user)
    "toilet", "shower",
)


# Pieces that should NEVER be the target of a rotation perturbation
# (`face_wall` / `rotate_180` / `rotate_45_offset`). Either:
#   - decorative floor coverings (rotating a rug is invisible),
#   - rotationally near-symmetric pieces (round tables, ottomans,
#     planters, cylindrical lamps),
#   - small accessories where rotation isn't meaningful (vases, bowls,
#     decorative bookcase contents).
# Movement and removal perturbations ARE still allowed on these.
NON_ROTATABLE_TOKENS = (
    # decorative floor coverings
    "rug", "carpet", "doormat", "mat", "runner",
    # symmetric / no-front pieces
    "planter", "plant", "vase", "bowl", "sculpture",
    "ottoman", "pouf", "footstool", "bean_bag",
    "floor_lamp", "table_lamp", "pendant_lamp", "ceiling_lamp",
    # tables (typically square / round, no fixed front)
    "coffee_table", "dining_table", "side_table",
    "end_table", "accent_table",
    # decorative containers
    "toy_chest", "trunk", "basket",
    # art (no canonical yaw)
    "easel", "artwork", "art_work",
)


def _is_non_rotatable(name: str) -> bool:
    """True if the piece name matches any NON_ROTATABLE_TOKENS stem,
    i.e. rotation perturbations should reject it as a target."""
    n = (name or "").lower()
    return any(t in n for t in NON_ROTATABLE_TOKENS)
# Furniture that should sit against / near a wall.
WALL_HUGGING_TOKENS = (
    "bookshelf", "bookcase", "media_console", "tv_stand", "wardrobe",
    "dresser", "cabinet", "bed", "nightstand", "desk", "sideboard",
    "shelving", "credenza", "rack", "vanity", "shelf",
)
# Common focal pairs (token A pulled toward token B).
# Bidirectional: `_focal_partner` checks both directions.
FOCAL_PAIRS = [
    # Living room / seating clusters
    ("sofa", "coffee_table"),
    ("couch", "coffee_table"),
    ("armchair", "coffee_table"),
    ("accent_armchair", "coffee_table"),
    ("ottoman", "sofa"),
    ("ottoman", "armchair"),
    ("side_table", "sofa"),
    ("side_table", "armchair"),
    ("media_console", "sofa"),
    ("media_console", "couch"),
    ("tv_console", "sofa"),
    ("tv_stand", "sofa"),
    ("floor_lamp", "sofa"),
    ("floor_lamp", "armchair"),
    # Dining
    ("dining_chair", "dining_table"),
    ("dining_table", "kitchen_island"),
    # Kitchen / bar
    ("bar_stool", "kitchen_island"),
    ("bar_stool", "bar"),
    # Bedroom
    ("bed", "nightstand"),
    ("bed", "dresser"),
    ("bed", "wardrobe"),
    ("storage_bench", "bed"),
    ("bedroom_bench", "bed"),
    ("valet_stand", "bed"),
    ("vanity", "bed"),
    # Bathroom
    ("toilet", "vanity"),
    # Nursery
    ("nursery_glider_armchair", "changing_table"),
    # Office / study
    ("office_chair", "desk"),
    ("study_chair", "desk"),
    ("study_chair", "study_desk"),
    ("office_chair", "study_desk"),
    ("bookcase", "desk"),
    ("bookshelf", "desk"),
    # Music
    ("piano_bench", "piano"),
    ("piano_bench", "digital_piano"),
    ("piano_bench", "upright_piano"),
    ("piano_bench", "grand_piano"),
    ("music_stand", "piano"),
    ("music_stand", "digital_piano"),
    # Meditation / reading
    ("meditation_cushion_set", "meditation_bench"),
    ("yoga_mat", "yoga_props"),
    ("yoga_props", "yoga_mat"),
    # Reading nook
    ("bookcase", "armchair"),
    ("bookshelf", "armchair"),
    # Art / hobby
    ("art_work_table", "storage_cabinet"),
]


PerturbKind = Literal[
    "face_wall",
    "rotate_180",
    "rotate_45_offset",
    "move_off_wall",
    "move_far_from_focal",
    "move_to_corner",
    "move_into_wall",
    "move_blocking_door",
    "move_overlap",
    "swap_two",
    "remove_object",
]

GtStrength = Literal["hard", "directional", "soft"]


@dataclass
class Perturbation:
    object_id: str
    kind: PerturbKind
    description: str
    original_pose: tuple[float, float, float]
    new_pose: tuple[float, float, float]
    reason_bad: str
    gt_strength: GtStrength
    secondary_change: dict | None = None
    # `remove_object` doesn't have a meaningful new_pose — mark it explicitly.
    is_removal: bool = False


def _has_token(name: str, tokens: tuple[str, ...] | list[str]) -> bool:
    n = name.lower()
    return any(t in n for t in tokens)


def _aabb_overlap(a, b, pad: float = 0.0) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                or a[3] + pad < b[1] or b[3] + pad < a[1])


def _door_clearance_band(scene: Scene) -> list[tuple[float, float, float, float]]:
    bands = []
    room = scene.room
    for d in room.doors:
        direction = {"A": "north", "B": "south", "C": "east", "D": "west"}.get(
            d.get("boundary_label", "A"), "north"
        )
        pos = float(d.get("position_exact", 0.0))
        w = float(d.get("width", 0.9))
        band_depth = 0.6
        if direction in ("north", "south"):
            x0 = room.x_min + pos - w / 2.0
            x1 = room.x_min + pos + w / 2.0
            y = room.y_max if direction == "north" else room.y_min
            if direction == "north":
                bands.append((x0, y - band_depth, x1, y))
            else:
                bands.append((x0, y, x1, y + band_depth))
        else:
            y0 = room.y_min + pos - w / 2.0
            y1 = room.y_min + pos + w / 2.0
            x = room.x_max if direction == "east" else room.x_min
            if direction == "east":
                bands.append((x - band_depth, y0, x, y1))
            else:
                bands.append((x, y0, x + band_depth, y1))
    return bands


def _is_out_of_room(f: Furniture, scene: Scene, slack: float = 0.05) -> bool:
    x0, y0, x1, y1 = f.world_aabb()
    r = scene.room
    return x0 < r.x_min - slack or y0 < r.y_min - slack or x1 > r.x_max + slack or y1 > r.y_max + slack


_FLOOR_PIECE_TOKENS = ("rug", "carpet", "doormat", "mat", "runner")


def _is_decorative_floor(f: Furniture) -> bool:
    """Returns True for pieces that are decorative floor coverings (rugs,
    carpets, mats, runners, doormats). Other furniture is naturally
    PLACED ON these — a bench resting on a rug, a coffee table over
    a carpet — so their footprint must NOT count as a 'collision' for
    any of the overlap-driven decisions (upgrade-to-tier-1, clean-
    placement search, swap_two upgrade).

    Matches whole-word tokens after splitting the (lower-cased) `name`
    on `_`, `-`, or whitespace, so e.g. `rug_0`, `area_rug`, `bath_mat`,
    `floor-mat` all match, but `format` (no rug/carpet/etc. token)
    doesn't."""
    import re
    parts = re.split(r"[_\s\-]+", (f.name or "").lower())
    return bool(set(parts) & set(_FLOOR_PIECE_TOKENS))


def _overlaps_any(f: Furniture, others: list[Furniture], pad: float = 0.0) -> bool:
    """Aabb-overlap test against `others`. Skips:
      * `f` itself (object-id match), and
      * decorative floor pieces (rugs / carpets / mats / runners / doormats),
        since other furniture is meant to sit ON them.
    The skip is global because every caller's semantics is "would these
    two real pieces collide?" — rugs never qualify."""
    a = f.world_aabb()
    for g in others:
        if g.object_id == f.object_id:
            continue
        if _is_decorative_floor(g):
            continue
        if _aabb_overlap(a, g.world_aabb(), pad=pad):
            return True
    return False


def _aabb_area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _aabb_intersection_area(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def physical_violations(
    scene: Scene,
    *,
    room_slack: float = 0.05,
    overlap_frac: float = 0.35,
    door_cover_frac: float = 0.30,
    check_doors: bool = False,
) -> list[tuple[str, str]]:
    """Geometric physical-legality check for a FINISHED scene.

    Returns a list of ``(kind, object_id)`` violations; empty ⇒ physically
    legal. Detected from geometry alone (no model):

      * ``out_of_room`` — a piece's AABB pokes outside the room interior
        (beyond ``room_slack`` m).
      * ``overlap``     — two non-decorative pieces **mutually deeply**
        overlap: the intersection covers ≥ ``overlap_frac`` of BOTH
        footprints. The "both" requirement excludes legitimate relationships
        (a chair tucked under a desk overlaps a large fraction of the *chair*
        but only a small fraction of the *desk*, so it is NOT flagged), while
        two comparably-sized pieces crashing through each other are.

    Door-clearance blocking is NOT treated as a physical violation (it is a
    soft/functional issue the critic handles, not a hard geometric illegality)
    and is OFF by default. The ``blocks_door`` check can still be enabled with
    ``check_doors=True`` if a caller wants it.

    Decorative floor pieces (rugs / carpets / mats) are excluded as collision
    targets. Used by the GRPO reward path: any non-empty result ⇒ the rollout
    is physically illegal and gets the maximum negative reward.
    """
    viols: list[tuple[str, str]] = []
    fs = [f for f in scene.furniture if not _is_decorative_floor(f)]

    for f in fs:
        if _is_out_of_room(f, scene, slack=room_slack):
            viols.append(("out_of_room", f.object_id))

    for i in range(len(fs)):
        ai = fs[i].world_aabb()
        area_i = _aabb_area(ai)
        for j in range(i + 1, len(fs)):
            aj = fs[j].world_aabb()
            inter = _aabb_intersection_area(ai, aj)
            if inter <= 0.0:
                continue
            area_j = _aabb_area(aj)
            fi = inter / area_i if area_i > 0 else 0.0
            fj = inter / area_j if area_j > 0 else 0.0
            if fi >= overlap_frac and fj >= overlap_frac:
                viols.append(("overlap", f"{fs[i].object_id}|{fs[j].object_id}"))

    if check_doors:
        for b in _door_clearance_band(scene):
            band_area = _aabb_area(b)
            if band_area <= 0:
                continue
            for f in fs:
                if _aabb_intersection_area(f.world_aabb(), b) / band_area >= door_cover_frac:
                    viols.append(("blocks_door", f.object_id))

    return viols


def is_physically_legal(scene: Scene, **kw) -> bool:
    """Convenience boolean wrapper around :func:`physical_violations`."""
    return not physical_violations(scene, **kw)


def _apply(scene: Scene, target_id: str, tx: float, ty: float, yaw_deg: float):
    s = scene.clone()
    f = s.by_id(target_id)
    assert f is not None
    f.tx, f.ty, f.yaw_deg = tx, ty, yaw_deg
    return s, f


def _nearest_wall(f: Furniture, room) -> str:
    dists = {
        "north": room.y_max - f.ty,
        "south": f.ty - room.y_min,
        "east": room.x_max - f.tx,
        "west": f.tx - room.x_min,
    }
    return min(dists, key=dists.get)


def _yaw_facing(direction: str) -> float:
    """yaw (deg) such that local +y points into the named wall."""
    return {"north": 0.0, "south": 180.0, "east": -90.0, "west": 90.0}[direction]


# Kinds where the perturbation IS a yaw change. Used downstream to:
#   - decide red-arrow / FRONT-BACK marker rendering on judge images
#     (only rotation kinds get a red arrow + FRONT/BACK boxes)
#   - rare-sample rotation on visually-symmetric targets via _is_visually_symmetric
ROTATION_KINDS = (
    "rotate_180", "face_wall", "rotate_45_offset",
    "rotate_out_of_room", "rotate_into_collision",
)


# Minimum Euclidean displacement (metres) for any movement perturbation
# to count as a valid perturbation. Below this, the judge can't visually
# distinguish the perturbed scene from the original, so the record is
# unhelpful as training signal. Applied centrally in `_emit_with_upgrade`
# so every move generator inherits the rule.
_MIN_MOVE_DELTA = 0.5


def _emit_with_upgrade(
    s2, f2, target, original_pose, new_pose,
    *,
    soft_kind: str,
    soft_description: str,
    soft_reason_bad: str,
    soft_strength: str,
    is_rotation: bool = False,
    secondary_change: dict | None = None,
    pre_scene=None,
):
    """Emit as the soft kind if the post-op pose is clean; otherwise
    UPGRADE to a tier-1 hard kind reflecting the incidental violation
    (overlap → `move_overlap`/`rotate_into_collision`; OOB →
    `move_into_wall`/`rotate_out_of_room`). The user directive is to
    take the more-severe tier-1 outcome rather than skip-and-retry,
    so soft generators don't waste budget rejecting double-violation
    candidates.

    For ROTATIONS, when `pre_scene` is supplied, we compare against the
    pre-perturbation state and SKIP the upgrade if the violation already
    existed in the original layout. (E.g. a bench resting on a rug that
    the engine treats as furniture would otherwise be wrongly tagged as
    "rotate_into_collision" the moment we rotate it.)

    For MOVEMENTS, we also reject perturbations whose Euclidean
    displacement is below `_MIN_MOVE_DELTA`. Tiny moves (< 0.5 m) are
    visually ambiguous at the rendered resolution and the judge often
    can't tell the perturbed scene from the original."""
    if not is_rotation:
        dx = new_pose[0] - original_pose[0]
        dy = new_pose[1] - original_pose[1]
        if (dx * dx + dy * dy) < _MIN_MOVE_DELTA * _MIN_MOVE_DELTA:
            return None
    oob = _is_out_of_room(f2, s2)
    overlap = _overlaps_any(f2, s2.furniture)
    if is_rotation and pre_scene is not None:
        # PER-PARTNER mask: only ignore overlaps that ALREADY existed
        # between the SAME pair of pieces in the pre-rotation scene.
        # A new overlap with a different partner (introduced by the
        # rotation widening the AABB) is still a real violation → keep
        # overlap=True so the caller drops the record.
        if overlap:
            pre_ids = set()
            tgt_aabb = target.world_aabb()
            for g in pre_scene.furniture:
                if g.object_id == target.object_id:
                    continue
                if _is_decorative_floor(g):
                    continue
                if _aabb_overlap(tgt_aabb, g.world_aabb()):
                    pre_ids.add(g.object_id)
            post_ids = set()
            f2_aabb = f2.world_aabb()
            for g in s2.furniture:
                if g.object_id == f2.object_id:
                    continue
                if _is_decorative_floor(g):
                    continue
                if _aabb_overlap(f2_aabb, g.world_aabb()):
                    post_ids.add(g.object_id)
            new_overlaps = post_ids - pre_ids
            if not new_overlaps:
                overlap = False
        if oob and _is_out_of_room(target, pre_scene):
            oob = False
    # Movements can also incidentally land in a door clearance band — that
    # is a tier-1 violation (move_blocking_door). Rotations can't move
    # the centre, so this check is movement-only.
    in_door_band = False
    if not is_rotation:
        bands = _door_clearance_band(s2)
        a = f2.world_aabb()
        for band in bands:
            if _aabb_overlap(a, band):
                in_door_band = True
                break
    # NEW POLICY: physical violations (OOB, overlap) → DROP, do NOT upgrade.
    # Physics is handled by the simulator (geometric check at gen time), so
    # incidentally-broken perturbations are unhelpful — they conflate the
    # intended kind with the violation. Door-band remains an upgrade target
    # because it is a "circulation" issue, not a pure-physics collision.
    if oob or overlap:
        return None
    if in_door_band:
        return s2, Perturbation(
            object_id=target.object_id,
            kind="move_blocking_door",
            description=(
                f"move {target.object_id} into a position that blocks a "
                "door's clearance band"
            ),
            original_pose=original_pose,
            new_pose=new_pose,
            reason_bad=(
                f"{target.object_id} now sits inside the door clearance "
                "band, blocking entry."
            ),
            gt_strength="hard",
            secondary_change=secondary_change,
        )
    return s2, Perturbation(
        object_id=target.object_id,
        kind=soft_kind,
        description=soft_description,
        original_pose=original_pose,
        new_pose=new_pose,
        reason_bad=soft_reason_bad,
        gt_strength=soft_strength,
        secondary_change=secondary_change,
    )


def _is_visually_symmetric(target, ar_threshold: float = 0.85,
                           max_dim_threshold: float = 0.7) -> bool:
    """Returns True when a yaw change is hard to see on this target. NOT a
    hard reject — used by sibling/cumulative samplers to RARE-sample
    rotation kinds on such targets (e.g. 70% skip and retry) rather than
    ban them outright. The training signal is still the FRONT-arrow
    direction overlay; we just don't want the dataset dominated by tiny
    cylindrical pieces whose silhouette never changes."""
    w = getattr(target, "local_width", None)
    d = getattr(target, "local_depth", None)
    if not w or not d or w <= 0 or d <= 0:
        return True
    if max(w, d) < max_dim_threshold:
        return True
    return (min(w, d) / max(w, d)) > ar_threshold


# ---------------------------------------------------------------------------
# Individual generators. Each returns (Scene, Perturbation) on success or None.
# ---------------------------------------------------------------------------

def _gen_face_wall(scene, target, rng) -> tuple | None:
    if _is_non_rotatable(target.name):
        return None
    wall = _nearest_wall(target, scene.room)
    new_yaw = _yaw_facing(wall)
    if abs(((new_yaw - target.yaw_deg + 180) % 360) - 180) < 5:
        return None
    s2, f2 = _apply(scene, target.object_id, target.tx, target.ty, new_yaw)
    return _emit_with_upgrade(
        s2, f2, target,
        original_pose=(target.tx, target.ty, target.yaw_deg),
        new_pose=(target.tx, target.ty, new_yaw),
        soft_kind="face_wall",
        soft_description=f"rotate {target.object_id} so its front faces the {wall} wall",
        soft_reason_bad=(
            f"{target.object_id}'s functional front (back of sofa, screen of TV, "
            f"drawers, shelves) is rotated to point at the {wall} wall, which "
            "breaks normal use."
        ),
        soft_strength="directional",
        is_rotation=True,
        pre_scene=scene,
    )


def _gen_rotate_180(scene, target, rng) -> tuple | None:
    if _is_non_rotatable(target.name):
        return None
    new_yaw = (target.yaw_deg + 180.0) % 360.0
    s2, f2 = _apply(scene, target.object_id, target.tx, target.ty, new_yaw)
    return _emit_with_upgrade(
        s2, f2, target,
        original_pose=(target.tx, target.ty, target.yaw_deg),
        new_pose=(target.tx, target.ty, new_yaw),
        soft_kind="rotate_180",
        soft_description=f"rotate {target.object_id} by 180°",
        soft_reason_bad=(
            f"{target.object_id} is flipped 180°, so its functional front "
            "is reversed (e.g. backrest toward the room, shelves toward a wall)."
        ),
        soft_strength="directional",
        is_rotation=True,
        pre_scene=scene,
    )


def _gen_rotate_45_offset(scene, target, rng) -> tuple | None:
    """Off-axis rotation generator. Two outcome classes:
      * In-bounds rotation that doesn't clip walls → emit as a regular
        soft `rotate_45_offset` (subjective awkward angle; tier 4).
      * Rotation that pushes the footprint past a room wall → emit as a
        HARD `rotate_out_of_room` (tier 1, gt_strength="hard"). These are
        physically invalid layouts even though the user-facing pose change
        was "just a rotation"; severity should match `move_into_wall`,
        not the soft layout tier."""
    if _is_non_rotatable(target.name):
        return None
    # Try a wider sweep of deltas in random order to maximise chance of
    # finding a clean rotation (in-bounds, no new overlap) within the
    # generator. Only emit if a clean delta is found — no fallback to
    # upgrade kinds.
    deltas = [30.0, -30.0, 40.0, -40.0, 35.0, -35.0, 25.0, -25.0,
              45.0, -45.0, 50.0, -50.0]
    rng.shuffle(deltas)
    chosen = None
    for delta in deltas:
        new_yaw = (target.yaw_deg + delta) % 360.0
        s2, f2 = _apply(scene, target.object_id, target.tx, target.ty, new_yaw)
        if _is_out_of_room(f2, s2, slack=0.0): continue
        if _overlaps_any(f2, s2.furniture): continue
        chosen = (delta, s2, f2, new_yaw)
        break
    if chosen is None:
        return None
    delta, s2, f2, new_yaw = chosen
    return _emit_with_upgrade(
        s2, f2, target,
        original_pose=(target.tx, target.ty, target.yaw_deg),
        new_pose=(target.tx, target.ty, new_yaw),
        soft_kind="rotate_45_offset",
        soft_description=f"rotate {target.object_id} by {int(delta)}° off its current axis",
        soft_reason_bad=(
            f"{target.object_id} is now rotated {int(delta)}° off its original "
            "axis, creating an awkward angle that doesn't match the rest of "
            "the room layout."
        ),
        soft_strength="directional",
        is_rotation=True,
        pre_scene=scene,
    )


def _gen_move_off_wall(scene, target, rng) -> tuple | None:
    if not _has_token(target.name, WALL_HUGGING_TOKENS):
        return None
    room = scene.room
    inset_x = max(0.4, 0.20 * room.width)
    inset_y = max(0.4, 0.20 * room.length)
    # Try up to 30 random interior positions; return the first that's
    # in-bounds and overlap-free. Don't waste records on physics fails.
    for _ in range(30):
        new_tx = room.cx + rng.uniform(-inset_x * 0.4, inset_x * 0.4)
        new_ty = room.cy + rng.uniform(-inset_y * 0.4, inset_y * 0.4)
        s2, f2 = _apply(scene, target.object_id, new_tx, new_ty, target.yaw_deg)
        if _is_out_of_room(f2, s2): continue
        if _overlaps_any(f2, s2.furniture): continue
        return _emit_with_upgrade(
            s2, f2, target,
            original_pose=(target.tx, target.ty, target.yaw_deg),
            new_pose=(new_tx, new_ty, target.yaw_deg),
            soft_kind="move_off_wall",
            soft_description=f"move {target.object_id} away from the wall into the room interior",
            soft_reason_bad=(
                f"{target.object_id} is a wall-hugging piece (storage / TV stand / bed-style); "
                "placing it in the middle of the room blocks circulation and looks wrong."
            ),
            soft_strength="soft",
        )
    return None


def _focal_partner(target, others) -> Furniture | None:
    for a, b in FOCAL_PAIRS:
        if a in target.name.lower():
            for o in others:
                if b in o.name.lower():
                    return o
        if b in target.name.lower():
            for o in others:
                if a in o.name.lower():
                    return o
    return None


def target_eligible_for_kind(scene: Scene, target: Furniture, kind: str) -> bool:
    """Cheap precondition check matching what `_GENERATORS[kind]` will
    reject up-front. Used by multitarget candidate selection so we don't
    waste 4-tuple slots on targets the perturb generator will immediately
    None on.

    Conservative — only declines when the generator has a SYNCHRONOUS
    hard filter (token / focal partner / corner adjacency / scene-level
    requirement). Physics-style filters (overlap, OOB) remain deferred
    to the generator's retry loop."""
    if kind == "move_off_wall":
        return _has_token(target.name, WALL_HUGGING_TOKENS)
    if kind == "move_far_from_focal":
        others = [g for g in scene.furniture if g.object_id != target.object_id]
        return _focal_partner(target, others) is not None
    if kind == "move_to_corner":
        room = scene.room
        near_x = ((target.tx - room.x_min < 0.25 * (room.x_max - room.x_min))
                  or (room.x_max - target.tx < 0.25 * (room.x_max - room.x_min)))
        near_y = ((target.ty - room.y_min < 0.25 * (room.y_max - room.y_min))
                  or (room.y_max - target.ty < 0.25 * (room.y_max - room.y_min)))
        return not (near_x and near_y)
    if kind == "move_blocking_door":
        # Scene-level: needs at least one door in the room.
        return len(_door_clearance_band(scene)) > 0
    # remove_object / face_wall / rotate_180 / rotate_45_offset
    # have no hard-coded target preconditions beyond physics retries.
    return True


def _gen_move_far_from_focal(scene, target, rng) -> tuple | None:
    others = [g for g in scene.furniture if g.object_id != target.object_id]
    partner = _focal_partner(target, others)
    if partner is None:
        return None
    room = scene.room
    # Aim for the half of the room AWAY from the partner — but explicitly
    # AVOID the corners (we have a separate `move_to_corner` kind for
    # those, and overlapping the two semantics produces visually-identical
    # records that aren't distinct training signals).
    margin_x = max(0.45 * target.local_width, 0.4)
    margin_y = max(0.45 * target.local_depth, 0.4)
    # A "corner" is a position close to BOTH x-extremes AND y-extremes of
    # the room interior. Use a generous ban radius (longer of target's
    # half-extent or 0.6 m) so even small pieces can't slip into a corner.
    corner_ban = max(0.6, target.local_width / 2.0, target.local_depth / 2.0)
    def _in_corner(tx: float, ty: float) -> bool:
        near_x_edge = (abs(tx - room.x_min) < corner_ban
                       or abs(tx - room.x_max) < corner_ban)
        near_y_edge = (abs(ty - room.y_min) < corner_ban
                       or abs(ty - room.y_max) < corner_ban)
        return near_x_edge and near_y_edge

    # Mid-far targets: midpoint of the wall opposite the partner, biased
    # toward the centre of that wall (away from corners).
    far_x = room.x_max - margin_x if partner.tx < room.cx else room.x_min + margin_x
    far_y = room.y_max - margin_y if partner.ty < room.cy else room.y_min + margin_y
    candidates = [
        (far_x, room.cy),       # mid of far wall along y
        (room.cx, far_y),       # mid of far wall along x
        ((room.cx + far_x) / 2, room.cy),
        (room.cx, (room.cy + far_y) / 2),
    ]
    for _ in range(30):
        rx = rng.uniform(room.x_min + margin_x, room.x_max - margin_x)
        ry = rng.uniform(room.y_min + margin_y, room.y_max - margin_y)
        if abs(rx - partner.tx) < room.width * 0.35 and abs(ry - partner.ty) < room.length * 0.35:
            continue  # too close to partner — not "far"
        if _in_corner(rx, ry):
            continue  # we don't do corners here — `move_to_corner` exists for that
        candidates.append((rx, ry))
    # Try all candidates; only accept a clean (in-bounds, no overlap)
    # placement. Drop the record entirely if none works — no fallback
    # to upgrade kinds. This keeps the generator from emitting physics-
    # violating layouts.
    chosen = None
    for new_tx, new_ty in candidates:
        s2, f2 = _apply(scene, target.object_id, new_tx, new_ty, target.yaw_deg)
        if _is_out_of_room(f2, s2): continue
        if _overlaps_any(f2, s2.furniture): continue
        chosen = (s2, f2, new_tx, new_ty)
        break
    if chosen is None:
        return None
    s2, f2, new_tx, new_ty = chosen
    return _emit_with_upgrade(
        s2, f2, target,
        original_pose=(target.tx, target.ty, target.yaw_deg),
        new_pose=(new_tx, new_ty, target.yaw_deg),
        soft_kind="move_far_from_focal",
        soft_description=f"move {target.object_id} far from its functional partner {partner.object_id}",
        soft_reason_bad=(
            f"{target.object_id} should sit close to {partner.object_id} for normal use; "
            "moving it to the far side of the room breaks the seating / dining / sleeping group."
        ),
        soft_strength="soft",
    )


def _gen_move_to_corner(scene, target, rng) -> tuple | None:
    room = scene.room
    # Reject targets that are already corner-adjacent — moving them
    # "into the corner" would barely change their position and produces
    # a near-identical scene, hurting the cumulative/sibling ordering
    # signal. We define "already in a corner" as being within 25% of
    # the room dimension from BOTH a horizontal AND a vertical wall.
    near_x = (target.tx - room.x_min < 0.25 * (room.x_max - room.x_min)) or \
             (room.x_max - target.tx < 0.25 * (room.x_max - room.x_min))
    near_y = (target.ty - room.y_min < 0.25 * (room.y_max - room.y_min)) or \
             (room.y_max - target.ty < 0.25 * (room.y_max - room.y_min))
    if near_x and near_y:
        return None
    margin_x = max(0.45 * target.local_width, 0.4)
    margin_y = max(0.45 * target.local_depth, 0.4)
    # For each corner: try the base "corner-most" placement, then up to 6
    # jittered positions that pull toward the room interior. Previous
    # behaviour tried ONE position per corner and bailed if all 4 failed
    # → ~4% yield. With jitter retries, more candidate placements satisfy
    # the no-overlap constraint while staying "in a corner".
    corners = ["nw", "ne", "sw", "se"]
    rng.shuffle(corners)
    chosen = None
    for corner in corners:
        base_cx = room.x_min + margin_x if "w" in corner else room.x_max - margin_x
        base_cy = room.y_min + margin_y if "s" in corner else room.y_max - margin_y
        # Jitter direction is INTO the room (away from the corner walls).
        inward_x = 1.0 if "w" in corner else -1.0
        inward_y = 1.0 if "s" in corner else -1.0
        positions = [(base_cx, base_cy)]
        for _ in range(6):
            jx = base_cx + inward_x * rng.uniform(0.05, 0.5)
            jy = base_cy + inward_y * rng.uniform(0.05, 0.5)
            positions.append((jx, jy))
        for cx, cy in positions:
            s2, f2 = _apply(scene, target.object_id, cx, cy, target.yaw_deg)
            if _is_out_of_room(f2, s2):
                continue
            if _overlaps_any(f2, s2.furniture):
                continue
            chosen = (s2, f2, cx, cy, corner)
            break
        if chosen is not None:
            break
    if chosen is None:
        return None
    s2, f2, cx, cy, corner = chosen
    return _emit_with_upgrade(
        s2, f2, target,
        original_pose=(target.tx, target.ty, target.yaw_deg),
        new_pose=(cx, cy, target.yaw_deg),
        soft_kind="move_to_corner",
        soft_description=f"move {target.object_id} into the {corner} corner",
        soft_reason_bad=(
            f"{target.object_id} is now jammed into the {corner} corner, far from "
            "its natural functional location."
        ),
        soft_strength="soft",
    )


def _gen_move_into_wall(scene, target, rng) -> tuple | None:
    """Place target so a meaningful slice of its footprint is OUTSIDE the
    room and the rest is INSIDE — the wall cuts through the piece. A pure
    "push past the wall" produces a render where the bench disappears
    entirely, leaving only an empty red label marker. The judge needs to
    see the protrusion to understand the perturbation, so center the
    piece roughly on the wall plane (slightly outside) with the wall
    bisecting the footprint."""
    direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    room = scene.room
    ax0, ay0, ax1, ay1 = target.world_aabb()
    half_x = (ax1 - ax0) / 2.0
    half_y = (ay1 - ay0) / 2.0
    # Wall coordinate on the push axis.
    if direction[0] == 1:
        wall_coord, sign, half = room.x_max, 1, half_x
        axis = 'x'
    elif direction[0] == -1:
        wall_coord, sign, half = room.x_min, -1, half_x
        axis = 'x'
    elif direction[1] == 1:
        wall_coord, sign, half = room.y_max, 1, half_y
        axis = 'y'
    else:
        wall_coord, sign, half = room.y_min, -1, half_y
        axis = 'y'
    # Push factor: 0.0 = center on wall (half in / half out), 0.5 = ~25% inside.
    push_factor = rng.uniform(0.05, 0.45)
    offset = sign * half * push_factor
    if axis == 'x':
        new_tx = wall_coord + offset
        new_ty = target.ty
    else:
        new_tx = target.tx
        new_ty = wall_coord + offset
    # Sanity: protrusion must be visible (≥10cm) and some still inside (≥10cm).
    if half < 0.20:
        return None
    s2, f2 = _apply(scene, target.object_id, new_tx, new_ty, target.yaw_deg)
    if not _is_out_of_room(f2, s2):
        return None
    return s2, Perturbation(
        object_id=target.object_id,
        kind="move_into_wall",
        description=f"push {target.object_id} into the wall",
        original_pose=(target.tx, target.ty, target.yaw_deg),
        new_pose=(new_tx, new_ty, target.yaw_deg),
        reason_bad=(
            f"{target.object_id}'s footprint extends past the room boundary "
            "(the wall cuts through it), which is physically invalid."
        ),
        gt_strength="hard",
    )


def _gen_move_blocking_door(scene, target, rng) -> tuple | None:
    bands = _door_clearance_band(scene)
    if not bands:
        return None
    # Try every band with jittered positions within the band footprint.
    # Single-center placement (previous behaviour) failed often because the
    # band centre commonly overlaps the door-leaf or a piece already near
    # the door — yielding only ~3% of attempts. With ~9 candidate spots
    # per band (1 center + 8 jitters), yield rises to ~25%+.
    band_order = list(range(len(bands)))
    rng.shuffle(band_order)
    for band_idx in band_order:
        band = bands[band_idx]
        bx_center = (band[0] + band[2]) / 2.0
        by_center = (band[1] + band[3]) / 2.0
        band_w = band[2] - band[0]
        band_h = band[3] - band[1]
        positions = [(bx_center, by_center)]
        for _ in range(8):
            jx = bx_center + rng.uniform(-band_w * 0.35, band_w * 0.35)
            jy = by_center + rng.uniform(-band_h * 0.35, band_h * 0.35)
            positions.append((jx, jy))
        for bx, by in positions:
            s2, f2 = _apply(scene, target.object_id, bx, by, target.yaw_deg)
            if _is_out_of_room(f2, s2):
                continue
            if _overlaps_any(f2, s2.furniture):
                continue
            a = f2.world_aabb()
            if not _aabb_overlap(a, band):
                continue
            return s2, Perturbation(
                object_id=target.object_id,
                kind="move_blocking_door",
                description=f"move {target.object_id} into the door clearance",
                original_pose=(target.tx, target.ty, target.yaw_deg),
                new_pose=(bx, by, target.yaw_deg),
                reason_bad=(
                    f"{target.object_id} now sits inside the door clearance band, "
                    "blocking entry."
                ),
                gt_strength="hard",
            )
    return None


def _gen_move_overlap(scene, target, rng) -> tuple | None:
    """Pick a NEW collision: place target so its footprint overlaps another
    piece that it does NOT already overlap, with a placement that produces
    a clearly visible top-down change.

    Earlier versions placed target at `(other.cx ± 5cm)`. That broke for two
    common cases: (1) when target is naturally already on/under `other`
    (rugs under beds, doormats by doors) the perturbation barely moves the
    object, and (2) when target is much smaller than `other`, target ends
    up fully occluded under `other` in the top-down render. Both produce
    near-identical SFT images that give the trained model no signal.
    """
    target_aabb = target.world_aabb()
    target_w = target_aabb[2] - target_aabb[0]
    target_h = target_aabb[3] - target_aabb[1]
    min_disp = max(target_w, target_h) * 0.6  # at least ~60% of own extent

    candidates = [
        g for g in scene.furniture
        if g.object_id != target.object_id
        and not _aabb_overlap(target_aabb, g.world_aabb())
    ]
    rng.shuffle(candidates)

    for other in candidates[:6]:
        ax0, ay0, ax1, ay1 = other.world_aabb()
        other_half_w = (ax1 - ax0) / 2.0
        other_half_h = (ay1 - ay0) / 2.0
        # Try several offset directions; place target so its centre sits at
        # `other.centre + edge_offset`, producing a clear half-overlap rather
        # than a hidden full overlap.
        for _ in range(8):
            sign_x, sign_y = rng.choice([
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, 1), (-1, 1), (1, -1),
            ])
            frac = rng.uniform(0.45, 0.85)
            new_tx = other.tx + sign_x * other_half_w * frac
            new_ty = other.ty + sign_y * other_half_h * frac
            # Reject negligible displacements from target's original pose.
            dx = new_tx - target.tx
            dy = new_ty - target.ty
            if dx * dx + dy * dy < min_disp * min_disp:
                continue
            s2, f2 = _apply(scene, target.object_id, new_tx, new_ty, target.yaw_deg)
            if not _overlaps_any(f2, s2.furniture):
                continue
            if _is_out_of_room(f2, s2):
                continue
            return s2, Perturbation(
                object_id=target.object_id,
                kind="move_overlap",
                description=f"move {target.object_id} on top of {other.object_id}",
                original_pose=(target.tx, target.ty, target.yaw_deg),
                new_pose=(new_tx, new_ty, target.yaw_deg),
                reason_bad=(
                    f"{target.object_id} now collides with {other.object_id} (overlapping "
                    "footprints), which is physically invalid."
                ),
                gt_strength="hard",
            )
    return None


def _gen_swap_two(scene, target, rng) -> tuple | None:
    # Reject swaps between two pieces of the SAME functional class
    # (e.g. chair_0 ↔ chair_1) — visually the swap is near-invisible
    # and the rule's "swap_two" label would be unfair signal. Use the
    # `name` field which is the human-readable category (e.g. "chair",
    # "sofa", "dining_table") shared across instances of the same type.
    same_name = target.name
    # Pair must be different category, similar size, AND far enough apart
    # that the swap produces a visible delta (≥ `_MIN_MOVE_DELTA`).
    others = [
        g for g in scene.furniture
        if g.object_id != target.object_id
        and g.name != same_name
        and abs(g.local_width * g.local_depth - target.local_width * target.local_depth)
            / max(g.local_width * g.local_depth, target.local_width * target.local_depth) < 0.6
        and ((g.tx - target.tx) ** 2 + (g.ty - target.ty) ** 2)
            >= _MIN_MOVE_DELTA * _MIN_MOVE_DELTA
    ]
    if not others:
        return None
    other = rng.choice(others)
    s = scene.clone()
    a = s.by_id(target.object_id); b = s.by_id(other.object_id)
    a.tx, b.tx = other.tx, target.tx
    a.ty, b.ty = other.ty, target.ty
    a.yaw_deg, b.yaw_deg = other.yaw_deg, target.yaw_deg
    # Two-piece move: if EITHER piece ends up overlapping or OOB, upgrade
    # to the corresponding tier-1 kind with the violating piece as the
    # primary `object_id`. (Both tier-1 same severity, so the picked
    # symptom is whichever fired first.)
    a_overlap = _overlaps_any(a, s.furniture)
    b_overlap = _overlaps_any(b, s.furniture)
    a_oob = _is_out_of_room(a, s)
    b_oob = _is_out_of_room(b, s)

    # Pre-swap positions, keyed by object_id. The cloned `a`/`b` have
    # already been swapped, so for their PRE-swap pose we need to look
    # back at the original `target` / `other`.
    pre_swap = {
        target.object_id: (target.tx, target.ty, target.yaw_deg),
        other.object_id: (other.tx, other.ty, other.yaw_deg),
    }

    def _swap_perturbation(primary_obj, partner_obj, kind, description, reason_bad):
        """Build a Perturbation whose `original_pose` / `new_pose` and
        `secondary_change` correctly describe a swap (possibly upgraded
        to a tier-1 kind). KEY INVARIANT — both the primary's pose pair
        AND the partner's pose pair must reflect THAT piece's own pre-
        and post-swap positions. Earlier code wrote the chosen-`target`
        pose into primary's field even when primary was actually `other`,
        which made Blender's `identify_xy` lookup miss the real piece."""
        return Perturbation(
            object_id=primary_obj.object_id,
            kind=kind,
            description=description,
            original_pose=pre_swap[primary_obj.object_id],
            new_pose=(primary_obj.tx, primary_obj.ty, primary_obj.yaw_deg),
            reason_bad=reason_bad,
            gt_strength="hard" if kind != "swap_two" else "soft",
            secondary_change={
                "object_id": partner_obj.object_id,
                "original_pose": list(pre_swap[partner_obj.object_id]),
                "new_pose": [partner_obj.tx, partner_obj.ty, partner_obj.yaw_deg],
            },
        )

    # NEW POLICY: if the raw swap is dirty, try micro-jitter on both
    # pieces (±0.1m → ±0.5m) to find a clean placement. Only drop if
    # all jitter attempts still violate physics.
    if a_oob or b_oob or a_overlap or b_overlap:
        a_tx0, a_ty0 = a.tx, a.ty
        b_tx0, b_ty0 = b.tx, b.ty
        clean = False
        for j in (0.1, 0.2, 0.3, 0.5):
            for _ in range(12):
                a.tx = a_tx0 + rng.uniform(-j, j)
                a.ty = a_ty0 + rng.uniform(-j, j)
                b.tx = b_tx0 + rng.uniform(-j, j)
                b.ty = b_ty0 + rng.uniform(-j, j)
                if (not _is_out_of_room(a, s) and
                        not _is_out_of_room(b, s) and
                        not _overlaps_any(a, s.furniture) and
                        not _overlaps_any(b, s.furniture)):
                    clean = True
                    break
            if clean:
                break
        if not clean:
            return None
    # Clean swap (no upgrade): primary is the originally-chosen `target`
    # (and `a`), partner is `other` (and `b`).
    return s, _swap_perturbation(
        a, b,
        kind="swap_two",
        description=f"swap positions of {target.object_id} and {other.object_id}",
        reason_bad=(
            f"{target.object_id} and {other.object_id} have been swapped; the new "
            "placement breaks the layout (e.g. seating no longer faces the coffee table, "
            "storage no longer sits on its wall)."
        ),
    )


def _gen_remove_object(scene, target, rng) -> tuple | None:
    """Remove `target` from the scene entirely. The original keeps it; the
    perturbed scene is missing it. Original is GT-better (more complete)."""
    if len(scene.furniture) <= 2:
        return None  # don't strip a near-empty room down to nothing
    s = scene.clone()
    s.furniture = [f for f in s.furniture if f.object_id != target.object_id]
    return s, Perturbation(
        object_id=target.object_id,
        kind="remove_object",
        description=f"remove {target.object_id} from the scene entirely",
        original_pose=(target.tx, target.ty, target.yaw_deg),
        new_pose=(target.tx, target.ty, target.yaw_deg),  # unused
        reason_bad=(
            f"`{target.object_id}` (a {target.name}) has been removed entirely. The "
            "room loses a piece of functional furniture without replacement, leaving "
            "the layout incomplete."
        ),
        gt_strength="soft",
        is_removal=True,
    )


_GENERATORS = {
    "face_wall": _gen_face_wall,
    "rotate_180": _gen_rotate_180,
    "rotate_45_offset": _gen_rotate_45_offset,
    "move_off_wall": _gen_move_off_wall,
    "move_far_from_focal": _gen_move_far_from_focal,
    "move_to_corner": _gen_move_to_corner,
    "move_into_wall": _gen_move_into_wall,
    "move_blocking_door": _gen_move_blocking_door,
    "move_overlap": _gen_move_overlap,
    "swap_two": _gen_swap_two,
    "remove_object": _gen_remove_object,
}


# Default sampling weights, biased toward kinds that are HARDER for a small
# multimodal model to learn from photo-real top-downs:
#   * Visually drastic (collision, missing piece) → easy to spot → low weight.
#   * Subtle (off-axis rotation, slight position shift, off-wall placement) →
#     harder for the model → high weight.
_KIND_WEIGHTS = {
    # Uniform sampling — every kind has the same chance of being picked
    # at every attempt. Per the latest user directive: physics-violation
    # kinds (move_into_wall, move_overlap, rotate_out_of_room,
    # rotate_into_collision) are EXCLUDED from synthesis — physics is
    # handled by simulator feedback at deployment time. Weight 0 means
    # the sampler never picks them.
    "move_into_wall":      0.0,
    "move_blocking_door":  0.0,  # disabled: user directive to stop synthesizing door-blocking records
    "move_overlap":        0.0,
    "remove_object":       1.0,
    "rotate_180":          1.0,
    "swap_two":            1.0,
    "face_wall":           1.0,
    "move_to_corner":      1.0,
    "rotate_45_offset":    1.0,
    "move_off_wall":       1.0,
    "move_far_from_focal": 1.0,
}


# Severity tier — used to ORDER sibling perturbations branching from the same
# original (lower number = more severe = WORSE in the ranking). Only relative
# ordering matters, not absolute values. Two kinds in the same tier are
# considered ambiguous; their order between each other can come from Gemini
# or a tie-breaker random.
_SEVERITY_TIER = {
    # Tier 1 (worst) — strict physical violation. Always worse than every
    # softer perturbation.
    "move_into_wall":        1,
    "move_blocking_door":    1,
    "move_overlap":          1,
    # Emergent kinds (not sampled directly; produced by rotation
    # generators when the rotation itself yields a hard violation).
    "rotate_out_of_room":    1,  # rotation pushes footprint past a wall
    "rotate_into_collision": 1,  # rotation overlaps another piece
    # Tier 2 — severe but recoverable / non-physical.
    "face_wall":             2,
    "rotate_180":            2,
    "remove_object":         2,
    "swap_two":              2,  # also flexible — see _FLEXIBLE_TIER_SETS
    # Tier 3 — soft layout regression.
    "move_off_wall":         3,
    "move_far_from_focal":   3,
    "move_to_corner":        3,  # also flexible — see _FLEXIBLE_TIER_SETS
    "rotate_45_offset":      3,  # also flexible — see below
}


# Some kinds are subjective enough that their position in the ranking is
# allowed to slot anywhere within a SET of tiers. The compatibility check
# in `process_scene_ranking` treats Gemini's order as accepted whenever
# there's a consistent tier assignment from these sets that makes the
# rule and Gemini sequences match.
#
# Crucially, NONE of the flexible kinds may slot into tier 1 — tier 1 is
# strictly reserved for objective physical violations (collisions, OOB).
# Gemini may legitimately push swap_two / move_to_corner up to tier 2 or
# down to tier 3, but never as high as tier 1.
#
# rotate_45_offset is fully flexible across non-physical-violation tiers
# (and may even be ranked above the unperturbed original) but also never
# pretends to be a tier-1 violation.
_FLEXIBLE_TIER_SETS = {
    # swap_two and move_to_corner: never as severe as tier 1, but their
    # impact can range from tier 2 (significant disruption) to tier 3
    # (mild displacement). Gemini decides where to slot them.
    "swap_two":          frozenset({2, 3}),
    "move_to_corner":    frozenset({2, 3}),
    # rotate_45_offset: now always a clear off-axis rotation (≥ 30°,
    # see `_gen_rotate_45_offset` deltas). Locked to tier 3 — never
    # tier 0 (which previously allowed the perturbed scene to be
    # ranked as good as / better than the original; that produced
    # counter-intuitive labels and confused n=2 test scoring).
    "rotate_45_offset":  frozenset({3}),
}


def severity_tier(kind: str) -> int:
    return _SEVERITY_TIER.get(kind, 5)


def flexible_tier_set(kind: str) -> frozenset:
    """Tiers this kind is allowed to occupy in the compatibility check.
    Defaults to the singleton from _SEVERITY_TIER for kinds without a
    flexible-tier override."""
    if kind in _FLEXIBLE_TIER_SETS:
        return _FLEXIBLE_TIER_SETS[kind]
    return frozenset({_SEVERITY_TIER.get(kind, 5)})


def _resolve_kind_weights(
    kinds: list[str] | None,
    kind_weights: dict[str, float] | None,
) -> tuple[list[str], list[float]]:
    """Merge user-supplied kind_weights with the package defaults; return
    aligned (kinds_list, weights_list)."""
    kinds_list = list(kinds or _GENERATORS.keys())
    if kind_weights is None:
        weights = [_KIND_WEIGHTS.get(k, 1.0) for k in kinds_list]
    else:
        weights = [kind_weights.get(k, _KIND_WEIGHTS.get(k, 1.0)) for k in kinds_list]
    if not weights or all(w <= 0 for w in weights):
        raise ValueError("All kind weights are non-positive — nothing to sample.")
    return kinds_list, weights


def propose_perturbation(
    scene: Scene,
    *,
    rng: random.Random | None = None,
    kinds: list[PerturbKind] | None = None,
    kind_weights: dict[str, float] | None = None,
    max_tries: int = 60,
) -> tuple[Scene, Perturbation]:
    rng = rng or random.Random()
    candidates = list(scene.furniture)
    rng.shuffle(candidates)
    kinds_list, weights = _resolve_kind_weights(kinds, kind_weights)
    for _ in range(max_tries):
        target = rng.choice(candidates)
        kind = rng.choices(kinds_list, weights=weights, k=1)[0]
        gen = _GENERATORS[kind]
        out = gen(scene, target, rng)
        if out is not None:
            return out
    raise RuntimeError("Could not generate a valid perturbation for this scene.")


def _replay_perturbation(scene: Scene, pert_dict: dict) -> tuple[Scene, Perturbation] | None:
    """Re-apply a previously-generated perturbation (recorded as a dict in
    an n2 verified pool) to `scene`. Returns (new_scene, Perturbation) or
    None on failure (target missing, missing pose, etc.).

    The recorded `original_pose`/`new_pose` were captured in the pool's
    base scene. Replay is safe iff the target object has NOT been touched
    by any earlier step in the current chain — the caller is responsible
    for enforcing same-object-forbidden so this invariant holds."""
    kind = pert_dict.get("kind")
    obj_id = pert_dict.get("object_id")
    if not kind or not obj_id:
        return None
    new_scene = scene.clone()
    target = new_scene.by_id(obj_id)
    if target is None:
        return None
    is_removal = (kind == "remove_object")
    if is_removal:
        new_scene.furniture = [f for f in new_scene.furniture if f.object_id != obj_id]
    else:
        new_pose = pert_dict.get("new_pose")
        if new_pose is None or len(new_pose) < 3:
            return None
        target.tx = float(new_pose[0])
        target.ty = float(new_pose[1])
        target.yaw_deg = float(new_pose[2])
    # swap_two: also update the partner pose
    sc = pert_dict.get("secondary_change")
    if isinstance(sc, dict):
        partner_id = sc.get("object_id")
        partner_new = sc.get("new_pose")
        if partner_id and partner_new and len(partner_new) >= 3:
            partner = new_scene.by_id(partner_id)
            if partner is not None:
                partner.tx = float(partner_new[0])
                partner.ty = float(partner_new[1])
                partner.yaw_deg = float(partner_new[2])
    orig = pert_dict.get("original_pose") or (target.tx, target.ty, target.yaw_deg)
    np = pert_dict.get("new_pose") or (target.tx, target.ty, target.yaw_deg)
    return new_scene, Perturbation(
        object_id=obj_id,
        kind=kind,
        description=pert_dict.get("description", "") or "",
        original_pose=(float(orig[0]), float(orig[1]), float(orig[2])),
        new_pose=(float(np[0]), float(np[1]), float(np[2])),
        reason_bad=pert_dict.get("reason_bad", "") or "",
        gt_strength=pert_dict.get("gt_strength", "medium"),
        secondary_change=sc if isinstance(sc, dict) else None,
        is_removal=is_removal,
    )


def propose_cum_from_pool(
    scene: Scene,
    pool: list[dict],
    *,
    size: int = 4,
    rng: random.Random | None = None,
    kinds_blocklist: tuple[str, ...] = ("rotate_45_offset",),
    max_tries: int = 200,
) -> tuple[list[tuple[Scene, list[Perturbation]]], dict]:
    """Build a cum chain by REPLAYING `size - 1` perturbations drawn from
    a pre-verified pool of n2 perturbations on the same base scene.

    Each pool entry is a dict (the json form of a `Perturbation`) — only
    `kind`, `object_id`, `original_pose`, `new_pose`, and
    `secondary_change` are required for replay. Same-object reuse is
    forbidden (so each replay safely lands on an untouched object).

    Returns the same (chain, meta) shape as `propose_ranking`."""
    rng = rng or random.Random()
    chain: list[tuple[Scene, list[Perturbation]]] = [(scene, [])]
    current = scene
    used_objects: set[str] = set()
    # Eligible pool filtered by kind blocklist.
    eligible = [p for p in pool if p.get("kind") not in kinds_blocklist]
    rng.shuffle(eligible)
    tries = 0
    for _ in range(1, size):
        succeeded = False
        for cand in eligible:
            tries += 1
            if tries > max_tries:
                break
            obj_id = cand.get("object_id")
            if obj_id in used_objects:
                continue
            sc = cand.get("secondary_change") or {}
            partner_id = sc.get("object_id") if isinstance(sc, dict) else None
            if partner_id and partner_id in used_objects:
                continue
            out = _replay_perturbation(current, cand)
            if out is None:
                continue
            new_scene, p = out
            used_objects.add(p.object_id)
            if partner_id:
                used_objects.add(partner_id)
            chain.append((new_scene, chain[-1][1] + [p]))
            current = new_scene
            # Remove the consumed candidate so we don't pick it twice.
            eligible.remove(cand)
            succeeded = True
            break
        if not succeeded:
            raise RuntimeError(
                f"propose_cum_from_pool: pool exhausted before reaching size {size}; "
                f"only got chain length {len(chain)}"
            )
    return chain, {"mode": "cumulative", "tie_set": []}


def propose_ranking(
    scene: Scene,
    size: int,
    *,
    mode: str = "cumulative",   # "cumulative" | "sibling"
    rng: random.Random | None = None,
    kinds: list[PerturbKind] | None = None,
    kind_weights: dict[str, float] | None = None,
    max_tries_per_step: int = 120,
) -> tuple[list[tuple[Scene, list[Perturbation]]], dict]:
    """Generate `size` progressively-worsening scenes for a ranking.

    Returns `(chain, meta)`:
      chain — list of length `size`; chain[0] is the original (best),
              chain[size-1] is the worst. Each entry is
              `(scene_at_this_rank, perturbations_describing_this_rank)`.
      meta  — `{"mode": str, "tie_set": list[set[int]]}`. tie_set lists
              indices that are at the same severity tier (i.e. their
              relative order is interchangeable / "soft"). mode is the
              actual mode used.

    `mode = "cumulative"` (legacy):
        chain[k] is chain[k-1] with one MORE perturbation. The k-th
        perturbation is appended to the existing list.

    `mode = "sibling"`:
        chain[1..] are independent SIBLING perturbations all branching
        directly from the original scene. They are sorted by severity
        tier (less severe first → more severe last). Their COT can rely
        purely on what's visible in each scene.
    """
    if size < 2:
        raise ValueError(f"ranking size must be >= 2, got {size}")
    rng = rng or random.Random()
    kinds_list, weights = _resolve_kind_weights(kinds, kind_weights)

    if mode == "cumulative":
        # Cumulative chains exclude rotate_45_offset entirely — the
        # subjective "is a 45° tilt actually worse?" tension makes it
        # unsuitable for a chain whose semantics is "scene k+1 strictly
        # adds an issue to scene k". rotate_45_offset is generated only
        # in sibling mode going forward.
        cum_kinds = [k for k in kinds_list if k != "rotate_45_offset"]
        cum_weights = [w for k, w in zip(kinds_list, weights) if k != "rotate_45_offset"]
        chain: list[tuple[Scene, list[Perturbation]]] = [(scene, [])]
        current = scene
        # Same-object reuse is FORBIDDEN. Each step of a cum chain must
        # touch a fresh object (and a fresh swap-partner, where applicable).
        # Rationale: across different kinds, severity is not a priori
        # predictable, so the previous tier-escalation rule produced
        # ambiguous chains (e.g. `move_to_corner(chair)` → `remove(chair)`
        # — supposedly "more severe", but with the chair already shoved
        # into a corner, removing it visually changes very little).
        # Forbidding reuse makes every chain step a clean independent
        # addition of damage to a distinct piece.
        # Decorative floor pieces (rugs / mats / carpets) are filtered
        # out of the candidate list — they're not meant to be the
        # primary target of any perturbation (moving a rug barely
        # changes the photoreal render).
        used_objects: set[str] = set()
        # Pairs that have already been swap_two'd in this chain, stored
        # as frozensets to make order-independent. Explicit safeguard on
        # top of the tier-escalation rule (which already implicitly
        # rejects swap-pair-twice because the second swap has the same
        # tier).
        used_swap_pairs: set[frozenset] = set()
        for step in range(1, size):
            succeeded = False
            for _ in range(max_tries_per_step):
                candidates = [
                    f for f in current.furniture
                    if not _is_decorative_floor(f)
                ]
                if not candidates:
                    break
                target = rng.choice(candidates)
                kind = rng.choices(cum_kinds, weights=cum_weights, k=1)[0]
                # Soft-rare-sample rotation kinds on visually-symmetric
                # targets: 70% skip and retry.
                if kind in ROTATION_KINDS and _is_visually_symmetric(target):
                    if rng.random() < 0.7:
                        continue
                # Same-object reuse FORBIDDEN (pre-filter on the sampled
                # target).
                if target.object_id in used_objects:
                    continue
                gen = _GENERATORS[kind]
                out = gen(current, target, rng)
                if out is None:
                    continue
                new_scene, p = out
                # Recheck against the FINAL perturbation (after any
                # `_emit_with_upgrade` rewrite). Also enforce on the
                # swap partner if applicable.
                partner_id = (p.secondary_change or {}).get("object_id")
                if p.object_id in used_objects:
                    continue
                if partner_id and partner_id in used_objects:
                    continue
                # used_swap_pairs is now redundant (partner_id can't be
                # reused), but keep the guard for safety in case the
                # generator emits swap-with-self.
                if p.kind == "swap_two" and partner_id:
                    pair = frozenset({p.object_id, partner_id})
                    if pair in used_swap_pairs:
                        continue
                    used_swap_pairs.add(pair)
                used_objects.add(p.object_id)
                if partner_id:
                    used_objects.add(partner_id)
                chain.append((new_scene, chain[-1][1] + [p]))
                current = new_scene
                succeeded = True
                break
            if not succeeded:
                raise RuntimeError(
                    f"Could not extend cumulative ranking to size {size} at step {step}"
                )
        # tie_set: in cumulative mode, every step adds strictly more damage,
        # so the only "tie" possibility is between siblings of the same
        # severity (which doesn't apply here). Treat as no ties.
        meta = {"mode": "cumulative", "tie_set": []}
        return chain, meta

    if mode == "sibling":
        # Generate (size - 1) DISTINCT sibling perturbations from `scene`.
        siblings: list[tuple[Scene, Perturbation]] = []
        seen_kinds: set[str] = set()
        seen_targets: set[str] = set()
        # Per-object tier(s) already used in the sibling group. Required
        # to be STRICT-different across reuses (no two same-tier kinds
        # on the same object), so the sibling-test tier-flex check can
        # never collapse them into "swappable" pairs.
        target_tier_used: dict[str, set[int]] = {}
        attempts = 0
        # Lower than the previous 0.3: most sibling groups should hit
        # distinct objects. Same-obj cases are still allowed but now
        # require strictly different tier numbers (not flex sets), so we
        # keep the probability low to avoid lots of retry rejection.
        TARGET_REUSE_PROB = 0.15
        # Decorative floor pieces (rugs / mats) excluded from the
        # candidate pool — they make for invisible perturbations.
        candidate_pool = [f for f in scene.furniture if not _is_decorative_floor(f)]
        while len(siblings) < size - 1 and attempts < (size - 1) * max_tries_per_step:
            attempts += 1
            if seen_targets and rng.random() < TARGET_REUSE_PROB:
                reusable = [f for f in candidate_pool
                            if f.object_id in seen_targets]
                target = (rng.choice(reusable) if reusable
                          else rng.choice(candidate_pool))
            else:
                target = rng.choice(candidate_pool)
            kind = rng.choices(kinds_list, weights=weights, k=1)[0]
            # Soft-rare-sample rotation kinds on visually-symmetric
            # targets (see comment in cumulative branch).
            if kind in ROTATION_KINDS and _is_visually_symmetric(target):
                if rng.random() < 0.7:
                    continue
            # Diversity: avoid duplicate (kind, target) within one ranking,
            # and prefer different kinds when possible.
            if (kind, target.object_id) in {
                (p.kind, p.object_id) for _, p in siblings
            }:
                continue
            if len(siblings) > 0 and kind in seen_kinds and len(seen_kinds) < min(size - 1, 4):
                # bias toward fresh kinds while we still have budget
                continue
            # Same-target tier strictness: if this target was already
            # perturbed in this sibling group, the new kind's
            # `severity_tier()` must be different from EVERY tier already
            # used on it. This avoids "two flex {2,3} kinds on same obj"
            # collapsing under sibling-test tier-flex matching.
            if target.object_id in target_tier_used:
                if severity_tier(kind) in target_tier_used[target.object_id]:
                    continue
            gen = _GENERATORS[kind]
            out = gen(scene, target, rng)
            if out is None:
                continue
            sibling_scene, p = out
            # Recheck tier strictness against the FINAL kind (post-upgrade).
            final_tier = severity_tier(p.kind)
            if p.object_id in target_tier_used and \
                    final_tier in target_tier_used[p.object_id]:
                continue
            # Visual de-duplication: two different kinds applied to the
            # SAME object can land on the same final pose (e.g. `face_wall`
            # vs `rotate_180` on a piece whose original yaw is exactly
            # opposite the nearest wall — both become a 180° flip). Reject
            # the new sibling if its pose is within ~5cm / ~5° of an
            # existing sibling on the same target. Removals are exempt
            # (no meaningful pose; tuple-uniqueness already handles them).
            is_visual_dup = False
            if not p.is_removal:
                for _, prev_p in siblings:
                    if prev_p.is_removal or prev_p.object_id != p.object_id:
                        continue
                    dx = abs(prev_p.new_pose[0] - p.new_pose[0])
                    dy = abs(prev_p.new_pose[1] - p.new_pose[1])
                    dyaw = abs(((prev_p.new_pose[2] - p.new_pose[2]) + 180) % 360 - 180)
                    if dx < 0.05 and dy < 0.05 and dyaw < 5.0:
                        is_visual_dup = True
                        break
            if is_visual_dup:
                continue
            siblings.append((sibling_scene, p))
            seen_kinds.add(kind)
            seen_targets.add(target.object_id)
            target_tier_used.setdefault(p.object_id, set()).add(final_tier)
            partner_id = (p.secondary_change or {}).get("object_id")
            if partner_id:
                target_tier_used.setdefault(partner_id, set()).add(final_tier)
        if len(siblings) < size - 1:
            raise RuntimeError(
                f"Could not generate {size - 1} distinct sibling perturbations "
                f"after {attempts} tries"
            )
        # Order by severity: less severe FIRST, more severe LAST.
        # severity_tier: smaller number = WORSE; we want best→worst, i.e.
        # the worst (smallest tier) goes to the BACK of the chain (after
        # the original). Sort ascending by negative tier so the LARGEST
        # tier (least severe) comes first, then descending severity.
        siblings.sort(key=lambda sp: (-_SEVERITY_TIER.get(sp[1].kind, 5),
                                       sp[1].object_id))
        chain = [(scene, [])]
        for s_scene, p in siblings:
            chain.append((s_scene, [p]))

        # Compute "tie sets" — groups of indices in the chain whose order
        # is interchangeable because they share a severity tier.
        tie_sets: list[set[int]] = []
        cur_set: set[int] = set()
        cur_tier: int | None = None
        for k in range(1, len(chain)):
            t = _SEVERITY_TIER.get(chain[k][1][0].kind, 5)
            if t == cur_tier:
                cur_set.add(k)
            else:
                if len(cur_set) > 1:
                    tie_sets.append(cur_set)
                cur_set = {k}
                cur_tier = t
        if len(cur_set) > 1:
            tie_sets.append(cur_set)
        return chain, {"mode": "sibling", "tie_set": tie_sets}

    raise ValueError(f"unknown ranking mode: {mode}")
