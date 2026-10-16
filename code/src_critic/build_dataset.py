"""End-to-end driver: load scene → perturb → render both → judge → write SFT record."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from .judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_POLISH_MODEL,
    JudgeResult,
    RankingJudgeResult,
    judge_pair,
    judge_ranking,
    make_client,
    polish_ranking_reasoning,
    polish_reasoning,
)
from .perturb import (
    Perturbation,
    propose_perturbation,
    propose_ranking,
    propose_cum_from_pool,
)
from .scene_io import Scene, load_scene
from .pil_annotate import annotate_topdown
from .topdown import render_scene as render_scene_schematic


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "datasets" / "raw"
DEFAULT_PER_STEP_DIR = REPO_ROOT / "datasets" / "per_step"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "critic_pairs"


def _find_latest_blend(per_step_room_dir: Path) -> Path | None:
    """Pick the .blend with the highest step number among furniture/wall/ceiling.
    The ceiling stage gives the fully-decorated room; furniture is the cleanest
    top-down (no ceiling decor). We prefer the latest furniture step here so
    the perturbation transform applies to a parent EMPTY whose children we
    actually care about (no wall/ceiling overlays)."""
    if not per_step_room_dir.exists():
        return None
    candidates = []
    for p in per_step_room_dir.glob("furniture_*.blend"):
        try:
            n = int(p.name.split("_")[1])
        except Exception:
            continue
        candidates.append((n, p))
    if not candidates:
        return None
    return max(candidates)[1]


def _per_step_room_dir(scene_dir: Path, scene: Scene) -> Path:
    return DEFAULT_PER_STEP_DIR / scene_dir.name / scene.room.room_id


def _run_blender_render(
    blend: Path,
    out_png: Path,
    *,
    perturbations: list[dict],
    room_box: tuple[float, float, float, float],
    resolution: int = 1024,
) -> dict:
    """Run blender_render.py in a fresh subprocess. Returns the JSON metadata."""
    import json as _json
    import subprocess as _sub
    import sys as _sys
    cmd = [
        _sys.executable, "-m", "src_critic.blender_render",
        "--blend", str(blend),
        "--out", str(out_png),
        "--resolution", str(resolution),
        "--perturbations", _json.dumps(perturbations),
        "--room-box", _json.dumps(list(room_box)),
    ]
    proc = _sub.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(
            f"blender_render failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout[-1500:]}\nstderr: {proc.stderr[-1500:]}"
        )
    # last non-blank stdout line is the JSON blob
    blob = ""
    for line in reversed(proc.stdout.splitlines()):
        if line.strip().startswith("{"):
            blob = line.strip()
            break
    if not blob:
        raise RuntimeError(f"blender_render produced no JSON output: {proc.stdout[-1500:]}")
    return _json.loads(blob)


def _perturbation_payload(
    perturbation: Perturbation,
    orig_xy: dict | None = None,
) -> list[dict]:
    """Translate a Perturbation into the list of payloads consumed by
    blender_render's per-target loop.

    `orig_xy` (object_id → (tx, ty)) overrides each payload's `identify_xy`
    with the scene-original position. Required for cumulative chains: a
    later perturbation's `original_pose` reflects POST-prior-mutation
    state (e.g. side_table at its post-swap location), and blender_render
    pre-resolves every payload against the un-mutated blend file —
    pre-resolution would otherwise alias the wrong piece (the swap
    partner, still sitting at the now-vacated original location). Passing
    `orig_xy` pins identify_xy to the blend-file positions, which are
    unique and unambiguous.
    """
    def _ident(default_xy, oid):
        if orig_xy is not None and oid in orig_xy:
            return [float(orig_xy[oid][0]), float(orig_xy[oid][1])]
        return [float(default_xy[0]), float(default_xy[1])]
    obj_id = perturbation.object_id
    if perturbation.is_removal:
        return [{
            "identify_xy": _ident(perturbation.original_pose, obj_id),
            "remove": True,
        }]
    payload = [{
        "identify_xy": _ident(perturbation.original_pose, obj_id),
        "new_xy": [perturbation.new_pose[0], perturbation.new_pose[1]],
        "delta_yaw_deg": perturbation.new_pose[2] - perturbation.original_pose[2],
    }]
    sec = perturbation.secondary_change or {}
    if sec:
        sec_id = sec.get("object_id", "")
        payload.append({
            "identify_xy": _ident(sec["original_pose"], sec_id),
            "new_xy": sec["new_pose"][:2],
            "delta_yaw_deg": sec["new_pose"][2] - sec["original_pose"][2],
        })
    return payload


# Physics-violation kinds: GT is already correct from geometric simulation,
# so we skip the LLM judge for these (LLM would just confirm what the
# physics check already said, wasting an API call and risking false drops).
_PHYSICS_VIOLATION_KINDS = frozenset({
    "move_into_wall",
    "move_overlap",
    "rotate_out_of_room",
    "rotate_into_collision",
})


def _neutral_diff_description(perturbation: Perturbation) -> str:
    """Build a NEUTRAL one-line description of the difference between the
    two scenes for the new judge prompt. Goal: tell the judge WHAT changes
    between SCENE A and SCENE B, but NEVER characterise either side as
    good/bad. The judge must form its own verdict from the visuals."""
    obj = perturbation.object_id
    sec = (perturbation.secondary_change or {}).get("object_id")
    kind = perturbation.kind
    if kind == "swap_two" and sec:
        return (f"objects `{obj}` and `{sec}` have their positions and "
                "orientations exchanged between the two scenes")
    if kind == "remove_object":
        return (f"object `{obj}` is present in one scene and entirely "
                "absent (removed) in the other")
    if kind == "rotate_180":
        return f"object `{obj}` has its yaw rotated 180° between the two scenes"
    if kind == "rotate_45_offset":
        return (f"object `{obj}` has its yaw rotated by about 30-40° "
                "between the two scenes")
    if kind == "face_wall":
        return (f"object `{obj}` has a different yaw between the two "
                "scenes (one orientation faces a nearby wall, the other does not)")
    if kind == "rotate_out_of_room":
        return (f"object `{obj}` has a different yaw between the two "
                "scenes (one orientation pushes its footprint outside the room boundary)")
    if kind == "rotate_into_collision":
        return (f"object `{obj}` has a different yaw between the two "
                "scenes (one orientation makes it overlap another piece)")
    if kind == "move_off_wall":
        return (f"object `{obj}` is at different positions between the "
                "two scenes (one position is against a wall, the other free-standing)")
    if kind == "move_far_from_focal":
        return (f"object `{obj}` is at different positions between the "
                "two scenes (one position is close to its functional group, "
                "the other farther away)")
    if kind == "move_to_corner":
        return (f"object `{obj}` is at different positions between the "
                "two scenes (one is centrally placed, the other in a room corner)")
    if kind == "move_blocking_door":
        return (f"object `{obj}` is at different positions between the "
                "two scenes (one position sits in a door's clearance band)")
    if kind == "move_into_wall":
        return (f"object `{obj}` is at different positions between the "
                "two scenes (one position has its footprint extending past a room wall)")
    if kind == "move_overlap":
        return (f"object `{obj}` is at different positions between the "
                "two scenes (one position overlaps another furniture piece)")
    return f"object `{obj}`'s pose differs between the two scenes"


def _qualitative_zone(tx: float, ty: float, room) -> str:
    """Translate a (tx, ty) into a coarse natural-language room zone."""
    cx, cy = (room.x_min + room.x_max) / 2.0, (room.y_min + room.y_max) / 2.0
    dx = tx - cx
    dy = ty - cy
    half_w = room.width / 2.0
    half_l = room.length / 2.0
    near_thresh_x = 0.30 * half_w
    near_thresh_y = 0.30 * half_l
    wall_thresh = 0.65
    parts = []
    if abs(dx) > near_thresh_x or abs(dy) > near_thresh_y:
        ns = "north" if dy > 0 else "south"
        ew = "east" if dx > 0 else "west"
        if abs(dx) > wall_thresh * half_w and abs(dy) > wall_thresh * half_l:
            parts.append(f"the {ns}-{ew} corner")
        elif abs(dx) > wall_thresh * half_w:
            parts.append(f"against the {ew} wall")
        elif abs(dy) > wall_thresh * half_l:
            parts.append(f"against the {ns} wall")
        else:
            parts.append(f"in the {ns}-{ew} part of the room")
    else:
        parts.append("near the centre of the room")
    return parts[0]


def _facing_phrase(yaw_deg: float) -> str:
    """Map a yaw to which cardinal the object's front faces."""
    y = ((yaw_deg + 180.0) % 360.0) - 180.0
    if -45 < y <= 45:
        return "north"
    if 45 < y <= 135:
        return "west"
    if -135 < y <= -45:
        return "east"
    return "south"


def _rule_based_cot(perturbation, winner: str, changed_name: str, scene) -> str:
    """Qualitative chain-of-thought that does NOT mention raw coordinates.

    The SFT model will only see two clean photo-real top-downs, so any
    numeric pose data leaked into the reasoning would be unrecoverable from
    the images and would teach the model to hallucinate coordinates.
    """
    loser = "B" if winner == "A" else "A"
    op = perturbation.original_pose
    np = perturbation.new_pose
    sec = perturbation.secondary_change or {}
    obj = perturbation.object_id

    orig_zone = _qualitative_zone(op[0], op[1], scene.room)
    new_zone = _qualitative_zone(np[0], np[1], scene.room)
    orig_face = _facing_phrase(op[2])
    new_face = _facing_phrase(np[2])

    if perturbation.kind == "remove_object":
        diff = (
            f"In SCENE {winner}, `{obj}` (a {changed_name}) is present {orig_zone}, "
            "completing the layout. In SCENE "
            f"{loser}, `{obj}` has been removed entirely, leaving an empty spot "
            "and a less complete arrangement."
        )
    elif perturbation.kind in ("rotate_180", "face_wall", "rotate_45_offset"):
        diff = (
            f"In SCENE {winner}, `{obj}` is {orig_zone} with its front facing {orig_face}, "
            f"which is the natural orientation for a {changed_name}. "
            f"In SCENE {loser}, the same `{obj}` has been rotated so its front now faces "
            f"{new_face}, which points its functional side the wrong way."
        )
    elif perturbation.kind == "swap_two":
        sec_id = sec.get("object_id", "")
        diff = (
            f"In SCENE {winner}, `{obj}` is {orig_zone} and `{sec_id}` is in its own natural "
            f"location — the two pieces sit in the spots that fit their function. "
            f"In SCENE {loser}, `{obj}` and `{sec_id}` have been swapped, so the `{obj}` "
            f"ends up {new_zone} (where the other piece used to be) and `{sec_id}` is "
            f"pushed into `{obj}`'s old spot, breaking the layout group."
        )
    elif perturbation.kind in ("move_off_wall", "move_far_from_focal", "move_to_corner"):
        diff = (
            f"In SCENE {winner}, `{obj}` is {orig_zone}, which is the natural placement for "
            f"a {changed_name}. In SCENE {loser}, `{obj}` has been moved so it now sits "
            f"{new_zone}, which separates it from the rest of its functional group."
        )
    elif perturbation.kind == "move_into_wall":
        diff = (
            f"In SCENE {winner}, `{obj}` sits cleanly inside the room {orig_zone}. "
            f"In SCENE {loser}, `{obj}` has been pushed into the wall and its footprint "
            f"now extends past the room boundary — a physically invalid placement."
        )
    elif perturbation.kind == "move_blocking_door":
        diff = (
            f"In SCENE {winner}, `{obj}` is {orig_zone} and the door swing is clear. "
            f"In SCENE {loser}, `{obj}` has been moved into the doorway, blocking entry."
        )
    elif perturbation.kind == "move_overlap":
        diff = (
            f"In SCENE {winner}, `{obj}` is {orig_zone}, with no overlap with other furniture. "
            f"In SCENE {loser}, `{obj}` has been moved so it now collides with another "
            f"piece — overlapping footprints, which is physically invalid."
        )
    else:
        diff = (
            f"In SCENE {winner}, `{obj}` sits {orig_zone}; in SCENE {loser} it has been "
            f"{perturbation.description}, which makes the layout worse."
        )

    return (
        f"The only difference between the two scenes concerns `{obj}` "
        f"(a {changed_name}). {diff} "
        f"All other furniture, the room geometry and the openings (doors / windows) are "
        f"identical in both scenes. Because of the issue above, SCENE {winner} is the "
        f"better layout."
    )


def _scrub_coordinates(text: str) -> str:
    """Remove cues from a Gemini COT that the SFT-time model can't see.

    The SFT image is a CLEAN photoreal top-down — no red border, no
    FRONT/BACK boxes, no coordinate dots, no `(x,y)` numbers. Any reference
    to those markup elements would teach the model to hallucinate cues that
    aren't in its input. We also drop meter-precise pose data for the same
    reason. Qualitative spatial language ("south wall", "centre of the
    room", "in front of the sofa") is left untouched.
    """
    import re

    # Remove parenthesised coordinate pairs first.
    text = re.sub(
        r"\s*\(\s*-?\d+(?:\.\d+)?\s*(?:m|meters?)?\s*[, ]\s*-?\d+(?:\.\d+)?\s*(?:m|meters?)?\s*\)",
        "", text)
    # Remove "roughly at <coord>" or "located at <coord>" leftovers that may
    # remain after the parenthesised tuple is stripped.
    text = re.sub(
        r"\b(?:roughly\s+|located\s+|positioned\s+)?at\s+(?:the\s+(?:indicated|annotated)\s+(?:location|spot|coordinate)s?)\.?",
        "", text, flags=re.IGNORECASE)

    # yaw=NN° / yaw=NN deg / yaw of NN°
    text = re.sub(
        r"yaw\s*(?:=|of|:)?\s*-?\d+(?:\.\d+)?\s*(?:°|deg(?:rees)?)?",
        "", text, flags=re.IGNORECASE)
    # rotated by NN° / rotation of NN°
    text = re.sub(
        r"(rotat(?:ed|ion)\s+(?:by|of))\s+-?\d+(?:\.\d+)?\s*(?:°|deg(?:rees)?)",
        r"\1 a noticeable angle", text, flags=re.IGNORECASE)
    # bare "1.23 m" tokens
    text = re.sub(r"-?\d+(?:\.\d+)?\s*(?:m|meters?)\b", "", text)
    # bare "at -1.23, 4.56" patterns
    text = re.sub(
        r"\bat\s+-?\d+(?:\.\d+)?\s*[, ]\s*-?\d+(?:\.\d+)?",
        "", text)

    # Drop references to overlay-only annotations the SFT image lacks.
    overlay_patterns = [
        r"\(?\s*indicated\s+by\s+the\s+red\s+arrow\s*\)?",
        r"\(?\s*shown\s+by\s+the\s+red\s+arrow\s*\)?",
        r"\(?\s*marked\s+by\s+the\s+red\s+border\s*\)?",
        r"\bthe\s+(?:red|coloured|blue)\s+(?:arrow|border|label|dot|box)s?\b",
        r"\b(?:red|coloured|blue)\s+(?:arrow|border|label|dot|box)s?\b",
        r"\bthe\s+`?FRONT`?\s+(?:label|tag|marker|box)\b",
        r"\bthe\s+`?BACK`?\s+(?:label|tag|marker|box)\b",
        r"\b`?FRONT`?\s+/\s+`?BACK`?\s+(?:label|tag|marker)s?\b",
        r"\bthe\s+coordinate\s+(?:dot|grid|marker)s?\b",
        r"\bthe\s+grid\s+marker\s+labelled\s+`?\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)`?",
    ]
    for p in overlay_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # Tidy up double spaces / orphan punctuation introduced by the deletions.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    return text.strip()


def _build_sft_record(
    *,
    pair_id: str,
    scene_dir: Path,
    image_a: Path,
    image_b: Path,
    image_a_judge: Path,
    image_b_judge: Path,
    answer_label: str,        # "A" or "B" — ground-truth (= original)
    gemini: JudgeResult,
    perturbation: Perturbation,
    changed_obj_name: str,
    scene,
    polish_client=None,
    judge_called=True,
) -> dict:
    """Single SFT example. The completion contains Gemini's COT plus the answer."""
    sec_id = (perturbation.secondary_change or {}).get("object_id")
    if perturbation.is_removal:
        diff_clause = (
            f"`{perturbation.object_id}` (a {changed_obj_name}) is PRESENT "
            "in one scene and ABSENT (removed entirely) in the other"
        )
    elif sec_id:
        diff_clause = f"`{perturbation.object_id}` and `{sec_id}` swapped positions"
    else:
        diff_clause = (
            f"`{perturbation.object_id}` (a {changed_obj_name}) differs in "
            "position and/or orientation"
        )
    user_prompt = (
        "You are a scene-quality judge. Two top-down photoreal renderings of the "
        "same room are shown as SCENE A and SCENE B. They share the same "
        "annotations: a 5x5 grid of red coordinate dots labelled `(x, y)`; blue "
        "object_id labels (e.g. `sofa_0`, `coffee_table_0`); blue wall-surface "
        "labels; green door labels; orange window labels; and a black arrow on "
        "every furniture rectangle.\n\n"
        "Arrow convention: the arrow tail is the object's geometric centre, the "
        "arrow HEAD points to the FRONT of the piece. FRONT means the side with "
        "the functional / interactive surface — the side the user / room should "
        "face: cabinet doors, drawer fronts, open shelves, seating cushions, the "
        "keyboard side of a desk, the screen of a TV, the foot of a bed, etc. "
        "BACK is the opposite side: the back panel / backrest / headboard side, "
        "i.e. what should be turned toward a wall. Apply this concept generally, "
        "even for furniture types not explicitly listed (e.g. an unfamiliar "
        "console or seat: the FRONT is whichever side a user would interact "
        "with, the BACK is the side that hides cables / structure).\n"
        "Concrete implications: a piece against a wall should have its BACK to "
        "the wall and its FRONT (arrow) into the room; a sofa and its TV / "
        "coffee table should have arrows pointing toward each other; an arrow "
        "pointing at a wall or at another piece's back is wrong orientation.\n\n"
        "The room geometry, doors and windows are identical in both scenes. "
        f"Between the two scenes, {diff_clause}; everything else is unchanged.\n"
        "Decide which scene is the better layout, considering (1) physical "
        "validity (no wall / object intersections), (2) door / circulation "
        "clearance, (3) functional orientation (apply the FRONT / BACK concept "
        "above), and (4) overall layout coherence. Respond with "
        "<reasoning>...</reasoning> and end with EXACTLY one tag of the "
        "form `<answer>X</answer>` where X is the single letter `A` or "
        "`B` for the better scene. Do NOT prefix with `SCENE`, do NOT "
        "use `>`, `=`, `tie`, `equal`, or any extra wording inside the "
        "answer tag — just the bare letter."
    )

    if gemini.answer == answer_label and gemini.reasoning.strip():
        raw_cot = _scrub_coordinates(gemini.reasoning.strip())
        cot_source = "gemini"
    else:
        raw_cot = _rule_based_cot(perturbation, answer_label, changed_obj_name, scene)
        cot_source = "rule_fallback"

    # Second-pass polish: rewrite the (potentially stiff / structured) COT
    # into one concise designer-tone paragraph. Verdict is unchanged.
    polished = polish_reasoning(
        raw_cot,
        winner=answer_label,
        changed_object_id=perturbation.object_id,
        changed_object_name=changed_obj_name,
        secondary_object_id=(perturbation.secondary_change or {}).get("object_id"),
        client=polish_client,
    ) if polish_client is not None else raw_cot
    cot = _scrub_coordinates(polished) if polished else raw_cot
    # `answer_label` is already a single letter for n=2; concatenated form
    # is the same. (We keep this branch consistent with the n>=3 record's
    # `''.join(gt_ranking_labels)` output below.)
    completion = f"<reasoning>\n{cot}\n</reasoning>\n<answer>{answer_label}</answer>"

    return {
        "id": pair_id,
        "format": "sft_pairwise_judge",
        "scene_dir": str(scene_dir),
        # Clean photoreal renders — used in the SFT messages.
        "images": {"A": str(image_a), "B": str(image_b)},
        # Annotated copies (red highlight + FRONT/BACK + grid + labels) used
        # only to elicit Gemini's reasoning; NOT included in the SFT message.
        "judge_images": {"A": str(image_a_judge), "B": str(image_b_judge)},
        "ground_truth": {
            "better_scene": answer_label,
            "changed_object_id": perturbation.object_id,
            "changed_object_name": changed_obj_name,
            "perturbation_kind": perturbation.kind,
            "perturbation_description": perturbation.description,
            "reason_bad": perturbation.reason_bad,
            "gt_strength": perturbation.gt_strength,
            "original_pose": list(perturbation.original_pose),
            "new_pose": list(perturbation.new_pose),
            "secondary_change": perturbation.secondary_change,
        },
        "gemini": {
            "model": DEFAULT_JUDGE_MODEL,
            "raw": gemini.raw,
            "reasoning": gemini.reasoning,
            "answer": gemini.answer,
            "called": judge_called,
            "agrees_with_ground_truth": (gemini.answer == answer_label) if judge_called else None,
        },
        "cot_source": cot_source,
        # SFT messages reference the CLEAN images only.
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_a), "tag": "SCENE_A"},
                    {"type": "image", "path": str(image_b), "tag": "SCENE_B"},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {"role": "assistant", "content": completion},
        ],
    }


def process_scene(
    raw_scene_dir: Path,
    out_dir: Path,
    *,
    n_pairs: int,
    seed: int,
    skip_judge: bool,
    use_blender: bool = True,
    blender_resolution: int = 1024,
    drop_on_disagreement: bool = True,
    forced_kinds: list[str] | None = None,
    client=None,
) -> list[dict]:
    scene = load_scene(raw_scene_dir)
    rng = random.Random(seed)
    records: list[dict] = []

    blend_path: Path | None = None
    room_box: tuple[float, float, float, float] | None = None
    cached_orig_png: Path | None = None
    cached_camera_meta: dict | None = None
    if use_blender:
        ps_dir = _per_step_room_dir(raw_scene_dir, scene)
        blend_path = _find_latest_blend(ps_dir)
        if blend_path is None:
            raise FileNotFoundError(f"No furniture .blend in {ps_dir}; use --no-blender only for schematic smoke tests")
        else:
            room_box = (
                scene.room.x_min, scene.room.y_min,
                scene.room.x_max, scene.room.y_max,
            )

    for i in range(n_pairs):
        try:
            kinds_filter = None
            if forced_kinds and i < len(forced_kinds):
                kinds_filter = [forced_kinds[i]]
            perturbed_scene, perturbation = propose_perturbation(
                scene, rng=rng, kinds=kinds_filter
            )
        except RuntimeError as e:
            print(f"[skip] {raw_scene_dir.name} #{i}: {e}", file=sys.stderr)
            continue

        # Randomize which is A vs B so the judge can't shortcut on order.
        original_is_a = rng.random() < 0.5
        scene_a, scene_b = (scene, perturbed_scene) if original_is_a else (perturbed_scene, scene)
        answer_label = "A" if original_is_a else "B"

        pair_id = f"{raw_scene_dir.name}__pair_{i:02d}"
        pair_dir = out_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        title_a = "SCENE A" + (" (original)" if original_is_a else " (perturbed)")
        title_b = "SCENE B" + (" (perturbed)" if original_is_a else " (original)")

        if use_blender:
            # Render the original once per scene (cached), perturbed per pair.
            if cached_orig_png is None:
                cache_dir = out_dir / "_renders" / raw_scene_dir.name
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_orig_png = cache_dir / "original.png"
                cached_meta_path = cache_dir / "original.meta.json"
                if not cached_orig_png.exists() or not cached_meta_path.exists():
                    print(f"[blender] rendering original {raw_scene_dir.name} "
                          f"({blend_path.name})", file=sys.stderr)
                    meta_o = _run_blender_render(
                        blend_path, cached_orig_png,
                        perturbations=[], room_box=room_box,
                        resolution=blender_resolution,
                    )
                    cached_meta_path.write_text(json.dumps(meta_o["camera"]))
                cached_camera_meta = json.loads(cached_meta_path.read_text())

            pert_png = pair_dir / "_pert_raw.png"
            print(f"[blender] rendering perturbed {pair_id} "
                  f"({perturbation.kind} on {perturbation.object_id})",
                  file=sys.stderr)
            try:
                meta_p = _run_blender_render(
                    blend_path, pert_png,
                    perturbations=_perturbation_payload(perturbation),
                    room_box=room_box,
                    resolution=blender_resolution,
                )
                pert_camera_meta = meta_p["camera"]
            except Exception as e:
                raise RuntimeError(f"Blender render failed for {pair_id}") from e

            if pert_png is not None:
                # Two annotated copies per scene:
                # * scene_A.png / scene_B.png  — annotated with grid + object_ids
                #   + door / window / wall labels + per-object FRONT arrows.
                #   NO red border, NO FRONT/BACK boxes, NO red arrow — the
                #   SFT model must figure out the change from photo + IDs only.
                # * scene_A_judge.png / scene_B_judge.png — same annotations
                #   PLUS the change pointers (red border, FRONT/BACK, red arrow).
                orig_sft = pair_dir / ("scene_A.png" if original_is_a else "scene_B.png")
                pert_sft = pair_dir / ("scene_B.png" if original_is_a else "scene_A.png")
                orig_judge = pair_dir / ("scene_A_judge.png" if original_is_a else "scene_B_judge.png")
                pert_judge = pair_dir / ("scene_B_judge.png" if original_is_a else "scene_A_judge.png")

                # SFT copies = annotated WITHOUT change-pointing highlights.
                annotate_topdown(
                    cached_orig_png, scene=scene,
                    camera_meta=cached_camera_meta,
                    highlight_ids=[],
                    out_path=orig_sft,
                )
                annotate_topdown(
                    pert_png, scene=perturbed_scene,
                    camera_meta=pert_camera_meta,
                    highlight_ids=[],
                    out_path=pert_sft,
                )

                # Judge copies = full annotation including red border + FRONT/BACK.
                hi_ids = [perturbation.object_id]
                if (perturbation.secondary_change or {}).get("object_id"):
                    hi_ids.append(perturbation.secondary_change["object_id"])
                annotate_topdown(
                    cached_orig_png, scene=scene,
                    camera_meta=cached_camera_meta,
                    highlight_ids=hi_ids,
                    out_path=orig_judge,
                )
                annotate_topdown(
                    pert_png, scene=perturbed_scene,
                    camera_meta=pert_camera_meta,
                    highlight_ids=hi_ids,
                    out_path=pert_judge,
                )
                pert_png.unlink(missing_ok=True)
                img_a = pair_dir / "scene_A.png"               # clean — for SFT
                img_b = pair_dir / "scene_B.png"
                img_a_judge = pair_dir / "scene_A_judge.png"   # annotated — for Gemini
                img_b_judge = pair_dir / "scene_B_judge.png"
            else:
                # Schematic fallback — judge and SFT use the same image
                img_a = render_scene_schematic(
                    scene_a, pair_dir / "scene_A.png",
                    title="SCENE A", highlight_id=None,
                )
                img_b = render_scene_schematic(
                    scene_b, pair_dir / "scene_B.png",
                    title="SCENE B", highlight_id=None,
                )
                img_a_judge, img_b_judge = img_a, img_b
        else:
            img_a = render_scene_schematic(
                scene_a, pair_dir / "scene_A.png",
                title="SCENE A", highlight_id=None,
            )
            img_b = render_scene_schematic(
                scene_b, pair_dir / "scene_B.png",
                title="SCENE B", highlight_id=None,
            )
            img_a_judge, img_b_judge = img_a, img_b

        changed = scene.by_id(perturbation.object_id)
        changed_name = changed.name if changed else perturbation.object_id

        if skip_judge:
            gemini = JudgeResult(raw="", reasoning="(skipped)", answer="UNKNOWN")
        else:
            try:
                gemini = judge_pair(
                    img_a_judge, img_b_judge,
                    changed_object_id=perturbation.object_id,
                    changed_object_name=changed_name,
                    secondary_object_id=(perturbation.secondary_change or {}).get("object_id"),
                    is_removal=perturbation.is_removal,
                    client=client,
                )
            except Exception as e:
                print(f"[judge_error] {pair_id}: {e!r}", file=sys.stderr)
                traceback.print_exc()
                gemini = JudgeResult(raw=f"ERROR: {e}", reasoning="(error)", answer="UNKNOWN")

        verdict = "✓" if gemini.answer == answer_label else ("✗" if gemini.answer in ("A", "B") else "?")

        # For HARD (collision) and DIRECTIONAL (orientation) perturbations
        # the ground truth is unambiguous, so we keep the pair regardless of
        # Gemini's verdict and fall back to the rule-based COT when Gemini
        # disagrees. Only SOFT perturbations (move-off-wall, swap, far-from-
        # focal, corner, removal) are genuinely subjective; for those we
        # drop the pair when Gemini disagrees.
        disagrees = (gemini.answer in ("A", "B")) and gemini.answer != answer_label
        is_soft = perturbation.gt_strength == "soft"
        if disagrees and drop_on_disagreement and is_soft and not skip_judge:
            print(
                f"[drop] {pair_id}  changed={perturbation.object_id} ({perturbation.kind})  "
                f"gt={answer_label}  gemini={gemini.answer}  ✗ — soft+disagree, pair_dir removed"
            )
            shutil.rmtree(pair_dir, ignore_errors=True)
            continue

        record = _build_sft_record(
            pair_id=pair_id,
            scene_dir=raw_scene_dir,
            image_a=img_a,
            image_b=img_b,
            image_a_judge=img_a_judge,
            image_b_judge=img_b_judge,
            answer_label=answer_label,
            gemini=gemini,
            perturbation=perturbation,
            changed_obj_name=changed_name,
            scene=scene,
            polish_client=client,
            judge_called=not skip_judge,
        )
        (pair_dir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        records.append(record)
        print(
            f"[ok] {pair_id}  changed={perturbation.object_id} ({perturbation.kind})  "
            f"gt={answer_label}  gemini={gemini.answer}  {verdict}"
        )
    return records


def _kendall_tau_distance(a: list[str], b: list[str]) -> int:
    """Number of pairwise inversions between two orderings of the same items.
    Returns the Kendall tau distance (0 = identical, max = fully reversed)."""
    if not a or set(a) != set(b):
        return -1
    pos_b = {x: i for i, x in enumerate(b)}
    inv = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if pos_b[a[i]] > pos_b[a[j]]:
                inv += 1
    return inv


def _severity_label_for_step(chain, k: int) -> int:
    from .perturb import severity_tier
    if k == 0:
        return 0
    perts = chain[k][1]
    if not perts:
        return 0
    # For cumulative mode the "step" tier is the LAST added perturbation;
    # for sibling mode the chain[k] only has one perturbation.
    return severity_tier(perts[-1].kind)


def _summarise_perturbations(
    perturbations: list["Perturbation"],
) -> str:
    """Short bullet-list of cumulative perturbations for the polish call."""
    if not perturbations:
        return "(no perturbations)"
    lines = []
    for i, p in enumerate(perturbations, 1):
        lines.append(f"{i}. {p.kind} on `{p.object_id}` — {p.description}")
    return "\n".join(lines)


def _record_fingerprint(scene_id: str, p: Perturbation) -> tuple:
    """Compact key identifying the same perturbation across workers /
    seeds. Two records sharing this fingerprint are logical duplicates."""
    obj = p.object_id
    sec = (p.secondary_change or {}).get("object_id", "")
    np = p.new_pose
    if np:
        np_key = (round(np[0] * 5) / 5,
                  round(np[1] * 5) / 5,
                  round(np[2] / 10) * 10)
    else:
        np_key = None
    return (scene_id, p.kind, obj, sec, np_key)


def _cum_chain_fingerprint(scene_id: str, perts) -> tuple:
    """Whole-chain fingerprint for a cum ranking. Two cum records that
    apply the SAME SET of (kind, target_id, partner_id) tuples to the
    SAME base scene are logical duplicates regardless of step order.

    Accepts either a list of `Perturbation` dataclasses or a list of
    dicts (as read back from JSON)."""
    items = []
    for p in perts:
        if hasattr(p, "kind"):
            kind = p.kind; obj = p.object_id
            sec = (p.secondary_change or {}).get("object_id", "")
        else:
            kind = p["kind"]; obj = p["object_id"]
            sec = (p.get("secondary_change") or {}).get("object_id", "")
        items.append((kind, obj, sec))
    return (scene_id, "__CUM_CHAIN__", frozenset(items))


def _record_fingerprint_from_json(rec: dict) -> tuple | None:
    scene_id = rec["id"].split("__ranking_")[0]
    perts = rec.get("ground_truth", {}).get("perturbations", [])
    if not perts:
        return None
    # If this is a cum n=4 record, return the whole-chain fingerprint so
    # the dedup set can block other cum chains that touch the same
    # (kind, target) set on the same base scene.
    if "_n4_cum" in rec.get("id", "") and len(perts) >= 2:
        return _cum_chain_fingerprint(scene_id, perts)
    p = perts[0]
    obj = p["object_id"]
    sec = (p.get("secondary_change") or {}).get("object_id", "")
    kind = p["kind"]
    np = p.get("new_pose")
    if np:
        np_key = (round(np[0] * 5) / 5,
                  round(np[1] * 5) / 5,
                  round(np[2] / 10) * 10)
    else:
        np_key = None
    return (scene_id, kind, obj, sec, np_key)


def _load_n2_verified_pool(out_dir: Path) -> dict[str, list[dict]]:
    """Scan canonical jsonl files for n=2 sibling records that were
    accepted by the LLM judge (gemini.matches_ground_truth == True). The
    perturbation dict from each accepted record is grouped by base scene
    so a cum-from-pool generator can draw from same-scene verified
    perturbations.
    """
    pool: dict[str, list[dict]] = {}
    sources = [
        out_dir / "sft.jsonl",
        out_dir / "sft_aug.jsonl",
        out_dir / "sft_aug_perm.jsonl",
        out_dir / "sft_aug_perm_fixed.jsonl",
        out_dir / "sft_train.jsonl",
        out_dir / "sft_train_v4b.jsonl",
        out_dir / "sft_train_v5.jsonl",
    ]
    seen_ids: set[str] = set()
    for path in sources:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("n") != 2:
                        continue
                    rid = rec.get("id", "")
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    gem = rec.get("gemini") or {}
                    if not gem.get("matches_ground_truth"):
                        continue
                    perts = rec.get("ground_truth", {}).get("perturbations") or []
                    if not perts:
                        continue
                    scene_key = rid.split("__")[0]
                    pool.setdefault(scene_key, []).append(perts[0])
        except Exception:
            pass
    return pool


def _load_existing_fingerprints(out_dir: Path) -> set[tuple]:
    """Build the set of fingerprints already represented on disk so the
    generator can avoid re-creating logically-identical records."""
    fps: set[tuple] = set()
    # Scan ALL canonical jsonl sources used in this project so the
    # dedup set covers cum chains that landed in train/test merges,
    # n4 supplements, or worker shards. Add new entries here when new
    # canonical sources are introduced.
    canonical = [
        out_dir / "sft.jsonl",
        out_dir / "sft_train.jsonl",
        out_dir / "sft_n4_train.jsonl",
        out_dir / "sft_n4_test.jsonl",
        out_dir / "sft_train_v4b.jsonl",
        out_dir / "sft_train_v5.jsonl",
        out_dir / "sft_train_v5_clean.jsonl",
        out_dir / "_permuted_test" / "sft_test_cumulative.jsonl",
        out_dir / "_permuted_test" / "sft_test_cumulative_clean.jsonl",
    ]
    globs = (
        sorted(out_dir.glob("sft_w*.jsonl"))
        + sorted(out_dir.glob("sft_aug_w*.jsonl"))
        + sorted(out_dir.glob("sft_n4_w*.jsonl"))
        + sorted(out_dir.glob("sft_n4_severity_w*.jsonl"))
    )
    for path in canonical + globs:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    fp = _record_fingerprint_from_json(rec)
                    if fp:
                        fps.add(fp)
        except Exception:
            pass
    return fps


def process_scene_ranking(
    raw_scene_dir: Path,
    out_dir: Path,
    *,
    n_rankings: int,
    seed: int,
    skip_judge: bool,
    ranking_size_dist: list[tuple[int, float]],
    kind_weights: dict[str, float] | None,
    mode_dist: list[tuple[str, float]] | None = None,
    use_blender: bool = True,
    blender_resolution: int = 1024,
    drop_on_disagreement_soft_only: bool = True,
    client=None,
    sft_fh=None,
    fingerprint_set: set[tuple] | None = None,
    no_judge_keep_polish: bool = False,
    cum_pool: list[dict] | None = None,
) -> list[dict]:
    """Generate `n_rankings` ranking-mode tasks for one scene.

    `no_judge_keep_polish`: when True, skip the LLM judge call (we trust
    the rule-based GT directly) but still run the polish pass so the
    final reasoning reads as natural prose. Used for cum chains under
    the same-object-forbidden rule where the ordering is unambiguous
    and judge agreement isn't a useful filter."""
    scene = load_scene(raw_scene_dir)
    rng = random.Random(seed)
    records: list[dict] = []

    blend_path: Path | None = None
    room_box: tuple[float, float, float, float] | None = None
    cached_orig_png: Path | None = None
    cached_camera_meta: dict | None = None
    if use_blender:
        ps_dir = _per_step_room_dir(raw_scene_dir, scene)
        blend_path = _find_latest_blend(ps_dir)
        if blend_path is None:
            raise FileNotFoundError(f"No furniture .blend in {ps_dir}")
        else:
            room_box = (scene.room.x_min, scene.room.y_min,
                        scene.room.x_max, scene.room.y_max)

    mode_dist_local = mode_dist or [("cumulative", 1.0), ("sibling", 1.0)]
    mode_names, mode_weights = zip(*mode_dist_local)

    for i in range(n_rankings):
        # Sample size from the (size, weight) distribution.
        sizes, weights = zip(*ranking_size_dist)
        size = rng.choices(sizes, weights=weights, k=1)[0]
        mode = rng.choices(mode_names, weights=mode_weights, k=1)[0]
        try:
            if cum_pool is not None and mode == "cumulative" and size >= 3:
                # Pool-based replay: draw verified n2 perturbations.
                # max_tries scaled by pool size so a small pool isn't a
                # silent dead-end. Pool entries are dicts (json form).
                chain, meta = propose_cum_from_pool(
                    scene, cum_pool, size=size, rng=rng,
                )
            else:
                chain, meta = propose_ranking(
                    scene, size, mode=mode, rng=rng, kind_weights=kind_weights
                )
        except RuntimeError as e:
            print(f"[skip] {raw_scene_dir.name} ranking #{i}: {e}", file=sys.stderr)
            continue

        # Fingerprint check: skip logically-duplicate records (same
        # scene + kind + target + partner + grid-rounded new pose).
        if fingerprint_set is not None and size == 2 and len(chain) >= 2:
            last_p = chain[-1][1][-1]
            fp = _record_fingerprint(raw_scene_dir.name, last_p)
            if fp in fingerprint_set:
                print(f"[dup] {raw_scene_dir.name} #{i} {last_p.kind}/{last_p.object_id} — duplicate, skip",
                      file=sys.stderr)
                continue
            fingerprint_set.add(fp)
        # cum chain fingerprint: skip when an existing cum record on the
        # same scene already covers this exact set of (kind, target,
        # partner) tuples (regardless of step ordering).
        if (fingerprint_set is not None and mode == "cumulative"
                and size >= 3 and len(chain) >= 2):
            # chain[-1] = (scene, list of all perturbations applied)
            full_perts = chain[-1][1]
            fp = _cum_chain_fingerprint(raw_scene_dir.name, full_perts)
            if fp in fingerprint_set:
                print(f"[dup-cum] {raw_scene_dir.name} #{i}: same (kind,target) "
                      f"set as existing cum — skip", file=sys.stderr)
                continue
            fingerprint_set.add(fp)

        # Permute visible labels A..N.
        labels = ["A", "B", "C", "D"][:size]
        perm = list(range(size))
        if size == 2:
            # Strict 50/50 alternation between [0,1] and [1,0] within a
            # worker — the random shuffle already gave ~49/51 in practice
            # but explicit alternation removes ANY drift toward an A- or
            # B-position bias in the training data.
            if i % 2 == 1:
                perm = [1, 0]
        else:
            rng.shuffle(perm)
        # visible_label[i] = labels[i], displayed image = chain[perm[i]]
        # GT ranking (best -> worst) by chain index = [0, 1, ..., size-1].
        # Mapped to visible labels: for chain index k, the visible label is
        #   labels[perm.index(k)]
        gt_ranking_labels = [labels[perm.index(k)] for k in range(size)]

        # Include the per-scene seed in the ranking_id so concurrent passes
        # of run_until_n don't reuse the same directory (and accidentally
        # delete records-in-flight on a soft+disagree from a later pass).
        ranking_id = f"{raw_scene_dir.name}__ranking_s{seed % 1000000}_{i:02d}_n{size}_{mode[:3]}"
        ranking_dir = out_dir / ranking_id
        ranking_dir.mkdir(parents=True, exist_ok=True)

        sft_imgs: list[Path] = []
        judge_imgs: list[Path] = []
        per_scene_perts: list[list[Perturbation]] = []

        if use_blender:
            # Cache the all-zeros "original" render for this scene.
            if cached_orig_png is None:
                cache_dir = out_dir / "_renders" / raw_scene_dir.name
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_orig_png = cache_dir / "original.png"
                cached_meta_path = cache_dir / "original.meta.json"
                if not cached_orig_png.exists() or not cached_meta_path.exists():
                    print(f"[blender] rendering original {raw_scene_dir.name}", file=sys.stderr)
                    try:
                        meta_o = _run_blender_render(
                            blend_path, cached_orig_png,
                            perturbations=[], room_box=room_box,
                            resolution=blender_resolution,
                        )
                        cached_meta_path.write_text(json.dumps(meta_o["camera"]))
                    except Exception as e:
                        print(f"[blender_skip_scene] {raw_scene_dir.name}: original render failed: {e!r}",
                              file=sys.stderr)
                        # disable Blender for this whole scene; the per-ranking
                        # loop falls back to schematic.
                        raise RuntimeError("Original Blender render failed") from e
                if use_blender:
                    cached_camera_meta = json.loads(cached_meta_path.read_text())

            ranking_render_failed = False
            if use_blender:
                for pos in range(size):
                    chain_idx = perm[pos]
                    visible_label = labels[pos]
                    cur_scene, cur_perts = chain[chain_idx]
                    per_scene_perts.append(cur_perts)
                    sft_path = ranking_dir / f"scene_{visible_label}.png"
                    judge_path = ranking_dir / f"scene_{visible_label}_judge.png"
                    if chain_idx == 0:
                        raw_png = cached_orig_png
                        cam_meta = cached_camera_meta
                    else:
                        raw_png = ranking_dir / f"_raw_{visible_label}.png"
                        # In a cum chain, later perturbations' identify_xy
                        # reflects post-prior-mutation state. blender_render
                        # pre-resolves every group at once against the blend
                        # file — pin identify_xy to the scene-original
                        # positions so the resolution is unambiguous.
                        orig_xy = {
                            f.object_id: (f.tx, f.ty)
                            for f in chain[0][0].furniture
                        }
                        payload = []
                        for p in cur_perts:
                            payload.extend(_perturbation_payload(p, orig_xy=orig_xy))
                        print(f"[blender] rendering {ranking_id} pos={visible_label} "
                              f"(chain step {chain_idx}, {len(cur_perts)} cumulative)",
                              file=sys.stderr)
                        try:
                            meta_p = _run_blender_render(
                                blend_path, raw_png,
                                perturbations=payload, room_box=room_box,
                                resolution=blender_resolution,
                            )
                            cam_meta = meta_p["camera"]
                        except Exception as e:
                            print(f"[blender_error_ranking] {ranking_id} pos={visible_label}: "
                                  f"{e!r} — skipping this ranking", file=sys.stderr)
                            raise RuntimeError(f"Ranking render failed: {ranking_id} {visible_label}") from e

                    # Clean SFT image — no change-pointing markup. The model
                    # must figure out the difference from the photo + ids only.
                    annotate_topdown(raw_png, scene=cur_scene, camera_meta=cam_meta,
                                     highlight_ids=[], out_path=sft_path)
                    # Judge image — full markup. We split highlights into:
                    #   * rotation_hi: perturbed-by-rotation pieces → get a
                    #     RED arrow (same length as black) + FRONT/BACK boxes
                    #   * judge_hi (all): get a red border around the bbox
                    # so the only objects with FRONT/BACK / red arrow are
                    # those whose orientation actually changed.
                    from .perturb import ROTATION_KINDS
                    judge_hi: list[str] = []
                    rotation_hi: list[str] = []
                    for p in cur_perts:
                        judge_hi.append(p.object_id)
                        if p.kind in ROTATION_KINDS:
                            rotation_hi.append(p.object_id)
                        sec = (p.secondary_change or {}).get("object_id")
                        if sec:
                            judge_hi.append(sec)
                    annotate_topdown(raw_png, scene=cur_scene, camera_meta=cam_meta,
                                     highlight_ids=judge_hi,
                                     rotation_highlight_ids=rotation_hi,
                                     out_path=judge_path)
                    if chain_idx != 0:
                        raw_png.unlink(missing_ok=True)
                    sft_imgs.append(sft_path)
                    judge_imgs.append(judge_path)

            if ranking_render_failed:
                shutil.rmtree(ranking_dir, ignore_errors=True)
                # reset accumulator state for the next ranking attempt
                sft_imgs.clear(); judge_imgs.clear(); per_scene_perts.clear()
                continue
        else:
            # Schematic fallback (no Blender): use matplotlib renderer.
            for pos in range(size):
                chain_idx = perm[pos]
                cur_scene, cur_perts = chain[chain_idx]
                per_scene_perts.append(cur_perts)
                visible_label = labels[pos]
                p = render_scene_schematic(
                    cur_scene, ranking_dir / f"scene_{visible_label}.png",
                    title=f"SCENE {visible_label}",
                    highlight_id=None,
                )
                sft_imgs.append(p)
                judge_imgs.append(p)

        # `perts_per_step` is the canonical "one perturbation per chain step"
        # list, valid for BOTH ranking modes. (Computed up-front so the
        # judge call can use it for the per-scene perturbation summary.)
        perts_per_step = [chain[k][1][-1] for k in range(1, len(chain))]
        non_45_perts = [p for p in perts_per_step if p.kind != "rotate_45_offset"]
        pure_45_chain = perts_per_step and not non_45_perts

        # Build per-scene perturbation descriptions (ordered by label
        # position). Used as a privileged hint for the judge.
        per_scene_descr: list[str] = []
        for pos in range(size):
            plist = per_scene_perts[pos]
            if not plist:
                per_scene_descr.append("")  # original
            else:
                per_scene_descr.append("; ".join(p.description for p in plist))

        # NEW: neutral diff description for the judge (n=2 only). Tells
        # the judge what changes WITHOUT revealing which scene is the
        # original or biasing toward either side. For n>=3 we keep
        # `per_scene_descr` as the legacy hint.
        neutral_diff_text: str | None = None
        skip_llm_judge = False
        if size == 2 and perts_per_step:
            neutral_diff_text = _neutral_diff_description(perts_per_step[0])
            # Physics-violation kinds: GT verified by geometric simulator,
            # no need to ask the LLM (would just confirm the obvious).
            if perts_per_step[0].kind in _PHYSICS_VIOLATION_KINDS:
                skip_llm_judge = True

        # ----- 2-attempt judge with hints -----
        # Per the user's directive: gpt-5.2 (default judge model) gets the
        # annotated images PLUS the per-scene perturbation list PLUS the
        # severity-tier rule. If attempt 1 produces a compatible ranking,
        # use its reasoning. Otherwise retry once with higher temperature
        # for diversity. If attempt 2 is also wrong, drop the record.
        # `judge_ranking` itself does internal retry on transient API
        # errors, so each attempt below already absorbs network blips.

        def _is_compatible(ranking: list[str]) -> tuple[bool, list[str]]:
            """Returns (compatible, possibly_updated_gt). For pure_45 we
            accept any well-formed ranking and adopt it as GT. For sibling
            we run the union-find tier-flexibility check (and adopt the
            judge's order iff compatible). For cumulative/n2 we require an
            exact match against the rule's GT."""
            if not ranking or len(ranking) != size or set(ranking) != set(labels):
                return False, gt_ranking_labels
            # NEW POLICY: for n=2 records (using neutral_diff prompt) we
            # require an EXACT match against the rule's GT. No
            # pure_45_chain acceptance, no sibling tier-flex relaxation —
            # any disagreement drops the record.
            if neutral_diff_text is not None:
                if ranking == gt_ranking_labels:
                    return True, gt_ranking_labels
                return False, gt_ranking_labels
            if pure_45_chain:
                return True, list(ranking)
            if ranking == gt_ranking_labels:
                return True, gt_ranking_labels
            if meta.get("mode") == "sibling":
                from .perturb import flexible_tier_set
                ranks_in_chain = {labels[i]: perm[i] for i in range(size)}
                label_tier_set = {}
                for lab in labels:
                    k = ranks_in_chain[lab]
                    if k == 0:
                        label_tier_set[lab] = frozenset({0})
                    else:
                        p = chain[k][1][-1]
                        label_tier_set[lab] = flexible_tier_set(p.kind)
                parent = {lab: lab for lab in labels}
                def _find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def _union(a, b):
                    ra, rb = _find(a), _find(b)
                    if ra != rb:
                        parent[ra] = rb
                for i in range(size):
                    _union(gt_ranking_labels[i], ranking[i])
                classes: dict = {}
                for lab in labels:
                    root = _find(lab)
                    if root not in classes:
                        classes[root] = set(label_tier_set[lab])
                    else:
                        classes[root] &= label_tier_set[lab]
                if all(classes[k] for k in classes):
                    return True, list(ranking)
            return False, gt_ranking_labels

        def _judge_attempt(temperature: float) -> RankingJudgeResult:
            """One judge call wrapped in transient-error retry."""
            JUDGE_MAX_ATTEMPTS = 5
            JUDGE_BACKOFF = [2, 4, 8, 16, 32]
            for attempt in range(JUDGE_MAX_ATTEMPTS):
                try:
                    # For n=2 records use the new neutral-diff prompt
                    # (no per-scene hints, no reason_bad leakage). For
                    # n>=3 keep the legacy per-scene descriptions.
                    if neutral_diff_text is not None:
                        return judge_ranking(
                            judge_imgs, client=client,
                            neutral_diff=neutral_diff_text,
                            include_tier_rule=False,
                            temperature=temperature,
                        )
                    return judge_ranking(
                        judge_imgs, client=client,
                        per_scene_perts=per_scene_descr,
                        include_tier_rule=False,
                        temperature=temperature,
                    )
                except Exception as e:
                    print(f"[judge_error] {ranking_id} t={temperature} "
                          f"net-retry {attempt + 1}: {e!r}", file=sys.stderr)
                    if attempt == JUDGE_MAX_ATTEMPTS - 1:
                        traceback.print_exc()
                        return RankingJudgeResult(
                            raw=f"ERROR: {e}", reasoning="", ranking=[]
                        )
                    time.sleep(JUDGE_BACKOFF[attempt])
            return RankingJudgeResult(raw="", reasoning="", ranking=[])

        if skip_judge or skip_llm_judge or no_judge_keep_polish:
            # No-judge path: GT is authoritative (rule-based for cum
            # under the same-object-forbidden rule; physics-simulator for
            # tier-1 violations). Skip the judge call entirely; we still
            # populate a stub RankingJudgeResult so downstream code can
            # consult it uniformly.
            reason_stub = (
                "(physics-violation: geometric simulator GT, judge skipped)"
                if skip_llm_judge else
                "(judge skipped; using rule-based draft + polish)"
            )
            gemini = RankingJudgeResult(
                raw="",
                reasoning=reason_stub,
                ranking=list(gt_ranking_labels),
            )
            compatible = True
        else:
            # Attempt 1 — temperature 0.2 (focused).
            gemini = _judge_attempt(temperature=0.2)
            compatible, gt_ranking_labels = _is_compatible(gemini.ranking)
            if not compatible:
                # Attempt 2 — temperature 0.6 for diversity.
                gemini2 = _judge_attempt(temperature=0.6)
                compat2, new_gt2 = _is_compatible(gemini2.ranking)
                if compat2:
                    gemini = gemini2
                    compatible = True
                    gt_ranking_labels = new_gt2
                else:
                    print(f"[drop] {ranking_id}  gt={'>'.join(gt_ranking_labels)}  "
                          f"a1={'>'.join(gemini.ranking) if gemini.ranking else '?'}  "
                          f"a2={'>'.join(gemini2.ranking) if gemini2.ranking else '?'}  "
                          f"✗ — both attempts wrong, drop", file=sys.stderr)
                    shutil.rmtree(ranking_dir, ignore_errors=True)
                    continue

        kt = _kendall_tau_distance(gemini.ranking, gt_ranking_labels) if gemini.ranking else -1
        verdict = "✓" if gemini.ranking == gt_ranking_labels else "~"
        full_chain_summary = _summarise_perturbations(perts_per_step)

        # Successful path: use the winning attempt's reasoning. Fall back
        # to rule prose only if the judge's reasoning came back empty
        # (rare — would mean the model produced an answer with no prose).
        # In no_judge_keep_polish mode we always go to rule_fallback so
        # the polish step rewrites a faithful GT-aligned draft.
        if (no_judge_keep_polish or skip_judge or skip_llm_judge
                or not gemini.reasoning.strip()):
            cot_source = "geometry_rule" if skip_llm_judge else "rule_fallback"
            raw_cot = _rule_based_ranking_cot(
                chain, gt_ranking_labels, perm, mode=meta.get("mode", "cumulative")
            )
        else:
            cot_source = "llm_judge"
            raw_cot = gemini.reasoning.strip()

        polished = _scrub_coordinates(raw_cot)
        # Polish runs whenever a client is available (judge skipped does
        # NOT imply polish skipped — we still want natural-language CoT
        # aligned to the rule GT). max_tokens=4000 matches the budget
        # used by generate_n4_reasoning.py because n=4 cum records list
        # up to 3 perturbations × 4 scenes; the default 2400 truncated
        # the trailing scene description on dense chains.
        if client is not None:
            polished = polish_ranking_reasoning(
                polished, ranking=gt_ranking_labels,
                perturbation_summary=full_chain_summary,
                client=client,
                record_id=ranking_id,
                max_tokens=4000,
            )
            polished = _scrub_coordinates(polished)
        completion = (
            f"<reasoning>\n{polished}\n</reasoning>\n"
            f"<answer>{''.join(gt_ranking_labels)}</answer>"
        )

        record = {
            "id": ranking_id,
            "format": "sft_nway_ranking",
            "scene_dir": str(raw_scene_dir),
            "n": size,
            "images": {labels[i]: str(sft_imgs[i]) for i in range(size)},
            "judge_images": {labels[i]: str(judge_imgs[i]) for i in range(size)},
            "ground_truth": {
                "ranking_best_to_worst": gt_ranking_labels,
                "chain_index_at_label": {labels[i]: perm[i] for i in range(size)},
                "ranking_mode": meta.get("mode", "cumulative"),
                "tie_sets_chain_indices": [sorted(s) for s in meta.get("tie_set", [])],
                # One perturbation per chain step (1..size-1). For
                # cumulative mode this is "what was newly broken at this
                # chain step"; for sibling mode it is "this sibling's
                # perturbation". Use `perturbations_by_label` below for the
                # per-scene view.
                "perturbations": [
                    {
                        "kind": p.kind,
                        "object_id": p.object_id,
                        "gt_strength": p.gt_strength,
                        "description": p.description,
                        "reason_bad": p.reason_bad,
                        "secondary_change": p.secondary_change,
                        "original_pose": list(p.original_pose) if p.original_pose else None,
                        "new_pose": list(p.new_pose) if p.new_pose else None,
                    }
                    for p in perts_per_step
                ],
                "perturbations_by_label": {
                    labels[i]: [
                        {
                            "kind": p.kind,
                            "object_id": p.object_id,
                            "gt_strength": p.gt_strength,
                            "description": p.description,
                            "reason_bad": p.reason_bad,
                            "secondary_change": p.secondary_change,
                        }
                        for p in chain[perm[i]][1]
                    ]
                    for i in range(size)
                },
            },
            "gemini": {
                "model": DEFAULT_JUDGE_MODEL,
                "raw": gemini.raw,
                "reasoning": gemini.reasoning,
                "ranking": gemini.ranking,
                "kendall_tau_distance": kt,
                "called": not (skip_judge or skip_llm_judge or no_judge_keep_polish),
                "matches_ground_truth": (gemini.ranking == gt_ranking_labels) if not (skip_judge or skip_llm_judge or no_judge_keep_polish) else None,
            },
            "cot_source": cot_source,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "path": str(sft_imgs[i]),
                          "tag": f"SCENE_{labels[i]}"} for i in range(size)]
                        + [{"type": "text", "text": _ranking_user_text(size, chain[-1][1])}]
                    ),
                },
                {"role": "assistant", "content": completion},
            ],
        }
        (ranking_dir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        records.append(record)
        if sft_fh is not None:
            sft_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            sft_fh.flush()
        print(f"[ok] {ranking_id}  gt={'>'.join(gt_ranking_labels)}  "
              f"gemini={'>'.join(gemini.ranking) if gemini.ranking else '?'}  "
              f"kt={kt}  {verdict}")
    return records


def _visible_perturbation_phrase(p: Perturbation) -> str:
    """A description of what someone looking ONLY at the SFT image sees.
    Keeps the COT free of privileged info (no coordinates, yaw degrees,
    references to the perturbation kind names, etc.)."""
    obj = f"`{p.object_id}`"
    if p.kind == "move_into_wall":
        return f"{obj} is pushed past the room boundary, with part of its footprint outside the wall — physically invalid."
    if p.kind == "move_blocking_door":
        return f"{obj} sits directly in the doorway, blocking entry/exit through the door."
    if p.kind == "move_overlap":
        sec = p.secondary_change.get("object_id") if p.secondary_change else None
        return f"{obj} is placed on top of another piece, with overlapping footprints — physically invalid."
    if p.kind == "face_wall":
        return f"{obj}'s FRONT arrow points directly at a wall, so its functional face (shelves / cushions / screen) is unusable."
    if p.kind == "rotate_180":
        return f"{obj} is flipped, so its back is toward the room and its front toward what should be the back side — using it from the room is no longer possible."
    if p.kind == "rotate_45_offset":
        return f"{obj} is rotated off-axis, sitting at an awkward angle that doesn't align with the room's main directions."
    if p.kind == "rotate_out_of_room":
        return f"{obj} is rotated to an angle where part of its footprint extends past a room wall — physically invalid."
    if p.kind == "rotate_into_collision":
        return f"{obj} is rotated to an angle where its footprint now overlaps a neighbouring piece — physically invalid."
    if p.kind == "move_off_wall":
        return f"{obj} is wall-storage that has been pulled into the centre of the room, no longer anchored against any wall."
    if p.kind == "move_far_from_focal":
        return f"{obj} has been moved away from its functional partner, breaking the natural seating / dining / sleeping group."
    if p.kind == "move_to_corner":
        return f"{obj} is jammed into a far corner, isolated from where it would actually be used."
    if p.kind == "swap_two":
        sec = p.secondary_change.get("object_id") if p.secondary_change else "another piece"
        return f"{obj} and `{sec}` have exchanged positions — both pieces now sit in the wrong functional zone for their type."
    if p.kind == "remove_object":
        return f"{obj} is missing from the room entirely; the layout is incomplete without it."
    return f"{obj} has been moved or rotated in a way that breaks the original layout."


def _rule_based_ranking_cot(
    chain,
    gt_ranking: list[str],
    perm: list[int],
    *,
    mode: str = "cumulative",
) -> str:
    """Pure-text COT that the SFT model could in principle infer from the
    images alone. We avoid mentioning kind names, coordinates, yaw degrees,
    or anything that wouldn't be visible to the model."""
    n = len(chain)
    pieces = []
    if mode == "sibling":
        pieces.append(
            f"All {n} scenes share the same room layout, but each one (other than the "
            "original) has ONE different thing changed. Comparing the visible state "
            "of each scene against the natural layout gives a clear ranking from "
            "most coherent to least coherent."
        )
        for k in range(n):
            label = gt_ranking[k]
            if k == 0:
                pieces.append(
                    f"Scene {label} is the most coherent: every furniture piece "
                    "sits where you'd expect it; storage and TV/console pieces have "
                    "their backs to walls; seating faces the coffee table; the door "
                    "swing is clear."
                )
            else:
                p = chain[k][1][-1]
                pieces.append(f"Scene {label} is worse than the better-ranked scenes — {_visible_perturbation_phrase(p)}")
    else:
        pieces.append(
            f"All {n} scenes share the same room geometry, but each one in the "
            "ranking has progressively MORE issues than the previous: the original "
            "is the most coherent, and each subsequent scene introduces another "
            "fault on top of the earlier ones."
        )
        for k in range(1, n):
            cur_perts = chain[k][1]
            last = cur_perts[-1]
            label = gt_ranking[k]
            pieces.append(f"In Scene {label}, in addition to the issues above, {_visible_perturbation_phrase(last)}")
    pieces.append(f"Ranking from best to worst: {' > '.join(gt_ranking)}.")
    return " ".join(pieces)


def _ranking_user_text(size: int, perturbations: list[Perturbation]) -> str:
    labels = ["A", "B", "C", "D"][:size]
    return (
        f"You are a scene-quality judge. {size} top-down photoreal renderings of "
        f"the same room are shown as SCENE {', SCENE '.join(labels)}. They share "
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


def main(argv: list[str] | None = None) -> int:
    global DEFAULT_PER_STEP_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                    help="Directory containing scene_xxx subfolders.")
    ap.add_argument("--per-step-dir", type=Path, default=DEFAULT_PER_STEP_DIR,
                    help="Root of scene/room/furniture_*.blend snapshots.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Where to write per-pair folders + sft.jsonl.")
    ap.add_argument("--scene", type=str, default=None,
                    help="Single scene folder name. If omitted, all scenes are processed.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most this many scenes (0 = all).")
    ap.add_argument("--n-pairs", type=int, default=2,
                    help="Number of (good, bad) pairs to generate per scene.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-judge", action="store_true",
                    help="Skip the Gemini API call (just render + write GT).")
    ap.add_argument("--no-judge-keep-polish", action="store_true",
                    help="Skip the LLM judge call but still run polish_ranking_reasoning "
                         "on the rule-based draft. Use when the rule GT is fully "
                         "authoritative (e.g. cum chains under same-object-forbidden "
                         "rule) so judge agreement no longer filters useful samples.")
    ap.add_argument("--cum-from-n2-pool", action="store_true",
                    help="For cum chains, draw each step's perturbation from the n2 "
                         "verified pool (gemini.matches_ground_truth=True) for the "
                         "same base scene, replaying recorded pose. Skips live "
                         "_GENERATORS sampling; same-object reuse is forbidden, so "
                         "each replay lands on an untouched piece.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip scenes whose pair_00 record.json already exists in --out-dir.")
    ap.add_argument("--no-blender", action="store_true",
                    help="Use the matplotlib schematic renderer instead of Blender.")
    ap.add_argument("--blender-resolution", type=int, default=1024,
                    help="Pixel resolution for Blender top-downs (square).")
    ap.add_argument("--keep-disagreements", action="store_true",
                    help="By default, pairs where Gemini's verdict disagrees "
                         "with our ground truth are skipped (and their pair_dir "
                         "deleted). Set this flag to keep them with rule-fallback "
                         "COTs.")
    ap.add_argument("--force-kinds", type=str, default=None,
                    help="Comma-separated list of perturbation kinds to use, in "
                         "order. Overrides random sampling: pair_NN uses the "
                         "kind at index NN. n_pairs is set to len(kinds) when "
                         "this flag is provided.")
    ap.add_argument("--ranking-sizes", type=str, default=None,
                    help="Switch to RANKING MODE. Comma-separated list of "
                         "`size:weight` pairs, e.g. `3:1,4:1` for uniform 3-way "
                         "vs 4-way. Sizes must be in [2, 4].")
    ap.add_argument("--n-rankings", type=int, default=2,
                    help="In ranking mode: number of rankings to generate per scene.")
    ap.add_argument("--ranking-modes", type=str, default="cumulative:1,sibling:1",
                    help="Comma-separated `mode:weight` pairs. modes: cumulative, sibling.")
    ap.add_argument("--kind-weights", type=str, default=None,
                    help="Comma-separated list of `kind:weight` pairs to override "
                         "default sampling weights, e.g. "
                         "`face_wall:1,rotate_180:1,...`. Unspecified kinds keep "
                         "their default weight.")
    ap.add_argument("--sft-name", type=str, default="sft.jsonl",
                    help="Filename of the SFT JSONL inside --out-dir. Each "
                         "parallel worker must use a unique name (e.g. "
                         "sft_w01.jsonl) so concurrent appends don't interleave.")
    args = ap.parse_args(argv)

    DEFAULT_PER_STEP_DIR = args.per_step_dir.resolve()
    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    forced_kinds = None
    if args.force_kinds:
        forced_kinds = [k.strip() for k in args.force_kinds.split(",") if k.strip()]
        if args.n_pairs != len(forced_kinds):
            print(f"[force-kinds] overriding n_pairs from {args.n_pairs} → {len(forced_kinds)}",
                  file=sys.stderr)
            args.n_pairs = len(forced_kinds)

    # Parse --kind-weights "face_wall:1,rotate_180:1,..." into a dict
    kind_weights = None
    if args.kind_weights:
        kind_weights = {}
        for tok in args.kind_weights.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" not in tok:
                raise ValueError(f"--kind-weights item missing ':' weight: {tok}")
            k, w = tok.split(":", 1)
            kind_weights[k.strip()] = float(w)

    # Parse --ranking-modes "cumulative:1,sibling:1"
    mode_dist: list[tuple[str, float]] | None = None
    if args.ranking_modes:
        mode_dist = []
        for tok in args.ranking_modes.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                m, w = tok.split(":", 1)
                mode_dist.append((m.strip(), float(w)))
            else:
                mode_dist.append((tok, 1.0))
        for m, _ in mode_dist:
            if m not in ("cumulative", "sibling"):
                raise ValueError(f"unknown ranking mode: {m}")

    # Parse --ranking-sizes "3:1,4:1" — switches to ranking mode.
    ranking_size_dist = None
    if args.ranking_sizes:
        ranking_size_dist = []
        for tok in args.ranking_sizes.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                s, w = tok.split(":", 1)
                ranking_size_dist.append((int(s), float(w)))
            else:
                ranking_size_dist.append((int(tok), 1.0))
        for s, _ in ranking_size_dist:
            if not 2 <= s <= 4:
                raise ValueError(f"ranking size {s} not in [2, 4]")

    scene_dirs: list[Path]
    if args.scene:
        scene_dirs = [raw_dir / args.scene]
    else:
        scene_dirs = sorted([p for p in raw_dir.iterdir()
                             if p.is_dir() and not p.name.startswith("_")])
    # Apply blocklist (raw_dir/_blocklist.txt) — one scene name per line.
    blocklist_path = raw_dir / "_blocklist.txt"
    if blocklist_path.exists():
        blocked = set()
        for line in blocklist_path.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                blocked.add(s)
        if blocked:
            before = len(scene_dirs)
            scene_dirs = [p for p in scene_dirs if p.name not in blocked]
            print(f"[blocklist] excluded {before - len(scene_dirs)} scene(s); "
                  f"{len(scene_dirs)} remaining", file=sys.stderr)
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]

    # We need the API client whenever we want polish (so any time
    # skip-judge is NOT fully on its own). The flag combinations:
    #   --skip-judge alone           → no client, no judge, no polish
    #   --no-judge-keep-polish       → client (for polish), no judge
    #   (neither flag)               → client, judge runs
    client = None if (args.skip_judge and not args.no_judge_keep_polish) else make_client()

    sft_path = out_dir / args.sft_name
    # Pass an open file handle through to the per-scene loop so each
    # finished record is appended immediately. This matters because a
    # single scene with 60 rankings can take an hour+, and we want
    # downstream training to start consuming partial output.
    sft_fh = sft_path.open("a", encoding="utf-8")
    # Build the de-duplication fingerprint set ONCE per process. Each
    # record we accept adds its FP so within-process dupes are blocked;
    # cross-process dupes are bounded because every worker re-reads
    # sft.jsonl + shards here and we re-read at the start of each scene
    # below (cheap — at most a few thousand records).
    fingerprint_set = _load_existing_fingerprints(out_dir)
    print(f"[dedup] loaded {len(fingerprint_set)} fingerprints", file=sys.stderr)
    n2_pool_by_scene: dict[str, list[dict]] | None = None
    if args.cum_from_n2_pool:
        n2_pool_by_scene = _load_n2_verified_pool(out_dir)
        total_p = sum(len(v) for v in n2_pool_by_scene.values())
        print(f"[pool] loaded n2 verified pool: {len(n2_pool_by_scene)} scenes, "
              f"{total_p} perturbations total", file=sys.stderr)
    n_errors = 0
    n_written = 0
    try:
        for i, sd in enumerate(scene_dirs):
            if args.resume and (out_dir / f"{sd.name}__pair_00" / "record.json").exists():
                print(f"[skip] {sd.name}: already done (--resume)")
                continue
            seed = args.seed + 1000 * i
            try:
                if ranking_size_dist is not None:
                    # Look up this scene's verified n2 pool (only when
                    # the cum-from-pool flag is on).
                    scene_pool = None
                    if n2_pool_by_scene is not None:
                        scene_pool = n2_pool_by_scene.get(sd.name)
                        if not scene_pool:
                            print(f"[pool] {sd.name}: no verified n2 perturbations — "
                                  f"skipping this scene", file=sys.stderr)
                            continue
                    recs = process_scene_ranking(
                        sd, out_dir,
                        n_rankings=args.n_rankings,
                        seed=seed,
                        skip_judge=args.skip_judge,
                        ranking_size_dist=ranking_size_dist,
                        kind_weights=kind_weights,
                        mode_dist=mode_dist,
                        use_blender=not args.no_blender,
                        blender_resolution=args.blender_resolution,
                        drop_on_disagreement_soft_only=not args.keep_disagreements,
                        client=client,
                        sft_fh=sft_fh,
                        fingerprint_set=fingerprint_set,
                        no_judge_keep_polish=args.no_judge_keep_polish,
                        cum_pool=scene_pool,
                    )
                else:
                    recs = process_scene(
                        sd, out_dir,
                        n_pairs=args.n_pairs,
                        seed=seed,
                        skip_judge=args.skip_judge,
                        use_blender=not args.no_blender,
                        blender_resolution=args.blender_resolution,
                        drop_on_disagreement=not args.keep_disagreements,
                        forced_kinds=forced_kinds,
                        client=client,
                    )
                    for r in recs:
                        sft_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    sft_fh.flush()
                n_written += len(recs)
            except Exception as e:
                n_errors += 1
                print(f"[scene_error] {sd.name}: {e!r}", file=sys.stderr)
                traceback.print_exc()
                continue
    finally:
        sft_fh.close()
    print(f"\nWrote SFT records to {sft_path}")
    return 1 if n_errors or (not n_written and not args.resume) else 0


if __name__ == "__main__":
    sys.exit(main())
