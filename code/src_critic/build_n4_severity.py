"""Generate n4 training records along two new severity-discrimination axes:

  * **Type 1 (multikind):** same target object, 4 different perturbation
    KINDS chosen so their LLM-derived severities are well-separated.
    Teaches the model that kind-of-fault matters.

  * **Type 2 (multitarget):** same kind, applied to 4 different TARGET
    objects whose LLM-derived severities under that kind are
    well-separated. Teaches the model which piece is functionally
    important.

GT is the rank order of `severity(kind, obj)` from the
`_severity_score.py` scalar; selection is constrained to gap ≥ 0.15 so
that ties (which give no discriminable training signal) are dropped.

Output records share the same SFT format as `build_n4_from_n2.py`
(messages with 4 images + prompt; deterministic ranking_best_to_worst).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

from .build_dataset import (
    DEFAULT_RAW_DIR,
    _find_latest_blend,
    _per_step_room_dir,
    _perturbation_payload,
    _run_blender_render,
)
from .build_n4_from_n2 import (
    OUT_DIR,
    _apply_chain,
    _orig_xy_map,
    _pert_to_dict,
    _physics_ok,
    _sft_prompt,
)
from .perturb import _GENERATORS, Perturbation, target_eligible_for_kind
from .pil_annotate import annotate_topdown
from .scene_io import load_scene
from .severity_score import (
    _KIND_BASE_SCORE,
    enumerate_3_tuples_above_anchor,
    enumerate_4_tuples,
    load_properties,
    object_severity_for_kind,
    pick_4_max_gap,
    severity,
)


# Kinds eligible for Type 1 / Type 2. Excluded:
#   * swap_two (2-object kind doesn't fit the (kind, single_obj) formula)
#   * move_into_wall / move_overlap / rotate_into_collision /
#     rotate_out_of_room (physics violations)
#   * move_blocking_door (per user directive: don't synthesize new
#     door-blocking records going forward)
ELIGIBLE_KINDS = (
    # Per user directive: multitarget only uses these 3 kinds.
    # Others excluded because their severity depends heavily on
    # the target's own properties (e.g. rotate_180 severity tied
    # to the object's directionality), making cross-object
    # severity comparison unreliable. The 3 kept kinds rank
    # cleanly by object importance alone.
    "remove_object", "move_to_corner", "move_far_from_focal",
)


# -------------------- Per-scene blend context cache --------------------

def _scene_ctx(scene_id: str, resolution: int) -> tuple | None:
    """Replicate build_n4_from_n2._scene_ctx logic. Returns
    (scene, blend_path, room_box, cached_orig_png, cam_meta) or None on
    error. Caches the unperturbed render under
    `datasets/critic_pairs/_renders/<scene_id>/original.png`."""
    raw_scene_dir = DEFAULT_RAW_DIR / scene_id
    if not raw_scene_dir.exists():
        return None
    try:
        scene = load_scene(raw_scene_dir)
    except Exception as e:
        print(f"[skip] {scene_id}: load_scene: {e!r}", file=sys.stderr)
        return None
    ps_dir = _per_step_room_dir(raw_scene_dir, scene)
    blend_path = _find_latest_blend(ps_dir)
    if blend_path is None:
        return None
    room_box = (scene.room.x_min, scene.room.y_min,
                scene.room.x_max, scene.room.y_max)
    cache_dir = OUT_DIR / "_renders" / scene_id
    cached_orig = cache_dir / "original.png"
    cached_meta = cache_dir / "original.meta.json"
    if not (cached_orig.exists() and cached_meta.exists()):
        try:
            meta = _run_blender_render(
                blend_path, cached_orig,
                perturbations=[], room_box=room_box,
                resolution=resolution,
            )
            cached_meta.write_text(json.dumps(meta["camera"]))
        except Exception as e:
            print(f"[skip] {scene_id}: original render fail: {e!r}", file=sys.stderr)
            return None
    cam_meta = json.loads(cached_meta.read_text())
    return (scene, blend_path, room_box, cached_orig, cam_meta)


def _try_generate(scene, kind: str, target_obj_id: str, rng) -> Perturbation | None:
    """Run the perturb generator for (kind, target) with retries until
    we get a physics-valid Perturbation MATCHING `kind` or fall through
    `max_tries`.

    Important: `_emit_with_upgrade` inside the soft generators can
    silently change the kind (e.g. `move_to_corner` → `move_blocking_door`
    when the picked corner overlaps a door's clearance band). For
    multitarget records we need EVERY B/C/D to share the requested
    kind, so any mismatched return is discarded."""
    gen = _GENERATORS.get(kind)
    if gen is None:
        return None
    target = scene.by_id(target_obj_id)
    if target is None:
        return None
    for _ in range(40):
        out = gen(scene, target, rng)
        if out is None:
            continue
        if isinstance(out, tuple):
            mut_scene, p = out
        else:
            p = out
        if p.kind != kind:
            # Soft-kind generator upgraded into a tier-1 violation kind —
            # not what we asked for; retry.
            continue
        if not _physics_ok(scene, p):
            continue
        return p
    return None


def _render_and_annotate(
    *,
    record_dir: Path,
    label: str,
    is_original: bool,
    scene_with_pert,
    perts_at_label: list[Perturbation],
    blend_path: Path,
    cached_orig: Path,
    cam_meta_orig: dict,
    room_box: tuple,
    orig_xy: dict,
    resolution: int,
) -> tuple[Path, Path]:
    """Render one of the 4 scenes (raw png + sft annotated + judge
    annotated). Returns (sft_path, judge_path)."""
    sft_path = record_dir / f"scene_{label}.png"
    judge_path = record_dir / f"scene_{label}_judge.png"

    if is_original:
        raw_png = cached_orig
        camera_meta = cam_meta_orig
    else:
        raw_png = record_dir / f"_raw_{label}.png"
        payload = []
        for p in perts_at_label:
            payload.extend(_perturbation_payload(p, orig_xy=orig_xy))
        meta = _run_blender_render(
            blend_path, raw_png,
            perturbations=payload, room_box=room_box,
            resolution=resolution,
        )
        camera_meta = meta["camera"]

    annotate_topdown(
        raw_png, scene=scene_with_pert, camera_meta=camera_meta,
        highlight_ids=[], out_path=sft_path,
    )
    hi = [p.object_id for p in perts_at_label]
    if perts_at_label:
        last_p = perts_at_label[-1]
        if last_p.secondary_change:
            hi.append(last_p.secondary_change["object_id"])
    rot_kinds = ("rotate_45_offset", "rotate_180", "face_wall")
    annotate_topdown(
        raw_png, scene=scene_with_pert, camera_meta=camera_meta,
        highlight_ids=hi, out_path=judge_path,
        rotation_highlight_ids=hi if any(p.kind in rot_kinds for p in perts_at_label) else [],
    )
    return sft_path, judge_path


def _build_sft_record(
    record_id: str,
    record_dir: Path,
    ranked_perts: list[list[Perturbation]],  # length 4: A..D, A is [] (anchor)
    mode_label: str,
    sft_images: dict[str, Path],
    judge_images: dict[str, Path],
    severity_per_label: dict[str, float],
) -> dict:
    """Same shape as build_n4_from_n2 records but `mode_label` is
    'multikind' or 'multitarget' and we stash per-label severity so
    downstream tooling can sanity-check the ordering."""
    labels = ["A", "B", "C", "D"]
    content = []
    for lab in labels:
        content.append({"type": "image", "path": str(sft_images[lab])})
    content.append({"type": "text", "text": _sft_prompt(4)})

    perts_by_label = {lab: [_pert_to_dict(p) for p in ranked_perts[i]]
                      for i, lab in enumerate(labels)}
    # Flatten the last scene's perts for the legacy `perturbations` field
    perts_flat = [_pert_to_dict(p) for p in ranked_perts[-1]]

    return {
        "id": record_id,
        "format": "ranking_n4",
        "scene_dir": str(record_dir.relative_to(OUT_DIR.parent.parent)),
        "n": 4,
        "images": {lab: str(sft_images[lab]) for lab in labels},
        "judge_images": {lab: str(judge_images[lab]) for lab in labels},
        "ground_truth": {
            "ranking_best_to_worst": labels.copy(),
            "chain_index_at_label": {lab: i for i, lab in enumerate(labels)},
            "ranking_mode": mode_label,
            "tie_sets_chain_indices": [],
            "perturbations": perts_flat,
            "perturbations_by_label": perts_by_label,
            "severity_per_label": severity_per_label,
        },
        "gemini": {
            "model": "n4_severity_rule",
            "raw": "",
            "reasoning": "(deterministic GT: severity gap >= 0.15)",
            "ranking": labels.copy(),
        },
        "cot_source": "deterministic",
        "messages": [{"role": "user", "content": content}],
    }


def _render_4_scenes(
    *,
    ctx,
    record_dir: Path,
    perts_per_label: list[list[Perturbation]],
    resolution: int,
) -> tuple[dict[str, Path], dict[str, Path]] | None:
    """Render A/B/C/D scenes (anchor + 3 perturbed). Returns
    (sft_images, judge_images) or None on render failure."""
    scene, blend_path, room_box, cached_orig, cam_meta = ctx
    orig_xy = _orig_xy_map(scene)
    sft_imgs: dict[str, Path] = {}
    judge_imgs: dict[str, Path] = {}
    labels = ["A", "B", "C", "D"]
    record_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, lab in enumerate(labels):
            cur = _apply_chain(scene, perts_per_label[i]) or scene
            sft_p, judge_p = _render_and_annotate(
                record_dir=record_dir, label=lab,
                is_original=(i == 0),
                scene_with_pert=cur,
                perts_at_label=perts_per_label[i],
                blend_path=blend_path, cached_orig=cached_orig,
                cam_meta_orig=cam_meta, room_box=room_box,
                orig_xy=orig_xy, resolution=resolution,
            )
            sft_imgs[lab] = sft_p
            judge_imgs[lab] = judge_p
        return sft_imgs, judge_imgs
    except Exception as e:
        print(f"[render_fail] {record_dir.name}: {e!r}", file=sys.stderr)
        shutil.rmtree(record_dir, ignore_errors=True)
        return None


# -------------------- TYPE 1: same target, multi-kind --------------------

def generate_multikind(
    *, target_count: int, resolution: int, seed: int,
    out_jsonl: Path, min_gap: float = 0.15,
) -> int:
    rng = random.Random(seed)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fh = out_jsonl.open("a", encoding="utf-8")

    scene_dirs = sorted(p for p in DEFAULT_RAW_DIR.iterdir()
                        if p.is_dir() and not p.name.startswith("_"))
    # Apply blocklist
    bl = DEFAULT_RAW_DIR / "_blocklist.txt"
    if bl.exists():
        blocked = {ln.strip() for ln in bl.read_text().splitlines()
                   if ln.strip() and not ln.strip().startswith("#")}
        scene_dirs = [p for p in scene_dirs if p.name not in blocked]
    rng.shuffle(scene_dirs)

    n_emit = 0
    seed_counter = 0
    for scene_path in scene_dirs:
        if n_emit >= target_count:
            break
        scene_id = scene_path.name
        props = load_properties(scene_path)
        if not props:
            continue

        # Score per object × kind
        candidates_by_obj = {}
        for obj_id in props.keys():
            row = {k: severity(k, props[obj_id]) for k in ELIGIBLE_KINDS}
            sel = pick_4_max_gap(row, min_gap=min_gap)
            if sel is not None:
                candidates_by_obj[obj_id] = sel

        if not candidates_by_obj:
            continue

        ctx = _scene_ctx(scene_id, resolution)
        if ctx is None:
            continue
        scene = ctx[0]

        # For each target object that yielded a 4-tuple, try to actually
        # generate the 3 perturbations (anchor stays empty)
        objs = list(candidates_by_obj.keys())
        rng.shuffle(objs)
        for obj_id in objs:
            if n_emit >= target_count:
                break
            chosen_kinds = candidates_by_obj[obj_id]  # ordered by ascending severity
            # Generate Perturbation objects for B/C/D (skip A = anchor)
            perts_BCD: list[Perturbation] = []
            for k in chosen_kinds[1:]:
                p = _try_generate(scene, k, obj_id, rng)
                if p is None:
                    break
                perts_BCD.append(p)
            if len(perts_BCD) != 3:
                continue
            perts_per_label = [[], [perts_BCD[0]], [perts_BCD[1]], [perts_BCD[2]]]
            seed_counter += 1
            record_id = (
                f"{scene_id}__ranking_s{seed_counter:06d}_n4_multikind"
            )
            record_dir = OUT_DIR / record_id
            render = _render_4_scenes(
                ctx=ctx, record_dir=record_dir,
                perts_per_label=perts_per_label, resolution=resolution,
            )
            if render is None:
                continue
            sft_imgs, judge_imgs = render
            sev_per_label = {
                lab: round(severity(chosen_kinds[i], props[obj_id]), 3)
                for i, lab in enumerate(("A", "B", "C", "D"))
            }
            sev_per_label["A"] = 0.0  # anchor severity is 0 by definition
            rec = _build_sft_record(
                record_id, record_dir, perts_per_label,
                mode_label="multikind", sft_images=sft_imgs,
                judge_images=judge_imgs, severity_per_label=sev_per_label,
            )
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_emit += 1
            if n_emit % 5 == 0:
                print(f"[multikind] {n_emit}/{target_count}", file=sys.stderr)

    fh.close()
    print(f"[done multikind] {n_emit}/{target_count}", file=sys.stderr)
    return n_emit


# -------------------- TYPE 2: same kind, multi-target --------------------

def generate_multitarget(
    *, target_count: int, resolution: int, seed: int,
    out_jsonl: Path, min_gap: float = 0.15,
    max_tuples_per_kind: int = 20,
    min_distinct: int = 1,
    existing_fingerprints: set | None = None,
    seed_counter_offset: int = 0,
) -> int:
    rng = random.Random(seed + 1)  # avoid colliding with multikind seed
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fh = out_jsonl.open("a", encoding="utf-8")
    skipped_existing = 0

    scene_dirs = sorted(p for p in DEFAULT_RAW_DIR.iterdir()
                        if p.is_dir() and not p.name.startswith("_"))
    bl = DEFAULT_RAW_DIR / "_blocklist.txt"
    if bl.exists():
        blocked = {ln.strip() for ln in bl.read_text().splitlines()
                   if ln.strip() and not ln.strip().startswith("#")}
        scene_dirs = [p for p in scene_dirs if p.name not in blocked]
    rng.shuffle(scene_dirs)

    n_emit = 0
    seed_counter = seed_counter_offset
    for scene_path in scene_dirs:
        if n_emit >= target_count:
            break
        scene_id = scene_path.name
        props = load_properties(scene_path)
        if not props:
            continue

        # For each kind, enumerate up to `max_tuples_per_kind` diverse
        # 4-object subsets (sorted by bottleneck gap desc, each ≥2 obj
        # apart from previously picked). Use OBJECT-LEVEL severity so
        # the kind-base offset doesn't compress the available range.
        # Per-kind eligibility filter: a target the perturb generator
        # will synchronously reject (e.g. move_off_wall on a sofa, since
        # sofa.name has no WALL_HUGGING_TOKENS) is excluded BEFORE the
        # 4-tuple search — otherwise the picked tuple will lose 1-3
        # targets to generator-None and the whole candidate is dropped.
        # We need a scene object to do scene-level eligibility (doors).
        scene_for_filter = None  # lazy
        candidates_by_kind: dict[str, list[list[str]]] = {}
        for k in ELIGIBLE_KINDS:
            # Lazy-load the scene only when we need it (most filters are
            # target-only; move_blocking_door is the only scene-level one).
            if scene_for_filter is None:
                ctx_tmp = _scene_ctx(scene_id, resolution)
                if ctx_tmp is None:
                    break
                scene_for_filter = ctx_tmp[0]
            eligible_obj_ids = [
                oid for oid in props
                if scene_for_filter.by_id(oid) is not None
                and target_eligible_for_kind(scene_for_filter,
                                             scene_for_filter.by_id(oid), k)
            ]
            if len(eligible_obj_ids) < 4:
                continue
            row = {oid: object_severity_for_kind(k, props[oid])
                   for oid in eligible_obj_ids}
            # Return 3-tuples of B/C/D objects (anchor is implicit at
            # severity 0 — never rendered as a target, so it must NOT
            # consume a slot in the candidate enumeration).
            tuples = enumerate_3_tuples_above_anchor(
                row, min_gap=min_gap,
                max_tuples=max_tuples_per_kind, min_distinct=min_distinct,
            )
            if tuples:
                candidates_by_kind[k] = tuples

        if not candidates_by_kind:
            continue

        ctx = _scene_ctx(scene_id, resolution)
        if ctx is None:
            continue
        scene = ctx[0]

        # Flatten to (kind, tuple) pairs and shuffle so different runs
        # explore different orderings.
        all_tasks = [(kind, tup) for kind, tups in candidates_by_kind.items()
                     for tup in tups]
        rng.shuffle(all_tasks)

        # In-session dedup so the same 3-object set within a (scene, kind)
        # isn't emitted twice if enumerate_3_tuples_above_anchor's
        # min_distinct allows close variants. Tracks fingerprint =
        # (scene, kind, sorted tuple of 3 B/C/D objects).
        session_fps: set = set()
        for kind, chosen_objs in all_tasks:
            if n_emit >= target_count:
                break
            # chosen_objs is now exactly 3 objects (B, C, D), already
            # sorted ascending by severity.
            fp = (scene_id, kind, tuple(sorted(chosen_objs)))
            if fp in session_fps:
                continue  # already emitted in this scene-loop iteration
            if existing_fingerprints is not None and fp in existing_fingerprints:
                skipped_existing += 1
                continue
            perts_BCD: list[Perturbation] = []
            for obj_id in chosen_objs:
                p = _try_generate(scene, kind, obj_id, rng)
                if p is None:
                    break
                perts_BCD.append(p)
            if len(perts_BCD) != 3:
                continue
            session_fps.add(fp)
            perts_per_label = [[], [perts_BCD[0]], [perts_BCD[1]], [perts_BCD[2]]]
            seed_counter += 1
            record_id = (
                f"{scene_id}__ranking_s{seed_counter:06d}_n4_multitarget"
            )
            record_dir = OUT_DIR / record_id
            render = _render_4_scenes(
                ctx=ctx, record_dir=record_dir,
                perts_per_label=perts_per_label, resolution=resolution,
            )
            if render is None:
                continue
            sft_imgs, judge_imgs = render
            # chosen_objs is now length 3 (B/C/D). A is the anchor at 0.
            sev_per_label = {
                "A": 0.0,
                "B": round(object_severity_for_kind(kind, props[chosen_objs[0]]), 3),
                "C": round(object_severity_for_kind(kind, props[chosen_objs[1]]), 3),
                "D": round(object_severity_for_kind(kind, props[chosen_objs[2]]), 3),
            }
            rec = _build_sft_record(
                record_id, record_dir, perts_per_label,
                mode_label="multitarget", sft_images=sft_imgs,
                judge_images=judge_imgs, severity_per_label=sev_per_label,
            )
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_emit += 1
            if n_emit % 10 == 0:
                print(f"[multitarget] {n_emit}/{target_count}", file=sys.stderr)

    fh.close()
    print(f"[done multitarget] {n_emit}/{target_count} "
          f"(skipped {skipped_existing} pre-existing fingerprints)",
          file=sys.stderr)
    return n_emit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("multikind", "multitarget", "both"),
                    default="both")
    ap.add_argument("--target", type=int, default=50,
                    help="number of records per mode")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--out-jsonl", type=Path,
                    default=OUT_DIR / "sft_n4_severity.jsonl")
    ap.add_argument("--min-gap", type=float, default=0.15)
    ap.add_argument("--existing-fps-from", type=Path, default=None,
                    help="If set, read all records from this jsonl and "
                         "skip generating any combo whose fingerprint "
                         "already appears there.")
    ap.add_argument("--seed-counter-offset", type=int, default=0,
                    help="Initial value for the per-record id counter.")
    args = ap.parse_args()
    existing_fps = None
    if args.existing_fps_from is not None and args.existing_fps_from.exists():
        existing_fps = set()
        with args.existing_fps_from.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rr = json.loads(line)
                if not rr["id"].endswith("_n4_multitarget"):
                    continue
                scene = rr["id"].split("__")[0]
                plb = rr["ground_truth"]["perturbations_by_label"]
                # Anchor is the label whose perts list is empty;
                # the other 3 labels each have 1 perturbation.
                kind = None
                objs = []
                for lab, perts in plb.items():
                    if perts:
                        kind = perts[0]["kind"]
                        objs.append(perts[0]["object_id"])
                if kind is not None and len(objs) == 3:
                    existing_fps.add((scene, kind, tuple(sorted(objs))))
        print(f"[fp-prefilter] loaded {len(existing_fps)} existing "
              f"fingerprints from {args.existing_fps_from}", file=sys.stderr)
    if args.mode in ("multikind", "both"):
        generate_multikind(
            target_count=args.target, resolution=args.resolution,
            seed=args.seed, out_jsonl=args.out_jsonl,
            min_gap=args.min_gap,
        )
    if args.mode in ("multitarget", "both"):
        generate_multitarget(
            target_count=args.target, resolution=args.resolution,
            seed=args.seed, out_jsonl=args.out_jsonl,
            min_gap=args.min_gap,
            existing_fingerprints=existing_fps,
            seed_counter_offset=args.seed_counter_offset,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
