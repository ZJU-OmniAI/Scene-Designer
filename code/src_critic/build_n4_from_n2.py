"""Build 4-option (n=4) training records by extending the existing
confirmed 2-option (n=2) records. Two modes:

 1. `param_sweep` (n4_psw): same kind, same target, 4 parameter values
    (original + 3 interpolated). GT order = by parameter magnitude.

 2. `cumulative`  (n4_cum):  3 distinct (kind, target) perturbations on
    the same scene, stacked B→C→D. GT order = by # of perturbations
    applied (original > 1 > 2 > 3).

Both produce DETERMINISTIC ground-truth — no LLM call needed. Physics
violations (overlap / out-of-room) drop the entire record.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from .build_dataset import (
    DEFAULT_RAW_DIR,
    _find_latest_blend,
    _per_step_room_dir,
    _perturbation_payload,
    _run_blender_render,
)
from .perturb import (
    Perturbation,
    _is_decorative_floor,
    _is_out_of_room,
    _overlaps_any,
)
from .pil_annotate import annotate_topdown
from .scene_io import load_scene


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "datasets" / "critic_pairs"

# Kinds with a continuous parameter — used as n4_psw seeds.
PSW_KINDS = frozenset({
    "rotate_45_offset",
    "move_off_wall",
    "move_far_from_focal",
    "move_to_corner",
})

# SFT prompt (mirrors `_ranking_user_text` in build_dataset.py)
def _sft_prompt(n: int) -> str:
    labels = ["A", "B", "C", "D"][:n]
    return (
        f"You are a scene-quality judge. {n} top-down photoreal renderings of "
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


def load_n2_records() -> list[dict]:
    recs = []
    with (OUT_DIR / "sft.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("n") == 2:
                    recs.append(r)
            except Exception:
                pass
    return recs


def group_by_scene(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        scene_id = r["id"].split("__ranking_")[0]
        out[scene_id].append(r)
    return out


def _interp_pose(orig: list[float], new: list[float], t: float) -> tuple[float, float, float]:
    """Linear interp between two poses (x, y, yaw_deg). Yaw uses shortest-arc."""
    x = orig[0] + t * (new[0] - orig[0])
    y = orig[1] + t * (new[1] - orig[1])
    # Yaw via shortest angular distance:
    dyaw = ((new[2] - orig[2] + 180) % 360) - 180
    yaw = orig[2] + t * dyaw
    return (float(x), float(y), float(yaw))


# Only `rotate_45_offset` among rotation kinds supports the param sweep.
# `face_wall` and `rotate_180` snap to fixed yaw targets and don't have
# a meaningful continuous-magnitude axis to interpolate along.
_ROTATION_KINDS = frozenset({"rotate_45_offset"})


def _build_perturbation(seed_pert: dict, t: float) -> Perturbation:
    """Build a t-interpolated Perturbation from a seed (the n2's pert).
    For rotate_45_offset, the absolute yaw delta is clipped to 90° because
    severity is only monotone in [0°, 90°] — past 90° the rotation wraps
    around and a 180° flip may look as natural as the original."""
    op = seed_pert["original_pose"]
    np_ = seed_pert["new_pose"]
    if seed_pert["kind"] in _ROTATION_KINDS:
        # Shortest-arc original delta
        anchor_delta = ((np_[2] - op[2] + 180) % 360) - 180
        target_delta = max(-90.0, min(90.0, t * anchor_delta))
        new_yaw = op[2] + target_delta
        new_pose = (float(np_[0]), float(np_[1]), float(new_yaw))
    else:
        new_pose = _interp_pose(op, np_, t)
    return Perturbation(
        object_id=seed_pert["object_id"],
        kind=seed_pert["kind"],
        description=f"{seed_pert['kind']} on {seed_pert['object_id']} (sweep t={t:.2f})",
        original_pose=tuple(op),
        new_pose=new_pose,
        reason_bad=seed_pert.get("reason_bad", ""),
        gt_strength=seed_pert.get("gt_strength", "soft"),
        secondary_change=seed_pert.get("secondary_change"),
        is_removal=(seed_pert["kind"] == "remove_object"),
    )


def _apply_to_scene(scene, pert: Perturbation):
    """Mutate a scene clone by applying a perturbation. Returns (clone, target)."""
    s = scene.clone()
    f = s.by_id(pert.object_id)
    if f is None:
        return None, None
    if pert.is_removal:
        s.furniture = [g for g in s.furniture if g.object_id != pert.object_id]
        return s, None
    f.tx, f.ty, f.yaw_deg = pert.new_pose
    if pert.secondary_change:
        sec_id = pert.secondary_change["object_id"]
        sf = s.by_id(sec_id)
        if sf is not None:
            np_ = pert.secondary_change["new_pose"]
            sf.tx, sf.ty, sf.yaw_deg = np_[0], np_[1], np_[2]
    return s, f


def _physics_ok(scene, pert: Perturbation) -> bool:
    """Apply pert to clone, check physics."""
    s, f = _apply_to_scene(scene, pert)
    if s is None:
        return False
    if f is None:  # removal
        return True
    if _is_out_of_room(f, s):
        return False
    if _overlaps_any(f, s.furniture):
        return False
    # secondary
    if pert.secondary_change:
        sec_id = pert.secondary_change["object_id"]
        sf = s.by_id(sec_id)
        if sf is not None:
            if _is_out_of_room(sf, s):
                return False
            if _overlaps_any(sf, s.furniture):
                return False
    return True


def _apply_chain(scene, perts: list[Perturbation]):
    """Apply a list of perts in order, returning final scene or None on phys-fail."""
    s = scene.clone()
    for p in perts:
        if not _physics_ok(s, p):
            return None
        s2, _ = _apply_to_scene(s, p)
        if s2 is None:
            return None
        s = s2
    return s


def _record_id(scene_id: str, seed_idx: int, mode_tag: str) -> str:
    return f"{scene_id}__ranking_s{seed_idx:06d}_n4_{mode_tag}"


def _orig_xy_map(scene) -> dict[str, tuple[float, float]]:
    return {f.object_id: (f.tx, f.ty) for f in scene.furniture}


def _render_chain(
    blend_path, out_png, room_box, perts, orig_xy, resolution,
):
    payload = []
    for p in perts:
        payload.extend(_perturbation_payload(p, orig_xy=orig_xy))
    meta = _run_blender_render(
        blend_path, out_png,
        perturbations=payload, room_box=room_box,
        resolution=resolution,
    )
    return meta["camera"]


def _write_record(
    record_id: str,
    record_dir: Path,
    ranked_perts_per_scene: list[list[Perturbation]],  # length 4: A=[], B=[p1], C=[p1,p2], D=[p1,p2,p3]
    mode_label: str,
    scenes_for_label: dict,  # label → cloned scene with all perts applied
    images: dict,            # label → sft png path
    judge_images: dict,      # label → judge png path
):
    labels = ["A", "B", "C", "D"]
    gt = labels.copy()
    perts_by_label = {labels[i]: [_pert_to_dict(p) for p in ranked_perts_per_scene[i]]
                       for i in range(4)}
    last_perts = [pl[-1] if pl else None for pl in ranked_perts_per_scene]
    perturbations_flat = [
        _pert_to_dict(ranked_perts_per_scene[-1][i]) if i < len(ranked_perts_per_scene[-1]) else None
        for i in range(3)
    ]
    perturbations_flat = [p for p in perturbations_flat if p is not None]

    # Build messages: prompt + 4 images
    content = []
    for lab in labels:
        content.append({"type": "image", "path": str(images[lab])})
    content.append({"type": "text", "text": _sft_prompt(4)})

    rec = {
        "id": record_id,
        "format": "ranking_n4",
        "scene_dir": str(record_dir.relative_to(OUT_DIR.parent.parent)),
        "n": 4,
        "images": {lab: str(images[lab]) for lab in labels},
        "judge_images": {lab: str(judge_images[lab]) for lab in labels},
        "ground_truth": {
            "ranking_best_to_worst": gt,
            "chain_index_at_label": {lab: i for i, lab in enumerate(gt)},
            "ranking_mode": mode_label,
            "tie_sets_chain_indices": [],
            "perturbations": perturbations_flat,
            "perturbations_by_label": perts_by_label,
        },
        "gemini": {
            "model": "n4_deterministic",
            "raw": "",
            "reasoning": "(deterministic GT: param sweep / cumulative)",
            "ranking": gt,
        },
        "cot_source": "deterministic",
        "messages": [{"role": "user", "content": content}],
    }
    return rec


def _pert_to_dict(p: Perturbation) -> dict:
    return {
        "kind": p.kind,
        "object_id": p.object_id,
        "gt_strength": p.gt_strength,
        "description": p.description,
        "reason_bad": p.reason_bad,
        "secondary_change": p.secondary_change,
        "original_pose": list(p.original_pose) if p.original_pose else None,
        "new_pose": list(p.new_pose) if p.new_pose else None,
    }


def generate(
    target_psw: int = 500,
    target_cum: int = 500,
    resolution: int = 1024,
    seed: int = 42,
    out_jsonl: Path = OUT_DIR / "sft_n4.jsonl",
    worker_id: int = 0,
    num_workers: int = 1,
):
    rng = random.Random(seed + worker_id * 7919)
    n2_records = load_n2_records()
    by_scene = group_by_scene(n2_records)
    # ALL workers see ALL scenes — each worker uses its own RNG order so
    # they explore different combinations. Final dedup at merge time
    # removes any cross-worker overlap.
    scene_ids = sorted(by_scene)
    rng.shuffle(scene_ids)

    print(f"loaded {len(n2_records)} n2 records across {len(by_scene)} scenes",
          file=sys.stderr)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fh = out_jsonl.open("a", encoding="utf-8")

    n_psw = 0
    n_cum = 0
    seed_counter = 0
    # Cache: scene_id → (scene, blend_path, room_box, cached_orig_png, cam_meta)
    scene_cache: dict[str, tuple] = {}

    def _scene_ctx(scene_id: str):
        if scene_id in scene_cache:
            return scene_cache[scene_id]
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
        ctx = (scene, blend_path, room_box, cached_orig, cam_meta)
        scene_cache[scene_id] = ctx
        return ctx

    # ============ PARAM SWEEP ============
    print(f"\n=== param sweep target {target_psw} ===", file=sys.stderr)
    for scene_id in scene_ids:
        if n_psw >= target_psw:
            break
        # Pick seed records with continuous-param kinds. Dedup by
        # (kind, target) so two n2 records on the same scene with the
        # same (kind, target) don't produce identical n4 sweeps.
        seen_kt = set()
        seeds = []
        for r in by_scene[scene_id]:
            perts = r["ground_truth"]["perturbations"]
            if not perts or perts[0]["kind"] not in PSW_KINDS:
                continue
            kt = (perts[0]["kind"], perts[0]["object_id"])
            if kt in seen_kt:
                continue
            seen_kt.add(kt)
            seeds.append(r)
        if not seeds:
            continue
        rng.shuffle(seeds)
        ctx = _scene_ctx(scene_id)
        if ctx is None:
            continue
        scene, blend_path, room_box, cached_orig, cam_meta = ctx
        orig_xy = _orig_xy_map(scene)

        for seed_rec in seeds:
            if n_psw >= target_psw:
                break
            seed_pert = seed_rec["ground_truth"]["perturbations"][0]
            if not seed_pert.get("original_pose") or not seed_pert.get("new_pose"):
                continue
            # Try increasing parameter ranges from MOST aggressive (2x
            # anchor) down to safest (1x anchor). Stops at the first
            # range where all 3 non-original variants pass physics
            # AND produce strictly-monotone visual change (no clip
            # collapse, e.g. rotation hitting 90° cap on multiple steps).
            range_options = [
                [0.0, 0.67, 1.33, 2.0],
                [0.0, 0.5,  1.0,  1.5],
                [0.0, 1.0/3, 2.0/3, 1.0],
            ]
            perts_per_scene = None
            for ts in range_options:
                cand = [_build_perturbation(seed_pert, t) for t in ts]
                # Check pose distinctness between B, C, D (>= 5° yaw or
                # >= 0.1m position delta between consecutive samples).
                def _distinct(p1, p2):
                    if not p1 or not p2:
                        return True
                    a = p1[-1].new_pose
                    b = p2[-1].new_pose
                    dx, dy = b[0]-a[0], b[1]-a[1]
                    if dx*dx + dy*dy >= 0.01:  # 0.1m
                        return True
                    dyaw = abs(((b[2]-a[2]+180) % 360) - 180)
                    return dyaw >= 5.0
                cand_per_scene = [[], [cand[1]], [cand[2]], [cand[3]]]
                if not (_distinct(cand_per_scene[1], cand_per_scene[2])
                        and _distinct(cand_per_scene[2], cand_per_scene[3])):
                    continue
                if all(_apply_chain(scene, pp) is not None for pp in cand_per_scene[1:]):
                    perts_per_scene = cand_per_scene
                    break
            if perts_per_scene is None:
                continue
            # Render + write
            seed_counter += 1
            record_id = _record_id(scene_id, seed_counter, "psw")
            record_dir = OUT_DIR / record_id
            record_dir.mkdir(parents=True, exist_ok=True)
            labels = ["A", "B", "C", "D"]
            images = {}; judge_images = {}
            try:
                for i, lab in enumerate(labels):
                    sft_path = record_dir / f"scene_{lab}.png"
                    judge_path = record_dir / f"scene_{lab}_judge.png"
                    if i == 0:
                        raw_png = cached_orig
                        camera_meta = cam_meta
                    else:
                        raw_png = record_dir / f"_raw_{lab}.png"
                        camera_meta = _render_chain(
                            blend_path, raw_png, room_box,
                            perts_per_scene[i], orig_xy, resolution,
                        )
                    # SFT clean image (no highlights)
                    cur_scene = _apply_chain(scene, perts_per_scene[i]) or scene
                    annotate_topdown(
                        raw_png, scene=cur_scene, camera_meta=camera_meta,
                        highlight_ids=[], out_path=sft_path,
                    )
                    # Judge image (highlights on perturbed pieces)
                    hi = [p.object_id for p in perts_per_scene[i]]
                    if perts_per_scene[i]:
                        last_p = perts_per_scene[i][-1]
                        if last_p.secondary_change:
                            hi.append(last_p.secondary_change["object_id"])
                    annotate_topdown(
                        raw_png, scene=cur_scene, camera_meta=camera_meta,
                        highlight_ids=hi, out_path=judge_path,
                        rotation_highlight_ids=hi if any(p.kind in ("rotate_45_offset", "rotate_180", "face_wall") for p in perts_per_scene[i]) else [],
                    )
                    images[lab] = sft_path
                    judge_images[lab] = judge_path
            except Exception as e:
                print(f"[fail] psw {record_id}: {e!r}", file=sys.stderr)
                shutil.rmtree(record_dir, ignore_errors=True)
                continue
            # Write record
            rec = _write_record(
                record_id, record_dir, perts_per_scene,
                "cumulative", scenes_for_label={}, images=images,
                judge_images=judge_images,
            )
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_psw += 1
            if n_psw % 10 == 0:
                print(f"[psw] {n_psw}/{target_psw}", file=sys.stderr)

    # ============ CUMULATIVE ============
    print(f"\n=== cumulative target {target_cum} ===", file=sys.stderr)
    for scene_id in scene_ids:
        if n_cum >= target_cum:
            break
        scene_recs = by_scene[scene_id]
        if len(scene_recs) < 3:
            continue
        ctx = _scene_ctx(scene_id)
        if ctx is None:
            continue
        scene, blend_path, room_box, cached_orig, cam_meta = ctx
        orig_xy = _orig_xy_map(scene)

        # Build candidate set: each rec has one pert. Need DISTINCT (kind, target).
        per_kt = {}  # (kind, target) → record
        for r in scene_recs:
            p = r["ground_truth"]["perturbations"]
            if not p:
                continue
            key = (p[0]["kind"], p[0]["object_id"])
            if key not in per_kt:
                per_kt[key] = p[0]
        candidates = list(per_kt.values())
        if len(candidates) < 3:
            continue
        rng.shuffle(candidates)

        # Try combinations of 3 perts
        for triple_start in range(0, len(candidates) - 2):
            if n_cum >= target_cum:
                break
            triple = candidates[triple_start:triple_start + 3]
            if len({(p["kind"], p["object_id"]) for p in triple}) != 3:
                continue
            # Build Perturbation objects (use raw new_pose from n2)
            perts = []
            for p in triple:
                pp = Perturbation(
                    object_id=p["object_id"],
                    kind=p["kind"],
                    description=p.get("description", ""),
                    original_pose=tuple(p["original_pose"]) if p.get("original_pose") else (0,0,0),
                    new_pose=tuple(p["new_pose"]) if p.get("new_pose") else (0,0,0),
                    reason_bad=p.get("reason_bad", ""),
                    gt_strength=p.get("gt_strength", "soft"),
                    secondary_change=p.get("secondary_change"),
                    is_removal=(p["kind"] == "remove_object"),
                )
                perts.append(pp)
            perts_per_scene = [[], perts[:1], perts[:2], perts[:3]]
            phys_ok = True
            for pp in perts_per_scene[1:]:
                if _apply_chain(scene, pp) is None:
                    phys_ok = False
                    break
            if not phys_ok:
                continue
            seed_counter += 1
            record_id = _record_id(scene_id, seed_counter, "cum")
            record_dir = OUT_DIR / record_id
            record_dir.mkdir(parents=True, exist_ok=True)
            labels = ["A", "B", "C", "D"]
            images = {}; judge_images = {}
            try:
                for i, lab in enumerate(labels):
                    sft_path = record_dir / f"scene_{lab}.png"
                    judge_path = record_dir / f"scene_{lab}_judge.png"
                    if i == 0:
                        raw_png = cached_orig
                        camera_meta = cam_meta
                    else:
                        raw_png = record_dir / f"_raw_{lab}.png"
                        camera_meta = _render_chain(
                            blend_path, raw_png, room_box,
                            perts_per_scene[i], orig_xy, resolution,
                        )
                    cur_scene = _apply_chain(scene, perts_per_scene[i]) or scene
                    annotate_topdown(
                        raw_png, scene=cur_scene, camera_meta=camera_meta,
                        highlight_ids=[], out_path=sft_path,
                    )
                    hi = [p.object_id for p in perts_per_scene[i]]
                    for p in perts_per_scene[i]:
                        if p.secondary_change:
                            hi.append(p.secondary_change["object_id"])
                    rot_hi = [p.object_id for p in perts_per_scene[i]
                              if p.kind in ("rotate_45_offset", "rotate_180", "face_wall")]
                    annotate_topdown(
                        raw_png, scene=cur_scene, camera_meta=camera_meta,
                        highlight_ids=hi, out_path=judge_path,
                        rotation_highlight_ids=rot_hi,
                    )
                    images[lab] = sft_path
                    judge_images[lab] = judge_path
            except Exception as e:
                print(f"[fail] cum {record_id}: {e!r}", file=sys.stderr)
                shutil.rmtree(record_dir, ignore_errors=True)
                continue
            rec = _write_record(
                record_id, record_dir, perts_per_scene,
                "cumulative", scenes_for_label={}, images=images,
                judge_images=judge_images,
            )
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_cum += 1
            if n_cum % 10 == 0:
                print(f"[cum] {n_cum}/{target_cum}", file=sys.stderr)
            # Continue trying more triplets from this scene (no break).

    fh.close()
    print(f"\nDONE: psw={n_psw}, cum={n_cum}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-psw", type=int, default=500)
    ap.add_argument("--target-cum", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "sft_n4.jsonl")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()
    # Per-worker quota
    target_psw = (args.target_psw + args.num_workers - 1) // args.num_workers
    target_cum = (args.target_cum + args.num_workers - 1) // args.num_workers
    generate(
        target_psw=target_psw,
        target_cum=target_cum,
        resolution=args.resolution,
        seed=args.seed,
        out_jsonl=args.out,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
