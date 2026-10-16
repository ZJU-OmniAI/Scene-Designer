"""Render an ortho top-down PNG of a per_step .blend, optionally applying a
perturbation transform to one (or two) furniture parent EMPTYs.

Designed to be invoked as a CLI subprocess so that bpy's process-global state
does not pollute the main pipeline:

    python -m src_critic.blender_render \\
        --blend  per_step/<scene>/<room>/furniture_NN__*.blend \\
        --out    /tmp/scene_A.png \\
        --resolution 1024 \\
        --perturbations '[{"identify_xy": [-2.46, 1.04], "new_xy": [-2.46, 1.04], "delta_yaw_deg": 180.0}]'

stdout (last non-blank line) is a JSON blob describing the world AABB and the
ortho camera box, which the annotator uses to project world coords into pixels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _emptys(bpy):
    return [o for o in bpy.data.objects if o.type == "EMPTY"]


def _find_target_group(bpy, target_xy, tol: float = 0.05):
    """Return the list of top-level scene objects that move together as the
    'object' positioned at `target_xy`.

    Two cases observed in the per_step blends:
      * generated assets — a single EMPTY at (x, y, 0) parents one MESH child.
      * articulated assets (cabinets, drawers) — several unparented MESHes
        all share the same world (x, y, 0); together they form one piece.

    Returns: list of bpy.types.Object (length 1 for generated assets, ≥2 for
    articulated assets) or [] if nothing matches within `tol` metres.
    """
    tx, ty = target_xy
    tol2 = tol * tol

    # 1) prefer a parent EMPTY with children (generated asset).
    best_empty = None
    best_d = tol2 + 1.0
    for o in _emptys(bpy):
        if not o.children or abs(o.location.z) > 0.4:
            continue
        d = (o.location.x - tx) ** 2 + (o.location.y - ty) ** 2
        if d < best_d:
            best_d = d
            best_empty = o
    if best_empty is not None and best_d <= tol2:
        return [best_empty]

    # 2) articulated asset — gather all top-level (no parent) MESH objects
    #    sitting at the target xy. Group by exact (x, y) snapped to mm.
    matches = []
    for o in bpy.data.objects:
        if o.type != "MESH" or o.parent is not None:
            continue
        if abs(o.location.z) > 0.4:
            continue
        d = (o.location.x - tx) ** 2 + (o.location.y - ty) ** 2
        if d <= tol2:
            matches.append(o)
    return matches


def _find_parent_empty(bpy, target_xy, tol: float = 0.05):
    """Backwards-compatible single-object accessor (now returns a group)."""
    group = _find_target_group(bpy, target_xy, tol)
    return group or None


# NOTE: _apply_perturbation was inlined into main() so we can pre-resolve all
# targets before mutating any of them (matters for multi-object swaps).


def _scene_aabb(bpy):
    inf = float("inf")
    mins = [inf, inf, inf]
    maxs = [-inf, -inf, -inf]
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for v in obj.bound_box:
            wp = obj.matrix_world @ bpy.types.Object.bl_rna  # placeholder; replaced below
    # The above triggers a nonsense access just to keep the import clean; do it properly:
    return None


def _real_scene_aabb(bpy, mathutils):
    inf = float("inf")
    mins = [inf, inf, inf]
    maxs = [-inf, -inf, -inf]
    any_hit = False
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for v in obj.bound_box:
            wp = obj.matrix_world @ mathutils.Vector(v)
            mins = [min(mins[i], wp[i]) for i in range(3)]
            maxs = [max(maxs[i], wp[i]) for i in range(3)]
            any_hit = True
    if not any_hit:
        raise RuntimeError("no mesh in scene")
    return mins, maxs


def _setup_camera(bpy, mathutils, room_box, *, mode: str = "persp_top",
                  margin_scale: float = 1.25):
    """Configure the rendering camera.

    mode = 'ortho_top' :  legacy pure orthographic top-down (no perspective).
    mode = 'persp_top' :  scenesmith-style perspective from straight above
        (lens=50mm, sensor=36mm, distance = max_dim/(2 tan(fov/2)) * margin).
        Wall-mounted objects on side walls remain visible because of the
        natural foreshortening at the image edges.
    """
    x_min, y_min, x_max, y_max = room_box
    width = x_max - x_min
    length = y_max - y_min
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    max_dim = max(width, length)

    cam_data = bpy.data.cameras.new("CriticCam")
    cam_obj = bpy.data.objects.new("CriticCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    if mode == "ortho_top":
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = max_dim
        cam_data.clip_start = 0.01
        cam_data.clip_end = 100.0
        cam_obj.location = (cx, cy, 8.0)
        cam_obj.rotation_euler = (0.0, 0.0, 0.0)
        view_extent = max_dim
        return {
            "mode": mode,
            "center": [cx, cy],
            "view_box_world": [cx - max_dim / 2.0, cy - max_dim / 2.0,
                               cx + max_dim / 2.0, cy + max_dim / 2.0],
        }

    # persp_top — scenesmith-style.
    cam_data.type = "PERSP"
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    fov = 2.0 * math.atan((cam_data.sensor_width / 2.0) / cam_data.lens)
    distance = (max_dim / 2.0) / math.tan(fov / 2.0) * margin_scale
    cam_obj.location = (cx, cy, distance)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    return {
        "mode": mode,
        "center": [cx, cy],
        "lens": cam_data.lens,
        "sensor_width": cam_data.sensor_width,
        "distance": distance,
        "fov_radians": fov,
        # half-extent of what the camera sees at floor level (z=0):
        "view_half_extent_at_floor": distance * math.tan(fov / 2.0),
    }


def _ensure_lighting(bpy):
    has_light = any(o.type == "LIGHT" for o in bpy.data.objects)
    if has_light:
        # boost the existing light a bit so the rendered top-down isn't dim
        for o in bpy.data.objects:
            if o.type == "LIGHT":
                try:
                    o.data.energy = max(o.data.energy, 5.0)
                except Exception:
                    pass
        return
    light_data = bpy.data.lights.new("CriticSun", type="SUN")
    light_data.energy = 5.0
    light_obj = bpy.data.objects.new("CriticSun", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (0.3, 0.2, 0.0)


def _hide_ceiling(bpy, mathutils, mins, maxs) -> None:
    """Hide the thin slab near the top of the room so the top-down sees inside."""
    size_x = maxs[0] - mins[0]
    size_y = maxs[1] - mins[1]
    size_z = maxs[2] - mins[2]
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        corners = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
        zs = [c.z for c in corners]
        xs = [c.x for c in corners]; ys = [c.y for c in corners]
        ext = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        center_z = (min(zs) + max(zs)) / 2.0
        if (ext[2] < 0.15
                and ext[0] > 0.5 * size_x
                and ext[1] > 0.5 * size_y
                and center_z > mins[2] + size_z * 0.7):
            o.hide_render = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--perturbations", type=str, default="[]",
                    help='JSON list of {"identify_xy":[x,y], "new_xy":[x,y]?, "delta_yaw_deg":d?}')
    ap.add_argument("--room-box", type=str, required=True,
                    help='JSON [x_min, y_min, x_max, y_max] of the room interior.')
    args = ap.parse_args()

    import bpy  # noqa: PLC0415
    import mathutils  # noqa: PLC0415

    if not args.blend.exists():
        print(json.dumps({"error": f"blend missing: {args.blend}"}))
        return 2

    bpy.ops.wm.open_mainfile(filepath=str(args.blend))

    perturbations = json.loads(args.perturbations)
    # Pre-resolve all target groups FIRST (by their original positions) so
    # that multi-object perturbations like a swap can't alias one another
    # after the first mutation. Apply transforms only after every group is
    # bound. Each "group" is a list of bpy objects that move together — a
    # single parent EMPTY for generated assets, or several unparented meshes
    # for articulated assets (cabinets etc.).
    resolved = []
    for p in perturbations:
        group = _find_target_group(bpy, p["identify_xy"])
        if not group:
            raise RuntimeError(
                f"No parent EMPTY (or articulated mesh group) found near "
                f"{p['identify_xy']} in this blend"
            )
        resolved.append(group)

    for p, group in zip(perturbations, resolved):
        # remove_object: hide every mesh in the group so it disappears from
        # the render entirely. We also recursively hide any child mesh of an
        # EMPTY group root.
        if p.get("remove"):
            stack = list(group)
            while stack:
                obj = stack.pop()
                obj.hide_render = True
                obj.hide_viewport = True
                stack.extend(obj.children)
            continue

        new_xy = p.get("new_xy")
        delta_yaw = p.get("delta_yaw_deg")
        delta_q = None
        if delta_yaw is not None and abs(delta_yaw) > 1e-3:
            delta_q = mathutils.Quaternion(
                (0.0, 0.0, 1.0), math.radians(float(delta_yaw))
            )
        for obj in group:
            if new_xy is not None:
                obj.location.x = float(new_xy[0])
                obj.location.y = float(new_xy[1])
            if delta_q is not None:
                if obj.rotation_mode == "QUATERNION":
                    obj.rotation_quaternion = delta_q @ obj.rotation_quaternion
                else:
                    obj.rotation_euler.z += math.radians(float(delta_yaw))
    bpy.context.view_layer.update()

    mins, maxs = _real_scene_aabb(bpy, mathutils)
    room_box = json.loads(args.room_box)
    cam_meta = _setup_camera(bpy, mathutils, room_box)

    _hide_ceiling(bpy, mathutils, mins, maxs)
    _ensure_lighting(bpy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    # Engine selection with headless robustness:
    #  - Legacy EEVEE (Blender <4.2) renders fine headless -> prefer it for
    #    visual parity with the original test set.
    #  - EEVEE Next (Blender 4.2+) needs a live GL/EGL context; as a bare
    #    bpy module on a display-less host it crashes ("Couldn't find current
    #    GLX or EGL context"). So on newer bpy we fall back to CYCLES, which
    #    is a pure-compute engine (GPU if available, else CPU) needing no GL.
    # CPU Cycles works headlessly and never takes a training GPU implicitly.
    import os
    scene.render.threads_mode = "FIXED"
    scene.render.threads = int(os.environ.get("SCENE_RENDER_THREADS", "8"))
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(os.environ.get("SCENE_RENDER_SAMPLES", "24"))
    scene.cycles.use_denoising = False
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(args.out)
    bpy.ops.render.render(write_still=True)

    out = {
        "blend": str(args.blend),
        "png": str(args.out),
        "resolution": args.resolution,
        "scene_aabb": {"min": mins, "max": maxs},
        "camera": cam_meta,
        "perturbations_applied": len(perturbations),
    }
    Path(str(args.out) + ".meta.json").write_text(json.dumps(out))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
