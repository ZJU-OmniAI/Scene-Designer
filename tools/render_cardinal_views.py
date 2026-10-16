#!/usr/bin/env python3
"""Render tuned cardinal interior views from a SceneSmith .blend file.

Usage (with the project's Blender Python):
  python tools/render_cardinal_views.py scene.blend --out renders
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


def render(blend: Path, out: Path, width: int, height: int, elevation: float, overwrite: bool = False, views: tuple[str, ...] = ('east','north','west','south')) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.threads_mode = "FIXED"
    scene.render.threads = int(os.environ.get("SCENE_RENDER_THREADS", "8"))
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(os.environ.get("SCENE_RENDER_SAMPLES", "24"))
    scene.cycles.use_denoising = False
    scene.render.image_settings.file_format = "PNG"

    # Room bounds are computed from the room meshes. Furniture must not change
    # the camera framing.
    room_meshes = [
        o for o in scene.objects
        if o.type == "MESH" and (o.name.startswith("Mesh_") or o.name.startswith("geometry_"))
    ]
    if not room_meshes:
        room_meshes = [o for o in scene.objects if o.type == "MESH"]
    points = [o.matrix_world @ Vector(c) for o in room_meshes for c in o.bound_box]
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (lo + hi) / 2
    max_dim = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)

    cam_data = bpy.data.cameras.new("CardinalCamera")
    camera = bpy.data.objects.new("CardinalCamera", cam_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    cam_data.type = "PERSP"
    cam_data.lens = 50
    cam_data.sensor_width = 36
    fov = 2 * math.atan((cam_data.sensor_width / 2) / cam_data.lens)
    distance = (max_dim / 2) / math.tan(fov / 2) / 0.8

    out.mkdir(parents=True, exist_ok=True)
    original_visibility = {obj: obj.hide_render for obj in scene.objects}
    elev = math.radians(elevation)
    for label, azimuth in (("east", 0), ("north", 90), ("west", 180), ("south", 270)):
        if label not in views:
            continue
        # Completing a missing direction must not overwrite an already bound
        # training image. Keep every valid canonical view at its native size.
        target = out / f"{blend.stem}.{label}.png"
        if not overwrite and blend.stem not in ('furniture_final','wall_final','ceiling_final') and target.is_file():
            try:
                from PIL import Image
                with Image.open(target) as existing:
                    matches = existing.format == "PNG" and min(existing.size) > 0
                    existing.verify()
                if matches:
                    print(f"[render] keep {target}")
                    continue
            except (OSError,ValueError,SyntaxError):
                pass
        for obj, hidden in original_visibility.items():
            obj.hide_render = hidden
        az = math.radians(azimuth)
        direction = Vector((math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)))

        # Hide the camera-side wall panels, retaining the side walls, floor,
        # windows, and all furniture as in the original SFT renderer.
        candidates = []
        for obj in room_meshes:
            if obj.dimensions.z > 2.0 and min(obj.dimensions.x, obj.dimensions.y) < 0.1:
                wall_center = sum((obj.matrix_world @ Vector(v) for v in obj.bound_box), Vector()) / 8
                candidates.append((wall_center.dot(direction), obj))
        if candidates:
            nearest = max(value for value, _ in candidates)
            for value, obj in candidates:
                if value > nearest - 0.3:
                    obj.hide_render = True

        camera.location = center + direction * distance
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(out / f"{blend.stem}.{label}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[render] wrote {scene.render.filepath}")

    # Final captures have explicit post-action provenance. Register the views
    # only when they were rendered from that exact captured blend. This does
    # not change camera settings or the existing per-step rendering workflow.
    manifest = blend.with_suffix('.json')
    if blend.stem in ('furniture_final', 'wall_final', 'ceiling_final') and manifest.is_file():
        record = json.loads(manifest.read_text())
        if record.get('blend_sha256') == hashlib.sha256(blend.read_bytes()).hexdigest():
            for view in views:
                path = out / f'{blend.stem}.{view}.png'
                record.setdefault('images',{})[view] = dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            temporary=manifest.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            temporary.replace(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blend", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rerender existing views")
    parser.add_argument("--views", nargs="+", choices=("south","east","north","west"), default=["east","north","west","south"])
    args = parser.parse_args()
    render(args.blend.resolve(), args.out.resolve(), args.width, args.height, args.elevation, args.overwrite, tuple(args.views))


if __name__ == "__main__":
    main()
