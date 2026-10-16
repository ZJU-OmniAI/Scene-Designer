"""Render topdown plus south/east/west/north oblique views from one .blend."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.render_cardinal_views import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('blend', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--top-resolution', type=int, default=1024)
    parser.add_argument('--elevation', type=float, default=30)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    render(args.blend.resolve(), args.out.resolve(), args.width, args.height,
           args.elevation, args.overwrite, ('south','east','west','north'))
    # Use the existing Blender renderer and annotation projection for topdown.
    from src_critic.blender_render import main as render_top
    target = args.out.resolve() / f'{args.blend.stem}.topdown.png'
    if args.overwrite or not target.exists():
        import bpy
        from mathutils import Vector
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and (o.name.startswith("Mesh_") or o.name.startswith("geometry_"))]
        if not meshes: meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        points = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
        box = [min(p.x for p in points), min(p.y for p in points), max(p.x for p in points), max(p.y for p in points)]
        sys.argv = [sys.argv[0], '--blend', str(args.blend.resolve()), '--out', str(target),
                    '--resolution', str(args.top_resolution), '--room-box', json.dumps(box)]
        render_top()

if __name__ == '__main__':
    main()
