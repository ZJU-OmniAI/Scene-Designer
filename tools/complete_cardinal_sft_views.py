"""Attach four same-state oblique renders to existing topdown SFT observations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scene_observation import VIEW_ORDER


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sft-root', type=Path, required=True)
    ap.add_argument('--repo-root', type=Path, default=ROOT)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--execute', action='store_true', help='Otherwise validate and list jobs only')
    args = ap.parse_args()
    base = args.repo_root.resolve()
    jobs = 0
    for jf in sorted(args.sft_root.rglob('*.json')):
        data = json.loads(jf.read_text())
        if data.get('stage') not in ('furniture', 'wall', 'ceiling'):
            continue
        changed = False
        for turn in data.get('turns', []):
            if turn.get('role') != 'assistant':
                continue
            images = turn.get('images') or ([turn['image']] if turn.get('image') else [])
            if not images:  # the SFT loader separately handles missing terminal images
                continue
            top = next((base / p for p in images if p.endswith('.topdown.png')), None)
            if top is None or not top.is_file():
                raise FileNotFoundError(f'{jf}: missing canonical .topdown.png')
            blend = top.with_name(top.name.removesuffix('.topdown.png') + '.blend')
            if not blend.is_file():
                raise FileNotFoundError(f'Need the SAME-STATE Blender snapshot: {blend}')
            paths = [blend.with_name(f'{blend.stem}.{v}.png') for v in VIEW_ORDER]
            jobs += 1
            if args.execute:
                subprocess.run([args.python, str(ROOT/'tools/render_cardinal_views.py'),
                                str(blend), '--out', str(blend.parent)], check=True, timeout=600)
                from PIL import Image
                for path in paths:
                    with Image.open(path) as image:
                        image.verify()
                turn['images'] = [os.path.relpath(p, base) for p in paths]
                turn.pop('image', None)
                changed = True
        if changed:
            temp = jf.with_suffix('.json.tmp')
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
            temp.replace(jf)
    print(f'{jobs} observations {"updated" if args.execute else "validated; pass --execute to render"}')

if __name__ == '__main__':
    main()
