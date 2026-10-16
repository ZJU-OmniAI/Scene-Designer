"""Render the same five native views for both policy rollout and training data."""
import os
import subprocess
import sys
from pathlib import Path
from src.runtime import ROOT
from src.scene_observation import VIEW_ORDER, validate_scene_views


def render_scene_views(agent, scene, save_dir, step_idx):
    from OpenSceneAgent.replay_chained import _save_roomscene_blend
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    blend = save_dir / f'step_{step_idx:03d}.blend'
    # Export the actual current state, never a previous observation.
    _save_roomscene_blend(agent, blend)
    proc = subprocess.run(
        [os.environ.get('BLENDER_PYTHON', sys.executable), str(ROOT/'tools/render_five_views.py'),
         str(blend), '--out', str(save_dir), '--overwrite'],
        cwd=ROOT, text=True, capture_output=True, timeout=600,
    )
    if proc.returncode:
        raise RuntimeError('Five-view rendering failed:\n' + proc.stderr[-3000:] + proc.stdout[-1500:])
    paths = [save_dir / f'{blend.stem}.{view}.png' for view in VIEW_ORDER]
    if not all(p.is_file() for p in paths):
        raise RuntimeError('Renderer did not produce all five current-state views')
    annotate_live_topdown(scene, paths[0])
    return validate_scene_views(paths)


def annotate_live_topdown(scene, path):
    """Add object IDs, openings and coordinates without perturbation hints."""
    import json
    from src_critic.scene_io import Scene, Room, Furniture, _quat_to_yaw_deg
    from src_critic.pil_annotate import annotate_topdown
    state = scene.to_state_dict()
    geometry = state['room_geometry']
    floor = geometry.get('floor') or {}
    center = floor.get('transform', {}).get('translation', [0, 0, 0])
    doors, windows = [], []
    boundaries = dict(north='A', south='B', east='C', west='D')
    for opening in geometry.get('openings', []):
        item = dict(id=opening['opening_id'], wall_direction=opening['wall_direction'],
                    boundary_label=boundaries[opening['wall_direction']],
                    position_exact=opening['position_along_wall'],
                    position_along_wall=opening['position_along_wall'])
        (doors if opening['opening_type'] == 'door' else windows).append(item)
    room = Room(str(getattr(scene, 'room_id', 'room')), float(geometry['length']),
                float(geometry['width']), float(center[0]), float(center[1]), doors, windows)
    objects = []
    for obj in state['objects'].values():
        if obj.get('object_type') in ('wall', 'floor'): continue
        if not obj.get('bbox_min') or not obj.get('bbox_max'): continue
        transform = obj['transform']
        objects.append(Furniture(obj['object_id'], obj.get('name', ''), obj.get('description', ''),
            *transform['translation'][:2], _quat_to_yaw_deg(transform['rotation_wxyz']),
            tuple(obj['bbox_min']), tuple(obj['bbox_max'])))
    metadata = json.loads(Path(str(path)+'.meta.json').read_text())
    annotate_topdown(path, scene=Scene(Path(scene.scene_dir), room, objects),
                     camera_meta=metadata['camera'], highlight_ids=[], out_path=path)
