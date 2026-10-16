"""Isolated Drake/Blender export; a native renderer failure cannot kill training."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import runtime
from scenesmith.agent_utils.room import RoomScene
from scenesmith.agent_utils.rendering import save_scene_as_blend

if __name__ == '__main__':
    scene_dir, state_path, output = map(Path, sys.argv[1:])
    scene = RoomScene(room_geometry=None, scene_dir=scene_dir)
    scene.restore_from_state_dict(json.loads(state_path.read_text()))
    save_scene_as_blend(scene=scene, output_path=output,
                        blender_server_host='127.0.0.1', blender_server_port_range=(8000,8050))
