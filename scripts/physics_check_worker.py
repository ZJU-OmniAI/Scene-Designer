"""
Subprocess worker for Drake physics check.
Isolated so that libdrake.so segfaults kill only this process, not the parent.

Usage:
    python physics_check_worker.py <scene_dir> <state_dict_json_path>

Prints the check_physics_violations result to stdout.
Exit code 0 = success, non-zero (or -11 SIGSEGV) = failure.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import runtime

def main():
    if len(sys.argv) != 3:
        print("Physics check unavailable (bad args).", flush=True)
        sys.exit(1)

    scene_dir = Path(sys.argv[1])
    state_dict_path = Path(sys.argv[2])

    try:
        state_dict = json.loads(state_dict_path.read_text())
    except Exception as e:
        print(f"Physics check unavailable (failed to load state: {e}).", flush=True)
        sys.exit(1)

    try:
        from scenesmith.agent_utils.room import RoomScene, AgentType
        from scenesmith.agent_utils.physics_tools import check_physics_violations
        from omegaconf import OmegaConf
    except Exception as e:
        print(f"Physics check unavailable (import error: {e}).", flush=True)
        sys.exit(1)

    try:
        # RoomScene is a dataclass; room_geometry is required but will be
        # overwritten by restore_from_state_dict, so pass None as placeholder.
        scene = RoomScene(room_geometry=None, scene_dir=scene_dir)  # type: ignore[arg-type]
        scene.restore_from_state_dict(state_dict)
    except Exception as e:
        print(f"Physics check unavailable (scene restore error: {e}).", flush=True)
        sys.exit(1)

    try:
        cfg_json = state_dict.get("_cfg_json")
        if cfg_json:
            cfg = OmegaConf.create(json.loads(cfg_json))
        else:
            print("Physics check unavailable (no cfg).", flush=True)
            sys.exit(1)

        result = check_physics_violations(
            scene=scene, cfg=cfg, agent_type=AgentType.FURNITURE
        )
        print(result, end='', flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Physics check unavailable (runtime error: {e}).", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
