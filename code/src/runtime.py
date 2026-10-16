"""Portable roots for local modules and the separately installed SceneSmith."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SCENESMITH_ROOT = Path(os.environ.get("SCENESMITH_ROOT", ROOT / "scenesmith")).resolve()
for path in (SCENESMITH_ROOT, ROOT / "src", ROOT):
    value = str(path)
    if value in sys.path: sys.path.remove(value)
    sys.path.insert(0, value)


def relocate_scene_config(value):
    """Resolve upstream asset paths against the external SceneSmith checkout."""
    if isinstance(value, dict): return {k: relocate_scene_config(v) for k, v in value.items()}
    if isinstance(value, list): return [relocate_scene_config(v) for v in value]
    if isinstance(value, str) and value.startswith(('data/', 'external/', 'materials/', 'checkpoints/')):
        return str(SCENESMITH_ROOT / value)
    return value
