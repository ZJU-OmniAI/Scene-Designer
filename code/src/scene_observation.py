"""One ordered scene-observation contract for data, SFT, RL and inference."""
from pathlib import Path

VIEW_ORDER = ('topdown', 'south', 'east', 'west', 'north')


def resolve_image(raw, repo_root):
    path = Path(raw)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path if path.is_file() else None


def validate_scene_views(paths, *, allow_empty=False):
    paths = [str(p) for p in paths]
    if not paths and allow_empty:
        return paths
    if len(paths) != 5 or len(set(paths)) != 5:
        raise ValueError('Scene observations require 5 distinct images ordered topdown, south, east, west, north')
    # Filenames with explicit canonical view names must agree with the slot.
    for path, view in zip(paths, VIEW_ORDER):
        stem = Path(path).stem
        known = next((v for v in VIEW_ORDER if stem.endswith('.' + v)), None)
        if known is not None and known != view:
            raise ValueError(f'Expected {view} view, got {path}')
    return paths


def scene_image_paths(turn, repo_root):
    raw = turn.get('images') or ([turn['image']] if turn.get('image') else [])
    paths = []
    for name in raw:
        path = resolve_image(name, repo_root)
        if path is None:
            raise FileNotFoundError(f'Missing SFT image: {name}')
        paths.append(str(path))
    return paths


def message_image_paths(messages):
    """Extract images in exactly chat-template order, without hidden fallbacks."""
    paths = []
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get('type') == 'image':
                path = item.get('image', item.get('path'))
                if not isinstance(path, (str, Path)):
                    raise ValueError('Expected a local image path in the observation')
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
                paths.append(str(path))
    return paths


def load_message_images(messages):
    from PIL import Image
    images = []
    for path in message_image_paths(messages):
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    return images or None


def configure_image_budget(processor, max_pixels=65536):
    """Set both legacy and fast-processor limits (Transformers 4.57)."""
    ip = processor.image_processor
    ip.max_pixels = int(max_pixels)
    if hasattr(ip, 'size') and isinstance(ip.size, dict):
        ip.size = dict(ip.size, longest_edge=int(max_pixels))
        ip.size['shortest_edge'] = min(ip.size.get('shortest_edge', 4096), int(max_pixels))
    if hasattr(ip, 'min_pixels') and ip.min_pixels is not None:
        ip.min_pixels = min(ip.min_pixels, int(max_pixels))
