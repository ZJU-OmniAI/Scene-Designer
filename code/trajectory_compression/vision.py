"""Send actual observation bytes, preserving each source turn's views and size."""
import base64
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
def resolve_image(value):
    path = Path(value)
    if not path.is_absolute(): path = ROOT / path
    if not path.is_file(): raise FileNotFoundError(f'Missing source observation: {value}')
    return path

def build_user_content(segment):
    observations=segment.get('observations')
    if not observations:raise ValueError('No observations attached; refusing text-only review')
    required={event['turn_index'] for event in segment['events']}
    by_turn={o['turn_index']:o for o in observations}
    if not required<=by_turn.keys():raise ValueError('Missing boundary/intermediate observations')
    content=[{'type':'text','text':json.dumps(segment,ensure_ascii=False)}];manifest=[]
    for obs in observations:
        paths=obs.get('images')
        if not paths:
            if (segment.get('stage')=='ceiling'
                    and obs.get('availability')=='not_captured_after_terminal_action'
                    and obs['turn_index']>segment.get('protected_turn',float('inf'))):
                content.append({'type':'text','text':json.dumps(dict(turn_index=obs['turn_index'],availability=obs['availability'],note='天花板末尾未采集动作后图像；这里只提供已有工具反馈，不把先前输入图当作动作后画面。'),ensure_ascii=False)})
                continue
            raise ValueError(f'Missing images at turn {obs["turn_index"]}')
        for vi,value in enumerate(paths):
            path=resolve_image(value);data=path.read_bytes()
            if data.startswith(b'\x89PNG\r\n\x1a\n') and len(data)>=24:
                mime='image/png';size=struct.unpack('>II',data[16:24])
            elif data.startswith(b'\xff\xd8\xff'):
                mime='image/jpeg';size=None
            else:raise ValueError(f'Unsupported source image: {path}')
            metadata={'turn_index':obs['turn_index'],'relation':obs['relation'],'view_index':vi,
                      'path':str(path),'dimensions':size,'sha256':hashlib.sha256(data).hexdigest()}
            manifest.append(metadata)
            content.append({'type':'text','text':json.dumps(metadata,ensure_ascii=False)})
            content.append({'type':'image_url','image_url':{
                'url':f'data:{mime};base64,'+base64.b64encode(data).decode('ascii'),'detail':'high'}})
    if not manifest:raise ValueError('No actual observations; refusing text-only review')
    return content,manifest
