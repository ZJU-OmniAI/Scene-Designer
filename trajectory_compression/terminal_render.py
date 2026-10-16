"""Render a terminal observation from a verified existing next-stage entry blend.

Never executes scene actions, physics, or asset generation. Requires the raw
furniture trace to end at the requested batch and checks every furniture asset
against the donor mesh positions and world bounding boxes before rendering.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from .vision import ROOT
from .terminal_images import MUTATIONS


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-image',type=Path,required=True)
    p.add_argument('--last-step',type=int,required=True)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--height',type=int,required=True)
    p.add_argument('--views',nargs='+',choices=['topdown','south','east','north','west'],default=['topdown'])
    a=p.parse_args(argv);folder=a.source_image.resolve().parent;sid=folder.parent.name;room=folder.name
    stage=a.source_image.name.split('_')[0]
    if stage not in ('furniture','wall'):raise ValueError('requires a recorded next-stage donor; ceiling has no guaranteed donor')
    raw=ROOT/'datasets/raw'/sid;fpath=raw/f'{stage}_{room}.json';f=json.loads(fpath.read_text())
    furniture_path=raw/f'furniture_{room}.json';furniture=json.loads(furniture_path.read_text())
    ops=[t for t in f['designer_trace'] if t.get('role')=='tool_call' and t.get('name') in MUTATIONS]
    if len(ops)!=a.last_step:raise ValueError('requested state is not the end of the raw stage trace')
    next_stage='wall' if stage=='furniture' else 'ceiling'
    blends=list(folder.glob(f'{next_stage}_1__*.blend'))
    if len(blends)!=1:raise ValueError('no unique next-stage entry blend')
    blend=blends[0];floorpath=raw/'floor_plan.json';floor=json.loads(floorpath.read_text())
    spec=next(r for r in floor['rooms'] if r['id']==room)
    L=spec['length'];W=spec['width'];H=floor['wall_height']
    dependencies={str(x):digest(x) for x in (fpath,furniture_path,floorpath,blend)}
    sys.path.insert(0,str(ROOT/'scenesmith'))
    import bpy
    import numpy as np
    from mathutils import Vector,Quaternion
    from scipy.spatial.transform import Rotation
    from scenesmith.agent_utils.blender.renderer import BlenderRenderer
    from scenesmith.agent_utils.blender.params import RenderParams
    from scenesmith.agent_utils.house import ClearanceOpeningData
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    objects=[dict(x,object_type='furniture') for x in furniture['placed_furniture']]
    if stage=='wall':objects += [dict(x,name=x['object_id'],object_type='wall_mounted') for x in f['placed_wall_objects']]
    groups={o['object_id']:[] for o in objects};unmatched=[]
    for mesh in bpy.context.scene.objects:
        if mesh.type!='MESH':continue
        pos=mesh.matrix_world.translation
        dims=mesh.dimensions
        structural_floor=mesh.name.startswith('Mesh_') and dims.z<.15 and dims.x>=L-.1 and dims.y>=W-.1
        structural_wall=mesh.name.startswith('geometry_') and min(dims.x,dims.y)<.11 and abs(dims.z-H)<.1 and (abs(abs(pos.x)-L/2)<.1 or abs(abs(pos.y)-W/2)<.1)
        if structural_floor or structural_wall or mesh.name.startswith(('frame','glass')):continue
        best=min(objects,key=lambda o:(pos-Vector(o['translation'])).length)
        distance=(pos-Vector(best['translation'])).length
        if distance<=.03:
            groups[best['object_id']].append(mesh);continue
        unmatched.append(mesh.name)
    if unmatched:raise ValueError(f'donor contains unaccounted-for meshes: {unmatched[:8]}')
    comparisons=[];metadata=[]
    for o in objects:
        meshes=groups[o['object_id']]
        if not meshes:raise ValueError(f'donor missing object {o["object_id"]}')
        pts=[m.matrix_world@Vector(c) for m in meshes for c in m.bound_box]
        bounds=[min(p[i] for p in pts) for i in range(3)]+[max(p[i] for p in pts) for i in range(3)]
        rotation=Quaternion(o['rotation_wxyz']).to_matrix();position=Vector(o['translation'])
        recorded_bounds='bbox_min' in o and 'bbox_max' in o
        if recorded_bounds:lo=o['bbox_min'];hi=o['bbox_max']
        else:
            local=[rotation.transposed()@(point-position) for point in pts]
            lo=[min(p[i] for p in local) for i in range(3)];hi=[max(p[i] for p in local) for i in range(3)]
        corners=[position+rotation@Vector((x,y,z)) for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]
        expected=[min(p[i] for p in corners) for i in range(3)]+[max(p[i] for p in corners) for i in range(3)]
        error=max(abs(x-y) for x,y in zip(bounds,expected))
        if recorded_bounds and error>.03:raise ValueError(f'donor geometry disagrees with {o["object_id"]}: {error}m')
        comparisons.append(dict(object_id=o['object_id'],max_bbox_error_m=error if recorded_bounds else None,mesh_count=len(meshes),bounds_source='recorded' if recorded_bounds else 'donor_mesh',max_origin_error_m=max((m.matrix_world.translation-position).length for m in meshes)))
        rot=np.asarray(rotation);pos=np.asarray(position);lo=np.array(lo);hi=np.array(hi)
        metadata.append(dict(name=o['name'],object_id=o['object_id'],object_type=o['object_type'],position=pos.tolist(),rotation_matrix=rot.tolist(),bounding_box=dict(center=(pos+rot@((lo+hi)/2)).tolist(),extents=(hi-lo).tolist()),parent_surface_id=None))
    for name,pos,ext in [('north_wall',[0,W/2-.025,H/2],[L+.05,.05,H]),('south_wall',[0,-W/2+.025,H/2],[L+.05,.05,H]),('east_wall',[L/2-.025,0,H/2],[.05,W,H]),('west_wall',[-L/2+.025,0,H/2],[.05,W,H])]:
        metadata.append(dict(name=name,object_id=name,object_type='wall',position=pos,rotation_matrix=np.eye(3).tolist(),bounding_box=dict(center=pos,extents=ext)))
    openings=[]
    for x in floor['doors']+floor['windows']:
        if room not in (x.get('room_id'),x.get('room_a'),x.get('room_b')):continue
        direction=x.get('wall_direction') or floor['boundary_labels'][x['boundary_label']][2]
        offset=x.get('position_along_wall',x.get('position_exact'));half=x['width']/2;h=x['height'];s=x.get('sill_height',0)
        if direction in ('north','south'):
            y=W/2 if direction=='north' else -W/2;start=[-L/2,y];end=[L/2,y];center=[-L/2+offset+half,y,s+h/2]
        else:
            xx=L/2 if direction=='east' else -L/2;start=[xx,-W/2];end=[xx,W/2];center=[xx,-W/2+offset+half,s+h/2]
        openings.append(ClearanceOpeningData(x['id'],'door' if 'door_type' in x else 'window',direction,center,x['width'],s,h,None,None,start,end,offset+half))
    class ExistingBlend(BlenderRenderer):
        def _import_and_organize_gltf(self,path):self._client_objects=bpy.data.collections['ClientObjects']
    params=RenderParams(scene=blend,scene_sha256=dependencies[str(blend)],image_type='color',width=a.width,height=a.height,near=.01,far=100.,focal_x=500,focal_y=500,fov_x=math.pi/4,fov_y=math.pi/4,center_x=a.width/2,center_y=a.height/2)
    stem=f'{stage}_terminal_{a.last_step}'
    manifest=folder/f'{stem}.json'
    manifest.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix='.terminal-render-',dir=folder) as tmp:
        renderer=ExistingBlend(blend_file=blend)
        renderer.render_agent_observation_views(params,Path(tmp),'single_top',a.width,a.height,0,1024,768,
            scene_objects=metadata,annotations=dict(enable_set_of_mark_labels=True,enable_bounding_boxes=True,enable_direction_arrows=True,enable_partial_walls=False,rendering_mode='furniture',enable_support_surface_debug=False,enable_convex_hull_debug=False,annotate_object_types=None,enable_coordinate_grid=True,show_coordinate_frame=True),
            wall_normals={'north_wall':[0,-1,0],'south_wall':[0,1,0],'east_wall':[-1,0,0],'west_wall':[1,0,0]},openings=openings)
        if any(digest(Path(k))!=v for k,v in dependencies.items()):raise ValueError('donor evidence changed during render')
        image=folder/f'{stem}.topdown.png';shutil.copy2(Path(tmp)/'0_top.png',image)
    from .offline_output import image_info
    images={'topdown':image_info(image)}
    # Reuse already rendered cardinal views of the SAME verified donor blend.
    # Rendering code and pixel dimensions stay unchanged.
    for view in ('south','east','north','west'):
        if view not in a.views:continue
        source=blend.with_suffix('.'+view+'.png')
        if source.is_file():
            target=folder/f'{stem}.{view}.png';shutil.copy2(source,target);images[view]=image_info(target)
    record=dict(stage=stage,completed_mutations=a.last_step,relation='verified_existing_next_stage_entry',
        source_image=str(a.source_image.resolve()),raw_trace_sha256=dependencies[str(fpath)],dependencies=dependencies,
        raw_terminal_actions=[dict(name=x['name'],arguments=x['arguments']) for x in ops],
        geometry_comparisons=comparisons,images=images,tolerance_m=.03,full_physics_verified=False)
    temporary=manifest.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');temporary.replace(manifest)
    print(json.dumps(dict(status='rendered_verified_donor',manifest=str(manifest),objects_checked=len(comparisons))),flush=True)

if __name__=='__main__':main()
