"""Local pose arithmetic for recorded rooms; no renderer or scene execution."""
import copy
import math
from .offline_check import UnknownEffect, rotation, finite_vector, state_from_placed_objects

MUTATIONS = {
 'add_furniture_to_scene_tool':'_add_furniture_to_scene_impl',
 'move_furniture_tool':'_move_furniture_impl','remove_furniture_tool':'_remove_furniture_impl',
 'rescale_furniture_tool':'_rescale_furniture_impl','snap_to_object_tool':'_snap_to_object_impl',
 **{f'{verb}_{kind}_object':f'_{verb}_{kind}_object_impl' for kind in ('wall','ceiling') for verb in ('place','move','remove','rescale')}}
# Inspected vision/check implementations only query geometry/serialize a worker;
# these do not apply simulated poses back to the recorded RoomScene.
READ_ONLY={'check_physics','check_facing_tool','check_reachability','observe_scene','get_current_scene_state',
           'get_wall_surfaces','get_room_info','list_available_assets','list_available_wall_assets','list_available_ceiling_assets'}
PRODUCERS={'generate_assets','generate_wall_assets','generate_ceiling_assets'}


def snapshot_state(snapshot):
    records=snapshot['objects'];state=state_from_placed_objects([
        dict(object_id=k,translation=o['transform']['translation'],rotation_wxyz=o['transform']['rotation_wxyz'],scale_factor=o.get('scale_factor',1.))
        for k,o in records.items()])
    for key,o in records.items():
        state[key].update(immutable=o.get('immutable',False),object_type=o['object_type'],
                          asset_key=o.get('sdf_path') or o.get('geometry_path'),
                          dimensions=[b-a for a,b in zip(o['bbox_min'],o['bbox_max'])] if o.get('bbox_min') is not None else None,
                          placement_info=copy.deepcopy(o.get('placement_info')))
    return state


def mm(a,b):return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def rpy(matrix):
    pitch=math.asin(max(-1.,min(1.,-matrix[2][0])))
    if abs(math.cos(pitch))>1e-9:
        roll=math.atan2(matrix[2][1],matrix[2][2]);yaw=math.atan2(matrix[1][0],matrix[0][0])
    else:roll=0.;yaw=math.atan2(-matrix[0][1],matrix[1][1])
    return list(map(math.degrees,(roll,pitch,yaw)))


def wall_surfaces(layout,room,geometry):
    """Same origin, axes, wall thickness and opening convention as _create_wall_surface."""
    placed=next(r for r in layout['placed_rooms'] if r['room_id']==room)
    offset=[placed['position'][0]+placed['width']/2,placed['position'][1]+placed['depth']/2]
    axes={'north':(0,[0,-1]),'south':(180,[0,1]),'east':(-90,[-1,0]),'west':(90,[1,0])}
    result={}
    for w in placed['walls']:
        direction=w['direction'];yaw,normal=axes[direction]
        start=[w['start_point'][i]-offset[i] for i in range(2)];end=[w['end_point'][i]-offset[i] for i in range(2)]
        origin_start={'north':start[0]<end[0],'south':start[0]>end[0],'east':start[1]>end[1],'west':start[1]<end[1]}[direction]
        origin=start if origin_start else end
        origin=[origin[i]+normal[i]*geometry['wall_thickness'] for i in range(2)]+[0.]
        length=math.dist(start,end);excluded=[]
        for opening in w.get('openings',[]):
            center=opening['position_along_wall']+opening['width']/2
            if not origin_start:center=length-center
            z=opening.get('sill_height',0.)
            excluded.append([center-opening['width']/2,z,center+opening['width']/2,
                             geometry['wall_height']-z if opening['opening_type']=='open' else z+opening['height']])
        result[w['wall_id']]=dict(origin=origin,yaw=yaw,length=length,height=geometry['wall_height'],excluded=excluded)
    return result


def pose_for_mount(stage,args,obj,context):
    theta=args.get('rotation_degrees',0.)
    if stage=='ceiling':
        p=[args['position_x'],args['position_y'],context['geometry']['wall_height']]
        g=context['geometry']
        if not -g['length']/2<=p[0]<=g['length']/2 or not -g['width']/2<=p[1]<=g['width']/2:
            raise UnknownEffect('ceiling position outside room bounds')
        return p,[0.,0.,theta],None
    surface=context['surfaces'].get(args['wall_surface_id'])
    if surface is None:raise UnknownEffect('wall surface missing')
    x,z=args['position_x'],args['position_z'];dims=obj.get('dimensions')
    if not dims:raise UnknownEffect('wall object dimensions missing')
    lo,hi=x-dims[0]/2,x+dims[0]/2;bottom,top=z-dims[2]/2,z+dims[2]/2
    if lo<0 or hi>surface['length'] or bottom<0 or top>surface['height']:raise UnknownEffect('wall object outside surface bounds')
    if any(lo<b and hi>a and bottom<d and top>c for a,c,b,d in surface['excluded']):raise UnknownEffect('wall object overlaps opening')
    rw=rotation([0,0,surface['yaw']]);local=[x,0,z]
    p=[surface['origin'][i]+sum(rw[i][k]*local[k] for k in range(3)) for i in range(3)]
    orient=rpy(mm(rw,mm(rotation([0,-theta,0]),rotation([0,0,180]))))
    placement=dict(parent_surface_id=args['wall_surface_id'],position_2d=[x,z],rotation_2d=math.radians(theta),placement_method='wall_placement')
    return p,orient,placement


def calculate(state,call,output,stage,context,recorded_after=None):
    """Mutate a private state only. Recorded endpoints supply asset metadata, never poses."""
    from .candidates import decode
    output=decode(output);name=call['name'];args=call.get('arguments') or {}
    if name in READ_ONLY or name in PRODUCERS:return
    if name not in MUTATIONS:raise UnknownEffect(f'unsupported state action: {name}')
    if not isinstance(output,dict) or output.get('success') is not True:
        raise UnknownEffect(f'{name}: failed action needs unchanged-input recorded transition')
    creation=name.startswith(('place_','add_'))
    oid=output.get('object_id') if creation else args.get('object_id')
    if creation:
        if oid in state or not recorded_after or oid not in recorded_after:raise UnknownEffect('new asset identity/metadata missing')
        obj=copy.deepcopy(recorded_after[oid]);obj['position']=None;obj['rotation']=None
        state[oid]=obj
    if oid not in state:raise UnknownEffect(f'object missing: {oid}')
    obj=state[oid]
    if obj.get('immutable'):raise UnknownEffect('cannot modify immutable object')
    if name.startswith('remove_'):del state[oid];return
    if name.startswith('rescale_'):
        factor=args['scale_factor']
        if type(factor) not in (int,float) or not math.isfinite(factor) or factor<=0:raise UnknownEffect('invalid rescale')
        asset=obj.get('asset_key')
        if not asset:raise UnknownEffect('shared rescale asset identity missing')
        affected={k for k,o in state.items() if o.get('asset_key')==asset}
        if output.get('affected_object_ids') is not None and affected!=set(output['affected_object_ids']):raise UnknownEffect('rescale affected objects disagree')
        for k in affected:
            state[k]['scale']=[x*factor for x in state[k]['scale']]
            if state[k].get('dimensions'):state[k]['dimensions']=[x*factor for x in state[k]['dimensions']]
        return
    if name=='snap_to_object_tool':
        snap=context.get('snap')
        if snap is None:raise UnknownEffect('snap geometry unavailable')
        obj['position'],obj['rotation']=snap.calculate(state,args);return
    if stage=='furniture':
        p=[args['x'],args['y'],0.];orient=[0.,0.,args.get('yaw',0.)]
        if not finite_vector(p) or not finite_vector(orient):raise UnknownEffect('invalid furniture pose')
        from .offline_check import angle_error
        if not creation and obj.get('position') is not None and math.dist(obj['position'],p)<1e-6 and angle_error(obj['rotation'],orient)<1e-6:
            raise UnknownEffect('retained move becomes NO_MOVEMENT')
        g=context['geometry']
        if not -g['length']/2<=p[0]<=g['length']/2 or not -g['width']/2<=p[1]<=g['width']/2:raise UnknownEffect('furniture outside room bounds')
        obj['position'],obj['rotation']=p,orient
    else:
        expected='wall_mounted' if stage=='wall' else 'ceiling_mounted'
        if obj.get('object_type')!=expected:raise UnknownEffect('wrong mounted object type')
        obj['position'],obj['rotation'],placement=pose_for_mount(stage,args,obj,context)
        if stage=='wall':obj['placement_info']=placement
