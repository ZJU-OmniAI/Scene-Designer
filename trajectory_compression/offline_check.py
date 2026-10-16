"""Offline pose consistency, without rendering, scene execution or GPT.

Compare original/compressed action calculations. Optional independently recorded
post-action states strengthen this to an endpoint pose comparison. Nominal action
agreement alone is not physical verification or permission to reuse stale images.
Units: metres, XYZ Euler angles in degrees (Rz @ Ry @ Rx), multiplicative scale.
"""
import argparse
import copy
import json
import math
import re
from pathlib import Path
from .candidates import candidates, deletable_turns, decode

class UnknownEffect(ValueError): pass


def rotation(rpy):
    x,y,z=map(math.radians,rpy);cx,sx=math.cos(x),math.sin(x);cy,sy=math.cos(y),math.sin(y);cz,sz=math.cos(z),math.sin(z)
    return [[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],
            [sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]]


def angle_error(a,b):
    # Geodesic distance on SO(3), not subtraction of wrapped Euler angles.
    a,b=rotation(a),rotation(b)
    trace=sum(a[i][j]*b[i][j] for i in range(3) for j in range(3))
    return math.degrees(math.acos(max(-1.,min(1.,(trace-1)/2))))


def finite_vector(value,n=3):
    return isinstance(value,(list,tuple)) and len(value)==n and all(type(x) in (int,float) and math.isfinite(x) for x in value)


def validate_state(state):
    if not isinstance(state,dict):raise ValueError('state must map object IDs to asset states')
    for oid,obj in state.items():
        if not isinstance(obj,dict):raise ValueError(f'{oid}: invalid asset state')
        for key in ('position','rotation','scale','dimensions'):
            value=obj.get(key)
            if value is not None and not finite_vector(value):raise ValueError(f'{oid}: invalid {key}')
            if key in ('scale','dimensions') and value is not None and any(x<=0 for x in value):
                raise ValueError(f'{oid}: nonpositive {key}')


def differences(a,b,position_tol,angle_tol,scale_tol):
    result=[]
    for oid in sorted(a.keys()|b.keys()):
        if oid not in a or oid not in b:
            result.append({'object_id':oid,'field':'existence'});continue
        for key in set(a[oid])|set(b[oid]):
            x,y=a[oid].get(key),b[oid].get(key)
            if x==y:continue
            if x is None or y is None:
                result.append({'object_id':oid,'field':key,'reason':'unknown vs known'});continue
            if key=='position':error=math.dist(x,y);limit=position_tol
            elif key=='rotation':error=angle_error(x,y);limit=angle_tol
            elif key in ('scale','dimensions'):
                error=max(abs(v-w) for v,w in zip(x,y));limit=scale_tol
            else:
                result.append({'object_id':oid,'field':key,'reason':'value differs'});continue
            if error>limit:result.append({'object_id':oid,'field':key,'error':error,'tolerance':limit})
    return result


def apply_action(state,call,output,stage,snap_geometry=None):
    name=call['name'];args=call.get('arguments') or {};output=decode(output)
    if not isinstance(output,dict) or output.get('success') is not True:
        raise UnknownEffect(f'{name}: no successful recorded result; cannot assume no side effects')
    oid=args.get('object_id')
    if name=='add_furniture_to_scene_tool':
        oid=output.get('object_id')
        if not oid or oid in state:raise UnknownEffect('missing/new object identity cannot be verified')
        state[oid]={'position':None,'rotation':None,'scale':[1.,1.,1.],
                    'asset_id':args.get('asset_id')}
        name='move_furniture_tool'
    if oid not in state:raise UnknownEffect(f'{name}: object {oid} does not exist')
    obj=state[oid]
    if obj.get('immutable'):raise UnknownEffect(f'{oid}: immutable asset')
    if name=='move_furniture_tool':
        # Public tool signature fixes z=roll=pitch=0, yaw defaults to 0.
        if not {'x','y'}<=args.keys():raise UnknownEffect('incomplete absolute position')
        if set(args)-{'object_id','asset_id','x','y','yaw'}:raise UnknownEffect('unrecognized furniture move arguments')
        pose=[args['x'],args['y'],0.];rpy=[0.,0.,args.get('yaw',0.)]
        if not finite_vector(pose) or not finite_vector(rpy):raise UnknownEffect('nonfinite pose')
        if obj.get('position') is not None and obj.get('rotation') is not None:
            if math.dist(obj['position'],pose)<1e-6 and angle_error(obj['rotation'],rpy)<1e-6:
                raise UnknownEffect('move may become NO_MOVEMENT; recorded success cannot be reused')
        obj['position']=pose;obj['rotation']=rpy
    elif name=='rescale_furniture_tool':
        factor=args.get('scale_factor')
        if type(factor) not in (int,float) or not math.isfinite(factor) or factor<=0:raise UnknownEffect('invalid scale')
        if obj.get('shared_asset'):raise UnknownEffect('shared asset rescale requires affected-object evidence')
        affected=output.get('affected_object_ids')
        if affected is not None and set(affected)!={oid}:raise UnknownEffect('rescale affects additional objects')
        obj['scale']=[x*factor for x in obj.get('scale',[1.,1.,1.])]
        if obj.get('dimensions') is not None:obj['dimensions']=[x*factor for x in obj['dimensions']]
    elif name=='snap_to_object_tool':
        if snap_geometry is None:raise UnknownEffect('snap requires entry snapshot geometry and snap configuration')
        position,rpy=snap_geometry.calculate(state,args)
        obj['position']=position;obj['rotation']=rpy
    elif name=='remove_furniture_tool':del state[oid]
    else:raise UnknownEffect(f'{name}: relative/surface/physics-dependent semantics not implemented')


def check_segment(episode,segment,drops,initial=None,reference_after=None,
                  position_tol=.03,angle_tol=3.,scale_tol=1e-4,snap_geometry=None):
    """Check one original-index interval. No mutation of input or training data.

    initial: independently recorded asset states immediately before segment.start.
    reference_after: independently recorded state immediately after segment.end.
    Without both, results are only nominal-action consistency, not measured poses.
    """
    for t in (position_tol,angle_tol,scale_tol):
        if type(t) not in (int,float) or not math.isfinite(t) or t<0:raise ValueError('invalid tolerance')
    if initial is None and snap_geometry is not None:
        initial=snap_geometry.initial_state()
    drops=set(drops)
    report={'stage':episode.get('stage'),'start':segment['start'],'end':segment['end'],
            'drops':sorted(drops),'status':'unverifiable','training_ready':False,
            'basis':('measured_endpoint' if initial is not None else 'recorded_endpoint_only') if reference_after is not None else 'nominal_actions',
            'tolerances':{'position_m':position_tol,'rotation_deg':angle_tol,'scale_absolute':scale_tol},
            'changed_inputs':[],'warnings':['Collision, support and dynamic simulation are not verified.'],
            'can_reuse_observations':False}
    if not drops<=deletable_turns(segment):
        report['reason']='proposal deletes boundary/protected/out-of-interval action';return report
    if initial is None:
        # Symbolic entry: unknown original pose; scale tracks multiplier relative
        # to the SAME original asset. This is explicitly not a measured scene.
        initial={op['object_id']:{'position':None,'rotation':None,'scale':[1.,1.,1.]}
                 for op in segment['operations'] if op.get('object_id')}
        report['warnings'].append('Entry poses missing: using shared symbolic entry and relative scale multipliers.')
    validate_state(initial)
    if reference_after is not None:validate_state(reference_after)
    original=copy.deepcopy(initial);compressed=copy.deepcopy(initial)
    try:
        for ti in range(segment['start'],segment['end']+1):
            turn=episode['turns'][ti]
            if turn.get('role')!='assistant':continue
            dirty=original!=compressed
            if ti not in drops and dirty:
                report['changed_inputs'].append(ti)
            calls=turn.get('tool_calls',[])
            if not calls:continue
            results=episode['turns'][ti+1]['results']
            if len(calls)!=len(results) or any(c['name']!=r['name'] for c,r in zip(calls,results)):
                raise UnknownEffect('action/result pairing mismatch')
            if ti in drops and len(calls)!=1:raise UnknownEffect('cannot delete partial batch')
            for call,result in zip(calls,results):
                name=call['name']
                # Inspected furniture check: serializes scene to an isolated
                # subprocess and returns text, never writes worker state back.
                if name=='check_physics' and episode.get('stage')=='furniture':
                    if dirty:report['warnings'].append(f'turn {ti}: original physics feedback is stale and cannot be reused')
                    continue
                if name not in ('move_furniture_tool','rescale_furniture_tool','remove_furniture_tool','add_furniture_to_scene_tool','snap_to_object_tool'):
                    # A recorded transition is NOT transplanted into a changed
                    # input. Even when inputs agree, no unknown effect is invented.
                    raise UnknownEffect(f'turn {ti}: {name} cannot be calculated by this checker')
                apply_action(original,call,result['output'],episode.get('stage'),snap_geometry)
                if ti not in drops:apply_action(compressed,call,result['output'],episode.get('stage'),snap_geometry)
        report['original_calculated_state']=original
        report['compressed_calculated_state']=compressed
        delta=differences(original,compressed,position_tol,angle_tol,scale_tol)
        if reference_after is not None:
            source_delta=differences(original,reference_after,position_tol,angle_tol,scale_tol)
            report['original_vs_recorded']=source_delta
            if source_delta:
                report['reason']='original action calculation disagrees with recorded endpoint; model/data mismatch'
                return report
            delta=differences(compressed,reference_after,position_tol,angle_tol,scale_tol)
        report['differences']=delta
        report['status']='mismatch' if delta else 'consistent_under_model'
        report['reason']='asset endpoint differs' if delta else 'asset membership, position, orientation and scale agree within specified tolerances'
        report['can_reuse_observations']=not report['changed_inputs']
        if report['changed_inputs']:report['warnings'].append('Retained step inputs changed: old images/results/memory require matching existing observations or reverting these deletions.')
        if reference_after is None:report['warnings'].append('No measured endpoint supplied; small placement noise is not independently checked.')
    except (UnknownEffect,KeyError,IndexError,TypeError) as exc:
        report['reason']=str(exc)
    return report



def check_handoff(previous_exit,next_entry,position_tol=.03,angle_tol=3.,scale_tol=1e-4):
    """Compare recorded room asset states at a stage handoff, no tool execution."""
    for t in (position_tol,angle_tol,scale_tol):
        if type(t) not in (int,float) or not math.isfinite(t) or t<0:raise ValueError('invalid tolerance')
    validate_state(previous_exit);validate_state(next_entry)
    for state in (previous_exit,next_entry):
        if not state or any(o.get('position') is None or o.get('rotation') is None for o in state.values()):
            return {'status':'unverifiable','reason':'missing measured boundary poses'}
    delta=differences(previous_exit,next_entry,position_tol,angle_tol,scale_tol)
    return {'status':'mismatch' if delta else 'consistent_under_model','differences':delta,
            'scope':'supplied asset poses only; framework transitions must be accounted for separately'}


def state_from_placed_objects(objects):
    """Read raw placed_furniture-style records, converting wxyz quaternions to RPY."""
    result={}
    for obj in objects:
        q=obj['rotation_wxyz']
        if not finite_vector(q,4):raise ValueError('invalid quaternion')
        norm=math.sqrt(sum(x*x for x in q))
        if norm<1e-12:raise ValueError('zero quaternion')
        w,x,y,z=[v/norm for v in q]
        rpy=[math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
             math.asin(max(-1.,min(1.,2*(w*y-z*x)))),
             math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]
        oid=obj['object_id']
        if oid in result:raise ValueError('duplicate object ID')
        result[oid]={'position':obj['translation'],'rotation':list(map(math.degrees,rpy)),
                     'scale':[obj.get('scale_factor',1.)]*3}
    validate_state(result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--segment',required=True,help='segment_id from candidate extraction')
    ap.add_argument('--drop',type=int,nargs='*',default=[])
    ap.add_argument('--initial-state',type=Path)
    ap.add_argument('--reference-after',type=Path)
    ap.add_argument('--scene-before',type=Path,help='scene_state.json immediately before the segment')
    ap.add_argument('--asset-root',type=Path,help='room directory for relative snapshot asset paths')
    ap.add_argument('--snap-config',type=Path,help='original YAML config containing snap_to_object')
    ap.add_argument('--position-tol',type=float,default=.03)
    ap.add_argument('--angle-tol',type=float,default=3.)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args();data=json.loads(args.input.read_text())
    segment=next((s for s in candidates(data['stage'],data) if s['segment_id']==args.segment),None)
    if segment is None:ap.error('segment not found')
    snap_geometry=None
    if args.scene_before:
        if not args.snap_config:ap.error('--scene-before requires --snap-config')
        from .offline_snap import SnapGeometry
        from omegaconf import OmegaConf
        cfg=OmegaConf.to_container(OmegaConf.load(args.snap_config),resolve=False)
        step_match=re.fullmatch(r'per_step_furniture_(\d+)__.*',args.scene_before.parent.name)
        snap_geometry=SnapGeometry(json.loads(args.scene_before.read_text()),(cfg.get('furniture_agent',cfg))['snap_to_object'],asset_root=args.asset_root,
                                   entry_step=int(step_match[1]) if step_match else None)
    report=check_segment(data,segment,args.drop,snap_geometry=snap_geometry,
        initial=json.loads(args.initial_state.read_text()) if args.initial_state else None,
        reference_after=json.loads(args.reference_after.read_text()) if args.reference_after else None,
        position_tol=args.position_tol,angle_tol=args.angle_tol)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text,end='')
    if report['status']!='consistent_under_model':raise SystemExit(1)

if __name__=='__main__':main()
