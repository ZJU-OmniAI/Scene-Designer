"""Check numeric feedback without auxiliary scene files or inferred geometry."""
import math
from collections import Counter
from .candidates import decode, CHECKS


def position(value):
    if not isinstance(value, dict) or not all(k in value for k in ('x','y','z')):
        return None
    xyz = [value[k] for k in ('x','y','z')]
    if not all(type(x) in (int,float) and math.isfinite(x) for x in xyz):
        return None
    return xyz


def check_feedback_continuity(episodes, tolerance=.03):
    """Compare an exact prior returned pose to the next recorded input pose.

    Do not infer missing positions, approximate a snap, or reject absence of
    evidence. Unknown effects invalidate the tracked value. A zero-distance
    snap is ambiguous because native push-out may precede its early return.
    Stage boundaries are not inferred from these local feedback checks.
    """
    issues=[];checked=[]
    readonly=CHECKS | {'observe_scene','get_current_scene_state', 'generate_assets',
                     'generate_wall_assets','generate_ceiling_assets'}
    for stage,ep in episodes.items():
        if stage=='floor_plan':continue
        known={}
        for ti,turn in enumerate(ep['turns']):
            calls=turn.get('tool_calls') or []
            if not calls:continue
            results=ep['turns'][ti+1]['results']
            identities=[]
            for ci,c in enumerate(calls):
                if c['name'] in readonly:continue
                result=decode(results[ci]['output'])
                oid=(c.get('arguments') or {}).get('object_id')
                if c['name'].startswith(('add_','place_')) and isinstance(result,dict):oid=result.get('object_id')
                if oid:identities.append(oid)
            ambiguous={oid for oid,count in Counter(identities).items() if count>1}
            for oid in ambiguous:known.pop(oid,None)
            for ci,call in enumerate(calls):
                name=call['name'];args=call.get('arguments') or {};out=decode(results[ci]['output'])
                if name in readonly:continue
                oid=args.get('object_id')
                if not isinstance(out,dict) or out.get('success') is not True:
                    if oid:known.pop(oid,None)
                    else:known.clear()
                    continue
                if name.startswith(('add_','place_')):oid=out.get('object_id')
                # Parallel tool batches do not establish same-object order.
                if oid in ambiguous:continue
                old=known.get(oid);observed=position(out.get('original_position'))
                if old is not None and observed is not None:
                    error=math.dist(old['position'],observed)
                    row=dict(stage=stage,turn=ti,call_index=ci,object_id=oid,
                             previous_turn=old['turn'],expected=old['position'],observed=observed,
                             position_error_m=error,tolerance_m=tolerance)
                    checked.append(row)
                    if error>tolerance:
                        issues.append(dict(**row,reason='recorded output position disagrees with subsequent input position'))
                if name=='snap_to_object_tool':
                    new=position(out.get('new_position'))
                    # A positive snap reports the actual final transform; the
                    # early-return case may report a stale pre-push-out pose.
                    distance=out.get('distance_moved')
                    if new is not None and type(distance) in (int,float) and distance>=.001:
                        known[oid]=dict(position=new,turn=ti)
                    else:known.pop(oid,None)
                elif name.startswith(('add_','place_','move_')):
                    new=position(out.get('position')) or position(out.get('new_position'))
                    if new is not None:known[oid]=dict(position=new,turn=ti)
                    else:known.pop(oid,None)
                elif name.startswith('remove_'):known.pop(oid,None)
                elif name.startswith('rescale_'):pass  # Native scale does not translate objects.
                else:known.clear()
    return dict(issues=issues,comparisons=checked,scope='exact recorded positions within stages; unknown effects and missing evidence are not failures')


def _new_no_movement_failures(stage, ep, drops):
    """Detect proven native NO_MOVEMENT failures using explicit returned poses.

    Missing poses remain unknown. In particular, an absolute target is not an
    observed endpoint: native placement noise can change it.
    """
    known={};changed=set();failures=[]
    readonly=CHECKS | {'observe_scene','get_current_scene_state','generate_assets',
                     'generate_wall_assets','generate_ceiling_assets'}
    for ti,t in enumerate(ep['turns']):
        calls=t.get('tool_calls') or []
        if (stage,ti) in drops:
            changed.update(c.get('arguments',{}).get('object_id') for c in calls if c['name'] not in readonly)
            continue
        if not calls:continue
        # Parallel mutations have no authoritative same-object ordering.
        mutations=[c for c in calls if c['name'] not in readonly]
        if len(mutations)>1:
            known.clear();changed.clear();continue
        for ci,c in enumerate(calls):
            name=c['name'];args=c.get('arguments') or {};oid=args.get('object_id')
            if name in readonly:continue
            out=decode(ep['turns'][ti+1]['results'][ci]['output'])
            if name=='move_furniture_tool' and oid in changed and oid in known:
                target=[args.get('x'),args.get('y'),0.,0.,0.,args.get('yaw',0.)]
                if all(type(v) in (int,float) and math.isfinite(v) for v in target):
                    if all(abs(a-b)<=1e-6+1e-5*abs(b) for a,b in zip(known[oid],target)):
                        failures.append(dict(stage=stage,turn=ti,reason='deletion makes retained successful move return NO_MOVEMENT'))
            changed.discard(oid)
            if not isinstance(out,dict) or out.get('success') is not True:
                if oid:known.pop(oid,None)
                else:known.clear()
                continue
            if name.startswith(('add_','place_','move_')):
                oid=out.get('object_id') or oid
                xyz=position(out.get('position')) or position(out.get('new_position'))
                rot=out.get('rotation')
                rpy=[rot.get(k) for k in ('roll','pitch','yaw')] if isinstance(rot,dict) else []
                if xyz is not None and len(rpy)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in rpy) and rpy[:2]==[0.,0.]:
                    # Native compares Euler angles recovered from the transform,
                    # not the unwrapped yaw reported by a placement operation.
                    yaw=math.degrees(math.atan2(math.sin(math.radians(rpy[2])),math.cos(math.radians(rpy[2]))))
                    known[oid]=xyz+[0.,0.,yaw]
                else:known.pop(oid,None)
            elif name.startswith('rescale_'):pass
            elif oid:known.pop(oid,None)
            else:known.clear()
    return failures


def check_deletion_dependencies(episodes, drops, arithmetic_stages=()):
    """No files needed: an absolute overwrite is independent of earlier poses.

    Relative snaps do not generally overwrite all coordinates. Do not transplant
    their old original_position/distance feedback onto a changed input.
    """
    failures=[]
    absolute={'move_furniture_tool','move_wall_object','move_ceiling_object'}
    pose_only=absolute | {'snap_to_object_tool'}
    for stage,ep in episodes.items():
        if stage in arithmetic_stages:continue
        failures.extend(_new_no_movement_failures(stage,ep,drops))
        pending=[]
        for ti,t in enumerate(ep['turns']):
            if t.get('role')=='tool_results':continue
            calls=t.get('tool_calls') or []
            if (stage,ti) in drops:
                pending.extend(c for c in calls if c['name']!='check_physics')
                continue
            if not pending:continue
            reason=None
            if len(calls)!=1 or calls[0]['name'] not in absolute:
                reason='retained action is not an absolute pose overwrite; prior pose dependency/feedback cannot be preserved'
            else:
                c=calls[0];oid=(c.get('arguments') or {}).get('object_id')
                if any(x['name'] not in pose_only or (x.get('arguments') or {}).get('object_id')!=oid for x in pending):
                    reason='deleted actions change more than the overwritten object pose'
                out=decode(ep['turns'][ti+1]['results'][0]['output'])
                if not isinstance(out,dict) or out.get('success') is not True:
                    reason='retained overwrite is not recorded as successful'
                elif any(k.startswith(('original_','previous_','current_')) and v is not None for k,v in out.items()):
                    reason='retained feedback describes the old input state'
            if reason:failures.append(dict(stage=stage,turn=ti,reason=reason))
            pending=[]
        if pending:failures.append(dict(stage=stage,reason='deleted actions have no retained overwrite'))
    return failures
