"""Align SFT calls to existing pre-action snapshots, with explicit provenance."""
import copy
import hashlib
import json
import re
from pathlib import Path
from .model import STAGES
from .candidates import decode
from .offline_check import differences, UnknownEffect
from .offline_scene import MUTATIONS, snapshot_state, wall_surfaces
from .offline_snap import SnapGeometry


def state_delta(a,b,position_tol=.03,angle_tol=3.,scale_tol=1e-4):
    # Placement coordinates describe a pose, so use pose tolerances here too.
    aa=copy.deepcopy(a);bb=copy.deepcopy(b);extra=[]
    for oid in a.keys() & b.keys():
        p,q=aa[oid].pop('placement_info',None),bb[oid].pop('placement_info',None)
        if p==q:continue
        if not p or not q or p.get('parent_surface_id')!=q.get('parent_surface_id') or p.get('placement_method')!=q.get('placement_method'):
            extra.append(dict(object_id=oid,field='placement_info'));continue
        import math
        if math.dist(p['position_2d'],q['position_2d'])>position_tol or abs(math.degrees((p['rotation_2d']-q['rotation_2d']+math.pi)%(2*math.pi)-math.pi))>angle_tol:
            extra.append(dict(object_id=oid,field='placement_pose'))
    return differences(aa,bb,position_tol,angle_tol,scale_tol)+extra


def normalized(name,args):
    args=copy.deepcopy(args);args.pop('kwargs',None)
    if name in ('_move_furniture_impl','_add_furniture_to_scene_impl'):
        for k in ('yaw','z','roll','pitch'):args.setdefault(k,0.)
    if name=='_snap_to_object_impl':args.setdefault('orientation','none')
    if any(x in name for x in ('_move_wall','_place_wall','_move_ceiling','_place_ceiling')):args.setdefault('rotation_degrees',0.)
    return name,args


def action_stage(name):
    if 'wall_object' in name:return 'wall'
    if 'ceiling_object' in name:return 'ceiling'
    return 'furniture'


class MissingEvidence(UnknownEffect):
    """Absent verification material, not an observed state mismatch."""


class RecordedEvidence:
    def __init__(self,room_dir,room,episodes,position_tol=.03,angle_tol=3.,scale_tol=1e-4,allow_missing=False):
        self.root=Path(room_dir).resolve();self.room=room;self.episodes=episodes
        self.tolerances=(position_tol,angle_tol,scale_tol)
        self.hashes={};self.snapshots={};self.before={};self.after={};self.call_records={};self.call_inputs={};self.entries={};self.exits={};self.contexts={};self.issues=[]
        self.skipped_checks=[];self.complete_stages=set();self.handoffs=[]
        self.layout=self.read(self.root.parent/'house_layout.json')
        self.log=self.read(self.root/'action_log.json')
        from omegaconf import OmegaConf
        config_path=self.root.parent.parent/'resolved_config.yaml'
        self.record(config_path);self.config=OmegaConf.to_container(OmegaConf.load(config_path),resolve=False)
        grouped={s:[] for s in STAGES[1:]}
        for action in self.log:
            if action['tool_name'] in MUTATIONS.values():grouped[action_stage(action['tool_name'])].append(action)
        for stage in STAGES[1:]:
            try:
                self._load_stage(stage,grouped[stage])
                self.complete_stages.add(stage)
            except MissingEvidence as exc:
                if not allow_missing:raise
                self.skipped_checks.append(dict(stage=stage,status='unverified',reason=str(exc)))
        self.handoffs=self._handoffs()

    def record(self,path):
        path=Path(path).resolve()
        if not path.is_file():raise MissingEvidence(f'missing verification file: {path}')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        self.hashes[str(path)]=digest

    def read(self,path):
        self.record(path);return json.loads(Path(path).read_text())

    def delta(self,a,b):return state_delta(a,b,*self.tolerances)

    def _load_stage(self,stage,actions):
        paths={int(p.parent.name.split('_')[3]):p for p in (self.root/'scene_renders'/stage).glob(f'per_step_{stage}_*/scene_state.json')}
        if not actions:raise MissingEvidence(f'{stage}: no recorded actions')
        # observe_scene also saves genuine snapshots. Use one only when its
        # timestamp lies between consecutive mutation log entries; never infer
        # an earlier state by subtracting objects from a later snapshot.
        from datetime import datetime,timezone
        times=[datetime.fromisoformat(a['timestamp']).replace(tzinfo=timezone.utc).timestamp() for a in actions]
        observations=[]
        for p in (self.root/'scene_renders'/stage).glob('renders_*/scene_state.json'):
            snap=self.read(p);observations.append((snap.get('timestamp',float('inf')),p))
        for n in range(1,len(actions)+1):
            if n in paths:continue
            lo=times[n-2] if n>1 else float('-inf');hi=times[n-1]
            eligible=[(stamp,p) for stamp,p in observations if lo<stamp<hi]
            if eligible:paths[n]=max(eligible,key=lambda item:item[0])[1]
        if 1 not in paths:raise MissingEvidence(f'{stage}: entry snapshot missing')
        end_name={'furniture':'furniture_only_pre_projection','wall':'scene_after_wall_objects','ceiling':'scene_after_ceiling_objects'}[stage]
        endpoint=self.root/'scene_states'/end_name/'scene_state.json'
        if not endpoint.is_file():raise MissingEvidence(f'{stage}: missing unprojected stage endpoint {endpoint}')
        end_snap=self.read(endpoint);end_state=snapshot_state(end_snap)
        loaded={n:self.read(p) for n,p in paths.items()}
        states={n:snapshot_state(snap) for n,snap in loaded.items()}
        self.snapshots[stage]=loaded
        self.entries[stage]=states[1];self.exits[stage]=end_state
        self.contexts[stage]=dict(geometry=loaded[1]['room_geometry'],surfaces=wall_surfaces(self.layout,self.room,loaded[1]['room_geometry']))
        cursor=0;current=copy.deepcopy(states[1]);turns=self.episodes[stage]['turns']
        for ti,turn in enumerate(turns):
            if turn.get('role')!='assistant':continue
            self.before[stage,ti]=copy.deepcopy(current)
            for ci,call in enumerate(turn.get('tool_calls',[])):
                self.call_inputs[stage,ti,ci]=copy.deepcopy(current)
                if call['name'] not in MUTATIONS:continue
                signature=normalized(MUTATIONS[call['name']],call.get('arguments') or {})
                match=next((j for j in range(cursor,len(actions)) if normalized(actions[j]['tool_name'],actions[j]['arguments'])==signature),None)
                if match is None:
                    raise UnknownEffect(f'{stage} turn {ti}: call cannot be aligned to original action log')
                # Never silently jump over a discarded state-changing source step.
                for j in range(cursor,match):
                    pre=states.get(j+1);post=states.get(j+2) if j+1<len(actions) else end_state
                    if pre is None or post is None:
                        self.skipped_checks.append(dict(stage=stage,turn=ti,status='unverified',reason=f'omitted action {j+1}: snapshots unavailable'))
                    elif self.delta(pre,post):
                        self.issues.append(dict(stage=stage,turn=ti,reason=f'input omits state-changing original action {j+1}'))
                n=match+1
                if n not in states:raise MissingEvidence(f'{stage} action {n}: missing pre-action snapshot')
                post=states.get(n+1) if n<len(actions) else end_state
                if post is None:raise MissingEvidence(f'{stage} action {n}: missing next snapshot')
                delta=self.delta(current,states[n])
                if delta:self.issues.append(dict(stage=stage,turn=ti,reason='source step input is disconnected',differences=delta))
                # First call establishes the measured input for the whole batch.
                if ci==0:self.before[stage,ti]=copy.deepcopy(states[n])
                result=turns[ti+1]['results'][ci]['output']
                self.call_records[stage,ti,ci]=dict(before=states[n],after=post,step=n,snapshot=loaded[n],output=result)
                current=copy.deepcopy(post);cursor=match+1
            self.after[stage,ti]=copy.deepcopy(current)
        for j in range(cursor,len(actions)):
            pre=states.get(j+1);post=states.get(j+2) if j+1<len(actions) else end_state
            if pre is None or post is None:
                self.skipped_checks.append(dict(stage=stage,status='unverified',reason=f'tail action {j+1}: snapshots unavailable'))
            elif self.delta(pre,post):self.issues.append(dict(stage=stage,reason=f'input lost final original action {j+1}'))
        if self.delta(current,end_state):self.issues.append(dict(stage=stage,reason='source stage tail differs from recorded endpoint',differences=self.delta(current,end_state)))

    def context(self,stage,ti,ci=0):
        context=dict(self.contexts[stage]);record=self.call_records.get((stage,ti,ci))
        if record and stage=='furniture':
            context['snap']=SnapGeometry(record['snapshot'],self.config['furniture_agent']['snap_to_object'],self.root,entry_step=record['step'])
        return context

    def _handoffs(self):
        rows=[]
        floor=self._floor_check() if 'furniture' in self.contexts else dict(status='unverified',reason='furniture entry evidence unavailable')
        rows.append(dict(from_stage='floor_plan',to_stage='furniture',**floor))
        for a,b in (('furniture','wall'),('wall','ceiling')):
            if a not in self.exits or b not in self.entries or a not in self.contexts or b not in self.contexts:
                rows.append(dict(from_stage=a,to_stage=b,status='unverified',reason='stage boundary evidence unavailable'))
                continue
            delta=self.delta(self.exits[a],self.entries[b])
            ga=self.contexts[a]['geometry'];gb=self.contexts[b]['geometry']
            geometry_delta=[k for k in ('width','length','wall_height','wall_thickness','openings') if ga.get(k)!=gb.get(k)]
            rows.append(dict(from_stage=a,to_stage=b,status='mismatch' if delta or geometry_delta else 'consistent_under_model',differences=delta,geometry_differences=geometry_delta,
                             scope='recorded full-room asset poses and room geometry; framework projection must fit configured tolerances'))
        return rows

    def _floor_check(self):
        """Floor plan is kept intact. Validate final declared shell against room entry.

        No floor-plan tool is re-executed and no claim is made about materials or
        simulation. Door/window identity and size are matched to saved layout.
        """
        turns=self.episodes['floor_plan']['turns'];specs=None;opening_calls=[]
        for ti,t in enumerate(turns):
            for ci,c in enumerate(t.get('tool_calls',[])):
                result=decode(turns[ti+1]['results'][ci]['output'])
                success=result.get('success') is True if isinstance(result,dict) else 'success=True' in str(result)
                if not success:continue
                if c['name']=='generate_room_specs':
                    specs=json.loads(c['arguments']['room_specs_json']);opening_calls=[]
                if c['name'] in ('add_door','add_window'):opening_calls.append((c,str(result)))
                if c['name'] in ('remove_room','resize_room','move_room','remove_door','remove_window'):
                    return dict(status='unverifiable',reason='floor-plan editing operation requires explicit layout arithmetic')
        if specs is None:return dict(status='unverifiable',reason='missing successful floor-plan specification')
        spec=next((r for r in specs if r.get('id',r.get('type'))==self.room),None)
        if spec is None:return dict(status='unverifiable',reason='room missing from final floor-plan specs')
        geometry=self.contexts['furniture']['geometry'];placed=next((r for r in self.layout['placed_rooms'] if r['room_id']==self.room),None)
        if placed is None:return dict(status='unverifiable',reason='room missing from saved layout')
        delta=[];tol=self.tolerances[0]
        for name,x,y in [('room_x',spec['width'],geometry['length']),('room_y',spec['depth'],geometry['width']),('layout_x',placed['width'],geometry['length']),('layout_y',placed['depth'],geometry['width']),('height',self.layout['wall_height'],geometry['wall_height'])]:
            if abs(x-y)>tol:delta.append(dict(field=name,expected=x,actual=y))
        openings={o['opening_id']:o for w in placed['walls'] for o in w.get('openings',[])}
        recorded={o['opening_id']:o for o in geometry.get('openings',[])}
        if set(openings)!=set(recorded):delta.append(dict(field='opening_identity'))
        for oid in openings.keys() & recorded.keys():
            for key in ('width','height','sill_height'):
                if abs(openings[oid].get(key,0)-recorded[oid].get(key,0))>tol:delta.append(dict(field=f'{oid}.{key}'))
        for c,result in opening_calls:
            ids=re.findall(r"(?:door|window)_\d+",result)
            if not ids:return dict(status='unverifiable',reason='opening result identity unavailable')
            oid=ids[0]
            if oid not in openings:delta.append(dict(field='declared_opening_missing',object_id=oid));continue
            for key in ('width','height','sill_height'):
                if key in c['arguments'] and abs(c['arguments'][key]-openings[oid].get(key,0))>tol:delta.append(dict(field=f'declared_{oid}.{key}'))
        return dict(status='mismatch' if delta else 'consistent_under_model',differences=delta,scope='unchanged floor plan: final room dimensions, height and opening identities/sizes')

    def verify_unchanged(self):
        for path,digest in self.hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise UnknownEffect(f'source evidence changed during run: {path}')
