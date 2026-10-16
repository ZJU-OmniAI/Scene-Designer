"""Bind observations by recorded state and retain an auditable transcript."""
import copy
import hashlib
import re
import struct
from pathlib import Path
from .offline_check import UnknownEffect
from .vision import resolve_image


def views(turn):return turn.get('images') or ([turn['image']] if turn.get('image') else [])


def image_info(path):
    path=resolve_image(str(path));data=path.read_bytes()
    if data[:8]!=b'\x89PNG\r\n\x1a\n' or len(data)<24:raise UnknownEffect(f'not a valid PNG observation: {path}')
    # Full decoding, not only a signature check, catches truncated render output.
    from PIL import Image
    with Image.open(path) as im:im.verify()
    m=re.search(r'\.(topdown|south|east|north|west)\.png$',path.name)
    view=m[1] if m else 'topdown' if path.name=='0_top.png' else None
    if view is None:raise UnknownEffect(f'observation has no declared view identity: {path}')
    return dict(path=str(path),view=view,size=list(struct.unpack('>II',data[16:24])),sha256=hashlib.sha256(data).hexdigest())


class ObservationBank:
    def __init__(self,evidence,episodes,allow_missing=False):
        from .terminal_images import terminal_plan
        self.terminal=terminal_plan(episodes)
        self.missing_original=set()
        self.evidence=evidence;self.bank={};self.original={};self.manifest=[]
        for stage in ('furniture','wall','ceiling'):
            bank=[];seen=set()
            for ti,t in enumerate(episodes[stage]['turns']):
                raw=views(t)
                if not raw:continue
                info=[image_info(p) for p in raw]
                if hasattr(evidence,'root'):
                    scene=evidence.root.parent
                    expected=f'{scene.parent.parent.name}_{scene.parent.name}_{scene.name}'
                    for item in info:
                        parts=Path(item['path']).parts
                        if expected not in parts or evidence.root.name[5:] not in parts:
                            raise UnknownEffect('source image belongs to a different scene or room')
                self.original[stage,ti]=info
                if (stage,ti) in self.terminal:
                    continue  # Boundary observations use the next stage entry.
                # All images of a turn must refer to the SAME pre-action snapshot.
                steps=[]
                for item in info:
                    m=re.fullmatch(rf'{stage}_(\d+)__.+\.(?:topdown|south|east|north|west)\.png',Path(item['path']).name)
                    if not m:raise UnknownEffect(f'cannot associate image with original snapshot: {item["path"]}')
                    steps.append(int(m[1]))
                if len(set(steps))!=1:raise UnknownEffect('a turn mixes images from different states')
                n=steps[0]
                if n not in evidence.snapshots.get(stage,{}):
                    if allow_missing:
                        self.missing_original.add((stage,ti));continue
                    from .offline_evidence import MissingEvidence
                    raise MissingEvidence(f'{stage} image snapshot {n} missing')
                from .offline_scene import snapshot_state
                state=snapshot_state(evidence.snapshots[stage][n])
                marker=tuple(x['path'] for x in info)
                if marker not in seen:bank.append((state,info,ti));seen.add(marker)
            self.bank[stage]=bank

        # Search every existing canonical step image, not just the subset that
        # survived older SFT processing. A matching next-stage entry can also be
        # the correct previous-stage final observation (same complete room state).
        from .offline_scene import snapshot_state
        folders={Path(item['path']).parent for info in self.original.values() for item in info}
        modes={tuple(p['view'] for p in info) for info in self.original.values()}
        known={tuple(x['path'] for x in info) for entries in self.bank.values() for _,info,_ in entries}
        extra=[]
        for folder in folders:
            for path in sorted(folder.glob('*.topdown.png')):
                match=re.fullmatch(r'(furniture|wall|ceiling)_(\d+)__.+\.topdown\.png',path.name)
                if not match:continue
                source_stage,n=match[1],int(match[2])
                if n not in evidence.snapshots.get(source_stage,{}):continue
                for mode in modes:
                    paths=[path.with_name(path.name[:-len('topdown.png')]+view+'.png') for view in mode]
                    if not all(p.is_file() for p in paths):continue
                    marker=tuple(str(p.resolve()) for p in paths)
                    if marker in known:continue
                    try:info=[image_info(p) for p in paths]
                    except (UnknownEffect,OSError,ValueError):continue
                    extra.append((snapshot_state(evidence.snapshots[source_stage][n]),info,-1))
                    known.add(marker)
        # Cross-stage reuse is allowed only after full-state, view and resolution
        # matching in bind; never by timestamp proximity or filename alone.
        all_entries=[entry for entries in self.bank.values() for entry in entries]+extra
        for stage in self.bank:
            own=self.bank[stage]
            self.bank[stage]=own+[(state,info,-1) for state,info,_ in all_entries if not any(info==oldinfo for _,oldinfo,_ in own)]

        self.view_bank=[(state,item) for state,info,_ in all_entries for item in info]
        if hasattr(evidence,'root'):
            # Original furniture observe_scene top images use 0_top.png. Reuse
            # those bytes only with their saved state; do not resample a smaller
            # next-stage top image or change any cardinal-view camera.
            for path in sorted((evidence.root/'scene_renders/furniture').glob('renders_*/0_top.png')):
                snapshot=path.parent/'scene_state.json'
                if not snapshot.is_file():continue
                try:
                    state=snapshot_state(evidence.read(snapshot));info=image_info(path)
                except (UnknownEffect,OSError,ValueError):continue
                info['snapshot']=str(snapshot)
                self.view_bank.append((state,info))

    def bind(self,stage,ti,turn,state):
        if (stage,ti) in self.terminal:
            from .terminal_images import bind_terminal
            return bind_terminal(turn,self.terminal[stage,ti])
        info=self.original.get((stage,ti))
        if not info:
            # Preserve original no-image mode, e.g. tool-only/final messages.
            return copy.deepcopy(turn),None
        mode=[(p['view'],p['size']) for p in info]
        candidates=[]
        for recorded,donor,donor_ti in self.bank[stage]:
            if [(p['view'],p['size']) for p in donor]!=mode:continue
            if not self.evidence.delta(state,recorded):candidates.append((donor_ti!=ti,donor_ti,donor))
        if candidates:
            _,donor_ti,donor=min(candidates,key=lambda x:(x[0],x[1]))
        else:
            # A top view and cardinal views may come from different saved
            # observations, provided all show mutually equivalent full states.
            donor=[];selected=[];donor_ti=-1
            for view,size in mode:
                found=next(((recorded,item) for recorded,item in self.view_bank
                            if item['view']==view and item['size']==size
                            and not self.evidence.delta(state,recorded)
                            and all(not self.evidence.delta(other,recorded) for other in selected)),None)
                if found is None:raise UnknownEffect(f'{stage} turn {ti}: no existing images match calculated state, view order and resolution')
                recorded,item=found;selected.append(recorded);donor.append(item)
        result=copy.deepcopy(turn);paths=[p['path'] for p in donor]
        if 'image' in result:result['image']=paths[0]
        if 'images' in result:result['images']=paths
        return result,dict(stage=stage,original_turn=ti,donor_turn=donor_ti,images=copy.deepcopy(donor))


def opaque_memory(turn):
    return any(any(token in key.lower() for token in ('memory','summary','history','context')) for key in turn if key not in ('content',))


def materialize(episodes,drops,before,bank):
    output=copy.deepcopy(episodes);manifest=[];provenance={}
    for stage,ep in output.items():
        retained=[];mapping=[]
        changed=any(s==stage for s,_ in drops)
        if changed and (opaque_memory({k:v for k,v in ep.items() if k not in ('task_context','turns')}) or any(opaque_memory(t) for t in ep['turns'])):raise UnknownEffect(f'{stage}: opaque per-turn memory cannot be regenerated safely')
        first_drop=min((ti for s,ti in drops if s==stage),default=len(ep['turns']))
        for ti,turn in enumerate(ep['turns']):
            if (stage,ti) in drops or (turn.get('role')=='tool_results' and (stage,ti-1) in drops):continue
            if changed and ti>first_drop and turn.get('role') in ('user','system'):
                raise UnknownEffect(f'{stage} turn {ti}: later user/system memory requires explicit regeneration')
            if turn.get('role')=='assistant':
                if stage!='floor_plan':
                    turn,binding=bank.bind(stage,ti,turn,before[stage,ti])
                    if binding:manifest.append(binding)
                # Training target is action-only. Old reasoning can describe deleted
                # trials even when geometric endpoints agree. Re-annotate separately.
                turn['content']=None
            mapping.append(ti);retained.append(turn)
        ep['turns']=retained;provenance[stage]=mapping
        source=ep.get('source',{})
        if not isinstance(source,dict):source={'original_source':source}
        ep['source']=dict(original_source=source,offline_compression=dict(original_turn_indices=mapping,
            deleted_turn_indices=sorted(ti for s,ti in drops if s==stage),reasoning='cleared_for_reannotation'))
    return output,manifest,provenance
