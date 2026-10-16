"""Offline compression with recorded-state checks and transactional deletion trials."""
import copy
from pathlib import Path
from .model import STAGES
from .candidates import candidates,mutation_indices,deletable_turns,dependent_physics_checks

class ValidationError(ValueError):pass

def validate_input(episodes):
    if set(episodes) != set(STAGES):
        raise ValidationError('require exactly floor_plan -> furniture -> wall -> ceiling for one room')
    for stage, ep in episodes.items():
        if ep.get('stage') != stage: raise ValidationError(f'{stage}: wrong stage metadata')
        # Historical truncation metadata is not a rejection criterion.
        # Recorded action/state and stage-handoff checks decide continuity.
        turns = ep.get('turns')
        if not isinstance(turns, list) or not turns: raise ValidationError(f'{stage}: empty trajectory')
        for i, turn in enumerate(turns):
            if turn.get('role') == 'tool_results':
                if i == 0 or not turns[i-1].get('tool_calls'): raise ValidationError('orphan results')
            calls = turn.get('tool_calls') or []
            if not calls: continue
            if turn.get('role') != 'assistant' or i+1 >= len(turns) or turns[i+1].get('role') != 'tool_results':
                raise ValidationError('missing paired results')
            results = turns[i+1].get('results', [])
            if len(calls) != len(results) or any(c.get('name') != r.get('name') for c,r in zip(calls,results)):
                raise ValidationError('call/result pairing mismatch')

class Compressor:
    """Offline production path. Evidence and arithmetic replace scene replay."""
    def __init__(self,evidence,reviewer):
        self.evidence=evidence;self.reviewer=reviewer;self._baseline_math={}

    def _calculate(self,episodes,drops,stages=None):
        from .offline_check import UnknownEffect
        from .offline_scene import calculate, MUTATIONS, READ_ONLY, PRODUCERS
        ev=self.evidence;before={};after={};checks=[]
        for stage in (STAGES[1:] if stages is None else stages):
            state=copy.deepcopy(ev.entries[stage])
            for ti,t in enumerate(episodes[stage]['turns']):
                if t.get('role')!='assistant':continue
                if (stage,ti) in drops:continue
                before[stage,ti]=copy.deepcopy(state)
                for ci,c in enumerate(t.get('tool_calls',[])):
                    output=episodes[stage]['turns'][ti+1]['results'][ci]['output']
                    record=ev.call_records.get((stage,ti,ci))
                    expected_input=getattr(ev,'call_inputs',{}).get((stage,ti,ci),ev.before[stage,ti])
                    if c['name'] in MUTATIONS:
                        if record is None:raise UnknownEffect(f'{stage}:{ti}: missing call evidence')
                        # First verify the original action's model against its own
                        # recorded endpoint; a later overwrite cannot hide a bug.
                        key=(stage,ti,ci)
                        if key not in self._baseline_math:
                            baseline=copy.deepcopy(record['before'])
                            try:
                                calculate(baseline,c,output,stage,ev.context(stage,ti,ci),record['after'])
                                delta=ev.delta(baseline,record['after'])
                                if delta:raise UnknownEffect(f'{stage}:{ti}: original action math disagrees with snapshot: {delta[:3]}')
                                self._baseline_math[key]=(True,None)
                            except UnknownEffect as exc:
                                self._baseline_math[key]=(False,str(exc))
                        modelled,failure=self._baseline_math[key]
                        input_delta=ev.delta(state,record['before'])
                        input_changed=state!=record['before']
                        if not modelled:
                            # An untouched recorded transition may be preserved,
                            # never transplanted to a changed calculated input.
                            if input_changed:raise UnknownEffect(failure)
                            checks.append(dict(stage=stage,turn=ti,status='recorded_unchanged_input',reason=failure))
                            state=copy.deepcopy(record['after'])
                        else:
                            if input_changed:
                                from .candidates import decode
                                result=decode(output)
                                if c['name']=='snap_to_object_tool' or (isinstance(result,dict) and any(k.startswith(('original_','previous_','current_')) and v is not None for k,v in result.items())):
                                    raise UnknownEffect(f'{stage}:{ti}: original feedback describes a changed input pose')
                                calculate(state,c,output,stage,ev.context(stage,ti,ci),record['after'])
                            else:
                                state=copy.deepcopy(record['after'])
                            # Retained tool response is only reusable when its
                            # resulting state agrees with its recorded endpoint.
                            if ev.delta(state,record['after']):raise UnknownEffect(f'{stage}:{ti}: retained action result differs; original feedback cannot be reused')
                            checks.append(dict(stage=stage,turn=ti,status='arithmetic_verified'))
                            # Bounded original placement noise is independently
                            # checked above; reuse the measured matching endpoint.
                            state=copy.deepcopy(record['after'])
                    elif c['name'] in READ_ONLY or c['name'] in PRODUCERS:
                        if state!=expected_input:
                            raise UnknownEffect(f'{stage}:{ti}: {c["name"]} original feedback/context is stale; check must be removed with its trial or deletion reverted')
                    else:
                        if any(s==stage and index<ti for s,index in drops):raise UnknownEffect(f'{stage}:{ti}: opaque operation may depend on deleted history')
                        if state!=expected_input:raise UnknownEffect(f'{stage}:{ti}: unknown operation on changed input')
                        checks.append(dict(stage=stage,turn=ti,status='preserved_opaque_operation'))
                after[stage,ti]=copy.deepcopy(state)
            if ev.delta(state,ev.exits[stage]):raise UnknownEffect(f'{stage}: compressed final state changed')
        return before,after,checks

    def run(self,episodes,workdir=None):
        from .offline_check import UnknownEffect
        from .offline_output import ObservationBank,materialize
        validate_input(episodes);ev=self.evidence
        report=dict(terminal_observation_schema=3,validation='offline_pose_and_recorded_context',accepted=[],decisions=[],handoffs=ev.handoffs,
                    source_issues=ev.issues,tolerances=dict(position_m=ev.tolerances[0],rotation_deg=ev.tolerances[1],scale=ev.tolerances[2]),
                    training_target='actions_only',cot_status='requires_reannotation',full_physics_verified=False)
        if ev.issues or any(h['status']!='consistent_under_model' for h in ev.handoffs):
            report.update(status='rejected_source',reason='recorded state discontinuity or unmatched stage handoff');return None,report
        bank=ObservationBank(ev,episodes)
        # Reject/rebind source image errors before potentially expensive mesh math.
        materialize(episodes,set(),ev.before,bank)
        baseline_before,_,baseline_checks=self._calculate(episodes,set())
        # Check existing images too; keeping everything is not a waiver for stale
        # source images. Rebind only to a measured matching observation.
        review_episodes,_,_=materialize(episodes,set(),baseline_before,bank)
        accepted=set()
        for stage in STAGES:
            for seg in candidates(stage,review_episodes[stage]):
                allowed=deletable_turns(seg)
                if not allowed:continue
                proposed=self.reviewer(seg)
                if not isinstance(proposed,set) or not proposed<=allowed:
                    report['decisions'].append(dict(segment=seg['segment_id'],status='rejected',reason='invalid protected/out-of-range deletion'));continue
                wanted={(stage,ti) for ti in proposed}-accepted
                plans=[wanted]+[{x} for x in sorted(wanted)] if len(wanted)>1 else [wanted]
                for plan in plans:
                    plan-=accepted
                    if not plan:continue
                    trial_actions=accepted|plan
                    trial=trial_actions|dependent_physics_checks(episodes,trial_actions)
                    try:
                        for st,ti in plan:
                            if not self._baseline_math.get((st,ti,0),(False,None))[0]:
                                raise UnknownEffect(f'{st}:{ti}: proposed deletion lacks a verified original action calculation')
                        before,after,checks=self._calculate(episodes,trial)
                        out,manifest,mapping=materialize(episodes,trial,before,bank)
                        validate_input(out)
                    except (UnknownEffect,ValueError,KeyError) as exc:
                        report['decisions'].append(dict(segment=seg['segment_id'],proposed=sorted(plan),status='retained',reason=str(exc)));continue
                    accepted=trial_actions
                    report['decisions'].append(dict(segment=seg['segment_id'],accepted=sorted(plan),removed_physics_checks=sorted(trial-trial_actions),status='accepted'))
        removed_checks=dependent_physics_checks(episodes,accepted)
        all_drops=accepted|removed_checks
        before,after,checks=self._calculate(episodes,all_drops)
        out,manifest,mapping=materialize(episodes,all_drops,before,bank)
        validate_input(out)
        for stage in STAGES:
            indices=mutation_indices(episodes[stage]);tail=indices[-1] if indices else 0
            if any(s==stage and ti>=tail for s,ti in all_drops):raise ValidationError('protected stage tail deleted')
            # Compare every retained action/result and all non-assistant fields.
            for new_index,old_index in enumerate(mapping[stage]):
                old=copy.deepcopy(episodes[stage]['turns'][old_index]);new=out[stage]['turns'][new_index]
                for key in set(old)|set(new):
                    if old.get('role')=='assistant' and key in ('content','image','images'):continue
                    if old.get(key)!=new.get(key):raise ValidationError(f'{stage}:{old_index}: unapproved field changed: {key}')
        import hashlib
        for binding in manifest:
            for image in binding['images']:
                if hashlib.sha256(Path(image['path']).read_bytes()).hexdigest()!=image['sha256']:
                    raise ValidationError('observation changed during validation')
        ev.verify_unchanged()
        report.update(status='accepted',accepted=sorted(accepted),step_checks=checks,images=manifest,provenance=mapping,
                      source_hashes=ev.hashes,stages=list(STAGES),source_unchanged=True,
                      arithmetic_checked_stages=list(STAGES[1:]),continuity_verified=True,
                      removed_physics_checks=sorted(removed_checks),all_deleted_turns=sorted(all_drops))
        return out,report
