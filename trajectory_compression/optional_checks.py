"""Generate from source SFT when auxiliary verification evidence is absent.

Source observations remain mandatory. Missing geometric evidence is reported as
unverified, never as a successful check. Known state mismatches still reject.
"""
import copy
import hashlib
from pathlib import Path

from .candidates import candidates, deletable_turns, dependent_physics_checks
from .engine import Compressor, validate_input
from .model import STAGES
from .feedback_check import check_feedback_continuity, check_deletion_dependencies
from .offline_check import UnknownEffect
from .offline_output import image_info, materialize, views, ObservationBank


class SourceObservations:
    def __init__(self, episodes, evidence=None):
        self.verified=ObservationBank(evidence,episodes,allow_missing=True) if evidence is not None else None
        self.episodes = episodes
        self.original = {}
        self.drops = set()
        for stage, ep in episodes.items():
            for ti, turn in enumerate(ep['turns']):
                if views(turn):
                    self.original[stage, ti] = [image_info(path) for path in views(turn)]
        from .terminal_images import terminal_plan
        self.terminal = terminal_plan(episodes)

    def bind(self, stage, ti, turn, state):
        if (stage,ti) in self.terminal:
            from .terminal_images import bind_terminal
            return bind_terminal(turn,self.terminal[stage,ti])
        if state is not None and self.verified is not None:
            try:return self.verified.bind(stage,ti,turn,state)
            except UnknownEffect:
                # Missing evidence may be skipped, an observed image-state
                # mismatch may not. Preserve available per-stage verification.
                if (stage,ti) not in self.verified.missing_original:raise
        info = self.original.get((stage, ti))
        if not info:
            return copy.deepcopy(turn), None
        # The next retained action consumes the input of the first removed
        # action. Never leave it looking at the result of a deleted trial.
        donor_ti = ti
        for previous in range(ti-1, -1, -1):
            prev = self.episodes[stage]['turns'][previous]
            if prev.get('role') == 'tool_results':
                continue
            if (stage, previous) not in self.drops:
                break
            donor_ti = previous
        donor = self.original.get((stage, donor_ti))
        if donor is None:
            raise UnknownEffect('removed action has no input image to supply to the next retained action')
        if [(x['view'], x['size']) for x in info] != [(x['view'], x['size']) for x in donor]:
            raise UnknownEffect('deletion would change observation mode or resolution')
        result = copy.deepcopy(turn)
        paths = [x['path'] for x in donor]
        if 'image' in result:
            result['image'] = paths[0]
        if 'images' in result:
            result['images'] = paths
        return result, dict(stage=stage, original_turn=ti, donor_turn=donor_ti,
                            state_match='unverified_without_snapshots', images=copy.deepcopy(donor))

    def verify_unchanged(self):
        for items in self.original.values():
            for item in items:
                if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
                    raise UnknownEffect('source observation changed during processing')


def run_with_optional_checks(episodes, reviewer, missing, evidence=None, position_tol=.03):
    validate_input(episodes)
    report = dict(status='accepted', accepted=[], decisions=[],
                  validation='available_evidence_plus_gpt', full_physics_verified=False,
                  continuity_verified=False, training_target='actions_only',
                  cot_status='requires_reannotation', skipped_checks=list(missing),
                  handoffs=getattr(evidence, 'handoffs', []),
                  source_issues=getattr(evidence, 'issues', []),
                  source_hashes=getattr(evidence, 'hashes', {}))
    if report['source_issues'] or any(h['status'] == 'mismatch' for h in report['handoffs']):
        report.update(status='rejected_source', reason='recorded state discontinuity')
        return None, report
    feedback=check_feedback_continuity(episodes, position_tol)
    report['feedback_continuity']=feedback
    if feedback['issues']:
        report.update(status='rejected_source',reason='numeric tool feedback proves source position discontinuity')
        return None,report
    bank = SourceObservations(episodes,evidence)
    report['terminal_observation_schema'] = 3
    before = {(stage, ti): None for stage, ep in episodes.items()
              for ti, t in enumerate(ep['turns']) if t.get('role') == 'assistant'}
    complete = sorted(getattr(evidence, 'complete_stages', set()))
    checker = Compressor(evidence, reviewer) if complete else None
    if checker:
        known_before,_,_=checker._calculate(episodes, set(), stages=complete)
        before.update(known_before)
    accepted = set()
    out, manifest, mapping = materialize(episodes, set(), before, bank)
    review_episodes=copy.deepcopy(out)
    for stage in STAGES:
        for segment in candidates(stage, review_episodes[stage]):
            allowed = deletable_turns(segment)
            if not allowed:
                continue
            proposal = reviewer(segment)
            if not isinstance(proposal, set) or not proposal <= allowed:
                report['decisions'].append(dict(segment=segment['segment_id'], status='retained', reason='invalid deletion proposal'))
                continue
            wanted = {(stage, ti) for ti in proposal} - accepted
            plans = [wanted] + [{x} for x in sorted(wanted)] if len(wanted) > 1 else [wanted]
            for plan in plans:
                plan -= accepted
                if not plan:
                    continue
                actions = accepted | plan
                drops = actions | dependent_physics_checks(episodes, actions)
                try:
                    dependencies=check_deletion_dependencies(episodes,drops,complete)
                    if dependencies:raise UnknownEffect(str(dependencies))
                    trial_before=dict(before)
                    if checker:
                        known_before,_,_=checker._calculate(episodes, drops, stages=complete)
                        trial_before.update(known_before)
                    bank.drops = drops
                    candidate, images, provenance = materialize(episodes, drops, trial_before, bank)
                    validate_input(candidate)
                except (UnknownEffect, ValueError, KeyError) as exc:
                    report['decisions'].append(dict(segment=segment['segment_id'], status='retained', reason=str(exc), proposed=sorted(plan)))
                    continue
                accepted = actions
                out, manifest, mapping = candidate, images, provenance
                report['decisions'].append(dict(segment=segment['segment_id'], status='accepted', accepted=sorted(plan),
                    verification='available_stage_arithmetic' if stage in complete else 'gpt_judgment_without_recorded_pose_check'))
    checks = dependent_physics_checks(episodes, accepted)
    bank.verify_unchanged()
    if evidence is not None:
        evidence.verify_unchanged()
    report.update(accepted=sorted(accepted), removed_physics_checks=sorted(checks),
                  all_deleted_turns=sorted(accepted | checks), images=manifest, provenance=mapping,
                  arithmetic_checked_stages=complete, stages=list(STAGES))
    return out, report
