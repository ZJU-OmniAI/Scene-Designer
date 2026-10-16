"""Four-stage offline compression. Never invokes a scene replay worker."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from .model import STAGES
from .engine import Compressor, validate_input
from .review import GPTReviewer
from .offline_check import UnknownEffect
from .offline_evidence import RecordedEvidence, MissingEvidence
from .optional_checks import run_with_optional_checks
from .feedback_check import check_feedback_continuity

ROOT=Path(__file__).resolve().parents[1]


def configured_key():
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('Set OPENAI_API_KEY before requesting CoT annotations')
    return key


def infer_room_dir(episodes,room):
    runs=set()
    for ep in episodes.values():
        for turn in ep['turns']:
            for image in turn.get('images') or ([turn['image']] if turn.get('image') else []):
                for part in Path(image).parts:
                    import re
                    m=re.fullmatch(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(scene_\d+)',part)
                    if m:runs.add(ROOT/'scenesmith/outputs'/m[1]/m[2]/m[3]/f'room_{room}')
    existing=[r for r in runs if r.is_dir()]
    if len(runs)<=1 and not existing:raise MissingEvidence('original room directory unavailable')
    if len(existing)!=1 or len(runs)!=1:raise UnknownEffect('conflicting original room references; supply --room-dir')
    return existing[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir',type=Path,required=True)
    parser.add_argument('--room',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--room-dir',type=Path,help='original scenesmith room directory; inferred from source images when unique')
    parser.add_argument('--retain-input-trajectory',action='store_true',help='Explicitly accept the supplied SFT as baseline, without comparison to omitted original-run actions or historical snapshots; retain transcript and deletion checks')
    parser.add_argument('--model',default='gpt-5.5')
    parser.add_argument('--api-url',default=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/') + '/chat/completions')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--check-only',action='store_true',help='no GPT, no deletions; validate/rebind existing four-stage data')
    group.add_argument('--review-file',type=Path,help='JSON {segment_id:[original turn indices]} for reproducible checks')
    parser.add_argument('--review-cache',type=Path,default=ROOT/'outputs/sft_compression_reviews.sqlite3')
    parser.add_argument('--api-attempts',type=int,default=3)
    parser.add_argument('--position-tol',type=float,default=.03)
    parser.add_argument('--angle-tol',type=float,default=3.)
    args=parser.parse_args(argv)
    import math
    if any(not math.isfinite(t) or t<0 for t in (args.position_tol,args.angle_tol)):parser.error('invalid tolerance')
    if args.api_attempts<1:parser.error('api-attempts must be positive')
    source=args.scene_dir.resolve();dest=args.output_dir.resolve()
    if dest==source or source in dest.parents or dest in source.parents:parser.error('source and output must be separate non-nested paths')
    if dest.exists():parser.error('output already exists; choose a fresh directory')
    names={s:'floor_plan.json' if s=='floor_plan' else f'{s}_{args.room}.json' for s in STAGES}
    raw={s:(source/name).read_bytes() for s,name in names.items()}
    episodes={s:json.loads(data) for s,data in raw.items()};reviewer=None;out=None;evidence=None;feedback=None
    try:
        validate_input(episodes)
        feedback=check_feedback_continuity(episodes,args.position_tol)
        if feedback['issues']:raise ValueError('numeric tool feedback proves source position discontinuity')
        if args.check_only:reviewer=lambda segment:set()
        elif args.review_file:
            proposals=json.loads(args.review_file.read_text())
            def reviewer(segment):
                values=proposals.get(segment['segment_id'],[])
                if not isinstance(values,list) or any(type(x) is not int for x in values):raise ValueError('invalid review-file indices')
                return set(values)
        else:reviewer=GPTReviewer(args.model,args.api_url,key=configured_key(),cache_path=args.review_cache,
                                  max_attempts=args.api_attempts,label=f'{source.name}/{args.room}')
        missing=[]
        if args.retain_input_trajectory:
            missing.append(dict(status='unverified',
                reason='user-authorized retained SFT baseline; original-run omissions and overlapping historical state differences are not rejection criteria',
                scope='historical snapshot and original-run endpoint/handoff comparisons'))
        else:
            try:
                room_dir=args.room_dir or infer_room_dir(episodes,args.room)
                evidence=RecordedEvidence(room_dir,args.room,episodes,args.position_tol,args.angle_tol,allow_missing=True)
                missing.extend(evidence.skipped_checks)
                missing.extend(h for h in evidence.handoffs if h['status'] in ('unverified','unverifiable'))
            except MissingEvidence as exc:
                missing.append(dict(status='unverified',reason=str(exc),scope='recorded pose and stage-handoff checks'))
        if missing:
            out,report=run_with_optional_checks(episodes,reviewer,missing,evidence,args.position_tol)
        else:
            try:
                out,report=Compressor(evidence,reviewer).run(episodes)
            except MissingEvidence as exc:
                out,report=run_with_optional_checks(episodes,reviewer,[dict(status='unverified',reason=str(exc))],evidence,args.position_tol)
    except (UnknownEffect,ValueError,KeyError,FileNotFoundError) as exc:
        report=dict(status='rejected_source',reason=str(exc),accepted=[],validation='offline',full_physics_verified=False)
        if evidence is not None:
            report.update(handoffs=evidence.handoffs,source_issues=evidence.issues,source_hashes=evidence.hashes)
    report['source_policy']='retained_input_trajectory' if args.retain_input_trajectory else 'compare_original_run'
    if feedback is not None:report['feedback_continuity']=feedback
    report['reviews']=getattr(reviewer,'reviews',[])
    report['review_failures']=sum(r.get('status')!='reviewed' for r in report['reviews'])
    if report.get('status')=='accepted' and report['review_failures']:
        report['status']='accepted_with_review_failures'
    report['review_mode']='check_only' if args.check_only else 'saved_proposals' if args.review_file else 'live_gpt'
    report['source_sha256']={s:hashlib.sha256(data).hexdigest() for s,data in raw.items()}
    if any((source/names[s]).read_bytes()!=raw[s] for s in STAGES):raise RuntimeError('source SFT changed during run')
    report['verification_schema']=2
    report['source_unchanged']=True
    report['generation_config']=dict(model=args.model,api_url=args.api_url,position_tol=args.position_tol,angle_tol=args.angle_tol)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.offline-compression-',dir=dest.parent) as tmp:
        work=Path(tmp)
        if out is not None:
            for stage,ep in out.items():
                data=json.dumps(ep,ensure_ascii=False,indent=2)+'\n';(work/names[stage]).write_text(data)
                if json.loads((work/names[stage]).read_text())!=ep:raise RuntimeError('output serialization mismatch')
        (work/'compression_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        if dest.exists():raise RuntimeError('output appeared during validation')
        work.rename(dest)
    print(json.dumps(dict(status=report['status'],accepted=len(report['accepted']),output=str(dest),reason=report.get('reason')),ensure_ascii=False),flush=True)
    return 0 if out is not None and not report['review_failures'] else 1

if __name__=='__main__':raise SystemExit(main())
