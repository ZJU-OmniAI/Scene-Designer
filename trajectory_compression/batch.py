"""Batch compression with independent publication audit and verified resume."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
from pathlib import Path
import tempfile

from . import cli
from .quality import audit_scene


def write_json(path, value):
    """Readers see either the old report or the complete new report."""
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.report-',delete=False) as stream:
        temporary=Path(stream.name)
        try:
            json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def process_scene(scene, args):
    row={'scene':scene.name,'status':'rejected','rooms':{}}
    dest=args.output_root/scene.name
    stale_reason=None
    try:
        if dest.exists():
            if not args.resume:
                raise ValueError('output already exists; use --resume to revalidate and reuse it')
            try:
                quality=audit_scene(dest,args.input_root,{})
                if quality['findings']:
                    raise ValueError(f'existing output fails current audit: {quality["findings"]}')
                prior=json.loads((dest/'compression_report.json').read_text())
                for r in prior['rooms'].values():
                    if r.get('arithmetic_checked_stages') and r.get('verification_schema',1)<2:
                        raise ValueError('partial-state image validation upgraded; rebuild required')
                    policy='retained_input_trajectory' if args.retain_input_trajectory else 'compare_original_run'
                    if r.get('source_policy','compare_original_run')!=policy:
                        raise ValueError('existing output used a different source acceptance policy')
                    expected='check_only' if args.check_only else 'live_gpt'
                    if r.get('review_mode')!=expected:
                        raise ValueError('existing output used a different review mode')
                    config=r.get('generation_config')
                    wanted=dict(model=args.model,api_url=args.api_url,position_tol=args.position_tol,angle_tol=args.angle_tol)
                    if config is not None and config!=wanted:
                        raise ValueError('existing output used different model/tolerances')
                    if config is None and (args.model!='gpt-5.5' or args.api_url!='https://api.openai.com/v1/chat/completions' or args.position_tol!=.03 or args.angle_tol!=3.):
                        raise ValueError('legacy output has no settings provenance for these overrides')
                row.update(status='cached',output=str(dest),quality_audit=quality)
                return row
            except (ValueError,AssertionError,KeyError,OSError) as exc:
                if not args.refresh_stale:raise
                # Keep the old output until its replacement passes all checks.
                stale_reason=str(exc)
        room_sets=[{x.stem[len(stage)+1:] for x in scene.glob(f'{stage}_*.json')}
                   for stage in ('furniture','wall','ceiling')]
        if not room_sets[0] or any(s!=room_sets[0] for s in room_sets) or not (scene/'floor_plan.json').is_file():
            raise ValueError('incomplete four-stage room files')
        with tempfile.TemporaryDirectory(prefix='.sft-batch-',dir=args.output_root) as tmp:
            work=Path(tmp);publish=work/'publish';publish.mkdir()
            for room in sorted(room_sets[0]):
                room_out=work/f'room_{room}'
                options=['--scene-dir',str(scene),'--room',room,'--output-dir',str(room_out),
                         '--model',args.model,'--api-url',args.api_url,
                         '--position-tol',str(args.position_tol),'--angle-tol',str(args.angle_tol),
                         '--review-cache',str(args.review_cache),'--api-attempts',str(args.api_attempts)]
                if args.check_only:options.append('--check-only')
                if args.retain_input_trajectory:options.append('--retain-input-trajectory')
                code=cli.main(options)
                report=json.loads((room_out/'compression_report.json').read_text())
                row['rooms'][room]=report
                if code or report.get('status')!='accepted':
                    raise ValueError(f'{room}: {report.get("reason") or report.get("status")}')
                for name in ('floor_plan.json',f'furniture_{room}.json',f'wall_{room}.json',f'ceiling_{room}.json'):
                    target=publish/name;data=(room_out/name).read_bytes()
                    if target.exists() and target.read_bytes()!=data:
                        raise ValueError(f'rooms disagree on shared {name}')
                    target.write_bytes(data)
            row['status']='accepted'
            write_json(publish/'compression_report.json',row)
            # Independent file-level acceptance, including all original image
            # bytes and source feedback, before anything reaches train_datasets.
            quality=audit_scene(publish,args.input_root,{},source_name=scene.name)
            if quality['findings']:
                raise ValueError(f'publication audit failed: {quality["findings"]}')
            row['quality_audit']={k:v for k,v in quality.items() if k!='rooms'}
            write_json(publish/'compression_report.json',row)
            if dest.exists():
                if stale_reason is None:raise ValueError('output appeared during validation')
                archive_root=args.report_root/'superseded';archive_root.mkdir(parents=True,exist_ok=True)
                archive=Path(tempfile.mkdtemp(prefix=scene.name+'-',dir=archive_root))
                write_json(archive/'reason.json',dict(scene=scene.name,reason=stale_reason))
                dest.rename(archive/'previous_output')
                try:publish.rename(dest)
                except BaseException:
                    (archive/'previous_output').rename(dest)
                    raise
                row['superseded_output']=str(archive/'previous_output')
            else:publish.rename(dest)
    except Exception as exc:
        row.update(status='rejected',reason=str(exc))
    return row


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=cli.ROOT/'datasets/sft')
    p.add_argument('--output-root',type=Path,default=cli.ROOT/'train_datasets')
    p.add_argument('--report-root',type=Path,default=cli.ROOT/'outputs/sft_compression_reports')
    p.add_argument('--review-cache',type=Path,default=cli.ROOT/'outputs/sft_compression_reviews.sqlite3')
    p.add_argument('--scene',action='append',help='只处理指定场景，可重复')
    p.add_argument('--max-scenes',type=int,default=0,help='0 表示全部')
    p.add_argument('--model',default='gpt-5.5')
    p.add_argument('--api-url',default='https://api.openai.com/v1/chat/completions')
    p.add_argument('--api-attempts',type=int,default=3)
    p.add_argument('--workers',type=int,default=1,help='独立场景进程数量')
    p.add_argument('--resume',action='store_true',help='重新验收已有结果，通过后跳过，不覆盖')
    p.add_argument('--refresh-stale',action='store_true',help='将未通过复核的旧输出归档，然后从源数据重新处理')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--retain-input-trajectory',action='store_true',help='Use supplied SFT baseline instead of historical original-run state comparisons')
    p.add_argument('--position-tol',type=float,default=.03)
    p.add_argument('--angle-tol',type=float,default=3.)
    args=p.parse_args(argv)
    if args.refresh_stale and not args.resume:p.error('--refresh-stale requires --resume')
    roots=[x.resolve() for x in (args.input_root,args.output_root,args.report_root)]
    args.input_root,args.output_root,args.report_root=roots
    for i,a in enumerate(roots):
        for b in roots[i+1:]:
            if a==b or a in b.parents or b in a.parents:
                p.error('input/output/report roots must be separate non-nested directories')
    if not args.input_root.is_dir() or args.max_scenes<0 or args.workers<1 or args.api_attempts<1:
        p.error('invalid input directory or numeric options')
    import math
    if any(not math.isfinite(x) or x<0 for x in (args.position_tol,args.angle_tol)):
        p.error('invalid tolerance')
    scenes=sorted(x for x in args.input_root.iterdir() if x.is_dir() and any(x.glob('*.json')))
    if args.scene:
        missing=set(args.scene)-{x.name for x in scenes}
        if missing:p.error(f'unknown scenes: {sorted(missing)}')
        scenes=[x for x in scenes if x.name in args.scene]
    if args.max_scenes:scenes=scenes[:args.max_scenes]
    if not scenes:p.error('no matching scenes')
    args.output_root.mkdir(parents=True,exist_ok=True);args.report_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/'.sft-compression.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('another batch is already using this output directory')
        report_dir=Path(tempfile.mkdtemp(prefix='run-',dir=args.report_root));rows={}
        print(f'reports={report_dir} scenes={len(scenes)} workers={args.workers}',flush=True)
        def completed(row):
            rows[row['scene']]=row
            write_json(report_dir/f'{row["scene"]}.json',row)
            summary=[dict(scene=r['scene'],status=r['status'],reason=r.get('reason')) for _,r in sorted(rows.items())]
            write_json(report_dir/'summary.json',summary)
            print(f'[{len(rows)}/{len(scenes)}] {row["status"]}: {row["scene"]} {row.get("reason", "")}',flush=True)
        if args.workers==1:
            for scene in scenes:completed(process_scene(scene,args))
        else:
            # Separate processes keep native geometry/global caches isolated.
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs={pool.submit(process_scene,scene,args):scene for scene in scenes}
                for job in as_completed(jobs):
                    try:row=job.result()
                    except Exception as exc:row=dict(scene=jobs[job].name,status='rejected',reason=f'worker failed: {type(exc).__name__}')
                    completed(row)
        accepted=sum(r['status']=='accepted' for r in rows.values())
        cached=sum(r['status']=='cached' for r in rows.values())
        rejected=len(rows)-accepted-cached
        print(f'accepted={accepted} cached={cached} rejected={rejected} reports={report_dir}',flush=True)
        return int(rejected>0)


if __name__=='__main__':raise SystemExit(main())
