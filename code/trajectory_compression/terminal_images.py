"""Bind terminal observations to the next stage's recorded entry.

No independent final snapshot is required. Native donor pixels are never resized.
Ceiling's cached pre-action tail image is removed, not presented as a new capture.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
from .offline_output import image_info, views
from .vision import ROOT

MUTATIONS={'add_furniture_to_scene_tool','move_furniture_tool','remove_furniture_tool',
 'rescale_furniture_tool','snap_to_object_tool','place_wall_object','move_wall_object',
 'remove_wall_object','rescale_wall_object','place_ceiling_object','move_ceiling_object',
 'remove_ceiling_object','rescale_ceiling_object'}
PATTERN=re.compile(r'^(furniture|wall|ceiling)_(\d+)__.+\.(topdown|south|east|west|north)\.png$')


NEXT_STAGE={'furniture':'wall','wall':'ceiling'}


def terminal_plan(episodes):
    """Return a binding for each existing post-mutation assistant observation.

    The stage entry is taken from the same four-stage room trajectory, never by
    globbing a similarly named scene. Missing auxiliary raw traces do not matter.
    """
    plan={}
    for stage in ('furniture','wall','ceiling'):
        ep=episodes.get(stage)
        if ep is None:continue
        turns=ep['turns']
        mutations=[i for i,t in enumerate(turns) if any(c['name'] in MUTATIONS for c in t.get('tool_calls') or [])]
        if not mutations:continue
        last=mutations[-1]
        tail=[i for i in range(last+1,len(turns)) if turns[i].get('role')=='assistant' and views(turns[i])]
        if not tail:continue
        if stage=='ceiling':
            before=views(turns[last])
            hashes=[image_info(p)['sha256'] for p in before]
            for ti in tail:
                if hashes and [image_info(p)['sha256'] for p in views(turns[ti])]==hashes:
                    plan[stage,ti]=dict(stage=stage,original_turn=ti,images=[],
                        state_match='no_post_action_capture',donor_stage=None,donor_turn=None)
            continue
        next_stage=NEXT_STAGE[stage]
        if next_stage not in episodes:continue
        donor=None
        for di,t in enumerate(episodes[next_stage]['turns']):
            if t.get('role')=='assistant' and views(t):
                donor=(di,[image_info(p) for p in views(t)]);break
            if any(c['name'] in MUTATIONS for c in t.get('tool_calls') or []):
                raise ValueError(f'{next_stage}: no entry image before its first mutation')
        if donor is None:raise ValueError(f'{next_stage}: missing recorded entry image')
        di,images=donor
        # Images for paired stages must identify the same room/collection when
        # canonical per-step layout is used. Prompt paraphrases may have other IDs.
        old=views(turns[last])
        if old:
            a=Path(image_info(old[0])['path']);b=Path(images[0]['path'])
            if PATTERN.fullmatch(a.name) and PATTERN.fullmatch(b.name) and (a.parent.parent.name,a.parent.name)!=(b.parent.parent.name,b.parent.name):
                raise ValueError('stage entry image belongs to a different scene or room')
        for ti in tail:
            plan[stage,ti]=dict(stage=stage,original_turn=ti,images=copy.deepcopy(images),
                state_match='next_stage_entry',donor_stage=next_stage,donor_turn=di)
    return plan


def bind_terminal(turn,binding):
    result=copy.deepcopy(turn);paths=[x['path'] for x in binding['images']]
    if not paths:
        result.pop('image',None);result.pop('images',None)
    else:
        if 'image' in result:result['image']=paths[0]
        if 'images' in result:result['images']=paths
    return result,copy.deepcopy(binding)


def inspect_episode(episode,raw_root=None,episodes=None):
    """Compatibility inspection; standalone stages cannot prove a handoff."""
    context=episodes or {episode['stage']:episode}
    fixes=[]
    for (stage,ti),binding in terminal_plan(context).items():
        if stage!=episode['stage']:continue
        turn=episode['turns'][ti];new,_=bind_terminal(turn,binding)
        if new!=turn:fixes.append(dict(binding,affected_turns=[ti]))
    return fixes,[]


def repair_episode(episode,raw_root=None,episodes=None):
    fixes,issues=inspect_episode(episode,raw_root,episodes)
    result=copy.deepcopy(episode)
    for fix in fixes:
        for ti in fix['affected_turns']:result['turns'][ti],_=bind_terminal(result['turns'][ti],fix)
    return result,fixes,issues


def main(argv=None):
    import argparse
    import fcntl
    import shutil
    import tempfile
    from .batch import write_json
    from .quality import audit_scene
    from .model import STAGES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=ROOT/'train_datasets')
    p.add_argument('--source-root',type=Path,default=ROOT/'datasets/sft')
    p.add_argument('--report-root',type=Path,default=ROOT/'outputs/sft_stage_boundary_repairs')
    p.add_argument('--apply',action='store_true')
    p.add_argument('--refresh-rendered-images',action='store_true',help='Rebind reviewed canonical cardinal renders of unchanged size; never accept arbitrary changes')
    args=p.parse_args(argv);args.report_root.mkdir(parents=True,exist_ok=True)
    rows=[];cache={}
    with (args.input_root/'.sft-compression.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for scene in sorted(x for x in args.input_root.iterdir() if x.is_dir() and not x.name.startswith('.')):
            row=dict(scene=scene.name,status='failed',boundary_changes=[],image_updates=[])
            try:
                original_report=json.loads((scene/'compression_report.json').read_text());report=copy.deepcopy(original_report);changes={}
                for room,rr in report['rooms'].items():
                    names={s:'floor_plan.json' if s=='floor_plan' else f'{s}_{room}.json' for s in STAGES}
                    source={s:json.loads((args.source_root/scene.name/names[s]).read_text()) for s in STAGES}
                    output={s:json.loads((scene/names[s]).read_text()) for s in STAGES}
                    plan=terminal_plan(source)
                    for (stage,oi),binding in plan.items():
                        mapping=rr['provenance'][stage]
                        if oi not in mapping:continue
                        ti=mapping.index(oi);old=output[stage]['turns'][ti];new,b=bind_terminal(old,binding)
                        if new!=old:
                            output[stage]['turns'][ti]=new;changes[names[stage]]=output[stage]
                            row['boundary_changes'].append(dict(stage=stage,turn=ti,original_turn=oi,old_images=views(old),new_images=views(new),relation=b['state_match']))
                        bindings=[x for x in rr['images'] if x['stage']==stage and x['original_turn']==oi]
                        if len(bindings)>1:raise ValueError('duplicate image binding')
                        if bindings:bindings[0].clear();bindings[0].update(b)
                        else:rr['images'].append(b)
                    for b in rr['images']:
                        for index,old in enumerate(b['images']):
                            path=old['path']
                            if path not in cache:cache[path]=image_info(path)
                            new=cache[path]
                            if old['sha256']==new['sha256']:continue
                            if not args.refresh_rendered_images:raise ValueError(f'image changed; explicit reviewed refresh required: {path}')
                            match=PATTERN.fullmatch(Path(path).name)
                            if not match or match[3]=='topdown' or old['view']!=new['view'] or old['size']!=new['size']:
                                raise ValueError(f'not a same-size canonical cardinal rerender: {path}')
                            blend=Path(path).with_name(Path(path).name.rsplit('.',2)[0]+'.blend')
                            if not blend.is_file():raise ValueError(f'rerender has no corresponding blend: {path}')
                            change=dict(path=path,previous_sha256=old['sha256'],sha256=new['sha256'],view=new['view'],size=new['size'],blend=str(blend),blend_sha256=hashlib.sha256(blend.read_bytes()).hexdigest(),validation='decoded PNG; unchanged native view/size; canonical source blend exists; reviewed rerender, not a new full-physics claim')
                            if not any(x['path']==path for x in row['image_updates']):row['image_updates'].append(change)
                            b['images'][index]=copy.deepcopy(new)
                    rr.pop('terminal_image_unresolved',None)
                    rr.pop('terminal_image_repairs',None)
                    rr['terminal_observation_schema']=3
                    rr['terminal_observation_policy']='next_stage_entry_native_pixels; ceiling_cached_post_action_image_omitted'
                report['terminal_observation_status']='stage_boundaries_bound'
                report['terminal_observation_schema']=3
                report.setdefault('reviewed_image_updates',[]).extend(row['image_updates'])
                if report!=original_report:changes['compression_report.json']=report
                # Validate the exact staged output BEFORE replacing existing data.
                with tempfile.TemporaryDirectory(prefix='.boundary-repair-',dir=args.input_root) as tmp:
                    staging=Path(tmp)/'scene';shutil.copytree(scene,staging)
                    for name,value in changes.items():write_json(staging/name,value)
                    quality=audit_scene(staging,args.source_root,cache,source_name=scene.name)
                    if quality['findings']:raise ValueError(str(quality['findings']))
                    # Recheck all bytes at commit; the separate cardinal renderer
                    # may still be filling other scenes in the background.
                    for rr in report['rooms'].values():
                        for b in rr['images']:
                            for im in b['images']:
                                if hashlib.sha256(Path(im['path']).read_bytes()).hexdigest()!=im['sha256']:raise ValueError('image changed during repair')
                    if args.apply and changes:
                        archive=args.report_root/'previous'/scene.name;archive.parent.mkdir(parents=True,exist_ok=True)
                        if archive.exists():archive=Path(tempfile.mkdtemp(prefix=scene.name+'-',dir=archive.parent))/'previous_output'
                        scene.rename(archive)
                        try:staging.rename(scene)
                        except BaseException:archive.rename(scene);raise
                    row.update(status='applied' if args.apply and changes else 'validated',quality=quality['status'])
            except Exception as exc:row['reason']=str(exc)
            rows.append(row)
            write_json(args.report_root/'report.json',dict(scenes=rows,scene_count=len(rows),failed=sum(x['status']=='failed' for x in rows),changed_turns=sum(len(x['boundary_changes']) for x in rows)))
            print(f'{len(rows)} {scene.name} {row["status"]} {row.get("reason","")}',flush=True)
    return int(any(x['status']=='failed' for x in rows))


if __name__=='__main__':raise SystemExit(main())
