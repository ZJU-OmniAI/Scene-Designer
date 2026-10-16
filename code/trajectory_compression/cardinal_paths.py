"""Attach existing same-step cardinal renders to published training observations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

from .offline_output import image_info, views
from .vision import ROOT

ORDER = ('topdown', 'south', 'east', 'west', 'north')


def sibling_paths(top):
    top = Path(top)
    if not top.name.endswith('.topdown.png'):
        return None
    stem = top.name.removesuffix('.topdown.png')
    if not top.with_name(stem + '.blend').is_file():
        return None
    return [top.with_name(stem + '.' + view + '.png') for view in ORDER]


def validate_expansion(record, actual):
    """Return the old observation only after checking an additive same-step edit."""
    old = record['previous_images']
    assert old and old[0]['view'] == 'topdown', 'cardinal expansion needs original top view'
    paths = sibling_paths(old[0]['path'])
    assert paths is not None, 'cardinal expansion has no corresponding blend'
    assert [x['path'] for x in actual] == [str(x) for x in paths], 'cardinal images belong to another step or wrong order'
    assert [x['view'] for x in actual] == list(ORDER), 'incorrect cardinal view identities'
    assert actual == record['images'], 'cardinal image pixels or dimensions changed'
    by_view = {x['view']: x for x in actual}
    assert len({x['view'] for x in old}) == len(old), 'duplicate baseline view'
    for item in old:
        assert by_view.get(item['view']) == item, 'original view changed during cardinal expansion'
    return old


def augment(episodes, report, cache):
    episodes = copy.deepcopy(episodes);report = copy.deepcopy(report)
    changes=[];skips=[]
    def info(path):
        if path not in cache:cache[path]=image_info(path)
        return cache[path]
    for room,rr in report['rooms'].items():
        existing={(x['stage'],x['original_turn']):x for x in rr.get('cardinal_view_expansions',[])}
        bindings={(x['stage'],x['original_turn']):x for x in rr['images']}
        for stage in ('furniture','wall','ceiling'):
            filename=f'{stage}_{room}.json'
            for ti,turn in enumerate(episodes[filename]['turns']):
                raw=views(turn)
                if not raw:continue  # Never recreate uncaptured terminal observations.
                original=[info(p) for p in raw]
                paths=sibling_paths(original[0]['path'])
                if paths is None:
                    skips.append(dict(file=filename,turn=ti,reason='no matching per-step blend'));continue
                if not all(p.is_file() for p in paths):
                    raise FileNotFoundError(f'missing cardinal render for {filename}:{ti}')
                expanded=[info(str(p)) for p in paths]
                oi=rr['provenance'][stage][ti];key=(stage,oi)
                record=existing.get(key) or dict(stage=stage,original_turn=oi,previous_images=original,images=expanded)
                validate_expansion(record,expanded)
                if original==expanded:continue
                assert key in bindings, 'observation missing from original image manifest'
                if 'image' in turn:turn['image']=expanded[0]['path']
                turn['images']=[x['path'] for x in expanded]
                bindings[key]['images']=copy.deepcopy(expanded)
                if key not in existing:
                    rr.setdefault('cardinal_view_expansions',[]).append(record);existing[key]=record
                changes.append(dict(file=filename,turn=ti,original_turn=oi,previous_view_count=len(original),new_view_count=5))
    return episodes,report,changes,skips


def main():
    from concurrent.futures import ThreadPoolExecutor
    import fcntl
    from .quality import audit_scene
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-root',type=Path,default=ROOT/'train_datasets')
    ap.add_argument('--source-root',type=Path,default=ROOT/'datasets/sft')
    ap.add_argument('--report',type=Path,default=ROOT/'outputs/train_cardinal_paths_completion.json')
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args();cache={};rows=[]
    scenes=sorted(p for p in args.input_root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    def read_scene(scene):
        raw={p.name:p.read_bytes() for p in scene.glob('*.json')}
        data={n:json.loads(b) for n,b in raw.items()}
        report=data.pop('compression_report.json')
        return scene,raw,data,report
    with (args.input_root/'.sft-compression.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=16) as pool:loaded=list(pool.map(read_scene,scenes))
        # Validate every image before writing any training file.
        allpaths=set()
        for _,_,eps,_ in loaded:
            for ep in eps.values():
                for turn in ep.get('turns',[]):
                    for path in views(turn):allpaths.add(path)
        with ThreadPoolExecutor(max_workers=12) as pool:
            for path,item in zip(sorted(allpaths),pool.map(image_info,sorted(allpaths))):cache[path]=item
        additions=set()
        for item in list(cache.values()):
            if item['view']=='topdown':
                paths=sibling_paths(item['path'])
                if paths is not None:additions.update(map(str,paths))
        additions=sorted(additions-cache.keys())
        with ThreadPoolExecutor(max_workers=12) as pool:
            for path,item in zip(additions,pool.map(image_info,additions)):cache[path]=item
        plans=[]
        for scene,raw,eps,report in loaded:
            new,rep,changes,skips=augment(eps,report,cache)
            # Only image fields may change; tool calls, results, order and text stay identical.
            for name,old in eps.items():
                stripped=copy.deepcopy(new[name])
                for a,b in zip(old['turns'],stripped['turns']):
                    for k in ('image','images'):
                        b.pop(k,None)
                        if k in a:b[k]=a[k]
                assert stripped==old, f'non-image field changed: {scene.name}/{name}'
            plans.append((scene,raw,new,rep,changes,skips))
        for scene,raw,new,rep,changes,skips in plans:
            changed_files=sorted({x['file'] for x in changes})
            if changes:
                with tempfile.TemporaryDirectory(prefix='.cardinal-paths-',dir=args.input_root) as td:
                    tmp=Path(td)
                    for name,ep in new.items():(tmp/name).write_text(json.dumps(ep,ensure_ascii=False,indent=2)+'\n')
                    (tmp/'compression_report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n')
                    audit=audit_scene(tmp,args.source_root,cache,source_name=scene.name)
                    assert not audit['findings'],audit['findings']
                    rep['quality_audit']={k:v for k,v in audit.items() if k!='rooms'}
                    (tmp/'compression_report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n')
                    if args.apply:
                        for name,data in raw.items():assert (scene/name).read_bytes()==data,'training data changed concurrently'
                        # Same-filesystem atomic replacement; manifests follow episode writes.
                        for name in changed_files+['compression_report.json']:(tmp/name).replace(scene/name)
            rows.append(dict(scene=scene.name,changed_files=changed_files,changed_turns=len(changes),changes=changes,skips=skips,status='applied' if args.apply and changes else 'unchanged' if not changes else 'planned'))
            if len(rows)%50==0:print(f'checked {len(rows)}/{len(plans)}',flush=True)
        result=dict(applied=args.apply,scenes=len(rows),changed_scenes=sum(bool(x['changed_turns']) for x in rows),changed_stage_files=sum(len(x['changed_files']) for x in rows),changed_turns=sum(x['changed_turns'] for x in rows),unique_images_validated=len({x['path'] for x in cache.values()}),scene_reports=rows)
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='scene_reports'}),flush=True)

if __name__=='__main__':main()
