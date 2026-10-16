"""Read-only audit of all atomically published four-stage SFT scenes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .candidates import candidates, deletable_turns, mutation_indices, dependent_physics_checks, check_status
from .engine import validate_input
from .feedback_check import check_feedback_continuity, check_deletion_dependencies
from .model import STAGES
from .offline_output import image_info, views


def audit_scene(scene, source_root, cache, source_name=None):
    name=source_name or scene.name
    row=dict(scene=name,findings=[],rooms={},image_references=0,deleted_actions=0,deleted_checks=0)
    report=json.loads((scene/'compression_report.json').read_text())
    assert report['status']=='accepted','scene not accepted'
    expected_files={'compression_report.json','floor_plan.json'}
    for room,r in report['rooms'].items():
        from .terminal_images import terminal_plan
        source_episodes={stage:json.loads((source_root/name/('floor_plan.json' if stage=='floor_plan' else f'{stage}_{room}.json')).read_text()) for stage in STAGES}
        boundary=terminal_plan(source_episodes)
        expansions={(x['stage'],x['original_turn']):x for x in r.get('cardinal_view_expansions',[])}
        assert len(expansions)==len(r.get('cardinal_view_expansions',[])), 'duplicate cardinal expansion record'
        original={};output={};drops={tuple(x) for x in r['all_deleted_turns']};actions={tuple(x) for x in r['accepted']}
        assert r['status']=='accepted' and r['review_failures']==0,'incomplete GPT review'
        assert not r.get('source_issues') and not any(h['status']=='mismatch' for h in r.get('handoffs',[])),'known mismatch published'
        for path,digest in r.get('source_hashes',{}).items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'supporting evidence changed since processing'
        for stage in STAGES:
            filename='floor_plan.json' if stage=='floor_plan' else f'{stage}_{room}.json';expected_files.add(filename)
            raw=(source_root/name/filename).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==r['source_sha256'][stage],'source changed since processing'
            original[stage]=json.loads(raw);output[stage]=json.loads((scene/filename).read_text())
            old=original[stage]['turns'];new=output[stage]['turns'];mapping=r['provenance'][stage]
            wanted=[i for i,t in enumerate(old) if (stage,i) not in drops and not (t['role']=='tool_results' and (stage,i-1) in drops)]
            assert mapping==wanted and len(new)==len(mapping),'incorrect deletion/provenance mapping'
            for key in set(original[stage])|set(output[stage]):
                if key not in ('source','turns'):assert original[stage].get(key)==output[stage].get(key),'episode metadata changed'
            mutations=mutation_indices(original[stage]);assert not mutations or mutations[-1] in mapping,'stage tail deleted'
            for t,i in zip(new,mapping):
                if t['role']=='assistant':assert t.get('content') is None,'old reasoning remains'
                for key in set(old[i])|set(t):
                    if old[i]['role']=='assistant' and key in ('content','image','images'):continue
                    assert old[i].get(key)==t.get(key),f'changed retained field {stage}:{i}:{key}'
                pair=[]
                for turn in (old[i],t):
                    infos=[]
                    for path in views(turn):
                        if path not in cache:cache[path]=image_info(path)
                        infos.append(cache[path])
                    pair.append(infos)
                actual_count=len(pair[1])
                if (stage,i) in expansions:
                    from .cardinal_paths import validate_expansion
                    pair[1]=validate_expansion(expansions[stage,i],pair[1])
                if (stage,i) in boundary:
                    expected=boundary[stage,i]['images']
                    assert [x['path'] for x in pair[1]]==[x['path'] for x in expected],'incorrect stage-boundary observation'
                    assert [(x['view'],x['size']) for x in pair[1]]==[(x['view'],x['size']) for x in expected],'boundary donor pixels changed'
                else:
                    assert [(x['view'],x['size']) for x in pair[0]]==[(x['view'],x['size']) for x in pair[1]],'view order or resolution changed'
                row['image_references']+=actual_count
        validate_input(original);validate_input(output)
        assert actions|dependent_physics_checks(original,actions)==drops,'wrong dependent checks removed'
        for stage,ti in actions:
            assert any(ti in deletable_turns(seg) for seg in candidates(stage,original[stage])),'protected/out-of-range action removed'
        for binding in r['images']:
            for item in binding['images']:
                if item['path'] not in cache:cache[item['path']]=image_info(item['path'])
                assert cache[item['path']]['sha256']==item['sha256'],'bound image changed'
            actual=output[binding['stage']]['turns'][r['provenance'][binding['stage']].index(binding['original_turn'])]
            assert [str(Path(x).resolve()) for x in views(actual)]==[x['path'] for x in binding['images']],'image manifest disagrees with output'
        feedback=check_feedback_continuity(original,r.get('generation_config',{}).get('position_tol',.03))
        dependencies=check_deletion_dependencies(original,drops,r.get('arithmetic_checked_stages',[]))
        for issue in feedback['issues']:row['findings'].append(dict(kind='source_numeric_discontinuity',room=room,**issue))
        for issue in dependencies:row['findings'].append(dict(kind='unsafe_deletion_dependency',room=room,**issue))
        last_checks={}
        for stage in STAGES[1:]:
            states=[]
            for i,t in enumerate(output[stage]['turns']):
                for ci,c in enumerate(t.get('tool_calls') or []):
                    if c['name']=='check_physics':states.append(check_status(output[stage]['turns'][i+1]['results'][ci]['output']))
            last_checks[stage]=states[-1] if states else 'absent'
        row['rooms'][room]=dict(feedback_comparisons=len(feedback['comparisons']),
            final_physics_feedback=last_checks,skipped_checks=r.get('skipped_checks',[]),
            reviews=len(r['reviews']),source_continuity_fully_verified=r.get('continuity_verified',False),
            compressed_pose_overwrite_check='passed' if not dependencies else 'failed')
        row['deleted_actions']+=len(actions);row['deleted_checks']+=len(r['removed_physics_checks'])
    assert {p.name for p in scene.glob('*.json')}==expected_files,'unexpected or missing SFT JSON'
    row['status']='issues_found' if row['findings'] else 'passed_available_checks'
    return row


def audit(root, source):
    rows=[];cache={}
    scenes=sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    for scene in scenes:
        try:row=audit_scene(scene,source,cache)
        except Exception as exc:row=dict(scene=scene.name,status='structural_or_file_error',findings=[dict(kind='structural_or_file_error',reason=str(exc))])
        rows.append(row)
    return dict(scene_count=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
        unique_image_files=len({x['path'] for x in cache.values()}),
        image_references=sum(r.get('image_references',0) for r in rows),
        deleted_actions=sum(r.get('deleted_actions',0) for r in rows),deleted_checks=sum(r.get('deleted_checks',0) for r in rows),
        scenes=rows,scope='file integrity, protected boundaries, transcripts, numeric tool-feedback consistency and deletion dependencies; no full physical simulation or visual quality claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();report=audit(a.input_root,a.source_root)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},ensure_ascii=False))
    return int(any(r['status']!='passed_available_checks' for r in report['scenes']))

if __name__=='__main__':raise SystemExit(main())
