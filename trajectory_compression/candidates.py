"""Bounded correction intervals; never flatten away dependency barriers."""
import json
import math
import re

MOVES = {'move_furniture_tool', 'snap_to_object_tool', 'move_wall_object',
         'move_ceiling_object', 'rescale_furniture_tool', 'rescale_wall_object',
         'rescale_ceiling_object'}
CHECKS = {'check_physics', 'check_reachability', 'check_facing_tool'}
READ_ONLY = CHECKS | {'observe_scene', 'get_current_scene_state'}

def mutation_indices(episode):
    return [i for i, t in enumerate(episode['turns']) if any(
        c.get('name') not in READ_ONLY for c in t.get('tool_calls', []))]

def decode(value):
    if isinstance(value, str):
        try: return json.loads(value)
        except ValueError: return value
    return value

def check_status(value):
    """Parse native physics reports; conflicting evidence always prefers failure."""
    value = decode(value)
    if isinstance(value, dict):
        verdicts = []
        if isinstance(value.get('has_violations'), bool): verdicts.append(not value['has_violations'])
        if isinstance(value.get('passed'), bool): verdicts.append(value['passed'])
        for key in ('violations', 'collisions', 'door_clearance_violations', 'window_access_violations'):
            if isinstance(value.get(key), list): verdicts.append(not value[key])
        if False in verdicts: return 'bad'
        if verdicts: return 'good'
    if isinstance(value, str):
        text = value.strip()
        if re.match(r'^Physics violations detected \(\d+ issue\(s\)\):', text, re.I): return 'bad'
        if text.lower() in ('no physics violations detected.',
                            'no physics violations detected. all objects are properly placed.'):
            return 'good'
    return 'unknown'

def candidates(stage, episode):
    turns = episode['turns']; last = mutation_indices(episode)
    protected = last[-1] if last else -1
    found = []; active = []; checks = []; object_id = None
    last_check = None; entry_check = None
    def flush():
        nonlocal active, checks, object_id, entry_check
        if len(active) >= 2:
            end = max([active[-1]['turn_index']] + [x['turn_index'] for x in checks])
            observations=[]
            indexes=[i for i in range(active[0]['turn_index'],end+1)
                     if turns[i].get('role')=='assistant']
            following=next((i for i in range(end+1,len(turns)) if turns[i].get('role')=='assistant'),None)
            if following is not None:indexes.append(following)
            for i in indexes:
                t=turns[i]
                paths=t.get('images') or ([t['image']] if t.get('image') else [])
                observation={'turn_index':i,'relation':'following_assistant_input' if i==following else 'assistant_input_before_action',
                             'images':list(paths)}
                if stage=='ceiling' and i>protected and not paths:
                    observation['availability']='not_captured_after_terminal_action'
                observations.append(observation)
            found.append({'observations':observations,'segment_id': f'{stage}:{active[0]["turn_index"]}:{end}',
                          'stage': stage, 'start': active[0]['turn_index'], 'end': end,
                          'operations': active.copy(), 'checks': checks.copy(),
                          'repair_confirmed': bool(checks and checks[-1]['status'] == 'good'),
                          'protected_turn': protected, 'entry_check': entry_check,
                          'events': sorted(active + checks, key=lambda x:x['turn_index'])})
        active = []; checks = []; object_id = None; entry_check = None
    for ti, turn in enumerate(turns):
        if turn.get('role') == 'tool_results': continue
        if turn.get('role') != 'assistant':
            flush(); last_check = None; continue  # user messages / memory updates are barriers
        calls = turn.get('tool_calls', [])
        if len(calls) != 1:
            flush(); last_check = None; continue  # don't partially delete a concurrent batch
        call = calls[0]; name = call['name']; args = call.get('arguments') or {}
        if name in CHECKS:
            result = turns[ti + 1]['results'][0]['output']
            last_check = {'turn_index': ti, 'tool': name, 'status': check_status(result), 'output': result}
            if active:
                checks.append(last_check)
                if last_check['status'] == 'good': flush()
            continue
        oid = args.get('object_id')
        if name not in MOVES or not oid:
            flush(); last_check = None; continue
        if active and oid != object_id: flush()
        if not active: entry_check = last_check
        last_check = None
        object_id = oid
        metrics = {}
        if active:
            old = active[-1]['arguments']
            keys = [k for k in ('x', 'y', 'z', 'position_x', 'position_y', 'position_z')
                    if isinstance(args.get(k), (int, float)) and isinstance(old.get(k), (int, float))]
            same_frame = name == active[-1]['tool'] and args.get('wall_surface_id') == old.get('wall_surface_id')
            if keys and same_frame: metrics['parameter_distance'] = math.sqrt(sum((args[k]-old[k])**2 for k in keys))
            metrics['exact_repeat'] = name == active[-1]['tool'] and args == old
        active.append({'step': ti, 'turn_index': ti, 'call_index': 0, 'tool': name,
                       'arguments': args, 'object_id': oid, 'metrics': metrics,
                       'results': turns[ti+1]['results'], 'protected': ti >= protected})
    flush()
    return found


def deletable_turns(segment):
    """Only interior correction actions; entry/exit actions and stage tail stay."""
    return {op['turn_index'] for op in segment['operations'][1:-1]
            if not op['protected'] and op['turn_index'] not in (segment['start'],segment['end'])}


def dependent_physics_checks(episodes, action_drops):
    """Remove intermediate checks belonging to a removed trial, never boundaries.

    Only whole, single-call check_physics turns directly following a deleted
    interior operation qualify. Stop at any retained action/other call/context
    message. First/last correction operations, segment exit checks and stage
    tails keep their protection. Tool results are paired by the output writer.
    """
    checks=set()
    for stage,episode in episodes.items():
        turns=episode['turns'];mutations=mutation_indices(episode)
        tail=mutations[-1] if mutations else 0
        for segment in candidates(stage,episode):
            allowed=deletable_turns(segment)
            for st,index in action_drops:
                if st!=stage or index not in allowed:continue
                for ti in range(index+1,segment['end']):
                    turn=turns[ti]
                    if turn.get('role')=='tool_results':continue
                    calls=turn.get('tool_calls') or []
                    if ti>=tail or turn.get('role')!='assistant' or len(calls)!=1 or calls[0].get('name')!='check_physics':break
                    checks.add((stage,ti))
    return checks
