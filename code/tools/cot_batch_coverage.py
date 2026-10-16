"""Per-call coverage contract for post-hoc scene rationales (no API calls)."""
import json

VERSION = 2
TARGET_MARKER = 'ANNOTATION TARGET TOOL CALLS:\n'

GENERATION_NOTE = """
CURRENT-BATCH COVERAGE: Every numbered target call belongs to THIS assistant
turn, including additional asset-generation batches and repeated placements or
moves of the same asset/object. Explain the whole current plan, not just its
first call. Before writing, account for every call_index. Group related calls
by function or wall to stay concise; do not print this internal checklist or
call indices in the rationale. A group must identify the operations, objects
or instance counts, and intended arrangement/change. Mentioning an object in
the room inventory is not explaining an action on it. 'I will place everything'
alone is not coverage. For repeated moves, preserve their order and final target.
For multiple generation calls, cover each batch, including variants/retries.
'Then' may describe execution order WITHIN this turn. Never describe a target
call only as optional work for a later turn, or end with only the first action
as if it were the entire plan. For example, mirror + two sconces + seven other
wall placements must all be accounted for now, possibly grouped by wall; do
not say 'I place the mirror and sconces now; the rest can be arranged later'.
Full batch coverage takes priority over the usual word target. Cut repeated
scene inventories and generic caveats first; exceed the target when needed.
"""

REVIEW_NOTE = """
MANDATORY CURRENT-BATCH AUDIT: In addition to issues and any current_pose_claims,
return action_coverage: [{"call_index": 0, "status": "covered",
"quotes": ["exact contiguous rationale excerpt"],
"reason": "how this excerpt explains this particular target call"}].
Return exactly one entry for EACH numbered target call, including repeated
calls on one object and every generation batch; use [] for no target calls.
Allowed status: covered, grouped, omitted, deferred.
covered/grouped require nonempty exact quotes and a reason mapping this call's
operation and object/instance to the CURRENT plan. The same quote may cover
several calls if it genuinely describes their group, counts and arrangement.
Aliases, concise functional/wall groupings and within-turn 'then' are allowed;
do not demand a verbatim list of all arguments. But an inventory mention or
'place all pieces' without identifying their plan is not sufficient. Check
repeated instances, final coordinates after repeated moves, and later batches
of generated assets rather than treating one asset/type as the entire batch.
Use omitted if the operation has no explanation (quotes may be []). Use
deferred if it is explained ONLY as later/optional work outside this turn;
quote that wording. Look at the WHOLE paragraph: an introductory 'later' is
not an error if the action is then clearly included in this same turn.
Omissions are evidenced by the TARGET CALL, not a nonexistent erroneous sentence.
Do not put pure omissions in issues, whose claim field requires an exact quote.
'Do not demand exhaustive narration' permits concise grouping, NOT omitted or
deferred target calls. A correct first-action explanation is not a full pass.

Apply these strict boundaries before assigning covered/grouped:
1. A vague list of remaining object TYPES plus 'spread across remaining walls'
   does not identify repeated instances or their arrangement. Do not fill in
   missing counts, wall assignments, or plans from the target calls yourself.
   Evidence must come from the rationale. Mark the unaccounted calls omitted.
2. If the rationale limits 'this step' or 'the immediate move' to an initial
   subset and relegates the rest to 'after that I can ...' or generic future
   fine-tuning, mark those remaining calls deferred unless another sentence
   clearly commits them to the current batch with a concrete grouped plan.
   Merely naming the later objects does not override that scope restriction.
3. Two moves of the SAME object to different targets need evidence of BOTH
   moves and the final target (numeric or unambiguous relational description).
   A generic 'fine-tune pair spacing' cannot cover an unmentioned second move
   or override a stated endpoint that is only the first move's endpoint.
These rules do not reject ordinary within-turn sequencing: 'first A, then
place two B on the west wall and C on the north wall' is a concrete batch plan.
Do not demand all coordinates when the arrangement is otherwise identifiable.
"""


def normalize(text):
    return ' '.join(text.translate(str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"'})).split())


def target_calls(request):
    """Read the last target block, after any quoted system/task text."""
    text = '\n'.join(x.get('text', '') for x in request if x.get('type') == 'text')
    if TARGET_MARKER not in text:
        if request:
            raise ValueError('Missing annotation target block')
        return []  # Compatibility for isolated non-action reviewer unit tests.
    calls = json.loads(text.rsplit(TARGET_MARKER, 1)[1].split('\n', 1)[0])
    if not isinstance(calls, list):
        raise ValueError('Annotation targets must be a list')
    return calls


def coverage_errors(report, calls, answer):
    entries = report.get('action_coverage')
    if not calls and entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError('Missing action_coverage list')
    seen = set()
    errors = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Coverage entry must be an object')
        index = entry.get('call_index')
        if type(index) is not int or not 0 <= index < len(calls) or index in seen:
            raise ValueError('Invalid or duplicate coverage call_index')
        seen.add(index)
        status = entry.get('status')
        if status not in ('covered', 'grouped', 'omitted', 'deferred'):
            raise ValueError('Invalid coverage status')
        reason = entry.get('reason')
        quotes = entry.get('quotes')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Missing coverage reason')
        if not isinstance(quotes, list) or any(
            not isinstance(q, str) or not q.strip() or normalize(q) not in normalize(answer)
            for q in quotes
        ):
            raise ValueError('Coverage quotes must be exact rationale excerpts')
        if status != 'omitted' and not quotes:
            raise ValueError('Covered, grouped and deferred calls require quotes')
        if status in ('omitted', 'deferred'):
            errors.append(f'batch coverage: call_index={index} {status}; target='
                          f'{json.dumps(calls[index], ensure_ascii=False)}; {reason}')
    if seen != set(range(len(calls))):
        raise ValueError('Incomplete action_coverage: every call_index is required')
    return errors
