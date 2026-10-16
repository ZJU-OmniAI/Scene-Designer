"""Shared API and formatting helpers for construction CoT annotation."""
import base64, json, os, sys, time, urllib.request, urllib.error
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("COT_MODEL", "gpt-5.2")


STAGES = ["floor_plan", "furniture", "wall", "ceiling"]

IMG_STAGES = {"furniture", "wall", "ceiling"}

COT_SYS = """You write the assistant's first-person chain-of-thought (the `think`) for a SINGLE step of an indoor-scene design trajectory. A designer agent builds a room by calling tools (generate assets, add / move / rescale / snap / remove furniture, physics checks). At each step it first writes its reasoning, then calls a tool. Reconstruct that reasoning for one given step, at high quality.

TEMPORAL RULE (critical). You are shown: (a) the user's room request; (b) the scene as it looks RIGHT NOW, BEFORE this step's action (a top-down render, when available); (c) the decisions already made in earlier steps AND their tool results (including any physics-check findings you have already seen); (d) the tool call this step will make. You have NOT seen the result of THIS step's action. Write only what the designer could reason BEFORE acting. Never mention or assume the outcome of the current action, and never reference future steps. You MAY and SHOULD cite specific findings from the immediately preceding tool result (e.g. a physics check that reported collisions or blocked openings), since those are already known to you.

COORDINATE SYSTEM (authoritative - use these numbers; do NOT guess compass directions from the picture). The room is centred on the origin (0,0) and positions are in metres. +X points EAST, -X points WEST; +Y points NORTH, -Y points SOUTH (so the north wall is at maximum Y, south at minimum Y, east at maximum X, west at minimum X). yaw is in degrees, counter-clockwise, with 0deg facing NORTH (+Y), 90deg facing WEST (-X), 180deg facing SOUTH (-Y), and 270deg / -90deg facing EAST (+X). The red dots in the render are (x, y) coordinate markers in metres - read positions off them. CRITICAL: every compass word, wall reference, side, and facing you write MUST agree with the object's numeric (x, y, yaw) taken from the action or history. For example an object at y=-1.2 is toward the SOUTH (never call it "north"); an object at x=+1.5 is toward the EAST; yaw=180 faces SOUTH. Never state a direction that contradicts the coordinates, and never rely on a compass hunch from the image when the numbers disagree. Different tools carry position in different fields, all in this same frame: floor furniture (add / move) uses world x, y (metres) and yaw; ceiling fixtures (place_ceiling_object) use world position_x, position_y in the SAME +X-east / +Y-north frame (so a fixture at position_x=1.95, position_y=-1.65 is in the SOUTH-EAST corner, and position_x=0, position_y=0 is the room centre); wall objects (place_wall_object) come with wall_surface_id (e.g. "<room>_north", "_south", "_east", "_west") that tells you EXACTLY which wall the piece is on - use it directly, and note that its position_x is the horizontal offset ALONG that wall (a local distance, NOT a world east/west coordinate) while position_z is height; door / window placement uses wall_id letters (typically A=north, B=south, C=east, D=west). CRITICAL: always name the specific piece a step places by the asset_id / object_id in the action - never guess which object a place_wall_object / place_ceiling_object / add step is handling (e.g. do not call a landscape print a "pendant").

The think must develop, as ONE coherent first-person passage (not labeled sections):
1. Scene understanding - Read the top-down image (if given): what objects are already placed and roughly where, what the room is for, which functional zones exist, which areas are empty or crowded. Relate this to the user's request. Judge where and how the target asset should be placed so the result is both functionally sound (reachable, near its functional partners, sensible orientation) and visually coherent (balanced, aligned, not cramped).
2. Spatial reasoning - Reason concretely and quantitatively: which anchor the placement is relative to (a specific wall, or a specific existing object), the approximate position and facing (meters from a wall, degrees of yaw), needed clearances and distances, alignment with neighbours, and why these choices avoid collisions, keep doors/windows unblocked, and preserve circulation. When correcting an issue reported by a preceding physics check, name the specific violation (object pair, penetration in cm, blocked door/window) and explain how the adjustment resolves it.
3. Layout planning - How this step fits the overall plan and functional zoning: what has been established, what this step contributes, what still remains.
Conclude at the decision: state the chosen action (tool + key arguments) as the natural consequence of the reasoning. The action you justify MUST be exactly the one given for this step (same object, same tool). Do not invent alternatives or extra steps.

For a physics-check step: reason that it is time to verify the current layout for collisions / penetrations / blocked openings / reachability before proceeding (you cannot know the result yet). For a correction step (move / rescale / snap): ground the reasoning in the specific issues the last physics check reported and in what is visible, then decide the adjustment; do not claim the fix succeeded. For a final summary step (no tool call): briefly conclude that the room is complete, recapping the main pieces and that the layout is coherent and physically valid.

STYLE: first person, like the designer thinking aloud; fluent, specific, grounded in what the image shows; thorough but not padded; use approximate numbers (meters, degrees) where they add precision. Output ONLY the think text - no headers, no bullet lists, no preamble."""

_API_KEY = None

def _key():
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.environ.get("OPENAI_API_KEY", "")
    if not _API_KEY:
        raise SystemExit("Set OPENAI_API_KEY (or pass --api-key).")
    return _API_KEY

def gpt_call(system, user_content, retries=3, want_json=False):
    """Call the endpoint with bounded transient retries (0 explicitly means unlimited)."""
    body = {"model": MODEL, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}]}
    if want_json:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode()
    last = None
    a = 0
    while True:
        a += 1
        try:
            req = urllib.request.Request(BASE_URL, data=data, headers={
                "Authorization": "Bearer " + _key(), "Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=300)
            return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last = "HTTP %s %s" % (e.code, e.read().decode()[:200])
            if e.code < 500 and e.code not in (429, 520, 521, 522, 524):
                # non-retryable (bad request / unsupported model / not found)
                raise RuntimeError("gpt_call non-retryable: " + str(last))
        except Exception as e:
            last = "%s %s" % (type(e).__name__, e)
        if retries and a >= retries:
            raise RuntimeError("gpt_call failed after %d attempts: %s" % (a, last))
        if a % 5 == 0:
            print("  [gpt_call] attempt %d still failing (%s); backing off..."
                  % (a, str(last)[:90]), file=sys.stderr, flush=True)
        time.sleep(min(60, 5 * a))

def b64img(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

def action_desc(tc):
    if not tc:
        return ""
    parts = []
    for c in tc:
        args = c.get("arguments") or {}
        keep = {k: v for k, v in args.items()
                if k in ("object_id", "asset_id", "name", "short_names",
                         "x", "y", "yaw",
                         "wall_id", "wall_surface_id",
                         "position_x", "position_y", "position_z", "rotation_degrees",
                         "scale", "target_id", "asset_description")}
        parts.append(f"{c['name']}({json.dumps(keep, ensure_ascii=False)})")
    return "; ".join(parts)

def summarize_result(res):
    outs = []
    for r in (res or []):
        name = r.get("name", "?")
        out = r.get("output", "")
        if name == "check_physics":
            outs.append(str(out).strip())
            continue
        try:
            o = json.loads(out) if isinstance(out, str) else out
        except Exception:
            outs.append(f"{name}: {str(out)[:120]}")
            continue
        if isinstance(o, dict):
            if o.get("success") is False:
                outs.append(f"{name} FAILED: {o.get('message') or o.get('error_type')}")
            elif name == "generate_assets":
                a = o.get("assets") or []
                items = [f"{x.get('object_id')}({x.get('width',0):.2f}x{x.get('depth',0):.2f}x{x.get('height',0):.2f})" for x in a]
                outs.append("generated: " + ", ".join(items))
            else:
                oid = o.get("object_id") or o.get("asset_id") or ""
                pos = o.get("new_position") or o.get("position") or o.get("current_position")
                pstr = f" at ({pos['x']:.2f},{pos['y']:.2f})" if isinstance(pos, dict) else ""
                outs.append(f"{name} ok: {oid}{pstr}")
        else:
            outs.append(f"{name}: {str(out)[:120]}")
    return " | ".join(outs)
