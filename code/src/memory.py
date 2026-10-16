#!/usr/bin/env python3
"""Demo: walk an SFT file (or whole run dir) and show how a deterministic
plug-and-play memory state would evolve after each tool_result. READ-ONLY —
does not modify any SFT file.

Each tool name has a small hand-written handler that updates a structured
memory dict from (arguments, output). After every assistant→tool_results
turn pair, we print the rendered memory snapshot so you can eyeball what
the model would see if this memory were fed back into context.

Usage:
    python src/memory.py datasets/sft/2026-04-19_16-13-13_scene_000/furniture_bathroom.json
    python src/memory.py datasets/sft/2026-04-19_16-13-13_scene_000/                       # whole run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# =========================================================================
# helpers
# =========================================================================

def _as_text(out) -> str:
    """check_physics outputs are plain strings; others may be dicts that
    contain a `message` field. Reduce to a string for parsing."""
    if isinstance(out, str):
        parsed = _as_dict(out)
        if parsed and isinstance(parsed.get("message"), str):
            return parsed["message"]
        return out
    if isinstance(out, dict):
        return out.get("message", "") or ""
    return ""


def _as_dict(out) -> dict:
    """Normalize tool_result output to a dict. SFT files store outputs as
    raw JSON strings (from scenesmith); we must parse before structural
    access. Returns {} on failure."""
    if isinstance(out, dict):
        return out
    if isinstance(out, str):
        try:
            parsed = json.loads(out)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _success(out, name: str = "") -> bool:
    """True iff the tool output indicates success. Handles all 3 formats:
    dict (inference), JSON string (SFT room stages), Python repr (SFT floor_plan).

    Special case: generate_*_assets often returns top-level success=false
    when ANY asset fails (e.g. 1 of 7 fails). But the OTHER assets in the
    `assets` array DID succeed and ARE usable. So we treat the call as
    successful as long as at least one asset was returned."""
    if out is None or out == "":
        return False
    d = _as_dict(out)
    if d.get("error"):
        return False
    if name.startswith("generate_") and name.endswith("_assets"):
        assets = d.get("assets") or []
        return len(assets) > 0
    if d and "success" in d:
        return bool(d["success"])
    s = str(out) if out is not None else ""
    if 'success=False' in s: return False
    if '"success": false' in s: return False
    if "'success': false" in s: return False
    if s.startswith("ERROR"): return False
    return True


def _err_msg(out) -> str:
    """Extract a short error message from a failed tool output (for memory display)."""
    d = _as_dict(out)
    if d:
        return (d.get("message") or "")[:120]
    s = str(out) if out is not None else ""
    # try regex on Python repr or "message" field
    import re as _re
    m = _re.search(r"message\s*=\s*['\"]([^'\"]+)['\"]", s) or _re.search(r'"message"\s*:\s*"([^"]+)"', s)
    return (m.group(1) if m else s)[:120]


def _brief_result(name: str, out) -> str:
    """Short result summary for SUCCESSFUL tool_call (shown in recent_tool_calls).
    Per-tool format so the model sees ID returned by get_material, asset names
    from generate_*_assets, violation count from check_physics, etc."""
    if out is None:
        return ""
    d = _as_dict(out)
    s = str(out) if out is not None else ""

    if name == "check_physics":
        text = _as_text(out)
        result = _parse_physics_result(text)
        if result.get("status") == "error":
            return "UNKNOWN: physics check unavailable/failed"
        if result["passed"]:
            return "no violations"
        m = re.search(r"\((\d+)\s+issue", text)
        return f"{m.group(1)} violations" if m else text[:80].replace("\n", " ")

    if name == "get_material":
        mid = d.get("material_id") if d else None
        if not mid:
            m = re.search(r"material_id=['\"]([^'\"]+)['\"]", s) or re.search(r'"material_id"\s*:\s*"([^"]+)"', s)
            if m: mid = m.group(1)
        return f"→ {mid}" if mid else ""

    if name.startswith("generate_") and name.endswith("_assets"):
        if d:
            assets = d.get("assets") or []
            ids = [a.get("object_id") for a in assets if isinstance(a, dict) and a.get("object_id")]
            head = ", ".join(ids[:3])
            return f"→ {len(ids)} asset(s): {head}{'...' if len(ids) > 3 else ''}"
        return ""

    if name in ("add_furniture_to_scene_tool", "place_wall_object", "place_ceiling_object"):
        oid = d.get("object_id") if d else None
        return f"→ object_id={oid}" if oid else "placed"

    # Generic: pull message field, fallback to truncated str
    msg = d.get("message", "") if d else ""
    if not msg:
        m = re.search(r"message\s*=\s*['\"]([^'\"]+)['\"]", s) or re.search(r'"message"\s*:\s*"([^"]+)"', s)
        if m: msg = m.group(1)
    return msg[:100] if msg else ""


_PASS_PATTERNS = (
    "no physics violations",
    "no violations",
    "all clear",
    "all objects are properly placed",
)


def _parse_violations(text: str) -> list[str]:
    """Pull violation lines out of check_physics text. Returns empty list
    when text indicates a passing check."""
    low = text.lower()
    if any(p in low for p in _PASS_PATTERNS):
        return []
    out = []
    for line in text.splitlines():
        s = line.strip().lstrip("-•·").strip()
        if not s:
            continue
        if s.startswith(("Physics", "Collisions", "Door", "Please")):
            continue
        if s.endswith(":"):
            continue
        out.append(s)
    return out


_PHYSICS_ERROR_PATTERNS = ("physics check unavailable", "physics check timed out", "physics check failed", "physics check skipped")


def _parse_physics_result(text: str) -> dict:
    """Parse physics output without treating an unavailable check as a pass."""
    low = text.lower()
    if not text.strip():
        return {"status": "error", "passed": False, "violations": [], "error": "Missing physics result"}
    for pattern in _PHYSICS_ERROR_PATTERNS:
        if pattern in low:
            return {"status": "error", "passed": False, "violations": [], "error": text.strip()}
    viols = _parse_violations(text)
    if any(p in low for p in _PASS_PATTERNS):
        return {"status": "passed", "passed": True, "violations": []}
    return {"status": "failed", "passed": False, "violations": viols}


# =========================================================================
# per-stage handlers — pure Python, no LLM
# =========================================================================

_RECENT_MAX = 5  # how many past tool_calls to remember


def _track_recent(mem, name, args, output=None):
    """Append the just-executed tool_call (with its result) to
    `mem.recent_tool_calls`. The result/error is included so the model
    sees a clear signal of what happened, not just what was attempted."""
    if "recent_tool_calls" not in mem:
        mem["recent_tool_calls"] = []
    entry = {"name": name, "arguments": dict(args or {})}
    oid = (args or {}).get("object_id")
    for key in ("assets", "wall_assets", "ceiling_assets"):
        obj = mem.get(key, {}).get(oid)
        if obj and obj.get("placed"):
            entry["before_state"] = {k: obj[k] for k in ("pos", "rot", "w", "d", "h", "wall") if k in obj}
            break
    # A successful edit does not inherit the previous check's validity.
    # Keep this outside the truncated recent-history window.
    geometry_edit = name.startswith(("add_", "place_", "move_", "remove_", "rescale_", "snap_")) or name in ("resize_room", "set_wall_height", "generate_room_specs")
    if geometry_edit and _success(output, name) and mem.get("physics") is not None:
        p = mem["physics"]
        p["edits_since_check"] = p.get("edits_since_check", 0) + 1
        p.setdefault("edits_after_check", []).append(entry)
    if not _success(output, name):
        entry["failed"] = True
        entry["error"] = _err_msg(output)
    else:
        result = _brief_result(name, output)
        if result:
            entry["result"] = result
    mem["recent_tool_calls"].append(entry)
    if len(mem["recent_tool_calls"]) > _RECENT_MAX:
        mem["recent_tool_calls"] = mem["recent_tool_calls"][-_RECENT_MAX:]


def update_floor_plan(mem, name, args, output):
    _track_recent(mem, name, args, output)
    # If the tool FAILED, do NOT mutate scene memory — the model should see
    # ❌ FAILED in recent_tool_calls and realize the action didn't happen.
    if not _success(output, name) and name != "check_physics":
        return
    if name == "generate_room_specs":
        try:
            specs = json.loads(args.get("room_specs_json", "[]"))
        except json.JSONDecodeError:
            specs = []
        # generate_room_specs replaces the layout in the backend.  In
        # particular, it clears all existing doors/windows and rebuilds wall
        # labels; keeping the old entries would make subsequent tool calls
        # reason about openings that no longer exist.
        mem["rooms"].clear()
        mem["doors"].clear()
        mem["windows"].clear()
        mem["wall_segment_labels"] = {}
        room_type_counts = {}
        for s in specs:
            room_type = s.get("type", "room")
            room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
            # Match FloorPlanTools' room_id generation: the first room keeps
            # its type, repeated types receive _2, _3, ... suffixes.
            rid = room_type if room_type_counts[room_type] == 1 else f"{room_type}_{room_type_counts[room_type]}"
            mem["rooms"][rid] = {
                "type": room_type,
                "width": s.get("width", 5.0),
                "depth": s.get("depth", 4.0),
                "prompt": s.get("prompt", ""),
            }
        # floor_plan outputs are Python dataclass repr (NOT JSON), so _as_dict
        # fails. Extract wall_segment_labels via regex + ast.literal_eval.
        labels = _as_dict(output).get("wall_segment_labels")
        if not labels and isinstance(output, str):
            import ast
            m = re.search(r"wall_segment_labels\s*=\s*(\{[^}]*\})", output)
            if m:
                try:
                    labels = ast.literal_eval(m.group(1))
                except (ValueError, SyntaxError):
                    labels = None
        if labels:
            mem["wall_segment_labels"] = labels

    elif name == "resize_room":
        rid = args.get("room_id")
        if rid in mem["rooms"]:
            mem["rooms"][rid]["width"] = args["width"]
            mem["rooms"][rid]["depth"] = args["depth"]
        # Resizing re-applies openings in the backend and removes any door or
        # window that no longer fits.  The current Result exposes those IDs in
        # its human-readable message (e.g. "Removed 1 door(s): door_1 (...)").
        # Remove them from memory as well; otherwise later planning steps may
        # refer to openings that no longer exist.
        text = _as_text(output)
        if not text and isinstance(output, str):
            text = output
        for kind, store in (("door", mem["doors"]), ("window", mem["windows"])):
            match = re.search(
                rf"Removed\s+\d+\s+{kind}\(s\):\s*(.*?)(?=\s*\|\s*Removed\s+\d+\s+(?:door|window)\(s\):|$)",
                text,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            for item in match.group(1).split(";"):
                id_match = re.match(r"\s*([A-Za-z0-9_-]+)", item)
                if id_match:
                    store.pop(id_match.group(1), None)

    elif name == "set_wall_height":
        mem["wall_height"] = args.get("height_meters")

    elif name == "add_door":
        # Extract door_N from output (works on both SFT string repr and
        # inference dict — dict has 'message' field that contains the id).
        msg = output if isinstance(output, str) else (
            (output.get("message", "") if isinstance(output, dict) else str(output)) or ""
        )
        did = f"door_{len(mem['doors']) + 1}"
        m = re.search(r"door_(\d+)", msg)
        if m: did = f"door_{m.group(1)}"
        mem["doors"][did] = {
            "wall_id": args.get("wall_id"), "position": args.get("position"),
            "width": args.get("width", 0.9), "height": args.get("height", 2.1),
        }

    elif name == "remove_door":
        mem["doors"].pop(args.get("door_id"), None)

    elif name == "add_window":
        msg = output if isinstance(output, str) else (
            (output.get("message", "") if isinstance(output, dict) else str(output)) or ""
        )
        wid = f"window_{len(mem['windows']) + 1}"
        m = re.search(r"window_(\d+)", msg)
        if m: wid = f"window_{m.group(1)}"
        mem["windows"][wid] = {
            "wall_id": args.get("wall_id"), "position": args.get("position"),
            "width": args.get("width", 1.2), "height": args.get("height", 1.2),
            "sill_height": args.get("sill_height", 0.9),
        }

    elif name == "remove_window":
        mem["windows"].pop(args.get("window_id"), None)

    elif name == "get_material":
        desc = args.get("description", "")
        mat_id = None
        if isinstance(output, dict):
            mat_id = output.get("material_id")
        elif isinstance(output, str):
            m = re.search(r"material_id='([^']+)'", output) or re.search(r'"material_id"\s*:\s*"([^"]+)"', output)
            if m: mat_id = m.group(1)
        if mat_id:
            mem["material_cache"][desc] = mat_id

    elif name == "set_room_materials":
        rid = args.get("room_id")
        if rid not in mem["rooms"]:
            return
        r = mem["rooms"][rid].setdefault("materials", {})
        if args.get("wall_material_id"):
            r["wall"] = args["wall_material_id"]
        if args.get("floor_material_id"):
            r["floor"] = args["floor_material_id"]

    elif name == "set_exterior_material":
        mem["exterior_material"] = args.get("material_id")

    elif name == "check_physics":
        text = _as_text(output)
        mem["physics"] = _parse_physics_result(text)


def _invalidate_snap_relations(bag, changed_ids):
    """Snap is a one-time pose operation, not a persistent attachment."""
    changed = set(changed_ids)
    for oid, asset in bag.items():
        if oid in changed or asset.get("snapped_to") in changed:
            asset.pop("snapped_to", None)


def update_room_stage(mem, name, args, output, *, key, place_arg_pos_x="x", place_arg_pos_y="y", rot_arg="yaw"):
    """Generic body-level handler used by furniture/wall/ceiling — they all
    share the same shape: generate_*_assets, place/move/remove_*, check_physics."""
    _track_recent(mem, name, args, output)
    bag = mem[key]
    library = mem.setdefault("asset_library", {})
    success = _success(output, name)
    # generate_*_assets: partial success is COMMON (scenesmith returns
    # success=false when ANY asset fails, even if most succeed). We must
    # parse `assets` array regardless of top-level success flag — every
    # entry in the array IS a successfully-generated asset.
    if name.startswith("generate_") and name.endswith("_assets"):
        out_d = _as_dict(output)
        for a in out_d.get("assets") or []:
            if not isinstance(a, dict):
                continue
            aid = a.get("object_id")
            if aid:
                version = library.get(aid, {}).get("version", 0) + 1
                library[aid] = {
                    "name": a.get("name"), "w": a.get("width"),
                    "d": a.get("depth"), "h": a.get("height"),
                    "placed": False,
                    "version": version,
                }
                if not bag.get(aid, {}).get("placed"):
                    bag[aid] = dict(library[aid])
        return

    # add/place_* handle their own failure path (stamp error onto asset entry).
    # All other mutating handlers must abort on failure so memory doesn't
    # record fake state. check_physics has no success/failure concept.
    PLACE_TOOLS = ("add_furniture_to_scene_tool", "place_wall_object", "place_ceiling_object")
    if not success and name not in PLACE_TOOLS and name != "check_physics":
        return

    # place/add → first placement
    if name in ("add_furniture_to_scene_tool", "place_wall_object", "place_ceiling_object"):
        aid = args.get("asset_id")
        out_d = _as_dict(output)
        # On failure: stamp a short error onto the asset so the next step's
        # memory shows why it isn't placed, instead of silently pretending it is.
        if not out_d.get("success"):
            if aid in bag:
                bag[aid]["error"] = {
                    "type": out_d.get("error_type") or "error",
                    "msg":  (out_d.get("message") or "")[:80],
                }
            return
        # Asset IDs and live scene IDs have independent namespaces.
        scene_oid = out_d.get("object_id") or aid
        src = library.get(aid) or bag.get(aid) or {}
        entry = {k: src.get(k) for k in ("name", "w", "d", "h")}
        entry["source_asset"] = aid
        entry["source_asset_version"] = src.get("version")
        bag[scene_oid] = entry
        entry.pop("error", None)
        entry["placed"] = True
        # Record the authoritative final pose returned by the placement tool.
        # Placement noise may make it differ from the requested arguments.
        # Keep argument fallback for older result formats.
        if name == "add_furniture_to_scene_tool":
            position = out_d.get("position") or {}
            rotation = out_d.get("rotation") or {}
            final_x = position.get("x", args.get(place_arg_pos_x))
            final_y = position.get("y", args.get(place_arg_pos_y))
            final_rot = rotation.get("yaw", args.get(rot_arg, 0))
        elif name == "place_wall_object":
            final_x = out_d.get("position_x", args.get(place_arg_pos_x))
            final_y = out_d.get("position_z", args.get(place_arg_pos_y))
            # Wall DTO uses `rotation_deg`; accept the alternate spelling for
            # compatibility with older/custom result producers.
            final_rot = out_d.get(
                "rotation_deg",
                out_d.get("rotation_degrees", args.get(rot_arg, 0)),
            )
        else:  # place_ceiling_object
            final_x = out_d.get("position_x", args.get(place_arg_pos_x))
            final_y = out_d.get("position_y", args.get(place_arg_pos_y))
            final_rot = out_d.get("rotation_degrees", args.get(rot_arg, 0))
        entry["pos"] = (final_x, final_y)
        entry["rot"] = final_rot
        wall = args.get("wall_surface_id")
        if wall: entry["wall"] = wall
        return

    # snap → updates pos & rot from result (already guarded by top-level success check)
    if name == "snap_to_object_tool":
        oid = args.get("object_id")
        _invalidate_snap_relations(bag, [oid])
        if oid in bag:
            bag[oid]["snapped_to"] = args.get("target_id")
            out_d = _as_dict(output)
            np = out_d.get("new_position") or {}
            if np: bag[oid]["pos"] = (np.get("x"), np.get("y"))
            # SnapToObjectResult reports the final absolute yaw in
            # `rotation_angle_degrees` when orientation was applied.  Older
            # code looked for `world_rotation`, a field that this DTO does not
            # contain, leaving memory stale after a successful reorientation.
            if out_d.get("rotation_applied"):
                yaw = out_d.get("rotation_angle_degrees")
                if yaw is not None:
                    bag[oid]["rot"] = yaw
        return

    # move/update
    if name in ("move_furniture_tool", "move_wall_object", "move_ceiling_object"):
        oid = args.get("object_id")
        _invalidate_snap_relations(bag, [oid])
        if oid in bag:
            out_d = _as_dict(output)
            # Move tools use absolute rotations with a default of 0 degrees;
            # an omitted rotation therefore resets orientation rather than
            # preserving the previous memory value.  Furniture may expose the
            # authoritative final pose in current_position/current_rotation.
            if name == "move_furniture_tool":
                current_pos = out_d.get("current_position") or {}
                current_rot = out_d.get("current_rotation") or {}
                final_x = current_pos.get("x", args.get(place_arg_pos_x))
                final_y = current_pos.get("y", args.get(place_arg_pos_y))
                final_rot = current_rot.get("yaw", args.get(rot_arg, 0))
            else:
                final_x = args.get(place_arg_pos_x)
                final_y = args.get(place_arg_pos_y)
                final_rot = args.get(rot_arg, 0)
            bag[oid]["pos"] = (final_x, final_y)
            bag[oid]["rot"] = final_rot
            wall = args.get("wall_surface_id")
            if wall: bag[oid]["wall"] = wall
        return

    # remove
    if name in ("remove_furniture_tool", "remove_wall_object", "remove_ceiling_object"):
        oid = args.get("object_id")
        _invalidate_snap_relations(bag, [oid])
        bag.pop(oid, None)
        if oid in library:
            bag[oid] = dict(library[oid])
            bag[oid]["removed_from_scene"] = True
        return

    # rescale
    if name in ("rescale_furniture_tool", "rescale_wall_object", "rescale_ceiling_object"):
        oid = args.get("object_id")
        sf = args.get("scale_factor", 1.0)
        out_d = _as_dict(output)
        # Rescaling is applied to the shared SDF, so the tool updates every
        # scene object that references that asset.  Keep memory consistent
        # with the authoritative affected_object_ids list; old result formats
        # may omit it, in which case update the requested instance only.
        affected_ids = out_d.get("affected_object_ids") or [oid]
        _invalidate_snap_relations(bag, affected_ids)
        affected_assets = {
            bag[x].get("source_asset", x) for x in affected_ids if x in bag
            and bag[x].get("source_asset_version") == library.get(bag[x].get("source_asset", x), {}).get("version")
        }
        for aid in affected_assets:
            if aid in library:
                for dim in ("w", "d", "h"):
                    if library[aid].get(dim) is not None:
                        library[aid][dim] *= sf
                if aid in bag and not bag[aid].get("placed"):
                    removed = bag[aid].get("removed_from_scene", False)
                    bag[aid] = dict(library[aid])
                    if removed:
                        bag[aid]["removed_from_scene"] = True
        for affected_oid in affected_ids:
            if affected_oid in bag:
                for k in ("w", "d", "h"):
                    if bag[affected_oid].get(k) is not None:
                        bag[affected_oid][k] = bag[affected_oid][k] * sf
        return

    if name == "check_physics":
        text = _as_text(output)
        mem["physics"] = _parse_physics_result(text)


def update_furniture(mem, name, args, output):
    update_room_stage(mem, name, args, output, key="assets")


def update_wall(mem, name, args, output):
    update_room_stage(
        mem, name, args, output,
        key="wall_assets",
        place_arg_pos_x="position_x", place_arg_pos_y="position_z",
        rot_arg="rotation_degrees",
    )


def update_ceiling(mem, name, args, output):
    update_room_stage(
        mem, name, args, output,
        key="ceiling_assets",
        place_arg_pos_x="position_x", place_arg_pos_y="position_y",
        rot_arg="rotation_degrees",
    )


HANDLERS = {
    "floor_plan": update_floor_plan,
    "furniture":  update_furniture,
    "wall":       update_wall,
    "ceiling":    update_ceiling,
}


def empty_memory(stage):
    if stage == "floor_plan":
        return {"rooms": {}, "doors": {}, "windows": {},
                "wall_height": None, "wall_segment_labels": {},
                "material_cache": {}, "exterior_material": None,
                "physics": None, "recent_tool_calls": []}
    if stage == "furniture":
        return {"assets": {}, "physics": None, "recent_tool_calls": []}
    if stage == "wall":
        return {"wall_assets": {}, "physics": None, "recent_tool_calls": []}
    if stage == "ceiling":
        return {"ceiling_assets": {}, "physics": None, "recent_tool_calls": []}
    return {}


# =========================================================================
# renderer — the compact text the model would see
# =========================================================================

def _fmt_not_placed(a: dict) -> str:
    """Render the 'not placed' label, surfacing the last error if there is one
    so the model can tell apart 'never tried yet' from 'tried and failed'."""
    err = a.get("error")
    if a.get("removed_from_scene") and not err:
        return "[removed from scene; reusable asset, NOT a pending placement]"
    if not err:
        return "[generated, not placed]"
    etype = err.get("type") or "error"
    msg = err.get("msg") or ""
    base = f"[not placed: {etype} — {msg}]" if msg else f"[not placed: {etype}]"
    # Size-related failure: re-generate smaller, don't try rescale (it needs
    # a placed object_id).
    low = (msg or "").lower()
    if any(k in low for k in (
        "too large", "exceeds", "doesn't fit", "does not fit",
        "out of bounds", "out-of-bounds", "dimension", "too big",
    )):
        base += " → regenerate smaller; rescale won't work."
    return base


def _fmt(v, fmt=".2f"):
    return f"{v:{fmt}}" if isinstance(v, (int, float)) else "?"


def render_memory(stage, mem) -> str:
    parts: list[str] = ["# Memory snapshot"]

    if stage == "floor_plan":
        if mem["rooms"]:
            parts.append(f"Rooms ({len(mem['rooms'])}):")
            for rid, r in mem["rooms"].items():
                mats = r.get("materials", {})
                m = (f"  wall={mats.get('wall', '-')}  floor={mats.get('floor', '-')}"
                     if mats else "")
                parts.append(f"  - {rid} ({r['type']}): "
                             f"{_fmt(r['width'], '.1f')}×{_fmt(r['depth'], '.1f')}m{m}")
        if mem["wall_height"]:
            parts.append(f"Wall height: {_fmt(mem['wall_height'], '.1f')}m")
        if mem["wall_segment_labels"]:
            parts.append("Wall labels: " + ", ".join(
                f"{k}={v}" for k, v in mem["wall_segment_labels"].items()))
        if mem["doors"]:
            parts.append(f"Doors ({len(mem['doors'])}):")
            for did, d in mem["doors"].items():
                parts.append(f"  - {did}: wall {d['wall_id']} ({d['position']}), "
                             f"{_fmt(d['width'], '.1f')}×{_fmt(d['height'], '.1f')}m")
        if mem["windows"]:
            parts.append(f"Windows ({len(mem['windows'])}):")
            for wid, w in mem["windows"].items():
                parts.append(f"  - {wid}: wall {w['wall_id']} ({w['position']}), "
                             f"{_fmt(w['width'], '.1f')}×{_fmt(w['height'], '.1f')}m, "
                             f"sill={_fmt(w['sill_height'], '.1f')}m")
        if mem["exterior_material"]:
            parts.append(f"Exterior material: {mem['exterior_material']}")
        if mem["material_cache"]:
            parts.append(f"Material search results ({len(mem['material_cache'])}; query describes the request, not verified properties of the returned material):")
            for desc, mid in mem["material_cache"].items():
                parts.append(f"  - query={desc[:55]!r}{'…' if len(desc) > 55 else ''}  → returned material_id={mid}")

    elif stage == "furniture":
        bag = mem["assets"]
        if bag:
            placed = sum(1 for a in bag.values() if a.get("placed"))
            parts.append(f"Furniture assets ({placed}/{len(bag)} placed):")
            for aid, a in bag.items():
                size = (f"{_fmt(a['w'])}×{_fmt(a['d'])}×{_fmt(a['h'])}m"
                        if a.get('w') else "?")
                if a.get("placed"):
                    px, py = a.get("pos", (None, None))
                    snap = a.get("snapped_to")
                    snap_s = f", snap→{snap}" if snap else ""
                    parts.append(f"  - {aid:<24s} {size:<22s} @ ({_fmt(px, '+.2f')}, "
                                 f"{_fmt(py, '+.2f')}) yaw={_fmt(a.get('rot', 0), '.0f')}°{snap_s}")
                else:
                    parts.append(f"  - {aid:<24s} {size:<22s} {_fmt_not_placed(a)}")

    elif stage == "wall":
        bag = mem["wall_assets"]
        if bag:
            placed = sum(1 for a in bag.values() if a.get("placed"))
            parts.append(f"Wall objects ({placed}/{len(bag)} placed):")
            for aid, a in bag.items():
                size = (f"{_fmt(a['w'])}×{_fmt(a['d'])}×{_fmt(a['h'])}m"
                        if a.get('w') else "?")
                if a.get("placed"):
                    px, pz = a.get("pos", (None, None))
                    parts.append(f"  - {aid:<22s} {size:<22s} on {a.get('wall', '?')} "
                                 f"@ x={_fmt(px, '.2f')}, z={_fmt(pz, '.2f')}, "
                                 f"rot={_fmt(a.get('rot', 0), '.0f')}°")
                else:
                    parts.append(f"  - {aid:<22s} {size:<22s} {_fmt_not_placed(a)}")

    elif stage == "ceiling":
        bag = mem["ceiling_assets"]
        if bag:
            placed = sum(1 for a in bag.values() if a.get("placed"))
            parts.append(f"Ceiling objects ({placed}/{len(bag)} placed):")
            for aid, a in bag.items():
                size = (f"{_fmt(a['w'])}×{_fmt(a['d'])}×{_fmt(a['h'])}m"
                        if a.get('w') else "?")
                if a.get("placed"):
                    px, py = a.get("pos", (None, None))
                    parts.append(f"  - {aid:<26s} {size:<22s} @ ({_fmt(px, '+.2f')}, "
                                 f"{_fmt(py, '+.2f')}) rot={_fmt(a.get('rot', 0), '.0f')}°")
                else:
                    parts.append(f"  - {aid:<26s} {size:<22s} {_fmt_not_placed(a)}")

    if stage != "floor_plan" and mem.get("asset_library"):
        parts.append("Asset library (asset_id for placement; separate from scene object_id):")
        for aid, asset in mem["asset_library"].items():
            parts.append(f"  - asset_id={aid}: {_fmt(asset.get('w'))}×{_fmt(asset.get('d'))}×{_fmt(asset.get('h'))}m")

    p = mem.get("physics")
    if p is not None:
        parts.append("Last recorded physics check (latest check supersedes earlier checks):")
        parts.append("  Scope: collisions, penetrations and opening obstructions; not a full navigation-reachability check.")
        if p.get("status") == "error":
            parts.append("Physics: ? UNKNOWN (check unavailable/failed)")
        elif p["passed"]:
            parts.append("Physics: ✓ PASS")
        else:
            parts.append(f"Physics: ✗ FAIL ({len(p.get('violations', []))} issue(s))")
            for v in p.get("violations", []):
                parts.append(f"  · {v}")
        edits = p.get("edits_since_check", 0)
        if edits:
            parts.append(f"  {edits} successful geometry edit(s) since this check; current geometry is NOT yet rechecked. Listed violations describe the earlier checked state.")
            parts.append("  Exact edits AFTER that latest check (oldest → newest):")
            for edit in p.get("edits_after_check", []):
                parts.append("    " + json.dumps(edit, ensure_ascii=False))
        else:
            parts.append("  No successful geometry edits since this check.")

    # Recent tool_call history — what YOU just did, oldest → newest.
    # Prevents the "I forgot what I just did" hallucination loop where the
    # model thinks it moved something when in fact it just kept calling
    # check_physics (or kept searching for the same material).
    recent = mem.get("recent_tool_calls") or []
    if recent:
        parts.append(f"Your last {len(recent)} action(s) (oldest → newest):")
        n = len(recent)
        latest_check = max((i for i, c in enumerate(recent) if c.get("name") == "check_physics"), default=-1)
        for i, call in enumerate(recent):
            if p is not None:
                if i == latest_check:
                    parts.append("  --- LATEST PHYSICS CHECK ---")
                elif i == 0 and latest_check > 0:
                    parts.append("  --- BEFORE latest physics check (already included in that checked state) ---")
                if i == latest_check + 1:
                    parts.append("  --- AFTER latest physics check ---")
            offset = i - n  # -N, -N+1, ..., -1
            cname = call.get("name", "?")
            cargs = call.get("arguments") or {}
            kv = []
            for k, v in cargs.items():
                vs = repr(v) if isinstance(v, str) else str(v)
                if len(vs) > 40:
                    vs = vs[:37] + "..."
                kv.append(f"{k}={vs}")
            args_str = ", ".join(kv)
            if len(args_str) > 110:
                args_str = args_str[:107] + "..."
            suffix = ""
            if call.get("failed"):
                err = call.get("error", "")
                suffix = f"  ❌ FAILED: {err[:60]}" if err else "  ❌ FAILED"
            elif call.get("result"):
                suffix = f"  ✓ {call['result'][:100]}"
            parts.append(f"  step{offset:+d}: {cname}({args_str}){suffix}")
            if call.get("before_state"):
                parts.append("    Recorded state BEFORE this action: " + json.dumps(call['before_state'], ensure_ascii=False))

    return "\n".join(parts) if len(parts) > 1 else "# Memory snapshot (empty)"


def memory_before_turns(sft):
    """Training and CoT share exactly the same pre-action memory snapshots."""
    stage = sft["stage"]
    handler = HANDLERS[stage]
    state = empty_memory(stage)
    snapshots = {}
    turns = sft["turns"]
    for i, turn in enumerate(turns):
        if turn.get("role") == "assistant":
            snapshots[i] = render_memory(stage, state)
        elif turn.get("role") == "tool_results":
            calls = turns[i-1].get("tool_calls") or [] if i else []
            for j, result in enumerate(turn.get("results") or []):
                call = calls[j] if j < len(calls) else {}
                name = result.get("name") or call.get("name")
                if name:
                    handler(state, name, call.get("arguments") or {}, result.get("output"))
    return snapshots


# =========================================================================
# main loop — walk SFT, update memory, print snapshot after every batch
# =========================================================================

def process_sft(sft_path: Path, *, full_dump: bool = False,
                save_dir: Path | None = None) -> None:
    sft = json.loads(sft_path.read_text(encoding="utf-8"))
    stage = sft.get("stage")
    handler = HANDLERS.get(stage)
    if handler is None:
        print(f"[skip] unsupported stage: {stage}", file=sys.stderr)
        return

    lines: list[str] = []
    def out(s: str) -> None:
        lines.append(s)
        print(s)

    out(f"\n{'='*70}\n FILE: {sft_path}\n STAGE: {stage}\n{'='*70}")
    mem = empty_memory(stage)
    turns = sft.get("turns", [])
    step = 0
    for i, turn in enumerate(turns):
        if turn.get("role") != "tool_results":
            continue
        if i == 0:
            continue
        prev = turns[i - 1]
        tool_calls = prev.get("tool_calls") or []
        results = turn.get("results") or []
        if len(tool_calls) != len(results):
            print(f"[warn] turn {i}: {len(tool_calls)} tool_calls vs "
                  f"{len(results)} results — skipping", file=sys.stderr)
            continue
        for tc, r in zip(tool_calls, results):
            step += 1
            handler(mem, tc.get("name"), tc.get("arguments") or {}, r.get("output"))

        names = [tc.get("name") for tc in tool_calls]
        out(f"\n--- after turn {i-1} (step={step}, {len(tool_calls)} call(s): "
            f"{', '.join(names)}) ---")
        out(render_memory(stage, mem))

    out(f"\n{'='*70}\n FINAL MEMORY ({stage})\n{'='*70}")
    out(render_memory(stage, mem))
    if full_dump:
        out(f"\n--- raw memory dict ---\n{json.dumps(mem, indent=2, ensure_ascii=False)}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{sft_path.stem}.memory.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[save] {out_path}", file=sys.stderr)


REPO_LOGS = Path("./logs/memory_demo")


def dump_prompts(sft_path: Path, save_dir: Path) -> None:
    """For each assistant turn, write the EXACT prompt that would be fed to the
    model (system + ## Task + ## Memory) plus the expected assistant emit
    (<think> + <tool_call>...). Image bytes are NOT included — only the text
    that lives in the user message.

    Output: <save_dir>/<sft_stem>.prompts.txt
    """
    sft = json.loads(sft_path.read_text(encoding="utf-8"))
    stage = sft.get("stage")
    handler = HANDLERS.get(stage)
    if handler is None:
        print(f"[dump-prompts] skip {sft_path.name}: unsupported stage {stage}",
              file=sys.stderr)
        return

    system_prompt = sft.get("system", "")
    turns = sft.get("turns", [])
    # First user turn carries the room prompt.
    room_prompt = ""
    if turns and turns[0].get("role") == "user":
        room_prompt = str(turns[0].get("content", ""))

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{sft_path.stem}.prompts.txt"

    sections: list[str] = []
    sections.append(
        "# Per-step prompts dump\n"
        f"# Source: {sft_path}\n"
        f"# Stage:  {stage}\n"
        f"# Format: every block below is what the trained model would actually\n"
        f"#         see at one inference step — the SYSTEM prompt, the USER\n"
        f"#         message (## Task + ## Memory), and the GT assistant emit.\n"
        f"#         At inference time the USER message also carries the current\n"
        f"#         topdown render (omitted here — see the .png files alongside\n"
        f"#         the SFT in datasets/per_step/<run>/<room>/).\n"
    )

    # The SYSTEM prompt is identical at every step — print once at the top.
    sections.append("=" * 80)
    sections.append("SYSTEM PROMPT (identical at every step)")
    sections.append("=" * 80)
    sections.append(system_prompt.rstrip() + "\n")

    mem = empty_memory(stage)
    assistant_step = 0
    for i, turn in enumerate(turns):
        role = turn.get("role")

        if role == "tool_results":
            # Apply tool results from the preceding assistant turn — this
            # advances the memory state used for the NEXT assistant turn.
            if i > 0 and turns[i - 1].get("role") == "assistant":
                prev = turns[i - 1]
                tool_calls = prev.get("tool_calls") or []
                results = turn.get("results") or []
                if len(tool_calls) == len(results):
                    for tc, r in zip(tool_calls, results):
                        handler(mem, tc.get("name"),
                                tc.get("arguments") or {}, r.get("output"))
            continue

        if role != "assistant":
            continue

        # Render the prompt the model would see RIGHT BEFORE this assistant
        # turn — based on memory state up to (but not including) this turn.
        assistant_step += 1
        memory_str = render_memory(stage, mem)
        user_text = f"## Task\n{room_prompt}\n\n## Memory\n{memory_str}"

        # Build the GT assistant emit (what the model is trained to produce).
        content = turn.get("content") or ""
        gt_tool_calls = turn.get("tool_calls") or []
        emit_parts: list[str] = []
        if content:
            emit_parts.append(f"<think>\n{content}\n</think>")
        for tc in gt_tool_calls:
            args_json = json.dumps(tc.get("arguments") or {},
                                   ensure_ascii=False)
            emit_parts.append(
                f"<tool_call>{{\"name\": \"{tc.get('name')}\", "
                f"\"arguments\": {args_json}}}</tool_call>"
            )
        gt_emit = "\n".join(emit_parts) if emit_parts else "(empty — final summary turn)"

        sections.append("\n" + "=" * 80)
        sections.append(
            f"ASSISTANT TURN {assistant_step}  (raw turn index {i}, stage={stage})"
        )
        sections.append("=" * 80)
        sections.append("\n### USER MESSAGE (text portion)")
        sections.append(user_text)
        sections.append("\n### GT ASSISTANT EMIT")
        sections.append(gt_emit)

    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[dump-prompts] {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="SFT JSON file or run directory")
    ap.add_argument("--full", action="store_true", help="also dump raw memory dict")
    ap.add_argument("--save", action="store_true",
                    help="write output to logs/memory_demo/<basename>/<stage>.memory.txt "
                         "in addition to printing")
    ap.add_argument("--prompts", action="store_true",
                    help="for each assistant turn, dump the FULL prompt that "
                         "would be fed to the model (system + ## Task + ## Memory) "
                         "plus the GT assistant emit, to "
                         "logs/memory_demo/<basename>/<stage>.prompts.txt")
    args = ap.parse_args()

    p = args.input
    save_dir = None
    if args.save or args.prompts:
        # use the run directory name (or file's parent dir name) as the bucket
        bucket = p.name if p.is_dir() else p.parent.name
        save_dir = REPO_LOGS / bucket

    if p.is_dir():
        files = [f for f in sorted(p.glob("*.json")) if "_replayed" not in f.name]
        order = ["floor_plan", "furniture", "wall", "ceiling"]
        files.sort(key=lambda f: next((i for i, k in enumerate(order) if k in f.stem), 99))
        for f in files:
            process_sft(f, full_dump=args.full,
                        save_dir=save_dir if args.save else None)
            if args.prompts:
                dump_prompts(f, save_dir)
    elif p.is_file():
        process_sft(p, full_dump=args.full,
                    save_dir=save_dir if args.save else None)
        if args.prompts:
            dump_prompts(p, save_dir)
    else:
        sys.exit(f"not found: {p}")


if __name__ == "__main__":
    main()
