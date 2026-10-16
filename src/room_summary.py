"""Shared room-summary block builder.

Both offline tooling (tools/inject_room_summary_to_sft.py) and the runtime
inference path (Designer/agent_rollout.py via agent_rollout_decoupled.py)
produce the SAME compact text block describing the room created in the
floor_plan stage. Training and inference must see byte-for-byte identical
summaries so the model's input distribution stays the same.

Block format example:
    ## ROOM SUMMARY (from floor_plan stage):
    - Room: bathroom, 2.4m wide × 2m deep, wall_height ~2.7m (default).
    - Wall A (north): 2.4m long; no openings.
    - Wall B (south): 2.4m long; window pos=center, 0.6m × 1m, sill 1.6m.
    - Wall C (east):  2m long; no openings.
    - Wall D (west):  2m long; door pos=center, 0.9m × 2.1m.
    - World coords centered on room: x∈[-1.2, +1.2], y∈[-1, +1].
"""
from __future__ import annotations


MARKER = "## ROOM SUMMARY (from floor_plan stage):"

WALL_LABEL_TO_DIR = {"A": "north", "B": "south", "C": "east", "D": "west"}


def _fmt_num(v) -> str:
    if isinstance(v, (int, float)):
        s = f"{v:.2f}"
        return s.rstrip("0").rstrip(".") if "." in s else s
    return str(v)


def _wall_length(label: str, room_w, room_d) -> float | None:
    """scenesmith convention: A/B span room width (E-W), C/D span room depth (N-S)."""
    if not isinstance(room_w, (int, float)) or not isinstance(room_d, (int, float)):
        return None
    if label in ("A", "B"): return room_w
    if label in ("C", "D"): return room_d
    return None


def format_room_summary(room: dict | None,
                        wall_height: float | None,
                        doors: list[dict],
                        windows: list[dict]) -> str | None:
    """Format the room summary block.

    Args:
        room: dict with keys {type, w, d}. None → return None (no block).
        wall_height: float in meters, or None for default annotation.
        doors: list of {wall, pos, w, h}.
        windows: list of {wall, pos, w, h, sill}.
    """
    if room is None or room.get("w") is None or room.get("d") is None:
        return None
    w, d = room["w"], room["d"]
    # We deliberately omit wall_height: scenesmith varies it (2.6–3.0m) per
    # scene but the SFT trajectory may not record the actual value (the
    # set_wall_height tool is often skipped, defaulting at runtime).
    # Including a "wall_height Xm" line would diverge between offline and
    # runtime; the model can reason about z-positions without it.
    lines = [MARKER]
    lines.append(f"- Room: {room.get('type', '?')}, "
                 f"{_fmt_num(w)}m wide × {_fmt_num(d)}m deep.")
    doors_by_wall: dict[str, list[dict]] = {}
    wins_by_wall:  dict[str, list[dict]] = {}
    for o in doors: doors_by_wall.setdefault(o["wall"], []).append(o)
    for o in windows: wins_by_wall.setdefault(o["wall"], []).append(o)
    for label in ("A", "B", "C", "D"):
        direction = WALL_LABEL_TO_DIR[label]
        L = _wall_length(label, w, d)
        L_text = f"{_fmt_num(L)}m long" if L is not None else "length unknown"
        opening_parts = []
        for door in doors_by_wall.get(label, []):
            # Doors carry an authoritative segment label on both sides
            # (offline: tool_call.position; runtime: Door.position_segment),
            # so we include it.
            p = door.get("pos")
            p_text = f"pos={p}" if p else "pos unspecified"
            opening_parts.append(
                f"door {p_text}, {_fmt_num(door['w'])}m × {_fmt_num(door['h'])}m")
        for win in wins_by_wall.get(label, []):
            # Windows: scenesmith only persists the randomised
            # position_along_wall (no segment label), so we deliberately omit
            # position here to keep offline & runtime byte-aligned. Size +
            # sill are enough for the model's wall-object reasoning.
            opening_parts.append(
                f"window {_fmt_num(win['w'])}m × {_fmt_num(win['h'])}m, "
                f"sill {_fmt_num(win['sill'])}m")
        openings = "; ".join(opening_parts) if opening_parts else "no openings"
        lines.append(f"- Wall {label} ({direction}): {L_text}; {openings}.")
    hw, hd = w / 2, d / 2
    lines.append(f"- World coords centered on room: "
                 f"x∈[-{_fmt_num(hw)}, +{_fmt_num(hw)}], "
                 f"y∈[-{_fmt_num(hd)}, +{_fmt_num(hd)}]. "
                 f"(Width spans the north/south walls; "
                 f"depth spans the east/west walls.)")
    return "\n".join(lines)


# ===================================================================
# Adapters from the two upstream data shapes
# ===================================================================


def from_floor_plan_sft(sft: dict) -> str | None:
    """Build a ROOM SUMMARY block from a parsed floor_plan.json (SFT data).

    Walks the trajectory's assistant tool_calls for
        generate_room_specs / set_wall_height / add_door / add_window
    and only includes add_door / add_window that succeeded (per
    their immediately-following tool_results)."""
    import json
    room = None
    wall_height = None
    accepted_doors:   list[dict] = []
    accepted_windows: list[dict] = []
    turns = sft.get("turns") or []
    for i, t in enumerate(turns):
        if t.get("role") != "assistant": continue
        calls = t.get("tool_calls") or []
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        results = (nxt.get("results") or []) if (nxt and nxt.get("role") == "tool_results") else []
        for j, tc in enumerate(calls):
            n = tc.get("name") or ""
            a = tc.get("arguments") or {}
            if n == "generate_room_specs":
                try: specs = json.loads(a.get("room_specs_json") or "[]")
                except Exception: specs = []
                for s in specs:
                    if isinstance(s, dict) and room is None:
                        room = {"type": s.get("type"), "w": s.get("width"), "d": s.get("depth")}
            elif n == "set_wall_height":
                wall_height = a.get("height_meters") or a.get("height")
            elif n in ("add_door", "add_window"):
                res = results[j].get("output") if j < len(results) else None
                ok = False
                if isinstance(res, dict):
                    ok = bool(res.get("success"))
                elif isinstance(res, str):
                    try:
                        o = json.loads(res)
                        ok = bool(o.get("success")) if isinstance(o, dict) else False
                    except Exception:
                        low = res.lower()
                        ok = ("success=true" in low.replace(" ", "")
                              or "successfully added door" in low
                              or "successfully added window" in low)
                if not ok: continue
                entry = {
                    "wall": a.get("wall_id"),
                    "pos":  a.get("position"),
                    "w":    a.get("width", 0.9 if n == "add_door" else 1.2),
                    "h":    a.get("height", 2.1 if n == "add_door" else 1.2),
                }
                if n == "add_window":
                    entry["sill"] = a.get("sill_height", 0.9)
                    accepted_windows.append(entry)
                else:
                    accepted_doors.append(entry)
    return format_room_summary(room, wall_height, accepted_doors, accepted_windows)


def from_house_layout_dict(layout: dict, room_id: str | None = None) -> str | None:
    """Build a ROOM SUMMARY block from a HouseLayout.to_dict() result.

    Used at INFERENCE time: agent_rollout_decoupled.py calls this on the
    floor_plan agent's layout, after handover.

    IMPORTANT: we read from `placed_rooms[]`, NOT `rooms[]`. The two carry
    DIFFERENT field semantics in scenesmith:
      - `rooms[]` is the original RoomSpec where "width"/"length" can be
        re-keyed at output time (we observed width/length swapped vs the
        user's input width/depth).
      - `placed_rooms[]` is the materialised PlacedRoom with `width`+`depth`
        matching the SFT generate_room_specs args, plus an authoritative
        `walls[]` array each having `direction` + `length` + `openings[]`.
    This keeps the runtime adapter byte-for-byte aligned with the offline
    adapter (which reads `width`/`depth` straight from tool_calls).

    Args:
        layout: dict from HouseLayout.to_dict().
        room_id: which room to summarise (multi-room houses). If None,
                 picks the first placed_room.
    """
    placed = layout.get("placed_rooms") or []
    chosen = None
    for r in placed:
        if isinstance(r, dict) and (room_id is None or r.get("room_id") == room_id):
            chosen = r
            break
    if chosen is None and placed:
        chosen = placed[0] if isinstance(placed[0], dict) else None
    if chosen is None:
        return None

    room = {
        # placed_room dicts don't repeat the room_type; pull it from the
        # matching rooms[] entry by id.
        "type": "room",
        "w":    chosen.get("width"),
        "d":    chosen.get("depth"),
    }
    rid = chosen.get("room_id")
    for r in (layout.get("rooms") or []):
        if isinstance(r, dict) and (r.get("id") == rid or r.get("room_id") == rid):
            room["type"] = r.get("type") or r.get("room_type") or "room"
            break

    wall_height = layout.get("wall_height")

    # Map north/south/east/west back to scenesmith's A/B/C/D label so the
    # output format ("Wall A (north): ...") matches the offline adapter's.
    DIR_TO_LABEL = {v: k for k, v in WALL_LABEL_TO_DIR.items()}

    # Source openings from the TOP-LEVEL doors[] and windows[] (they carry
    # `position_segment` = the original left/center/right segment chosen by
    # the floor_plan agent, matching what the SFT add_door / add_window call
    # had as `position`). Filter to this room when room_id is known via
    # `room_a` / `room_id` fields.
    def _belongs_to_room(o: dict) -> bool:
        rid_ = chosen.get("room_id")
        if rid_ is None:
            return True
        return (o.get("room_a") == rid_ or o.get("room_b") == rid_
                or o.get("room_id") == rid_)

    doors: list[dict] = []
    for d in (layout.get("doors") or []):
        if not isinstance(d, dict): continue
        if not _belongs_to_room(d): continue
        doors.append({
            "wall": d.get("boundary_label", "?"),
            "pos":  d.get("position_segment", "center"),
            "w":    d.get("width", 0.9),
            "h":    d.get("height", 2.1),
        })
    windows: list[dict] = []
    for wnd in (layout.get("windows") or []):
        if not isinstance(wnd, dict): continue
        if not _belongs_to_room(wnd): continue
        windows.append({
            "wall": wnd.get("boundary_label", "?"),
            "pos":  wnd.get("position_segment", "center"),
            "w":    wnd.get("width", 1.2),
            "h":    wnd.get("height", 1.2),
            "sill": wnd.get("sill_height", 0.9),
        })

    return format_room_summary(room, wall_height, doors, windows)
