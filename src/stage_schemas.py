"""Per-stage tool schemas for the SFT pipeline.

Each STAGE_SCHEMAS[<stage>] maps tool_name → spec dict with fields:
    required: {field_name: type_or_tuple}
    optional: {field_name: type_or_tuple}    (type accepts a default)
    enums:    {field_name: set_of_allowed_values}     (optional)
    ranges:   {field_name: (lo, hi)}                  (optional, inclusive)
    checks:   "<name>"                                 (optional, custom validator key)

The validator in make_episode.py reads these to reject DeepSeek SFT that
deviates from scenesmith's real tool signatures.
"""
from __future__ import annotations

# ------------------------------------------------------------------ floor_plan

FLOOR_PLAN_SCHEMAS = {
    "generate_room_specs": {
        "required": {"room_specs_json": str},
        "optional": {},
        "checks": "__generate_room_specs",
    },
    "resize_room": {
        "required": {"room_id": str, "width": (int, float), "depth": (int, float)},
        "optional": {},
        "ranges":   {"width": (1.5, 20.0), "depth": (1.5, 20.0)},
    },
    "add_adjacency":          {"required": {"room_a": str, "room_b": str}, "optional": {}},
    "remove_adjacency":       {"required": {"room_a": str, "room_b": str}, "optional": {}},
    "add_open_connection":    {"required": {"room_a": str, "room_b": str}, "optional": {}},
    "remove_open_connection": {"required": {"room_a": str, "room_b": str}, "optional": {}},
    "set_wall_height": {
        "required": {"height_meters": (int, float)}, "optional": {},
        "ranges":   {"height_meters": (2.0, 4.5)},
    },
    "add_door": {
        "required": {"wall_id": str, "position": str},
        "optional": {"width": (int, float), "height": (int, float)},
        "enums":    {"position": {"left", "center", "right"}},
        # scenesmith allows narrower bathroom doors than the ergonomic default;
        # keep range wide enough to cover what agents actually emit in raw.
        "ranges":   {"width": (0.6, 1.9), "height": (1.8, 2.4)},
    },
    "remove_door":   {"required": {"door_id": str}, "optional": {}},
    "add_window": {
        "required": {"wall_id": str, "position": str},
        "optional": {"width": (int, float), "height": (int, float), "sill_height": (int, float)},
        "enums":    {"position": {"left", "center", "right"}},
        "ranges":   {"width": (0.6, 3.0), "height": (0.6, 2.0)},
    },
    "remove_window": {"required": {"window_id": str}, "optional": {}},
    "get_material":  {"required": {"description": str}, "optional": {}},
    "set_room_materials": {
        "required": {
            "room_id": str,
            "floor_material_id": str,
            "wall_material_id": str,
        },
        "optional": {},
    },
    "set_exterior_material": {"required": {"material_id": str}, "optional": {}},
    "check_physics":         {"required": {}, "optional": {}},
}

# ------------------------------------------------------------------- furniture

FURNITURE_SCHEMAS = {
    "generate_assets": {
        "required": {"object_descriptions": list, "short_names": list,
                     "desired_dimensions": list},
        "optional": {"style_context": str},
    },
    "add_furniture_to_scene_tool": {
        "required": {"asset_id": str, "x": (int, float), "y": (int, float)},
        "optional": {"yaw": (int, float)},
    },
    "snap_to_object_tool": {
        "required": {"object_id": str, "target_id": str},
        "optional": {"orientation": str},
        "enums":    {"orientation": {"toward", "away", "none"}},
    },
    "move_furniture_tool": {
        "required": {"object_id": str, "x": (int, float), "y": (int, float)},
        "optional": {"yaw": (int, float)},
    },
    "remove_furniture_tool": {
        "required": {"object_id": str}, "optional": {},
    },
    "rescale_furniture_tool": {
        "required": {"object_id": str, "scale_factor": (int, float)}, "optional": {},
    },
    "check_physics":          {"required": {}, "optional": {}},
}

# ------------------------------------------------------------------------ wall

WALL_SCHEMAS = {
    "generate_wall_assets": {
        "required": {"object_descriptions": list, "short_names": list,
                     "desired_dimensions": list},
        "optional": {"style_context": str},
    },
    "place_wall_object": {
        "required": {"asset_id": str, "wall_surface_id": str,
                     "position_x": (int, float), "position_z": (int, float)},
        "optional": {"rotation_degrees": (int, float)},
    },
    "move_wall_object": {
        "required": {"object_id": str, "wall_surface_id": str,
                     "position_x": (int, float), "position_z": (int, float)},
        "optional": {"rotation_degrees": (int, float)},
    },
    "remove_wall_object": {
        "required": {"object_id": str}, "optional": {},
    },
    "rescale_wall_object": {
        "required": {"object_id": str, "scale_factor": (int, float)}, "optional": {},
    },
    "check_physics":     {"required": {}, "optional": {}},
}

# --------------------------------------------------------------------- ceiling

CEILING_SCHEMAS = {
    "generate_ceiling_assets": {
        "required": {"object_descriptions": list, "short_names": list,
                     "desired_dimensions": list},
        "optional": {"style_context": str},
    },
    "place_ceiling_object": {
        "required": {"asset_id": str, "position_x": (int, float), "position_y": (int, float)},
        "optional": {"rotation_degrees": (int, float)},
    },
    "move_ceiling_object": {
        "required": {"object_id": str, "position_x": (int, float), "position_y": (int, float)},
        "optional": {"rotation_degrees": (int, float)},
    },
    "remove_ceiling_object": {
        "required": {"object_id": str}, "optional": {},
    },
    "rescale_ceiling_object": {
        "required": {"object_id": str, "scale_factor": (int, float)}, "optional": {},
    },
    "check_physics":          {"required": {}, "optional": {}},
}

# ---------------------------------------------------------------------- index

STAGE_SCHEMAS: dict[str, dict] = {
    "floor_plan": FLOOR_PLAN_SCHEMAS,
    "furniture":  FURNITURE_SCHEMAS,
    "wall":       WALL_SCHEMAS,
    "ceiling":    CEILING_SCHEMAS,
}

# Tools dropped from every compressed SFT trajectory regardless of stage.
# NOTE: check_physics is NOT in this list — it's kept (sparsely) to preserve
# agent error-detection and self-correcting behavior in the training data.
DROPPED_TOOLS = {
    "observe_scene", "render_ascii", "validate",
    "list_room_materials", "list_available_assets", "list_wall_surfaces",
    "list_support_surfaces", "get_current_scene_state",
    "check_reachability",
    "designer_todo_manager",
    # framework transitions
    "request_initial_design", "request_critique", "request_design_change",
}


def validate_tool_call(stage: str, name: str, arguments: dict) -> list[str]:
    """Validate one tool_call's args against STAGE_SCHEMAS[stage]. Returns errors."""
    import json as _json
    errs: list[str] = []
    schemas = STAGE_SCHEMAS.get(stage)
    if schemas is None:
        errs.append(f"unknown stage {stage!r}; allowed: {sorted(STAGE_SCHEMAS)}")
        return errs
    if name not in schemas:
        errs.append(f"unknown tool {name!r} for stage {stage!r}; allowed: {sorted(schemas)}")
        return errs
    if not isinstance(arguments, dict):
        errs.append(f"{name}: arguments must be an object, got {type(arguments).__name__}")
        return errs

    spec = schemas[name]
    req = spec.get("required", {})
    opt = spec.get("optional", {})
    all_fields = {**req, **opt}

    for f, typ in req.items():
        if f not in arguments:
            errs.append(f"{name}: missing required field {f!r}")

    for f in arguments:
        if f not in all_fields:
            close = [x for x in all_fields if x.lower().replace("_", "") == f.lower().replace("_", "")]
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            errs.append(f"{name}: unknown field {f!r}{hint}")

    for f, val in arguments.items():
        if f not in all_fields:
            continue
        typ = all_fields[f]
        if isinstance(typ, tuple):
            if not isinstance(val, typ):
                errs.append(f"{name}.{f}: expected {typ}, got {type(val).__name__} = {val!r}")
        elif not isinstance(val, typ):
            errs.append(f"{name}.{f}: expected {typ.__name__}, got {type(val).__name__} = {val!r}")

    for f, allowed in spec.get("enums", {}).items():
        if f in arguments and arguments[f] not in allowed:
            errs.append(f"{name}.{f}: {arguments[f]!r} not in enum {sorted(allowed)}")

    for f, (lo, hi) in spec.get("ranges", {}).items():
        v = arguments.get(f)
        if isinstance(v, (int, float)) and not (lo <= v <= hi):
            errs.append(f"{name}.{f}: {v} out of range [{lo}, {hi}]")

    # Custom checks
    if spec.get("checks") == "__generate_room_specs" and "room_specs_json" in arguments:
        rj = arguments["room_specs_json"]
        if not isinstance(rj, str):
            errs.append(f"{name}.room_specs_json: must be a STRING containing a JSON array")
        else:
            try:
                parsed = _json.loads(rj)
            except _json.JSONDecodeError as e:
                errs.append(f"{name}.room_specs_json: invalid JSON: {e}")
                parsed = None
            if parsed is not None:
                if not isinstance(parsed, list):
                    errs.append(f"{name}.room_specs_json: must be a JSON ARRAY, got {type(parsed).__name__}")
                else:
                    # `prompt` is injected by scenesmith from the user message at
                    # tool-call dispatch time, so raw tool_call.arguments may or
                    # may not carry it — don't require it here.
                    ROOM_REQUIRED = {"type"}
                    ROOM_ALLOWED = ROOM_REQUIRED | {
                        "prompt", "width", "depth", "connections", "exterior_walls",
                    }
                    for i, it in enumerate(parsed):
                        if not isinstance(it, dict):
                            errs.append(f"{name}.room_specs_json[{i}]: must be an object")
                            continue
                        for f in ROOM_REQUIRED:
                            if f not in it:
                                alt = "room_type" if f == "type" and "room_type" in it else None
                                hint = f" (found {alt!r})" if alt else ""
                                errs.append(f"{name}.room_specs_json[{i}]: missing {f!r}{hint}")
                        for f in it:
                            if f not in ROOM_ALLOWED:
                                alt = "type" if f == "room_type" else None
                                hint = f" (did you mean {alt!r}?)" if alt else ""
                                errs.append(f"{name}.room_specs_json[{i}]: unknown field {f!r}{hint}")

    return errs


def material_id_producing_tools(stage: str) -> set[str]:
    """Tools whose synthesized output.material_id can satisfy a set_*_material arg."""
    if stage == "floor_plan":
        return {"get_material"}
    return set()


def material_id_consuming_fields(stage: str) -> dict[str, list[str]]:
    """For each tool in stage, list arg fields that must be traceable material_ids."""
    if stage == "floor_plan":
        return {
            "set_room_materials": ["wall_material_id", "floor_material_id"],
            "set_exterior_material": ["material_id"],
        }
    return {}


def asset_id_producing_tools(stage: str) -> set[str]:
    """Tools whose synthesized output.assets[].object_id can satisfy an asset_id arg."""
    return {
        "floor_plan": set(),
        "furniture":  {"generate_assets"},
        "wall":       {"generate_wall_assets"},
        "ceiling":    {"generate_ceiling_assets"},
    }.get(stage, set())


def asset_id_consuming_fields(stage: str) -> dict[str, list[str]]:
    """Tools + fields whose values must be object_ids produced by a prior generate_* call."""
    return {
        "floor_plan": {},
        "furniture":  {"add_furniture_to_scene_tool": ["asset_id"]},
        "wall":       {"place_wall_object": ["asset_id"]},
        "ceiling":    {"place_ceiling_object": ["asset_id"]},
    }.get(stage, {})
