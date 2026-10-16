#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import copy
import json
import traceback
from datetime import datetime
from pathlib import Path

# When run inside Docker the repo is mounted at a container-specific path; on the host resolve
# relative to this file (__file__ = <repo>/src/tool_server.py → parent = src,
# grandparent = repo root).
from src.runtime import ROOT as _REPO_ROOT, SCENESMITH_ROOT

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_rollout_decoupled import RolloutSession, ToolEnvironment
from scenesmith.utils.logging import ConsoleLogger
from OpenSceneAgent.loop import dispatch


# In Docker, <container-results> is the container-side path and HOST_RESULTS_ROOT is
# the corresponding host path (passed via -e).  On the host both sides are the
# same, so we default CONTAINER_RESULTS to the real results dir and the
# HOST_RESULTS_ROOT env var is left unset (falls back to the same value).
_DEFAULT_RESULTS = str(_REPO_ROOT / "results")
CONTAINER_RESULTS = os.environ.get("CONTAINER_RESULTS_ROOT", _DEFAULT_RESULTS)
HOST_RESULTS = os.environ.get("HOST_RESULTS_ROOT", CONTAINER_RESULTS).rstrip("/")


def to_host(p: object) -> object:
    if p is None:
        return None
    s = str(p)
    if s.startswith(CONTAINER_RESULTS):
        return HOST_RESULTS + s[len(CONTAINER_RESULTS):]
    return s


def rewrite_obs(obs: dict, steps_dir: Path) -> dict:
    obs = dict(obs)
    obs["image_path"] = to_host(obs.get("image_path"))
    obs["image_paths"] = [to_host(p) for p in obs.get("image_paths", [])]
    msgs = obs.get("messages")
    if msgs:
        new_msgs = []
        for m in msgs:
            content = m.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        new_content.append({**part,
                                            "image": to_host(part.get("image"))})
                    else:
                        new_content.append(part)
                new_msgs.append({**m, "content": new_content})
            else:
                new_msgs.append(m)
        obs["messages"] = new_msgs
    obs["steps_dir_host"] = to_host(str(steps_dir))
    return obs


class _State:
    session: RolloutSession | None = None
    env: ToolEnvironment | None = None
    logger: ConsoleLogger | None = None
    scene_dir: Path | None = None
    pending_branch: dict | None = None


_state = _State()


def _close_all() -> None:
    if _state.env is not None:
        try: _state.env.cleanup()
        except Exception: pass
    if _state.session is not None:
        try: _state.session.close()
        except Exception: pass
    _state.env = None
    _state.session = None
    _state.logger = None
    _state.scene_dir = None
    _state.pending_branch = None


def _require_session() -> RolloutSession:
    if _state.session is None:
        raise HTTPException(400, "no active session — POST /session first")
    return _state.session


def _require_env() -> ToolEnvironment:
    if _state.env is None:
        raise HTTPException(400, "no active env — POST /session/stage first")
    return _state.env


class StartReq(BaseModel):
    prompt: str
    config: str = "scene_designer"
    results_root: str = _DEFAULT_RESULTS
    stub_assets: bool = False


class StageStartReq(BaseModel):
    stage: str
    render_gpu_id: int | None = None
    skip_render: bool = False
    stub_assets: bool = False


class HandoverReq(BaseModel):
    room_id: str | None = None


class StepReq(BaseModel):
    tool_calls: list[dict]


class BranchCandidateReq(BaseModel):
    raw: str | None = None
    think: str | None = None
    tool_calls: list[dict] = Field(default_factory=list)


class BranchEvalReq(BaseModel):
    branches: list[BranchCandidateReq]


class BranchChooseReq(BaseModel):
    branch_id: str
    chosen_idx: int


app = FastAPI(title="OpenSceneAgent Tool Server")


def _model_dump(obj: BaseModel) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()


async def _execute_tool_calls_for_mem(
    e: ToolEnvironment,
    tool_calls: list[dict],
    mem: dict,
) -> list[dict]:
    """Execute tool calls against the live scene while updating the supplied mem.

    This mirrors ToolEnvironment.step(), but deliberately does not increment
    e.step_idx and lets branch evaluation update isolated memory copies.
    """
    results: list[dict] = []
    for tc in tool_calls:
        if tc.get("_parse_error"):
            results.append({"name": "_parse_error",
                            "output": "ERROR: malformed tool_call JSON"})
            continue
        name = tc.get("name")
        if not name:
            results.append({"name": "_missing_name",
                            "output": "ERROR: tool_call missing name"})
            continue
        args = tc.get("arguments") or {}
        try:
            output_str = await dispatch(e.tool_map, name, args)
            if isinstance(output_str, str) and output_str.startswith("ERROR"):
                output: object = output_str
            else:
                try:
                    output = (json.loads(output_str)
                              if isinstance(output_str, str) else output_str)
                except (json.JSONDecodeError, TypeError):
                    output = output_str
            results.append({"name": name, "output": output})
            try:
                e._handler(mem, name, args, output)
            except Exception as exc:  # noqa: BLE001
                print(f"   [mem warn] branch handler {name}: {exc}",
                      file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            results.append({
                "name": name,
                "output": f"ERROR: {exc}\n{traceback.format_exc()}",
            })
    return results


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "host_results_root": HOST_RESULTS,
        "has_session": _state.session is not None,
        "has_env": _state.env is not None,
        "stage": _state.env.stage if _state.env is not None else None,
    }


@app.post("/session")
def new_session(req: StartReq):
    if (
        os.environ.get("OPENSCENEAGENT_REQUIRE_STUB_SESSION", "")
        .strip()
        .lower()
        in {"1", "true", "yes"}
        and not req.stub_assets
    ):
        raise HTTPException(
            423,
            "server is locked to stub-assets sessions for the active batch",
        )
    _close_all()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_dir = Path(req.results_root) / f"agent_rollout_{timestamp}" / "scene_000"
    scene_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENSCENEAGENT_STUB_ASSETS"] = "1" if req.stub_assets else "0"
    sess = RolloutSession(
        scene_dir, req.config, req.prompt,
        stub_assets=req.stub_assets,
    )
    sess.start()
    _state.session = sess
    _state.scene_dir = scene_dir
    return {
        "scene_dir_container": str(scene_dir),
        "scene_dir_host":      to_host(str(scene_dir)),
    }


@app.post("/session/stage")
def start_stage(req: StageStartReq):
    s = _require_session()
    if _state.env is not None:
        try: _state.env.cleanup()
        except Exception: pass
        _state.env = None
    _state.pending_branch = None
    os.environ["OPENSCENEAGENT_STUB_ASSETS"] = "1" if req.stub_assets else "0"

    if req.stage == "floor_plan":
        env = s.make_floor_plan_env()
    elif req.stage in {"furniture", "wall", "ceiling"}:
        if s.scene is None:
            raise HTTPException(
                400, "POST /session/handover before furniture/wall/ceiling",
            )
        if _state.logger is None:
            _state.logger = ConsoleLogger(output_dir=s.scene.scene_dir)
        env = s.make_roomscene_env(
            req.stage, req.render_gpu_id, _state.logger,
            skip_render=req.skip_render,
        )
    else:
        raise HTTPException(400, f"unknown stage: {req.stage!r}")
    _state.env = env
    return {
        "stage":               env.stage,
        "steps_dir_container": str(env.steps_dir),
        "steps_dir_host":      to_host(str(env.steps_dir)),
    }


@app.post("/session/handover")
def handover(req: HandoverReq):
    s = _require_session()
    ok = s.handover_after_floor_plan(req.room_id)
    if not ok:
        raise HTTPException(400, "handover failed (no rooms or unknown room_id)")
    return {
        "room_id":             s.scene.room_id,
        "scene_dir_container": str(s.scene.scene_dir),
        "scene_dir_host":      to_host(str(s.scene.scene_dir)),
    }


@app.post("/session/observation")
def observation():
    e = _require_env()
    obs = e.observation()
    return rewrite_obs(obs, e.steps_dir)


@app.post("/session/step")
async def step(req: StepReq):
    e = _require_env()
    results = await e.step(req.tool_calls)
    return {"results": results}


@app.post("/session/branch/evaluate")
@app.post("/session/branch_eval")
async def branch_evaluate(req: BranchEvalReq):
    e = _require_env()
    if e.stage != "furniture":
        raise HTTPException(400, "branch evaluation is currently furniture-only")
    if not req.branches:
        raise HTTPException(400, "no branches supplied")
    if e.scene is None:
        raise HTTPException(400, "active env has no RoomScene")
    if not hasattr(e.scene, "to_state_dict"):
        raise HTTPException(400, "active scene does not support snapshots")

    try:
        from grpo_rollout import _flush_furniture_json, _save_branch_scene
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"cannot import GRPO scene helpers: {exc}") from exc

    scene = e.scene
    branches = [_model_dump(b) for b in req.branches]
    branch_id = f"{e.stage}_{e.step_idx:04d}_{datetime.now().strftime('%H%M%S_%f')}"
    branch_root = e.steps_dir / "branch_eval" / branch_id
    parent_dir = branch_root / "parent"
    branch_root.mkdir(parents=True, exist_ok=True)

    scene_snapshot = scene.to_state_dict()
    mem_snapshot = copy.deepcopy(e.mem)

    try:
        _flush_furniture_json(scene)
        _save_branch_scene(scene.scene_dir, parent_dir, agent=e.agent)
    except Exception as exc:  # noqa: BLE001
        print(f"[branch warn] save parent scene: {exc}", file=sys.stderr)

    branch_records: list[dict] = []
    branch_mems: list[dict] = []
    branch_states: list[dict] = []
    for idx, branch in enumerate(branches):
        scene.restore_from_state_dict(scene_snapshot)
        branch_mem = copy.deepcopy(mem_snapshot)
        tool_calls = list(branch.get("tool_calls") or [])
        results = await _execute_tool_calls_for_mem(e, tool_calls, branch_mem)

        branch_dir = branch_root / f"branch_{idx:02d}"
        try:
            _flush_furniture_json(scene)
            _save_branch_scene(scene.scene_dir, branch_dir, agent=e.agent)
        except Exception as exc:  # noqa: BLE001
            print(f"[branch warn] save branch {idx}: {exc}", file=sys.stderr)

        record = {
            "idx": idx,
            "raw": branch.get("raw"),
            "think": branch.get("think"),
            "tool_calls": tool_calls,
            "tool_results": results,
            "scene_dir_container": str(branch_dir),
            "scene_dir_host": to_host(str(branch_dir)),
        }
        branch_records.append(record)
        branch_mems.append(branch_mem)
        branch_states.append(scene.to_state_dict())

    scene.restore_from_state_dict(scene_snapshot)
    e.mem = copy.deepcopy(mem_snapshot)
    _state.pending_branch = {
        "branch_id": branch_id,
        "stage": e.stage,
        "step_idx": e.step_idx,
        "scene_snapshot": scene_snapshot,
        "mem_snapshot": mem_snapshot,
        "branches": branches,
        "branch_records": branch_records,
        "branch_mems": branch_mems,
        "branch_states": branch_states,
        "parent_scene_dir_container": str(parent_dir),
        "parent_scene_dir_host": to_host(str(parent_dir)),
    }
    return {
        "branch_id": branch_id,
        "stage": e.stage,
        "step_idx": e.step_idx,
        "parent_scene_dir_container": str(parent_dir),
        "parent_scene_dir_host": to_host(str(parent_dir)),
        "branches": branch_records,
    }


@app.post("/session/branch/choose")
@app.post("/session/branch_choose")
async def branch_choose(req: BranchChooseReq):
    e = _require_env()
    pending = _state.pending_branch
    if pending is None:
        raise HTTPException(400, "no pending branch evaluation")
    if pending.get("branch_id") != req.branch_id:
        raise HTTPException(400, "branch_id does not match pending evaluation")
    if e.stage != pending.get("stage"):
        raise HTTPException(400, "pending branch belongs to a different stage")
    if not (0 <= req.chosen_idx < len(pending["branches"])):
        raise HTTPException(400, f"chosen_idx out of range: {req.chosen_idx}")
    if e.scene is None:
        raise HTTPException(400, "active env has no RoomScene")

    try:
        from grpo_rollout import _flush_furniture_json
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"cannot import GRPO scene helper: {exc}") from exc

    e.scene.restore_from_state_dict(pending["branch_states"][req.chosen_idx])
    e.mem = copy.deepcopy(pending["branch_mems"][req.chosen_idx])
    results = pending["branch_records"][req.chosen_idx]["tool_results"]
    e.step_idx += 1
    try:
        _flush_furniture_json(e.scene)
    except Exception as exc:  # noqa: BLE001
        print(f"[branch warn] flush chosen scene: {exc}", file=sys.stderr)

    response = {
        "branch_id": req.branch_id,
        "chosen_idx": req.chosen_idx,
        "step_idx": pending["step_idx"],
        "tool_results": results,
        "chosen_branch": pending["branch_records"][req.chosen_idx],
    }
    _state.pending_branch = None
    return response


@app.post("/session/finalize")
def finalize():
    e = _require_env()
    e.finalize_stage()
    if e.stage == "floor_plan":
        return {"stage": "floor_plan", "ok": True}
    blend_path = e.scene.scene_dir / f"final_{e.stage}" / f"{e.stage}.blend"
    return {
        "stage":            e.stage,
        "blend_path_host":  to_host(str(blend_path)),
    }


@app.post("/session/cleanup")
def cleanup_env():
    if _state.env is not None:
        try: _state.env.cleanup()
        except Exception: pass
        _state.env = None
    _state.pending_branch = None
    return {"ok": True}


@app.post("/session/close")
def close_session():
    if (
        os.environ.get("OPENSCENEAGENT_REQUIRE_STUB_SESSION", "")
        .strip()
        .lower()
        in {"1", "true", "yes"}
        and _state.session is not None
    ):
        return {"ok": True, "locked": True, "ignored": True}
    _close_all()
    return {"ok": True}


@app.post("/admin/exit")
def admin_exit():
    """Close any active session, then exit the process so a supervisor can
    respawn a clean uvicorn. Use this between independent test runs to avoid
    leaked scenesmith state (ports/CUDA worker pools/background threads)."""
    _close_all()
    import os as _os, threading as _t, time as _time
    def _delayed_exit() -> None:
        _time.sleep(0.5)  # let the 200 response flush back to the client
        _os._exit(0)
    _t.Thread(target=_delayed_exit, daemon=True).start()
    return {"ok": True, "msg": "server exiting; supervisor should respawn"}
