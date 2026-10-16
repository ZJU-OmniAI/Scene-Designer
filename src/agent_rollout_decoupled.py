#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from src.runtime import ROOT, SCENESMITH_ROOT

import torch

from scenesmith.llm_provider import configure as _configure_llm_gateway
_configure_llm_gateway()

from scenesmith.agent_utils.room import RoomScene
from scenesmith.utils.logging import ConsoleLogger
from scenesmith.experiments.base_experiment import BaseExperiment

from OpenSceneAgent.loop import (
    build_agent_and_tools as build_floor_plan_agent_tools,
    dispatch,
)
from OpenSceneAgent.replay_chained import (
    BackendServers,
    COMPATIBLE_CEILING_AGENTS,
    COMPATIBLE_FURNITURE_AGENTS,
    COMPATIBLE_WALL_AGENTS,
    _save_roomscene_blend,
    compose_full_cfg,
    derive_room_scene,
)

from OpenSceneAgent.agent_rollout import (
    DEFAULT_MODEL_DIR,
    build_messages,
    model_generate,
    parse_assistant_text,
    render_system,
    render_topdown,
    write_run_metrics,
)

import memory as _mem

from transformers import AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )


class ToolEnvironment:
    def __init__(
        self,
        *,
        stage: str,
        room_prompt: str,
        agent,
        scene,
        tool_map: dict,
        steps_dir: Path,
        room_summary: str | None = None,
        skip_render: bool = False,
    ):
        self.stage = stage
        self.room_prompt = room_prompt
        self.agent = agent
        self.scene = scene
        self.tool_map = tool_map
        self.steps_dir = Path(steps_dir)
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.room_summary = room_summary
        self.skip_render = skip_render

        self.system = render_system(stage, room_prompt, room_summary)
        self.mem = _mem.empty_memory(stage)
        self._handler = _mem.HANDLERS[stage]
        self.step_idx = 0

    def observation(self) -> dict:
        from src.render_observation import render_scene_views
        if self.stage != "floor_plan" and self.skip_render:
            raise ValueError('Five-view scene inference cannot skip observation rendering')
        image_paths = [] if self.stage == "floor_plan" else render_scene_views(
            self.agent, self.scene, self.steps_dir, self.step_idx)
        image_path = image_paths[0] if image_paths else None
        memory_str = _mem.render_memory(self.stage, self.mem)
        return {
            "stage":       self.stage,
            "step_idx":    self.step_idx,
            "system":      self.system,
            "room_prompt": self.room_prompt,
            "memory_str":  memory_str,
            "image_path":  str(image_path) if image_path else None,
            "image_paths": image_paths,
            "messages":    build_messages(self.system, self.room_prompt,
                                          memory_str, image_paths),
        }

    async def step(self, tool_calls: list[dict]) -> list[dict]:
        results: list[dict] = []
        for tc in tool_calls:
            if tc.get("_parse_error"):
                results.append({"name": "_parse_error",
                                "output": "ERROR: malformed tool_call JSON"})
                continue
            name = tc["name"]
            args = tc.get("arguments") or {}
            try:
                output_str = await dispatch(self.tool_map, name, args)
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
                    self._handler(self.mem, name, args, output)
                except Exception as e:
                    print(f"   [mem warn] handler {name}: {e}", file=sys.stderr)
            except Exception as e:
                results.append({"name": name,
                                "output": f"ERROR: {e}\n{traceback.format_exc()}"})

        self.step_idx += 1
        return results

    def finalize_stage(self) -> dict:
        if self.stage == "floor_plan":
            floor_plans_dir = self.agent.layout.house_dir / "floor_plans"
            floor_plans_dir.mkdir(parents=True, exist_ok=True)
            self.agent._generate_all_room_geometries(output_dir=floor_plans_dir)
            # Do not call _export_floor_plan() here. That path invokes Drake's
            # RenderEngineGltfClient / vtkEGLRenderWindow and can segfault on
            # this headless server. The downstream room stages only require
            # generated room_geometries plus house_layout.json.
            hl_path = self.agent.layout.house_dir / "house_layout.json"
            hl_path.write_text(json.dumps(self.agent.layout.to_dict(), indent=2),
                               encoding="utf-8")
            print("[floor_plan] geometries + house_layout.json written",
                  file=sys.stderr)
            return {"layout": self.agent.layout}

        # ``skip_render`` only controls the per-step observation render.  It
        # must not suppress stage checkpoints: downstream exporters and
        # recovery tools need the furniture JSON and a materialized .blend
        # even when rollout observations are text-only.
        if self.stage == "furniture":
            from grpo_rollout import _flush_furniture_json

            _flush_furniture_json(self.scene)
            print("[final] persisted furniture state", file=sys.stderr)

        # Per-step observation rendering and in-server Blender export share
        # the same headless EGL path.  With skip_render enabled, materialize
        # stage blends in the independent bpy checkpoint exporter instead of
        # risking a VTK/EGL segfault in the long-lived tool server.
        if self.skip_render:
            print(f"[final] deferred {self.stage} blend to stage exporter",
                  file=sys.stderr)
            return {"scene": self.scene, "blend_deferred": True}

        final_path = self.scene.scene_dir / f"final_{self.stage}" / f"{self.stage}.blend"
        try:
            _save_roomscene_blend(self.agent, final_path)
            print(f"[final] {final_path}", file=sys.stderr)
        except Exception as e:
            print(f"[final warn] {e}", file=sys.stderr)
            raise
        return {"scene": self.scene, "blend_path": str(final_path)}

    def cleanup(self) -> None:
        try:
            self.agent.cleanup()
        except Exception:
            pass


class ModelPolicy:
    def __init__(self, model_dir: Path, max_new_tokens: int = 8192):
        print(f"[init] loading model from {model_dir}", file=sys.stderr)
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        from src.scene_observation import configure_image_budget
        configure_image_budget(self.processor)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.model.eval()
        self.model.config.use_cache = True
        self.max_new_tokens = max_new_tokens
        n_b = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[init] model on cuda ({n_b:.2f}B params)", file=sys.stderr)

    def generate(self, obs: dict) -> dict:
        img = Path(obs["image_path"]) if obs.get("image_path") else None
        raw = model_generate(self.model, self.processor, obs["messages"],
                             img, self.max_new_tokens)
        think, tool_calls = parse_assistant_text(raw)
        return {"raw": raw, "think": think, "tool_calls": tool_calls}


class RolloutSession:
    def __init__(
        self,
        scene_dir: Path,
        config: str,
        room_prompt: str,
        *,
        stub_assets: bool = False,
    ):
        self.scene_dir = scene_dir
        self.config = config
        self.room_prompt = room_prompt
        self.stub_assets = stub_assets
        self.cfg = None
        self.cfg_dict = None
        self.servers: BackendServers | None = None
        self.fp_agent = None
        self.fp_tool_map: dict | None = None
        self.fp_logger = None
        self.fp_mat_server = None
        self.scene: RoomScene | None = None
        self.house_layout = None
        self.room_geometry = None
        # Cached ROOM SUMMARY block (computed at handover, fed to all
        # downstream stage agents' system prompts).
        self.room_summary: str | None = None

    def start(self) -> None:
        self.cfg, self.cfg_dict = compose_full_cfg(self.config, self.scene_dir)
        self.servers = BackendServers(
            self.cfg, self.scene_dir,
            skip_geometry=self.stub_assets,
        )
        self.servers.start_all()
        (self.fp_agent, self.fp_tool_map,
         self.fp_logger, self.fp_mat_server) = build_floor_plan_agent_tools(
            config_name=self.config,
            house_prompt=self.room_prompt,
            scene_dir=self.scene_dir,
        )

    def make_floor_plan_env(self) -> ToolEnvironment:
        return ToolEnvironment(
            stage="floor_plan",
            room_prompt=self.room_prompt,
            agent=self.fp_agent,
            scene=None,
            tool_map=self.fp_tool_map,
            steps_dir=self.scene_dir / "rollout_floor_plan",
        )

    def handover_after_floor_plan(self, room_id: str | None) -> bool:
        layout = self.fp_agent.layout
        if not layout.placed_rooms:
            print("[floor_plan] no rooms in layout; nothing downstream",
                  file=sys.stderr)
            return False
        chosen = room_id or layout.placed_rooms[0].room_id
        if chosen not in layout.room_geometries:
            print(f"[chained] requested room {chosen!r} not generated; "
                  f"have {sorted(layout.room_geometries)}", file=sys.stderr)
            return False
        print(f"[chained] proceeding with room {chosen!r}", file=sys.stderr)
        self.scene = derive_room_scene(self.fp_agent, chosen)
        self.room_geometry = layout.room_geometries[chosen]
        self.house_layout = layout

        # Compute the ROOM SUMMARY block once here — wall/furniture/ceiling
        # envs will all receive the same one. SAFE because the model would
        # see exactly the same block at train time (the SFT-side patcher
        # tools/inject_room_summary_to_sft.py uses the same formatter).
        try:
            from room_summary import from_house_layout_dict
            self.room_summary = from_house_layout_dict(layout.to_dict(), chosen)
            if self.room_summary:
                print(f"[handover] room_summary block: "
                      f"{len(self.room_summary)} chars", file=sys.stderr)
            else:
                print(f"[handover] room_summary not buildable (missing dims?)",
                      file=sys.stderr)
        except Exception as e:
            print(f"[handover] room_summary build failed: {e}", file=sys.stderr)
            self.room_summary = None
        return True

    def make_roomscene_env(self, stage: str, render_gpu_id: int | None,
                           logger, skip_render: bool = False) -> ToolEnvironment:
        if stage == "furniture":
            agent = BaseExperiment.build_furniture_agent(
                cfg_dict=self.cfg_dict,
                compatible_agents=COMPATIBLE_FURNITURE_AGENTS,
                logger=logger, render_gpu_id=render_gpu_id,
            )
        elif stage == "wall":
            agent = BaseExperiment.build_wall_agent(
                cfg_dict=self.cfg_dict,
                compatible_agents=COMPATIBLE_WALL_AGENTS,
                logger=logger,
                house_layout=self.house_layout,
                ceiling_height=self.room_geometry.wall_height,
                wall_thickness=self.room_geometry.wall_thickness,
                render_gpu_id=render_gpu_id,
            )
        elif stage == "ceiling":
            agent = BaseExperiment.build_ceiling_agent(
                cfg_dict=self.cfg_dict,
                compatible_agents=COMPATIBLE_CEILING_AGENTS,
                logger=logger,
                ceiling_height=self.room_geometry.wall_height,
                render_gpu_id=render_gpu_id,
            )
        else:
            raise ValueError(f"unsupported stage: {stage}")

        agent.scene = self.scene
        if stage == "wall":
            agent.wall_surfaces = agent._extract_wall_surfaces(
                room_id=self.scene.room_id,
            )
            if not agent.wall_surfaces:
                raise RuntimeError(
                    f"no wall surfaces for room {self.scene.room_id!r}"
                )
            designer_tools = agent._create_designer_tools(
                wall_surfaces=agent.wall_surfaces,
            )
        elif stage == "ceiling":
            agent.room_bounds = agent._extract_room_bounds(scene=self.scene)
            designer_tools = agent._create_designer_tools()
        else:
            designer_tools = agent._create_designer_tools()

        tool_map = {ft.name: ft for ft in designer_tools}
        print(f"[{stage}] agent ready ({len(tool_map)} tools)", file=sys.stderr)
        return ToolEnvironment(
            stage=stage,
            room_prompt=self.room_prompt,
            agent=agent,
            scene=self.scene,
            tool_map=tool_map,
            steps_dir=self.scene.scene_dir / f"rollout_{stage}",
            room_summary=self.room_summary,
            skip_render=skip_render,
        )

    def close(self) -> None:
        if self.servers is not None:
            try: self.servers.stop_all()
            except Exception: pass
        if self.fp_agent is not None:
            try: self.fp_agent.cleanup()
            except Exception: pass
        if self.fp_mat_server is not None:
            try: self.fp_mat_server.stop()
            except Exception: pass


async def driver_loop(
    *, env: ToolEnvironment, policy: ModelPolicy, max_steps: int,
) -> list[dict]:
    rollout_log: list[dict] = []
    io_log: list[dict] = []

    for _ in range(max_steps):
        obs = env.observation()
        step_idx = obs["step_idx"]

        print(f"\n[{env.stage}][step {step_idx}] generating...", file=sys.stderr)
        out = policy.generate(obs)
        raw, think, tool_calls = out["raw"], out["think"], out["tool_calls"]

        think_preview = (think or "(no <think>)").replace("\n", " ")[:140]
        print(f"[{env.stage}][step {step_idx}] think: {think_preview}",
              file=sys.stderr)

        io_entry: dict = {
            "turn": step_idx,
            "input": {
                "task":       env.room_prompt,
                "memory":     obs["memory_str"],
                "image_path": obs["image_path"],
            },
            "output": {"raw": raw, "think": think, "tool_calls": tool_calls},
            "tool_results": [],
        }

        if not tool_calls:
            print(f"[{env.stage}] step {step_idx}: no <tool_call> — stage done",
                  file=sys.stderr)
            rollout_log.append({"step": step_idx, "think": think,
                                "tool_calls": [], "tool_results": []})
            io_entry["stop_reason"] = "no_tool_call"
            io_log.append(io_entry)
            break

        for j, tc in enumerate(tool_calls):
            args_str = json.dumps(tc.get("arguments") or {}, ensure_ascii=False)
            preview = args_str if len(args_str) <= 120 else args_str[:117] + "..."
            print(f"[{env.stage}][step {step_idx}][call {j}] "
                  f"{tc.get('name')}({preview})", file=sys.stderr)

        results = await env.step(tool_calls)
        rollout_log.append({"step": step_idx, "think": think,
                            "tool_calls": tool_calls, "tool_results": results})
        io_entry["tool_results"] = results
        io_log.append(io_entry)
    else:
        print(f"[{env.stage}] hit max_steps={max_steps}, stopping",
              file=sys.stderr)

    (env.steps_dir / "rollout.json").write_text(
        json.dumps(rollout_log, indent=2, ensure_ascii=False), encoding="utf-8")
    (env.steps_dir / "turns_io.json").write_text(
        json.dumps(io_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{env.stage}] saved logs → {env.steps_dir}", file=sys.stderr)
    return rollout_log


async def main_async(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_dir = args.results_dir / f"agent_rollout_{timestamp}" / "scene_000"
    scene_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[metrics] rollout started at {start_human}", file=sys.stderr)

    session = RolloutSession(scene_dir, args.config, args.prompt)
    try:
        session.start()
        policy = ModelPolicy(args.model_dir, max_new_tokens=args.max_new_tokens)

        env = session.make_floor_plan_env()
        await driver_loop(env=env, policy=policy,
                          max_steps=args.max_steps_per_stage)
        try:
            env.finalize_stage()
        except Exception as e:
            print(f"[floor_plan] geometry generation FAILED: {e}", file=sys.stderr)
            print("[chained] cannot proceed without geometry. Stopping.",
                  file=sys.stderr)
            return

        if not session.handover_after_floor_plan(args.room):
            return

        logger = ConsoleLogger(output_dir=session.scene.scene_dir)
        for stage, skip in [("furniture", args.skip_furniture),
                            ("wall",      args.skip_wall),
                            ("ceiling",   args.skip_ceiling)]:
            if skip:
                continue
            env = session.make_roomscene_env(stage, args.render_gpu_id, logger)
            try:
                await driver_loop(env=env, policy=policy,
                                  max_steps=args.max_steps_per_stage)
                env.finalize_stage()
            finally:
                env.cleanup()

        print(f"\n[done] all artifacts under {scene_dir}", file=sys.stderr)
    finally:
        try:
            write_run_metrics(scene_dir.parent, t_start, start_human, args)
        except Exception as e:
            print(f"[metrics warn] failed to write metrics.json: {e}",
                  file=sys.stderr)
        session.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decoupled rollout — in-container, model + tools in one process.",
    )
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--config", default="scene_designer")
    ap.add_argument("--render-gpu-id", type=int, default=None)
    ap.add_argument("--max-steps-per-stage", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--room", default=None)
    ap.add_argument("--skip-furniture", action="store_true")
    ap.add_argument("--skip-wall",      action="store_true")
    ap.add_argument("--skip-ceiling",   action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
