#!/usr/bin/env python3
"""SceneSmith configuration, backend services and stage handover helpers."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.runtime import SCENESMITH_ROOT as REPO_ROOT

import requests  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, OmegaConf, open_dict  # noqa: E402

from pydrake.all import ApplyCameraConfig, CameraConfig, RenderEngineGltfClientParams  # noqa: E402
from pydrake.common.schema import Transform as DrakeTransform  # noqa: E402
from pydrake.math import RigidTransform as DrakeRigidTransform  # noqa: E402

# Route VLM / Agents-SDK traffic through SCENESMITH_LLM_* gateway. main.py
# does this at import time; we must too, otherwise asset_router's VLM calls
# fall back to api.openai.com and "Connection error" every description.
# Idempotent; no-op if SCENESMITH_LLM_BASE_URL isn't set.
from scenesmith.llm_provider import configure as _configure_llm_gateway  # noqa: E402
_configure_llm_gateway()

from scenesmith.agent_utils.drake_utils import create_plant_from_dmd  # noqa: E402
from scenesmith.agent_utils.room import RoomScene  # noqa: E402
from scenesmith.utils.logging import ConsoleLogger  # noqa: E402

# Agent classes (for building compatible_agents dicts)
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import StatefulFloorPlanAgent  # noqa: E402
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent  # noqa: E402
from scenesmith.wall_agents.stateful_wall_agent import StatefulWallAgent  # noqa: E402
from scenesmith.ceiling_agents.stateful_ceiling_agent import StatefulCeilingAgent  # noqa: E402

from scenesmith.experiments.base_experiment import BaseExperiment  # noqa: E402
from scenesmith.experiments.indoor_scene_generation import (  # noqa: E402
    _get_retrieval_gpu_device,
)

# Servers
from scenesmith.agent_utils.geometry_generation_server.server_manager import (  # noqa: E402
    GeometryGenerationServer,
)
from scenesmith.agent_utils.hssd_retrieval_server.server_manager import (  # noqa: E402
    HssdRetrievalServer,
)
from scenesmith.agent_utils.articulated_retrieval_server.server_manager import (  # noqa: E402
    ArticulatedRetrievalServer,
)

from Designer.loop import (  # noqa: E402
    build_agent_and_tools as build_floor_plan_agent_tools,
    dispatch,
)


COMPATIBLE_FURNITURE_AGENTS = {
    "stateful_furniture_agent": StatefulFurnitureAgent,
    # Some configs alias; add as needed.
}
COMPATIBLE_WALL_AGENTS = {
    "stateful_wall_agent": StatefulWallAgent,
}
COMPATIBLE_CEILING_AGENTS = {
    "stateful_ceiling_agent": StatefulCeilingAgent,
}


# =========================================================================
# Config / server bootstrap
# =========================================================================


def compose_full_cfg(config_name: str, output_dir: Path):
    """Return (cfg DictConfig, cfg_dict plain dict) with _name fields injected.

    Mirrors main.py:48-60 — scenesmith expects `<section>._name` to be set
    to the chosen yaml's stem. With plain compose() that info lives in
    HydraConfig.get().runtime.choices; since we compose outside @hydra.main,
    we hardcode the defaults that config.yaml declares.
    """
    cfg_dir = str(REPO_ROOT / "configurations")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=[f"+name=chained_replay_{datetime.now():%H%M%S}"],
        )

    defaults_map = {
        "experiment":       "indoor_scene_generation",
        "floor_plan_agent": "stateful_floor_plan_agent",
        "furniture_agent":  "stateful_furniture_agent",
        "wall_agent":       "stateful_wall_agent",
        "ceiling_agent":    "stateful_ceiling_agent",
        "manipuland_agent": "stateful_manipuland_agent",
    }
    with open_dict(cfg):
        for section, name in defaults_map.items():
            if section in cfg and "_name" not in cfg[section]:
                cfg[section]._name = name
        # experiment.output_dir is also required by many server / agent code paths
        if "experiment" in cfg and "output_dir" not in cfg.experiment:
            cfg.experiment.output_dir = str(output_dir)

    from src.runtime import relocate_scene_config
    cfg = OmegaConf.create(relocate_scene_config(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, OmegaConf.to_container(cfg, resolve=True)


class BackendServers:
    """Start / stop geometry / HSSD / articulated servers.

    Mirrors scenesmith's indoor_scene_generation.{_start_geometry_server,
    _start_hssd_server, _start_articulated_server} exactly — same conditionals,
    same constructor arg names, same timeouts. Use an OmegaConf DictConfig so
    attribute access (`cfg.experiment.geometry_generation_server.host`) matches
    scenesmith's style 1:1.
    """

    def __init__(
        self,
        cfg: DictConfig,
        output_dir: Path,
        *,
        skip_geometry: bool = False,
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.skip_geometry = skip_geometry
        self.geometry_server = None
        self.hssd_server = None
        self.articulated_server = None

    def start_all(self) -> None:
        self._start_geometry()
        self._start_hssd()
        self._start_articulated()

    # ------------------------------------------------------------- geometry

    def _start_geometry(self) -> None:
        if self.skip_geometry:
            print(f"[server] skip geometry_server (stub assets)", file=sys.stderr)
            return
        furniture_uses_generated = (
            self.cfg.furniture_agent.asset_manager.general_asset_source == "generated"
        )
        manipuland_uses_generated = (
            self.cfg.manipuland_agent.asset_manager.general_asset_source == "generated"
        )
        if not (furniture_uses_generated or manipuland_uses_generated):
            print(f"[server] skip geometry_server (no agent uses 'generated')", file=sys.stderr)
            return

        server_config = self.cfg.experiment.geometry_generation_server
        backend = self.cfg.furniture_agent.asset_manager.get("backend", "hunyuan3d")
        sam3d_config = None
        if backend == "sam3d":
            sam3d_cfg = self.cfg.furniture_agent.asset_manager.sam3d
            sam3d_config = {
                "sam3_checkpoint":  str(sam3d_cfg.sam3_checkpoint),
                "sam3d_checkpoint": str(sam3d_cfg.sam3d_checkpoint),
            }
        print(f"[server] starting geometry_server ({backend}) on "
              f"{server_config.host}:{server_config.port}", file=sys.stderr)
        self.geometry_server = GeometryGenerationServer(
            host=server_config.host, port=server_config.port,
            backend=backend, sam3d_config=sam3d_config,
            log_file=self.output_dir / "geometry_server.log",
        )
        self.geometry_server.start()
        self.geometry_server.wait_until_ready(timeout_s=120.0)
        print(f"[server] geometry_server ready", file=sys.stderr)

    # ----------------------------------------------------------------- hssd

    def _start_hssd(self) -> None:
        needs = any(
            getattr(self.cfg, stage).asset_manager.general_asset_source == "hssd"
            for stage in ("furniture_agent", "manipuland_agent", "wall_agent", "ceiling_agent")
        )
        if not needs:
            print(f"[server] skip hssd_server (no agent uses 'hssd')", file=sys.stderr)
            return
        server_config = self.cfg.experiment.hssd_retrieval_server
        hssd_config = self.cfg.furniture_agent.asset_manager.hssd
        retrieval_device = _get_retrieval_gpu_device()
        print(f"[server] starting hssd_server on "
              f"{server_config.host}:{server_config.port}  clip={retrieval_device or 'default'}",
              file=sys.stderr)
        self.hssd_server = HssdRetrievalServer(
            host=server_config.host, port=server_config.port,
            preload_retriever=True,
            hssd_data_path=str(hssd_config.data_path),
            hssd_preprocessed_path=str(hssd_config.preprocessed_path),
            hssd_top_k=hssd_config.use_top_k,
            clip_device=retrieval_device,
        )
        self.hssd_server.start()
        self.hssd_server.wait_until_ready(timeout_s=60.0)
        print(f"[server] hssd_server ready", file=sys.stderr)

    # ---------------------------------------------------------- articulated

    def _start_articulated(self) -> None:
        needs = any(
            getattr(self.cfg, stage).asset_manager.router.strategies.articulated.enabled
            for stage in ("furniture_agent", "manipuland_agent", "wall_agent", "ceiling_agent")
        )
        if not needs:
            print(f"[server] skip articulated_server (no agent enables it)", file=sys.stderr)
            return
        server_config = self.cfg.experiment.articulated_retrieval_server
        articulated_config = self.cfg.furniture_agent.asset_manager.articulated
        retrieval_device = _get_retrieval_gpu_device()
        print(f"[server] starting articulated_server on "
              f"{server_config.host}:{server_config.port}  clip={retrieval_device or 'default'}",
              file=sys.stderr)
        self.articulated_server = ArticulatedRetrievalServer(
            host=server_config.host, port=server_config.port,
            preload_retriever=True,
            articulated_config=articulated_config,
            clip_device=retrieval_device,
        )
        self.articulated_server.start()
        self.articulated_server.wait_until_ready(timeout_s=60.0)
        print(f"[server] articulated_server ready", file=sys.stderr)

    def stop_all(self) -> None:
        for name, srv in (("geometry", self.geometry_server),
                          ("hssd", self.hssd_server),
                          ("articulated", self.articulated_server)):
            if srv is None:
                continue
            try:
                if hasattr(srv, "is_running") and srv.is_running():
                    srv.stop()
                    print(f"[server] stopped {name}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[server] stop {name} warn: {e!r}", file=sys.stderr)


# =========================================================================
# Floor plan: replay + derive RoomScene
# =========================================================================




def derive_room_scene(fp_agent, room_id: str) -> RoomScene:
    """Build a pre-furniture RoomScene from floor_plan agent's HouseLayout.

    Mirrors scenesmith's indoor_scene_generation.py:524.
    """
    if room_id not in fp_agent.layout.room_geometries:
        raise RuntimeError(
            f"room_id {room_id!r} not in layout.room_geometries; "
            f"have: {sorted(fp_agent.layout.room_geometries)}"
        )
    room_geometry = fp_agent.layout.room_geometries[room_id]

    room_spec = next((r for r in fp_agent.layout.room_specs if r.room_id == room_id), None)
    room_prompt = (room_spec.prompt if room_spec else "") or fp_agent.layout.house_prompt

    room_dir = fp_agent.layout.house_dir / f"room_{room_id}"
    room_dir.mkdir(parents=True, exist_ok=True)

    scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=room_dir,
        room_id=room_id,
        text_description=room_prompt,
        action_log_path=room_dir / "action_log.json",
    )
    for wall in room_geometry.walls:
        scene.add_object(wall)
    print(f"[derive] RoomScene room_id={room_id} prompt={room_prompt!r} "
          f"walls={len(room_geometry.walls)}", file=sys.stderr)
    return scene


# =========================================================================
# Furniture / wall / ceiling: replay against an existing RoomScene
# =========================================================================


# ---- asset_id alias machinery --------------------------------------------
#
# scenesmith's asset_router uses a VLM inside generate_*_assets to pick a
# short_name for each description. The VLM is non-deterministic: original
# demo may produce `abstract_print_0`, replay may produce `print_0` for the
# SAME description. Downstream tool_calls in the SFT reference the ORIGINAL
# id, so they fail ("asset not found") on replay unless we remap.
#
# Strategy: for every generate_*_assets dispatched, pair (raw original id)
# ↔ (replay real id) by `description` (stable across runs), add to a session-
# wide alias map, then rewrite all subsequent tool_call arguments before
# dispatching.
#
# The raw original id comes from the un-compressed trace in
# datasets/raw/<run>/<stage>_<room>.json which has every generate_*_assets's
# full `output.assets` list (each with description + object_id).

_GENERATE_ASSETS_TOOLS = {
    "generate_assets",            # furniture / manipuland
    "generate_wall_assets",
    "generate_ceiling_assets",
}










def _save_roomscene_blend(agent, output_path: Path) -> None:
    import subprocess
    import tempfile
    from src.runtime import ROOT
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scene-export-") as temp:
        state_path = Path(temp) / "state.json"
        state_path.write_text(json.dumps(agent.scene.to_state_dict()))
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/export_scene_worker.py"),
             str(Path(agent.scene.scene_dir).resolve()), str(state_path), str(output_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=600)
        if proc.returncode or not output_path.is_file():
            raise RuntimeError(f"Scene export failed ({proc.returncode}): " + proc.stderr[-3000:] + proc.stdout[-1500:])
