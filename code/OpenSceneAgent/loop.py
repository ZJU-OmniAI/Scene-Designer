#!/usr/bin/env python3
"""SceneSmith floor-plan agent construction and tool dispatch."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys

from datetime import datetime
from pathlib import Path

from src.runtime import SCENESMITH_ROOT as REPO_ROOT

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from agents.tool_context import ToolContext  # noqa: E402

from scenesmith.agent_utils.house import HouseLayout  # noqa: E402
from scenesmith.agent_utils.materials_retrieval_server.server_manager import (  # noqa: E402
    MaterialsRetrievalServer,
)
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (  # noqa: E402
    StatefulFloorPlanAgent,
)
from scenesmith.floor_plan_agents.tools.geometry_cache import GeometryCache  # noqa: E402
from scenesmith.utils.logging import ConsoleLogger  # noqa: E402

DEFAULT_HOUSE_PROMPT = (
    "A bedroom with a bed, two nightstands, and a wardrobe in the corner of the room."
)






def build_agent_and_tools(
    config_name: str, house_prompt: str, scene_dir: Path
) -> tuple[StatefulFloorPlanAgent, dict, ConsoleLogger, MaterialsRetrievalServer | None]:
    """Compose hydra config and instantiate the floor-plan agent's designer tools.

    Also starts the materials retrieval server on 127.0.0.1:7008 so the
    `get_material` FunctionTool works. Without it the tool fails with
    "Failed to connect to materials retrieval server at http://127.0.0.1:7008"
    because the full pipeline (main.py) is what normally spins it up.
    """
    scene_dir.mkdir(parents=True, exist_ok=True)

    cfg_dir = str(REPO_ROOT / "configurations")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=[f"+name=manual_demo_{datetime.now():%H%M%S}"],
        )

    from src.runtime import relocate_scene_config
    cfg_dict = relocate_scene_config(OmegaConf.to_container(cfg, resolve=True))
    fp_cfg = OmegaConf.create(cfg_dict["floor_plan_agent"])
    mat_cfg = OmegaConf.create(cfg_dict["experiment"]["materials_retrieval_server"])

    logger = ConsoleLogger(output_dir=scene_dir)
    print(f"[osa] scene_dir = {scene_dir}")
    print(
        "[osa] floor_plan_agent: "
        f"mode={fp_cfg.mode} "
        f"use_retrieval_server={fp_cfg.materials.use_retrieval_server}"
    )

    # Start the materials retrieval server so `get_material` works. Bind to
    # the host/port from the experiment config (default 127.0.0.1:7008).
    mat_server: MaterialsRetrievalServer | None = None
    if fp_cfg.materials.use_retrieval_server:
        print(
            "[osa] starting materials retrieval server on "
            f"{mat_cfg.host}:{mat_cfg.port} (preloads CLIP, ~5s)..."
        )
        mat_server = MaterialsRetrievalServer(
            host=str(mat_cfg.host),
            port=int(mat_cfg.port),
            preload_retriever=True,
            materials_config=mat_cfg,
        )
        mat_server.start()
        mat_server.wait_until_ready()
        print("[osa] materials retrieval server ready.")

    agent = StatefulFloorPlanAgent(cfg=fp_cfg, logger=logger)

    agent.layout = HouseLayout(house_dir=scene_dir, house_prompt=house_prompt)
    agent.house_prompt = house_prompt
    agent._geometry_cache = GeometryCache(cache_dir=scene_dir / ".geometry_cache")

    tool_map = {ft.name: ft for ft in agent._create_designer_tools()}
    return agent, tool_map, logger, mat_server






async def dispatch(tool_map: dict, name: str, arguments: dict) -> str:
    """Invoke the FunctionTool. Returns the string/JSON result."""
    if name not in tool_map:
        return f"ERROR: no such tool '{name}'. Try 'help'."
    ft = tool_map[name]
    ctx = ToolContext(
        context=None,
        tool_name=name,
        tool_call_id=f"osa-{datetime.now():%H%M%S%f}",
        tool_arguments=json.dumps(arguments),
    )
    try:
        result = await ft.on_invoke_tool(ctx, json.dumps(arguments))
    except Exception as e:  # noqa: BLE001 - intentional: show any tool failure to the user
        import traceback

        return f"ERROR during tool call:\n{traceback.format_exc()}"
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "__dict__"):
        return json.dumps(
            result.__dict__, indent=2, ensure_ascii=False, default=str
        )
    if isinstance(result, (str, bytes)):
        return result if isinstance(result, str) else result.decode("utf-8", "replace")
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)
