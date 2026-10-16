#!/usr/bin/env python3
"""Shared prompt formatting, output parsing and rollout metrics."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from src.runtime import ROOT as REPO_ROOT
import torch  # noqa: E402
from PIL import Image  # noqa: E402

# Match replay_chained: route VLM gateway env at import.
from scenesmith.llm_provider import configure as _configure_llm_gateway  # noqa: E402
_configure_llm_gateway()

from scenesmith.agent_utils.room import RoomScene  # noqa: E402
from scenesmith.utils.logging import ConsoleLogger  # noqa: E402
from scenesmith.experiments.base_experiment import BaseExperiment  # noqa: E402

# Reuse replay_chained's stage-handover machinery — fewer surfaces to debug.
from OpenSceneAgent.loop import (  # noqa: E402
    build_agent_and_tools as build_floor_plan_agent_tools,
    dispatch,
)
from OpenSceneAgent.replay_chained import (  # noqa: E402
    BackendServers,
    COMPATIBLE_CEILING_AGENTS,
    COMPATIBLE_FURNITURE_AGENTS,
    COMPATIBLE_WALL_AGENTS,
    _save_roomscene_blend,
    compose_full_cfg,
    derive_room_scene,
)

# Memory builder lives at <repo>/src/memory.py.
# In Docker the repo is mounted at <release-root>; on the host derive from __file__
# (__file__ = <repo>/scenesmith/OpenSceneAgent/agent_rollout.py → parents[2] = repo root).
_REPO_ROOT = REPO_ROOT
import memory as _mem  # noqa: E402

# Loaded after the env routing above (transformers may pull in network code).
# Container's transformers may be pinned to 4.55.x (no top-level Qwen3VL
# export), so try a deeper-path import as fallback. Qwen3-VL was added in
# 4.56; if even the deeper import fails, the version is genuinely too old.
from transformers import AutoProcessor  # noqa: E402
try:
    from transformers import Qwen3VLForConditionalGeneration  # noqa: E402
except ImportError:
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
            Qwen3VLForConditionalGeneration,
        )
    except ImportError as e:
        raise ImportError(
            "Container's transformers does not have Qwen3VLForConditionalGeneration. "
            "Upgrade to >=4.56 (Qwen3-VL landed there). Either rebuild the scenesmith "
            "image with `transformers>=5.0` in pyproject.toml, or run "
            "`pip install -U 'transformers>=5.0'` inside the container before launching. "
            f"Underlying: {e}"
        )


# =========================================================================
# Constants & regex
# =========================================================================


PROMPTS_DIR = _REPO_ROOT / "prompts"
DEFAULT_MODEL_DIR = _REPO_ROOT / "model" / "OpenSceneAgent"

# Match Qwen3-VL chat template's tool_call format. Body must be a JSON object
# with `name` + `arguments` (per the template the model was trained against).
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


# =========================================================================
# Output parsing
# =========================================================================


def parse_assistant_text(text: str) -> tuple[str | None, list[dict]]:
    """Extract <think> body + every <tool_call> JSON block."""
    think_m = _THINK_RE.search(text)
    think = think_m.group(1).strip() if think_m else None

    tool_calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        body = m.group(1).strip()
        try:
            tc = json.loads(body)
            tool_calls.append({
                "name": tc.get("name"),
                "arguments": tc.get("arguments") or {},
            })
        except json.JSONDecodeError:
            tool_calls.append({"_parse_error": True, "_raw": body})
    return think, tool_calls


# =========================================================================
# Prompt rendering (system + memory + image  →  messages)
# =========================================================================


def render_system(stage: str, room_prompt: str,
                  room_summary: str | None = None) -> str:
    """Read prompts/<stage>_system_prompt + prompts/<stage>_tool, fill placeholders.
    Identical to what src/make_episode.py used at SFT data-build time, so the
    model sees the same system prompt at inference as during training.

    `room_summary` (optional, used by furniture/wall/ceiling stages): a compact
    text block produced from the floor_plan agent's HouseLayout describing
    room dimensions, walls, openings, world-coord convention. Appended verbatim
    to the system text so the model has the same room facts it would have
    seen if the SFT data was patched via tools/inject_room_summary_to_sft.py.
    Pass None for floor_plan (no summary yet) or to skip appending."""
    template = (PROMPTS_DIR / f"{stage}_system_prompt").read_text(encoding="utf-8")
    tools = (PROMPTS_DIR / f"{stage}_tool").read_text(encoding="utf-8").rstrip()
    out = (
        template
        .replace("{{ room_prompt }}", room_prompt)
        .replace("{{room_prompt}}", room_prompt)
        .replace("{{ tools }}", tools)
        .replace("{{tools}}", tools)
    )
    if room_summary:
        out = out.rstrip() + "\n\n" + room_summary + "\n"
    return out


def build_messages(system: str, room_prompt: str, memory_str: str,
                   image_path: Path | None = None) -> list[dict]:
    """Same layout as src/sft_dataset.py._build_messages: system + user(image+task+memory)."""
    user_content: list[dict] = []
    paths = image_path if isinstance(image_path, (list, tuple)) else ([image_path] if image_path else [])
    for path in paths:
        user_content.append({"type": "image", "image": str(path)})
    user_content.append({
        "type": "text",
        "text": f"## Task\n{room_prompt}\n\n## Memory\n{memory_str}",
    })
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# =========================================================================
# Top-down rendering (only for furniture/wall/ceiling stages)
# =========================================================================


def render_topdown(agent, scene, save_dir: Path, step_idx: int) -> Path | None:
    """Render the current RoomScene's annotated top-down PNG (bbox / labels /
    direction arrows / coord points). Same render path as observe_scene.

    Returns the saved PNG path, or None if rendering wasn't possible.
    """
    rm = getattr(agent, "rendering_manager", None)
    bs = getattr(agent, "blender_server", None)
    if rm is None or bs is None:
        return None

    # Wall agent's cfg.rendering doesn't ship with `layout`; inject single_top
    # so render_scene knows what layout to produce.
    try:
        from omegaconf import open_dict
        cfg = rm.cfg
        if "layout" not in cfg:
            with open_dict(cfg):
                cfg.layout = "single_top"
    except Exception:  # noqa: BLE001
        pass

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"step_{step_idx:03d}.topdown.png"
    try:
        render_dir = rm.render_scene(
            scene=scene,
            blender_server=bs,
            rendering_mode="furniture",
            render_name=f"rollout_{step_idx:03d}",
        )
        for cand in ("0_top.png", "wall_context_top.png", "topdown.png"):
            p = Path(render_dir) / cand
            if p.exists():
                save_path.write_bytes(p.read_bytes())
                return save_path
        print(f"[render warn] no top-down candidate in {render_dir}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[render warn] {e}", file=sys.stderr)
    return None


# =========================================================================
# Model generate
# =========================================================================


@torch.no_grad()
def model_generate(model, processor, messages: list[dict],
                   image_path: Path | None,
                   max_new_tokens: int) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from src.scene_observation import load_message_images
    images = load_message_images(messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[0, prompt_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=False)


# =========================================================================
# Per-stage loop
# =========================================================================




# =========================================================================
# RoomScene-stage runner (furniture / wall / ceiling)
# =========================================================================




# =========================================================================
# Main: 4-stage rollout
# =========================================================================


def write_run_metrics(rollout_root: Path, t_start: float, start_human: str,
                      args: argparse.Namespace) -> None:
    """Walk every rollout_<stage>/rollout.json under rollout_root, count turns
    (each entry == one model input/output cycle), and write metrics.json next
    to scene_000/. Safe to call even if stages failed partway."""
    elapsed = time.perf_counter() - t_start
    end_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    elapsed_human = f"{h:02d}:{m:02d}:{s:02d}"

    per_stage: dict[str, int] = {}
    for f in sorted(rollout_root.glob("**/rollout.json")):
        stage = f.parent.name.replace("rollout_", "")
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            per_stage[stage] = len(d) if isinstance(d, list) else len(d.get("steps", []))
        except Exception:
            per_stage[stage] = -1
    turns_total = sum(n for n in per_stage.values() if n > 0)

    metrics = {
        "prompt":          args.prompt,
        "model_dir":       str(args.model_dir),
        "start_human":     start_human,
        "end_human":       end_human,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_human":   elapsed_human,
        "turns_per_stage": per_stage,
        "turns_total":     turns_total,
    }
    out = rollout_root / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n================== rollout metrics ==================",
          file=sys.stderr)
    print(f"  prompt        : {args.prompt}", file=sys.stderr)
    print(f"  model_dir     : {args.model_dir}", file=sys.stderr)
    print(f"  start         : {start_human}", file=sys.stderr)
    print(f"  end           : {end_human}", file=sys.stderr)
    print(f"  elapsed       : {elapsed_human}   ({elapsed:.1f} s)",
          file=sys.stderr)
    print(f"  turns/stage   :", file=sys.stderr)
    for stage, n in per_stage.items():
        print(f"    {stage:12s} {n:>4d} turns", file=sys.stderr)
    print(f"  turns_total   : {turns_total}", file=sys.stderr)
    print(f"  saved → {out}", file=sys.stderr)
    print("=====================================================",
          file=sys.stderr)
