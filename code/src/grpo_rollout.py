"""Tree-structured GRPO rollout for OpenSceneAgent.

floor_plan stage: linear (one action per step).
furniture stage:  at every state-modifying step, sample N completions,
                  execute each from the SAME scene snapshot, collect
                  (context, completion, reward_placeholder) into a BranchGroup,
                  then randomly restore one branch state and continue.

The resulting GRPORolloutBuffer (list of BranchGroups) is returned to
train_grpo.py, which computes GRPO advantages and updates the policy.
"""
from __future__ import annotations

import copy
import json
import random
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.runtime import ROOT as REPO_ROOT, SCENESMITH_ROOT


import memory as _mem  # noqa: E402

# Reuse agent_rollout helpers (same container path assumptions).
from OpenSceneAgent.agent_rollout import (  # noqa: E402
    build_messages,
    parse_assistant_text,
    render_system,
    render_topdown,
)
from OpenSceneAgent.loop import dispatch  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tools that mutate RoomScene.objects (and therefore the Drake physics state).
# Non-modifying tools (check_physics, generate_assets) are NOT branch points.
STATE_MODIFYING_TOOLS: frozenset[str] = frozenset({
    "add_furniture_to_scene_tool",
    "move_furniture_tool",
    "snap_to_object_tool",
    "remove_furniture_tool",
    "rescale_furniture_tool",
})

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BranchGroup:
    """One branching step: N completions for the same (messages, image) context.

    rewards is initially filled with 0.0 placeholders; train_grpo.py overwrites
    them with real scores before computing GRPO advantages.
    """
    step_idx: int
    stage: str

    # Serialisable prompt context (for log-prob computation during training).
    # messages is the list[dict] passed to apply_chat_template.
    messages: list[dict]
    image_path: str | None          # str so it's JSON-serialisable

    completions: list[str]          # raw model outputs (including <think>)
    tool_calls_per_branch: list[list[dict]]  # parsed tool_calls for each completion
    tool_results_per_branch: list[list[dict]]

    rewards: list[float] = field(default_factory=list)  # filled in later
    chosen_idx: int = 0             # which branch was selected to continue
    # Per-branch scene snapshot dirs (floor_plan.json + furniture_*.json).
    # Used by reward_critic to render topdown images and run legality checks.
    branch_scene_dirs: list[str | None] = field(default_factory=list)
    # Scene dir BEFORE this branching step (shared parent state). Used by the
    # per-branch improvement reward: critic compares before vs after per branch.
    parent_scene_dir: str | None = None
    # Per-trajectory global advantage (filled by train_grpo.py after all G
    # trajectories of a prompt are collected). Added to the local advantage.
    global_adv: float = 0.0
    image_paths: list[str] = field(default_factory=list)


@dataclass
class GRPORolloutBuffer:
    """Collected branch groups from one episode (one prompt)."""
    prompt: str
    branch_groups: list[BranchGroup] = field(default_factory=list)
    # Number of furniture-stage turns actually taken in this trajectory
    # (modifying + non-modifying steps). Used by the length reward.
    traj_len: int = 0
    # Final scene dir (loadable via load_scene) after the trajectory ends.
    # Used by the object-count reward.
    final_scene_dir: str | None = None

    def num_samples(self) -> int:
        return sum(len(g.completions) for g in self.branch_groups)

    def to_json(self) -> str:
        import dataclasses
        return json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Model generate (single + batched)
# ---------------------------------------------------------------------------

class SampledCompletion(str):
    """Decoded text with the exact sampled IDs retained for the RL loss."""
    def __new__(cls, text, token_ids):
        value = super().__new__(cls, text)
        value.token_ids = list(token_ids)
        return value

    def __getnewargs__(self):
        return str(self), self.token_ids


@torch.no_grad()
def _generate_one(model, processor, messages: list[dict],
                  image_path: str | None,
                  max_new_tokens: int, temperature: float) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    from src.scene_observation import load_message_images
    images = load_message_images(messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    do_sample = temperature > 0.0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=pad_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    ids = out[0, prompt_len:].tolist()
    return SampledCompletion(processor.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False), ids)


@torch.no_grad()
def _generate_n(model, processor, messages: list[dict],
                image_path: str | None,
                n: int, max_new_tokens: int, temperature: float) -> list[str]:
    """Sample N completions independently (sequential; avoids OOM on large N)."""
    return [
        _generate_one(model, processor, messages, image_path, max_new_tokens, temperature)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------

def _is_state_modifying(tool_calls: list[dict]) -> bool:
    return any(tc.get("name") in STATE_MODIFYING_TOOLS for tc in tool_calls)


async def _execute_tool_calls(
    tool_map: dict, tool_calls: list[dict],
) -> list[dict]:
    """Execute a list of tool calls and return results."""
    results: list[dict] = []
    for tc in tool_calls:
        if tc.get("_parse_error"):
            results.append({"name": "_parse_error",
                            "output": "ERROR: malformed tool_call JSON"})
            continue
        name, args = tc["name"], tc.get("arguments") or {}
        try:
            out_str = await dispatch(tool_map, name, args)
            try:
                output = json.loads(out_str) if isinstance(out_str, str) else out_str
            except (json.JSONDecodeError, TypeError):
                output = out_str
        except Exception as e:  # noqa: BLE001
            output = f"ERROR: {e}\n{traceback.format_exc()}"
        results.append({"name": name, "output": output})
    return results


# ---------------------------------------------------------------------------
# Scene snapshot helpers (for critic reward)
# ---------------------------------------------------------------------------

def _flush_furniture_json(scene) -> None:
    """Serialize current room furniture state to furniture_{room_id}.json.

    Must be called before _save_branch_scene so the critic can find the file.
    The room's furniture lives in scene.objects (memory only); the critic's
    load_scene() expects furniture_{room_id}.json on disk in scene.scene_dir.
    """
    import json as _json
    room_id = getattr(scene, "room_id", "room")
    out_path = Path(scene.scene_dir) / f"furniture_{room_id}.json"
    placed = []
    for obj in getattr(scene, "objects", {}).values():
        try:
            d = obj.to_dict()
            if d.get("object_type") != "furniture":
                continue
            t = d.get("transform", {})
            entry = {
                "object_id": d.get("object_id", "unknown"),
                "name": d.get("name", ""),
                "description": d.get("description", ""),
                "translation": t.get("translation", [0.0, 0.0, 0.0]),
                "rotation_wxyz": t.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0]),
                "bbox_min": d.get("bbox_min") or [0.0, 0.0, 0.0],
                "bbox_max": d.get("bbox_max") or [1.0, 1.0, 1.0],
            }
            placed.append(entry)
        except Exception:  # noqa: BLE001
            pass
    out_path.write_text(_json.dumps({"placed_furniture": placed}, indent=2))


def _save_branch_scene(scene_dir: Path | str, dst_dir: Path | str, *, agent) -> None:
    """Copy floor_plan.json + furniture_*.json to dst_dir for critic evaluation.

    Creates an empty furniture file if none exists yet (first branch point).
    """
    import json as _json
    import shutil as _shutil
    scene_dir = Path(scene_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    from src.render_observation import render_scene_views
    topdown = render_scene_views(agent, agent.scene, dst_dir, 0)[0]
    _shutil.copy2(topdown, dst_dir / "_topdown.png")

    fp = scene_dir / "floor_plan.json"
    if not fp.exists():
        # floor_plan.json is stored at the scene level as house_layout.json
        alt_fp = scene_dir.parent / "house_layout.json"
        if alt_fp.exists():
            fp = alt_fp
    if fp.exists():
        _shutil.copy2(fp, dst_dir / "floor_plan.json")

    has_furniture = False
    for ff in sorted(scene_dir.glob("furniture_*.json")):
        if ".deepseek_raw" not in ff.name:
            _shutil.copy2(ff, dst_dir / ff.name)
            has_furniture = True

    # Create an empty furniture placeholder so load_scene doesn't fail.
    if not has_furniture and fp.exists():
        try:
            fp_data = _json.loads(fp.read_text())
            rooms = fp_data.get("rooms", [])
            room_id = rooms[0]["id"] if rooms else "room"
        except Exception:
            room_id = "room"
        (dst_dir / f"furniture_{room_id}.json").write_text(
            _json.dumps({"placed_furniture": []}, indent=2)
        )


# ---------------------------------------------------------------------------
# Core: tree furniture rollout
# ---------------------------------------------------------------------------

async def tree_furniture_rollout(
    *,
    model,
    processor,
    agent,
    tool_map: dict,
    scene,
    room_prompt: str,
    steps_dir: Path,
    n_branches: int = 4,
    max_steps: int = 40,
    max_new_tokens: int = 4096,
    temperature: float = 0.8,
    skip_render: bool = False,
) -> GRPORolloutBuffer:
    """Run the furniture stage with tree-structured branching at state-modifying steps.

    At every step where the model produces a state-modifying action:
      1. Sample n_branches completions (re-using the SAME scene snapshot).
      2. Execute each completion sequentially, restoring scene to snapshot before each.
      3. Add a BranchGroup to the buffer.
      4. Restore scene to the randomly chosen branch and continue.

    At steps that are NOT state-modifying, run a single greedy step.
    """
    steps_dir = Path(steps_dir)
    steps_dir.mkdir(parents=True, exist_ok=True)

    system = render_system("furniture", room_prompt)
    mem = _mem.empty_memory("furniture")
    handler = _mem.HANDLERS["furniture"]
    buffer = GRPORolloutBuffer(prompt=room_prompt)
    stage = "furniture"
    turns = 0   # furniture-stage turns actually taken (for the length reward)

    for step_idx in range(max_steps):
        # ---------- render + build messages ----------
        from src.render_observation import render_scene_views
        if skip_render:
            raise ValueError('Scene-GRPO requires current-state five-view observations')
        image_paths = render_scene_views(agent, scene, steps_dir, step_idx)
        image_path = image_paths[0]
        memory_str = _mem.render_memory(stage, mem)
        messages = build_messages(system, room_prompt, memory_str, image_paths)

        # ---------- greedy probe: is this a branching step? ----------
        probe = _generate_one(model, processor, messages, image_path,
                               max_new_tokens, temperature=0.0)
        _, probe_calls = parse_assistant_text(probe)

        if not probe_calls:
            if buffer.branch_groups:
                # We already have data and model wants to stop — done.
                print(f"[furniture][step {step_idx}] no tool_call — stage done",
                      file=sys.stderr)
                break
            # No data yet (step 0 often). Greedy probe stopped, but sampled
            # completions may still produce tool calls. Try once at temperature.
            probe = _generate_one(model, processor, messages, image_path,
                                   max_new_tokens, temperature)
            _, probe_calls = parse_assistant_text(probe)
            if not probe_calls:
                print(f"[furniture][step {step_idx}] no tool_call (greedy+sampled) — stage done",
                      file=sys.stderr)
                break

        if not _is_state_modifying(probe_calls):
            # Non-modifying step (check_physics / generate_assets): run once.
            results = await _execute_tool_calls(tool_map, probe_calls)
            for tc, res in zip(probe_calls, results):
                try:
                    handler(mem, tc["name"], tc.get("arguments") or {}, res["output"])
                except Exception as e:  # noqa: BLE001
                    print(f"[mem warn] {tc['name']}: {e}", file=sys.stderr)
            print(f"[furniture][step {step_idx}] non-modifying: "
                  f"{[tc['name'] for tc in probe_calls]}", file=sys.stderr)
            # Fix 17: If generate_assets returned 0 successes and we already have
            # branch data, stop. Prevents infinite retry loops when asset generation
            # is broken (e.g. Blender validation unavailable on headless servers).
            _stop = False
            if buffer.branch_groups:
                for _tc, _res in zip(probe_calls, results):
                    if _tc["name"] == "generate_assets":
                        try:
                            _out = json.loads(_res.get("output", "{}"))
                            if _out.get("successful_count", 1) == 0:
                                print(
                                    f"[furniture][step {step_idx}] generate_assets "
                                    f"returned 0 successes — branch data available, "
                                    f"stage done",
                                    file=sys.stderr,
                                )
                                _stop = True
                        except Exception:  # noqa: BLE001
                            pass
            if _stop:
                break
            turns += 1   # non-modifying step counts as a turn
            continue

        # ---------- branching step ----------
        print(f"[furniture][step {step_idx}] BRANCH (n={n_branches}): "
              f"{[tc['name'] for tc in probe_calls]}", file=sys.stderr)

        # Save current scene + memory state before any branch execution.
        scene_snapshot = scene.to_state_dict()
        mem_snapshot = copy.deepcopy(mem)

        # Save snapshot scene dir (used for no-op branches that change nothing).
        snapshot_scene_dir = steps_dir / f"step{step_idx:04d}_snapshot"
        try:
            _flush_furniture_json(scene)
            _save_branch_scene(scene.scene_dir, snapshot_scene_dir, agent=agent)
        except Exception as _e:
            print(f"[branch scene] snapshot save failed: {_e}", file=sys.stderr)
            snapshot_scene_dir = None  # type: ignore[assignment]

        completions: list[str] = []
        parsed_calls: list[list[dict]] = []
        branch_results: list[list[dict]] = []
        branch_states: list[dict] = []
        branch_mems: list[Any] = []   # mem state after each branch
        branch_scene_dirs: list[str | None] = []

        # Sample N completions (first one re-uses probe for speed).
        raw_completions = [probe] + _generate_n(
            model, processor, messages, image_path,
            n=n_branches - 1, max_new_tokens=max_new_tokens, temperature=temperature,
        )

        for branch_idx, completion in enumerate(raw_completions):
            # Restore scene + mem to the pre-branch snapshot for each branch.
            scene.restore_from_state_dict(scene_snapshot)
            mem_b = copy.deepcopy(mem_snapshot)

            _, tool_calls = parse_assistant_text(completion)
            if not tool_calls:
                # This completion ended the stage — treat as a no-op branch.
                completions.append(completion)
                parsed_calls.append([])
                branch_results.append([])
                branch_mems.append(mem_b)
                branch_states.append(scene.to_state_dict())
                # No-op branch: scene = snapshot state
                branch_scene_dirs.append(
                    str(snapshot_scene_dir) if snapshot_scene_dir else None
                )
                continue

            results = await _execute_tool_calls(tool_map, tool_calls)
            for tc, res in zip(tool_calls, results):
                # A malformed tool call may lack 'name' (rare model glitch).
                # Skip it instead of raising KeyError, which would kill the
                # whole trajectory — and don't reference tc['name'] in the
                # error path either (that re-raised the same KeyError before).
                tc_name = tc.get("name")
                if not tc_name:
                    print(f"[mem warn] branch {branch_idx}: tool call missing "
                          f"'name', skipping: {tc}", file=sys.stderr)
                    continue
                try:
                    handler(mem_b, tc_name, tc.get("arguments") or {}, res["output"])
                except Exception as e:  # noqa: BLE001
                    print(f"[mem warn] branch {branch_idx} {tc_name}: {e}",
                          file=sys.stderr)

            # Flush furniture state to disk before saving branch snapshot.
            # scene.objects lives in memory; critic needs furniture_*.json on disk.
            try:
                _flush_furniture_json(scene)
            except Exception as _fe:  # noqa: BLE001
                print(f"[branch scene] furniture flush failed branch {branch_idx}: {_fe}",
                      file=sys.stderr)

            # Save post-action scene for critic evaluation.
            branch_dir = steps_dir / f"step{step_idx:04d}_br{branch_idx:02d}"
            try:
                _save_branch_scene(scene.scene_dir, branch_dir, agent=agent)
                branch_scene_dirs.append(str(branch_dir))
            except Exception as _e:
                print(f"[branch scene] branch {branch_idx} save failed: {_e}",
                      file=sys.stderr)
                branch_scene_dirs.append(None)

            completions.append(completion)
            parsed_calls.append(tool_calls)
            branch_results.append(results)
            branch_mems.append(mem_b)
            branch_states.append(scene.to_state_dict())

            print(f"  branch {branch_idx}: {[tc['name'] for tc in tool_calls]}",
                  file=sys.stderr)

        # ---------- record branch group ----------
        chosen_idx = random.randrange(len(completions))
        bg = BranchGroup(
            step_idx=step_idx,
            stage=stage,
            messages=messages,
            image_path=image_path,
            image_paths=image_paths,
            completions=completions,
            tool_calls_per_branch=parsed_calls,
            tool_results_per_branch=branch_results,
            rewards=[0.0] * len(completions),  # placeholder; filled by reward fn
            chosen_idx=chosen_idx,
            branch_scene_dirs=branch_scene_dirs,
            parent_scene_dir=(str(snapshot_scene_dir)
                              if snapshot_scene_dir is not None else None),
        )
        buffer.branch_groups.append(bg)
        turns += 1   # branching step counts as a turn

        # Continue from exactly the state the Critic scored, without executing
        # potentially stateful tool calls a second time.
        scene.restore_from_state_dict(branch_states[chosen_idx])
        mem = copy.deepcopy(branch_mems[chosen_idx])
        _flush_furniture_json(scene)

    # ---------- record trajectory length + final scene (for global rewards) ----------
    buffer.traj_len = turns
    try:
        final_dir = steps_dir / "final_scene"
        _flush_furniture_json(scene)
        _save_branch_scene(scene.scene_dir, final_dir, agent=agent)
        buffer.final_scene_dir = str(final_dir)
    except Exception as _fe:  # noqa: BLE001
        print(f"[furniture] final scene save failed: {_fe}", file=sys.stderr)
        # Fallback: last chosen branch scene, if any.
        if buffer.branch_groups:
            last = buffer.branch_groups[-1]
            ci = last.chosen_idx
            if 0 <= ci < len(last.branch_scene_dirs):
                buffer.final_scene_dir = last.branch_scene_dirs[ci]

    print(f"[furniture] rollout done. "
          f"{len(buffer.branch_groups)} branch groups, "
          f"{buffer.num_samples()} total samples, traj_len={buffer.traj_len}.",
          file=sys.stderr)
    return buffer


# ---------------------------------------------------------------------------
# Linear floor_plan rollout (no branching needed)
# ---------------------------------------------------------------------------

async def linear_floor_plan_rollout(
    *,
    model,
    processor,
    fp_agent,
    fp_tool_map: dict,
    room_prompt: str,
    steps_dir: Path,
    max_steps: int = 20,
    max_new_tokens: int = 4096,
) -> list[dict]:
    """Run floor_plan stage linearly (greedy, no branching). Returns rollout log."""
    steps_dir = Path(steps_dir)
    steps_dir.mkdir(parents=True, exist_ok=True)

    system = render_system("floor_plan", room_prompt)
    mem = _mem.empty_memory("floor_plan")
    handler = _mem.HANDLERS["floor_plan"]
    rollout_log: list[dict] = []

    for step_idx in range(max_steps):
        memory_str = _mem.render_memory("floor_plan", mem)
        messages = build_messages(system, room_prompt, memory_str, image_path=None)

        generated = _generate_one(model, processor, messages, None,
                                  max_new_tokens, temperature=0.0)
        think, tool_calls = parse_assistant_text(generated)

        if not tool_calls:
            print(f"[floor_plan][step {step_idx}] no tool_call — stage done",
                  file=sys.stderr)
            rollout_log.append({"step": step_idx, "think": think,
                                "tool_calls": [], "tool_results": []})
            break

        results = await _execute_tool_calls(fp_tool_map, tool_calls)
        for tc, res in zip(tool_calls, results):
            try:
                handler(mem, tc["name"], tc.get("arguments") or {}, res["output"])
            except Exception as e:  # noqa: BLE001
                print(f"[mem warn] {tc['name']}: {e}", file=sys.stderr)

        rollout_log.append({
            "step": step_idx, "think": think,
            "tool_calls": tool_calls,
            "tool_results": results,
        })

    log_path = steps_dir / "floor_plan_rollout.json"
    log_path.write_text(json.dumps(rollout_log, indent=2, ensure_ascii=False))
    return rollout_log
