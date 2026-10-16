"""GRPO training for OpenSceneAgent.

Collects tree-structured rollouts from the scenesmith environment
(floor_plan + furniture stages only), computes GRPO advantages per branch
group, and updates the policy using clipped PPO-style gradient updates.

Usage (from the release root):
    python -m src.train_grpo \\
        --model-dir model/OpenSceneAgent \\
        --prompts-file scenesmith/scripts/prompts.jsonl \\
        --output-dir model/OpenSceneAgent_grpo_v1 \\
        --n-branches 4 \\
        --n-episodes 2 \\
        --config scene_designer

GRPO objective (per branch group of size N):
    A_i = (r_i - mean(r)) / (std(r) + eps)                  # advantage
    ratio_i = exp(logp_i - logp_behavior_i)                      # importance ratio
    loss_i = -min(ratio_i * A_i, clip(ratio_i, 1±eps) * A_i)
    L = mean_over_groups_and_tokens(loss_i) + kl_coef * KL(pi || pi_ref)

The reward function is a pluggable hook (--reward-fn). Default: frozen Scene-Builder Critic with ranking and improvement rewards.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor

from src.runtime import ROOT as REPO_ROOT, SCENESMITH_ROOT


try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

from scenesmith.llm_provider import configure as _configure_llm_gateway  # noqa: E402
_configure_llm_gateway()

# Heavy scenesmith/OpenSceneAgent imports live ONLY inside collect_episode() so
# that multiprocessing.spawn (used by SAM3D worker_pool) can re-import this
# file without triggering the bpy import chain (which fails in spawned workers).


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPOTrainConfig:
    model_dir: Path
    output_dir: Path
    prompts_file: Path
    config: str = "scene_designer"
    # Where periodic checkpoints go, named OpenSceneAgent_rl_<step>step.
    ckpt_dir: Path = Path("model")
    results_dir: Path = Path("results")
    reward_fn: str = "critic"

    # Rollout
    n_branches: int = 4
    n_episodes: int = 2           # prompts per update step
    max_steps_furniture: int = 40
    max_new_tokens: int = 2048
    temperature: float = 0.8

    # GRPO hyper-params
    lr: float = 1e-6
    kl_coef: float = 0.05
    clip_eps: float = 0.2
    grad_accum: int = 1
    reward_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Training
    total_update_steps: int = 50
    warmup_steps: int = 5
    save_steps: int = 10
    start_step: int = 1   # resume from this step (1 = fresh start)
    seed: int = 42

    # GPU
    render_gpu_id: int | None = None

    # Critic reward
    critic_model_dir: Path = Path("model/Qwen3-VL-8B-Instruct")
    critic_adapter_dir: Path = Path(
        "datasets/critic_pairs/runs/scene_critic_lora_v10/final"
    )
    critic_device: str = "cuda:1"
    # When True, the critic IS the policy being trained (shared weights). No
    # separate critic model is loaded; each gradient update makes the next
    # round's rollout AND critic use the updated weights. --critic-model-dir /
    # --critic-adapter-dir / --critic-device are ignored in this mode.
    critic_self: bool = False
    # When True, the critic is a FROZEN copy of v7 = the ref_model (reused, no
    # extra load). Policy keeps updating; critic always scores with old v7.
    critic_frozen_v7: bool = True

    # Trajectory-level GRPO: G full trajectories per prompt (the outer group).
    n_trajectories: int = 4
    # Global reward thresholds.
    len_full: int = 20     # traj_len <= this → length reward 1.0
    len_zero: int = 38     # traj_len >= this → length reward 0.0
    obj_full: int = 8      # n_obj >= this → objcount reward 1.0
    obj_zero: int = 3      # n_obj <= this → objcount reward 0.0


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reward_random(buffer: GRPORolloutBuffer) -> None:
    """Placeholder: assign uniform random rewards in [0, 1]."""
    for bg in buffer.branch_groups:
        bg.rewards = [random.random() for _ in bg.completions]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_TOOL_CALL_BODY_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Legacy strict format reward: 0.5 for EXACTLY one <think> + 0.5 for EXACTLY one
# <tool_call> tag (JSON not validated). Kept as an opt-in switch for reproducing
# older runs; see --strict-format-reward.
STRICT_FORMAT_REWARD = False

# Per-step critic reward: legality gate.
#   True (default): physically-illegal branches are floored to LEGALITY_FLOOR
#     (below the critic's [0,1] rank range) and never shown to the critic, so
#     reward_borda tracks physical legality. LEGALITY_FLOOR is -0.5 (a gentler
#     floor than the old -1.0, so the critic's aesthetic ranking isn't drowned).
#   False (--pure-critic): pure critic ranking on ALL branches, no floor.
CRITIC_LEGALITY_GATE = True
LEGALITY_FLOOR = -0.5   # raw score for a physically-illegal / unrenderable branch


def _count_valid_tool_calls(comp: str) -> int:
    """Number of <tool_call> blocks whose body is valid JSON with a "name"."""
    n = 0
    for m in _TOOL_CALL_BODY_RE.finditer(comp):
        try:
            tc = json.loads(m.group(1).strip())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(tc, dict) and tc.get("name"):
            n += 1
    return n


def _format_score(comp: str) -> float:
    """Per-completion format reward in [0, 1].

    think half (0.5): exactly one well-formed <think>...</think> block.

    tool-call half (0.5):
      - strict (legacy): exactly one <tool_call> tag, JSON not checked.
      - default: at least one tool call AND every <tool_call> tag is valid JSON
        with a "name". This credits legitimate *batched* action turns (>=1 call)
        instead of only a single call, and stops rewarding tag-only-but-broken
        JSON. A think-only completion (no tool call, e.g. a sampled branch that
        declined to act at a state-modifying step) still gets 0.5.
    """
    r = 0.0
    if len(_THINK_RE.findall(comp)) == 1:
        r += 0.5
    if STRICT_FORMAT_REWARD:
        if len(_TOOL_CALL_RE.findall(comp)) == 1:
            r += 0.5
    else:
        n_tags = len(_TOOL_CALL_RE.findall(comp))
        n_valid = _count_valid_tool_calls(comp)
        if n_valid >= 1 and n_valid == n_tags:
            r += 0.5
    return r


def reward_format(buffer: GRPORolloutBuffer) -> None:
    """Format reward: 0.5 for the <think> block + 0.5 for well-formed tool call(s).

    See _format_score for the exact rule. A well-formed completion looks like:
        <think> ... </think>
        <tool_call> {"name": "...", "arguments": {...}} </tool_call>
    """
    for bg in buffer.branch_groups:
        bg.rewards = [_format_score(c) for c in bg.completions]


# ---------------------------------------------------------------------------
# Global (per-trajectory) rewards
# ---------------------------------------------------------------------------

def length_reward(traj_len: int, full: int, zero: int) -> float:
    """Trajectory-length reward, in [0, 1]. Encourages short trajectories.

    full score (1.0) for traj_len <= `full`; 0.0 for traj_len >= `zero`; linear
    ramp-down in between. Default thresholds: full=20, zero=38.
    """
    if traj_len <= full:
        return 1.0
    if traj_len >= zero:
        return 0.0
    return (zero - traj_len) / max(zero - full, 1)


def objcount_reward(n_obj: int, full: int, zero: int) -> float:
    """Object-count reward, in [0, 1]. Encourages richer scenes.

    full score (1.0) for n_obj >= `full`; 0.0 for n_obj <= `zero`; linear ramp-up
    in between. Default thresholds: full=8, zero=3.
    """
    if n_obj >= full:
        return 1.0
    if n_obj <= zero:
        return 0.0
    return (n_obj - zero) / max(full - zero, 1)


def count_objects(scene_dir: str | None) -> int:
    """Count placed objects (furniture + wall/ceiling-mounted) in a final scene.

    The furniture rollout only places `placed_furniture`, so this is len(scene
    .furniture). Returns 0 on any failure (missing dir / unreadable scene).
    """
    if not scene_dir:
        return 0
    try:
        from src_critic.scene_io import load_scene  # type: ignore[import]
        return len(load_scene(scene_dir).furniture)
    except Exception as e:  # noqa: BLE001
        print(f"[global_reward] count_objects failed for {scene_dir}: {e}",
              file=sys.stderr)
        return 0


def _load_critic(cfg: "GRPOTrainConfig"):
    """Load critic model (Qwen3-VL-8B + LoRA v10) onto cfg.critic_device."""
    from peft import PeftModel  # type: ignore[import]
    print(f"[train] loading critic base from {cfg.critic_model_dir}", file=sys.stderr)
    c_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(cfg.critic_model_dir), torch_dtype=torch.bfloat16,
    ).to(cfg.critic_device)
    print(f"[train] loading critic LoRA from {cfg.critic_adapter_dir}", file=sys.stderr)
    c_model = PeftModel.from_pretrained(c_model, str(cfg.critic_adapter_dir))
    c_model.eval()
    for p in c_model.parameters():
        p.requires_grad_(False)
    c_proc = AutoProcessor.from_pretrained(str(cfg.critic_model_dir))
    from src.scene_observation import configure_image_budget
    configure_image_budget(c_proc)
    return c_model, c_proc


def make_reward_critic(
    critic_model, critic_processor, is_policy: bool = False
) -> Callable[[GRPORolloutBuffer], None]:
    """Return a reward function = format + per-step critic ranking (local advantage).

    By default this is PURE critic (no physical-legality gate): every rendered
    branch is ranked by the critic, so intermediate collisions don't destroy the
    signal. Legality is enforced separately as an episode-level global reward
    (see the global step in the training loop). Set CRITIC_LEGALITY_GATE=True
    (--critic-legality-gate) to restore the old hard-floor behaviour.

    When ``is_policy`` is True the critic model IS the policy being trained. In
    that case each scoring pass temporarily flips it to eval() + KV-cache (and
    disables gradient checkpointing) so the ranking is deterministic and fast,
    then restores train() mode — mirroring the toggle in collect_episode().
    """
    from src_critic.grpo_advantage import (  # type: ignore[import]
        advantage_from_critic,
        advantage_from_critic_with_legality,
        legality_flags_from_scenes,
    )
    def _render(scene_dir: str) -> str | None:
        out = Path(scene_dir) / "_topdown.png"
        if not out.is_file():
            raise FileNotFoundError(f"Missing current-state Critic observation: {out}")
        return str(out)

    def _improvement(parent_dir, topdown, valid_idx, K):
        """Per-branch ±0.5: did the action improve the scene vs the shared parent?

        For each valid branch, a 2-way critic comparison [parent, branch] (which
        the critic WAS trained on, n2) is run. +0.5 if the critic ranks the
        branch above the parent, -0.5 if below, 0.0 if undecidable (parse fail /
        render fail / no parent).
        """
        imp = [0.0] * K
        if not parent_dir:
            return imp
        parent_img = _render(parent_dir)
        if parent_img is None:
            return imp
        for orig_i in valid_idx:
            branch_img = topdown[orig_i]
            if branch_img is None:
                continue
            try:
                res = advantage_from_critic(
                    critic_model, critic_processor,
                    [parent_img, branch_img], method="borda",
                )
            except Exception as e:  # noqa: BLE001
                print(f"[reward_critic] improvement call failed: {e}",
                      file=sys.stderr)
                continue
            rank = res.ranking            # 2 labels best→worst; A=parent, B=branch
            if not rank:
                continue                  # parse fail → neutral 0.0
            imp[orig_i] = 0.5 if rank[0] == "B" else -0.5
        return imp

    def _score_buffer(buffer: GRPORolloutBuffer) -> None:
        for bg in buffer.branch_groups:
            K = len(bg.completions)
            labels = list("ABCD")[:K]

            # Format sub-reward (0..1 per branch).
            fmt: list[float] = [_format_score(comp) for comp in bg.completions]

            scene_dirs = bg.branch_scene_dirs
            if len(scene_dirs) < 2:
                bg.rewards = fmt
                bg._fmt_rewards = fmt           # type: ignore[attr-defined]
                bg._borda_rewards = [0.0] * K   # type: ignore[attr-defined]
                bg._imp_rewards = [0.0] * K     # type: ignore[attr-defined]
                continue

            # Render topdown images for each branch.
            topdown = [_render(sd) if sd else None for sd in scene_dirs]

            # Only pass valid (non-None) paths to critic.
            valid_idx = [i for i, p in enumerate(topdown) if p is not None]
            if len(valid_idx) < 2:
                bg.rewards = fmt
                bg._fmt_rewards = fmt           # type: ignore[attr-defined]
                bg._borda_rewards = [0.0] * K   # type: ignore[attr-defined]
                bg._imp_rewards = [0.0] * K     # type: ignore[attr-defined]
                continue

            valid_paths = [topdown[i] for i in valid_idx]

            try:
                if CRITIC_LEGALITY_GATE:
                    # floor physically-illegal branches (to LEGALITY_FLOOR) before
                    # the critic, so reward_borda tracks physical legality.
                    legal_flags, _ = legality_flags_from_scenes(scene_dirs)
                    valid_legal = [legal_flags[i] for i in valid_idx]
                    res = advantage_from_critic_with_legality(
                        critic_model, critic_processor,
                        valid_paths, valid_legal, method="borda",
                        illegal_floor=LEGALITY_FLOOR,
                    )
                else:
                    # default: pure per-step critic ranking (local advantage).
                    res = advantage_from_critic(
                        critic_model, critic_processor,
                        valid_paths, method="borda",
                    )
            except Exception as e:
                print(f"[reward_critic] critic call failed: {e}", file=sys.stderr)
                bg.rewards = fmt
                bg._fmt_rewards = fmt           # type: ignore[attr-defined]
                bg._borda_rewards = [0.0] * K   # type: ignore[attr-defined]
                bg._imp_rewards = [0.0] * K     # type: ignore[attr-defined]
                continue

            # Map raw Borda scores back to all K branches.
            # Critic assigns "A"→valid_idx[0], "B"→valid_idx[1], ...
            critic_labels = list("ABCD")[:len(valid_idx)]
            critic_raw = [LEGALITY_FLOOR] * K   # unrendered branches keep the floor
            for out_pos, orig_i in enumerate(valid_idx):
                lab = critic_labels[out_pos]
                critic_raw[orig_i] = res.raw_score.get(lab, LEGALITY_FLOOR)

            # Per-branch improvement (±0.5) vs the shared parent scene.
            imp = _improvement(bg.parent_scene_dir, topdown, valid_idx, K)

            # Combined LOCAL reward: format (0..1) + critic Borda (-1..1) +
            # improvement (±0.5).
            bg.rewards = [fmt[i] + critic_raw[i] + imp[i] for i in range(K)]
            # Store component rewards for per-step aggregation in training loop.
            bg._fmt_rewards = fmt          # type: ignore[attr-defined]
            bg._borda_rewards = critic_raw  # type: ignore[attr-defined]
            bg._imp_rewards = imp          # type: ignore[attr-defined]
            rank_str = ">".join(res.ranking)
            score_str = " ".join(
                f"br{i:02d}:fmt={fmt[i]:.1f}+borda={critic_raw[i]:.3f}"
                f"+imp={imp[i]:+.1f}={bg.rewards[i]:.3f}" for i in range(K))
            print(f"[reward_critic] step{bg.step_idx:04d} rank={rank_str} | {score_str}", file=sys.stderr)

    def reward_critic(buffer: GRPORolloutBuffer) -> None:
        # Separate frozen critic: score directly (already eval() + no grad).
        if not is_policy:
            _score_buffer(buffer)
            return
        # Self-critic (critic == policy): flip to eval + KV-cache for a
        # deterministic, fast ranking, then restore training state. Generation
        # inside advantage_from_critic already runs under torch.no_grad(), so no
        # graph is built on the policy.
        was_training = critic_model.training
        prev_use_cache = getattr(critic_model.config, "use_cache", None)
        was_gc = getattr(critic_model, "is_gradient_checkpointing", False)
        critic_model.eval()
        if was_gc:
            critic_model.gradient_checkpointing_disable()
        critic_model.config.use_cache = True
        try:
            _score_buffer(buffer)
        finally:
            critic_model.config.use_cache = prev_use_cache
            if was_gc:
                critic_model.gradient_checkpointing_enable()
            if was_training:
                critic_model.train()

    return reward_critic


# Reward registry — plug in your own function here later.
REWARD_FN_REGISTRY: dict[str, Callable[[GRPORolloutBuffer], None]] = {
    "random": reward_random,
    "format": reward_format,
}


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def compute_completion_token_log_probs(
    model,
    processor,
    messages: list[dict],
    image_path: str | None,
    completions: list[str],
) -> list[torch.Tensor]:
    """Return completion-token log probabilities, one tensor per completion.
    """
    from PIL import Image as PILImage

    log_probs: list[torch.Tensor] = []
    for completion in completions:
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        from src.scene_observation import load_message_images
        images = load_message_images(messages)
        prompt_inputs = processor(
            text=[prompt_text], images=images, return_tensors="pt", padding=False,
        ).to(model.device)
        ids = getattr(completion, 'token_ids', None)
        if ids is None:
            ids = processor.tokenizer.encode(str(completion), add_special_tokens=False)
        if not ids:
            raise ValueError('Cannot optimize an empty completion')
        completion_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        full_inputs = dict(prompt_inputs)
        full_inputs['input_ids'] = torch.cat([prompt_inputs['input_ids'], completion_ids], dim=1)
        full_inputs['attention_mask'] = torch.cat([prompt_inputs['attention_mask'], torch.ones_like(completion_ids)], dim=1)
        if 'mm_token_type_ids' in prompt_inputs:
            full_inputs['mm_token_type_ids'] = torch.cat([prompt_inputs['mm_token_type_ids'], torch.zeros_like(completion_ids)], dim=1)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        input_ids = full_inputs["input_ids"]  # [1, L]

        logits = model(**full_inputs).logits  # [1, L, V]

        shift_logits = logits[0, :-1, :]     # [L-1, V]
        shift_labels = input_ids[0, 1:]      # [L-1]

        completion_start = prompt_len - 1
        mask = torch.zeros(shift_labels.shape[0], dtype=torch.bool,
                           device=model.device)
        mask[completion_start:] = True

        token_log_probs = F.log_softmax(shift_logits, dim=-1)  # [L-1, V]
        selected = token_log_probs[
            torch.arange(shift_labels.shape[0], device=model.device), shift_labels
        ]  # [L-1]
        log_probs.append(selected[mask])

    return log_probs


def grpo_loss_for_group(
    bg: BranchGroup,
    model,
    ref_model,
    processor,
    clip_eps: float,
    kl_coef: float,
    reward_eps: float,
) -> torch.Tensor:
    """Compute GRPO loss for one BranchGroup.

    Advantage = LOCAL advantage (branch rewards z-normalized within this group)
    + GLOBAL advantage (this trajectory's global reward z-normalized across the
    G trajectories of the prompt; the same scalar for every branch here). The
    global term is broadcast to all branches so a globally-good trajectory lifts
    all its actions, a globally-bad one suppresses them — trajectory-level GRPO.

    Returns scalar loss tensor with grad.
    """
    rewards = torch.tensor(bg.rewards, dtype=torch.float32, device=model.device)
    global_adv = float(getattr(bg, "global_adv", 0.0))
    std = rewards.std(unbiased=False)
    if std < reward_eps:
        local_adv = torch.zeros_like(rewards)   # flat local rewards → no local signal
    else:
        local_adv = (rewards - rewards.mean()) / (std + reward_eps)

    advantages = local_adv + global_adv         # broadcast global scalar to all branches
    if advantages.abs().max() < reward_eps:
        # No local AND no global signal → nothing to learn from this group.
        return torch.tensor(0.0, requires_grad=True, device=model.device)

    from src.grpo_objective import grpo_objective
    log_probs_new = compute_completion_token_log_probs(
        model, processor, bg.messages, bg.image_path, bg.completions)
    # One optimizer step follows collection; cache the behavior policy per group.
    if not getattr(bg, "behavior_log_probs", None):
        bg.behavior_log_probs = [x.detach().clone() for x in log_probs_new]
    with torch.no_grad():
        log_probs_ref = compute_completion_token_log_probs(
            ref_model, processor, bg.messages, bg.image_path, bg.completions)
    return grpo_objective(log_probs_new, bg.behavior_log_probs, log_probs_ref,
                          advantages, clip_eps=clip_eps, kl_coef=kl_coef)


# ---------------------------------------------------------------------------
# One episode: floor_plan → furniture → rollout buffer
# ---------------------------------------------------------------------------

async def collect_episode(
    prompt: str,
    cfg: "GRPOTrainConfig",
    model,
    processor,
    episode_dir: Path,
) -> "GRPORolloutBuffer | None":
    """Run one full episode (floor_plan linear + furniture tree) and return buffer."""
    from grpo_rollout import (  # noqa: E402
        GRPORolloutBuffer,
        linear_floor_plan_rollout,
        tree_furniture_rollout,
    )
    from scenesmith.utils.logging import ConsoleLogger  # noqa: E402
    from scenesmith.experiments.base_experiment import BaseExperiment  # noqa: E402
    from OpenSceneAgent.loop import build_agent_and_tools  # noqa: E402
    from OpenSceneAgent.replay_chained import (  # noqa: E402
        BackendServers,
        COMPATIBLE_FURNITURE_AGENTS,
        compose_full_cfg,
        derive_room_scene,
    )

    episode_dir.mkdir(parents=True, exist_ok=True)

    full_cfg, cfg_dict = compose_full_cfg(cfg.config, episode_dir)
    servers = BackendServers(full_cfg, episode_dir)

    fp_agent = fp_mat_server = None
    _was_gc = getattr(model, 'is_gradient_checkpointing', False)
    try:
        if _was_gc:
            model.gradient_checkpointing_disable()
        servers.start_all()
        fp_agent, fp_tool_map, fp_logger, fp_mat_server = build_agent_and_tools(
            config_name=cfg.config, house_prompt=prompt, scene_dir=episode_dir,
        )

        # ---------- floor_plan (linear) ----------
        await linear_floor_plan_rollout(
            model=model, processor=processor,
            fp_agent=fp_agent, fp_tool_map=fp_tool_map,
            room_prompt=prompt,
            steps_dir=episode_dir / "rollout_floor_plan",
            max_steps=20, max_new_tokens=cfg.max_new_tokens,
        )

        fp_agent._generate_all_room_geometries(
            output_dir=fp_agent.layout.house_dir / "floor_plans"
        )
        # Skip _export_floor_plan(): it calls save_directive_as_blend() which invokes
        # Drake's RenderEngineGltfClient → vtkEGLRenderWindow → EGL init fails on
        # headless server → VTK calls exit(1) → terminates main training process.
        # The .blend file is only for inspection and is NOT needed for GRPO training.
        (fp_agent.layout.house_dir / "house_layout.json").write_text(
            json.dumps(fp_agent.layout.to_dict(), indent=2)
        )

        if not fp_agent.layout.placed_rooms:
            print(f"[episode] no rooms generated for prompt: {prompt[:60]}",
                  file=sys.stderr)
            return None

        room_id = fp_agent.layout.placed_rooms[0].room_id
        scene = derive_room_scene(fp_agent, room_id)
        logger = ConsoleLogger(output_dir=scene.scene_dir)

        # ---------- furniture (tree rollout) ----------
        furniture_agent = BaseExperiment.build_furniture_agent(
            cfg_dict=cfg_dict,
            compatible_agents=COMPATIBLE_FURNITURE_AGENTS,
            logger=logger,
            render_gpu_id=cfg.render_gpu_id,
        )
        furniture_agent.scene = scene
        designer_tools = furniture_agent._create_designer_tools()
        # Exclude observe_scene: calls RenderEngineGltfClient → vtkEGLRenderWindow
        # → EGL init fails on headless server → VTK calls C exit(1).
        # check_physics is now safe: it runs Drake in an isolated subprocess
        # (physics_check_worker.py), so libdrake.so segfaults cannot reach here.
        _SKIP_TOOLS = {"observe_scene"}
        tool_map = {ft.name: ft for ft in designer_tools
                    if ft.name not in _SKIP_TOOLS}

        buffer = await tree_furniture_rollout(
            model=model, processor=processor,
            agent=furniture_agent, tool_map=tool_map,
            scene=scene, room_prompt=prompt,
            steps_dir=episode_dir / "rollout_furniture",
            n_branches=cfg.n_branches,
            max_steps=cfg.max_steps_furniture,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            skip_render=False,
        )

        try:
            furniture_agent.cleanup()
        except Exception:  # noqa: BLE001
            pass

        return buffer

    except Exception as e:  # noqa: BLE001
        print(f"[episode ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None
    finally:
        servers.stop_all()
        if fp_agent is not None:
            try:
                fp_agent.cleanup()
            except Exception:  # noqa: BLE001
                pass
        if fp_mat_server is not None:
            try:
                fp_mat_server.stop()
            except Exception:  # noqa: BLE001
                pass
        if _was_gc:
            model.gradient_checkpointing_enable()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(cfg: GRPOTrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts.
    prompts = [
        json.loads(line)["prompt"]
        for line in cfg.prompts_file.read_text().splitlines()
        if line.strip()
    ]
    print(f"[train] {len(prompts)} prompts, {cfg.total_update_steps} update steps",
          file=sys.stderr)

    # Determine load dir: resume from last checkpoint if start_step > 1.
    if cfg.start_step > 1:
        ckpt_load_dir = cfg.ckpt_dir / f"OpenSceneAgent_rl_{cfg.start_step - 1}step"
        if not ckpt_load_dir.exists():
            raise FileNotFoundError(
                f"--start-step {cfg.start_step} requested but checkpoint not found: {ckpt_load_dir}"
            )
        _load_dir = ckpt_load_dir
        print(f"[train] resuming from checkpoint {_load_dir} (step {cfg.start_step})",
              file=sys.stderr)
    else:
        _load_dir = cfg.model_dir
        print(f"[train] loading model from {_load_dir}", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(str(_load_dir))
    from src.scene_observation import configure_image_budget
    configure_image_budget(processor)

    policy = Qwen3VLForConditionalGeneration.from_pretrained(
        str(_load_dir), torch_dtype=torch.bfloat16,
    ).to("cuda")
    policy.gradient_checkpointing_enable()
    policy.train()

    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(cfg.model_dir), torch_dtype=torch.bfloat16,  # ref always from original
    ).to("cuda")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = Adafactor(
        policy.parameters(),
        lr=cfg.lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    # Always create scheduler with last_epoch=-1 so PyTorch sets initial_lr in
    # param groups during __init__. Then overwrite with saved state if resuming.
    # (Using last_epoch=N>0 at creation triggers a PyTorch check for initial_lr
    # which fails on a freshly-constructed optimizer.)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.total_update_steps,
    )

    # Restore optimizer + scheduler state if resuming.
    if cfg.start_step > 1:
        _ckpt_load_dir = cfg.ckpt_dir / f"OpenSceneAgent_rl_{cfg.start_step - 1}step"
        _opt_path = _ckpt_load_dir / "optimizer.pt"
        _sched_path = _ckpt_load_dir / "scheduler.pt"
        if _opt_path.exists():
            optimizer.load_state_dict(torch.load(str(_opt_path), map_location="cpu"))
            print(f"[train] loaded optimizer state from {_opt_path}", file=sys.stderr)
        if _sched_path.exists():
            scheduler.load_state_dict(torch.load(str(_sched_path), map_location="cpu"))
            print(f"[train] loaded scheduler state from {_sched_path}", file=sys.stderr)

    log_path = cfg.output_dir / "train_grpo.log"
    log_file = open(log_path, "a" if cfg.start_step > 1 else "w", encoding="utf-8")

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, file=sys.stderr)
        print(line, file=log_file, flush=True)

    log(f"Starting GRPO training. Output: {cfg.output_dir}")

    # Set up the reward function.
    if cfg.reward_fn == "critic":
        if cfg.critic_frozen_v7:
            # Critic = frozen v7 = the ref_model (reused, no extra load). Policy
            # keeps updating; the critic always scores with the old v7 weights.
            critic_model, critic_processor = ref_model, processor
            reward_fn: Callable[[GRPORolloutBuffer], None] = make_reward_critic(
                critic_model, critic_processor, is_policy=False
            )
            log("Critic = frozen v7 (= ref_model, shared frozen weights)")
        elif cfg.critic_self:
            # Critic IS the policy: shared weights, no extra model loaded.
            critic_model, critic_processor = policy, processor
            reward_fn = make_reward_critic(
                critic_model, critic_processor, is_policy=True
            )
            log("Critic = policy (self-critic, shared weights on policy GPU)")
        else:
            critic_model, critic_processor = _load_critic(cfg)
            reward_fn = make_reward_critic(
                critic_model, critic_processor, is_policy=False
            )
            log(f"Critic reward loaded on {cfg.critic_device}")
    else:
        reward_fn = REWARD_FN_REGISTRY.get(cfg.reward_fn, reward_format)

    prompt_idx = (cfg.start_step - 1) * cfg.n_episodes  # keep prompt rotation consistent
    # Abort if the rollout stops producing usable groups for this many steps in
    # a row. A recoverable-looking failure (e.g. the geometry server losing a
    # port-bind race after a worker teardown) otherwise makes every remaining
    # step a silent no-op, and the loop still runs to the end and prints
    # "Training complete" with a zero exit code — so nothing restarts and no
    # alarm fires. Better to crash loudly and keep the last good checkpoint.
    MAX_CONSECUTIVE_EMPTY = 5
    applied_updates = 0
    consecutive_empty = 0
    for update_step in range(cfg.start_step, cfg.total_update_steps + 1):
        t0 = time.perf_counter()
        log(f"=== update step {update_step}/{cfg.total_update_steps} ===")

        # ---- collect: n_episodes prompts × G full trajectories each ----
        # Each prompt yields G trajectories = one trajectory-level GRPO group.
        # LOCAL reward (per branch): fmt + borda + improvement (reward_fn).
        # GLOBAL reward (per trajectory): length_reward + objcount_reward.
        all_groups: list[BranchGroup] = []
        all_global: list[float] = []
        traj_lens: list[int] = []
        obj_counts: list[int] = []
        for ep_i in range(cfg.n_episodes):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            log(f"  prompt {ep_i+1}/{cfg.n_episodes} × G={cfg.n_trajectories}: "
                f"{prompt[:55]}")

            trajs: list[GRPORolloutBuffer] = []
            for g in range(cfg.n_trajectories):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                episode_dir = cfg.results_dir / f"grpo_{ts}_{ep_i}_{g}" / "scene_000"
                buffer = asyncio.run(collect_episode(prompt, cfg, policy, processor,
                                                     episode_dir))
                if buffer is None:
                    log(f"    traj {g+1}/{cfg.n_trajectories} failed, skipping")
                    continue
                reward_fn(buffer)   # fills LOCAL per-branch rewards (fmt+borda+imp)
                N = count_objects(buffer.final_scene_dir)
                r_len = length_reward(buffer.traj_len, cfg.len_full, cfg.len_zero)
                r_obj = objcount_reward(N, cfg.obj_full, cfg.obj_zero)
                buffer._global_reward = r_len + r_obj  # type: ignore[attr-defined]
                traj_lens.append(buffer.traj_len)
                obj_counts.append(N)
                all_global.append(buffer._global_reward)  # type: ignore[attr-defined]
                log(f"    traj {g+1}/{cfg.n_trajectories}: "
                    f"{len(buffer.branch_groups)} groups, len={buffer.traj_len}, "
                    f"N={N}, global={buffer._global_reward:.3f} "
                    f"(len {r_len:.2f}+obj {r_obj:.2f})")
                trajs.append(buffer)

            # ---- global advantage across THIS prompt's G trajectories ----
            if not trajs:
                continue
            g_rewards = [b._global_reward for b in trajs]  # type: ignore[attr-defined]
            g_mean = sum(g_rewards) / len(g_rewards)
            g_std = (sum((r - g_mean) ** 2 for r in g_rewards) / len(g_rewards)) ** 0.5
            for b in trajs:
                if g_std > cfg.reward_eps:
                    g_adv = (b._global_reward - g_mean) / (g_std + cfg.reward_eps)  # type: ignore[attr-defined]
                else:
                    g_adv = 0.0
                for bg in b.branch_groups:
                    bg.global_adv = g_adv
                all_groups.extend(b.branch_groups)

        if not all_groups:
            consecutive_empty += 1
            log(f"  no valid groups this step, skipping update "
                f"({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY} in a row)")
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                raise RuntimeError(
                    f"{consecutive_empty} consecutive steps produced no valid "
                    f"groups (step {update_step}). The rollout/geometry pipeline "
                    f"is broken — aborting so the run fails loudly instead of "
                    f"burning the remaining steps as no-ops. Last good "
                    f"checkpoint is preserved.")
            continue
        consecutive_empty = 0

        # ---- Aggregate reward stats (local components + global) ----
        all_total: list[float] = []
        all_fmt: list[float] = []
        all_borda: list[float] = []
        all_imp: list[float] = []
        n_illegal_groups = 0
        for bg in all_groups:
            all_total.extend(bg.rewards)
            fmt_r  = getattr(bg, "_fmt_rewards",  [None] * len(bg.rewards))
            borda_r = getattr(bg, "_borda_rewards", [None] * len(bg.rewards))
            imp_r = getattr(bg, "_imp_rewards", [None] * len(bg.rewards))
            if all(v is not None for v in fmt_r):
                all_fmt.extend(fmt_r)
            if all(v is not None for v in borda_r):
                all_borda.extend(borda_r)
                if all(v < 0 for v in borda_r):   # every branch illegal/unrendered
                    n_illegal_groups += 1
            if all(v is not None for v in imp_r):
                all_imp.extend(imp_r)
        _n = max(len(all_total), 1)
        avg_total = sum(all_total) / _n
        avg_fmt   = sum(all_fmt)   / max(len(all_fmt), 1)
        avg_borda = sum(all_borda) / max(len(all_borda), 1)
        avg_imp   = sum(all_imp)   / max(len(all_imp), 1)
        avg_len   = sum(traj_lens) / max(len(traj_lens), 1)
        avg_obj   = sum(obj_counts) / max(len(obj_counts), 1)
        avg_global = sum(all_global) / max(len(all_global), 1)
        n_traj = len(traj_lens)

        # ---- GRPO gradient update ----
        optimizer.zero_grad()
        total_loss = 0.0
        n_groups_used = 0

        for gi, bg in enumerate(all_groups):
            if len(bg.completions) < 2:
                continue  # need at least 2 for meaningful advantage

            loss = grpo_loss_for_group(
                bg, policy, ref_model, processor,
                clip_eps=cfg.clip_eps,
                kl_coef=cfg.kl_coef,
                reward_eps=cfg.reward_eps,
            )
            (loss / cfg.grad_accum).backward()
            total_loss += loss.item()
            n_groups_used += 1

        if n_groups_used > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            applied_updates += 1

        elapsed = time.perf_counter() - t0
        log(
            f"  loss={total_loss/max(n_groups_used,1):.4f}  "
            f"reward_total={avg_total:.3f}  reward_fmt={avg_fmt:.3f}  "
            f"reward_borda={avg_borda:.3f}  reward_imp={avg_imp:+.3f}  "
            f"reward_global={avg_global:.3f}(len{avg_len:.1f} obj{avg_obj:.1f})  "
            f"illegal_groups={n_illegal_groups}/{len(all_groups)}  "
            f"groups={n_groups_used}  trajs={n_traj}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.0f}s"
        )

        # ---- save checkpoint every save_steps → model/OpenSceneAgent_rl_<step>step ----
        if update_step % cfg.save_steps == 0:
            ckpt_dir = cfg.ckpt_dir / f"OpenSceneAgent_rl_{update_step}step"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(str(ckpt_dir))
            processor.save_pretrained(str(ckpt_dir))
            torch.save(optimizer.state_dict(), str(ckpt_dir / "optimizer.pt"))
            torch.save(scheduler.state_dict(), str(ckpt_dir / "scheduler.pt"))
            log(f"  saved checkpoint → {ckpt_dir}")

    if applied_updates == 0:
        raise RuntimeError("No optimizer update completed; refusing to publish an unchanged RL checkpoint")
    # Final save.
    policy.save_pretrained(str(cfg.output_dir))
    processor.save_pretrained(str(cfg.output_dir))
    log(f"Training complete. Model saved to {cfg.output_dir}")
    log_file.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> GRPOTrainConfig:
    ap = argparse.ArgumentParser(description="GRPO training for OpenSceneAgent")
    ap.add_argument("--model-dir",    type=Path, required=True)
    ap.add_argument("--output-dir",   type=Path, required=True)
    ap.add_argument("--ckpt-dir",     type=Path, default=Path("model"),
                    help="dir for periodic checkpoints (OpenSceneAgent_rl_<step>step)")
    ap.add_argument("--prompts-file", type=Path,
                    default=Path("scenesmith/scripts/prompts.jsonl"))
    ap.add_argument("--config",       default="scene_designer")
    ap.add_argument("--results-dir",  type=Path, default=Path("results"))
    ap.add_argument("--n-branches",   type=int,  default=4)
    ap.add_argument("--n-episodes",   type=int,  default=2)
    ap.add_argument("--n-trajectories", type=int, default=4,
                    help="G: full trajectories per prompt (outer GRPO group). "
                         "Global advantage is computed across these G trajectories.")
    ap.add_argument("--max-steps-furniture", type=int, default=40)
    ap.add_argument("--max-new-tokens",      type=int, default=2048)
    ap.add_argument("--temperature",  type=float, default=0.8)
    ap.add_argument("--lr",           type=float, default=1e-6)
    ap.add_argument("--kl-coef",      type=float, default=0.05)
    ap.add_argument("--clip-eps",     type=float, default=0.2)
    ap.add_argument("--grad-accum",   type=int,   default=1)
    ap.add_argument("--total-update-steps", type=int, default=50)
    ap.add_argument("--warmup-steps", type=int,   default=5)
    ap.add_argument("--save-steps",   type=int,   default=10)
    ap.add_argument("--start-step",   type=int,   default=1,
                    help="Resume from this update step (1 = fresh start). "
                         "Model is loaded from output-dir/checkpoint-<start_step-1>.")
    ap.add_argument("--render-gpu-id", type=int,  default=None)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--reward-fn",    default="critic",
                    choices=list(REWARD_FN_REGISTRY.keys()) + ["critic"],
                    help="reward function name")
    ap.add_argument("--critic-model-dir",   type=Path,
                    default=Path("model/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--critic-adapter-dir", type=Path,
                    default=Path(
                        "datasets/critic_pairs/runs/scene_critic_lora_v10/final"
                    ))
    ap.add_argument("--critic-device",      default="cuda:1")
    ap.add_argument("--critic-self", action="store_true",
                    help="use the policy itself as the critic (shared weights, "
                         "self-critic). Ignores --critic-model-dir / "
                         "--critic-adapter-dir / --critic-device; the critic "
                         "co-evolves with the policy after every update.")
    ap.add_argument("--critic-frozen-v7", action=argparse.BooleanOptionalAction, default=True,
                    help="critic = frozen v7 = the ref_model (reused, no extra "
                         "load). Policy keeps updating; critic always scores "
                         "with old v7. Takes precedence over --critic-self.")
    ap.add_argument("--len-full", type=int, default=20,
                    help="traj_len <= this → length reward 1.0")
    ap.add_argument("--len-zero", type=int, default=38,
                    help="traj_len >= this → length reward 0.0")
    ap.add_argument("--obj-full", type=int, default=8,
                    help="n_obj >= this → objcount reward 1.0")
    ap.add_argument("--obj-zero", type=int, default=3,
                    help="n_obj <= this → objcount reward 0.0")
    ap.add_argument("--strict-format-reward", action="store_true",
                    help="use the legacy format reward (exactly one <think> + "
                         "exactly one <tool_call> tag, JSON not validated)")
    ap.add_argument("--pure-critic", action="store_true",
                    help="disable the physical-legality gate in the per-step "
                         "critic reward (default: gate ON, illegal branches "
                         "floored to --illegal-floor)")
    ap.add_argument("--illegal-floor", type=float, default=-0.5,
                    help="raw score for a physically-illegal branch (default -0.5)")
    args = ap.parse_args()
    kwargs = {k.replace("-", "_"): v for k, v in vars(args).items()}
    # module-level switches, not GRPOTrainConfig fields
    global STRICT_FORMAT_REWARD, CRITIC_LEGALITY_GATE, LEGALITY_FLOOR
    STRICT_FORMAT_REWARD = kwargs.pop("strict_format_reward", False)
    CRITIC_LEGALITY_GATE = not kwargs.pop("pure_critic", False)
    LEGALITY_FLOOR = kwargs.pop("illegal_floor", -0.5)
    return GRPOTrainConfig(**kwargs)


if __name__ == "__main__":
    main(parse_args())
