"""Step-wise SFT dataset for Qwen3-VL on scenesmith trajectories.

Each `datasets/sft/<run_id>/<stage>_<...>.json` is split into N independent
training samples — one per assistant turn. The k-th sample has:

    input  = system_prompt
           + (user) "## Task\n<task>\n\n## Memory\n<memory_k>" + image_k
    target = "<think>\n<assistant_content>\n</think>" + tool_calls (rendered
             by the chat template into <tool_call>...</tool_call> blocks)

`memory_k` is built incrementally by replaying turns[0..k-1].tool_results
through the per-stage handlers in `memory.py`. Image_k is the per_step
PNG attached to assistant turn k (None for floor_plan stage).

Loss is masked everywhere except the assistant span (between
`<|im_start|>assistant\n` and `<|im_end|>`).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory as _mem  # noqa: E402
from scene_observation import scene_image_paths, resolve_image as _resolve_image






@dataclass
class StepSample:
    stage: str
    system: str
    task_context: str
    memory_str: str
    image_path: Path | None
    assistant_content: str
    tool_calls: list[dict] | None
    source_file: str
    step_idx: int
    qa_answer: str = ""  # only used when stage == "qa"
    qa_images: list = field(default_factory=list)  # qa stage: all image paths (supports multi-image)
    scene_images: list = field(default_factory=list)  # scene stage: per-turn view paths in display order (topdown, south, ...)


def _build_qa_samples(sft_path: Path, sft: dict, repo_root: Path) -> list[StepSample]:
    """One StepSample from a QA JSON (stage == 'qa', no memory, no tool calls)."""
    system = sft.get("system", "You are a helpful assistant. Think carefully before answering.")
    turns = sft.get("turns") or []
    question, think, answer = "", "", ""
    qa_images: list[str] = []
    for t in turns:
        if t.get("role") == "user":
            question = t.get("content") or ""
            raw_img = t.get("image")
            if raw_img:
                # image may be a single path (str) or a list of paths (multi-image);
                # keep however many the sample ships with — resolve each, keep order.
                raws = raw_img if isinstance(raw_img, list) else [raw_img]
                for r in raws:
                    p = _resolve_image(r, repo_root)
                    if p is not None:
                        qa_images.append(str(p))
        elif t.get("role") == "assistant":
            think  = (t.get("content") or "").strip()
            answer = (t.get("answer") or "").strip()
    if not question or not answer:
        return []
    return [StepSample(
        stage="qa",
        system=system,
        task_context=question,
        memory_str="",
        image_path=Path(qa_images[0]) if qa_images else None,
        assistant_content=think,
        tool_calls=None,
        source_file=sft_path.name,
        step_idx=0,
        qa_answer=answer,
        qa_images=qa_images,
    )]


def _build_critic_samples(sft_path: Path, sft: dict, repo_root: Path) -> list[StepSample]:
    """One StepSample from a critic ranking JSON (stage == 'critic').

    N annotated top-down renders + one ranking prompt ->
    ``<reasoning>...</reasoning>\\n<answer>LETTERS</answer>``.

    The images are shown in list order; the letters A/B/C/D in the reasoning
    and answer are POSITIONAL slot labels (1st image = SCENE A, 2nd = SCENE B,
    ...), so the list order must be preserved verbatim. There is NO system
    turn — the ranking prompt lives in the user text block, matching how the
    critic was trained/evaluated in src_critic.
    """
    prompt = (sft.get("prompt") or sft.get("system") or "").strip()
    reasoning = (sft.get("reasoning") or "").strip()
    answer = (sft.get("answer") or "").strip()
    imgs: list[str] = []
    for raw in (sft.get("images") or []):
        p = _resolve_image(raw, repo_root)
        if p is None:
            # A missing render would desync the positional letter labels — drop
            # the whole ranking rather than train on a scrambled slot mapping.
            raise FileNotFoundError(f"{sft_path}: missing Critic image {raw}")
        imgs.append(str(p))
    if not prompt or not reasoning or not answer or not imgs:
        return []
    return [StepSample(
        stage="critic",
        system="",
        task_context=prompt,
        memory_str="",
        image_path=Path(imgs[0]),
        assistant_content=reasoning,
        tool_calls=None,
        source_file=sft_path.name,
        step_idx=0,
        qa_answer=answer,
        qa_images=imgs,
    )]


def build_step_samples(sft_path: Path, repo_root: Path) -> list[StepSample]:
    """Walk one SFT JSON, replay tool_results into memory, emit one StepSample per assistant turn."""
    sft = json.loads(sft_path.read_text(encoding="utf-8"))
    stage = sft.get("stage")
    if stage == "qa":
        return _build_qa_samples(sft_path, sft, repo_root)
    if stage == "critic":
        return _build_critic_samples(sft_path, sft, repo_root)
    if stage not in _mem.HANDLERS:
        raise ValueError(f"unknown stage in {sft_path.name}: {stage!r}")
    handler = _mem.HANDLERS[stage]
    system_prompt = sft["system"]
    task_context = sft.get("task_context", {}).get("scene_description", "")

    contexts = _mem.memory_before_turns(sft)
    samples: list[StepSample] = []
    step_idx = 0
    turns = sft["turns"]

    for i, t in enumerate(turns):
        role = t.get("role")
        if role == "assistant":
            memory_str = contexts[i]
            # A turn may ship one view (`image`, str) or several (`images`, list,
            # e.g. [top-down, south]); preserve order and fail on missing images.
            scene_images = scene_image_paths(t, repo_root)
            from scene_observation import validate_scene_views
            if stage != "floor_plan":
                if not scene_images:
                    # Terminal observations not captured by the source are excluded.
                    if not t.get("tool_calls") or all(c.get("name") == "check_physics" for c in t.get("tool_calls", [])):
                        continue
                validate_scene_views(scene_images)
            image_path = Path(scene_images[0]) if scene_images else None
            samples.append(StepSample(
                stage=stage,
                system=system_prompt,
                task_context=task_context,
                memory_str=memory_str,
                image_path=image_path,
                assistant_content=(t.get("content") or "").strip(),
                tool_calls=(t.get("tool_calls") or None),
                source_file=sft_path.name,
                step_idx=step_idx,
                scene_images=scene_images,
            ))
            step_idx += 1

    return samples


def discover_step_samples(sft_root: Path, repo_root: Path) -> list[StepSample]:
    """Walk datasets/sft/<run_id>/*.json and produce all step samples."""
    samples: list[StepSample] = []
    for sft_file in sorted(sft_root.rglob("*.json")):
        if sft_file.name == "compression_report.json":
            continue
        document = json.loads(sft_file.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "stage" not in document:
            continue  # metadata/report JSON, not an episode
        samples.extend(build_step_samples(sft_file, repo_root))
    scene_no_img = sum(1 for s in samples
                       if s.image_path is None and s.stage not in ("floor_plan", "qa"))
    qa_no_img = sum(1 for s in samples if s.stage == "qa" and s.image_path is None)
    qa_total  = sum(1 for s in samples if s.stage == "qa")
    critic_total = sum(1 for s in samples if s.stage == "critic")
    print(f"[init] {len(samples)} step samples found "
          f"(scene missing image: {scene_no_img}; "
          f"QA: {qa_total} total, {qa_no_img} text-only; "
          f"critic: {critic_total})", flush=True)
    return samples


def _build_messages(s: StepSample, no_thinking: bool = False) -> tuple[list[dict], list[dict]]:
    """Return (full_messages, prompt_messages_for_loss_mask).

    When no_thinking=True, the assistant message OMITS the <think>...</think>
    reasoning prefix — the assistant's content field is set to empty string
    so the chat template emits only the <tool_call> blocks (or just <|im_end|>
    for summary turns). This trains a "tool-calls-only" agent.

    For stage == "qa": user content is the raw question (no Task/Memory wrapper);
    assistant content is <think>reasoning</think>\\n<answer>answer</answer>.

    For stage == "critic": no system turn; user content is the N ranking images
    (in slot order A,B,C,D) followed by the ranking prompt; assistant content is
    <reasoning>...</reasoning>\\n<answer>LETTERS</answer>.
    """
    if s.stage == "critic":
        asst_content = (
            f"<reasoning>\n{s.assistant_content}\n</reasoning>\n"
            f"<answer>{s.qa_answer}</answer>"
        )
        user_content_c: list[dict] = []
        for im in (s.qa_images or []):
            user_content_c.append({"type": "image", "image": im})
        user_content_c.append({"type": "text", "text": s.task_context})
        full = [
            {"role": "user",      "content": user_content_c},
            {"role": "assistant", "content": asst_content},
        ]
        return full, full[:-1]

    if s.stage == "qa":
        think = (s.assistant_content or "").strip()
        if no_thinking or not think:
            # No reasoning available (e.g. ViRL39K) — emit the answer directly,
            # with no <think> block, matching a system prompt that asks only for
            # <answer>...</answer>.
            asst_content = f"<answer>\n{s.qa_answer}\n</answer>"
        else:
            asst_content = (
                f"<think>\n{s.assistant_content}\n</think>\n"
                f"<answer>\n{s.qa_answer}\n</answer>"
            )
        user_content_qa: list[dict] = []
        qa_imgs = s.qa_images or ([str(s.image_path)] if s.image_path is not None else [])
        for im in qa_imgs:
            user_content_qa.append({"type": "image", "image": im})
        user_content_qa.append({"type": "text", "text": s.task_context})
        full = [
            {"role": "system",    "content": s.system},
            {"role": "user",      "content": user_content_qa},
            {"role": "assistant", "content": asst_content},
        ]
        return full, full[:-1]

    user_text = f"## Task\n{s.task_context}\n\n## Memory\n{s.memory_str}"
    user_content: list[dict] = []
    scene_imgs = s.scene_images or ([str(s.image_path)] if s.image_path is not None else [])
    for im in scene_imgs:
        user_content.append({"type": "image", "image": im})
    user_content.append({"type": "text", "text": user_text})

    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "" if no_thinking else f"<think>\n{s.assistant_content}\n</think>",
    }
    if s.tool_calls:
        assistant_msg["tool_calls"] = [
            {"type": "function",
             "function": {"name": tc["name"], "arguments": tc.get("arguments") or {}}}
            for tc in s.tool_calls
        ]

    full = [
        {"role": "system", "content": s.system},
        {"role": "user", "content": user_content},
        assistant_msg,
    ]
    prompt = full[:-1]
    return full, prompt


class StepwiseSFTDataset(Dataset):
    """One example == one assistant decision (a "step").

    Args:
        sft_root: datasets/sft/ root (containing per-run subdirs).
        repo_root: repo root, used to resolve assistant.image (which is
            stored as a repo-relative path in the SFT JSON).
        processor: Qwen3VLProcessor instance (AutoProcessor.from_pretrained).
        max_length: token cap; samples longer than this raise.
    """

    def __init__(self, sft_root: Path, repo_root: Path, processor,
                 max_length: int = 8192, no_thinking: bool = False):
        self.processor = processor
        self.no_thinking = no_thinking
        self.repo_root = repo_root
        self.max_length = max_length
        self.samples = discover_step_samples(sft_root, repo_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        full_msgs, prompt_msgs = _build_messages(s, no_thinking=self.no_thinking)

        full_text = self.processor.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        # Load EVERY image referenced by the message so the count matches the
        # image placeholders _build_messages emitted (qa can be multi-image).
        if s.stage in ("qa", "critic"):
            img_paths = s.qa_images or ([str(s.image_path)] if s.image_path is not None else [])
        else:
            img_paths = s.scene_images or ([str(s.image_path)] if s.image_path is not None else [])
        images = [Image.open(p).convert("RGB") for p in img_paths] or None

        full_inputs = self.processor(
            text=[full_text], images=images, return_tensors="pt", padding=False
        )
        prompt_inputs = self.processor(
            text=[prompt_text], images=images, return_tensors="pt", padding=False
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        full_len = int(input_ids.shape[0])

        if full_len > self.max_length:
            raise RuntimeError(
                f"sample {s.source_file} step={s.step_idx} has {full_len} tokens > "
                f"max_length={self.max_length}. "
                f"({'QA: increase --max-length or shorten think traces' if s.stage == 'qa' else 'increase --max-length or shrink images via --max-pixels'})"
            )
        if prompt_len >= full_len:
            raise RuntimeError(
                f"sample {s.source_file} step={s.step_idx}: prompt_len={prompt_len} "
                f">= full_len={full_len}. assistant span empty — chat template bug?"
            )

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # Qwen3-VL needs mm_token_type_ids for M-RoPE (0=text, 1=image).
        # The processor always returns it, even for image-less inputs.
        if "mm_token_type_ids" in full_inputs:
            out["mm_token_type_ids"] = full_inputs["mm_token_type_ids"][0]
        if "pixel_values" in full_inputs:
            out["pixel_values"] = full_inputs["pixel_values"]
        if "image_grid_thw" in full_inputs:
            out["image_grid_thw"] = full_inputs["image_grid_thw"]
        return out


@dataclass
class StepwiseCollator:
    """Right-pad text fields, concat per-image vision fields across the batch."""
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        bsz = len(batch)

        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        has_mm_type = any("mm_token_type_ids" in b for b in batch)
        mm_token_type_ids = torch.zeros((bsz, max_len), dtype=torch.long) if has_mm_type else None

        for i, b in enumerate(batch):
            L = b["input_ids"].size(0)
            input_ids[i, :L] = b["input_ids"]
            attention_mask[i, :L] = b["attention_mask"]
            labels[i, :L] = b["labels"]
            if mm_token_type_ids is not None and "mm_token_type_ids" in b:
                mm_token_type_ids[i, :L] = b["mm_token_type_ids"]

        out: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if mm_token_type_ids is not None:
            out["mm_token_type_ids"] = mm_token_type_ids
        pvs = [b["pixel_values"] for b in batch if "pixel_values" in b]
        if pvs:
            out["pixel_values"] = torch.cat(pvs, dim=0)
        gthw = [b["image_grid_thw"] for b in batch if "image_grid_thw" in b]
        if gthw:
            out["image_grid_thw"] = torch.cat(gthw, dim=0)
        return out
