#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests
import torch
from PIL import Image

from src.runtime import ROOT as _REPO_ROOT, SCENESMITH_ROOT

from transformers import AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)

# Same branch points as GRPO. Non-mutating tools such as generate_assets and
# check_physics remain linear; only RoomScene object mutations are branched.
STATE_MODIFYING_TOOLS: frozenset[str] = frozenset({
    "add_furniture_to_scene_tool",
    "move_furniture_tool",
    "snap_to_object_tool",
    "remove_furniture_tool",
    "rescale_furniture_tool",
    "place_wall_object",
    "move_wall_object",
    "remove_wall_object",
    "rescale_wall_object",
    "place_ceiling_object",
    "move_ceiling_object",
    "remove_ceiling_object",
    "rescale_ceiling_object",
})
ASSET_GENERATION_TOOLS: frozenset[str] = frozenset({
    "generate_assets",
    "generate_wall_assets",
    "generate_ceiling_assets",
})


def parse_assistant_text(text: str) -> tuple[str | None, list[dict]]:
    think_m = _THINK_RE.search(text)
    think = think_m.group(1).strip() if think_m else None
    tool_calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        body = m.group(1).strip()
        try:
            tc = json.loads(body)
            tool_calls.append({"name": tc.get("name"),
                               "arguments": tc.get("arguments") or {}})
        except json.JSONDecodeError:
            # Include a sentinel 'name' so the many downstream `tc["name"]`
            # accesses (logging, handler dispatch) never KeyError on a
            # truncated/malformed tool call — it's caught as a parse error.
            tool_calls.append({"_parse_error": True, "_raw": body,
                               "name": "_parse_error"})
    return think, tool_calls


class ModelPolicy:
    def __init__(self, model_dir: Path, max_new_tokens: int = 8192,
                 device: str = "cuda:0", name: str = "policy"):
        print(f"[init] loading {name} model from {model_dir}", file=sys.stderr)
        self.name = name
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        from src.scene_observation import configure_image_budget
        configure_image_budget(self.processor)
        print(f"[init] {name} processor ready", file=sys.stderr)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16,
        )
        print(f"[init] {name} weights loaded; moving to {self.device}",
              file=sys.stderr)
        self.model = self.model.to(self.device)
        print(f"[init] {name} moved to {self.device}", file=sys.stderr)
        self.model.eval()
        self.model.config.use_cache = True
        self.max_new_tokens = max_new_tokens
        n_b = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[init] {name} model on {self.model.device} "
              f"({n_b:.2f}B params)", file=sys.stderr)

    @torch.no_grad()
    def generate(self, obs: dict, *, do_sample: bool = True,
                 temperature: float = 0.7, top_p: float = 0.9) -> dict:
        messages = obs["messages"]
        from src.scene_observation import message_image_paths, validate_scene_views
        if obs.get("stage") in ("furniture", "wall", "ceiling"):
            validate_scene_views(message_image_paths(messages))
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        img_path = obs.get("image_path")
        from src.scene_observation import load_message_images
        images = load_message_images(messages)
        inputs = self.processor(text=[text], images=images,
                                return_tensors="pt").to(self.model.device)
        pad_id = (self.processor.tokenizer.pad_token_id
                  or self.processor.tokenizer.eos_token_id)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        out = self.model.generate(
            **inputs, **gen_kwargs,
        )
        prompt_len = inputs["input_ids"].shape[1]
        raw = self.processor.tokenizer.decode(out[0, prompt_len:],
                                              skip_special_tokens=False)
        think, tool_calls = parse_assistant_text(raw)
        return {"raw": raw, "think": think, "tool_calls": tool_calls}


def _dedupe_tool_calls(stage: str, step_idx: int,
                       tool_calls: list[dict]) -> list[dict]:
    # Small VL models occasionally echo a tool_call verbatim within the same
    # turn. Sending the duplicate to scenesmith can create asset_id collisions.
    seen_sigs: set[str] = set()
    deduped: list[dict] = []
    for tc in tool_calls:
        sig = ((tc.get("name") or "") + "|" +
               json.dumps(tc.get("arguments") or {},
                          sort_keys=True, ensure_ascii=False))
        if sig in seen_sigs:
            print(f"[{stage}][step {step_idx}] [dedup] dropping duplicate "
                  f"{tc.get('name')}({sig[:80]}...)", file=sys.stderr)
            continue
        seen_sigs.add(sig)
        deduped.append(tc)
    return deduped


def _split_batched_asset_generation(stage: str, step_idx: int,
                                     tool_calls: list[dict]) -> list[dict]:
    expanded: list[dict] = []
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("arguments") or {}
        descriptions = args.get("object_descriptions")
        if (
            name not in ASSET_GENERATION_TOOLS
            or not isinstance(descriptions, list)
            or len(descriptions) <= 1
        ):
            expanded.append(tc)
            continue

        n = len(descriptions)
        required = ("object_descriptions", "short_names", "desired_dimensions")
        if any(not isinstance(args.get(k), list) or len(args[k]) != n
               for k in required):
            print(f"[{stage}][step {step_idx}] [asset split warn] "
                  f"cannot split malformed {name} batch", file=sys.stderr)
            expanded.append(tc)
            continue

        print(f"[{stage}][step {step_idx}] [asset split] {name}: "
              f"{n} assets -> {n} single-asset calls", file=sys.stderr)
        for i in range(n):
            one_args: dict = {}
            for key, value in args.items():
                if isinstance(value, list) and len(value) == n:
                    one_args[key] = [value[i]]
                else:
                    one_args[key] = value
            expanded.append({"name": name, "arguments": one_args})
    return expanded


def _has_asset_generation(tool_calls: list[dict]) -> bool:
    return any(tc.get("name") in ASSET_GENERATION_TOOLS for tc in tool_calls)


def _is_state_modifying(tool_calls: list[dict]) -> bool:
    return any(tc.get("name") in STATE_MODIFYING_TOOLS for tc in tool_calls)


def _print_tool_calls(stage: str, step_idx: int,
                      tool_calls: list[dict],
                      prefix: str = "call") -> None:
    for j, tc in enumerate(tool_calls):
        args_str = json.dumps(tc.get("arguments") or {}, ensure_ascii=False)
        preview = args_str if len(args_str) <= 120 else args_str[:117] + "..."
        print(f"[{stage}][step {step_idx}][{prefix} {j}] "
              f"{tc.get('name')}({preview})", file=sys.stderr)


class BranchRanker:
    def _render_branch(self, scene_dir: str, label: str) -> str | None:
        out = Path(scene_dir) / "_topdown.png"
        if not out.is_file():
            raise FileNotFoundError(f"Missing current-state Critic observation: {out}")
        return str(out)


class CriticRanker(BranchRanker):
    def __init__(self, model_dir: Path, *, device: str = "cuda:1",
                 max_new_tokens: int = 2048,
                 legality_gate: bool = True,
                 illegal_floor: float = -0.5):
        print(f"[init] loading critic model from {model_dir}", file=sys.stderr)
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        from src.scene_observation import configure_image_budget
        configure_image_budget(self.processor)
        print("[init] critic processor ready", file=sys.stderr)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16,
        )
        print(f"[init] critic weights loaded; moving to {device}",
              file=sys.stderr)
        self.model = self.model.to(device)
        print(f"[init] critic moved to {device}", file=sys.stderr)
        self.model.eval()
        self.model.config.use_cache = True
        self.max_new_tokens = max_new_tokens
        self.legality_gate = legality_gate
        self.illegal_floor = illegal_floor
        n_b = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[init] critic model on {self.model.device} "
              f"({n_b:.2f}B params)", file=sys.stderr)

    def choose(self, branch_records: list[dict]) -> dict:
        from src_critic.grpo_advantage import (
            advantage_from_critic,
            advantage_from_critic_with_legality,
            legality_flags_from_scenes,
        )

        labels = list("ABCD")[:len(branch_records)]
        scene_dirs = [
            b.get("scene_dir_host") or b.get("scene_dir_container")
            for b in branch_records
        ]
        topdown = [
            self._render_branch(sd, labels[i]) if sd else None
            for i, sd in enumerate(scene_dirs)
        ]
        valid_idx = [i for i, p in enumerate(topdown) if p is not None]
        if len(valid_idx) < 2:
            chosen = valid_idx[0] if valid_idx else 0
            return {
                "chosen_idx": chosen,
                "reason": "fewer_than_two_renderable_branches",
                "ranking": [labels[chosen]] if labels else [],
                "raw_score": {labels[i]: 0.0 for i in range(len(labels))},
                "topdown": topdown,
                "legal": [True] * len(branch_records),
                "violations": [[] for _ in branch_records],
            }

        valid_paths = [topdown[i] for i in valid_idx]
        valid_labels = list("ABCD")[:len(valid_idx)]
        legal_flags = [True] * len(branch_records)
        violations: list[list] = [[] for _ in branch_records]
        if self.legality_gate:
            try:
                legal_flags, violations = legality_flags_from_scenes(scene_dirs)
            except Exception as exc:  # noqa: BLE001
                print(f"[critic] legality check failed: {exc}", file=sys.stderr)
                legal_flags = [True] * len(branch_records)
                violations = [[("unchecked", repr(exc))]
                              for _ in branch_records]
            valid_legal = [legal_flags[i] for i in valid_idx]
            res = advantage_from_critic_with_legality(
                self.model, self.processor, valid_paths, valid_legal,
                method="borda", illegal_floor=self.illegal_floor,
                max_new_tokens=self.max_new_tokens, return_raw=True,
            )
        else:
            res = advantage_from_critic(
                self.model, self.processor, valid_paths,
                method="borda", max_new_tokens=self.max_new_tokens,
                return_raw=True,
            )

        raw_full = {labels[i]: self.illegal_floor for i in range(len(labels))}
        for out_pos, orig_i in enumerate(valid_idx):
            raw_full[labels[orig_i]] = res.raw_score.get(
                valid_labels[out_pos], self.illegal_floor,
            )

        # res labels are over the valid subset. Map back to original branch ids.
        rank_full: list[str] = []
        for lab in res.ranking:
            if lab in valid_labels:
                rank_full.append(labels[valid_idx[valid_labels.index(lab)]])
        rank_full += [
            labels[i] for i in range(len(labels)) if labels[i] not in rank_full
        ]
        if rank_full:
            chosen = labels.index(rank_full[0])
        else:
            chosen = max(range(len(labels)), key=lambda i: raw_full[labels[i]])

        print(f"[critic] rank={'>'.join(rank_full) or '(parse fail)'} "
              f"chosen=branch_{chosen:02d} scores={raw_full}",
              file=sys.stderr)
        for i, (ok, vs) in enumerate(zip(legal_flags, violations)):
            if not ok:
                print(f"[critic] branch_{i:02d} ILLEGAL: {vs}",
                      file=sys.stderr)

        return {
            "chosen_idx": chosen,
            "ranking": rank_full,
            "raw_score": raw_full,
            "method": res.method,
            "raw_text": res.raw_text,
            "topdown": topdown,
            "legal": legal_flags,
            "violations": violations,
        }


def _parse_api_ranking(raw: str, labels: list[str]) -> tuple[list[str], str]:
    """Parse a strict JSON ranking, with a text fallback for API gateways."""
    text = raw.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group(0))

    reason = ""
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        ranking = payload.get("ranking") if isinstance(payload, dict) else None
        if isinstance(ranking, str):
            ranking = re.findall(r"[A-D]", ranking.upper())
        if isinstance(ranking, list):
            normalized = [str(label).strip().upper() for label in ranking]
            if len(normalized) == len(labels) and set(normalized) == set(labels):
                reason = str(payload.get("reason") or payload.get("reasoning") or "")
                return normalized, reason

    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.I)
    answer_text = answer.group(1) if answer else text
    ranking = re.findall(r"\b[A-D]\b", answer_text.upper())
    if len(ranking) == len(labels) and set(ranking) == set(labels):
        return ranking, reason
    compact = [char for char in answer_text.upper() if char in labels]
    if len(compact) == len(labels) and set(compact) == set(labels):
        return compact, reason
    raise RuntimeError(f"GPT critic returned an invalid ranking: {raw[:500]!r}")


class GPTCriticRanker(BranchRanker):
    """Rank rendered furniture branches with GPT through an OpenAI API."""

    RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

    def __init__(self, *, api_url: str, api_key: str, model: str = "gpt-5.5",
                 protocol: str = "responses", max_output_tokens: int = 2048,
                 reasoning_effort: str = "medium", image_detail: str = "high",
                 timeout: float = 300.0, max_retries: int = 5,
                 legality_gate: bool = True, illegal_floor: float = -0.5,
                 session: requests.Session | None = None):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.protocol = protocol
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail
        self.timeout = timeout
        self.max_retries = max_retries
        self.legality_gate = legality_gate
        self.illegal_floor = illegal_floor
        self.s = session or requests.Session()
        print(f"[init] GPT critic model={model} protocol={protocol} "
              f"url={api_url}", file=sys.stderr)

    @staticmethod
    def _ranking_prompt(labels: list[str]) -> str:
        names = ", ".join(f"SCENE {label}" for label in labels)
        return (
            f"You are an expert indoor-scene layout critic. You see {len(labels)} "
            f"top-down renderings of the same room labelled {names}. Room geometry, "
            "doors, and windows are identical; only the furniture arrangement differs. "
            "Rank every scene from BEST to WORST by physical validity, collision and "
            "boundary avoidance, door/window accessibility, circulation, functional "
            "grouping, interaction orientation, scale, and overall spatial balance. "
            "Return each label exactly once. Respond only with a JSON object using "
            "the form {\"ranking\":[\"A\",\"B\"],\"reason\":\"brief explanation\"}."
        )

    @staticmethod
    def _schema(labels: list[str]) -> dict:
        return {
            "type": "object",
            "properties": {
                "ranking": {
                    "type": "array",
                    "items": {"type": "string", "enum": labels},
                    "minItems": len(labels),
                    "maxItems": len(labels),
                },
                "reason": {"type": "string"},
            },
            "required": ["ranking", "reason"],
            "additionalProperties": False,
        }

    @staticmethod
    def _data_url(path: str) -> str:
        image_path = Path(path)
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _request_body(self, image_paths: list[str], labels: list[str]) -> dict:
        prompt = self._ranking_prompt(labels)
        schema = self._schema(labels)
        if self.protocol == "responses":
            content = [{"type": "input_text", "text": prompt}]
            content += [
                {"type": "input_image", "image_url": self._data_url(path),
                 "detail": self.image_detail}
                for path in image_paths
            ]
            body = {
                "model": self.model,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": self.max_output_tokens,
                "text": {"format": {
                    "type": "json_schema",
                    "name": "scene_branch_ranking",
                    "strict": True,
                    "schema": schema,
                }},
            }
            if self.reasoning_effort != "default":
                body["reasoning"] = {"effort": self.reasoning_effort}
            return body

        content = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {
                "url": self._data_url(path), "detail": self.image_detail,
            }}
            for path in image_paths
        ]
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "scene_branch_ranking",
                    "strict": True,
                    "schema": schema,
                },
            },
        }

    def _extract_text(self, payload: dict) -> str:
        if self.protocol == "chat_completions":
            return str(
                ((payload.get("choices") or [{}])[0].get("message") or {})
                .get("content") or ""
            )
        if payload.get("output_text"):
            return str(payload["output_text"])
        texts: list[str] = []
        for output in payload.get("output") or []:
            for content in output.get("content") or []:
                if content.get("type") in {"output_text", "text"}:
                    texts.append(str(content.get("text") or ""))
        return "\n".join(texts)

    def _rank(self, image_paths: list[str], labels: list[str]) -> tuple[list[str], str, str]:
        body = self._request_body(image_paths, labels)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.s.post(
                    self.api_url, headers=headers, json=body,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    raw = self._extract_text(response.json())
                    ranking, reason = _parse_api_ranking(raw, labels)
                    return ranking, reason, raw
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code == 400 and self.protocol == "chat_completions":
                    if "max_completion_tokens" in body:
                        body["max_tokens"] = body.pop("max_completion_tokens")
                        print("[gpt-critic] gateway rejected max_completion_tokens; "
                              "retrying with max_tokens", file=sys.stderr)
                        continue
                    if "response_format" in body:
                        body.pop("response_format")
                        print("[gpt-critic] gateway rejected structured output; "
                              "retrying with prompt-enforced JSON", file=sys.stderr)
                        continue
                if response.status_code not in self.RETRYABLE_STATUS:
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
            if attempt < self.max_retries:
                delay = min(30.0, 2.0 * (attempt + 1))
                print(f"[gpt-critic] attempt {attempt + 1} failed: "
                      f"{last_error}; retrying in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay)
        raise RuntimeError(f"GPT critic API failed: {last_error}")

    def choose(self, branch_records: list[dict]) -> dict:
        from src_critic.grpo_advantage import borda_score, legality_flags_from_scenes

        labels = list("ABCD")[:len(branch_records)]
        scene_dirs = [
            branch.get("scene_dir_host") or branch.get("scene_dir_container")
            for branch in branch_records
        ]
        topdown = [
            self._render_branch(scene_dir, labels[index]) if scene_dir else None
            for index, scene_dir in enumerate(scene_dirs)
        ]
        renderable = [index for index, path in enumerate(topdown) if path is not None]
        legal_flags = [True] * len(branch_records)
        violations: list[list] = [[] for _ in branch_records]
        if self.legality_gate:
            try:
                legal_flags, violations = legality_flags_from_scenes(scene_dirs)
            except Exception as exc:  # noqa: BLE001
                print(f"[gpt-critic] legality check failed: {exc}", file=sys.stderr)
                violations = [[("unchecked", repr(exc))] for _ in branch_records]

        eligible = [
            index for index in renderable
            if legal_flags[index] or not self.legality_gate
        ]
        raw_text = ""
        reason = ""
        if len(eligible) >= 2:
            eligible_labels = [labels[index] for index in eligible]
            ranking, reason, raw_text = self._rank(
                [topdown[index] for index in eligible], eligible_labels,
            )
        elif len(eligible) == 1:
            ranking = [labels[eligible[0]]]
            reason = "only_one_renderable_legal_branch"
        else:
            fallback = renderable[0] if renderable else 0
            ranking = [labels[fallback]] if labels else []
            reason = "no_renderable_legal_branch"

        ranking += [label for label in labels if label not in ranking]
        chosen = labels.index(ranking[0]) if ranking else 0
        shown_scores = borda_score(ranking[:max(1, len(eligible))], max(1, len(eligible)))
        raw_score = {label: self.illegal_floor for label in labels}
        raw_score.update(shown_scores)

        print(f"[gpt-critic] rank={'>'.join(ranking)} "
              f"chosen=branch_{chosen:02d} model={self.model}", file=sys.stderr)
        return {
            "chosen_idx": chosen,
            "ranking": ranking,
            "raw_score": raw_score,
            "method": "gpt-api+borda",
            "raw_text": raw_text,
            "reason": reason,
            "api_model": self.model,
            "api_protocol": self.protocol,
            "topdown": topdown,
            "legal": legal_flags,
            "violations": violations,
        }


class ToolEnvClient:
    def __init__(self, base_url: str, timeout: float = 1800):
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Connection": "close"})
        self.timeout = timeout

    def reset_http_session(self) -> None:
        try:
            self.s.close()
        except Exception:
            pass
        self.s = requests.Session()
        self.s.headers.update({"Connection": "close"})

    def _post(self, path: str, body: dict | None = None) -> dict:
        try:
            r = self.s.post(f"{self.base}{path}", json=body or {},
                            timeout=self.timeout)
        except requests.RequestException as exc:
            self.reset_http_session()
            raise RuntimeError(f"{path} request failed: {exc}") from exc
        if r.status_code >= 400:
            raise RuntimeError(f"{path} → {r.status_code}: {r.text[:400]}")
        return r.json() if r.text else {}

    def healthz(self) -> dict:
        r = self.s.get(f"{self.base}/healthz", timeout=10)
        r.raise_for_status()
        return r.json()

    def wait_healthy(self, timeout_s: float = 600.0,
                     interval_s: float = 2.0) -> dict:
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                return self.healthz()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                self.reset_http_session()
                time.sleep(interval_s)
        raise RuntimeError(f"server did not become healthy: {last_err}")

    def new_session(
        self,
        prompt: str,
        config: str,
        results_root: str,
        *,
        stub_assets: bool = False,
    ) -> dict:
        return self._post("/session", {
            "prompt": prompt,
            "config": config,
            "results_root": results_root,
            "stub_assets": stub_assets,
        })

    def start_stage(self, stage: str, render_gpu_id: int | None = None,
                    skip_render: bool = False,
                    stub_assets: bool = False) -> dict:
        return self._post("/session/stage", {
            "stage": stage,
            "render_gpu_id": render_gpu_id,
            "skip_render": skip_render,
            "stub_assets": stub_assets,
        })

    def handover(self, room_id: str | None) -> dict:
        return self._post("/session/handover", {"room_id": room_id})

    def observation(self) -> dict:
        return self._post("/session/observation")

    def step(self, tool_calls: list[dict]) -> list[dict]:
        return self._post("/session/step", {"tool_calls": tool_calls})["results"]

    def branch_eval(self, branches: list[dict]) -> dict:
        return self._post("/session/branch/evaluate", {"branches": branches})

    def branch_choose(self, branch_id: str, chosen_idx: int) -> dict:
        return self._post("/session/branch/choose", {
            "branch_id": branch_id, "chosen_idx": chosen_idx,
        })

    def finalize(self) -> dict:
        return self._post("/session/finalize")

    def cleanup_env(self) -> dict:
        return self._post("/session/cleanup")

    def close(self) -> dict:
        return self._post("/session/close")

    def admin_exit(self) -> dict:
        return self._post("/admin/exit")


def _slugify(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return (s[:max_len] or "rollout").rstrip("_")


def _load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        prompt: str | None = None
        try:
            obj = json.loads(s)
            if isinstance(obj, str):
                prompt = obj
            elif isinstance(obj, dict):
                for key in ("prompt", "text", "instruction"):
                    if isinstance(obj.get(key), str):
                        prompt = obj[key]
                        break
        except Exception:
            pass
        if prompt is None:
            if s.startswith("{") and s.endswith("}") and ":" not in s:
                prompt = s[1:-1].strip()
            else:
                prompt = s
        if prompt:
            prompts.append(prompt)
    return prompts


def run_stage(env: ToolEnvClient, policy: ModelPolicy, stage: str,
              steps_dir_host: Path, max_steps: int,
              trace_dir: Path | None = None,
              trace_entries: list[dict] | None = None,
              critic: BranchRanker | None = None,
              n_branches: int = 4,
              branch_stages: set[str] | None = None,
              temperature: float = 0.7,
              branch_temperature: float = 0.8,
              strict_errors: bool = False) -> list[dict]:
    rollout_log: list[dict] = []
    io_log: list[dict] = []
    steps_dir_host.mkdir(parents=True, exist_ok=True)
    branch_stages = branch_stages or {"furniture"}

    for _ in range(max_steps):
        obs = env.observation()
        step_idx = obs["step_idx"]

        print(f"\n[{stage}][step {step_idx}] generating...", file=sys.stderr)
        out = policy.generate(obs, temperature=temperature)
        raw, think, tool_calls = out["raw"], out["think"], out["tool_calls"]
        preview = (think or "(no <think>)").replace("\n", " ")[:140]
        print(f"[{stage}][step {step_idx}] think: {preview}", file=sys.stderr)

        io_entry: dict = {
            "turn": step_idx,
            "input": {"task":       obs["room_prompt"],
                      "memory":     obs["memory_str"],
                      "image_path": obs.get("image_path")},
            "output": {"raw": raw, "think": think, "tool_calls": tool_calls},
            "tool_results": [],
        }

        trace_image_rel: str | None = None
        if trace_dir is not None and obs.get("image_path"):
            src = Path(obs["image_path"])
            if src.exists():
                dst_name = f"{stage}_turn{step_idx:03d}{src.suffix or '.png'}"
                dst = trace_dir / "images" / dst_name
                try:
                    shutil.copyfile(src, dst)
                    trace_image_rel = f"images/{dst_name}"
                except Exception as e:
                    print(f"[trace warn] copy image: {e}", file=sys.stderr)
        trace_entry: dict | None = None
        if trace_entries is not None:
            trace_entry = {
                "stage":        stage,
                "turn":         step_idx,
                "image":        trace_image_rel,
                "think":        think,
                "tool_calls":   tool_calls,
                "tool_results": [],
            }
            trace_entries.append(trace_entry)

        if not tool_calls:
            print(f"[{stage}] no <tool_call> — stage done", file=sys.stderr)
            rollout_log.append({"step": step_idx, "think": think,
                                "tool_calls": [], "tool_results": []})
            io_entry["stop_reason"] = "no_tool_call"
            io_log.append(io_entry)
            break

        tool_calls = _dedupe_tool_calls(stage, step_idx, tool_calls)
        tool_calls = _split_batched_asset_generation(stage, step_idx, tool_calls)
        io_entry["output"]["tool_calls"] = tool_calls
        if trace_entry is not None:
            trace_entry["tool_calls"] = tool_calls
        if not tool_calls:
            print(f"[{stage}] no tool calls after dedup — stage done",
                  file=sys.stderr)
            rollout_log.append({"step": step_idx, "think": think,
                                "tool_calls": [], "tool_results": []})
            io_entry["stop_reason"] = "no_tool_call_after_dedup"
            io_log.append(io_entry)
            break

        should_branch = (
            critic is not None
            and stage in branch_stages
            and _is_state_modifying(tool_calls)
            and n_branches >= 2
        )
        if should_branch:
            print(f"[{stage}][step {step_idx}] state-changing step; "
                  f"sampling {n_branches} branches", file=sys.stderr)
            candidates: list[dict] = [{
                "raw": raw, "think": think, "tool_calls": tool_calls,
            }]
            _print_tool_calls(stage, step_idx, tool_calls, prefix="branch0")
            for bi in range(1, n_branches):
                b_out = policy.generate(
                    obs, temperature=branch_temperature,
                )
                b_calls = _dedupe_tool_calls(
                    stage, step_idx, b_out["tool_calls"],
                )
                b_preview = (b_out["think"] or "(no <think>)").replace(
                    "\n", " ",
                )[:140]
                print(f"[{stage}][step {step_idx}][branch {bi}] think: "
                      f"{b_preview}", file=sys.stderr)
                _print_tool_calls(stage, step_idx, b_calls,
                                  prefix=f"branch{bi}")
                candidates.append({
                    "raw": b_out["raw"],
                    "think": b_out["think"],
                    "tool_calls": b_calls,
                })

            try:
                eval_resp = env.branch_eval(candidates)
                choice = critic.choose(eval_resp["branches"])
                choose_resp = env.branch_choose(
                    eval_resp["branch_id"], int(choice["chosen_idx"]),
                )
                chosen_idx = int(choice["chosen_idx"])
                chosen_calls = candidates[chosen_idx]["tool_calls"]
                results = choose_resp.get("tool_results", [])
                if trace_entry is not None:
                    trace_entry["tool_calls"] = chosen_calls
                    trace_entry["tool_results"] = results
                    trace_entry["branch_id"] = eval_resp["branch_id"]
                    trace_entry["branch_candidates"] = eval_resp["branches"]
                    trace_entry["critic_choice"] = choice
                io_entry["output"] = {
                    "raw": candidates[chosen_idx]["raw"],
                    "think": candidates[chosen_idx]["think"],
                    "tool_calls": chosen_calls,
                    "chosen_from_branch": chosen_idx,
                }
                io_entry["tool_results"] = results
                io_entry["branch_id"] = eval_resp["branch_id"]
                io_entry["branch_candidates"] = eval_resp["branches"]
                io_entry["critic_choice"] = choice
                rollout_log.append({
                    "step": step_idx,
                    "think": candidates[chosen_idx]["think"],
                    "tool_calls": chosen_calls,
                    "tool_results": results,
                    "branch_id": eval_resp["branch_id"],
                    "branch_candidates": eval_resp["branches"],
                    "critic_choice": choice,
                })
                io_log.append(io_entry)
                if not chosen_calls:
                    print(f"[{stage}] critic chose no-tool branch — stage done",
                          file=sys.stderr)
                    break
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[{stage}] branch/critic failed; falling back to "
                      f"linear step: {exc}", file=sys.stderr)
                if trace_entry is not None:
                    trace_entry["branch_error"] = (
                        f"{exc}\n{traceback.format_exc()}"
                    )
                if strict_errors:
                    raise RuntimeError(
                        f"{stage} branch/critic failed: {exc}"
                    ) from exc

        _print_tool_calls(stage, step_idx, tool_calls)
        try:
            if _has_asset_generation(tool_calls) and len(tool_calls) > 1:
                results = []
                for single_idx, single_call in enumerate(tool_calls, start=1):
                    print(f"[{stage}][step {step_idx}] "
                          f"asset request {single_idx}/{len(tool_calls)}",
                          file=sys.stderr)
                    results.extend(env.step([single_call]))
            else:
                results = env.step(tool_calls)
        except RuntimeError as e:
            print(f"[{stage}] step HTTP failed: {e}", file=sys.stderr)
            if strict_errors:
                raise
            results = [{"name": tc.get("name"),
                        "output": f"ERROR: client/server failure: {e}"}
                       for tc in tool_calls]

        if trace_entry is not None:
            trace_entry["tool_results"] = results

        rollout_log.append({"step": step_idx, "think": think,
                            "tool_calls": tool_calls, "tool_results": results})
        io_entry["tool_results"] = results
        io_log.append(io_entry)
    else:
        print(f"[{stage}] hit max_steps={max_steps}, stopping", file=sys.stderr)

    (steps_dir_host / "rollout.json").write_text(
        json.dumps(rollout_log, indent=2, ensure_ascii=False), encoding="utf-8")
    (steps_dir_host / "turns_io.json").write_text(
        json.dumps(io_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{stage}] saved logs → {steps_dir_host}", file=sys.stderr)
    return rollout_log


def run_rollout(
    args: argparse.Namespace,
    env: ToolEnvClient,
    policy: ModelPolicy,
    critic: BranchRanker | None,
    branch_stages: set[str],
    prompt: str,
    *,
    run_label: str | None = None,
) -> None:
    trace_root = (args.trace_root if args.trace_root is not None
                  else Path(__file__).resolve().parents[1] / "results")
    trace_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_label:
        trace_prefix = f"{trace_prefix}_{_slugify(run_label, 24)}"
    trace_dir = trace_root / f"{trace_prefix}_{_slugify(prompt)}"
    (trace_dir / "images").mkdir(parents=True, exist_ok=True)
    trace_entries: list[dict] = []
    print(f"[trace] writing to {trace_dir}", file=sys.stderr)

    t_start = time.perf_counter()
    start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scene_dir_host: Path | None = None

    try:
        sess_info = env.new_session(
            prompt, args.config, args.results_root_container,
            stub_assets=args.stub_assets,
        )
        scene_dir_host = Path(sess_info["scene_dir_host"])
        print(f"[session] scene_dir_host = {scene_dir_host}", file=sys.stderr)

        st = env.start_stage(
            "floor_plan", args.render_gpu_id,
            stub_assets=args.stub_assets,
        )
        run_stage(env, policy, "floor_plan",
                  Path(st["steps_dir_host"]), args.max_steps_per_stage,
                  trace_dir=trace_dir, trace_entries=trace_entries,
                  critic=critic, n_branches=args.n_branches,
                  branch_stages=branch_stages,
                  temperature=args.temperature,
                  branch_temperature=args.branch_temperature,
                  strict_errors=args.strict_stage_failure)
        try:
            env.finalize()
        except RuntimeError as e:
            print(f"[floor_plan] finalize failed: {e}", file=sys.stderr)
            print("[chained] cannot proceed without geometry. Stopping.",
                  file=sys.stderr)
            if args.strict_stage_failure:
                raise
            return

        try:
            hand = env.handover(args.room)
            print(f"[chained] proceeding with room {hand['room_id']!r}",
                  file=sys.stderr)
        except RuntimeError as e:
            print(f"[handover] {e}", file=sys.stderr)
            if args.strict_stage_failure:
                raise
            return

        for stage, skip in [("furniture", args.skip_furniture),
                            ("wall",      args.skip_wall),
                            ("ceiling",   args.skip_ceiling)]:
            if skip:
                continue
            try:
                st = env.start_stage(
                    stage, args.render_gpu_id,
                    skip_render=args.skip_observation_render,
                    stub_assets=args.stub_assets,
                )
                run_stage(env, policy, stage,
                          Path(st["steps_dir_host"]),
                          args.max_steps_per_stage,
                          trace_dir=trace_dir, trace_entries=trace_entries,
                          critic=critic, n_branches=args.n_branches,
                          branch_stages=branch_stages,
                          temperature=args.temperature,
                          branch_temperature=args.branch_temperature,
                          strict_errors=args.strict_stage_failure)
                env.finalize()
            except RuntimeError as e:
                print(f"[{stage}] stage failed: {e}", file=sys.stderr)
                if args.strict_stage_failure:
                    raise
            finally:
                try: env.cleanup_env()
                except Exception: pass

        print(f"\n[done] all artifacts under {scene_dir_host}", file=sys.stderr)
    finally:
        elapsed = time.perf_counter() - t_start
        end_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h = int(elapsed // 3600); m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        critic_backend = args.critic_backend if args.branch_critic else None
        critic_model = None
        critic_protocol = None
        if args.branch_critic:
            if args.critic_backend == "gpt":
                critic_model = args.api_critic_model
                critic_protocol = args.api_critic_protocol
            elif args.critic_model_dir is not None:
                critic_model = str(args.critic_model_dir)
        print(f"\n[metrics] {start_human} → {end_human}  "
              f"({h:02d}:{m:02d}:{s:02d}, {elapsed:.1f}s)", file=sys.stderr)
        if scene_dir_host is not None:
            try:
                (scene_dir_host.parent / "metrics.json").write_text(json.dumps({
                    "prompt":      prompt,
                    "model_dir":   str(args.model_dir),
                    "critic_backend": critic_backend,
                    "critic_model": critic_model,
                    "critic_protocol": critic_protocol,
                    "server_url":  args.server_url,
                    "start_human": start_human,
                    "end_human":   end_human,
                    "elapsed_seconds": round(elapsed, 2),
                    "elapsed_human":   f"{h:02d}:{m:02d}:{s:02d}",
                    "run_label":       run_label,
                }, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[metrics warn] {e}", file=sys.stderr)

        try:
            (trace_dir / "trace.json").write_text(
                json.dumps({
                    "prompt":      prompt,
                    "model_dir":   str(args.model_dir),
                    "critic_backend": critic_backend,
                    "critic_model": critic_model,
                    "critic_protocol": critic_protocol,
                    "start_human": start_human,
                    "end_human":   end_human,
                    "run_label":   run_label,
                    "turns":       trace_entries,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[trace] saved → {trace_dir / 'trace.json'} "
                  f"({len(trace_entries)} turns)", file=sys.stderr)
        except Exception as e:
            print(f"[trace warn] {e}", file=sys.stderr)

        try: env.close()
        except Exception: pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Host-side rollout client — VLM locally, scenesmith tools over HTTP.",
    )
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", type=Path, default=None,
                    help="Optional batch file. Supports JSONL records with "
                         "prompt/text/instruction fields, JSON strings, or "
                         "the legacy {prompt text} lines used by prompt.jsonl.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Number of times to run each prompt when "
                         "--prompt-file is supplied.")
    ap.add_argument("--prompt-index-offset", type=int, default=0,
                    help="Offset added to prompt indices in batch run labels.")
    ap.add_argument("--always-run-label", action="store_true",
                    help="Assign a promptXX_repYY label even for one job.")
    ap.add_argument("--server-url", default="http://localhost:8000")
    ap.add_argument("--model-dir", type=Path, required=True)
    _default_results = str(Path(__file__).resolve().parents[1] / "results")
    ap.add_argument("--results-root-container", default=_default_results,
                    help="results root as seen by the tool-server process. "
                         "When running locally (no Docker) this is the same as "
                         "the host path. In Docker it is the container-side mount.")
    ap.add_argument("--config", default="scene_designer")
    ap.add_argument("--render-gpu-id", type=int, default=None)
    ap.add_argument("--max-steps-per-stage", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--room", default=None)
    ap.add_argument("--skip-furniture", action="store_true")
    ap.add_argument("--skip-wall",      action="store_true")
    ap.add_argument("--skip-ceiling",   action="store_true")
    ap.add_argument("--keep-server-alive", action="store_true",
                    help="By default we POST /admin/exit at the end so the "
                         "supervisor respawns uvicorn for the next test. Pass "
                         "this flag to skip — useful when chaining several "
                         "rollouts against one warm server.")
    ap.add_argument("--model-gpu", default=None,
                    help="CUDA_VISIBLE_DEVICES for the model process, e.g. '0,1'. "
                         "If unset, uses whatever CUDA_VISIBLE_DEVICES is in env.")
    ap.add_argument("--policy-device", default="cuda:0")
    ap.add_argument("--branch-critic", action="store_true",
                    help="At state-changing steps, sample branches and let a "
                         "critic choose which branch to execute.")
    ap.add_argument("--critic-backend", choices=("local", "gpt"),
                    default="local",
                    help="Branch ranker implementation. Default 'local' "
                         "preserves the existing checkpoint-based critic.")
    ap.add_argument("--critic-model-dir", type=Path, default=None)
    ap.add_argument("--critic-device", default="cuda:1")
    ap.add_argument("--critic-max-new-tokens", type=int, default=2048)
    ap.add_argument("--api-critic-url", default=None,
                    help="Full GPT critic endpoint. Defaults to "
                         "$GPT_CRITIC_API_URL, or the OpenAI endpoint.")
    ap.add_argument("--api-critic-key", default=None,
                    help="Defaults to $GPT_CRITIC_API_KEY or $OPENAI_API_KEY.")
    ap.add_argument("--api-critic-model", default="gpt-5.5")
    ap.add_argument("--api-critic-protocol",
                    choices=("responses", "chat_completions"),
                    default="responses")
    ap.add_argument("--api-critic-reasoning-effort",
                    choices=("default", "none", "low", "medium", "high", "xhigh"),
                    default="medium")
    ap.add_argument("--api-critic-image-detail",
                    choices=("auto", "low", "high"), default="high")
    ap.add_argument("--api-critic-timeout", type=float, default=300.0)
    ap.add_argument("--api-critic-max-retries", type=int, default=5)
    ap.add_argument("--n-branches", type=int, default=4)
    ap.add_argument("--branch-temperature", type=float, default=0.8)
    ap.add_argument("--branch-stages", default="furniture",
                    help="Comma-separated stages that may branch. The server "
                         "supports branch execution for room object stages.")
    ap.add_argument("--skip-observation-render", action="store_true",
                    help="Skip Drake/Blender topdown observation renders for "
                         "room stages. This matches GRPO collection on "
                         "headless servers; branch critic still renders "
                         "schematic topdowns from saved scene JSON.")
    ap.add_argument("--stub-assets", action="store_true",
                    help="Register bbox-only assets for layout-only prompt "
                         "sweeps. Default off preserves full asset generation.")
    ap.add_argument("--pure-critic", action="store_true",
                    help="Disable the physical-legality gate when ranking "
                         "branches. Default matches GRPO training: gate ON.")
    ap.add_argument("--illegal-floor", type=float, default=-0.5)
    ap.add_argument("--trace-root", type=Path, default=None,
                    help="directory for per-run trace files. "
                         "Defaults to <repo>/results/")
    ap.add_argument("--server-ready-timeout", type=float, default=600.0,
                    help="Seconds to wait for a supervised tool server to "
                         "come back before each rollout.")
    ap.add_argument("--run-retries", type=int, default=0,
                    help="Extra retry attempts for each rollout after a "
                         "server/client failure. Default 0 preserves the "
                         "original single-run behavior.")
    ap.add_argument("--strict-stage-failure", action="store_true",
                    help="Propagate stage/branch/server failures so batch "
                         "runs can retry instead of recording partial traces "
                         "as successful.")
    args = ap.parse_args()

    if args.model_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.model_gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.branch_critic and not (2 <= args.n_branches <= 4):
        raise SystemExit("--branch-critic supports --n-branches in [2, 4]")

    env = ToolEnvClient(args.server_url)
    try:
        h = env.healthz()
        print(f"[client] server healthz: {h}", file=sys.stderr)
    except Exception as e:
        print(f"[client] cannot reach server at {args.server_url}: {e}",
              file=sys.stderr)
        sys.exit(2)

    policy = ModelPolicy(
        args.model_dir, max_new_tokens=args.max_new_tokens,
        device=args.policy_device, name="policy",
    )
    critic: BranchRanker | None = None
    if args.branch_critic:
        if args.critic_backend == "local":
            if args.critic_model_dir is None:
                raise SystemExit(
                    "--critic-backend local requires --critic-model-dir"
                )
            critic = CriticRanker(
                args.critic_model_dir,
                device=args.critic_device,
                max_new_tokens=args.critic_max_new_tokens,
                legality_gate=not args.pure_critic,
                illegal_floor=args.illegal_floor,
            )
        else:
            api_key = (
                args.api_critic_key
                or os.environ.get("GPT_CRITIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            if not api_key:
                raise SystemExit(
                    "GPT critic requires --api-critic-key, "
                    "$GPT_CRITIC_API_KEY, or $OPENAI_API_KEY"
                )
            api_url = (
                args.api_critic_url
                or os.environ.get("GPT_CRITIC_API_URL")
            )
            if not api_url:
                api_base = os.environ.get("OPENAI_BASE_URL")
                endpoint = (
                    "responses" if args.api_critic_protocol == "responses"
                    else "chat/completions"
                )
                api_url = (
                    f"{api_base.rstrip('/')}/{endpoint}"
                    if api_base else f"https://api.openai.com/v1/{endpoint}"
                )
            critic = GPTCriticRanker(
                api_url=api_url,
                api_key=api_key,
                model=args.api_critic_model,
                protocol=args.api_critic_protocol,
                max_output_tokens=args.critic_max_new_tokens,
                reasoning_effort=args.api_critic_reasoning_effort,
                image_detail=args.api_critic_image_detail,
                timeout=args.api_critic_timeout,
                max_retries=args.api_critic_max_retries,
                legality_gate=not args.pure_critic,
                illegal_floor=args.illegal_floor,
            )
    branch_stages = {
        s.strip() for s in args.branch_stages.split(",") if s.strip()
    }

    if args.prompt_file is not None:
        base_prompts = _load_prompts(args.prompt_file)
        if not base_prompts:
            raise SystemExit(f"no prompts parsed from {args.prompt_file}")
    elif args.prompt:
        base_prompts = [args.prompt]
    else:
        raise SystemExit("provide --prompt or --prompt-file")

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    jobs = [
        (prompt_idx, rep, prompt)
        for prompt_idx, prompt in enumerate(
            base_prompts, start=1 + args.prompt_index_offset
        )
        for rep in range(1, args.repeats + 1)
    ]
    failures = 0
    try:
        for job_idx, (prompt_idx, rep, prompt) in enumerate(jobs, start=1):
            labeled_batch = len(jobs) > 1 or args.always_run_label
            run_label = f"prompt{prompt_idx:02d}_rep{rep:02d}" if labeled_batch else None
            if labeled_batch:
                print(f"\n[batch] run {job_idx}/{len(jobs)} "
                      f"{run_label}: {prompt}", file=sys.stderr)
            ok = False
            max_attempts = args.run_retries + 1
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    print(f"[batch] retrying {run_label or 'single_run'} "
                          f"attempt {attempt}/{max_attempts}",
                          file=sys.stderr)
                try:
                    h = env.wait_healthy(args.server_ready_timeout)
                    print(f"[client] server healthz: {h}", file=sys.stderr)
                    run_rollout(
                        args, env, policy, critic, branch_stages,
                        prompt, run_label=run_label,
                    )
                    ok = True
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[batch] run attempt failed: {e}\n"
                          f"{traceback.format_exc()}", file=sys.stderr)
                    env.reset_http_session()
            if not ok:
                failures += 1
    finally:
        if not args.keep_server_alive:
            try:
                env.admin_exit()
                print("[client] requested server graceful exit; supervisor "
                      "will respawn uvicorn for the next run.", file=sys.stderr)
            except Exception as e:
                print(f"[client] /admin/exit warn: {e}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{failures}/{len(jobs)} rollout(s) failed")


if __name__ == "__main__":
    main()
