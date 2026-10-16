"""Full-parameter SFT for Qwen3-VL on scenesmith stepwise data.

Quick start:
    bash scripts/train_sft.sh

Smoke test (one-batch dry run, no training):
    python src/train_sft.py --dry-run --output-dir /tmp/sft_dry
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_dataset import StepwiseCollator, StepwiseSFTDataset  # noqa: E402


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = REPO_ROOT_DEFAULT / "model" / "Qwen3-VL-8B-Instruct"
SFT_ROOT_DEFAULT = REPO_ROOT_DEFAULT / "datasets" / "sft"
OUTPUT_DEFAULT = REPO_ROOT_DEFAULT / "model" / "Scene-Builder-Qwen3-VL-8B-Instruct"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(MODEL_DEFAULT))
    ap.add_argument("--sft-root", default=str(SFT_ROOT_DEFAULT))
    ap.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    ap.add_argument("--output-dir", default=str(OUTPUT_DEFAULT),
                    help="default: model/Scene-Builder-Qwen3-VL-8B-Instruct")
    ap.add_argument("--num-epochs", type=float, default=3.0)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-pixels", type=int, default=65536,
                    help="cap each image to this many pixels (default: 256×256)")
    ap.add_argument("--save-steps", type=int, default=0,
                    help="if >0, save a mid-training checkpoint every N steps. "
                         "default 0 = no mid-training saves; only the final model "
                         "is kept (via trainer.save_model at end).")
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--max-shard-size", default="5GB",
                    help="split saved weights into shards of this size (HF Hub convention).")
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deepspeed", default=None,
                    help="path to DeepSpeed config JSON (e.g. configs/ds_zero2.json)")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --output-dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="build dataset + print one example, do NOT train")
    # ----- LoRA -----
    ap.add_argument("--use-lora", action="store_true",
                    help="Enable LoRA (PEFT) fine-tuning. Trains adapters only; "
                         "base weights stay frozen — much less GPU memory + "
                         "saves a tiny ~100MB adapter file.")
    ap.add_argument("--lora-r", type=int, default=32,
                    help="LoRA rank. Higher = more capacity / more params. "
                         "Common: 8 (light), 16, 32, 64.")
    ap.add_argument("--lora-alpha", type=int, default=64,
                    help="LoRA alpha scaling. Common convention: 2 × lora_r.")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj",
                    help="Comma-separated module name patterns. Defaults to "
                         "attention projections only. Add MLP (gate_proj,up_proj,"
                         "down_proj) for more capacity at ~2-3× param cost.")
    ap.add_argument("--lora-include-mlp", action="store_true",
                    help="Shorthand: also LoRA the MLP layers (gate_proj/"
                         "up_proj/down_proj). Equivalent to appending them to "
                         "--lora-target-modules.")
    ap.add_argument("--lora-merge-on-save", action="store_true",
                    help="At end of training, merge adapter weights back into "
                         "the base model and save the FULL merged model "
                         "(needed for inference paths that load the model as "
                         "a regular HF checkpoint). Without this flag, only "
                         "the small adapter is saved; inference code then "
                         "needs to load base + adapter via PeftModel.")
    ap.add_argument("--max-steps", type=int, default=-1, help="Limit optimizer updates; useful for smoke testing")
    ap.add_argument("--use-cpu", action="store_true")
    ap.add_argument("--dataloader-workers", type=int, default=2)
    ap.add_argument("--report-to", default="none", choices=["none", "tensorboard"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[init] loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    from src.scene_observation import configure_image_budget
    configure_image_budget(processor, args.max_pixels)

    print(f"[init] discovering samples under {args.sft_root}")
    dataset = StepwiseSFTDataset(
        sft_root=Path(args.sft_root),
        repo_root=Path(args.repo_root),
        processor=processor,
        max_length=args.max_length,
    )
    print(f"[init] {len(dataset)} step samples loaded")
    if len(dataset) == 0:
        sys.exit("no samples — check --sft-root")

    if args.dry_run:
        print("\n[dry-run] inspecting one example...")
        ex = dataset[0]
        for k, v in ex.items():
            shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
            print(f"   {k}: {shape}")
        n_target = int((ex["labels"] != -100).sum().item())
        print(f"   loss tokens (labels != -100): {n_target}")
        print(f"   sample 0 metadata: stage={dataset.samples[0].stage} "
              f"step={dataset.samples[0].step_idx} "
              f"src={dataset.samples[0].source_file}")
        return

    print(f"[init] loading model from {args.model_path} (bf16)")
    dtype = torch.bfloat16 if not args.no_bf16 else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )

    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ----- LoRA wrap (PEFT) -----
    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        if args.lora_include_mlp:
            for m in ("gate_proj", "up_proj", "down_proj"):
                if m not in targets:
                    targets.append(m)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
        )
        model = get_peft_model(model, lora_cfg)
        # Re-enable input grads for gradient checkpointing through frozen base.
        if not args.no_grad_checkpoint and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print(f"[lora] r={args.lora_r} alpha={args.lora_alpha} "
              f"dropout={args.lora_dropout} targets={targets}")
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        use_cpu=args.use_cpu,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=not args.no_bf16,
        gradient_checkpointing=not args.no_grad_checkpoint,
        save_strategy=("steps" if args.save_steps > 0 else "no"),
        save_steps=(args.save_steps if args.save_steps > 0 else 999_999),
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        logging_steps=args.logging_steps,
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
        optim="adafactor",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        ignore_data_skip=True,  # on --resume, don't slow-replay batches to global_step
        ddp_find_unused_parameters=True,
        deepspeed=args.deepspeed,
    )

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    collator = StepwiseCollator(pad_token_id=pad_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(f"[train] starting; output → {args.output_dir}"
          + ("  (RESUME from last checkpoint)" if args.resume else ""))
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    # Custom save with explicit shard size — HF Trainer.save_model() doesn't
    # expose max_shard_size, but model.save_pretrained() does. We replicate
    # Trainer._save's gather-then-save pattern so DDP works.
    state_dict = trainer.accelerator.get_state_dict(trainer.model)
    if trainer.is_world_process_zero():
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)

        if args.use_lora:
            if args.lora_merge_on_save:
                # Merge LoRA into base, save full merged model. Loadable by
                # any vanilla HF code path — same as a full FT checkpoint.
                print(f"[lora] merging adapter weights into base model...")
                merged = unwrapped.merge_and_unload()
                merged.save_pretrained(
                    args.output_dir,
                    max_shard_size=args.max_shard_size,
                    safe_serialization=True,
                )
            else:
                # Save adapter only (~100 MB). To use at inference:
                #   from peft import PeftModel
                #   base = Qwen3VLForConditionalGeneration.from_pretrained(<base>)
                #   model = PeftModel.from_pretrained(base, <adapter_dir>)
                print(f"[lora] saving adapter only (base weights NOT written)")
                unwrapped.save_pretrained(args.output_dir, safe_serialization=True)
        else:
            unwrapped.save_pretrained(
                args.output_dir,
                state_dict=state_dict,
                max_shard_size=args.max_shard_size,
                safe_serialization=True,
            )
        processor.save_pretrained(args.output_dir)

    print(f"[done] saved training output to {args.output_dir}")


if __name__ == "__main__":
    main()
