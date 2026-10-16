# Scene-Designer Usage Guide

[← Project README](README.md)

## Layout

| Directory | Purpose |
|---|---|
| `trajectory_compression/` | Four-stage offline trajectory compression and evidence checks |
| `tools/` | CoT annotation, five-view preparation, Critic SFT conversion |
| `src_critic/` | Scene perturbations, rendering, ranking labels, CoT and rewards |
| `src/` | SFT, GRPO, model client, tool server, shared observation contract |
| `Designer/` | Runtime adapters for SceneSmith |
| `prompts/` | Stage-specific policy instructions and tool schemas |
| `integration/` | SceneSmith compatibility patch and configuration |
| `scripts/` | Setup, preflight, isolated physics/render workers |

Run the commands below **from this directory**. Model and dataset locations are configurable; examples use relative paths. Environment variables are read from the process environment; `.env.example` is a template and is not automatically loaded.

## Installation

Use Linux and Python 3.11. The tested model stack is PyTorch 2.8, Transformers 4.57.6 and PEFT 0.19.1. Install a CUDA-compatible PyTorch build for training on GPUs.

For data processing and SFT without scene execution:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For tool-based inference, RL, and rendering, prepare **SceneSmith separately**:

```bash
git clone https://github.com/nepfaff/scenesmith.git scenesmith
git -C scenesmith checkout e888297a8b0a2f3acbb8634edf9def59cceab836
python scripts/setup_scenesmith.py --path scenesmith
```

Follow [SceneSmith's installation and asset setup instructions](https://github.com/nepfaff/scenesmith/tree/e888297a8b0a2f3acbb8634edf9def59cceab836), then install this release's `requirements.txt` into that environment. In a local installation:

```bash
cd scenesmith
uv sync --no-dev
cd ..
source scenesmith/.venv/bin/activate
uv pip install -r requirements.txt
```

The patch also aligns SceneSmith’s older PyTorch/Torchvision/Transformers constraints with this release’s tested model stack. Rebuild optional CUDA extensions for the installed PyTorch version when preparing the geometry backend.

The setup script applies `integration/scenesmith.patch` and copies `integration/scene_designer.yaml` into `scenesmith/configurations/scene_designer.yaml`. **Do not copy our `src/` or `Designer/` into SceneSmith.** They remain here; `src/runtime.py` puts the correct release and external checkout on the import path. For a different checkout location, set `SCENESMITH_ROOT` before launching. The patch is checked against the exact upstream revision and refuses an incompatible checkout.

The integration patch preserves the research configuration’s **0.2 m floor-penetration tolerance** for furniture, wall and ceiling (upstream uses 0.05 m). This affects physical validation; override `physics_validation.floor_penetration_tolerance_m` per stage if you need the stricter setting. Failed physics checks return “unknown”, never a successful validation. Manipuland generation is outside this release’s four-stage scope.

The default scene configuration needs SAM3/SAM3D checkpoints, materials and CLIP embeddings, and articulated-object assets. These must be prepared under SceneSmith according to its instructions and licenses; installing Python packages alone is insufficient. Asset paths beginning with `data/`, `external/`, `materials/` or `checkpoints/` resolve against the external checkout. Configure the asset sources and auxiliary models in `scenesmith/configurations/scene_designer.yaml` for your installation.

Drake's scene export also needs EGL. On Ubuntu, install `libegl1` and a working Mesa/NVIDIA EGL implementation (`libegl-mesa0` for software rendering); see [Drake's rendering troubleshooting](https://drake.mit.edu/troubleshooting.html#gl-init). Five-view rasterization uses headless CPU Cycles, with `SCENE_RENDER_SAMPLES=24` by default. Increase this for cleaner images. Native scene export and physics checks run in child processes so a native-library error does not terminate the training process silently.

```bash
export OPENAI_API_KEY='YOUR_KEY'
export OPENAI_BASE_URL='https://api.openai.com/v1'
export COT_MODEL='gpt-5.2'
export CRITIC_JUDGE_MODEL='gpt-5.2'
export CRITIC_POLISH_MODEL='gpt-5.2'
export SCENESMITH_MODEL='gpt-5.2'
python scripts/preflight.py --runtime --model model/Scene-Builder-Qwen3-VL-8B-Instruct
```

`CRITIC_BASE_URL` / `CRITIC_API_KEY` can override the endpoint for annotation. SceneSmith supports its own `SCENESMITH_LLM_*` settings. Auxiliary API models are separate from the local Scene-Builder policy.

Download the checkpoint into the example model directory using the Hugging Face CLI:

```bash
hf download XuanyuLiu/Scene-Builder-Qwen3-VL-8B-Instruct \
  --local-dir model/Scene-Builder-Qwen3-VL-8B-Instruct
```

The published checkpoint is the existing SFT checkpoint. Updating this code to five views does **not** retroactively retrain those weights; fine-tune on the prepared five-view data when training the revised policy.

## Observation contract

Every furniture, wall and ceiling policy observation contains **five separate images**, in this order:

1. topdown (with object IDs and geometric annotations);
2. south oblique;
3. east oblique;
4. west oblique;
5. north oblique.

All five must depict the same pre-action state. They are passed separately to the VL processor, not tiled into one image. Floor-plan generation is text-only because the room has not been constructed yet. Critic inputs are different: A/B or A/B/C/D denote **candidate scenes**, each represented by one annotated topdown image; they are not camera directions.

The default model preprocessing budget is 65,536 pixels **per image**, consistently applied to the fast processor's size settings. Rendering retains native 1024-pixel images. Missing or misordered scene views fail validation; missing-image terminal/check-only source turns are omitted by the SFT loader instead of reusing a stale observation.

## 1. Compress trajectories and annotate CoT

Input episodes are existing four-stage tool transcripts:

```text
datasets/episodes/run_001/
  floor_plan.json
  furniture_bedroom.json
  wall_bedroom.json
  ceiling_bedroom.json
```

Each episode contains `stage`, `system`, `task_context`, and `turns`; assistant turns contain reasoning, tool calls/results and relative `image` / `images` references. Supply the original SceneSmith room directory for recorded-state checks:

```bash
python -m trajectory_compression.cli \
  --scene-dir datasets/episodes/run_001 --room bedroom \
  --room-dir scenesmith/outputs/RUN/scene_000/room_bedroom \
  --output-dir datasets/compressed/run_001 --model "$COT_MODEL"
```

Compression works offline against recorded state and tool effects; it does not collect new trajectories or replay API-driven generation. `--check-only` performs validation without requesting deletions. Reports explicitly distinguish recorded-evidence checks from weaker available checks. `--retain-input-trajectory` explicitly accepts the supplied transcript as the baseline when original-run evidence is unavailable; this is not a claim of equivalence to an unobserved original run. Keep `compression_report.json` for auditing.

Complete oblique observations using the **matching step's** `.blend` beside each existing `.topdown.png`:

```bash
python tools/complete_cardinal_sft_views.py --sft-root datasets/compressed
python tools/complete_cardinal_sft_views.py --sft-root datasets/compressed --execute
```

A missing same-state snapshot is an error. This command preserves the existing annotated topdown and adds four views. Then annotate CoT without changing the retained actions:

```bash
python -m tools.rewrite_cot_gpt --src datasets/compressed \
  --out datasets/sft --max-workers 2 --max-rounds 3
```

`--plan-only` inspects coverage without API calls. Do not use an unbounded retry loop for unattended jobs. Run CoT annotation before final SFT validation so reasoning matches the retained trajectory and images.

## 2. Construct Scene-OTC

Input scenes use `datasets/raw/<scene>/floor_plan.json` and `furniture_<room>.json`. The floor-plan file contains `rooms`, `doors`, `windows`; the furniture file contains `placed_furniture` entries with `object_id`, name/description, `translation`, `rotation_wxyz`, `bbox_min`, `bbox_max`. This pipeline currently takes the first room/furniture file per input scene: split multiroom scenes into separate inputs.

Place matching final furniture snapshots at `datasets/per_step/<scene>/<room>/furniture_<step>*.blend`. Scene JSON and Blender snapshot must describe the same state. The renderer uses the latest furniture step.

```bash
python -m src_critic.build_dataset \
  --raw-dir datasets/raw --per-step-dir datasets/per_step \
  --out-dir datasets/critic_pairs \
  --ranking-sizes 2:1,4:1 --ranking-modes cumulative:1,sibling:1 \
  --n-rankings 8 --seed 42
```

The pipeline designs eligible perturbations, applies their transformations, renders clean candidate images and separate judge images, checks judgments, annotates reasoning, and writes `sft.jsonl` plus per-record assets. Perturbation pointers appear only in judge images, not the candidate images used for SFT. Ranking-mode judge disagreements are filtered after retries. The legacy binary-pair mode retains hard/directional rule labels and filters soft disagreements. The default sampler excludes collision and door-blocking kinds; explicitly enabled geometric negatives use rule CoT and record that no API judge ran. Rule-based orderings are synthetic supervision assumptions, not human ground truth. Inspect accepted examples before training.

`--skip-judge --no-blender` is an **offline schematic smoke test**, not the production dataset recipe. Production rendering failures are errors, not automatic substitution with schematic images. Use an empty output directory for a fresh run; output JSONL is appended for interrupted work, so do not merge overlapping runs blindly.

Optional severity construction uses cached object importance/directionality and the existing parameter-sweep, cumulative, multikind and multitarget generators:

```bash
python -m src_critic.score_scene_objects --raw-dir datasets/raw
python -m src_critic.build_n4_from_n2 --help
python -m src_critic.build_n4_severity --mode both --target 50
python -m src_critic.generate_image_grounded_reason \
  --inp datasets/critic_pairs/sft_n4_severity.jsonl \
  --out datasets/critic_pairs/sft_n4_reasoned.jsonl --base-dir .
```

The optional severity generators use the documented `datasets/raw`, `datasets/per_step`, and `datasets/critic_pairs` layout. Their intermediate rule-based reasoning is not the final image-grounded CoT. `permute_n4_labels`, `fix_perm_letter_sequences`, `build_multitarget_pairwise`, and `assemble_train_set` provide label permutation, pairwise derivation and assembly; inspect `--help` for their inputs. Split by base scene **before** augmentation to avoid related-scene leakage across train/test. The provided Scene-OTC-Bench is held out; do not mix it into training.

Convert final accepted JSONL records to the SFT loader format and copy only their candidate images:

```bash
python tools/convert_otc_to_sft.py --input datasets/critic_pairs/sft.jsonl \
  --source-root . --output datasets/otc_sft
```

This produces `sft/critic_*.json` and `images/` with relative paths. Train that set using `--sft-root datasets/otc_sft/sft --repo-root datasets/otc_sft`. For joint OTA+OTC SFT, place both sets' episode JSONs beneath one SFT root and express all image paths relative to the same `--repo-root`. Candidate order, answer letters and reasoning must stay aligned.

## 3. SFT

Validate dataset encoding before allocating a model:

```bash
python -m src.train_sft --model-path model/Scene-Builder-Qwen3-VL-8B-Instruct \
  --sft-root datasets/sft --repo-root . --dry-run
```

Full-parameter training:

```bash
python -m src.train_sft --model-path model/Scene-Builder-Qwen3-VL-8B-Instruct \
  --sft-root datasets/sft --repo-root . --output-dir model/scene_builder_sft \
  --num-epochs 3 --per-device-batch-size 1 --grad-accum 8 --max-pixels 65536
```

For LoRA, append `--use-lora --lora-r 32 --lora-alpha 64`. Add `--lora-merge-on-save` to produce a standalone checkpoint usable by the inference commands. Adapter-only output must be loaded with its base checkpoint via PEFT; it is not a standalone model directory. `--max-steps 1 --warmup-ratio 0` limits an optimizer smoke check. CPU checks can use `--use-cpu --dataloader-workers 0`; CPU execution is not a substitute for throughput or GPU memory validation.

## 4. Scene-GRPO

Provide a JSONL prompt file with one `{"prompt": "Design a ..."}` object per line. RL collects floor-plan and furniture trajectories with real tools, branches at scene-editing decisions, and combines local Critic reward with trajectory-level length/object-count reward. Wall/ceiling generation is supported by inference and SFT; this RL entrypoint optimizes the **furniture stage**.

```bash
python -m src.train_grpo \
  --model-dir model/scene_builder_sft --prompts-file datasets/rl_prompts.jsonl \
  --output-dir model/scene_builder_rl --ckpt-dir model/rl_checkpoints \
  --results-dir results/rl --reward-fn critic --critic-frozen-v7 \
  --n-branches 4 --n-trajectories 4 --n-episodes 1 --total-update-steps 50
```

The legacy flag name `--critic-frozen-v7` means “reuse the frozen initial reference model as Critic”; it is not tied to a local v7 path. The starting checkpoint must have Critic supervision. For a separate Critic, pass `--no-critic-frozen-v7`, `--critic-model-dir`, `--critic-adapter-dir`, and `--critic-device`. `--reward-fn random` / `format` are debugging modes, not the method's Critic reward.

This runner uses a policy and frozen reference on the default CUDA device, plus the configured geometry services. Reserve enough GPU memory for all of them; it does not implement distributed RL. The release corrects the old objective's sequence-summed reference ratio to completion-token clipping against detached behavior log probabilities, with a separate nonnegative sampled KL term against the frozen reference. It makes one optimizer update after each collected batch. Exact sampled token IDs are retained for the objective. Branch selection restores the scored scene and memory instead of executing the chosen action twice. This is an implementation correction, not a claim that historical experiment numbers have been rerun.

## 5. Generate scenes

Start the tool server in one terminal, bound to loopback (it is a local trusted tool service, not an authenticated public API):

```bash
python -m uvicorn src.tool_server:app --host 127.0.0.1 --port 18000
```

In another terminal, activate the same environment and run:

```bash
python -m src.rollout_client --server-url http://127.0.0.1:18000 \
  --model-dir model/Scene-Builder-Qwen3-VL-8B-Instruct \
  --prompt 'Design a warm bedroom with a bed, bedside storage, and a reading corner.' \
  --config scene_designer --max-steps-per-stage 30 --keep-server-alive
```

The policy generates tools through floor plan → furniture → wall → ceiling, and receives five current-state images at each visual stage. `--branch-critic --critic-model-dir model/scene_builder_sft` enables Critic-guided branch selection. Results and tool traces are saved under `results/`. Tool-server and policy processes must share the filesystem; separate container mounts can use `CONTAINER_RESULTS_ROOT` and `HOST_RESULTS_ROOT`.

For a cheaper model-only check on existing observations (this does **not** execute scene tools):

```bash
python -m src.infer_step --model-dir model/Scene-Builder-Qwen3-VL-8B-Instruct \
  --sft-root datasets/sft --repo-root . --stage furniture --limit 1
```

## Validation and limitations

See [VALIDATION.md](VALIDATION.md) for executed checks and remaining integration limits. Development tests are maintained separately and are not included in this release. Check the installation before running:

```bash
python scripts/preflight.py --runtime --model model/Scene-Builder-Qwen3-VL-8B-Instruct
```

API annotation, assets, model downloads and full scene rollouts require the external resources described above. Tests of tensor shapes or a short optimizer step do not validate scene quality or reproduce paper results.

## Third-party code

SceneSmith is an external dependency under its own license. Its compatibility patch includes upstream context; the upstream license notice is retained in `integration/SceneSmith-LICENSE`. Model checkpoints, pretrained asset generators and scene assets have their own licenses. This release does not redistribute them.
