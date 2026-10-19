# Release validation

Checks performed on 2026-10-16. These establish execution and data contracts, not paper-result reproduction or a guarantee that no bugs remain.

## Executed checks

| Check | Result and scope |
|---|---|
| Python compilation | All release Python files compile |
| Offline unit tests | 53 tests passed: CoT coverage, five-view ordering, GRPO objective, portable OTC conversion, legal-subset label mapping and geometric-label provenance |
| Compression tests | 83 tests passed, including the four-stage CLI, retained-action checks, recorded evidence and missing-image handling |
| Branch integration regressions | 3 tests: exact selected-state restoration without executing actions twice, tool-server branch commit, sampled-token copying/serialization |
| CLI entrypoints | 17 documented data/training/inference entrypoints returned successfully for `--help` |
| External SceneSmith | Integration patch applied to an independent checkout at the documented upstream revision; main modules import from the curated release and that checkout |
| Real scene rendering | Existing bedroom exported through Drake/Blender, then rendered to topdown + south/east/west/north; output images inspected, with live object/coordinate annotations |
| Real Scene-OTC geometry path | Existing bedroom → perturbation → Blender candidate renders → ranking JSONL → portable Critic SFT JSON and copied relative image paths; API judge deliberately skipped in this smoke check |
| Real checkpoint SFT | Published Scene-Builder weights on CPU, five real scene images, LoRA rank 2, one optimizer update at learning rate 2e-5, finite loss/gradients, saved adapter; nonzero LoRA B weights checked |
| Merged LoRA save/reload | SFT CLI on a tiny randomly initialized Qwen3-VL with the real processor; merged checkpoint reloads with no missing/unexpected keys and changed weights |
| RL tensor/gradient path | Real VL processor + tiny random Qwen3-VL, five images, exact completion-token boundary, finite nonzero SFT and GRPO gradients |
| Real model inference | Both independent-step inference and the actual rollout client's ModelPolicy ran with Scene-Builder weights on CPU; model-forward hooks confirmed five image grids |
| One model/tool/observation cycle | Five real images → Scene-Builder-generated `move_furniture_tool` → real SceneSmith furniture movement → five new current-state views. Uses an existing asset; no asset-generation stand-in is reported as real generation |
| Release scan | No embedded credential tokens, personal machine paths, model weights, scene snapshots, datasets, upstream SceneSmith checkout or manuscript PDF in the release |

The smoke images used reduced pixel/sample budgets to keep CPU checks practical. The real Scene-Builder SFT check used a small observation fixture, not the full training corpus. The tiny random model checks validate code and tensors only.

## Fixes made during verification

- All visual policy paths pass five images in the same order; floor-plan and Critic candidate semantics remain separate.
- Fast-processor pixel limits are set correctly; missing Critic images fail rather than silently scrambling/dropping slots.
- LoRA merged saving uses merged weights; optimizer checkpoints are retained for SFT resume.
- The RL surrogate uses behavior log probabilities and a separate reference KL, on exact sampled tokens without an extra templated EOS.
- Branch selection restores the scored scene/memory. Structural walls do not inflate furniture-count rewards; filtered candidate labels map back to their original slots.
- Critic branches use actual annotated rendered scenes. Failed renders do not silently become production schematic training data.
- Geometric rule supervision is labeled honestly: no fabricated API-judge agreement. Physics-worker failure is “unknown”, not a successful check.
- Relative configuration paths resolve against the separately installed SceneSmith checkout. Its dependency constraints are aligned with the tested model stack.

The preserved research SceneSmith configuration uses a 0.2 m floor-penetration tolerance for furniture/wall/ceiling; this differs from upstream's 0.05 m and is documented in the README. Optional perturbation orderings remain synthetic supervision assumptions. The RL default Critic is the frozen initial checkpoint, which must have Critic supervision.

## Not executed in this verification session

- Full GPU SFT or a full GPU Scene-GRPO optimizer run with generated scene trajectories. All four available training GPUs were occupied; existing jobs were left running.
- An entire floor-plan → furniture → wall → ceiling generation starting from a new prompt with fresh SAM3D/articulated assets.
- Live paid API CoT/judge annotation or full-corpus generation for every optional severity recipe.
- A clean-machine installation of all CUDA geometry extensions, distributed training, performance benchmarking, or reproduction of paper tables.

A fresh same-family static review identified additional reward/provenance issues; fixes and regression checks were made afterward. That review is provisional, not an independent correctness certification. Historical experiment results were not regenerated by these code fixes.

## Installation check

The development test suite and model smoke-test script used for the checks above are kept outside this release. The counts above describe pre-release validation.

From the code root, check your prepared environment with:

```bash
python scripts/preflight.py --runtime --model model/Scene-Builder-Qwen3-VL-8B-Instruct
```

Real-scene execution requires your own snapshots/assets; private research scenes and temporary validation checkpoints are not distributed.
