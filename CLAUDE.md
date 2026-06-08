# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Visual Planning is a research codebase for training vision-language models to plan **entirely in the visual domain** — producing sequences of images as reasoning steps rather than text. It implements VPRL (Visual Planning Reinforcement Learning), a two-stage training framework using SFT + GRPO, evaluated on three spatial navigation environments: FrozenLake, Maze, and MiniBehaviour.

## Environment Setup

```bash
conda create -n visualplanning python=3.12.3
conda activate visualplanning
bash scripts/install.sh
```

`install.sh` also clones datasets from HuggingFace (`SFT_random_*`) and base model checkpoints (`LVM_ckpts`, `vqvae_ckpts`) — these must be present under `./models/` and `./vqlm/vqvae_ckpts` before training.

## Training Commands

All scripts accept `frozenlake`, `maze`, or `minibehaviour` as the task argument.

**VPFT — SFT on optimal trajectories:**
```bash
bash scripts/sft_optimal.sh <task>
```

**VPRL Stage 1 — SFT on random trajectories (policy initialization):**
```bash
bash scripts/sft_random.sh <task>
```

**VPRL Stage 2 — GRPO reinforcement learning:**
```bash
bash scripts/grpo.sh <task>
```

Stage 2 expects a merged checkpoint from Stage 1 at `./models/<task>/SFT_LVM_random_merged_ckpts`.

**Hyperparameter overrides** use Hydra dot-notation on the CLI, e.g.:
```bash
python train_sft.py SFT.num_train_epochs=5 SFT.dataset_pth=... SFT.model_path=... SFT.output_dir=... SFT.run_name=...
```

## Evaluation

```bash
python evaluation/visual_planning_evaluator.py  # requires configs/eval.yaml
```

To aggregate results from an already-evaluated output folder without running inference:
```bash
python evaluation/visual_planning_evaluator.py show_stats=true evaluation_result_folder_pth=<path>
```

## Architecture

### Core abstractions

- **VQ-VAE tokenizer** (`vqlm/vqvae_muse.py`): Encodes/decodes 256×256 images to/from 256 discrete integer tokens. Each token sequence represents one visual frame. The vocabulary range 0–8191 is valid; 8192+ tokens are suppressed during generation.

- **Base model**: `LlamaForCausalLM` from `./models/LVM_ckpts`. Trained autoregressively over flattened image token sequences. All training uses LoRA (r=32, α=64) applied to all attention and MLP projections; weights are merged and saved as full checkpoints after training.

- **ActionParser** (`layout_parser.py`): Converts raw image tensors into agent action labels by dividing each 256×256 image into a `level × level` patch grid and comparing agent position between consecutive frames. Environment-specific methods: `parse_action_in_imgs` (FrozenLake), `parse_maze_action_in_imgs` (Maze), `parse_mini_action_in_imgs` (MiniBehaviour).

- **VisionGRPOTrainer** (`CustomGRPO.py`): Extends TRL's `GRPOTrainer` with a `KLAnnealer` for dynamic KL-penalty scheduling. Used by all Stage 2 training scripts.

### Dataset format

JSONL files where each line has:
- `input_tokens`: list[int] — the prompt image tokens (256 per frame)
- `output_tokens`: list[int] — the target next-frame tokens (256)
- `input_state`: agent position(s)
- `meta`: dict containing `level`, `target_pos`, `distance_map`, `layout`, etc.

### Training flow

1. `train_sft.py`: Loads `TokenizedDataset`, pads sequences with `-100`, masks the first 256 tokens (input frame) from loss, trains with HuggingFace `Trainer`. Logs image reconstructions to W&B every 50 steps.

2. `train_rl_{frozen,maze,mini}.py`: Loads `TokenizedDataset` without padding (GRPO handles collation). The reward function (`action_reward_func`) decodes each generated 256-token frame, infers the agent action via `ActionParser`, and returns:
   - `+1` if the action moves toward the goal (Progress Reward)
   - `0` if no progress
   - `-5` for invalid actions or out-of-vocabulary tokens

### Config system

Hydra with `configs/train.yaml`. The top-level keys are `SFT` and `GRPO`. All training scripts use `@hydra.main(config_path="configs", config_name="train")`.

Evaluation uses a separate `configs/eval.yaml` (not tracked in the repo; must be created manually).

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| `transformers` | 4.49.0 | LlamaForCausalLM, Trainer |
| `trl` | 0.15.2 | GRPOConfig, GRPOTrainer |
| `peft` | 0.14.0 | LoRA |
| `hydra-core` | 1.3.2 | Config management |
| `wandb` | 0.19.7 | Experiment tracking (required) |
| PyTorch | 2.5.1+cu124 | CUDA 12.4 required |
