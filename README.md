# SignVLM: Vision-Faithful Sign-to-Lane Rule Binding

[![Pretrained adapters on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ray90100%2FSignVLM--public-yellow)](https://huggingface.co/ray90100/SignVLM-public/tree/main)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> Use a Vision-Language Model to read traffic-sign content and bind the parsed
> rule to specific lanes on the map, so a downstream planner can change lanes
> correctly — even when the prior map and the camera-observed sign disagree.

Qwen2.5-VL-7B + LoRA SFT + **CAVP** (Conflict-Aware Vision-Prior training)
substantially outperforms zero-shot and the RuleVLM baseline on MapDR Test
under vision-vs-map conflict — see §4 for the reproduction recipe.

**Pretrained LoRA adapters** for all paper-cited runs (10 adapters, ~23 GB)
are released on Hugging Face:
**https://huggingface.co/ray90100/SignVLM-public/tree/main**
— see §3 "Pretrained adapters" for a one-liner download / load snippet.

---

## 1. What this repo does

Autonomous-driving stacks commonly consume map priors (speed limits, allowed
vehicle types, time-of-day rules, etc.) as structured input. When the
on-road sign disagrees with the prior, the model tends to trust the prior
text — the visual evidence loses. SignVLM trains the VLM to cross-check the
cropped sign image against the HD-map-rendered visual prompt and prefer the
visual evidence when they conflict.

Sub-tasks:

1. **Sign content extraction** — parse the rule from the sign image (speed,
   direction, vehicle type, effective time, …).
2. **Rule–lane correspondence** — bind the rule to the specific lane(s) it
   applies to using the projected HD-map lanes (centerlines) as a visual
   prompt.

Method:

- Backbone: `Qwen2.5-VL-7B-Instruct`
- Input: cropped sign image + HD-map visual prompt (centerlines + a rendered
  rule panel of *prior* speed/direction/vehicle/time).
- Training: LoRA SFT on MapDR with **CAVP** — at training time, the rendered
  prior panel is corrupted with probability *p* in one or more fields; the
  ground-truth answer always follows the *visual* sign content. The model
  learns to disregard the panel text when it contradicts the sign.

## 2. Environment setup

Tested on CUDA 12.x / Python 3.10, single-machine with 4 × 24 GB GPUs
(RTX 4090 class). Single 24 GB card works with `LOAD_IN_4BIT=1` and reduced
image pixels.

```bash
conda create -n signvlm python=3.10 -y
conda activate signvlm
pip install -r requirements.txt
# flash-attn does not provide prebuilt wheels for many envs; install separately:
pip install flash-attn --no-build-isolation     # 5–10 min compile, needs ninja+nvcc
```

`bitsandbytes` is optional but recommended (used for `--optim adamw8bit` and
`--load-in-4bit`).

## 3. Data preparation

Download MapDR (CVPR 2025 Highlight) from the official repo and symlink it
into the `data/` directory that ships with this repo:

```bash
# https://github.com/MIV-XJTU/MapDR — follow their instructions to obtain data
ln -s /path/to/your/MapDR data/MapDR
```

(`data/`, `ckpts/`, `runs/` are placeholder directories shipped with the
repo, each with a short README explaining what to put inside.)

Expected layout:

```
data/MapDR/
  <scene_id>/
    data.json
    label.json
    img/*.jpg
  ...
  split.json          # Train/Test scene-id lists
```

Then download the base VLM checkpoint:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir ckpts/Qwen2.5-VL-7B-Instruct
```

### Pretrained adapters (skip training)

The LoRA adapters used in the paper are released on Hugging Face at
[**ray90100/SignVLM-public**](https://huggingface.co/ray90100/SignVLM-public).
Use them directly if you just want to evaluate / inspect, without
re-running the SFT / DPO / GRPO training jobs:

```bash
# Headline adapter — Qwen2.5-VL-7B + LoRA SFT + CAVP, seed 42
huggingface-cli download ray90100/SignVLM-public \
    --include "sft-7B-CAVP-p0.3-s42/*" \
    --local-dir ckpts/
```

```python
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="bfloat16", device_map="cuda")
model = PeftModel.from_pretrained(
    base, "ray90100/SignVLM-public", subfolder="sft-7B-CAVP-p0.3-s42")
```

See the Hugging Face repo's model card for the full adapter index and
which subfolder maps to which paper table row.

## 4. Reproduce the headline result

### 4.1 Train SFT baseline (clean, no conflict perturbation)

```bash
./scripts/sft.sh
# outputs to runs/sft/<timestamp>/final/
```

### 4.2 Train SFT + CAVP (the main method)

```bash
PERTURB_MODE=conflict PERTURB_PROB=0.3 ./scripts/sft.sh
```

### 4.3 Evaluate

```bash
# Inference on MapDR Test (1076 scenes, shard across GPUs)
python scripts/eval_sft.py --adapter runs/sft/<run>/final
# Apply canonical (v3) post-processing for robust string matching
python scripts/eval_canonical.py runs/sft/<run>/final/eval_<...> --rule-version v3
```

As a sanity check, the 7B + CAVP main run on the full Test split (1076
scenes) with `seed=42` should land near **Overall F1 ≈ 0.80** under v3
canonical conflict-mode evaluation. ±1pt is normal seed/framework noise;
the gap between *clean SFT* and *SFT + CAVP* under panel-conflict input is
what reproduces most stably.

### 4.4 Common knobs

The launcher scripts read environment variables (see comments at the top of
each `.sh` file). Key ones:

| var | default | meaning |
|---|---|---|
| `GPUS` | `1,2,3,4,5,6` | CUDA device IDs (comma-separated) |
| `EPOCHS` | `1` | SFT epochs (`2` for DPO) |
| `MAX_PIXELS` | `802816` | per-image cap; `1605632` for option-b |
| `PERTURB_MODE` | `none` | `none` / `noise` / `conflict` (CAVP) |
| `PERTURB_PROB` | `0.0` | per-sample perturb probability |
| `LOAD_IN_4BIT` | `0` | 1 → QLoRA NF4 base (single-24G runs) |
| `MODEL_NAME` | `Qwen2.5-VL-7B-Instruct` | also supports `Qwen2.5-VL-3B-Instruct` |

## 5. Inspect visual prompts (optional)

`scripts/render_visual_prompts.py` is an offline preprocessing utility that
renders the cropped sign + HD-map centerline + optional rule-panel overlay
for a handful of MapDR scenes you place in `data/zeroshot_demo/`. Useful
for sanity-checking the projection pipeline before training.

```bash
# Copy ~8 MapDR scenes into data/zeroshot_demo/ first
python scripts/render_visual_prompts.py
# Outputs to experiments/visual_prompts/<scene>/{cropped_sign.jpg, visual_prompt.jpg, ...}
```

## 6. RuleVLM baseline (optional, for comparison)

SignVLM's main competitor is **RuleVLM** (MapDR's official end-to-end
method). To reproduce the RuleVLM baseline, you must
clone the official repo and use their pinned environment — this repo only
provides thin wrapper scripts.

> The official RuleVLM metric implementation is **not** included in this
> repo (to respect their CC BY-NC-SA license). For an apples-to-apples
> "official" number, run their `evaluate.py` on the wrapper's output.

```bash
# 1. Clone the official RuleVLM repo (sibling directory by default)
git clone https://github.com/MIV-XJTU/MapDR ../RuleVLM_ref
# Follow their README to obtain the RuleVLM adapter ckpt and Qwen-VL-Chat base.

# 2. Their environment uses transformers==4.42.3, a separate conda env:
conda create -n rulevlm python=3.10 -y
conda activate rulevlm
# (install per RuleVLM repo instructions)

# 3. Run our wrappers (point RULEVLM_REPO at the cloned repo)
RULEVLM_REPO=../RuleVLM_ref/RuleVLM \
python scripts/eval_rulevlm.py \
    --mapdr-dir ./data \
    --split-file ./data/MapDR/split.json \
    --qa-dataset-dir /tmp/rulevlm_qa_data \
    --adapter <path/to/rulevlm-adapter> \
    --model-dir <path/to/qwen-vl-chat> \
    --out-dir ./runs/rulevlm/eval_native

# Then optionally feed the output back through their official evaluate.py.
```

`scripts/eval_rulevlm_setupB.py` runs the RuleVLM model on **SignVLM's**
panel-aware conflict-perturbed visual prompts, for apples-to-apples
comparison with SignVLM-CAVP under identical conflict.

## 7. Stage 2 — DPO / GRPO (experimental, results withheld)

Stage-2 reinforcement-fine-tuning code is provided for transparency; the
corresponding paper numbers are withheld pending an upcoming submission.

```bash
# DPO — requires a trained SFT adapter as init
INIT_ADAPTER=runs/sft/<your-cavp-run>/final ./scripts/dpo.sh

# GRPO — no shell launcher; see scripts/train_grpo.py docstring for CLI flags
python scripts/train_grpo.py --help
```

## 8. Downstream planner extension (experimental, results withheld)

A small MLP planner consumes SignVLM's structured output (`target_lane` +
parsed rules) along with ego history and scene centerlines, and predicts a
short-horizon ego trajectory. This shows the parsed sign rule can be
plumbed end-to-end to a downstream behavioral module. Code is provided;
quantitative results are withheld pending submission.

```bash
# 1. Extract (feature_dict, gt_trajectory) pairs from MapDR Train split
python scripts/planner_extract_data.py        # writes data/planner_data/{train,val}/*.pt

# 2. Train the MLP planner
python scripts/train_planner.py --epochs 100  # writes runs/planner/<tag>/best.pt

# 3. Run SignVLM inference on a scene to produce target_lane + rule json
python scripts/trajectory_signvlm_infer.py \
    --adapter runs/sft/<your-cavp-run>/final \
    --scene <scene_hash> \
    --out runs/trajectory/<scene_hash>.jsonl

# 4. Render side-by-side BEV / dashcam comparison with the geometric baseline
python scripts/trajectory_planner_render.py \
    --scene <scene_hash> \
    --ckpt runs/planner/<tag>/best.pt \
    --signvlm-jsonl runs/trajectory/<scene_hash>.jsonl \
    --rule-from-signvlm
```

`scripts/trajectory_demo_smoke.py` is the geometry-only baseline (no
planner, no SignVLM); `scripts/trajectory_filter_ego_on_signlane.py`
selects scenes where the ego actually overlaps a sign-affected lane.

## 9. Repository layout

```
src/                 core modules
  dataset.py         MapDR dataset (reads scene + renders visual prompt + optional CAVP perturb)
  perturb_conflict.py  CAVP: vision-vs-prior conflict generators
  projection.py      HD-map → image projection, centerline rendering
  prompt.py          prompt template
  metric.py          clean-room reimplementation of the 5 MapDR metrics
  dpo_dataset.py     DPO preference-pair dataset
  grpo_dataset.py    GRPO rollout dataset (conflict-only)
  grpo_reward.py     GRPO reward function (mirrors eval_canonical v3)
  token_mask.py      token-level DPO mask helper
  planner_model.py   downstream-extension MLP planner
scripts/             training, evaluation, preprocessing, planner extension
data/                MapDR + derived planner_data (symlinks; not in repo)
ckpts/               base VLM checkpoints (downloaded; not in repo)
runs/                training outputs (not in repo)
```

## 10. License and citation

- **Code in this repo**: Apache License 2.0 — see `LICENSE`.
- **MapDR dataset**: CC BY-NC-SA 4.0 (non-commercial). Cite the MapDR paper.
- **RuleVLM**: CC BY-NC-SA 4.0. We do not redistribute their code; we provide
  thin wrappers that *call* their cloned repo and adapter.

Citation:

```
TBD — paper under review.
```

## Acknowledgements

This work builds on [MapDR](https://github.com/MIV-XJTU/MapDR) and
[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
