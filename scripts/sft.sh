#!/usr/bin/env bash
# scripts/sft.sh — minimal launcher for SignVLM Stage 1 SFT
#
# Defaults: 6 cards (GPUs 1-6, leaving GPU 0 free), 1 epoch, option-a image cap.
# Forwards any extra positional args to scripts/train_sft.py.
#
# Usage:
#   ./scripts/sft.sh                                # foreground tqdm progress
#   BG=1 ./scripts/sft.sh                           # background (nohup) + tail log
#   GPUS=6 ./scripts/sft.sh                         # single-card override
#   ./scripts/sft.sh --map-perturb 0.3              # extra args forwarded (deprecated)
#   PERTURB_MODE=conflict PERTURB_PROB=0.3 ./scripts/sft.sh   # W2 conflict-aware SFT
#   EPOCHS=2 GRAD_ACCUM=8 ./scripts/sft.sh
#
# Env-var defaults (override with VAR=value prefix):
#   GPUS=1,2,3,4,5,6        comma-separated CUDA device ids
#   EPOCHS=1
#   GRAD_ACCUM=4            gradient accumulation steps (per process)
#   MAX_PIXELS=802816       per-image max pixels (option a; b: 1605632)
#   OPTIM=adamw             'adamw' (fp32) or 'adamw8bit' (bitsandbytes, save 2-4 GB)
#   PERTURB_MODE=none       'none' / 'noise' / 'conflict' (Task 6.10 + W2 main path)
#   PERTURB_PROB=0.0        per-sample perturb probability (used with PERTURB_MODE)
#   LOG_EVERY=25            jsonl log frequency
#   SAVE_EVERY=500          ckpt frequency (0 = end-of-run only)
#   LOAD_IN_4BIT=0          1 = QLoRA NF4 base (Tab 4 4bit baseline row)
#   MODEL_NAME=Qwen2.5-VL-7B-Instruct   under ckpts/; use Qwen2.5-VL-3B-Instruct for 3B twin runs
#   BG=0                    1 = nohup in background, then tail -f the log
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-1,2,3,4,5,6}"
NPROC="$(echo "$GPUS" | tr ',' '\n' | wc -l)"
EPOCHS="${EPOCHS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_PIXELS="${MAX_PIXELS:-802816}"
OPTIM="${OPTIM:-adamw}"
PERTURB_MODE="${PERTURB_MODE:-none}"
PERTURB_PROB="${PERTURB_PROB:-0.0}"
CONFLICT_TYPE_WEIGHTS="${CONFLICT_TYPE_WEIGHTS:-}"
LOG_EVERY="${LOG_EVERY:-25}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B-Instruct}"

PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"

CMD=(
  "$ACCELERATE" launch
  --num_processes "$NPROC"
  --mixed_precision bf16
  scripts/train_sft.py
    --epochs "$EPOCHS"
    --grad-accum "$GRAD_ACCUM"
    --shuffle-idx
    --max-image-pixels "$MAX_PIXELS"
    --optim "$OPTIM"
    --perturb-mode "$PERTURB_MODE"
    --perturb-prob "$PERTURB_PROB"
    --conflict-type-weights "$CONFLICT_TYPE_WEIGHTS"
    --log-every "$LOG_EVERY"
    --save-every "$SAVE_EVERY"
    --model-name "$MODEL_NAME"
)
if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  CMD+=(--load-in-4bit)
fi
CMD+=("$@")

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# RTX 4090 (Ada) does not support P2P / IB — accelerate auto-disables for multi-card
# but not always for num_processes=1; set explicitly for safety.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "[sft] CUDA_VISIBLE_DEVICES=$GPUS  num_processes=$NPROC  model=$MODEL_NAME  epochs=$EPOCHS  grad_accum=$GRAD_ACCUM  max_pixels=$MAX_PIXELS  optim=$OPTIM  perturb=${PERTURB_MODE}@${PERTURB_PROB}  load_in_4bit=$LOAD_IN_4BIT"

if [[ "${BG:-0}" == "1" ]]; then
  LOG="/tmp/sft_$(date +%Y%m%d_%H%M%S).log"
  echo "[sft] background mode, log → $LOG"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  PID=$!
  disown
  echo "[sft] training PID $PID (Ctrl+C exits tail; training keeps running)"
  sleep 1
  exec tail -f "$LOG"
else
  echo "[sft] foreground mode (Ctrl+C kills training)"
  exec "${CMD[@]}"
fi
