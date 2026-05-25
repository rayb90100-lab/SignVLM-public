#!/usr/bin/env bash
# scripts/dpo.sh — minimal launcher for SignVLM Stage 2 DPO RFT
#
# Defaults: 6 cards (GPUs 1-6), 2 epochs, init from C0.3 SFT adapter.
# Forwards any extra positional args to scripts/train_dpo.py.
#
# Usage:
#   ./scripts/dpo.sh                                # foreground tqdm progress
#   BG=1 ./scripts/dpo.sh                           # background (nohup) + tail log
#   MAX_STEPS=5 GPUS=1 ./scripts/dpo.sh             # single-card dry-run
#   BETA=0.3 ./scripts/dpo.sh
#   CONFLICT_TYPE_WEIGHTS=0.15,0.55,0.15,0.15 ./scripts/dpo.sh
#
# Env-var defaults (override with VAR=value prefix):
#   GPUS=1,2,3,4,5,6        comma-separated CUDA device ids
#   EPOCHS=2
#   GRAD_ACCUM=4
#   MAX_PIXELS=802816       per-image max pixels (DPO forward 2× SFT — keep conservative)
#   OPTIM=adamw             'adamw' or 'adamw8bit'
#   INIT_ADAPTER=runs/sft/20260512_021549/final     C0.3 adapter (best W2 SFT)
#   CONFLICT_RATIO=0.7      fraction of conflict pairs vs control
#   CONFLICT_TYPE_WEIGHTS=  empty=uniform; '0.15,0.55,0.15,0.15' direction-up
#   BETA=0.1                DPO temperature β
#   LR=5e-6                 DPO recommended ~40x smaller than SFT
#   LOG_EVERY=10
#   SAVE_EVERY=500
#   MAX_STEPS=0             >0 = stop after N steps (dry-run / sanity)
#   LOAD_IN_4BIT=0          1 = QLoRA NF4 base (saves ~10 GB; row 5 4bit Tab 4)
#   MODEL_NAME=Qwen2.5-VL-7B-Instruct   under ckpts/; use Qwen2.5-VL-3B-Instruct for 3B twin runs
#   SINGLE_PASS=0           1 = v1 concat batch + single backward (3B multi-card); 0 = plan A (7B single-card default)
#   BG=0                    1 = nohup background + tail
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-1,2,3,4,5,6}"
NPROC="$(echo "$GPUS" | tr ',' '\n' | wc -l)"
EPOCHS="${EPOCHS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_PIXELS="${MAX_PIXELS:-802816}"
OPTIM="${OPTIM:-adamw}"
INIT_ADAPTER="${INIT_ADAPTER:?must set INIT_ADAPTER, e.g. INIT_ADAPTER=runs/sft/<your-sft-run>/final ./scripts/dpo.sh}"
CONFLICT_RATIO="${CONFLICT_RATIO:-0.7}"
CONFLICT_TYPE_WEIGHTS="${CONFLICT_TYPE_WEIGHTS:-}"
BETA="${BETA:-0.1}"
LR="${LR:-5e-6}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-500}"
MAX_STEPS="${MAX_STEPS:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B-Instruct}"
SINGLE_PASS="${SINGLE_PASS:-0}"
TOKEN_MASK_DPO="${TOKEN_MASK_DPO:-0}"
KL_WEIGHT="${KL_WEIGHT:-0.1}"

PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"

CMD=(
  "$ACCELERATE" launch
  --num_processes "$NPROC"
  --mixed_precision bf16
  scripts/train_dpo.py
    --epochs "$EPOCHS"
    --grad-accum "$GRAD_ACCUM"
    --shuffle-idx
    --max-image-pixels "$MAX_PIXELS"
    --optim "$OPTIM"
    --init-adapter "$INIT_ADAPTER"
    --conflict-ratio "$CONFLICT_RATIO"
    --conflict-type-weights "$CONFLICT_TYPE_WEIGHTS"
    --beta "$BETA"
    --lr "$LR"
    --log-every "$LOG_EVERY"
    --save-every "$SAVE_EVERY"
    --max-steps "$MAX_STEPS"
    --model-name "$MODEL_NAME"
)
if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  CMD+=(--load-in-4bit)
fi
if [[ "$SINGLE_PASS" == "1" ]]; then
  CMD+=(--single-pass)
fi
if [[ "$TOKEN_MASK_DPO" == "1" ]]; then
  CMD+=(--token-mask-dpo --kl-weight "$KL_WEIGHT")
fi
CMD+=("$@")

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "[dpo] CUDA_VISIBLE_DEVICES=$GPUS  num_processes=$NPROC  model=$MODEL_NAME  single_pass=$SINGLE_PASS  epochs=$EPOCHS  grad_accum=$GRAD_ACCUM  max_pixels=$MAX_PIXELS  beta=$BETA  lr=$LR  conflict_ratio=$CONFLICT_RATIO  load_in_4bit=$LOAD_IN_4BIT  init=$INIT_ADAPTER"

if [[ "${BG:-0}" == "1" ]]; then
  LOG="/tmp/dpo_$(date +%Y%m%d_%H%M%S).log"
  echo "[dpo] background mode, log → $LOG"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  PID=$!
  disown
  echo "[dpo] training PID $PID (Ctrl+C exits tail; training keeps running)"
  sleep 1
  exec tail -f "$LOG"
else
  echo "[dpo] foreground mode (Ctrl+C kills training)"
  exec "${CMD[@]}"
fi
