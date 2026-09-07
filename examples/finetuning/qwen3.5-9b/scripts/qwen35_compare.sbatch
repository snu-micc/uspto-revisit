#!/bin/bash
#SBATCH --job-name=qwen35-qlora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=qwen35-%j.out
#SBATCH --error=qwen35-%j.err

set -euo pipefail

# Submit from examples/finetuning/qwen3.5-9b after activating the training environment.
PROJECT="${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the Qwen directory}"
cd "$PROJECT"
PYTHON="${QWEN_PYTHON:-python}"
mkdir -p "$PROJECT/tmp" "$PROJECT/hf-cache"

export HF_HOME="$PROJECT/hf-cache"
export HUGGINGFACE_HUB_CACHE="$PROJECT/hf-cache/hub"
export TRANSFORMERS_CACHE="$PROJECT/hf-cache/transformers"
export TMPDIR="$PROJECT/tmp"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=$SLURM_JOB_ID host=$(hostname) visible_gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

"$PYTHON" -m src.infer \
  --data data/eval.jsonl \
  --output outputs/baseline.jsonl \
  --max-new-tokens 1024

# The historical run resumed from checkpoint-14. A fresh reproduction starts
# without it; set QWEN_RESUME only to a complete Trainer checkpoint if resuming.
resume_args=()
if [[ -n "${QWEN_RESUME:-}" ]]; then
  resume_args=(--resume-from-checkpoint "$QWEN_RESUME")
fi
"$PYTHON" -m src.train_qlora \
  --train-data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --output outputs/qlora \
  --max-length 2048 \
  --epochs 5 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4 \
  --save-steps 14 \
  --eval-steps 14 \
  --seed 42 \
  "${resume_args[@]}"

"$PYTHON" -m src.infer \
  --data data/eval.jsonl \
  --adapter outputs/qlora/final \
  --output outputs/finetuned.jsonl \
  --max-new-tokens 1024

"$PYTHON" -m src.compare \
  --baseline outputs/baseline.jsonl \
  --finetuned outputs/finetuned.jsonl \
  --output outputs/comparison.md

echo "completed_at=$(date --iso-8601=seconds)"
