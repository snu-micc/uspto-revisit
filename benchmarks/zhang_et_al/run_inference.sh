#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_inference.sh BASE_MODEL_PATH [INPUT_CSV] [OUTPUT_DIR]
BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$BENCH_DIR/../.." && pwd)"
MODEL_PATH="${1:?Provide the base model path or Hugging Face model ID}"
INPUT_CSV="${2:-$REPO_ROOT/analysis/data/input.csv}"
OUTPUT_DIR="${3:-$REPO_ROOT/analysis/results/zhang_generation}"
PYTHON_BIN="${PYTHON:-python}"
for task in prod role; do
  test -f "$BENCH_DIR/adapters/$task/adapter_model.safetensors" || { echo "Extract llama3_8b_adapters.zip into $BENCH_DIR first." >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
"$PYTHON_BIN" "$BENCH_DIR/scripts/predict_csv.py" --model "$MODEL_PATH" --prod_adapter "$BENCH_DIR/adapters/prod" --role_adapter "$BENCH_DIR/adapters/role" --csv "$INPUT_CSV" --text_col paragraph --id_col idx --prod_unit paragraph --out "$OUTPUT_DIR/per_product.csv"
"$PYTHON_BIN" "$BENCH_DIR/scripts/collapse_predictions.py" "$OUTPUT_DIR/per_product.csv" --input "$INPUT_CSV" --out "$OUTPUT_DIR/per_paragraph.csv"
