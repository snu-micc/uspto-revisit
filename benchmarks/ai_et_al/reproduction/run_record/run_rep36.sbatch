#!/bin/bash
#SBATCH -J llama_rep36
#SBATCH -o <server_home>/llama_benchmark/rep36/logs/%x_%j.out
#SBATCH -e <server_home>/llama_benchmark/rep36/logs/%x_%j.err
#SBATCH -p 5000_ada
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --gres=shard:5000ada:31
#SBATCH --time=12:00:00

# llama_infer_accessory.sh 와 같은 환경/자원으로 실행
START_TIME=$(date +%s)
echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Start    : $(date)"
echo "Node     : $(hostname)"
echo "========================================"

module purge
module load cuda/11.8 miniconda oneAPI/2025.2.1

RUN_DIR=<server_home>/llama_benchmark/rep36
cd <server_home>/llama_benchmark
ENV_PYTHON="<server_home>/.conda/envs/llama_env/bin/python"
mkdir -p "$RUN_DIR/output" "$RUN_DIR/run_record"

# ── 실행 환경 기록 ─────────────────────────────────────────────────────────
{
  echo "run_date: $(date --iso-8601=seconds)"
  echo "slurm_job_id: $SLURM_JOB_ID"
  echo "node: $(hostname)"
  echo "slurm_alloc: $(scontrol show job $SLURM_JOB_ID | grep -o 'AllocTRES=[^ ]*')"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
  $ENV_PYTHON - <<'EOF'
import sys, torch, importlib.metadata as md
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cudnn:", torch.backends.cudnn.version())
for p in ("transformers", "fairscale", "sentencepiece", "numpy"):
    try: print(f"{p}:", md.version(p))
    except Exception as e: print(f"{p}: not installed")
if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0))
EOF
  echo "accessory_commit: $(git -C LLaMA2-Accessory rev-parse HEAD)"
  echo "accessory_remote: $(git -C LLaMA2-Accessory remote get-url origin)"
  echo "accessory_local_changes:"; git -C LLaMA2-Accessory status --short
} > "$RUN_DIR/run_record/environment.txt" 2>&1
git -C LLaMA2-Accessory diff > "$RUN_DIR/run_record/LLaMA2-Accessory_local.diff"
cp ai_compare_accessory.py llama_infer_accessory.sh "$RUN_DIR/run_record/"
cp "$RUN_DIR/ai_compare_accessory_rep36.py" "$RUN_DIR/run_rep36.sbatch" "$RUN_DIR/run_record/"

# 체크포인트 해시 (추론과 병행)
( cd finetuned_model && sha256sum consolidated.00-of-01.model_fixed.pth tokenizer.model \
    accessory_pretrained/config.json accessory_pretrained/meta.json \
    > "$RUN_DIR/run_record/checkpoint_sha256.txt" ) &
HASH_PID=$!

# shard 할당은 GPU 메모리를 보장하지 않으므로, 모델(bf16 7B ≈ 14 GiB)이 올라갈 여유가 생길 때까지 대기
NEED_MIB=16384
WAIT_LIMIT=$((12 * 60 * 60))
WAITED=0
while true; do
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE_MIB" -ge "$NEED_MIB" ]; then
    echo "GPU free ${FREE_MIB} MiB >= ${NEED_MIB} MiB, starting at $(date --iso-8601=seconds)"
    break
  fi
  if [ "$WAITED" -ge "$WAIT_LIMIT" ]; then
    echo "GPU free ${FREE_MIB} MiB < ${NEED_MIB} MiB after ${WAITED}s, giving up"
    exit 1
  fi
  [ $((WAITED % 300)) -eq 0 ] && echo "waiting for GPU memory: free ${FREE_MIB} MiB ($(date +%T))"
  sleep 30
  WAITED=$((WAITED + 30))
done

srun $ENV_PYTHON -u "$RUN_DIR/ai_compare_accessory_rep36.py"
RC=$?
wait $HASH_PID

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "========================================"
echo "End      : $(date)"
echo "Elapsed  : $(($ELAPSED/3600))h $(($ELAPSED%3600/60))m $(($ELAPSED%60))s"
echo "Exit     : $RC"
echo "========================================"
exit $RC
