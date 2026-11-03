#!/bin/bash
#SBATCH -J llama_acc
#SBATCH -o logs_test/%x_%j.out
#SBATCH -e logs_test/%x_%j.err
#SBATCH -p 5000_ada
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --gres=shard:30
#SBATCH --time=72:00:00

# ── 시작 시간 기록 ────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Start    : $(date)"
echo "Node     : $(hostname)"
echo "========================================"

# 1. module 가져오기
module purge
module load cuda/11.8 miniconda oneAPI/2025.2.1

# 2. log directory 만들기
mkdir -p <server_home>/llama_benchmark/logs_test
cd <server_home>/llama_benchmark

# 3. 시스템의 공용 콘다 경로 설정
ENV_PYTHON="<server_home>/.conda/envs/llama_env/bin/python"
echo "Python   : $ENV_PYTHON"
echo "Starting inference (accessory / LLaMA-Adapter)..."
echo "========================================"

# 4. slurm 작업 제출
srun $ENV_PYTHON -u ai_compare_accessory.py

# ── 종료 시간 및 경과 시간 계산 ──────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================"
echo "End      : $(date)"
echo "Elapsed  : $(($ELAPSED/3600))h $(($ELAPSED%3600/60))m $(($ELAPSED%60))s"
echo "========================================"
