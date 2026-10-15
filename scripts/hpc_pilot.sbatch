#!/bin/bash
# SJSU CoE HPC batch job: start vLLM on the allocated GPU, run the pilot
# scripts against it for one or more conditions, shut it down. GPU nodes have
# no interactive shells, so server and client both live inside this job.
#
# HPC rules (course email, Oct 2026): one GPU, keep it busy, release it when
# done. This job therefore (a) refuses to start unless the model weights are
# already cached (download them on the head node first, see PILOT.md),
# (b) loads the model once and runs every requested condition, (c) exits as
# soon as the last script finishes.
#
#   mkdir -p logs
#   sbatch scripts/hpc_pilot.sbatch bf16 clean 20                      # precision, conditions, n (0 = all)
#   sbatch scripts/hpc_pilot.sbatch int4 clean,NoImage,LowLight,Fog 0  # full matrix for one precision
#   squeue -u $USER ; tail -f logs/qflip-<jobid>.out
#
# Edit the paths under "Site settings" once. Submit from the repo root.

#SBATCH --job-name=qflip
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PRECISION="${1:-bf16}"
CONDITIONS="${2:-clean}"
N="${3:-20}"

# ---- Site settings -----------------------------------------------------------
STUDENT_DIR="/data/cmpe249-fa26/students/$USER"
VENV="$STUDENT_DIR/qflip-venv"            # created once, see PILOT.md "HPC"
# /data is >90% full; weights live on /scratch (wiped 21 days after semester end).
export HF_HOME="${HF_HOME:-/scratch/cmpe249-fa26/$USER/hf}"   # adjust once `ls /scratch` shows the real group path
export HF_HUB_OFFLINE=1                    # never download inside a GPU job
# -----------------------------------------------------------------------------

module load python3/3.12.12
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs manifests results/pilot

case "$PRECISION" in
  bf16) REPO="Qwen/Qwen2.5-VL-7B-Instruct" ;;
  int4) REPO="Qwen/Qwen2.5-VL-7B-Instruct-AWQ" ;;
  *) echo "unknown precision $PRECISION"; exit 1 ;;
esac
if ! python -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', local_files_only=True)" 2>/dev/null; then
  echo "weights for $REPO not in $HF_HOME. Download on the head node first (PILOT.md), then resubmit."
  exit 1
fi

echo "job $SLURM_JOB_ID on $(hostname), gpu: ${CUDA_VISIBLE_DEVICES:-?}, precision=$PRECISION conditions=$CONDITIONS n=$N"
bash scripts/00_check_machine.sh > "manifests/machine-$SLURM_JOB_ID.txt"

PORT=$((20000 + SLURM_JOB_ID % 10000))
export PORT
export QFLIP_BASE_URL="http://127.0.0.1:$PORT/v1"
export GPU_UTIL="${GPU_UTIL:-0.85}"

bash scripts/01_serve.sh "$PRECISION" > "logs/serve-$PRECISION-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping server"; kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "waiting for vLLM on port $PORT"
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then echo "server up after ~$((i*5))s"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died, see logs/serve-$PRECISION-$SLURM_JOB_ID.log"; exit 1; fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "server not up after 5 min"; exit 1; }

IFS=',' read -ra COND_LIST <<< "$CONDITIONS"
python scripts/02_smoke_request.py --precision "$PRECISION" --condition "${COND_LIST[0]}"
for c in "${COND_LIST[@]}"; do
  python scripts/03_smoke_drivebench.py --precision "$PRECISION" --condition "$c" --n "$N"
done

echo "done"
