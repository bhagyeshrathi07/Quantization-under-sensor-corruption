#!/usr/bin/env bash
# Serve an open-weights judge model with vLLM on a second port, OpenAI-compatible,
# for the VQA rubric judge (scripts/08_judge_vqa.py --base-url http://localhost:8001/v1
# --model judge-local). Run after the study runs are done: it shares the GPU.
#
#   bash scripts/serve_judge.sh                                   # default JUDGE_MODEL below
#   JUDGE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 bash scripts/serve_judge.sh
#
# JUDGE_REVISION (the checkpoint commit) is required and written to the manifest
# (manifests/serve_judge.txt), which the judge freeze and final-split judging check;
# JUDGE_ALLOW_UNPINNED=1 serves the current revision for development trials only.
# Not a Qwen model: the model under test is Qwen, and a same-family judge invites an
# obvious objection.
set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-20b}"
JUDGE_REVISION="${JUDGE_REVISION:-}"
SERVED="${JUDGE_SERVED:-judge-local}"
PORT="${JUDGE_PORT:-8001}"
MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${JUDGE_GPU_UTIL:-0.85}"

REV_ARGS=()
if [[ -n "$JUDGE_REVISION" ]]; then
  REV_ARGS=(--revision "$JUDGE_REVISION")
elif [[ "${JUDGE_ALLOW_UNPINNED:-0}" != "1" ]]; then
  echo "serve_judge.sh: JUDGE_REVISION is empty. Pin the checkpoint revision (judge freeze and final-split judging require it), or set JUDGE_ALLOW_UNPINNED=1 for a development trial." >&2
  exit 2
fi

mkdir -p manifests
{
  echo "date: $(date -Iseconds)"
  echo "role: judge"
  echo "model: $JUDGE_MODEL"
  echo "revision: $JUDGE_REVISION"
  echo "pinned: $([[ -n "$JUDGE_REVISION" ]] && echo yes || echo no)"
  echo "served_model_name: $SERVED"
  echo "max_model_len: $MAX_MODEL_LEN"
  echo "gpu_memory_utilization: $GPU_UTIL"
  echo "port: $PORT"
  echo "pid: $$"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
  python "$(dirname "$0")/runtime_versions.py"
} > manifests/serve_judge.txt

exec vllm serve "$JUDGE_MODEL" "${REV_ARGS[@]}" \
  --served-model-name "$SERVED" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs 4 \
  --seed 0
