#!/usr/bin/env bash
# Start the vLLM server for one precision. Run in its own terminal; leave it up
# while running scripts 02/03. Only one server at a time on the 24 GB card.
#
#   bash scripts/01_serve.sh bf16
#   bash scripts/01_serve.sh int4
#
# Flags follow C_PROJECT_PROPOSAL.md section 4.1. Adjust MAX_PIXELS after the
# pilot (200704 ~ 252 tokens per 1600x900 view; 401408 ~ 512 tokens).
set -euo pipefail

PRECISION="${1:?usage: 01_serve.sh bf16|int4}"
PORT="${PORT:-8000}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"
# Equal KV-cache byte capacity across precisions (proposal 5.2). Requires a
# vLLM version that supports --kv-cache-memory-bytes; leave empty to skip and
# use --gpu-memory-utilization for the pilot only.
KV_BYTES="${KV_BYTES:-}"

case "$PRECISION" in
  bf16)
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    SERVED="qwen2.5-vl-7b-bf16"
    EXTRA=(--dtype bfloat16)
    ;;
  int4)
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"   # official checkpoint, pilot only
    SERVED="qwen2.5-vl-7b-awq"
    EXTRA=(--dtype float16 --quantization awq_marlin)
    ;;
  *) echo "unknown precision: $PRECISION" >&2; exit 1 ;;
esac

KV_ARGS=()
if [[ -n "$KV_BYTES" ]]; then
  KV_ARGS=(--kv-cache-memory-bytes "$KV_BYTES")
else
  KV_ARGS=(--gpu-memory-utilization "$GPU_UTIL")
fi

mkdir -p manifests
{
  echo "date: $(date -Iseconds)"
  echo "precision: $PRECISION"
  echo "model: $MODEL"
  echo "served_model_name: $SERVED"
  echo "max_pixels: $MAX_PIXELS"
  echo "max_model_len: $MAX_MODEL_LEN"
  echo "kv_args: ${KV_ARGS[*]}"
  echo "vllm: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo unknown)"
} | tee "manifests/serve_${PRECISION}.txt"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt '{"image":6}' \
  --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
  --seed 0 \
  --max-num-seqs 1 \
  "${KV_ARGS[@]}" \
  "${EXTRA[@]}"
