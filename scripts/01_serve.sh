#!/usr/bin/env bash
# Start the vLLM server for one precision. Run in its own terminal; leave it up
# while running scripts 02/03. Only one server at a time on the 24 GB card.
#
#   bash scripts/01_serve.sh bf16
#   bash scripts/01_serve.sh int4
#   INT4_MODEL=checkpoints/qwen2.5-vl-7b-awq-w4a16-g128 bash scripts/01_serve.sh int4   # own checkpoint
#
# Flags follow C_PROJECT_PROPOSAL.md section 4.1. Adjust MAX_PIXELS after the
# pilot (200704 ~ 252 tokens per 1600x900 view; 401408 ~ 512 tokens).
set -euo pipefail

PRECISION="${1:?usage: 01_serve.sh bf16|int4}"
CONFIG="${CONFIG:-config/pilot.yaml}"
# Defaults come from the config's serving: section so the frozen copy can be
# enforced; environment variables override them for pilots only. Every
# serving setting is written to the manifest, and the manifest records this
# process's pid (exec keeps it) so the run guard can bind it to the server.
cfgval() { python -c "import sys,yaml; c=yaml.safe_load(open('$CONFIG')); print(c$1)"; }
PORT="${PORT:-8000}"
MAX_PIXELS="${MAX_PIXELS:-$(cfgval "['serving']['max_pixels']")}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(cfgval "['serving']['max_model_len']")}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(cfgval "['serving']['max_num_seqs']")}"
SEED="${SEED:-$(cfgval "['serving']['seed']")}"
PREFIX_CACHING="${PREFIX_CACHING:-$(cfgval "['serving']['prefix_caching']" | tr 'A-Z' 'a-z')}"
GPU_UTIL="${GPU_UTIL:-0.90}"
# Equal KV-cache byte capacity across precisions (proposal 5.2), so peak memory
# reflects weights plus a fixed cache rather than a fraction of the card.
# 2 GiB is ~9x what one 4096-token sequence needs for this model (~57 KB per
# token) and fits beside BF16 weights on 24 GB. Set KV_BYTES= (empty) to fall
# back to --gpu-memory-utilization, which was the pilot behavior.
KV_BYTES="${KV_BYTES-$(cfgval "['serving']['kv_bytes']")}"
BF16_REVISION="${BF16_REVISION:-$(cfgval "['quantize']['revision']")}"

case "$PRECISION" in
  bf16)
    MODEL="$(cfgval "['quantize']['base_model']")"
    SERVED="qwen2.5-vl-7b-bf16"
    REVISION="$BF16_REVISION"
    EXTRA=(--dtype bfloat16 --revision "$REVISION")
    ;;
  int4)
    # Official checkpoint by default (pilot). INT4_MODEL=<dir> serves our own
    # llm-compressor checkpoint (scripts/quantize_awq.py); its config.json
    # carries quant_method compressed-tensors, which vLLM needs told explicitly
    # because awq_marlin would reject it. Same served name, so config and
    # results do not change; the manifest records which checkpoint ran.
    MODEL="${INT4_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"
    SERVED="qwen2.5-vl-7b-awq"
    REVISION=""
    INT4_QUANT="${INT4_QUANT:-}"
    if [[ -z "$INT4_QUANT" ]]; then
      if [[ -f "$MODEL/config.json" ]] && grep -q compressed-tensors "$MODEL/config.json"; then
        INT4_QUANT=compressed-tensors
      else
        INT4_QUANT=awq_marlin
      fi
    fi
    EXTRA=(--dtype float16 --quantization "$INT4_QUANT")
    ;;
  *) echo "unknown precision: $PRECISION" >&2; exit 1 ;;
esac

case "$PREFIX_CACHING" in
  true) PREFIX_ARGS=(--enable-prefix-caching) ;;
  false) PREFIX_ARGS=(--no-enable-prefix-caching) ;;
  *) echo "PREFIX_CACHING must be true or false, got $PREFIX_CACHING" >&2; exit 1 ;;
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
  echo "revision: $REVISION"
  echo "served_model_name: $SERVED"
  echo "max_pixels: $MAX_PIXELS"
  echo "max_model_len: $MAX_MODEL_LEN"
  echo "kv_args: ${KV_ARGS[*]}"
  echo "seed: $SEED"
  echo "max_num_seqs: $MAX_NUM_SEQS"
  echo "prefix_caching: $PREFIX_CACHING"
  echo "extra_args: ${EXTRA[*]}"
  echo "port: $PORT"
  echo "pid: $$"
  echo "power_limit_w: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
  # the four pinned serving packages (PROTOCOL section 2), through the same collector the
  # freeze uses (module __version__, build tag kept); the freeze check requires every one
  python "$(dirname "$0")/runtime_versions.py"
} | tee "manifests/serve_${PRECISION}.txt"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt '{"image":6}' \
  --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
  --seed "$SEED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  "${PREFIX_ARGS[@]}" \
  "${KV_ARGS[@]}" \
  "${EXTRA[@]}"
