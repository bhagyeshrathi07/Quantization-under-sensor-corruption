#!/usr/bin/env bash
# One-shot setup on a fresh Vast.ai (or any Ubuntu + CUDA) instance.
# Run as root or a sudo user from any directory:
#
#   curl -sL https://raw.githubusercontent.com/bhagyeshrathi07/Quantization-under-sensor-corruption/main/scripts/vast_setup.sh | bash
#
# Installs uv + the project + vLLM, downloads DriveBench annotations and the
# model weights, and prints the next commands. Images (DriveBench.zip) are
# started in the background because they take the longest.
set -euo pipefail

WORK="${WORK:-/workspace}"            # Vast mounts the instance disk here
REPO="https://github.com/bhagyeshrathi07/Quantization-under-sensor-corruption.git"
export HF_HOME="$WORK/hf"
export DEBIAN_FRONTEND=noninteractive

mkdir -p "$WORK" && cd "$WORK"
command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip tmux git curl > /dev/null)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d qflip ] || git clone -q "$REPO" qflip
cd qflip
uv venv -q .venv --python 3.12
source .venv/bin/activate
uv pip install -q -e ".[dev,data]" vllm
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
pytest -q 2>&1 | tail -1

python scripts/download_drivebench.py --json-only 2>&1 | tail -1
(hf download Qwen/Qwen2.5-VL-7B-Instruct > "$WORK/dl-bf16.log" 2>&1 \
  || huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct > "$WORK/dl-bf16.log" 2>&1) &
(python scripts/download_drivebench.py > "$WORK/dl-images.log" 2>&1) &

cat <<EOF

setup done. downloads running in background:
  tail -f $WORK/dl-bf16.log      # ~17 GB weights
  tail -f $WORK/dl-images.log    # DriveBench.zip + extract + verify

every new shell:
  cd $WORK/qflip && source .venv/bin/activate && export HF_HOME=$WORK/hf

then (two tmux panes):
  bash scripts/01_serve.sh bf16
  python scripts/02_smoke_request.py --precision bf16 --condition NoImage   # works before images finish
  python scripts/03_smoke_drivebench.py --precision bf16
EOF
