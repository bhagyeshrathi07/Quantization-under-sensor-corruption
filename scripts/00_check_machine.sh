#!/usr/bin/env bash
# Records host/GPU facts for the run manifest. Output goes to stdout and manifests/machine.txt.
set -u
mkdir -p manifests
{
  echo "date: $(date -Iseconds)"
  echo "host: $(hostname)"
  echo "os: $(uname -srmo)"
  echo "python: $(python3 --version 2>&1)"
  echo "uv: $(uv --version 2>&1 || echo missing)"
  echo
  echo "== GPU =="
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,power.limit --format=csv 2>&1
  echo
  echo "== CUDA toolkit (nvcc) =="
  nvcc --version 2>&1 | tail -1 || echo "nvcc missing (fine; vLLM wheels bundle CUDA)"
  echo
  echo "== RAM =="
  free -h
  echo
  echo "== Disk (need ~50 GB for models + DriveBench images) =="
  df -h . "$HOME" 2>/dev/null | sort -u
  echo
  echo "== HF cache =="
  du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || echo "no HF cache yet"
} | tee manifests/machine.txt
