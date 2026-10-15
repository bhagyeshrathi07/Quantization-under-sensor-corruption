# Pilot runbook (weeks 4–5)

Covers proposal timeline rows "4–5": serve BF16 and INT4, smoke request, 20-question DriveBench run, scene split. Everything writes to `results/pilot/` and `manifests/`.

## 0. One-time setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]" vllm            # pin the vllm version that works; record it in manifests/serve_*.txt
pytest                                     # offline unit tests, no server needed
bash scripts/00_check_machine.sh           # -> manifests/machine.txt
```

Data:

```bash
uv pip install -e ".[data]"
python scripts/download_drivebench.py --json-only   # 32 MB of annotations, enough for 04_scene_split.py
python scripts/download_drivebench.py               # + DriveBench.zip from Google Drive (large), extract, verify
python scripts/download_drivebench.py --verify      # re-check layout any time
```

Resulting layout (upstream `docs/DATA_PREPAER.md`):

```
data/
  drivebench-test.json        # 1461 questions, 174 scenes, 1200 clean images
  nuscenes/samples/CAM_*/     # clean images
  corruption/LowLight/CAM_*/  # corrupted images, same file names
  corruption/Fog/CAM_*/
```

If Google Drive blocks the large download, get `DriveBench.zip` in a browser, drop it in `data/`, rerun. Config lives in `config/pilot.yaml`.

## 1. Serve (own terminal, one precision at a time)

```bash
bash scripts/01_serve.sh bf16      # Qwen/Qwen2.5-VL-7B-Instruct
# later:
bash scripts/01_serve.sh int4      # Qwen/Qwen2.5-VL-7B-Instruct-AWQ (official, pilot only)
```

Wait for `Application startup complete`. If BF16 fails to start or OOMs at startup, first lower `MAX_PIXELS`; if it still fails, switch `MODEL` to the 3B checkpoints (gate decision, record it).

## 2. Smoke request

```bash
python scripts/02_smoke_request.py --precision bf16
```

Gate checks printed at the end: six views accepted, logprobs returned, option letter in top-k, peak memory. `prompt_tokens / n_views` gives the image-token budget actually in effect.

## 3. Twenty-question run

```bash
python scripts/03_smoke_drivebench.py --precision bf16
python scripts/03_smoke_drivebench.py --precision bf16 --condition NoImage
# swap server to int4, then:
python scripts/03_smoke_drivebench.py --precision int4
python scripts/05_compare_flips.py results/pilot/bf16_clean.jsonl results/pilot/int4_clean.jsonl
```

Runs are resumable (rerun the same command to continue). `--n 0` runs every eligible question. The `.drivebench.json` output feeds the upstream scorer:

```bash
# in a DriveBench checkout, with data/visual_description.json present
PYTHONPATH=. python evaluate/eval.py /abs/path/results/pilot/bf16_clean.drivebench.json
```

## 4. Scene split

```bash
python scripts/04_scene_split.py    # -> manifests/scene_split.json
```

## Vast.ai (rented RTX 4090)

Rent: filter GPU = RTX 4090, disk ≥ 100 GB, on-demand (not interruptible), reliability ≥ 98%, download speed ≥ 500 Mbps, DLPerf doesn't matter. Template: **PyTorch (Vast)** or any `nvidia/cuda:12.x` Ubuntu image; launch mode **SSH**. Add your laptop's public key under Account → Keys first. Avoid the cheapest hosts; they are often power-limited, which skews latency.

Then, over SSH:

```bash
curl -sL https://raw.githubusercontent.com/bhagyeshrathi07/Quantization-under-sensor-corruption/main/scripts/vast_setup.sh | bash
```

That installs everything under `/workspace`, starts the weight and image downloads in the background, and prints the next commands. Record the host's machine ID (instance page) in `manifests/machine.txt`; profiling sessions must all run on the same host. Stop (not destroy) the instance between sessions to keep `/workspace`; storage is billed per GB-month while stopped. `rsync -avz vast:/workspace/qflip/results/ results/` before destroying.

## HPC (SJSU CoE-HPC3, `/data/cmpe249-fa26`)

Kept for reference; the team chose Vast.ai on October 14, 2026.

Facts verified October 14, 2026:

- Login is a two-hop: laptop to `coe-hpc1.sjsu.edu` (needs SJSU VPN), then `ssh coe-hpc3` (node `g17`). SLURM, `/data`, `/scratch` are on hpc3. An ssh config with `ProxyJump` makes `ssh hpc` one step.
- No direct internet anywhere on the cluster; `export http_proxy=http://172.16.1.2:3128 https_proxy=$http_proxy` works on g17.
- `/data` is >90% full and the README forbids dataset copies there. Use `/scratch/cmpe249-fa26/$USER` for weights and the DriveBench zip (13 TB free; wiped 21 days after semester end).
- nuScenes `samples/` is already at `/data/cmpe249-fa26/datasets/nuscenes/v1.0-trainval`. DriveBench clean images are those files, so symlink instead of downloading. Only the corrupted images come from the zip.
- `module load python3/3.12.12`. The course `mmdet3d-cu118` env is irrelevant to us.
- A100 nodes `cs[001-004]` (4 GPUs each) were idle; `sbatch --test-only` quoted an immediate start. H100s had a ~9 h wait. P100 nodes cannot run vLLM.
- GPU work is batch-only, one GPU per job, release promptly (course email, October 2026).

One-time setup on **g17** (CPU login node):

```bash
export http_proxy=http://172.16.1.2:3128 https_proxy=$http_proxy
S=/data/cmpe249-fa26/students/$USER; mkdir -p $S && cd $S
module load python3/3.12.12
python3 -m venv $S/qflip-venv && source $S/qflip-venv/bin/activate
git clone https://github.com/bhagyeshrathi07/Quantization-under-sensor-corruption.git qflip && cd qflip
pip install -e ".[dev,data]" vllm
pytest

python scripts/download_drivebench.py --json-only       # annotations, 32 MB
mkdir -p data && ln -s /data/cmpe249-fa26/datasets/nuscenes/v1.0-trainval data/nuscenes
python scripts/download_drivebench.py --verify          # expect "all clean images present"

export HF_HOME=/scratch/cmpe249-fa26/$USER/hf; mkdir -p $HF_HOME
hf download Qwen/Qwen2.5-VL-7B-Instruct                 # ~17 GB via proxy, on g17, NOT in a GPU job
hf download Qwen/Qwen2.5-VL-7B-Instruct-AWQ             # ~6 GB  (older hub: huggingface-cli download ...)
```

Corrupted images (LowLight, Fog) come from `DriveBench.zip` (Google Drive, size unknown). Download it to `/scratch/cmpe249-fa26/$USER/` via the proxy with `gdown`, or on a laptop and `scp` to `hpc:/scratch/cmpe249-fa26/<id>/`, then extract only what is needed:

```bash
cd $S/qflip && unzip -q /scratch/cmpe249-fa26/$USER/DriveBench.zip 'corruption/LowLight/*' 'corruption/Fog/*' -d data/
```

Until then, the `NoImage` condition needs no image files and exercises the full pipeline: `sbatch scripts/hpc_pilot.sbatch bf16 NoImage 20`.

Every new shell: `module load python3/3.12.12 && source $S/qflip-venv/bin/activate && export http_proxy=... https_proxy=...`.

Submit (one GPU, model loaded once, all conditions in one job, job exits when done):

```bash
mkdir -p logs
sbatch scripts/hpc_pilot.sbatch bf16 clean 20
sbatch scripts/hpc_pilot.sbatch int4 clean 20
squeue -u $USER
tail -f logs/qflip-<jobid>.out
python scripts/05_compare_flips.py results/pilot/bf16_clean.jsonl results/pilot/int4_clean.jsonl
# later, full matrix per precision in a single job:
sbatch scripts/hpc_pilot.sbatch bf16 clean,NoImage,LowLight,Fog 0
```

The job refuses to start if weights are not cached (`HF_HUB_OFFLINE=1`), so the GPU never sits idle on a download. `--time=02:00:00` is a cap, not a hold; the job ends when the scripts end. For `--n 0` full runs raise it to 3 h. The job pins `--gres=gpu:a100:1`; switch to `h100` only if the A100 queue backs up. Do not submit interactive GPU sessions.

Results and manifests land in the repo checkout under `$S`; `git add results`-style copying is not needed, rsync `results/` and `manifests/` to your laptop or commit manifests from the head node.

## What to record in the week-5 gate note

- vLLM / transformers versions, `MAX_PIXELS`, tokens per view, peak MiB for each precision
- whether option-letter logprobs came back on MCQ (margin feasibility)
- MCQ accuracy and harmful flips on the 20-question slice (sanity only, not a result)
- p50/p95 latency per precision
- decision: 7B or 3B

## Known differences from the released DriveBench adapter

- The released adapter resizes every view to 224×224. We send originals and cap tokens with `max_pixels` (`images.resize_to: null`). Set `resize_to: [224, 224]` to reproduce upstream exactly.
- Released decoding is temperature 0.2 / top_p 0.2 / 512 tokens; we use greedy / 128 tokens so flips are not sampling noise.
- We append "Begin your answer with the option letter." to MCQ questions so the margin can be read from the first token. Turn off with `mcq_suffix: ""`.
- The released MCQ normalizer matches the article "a" as option A; we record both that rule (`mcq_pred`) and a strict one (`mcq_pred_strict`).
- DriveBench "Dark" is the `LowLight` corruption directory.
