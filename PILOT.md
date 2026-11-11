# Runbook

Setup, data, serving, runs, stability, profiling, and host notes. Written for the week 4–5 pilot and kept current; the week-5 gate results are in `manifests/gate_week5.md` and the protocol itself in `PROTOCOL.md`. Everything writes to `results/` (not tracked in git) and `manifests/`.

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
python scripts/download_drivebench.py               # + DriveBench.zip from Google Drive (7.2 GB, corruption images only), extract, verify
python scripts/download_clean_images.py             # clean nuScenes keyframes from the public AWS mirror (streams ~45 GB, keeps ~200 MB, no account)
python scripts/download_drivebench.py --verify      # re-check layout any time
```

Resulting layout (upstream `docs/DATA_PREPAER.md`):

```
data/
  drivebench-test.json        # 1461 questions, 174 scenes, 1200 clean images
  nuscenes/samples/CAM_*/     # clean images: NOT in the zip; fetched by download_clean_images.py (or a symlink on the HPC)
  corruption/LowLight/CAM_*/  # corrupted images, same file names
  corruption/Fog/CAM_*/
```

If Google Drive blocks the large download, get `DriveBench.zip` in a browser, drop it in `data/`, rerun. Config lives in `config/pilot.yaml`.

Clean images come from Motional's public mirror on the AWS Open Data registry (`s3://motional-nuscenes`, anonymous, CC BY-NC-SA 4.0). The 1200 keyframes sit inside ten `v1.0-trainvalNN_keyframes.tgz` tarballs (4 to 6 GB each) with no scene index, so the script streams each tarball through `tar` and keeps only the referenced files, stopping when none are missing. Worst case ~45 GB of transfer: about 10 minutes on the Vast box, about an hour on a home connection. Rerunning skips files already present.

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

Runs are resumable (rerun the same command to continue) but only under the same settings: each result file has a `.meta.json` sidecar and a rerun with a different split, config, prompt, checkpoint, or serving flag is refused (use another `--out-dir`). The default split is development. `--n 0` runs every eligible question; `--split final --out-dir results/final` is the full run and needs the core freeze (`python scripts/freeze_config.py`), which the script checks together with the running server's manifest. Pilot files from before November 4 have no sidecar; `05_compare_flips.py --no-provenance` reads them as exploratory output and refuses any file whose sidecar declares the final split. Pass the files as reference BF16 then quantized INT4; the reverse order is refused. `05_compare_flips.py` takes any number of ref/quant pairs, reports flip counts with paired bootstrap CIs over scenes, breaks them down with `--by cell n_views margin condition`, and prints groups below the coverage minimum (`analysis.min_questions` / `min_scenes` in config, or `--min-n` / `--min-scenes`) as UNASSESSED. The `.drivebench.json` output feeds the upstream scorer's MCQ accuracy and language metrics:

```bash
# in a DriveBench checkout (its GPT path does not run as shipped; use 08_judge_vqa.py for the rubric)
PYTHONPATH=. python evaluate/eval.py /abs/path/results/pilot/bf16_clean.drivebench.json
```

### Full matrix

```bash
DRY_RUN=1 bash scripts/run_matrix.sh                     # prints the 8 commands
bash scripts/run_matrix.sh                               # bf16 then int4, 4 conditions, final split -> results/final
CONDITIONS="clean Fog" SPLIT=development OUT_DIR=results/dev N=20 bash scripts/run_matrix.sh   # development pilot
INT4_MODEL=checkpoints/qwen2.5-vl-7b-awq-w4a16-g128 bash scripts/run_matrix.sh int4            # own checkpoint
```

The driver starts `01_serve.sh` itself, waits for `/health` (up to 20 min), runs every condition, copies `manifests/serve_<p>.txt` next to the results, and stops the server before the next precision. Rerunning continues where it stopped. Run it inside `tmux`; a full precision on final scenes is roughly 1116 questions × p50 latency, about 45 min for BF16 and 20 min for INT4 per condition at pilot speeds. `RSYNC_TO=user@host:/path` syncs results after each run.

### VQA judge

```bash
export OPENAI_API_KEY=...                                                  # this shell only; never in a file
python scripts/08_judge_vqa.py results/pilot/bf16_LowLight.jsonl --dry-run   # counts and an example prompt, no key needed
python scripts/08_judge_vqa.py results/final/bf16_clean.jsonl results/final/int4_clean.jsonl --limit 20   # cost check
python scripts/08_judge_vqa.py results/final/*_*.jsonl --workers 8
python scripts/05_compare_flips.py results/final/bf16_clean.jsonl results/final/int4_clean.jsonl --vqa --judge-threshold 60
```

Writes `<file>.judge.jsonl` next to each result file; resumable. The pilot's 12 LowLight VQA answers are about 13k prompt tokens per precision, so the full matrix (657 VQA questions × 8 runs) is roughly 5.5M prompt tokens plus about 300 output tokens per call. Judge model and threshold are team item 2.

### Own INT4 checkpoint

```bash
# separate venv: llm-compressor pins its own torch/transformers and must not touch the vLLM venv
uv venv .venv-quant --python 3.12 && VIRTUAL_ENV=$PWD/.venv-quant uv pip install -e ".[quant]"
.venv-quant/bin/python scripts/quantize_awq.py --dry-run  # manifest + calibration list, no model load
TOKENIZERS_PARALLELISM=false .venv-quant/bin/python scripts/quantize_awq.py   # 8 min on the 4090 (with the server stopped)
INT4_MODEL=checkpoints/qwen2.5-vl-7b-awq-w4a16-g128 bash scripts/01_serve.sh int4
.venv-quant/bin/python scripts/quantize_awq.py --retain "re:.*language_model\.layers\.(3|7)\..*" --out-dir checkpoints/retain-3-7   # mitigation
```

Measured November 3: bf16 load on CPU, 107 six-view calibration samples, 29 sequential stages, 488 s total, 11 GB GPU peak, checkpoint 6.5 GB. The warning "28 mappings were skipped due to incompatible shapes" is the per-layer `v_proj -> o_proj` balance that grouped-query attention rules out; expected. The serve script detects a compressed-tensors checkpoint from its `config.json` and passes `--quantization compressed-tensors`. Run own-checkpoint slices with `--out-dir results/pilot_ownawq` so they do not overwrite the official-AWQ files (same `int4_` prefix).

## 4. Stability floor and deployment profile

```bash
# same server as above; 60 development-scene questions, 3 repeats, per precision and condition
python scripts/06_stability.py --precision bf16 --condition clean
python scripts/06_stability.py --precision bf16 --condition clean --report      # summarize only

# profile: 50 six-view development questions, streamed for TTFT; one invocation = one session, run three
python scripts/07_profile.py --precision bf16 --session 1
python scripts/07_profile.py --report results/profile/*.json
```

`01_serve.sh` now pins `--kv-cache-memory-bytes` to 2 GiB for both precisions, so peak memory from `07_profile.py` compares weights plus a fixed cache. The pilot used `--gpu-memory-utilization 0.9` instead (`KV_BYTES=` restores that), which is why its memory numbers are not a result. The stability summary's `self_change_rate` is the floor any BF16-versus-INT4 `change_rate` must clear.

## 5. Scene split

```bash
python scripts/04_scene_split.py    # -> manifests/scene_split.json
```

## Vast.ai (rented RTX 4090)

Rent: filter GPU = RTX 4090 (24 GB; skip the 48 GB modded cards), disk ≥ 100 GB, on-demand (not interruptible), reliability ≥ 98%, download speed ≥ 500 Mbps, **Max CUDA ≥ 13.0** (vLLM wheels are cu130). DLPerf matters as a power-cap tell: about 97 means a stock 450 W card, about 74 means a capped one (the October host). Template: **PyTorch (Vast)**; launch mode **SSH**. Add your laptop's public key under Account → Keys first. Stopped instances can be destroyed by the host (happened October 2026), so sync results after every session.

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

## GPU session log

- **October 15 (host m:3752, 300 W cap, vLLM 0.31):** gate pilot, LowLight 20 questions both precisions. `manifests/gate_week5.md`.
- **November 7–8 (same host):** core freeze, server verification, own-checkpoint stability, profile ×3, the 8-run matrix (zero errors), confirmatory analysis, judge trials. `manifests/session2_2026-11-08.md`.
- **November 3 (host m:11851, 450 W, vLLM 0.33):** the October instance had been destroyed by its host; rebuilt from `vast_setup.sh`. Clean images, KV flag and streaming verified, clean/Fog/NoImage slices both precisions, stability, profile session 1, own AWQ checkpoint built and smoke-tested. `manifests/session1_2026-11-03.md`. Everything below in the checklist is done except the freeze.

## Next GPU session checklist (written after the October 15 pilot; completed November 3)

Resume the same Vast host (profiling must stay on one host), then in order:

```bash
cd /workspace/qflip && git pull && source .venv/bin/activate && export HF_HOME=/workspace/hf
python scripts/download_clean_images.py                       # ~10 min at 800 Mbps; all ten tarballs needed
python scripts/download_drivebench.py --verify                # expect "all clean images present"
bash scripts/01_serve.sh bf16                                 # separate pane; confirms --kv-cache-memory-bytes is accepted
python scripts/02_smoke_request.py --precision bf16           # clean condition now possible
python scripts/03_smoke_drivebench.py --precision bf16 --condition clean
python scripts/03_smoke_drivebench.py --precision bf16 --condition Fog
python scripts/06_stability.py --precision bf16 --condition clean
python scripts/07_profile.py --precision bf16 --session 1     # first real use of the streaming path
# swap the server to int4 and repeat 03 (clean, Fog), 06, 07
python scripts/05_compare_flips.py results/pilot/bf16_clean.jsonl results/pilot/int4_clean.jsonl results/pilot/bf16_Fog.jsonl results/pilot/int4_Fog.jsonl --by cell n_views margin condition --min-n 3 --min-scenes 1
# with the server stopped (needs the whole GPU):
free -g && uv pip install -e ".[quant]" && python scripts/quantize_awq.py      # own checkpoint, ~30-60 min
INT4_MODEL=checkpoints/qwen2.5-vl-7b-awq-w4a16-g128 bash scripts/01_serve.sh int4 && python scripts/02_smoke_request.py --precision int4
uv pip freeze > manifests/requirements_frozen.txt           # version pins for the freeze
```

Before stopping the instance: `rsync -avz vast:/workspace/qflip/results/ results/` and `rsync -avz vast:/workspace/qflip/manifests/ manifests/`, then commit `manifests/`.

The week-5 gate note (`manifests/gate_week5.md`) recorded: vLLM and transformers versions, `MAX_PIXELS`, tokens per view, peak MiB per precision, logprob availability, MCQ accuracy and flips on the 20-question slice, p50/p95 latency, and the 7B decision.

## Known differences from the released DriveBench adapter

- The released adapter resizes every view to 224×224. We send originals and cap tokens with `max_pixels` (`images.resize_to: null`). Set `resize_to: [224, 224]` to reproduce upstream exactly.
- Released decoding is temperature 0.2 / top_p 0.2 / 512 tokens; we use greedy / 128 tokens so flips are not sampling noise.
- We append "Begin your answer with the option letter." to MCQ questions so the margin can be read from the first token. Turn off with `mcq_suffix: ""`.
- The released MCQ normalizer matches the article "a" as option A; we record both that rule (`mcq_pred`) and a strict one (`mcq_pred_strict`).
- DriveBench "Dark" is the `LowLight` corruption directory.
