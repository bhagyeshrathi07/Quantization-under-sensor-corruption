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
