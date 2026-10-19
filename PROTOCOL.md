# Experiment protocol (draft for the week-7 freeze)

**Status: DRAFT, October 19, 2026.** Each item is marked **FROZEN** (decided and tested), **PILOT DEFAULT** (in use, expected to stand, confirm at freeze), or **TO DECIDE** (needs a development-scene pilot or a team decision). When the freeze happens, every TO DECIDE item is resolved, the date is written here, and a copy of `config/pilot.yaml` is saved as `manifests/config_frozen.yaml`. After that, changes to anything in this file require a dated entry in the change log at the bottom. Analysis of final-scene results does not begin before the freeze.

The proposal ([C_PROJECT_PROPOSAL.md](C_PROJECT_PROPOSAL.md)) states what we intend; this file states exactly how it is executed, so that a reader can rerun the study from the repository.

## 1. Model and checkpoints

| Item | Value | Status |
|---|---|---|
| Model | Qwen2.5-VL-7B-Instruct (not the 3B fallback; week-5 gate passed) | FROZEN |
| BF16 reference | `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` | FROZEN |
| INT4 for the pilot | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (official; decoder INT4, vision encoder BF16) | FROZEN (pilot only) |
| INT4 for the final comparison | Own AWQ W4A16 checkpoint from the base revision above, built with llm-compressor, calibration prompts from the calibration scenes only; `ignore` list recorded (vision encoder, embeddings, lm_head) | TO DECIDE: tool version, group size (128 expected), calibration prompt count |
| Fallback | If the own checkpoint does not run in vLLM by end of week 7, the official AWQ checkpoint becomes the final INT4 and the mitigation is reported as not executed | FROZEN rule |

## 2. Serving

All runs use `scripts/01_serve.sh`, one precision at a time, one server per GPU.

| Flag | Value | Status |
|---|---|---|
| vLLM / torch / transformers | 0.31.0 / 2.13.0+cu130 / 5.17.0 (pilot versions; pin in `pyproject.toml` at freeze) | PILOT DEFAULT |
| `--max-model-len` | 4096 | PILOT DEFAULT |
| `--limit-mm-per-prompt` | `{"image": 6}` | FROZEN |
| `--mm-processor-kwargs max_pixels` | 200704 (measured ~165 image tokens per 1600×900 view, ~1293 prompt tokens for six views) | TO DECIDE: keep, or raise to 401408 (~512 tokens per view) after a development-scene pilot on both precisions |
| `--kv-cache-memory-bytes` | 2 GiB, identical for both precisions | PILOT DEFAULT (verify accepted by vLLM 0.31 at the next session) |
| `--max-num-seqs` | 1 (batch size 1, concurrency 1) | FROZEN |
| `--seed` | 0 | FROZEN |
| `--dtype` | bfloat16 for BF16; float16 with `--quantization awq_marlin` for INT4 | FROZEN |
| Prefix caching | vLLM default (on) for quality runs; stability runs are reported with it on, and optionally once with `--no-enable-prefix-caching` | TO DECIDE |
| Host | One pinned Vast.ai host (machine id m:3752, RTX 4090 24 GB, power limit 300 W, driver 580.178.04). All profiling sessions on this host. If the host becomes unavailable, re-run the profile on the replacement host and report both | FROZEN rule |

## 3. Prompt and images

| Item | Value | Status |
|---|---|---|
| System prompt | `prompt.txt`, the DriveBench system prompt verbatim, with the camera sentence rewritten per question exactly as upstream `inference/utils.py` does | FROZEN |
| View selection | Cameras referenced by `<c1,CAM_X,...>` tags in the question; all six if none referenced; ordered FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT (upstream `tools/preprocess.py`) | FROZEN |
| Message layout | Images first, then system prompt plus question as one user turn | FROZEN |
| MCQ suffix | `" Begin your answer with the option letter."` appended to MCQ questions only, so the first generated token is the option letter | PILOT DEFAULT |
| Image handling | Originals (1600×900 JPEG) sent as data URLs; token count set by `max_pixels`. The released adapter's 224×224 resize is available as `images.resize_to: [224, 224]` and is not the default | TO DECIDE (team item 5) |
| NoImage control | One 224×224 black PNG per selected view, same view count as the clean question (released adapter behavior) | FROZEN |
| Corruptions | Released DriveBench images: `LowLight` (paper name "Dark") and `Fog`, by path replacement `nuscenes/samples` to `corruption/<Type>` | FROZEN |

## 4. Decoding

| Item | Value | Status |
|---|---|---|
| Temperature | 0.0 (greedy) | FROZEN |
| `max_tokens` | 128 | PILOT DEFAULT |
| `seed` | 0 | FROZEN |
| `logprobs` | on, `top_logprobs` 5, read at the first generated token | FROZEN |

## 5. Question set and splits

| Item | Value | Status |
|---|---|---|
| Annotations | `data/drivebench-test.json` from HF `drive-bench/arena`: 1461 questions, 174 scenes, 1200 clean images | FROZEN |
| Eligible cells | perception/MCQ, perception/VQA, prediction/VQA, planning/VQA, behavior/MCQ (released scorer routing). Prediction MCQ is dropped by the released scorer and is not run | FROZEN |
| Scene split | `manifests/scene_split.json`, seed 0: calibration 10% (17 scenes), development 15% (26 scenes), final 75% (131 scenes, 1116 questions, of which prediction/VQA 45) | FROZEN |
| Use of splits | Calibration scenes: quantization calibration only. Development scenes: every prompt, image, decoding, bin, and mitigation choice; stability slice; profile slice. Final scenes: the 8-run matrix, touched once | FROZEN rule |
| Coverage minimum | A cell with fewer than N final-scene questions or fewer than M scenes is reported as unassessed | TO DECIDE: N and M (prediction/VQA at 45 questions is the cell at risk) |

## 6. Conditions and matrix

Clean, NoImage, LowLight, Fog × BF16, INT4 = 8 quality runs on final scenes with `scripts/03_smoke_drivebench.py --n 0 --split final`. Mitigation, if executed, adds one precision configuration × the same conditions. **FROZEN** (conditions pending team item 1; default stands if no objection by the freeze).

## 7. Scoring

| Item | Rule | Status |
|---|---|---|
| MCQ, primary | **Strict rule:** the normalized answer is the option letter at the start of the response (`normalize_mcq_strict`); correct if it equals the ground-truth letter. Stored as `mcq_correct` | FROZEN |
| MCQ, secondary | **Released rule:** upstream `evaluate/utils.py` regex, which also matches the article "a" as option A. Stored as `mcq_correct_released` and reported alongside for comparability with published DriveBench numbers, with the defect stated | FROZEN |
| VQA | Rubric score via a fixed judge model, version, and prompt, from the released evaluator with `--eval-gpt`; correctness threshold frozen on development scenes. Only if judge budget is approved; otherwise VQA cells are unassessed and a 100-pair human-scored sample is reported | TO DECIDE (team item 2) |
| Errors | Requests that error are retried on rerun; a question still errored after the run is counted in the failed total and excluded from denominators with the count published | FROZEN |

## 8. Flip classification and factors

Per question, per condition, paired by `question_id` (`src/qflip/flips.py`):

| Class | Definition |
|---|---|
| harmful | BF16 correct, INT4 incorrect |
| beneficial | BF16 incorrect, INT4 correct |
| neutral_change | same correctness, different letter |
| unchanged | same letter |

`harmful_rate` F_h = harmful / BF16-correct. `acc_delta` = INT4 accuracy minus BF16 accuracy. `change_rate` = 1 minus unchanged fraction.

Factors: cell (task × format), `n_views` (1 to 6), BF16 option margin (probability of the top option letter minus the second, from first-token `top_logprobs`), condition. Margin bins: edges 0.25 and 0.5 are the pilot default; **TO DECIDE** on development scenes before the freeze.

**Prespecified confirmatory comparisons (one per factor, stated here before the final run):**

1. Margin: lowest bin versus the rest, F_h.
2. Task: planning + behavior (decision cells) versus perception + prediction, F_h. MCQ only unless VQA is scored.
3. Views: 1 view versus 6 views, F_h.
4. Condition: each stress condition versus clean, F_h.

Everything else is exploratory and labeled as such.

## 9. Uncertainty

Paired bootstrap over scenes: resample scenes with replacement, keep every question and both precisions of a resampled scene together, 2000 draws, seed 0, 95% percentile intervals for F_h, `acc_delta`, `change_rate`, and each subgroup contrast. A contrast supports the hypothesis only when its interval excludes zero. Implemented in `bootstrap_scenes`; invoked by `scripts/05_compare_flips.py`. **FROZEN.**

## 10. Stability floor

`scripts/06_stability.py`: 60 stratified development-scene questions (seed 1), three repeats per precision in the clean condition, same server settings as the quality runs. Reports `self_change_rate` (letter disagreement between repeats) and `self_correctness_change_rate`. A BF16-versus-INT4 `change_rate` below the larger of the two precisions' `self_change_rate` is reported as within the nondeterminism floor. **PILOT DEFAULT** (slice size and conditions to confirm).

## 11. Deployment profile

`scripts/07_profile.py`: 50 six-view development questions in fixed order, 3 warm-up requests excluded, streamed requests. Per session: TTFT p50/p95/p99, total latency p50/p95/p99, decode tokens per second, peak GPU memory sampled at 0.2 s, with the server manifest embedded. Three separate sessions per precision on the pinned host, all with `--kv-cache-memory-bytes` 2 GiB. Budgets from the proposal: p95 TTFT ≤ 2 s, median decode ≥ 15 tok/s, INT4 peak memory ≥ 25% below BF16. The pilot's memory numbers (both ~22 GB under `--gpu-memory-utilization 0.9`) are not a result. **PILOT DEFAULT.**

## 12. Mitigation (week 12, if time permits)

Per-layer sensitivity on development scenes: re-quantize with one decoder block group held at BF16 at a time (or output-divergence measurement), rank, keep the smallest set that reduces F_h on development scenes, freeze the list, re-quantize from the base checkpoint with the same recipe and calibration set, then evaluate on final scenes like any other precision configuration. **TO DECIDE**: block grouping and the number of retained layers.

## 13. Record schema

One JSON object per line in `results/<run>/<precision>_<condition>.jsonl`, written by `qflip.runner.run_question`:

```
question_id, scene_token, frame_token, question_type, tag, cell, views, n_views,
condition, precision, model, gt, pred, error, prompt_tokens, completion_tokens,
latency_s, first_token_top_logprobs,
# MCQ only:
mcq_pred (released rule), mcq_pred_strict, mcq_gt, mcq_correct (strict),
mcq_correct_released, margin_letter, margin, option_probs
```

`question_id` = first 8 characters of scene token, first 8 of frame token, 10-character SHA-1 of the question text. Reruns append; the last record per `question_id` wins; errored records are retried. A `.drivebench.json` companion file carries the upstream fields plus `pred` for the released evaluator.

## 14. Provenance per run

`manifests/serve_<precision>.txt` (model, flags, versions, power limit, GPU name) and `manifests/machine.txt` (host facts) are written by the serve and check scripts. Pilot results under `results/pilot/` are tracked in git; full-run results are not (synced from the host after each session) and the raw files for the final report are archived with the submission.

## Change log

- 2026-10-19: first draft, after the week-5 pilot. Nothing frozen beyond what the pilot tested.
