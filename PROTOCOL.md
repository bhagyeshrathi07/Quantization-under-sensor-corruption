# Experiment protocol (draft for the week-7 freeze)

**Status: DRAFT, November 3, 2026.** Each item is marked **FROZEN** (decided and tested), **PILOT DEFAULT** (in use, expected to stand, confirm at freeze), or **TO DECIDE** (needs a development-scene pilot or a team decision). When the freeze happens, every TO DECIDE item is resolved, the date is written here, and a copy of `config/pilot.yaml` is saved as `manifests/config_frozen.yaml`. After that, changes to anything in this file require a dated entry in the change log at the bottom. Analysis of final-scene results does not begin before the freeze.

The proposal ([C_PROJECT_PROPOSAL.md](C_PROJECT_PROPOSAL.md)) states what we intend; this file states exactly how it is executed, so that a reader can rerun the study from the repository.

## 1. Model and checkpoints

| Item | Value | Status |
|---|---|---|
| Model | Qwen2.5-VL-7B-Instruct (not the 3B fallback; week-5 gate passed) | FROZEN |
| BF16 reference | `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` | FROZEN |
| INT4 for the pilot | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (official; decoder INT4, vision encoder BF16) | FROZEN (pilot only) |
| INT4 for the final comparison | Own checkpoint from `scripts/quantize_awq.py`: llm-compressor `AWQModifier` + `QuantizationModifier`, scheme `W4A16_ASYM` (4-bit weights with zero point, group size 128, the official checkpoint's convention), targets `Linear`, ignore `lm_head`, embeddings, vision tower (`config/pilot.yaml` `quantize:`). Calibration: every eligible calibration-split question (107 from 17 scenes), rendered exactly as the runs render them (same system prompt, views, `max_pixels` 200704, clean condition), `max_seq_len` 4096. Saved as compressed-tensors; served with `INT4_MODEL=<dir>`; `manifests/quantize_<name>.json` records recipe, tool versions, and calibration question ids | PILOT DEFAULT (script written and dry-run tested October 20; first real build next GPU session). TO DECIDE: llm-compressor version pin |
| Fallback | If the own checkpoint does not run in vLLM by end of week 7, the official AWQ checkpoint becomes the final INT4 and the mitigation is reported as not executed | FROZEN rule |

## 2. Serving

All runs use `scripts/01_serve.sh`, one precision at a time, one server per GPU.

| Flag | Value | Status |
|---|---|---|
| vLLM / torch / transformers | 0.33.0 / 2.14.0+cu130 / 5.20.2, installed November 3 on the replacement host and captured in `manifests/requirements_frozen.txt` (the October pilot ran 0.31.0 / 2.13.0 / 5.17.0; nothing from that stack is reused). Pin these in `pyproject.toml` at the freeze | PILOT DEFAULT |
| `--max-model-len` | 4096 | PILOT DEFAULT |
| `--limit-mm-per-prompt` | `{"image": 6}` | FROZEN |
| `--mm-processor-kwargs max_pixels` | 200704 (measured ~165 image tokens per 1600×900 view, ~1293 prompt tokens for six views) | TO DECIDE: keep, or raise to 401408 (~512 tokens per view) after a development-scene pilot on both precisions |
| `--kv-cache-memory-bytes` | 2 GiB, identical for both precisions; verified November 3 on vLLM 0.33 (2.0 GiB reserved, 37,440 KV tokens, both precisions) | FROZEN |
| `--max-num-seqs` | 1 (batch size 1, concurrency 1) | FROZEN |
| `--seed` | 0 | FROZEN |
| `--dtype` | bfloat16 for BF16; float16 with `--quantization awq_marlin` for INT4 | FROZEN |
| Prefix caching | vLLM default (on) for quality runs; stability runs are reported with it on, and optionally once with `--no-enable-prefix-caching` | TO DECIDE |
| Host | One pinned Vast.ai host. Since November 3, 2026: offer m:11851 (Spain, RTX 4090 24 GB, stock 450 W power limit, driver 615.71.09); the October host m:3752 (300 W cap) destroyed the stopped instance, and its latency numbers are pilot-only. All profiling sessions on the current host. If it becomes unavailable, re-run all profile sessions on the replacement and report both | FROZEN rule |

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
| Exclusions | `manifests/scene_exclusions.json`: 15 final-split scenes that the October 15 and November 3 pilot slices inspected (03 then defaulted to `--split all`). The confirmatory population is final minus exclusions: 116 scenes, 838 eligible questions (behavior/MCQ 133, perception/MCQ 133, perception/VQA 133, planning/VQA 399, prediction/VQA 40 in 37 scenes). The split is never regenerated; exclusions are additive with their own provenance. `--split final_exposed` returns the 15 for exploratory use only | FROZEN (added 2026-11-04) |
| Use of splits | Calibration scenes: quantization calibration only. Development scenes: every prompt, image, decoding, bin, and mitigation choice; stability slice; profile slice; the default for every run script. Final scenes: the 8-run matrix, touched once, and only through the freeze guard (section 14) | FROZEN rule |
| Coverage minimum | A group with fewer than N questions or fewer than M scenes is reported as unassessed (`05_compare_flips.py` prints UNASSESSED and writes `assessed: false`; defaults `analysis.min_questions` 30 and `analysis.min_scenes` 10 in config, overridable with `--min-n` / `--min-scenes`) | TO DECIDE: N and M (prediction/VQA at 45 final questions is the cell at risk) |

## 6. Conditions and matrix

Clean, NoImage, LowLight, Fog × BF16, INT4 = 8 quality runs on final scenes, driven by `scripts/run_matrix.sh` (per precision: start the server, wait for `/health`, run each condition with `03_smoke_drivebench.py --n 0 --split final --out-dir results/final`, stop the server; resumable; the serve manifest is copied next to the results). Mitigation, if executed, adds one precision configuration × the same conditions. **FROZEN** (conditions pending team item 1; default stands if no objection by the freeze).

## 7. Scoring

| Item | Rule | Status |
|---|---|---|
| MCQ, primary | **Strict rule:** the normalized answer is the option letter at the start of the response (`normalize_mcq_strict`); correct if it equals the ground-truth letter. Stored as `mcq_correct` | FROZEN |
| MCQ, secondary | **Released rule:** upstream `evaluate/utils.py` regex, which also matches the article "a" as option A. Stored as `mcq_correct_released` and reported alongside for comparability with published DriveBench numbers, with the defect stated | FROZEN |
| VQA | Rubric score 0–100 from `scripts/08_judge_vqa.py`: the released judge templates verbatim (`src/qflip/judge_prompts.py`, per-cell routing as upstream `evaluate/eval.py`), the released system line, temperature 0, `max_tokens` 512, object visual description from the toolkit's `data/visual_description.json` (planning template). The released `--eval-gpt` path is not runnable as shipped (its `request.py` passes `max_tokens` to a method that does not accept it and retries forever), so the request is made here with the openai client; model name and template hash are written into every record. Judge model: `judge.model` in config (released code hard-codes gpt-3.5-turbo). Correct = score ≥ `judge.correct_threshold`, frozen on development scenes. VQA flip classes: harmful, beneficial, unchanged (no neutral_change; wording always differs). Only if judge budget is approved; otherwise VQA cells are unassessed and a 100-pair human-scored sample is reported | TO DECIDE (team item 2): judge model, threshold, budget |
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

**Prespecified confirmatory contrasts (family of five on the MCQ harmful rate F_h, Δ = F_h(A) − F_h(B); `qflip.contrasts`, `05 --contrasts`; stated here before results from the remaining confirmatory population exist):**

| # | Contrast | A | B | Population |
|---|---|---|---|---|
| C1 | margin | BF16 margin in the lowest bin | all other bins | clean, pairs with a margin |
| C2 | task | behavior/MCQ | perception/MCQ | clean |
| C4a | NoImage vs clean | NoImage pairs | clean pairs | common support: ids paired without error under both |
| C4b | LowLight vs clean | LowLight pairs | clean pairs | common support |
| C4c | Fog vs clean | Fog pairs | clean pairs | common support |

The views comparison (1 vs 6 views) is not in the family: in the final split every behavior MCQ has six views and every perception MCQ has one, so Δ_views = −Δ_task in every draw; it is reported as descriptive with that sentence. C1 and C2 are clean-only (amendment of 2026-11-04; the proposal's clean-primary language did not specify contrast scope). Support rule: a contrast is assessed only with ordinary coverage on both sides (section 5), at least 20 reference-correct pairs from at least 10 scenes on each side, and at least half of the contrast draws valid (a draw is invalid when either side has no reference-correct pair); otherwise `assessed: false` with the failed condition named and the point estimate still reported. The support rule is never lowered to rescue a contrast. Change rate, accuracy delta, VQA contrasts, pooled-condition versions of C1/C2, and all breakdowns are exploratory and labeled as such. **FROZEN** (family, statistic, support rule); margin edges per the row above.

## 9. Uncertainty

Paired bootstrap over scenes: resample scenes with replacement, keep every question and both precisions of a resampled scene together, seed 0. Group intervals (F_h, `acc_delta`, `change_rate` per group): 2,000 draws, 95% percentile (`bootstrap_scenes`). Prespecified contrasts (section 8): 20,000 draws per contrast, scenes drawn from the union of both sides with every pair of a drawn scene entering on its own side; raw percentile 95% and Bonferroni-adjusted 99% (α = 0.05 / 5) intervals both stored. A contrast supports the hypothesis only when the adjusted interval excludes zero; the family stays five whatever happens to any member (`bootstrap_contrast`). Draw counts, the bootstrap seed, and support minimums live in config `analysis:` and are part of the core freeze; `05` refuses a CLI override of a frozen value on final-split files. Below half valid draws a contrast's intervals and exclusion flags are reported as unavailable, not as numbers. `05` refuses an input file given twice and any (condition, question) pair that appears in two file pairs, so repeated inputs cannot manufacture support. **FROZEN.**

## 10. Stability floor

`scripts/06_stability.py`: 60 stratified development-scene questions (seed 1), three repeats per precision in the clean condition, same server settings as the quality runs. Reports `self_change_rate` (letter disagreement between repeats) and `self_correctness_change_rate`. A BF16-versus-INT4 `change_rate` below the larger of the two precisions' `self_change_rate` is reported as within the nondeterminism floor. Measured November 3 on the pinned host: zero letter, correctness, or text changes across three repeats for both precisions (25 MCQ); margin range max 0.039 BF16, 0.004 INT4 (`results/stability/`). **PILOT DEFAULT** (slice size and conditions to confirm; a repeat with `--no-enable-prefix-caching` is optional).

## 11. Deployment profile

`scripts/07_profile.py`: 50 six-view development questions in fixed order, 3 warm-up requests excluded, streamed requests. Per session: TTFT p50/p95/p99, total latency p50/p95/p99, decode tokens per second, peak GPU memory sampled at 0.2 s, with the server manifest embedded. Three separate sessions per precision on the pinned host, all with `--kv-cache-memory-bytes` 2 GiB. Budgets from the proposal: p95 TTFT ≤ 2 s, median decode ≥ 15 tok/s, INT4 peak memory ≥ 25% below BF16. The October pilot's memory numbers (both ~22 GB under `--gpu-memory-utilization 0.9`) are not a result. Session 1 of 3 ran November 3 on the pinned host (`results/profile/*_s1.json`, `manifests/session1_2026-11-03.md`). **PILOT DEFAULT.**

## 12. Mitigation (week 12, if time permits)

Per-layer sensitivity on development scenes: re-quantize with one decoder block group held at BF16 at a time (or output-divergence measurement), rank, keep the smallest set that reduces F_h on development scenes, freeze the list, re-quantize from the base checkpoint with the same recipe and calibration set, then evaluate on final scenes like any other precision configuration. **TO DECIDE**: block grouping and the number of retained layers.

## 13. Record schema

One JSON object per line in `results/<run>/<precision>_<condition>.jsonl`, written by `qflip.runner.run_question`:

```
question_id, scene_token, frame_token, question_type, tag, cell, question, views, n_views,
condition, precision, model, gt, pred, error, prompt_tokens, completion_tokens,
latency_s, first_token_top_logprobs,
# MCQ only:
mcq_pred (released rule), mcq_pred_strict, mcq_gt, mcq_correct (strict),
mcq_correct_released, margin_letter, margin, option_probs
```

`question_id` = first 8 characters of scene token, first 8 of frame token, 10-character SHA-1 of the question text. Reruns append; the last record per `question_id` wins; errored records are retried. A `.drivebench.json` companion file carries the upstream fields plus `pred` for the released evaluator. (`question` was added October 20; pilot records lack it and the judge falls back to the annotation file.)

Judge sidecar, one per result file (`<precision>_<condition>.judge.jsonl`): `question_id, scene_token, cell, condition, precision, judge_model, judge_template, judge_template_sha, desc, reply, score, prompt_tokens, completion_tokens, latency_s, error`. Same append-and-last-wins rule.

## 14. Provenance per run

`manifests/serve_<precision>.txt` (model path, revision, flags including quantization method, versions, power limit, GPU name) and `manifests/machine.txt` (host facts) are written by the serve and check scripts; `manifests/quantize_<name>.json` by the quantization script; `logs/matrix_<timestamp>.txt` by the matrix driver.

**Result-file sidecars (added 2026-11-04).** Every result file has a `.meta.json` written before its first record (`qflip.provenance`): split and the sha256 of the split file, exclusion list, annotations, and prompt; a canonical hash of the core config sections (`core_config_sha256`, computed over the sections listed in section 15, so a judge-only edit of the YAML does not change it); precision, condition, served name, model path or id, revision, checkpoint fingerprint (sha256 over config.json and every safetensors file for a local checkpoint), quantization method, dtype, max_pixels, max_model_len, KV-cache args, seed, max_num_seqs, prefix caching, vLLM version. These are the compatibility fields: a rerun refuses to append to a file whose sidecar differs in any of them, and also refuses when any raw row already in the file (retries included, checked before the last-record-wins deduplication) carries another precision or condition or a scene outside the declared population; `05` applies the same row check to every analyzed file. The full config-file hash, code revision, dirty flag, host, server pid and port, and the serve manifest copy are recorded as history only, so a server restart does not block a resume. Files without a sidecar are never analyzed; `05 --no-provenance` exists for the pre-sidecar pilot files only, refuses any file whose sidecar declares the final split, and labels its output exploratory.

**Live-server check.** `03` and `06` query `/v1/models` at the endpoint the inference client uses (`QFLIP_BASE_URL`, else the config) and refuse a server that does not list the expected served name or whose reported model path differs from the manifest. The served alias is shared by the official and the own INT4 checkpoints, so the manifest is also bound to the running process: `01_serve.sh` records its pid and port; the guard requires that pid to be alive on this host with a command line naming the manifest's model and exactly that port, the client's endpoint port (parsed, not matched as text) to equal it, and the process holding the LISTEN socket serving that endpoint's address and port (found through `/proc/net/tcp*` and `/proc/<pid>/fd`; `0.0.0.0` counts for IPv4 addresses and `::` for both families) to be that pid or one of its descendants. Ownership is tracked per matching socket: a matching socket whose owner is not visible refuses the run. The endpoint must be a loopback address (`localhost`, `127.0.0.1`, `::1`); other hostnames are refused rather than resolved. Ownership that cannot be determined for any matching socket, or is shared by several processes, is refused, not skipped. The matrix driver additionally requires the port to be free before it starts a server, the manifest pid to be the server it started, and the port's listener (from `ss -ltnp`, required) to be that process or its child.

**Final-run guard.** A `--split final` run, from the driver or from `03` directly, requires `manifests/config_frozen.yaml` (section 15), a live config, inputs, and protocol snapshot identical to it, and a serve manifest whose model, revision (BF16), checkpoint fingerprint and quantization method (INT4), dtype (bfloat16 for BF16, float16 for INT4, recorded in the freeze per model), max_pixels, max_model_len, KV bytes, seed, max_num_seqs, and prefix caching match it. `05` on final-split files requires the freeze, checks every file's sidecar (input hashes, core config hash, all serving settings, model identity) against the frozen values rather than only the live config, verifies every record's scene is in the population, and refuses CLI overrides of frozen analysis settings. Pairing requires the records in the order reference BF16 then quantized INT4; a reversed pair is refused rather than silently swapping harmful and beneficial.

## 15. Freeze mechanics

`scripts/freeze_config.py` writes `manifests/config_frozen.yaml` (core config sections `server, models, data, prompt, images, generation, serving, quantize, analysis` and their canonical hash; fingerprints of split, exclusions, annotations, prompt; the snapshot path and sha256 of `manifests/PROTOCOL_frozen.md`, required; the BF16 id and revision; the INT4 path and checkpoint fingerprint; git commit) and copies this file to the snapshot path. `--check` fails if the snapshot is edited after the freeze. It refuses while any table row outside sections 7 and 12 has status TO DECIDE, or while the working tree is dirty. `--check` compares the live state to the frozen copy; the run and analysis guards call it. The judge protocol (section 7, VQA row) is frozen separately in `manifests/judge_frozen.yaml` after development validation; changes there never touch the core freeze. Pilot results under `results/pilot/` are tracked in git; full-run results are not (synced from the host after each session) and the raw files for the final report are archived with the submission.

## Change log

- 2026-10-19: first draft, after the week-5 pilot. Nothing frozen beyond what the pilot tested.
- 2026-10-20: own-checkpoint recipe made concrete (section 1), matrix driver (6), VQA judge path and VQA flip classes (7), coverage rule mechanics (5), `question` field and judge sidecar schema (13). Nothing frozen.
- 2026-11-03: replacement host and new software versions (section 2); KV-cache flag verified and frozen; stability floor measured (10); profile session 1 done (11); own AWQ checkpoint built (1). See `manifests/session1_2026-11-03.md`.
- 2026-11-04: review found that the November 3 slices had drawn from the final split; 15 scenes excluded (section 5), run default moved to development, sidecars, live-server check, final-run guard, and freeze mechanics added (sections 14, 15); a step-1 audit closed eight guard gaps the same day (core-config hash, per-file freeze checks in `05`, pairing order, process binding of the serve manifest, seed / max_num_seqs / prefix caching in the manifest, resume record inspection, required snapshot hash); a follow-up audit added listener ownership, frozen dtype per precision, and raw-row validation shared by resume and analysis. Section 8 contrasts rewritten as the family of five with the support rule; section 9 gains the contrast intervals (plan step 2). The November 3 profile and stability numbers are official-checkpoint reference measurements, not sessions of the frozen study. Plan and acceptance criteria in `WEEK9_ANALYSIS_PLAN.md`.
