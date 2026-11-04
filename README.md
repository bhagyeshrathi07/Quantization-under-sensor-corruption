# Quantization-Induced Decision Failures in a Driving Vision-Language Model: A Paired BF16-versus-INT4 Deployment Study

**Institution:** San Jose State University (SJSU)  
**Team:** Ronak, Bhagyesh, and Akshay  
**Track:** System / Deployment (supported by experimental research)  
**Duration:** 14 weeks  
**Prepared:** September 8, 2026. **Revised:** September 29, 2026, after instructor feedback. **Status updated:** November 3, 2026.

## Abstract

Quantization reduces the precision of the numbers used inside a neural network, potentially reducing memory use and inference time. Average accuracy on a benchmark may stay almost unchanged after quantization while individual answers still change, and some of those changes turn a correct driving decision into an incorrect one. This project deploys one driving-scene vision-language model, Qwen2.5-VL-7B-Instruct, in vLLM at BF16 and at INT4, runs the same DriveBench questions through both, and identifies every case in which the full-precision model is correct but the quantized model changes the answer or the driving decision. We analyze these failures by driving task, answer format, the full-precision model's confidence margin, the number of camera views, and, as a stress test, a small set of sensor-corruption conditions. We measure the latency and memory savings that INT4 provides under matched serving settings, and, if time permits, test one simple mitigation: retaining a small set of sensitive decoder layers at BF16. The contribution is a reproducible, hardware-measured, paired failure analysis of a quantized driving VLM, extending prior work that reports only aggregate scores. The study runs on one rented RTX 4090 (24 GB) pinned for all measurements; SJSU's course HPC is the fallback, and no Jetson is required.

## Repository contents

| Course item | File |
|---|---|
| B. Literature & SOTA Survey | [B_LITERATURE_SURVEY.md](B_LITERATURE_SURVEY.md) |
| C. Project Proposal | [C_PROJECT_PROPOSAL.md](C_PROJECT_PROPOSAL.md) |
| D. AI Novelty & Feasibility Audit | [D_AI_NOVELTY_FEASIBILITY_AUDIT.md](D_AI_NOVELTY_FEASIBILITY_AUDIT.md) |

Working documents:

| File | Purpose |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | Exact execution protocol: checkpoints, serving flags, prompt, scoring rules, flip classes, statistics, record schema. Draft until the week-7 freeze; each item marked frozen, pilot default, or to decide. |
| [PILOT.md](PILOT.md) | Runbook: setup, data, serve, smoke test, runs, stability, profile, Vast.ai and HPC notes. |
| [TEAM_REVIEW.md](TEAM_REVIEW.md) | Summary for teammates of the scope change, every decision and its reason, and the items that still need sign-off. |
| [manifests/gate_week5.md](manifests/gate_week5.md) | Week-5 baseline gate: pilot measurements and the findings that changed the protocol. |

Code (package `qflip`, Python 3.12, vLLM):

```
config/pilot.yaml                  every tunable in one place; frozen copy goes to manifests/ at the freeze
prompt.txt                         DriveBench system prompt, verbatim
src/qflip/drivebench.py            annotations, view selection, prompt building, corruption paths, MCQ normalization
src/qflip/client.py                vLLM OpenAI-compatible client: logprobs, option margin, streaming for TTFT
src/qflip/runner.py                one question in, one result record out (shared by all run scripts)
src/qflip/flips.py                 pairing (MCQ and judged VQA), flip classes, paired bootstrap over scenes, coverage rule
src/qflip/judge.py                 DriveBench rubric judge: description lookup, prompt fill, score parsing, sidecars
src/qflip/judge_prompts.py         released judge templates, verbatim
src/qflip/provenance.py            result-file sidecars (input hashes, checkpoint fingerprint), resume compatibility, live-server check
src/qflip/freeze.py, guard.py      core freeze (frozen config + protocol snapshot) and the run/analysis guards that enforce it
src/qflip/split.py, stats.py, gpu.py, records.py
scripts/download_drivebench.py     annotations + visual descriptions (HF, toolkit) + corruption images (Google Drive zip)
scripts/download_clean_images.py   clean nuScenes keyframes streamed from the public AWS mirror
scripts/00_check_machine.sh        host facts -> manifests/machine.txt
scripts/01_serve.sh bf16|int4      vllm serve with the protocol's flags, equal KV-cache bytes
scripts/02_smoke_request.py        one six-view request: tokens per view, logprobs, margin, memory
scripts/03_smoke_drivebench.py     N or all questions per (precision, condition), resumable, --split
scripts/04_scene_split.py          calibration / development / final scene manifest
scripts/exclude_exposed_scenes.py  record final scenes a pilot inspected -> manifests/scene_exclusions.json
scripts/freeze_config.py           write / check the core freeze
scripts/05_compare_flips.py        flips with bootstrap CIs, breakdowns by cell, views, margin, condition; --vqa; unassessed groups
scripts/06_stability.py            three-repeat slice: nondeterminism floor
scripts/07_profile.py              TTFT, latency percentiles, decode tok/s, peak memory per session
scripts/08_judge_vqa.py            VQA rubric judge (released prompts, fixed model, key from the environment), resumable sidecars
scripts/quantize_awq.py            own AWQ W4A16 checkpoint with llm-compressor, calibration-split prompts, manifest
scripts/run_matrix.sh              serve -> all conditions -> stop, per precision; resumable; DRY_RUN=1 prints the plan
scripts/rescore.py                 recompute MCQ fields after a rule change
scripts/vast_setup.sh, hpc_pilot.sbatch
tests/                             37 offline unit tests (pytest)
```

## Scope

**Required:** Qwen2.5-VL-7B-Instruct at BF16 versus INT4 (AWQ) in vLLM, on DriveBench, in the clean condition plus a three-condition stress test (Dark, Fog, NoImage). Per-question flip classification, breakdowns by task, format, confidence margin, view count, and condition, and matched latency and memory measurements. The 3B fallback is no longer needed: the 7B model passed the week-5 gate.

**If time permits:** one mitigation that retains selected decoder layers at BF16.

**Removed on September 29, 2026:** the camera-based occupancy perception strand (FlashOcc-M0 in TensorRT), the full corruption matrix, and GPU energy as a required measurement. FP8, additional models, the remaining DriveBench conditions, energy, and Jetson measurements are extensions.

## Status (November 3, 2026, semester week 9)

**Done.**
- Week-5 baseline gate passed on October 15 ([manifests/gate_week5.md](manifests/gate_week5.md)): BF16 7B serves six views on the 24 GB card; official AWQ INT4 serves; option-letter log-probabilities are available, so the margin analysis is feasible; 20-question LowLight run on both precisions with zero errors; BF16 p50 2.39 s versus INT4 0.91 s per question at batch size 1.
- Two protocol findings from the pilot: the released DriveBench MCQ normalizer scores the article "a" in an explanation as option A, so a strict letter-first rule is now primary and the released rule is recorded alongside; and the DriveBench image archive contains only corrupted images, so clean nuScenes keyframes are fetched from Motional's public AWS mirror (no account; CC BY-NC-SA 4.0).
- Analysis tooling built and unit-tested offline: paired bootstrap, stability slice, deployment profile with streaming TTFT, equal KV-cache serving, scene-split filtering.
- Build for the final runs written offline on October 20, dry-run tested, not yet run on a GPU: own-AWQ quantization script (recipe and calibration manifest), matrix driver, VQA judge with the released rubric prompts (the released `--eval-gpt` path cannot run as shipped; see PROTOCOL section 7), coverage rule in the comparison. The toolkit's `data/visual_description.json`, needed by the planning judge prompt, was located and is fetched automatically.

- GPU session 1 on the replacement host, November 3 ([manifests/session1_2026-11-03.md](manifests/session1_2026-11-03.md)): the October instance had been destroyed by its host and was rebuilt; clean nuScenes images fetched; equal KV cache and streaming verified on vLLM 0.33; clean, Fog, NoImage 20-question slices on both precisions (zero errors; 1 harmful and 1 beneficial flip in 32 MCQ pairs, both NoImage with BF16 margin below 0.25); stability floor measured at zero self-changes; deployment profile session 1 (INT4 2.4× faster at batch 1, 47% less peak memory, equal 2 GiB KV cache); own AWQ W4A16 checkpoint built in 8 minutes and served.

- November 4 review (two rounds, [WEEK9_ANALYSIS_PLAN.md](WEEK9_ANALYSIS_PLAN.md)): the November 3 slices had drawn from the final holdout; 15 scenes are now excluded (confirmatory population 116 scenes, 838 questions), every result file carries a provenance sidecar, final-split runs and analyses are refused without a matching freeze, pairing validates eight fields, and the November 3 profile and stability numbers are reclassified as official-checkpoint reference measurements.

**In progress (week 9, two weeks behind the plan).** Plan steps 2–5 (prespecified contrasts, pooled profile and tables, judge changes, freeze script run), then the core freeze, then one GPU session (own-checkpoint stability and profile, the 8-run matrix).

**Not started.** Profile sessions 2 and 3, VQA judging (budget decision), analysis and explorer (weeks 10–11), mitigation (12), report (13–14).

**Open team decisions** are listed in [TEAM_REVIEW.md](TEAM_REVIEW.md), section 8 and the October 19 addendum.

Pilot results (`results/pilot/`, 152 KB) are tracked so teammates can run `05_compare_flips.py` on them. Full-run results stay untracked and are synced from the host; the final raw files are archived with the submission.
