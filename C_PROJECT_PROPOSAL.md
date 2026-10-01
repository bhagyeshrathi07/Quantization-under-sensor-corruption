# C. Project Proposal

## Quantization-Induced Decision Failures in a Driving Vision-Language Model: A Paired BF16-versus-INT4 Deployment Study

**Institution:** San Jose State University  
**Team:** Ronak, Bhagyesh, and Akshay  
**Duration:** 14 weeks  
**Track:** System / Deployment (supported by experimental research)  
**Prepared:** September 8, 2026. **Revised:** September 29, 2026, after instructor feedback (see [D_AI_NOVELTY_FEASIBILITY_AUDIT.md](D_AI_NOVELTY_FEASIBILITY_AUDIT.md), "Revision after instructor feedback").  
**Scope:** One model family, the driving vision-language model (VLM) strand. The camera-based occupancy perception strand from the September 8 proposal has been removed. Sensor corruption is retained as one stress test, not as the primary contribution.  
**Status:** Proposed experiments; no results or hardware measurements are claimed in this document.

## Abstract

Quantization reduces the precision of the numbers used inside a neural network, potentially reducing memory use and inference time. Average accuracy on a benchmark may stay almost unchanged after quantization while individual answers still change, and some of those changes turn a correct driving decision into an incorrect one. This project will deploy one driving-scene vision-language model, Qwen2.5-VL-7B-Instruct, in vLLM at BF16 and at INT4, run the same DriveBench questions through both, and identify every case in which the full-precision model is correct but the quantized model changes the answer or the driving decision. We will analyze these failures by driving task, answer format, the full-precision model's answer confidence (margin), the number of camera views, and, as a stress test, a small set of sensor-corruption conditions. We will measure the latency and memory savings that INT4 provides under matched serving settings. If time permits, we will test one simple mitigation: retaining a small set of sensitive decoder layers at BF16. The contribution is a reproducible, hardware-measured, paired failure analysis of a quantized driving VLM, extending existing quantized-driving-VLM research that reports only aggregate scores. The study runs on one RTX 4090 (24 GB); an RTX 3060 Ti (8 GB) is supplementary, and neither HPC access nor a Jetson is required.

## 1. Problem formulation

### Domain problem

An engineer selecting a lower-precision VLM for a driving assistant needs to know more than its average benchmark score. Two models with nearly equal average accuracy can disagree on many individual questions, and a disagreement on a planning or behavior question is a changed driving decision. Prior research has already quantized driving VLMs and reported that the aggregate 4-bit gap is "not very large" ([Mathew and Gu](https://doi.org/10.1109/ICDMW69685.2025.00440)). The unanswered project question is what that aggregate hides: which answers change, whether those changes are harmful, and which properties of a question predict that its answer will flip. We do not claim to introduce quantization for driving VLMs.

**Research question:** When the BF16 model answers a DriveBench question correctly, how often does the INT4 model change the answer or the driving decision, and which question properties (driving task, answer format, BF16 confidence margin, number of camera views, corruption condition) are associated with those harmful changes?

**Primary hypothesis:** Harmful answer flips (BF16 correct, INT4 incorrect) are not spread uniformly. Their rate will be measurably higher in at least one predefined subgroup, most plausibly questions on which the BF16 model's confidence margin is small, than the overall harmful-flip rate. This is a hypothesis to test, not an expected finding presented as fact.

**Secondary hypothesis (mitigation, if time permits):** Retaining a small set of sensitive decoder layers at BF16 reduces the harmful-flip rate while retaining most of the INT4 memory saving. This evaluates an established mitigation principle ([MBQ](https://arxiv.org/abs/2412.19509)); it is not a new quantization algorithm.

### Inputs and outputs

| Item | Content |
|---|---|
| Model input | The benchmark-selected camera views in their recorded order (one to six views), a driving question, and a fixed prompt |
| Model output | A multiple-choice (MCQ) or open-ended (VQA) answer; for MCQ, the token log-probabilities over the option letters |
| Study output per question | BF16 answer and correctness, INT4 answer and correctness, flip class (harmful / beneficial / neutral / unchanged), BF16 margin, INT4 margin, view count, task, format, condition |
| Study output aggregate | Harmful- and beneficial-flip rates with uncertainty, broken down by each factor; matched latency, throughput, and peak GPU memory for BF16 and INT4; mitigation tradeoff if executed |

Reference answers are evaluation inputs, not inputs supplied to the model. Corruptions modify the camera images while retaining the question, view selection, and reference answer.

## 2. Model, precision pair, and runtime

**Model: Qwen2.5-VL-7B-Instruct.** A public, ViT-based VLM with an accessible deployment path in vLLM. Qwen3-VL is newer and is covered in the survey; Qwen2.5-VL is a reproducibility choice, not a claim about current SOTA. If the full BF16 workload (about 16.6 GB of bf16 weights plus activations and KV cache) does not fit in 24 GB or does not execute reliably in the pinned vLLM after setting a fixed image-token budget, use Qwen2.5-VL-3B-Instruct for every comparison. Reducing model size preserves a complete BF16/INT4 comparison. [Model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [vLLM model support](https://docs.vllm.ai/en/stable/models/supported_models/).

| Required experiment | Reference | Quantized configuration | Runtime |
|---|---|---|---|
| One VLM | BF16, same processor and prompt | One INT4 weight-only method: AWQ preferred, GPTQ contingency | vLLM |

**Quantization recipe.** The official `Qwen2.5-VL-7B-Instruct-AWQ` checkpoint is used only for the pilot. For the final comparison and for the mitigation, we quantize the base checkpoint ourselves with one pinned tool (llm-compressor preferred, AutoAWQ contingency) so that the INT4 baseline and any mitigation variant share the same recipe, calibration set, and base weights. Record which components are quantized: the official recipe quantizes the language decoder and leaves the vision encoder at BF16, so the INT4 model is a mixed-precision model, and an unquantized vision encoder must not later be credited as a mitigation. Exact checkpoint revision, quantization settings, package versions, and container digests will be pinned after the pilot; they are not fabricated at proposal time.

**Extensions (not required):** FP8 on the 4090, a second VLM (for example a small Qwen3-VL variant, each paired with its own BF16 reference), the remaining DriveBench corruption conditions, GPU energy per answer, and Jetson measurements. W4A8, NF4, custom kernels, QAT, and the removed perception strand are out of scope.

## 3. Data and experiment matrix

**Benchmark: DriveBench.** Required driving-task cells are perception MCQ/VQA, prediction VQA, planning VQA, and behavior MCQ, subject to validation against the pinned annotations and scorer. Caption questions belong to the separate corruption-awareness data and are optional. The released preprocessing extracts camera names from `<c1,CAM_BACK,x,y>` references in each question and keeps only those views, retaining all six otherwise; we preserve each question's selected view subset and order identically across precisions and conditions and record the resulting view count as an analysis factor. [DriveBench task tables](https://arxiv.org/html/2501.04003v1#A1), [preprocessing code](https://github.com/worldbench/DriveBench/blob/main/tools/preprocess.py), [evaluation code](https://github.com/worldbench/DriveBench/blob/main/evaluate/eval.py).

**Scene separation.** Before any experiment, reserve disjoint calibration and development scene lists, publish them, and exclude both from final evaluation. Use compatible DriveLM training-scene examples for calibration and development where available, removing any scene that overlaps DriveBench; otherwise reserve scenes from DriveBench itself and publish the reduced final-evaluation protocol. All prompt, image-token, serving, and mitigation choices are made on development scenes only. [DriveLM](https://github.com/OpenDriveLab/DriveLM).

**Conditions.** The primary experiment is the **clean** condition. The stress test adds three released DriveBench conditions: **Dark**, **Fog**, and the **NoImage** control. Use the published corruption realizations; do not introduce new severities. The released adapter implements NoImage by substituting a 224×224 all-black image per view while keeping one `<image>` placeholder per view; reproduce that consistently with the Qwen processor and record the resulting image-token counts. The remaining DriveBench conditions are an extension. [Released NoImage adapter](https://github.com/worldbench/DriveBench/blob/main/inference/llava1.5.py).

**Minimum matrix:** 2 precisions × 4 conditions = 8 quality runs over the eligible final-evaluation question set, plus profiling repetitions. Mitigation adds 1 precision configuration × up to 4 conditions after development choices are frozen. This is a deliberately narrow matrix, not a reproduction of the full DriveBench leaderboard.

**Eligibility and reporting.** Audit the pinned scorer: it routes five cells (perception MCQ and VQA, prediction VQA, planning VQA, behavior MCQ), silently drops prediction MCQ (`pass`), raises on unknown tags, and runs GPT scoring only with `--eval-gpt`. The clean prediction-VQA cell has only 61 questions and may fall below the declared coverage minimum after scene holdout, in which case it is reported as unassessed. Publish eligible, excluded, skipped, and failed counts with reasons. Do not silently drop inference failures from denominators. Do not compare a held-out-subset score with a published full-benchmark score as if the protocols matched.

**Storage and access.** DriveBench images derive from nuScenes; check redistribution rights and judge-service costs before acquisition or publication. Publish manifests and derived results rather than assuming images may be redistributed.

## 4. Proposed technical approach

```text
DriveBench questions + immutable scene/question manifest
                      |
      clean / Dark / Fog / NoImage inputs (same view subset per question)
                      |
              vLLM adapter (fixed prompt, decoding, image budget)
           BF16 reference   |   INT4 (AWQ)   |   [INT4 + retained BF16 layers]
                      |
   answers + option logprobs + latency + memory + run metadata
                      |
      scoring -> per-question flip classification -> margin / views / task / format
                      |
     paired bootstrap over scenes -> breakdown tables + failure explorer
```

### 4.1 Serving adapter

Build a vLLM adapter that preserves each question's view subset and order, image references, question text, and response-format instructions. Start with batch size 1, concurrency 1, `--max-model-len 4096`, `--limit-mm-per-prompt '{"image":6}'`, an image-token cap targeting at most 256 tokens per view (`max_pixels = 200704`, about 252 tokens for a 1600×900 frame; a 512-token setting will be piloted because 252 tokens is coarse for driving imagery), greedy decoding, and a 128-token answer limit. Without these flags vLLM's start-up profiling assumes 1,280 tokens per image and can exhaust 24 GB. Validate settings on development scenes and use the same settings for every precision. Record truncated and invalid answers. A small Transformers comparison on a handful of questions checks adapter correctness.

### 4.2 Flip classification

For every eligible question and condition, record the BF16 and INT4 answers and score both with the same pinned scorer. Classify each pair:

- **Harmful flip:** BF16 correct, INT4 incorrect. Primary outcome.
- **Beneficial flip:** BF16 incorrect, INT4 correct.
- **Neutral change:** both correct or both incorrect but the answer differs.
- **Unchanged:** same normalized answer.

For MCQ, normalize option letters before comparison, so a flip is exact. For open-ended VQA, correctness comes from the rubric score against a fixed threshold frozen on development data, and a paraphrase alone does not count as a change; a "decision change" in planning VQA is a change in the recommended action, judged by the same rubric. Report flip rates separately from rubric-score changes. Repeat a small fixed slice three times per precision to estimate runtime/decoding instability, so that flips are not confused with nondeterminism.

### 4.3 Analysis factors

- **Driving task:** perception, prediction, planning, behavior. Planning and behavior are the "driving decision" cells.
- **Answer format:** MCQ versus VQA, never pooled.
- **Confidence margin:** for MCQ, request `logprobs` at the answer position and compute the BF16 model's probability gap between its top two option letters. Bin questions by margin (bins frozen on development data) and report the harmful-flip rate per bin. Record the INT4 margin as well. This factor is the most direct test of the primary hypothesis and yields a cheap deployment-time warning signal if it holds.
- **Number of camera views:** one to six, as selected by the question.
- **Condition:** clean versus Dark, Fog, NoImage. Report the stress test as a secondary breakdown; it is not the primary contribution.

### 4.4 Mitigation (if time permits)

Estimate per-layer sensitivity on development scenes (for example, by re-quantizing with one decoder block group excluded at a time, or by output-divergence measurement), select one small set of decoder layers to retain at BF16 using the quantizer's supported exclusion mechanism, freeze that choice, then re-quantize from the base checkpoint with the same recipe. Evaluate harmful-flip rate, MCQ accuracy, peak memory, and throughput against the unmodified INT4 model. If the pinned runtime cannot execute a mixed-precision checkpoint, use a fixed-size balanced calibration set as the controlled mitigation instead and state the substitution.

### 4.5 Shared infrastructure

Every run saves a configuration hash, model and code revisions, question and scene IDs, seeds, precision coverage, device identity, elapsed time, error records, raw answers, and logprobs. Runs resume from completed questions. A public result explorer filters by task, format, margin bin, view count, and condition and shows paired BF16/INT4 examples of harmful and beneficial flips. It is an evaluation tool, not a vehicle-control application.

## 5. Metrics and success criteria

### 5.1 Quality and flips

Let S be a higher-is-better score. Quantization loss at condition c is L(c) = S(BF16, c) − S(INT4, c), computed within one format and scale (MCQ accuracy, or VQA rubric score on 0–100). Harmful-flip rate F_h(c) is the number of harmful flips divided by the number of eligible questions the BF16 model answered correctly at c. Beneficial-flip rate F_b(c) is defined symmetrically. The primary comparison is F_h in the clean condition, overall and per predefined subgroup.

For example, if BF16 and INT4 both score 80% on 500 MCQ questions, L = 0, yet if 40 questions flipped harmfully and 40 beneficially, F_h = 40/400 = 10%. These numbers illustrate the calculation and are not results.

**Uncertainty.** Use a paired bootstrap over scenes, resampling whole scenes and keeping all questions, precisions, and conditions within a resampled scene together. Report 95% percentile intervals for F_h, F_b, L, and each subgroup contrast. One prespecified confirmatory comparison per factor (for example, lowest-margin bin versus the rest) is stated before the final run; all other breakdowns are exploratory. A subgroup contrast supports the primary hypothesis only when its interval excludes zero. Cells lacking the frozen minimum scene coverage are **unassessed**, not passes. A nonsignificant difference is not evidence of equivalence or of driving safety.

**Scoring.** MCQ accuracy is the primary objective metric and the basis for exact flips. VQA cells use the benchmark-compatible rubric via a fixed judge version and prompt if judge access and budget are available; otherwise, publish MCQ results plus a human-scored sample and mark unscored VQA cells unassessed. Independently double-review a balanced sample of 100 answer pairs, including apparent harmful flips, and report agreement.

### 5.2 Deployment budgets

Project targets, not achieved results or safety standards. Freeze interpretation after the pilot and before inspecting final results.

| Criterion | Target |
|---|---|
| Local execution | Complete evaluation on one 24 GB RTX 4090 |
| Memory | INT4 peak GPU memory at least 25% lower than BF16 under equal KV-cache byte capacity, dtype, context, workload, and concurrency |
| Response | p95 time to first token ≤2 s; median generation throughput ≥15 tokens/s |
| Quality tolerance | Bootstrap upper bound on MCQ loss ≤3 percentage points and on VQA rubric loss ≤5 points for every required eligible cell in the clean condition; stress-test cells reported against the same limits but not required to pass |
| Transparency | Report every failed budget |

Set the same explicit `--kv-cache-memory-bytes`, KV dtype, and context for both precisions; an equal fraction-of-GPU reservation is not equal cache capacity. Separate reserved memory, model allocations, and observed peak. Publish p50/p95/p99 latency, time to first token, and total answer time from at least three independent timing sessions with warm-up on a fixed workload. GPU energy per answer via NVML is an extension.

**Scientific completion is distinct from hitting every budget.** Both actual-runtime precision configurations, every required eligible cell with its declared coverage, raw logs, flip classification, uncertainty analysis, and the reproducibility package must be completed. A result that misses a budget or is inconclusive is valid evidence; a missing run or unscored cell is incomplete work and is reported as such.

## 6. Feasibility, ownership, and timeline

Primary resource: a team-owned workstation with one RTX 4090 (24 GB); a second machine with an RTX 3060 Ti (8 GB) for development. Host details, driver, CUDA, and free disk are recorded in the pilot run manifest, not asserted here. Plan around 8–10 hours per member per week. Estimated ramp-up: 2–3 weeks for serving/quantization, 1–2 weeks for scoring and statistics infrastructure, overlapping.

The revised plan starts at semester week 4 (week of September 28, 2026); weeks 1–3 were spent on the original proposal, survey, and audit.

| Weeks | Required milestone |
|---|---|
| 4–5 | Pilot: pin vLLM; serve BF16 and official AWQ INT4; run 20-question smoke test; confirm 24 GB fit (else switch to 3B); confirm `logprobs` available; publish scene splits |
| 6–7 | Freeze prompt, image budget, decoding, flip definitions, margin bins, eligibility; build adapter, scorer integration, logging; quantize own INT4 from base checkpoint; dry run on development scenes |
| 8–9 | Run the frozen 8-run matrix on final-evaluation scenes; profiling sessions; repeat-slice stability runs |
| 10–11 | Flip classification, bootstrap intervals, factor breakdowns, human-review sample, result explorer |
| 12 | Mitigation: sensitivity on development scenes, re-quantize, evaluate (skipped if behind schedule; reported as not executed) |
| 13–14 | Reproduce representative runs from a clean setup; documentation, final report, presentation, demo video |

**Roles (proposed; to be confirmed by the team):** **Ronak:** INT4 quantization pipeline, vLLM serving, latency/memory profiling, mitigation. **Akshay:** DriveBench adapter, prompt and view handling, scorer integration, flip classification, human-review sample. **Bhagyesh (maintainer):** manifests and scene splits, run logging and schemas, bootstrap statistics, factor analysis, Docker orchestration, result explorer. All members review design and writing.

**Failure gates:** End of week 5, if BF16 7B does not fit or serve reliably, switch to 3B immediately. End of week 7, if no executable INT4 runs in vLLM, request advisor review of the blocked component; do not relabel fake quantization as deployment. Mitigation and all extensions stop until the required pair and matrix are complete.

## 7. Expected deliverables and academic contribution

Required deliverables: the ten-paper literature survey; this proposal; the AI novelty/feasibility audit with its revision record; a public codebase with reproducible setup and commands; raw logs, manifests, and per-question flip tables; breakdown tables with uncertainty; a result explorer with paired failure examples; and the final report, presentation, and demo video. Model and data downloads may require registration, so one-command execution begins after documented asset acquisition.

The expected contribution is evidence about which driving-VLM answers a BF16-to-INT4 deployment changes, whether those changes are harmful, and which observable question properties predict them, together with the measured memory and latency the deployment buys. Findings are specific to the tested model, quantizer, and benchmark and will not be generalized to all driving VLMs. A workshop submission, an upstream issue, and a Jetson row are extensions; positive results and real-world driving safety are not promised outcomes.

## 8. Model and software stewardship

Development and measurement machine: team-owned RTX 4090 workstation; RTX 3060 Ti machine for development. Bhagyesh is repository maintainer and the point of contact for the codebase, CI/Docker, and any agent-assisted tooling; Ronak owns serving and quantization; Akshay owns evaluation. Upstream contact points: [Qwen](https://github.com/QwenLM/Qwen3-VL/issues), [DriveBench](https://github.com/worldbench/DriveBench/issues), [vLLM](https://github.com/vllm-project/vllm/issues), [llm-compressor](https://github.com/vllm-project/llm-compressor/issues). Actual revisions, license observations, device details, and successful smoke-test commands will be recorded in a machine-readable run manifest.

The companion files [B_LITERATURE_SURVEY.md](B_LITERATURE_SURVEY.md) and [D_AI_NOVELTY_FEASIBILITY_AUDIT.md](D_AI_NOVELTY_FEASIBILITY_AUDIT.md) provide the literature comparison, the audit, and the record of the September 29 scope revision. Course number and instructor attribution are added at submission; no institutional approval is implied.
