# Team review: scope change and decisions (September 29 – October 1, 2026)

For Bhagyesh and Akshay. This summarizes what happened after the professor reviewed our September 8 proposal, what was changed in the repo, what has been built so far, and what still needs your sign-off. Read the "Decisions that need your confirmation" section first if you are short on time.

## 1. The project in one paragraph

We take one driving vision-language model (Qwen2.5-VL-7B-Instruct), run it at full precision (BF16) and at compressed precision (INT4), ask both the same DriveBench driving questions, and find every question where the full-precision model was right but the compressed model changed its answer. We then look for what those failures have in common: the kind of question, the answer format, how confident the full model was, how many camera images the question used, and whether the images were degraded. Alongside that, we measure how much memory and time INT4 actually saves. If there is time, we test one fix: keep a few sensitive layers at full precision and see whether fewer answers flip.

## 2. What the professor said and why we followed it

The professor's feedback on the September 8 proposal, paraphrased:

- Sensor-corruption robustness is already well studied, so it is a weak main contribution.
- Two deployment stacks (FlashOcc in TensorRT and Qwen-VL in vLLM) plus corruption evaluation, mitigation, energy profiling, and the statistics is too much for one semester.
- Keep one model family, preferably the VLM.
- Make the central question quantization-induced decision failures: same DriveBench examples through BF16 and INT4, find where full precision is correct but quantization changes the answer or driving decision.
- Analyze those failures by task type, answer format, confidence/margin, number of camera views, and optionally corruption. Corruption stays as one stress test.
- If time permits, test one simple mitigation such as retaining sensitive layers at higher precision.
- A careful paired BF16-vs-INT4 failure analysis with measured latency and memory would already be a strong, much more feasible project.

We adopted all of it. Two reasons beyond "the professor said so":

1. The perception strand carried the biggest engineering risk in the old plan. FlashOcc pins TensorRT 8.4 and CUDA 11.6, which predate the RTX 4090, so we would have had to rebuild its custom CUDA plugin before measuring anything. That was Ronak's strand, and it is the one most likely to eat the semester.
2. The literature survey had already found direct prior work on both strands (BRCB for compressed BEV models under corruption; Mathew and Gu for 4-bit VLMs on DriveBench). The one thing nobody reports is per-question pairing. That is now the whole project.

## 3. The new research question, with an example

**Question:** When the BF16 model answers a DriveBench question correctly, how often does the INT4 model change the answer, and which question properties predict those harmful changes?

**Why averages hide this.** Suppose both models score 80% on 500 multiple-choice questions. The accuracy loss is 0. But if 40 questions flipped from right to wrong and 40 other questions flipped from wrong to right, 10% of the BF16-correct answers were lost. Mathew and Gu reported that the 4-bit gap is "not very large"; that is an aggregate statement, and this is what it can hide.

**Flip classes, per question:**

| BF16 | INT4 | Class |
|---|---|---|
| correct | incorrect | **harmful flip** (primary outcome) |
| incorrect | correct | beneficial flip |
| same correctness, different answer | | neutral change |
| same normalized answer | | unchanged |

**Example of a harmful flip (from the fake-server dry run, so the numbers mean nothing, but the shape is what we will report):**

```
perception/MCQ  gt=A  bf16=A  int4=C  margin=0.97
```

Here BF16 answered A with 97% probability margin over its second choice, and INT4 answered C. That is a harmful flip on a high-confidence question, which would be a surprising and interesting case if it happens on real hardware.

**Primary hypothesis:** harmful flips are concentrated in a predefined subgroup, most plausibly questions where the BF16 model's confidence margin is small. If this holds, the margin becomes a cheap deployment-time warning signal: "do not trust the INT4 answer when the full model was unsure."

**Analysis factors:**

- Task: perception, prediction, planning, behavior. Planning and behavior are the "driving decision" cells.
- Format: multiple-choice (MCQ) versus open-ended (VQA). Never pooled.
- Confidence margin: for MCQ, probability gap between the top two option letters at the first generated token, read from vLLM `logprobs`.
- Number of camera views: 1 to 6, as selected by the question.
- Condition: clean versus Dark (`LowLight`), Fog, NoImage.

## 4. Every decision made, with reasoning

### Scope

| Decision | Reason |
|---|---|
| Drop the perception strand entirely (FlashOcc, BEVDet, TensorRT, INT8 calibration, RoboBEV) | Professor's recommendation; largest engineering risk; area already occupied by BRCB and PTQAT. |
| Corruption is a stress test: clean + LowLight + Fog + NoImage, not the full 17-condition suite | Professor said "optionally corruption condition." Three conditions keep the matrix small (8 runs instead of 26) while still allowing an image-dependence check via NoImage. **Pending your confirmation.** |
| GPU energy moved to an extension | Professor's list of required measurements was latency and memory. Energy adds NVML integration work for no change in the research question. **Pending your confirmation.** |
| Mitigation is "if time permits," scheduled for week 12, first thing cut if behind | Professor's wording. |
| Timeline restarted at semester week 4 (week of September 28). Pilot in weeks 4–5, freeze and build 6–7, full runs 8–9, analysis 10–11, mitigation 12, report 13–14 | Weeks 1–3 went to the original proposal. |

### Model and runtime

| Decision | Reason |
|---|---|
| Qwen2.5-VL-7B-Instruct, BF16 reference, AWQ INT4 in vLLM. Fallback Qwen2.5-VL-3B if 7B does not fit 24 GB with six views | Unchanged from the original proposal. |
| Official `Qwen2.5-VL-7B-Instruct-AWQ` checkpoint for the pilot only. Final INT4 and any mitigation variant are quantized by us from the base checkpoint with one pinned tool (llm-compressor preferred) | The mitigation must share the same recipe, calibration set, and base weights as the baseline INT4, otherwise the comparison is confounded. This was not in the professor's note; it is our addition. Fallback rule: if own quantization is blocked by week 7, use the official checkpoint and report mitigation as not executed. |
| Greedy decoding, 128-token answers, fixed seed | The released DriveBench adapter samples at temperature 0.2. Sampling noise would be indistinguishable from quantization flips. A three-repeat slice per precision measures the remaining nondeterminism floor. |

### Evaluation protocol

| Decision | Reason |
|---|---|
| MCQ accuracy is the primary flip signal. VQA rubric scoring is secondary and depends on judge access and budget | MCQ flips are exact (letter changed or not). VQA requires a GPT judge that costs money; Mathew and Gu used GPT-5-mini. **Pending your confirmation.** |
| Append "Begin your answer with the option letter." to MCQ questions | The margin must be read from the first generated token. Without this, the model may begin with "The" and the option-letter probabilities are not at the first position. Configurable; turn off with `mcq_suffix: ""`. |
| Send original images and cap tokens with vLLM `max_pixels` (about 252 tokens per view). Do not reproduce the released adapter's 224×224 resize by default | 224×224 is very coarse for driving imagery. But this is a deviation from the released protocol, so it is a config switch (`resize_to: [224, 224]`) and must be stated in the report. **Pending team discussion.** |
| Record both the released MCQ normalizer and a strict one | The released `preprocess_answer` regex `\b(a|option a)\b` matches the article "a" in free text as option A. We report scorer-compatible accuracy and also our strict version. |
| Scene split: 10% calibration, 15% development, 75% final, seeded, disjoint. All prompt/budget/mitigation choices made on development scenes only | Standard held-out discipline. On the real annotations this leaves 131 final scenes with 1116 questions, but only 45 prediction-VQA questions; that cell may end up "unassessed." |
| Paired bootstrap over scenes, 95% intervals, one prespecified confirmatory comparison per factor; everything else exploratory | Replaces the Bonferroni family machinery from the original proposal. Many questions come from one scene, so resampling questions would overstate precision. |

### Roles (proposed, pending confirmation)

| Person | Owns |
|---|---|
| Ronak | INT4 quantization pipeline, vLLM serving, latency and memory profiling, mitigation |
| Akshay | DriveBench adapter, prompt and view handling, scorer integration, flip classification, human-review sample of 100 answer pairs |
| Bhagyesh (maintainer) | Manifests and scene splits, run logging and schemas, bootstrap statistics, factor analysis, Docker, result explorer |

## 5. What was changed in the repo

All four course documents were rewritten on September 29:

- `C_PROJECT_PROPOSAL.md`: new title, abstract, research question, hypotheses, flip-classification protocol, analysis factors, 8-run matrix, trimmed statistics, VLM-only budgets, restarted timeline, roles.
- `README.md`: new title, abstract, scope with an explicit "removed" list.
- `B_LITERATURE_SURVEY.md`: same ten papers (course requirement), but the four perception papers now explain why that strand was dropped, and Mathew and Gu is the main comparison with three concrete differences from our design listed.
- `D_AI_NOVELTY_FEASIBILITY_AUDIT.md`: new top section with the instructor feedback, our response, a revised critique, and the pending-decisions list. The original September 8 Claude critique is kept verbatim as a record. Scope table, risks, gates, and claims-to-avoid redone.

## 6. What has been built (pilot skeleton)

Code for the week 4–5 pilot is in place and tested offline. Nothing has been run on the GPU yet.

```
PILOT.md                      runbook: setup, serve, smoke test, 20-question run, scene split
config/pilot.yaml             every tunable in one place
prompt.txt                    DriveBench system prompt, verbatim
src/qflip/                    annotation loading, view selection, prompt building,
                              vLLM client with logprobs, margin, GPU memory sampler, resumable records
scripts/download_drivebench.py   HF annotations + Google Drive image archive + verify
scripts/00_check_machine.sh      host/GPU facts
scripts/01_serve.sh bf16|int4    vllm serve with the proposal's flags
scripts/02_smoke_request.py      one six-view request: tokens/view, logprobs, margin, peak memory
scripts/03_smoke_drivebench.py   20 stratified questions per (precision, condition), resumable
scripts/04_scene_split.py        calibration/development/final manifest
scripts/05_compare_flips.py      pair two result files, count flips by cell and margin
tests/test_drivebench.py         7 unit tests, pass
```

Verified so far:

- Unit tests pass.
- All scripts ran end to end against a fake OpenAI-compatible server with 312 real DriveBench records and dummy images. The wiring works; no model numbers exist yet.
- Annotations downloaded from HF: 1461 questions, 174 scenes, 1200 clean images referenced. Cell counts: behavior MCQ 200, perception MCQ 200, perception VQA 200, planning VQA 600, prediction VQA 61, prediction MCQ 200 (dropped by the released scorer).
- Scene split generated on the real annotations.

Not yet done: the image archive download (size unknown, from Google Drive), vLLM install, anything on the 4090.

## 7. Facts learned from the DriveBench source that affect the protocol

- Annotation fields: `scene_token`, `frame_token`, `question_type`, `question`, `answer` (ground truth), `tag`, `image_path` (dict camera to path). Model output goes in a `pred` field; our scripts write that format so the upstream scorer runs on it.
- The paper's "Dark" condition is the `LowLight` directory in the released code.
- The released adapter resizes every image to 224×224 before inference (see the decision above).
- The released scorer silently drops prediction MCQ, raises on unknown tags, and only calls GPT with `--eval-gpt`. It also needs `data/visual_description.json`, which is not in the HF dataset; we still need to locate it.
- Corruption images are found by replacing `nuscenes/samples` with `corruption/<Type>` in each path, so corrupted and clean images share file names.
- The HF dataset also has per-condition JSONs (`fog.json`, `lowlight.json`, ...). The released run script does not use them; it uses one JSON plus path replacement. What those files are for is unverified.

## 8. Decisions that need your confirmation

Please reply yes/no or propose an alternative for each.

1. Stress-test conditions: clean + LowLight + Fog + NoImage. (Alternative: add CameraCrash, or use only clean + NoImage.)
2. MCQ accuracy primary; VQA secondary and conditional on judge budget. (Alternative: budget for GPT judging of all VQA cells from the start. Needs an API key and a cost estimate from the pilot.)
3. Energy measurement is an extension, not required.
4. Roles as listed in section 4.
5. Image handling: originals with a token cap (our default) versus the released 224×224 resize. This changes comparability with Mathew and Gu and DriveBench's own numbers.
6. Send the revised proposal to the professor now, before the pilot, or after the week-5 pilot numbers exist.

## 9. Immediate next steps

1. Confirm the six items above.
2. Commit the revised docs and skeleton (nothing is committed yet).
3. On the 4090 machine: `bash scripts/00_check_machine.sh`, install vLLM, `python scripts/download_drivebench.py`.
4. `bash scripts/01_serve.sh bf16`, then `02_smoke_request.py`. The gate checks printed at the end decide 7B versus 3B and whether the margin analysis is feasible.
5. Twenty-question run on both precisions, `05_compare_flips.py`, and write the week-5 gate note (items listed at the bottom of `PILOT.md`).
