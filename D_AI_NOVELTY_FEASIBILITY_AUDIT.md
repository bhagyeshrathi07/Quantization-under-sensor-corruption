# D. AI Novelty & Feasibility Audit

**Project:** Quantization-Induced Decision Failures in a Driving Vision-Language Model: A Paired BF16-versus-INT4 Deployment Study (formerly "Quantization Under Sensor Corruption: Deployment and Robustness of Driving Perception and Vision-Language Models")  
**Audit date:** September 8, 2026. **Revised:** September 29, 2026, after instructor feedback.  
**Assessment basis:** The team's pre-audit project brief (not submitted), a targeted search of recent papers and official repositories, the proposed 14-week experiment plan, and the instructor's written feedback on the September 8 proposal. Quoted "original claims" below refer to the pre-audit brief. This audit reports a literature-based assessment; no project experiments or performance measurements have been completed.

## Revision after instructor feedback (September 29, 2026)

The instructor reviewed the September 8 proposal and recommended reducing scope substantially. The feedback, paraphrased:

1. Sensor-corruption robustness is already well studied, so it is a weak primary contribution.
2. Two deployment stacks (FlashOcc/TensorRT INT8 and Qwen-VL/vLLM INT4), plus corruption evaluation, mixed-precision mitigation, energy profiling, and the statistical analysis, is too much for one semester.
3. Focus on one model family, preferably the VLM strand.
4. Make the central question quantization-induced decision failures: run the same DriveBench examples through BF16 and INT4 and identify cases where the full-precision model is correct but quantization changes the answer or driving decision.
5. Analyze those failures by task type, answer format, confidence/margin, number of camera views, and optionally corruption condition. Sensor corruption remains as one stress test.
6. If time permits, test one simple mitigation such as selectively retaining sensitive layers at higher precision.
7. A careful paired BF16-versus-INT4 failure analysis with measured latency and memory savings is already a strong and much more feasible project.

**Team response.** The proposal was revised on September 29, 2026 to adopt all seven points. The perception strand, TensorRT, INT8 calibration, the full corruption matrix, and required energy measurement were removed. The research question, hypotheses, flip-classification protocol, analysis factors, timeline, and roles in [C_PROJECT_PROPOSAL.md](C_PROJECT_PROPOSAL.md) were rewritten. The literature survey keeps its ten entries but reframes the four perception papers as context for the scope decision and makes Mathew and Gu the main comparison. Decisions the team still needs to confirm are listed at the end of this section.

**Revised critique summary (team-written, September 29, 2026).** The narrowed project is a paired, per-question deployment study of one quantized driving VLM. Its defensible contribution is not quantization, not DriveBench, and not corruption robustness, all of which exist; it is the per-question pairing of BF16 and INT4 answers, the separation of harmful from beneficial flips, and the test of whether observable question properties (in particular the BF16 confidence margin) predict harmful flips, together with matched memory and latency measurements. The closest prior work, Mathew and Gu (ICDMW 2025), reports aggregate scores and finds the 4-bit gap small; our design is built to look beneath that aggregate. The main remaining novelty risk is that a "prediction churn" or "answer flip" analysis of quantized VLMs may exist outside the driving domain; the survey refresh before the final report must search for that specifically. The main feasibility risk moved from the TensorRT plugin rebuild to two smaller items: fitting the 7B BF16 six-view workload in 24 GB, and obtaining option-level log-probabilities from vLLM for the margin analysis. Both are tested in the week 4–5 pilot and both have fallbacks (3B model; margin from a constrained single-token answer format).

**Decisions pending team confirmation (assumed in the revised documents):**

- Stress-test conditions: clean plus Dark, Fog, and NoImage (three of the seventeen DriveBench conditions).
- MCQ accuracy as the primary flip signal; VQA rubric scoring secondary and dependent on judge access and budget.
- GPU energy demoted to an extension.
- Roles: Ronak owns quantization, serving, profiling, and mitigation; Akshay owns the DriveBench adapter, scoring, and flip classification; Bhagyesh owns infrastructure, statistics, and the explorer.

## Original paste-ready AI critique summary (September 8, 2026)

*Critique generated with Claude (Anthropic) on September 8, 2026, from the project brief, B_LITERATURE_SURVEY.md, and C_PROJECT_PROPOSAL.md, then verified claim-by-claim against primary sources on the same date; lightly edited for length. Retained unchanged as the record of the original audit. Passages about the perception strand are superseded by the September 29 revision above.*

> **Verdict: a viable graduate systems/deployment project after narrowing, with substantial overlap with existing research.** The proposal should retain both required strands: one camera-based perception model executed in TensorRT and one vision-language model executed in vLLM. Quantized BEV robustness and quantized driving VLM evaluation are already studied, so neither strand should be described as a previously untouched topic. BRCB evaluates compressed BEV models under corruption; PTQAT includes quantized 3D perception and a TensorRT deployment; Mathew and Gu evaluate full-precision and 4-bit VLMs on DriveBench. These papers must be acknowledged as direct prior work. [BRCB](https://www.ijcai.org/proceedings/2025/766), [PTQAT](https://arxiv.org/abs/2508.10557), [Mathew and Gu](https://doi.org/10.1109/ICDMW69685.2025.00440)
>
> The proposed contribution is a controlled, reproducible study of a deployment decision: whether a precision chosen using clean overall quality continues to meet the project's predefined quality and runtime budgets under selected sensor corruptions and driving question types. The analysis should separate model conversion errors from quantization losses, compare the same examples, report uncertainty across driving scenes, and measure actual execution on the named hardware. One controlled mitigation should test whether retaining selected components at higher precision improves this tradeoff. This is a candidate empirical and engineering contribution, not a claim to have invented quantization or mixed precision.
>
> **The main red-ocean risk is producing another table of average accuracy and speed without explaining what the average hides.** The stronger result is an auditable explanation of which precision choices remain acceptable under the tested conditions, which fail, and what a mitigation costs. General VLM quantization and sensitivity-aware calibration are active research areas: MBQ and QIG already address unequal quantization sensitivity. [MBQ](https://arxiv.org/abs/2412.19509), [QIG official implementation](https://github.com/ucas-xiang/QIG)
>
> **Oversaturated areas to avoid as the headline claim:** (1) generic "quantize a VLM and report accuracy/latency" tables; (2) new sensitivity-aware PTQ calibration methods (MBQ, QIG, PTQAT already occupy this); (3) corruption-robustness benchmarking of BEV models by itself (RoboBEV, BRCB); (4) "first quantized driving VLM evaluation" (EM-VLM4AD, CVPR 2024 workshop; Mathew and Gu, ICDMW 2025). The defensible position is the paired, conversion-controlled, per-task and per-corruption deployment-budget analysis on named hardware.
>
> **Feasibility is conditional on early deployment tests.** A three-person team can pursue one model per strand in 14 weeks if the complete experiments fit the RTX 4090 and both actual runtimes work early. The additional architectures, Jetson, FP8, and W4A8 are extensions. An unsuccessful compatibility test must activate a smaller model or supported model fallback within the affected strand; it must not silently remove that strand. HPC access, a borrowed device, merged upstream contributions, and paper acceptance are not dependencies. A negative scientific finding can still be a successful project if both pipelines, the agreed experiments, and the analysis are completed.

The instructor's feedback agreed with the "what the average hides" point and went further than the original critique on scope: rather than keeping both strands, it concluded that one strand with a sharper question was the feasible project. The revised proposal follows the instructor.

## What is already known, and what this changes

| Prior work | Verified relevance | Consequence for the revised proposal |
|---|---|---|
| **Can Small Quantized VLMs Drive?, ICDMW 2025** | Evaluates nine VLMs, including Qwen2.5-VL-3B/7B, at FP16 and as BitsAndBytes 4-bit variants using vLLM on all 17 DriveBench conditions, scored by GPT-5-mini (verified against the published PDF). Main summaries average across driving tasks; finds the 4-bit gap "not very large" and mixed latency effects. Presented in the ICDM 2025 Undergraduate and High School Symposium; cited as an empirical data point rather than a peer-reviewed main-track result. [Publication DOI](https://doi.org/10.1109/ICDMW69685.2025.00440) | **Main comparison.** Quantizing Qwen2.5-VL-7B and evaluating it on DriveBench in vLLM is established. Our contribution is what their aggregate does not report: per-question pairing, harmful versus beneficial flips, and margin/view-count/task/format breakdowns. Different quantizer (AWQ versus BitsAndBytes) means runtime numbers are not comparable across studies. |
| **Are VLMs Ready for Autonomous Driving? (DriveBench), ICCV 2025** | Introduces the benchmark, tasks, corruption conditions, and NoImage control; shows models answer plausibly without image evidence. [Paper](https://arxiv.org/abs/2501.04003) | Required framework, not a contribution. Its per-question view selection provides our view-count factor; its NoImage control lets us check image dependence of flips. |
| **MBQ, CVPR 2025** and **QIG, CVPR 2026** | Sensitivity-aware calibration for VLMs; official code for other model families; pseudo-quantized evaluation commands. [MBQ](https://arxiv.org/abs/2412.19509), [QIG](https://github.com/ucas-xiang/QIG) | Sensitivity-aware quantization is prior art. Our mitigation is a controlled evaluation of retaining sensitive layers, not a new method. |
| **BRCB, IJCAI 2025** and **PTQAT, ICCV Workshops 2025** | Compression of BEV/3D perception models, including under corruption (BRCB) and with a TensorRT W8A8 deployment (PTQAT). [BRCB](https://www.ijcai.org/proceedings/2025/0766.pdf), [PTQAT](https://arxiv.org/html/2508.10557v2) | Evidence supporting removal of the perception strand: that area is occupied and its deployment stack was the project's largest engineering risk. |
| **Less Precise Can Be More Reliable, ICML 2026** | Quantization's effect on CLIP-style VLM reliability, including calibration and noise robustness. [Paper](https://arxiv.org/abs/2509.21173) | Closest existing work to the confidence-margin analysis, on a different model class and task. Must be cited when interpreting the margin result. |

**Tentative gap:** In the inspected studies, we did not find a per-question paired flip analysis of a quantized generative driving VLM with confidence-margin, view-count, task, and format breakdowns and matched serving measurements. This observation supports investigating the project; it does not prove priority. EM-VLM4AD (CVPR 2024 workshop, [arXiv:2403.19838](https://arxiv.org/abs/2403.19838)) evaluated an 8-bit quantized driving VLM on DriveLM without corruption tests or per-question analysis.

## Newer overlap and remaining uncertainty

- **QIG, CVPR 2026:** advances token-level sensitivity estimation; official code released April 3, 2026. A new sensitivity method is outside the required contribution. [Official repository](https://github.com/ucas-xiang/QIG)
- **Less Precise Can Be More Reliable, ICML 2026:** further evidence that quantization robustness beyond average accuracy is already researched; its setting differs from generative driving question answering. [Primary paper](https://arxiv.org/abs/2509.21173)
- **PeLiC-VLM, April 2026:** the publisher abstract describes a quantized image-text adapter for driving VQA. Full-text access was blocked during this audit. Whether it contains a paired per-task analysis is **unverified** and remains an unresolved overlap item. [DOI](https://doi.org/10.1016/j.aei.2026.104424)
- **Prediction churn / answer-flip literature (not yet searched as of September 29):** "prediction churn" and "model disagreement" between compressed and uncompressed classifiers is an existing line of work outside VLMs. The survey refresh must search for churn or flip analyses of quantized LLMs/VLMs specifically before the final report; if one exists, the contribution wording narrows to the driving-decision setting and the margin/view-count factors.

Searches so far covered exact paper titles, official-code availability, DriveBench plus quantization, driving quantization plus question type, and 2026 VLM quantization robustness. This was a targeted audit, not an exhaustive systematic review. Refresh the search before freezing the final report; record new overlaps and adjust the contribution wording rather than claiming that search silence proves absence.

## Required scope and fallback rules (revised September 29, 2026)

| Item | Required semester commitment | Contingency or extension |
|---|---|---|
| **Model and precisions** | Qwen2.5-VL-7B-Instruct BF16 reference and one INT4 configuration (AWQ preferred, GPTQ contingency) executed in vLLM. | If the complete fixed six-view workload cannot fit or execute reliably in 24 GB, use Qwen2.5-VL-3B-Instruct with the same design. FP8 is an extension after the required pair works. |
| **Quantization recipe** | Own quantization from the base checkpoint with one pinned tool (llm-compressor preferred, AutoAWQ contingency), so baseline INT4 and mitigation share a recipe. Official AWQ checkpoint for the pilot only. | If own quantization is blocked by week 7, use the official AWQ checkpoint for the required comparison and report the mitigation as not executed. |
| **Evaluation** | DriveBench clean condition on all eligible required cells; per-question flip classification; breakdowns by task, format, BF16 margin, view count. | Cells below frozen coverage are unassessed. If judge access is unavailable, MCQ is primary and VQA cells are marked unassessed with a human-scored sample. |
| **Stress test** | Dark, Fog, and NoImage from the released DriveBench conditions. | Remaining fourteen conditions are an extension. |
| **Deployment measurement** | Peak GPU memory, time to first token, throughput, p50/p95/p99 latency under matched KV-cache settings. | GPU energy per answer is an extension. |
| **Mitigation** | Not required. | If time permits after week 11: retain selected decoder layers at BF16; if the runtime cannot execute it, use a controlled calibration-set change instead. |
| **Compute** | One RTX 4090 with 24 GB VRAM. | RTX 3060 Ti for development only. HPC and Jetson are optional. |
| **Removed** | Perception strand (FlashOcc-M0, BEVDet, TensorRT, INT8 calibration, RoboBEV protocol). | Not an extension this semester. |
| **Outputs** | Executable BF16 and INT4 pipelines, manifests, logs, per-question flip tables, analysis with uncertainty, documented limitations, report, and demo. | Upstream merges, paper acceptance, and a public submission community are stretch outcomes. |

Keep all quality tolerances, performance targets, and statistical decision rules consistent with the Project Proposal Document; this audit introduces no alternative thresholds.

## Feasibility risks and mitigation (revised)

| Risk | Assessment and response |
|---|---|
| **Lost time** | Weeks 1–3 went to the original proposal. The revised plan has 11 weeks. The pilot starts immediately and in parallel with document revision; mitigation is the first item cut if the schedule slips. |
| **Learning cost** | Serving/quantization ramp-up **2–3 weeks**; scoring and statistics infrastructure **1–2 weeks**, overlapping across the three students. Not measured completion guarantees. |
| **24 GB memory limit** | Test the entire six-view workload, including activations, context, generated tokens, and runtime allocations, in the week 4–5 pilot. Freeze a workable image-token budget before the main evaluation. A 3B result must be labeled as such, with both precisions matched. |
| **Margin measurement** | vLLM must return log-probabilities over the option letters at the answer position. Verify in the pilot with the Qwen chat template. Fallback: constrain the MCQ answer format to a single option token and read the first-token distribution. If neither works, margin is reported as unassessed and the primary hypothesis is tested on task/format/view count. |
| **Runtime versus simulated quantization** | Record which tensors actually execute at INT4 (decoder only; vision encoder BF16). Fake-quantized results do not satisfy the deployment requirement. |
| **Flips versus nondeterminism** | Greedy decoding, fixed seeds, and a three-repeat slice per precision quantify runtime instability; flips smaller than that noise floor are not attributed to quantization. |
| **One shared GPU** | Reserve GPU windows; run scoring, statistics, and explorer work on CPU. Project the full matrix (8 quality runs plus profiling) from pilot throughput. |
| **Evaluation expense** | Pilot the judge pipeline and record cost before committing VQA cells. Retain raw answers. MCQ needs no judge. |
| **Scene dependence** | Resample by scene in the bootstrap; many questions from one scene are not independent. Freeze minimum scene coverage per cell on development data. |
| **Mitigation overfitting** | Select retained layers on development scenes only; freeze before the final run; compare measured cost with the unmodified INT4 model. |
| **Hardware extrapolation** | 4090 results do not establish Orin or other-GPU behavior. Report only tested devices. |

## Prior-work reproducibility status

- **Mathew and Gu:** an official experiment repository was not located. Discuss the study and implement the project's own documented paired comparison; do not imply an exact reproduction of undisclosed settings.
- **DriveBench:** official preprocessing, inference adapters, and evaluator are released and are project dependencies.
- **MBQ/QIG:** official implementations exist, but documented supported models include Qwen2-VL rather than verified turnkey Qwen2.5-VL support. Adaptation is optional. [MBQ code](https://github.com/thu-nics/MBQ), [QIG code](https://github.com/ucas-xiang/QIG)
- **BRCB / PTQAT:** dedicated official implementations were not located; no longer project dependencies after the strand removal.

"Not located" describes this search result. It is not proof that code is unavailable everywhere.

## Decision gates (revised)

1. **End of week 5, baseline gate:** BF16 7B serves the six-view workload in 24 GB with the frozen image budget; official AWQ INT4 serves; 20-question smoke test produces valid answers and scorer output; `logprobs` available for MCQ options; scene splits published. Otherwise switch to 3B or the margin fallback immediately.
2. **End of week 7, executable quantization gate:** own INT4 checkpoint from the base model runs in vLLM and reproduces the pilot; prompt, decoding, flip definitions, margin bins, eligibility, and coverage minimums frozen. Otherwise fall back to the official AWQ checkpoint and drop mitigation.
3. **Before the main sweep, experiment gate:** confirmatory comparisons stated; calibration/development/test separation verified; judge cost recorded or VQA marked unassessed.
4. **Before final reporting, evidence gate:** distinguish experiments completed from planned extensions; report uncertainty, failed budgets, and unexecuted mitigation explicitly; refresh the overlap search.

## Claims to avoid

| Unsupported claim | Evidence-respecting replacement |
|---|---|
| "First quantized driving-VLM evaluation." | "Extends Mathew and Gu (2025) and EM-VLM4AD (2024) with per-question paired flip analysis." |
| "INT4 accuracy is unchanged, so deployment is safe." | "Aggregate loss and harmful-flip rate are reported separately; a small aggregate loss can coexist with a nontrivial harmful-flip rate." |
| "Low-margin questions cause flips." | "Low BF16 margin is associated with a higher harmful-flip rate in the tested model and benchmark, if the interval supports it." |
| "Quantization makes the model faster." | "We report measured latency and memory for each executable configuration under matched settings." |
| "The mitigation is a novel quantization algorithm." | "The study evaluates the flip-rate/memory tradeoff of a controlled layer-retention mitigation informed by existing methods." |
| "No harmful flips means safe to drive." | "Results bound the harmful-flip rate on a benchmark; they do not establish driving safety." |

## Source-code checks that affect the protocol

Observed defaults and interfaces, not claims that a published result is erroneous.

- DriveBench preprocessing selects camera views referenced by the question via `<c1,CAM_BACK,x,y>` tags. Preserve that mapping, record the actual view count per question, and use it as the view-count factor. The released NoImage adapter supplies 224×224 black images while retaining `<image>` placeholders; removing image tokens is a different intervention. [Preprocessor](https://github.com/worldbench/DriveBench/blob/main/tools/preprocess.py), [released adapter](https://github.com/worldbench/DriveBench/blob/main/inference/llava1.5.py)
- DriveBench's released evaluator routes five cells (perception MCQ and VQA, prediction VQA, planning VQA, behavior MCQ), silently drops prediction MCQ (`pass`), raises on unknown tags, and runs GPT scoring only with `--eval-gpt`; the clean prediction-VQA cell has only 61 questions. Publish included/skipped question IDs and the tag-to-score mapping. Unsupported or empty cells are unassessed or N/A, never converted into scores. [Evaluator](https://github.com/worldbench/DriveBench/blob/main/evaluate/eval.py), [paper supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Xie_Are_VLMs_Ready_ICCV_2025_supplemental.pdf)
- The official `Qwen2.5-VL-7B-Instruct-AWQ` checkpoint quantizes the language decoder and leaves the vision encoder at BF16. Any own-quantized checkpoint must record its `modules_to_not_convert` (or equivalent ignore list) so that the baseline INT4 and the mitigation differ only in the retained decoder layers. [Model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ)
- vLLM's start-up profiling assumes 1,280 image tokens per image unless `max_pixels` and `--limit-mm-per-prompt` are set; the pilot must record the actual image-token count per view under the frozen budget.
- The FlashOcc-specific checks from the September 8 audit (M0 config/checkpoint pairing, `dataset_type='val'` calibration default, TensorRT 8.4 plugin compatibility) no longer apply and are retained only in git history.
