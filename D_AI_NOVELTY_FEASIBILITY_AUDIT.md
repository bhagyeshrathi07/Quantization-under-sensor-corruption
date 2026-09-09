# D. AI Novelty & Feasibility Audit

**Project:** Quantization Under Sensor Corruption: Deployment and Robustness of Driving Perception and Vision-Language Models  
**Audit date:** September 8, 2026  
**Assessment basis:** The team's pre-audit project brief (not submitted), a targeted search of recent papers and official repositories, and the proposed 14-week experiment plan. Quoted "original claims" below refer to that pre-audit brief. This audit reports a literature-based assessment; no project experiments or performance measurements have been completed.

*Critique generated with Claude (Anthropic) on September 8, 2026, from the project brief, B_LITERATURE_SURVEY.md, and C_PROJECT_PROPOSAL.md, then verified claim-by-claim against primary sources on the same date; lightly edited for length.*

## Paste-ready AI critique summary

> **Verdict: a viable graduate systems/deployment project after narrowing, with substantial overlap with existing research.** The proposal should retain both required strands: one camera-based perception model executed in TensorRT and one vision-language model executed in vLLM. Quantized BEV robustness and quantized driving VLM evaluation are already studied, so neither strand should be described as a previously untouched topic. BRCB evaluates compressed BEV models under corruption; PTQAT includes quantized 3D perception and a TensorRT deployment; Mathew and Gu evaluate full-precision and 4-bit VLMs on DriveBench. These papers must be acknowledged as direct prior work. [BRCB](https://www.ijcai.org/proceedings/2025/766), [PTQAT](https://arxiv.org/abs/2508.10557), [Mathew and Gu](https://doi.org/10.1109/ICDMW69685.2025.00440)
>
> The proposed contribution is a controlled, reproducible study of a deployment decision: whether a precision chosen using clean overall quality continues to meet the project's predefined quality and runtime budgets under selected sensor corruptions and driving question types. The analysis should separate model conversion errors from quantization losses, compare the same examples, report uncertainty across driving scenes, and measure actual execution on the named hardware. One controlled mitigation should test whether retaining selected components at higher precision improves this tradeoff. This is a candidate empirical and engineering contribution, not a claim to have invented quantization or mixed precision.
>
> **The main red-ocean risk is producing another table of average accuracy and speed without explaining what the average hides.** The stronger result is an auditable explanation of which precision choices remain acceptable under the tested conditions, which fail, and what a mitigation costs. General VLM quantization and sensitivity-aware calibration are active research areas: MBQ and QIG already address unequal quantization sensitivity. [MBQ](https://arxiv.org/abs/2412.19509), [QIG official implementation](https://github.com/ucas-xiang/QIG)
>
> **Oversaturated areas to avoid as the headline claim:** (1) generic "quantize a VLM and report accuracy/latency" tables; (2) new sensitivity-aware PTQ calibration methods (MBQ, QIG, PTQAT already occupy this); (3) corruption-robustness benchmarking of BEV models by itself (RoboBEV, BRCB); (4) "first quantized driving VLM evaluation" (EM-VLM4AD, CVPR 2024 workshop; Mathew and Gu, ICDMW 2025). The defensible position is the paired, conversion-controlled, per-task and per-corruption deployment-budget analysis on named hardware.
>
> **Feasibility is conditional on early deployment tests.** A three-person team can pursue one model per strand in 14 weeks if the complete experiments fit the RTX 4090 and both actual runtimes work early. The additional architectures, Jetson, FP8, and W4A8 are extensions. An unsuccessful compatibility test must activate a smaller model or supported model fallback within the affected strand; it must not silently remove that strand. HPC access, a borrowed device, merged upstream contributions, and paper acceptance are not dependencies. A negative scientific finding can still be a successful project if both pipelines, the agreed experiments, and the analysis are completed.

## What is already known, and what this changes

| Prior work | Verified relevance | Consequence for this proposal |
|---|---|---|
| **Boost Embodied AI Models with Robust Compression Boundary (BRCB), IJCAI 2025** | Table 1 of the paper compares FP32 baselines with INT8 compressed BEVFormer, BEVFormer v2, and BEVFusion (NDS/mAP) on benign, rainy, snowy, and dark inputs; it develops a gated compression-tuning method for corruption robustness and received an IJCAI 2025 Distinguished Paper Award. [Official paper](https://www.ijcai.org/proceedings/2025/0766.pdf) | The original claim that no quantized BEV model has been evaluated under corruption is incorrect. BRCB uses its own rainy/snowy/dark sets rather than the standardized RoboBEV/nuScenes-C protocol, reports INT8 only, and gives no TensorRT latency, which leaves the proposed paired, protocol-standardized deployment study open. |
| **PTQAT: A Hybrid Parameter-Efficient Quantization Algorithm for 3D Perception Tasks, ICCV Workshops 2025** | Studies selective QAT combined with PTQ for detection, BEV segmentation, and occupancy. Its principal low-bit experiments are W4 weight-only; a separate BEVDepth4D experiment executes W8A8 in TensorRT. [Paper](https://arxiv.org/html/2508.10557v2) | Quantized perception deployment is established prior work. The project must not present W4 weight-only results as evidence that arbitrary perception models support executable W4A8. |
| **Can Small Quantized VLMs Drive? An Experimental Evaluation of Small Quantized VLMs for Autonomous Driving, ICDMW 2025** | Evaluates nine VLMs (SmolVLM, Qwen2.5-VL-3B/7B, Gemma-3, LLaVA-1.5) at FP16 and as BitsAndBytes 4-bit variants using vLLM on all 17 DriveBench conditions, scored by GPT-5-mini (verified against the published PDF). Its main answer-quality summaries average across driving tasks; it finds the 4-bit gap "not very large" and reports mixed latency effects. Presented in the ICDM 2025 Undergraduate and High School Symposium; cited as an empirical data point rather than a peer-reviewed main-track result. [Publication DOI](https://doi.org/10.1109/ICDMW69685.2025.00440) | The VLM strand is an extension of an existing study. Paired quality losses by task and condition are a narrower candidate contribution; “first question-type analysis” would be too broad. |
| **MBQ: Modality-Balanced Quantization for Large Vision-Language Models, CVPR 2025, pp. 4167–4177** | Uses differing vision/language sensitivity during calibration. Its repository includes pseudo-quantized evaluation; a specialized W3 kernel demonstration is distinct from general native W4A8 deployment. [Paper](https://arxiv.org/abs/2412.19509), [code](https://github.com/thu-nics/MBQ) | Sensitivity-aware mitigation is prior art. A component-preservation ablation can be useful engineering, but should not be marketed as a new quantization algorithm. |

**Tentative gap:** In the inspected studies, we did not establish a complete match to the proposed combination of paired quantization–corruption analysis, driving-task breakdowns, conversion controls, measured consumer-GPU execution, and a budgeted mitigation across both strands. This observation supports investigating the project; it does not prove priority. EM-VLM4AD (CVPR 2024 workshop, [arXiv:2403.19838](https://arxiv.org/abs/2403.19838)) also evaluated an 8-bit quantized driving VLM on DriveLM without corruption tests, so quantized driving VQA predates Mathew and Gu.

## Newer overlap and remaining uncertainty

- **QIG, CVPR 2026:** Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients advances token-level sensitivity estimation. Its official repository announces code release on April 3, 2026. A new sensitivity method is therefore outside the required contribution. [Official repository](https://github.com/ucas-xiang/QIG)
- **Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy, ICML 2026:** examines CLIP-style model reliability, including robustness to noise. The inspected arXiv revision is dated May 15, 2026. It is further evidence that quantization robustness beyond average accuracy is already researched; its setting differs from generative driving question answering. [Primary paper](https://arxiv.org/abs/2509.21173)
- **Performance and lightweight coexistence: Vision-language model for visual scene question answering in autonomous driving, April 2026:** the publisher abstract describes PeLiC-VLM and a quantized image-text adapter for driving VQA. Full-text access was blocked during this audit. Whether it contains the proposed paired corruption and question-type analysis is **unverified** and must remain an unresolved overlap item. [DOI](https://doi.org/10.1016/j.aei.2026.104424)

Searches covered exact paper titles, official-code availability, DriveBench plus quantization, driving quantization plus question type, and 2026 VLM quantization robustness. Sources included conference proceedings, arXiv, author-uploaded full text, publishers, and official GitHub repositories. This was a targeted audit, not an exhaustive systematic review. Refresh the search before freezing the final report; record new overlaps and adjust the contribution wording rather than claiming that search silence proves absence.

## Required scope and fallback rules

| Item | Required semester commitment | Contingency or extension |
|---|---|---|
| **Strand A: perception** | FlashOcc-M0 reference and INT8 execution in an actual TensorRT pipeline, with paired quality and runtime evaluation. | Required conditions: clean plus low light, fog, motion blur, color quantization, and camera crash at RoboBEV severity 2. If the deployment route cannot be made usable at the early gate, switch to BEVDet-R50 and verify its complete path. If that also fails, report the deployment requirement as unresolved; fake quantization alone does not fulfill it. |
| **Strand B: driving VLM** | Qwen2.5-VL-7B BF16 reference and one compatible INT4 configuration executed in vLLM, with driving-task breakdowns. | If the complete fixed workload cannot fit or execute reliably, test Qwen2.5-VL-3B with the same study design. FP8 is a desired extension after the required pair works. |
| **Compute** | One RTX 4090 with 24 GB VRAM; schedule the two strands on this shared GPU. | RTX 3060 Ti is for development and workloads that pass a memory test. HPC GPUs and Jetson are optional resources. |
| **Other models and formats** | No additional required architectures. | BEVFormer, StreamPETR, other occupancy models, Senna, DriveMM, InternVL, Jetson, and W4A8 follow only after both required strands are complete. |
| **Outputs** | Both executable pipelines, configuration manifests, experiment logs, analysis, documented limitations, report, and demo. | Upstream merges, paper acceptance, a confirmed published bug and fix, and a public submission community are stretch outcomes. |

These are implementation commitments and fallback decisions, not assertions that compatibility has already been demonstrated. Keep all quality tolerances, performance targets, and statistical decision rules consistent with the Project Proposal Document; this audit introduces no alternative thresholds.

## Feasibility risks and mitigation

| Risk | Assessment and response |
|---|---|
| **Learning cost** | Planning estimates: Strand A needs **3–4 weeks** of TensorRT/export/plugin and profiling ramp-up; Strand B needs **2–3 weeks** of serving/quantization/evaluation ramp-up; infrastructure/statistics needs **1–2 weeks**. These activities overlap across the three students. They are not measured completion guarantees. |
| **Runtime versus simulated quantization** | Record which tensors/operators actually execute at low precision. Fake-quantized accuracy results may diagnose numerical sensitivity, but must be labeled separately from actual TensorRT/vLLM performance. Unsupported low-bit operators are a feasibility failure, not a speedup. |
| **24 GB memory limit** | Test the entire workload, including images, activations, context, generated tokens, and runtime allocations. Freeze a workable input budget before the main evaluation. A lower-resolution or smaller-model result must be labeled as that workload, with both reference and quantized runs matched. |
| **One shared primary GPU** | Reserve GPU windows by strand. Run preprocessing, statistics, and artifact work on CPU where practical. Measure the pilot's actual runtime and project the full matrix, including calibration, compilation, retries, and grading. Original GPU-hour estimates remain unverified. |
| **Disk and network logistics** | Confirm exact required downloads and available space before starting. Prefer deterministic corruption generation and bounded caching over storing the entire expanded image matrix. Publish manifests and generation instructions; do not assume all source images or checkpoints can be redistributed. |
| **Evaluation expense and uncertainty** | Pilot the grading pipeline, record its cost, and retain raw answers. Use the same scoring procedure across precisions, check a human-reviewed sample, and disclose invalid/blank answers. Resample by driving scene when estimating uncertainty; many questions from one scene are not independent scenes. |
| **Temporal corruption validity** | Confirm that the chosen data contain the frames and sequence ordering required by any temporal fault. A black image, a dropped camera, and a missing temporal frame are different interventions and must be named accordingly. |
| **Mitigation overfitting** | Choose preserved components or calibration changes using held-out development data. Freeze the mitigation before the final test and compare its measured cost with the unmodified quantized model. |
| **Hardware extrapolation** | Results on the 4090 do not establish latency, energy, or deployability on Orin or another GPU. Report measurements only for devices actually tested. A consumer-GPU result is still valid evidence within its stated setting. |

## Prior-work reproducibility status

- **BRCB:** a dedicated official method implementation was not located. The [proceedings PDF](https://www.ijcai.org/proceedings/2025/0766.pdf) links only the target-model repositories (BEVFormer, BEVFusion, OpenVLA), which is not equivalent to releasing BRCB. Required as a literature comparison; not a mandatory reproduction.
- **PTQAT:** a dedicated official implementation was not located in the inspected paper and targeted repository search. The paper reports A100 fine-tuning. Required as prior work; reproducing its training is an optional addition. [Paper](https://arxiv.org/html/2508.10557v2)
- **Mathew and Gu:** an official experiment repository was not located. Discuss the study and implement the project's own documented paired comparison; do not imply an exact reproduction of undisclosed settings.
- **MBQ/QIG:** official implementations exist, but the documented supported model choices include Qwen2-VL rather than verified turnkey Qwen2.5-VL support. Their pseudo-quantized evaluation commands do not establish native deployment of the proposed configuration. Adaptation or a separate supported-model experiment would be extra work. [MBQ code](https://github.com/thu-nics/MBQ), [QIG code](https://github.com/ucas-xiang/QIG)

“Not located” describes this search result. It is not proof that code is unavailable everywhere.

## Decision gates

1. **End of week 2 — baseline gate:** run representative recorded inputs through both reference models; check answer/output validity, memory, image processing, data access, and pilot cost. Begin the stated model fallback within any blocked strand.
2. **End of week 4 — executable quantization gate:** obtain a reproducible actual-runtime quantized pilot for both strands. Freeze the viable model/runtime configuration. Do not replace an unresolved runtime with simulated speed claims.
3. **Before the main sweep — experiment gate:** freeze calibration/test separation, corruption definitions, input/output budgets, grading rubric, metric scales, statistical comparisons, and the proposal's success thresholds. Size the full matrix using the pilot.
4. **Before final reporting — evidence gate:** distinguish experiments completed from planned extensions; report uncertainty and failed configurations. If one required strand remains incomplete, state this explicitly and discuss the remaining scope with the instructor.

## Claims to avoid

| Unsupported claim | Evidence-respecting replacement |
|---|---|
| “Both ideas are verified blue ocean.” | “The proposal overlaps with established work; its narrower empirical contribution remains a research hypothesis.” |
| “INT4 will fit the 8 GB GPU.” | “Fit will be tested for the complete specified workload; the 4090 is the required execution platform.” |
| “Quantization makes the model faster.” | “We will measure speed and memory for each executable configuration.” |
| “No significant INT8 interaction means safe to deploy.” | “We will assess whether uncertainty bounds satisfy the predefined degradation tolerance on the tested conditions.” |
| “The mitigation is a novel quantization algorithm.” | “The study evaluates the quality/runtime tradeoff of a controlled mitigation informed by existing methods.” |

Topic-selection considerations outside the literature and the experiment plan were not audited here and are not evidence for scientific novelty or measured feasibility.

## Source-code checks that affect the protocol

A source review found several details that must be addressed in the implementation. These are observed defaults and interfaces, not claims that a published result is erroneous.

- FlashOcc's README links both the M0 and M1 table rows to the M1 config file; only `doc/mmdeploy_test.md` pairs `flashocc-r50-M0-trt.py` with the M0 checkpoint. Pin the M0 configuration and weight together from that file. The pinned stack (torch 1.10.0+cu111, TensorRT 8.4.0.6, CUDA 11.6) predates the RTX 4090. [TensorRT test commands](https://github.com/Yzichen/FlashOCC/blob/master/doc/mmdeploy_test.md), [pinned environment](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md)
- The upstream TensorRT converter passes `dataset_type='val'` at its calibration call site. Our project must override that default and use separate training scenes to preserve a held-out final test. [Conversion script](https://github.com/Yzichen/FlashOCC/blob/master/tools/convert_bevdet_to_TRT.py)
- DriveBench preprocessing selects camera views referenced by the question. Preserve that mapping and record the actual view count. The released NoImage adapter supplies blank images while retaining image placeholders; removing image tokens is a different intervention. [Preprocessor](https://github.com/worldbench/DriveBench/blob/main/tools/preprocess.py), [released adapter](https://github.com/worldbench/DriveBench/blob/main/inference/llava1.5.py)
- DriveBench's released evaluator routes five cells (perception MCQ and VQA, prediction VQA, planning VQA, behavior MCQ), silently drops prediction MCQ, and raises on unknown tags; the clean prediction-VQA cell has only 61 questions. Publish included/skipped question IDs and the tag-to-score mapping. Captioning in the separate corruption-awareness task is outside the required main driving QA comparison. Unsupported or empty cells must be reported as unassessed or not applicable, never converted into accuracy scores. [Evaluator](https://github.com/worldbench/DriveBench/blob/main/evaluate/eval.py), [paper supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Xie_Are_VLMs_Ready_ICCV_2025_supplemental.pdf)
