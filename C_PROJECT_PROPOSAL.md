# C. Project Proposal

## Quantization Under Sensor Corruption: Deployment and Robustness of Driving Perception and Vision-Language Models

**Institution:** San Jose State University  
**Team:** Ronak, Bhagyesh, and Akshay  
**Duration:** 14 weeks  
**Track:** System / Deployment (supported by experimental research)  
**Prepared:** September 8, 2026  
**Scope:** Both Strand A (perception) and Strand B (vision-language reasoning) are required. Additional models and hardware are extensions.  
**Status:** Proposed experiments; no results or hardware measurements are claimed in this document.

## Abstract

Quantization reduces the precision of the numbers used inside a neural network, potentially reducing memory use and inference time. A small accuracy loss on clear images may conceal larger losses in fog, darkness, or particular driving questions. This project will build and evaluate two deployed inference paths: a camera-based occupancy model in TensorRT and a driving-scene vision-language model in vLLM. Using recorded driving data, we will compare each quantized model with the same model at higher precision, keeping inputs and evaluation settings fixed. We will measure additional accuracy loss under sensor corruption, changes in answers across driving tasks, latency, memory, and GPU energy. We will also test one targeted mitigation in each strand. The contribution is a reproducible, hardware-measured study of deployment tradeoffs and failure patterns, extending existing quantized-perception and driving-VLM research. Both strands will run on one RTX 4090 with 24 GB VRAM; an RTX 3060 Ti is supplementary, and neither HPC access nor a Jetson is required.

## 1. Problem formulation

### Domain problem

An engineer selecting a lower-precision model needs to know more than its average score on clean data. The model must remain useful when camera inputs degrade and when questions require different kinds of reasoning. Prior research already studies quantized BEV robustness and quantized driving VLMs; the unanswered project question is whether a particular precision choice meets explicit quality and runtime budgets across our tested conditions. We do not claim to introduce quantization for autonomous driving. [BRCB](https://www.ijcai.org/proceedings/2025/766), [quantized driving-VLM study](https://doi.org/10.1109/ICDMW69685.2025.00440).

**Research question:** Does selecting model precision using clean aggregate quality conceal larger precision-related losses under sensor corruption, or within particular driving tasks and answer formats?

**Primary hypothesis:** For at least one tested lower-precision configuration, the loss relative to its matching higher-precision reference will be larger under a predefined corruption or driving-task group than the loss suggested by the clean overall result. This is a hypothesis to test, not an expected finding presented as fact.

**Secondary hypothesis:** Preserving a small set of sensitive operations at higher precision can reduce the largest measured loss while retaining part of the deployment benefit. This is an evaluation of an established mitigation principle, not a claim of a new quantization algorithm. [MBQ](https://arxiv.org/abs/2412.19509).

### Inputs and outputs

| Strand | Model input | Model output | Study output |
|---|---|---|---|
| A: perception | Six synchronized camera images, camera calibration, and model-required metadata for a recorded sample | A 3D occupancy grid with semantic classes; 3D detection boxes if the BEVDet fallback is activated | Task quality, conversion and quantization losses, latency, memory, energy, and per-component sensitivity |
| B: VLM | The benchmark-selected camera views in their recorded order, a driving question, and a fixed prompt | A multiple-choice or open-ended answer | Scores by driving task and answer format, harmful/beneficial answer changes, latency, throughput, memory, and energy |

Ground-truth occupancy labels and reference answers are evaluation inputs, not inputs supplied to the model when answering. Corruptions modify the camera images while retaining the sample/question pairing and ground truth. Results from the two strands remain in separate metric tables; mIoU and answer scores will not be averaged together.

## 2. Required scope and selected models

**Strand A: FlashOcc-M0.** Use the released single-frame, ResNet-50 configuration at 256 × 704 input resolution as the primary occupancy baseline. The intended identities are `projects/configs/flashocc/flashocc-r50-M0.py`, its `flashocc-r50-M0-trt.py` deployment configuration, and `ckpts/flashocc-r50-M0-256x704.pth`; keep the single-frame M0 distinct from the multi-frame M1 configuration in the same repository. Note that the upstream README links both the M0 and M1 table rows to the M1 config file; the correct M0 config/checkpoint pairing is taken from `doc/mmdeploy_test.md`. The M0 weight is distributed as an unnamed Google Drive file; record its Drive id, SHA-256 checksum, and source revision after acquisition. Its architecture was selected for a practical deployment experiment; it is not described as a foundation model or the newest occupancy architecture. If its runtime cannot be made reproducible within the initial gate, use BEVDet-R50 as a detection-based contingency and replace occupancy metrics with NDS/mAP. Strand A remains required, and its input/output documentation, evaluator, and declared quality targets must reflect that change. [M0 configuration](https://github.com/Yzichen/FlashOCC/blob/master/projects/configs/flashocc/flashocc-r50-M0.py), [TensorRT test commands](https://github.com/Yzichen/FlashOCC/blob/master/doc/mmdeploy_test.md), [pinned environment](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md), [BEVDet repository](https://github.com/HuangJunJie2017/BEVDet).

**Strand B: Qwen2.5-VL-7B-Instruct.** Use this public, ViT-based VLM as the reproducibility baseline. A BF16 reference and one INT4 deployment are required. FP8 is an optional extension once the required pair works. Qwen3-VL is newer and is covered in the survey; Qwen2.5-VL is not claimed to be current SOTA. If the full BF16 workload (≈16.6 GB of bf16 weights plus activations and KV cache) does not fit 24 GB or does not execute reliably in the pinned vLLM after setting a fixed image-token budget, use Qwen2.5-VL-3B-Instruct for every precision comparison in the main experiment. Reducing model size preserves a complete reference/quantized comparison. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [vLLM model support](https://docs.vllm.ai/en/stable/models/supported_models/), [Qwen3-VL report](https://arxiv.org/abs/2511.21631).

| Required experiment | Reference | Quantized configuration | Runtime |
|---|---|---|---|
| A: one perception model | TensorRT FP16; matched-precision PyTorch/TensorRT conversion control and PyTorch FP32 diagnostic | INT8 with any retained FP16 operations explicitly listed | TensorRT |
| B: one VLM | BF16, with the same processor and prompt | One INT4 method: AWQ preferred, GPTQ contingency if supported | vLLM |

The exact checkpoint revision, quantization settings, package versions, and container digests will be pinned after smoke testing in weeks 1–2. They are not fabricated at proposal time. Public prequantized weights may be used for setup, but the final comparison must document that they derive from the same base checkpoint. Record which components are quantized; an INT4 language decoder with a BF16 vision encoder is a mixed-precision model.

**Extensions:** FP8 on the 4090, a second model per strand, a modern ViT perception model, additional quantizers, a broader corruption sweep, and Jetson measurements. W4A8, NF4, custom CUDA kernels, QAT, and specialist VLM ports are not prerequisites for completing both strands.

## 3. Data and experiment matrix

**Perception data:** nuScenes camera data and metadata plus the occupancy ground truth required by FlashOcc/Occ3D. The upstream layout lists `samples`, `sweeps`, and a separate `gts` folder downloaded from the CVPR 2023 occupancy challenge; camera keyframes alone are not a complete occupancy dataset. Whether M0's single-frame loader actually reads `sweeps` is checked in week 1 before deciding whether that download can be omitted. Use the official validation split for final evaluation. Before experiments, publish separate, nonoverlapping training-scene lists for calibration and development; development scenes are used for all configuration and mitigation choices. [FlashOcc data instructions](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md), [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D).

**VLM data:** DriveBench is the primary evaluation benchmark. Required driving-task cells are perception MCQ/VQA, prediction VQA, planning VQA, and behavior MCQ, subject to validation against the pinned annotations and scorer. Caption questions belong to the separate corruption-awareness data and are optional; they are not part of the required driving matrix. The released preprocessing extracts camera names from `<c1,CAM_BACK,x,y>` references in the question and keeps only those views, retaining all views otherwise. Preserve each question's selected subset and order, which may contain one to six views, identically across precision/condition pairs. [DriveBench task tables](https://arxiv.org/html/2501.04003v1#A1), [preprocessing code](https://github.com/worldbench/DriveBench/blob/main/tools/preprocess.py), [evaluation code](https://github.com/worldbench/DriveBench/blob/main/evaluate/eval.py).

Use compatible DriveLM training-scene examples as B's calibration and development source where available, removing any scene that overlaps DriveBench final evaluation and dividing the remaining scenes into disjoint calibration and development lists. If compatible external examples or reliable scene mappings are unavailable, reserve disjoint calibration and development scenes from DriveBench before any experiments, exclude both lists from final evaluation, and publish the resulting reduced benchmark protocol. All prompt, token-budget, runtime-budget, and mitigation pilots use development scenes. DriveLM itself is an optional additional final evaluation, not a requirement. [DriveLM](https://github.com/OpenDriveLab/DriveLM).

**Required conditions for A:** clean plus five RoboBEV corruption types at their middle severity (2 of 3): low light (RoboBEV `Low Light`; DriveBench `Dark`), fog, motion blur, color quantization, and camera crash. Camera crash tests missing camera input; FlashOcc-M0 is single-frame, so this study does not claim to assess temporal recovery. Image color quantization is a data corruption and must be labeled separately from model quantization. The complete eight-corruption, three-severity suite is an extension. Camera Crash and Frame Lost are already part of those eight. [RoboBEV](https://github.com/worldbench/RoboBEV).

**Required conditions for B:** clean plus the five corresponding released DriveBench conditions (Dark, Fog, Motion Blur, Color Quantization, Camera Crash), and its NoImage control. Use the published corruption realization rather than silently introducing a new severity setting. The inspected released adapter implements NoImage by substituting a 224×224 all-zero (black) image per view while still appending one `<image>` placeholder per view; reproduce that intervention consistently with the selected Qwen processor and record the resulting image-token counts. A prompt with no image tokens is a separate optional intervention. Compare precision pairs under the same control implementation; differences between controls must not be used to claim a quantization speedup. [Released NoImage adapter](https://github.com/worldbench/DriveBench/blob/main/inference/llava1.5.py).

The minimum matrix is 12 A quality runs (2 precisions × 6 conditions) and 14 B quality runs (2 precisions × 7 conditions), before mitigation and profiling repetitions. This is a deliberately selected subset, not a claim to reproduce the entire robustness leaderboard. Add mitigation variants for the same conditions only after freezing the development choices.

Use the full eligible final-evaluation split where practical. Measure throughput on development scenes, then freeze any necessary final scene-stratified subset before inspecting final answers or quality scores. Publish its sample IDs, scene counts, question counts per eligible task/format cell, and selection rule. Audit the pinned scorer's eligibility filters: the released scorer routes five cells (perception MCQ and VQA, prediction VQA, planning VQA, behavior MCQ), silently drops prediction MCQ (`pass`), raises on unknown tags, and only runs GPT scoring with `--eval-gpt`. The clean prediction-VQA cell has only 61 questions, so after scene holdout it may fall below the declared coverage minimum and be reported as unassessed. Publish eligible, excluded, skipped, and failed counts with reasons; mark structurally unsupported cells N/A. Do not silently drop inference failures from eligible denominators. Do not compare a subset score with a published full-benchmark score as if the protocols matched. Final evaluation scenes must not enter calibration, mitigation selection, or prompt tuning. [DriveBench scorer](https://github.com/worldbench/DriveBench/blob/main/evaluate/eval.py).

**Storage and access:** nuScenes requires registration. Measure download and extracted sizes in week 1; the ~45 GB keyframe-only figure is not a full-project storage budget. Plan a 250 GB working allowance as an estimate and a 500 GB ceiling; if the required assets exceed it, use a documented validation subset or revise the storage plan before downloading. Generate/cache one corruption condition at a time and remove only reproducible project caches under a documented retention policy. Dataset approval delays, image redistribution rights, checkpoint terms, and judge-service costs must be checked before acquisition or publication. Publish manifests and derived results rather than assuming all underlying images may be redistributed. [nuScenes access page](https://www.nuscenes.org/nuscenes).

## 4. Proposed technical approach

The system has two model adapters and a shared experiment layer. A sample manifest identifies each scene and input; deterministic corruption adapters produce paired conditions; a runtime adapter executes the selected precision; evaluators record task quality and operational measurements. Analysis code then creates separate perception and VLM tables and a shared failure explorer.

```text
Recorded data + immutable sample/scene manifest
                      |
          clean / corruption / control inputs
                      |
       +--------------+----------------+
       |                               |
 A: perception adapter             B: VLM adapter
 export -> INT8 recipe             BF16 -> INT4 recipe
 TensorRT FP16 / INT8              vLLM BF16 / INT4
       |                               |
 occupancy/detection evaluator    QA + answer-change evaluator
       +--------------+----------------+
                      |
        timings / memory / energy / run metadata
                      |
        paired analysis -> tables + failure explorer
```

### A. Perception deployment

Reproduce the checkpoint's reference outputs, export through the supported path, and build an FP16 engine. Use a matched-precision control, such as PyTorch FP16 versus TensorRT FP16 or FP32 on both, to measure conversion/runtime differences; retain PyTorch FP32 as a diagnostic reference. Comparing PyTorch FP32 directly with TensorRT FP16 combines conversion/runtime and precision effects and must be labeled accordingly if a matched control cannot execute. The primary quantization comparison is TensorRT FP16 versus TensorRT INT8 on the same examples.

Next, calibrate INT8 using the declared training-scene calibration manifest. The inspected upstream converter hard-codes `dataset_type='val'` at its `create_calib_input_data` call site and builds `calib_data.h5` from that split; explicitly override that selection and verify logged calibration scene IDs before conversion. The repository-supported TensorRT calibration path is the primary INT8 route, because FlashOcc's custom `bev_pool_v2` plugin op is not recognized by NVIDIA ModelOpt's ONNX quantizer; ModelOpt is used only if it can skip that op cleanly. Record the recipe actually executed, including retained FP16 layers and custom plugins. The required artifact is a real INT8 TensorRT path; a ModelOpt-only integration is not assumed to be turnkey. [Converter source](https://github.com/Yzichen/FlashOCC/blob/master/tools/convert_bevdet_to_TRT.py).

Use separate, pinned environments for legacy model/export dependencies and modern quantization/runtime dependencies where necessary. FlashOcc pins Python 3.8.5, torch 1.10.0+cu111, mmcv-full 1.5.3, mmdet3d v1.0.0rc4, and TensorRT 8.4.0.6 with CUDA 11.6, a stack that predates the RTX 4090 (Ada support arrived in TensorRT 8.5), so the `bev_pool_v2` plugin must be rebuilt against a newer TensorRT. ModelOpt deprecated CUDA 11.x and torch 2.0 in 0.23.0 (January 2025) and requires torch ≥2.8 as of 0.44.0 (May 2026), so the two cannot share an environment. A successful container build, plugin build, and FP16 engine on the 4090 are therefore the explicit week-2 gate. Upstream reference numbers on an RTX 3090: PyTorch FP32 31.95 mIoU → TensorRT FP32/FP16 30.78 → TensorRT INT8 29.59; engine throughput 197.6 Hz FP16 → 397.0 Hz INT8. [FlashOcc installation](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md), [ModelOpt changelog](https://github.com/NVIDIA/Model-Optimizer/blob/main/CHANGELOG.rst).

For sensitivity analysis, restore selected supported modules or operations to FP16 one group at a time on development data, then select one small mixed-precision configuration. Verify that its actual engine reproduces its numerical behavior before benchmarking it. Fake-quantized PyTorch results may diagnose sensitivity but cannot satisfy the runtime requirement.

### B. VLM deployment

Build a vLLM adapter that preserves each question's selected camera-view subset and order, image references, question text, and response-format instructions. Start with batch size 1, concurrency 1, a 4,096-token context cap, an image-token cap targeting at most 256 tokens per view (`max_pixels = 200704`, about 252 tokens for a 1600×900 frame, a 2.7× per-axis downscale; a 512-token setting will be piloted because 252 tokens is coarse for driving imagery), `--limit-mm-per-prompt '{"image":6}'`, `--max-model-len 4096`, and a 128-token answer limit. Without these flags vLLM's start-up profiling assumes the default 1,280 tokens per image and can exhaust 24 GB. Validate these settings on the chosen processor and use the same settings for every precision. Record truncation and invalid answers; revise the common limits using development data if necessary before the final run. Qwen supports controlling image resolution to trade computation and memory against visual detail. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

Use one supported INT4 recipe, keeping calibration examples separate from final scenes. The same BF16 reference is run in vLLM; a small Transformers comparison checks adapter correctness. FP8 is an additional configuration only after the primary pair works. Hardware support for a quantization family does not establish support for every model/operator combination. [vLLM quantization support](https://docs.vllm.ai/en/stable/features/quantization/).

For mitigation, preserve one selected decoder block group at BF16 using a supported exclusion mechanism and evaluate the tradeoff. If the selected runtime cannot execute that configuration, use a fixed-size, balanced calibration set as the controlled mitigation instead. Freeze the choice on development scenes. An unchanged vision encoder must not be credited as a new mitigation when it was already high precision in the baseline.

### Shared advanced capabilities

Every run will save a configuration hash, source/model revisions, scene and question IDs, random seeds, precision coverage, device identity, elapsed time, error records, and raw predictions. Runs will resume from completed samples and validate schema compatibility. A public result explorer will filter by model, condition, task, and response format and show paired examples of regressions and improvements. It is an evaluation tool for researchers and developers, not a vehicle-control application.

## 5. Metrics and success criteria

### Quality and interaction

For a higher-is-better score S, define quantization loss at condition c as L(c) = S(reference, c) − S(quantized, c). Define additional corruption loss as I(c) = L(c) − L(clean). Positive I means the quantization penalty is greater under corruption. Within a VLM task group g and a single response format, use I_g(c) = L_g(c) − L_g(clean).

Separately, test whether clean overall results conceal a task-group loss using H_g = L_g(clean) − L_all(clean). Compute L_all only within the same response format and scoring scale, using empirical task weights from the frozen, common eligible question set. Keep the question set and empirical task weights fixed across precisions and conditions. Retain those weights in bootstrap replicates while recomputing subgroup scores from resampled scenes. Positive H_g indicates a larger clean loss in that group than in the corresponding overall result. Do not compare MCQ accuracy with VQA rubric scores or mix incompatible scales.

For example, if clean scores are 90 and 89, but fog scores are 70 and 60, then L(clean) = 1, L(fog) = 10, and I(fog) = 9 points. These numbers illustrate the calculation and are not experimental results.

For A, primary quality is mIoU under a fixed occupancy mask/evaluation convention; per-class IoU is diagnostic, and RayIoU is secondary if the selected model/evaluator supports it. If the documented BEVDet contingency is used, NDS and mAP replace occupancy metrics; express both on a 0–100 scale and apply the fallback tolerances below. State the scope change before final testing.

For B, multiple-choice accuracy is the primary objective metric for eligible MCQ cells. Required open-ended VQA cells retain the benchmark-compatible rubric score, reported separately; rescaling a score to 0–100 does not turn it into percentage accuracy. Optional caption results use a separate corruption-awareness table. Use a fixed evaluator version and prompt. Independently double-review a balanced sample of 100 answer pairs, including apparent regressions, and report agreement and disagreements; this sample audits grading and does not automatically establish every cell's accuracy. If judge access is unavailable, publish automated MCQ results plus the separately identified human-scored sample, and mark any insufficiently scored required VQA cell unassessed. Such a cell cannot receive a quality pass or count as completed required evaluation. Recover adequate scoring coverage or explicitly report the outstanding requirement; never silently substitute a different metric.

Report semantic answer changes, original-correct to quantized-incorrect regressions, and the reverse improvements. For MCQ, normalize answer choices. For free text, paraphrases alone do not count as semantic flips. Separate these outcomes from rubric-score changes. Use the same questions and decoding settings, with repeat runs on a small fixed slice to estimate runtime/decoding instability.

Use a paired bootstrap over scenes, preserving all relevant samples, questions, and precision/condition pairs within a resampled scene. For occupancy, aggregate per-scene confusion matrices and recompute mIoU rather than pretending individual voxels are independent. Report ordinary 95% intervals as descriptive uncertainty. Separately freeze a confirmatory family for each strand, including its required loss-tolerance comparisons and selected interaction/group contrasts. If that family has K intervals, construct Bonferroni-adjusted two-sided intervals with error probability 0.05/K per interval (0.025/K in each tail), giving a nominal simultaneous 95% family confidence level. Use enough bootstrap repetitions to resolve those tails; document that bootstrap coverage is approximate. Do not shrink the family after seeing results. Additional subgroup results are exploratory.

Use the adjusted interval to classify a quality loss: **meets tolerance** if its upper bound is at or below the tolerance; **exceeds tolerance** if its lower bound is above it; otherwise **inconclusive**. An interaction or group contrast supports a positive effect only when its adjusted lower bound exceeds zero. Before final testing, freeze per-cell minimum independent-scene coverage and scoring-completeness requirements using development data. Unsupported task/format cells are N/A; eligible cells lacking the declared coverage or reliable interval estimation are **unassessed**, not passes. This distinguishes inadequate evidence from an adequately measured result that misses a deployment budget.

### Proposed deployment budgets

These are project targets, not achieved results or industry safety standards. Freeze their exact interpretation after the setup pilot and before inspecting final-test results; any revision must be dated and justified.

| Criterion | A: perception target | B: VLM target |
|---|---|---|
| Local execution | Complete evaluation on one 24 GB RTX 4090 | Complete evaluation on one 24 GB RTX 4090 |
| Runtime improvement | At least 1.25× speedup over FP16 on engine plus postprocessing, with preprocessing and end-to-end reported separately (upstream engine-only speedup is 2.0×, but INT8 saves only ≈2.5 ms per sample, so an end-to-end 1.25× requires non-engine overhead ≤≈7.6 ms) | At least 25% lower peak GPU memory than BF16 under equal cache/workload settings |
| Response budget | p95 latency ≤20 ms per six-camera sample on engine plus postprocessing; end-to-end reported separately | p95 time to first token ≤2 s; median generation throughput ≥15 tokens/s |
| Quality tolerance | Adjusted upper bound on mIoU loss ≤2 points clean and ≤3 points for each required corruption; if BEVDet is used, the same limits apply separately to NDS and mAP on 0–100 scales | Adjusted upper bound on MCQ loss ≤3 percentage points and on VQA rubric-score loss ≤5 points on a stated 0–100 scale, for every respective required eligible task/condition cell |
| Transparency | Report failed quality/runtime budgets | Report failed quality/runtime budgets |

Memory measurements must count total GPU use as well as model allocations. For vLLM, set the same explicit absolute KV-cache byte capacity (`--kv-cache-memory-bytes`), KV-cache dtype, maximum context, workload, and concurrency across BF16 and INT4; record actual allocations and the configuration mechanism supported by the pinned runtime. Do not use an equal fraction-of-GPU reservation as a substitute for equal cache capacity. Separate reserved memory, model allocations, and total observed peak GPU use; the reserved pool alone is not model memory savings. Publish p50/p95/p99 latency, even where the target uses p95, along with sample counts and repetitions. Report uncertainty for tail estimates.

**Scientific completion is distinct from hitting every deployment budget.** Both actual-runtime model pairs, every required eligible evaluation cell with its declared scoring and scene coverage, raw logs, uncertainty analysis, mitigation comparisons, and the reproducibility package must be completed. A sufficiently scored result may be inconclusive about the tolerance or establish that a deployment exceeds it; either is valid research evidence. Missing runs, unsupported required execution, and insufficiently scored cells are incomplete work rather than negative findings. Report them explicitly instead of counting them as completed. A nonsignificant difference alone is not evidence of equivalence or driving safety.

### Runtime and energy protocol

Measure perception engine time and end-to-end time separately. Define end-to-end as decoded host images through preprocessing, transfers, model execution, and postprocessing; report disk decoding separately. For VLMs, measure client-observed time to first token and total answer time, plus server-side prefill/generation metrics. Use representative inputs, warm-up, at least three independent timing sessions, and a fixed workload. Keep detailed profiler collection separate from headline timing runs and archive representative Nsight traces.

Record power limits, clocks, temperatures, background GPU use, and software versions. Integrate supported NVML power/energy measurements over sustained repeated workloads and divide by completed samples or answers. Report GPU-board joules per sample, with gross and idle-adjusted definitions stated separately. This is not whole-machine energy. If a Jetson becomes available, use its documented rail telemetry and state the different measurement boundary. [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html), [Jetson telemetry](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).

## 6. Feasibility, ownership, and timeline

The primary resource is a team-owned workstation with one RTX 4090 (24 GB), with a second machine holding an RTX 3060 Ti (8 GB) for development and only measured workloads that fit. Host CPU, RAM, free NVMe, driver and CUDA versions, OS, and access mode for both machines will be recorded in the week-1 run manifest. SJSU HPC and Jetson access are optional accelerants. GPU availability, free disk, host RAM, and driver compatibility must be checked in week 1; the proposal does not assert that these checks have passed.

Plan around 8–10 hours per member per week, or 336–420 person-hours across the team, as a scheduling assumption. Estimated ramp-up is 3–4 weeks for A's export/runtime stack, 2–3 weeks for B's serving/quantization stack, and 1–2 weeks for shared measurement infrastructure, overlapping with implementation. Replace prior GPU-hour guesses with measured pilot throughput and remaining sample counts. Reserve separate time slots on the shared 4090.

| Weeks | Required milestone |
|---|---|
| 1–2 | Freeze data manifests and environments; reproduce A reference outputs and B BF16 answers; measure memory and pilot throughput |
| 3–4 | Execute real A FP16/INT8 engines and B BF16/INT4 serving; complete adapter and conversion checks |
| 5–6 | Finish both clean/corrupted pilots; finalize scoring, scene grouping, and development-only mitigation selection |
| 7–9 | Run both frozen experiment matrices; profile representative runs; collect sustained energy measurements |
| 10–11 | Complete mitigation comparisons, uncertainty estimates, failure analysis, and public result explorer |
| 12–14 | Reproduce representative runs from a clean setup; finish GitHub documentation, final report, presentation, and demo video |

**Ronak (Strand A):** perception export, quantization, TensorRT integration, component sensitivity, and profiling. **Akshay (Strand B):** VLM adapter, vLLM serving, quantization, answer scoring, and task/format analysis. **Bhagyesh (infrastructure and maintainer):** manifests, corruption adapters, experiment logging, statistics, energy measurement, Docker orchestration, and result explorer. All members review experiment design and writing. Ronak and Akshay produce their own raw runtime logs; Bhagyesh defines shared schemas and analysis to avoid becoming the sole integration bottleneck.

**Failure gates:** If an initial checkpoint/runtime pair fails the week-2 feasibility check, use its documented contingency immediately. If either strand still lacks an actual lower-precision runtime by the end of week 4, reassign effort and revise the model/precision implementation while retaining both strands; request advisor review of the concrete blocked component. Do not relabel fake quantization as completed deployment. Extra models, FP8, and hardware comparisons stop until both minimum paths work.

## 7. Expected deliverables and academic contribution

Required deliverables are the ten-paper literature survey; this proposal; the AI novelty/feasibility audit; a public codebase with reproducible setup and per-strand commands; raw experiment logs and sample manifests; paired analysis tables and representative traces; a result explorer with failure examples; and the final report, presentation, and demo video. Model/data downloads may require their own registration, so one-command execution begins after documented asset acquisition.

The expected contribution is evidence about when a clean-data precision choice does or does not remain acceptable under the selected corruptions and driving questions, with conversion errors separated from quantization effects and measured hardware costs. Model-specific findings will not be generalized to all driving models. Both strands share methodology and infrastructure while retaining their distinct outputs and metrics.

An upstream issue or pull request to a dependency, a resolved reproduction discrepancy, a Jetson row, and a workshop submission are extensions. External acceptance, a discovered bug, a positive result, and real-world driving safety are not promised outcomes. GPU measurements without Jetson will be described as consumer-GPU deployment measurements relevant to resource constraints, not proof of onboard edge performance.

## 8. Model and software stewardship

**Available device and maintainer.** The development and measurement machine is a team-owned workstation with one NVIDIA RTX 4090 (24 GB); a second machine with an RTX 3060 Ti (8 GB) supports development. Bhagyesh is the repository maintainer and the point of contact for the codebase, CI/Docker, and any Claude Code or agent-assisted tooling used during the project; Ronak owns Strand A and Akshay owns Strand B. Upstream issue trackers are the contact points for the third-party components: [FlashOcc](https://github.com/Yzichen/FlashOCC/issues), [Qwen](https://github.com/QwenLM/Qwen3-VL/issues) (the former `QwenLM/Qwen2.5-VL` repository, renamed upstream), [DriveBench](https://github.com/worldbench/DriveBench/issues), [RoboBEV](https://github.com/worldbench/RoboBEV/issues), [ModelOpt](https://github.com/NVIDIA/Model-Optimizer/issues), and [vLLM](https://github.com/vllm-project/vllm/issues). Actual revisions, license observations, device details, and successful smoke-test commands will be recorded in a machine-readable run manifest.

The companion files B_LITERATURE_SURVEY.md and D_AI_NOVELTY_FEASIBILITY_AUDIT.md provide the literature comparison and evidence behind the scope. Course number and instructor attribution are added at submission; no institutional approval is implied.
