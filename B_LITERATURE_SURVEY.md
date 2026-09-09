# B. Literature and SOTA Survey Results

**Project:** Quantization Under Sensor Corruption: Deployment and Robustness of Driving Perception and Vision-Language Models  
**Survey date:** September 8, 2026  
**Track:** System / Deployment (supported by experimental research)  
**Required strands:** A — camera-based occupancy perception; B — driving vision-language question answering.

This survey covers ten papers (plus one earlier quantized-driving-VLM citation and three 2026 context items) published or first made public between September 8, 2023 and September 8, 2026. Dates distinguish first preprints from later conference or journal publication. Technical reports are identified as such. Sources were checked through primary papers, official proceedings, and project repositories; the Mathew–Gu study is available through an author-uploaded manuscript and the ICDMW 2025 proceedings record (IEEE Xplore, paywalled). Published results have not been independently reproduced by this team. Statements labeled **project assessment** are our interpretation.

The proposal requires both strands. **FlashOcc-M0** and **Qwen2.5-VL-7B-Instruct** are the primary planned models, with model contingencies defined in [the project proposal](C_PROJECT_PROPOSAL.md). They are selected for a reproducible, bounded deployment study; neither is described as the overall accuracy leader in September 2026. The survey includes newer model context without making additional models mandatory.

## Comparison of the ten selected works

| # | Paper / model | Date and venue | Main contribution | Role in our project |
|---|---|---|---|---|
| 1 | [FlashOcc](https://arxiv.org/abs/2311.12058) | Nov. 18, 2023; preprint | Efficient occupancy architecture | Primary Strand A baseline |
| 2 | [Expanded RoboBEV study](https://arxiv.org/abs/2405.17426) | May 27, 2024; TPAMI vol. 47 no. 5, May 2025 | Perception corruption benchmark and analysis | Strand A corruption protocol |
| 3 | [BRCB](https://www.ijcai.org/proceedings/2025/766) | Aug. 2025; IJCAI 2025 Distinguished Paper | Robust compression of embodied models | Direct Strand A prior work |
| 4 | [PTQAT](https://arxiv.org/abs/2508.10557) | Aug. 14, 2025; ICCV 2025 BiVision Workshop | Selective quantization-aware tuning for 3D perception | Quantization method and feasibility context |
| 5 | [DriveLM](https://arxiv.org/abs/2312.14150) | Dec. 21, 2023; ECCV 2024 Oral | Graph-structured driving question answering | Driving task structure and secondary data source |
| 6 | [DriveBench](https://arxiv.org/abs/2501.04003) | Jan. 7, 2025; ICCV 2025 | VLM reliability under corruptions and missing images | Required Strand B evaluation framework |
| 7 | [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) | Feb. 19, 2025; technical report | Dynamic-resolution vision-language foundation models | Primary Strand B baseline |
| 8 | [Qwen3-VL](https://arxiv.org/abs/2511.21631) | Nov. 26, 2025; technical report | Updated visual features and multimodal reasoning | Newer model context; optional extension |
| 9 | [Can Small Quantized VLMs Drive?](https://doi.org/10.1109/ICDMW69685.2025.00440) | Nov. 12–15, 2025; ICDMW 2025 (ICDM Undergraduate and High School Symposium) | Quantized VLM evaluation on DriveBench | Direct Strand B prior work |
| 10 | [MBQ](https://arxiv.org/abs/2412.19509) | Dec. 27, 2024; CVPR 2025, pp. 4167–4177 | Calibration that accounts for modality sensitivity | Sensitivity analysis and mitigation context |

## 1. FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin

**Authors / date:** Zichen Yu et al.; first preprint November 18, 2023. A later conference venue was not verified. [Paper](https://arxiv.org/abs/2311.12058)

FlashOcc reduces the computation required to predict which parts of a three-dimensional scene are occupied. It retains intermediate features in bird's-eye view, uses two-dimensional convolutions, and transforms channels into height at the output. The paper evaluates its accuracy, runtime, and memory tradeoffs on Occ3D-nuScenes. This makes it relevant to deployment experiments that must measure both output quality and actual execution cost. [Paper](https://arxiv.org/abs/2311.12058)

The official repository supplies models, training code, TensorRT instructions, and a linked C++/CUDA implementation. M0 uses a ResNet-50 backbone and one frame. Its reported PyTorch and TensorRT accuracy differ even before INT8 is enabled. **Project assessment:** reproduce this conversion difference before attributing all loss to quantization. M0 is our primary Strand A baseline, but its results cannot establish temporal-memory robustness or performance for ViT-based perception models. Published RTX 3090 speeds are reference measurements, not predictions for our hardware. [Code and deployment results](https://github.com/Yzichen/FlashOCC)

## 2. Benchmarking and Improving Bird's Eye View Perception Robustness in Autonomous Driving

**Authors / date:** Shaoyuan Xie et al.; first preprint May 27, 2024; published in IEEE TPAMI, vol. 47, no. 5, pp. 3878–3894 (May 2025). This is the expanded RoboBEV study, rather than the original April 2023 preprint. [Paper](https://arxiv.org/abs/2405.17426), [journal DOI](https://doi.org/10.1109/TPAMI.2025.3535960)

The study evaluates perception models across detection, segmentation, depth, and occupancy tasks under corruptions and sensor failures. It investigates how pretraining, image-to-BEV transformation, and temporal information relate to robustness, and presents a CLIP-based enhancement. Its findings show why a clean benchmark score is insufficient to describe behavior under degraded inputs. [Paper](https://arxiv.org/abs/2405.17426)

The public RoboBEV repository provides corruption-generation and evaluation resources. Its eight corruption types already include Camera Crash and Frame Lost; these must not be counted twice. Image color quantization is an input corruption and differs from reducing model weight precision. **Project assessment:** use the released protocol for Strand A, while testing the additional degradation caused by model quantization. Corruption benchmarking itself is established work. Single-frame results should be interpreted according to the inputs actually consumed by FlashOcc-M0. [Official code](https://github.com/worldbench/RoboBEV)

## 3. Boost Embodied AI Models with Robust Compression Boundary

**Authors / date:** Chong Yu, Tao Chen, and Zhongxue Gan (Fudan University); IJCAI 2025 Main Track, pages 6886–6894, DOI 10.24963/ijcai.2025/766; one of three IJCAI 2025 Distinguished Papers, presented August 20, 2025. No arXiv preprint exists. [Official proceedings](https://www.ijcai.org/proceedings/2025/766), [award announcement](https://aihub.org/2025/08/20/congratulations-to-the-ijcai2025-distinguished-paper-award-winners/)

BRCB studies compression while accounting for how embodied models respond to corrupted inputs. The method distinguishes components according to their corruption sensitivity and uses a gating mechanism during compression tuning. Its Table 1 reports FP32 baselines and INT8 compressed variants of BEVFormer, BEVFormer v2, and BEVFusion on nuScenes detection under benign, rainy, snowy, and dark samples; Tables 2–4 extend this to BEV map segmentation and OpenVLA robotics tasks (verified against the proceedings PDF). This is direct evidence that quantized BEV robustness has already been investigated. [Official paper](https://www.ijcai.org/proceedings/2025/0766.pdf)

**Project assessment:** the proposal must not claim to be the first study of BEV quantization under corruption. Its candidate extension is a reproducible occupancy deployment experiment that separates conversion loss, quantization loss, and measured hardware costs. BRCB uses its own rainy/snowy/dark corruption sets, not the standardized RoboBEV/nuScenes-C protocol, and reports only INT8 (no W4 or FP8) with no TensorRT latency. The proceedings PDF links only the target-model repositories (BEVFormer, BEVFusion, OpenVLA); no BRCB implementation appears on the IJCAI page, the authors' lab page, or in a GitHub repository search as of September 8, 2026, so BRCB is essential related work but not a reproduction dependency. These availability limits do not reduce its importance to the novelty assessment.

## 4. PTQAT: A Hybrid Parameter-Efficient Quantization Algorithm for 3D Perception Tasks

**Authors / date:** Xinhao Wang, Zhiwei Lin, Zhongyu Xia, and Yongtao Wang; first preprint August 14, 2025 (v2 August 15, 2025); ICCV 2025 BiVision Workshop; Peking University. [Paper](https://arxiv.org/abs/2508.10557)

PTQAT combines post-training quantization with selective quantization-aware fine-tuning. Rather than updating every quantizable layer, it chooses a subset for training while freezing other layers. The authors evaluate this approach on nuScenes detection, segmentation, and occupancy models. They report that selective tuning can compensate for quantization errors while updating fewer parameters than their QAT comparison. [Paper](https://arxiv.org/abs/2508.10557)

The work distinguishes its weight-only low-bit experiments from a TensorRT W8A8 deployment experiment. That distinction matters: a four-bit accuracy result does not automatically establish executable W4A8 acceleration. **Project assessment:** PTQAT supports our focus on layer sensitivity and confirms that quantized 3D perception is an active research area. However, its tuning procedure and A100 experiments are not a verified turnkey path for our 4090. A dedicated official implementation was not located. We therefore use it as methodological context, with selective tuning optional after the required PTQ experiments work. [Official workshop paper](https://openaccess.thecvf.com/content/ICCV2025W/BiVision/papers/Wang_PTQAT_A_Hybrid_Parameter-Efficient_Quantization_Algorithm_for_3D_Perception_Tasks_ICCVW_2025_paper.pdf)

## 5. DriveLM: Driving with Graph Visual Question Answering

**Authors / date:** Chonghao Sima et al.; first preprint December 21, 2023; ECCV 2024 Oral. [Paper](https://arxiv.org/abs/2312.14150)

DriveLM organizes driving questions into a graph that connects perception, prediction, and planning. Instead of treating every answer as unrelated, its task structure represents how recognizing objects can support reasoning about motion and driving decisions. The authors build datasets using nuScenes and CARLA and introduce DriveLM-Agent. Their experiments report benefits when generalizing to unseen objects or sensor configurations. [Paper](https://arxiv.org/abs/2312.14150)

Code and data-preparation scripts are publicly released; checkpoint availability should be confirmed in the repository docs before use. **Project assessment:** DriveLM supplies useful task definitions and a possible secondary evaluation source for Strand B. It helps distinguish failures in scene recognition from failures in predicting behavior or explaining a plan. We will verify training/evaluation overlap before using a driving-specialized checkpoint. A strong question-answering score remains a limited proxy for driving behavior; it does not demonstrate safe control of a vehicle. Open-ended responses also require scoring by meaning rather than matching exact wording. [Official code and data](https://github.com/OpenDriveLab/DriveLM)

## 6. Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives

**Authors / date:** Shaoyuan Xie et al.; first preprint January 7, 2025; ICCV 2025. [Paper](https://arxiv.org/abs/2501.04003), [official code and publication record](https://github.com/worldbench/DriveBench)

This paper introduces DriveBench, which tests VLMs with clean images, corrupted images, and absent visual information. It studies four driving tasks and separate corruption-awareness questions. Its central finding is that plausible responses can reflect textual cues or general knowledge even when useful image evidence is absent. Dataset imbalance and evaluation choices can conceal this behavior. [Paper](https://arxiv.org/abs/2501.04003)

DriveBench provides public evaluation code and data resources. Its core driving table uses perception MCQ/VQA, prediction VQA, planning VQA, and behavior MCQ. Captioning appears in the separate corruption-awareness study, giving three formats across the broader benchmark. **Project assessment:** use the core driving cells, audit skipped questions, and mark unavailable combinations N/A. Our extension concerns paired precision losses within supported groups, retaining distinct accuracy and judged-quality metrics. [Paper, Tables 6–7](https://arxiv.org/html/2501.04003v1#A1), [released evaluator](https://github.com/worldbench/DriveBench/blob/main/evaluate/eval.py)

## 7. Qwen2.5-VL Technical Report

**Authors / date:** Shuai Bai et al.; first preprint February 19, 2025. This is a technical report; no conference venue is claimed. [Paper](https://arxiv.org/abs/2502.13923)

Qwen2.5-VL combines a dynamic-resolution vision transformer with a language model. Its design includes window attention and video time information, supporting image understanding, object localization, and video tasks. The report evaluates broad multimodal capabilities across model sizes. Its largest-model results should not be treated as measured performance of the 7B model or evidence of driving reliability. [Paper](https://arxiv.org/abs/2502.13923)

The team releases a 7B instruction-tuned checkpoint and official AWQ variants. **Project assessment:** Qwen2.5-VL-7B-Instruct is our primary Strand B baseline because it offers a modern visual backbone and an accessible deployment starting point. We will hold the image-token limit, decoding settings, and prompt template fixed across precision variants. Fitting the complete multi-image workload in 24 GB must be demonstrated in the pilot. Qwen2.5-VL is a reproducibility choice, not a claim about the latest model ranking. [Model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [official release history](https://github.com/QwenLM/Qwen3-VL#news)

## 8. Qwen3-VL Technical Report

**Authors / date:** Shuai Bai et al.; first preprint November 26, 2025. This is a technical report. [Paper](https://arxiv.org/abs/2511.21631)

Qwen3-VL updates the model family through revised spatial-temporal position encoding, integration of features from several vision-transformer layers, and explicit video timestamp alignment. It offers multiple dense and mixture-of-experts sizes. The authors report improvements across multimodal evaluations and discuss quality under token and latency constraints. These claims concern their reported workloads, rather than our proposed driving-corruption experiment. [Paper](https://arxiv.org/abs/2511.21631)

The official repository links released models, including the 4B and 8B variants (released October 15, 2025, before the technical report), and provides inference examples. **Project assessment:** this paper establishes newer architectural context and prevents us from describing Qwen2.5-VL as the newest Qwen vision-language model. A small Qwen3-VL variant could test whether a finding transfers to another generation, but it is optional. Any comparison must pair each model with its own unquantized reference. Changing model generation together with precision would otherwise mix architecture, training, and quantization effects. Its largest models are outside our required hardware scope. [Official code and models](https://github.com/QwenLM/Qwen3-VL)

## 9. Can Small Quantized VLMs Drive? An Experimental Evaluation of Small Quantized VLMs for Autonomous Driving

**Authors / date:** Samson Mathew and Zonghua Gu (Hofstra University); 2025 IEEE International Conference on Data Mining Workshops (ICDMW), Washington, DC, November 12–15, 2025, pages 3255–3262, DOI 10.1109/ICDMW69685.2025.00440 (added to IEEE Xplore March 10, 2026). The official program lists the paper in the ICDM 2025 Undergraduate and High School Symposium (oral session UH1, November 15, 2025); it is cited here as an empirical data point, not as a peer-reviewed main-track result. The team holds the published PDF; the claims above were verified against it. [Publication DOI](https://doi.org/10.1109/ICDMW69685.2025.00440), [official conference program](https://www3.cs.stonybrook.edu/~icdm2025/program.html)

This study evaluates nine general-purpose VLMs (SmolVLM 256M/500M/2.2B, Qwen2.5-VL-3B/7B-Instruct, Gemma-3 4B/12B, LLaVA-1.5 7B/13B) at FP16 and as 4-bit BitsAndBytes variants, 18 configurations in total, served through vLLM on DriveBench (Table I and Sections II-B and III of the paper). Answers are scored 0–100 by GPT-5-mini. The experiments cover clean, corrupted, and missing-image conditions, answer quality, and inference timing. Reported latency benefits are mixed, showing why reduced weight precision cannot substitute for runtime measurement. The main quality summaries aggregate across driving tasks. [Author-uploaded manuscript](https://www.researchgate.net/publication/396983942_Can_Small_Quantized_VLMs_Drive_An_Experimental_Evaluation_of_Small_Quantized_VLMs_for_Autonomous_Driving)

**Project assessment:** this is direct prior work for Strand B, so applying quantization to driving VLMs is not our novelty claim. We will extend the analysis through paired quality changes, task-specific degradation, corruption interactions, and measurements on our named hardware. A public experiment repository was not located. The paper's introduction says "eight small-scale VLMs" while Table I lists nine (18 with 4-bit variants); we plan comparisons from Table I. The paper's own conclusion is that the 4-bit gap is "not very large" and some 4-bit variants score slightly higher than FP16, which is exactly the aggregate result our per-task and per-corruption pairing is designed to look beneath. An earlier data point exists: EM-VLM4AD (Gopalkrishnan, Greer, and Trivedi; CVPR 2024 Workshop on Vision and Language for Autonomous Driving and Robotics; [arXiv:2403.19838](https://arxiv.org/abs/2403.19838), [code](https://github.com/akshaygopalkr/EM-VLM4AD)) evaluated an 8-bit quantized T5-Large driving VLM on DriveLM, so quantized driving VQA predates Mathew–Gu; neither work tests corruption interactions.

## 10. MBQ: Modality-Balanced Quantization for Large Vision-Language Models

**Authors / date:** Shiyao Li et al.; first preprint December 27, 2024 (v2 March 21, 2025); CVPR 2025, pages 4167–4177. [Paper](https://arxiv.org/abs/2412.19509)

MBQ examines differences between the quantization sensitivity of visual and language tokens. It adjusts calibration so that reconstruction objectives account for those differences, rather than weighting all modalities equally. The paper studies weight-only and weight-and-activation quantization, and separately demonstrates a specialized low-bit GPU kernel. This connects calibration choices to accuracy while distinguishing a numerical quantization experiment from an executable speed demonstration. [Paper](https://arxiv.org/abs/2412.19509)

The official implementation includes quantization search and evaluation code. Its documented model choices include Qwen2-VL and several other families; turnkey Qwen2.5-VL support was not verified. Several example evaluation commands use pseudo quantization, and the W3 GPU kernel reported in the paper is not included in the public repository. **Project assessment:** MBQ motivates component-level sensitivity analysis and a controlled higher-precision mitigation in Strand B. It does not establish a ready-made W4A8 vLLM deployment for our selected checkpoint. Adapting MBQ is therefore optional, while the required experiment uses a precision configuration whose actual runtime execution can be verified. [Official code](https://github.com/thu-nics/MBQ)

## Synthesis and implications for the proposal

The literature supports the importance of this problem, but it also establishes substantial overlap. BRCB already combines BEV compression with corruption evaluation; PTQAT studies quantized 3D perception; DriveBench already analyzes unreliable driving answers; and Mathew–Gu already evaluate quantized driving VLMs. Adjacent 2025–2026 work tests precision × input-degradation interactions on 2D detection (Karimov et al., [arXiv:2508.19600](https://arxiv.org/abs/2508.19600), YOLO with TensorRT INT8) and on non-driving VLM uncertainty (Ferdous, [arXiv:2607.24440](https://arxiv.org/abs/2607.24440)), but none on occupancy/BEV under RoboBEV corruptions with TensorRT or on DriveBench stratified by task and corruption. A project consisting only of standard quantization and average accuracy/FPS tables would contribute limited new evidence. [BRCB](https://www.ijcai.org/proceedings/2025/766), [PTQAT](https://arxiv.org/abs/2508.10557), [DriveBench](https://arxiv.org/abs/2501.04003), [Mathew–Gu](https://doi.org/10.1109/ICDMW69685.2025.00440)

**Our candidate contribution is a controlled deployment study across both required strands:** does a precision choice that meets a quality budget on clean inputs still meet that budget under selected corruptions, and which VLM task groups suffer the largest additional losses? We will separate backend conversion from quantization, pair results on identical examples, report uncertainty by driving scene, and measure the memory/latency cost of one targeted mitigation. This is a proposed extension, not an established first-of-its-kind claim.

The minimum implementation remains one perception model and one VLM, with FlashOcc-M0 and Qwen2.5-VL-7B-Instruct as the primary choices and the contingencies in C available if a pilot fails. Both strands contribute required results. Adding architectures, all precision formats, custom kernels, a complete new benchmark, or reproductions of methods without released code is outside the minimum commitment. A null interaction is useful if its uncertainty is sufficiently small to assess a predefined tolerance; it does not establish safe autonomous driving.

## Latest-context check beyond the ten selected entries

The main survey is intentionally limited to ten substantive entries. Three additional 2026 developments inform its limitations:

- **QIG (Xiang et al., CVPR 2026; [arXiv:2603.17809](https://arxiv.org/abs/2603.17809), March 18, 2026)** estimates token-level quantization sensitivity with integrated gradients; official code and evaluation pipeline released April 3, 2026, documenting InternVL2, LLaVA-OneVision, and Qwen2-VL. It reinforces that sensitivity-aware quantization is active prior art; its documented paths still require compatibility checks for our selected deployment. [Official repository](https://github.com/ucas-xiang/QIG)
- **Less Precise Can Be More Reliable (ICML 2026; [arXiv:2509.21173](https://arxiv.org/abs/2509.21173), revised May 15, 2026)** evaluates quantization's effect on CLIP-style VLM reliability, including robustness to noise. Its setting differs from generative driving question answering, but it is further evidence that quantization robustness beyond average accuracy is already studied.
- **PeLiC-VLM (Gan et al., *Advanced Engineering Informatics* 71:104424, April 2026)**, "Performance and lightweight coexistence: Vision-language model for visual scene question answering in autonomous driving," combines a multi-view aggregation network, a visual refinement module, and an efficient quantized image-text adapter (268 M parameters, 4.7 GFLOPs, 0.49 GB). Its publisher abstract was available, but full-text access was blocked during this survey. Whether it already reports paired precision losses by task and corruption is **unverified**, and remains an explicit overlap question. [Publisher record](https://www.sciencedirect.com/science/article/pii/S1474034626001163), [DOI](https://doi.org/10.1016/j.aei.2026.104424)

This is a targeted recent-literature survey, not an exhaustive proof that no overlapping paper exists. The team will update the audit before freezing its final contribution claims and will report baseline reproductions under pinned software, checkpoint, dataset, and evaluation versions.
