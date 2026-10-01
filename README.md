# Quantization-Induced Decision Failures in a Driving Vision-Language Model: A Paired BF16-versus-INT4 Deployment Study

**Institution:** San Jose State University (SJSU)  
**Team:** Ronak, Bhagyesh, and Akshay  
**Track:** System / Deployment (supported by experimental research)  
**Duration:** 14 weeks  
**Prepared:** September 8, 2026. **Revised:** September 29, 2026, after instructor feedback.

## Abstract

Quantization reduces the precision of the numbers used inside a neural network, potentially reducing memory use and inference time. Average accuracy on a benchmark may stay almost unchanged after quantization while individual answers still change, and some of those changes turn a correct driving decision into an incorrect one. This project will deploy one driving-scene vision-language model, Qwen2.5-VL-7B-Instruct, in vLLM at BF16 and at INT4, run the same DriveBench questions through both, and identify every case in which the full-precision model is correct but the quantized model changes the answer or the driving decision. We will analyze these failures by driving task, answer format, the full-precision model's confidence margin, the number of camera views, and, as a stress test, a small set of sensor-corruption conditions. We will measure the latency and memory savings that INT4 provides under matched serving settings, and, if time permits, test one simple mitigation: retaining a small set of sensitive decoder layers at BF16. The contribution is a reproducible, hardware-measured, paired failure analysis of a quantized driving VLM, extending prior work that reports only aggregate scores. The study runs on one RTX 4090 (24 GB); an RTX 3060 Ti (8 GB) is supplementary, and neither HPC access nor a Jetson is required.

## Repository contents

| Course item | File |
|---|---|
| B. Literature & SOTA Survey | [B_LITERATURE_SURVEY.md](B_LITERATURE_SURVEY.md) |
| C. Project Proposal | [C_PROJECT_PROPOSAL.md](C_PROJECT_PROPOSAL.md) |
| D. AI Novelty & Feasibility Audit | [D_AI_NOVELTY_FEASIBILITY_AUDIT.md](D_AI_NOVELTY_FEASIBILITY_AUDIT.md) |

The survey covers ten papers from November 2023 to November 2025 plus 2026 context. The proposal specifies the model and precision pair, the flip-classification protocol, the analysis factors, numerical targets, data handling, team responsibilities, and the revised timeline. The audit contains the original AI critique, the instructor feedback that narrowed the scope, and the documented prior-work overlap and remaining uncertainty.

## Scope

**Required:** Qwen2.5-VL-7B-Instruct at BF16 versus INT4 (AWQ preferred, GPTQ contingency) in vLLM, on DriveBench, in the clean condition plus a three-condition stress test (Dark, Fog, NoImage). Per-question flip classification, breakdowns by task, format, confidence margin, view count, and condition, and matched latency and memory measurements. Fallback model: Qwen2.5-VL-3B-Instruct.

**If time permits:** one mitigation that retains selected decoder layers at BF16.

**Removed on September 29, 2026:** the camera-based occupancy perception strand (FlashOcc-M0 in TensorRT), the full corruption matrix, and GPU energy as a required measurement. FP8, additional models, the remaining DriveBench conditions, energy, and Jetson measurements are extensions.

## Status

Proposal stage, revised. No model has been downloaded, quantized, or benchmarked yet. Software revisions and resource measurements will be recorded during the week 4–5 pilot rather than estimated here.
