# Quantization Under Sensor Corruption: Deployment and Robustness of Driving Perception and Vision-Language Models

**Institution:** San Jose State University (SJSU)  
**Team:** Ronak, Bhagyesh, and Akshay  
**Track:** System / Deployment (supported by experimental research)  
**Duration:** 14 weeks  
**Prepared:** September 8, 2026

## Abstract

Quantization reduces the precision of the numbers used inside a neural network, potentially reducing memory use and inference time. A small accuracy loss on clear images may conceal larger losses in fog, darkness, or particular driving questions. This project will build and evaluate two deployed inference paths: a camera-based occupancy model (FlashOcc-M0) in TensorRT and a driving-scene vision-language model (Qwen2.5-VL-7B-Instruct) in vLLM. Using recorded driving data (nuScenes with Occ3D labels, and DriveBench), we will compare each quantized model with the same model at higher precision, keeping inputs and evaluation settings fixed. We will measure additional accuracy loss under sensor corruption, changes in answers across driving tasks, latency, memory, and GPU energy, and test one targeted mitigation in each strand. The contribution is a reproducible, hardware-measured study of deployment tradeoffs and failure patterns, extending existing quantized-perception and driving-VLM research. Both strands run on one RTX 4090 (24 GB); an RTX 3060 Ti (8 GB) is supplementary, and neither HPC access nor a Jetson is required.

## Repository contents

| Course item | File |
|---|---|
| B. Literature & SOTA Survey | [B_LITERATURE_SURVEY.md](B_LITERATURE_SURVEY.md) |
| C. Project Proposal | [C_PROJECT_PROPOSAL.md](C_PROJECT_PROPOSAL.md) |
| D. AI Novelty & Feasibility Audit | [D_AI_NOVELTY_FEASIBILITY_AUDIT.md](D_AI_NOVELTY_FEASIBILITY_AUDIT.md) |

The survey covers ten papers from November 2023 to November 2025 plus 2026 context. The proposal specifies inputs, outputs, paired experiments, numerical targets, two real deployment paths, data, architecture, team responsibilities, and a timeline. The audit contains the AI critique and documents prior-work overlap and remaining uncertainty.

## Scope

Both strands are required. The minimum commitment is FlashOcc-M0 at TensorRT FP16 versus INT8, and Qwen2.5-VL-7B-Instruct at BF16 versus INT4 in vLLM. Each strand has a documented fallback model (BEVDet-R50; Qwen2.5-VL-3B-Instruct). FP8, W4A8, additional models, and Jetson measurements are extensions and do not displace the required work.

## Status

Proposal stage. No model has been downloaded, quantized, or benchmarked yet. Software revisions and resource measurements will be recorded during the week 1–2 pilot rather than estimated here.

