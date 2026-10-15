# Week-5 baseline gate: October 14–15, 2026

Pilot run on a rented RTX 4090 (Vast.ai, host machine id m:3752, Norway; power limit 300 W, not stock 450 W). Software: vLLM 0.31.0, torch 2.13.0+cu130, Python 3.12. BF16 checkpoint revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. INT4 = official `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (pilot only).

## Gate results

| Check | Result |
|---|---|
| BF16 7B serves six views in 24 GB | Yes. Peak 21.9 GB total GPU use with `--gpu-memory-utilization 0.9`; that is the reservation, not model memory. No 3B fallback needed. |
| INT4 (AWQ) serves in vLLM | Yes, `--quantization awq_marlin`. |
| Option-letter logprobs available | Yes. With the MCQ suffix, the first generated token is the letter and all four options appear in top-5. |
| Margin computable | Yes, e.g. 0.35 on a NoImage behavior question. |
| Image tokens per view | ~1293 prompt tokens for a full six-view question, so ~165 image tokens per view at `max_pixels=200704` (nuScenes frames are 1600×900; the cap lands below the 252 estimate). Revisit the 512-token setting on development scenes. |
| 20-question run, both precisions, LowLight | 0 errors, all five eligible cells present. |
| Latency, batch 1 | BF16 p50 2.39 s, INT4 p50 0.91 s (2.6×). BF16 decode ~65 tok/s on an 80-token answer. Target of ≥15 tok/s met for both. |
| Scene split | `manifests/scene_split.json`: 131 final scenes, 1116 questions; prediction-VQA has 45 in final (coverage risk as predicted). |

**Decision: 7B, not 3B. Margin analysis feasible. Proceed to protocol freeze.**

## Findings that change the protocol

1. **Released DriveBench MCQ normalizer misreads explanations.** `evaluate/utils.py` matches `\b(a|option a)\b`, so "D. ... appears to be **a** vehicle" scores as option A. On the 8 pilot MCQs: strict rule 4/8 for both precisions and zero flips; released rule 2/8 vs 1/8 and one phantom harmful flip. Our primary rule is now "letter at the start of the answer"; the released rule is stored alongside (`mcq_correct_released`) for comparability. This must be reported, and it casts doubt on published DriveBench MCQ numbers for any model that explains its answers.
2. **DriveBench.zip has no clean images.** It holds the 15 corruption types only (7.2 GB; 1800 files per type). Clean images are nuScenes keyframes (1200 needed). Options: ask the instructor to allow copying those 1200 files off the class HPC; register at nuscenes.org; or run the clean condition on the HPC. Pending.
3. **Memory comparison needs equal KV-cache bytes.** With the 0.9 utilization setting both precisions show ~22 GB. Profiling runs must set `--kv-cache-memory-bytes` identically; the pilot numbers are not a memory result.
4. **Power-limited host.** 300 W cap. Latency numbers are for this host; profiling sessions must all use one pinned host, and the cap must be stated.

## Not yet done

- Clean condition (blocked on images).
- Fog condition (images present, not run).
- Own INT4 quantization from the base checkpoint (llm-compressor); official AWQ used for the pilot.
- Three-repeat stability slice.

## Raw files

`results/pilot/{bf16,int4}_LowLight.jsonl`, `results/pilot/smoke_*.json`, `logs/03_*_lowlight.txt`, `logs/05_lowlight.txt`, `manifests/machine.txt` (from `00_check_machine.sh` on the instance).
