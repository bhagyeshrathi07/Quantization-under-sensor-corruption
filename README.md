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
| [PROTOCOL.md](PROTOCOL.md) | Exact execution protocol: checkpoints, serving flags, prompt, scoring rules, flip classes, statistics, record schema, provenance and freeze mechanics. Ready for the core freeze since November 6; the judge protocol is frozen separately. |
| [REPORT.md](REPORT.md) | The report: question, method, results (decision flips, the margin contrast, deployment profile, free-text evidence), discussion, limitations, reproducibility. |
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
src/qflip/contrasts.py             the five prespecified F_h contrasts: paired scene bootstrap, raw and Bonferroni intervals, support rule
src/qflip/judge.py                 DriveBench rubric judge: description lookup, prompt fill, score parsing, sidecars
src/qflip/judge_prompts.py         released judge templates, verbatim
src/qflip/provenance.py            result-file sidecars (input hashes, checkpoint fingerprint), resume compatibility, live-server check
src/qflip/freeze.py, guard.py      core freeze (frozen config + protocol snapshot) and the run/analysis guards that enforce it
src/qflip/report.py                coverage per cell, harmful examples, pooled deployment profile (pooling rule)
src/qflip/tables.py                report tables T1-T7 as markdown; formatting only
src/qflip/versions.py              the one runtime-version collector (module __version__, build tag kept) for the serve manifest and the freeze
src/qflip/split.py, stats.py, gpu.py, records.py
scripts/download_drivebench.py     annotations + visual descriptions (HF, toolkit) + corruption images (Google Drive zip)
scripts/download_clean_images.py   clean nuScenes keyframes streamed from the public AWS mirror
scripts/00_check_machine.sh        host facts -> manifests/machine.txt
scripts/01_serve.sh bf16|int4      vllm serve with the protocol's flags, equal KV-cache bytes
scripts/02_smoke_request.py        one six-view request: tokens per view, logprobs, margin, memory
scripts/03_smoke_drivebench.py     N or all questions per (precision, condition), resumable, --split
scripts/04_scene_split.py          calibration / development / final scene manifest
scripts/exclude_exposed_scenes.py  record final scenes a pilot inspected -> manifests/scene_exclusions.json
scripts/freeze_config.py           write / check the core freeze (pins, exact runtime builds, checkpoint fingerprint, snapshot)
scripts/runtime_versions.py        serve-manifest version lines from qflip.versions (called by 01_serve.sh)
scripts/freeze_judge.py            write / check the judge freeze (settings, threshold, template and system-line hashes)
scripts/05_compare_flips.py        flips with bootstrap CIs, breakdowns, contrasts, coverage per cell, harmful examples; --vqa; --out JSON
scripts/06_stability.py            three-repeat slice: nondeterminism floor
scripts/07_profile.py              TTFT, latency percentiles, decode tok/s, peak memory per session; --report pools sessions (--out JSON)
scripts/08_judge_vqa.py            VQA rubric judge (released prompts, settings hashed into each record, --compat-check, replicates), resumable sidecars
scripts/09_tables.py               markdown tables T1-T7 from the 05, 07 and 06 JSON outputs
scripts/10_build_label_sheet.py    blind label sheet for the judge calibration (development answers under the core freeze only; one item per VQA answer, shuffled; one copy per labeller; key kept apart; never overwritten)
scripts/11_calibrate_judge.py      --labels-only: pre-discussion agreement and Fleiss' kappa from the label columns alone (consensus columns not read); full: decision 4 against the consensus labels on the rows every labeller reached (issued items, sheet order, displayed text and answers bound by digest; agreement curve, leave-one-scene-out acceptance on judge.acceptance as exact fractions, sensitivity rows)
scripts/12_merge_label_copies.py   merge the labellers' copies into the master sheet (same items, order and text; conflicts refused)
config/explore_512tok.yaml         exploratory copy of the pilot config at the released 512-token decoding length (manifests/explore_512tok_2026-11-16.md); never study data
config/vqa512.yaml                 the VQA amendment: pilot config with max_tokens 512, frozen separately (freeze_config.py --amendment vqa512); 03/05/08/10/11 take --amendment vqa512, run_matrix.sh AMENDMENT=vqa512
scripts/13_label_ui.py             local browser tool for labelling one person's copy (manifests/labelling_guide.md beside each item; the labeller's assigned block, in order; reference words highlighted in the model answer; backed-up atomic writes of the label column only; --check); --consensus: the discussion mode on the master sheet (human and model labels with reasons side by side, consensus and comment per item, "accept the human label on every agreed item"; writes the consensus columns only)
scripts/14_llm_label.py            side study: a model labels its own copy under the same guide (prompts per batch; --backend codex; never merged into the master sheet)
scripts/15_label_assignment.py     who labels which items: contiguous blocks in sheet order, written beside the sheet and picked up by 13 and 11
scripts/16_disagreements.py        a human's disagreements with the model labellers, with the models' reasons, for the consensus discussion (only after every block is complete)
scripts/17_vqa_tables.py           report tables V1-V4 from the consensus sheet, its key and the calibration JSON (human-adjudicated VQA correctness per cell and precision, paired outcomes, calibration summary, labellers against the consensus)
scripts/19_figures.py              report figure F1 (harmful-flip rate by BF16 margin bin with intervals and counts) as SVG from 05's JSON
scripts/18_failure_examples.py     harmful MCQ flips for the report from 05's JSON and the raw answers (lowest BF16 margin first, caps per cell and condition; question, options, both answers, option probabilities, image paths with what is missing locally)
scripts/serve_judge.sh             open-weights judge on a second vLLM port for the VQA judge trial (any OpenAI-compatible endpoint works via --base-url)
scripts/host_pull.sh               update the GPU host's checkout: backs up colliding untracked and modified files, fast-forwards, compares
scripts/quantize_awq.py            own AWQ W4A16 checkpoint with llm-compressor, calibration-split prompts, manifest
scripts/run_matrix.sh              serve -> all conditions -> stop, per precision; resumable; DRY_RUN=1 prints the plan
scripts/rescore.py                 recompute MCQ fields after a rule change
scripts/vast_setup.sh, hpc_pilot.sbatch
tests/                             130 offline unit tests (pytest)
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

- GPU session 1 on the replacement host, November 3 ([manifests/session1_2026-11-03.md](manifests/session1_2026-11-03.md)): the October instance had been destroyed by its host and was rebuilt; clean nuScenes images fetched; equal KV cache and streaming verified on vLLM 0.33; clean, Fog, NoImage 20-question slices on both precisions (zero errors; 1 harmful and 1 beneficial flip in 24 MCQ pairs, both NoImage with BF16 margin below 0.25; exploratory, and these scenes are now excluded from the confirmatory population); official-checkpoint reference measurements, kept under `results/reference_official/` and not pooled into the study: MCQ stability at zero self-changes with VQA text varying in 6 of 35 BF16 and 1 of 35 INT4 repeats, and a deployment profile (INT4 2.4× faster at batch 1, 47% less peak memory, equal 2 GiB KV cache); own AWQ W4A16 checkpoint built in 8 minutes and served.

- November 4 review (two rounds, [WEEK9_ANALYSIS_PLAN.md](WEEK9_ANALYSIS_PLAN.md)): the November 3 slices had drawn from the final holdout; 15 scenes are now excluded (confirmatory population 116 scenes, 838 questions), every result file carries a provenance sidecar, final-split runs and analyses are refused without a matching freeze, pairing validates eight fields, and the November 3 profile and stability numbers are reclassified as official-checkpoint reference measurements.

- Steps 1 and 2 of the plan are committed: integrity fixes (four review rounds), then the five prespecified contrasts with both intervals and the support rule (`05 --contrasts`), the views comparison as descriptive only, PROTOCOL sections 8 and 9 rewritten (two review rounds). Step 4 (judge, November 5): optional request settings, `input_sha256` and `replicate` as the record key, production scores bound to the current answer and judge settings, `08 --compat-check`, measured usage, and the cost note [manifests/judge_cost.md](manifests/judge_cost.md) with current prices; after review, `08` validates provenance and both freezes before any call and the judge freeze binds the rubric templates by hash. The judge model itself is still the team's decision. Step 3 (November 6): `05` writes coverage per cell and harmful examples, `07 --report --out` pools profile sessions under the plan's rule, and `09_tables.py` formats T1–T7.

- GPU session 2, November 7–8 ([manifests/session2_2026-11-08.md](manifests/session2_2026-11-08.md)): core freeze written on the host and committed; both servers verified against it; own-checkpoint stability: zero observed MCQ letter or correctness changes across three repeats for both precisions (VQA text varies in 6 of 35 BF16 and 5 of 35 INT4 questions; judge-score noise unmeasured); three profile sessions per precision pooled (INT4 peak memory 10,292 vs 19,400 MiB, latency p50 0.88 vs 2.10 s); the 8-run matrix on the 838 confirmatory questions with zero errors; analysis under the frozen settings: 1,064 MCQ pairs; descriptive summaries: pooled F_h 0.130 [0.096, 0.168], 53 of 55 harmful flips at BF16 margin below 0.25; confirmatory family: contrast C1 (low margin vs other bins) excludes zero after Bonferroni adjustment, the other four do not. Judge trial on 40 development answers: local gpt-oss-20b scored 40 of 40 after settings and extraction changes; DeepSeek V4 flash and pro through the team's proxy scored 40 of 40 each with thinking turned off (`reasoning_effort: none`, verified through the proxy), projected matrix cost $2.97 and $9.99 from measured tokens. The reviewer's two passes on the session fixed the score extractor (a complete expression or nothing, decimals rejected; version 4), bound it to the judge freeze, added backend identity to every record and to the freeze with one validity predicate shared by resume, freeze evidence and analysis, and bound a local judge to its running server; a third pass made one function (`usable_score`: expected backend plus a reply the current extractor parses) the criterion for resume, freeze evidence, the judging summary and analysis alike, and replaced the host pull checklist with `scripts/host_pull.sh` (93 tests).

**In progress (week 10).** Judge chosen (DeepSeek V4 flash via the team's proxy) and validated on development answers ([manifests/judge_validation_2026-11-11.md](manifests/judge_validation_2026-11-11.md)); human labels for the threshold, then the judge freeze and VQA judging of the matrix; report tables from `results/final/tables.md`; mitigation if time permits.

**Decided November 13** ([manifests/judge_decisions_2026-11-13.md](manifests/judge_decisions_2026-11-13.md)): production VQA score = mean of three judgings (`judge.production_replicates`, frozen with the judge), label pool = every development VQA answer from a development run on the frozen servers (262 items), three blind labellers with a consensus label, threshold rule fixed before any label exists. **November 14** ([manifests/session3_2026-11-14.md](manifests/session3_2026-11-14.md)): development run done under the core freeze (191 answers per precision, zero errors), every development VQA answer judged three times (786 calls, all distinct, nothing stale), the blind sheet issued (`manifests/withdrawn/judge_label_sheet_2026-11-14.csv`, 262 items, one copy per labeller beside it, key in `results/dev/label_sheet_key_withdrawn_2026-11-14.json`), and the calibration implemented (`qflip.calibrate`, scripts 10, 11 and 12, bars in `judge.acceptance`) with tests on synthetic labels; the reviewer's pass on the tooling (five findings: displayed text not bound to the labels, no frozen-development input guard, sheet order and item set not enforced, a kappa of exactly 0.40 failing in floating point, no pre-discussion mode) is fixed. Open: the three labellings, the discussion, the calibration run, the judge freeze.

**VQA amendment (November 16, [manifests/explore_512tok_2026-11-16.md](manifests/explore_512tok_2026-11-16.md)).** 99% of the VQA answers stopped at the frozen 128-token cap (the released decoding is 512). VQA is regenerated at 512 under a separate, enforced amendment freeze into its own directories; the MCQ study and its results are untouched. Sequence (steps after the calibration superseded by its outcome, see below): amendment freeze from a clean commit, development run (262 answers), 786 fresh judgings, replacement blind sheets (all done November 17, [manifests/session4_2026-11-17.md](manifests/session4_2026-11-17.md)), labelling with `scripts/13_label_ui.py` by `manifests/labelling_guide.md`, calibration under the unchanged acceptance rules, judge freeze, final VQA run (4,576 answers, about 5.75 GPU hours), matrix judging. The November 14 sheets are withdrawn unlabelled.

**Judge calibration, November 22 and 23** ([manifests/judge_calibration_2026-11-22.md](manifests/judge_calibration_2026-11-22.md)): one human per item plus two model labellers as the second opinion, consensus on all 262 items (22 correct after one correction); the judge is not accepted (held-out recall on human-correct items 0.59, 13 of 22, bar 0.60). VQA stays unassessed by the judge; the human labels are the VQA evidence; no final VQA run, no matrix judging; no second judge attempt (decided November 23). The amendment sequence's judge-freeze and final-run steps are superseded.

**Not started.** Explorer (weeks 10–11), mitigation (12, to be reported as not executed), report (13–14).

**Open team decisions** are listed in [TEAM_REVIEW.md](TEAM_REVIEW.md), section 8 and the October 19 addendum.

Pilot results (`results/pilot/`, 152 KB) are tracked so teammates can run `05_compare_flips.py` on them. Full-run results stay untracked and are synced from the host; the final raw files are archived with the submission.
