# Quantization-Induced Decision Failures in a Driving Vision-Language Model: A Paired BF16-versus-INT4 Deployment Study

CMPE 249, San José State University, Fall 2026. Ronak Patel, Bhagyesh Rathi, Akshay Kumar. Draft of November 23, 2026. Core protocol frozen November 8, 2026; judge calibration closed November 23. Repository: `github.com/bhagyeshrathi07/Quantization-under-sensor-corruption` (the code and result revision at submission: pending, to be recorded in the README).

## Abstract

Weight-only 4-bit quantization is a common way to reduce the memory a 7-billion-parameter vision-language model (VLM) needs, and published benchmark averages suggest it costs little accuracy. We ask what the average hides for a driving assistant: when the full-precision model answers a driving question correctly, how often does the quantized model change that answer, and which questions are at risk? We quantized Qwen2.5-VL-7B-Instruct ourselves (AWQ, 4-bit weights, group size 128), served both precisions in vLLM under matched, frozen non-precision serving settings on one RTX 4090, and answered every eligible DriveBench question on 116 held-out scenes under four image conditions (clean, no image, low light, fog), pairing the two models' answers question by question. Five contrasts on the harmful-flip rate were pre-registered before the final run. On the 266 multiple-choice questions, taken under each of the four conditions (1,064 question-condition pairs), mean accuracy fell from 39.8% to 36.7%, but 55 answers the BF16 model had right were wrong under INT4 (13.0% of BF16-correct answers, 95% interval 9.6% to 16.8%; 14.2% under clean images), 21 went the other way, and 55 more changed between wrong options. The one pre-registered contrast whose Bonferroni-adjusted interval excluded zero concerns the full-precision model's own confidence: on clean images, among questions where BF16's first-token probability gap between its top two options was below 0.25, 15 of its 47 correct answers were lost under INT4; among the other questions, 0 of 59 (difference 0.319, adjusted 99% interval 0.140 to 0.512). The task and corruption contrasts were not resolved; their adjusted intervals allow effects of either sign of up to 25 points. INT4 cut peak GPU memory by 46.9% and median latency by a factor of 2.4, and the flip rate exceeds the measured stability floor (zero changes over three repeats of a 25-question slice). The free-text questions could not be scored at scale: the pre-registered calibration of an LLM judge against human labels failed its recall bar (13 of 22 human-correct answers recognized, bar 0.60), so that part of the benchmark is reported from 262 human-adjudicated development answers (8.4% correct) and is otherwise unassessed. Harmful flips concentrated among the clean questions on which the full-precision model was least sure; whether the quantized model's own margin could serve as a deployment-time signal was not tested.

## 1. Introduction

Quantization reduces the numeric precision of a network's weights so that the model needs less memory and, with suitable kernels, runs faster. For large language and vision-language models, 4-bit weight-only quantization has become routine, and the usual evidence that it is safe is a benchmark average that moves by a point or two. A driving assistant does not experience averages. It experiences individual decisions, and a decision that changes from right to wrong is a failure regardless of what happens elsewhere in the benchmark. Two models with the same average accuracy can disagree on many individual questions.

This study measures that disagreement directly. We take one open VLM, Qwen2.5-VL-7B-Instruct, produce a 4-bit version of it with a documented recipe, and ask the same driving questions of both under the same conditions. Every question yields a pair of answers, and each pair falls into one of four classes: unchanged, a change between two wrong answers, a beneficial flip (wrong to right) or a harmful flip (right to wrong). Throughout, "correct" means agreement with the benchmark's reference answer; "harmful" means a loss of that reference correctness, not a measured driving outcome. The harmful-flip rate, the share of BF16-correct answers that INT4 gets wrong, is the quantity of interest, and the research question is whether it concentrates in predictable subgroups: driving task, answer format, number of camera views, image corruption, and the full-precision model's own confidence.

The study was designed as a deployment experiment rather than a modelling one. The quantization, the serving stack, the prompt, the decoding settings, the question population and the analysis were fixed and recorded before the final run, and five contrasts were pre-registered with their statistical procedure. Mitigation (keeping a few decoder layers at full precision) was planned as an optional second stage and was not executed. Everything described here is reproducible from the repository, where every result file carries a provenance record that the analysis code checks against the frozen configuration before it will read it.

## 2. Related work

DriveBench (Xie et al., ICCV 2025) is the evaluation framework. It poses driving questions from DriveLM (Sima et al., ECCV 2024) over nuScenes keyframes in four tasks (perception, prediction, planning, behavior), under clean images, image corruptions and a no-image control, and its central finding is that general-purpose VLMs answer many driving questions from text priors rather than from the image. The benchmark has three question types: multiple choice, open-ended text and visual grounding. This study uses the first two; grounding is not evaluated. The released scorer defines the question cells we use and the rubric prompts for free-text scoring.

The closest prior work is Mathew and Gu (ICDMW 2025), who evaluated nine small VLMs, Qwen2.5-VL-7B among them, at 16-bit and 4-bit precision on DriveBench and concluded that the aggregate 4-bit gap is "not very large". Our study starts where theirs stops: it quantizes the same model, but pairs the two precisions' answers per question on a fixed population with a pre-registered analysis, so that the average gap is decomposed into harmful flips, beneficial flips and neutral changes, and related to question properties.

The quantization method is AWQ (Lin et al., 2024), activation-aware weight-only quantization, in the llm-compressor implementation, following the conventions of the Qwen2.5-VL technical report's official AWQ checkpoint (decoder weights quantized, vision encoder kept at BF16) but built from the base checkpoint with one pinned tool so that the recipe is known. On the sensitivity side, MBQ (Li et al., CVPR 2025) shows that the vision and language tokens of a VLM differ in quantization sensitivity and that calibration should account for it; PTQAT (Wang et al., ICCV 2025 workshop) shows selective quantization-aware tuning of sensitive layers for 3D perception models; BRCB (Yu et al., IJCAI 2025) studies compression of embodied models under corrupted inputs. These motivated the planned, unexecuted mitigation; none is reproduced here.

## 3. Method

### 3.1 Model, precisions and serving

The reference is `Qwen/Qwen2.5-VL-7B-Instruct` at a pinned revision, served in BF16. The INT4 model is our own checkpoint, built with llm-compressor 0.15.0 (AWQ, scheme W4A16 asymmetric, group size 128, `Linear` layers quantized, language-model head, embeddings and vision tower left at BF16), calibrated on every eligible question of a reserved calibration split (107 questions from 17 scenes, clean images, at the study's image budget), served in FP16 with the compressed-tensors kernels. The comparison is therefore between two deployment configurations, BF16 weights with BF16 kernels against 4-bit weights with FP16 compute and the quantized kernels, not between weight precisions alone. The official AWQ release of the same model was used only during the pilot.

Both precisions run in vLLM 0.33.0 (torch 2.14.0+cu130, transformers 5.20.2) on one rented RTX 4090 (24 GB, stock 450 W limit), one server at a time, with identical serving settings: context 4096 tokens, up to six images per prompt, image budget `max_pixels` 200,704 (about 165 image tokens per 1600×900 view), a KV cache of exactly 2 GiB for both precisions so that peak memory reflects weights rather than cache allocation, batch size 1, seed 0, prefix caching on. Decoding is greedy with a 128-token cap, seed 0, with the top-5 log-probabilities of the first generated token recorded.

### 3.2 Questions, prompt and conditions

The population is the DriveBench test set (1,461 questions over 174 scenes). The scenes were split once, with a fixed seed, into calibration (17 scenes, quantization only), development (26 scenes, every design choice, the stability and profiling slices, and the judge work of Section 3.5) and final (131 scenes). Fifteen final scenes that pilot runs had touched before the split was enforced were excluded, leaving a confirmatory population of 116 scenes and 838 eligible questions: behavior MCQ 133, perception MCQ 133, perception VQA 133, planning VQA 399, prediction VQA 40. Prediction MCQ is dropped by DriveBench's own scorer and was not run.

Each question is sent with the DriveBench system prompt, the camera views the question references (one for perception questions, six for behavior questions in this population), the original 1600×900 frames as data URLs, and, for multiple-choice questions, the suffix "Begin your answer with the option letter." so that the first generated token is the answer letter and its probability distribution over the letters can be read. Four conditions are run: clean, NoImage (a black 224×224 image per view, the benchmark's control for text-prior answering), LowLight and Fog (the benchmark's released corrupted frames). The matrix is 2 precisions × 4 conditions = 8 runs over the same questions. Because of the image handling, the answer suffix, the strict normalizer below and the study's own free-text criterion, the absolute accuracies reported here are not directly comparable with published DriveBench scores.

### 3.3 Outcomes

A multiple-choice answer is correct when the letter at the start of the response equals the reference (strict rule). The benchmark's released normalizer, a regular expression that also matches the article "a" as option A, is recorded as a secondary score and reported in Appendix A; the strict rule is primary. For each question and condition the BF16 and INT4 answers are paired and classified: harmful (BF16 correct, INT4 incorrect), beneficial (the reverse), neutral change (same correctness, different letter), unchanged. The harmful-flip rate F_h is harmful divided by BF16-correct; the change rate is one minus the unchanged fraction. Each pair also carries the BF16 margin, the probability of the top option letter minus the second at the first token, binned at 0.25 and 0.5, the task cell, the number of views and the condition.

### 3.4 Pre-registered analysis

Uncertainty comes from a paired bootstrap over scenes: scenes are resampled with replacement and every question of a drawn scene enters with both of its answers. Group intervals use 2,000 draws; the confirmatory contrasts use 20,000. Five contrasts on F_h were fixed before the final run, with Bonferroni correction over the family of five (adjusted 99% intervals; a contrast counts as detected when its adjusted interval excludes zero):

- C1: clean questions whose BF16 margin is below 0.25 versus all other margins;
- C2: behavior MCQ versus perception MCQ, clean;
- C4a, C4b, C4c: NoImage, LowLight and Fog versus clean, on the questions paired without error under both conditions.

A sixth comparison, one view versus six views, is reported as descriptive only: in this population every behavior question has six views and every perception question has one, so it is the task contrast under another name. Only these five contrasts are confirmatory; the per-condition and pooled rates, the accuracy differences, the change rates and the breakdowns are exploratory descriptions of the same data. Coverage rule: a group is reported as unassessed when it has fewer than 30 paired questions or fewer than 10 scenes (either suffices). Contrast support rule: each side of a contrast must first meet the ordinary coverage rule (30 paired questions from 10 scenes) and then hold at least 20 BF16-correct pairs from at least 10 scenes, and at least half of the bootstrap draws must be valid.

The configuration, the input fingerprints (annotations, split, exclusions, prompt), the pinned serving versions, the model identities and the protocol text were frozen on November 8, 2026 before the final run (`manifests/config_frozen.yaml`). Every run writes a provenance record beside its result file, and the analysis refuses final-split files whose record differs from the freeze in any setting. Three repeats of a 25-question clean development slice measure the stability floor. Three profiling sessions per precision measure time to first token, latency, decode speed and peak GPU memory; each session is 50 six-view development questions at batch size 1 with three warm-up requests excluded, the 128-token cap and prefix caching on, one fresh server start per session.

### 3.5 Free-text questions

The free-text (VQA) cells are scored by DriveBench with an LLM judge that applies a released rubric and returns a score from 0 to 100. We planned to use such a judge (DeepSeek V4 flash, through the team's proxy, the released rubric prompts verbatim, three judgings per answer averaged) and to count an answer correct when its mean score reached a threshold. The threshold rule was fixed in advance: over the integer thresholds 0 to 100, choose the one that maximizes agreement with the human labels on the development answers, the lowest on a tie; accept the judge only if, under leave-one-scene-out cross-validation with the threshold refitted in each fold, the pooled held-out predictions reach agreement 0.80, Cohen's kappa 0.40 and recall 0.60 on both the human-correct and the human-incorrect classes, with at least 100 calibration items and 15 per class. The fallback (no threshold, the free-text cells unassessed by the judge, the human labels as the evidence) was written down before any label existed. Because the procedure is fixed, no other threshold can be selected after a rejection.

Two things happened on the way. First, inspection of the development answers showed that the frozen 128-token decoding cap, chosen with the multiple-choice questions in mind, had been reached by 260 of 262 free-text answers (4,551 of 4,576 in the final matrix); the released benchmark decodes 512 tokens. Rather than alter the core freeze, we registered an amendment: a second, separately frozen configuration that differs from the core freeze in the decoding cap alone and applies to the free-text cells alone, under which the development answers were regenerated (23 of 262 still reach the cap) and judged three times each. Second, for time, the human labelling was done by one labeller per item in three blocks (88, 87 and 87 items) under a written criterion agreed before labelling (the model answer must contain the reference answer's core content, judged against the reference text without the images), blind to precision and to the judge's scores. Two language models, Claude Fable 5.1 at medium reasoning effort and the Codex model `gpt-6-astra` at high effort, labelled the same sheet under the same guide and the same prompt as a registered side study; the 30 items where a model disagreed with the human were settled by the three humans together, with the models' reasons visible, and the 232 agreed items took the human label. What is verified about the model labellers is the content of their prompts (the guide and the item's cell, question, reference and answer, nothing else); their execution details are operator-reported. The design, its limitations and a disclosure audit are recorded in `manifests/judge_decisions_2026-11-13.md` and `manifests/judge_calibration_2026-11-22.md`.

## 4. Results

All numbers below are from the frozen final run (`results/final/`, tables T1 to T7 in `results/final/tables.md`) and the development judge work (`results/dev_vqa512/vqa_tables.md`). Coverage was complete: every eligible question was answered by both precisions under every condition with zero errors.

### 4.1 Decisions change behind a small average loss (exploratory description)

Over the 266 multiple-choice questions under four conditions, 1,064 question-condition pairs, accuracy fell from 39.8% (BF16) to 36.7% (INT4), a difference of 3.2 points. The pairing shows what the average hides.

| condition | pairs | acc BF16 / INT4 | Δacc [95%] | harmful | beneficial | neutral | F_h [95%] | change rate [95%] |
|---|---|---|---|---|---|---|---|---|
| clean | 266 | 0.398 / 0.357 | −0.041 [−0.076, −0.011] | 15 | 4 | 14 | 0.142 [0.081, 0.211] | 0.124 [0.086, 0.162] |
| NoImage | 266 | 0.357 / 0.353 | −0.004 [−0.031, +0.023] | 7 | 6 | 14 | 0.074 [0.026, 0.128] | 0.102 [0.068, 0.135] |
| LowLight | 266 | 0.402 / 0.376 | −0.026 [−0.054, +0.003] | 12 | 5 | 15 | 0.112 [0.056, 0.178] | 0.120 [0.083, 0.161] |
| Fog | 266 | 0.436 / 0.380 | −0.056 [−0.094, −0.019] | 21 | 6 | 12 | 0.181 [0.118, 0.258] | 0.147 [0.107, 0.192] |
| pooled | 1064 | 0.398 / 0.367 | −0.032 [−0.050, −0.014] | 55 | 21 | 55 | 0.130 [0.096, 0.168] | 0.123 [0.101, 0.147] |

Pooled over the four conditions, which contribute equal numbers of pairs, 55 of the 424 answers that BF16 had right were wrong under INT4, a harmful-flip rate of 13.0% (95% interval 9.6% to 16.8%; 14.2% on clean images alone); 21 answers went from wrong to right, and 55 changed between two wrong options. One pair in eight changed (131 of 1,064, 12.3%). The net accuracy loss of 3.2 points is the difference between 55 harmful and 21 beneficial flips over 1,064 pairs, which is why the average understates the instability: for every answer lost on net, almost four decisions changed.

By task, behavior questions (six views, "what is the ego vehicle doing") had F_h 0.179 [0.125, 0.243] against 0.090 [0.048, 0.140] for perception questions (one view, "what is the state of this object"). Accuracy is low in absolute terms for both precisions, as DriveBench reports for general-purpose VLMs; the study's claim concerns the pairing, not the level.

### 4.2 The pre-registered contrasts: confidence predicts flips; task and corruption are not resolved

| id | contrast | F_h A | F_h B | Δ | 95% | 99% Bonferroni | status |
|---|---|---|---|---|---|---|---|
| C1 | margin below 0.25 vs other, clean | 0.319 (15 of 47 BF16-correct; 133 questions) | 0.000 (0 of 59; 133 questions) | +0.319 | [+0.184, +0.464] | [+0.140, +0.512] | detected |
| C2 | behavior vs perception, clean | 0.176 | 0.109 | +0.067 | [−0.063, +0.200] | [−0.103, +0.247] | not resolved |
| C4a | NoImage vs clean | 0.074 | 0.142 | −0.068 | [−0.159, +0.019] | [−0.192, +0.043] | not resolved |
| C4b | LowLight vs clean | 0.112 | 0.142 | −0.029 | [−0.110, +0.050] | [−0.137, +0.075] | not resolved |
| C4c | Fog vs clean | 0.181 | 0.142 | +0.040 | [−0.052, +0.132] | [−0.081, +0.161] | not resolved |

C1 is the result of the study. On clean images, the 266 questions split evenly by whether BF16's first-token probability gap between its two top options was below 0.25. Among the 133 low-margin questions BF16 answered 47 correctly and INT4 lost 15 of those; among the 133 others BF16 answered 59 correctly and INT4 lost none. The adjusted 99% interval for the difference runs from 0.140 to 0.512. Pooled over all four conditions the same pattern holds (table T4 and Figure 1): 53 of the 55 harmful flips, and all 21 beneficial ones, occurred in the lowest margin bin; the two remaining harmful flips are in the middle bin and the highest bin has none. Accuracy was not monotonic across the margin bins: BF16 accuracy is 41.8% in the lowest bin, 48.6% in the middle and 20.8% (43 of 207 pairs) in the highest. The margin predicts whether the quantized model will agree with the full-precision one, not whether either is correct.

![Figure 1. Harmful-flip rate by BF16 margin bin, exploratory, pooled over the four conditions, with 95% scene-bootstrap intervals and harmful / BF16-correct counts (table T4): 53 of 219, 2 of 162, 0 of 43. The [0, 0] interval of the highest bin means no harmful flip was observed there, not that the population risk is zero.](results/final/figures/fh_by_margin.svg)

The other four contrasts are not resolved, and their uncertainty is wide. In the adjusted 99% intervals, the task difference (behavior minus perception) lies between −10.3 and +24.7 points of F_h; NoImage minus clean between −19.2 and +4.3; LowLight minus clean between −13.7 and +7.5; Fog minus clean between −8.1 and +16.1. The point estimates are small, but with 116 scenes the data are consistent with substantial effects of either sign for the task and for each corruption. No power analysis was performed, so no claim is made about which effect sizes would have been detectable. The NoImage condition, where the model answers without any image, has the lowest observed harmful rate and the lowest accuracy; one hypothesis is that answers drawn from text priors are less affected by quantization, but the contrast does not test it, and the BF16-correct denominator differs by condition.

### 4.3 Deployment profile and stability

| precision | TTFT p50 / p95 (s) | latency p50 / p95 (s) | decode p50 (tok/s) | peak GPU memory (MiB) |
|---|---|---|---|---|
| BF16 | 0.11 / 0.39 | 2.10 / 2.39 | 63.3 | 19,400 |
| INT4 | 0.14 / 0.42 | 0.88 / 1.17 | 168.4 | 10,292 |

Pooled over three sessions of 50 requests per precision on the same host, INT4 reduced peak memory by 46.9% and median latency by a factor of 2.4, with decode speed up from 63 to 168 tokens per second; time to first token was slightly higher for INT4 (0.14 s against 0.11 s at the median), a difference this profile does not explain. Sessions agreed within 0.02 s at the median. The three performance targets set in the proposal were met (TTFT p95 at most 2 s, decode at least 15 tokens per second, memory saving at least 25%). The proposal also set a quality target for the clean multiple-choice cells, an upper confidence bound on the accuracy loss of at most 3 points per cell. That target is not met by either cell: on clean images behavior MCQ fell from 51 to 44 of 133 (−5.3 points, 95% interval −10.3 to −0.8, upper bound on the loss 10.3 points) and perception MCQ from 55 to 51 of 133 (41.4% to 38.3%, −3.0 points, interval −7.4 to +0.8, upper bound on the loss 7.4 points); both bounds exceed 3 points. The free-text quality target is unassessed (Section 4.4). The targets were planning tolerances, not acceptance criteria for the study.

Three repeated runs of the 25-question clean development slice gave identical option letters, correctness and text for both precisions (margin range at most 0.039). The measured stability floor is therefore zero changes, and the final flip rates are well above it. The floor was measured on 25 clean development questions, not on the final population or the corrupted conditions; it measures the observed instability on that slice and does not prove that every final flip arose from the weight change alone, and the two configurations also differ in serving kernels (Section 3.1).

### 4.4 Free-text questions: judge not validated, human labels as the evidence

The judge calibration did not pass. On 262 development answers (131 questions, both precisions, clean condition, 512-token amendment) with a human-adjudicated consensus of 22 correct and 240 incorrect, the pre-registered procedure chose a threshold of 56 and, under leave-one-scene-out cross-validation, reached agreement 0.947, kappa 0.62, recall 0.98 on human-incorrect answers and recall 0.59 (13 of 22) on human-correct answers, against a bar of 0.60; passing required 14 of 22. Agreement, kappa and the counts passed; recall on the correct class did not. One consensus label (D246) was corrected by the three labellers after the first run, for consistency with their reading of seven similar items; the correction moved held-out recall from 0.381 to 0.591 with the verdict unchanged, which shows how much the estimate rests on 22 positive items and on the thresholds fitted in each fold. By the rule fixed in advance, no threshold was frozen, the final free-text answers were not judged, and the free-text cells of the matrix are reported as unassessed. The rubric the judge applies rewards elements the study's criterion does not (reasoning quality, a stated probability that an action is safe); whether that mismatch explains the missed answers was not examined and remains a hypothesis.

What the human labels show on the development answers, descriptively:

| cell | BF16 correct | INT4 correct | both | paired: both right / harmful / beneficial / both wrong |
|---|---|---|---|---|
| perception VQA | 0 / 30 | 0 / 30 | 0 / 60 | 0 / 0 / 0 / 30 |
| planning VQA | 9 / 90 | 13 / 90 | 22 / 180 | 4 / 5 / 9 / 72 |
| prediction VQA | 0 / 11 | 0 / 11 | 0 / 22 | 0 / 0 / 0 / 11 |
| all | 9 / 131 (6.9%) | 13 / 131 (9.9%) | 22 / 262 (8.4%) | 4 / 5 / 9 / 113 |

Under the study's written criterion, 8.4% of the free-text answers were correct, all of them planning answers; no perception or prediction answer met the criterion, and the causes of those failures were not analyzed beyond the labellers' comments. INT4 was correct slightly more often than BF16 (13 against 9) with five harmful and nine beneficial flips among 131 questions. These are 26 development scenes under clean images and carry no interval; they say nothing about the final scenes or the corrupted conditions.

The side study of model labellers is reported for what it is. Against the final consensus, the human labellers' pre-discussion labels agreed on 247 of 262 items, Claude Fable 5.1 (medium effort) on 256, `gpt-6-astra` (high effort) on 251, and the rubric judge at its fitted threshold on 249. None of these is an independent validation: the consensus inherits the human label on the 232 agreed items and was settled with the models' reasons in view on the other 30. These counts describe agreement with the assisted consensus; they establish neither comparative labelling accuracy nor repeatability.

### 4.5 Failure examples

Two of the 55 harmful flips, from `results/final/failure_examples.md` (lowest BF16 margin first; the full list with option probabilities and view files is in the repository):

1. Perception MCQ, clean, margin 0.000. "What is the moving status of object <c1, CAM_FRONT, 0.30, 0.54>?" Options: A going ahead (correct), B turn left, C turn right. BF16 put 0.34 on A, 0.34 on B and 0.27 on C and answered A. INT4 answered C, and its explanation then argues that the vehicle "seems to be moving straight ahead" before concluding "C. Turn right": the letter and the explanation disagree. Under the declared letter scorer this is a harmful flip; the text does not show a changed motion judgment.
2. Behavior MCQ, Fog, margin 0.000. "Predict the behavior of the ego vehicle." Options: A steering right, slowly; B slightly steering right, slowly (correct); C steering left, normal speed; D steering right, very fast. BF16 put 0.29 on B and 0.29 on C and answered B; INT4 answered A.

Twelve such examples are selected in the repository with caps per cell and condition; their camera images are on the GPU host and are to be added to the submission.

## 5. Discussion

The headline is that 4-bit quantization of this driving VLM is not a uniform 3-point tax. It leaves most decisions untouched and removes about one in eight of the correct ones on this population, and the decisions it removes are the ones the full-precision model was uncertain about: on clean images nothing with a first-token margin above 0.25 flipped. Two consequences follow. The first is that the average accuracy of a quantized model is a poor proxy for its decision-level agreement with the full-precision model; a 3-point average loss hid a 13% loss of correct decisions plus a comparable number of changed wrong ones. The second is a hypothesis rather than a result: harmful flips concentrated among clean questions with low BF16 margin, which suggests that a margin-based signal might identify at-risk answers at inference time. The contrast used BF16's margin, so whether INT4's own margin supports such routing, at what cost, and for what share of answers, remains untested.

The unresolved contrasts should be read as what they are. Within the adjusted intervals the task difference (behavior minus perception) lies between −10.3 and +24.7 points of F_h, NoImage minus clean between −19.2 and +4.3, LowLight minus clean between −13.7 and +7.5, and Fog minus clean between −8.1 and +16.1; the bounds differ substantially by condition. The point estimates are small; the data do not establish that the effects are.

The free-text part of the study did not reach a verdict, and the reasons are a methods result. A decoding cap chosen with multiple-choice questions in mind had been reached by nearly every free-text answer, and nobody checked completion lengths against the cap before the freeze; the `completion_tokens` field recorded in every answer is what made the problem detectable after the fact, and the finish reason is now recorded as well. The rubric judge, applied verbatim as released, came one item short of the pre-registered acceptance bar on a positive class of 22. Its scores separate correct from incorrect answers in aggregate (kappa 0.62), but the acceptance rule asks for recall on the correct class, and that is where it fell short. A judge built on the study's own criterion, validated on an independently labelled set, is the natural next attempt; the team chose not to make it within the course.

## 6. Limitations

- "Harmful" is loss of benchmark-reference correctness. No closed-loop driving performance or safety outcome was evaluated.
- One model, one quantization recipe, one hardware target. The concentration of flips at low margin may be specific to AWQ on Qwen2.5-VL-7B and to the first-token margin under this prompt design (greedy decoding, answer-letter suffix).
- The confirmatory population is 116 scenes. The task and corruption contrasts are unresolved with intervals that allow large effects; no power analysis was performed. Absolute accuracies are low for both precisions, which limits the number of reference-correct pairs in some subgroups (the highest margin bin has 207 pairs but 43 BF16-correct).
- Absolute accuracies depend on the study's image handling, answer suffix, strict normalizer and free-text criterion and are not comparable with published DriveBench scores.
- The free-text cells are unassessed by the judge. The human-adjudicated evidence covers 131 development questions under clean images only, was produced by one human per item with two language models as the second opinion, and its consensus was settled with the models' reasons visible; no three-way human agreement was measured; the 128-token free-text answers of the final matrix remain unscored.
- The stability floor was measured on 25 clean development questions; the latency and memory figures are for batch size 1 on one RTX 4090 with a fixed 2 GiB KV cache and are not a throughput benchmark.
- Mitigation (retaining sensitive decoder layers at BF16) was planned conditionally and not executed.

## 7. Reproducibility and status of deliverables

The repository records every setting with its frozen value (`PROTOCOL.md`, `manifests/config_frozen.yaml`, `manifests/config_frozen_vqa512.yaml`). Every result file carries a provenance sidecar, and the analysis refuses files that do not match the freeze. From the raw answer files, the judge replies, the label sheets, the profiling logs and the manifests, the scripts regenerate every table and figure: `05_compare_flips.py --contrasts` (flip analysis and intervals), `07_profile.py --report` (profile), `06_stability.py --report` (stability floor), `09_tables.py` (tables T1 to T7), `17_vqa_tables.py` (free-text tables), `18_failure_examples.py` (examples) and `19_figures.py` (Figure 1). The offline test suite (130 tests) covers the flip classification, the bootstrap and contrasts, the provenance and freeze guards, the judge pipeline and the labelling tools. The judge decisions, the labelling guide, the amendment and the calibration are documented in dated notes under `manifests/`, including every reviewer finding and its resolution.

Status of the deliverables promised in the proposal: the BF16/INT4 pair, the 8-run matrix, the profiling and the stability floor are complete; the free-text scoring stopped at the calibration gate as described; the failure explorer was not built (the examples are a generated list); mitigation was not executed; the reproduction from a fresh checkout is pending at the time of this draft.

## 8. Conclusion

Quantizing Qwen2.5-VL-7B to 4-bit weights nearly halved its peak memory and more than halved its median latency at a cost of about 3 points of multiple-choice accuracy on DriveBench, but behind that average one correct decision in eight was lost and as many wrong answers changed. The losses were not spread evenly: on clean images every lost decision was one on which the full-precision model had been unsure, and none of its confident decisions flipped. Whether the task or the image condition changes the risk could not be resolved with 116 scenes. The final free-text portion of the benchmark remains unscored because the judge did not pass its pre-registered validation, which we report as it happened. For deployment, the result argues that a quantized driving VLM should be evaluated at the decision level, paired against its full-precision reference, and that the full-precision model's first-token margin is the first place to look for the decisions quantization will change; whether the quantized model's own margin carries the same information is untested.

## 9. Contributions

Ronak Patel: quantization pipeline and checkpoint, vLLM serving and host, profiling, GPU sessions, judge proxy, labelling. Akshay Kumar: DriveBench adapter, prompt and view handling, scorer integration, flip classification, labelling. Bhagyesh Rathi: manifests and scene split, run logging and provenance schema, bootstrap statistics, labelling. The analysis code, freeze mechanics, judge calibration and labelling tools were developed with AI assistance (Claude) and reviewed round by round by a second model (Codex); all findings and their resolutions are recorded in the protocol change log.

## Appendix A. Secondary multiple-choice score (released normalizer)

The benchmark's released normalizer searches the whole lower-cased answer for the patterns `a`, `b`, `c`, `d` (or `option a` and so on) as whole words, in that order, and returns the first that matches. Because `a` is also the article and the search is not anchored to the start, an answer such as "D. There is a vehicle ahead." normalizes to A, so the rule can override an explicit initial letter and change correctness in either direction. It is recorded alongside the strict rule; the strict rule is the primary score throughout this report. Correct answers of 266 per condition:

| condition | BF16 strict | BF16 released | INT4 strict | INT4 released |
|---|---|---|---|---|
| clean | 106 | 125 | 95 | 119 |
| NoImage | 95 | 106 | 94 | 104 |
| LowLight | 107 | 115 | 100 | 125 |
| Fog | 116 | 133 | 101 | 129 |

## References

1. Shaoyuan Xie et al. Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives (DriveBench). ICCV 2025. arXiv:2501.04003.
2. Chonghao Sima et al. DriveLM: Driving with Graph Visual Question Answering. ECCV 2024. arXiv:2312.14150.
3. Samson Mathew and Zonghua Gu. Can Small Quantized VLMs Drive? An Experimental Evaluation of Small Quantized VLMs for Autonomous Driving. ICDMW 2025. doi:10.1109/ICDMW69685.2025.00440.
4. Ji Lin et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024. arXiv:2306.00978.
5. Shuai Bai et al. Qwen2.5-VL Technical Report. arXiv:2502.13923, 2025.
6. Shiyao Li et al. MBQ: Modality-Balanced Quantization for Large Vision-Language Models. CVPR 2025. arXiv:2412.19509.
7. Xinhao Wang et al. PTQAT: A Hybrid Parameter-Efficient Quantization Algorithm for 3D Perception Tasks. ICCV 2025 BiVision Workshop. arXiv:2508.10557.
8. Chong Yu, Tao Chen and Zhongxue Gan. Boost Embodied AI Models with Robust Compression Boundary. IJCAI 2025. doi:10.24963/ijcai.2025/766.
