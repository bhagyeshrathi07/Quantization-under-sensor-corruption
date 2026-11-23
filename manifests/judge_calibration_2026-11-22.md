# Judge calibration on the development answers (November 22, 2026): not accepted

Rule: decision 4 of `judge_decisions_2026-11-13.md`, unchanged; acceptance bars from `config/pilot.yaml` (`judge.acceptance`: held-out agreement 0.80, Cohen's kappa 0.40, recall 0.60 on both classes, 15 items per class, 100 items). Inputs: the 512-token development answers under the VQA amendment (`results/dev_vqa512/`, freeze `vqa512` of November 17), their three judgings each, and the consensus labels of `manifests/judge_label_sheet_2026-11-17.csv`. Command and report: `scripts/11_calibrate_judge.py ... --amendment vqa512 --out results/dev_vqa512/judge_calibration.json`.

## Labels

One human labeller per item (amendment of November 19 to decision 3): Ronak D001 to D088, Bhagyesh D089 to D175, Akshay D176 to D262; 31 human-correct of 262 before discussion (6, 8, 17). The two model labellers of the side study (Fable at medium effort, the Codex model at high effort, same guide and prompts) disagreed with the human on 30 items. The three humans settled those 30 on November 22 with `13 --consensus`, each with a comment; the consensus moved 16 of them (14 from correct to incorrect, 2 from incorrect to correct); the 232 agreed items took the human label by the rule. Consensus: 21 correct, 241 incorrect. No three-way human agreement exists (no shared items).

## Calibration

| | value | bar |
|---|---|---|
| items | 262 (26 scenes), 21 human-correct, 241 human-incorrect | 100, 15 per class |
| fit on all items | threshold 56, agreement 0.947, kappa 0.603, recall correct 0.571, recall incorrect 0.979 | |
| leave-one-scene-out, pooled held-out | agreement 0.927, kappa 0.42, recall correct 0.381, recall incorrect 0.975; fold thresholds 55 to 68 | 0.80, 0.40, 0.60, 0.60 |
| sensitivity, majority label (equal to the human label here) | threshold 56, held-out kappa 0.465, not accepted | |
| sensitivity, replicate 0 only | threshold 64, held-out kappa 0.524, not accepted | |

**Not accepted**: held-out recall on the human-correct items is 0.381, below 0.60. Agreement, kappa and the counts pass; the judge misses most of the answers the humans call correct (9 of 21 at the fitted threshold on the full fit, more under cross-validation) while almost never calling an incorrect one correct. The judge's rubric scores do not separate the 21 correct answers from the rest well enough at any single threshold: the fold thresholds range from 55 to 68.

By the pre-registered fallback, no judge threshold is frozen, every VQA cell stays unassessed, and the human labels are the VQA evidence: 21 of 262 development answers correct under the study's criterion (BF16 and INT4 together; per precision from the key, in the report). The final VQA run under the amendment and the matrix judging are not performed. The five MCQ contrasts and the core results are untouched.

## Side study: model labellers against the consensus

Agreement with the final consensus on the 262 items. Caveat: the consensus was settled with the models' reasons in view, so it is not independent of them; the human row is the only independent one.

| labeller | agrees | kappa | recall correct | recall incorrect | says correct |
|---|---|---|---|---|---|
| human (pre-discussion) | 246 | 0.660 | 0.857 | 0.946 | 31 |
| Fable, medium | 255 | 0.843 | 1.000 | 0.971 | 28 |
| Astra (Codex), high | 252 | 0.669 | 0.524 | 1.000 | 11 |
| judge at threshold 56 | 248 | 0.603 | 0.571 | 0.979 | 17 |

Fable and Astra agree with each other on 245 of 262. Fable flagged 36 items as ambiguous under the guide (29 of its labels there match the consensus), Astra 12 (9 match). Both models apply the written criterion at least as consistently as a single human reading it for the first time; Astra is the strictest reader of all four, the judge's rubric scores the least aligned with the criterion.

## What follows

- Report: VQA unassessed by the judge; human-labelled accuracy on the development answers as the VQA evidence, with the cap finding (`explore_512tok_2026-11-16.md`) and this calibration as the account of why. MCQ decision flips remain the primary result.
- No GPU work remains for VQA. The remaining work is tables, the side-study paragraph, the failure examples, and the write-up.
