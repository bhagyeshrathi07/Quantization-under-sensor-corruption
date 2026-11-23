# Judge calibration on the development answers (November 22 and 23, 2026): not accepted

Rule: decision 4 of `judge_decisions_2026-11-13.md`, unchanged; acceptance bars from `config/pilot.yaml` (`judge.acceptance`: held-out agreement 0.80, Cohen's kappa 0.40, recall 0.60 on both classes, 15 items per class, 100 items). Inputs: the 512-token development answers under the VQA amendment (`results/dev_vqa512/`, freeze `vqa512` of November 17), their three judgings each, and the consensus labels of `manifests/judge_label_sheet_2026-11-17.csv`. Command and report: `scripts/11_calibrate_judge.py ... --amendment vqa512 --out results/dev_vqa512/judge_calibration.json`.

## Labels

One human labeller per item (amendment of November 19 to decision 3): Ronak D001 to D088, Bhagyesh D089 to D175, Akshay D176 to D262; 31 human-correct of 262 before discussion (6, 8, 17). The two model labellers of the side study (Fable at medium effort, the Codex model at high effort, same guide and prompts) disagreed with the human on 30 items. The three humans settled those 30 on November 22 with `13 --consensus`, each with a comment; the 232 agreed items took the human label by the rule. On November 23 the reviewer found that D246's comment contradicted its recorded 0 (Fable's reason had been pasted next to a 0); the three humans confirmed 1, the reading they had applied to D048, D053, D096, D149, D175, D190 and D227, and the correction is recorded in the item's comment with the date. Final consensus: 22 correct, 240 incorrect; 15 items differ from the human label (12 from correct to incorrect, 3 from incorrect to correct: D074, D091, D099); on November 22, before the D246 correction, it was 16 (13 and 3). No three-way human agreement exists (no shared items).

Audit note on the disclosure order: each human's block was complete on disk and verified before that person's disagreement list was produced, but Ronak's copy was committed only together with the consensus (`4738480`), after the first calibration run; Bhagyesh's and Akshay's copies were committed before any list was produced. The git history therefore shows the order for two of the three labellers only. Consensus mode now refuses to open until every block is fully labelled, and the disagreement list refuses until every block is complete.

## Calibration

| | November 22 (D246 at 0) | November 23 (D246 at 1, final) | bar |
|---|---|---|---|
| items | 262 (26 scenes), 21 / 241 | 262 (26 scenes), 22 / 240 | 100, 15 per class |
| fit on all items | threshold 56, agreement 0.947, kappa 0.603, recall correct 0.571 (12 of 21), recall incorrect 0.979 | threshold 56, agreement 0.950, kappa 0.640, recall correct 0.591 (13 of 22), recall incorrect 0.983 | |
| leave-one-scene-out, pooled held-out | agreement 0.927, kappa 0.42, recall correct 0.381 (8 of 21), recall incorrect 0.975; fold thresholds 55 to 68 | agreement 0.947, kappa 0.621, recall correct 0.591 (13 of 22), recall incorrect 0.979; fold thresholds 55 and 56 | 0.80, 0.40, 0.60, 0.60 |
| sensitivity, majority label | threshold 56, held-out kappa 0.465, not accepted | threshold 56, held-out kappa 0.465, not accepted | |
| sensitivity, replicate 0 only | threshold 64, held-out kappa 0.524, not accepted | threshold 64, held-out kappa 0.49, not accepted | |

**Not accepted**: the registered threshold-selection procedure (the integer threshold maximizing agreement, lowest on a tie, evaluated by leave-one-scene-out cross-validation) fails the held-out recall bar on the human-correct items, 0.591 against 0.60, 13 of 22 caught, one item short. Agreement, kappa, recall on the incorrect items and the counts pass. In-sample the fitted threshold misses 9 of 22 human-correct answers and calls 4 of 240 human-incorrect ones correct. Other thresholds pass the rate bars in-sample (on the November 22 labels, 50 gave agreement 0.931, kappa 0.571, recalls 0.667 and 0.954), which says nothing about acceptance: the rule fixes the procedure, and choosing a threshold after seeing the held-out result is what the rule exists to prevent. The D246 correction moved held-out recall from 0.381 to 0.591 because the fold thresholds changed with it; it was made for consistency with the other items, not for the bar, and the verdict is the same either way.

By the pre-registered fallback, no judge threshold is frozen, every VQA cell stays unassessed by the judge, and the human labels are the VQA evidence. The final VQA run under the amendment and the matrix judging are not performed. Ronak decided on November 23 not to attempt a second judge. The five MCQ contrasts and the core results are untouched.

## Side study: model labellers against the consensus

Agreement with the final consensus on the 262 items. None of these rows is an independent validation: the consensus inherits the human label on 232 items and was settled with the models' reasons in view on the other 30, and the judge row is the in-sample fit. The numbers describe agreement with an assisted consensus, not independently established correctness or repeatability.

| labeller | agrees | kappa | recall correct | recall incorrect | says correct |
|---|---|---|---|---|---|
| human (pre-discussion) | 247 | 0.69 | 0.864 | 0.950 | 31 |
| Fable, medium | 256 | 0.87 | 1.000 | 0.975 | 28 |
| Astra (Codex), high | 251 | 0.65 | 0.500 | 1.000 | 11 |
| judge at threshold 56 (in-sample) | 249 | 0.64 | 0.591 | 0.983 | 17 |

(Rows recomputed on the final consensus with D246 at 1; the November 22 note had 246 / 255 / 252 / 248.) Fable and Astra agree with each other on 245 of 262. Fable flagged 36 items as ambiguous under the guide, Astra 12.

What is verified about the model labellers: every saved prompt holds the guide verbatim and, per item, the cell, question, reference answer and model answer, nothing else. What is operator-reported, not evidenced by a retained log: Fable ran as Claude Code agents (`claude-fable-5-1`, medium effort), Astra through `codex exec` from an empty temporary folder in a read-only sandbox with the model from the Codex configuration (`gpt-6-astra`, not passed explicitly); neither run kept its execution stream. Both facts are recorded in `results/dev_vqa512/llm_<model>/labels.json` as operator-reported. `14` now passes and records the model explicitly and keeps the Codex event stream.

## What follows

- Report: VQA unassessed by the judge. The VQA evidence is human-adjudicated, model-assisted labels on the clean development answers: 131 paired questions across 26 scenes, both precisions, with per-cell and per-precision counts and paired flips; it says nothing about the final set or the corrupted conditions. The cap finding (`explore_512tok_2026-11-16.md`) and this calibration are the account of why.
- No GPU work remains for VQA. The remaining work is tables, the side-study paragraph, the failure examples, and the write-up.
