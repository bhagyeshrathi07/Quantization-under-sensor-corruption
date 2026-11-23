### V1. Human-adjudicated correctness of the clean development VQA answers (consensus labels)

Source: `manifests/judge_label_sheet_2026-11-17.csv` (consensus column), key `results/dev_vqa512/label_sheet_key.json`. One human labeller per item, two model labellers as the second opinion, disagreements settled by the three humans; the criterion is `manifests/labelling_guide.md`. Development scenes, clean condition, 512-token amendment; no claim about the final scenes or the corrupted conditions.

| cell | BF16 | INT4 | both |
|---|---|---|---|
| perception/VQA | 0/30 (0.0%) | 0/30 (0.0%) | 0/60 (0.0%) |
| planning/VQA | 9/90 (10.0%) | 13/90 (14.4%) | 22/180 (12.2%) |
| prediction/VQA | 0/11 (0.0%) | 0/11 (0.0%) | 0/22 (0.0%) |
| all | 9/131 (6.9%) | 13/131 (9.9%) | 22/262 (8.4%) |

### V2. Paired outcome per development question (BF16 against INT4)

Descriptive: 131 questions on 26 development scenes, no interval. Harmful = BF16 correct and INT4 incorrect; beneficial = the reverse.

| cell | questions | both correct | harmful | beneficial | both incorrect |
|---|---|---|---|---|---|
| perception/VQA | 30 | 0 | 0 | 0 | 30 |
| planning/VQA | 90 | 4 | 5 | 9 | 72 |
| prediction/VQA | 11 | 0 | 0 | 0 | 11 |
| all | 131 | 4 | 5 | 9 | 113 |

### V3. Judge calibration (decision 4, pre-registered)

Source: `results/dev_vqa512/judge_calibration.json`. Verdict: **NOT ACCEPTED**; reasons: held-out recall on human-correct items 0.591 below 0.60.

| | fit on all items | leave-one-scene-out, pooled held-out | bar |
|---|---|---|---|
| threshold | 56 |  |  |
| agreement | 0.950 | 0.947 | 0.8 |
| Cohen's kappa | 0.640 | 0.621 | 0.4 |
| recall, human-correct | 0.591 | 0.591 | 0.6 |
| recall, human-incorrect | 0.983 | 0.979 | 0.6 |
| items (human-correct / human-incorrect) | 262 (22 / 240) | | 100 (15 per class) |
| fold thresholds | | 55, 56 | |

Held-out: leave-one-scene-out over 26 scenes, predictions pooled; the fit of each fold chooses its own threshold (fold thresholds 55, 56).

### V4. Labellers against the final consensus (not an independent validation)

The consensus inherits the human label on the agreed items and was settled with the models' reasons in view on the others; these rows measure agreement with that assisted consensus.

| labeller | agrees | of | says correct |
|---|---|---|---|
| human (pre-discussion) | 247 | 262 | 31 |
| fable | 256 | 262 | 28 |
| astra | 251 | 262 | 11 |
| consensus | 262 | 262 | 22 |
