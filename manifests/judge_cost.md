# VQA judge: cost note (skeleton, November 5, 2026)

Plan `WEEK9_ANALYSIS_PLAN.md` section 7. The team decides budget and model (team item 2); the judge freeze (`manifests/judge_frozen.yaml`) records the decision. Measured token counts come from the `--limit 20` trial of `scripts/08_judge_vqa.py`; until that trial runs, the rows marked *provisional* use an assumed token budget and must not be quoted as a measurement.

## Call counts

| Scope | VQA answers per run | Runs | Calls |
|---|---|---|---|
| 8-run matrix (2 precisions × 4 conditions), confirmatory population | 572 | 8 | 4,576 |
| Mitigation (if time permits): 1 extra precision × 4 conditions | 572 | 4 | 2,288 |
| Development validation: noise on fixed answers (40 answers × 3 replicates) | 40 | 3 | 120 |
| Development validation: stability repeats (35 VQA answers × 3 repeats × 2 precisions) | 35 | 6 | 210 |
| Development validation: threshold sample (100 answers, replicate 0) | 100 | 1 | 100 |
| Compat check and trial | — | — | ≤ 30 |

Matrix plus development validation: about 5,040 calls. With mitigation: about 7,330.

## Token budget per call

| Quantity | Provisional assumption | Measured (`--limit 20`) |
|---|---|---|
| Prompt tokens (system line + filled rubric template + answer) | 1,100 (dry run on 12 pilot answers: 4,451 chars ÷ 4 for perception, 1,081 mean) | TBD |
| Completion tokens (rubric breakdown + total score + summary) | 250 | TBD |
| Reasoning tokens (reasoning models only; billed as output) | 0 with `reasoning_effort: none` | TBD |

The dry run reports characters ÷ 4; the planning template is the longest (its object description is appended). The trial replaces both rows with the mean of the measured `usage` fields written into every judge record.

## Prices

Source: https://developers.openai.com/api/docs/pricing (Standard tier, USD per 1M tokens), read November 5, 2026; shutdown dates from https://developers.openai.com/api/docs/deprecations, read November 6, 2026. Re-read both on the day of the trial and record the date here.

| Candidate | Input | Output | Status (plan section 7, verified November 4) | Matrix, provisional (4,576 calls) | Matrix + mitigation + validation (7,330 calls) |
|---|---|---|---|---|---|
| gpt-5.6-terra | $2.00 | $12.00 | OpenAI's named replacement for the retired gpt-3.5-turbo path | $23.80 | $38.12 |
| gpt-5.6-luna | $0.20 | $1.20 | current small model; parameters undocumented, compat check required | $2.38 | $3.81 |
| gpt-5.4-mini | $0.75 | $4.50 | accepts `temperature` only with `reasoning_effort: none` | $8.92 | $14.29 |
| gpt-5.4-nano | $0.20 | $1.25 | same family; cheapest current; shuts down April 1, 2027 | $2.44 | $3.90 |
| gpt-4.1-mini | $0.40 | $1.60 | non-reasoning; accepts the released settings as-is; shuts down January 14, 2027 (deprecations page, read November 6): usable for this semester only | $3.84 | $6.16 |
| gpt-4o-mini | $0.15 | $0.60 | shuts down January 14, 2027: usable for this semester only | $1.44 | $2.31 |
| gpt-5-mini | $0.25 | $2.00 | shuts down December 11, 2026; rejects `temperature` | $3.55 | $5.68 |

Provisional cost per call = (1,100 × input + 250 × output) / 1,000,000. Example, gpt-5.6-terra: (1,100 × 2.00 + 250 × 12.00) / 1e6 = $0.0052 per call.

## Compatibility (to fill from `08 --compat-check`)

| Model | `max_tokens` | `max_completion_tokens` | `temperature` | `top_p` | `reasoning_effort` | `seed` | Usage of the probe call |
|---|---|---|---|---|---|---|---|
| TBD | | | | | | | |

## Recommendation (to fill after the trial)

Criteria, in order: the settings the released judge used (temperature 0, max_tokens 512) are accepted or have a documented equivalent; the model is available through the report date (mid-December 2026); cost of the full programme under the team budget; replicate noise (40 × 3) at the candidate threshold.

Open: budget ceiling (team item 2); whether the mitigation strand runs; threshold from the 100-answer human-labelled sample.
