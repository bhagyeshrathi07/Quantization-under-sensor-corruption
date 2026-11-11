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
| Prompt tokens (system line + filled rubric template + answer) | 1,100 (dry run on 12 pilot answers: 4,451 chars ÷ 4 for perception, 1,081 mean) | 953 (local judge, 40 development answers, November 8); 895–898 (DeepSeek V4 flash and pro, the same 40 answers) |
| Completion tokens (rubric breakdown + total score + summary) | 250 | 392 (local judge, reasoning effort low; 62 of them reasoning tokens); 313–321 (DeepSeek V4 flash, thinking off); 244–261 (DeepSeek V4 pro, thinking off) |
| Reasoning tokens (reasoning models only; billed as output) | 0 with `reasoning_effort: none` | local judge: ~400 at the default effort (cuts the reply off at 512 tokens), 46–62 at `low`. DeepSeek V4 flash: 0 with `none`; at the default and at `low` the whole budget (512, then 2048 tokens) went to reasoning and no score was returned |

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

## Non-OpenAI candidates (decided November 7: trial two during the GPU session)

The code is provider-agnostic: any OpenAI-compatible endpoint through `judge.base_url` / `08 --base-url`, key from `JUDGE_API_KEY` per session, endpoint and model frozen in the judge freeze and hashed into every record. The argument for a non-OpenAI judge is reproducibility more than price: every closed model above retires, and OpenAI retired the released judge mid-project.

| Candidate | How | Cost basis | Notes |
|---|---|---|---|
| DeepSeek V4 flash via the team's own LiteLLM proxy (alias `deepseek-v4-flash-ron` → upstream `deepseek/deepseek-v4-flash`) | `--base-url https://aiapi.paid.lol/v1 --model deepseek-v4-flash-ron` | proxy price table (set October 31, 2026; equals DeepSeek's peak cache-miss rate): $0.30 in / $1.20 out per 1M; projected from the trial's measured tokens, not a measured charge: matrix $2.97 unweighted, $2.66 weighted by the matrix cell mix; with mitigation and validation $4.76 | different family from Qwen; the proxy is ours (LiteLLM control plane), the upstream is DeepSeek's API; the record carries the alias, the endpoint, the `judge.backend` description and the response model id the proxy returns, while the alias-to-upstream mapping is the proxy's own routing table, copied by hand into `judge.backend.upstream_model`; closed API, rate limits unknown; V4 is a reasoning model: `reasoning_effort: none` turns thinking off as verified through this proxy (DeepSeek's own API documents `thinking: {type: disabled}` and `reasoning_effort` low/high/max; the proxy translates), the default and `low` return no score within 2048 tokens |
| DeepSeek V4 pro via the same proxy (alias `deepseek-v4-pro-ron` → `deepseek/deepseek-v4-pro`) | `--base-url https://aiapi.paid.lol/v1 --model deepseek-v4-pro-ron` | $1.32 in / $3.96 out per 1M; projected: matrix $9.99 unweighted, $8.82 cell-weighted; full programme $16.00 | trialled as well on November 8 (marginal cost under $0.10); same thinking-off requirement and identity caveat as flash |
| Local open weights on the study GPU (`scripts/serve_judge.sh`, default `openai/gpt-oss-20b`; fallback `mistralai/Mistral-7B-Instruct-v0.3`) | `--base-url http://localhost:8001/v1 --model judge-local`, key `EMPTY` | GPU time only: 4,576 calls at a few calls/s ≈ 0.5–1 h, under $1 at $0.617/h | most reproducible (pinned revision, same vLLM); runs after the study servers; rubric-following to be judged by the 40 × 3 noise test and the 100-answer human agreement, like any candidate; no Qwen-family judge |

Trial outcome, November 8 (`manifests/session2_2026-11-08.md` section 6). The first DeepSeek attempt was blocked by the upstream account balance (402 on every call); a reissued key later the same day exposed the `-ron` aliases and both DeepSeek lines were trialled from the local machine (no GPU needed; the key never touched the host). The local judge scored 40 of 40 answers only after `max_tokens` 2048 and `reasoning_effort: low`, and only with the markdown-tolerant score extraction added the same day. All three candidates scored the same 20 BF16 and 20 INT4 development answers (stability repeat 1), 40 of 40 each, zero errors:

| Candidate (settings) | Score range | Cell means BF16 (perception / planning / prediction) | Cell means INT4 | Extraction rule | Mean latency, 4 workers | Mean tokens per call (prompt / completion) |
|---|---|---|---|---|---|---|
| judge-local, gpt-oss-20b (`max_tokens` 2048, `reasoning_effort: low`; revision unpinned in this trial) | 0–31 | 20.9 / 8.4 / 6.3 | 20.1 / 6.0 / 12.0 | released 14, markdown 23, arithmetic 3 | 3.3 s | 953 / 392 |
| deepseek-v4-flash-ron (`max_tokens` 2048, `reasoning_effort: none`) | 4–36 | 26.3 / 13.0 / 21.8 | 26.4 / 9.1 / 17.3 | released 38, markdown 2 | 3.4 s | 897 / 317 |
| deepseek-v4-pro-ron (same) | 3–46 | 32.3 / 19.3 / 20.7 | 31.0 / 11.4 / 13.2 | released 38, markdown 2 | 4.7 s | 897 / 252 |

Agreement on the 20 shared answers per precision (absolute disagreement and signed difference are different things): flash vs pro mean absolute difference 6.55 (BF16) / 5.05 (INT4) points, flash lower by 3.95 / 1.15 on average, Pearson 0.73–0.79. Local vs flash: mean absolute difference 9.45 / 9.0, local lower by 8.15 / 4.9 on average, local higher on 3 / 7 of 20 answers. Local vs pro: 12.7 / 8.85, local lower by 12.1 / 6.05, local higher on 2 / 4 of 20. Agreement between candidates does not establish validity; all three are harsh relative to the 0–100 scale, and the threshold comes from the human-labelled sample, not from these means. Replicate noise (40 × 3) is still to be measured for whichever candidate the team picks.

The cost figures above are projections from the trial's measured tokens at the proxy's table rates, not measured charges: the trial is 14 perception / 14 planning / 12 prediction answers while a matrix run is 133 / 399 / 40, and planning replies are the shortest, so the cell-weighted projection is lower than the unweighted one; both come from clean development answers. DeepSeek's own price page (read November 8) distinguishes cache-hit and cache-miss input and halves every rate off-peak; the proxy's tariff and its bill were not verified here.

Backend identity (review of November 8). DeepSeek's model page (read November 8) says the legacy name `deepseek-v4-flash` is still accepted and routes to DeepSeek-V4.1-Flash-1017 at the Flash price, and that `deepseek-v4-pro` is DeepSeek-V4-Pro-0813. That describes DeepSeek's official routing, not what this proxy's backend actually serves: the proxy's routing table maps the `-ron` aliases to `deepseek/deepseek-v4-flash` and `deepseek/deepseek-v4-pro`, and the response model id the proxy returns is whatever it chooses to report. Every record now carries the response model id, response id, system fingerprint and request time, plus the `judge.backend` description from config; the judge freeze binds that description and the one response model id observed during validation, and final-split replies from any other response model are kept but never scored. Before the freeze, capture the proxy's non-secret routing configuration for the chosen alias (LiteLLM `/model/info`) into `judge.backend` and the session note. A local judge is bound by its checkpoint revision (`JUDGE_REVISION`, written to `manifests/serve_judge.txt` and checked against the freeze); the November 8 trial served gpt-oss-20b unpinned, which `serve_judge.sh` now refuses outside `JUDGE_ALLOW_UNPINNED=1` development trials. Trial procedure (development files only, the two candidates; the development-split stability repeats `results/stability/*_clean_rep1.jsonl` carry 35 VQA answers per precision with sidecars): `08 --compat-check --model <m> --base-url <u>`, then `08 results/dev/bf16_clean.jsonl results/dev/int4_clean.jsonl --limit 20 --model <m> --base-url <u>`; record measured usage, latency, score spread and the compatibility table below; the team then picks with real numbers.

## Compatibility (to fill from `08 --compat-check`)

| Model | `max_tokens` | `max_completion_tokens` | `temperature` | `top_p` | `reasoning_effort` | `seed` | Usage of the probe call |
|---|---|---|---|---|---|---|---|
| judge-local (gpt-oss-20b, vLLM 0.33.0, November 8) | accepted | not probed | not probed | not probed | accepted (`low`); default effort overruns a 512-token budget | not probed | prompt 742, completion 406 (75 reasoning), 2.2 s |
| deepseek-v4-flash-ron via proxy (November 8; the first attempt that day got 402 Insufficient Balance upstream on every call) | accepted | not probed | not probed | not probed | accepted; `none` turns thinking off through this proxy and is required: the default and `low` spent the whole budget (512, then 2048 tokens) on reasoning and returned no score | not probed | prompt 921, completion 445 (0 reasoning), 4.3 s, score 36 |
| deepseek-v4-pro-ron via proxy (November 8) | accepted | not probed | not probed | not probed | accepted (`none`) | not probed | prompt 921, completion 216 (0 reasoning), 4.8 s, score 45 |

## Recommendation (to fill after the trial)

Criteria, in order: the settings the released judge used (temperature 0, max_tokens 512) are accepted or have a documented equivalent; the model is available through the report date (mid-December 2026); cost of the full programme under the team budget; replicate noise (40 × 3) at the candidate threshold.

Open: budget ceiling (team item 2); whether the mitigation strand runs; threshold from the 100-answer human-labelled sample.
