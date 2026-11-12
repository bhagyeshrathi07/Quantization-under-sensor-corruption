# Judge validation, development split (November 11, 2026)

Judge chosen by the team on November 11: DeepSeek V4 flash through the team's LiteLLM proxy (`deepseek-v4-flash-ron` at `https://aiapi.paid.lol/v1`), `max_tokens` 2048, `reasoning_effort: none`, backend description in `config/pilot.yaml` (`judge.backend`) from the proxy's routing table (`manifests/judge_proxy_routing.json`, read the same day). Everything below ran from the local machine with the key in a file outside the repository; no GPU, no host.

## 1. Proxy cache: replicates were copies until the bypass was sent

The first replicate runs (`--replicate 0/1/2` on the same 40 answers) returned the same `response_id` and `created` for all 40 questions across the three replicates, and the rep2/rep3 stability files (whose answers are byte-identical) received the same responses as each other: the proxy served identical requests from its cache. Those records are superseded (they stay in the sidecars under their old settings hash, as stale). Fix: every judge request now carries LiteLLM's per-request cache control `cache: {"no-cache": true, "no-store": true}` (`judge.no_cache`, default true; part of the wire request, the settings and the hash; travels in the SDK's `extra_body`). Verified: two identical requests returned different response ids; every replicate below has a distinct response id. A judge record is therefore a judging, not a lookup, and the measured usage is a real call.

## 2. Judge noise on fixed answers (plan section 7, item 1)

20 BF16 and 20 INT4 development answers (stability repeat 1, the first 20 VQA answers per file: 7 perception, 7 planning, 6 prediction) judged three times each (replicates 0, 1, 2), 120 calls, zero errors.

| | BF16 | INT4 |
|---|---|---|
| Score range across 3 judgings, mean / median / max | 7.5 / 6.5 / 18 | 7.0 / 5.0 / 20 |
| Within-answer SD, mean | 3.2 | 3.0 |
| Answers whose correctness changes across judgings at threshold 10 / 20 / 30 / 40 / 50 | 4 / 4 / 8 / 2 / 0 of 20 | 3 / 4 / 3 / 2 / 0 of 20 |
| Mean correct rate at threshold 10 / 20 / 30 / 40 / 50 | 0.87 / 0.63 / 0.18 / 0.03 / 0.00 | 0.80 / 0.48 / 0.07 / 0.03 / 0.00 |

Score level: the 70 rep-1 answers score between 4 and 66, quartiles 12 / 18 / 23.5, cell means BF16 perception 26.3, planning 18.1, prediction 18.5; INT4 26.7 / 14.1 / 12.5. Whether these answers are in fact mostly wrong, or the judge scores low, is what the human labels decide (section 4); the numbers above say nothing about which. Counts above candidate thresholds, rep 1 replicate 0: BF16 6 / 3 / 1 of 35 at 30 / 40 / 50, INT4 3 / 2 / 0 of 35. Flip classification stays defined at such counts. The 20-pair / 10-scene support rule belongs to the five prespecified MCQ contrasts (`05 --contrasts` runs over MCQ pairs only); `05 --vqa` pairs the judged VQA cells and reports them under the coverage minimum (30 questions, 10 scenes), and VQA output is exploratory in the protocol (section 8), with no VQA contrast computed. The flip counts in the table mean an answer crossed the threshold at least once in three judgings, not a per-judging error rate.

## 3. Judge scores across repeated model answers (plan section 7, item 2)

The three stability repeats' VQA answers (35 per precision per repeat, 210 calls, replicate 0, zero errors). Answer text differs across repeats for 6 of 35 BF16 and 5 of 35 INT4 questions; repeats 2 and 3 are byte-identical.

| | BF16 | INT4 |
|---|---|---|
| Score range across repeats, mean (all / same text / changed text) | 7.9 / 7.4 / 10.3 | 7.3 / 7.2 / 8.0 |
| Max range (same text / changed text) | 28 / 18 | 22 / 18 |
| Correctness changes across repeats at threshold 10 / 20 / 30 / 40 / 50 | 6 / 8 / 6 / 5 / 1 of 35 | 9 / 9 / 5 / 3 / 0 of 35 |
| of which with unchanged answer text | 4 / 6 / 3 / 3 / 1 of 29 | 8 / 6 / 5 / 3 / 0 of 30 |

Most of the across-repeat variation is judge noise on identical text, not answer variation: the same-text range (7.2–7.4) matches the fixed-answer range of section 2.

## 4. Human-label threshold sample (plan section 7, item 3)

`manifests/judge_threshold_sample_2026-11-11.csv`: 81 distinct development VQA answers (every rep-1 answer for both precisions, 70, plus the 11 rep-2 answers whose text differs), shuffled with a fixed seed, no scores, no precision, columns for a 1/0 correctness label per labeller. The key (item to question id, precision, repeat, judge score) is `results/stability/threshold_sample_key.json`; labellers must not open it. This is a sample limitation, not the development population: the development split holds 131 VQA questions in 26 scenes (90 planning, 30 perception, 11 prediction); model answers exist for 35 of them (the stability slice, 19 scenes), two precisions each, and the November 3 pilot files add one development-scene VQA answer. The 11 variant answers belong to 8 distinct questions, so the 81 items are unevenly spread over the 35 questions: 27 questions have two items (one per precision), 5 have three and 3 have four. Primary calibration, fixed here before any label is compared with a score: the 70 rep-1 items only, one BF16 and one INT4 answer per question, every question weighted equally; the threshold maximizing agreement on those 70 is the one frozen. Sensitivity analysis: the same calibration on all 81 items. The key marks the variants (`repeat: 2`); the labelling itself covers all 81 items blind. The plan asked for 100 answers: reaching it means a development-split run on the GPU (the 96 remaining development VQA questions, or a subset, both precisions; minutes of GPU plus the instance start), or the team amends the plan to this 81-answer sample and records the amendment. Procedure unchanged: two team members label blind on the rubric's correctness criterion; the threshold is the score that maximizes agreement with the labels; disagreements between labellers are recorded.

## 5. Cost

The six sidecars hold 580 records requested on November 11: 290 with the cache bypass (the fresh judgings: the 120 noise calls and the 210 repeat calls share the 40 replicate-0 records of rep 1) and 290 without it (the superseded, cache-served first runs); one compat probe and two bypass probes are not in the sidecars, so 583 requests left this machine. The 290 fresh judgings cost about $0.19 at the proxy's table rates by their measured usage; the 290 cache-served requests carry usage fields too but whether the proxy billed them is unknown, so total billed spend is between about $0.19 and about $0.38 and is read from the proxy's own spend log, not from these records.

## 6. Left

- Decide, before calibrating, whether production scoring stays replicate 0 (the plan and the code today) or becomes an average over k judgings. Averaging changes what the labels are calibrated against (the aggregate), needs the replicate count and a missing-score policy implemented and frozen, and multiplies the matrix cost: three judgings per answer is 13,728 calls, about $8.9 instead of $3.
- Labels (two people, 81 items, or more after a development run); choose the threshold against the chosen score (single or aggregate); decide whether to run the extra development questions to reach 100 or amend the plan.
- Judge freeze: `python scripts/freeze_judge.py --threshold <N> --records results/stability/*.judge.jsonl --note "..."` after committing the judge config (the freeze records the commit).
- Then the matrix: `08 results/final/<precision>_<condition>.jsonl` for the eight files (4,576 calls, about $3), `05 --contrasts --vqa`, `09_tables.py`.
