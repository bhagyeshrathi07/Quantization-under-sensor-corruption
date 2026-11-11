Analysis class: **confirmatory**; core freeze 2026-11-08T00:06:06+00:00.

### T1. Coverage per cell (confirmatory analysis)

| Condition | Cell | Eligible | Present BF16 | Present INT4 | Errored BF16 / INT4 | Paired | Unpaired | Scenes | Assessed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | behavior/MCQ | 133 | 133 | 133 | 0 / 0 | 133 | 0 | 116 | yes |
| clean | perception/MCQ | 133 | 133 | 133 | 0 / 0 | 133 | 0 | 116 | yes |
| clean | perception/VQA | 133 | 133 | 133 | 0 / 0 | 0 | 133 | 0 | no |
| clean | planning/VQA | 399 | 399 | 399 | 0 / 0 | 0 | 399 | 0 | no |
| clean | prediction/VQA | 40 | 40 | 40 | 0 / 0 | 0 | 40 | 0 | no |
| NoImage | behavior/MCQ | 133 | 133 | 133 | 0 / 0 | 133 | 0 | 116 | yes |
| NoImage | perception/MCQ | 133 | 133 | 133 | 0 / 0 | 133 | 0 | 116 | yes |
| NoImage | perception/VQA | 133 | 133 | 133 | 0 / 0 | 0 | 133 | 0 | no |
| NoImage | planning/VQA | 399 | 399 | 399 | 0 / 0 | 0 | 399 | 0 | no |
| NoImage | prediction/VQA | 40 | 40 | 40 | 0 / 0 | 0 | 40 | 0 | no |
| LowLight | behavior/MCQ | 133 | 133 | 133 | 0 / 0 | 133 | 0 | 116 | yes |
| LowLight | perception/MCQ | 133 | 133 | 133 | 0 / 0 | 133 | 0 | 116 | yes |
| LowLight | perception/VQA | 133 | 133 | 133 | 0 / 0 | 0 | 133 | 0 | no |
| LowLight | planning/VQA | 399 | 399 | 399 | 0 / 0 | 0 | 399 | 0 | no |
| LowLight | prediction/VQA | 40 | 40 | 40 | 0 / 0 | 0 | 40 | 0 | no |
| Fog | behavior/MCQ | 133 | 133 | 133 | 0 / 0 | 133 | 0 | 116 | yes |
| Fog | perception/MCQ | 133 | 133 | 133 | 0 / 0 | 133 | 0 | 116 | yes |
| Fog | perception/VQA | 133 | 133 | 133 | 0 / 0 | 0 | 133 | 0 | no |
| Fog | planning/VQA | 399 | 399 | 399 | 0 / 0 | 0 | 399 | 0 | no |
| Fog | prediction/VQA | 40 | 40 | 40 | 0 / 0 | 0 | 40 | 0 | no |

Eligible = questions of the cell in the declared population (n/a when the population is not declared). Assessed = at least 30 paired questions from 10 scenes.

### T2. Headline per condition, MCQ

| Group | n | Scenes | Acc BF16 / INT4 | Δacc [95%] | Harmful | Beneficial | Neutral | F_h [95%] | Change [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fog | 266 | 116 | 0.436 / 0.380 | -0.056 [-0.094, -0.019] | 21 | 6 | 12 | 0.181 [0.118, 0.258] | 0.147 [0.107, 0.192] |
| LowLight | 266 | 116 | 0.402 / 0.376 | -0.026 [-0.054, +0.003] | 12 | 5 | 15 | 0.112 [0.056, 0.178] | 0.120 [0.083, 0.161] |
| NoImage | 266 | 116 | 0.357 / 0.353 | -0.004 [-0.031, +0.023] | 7 | 6 | 14 | 0.074 [0.026, 0.128] | 0.102 [0.068, 0.135] |
| clean | 266 | 116 | 0.398 / 0.357 | -0.041 [-0.076, -0.011] | 15 | 4 | 14 | 0.142 [0.081, 0.211] | 0.124 [0.086, 0.162] |

Group intervals: scene-level paired bootstrap, 2000 draws, seed 0. F_h = harmful / BF16-correct.

### T3. Prespecified contrasts (family of 5, Bonferroni α = 0.010)

| Id | Contrast | Population | F_h A | F_h B | Δ | 95% raw | 99% Bonferroni | Valid draws | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | margin [0,0.25) vs other margin bins | clean, pairs with a margin | 0.319 (n=133, ref-correct 47/41 sc.) | 0.000 (n=133, ref-correct 59/47 sc.) | +0.319 | [+0.184, +0.464] | [+0.140, +0.512] | 20000/20000 | assessed * |
| C2 | behavior/MCQ vs perception/MCQ | clean | 0.176 (n=133, ref-correct 51/47 sc.) | 0.109 (n=133, ref-correct 55/51 sc.) | +0.067 | [-0.063, +0.200] | [-0.103, +0.247] | 20000/20000 | assessed |
| C4a | NoImage vs clean | common support (ids paired without error under both) | 0.074 (n=266, ref-correct 95/72 sc.) | 0.142 (n=266, ref-correct 106/78 sc.) | -0.068 | [-0.159, +0.019] | [-0.192, +0.043] | 20000/20000 | assessed |
| C4b | LowLight vs clean | common support (ids paired without error under both) | 0.112 (n=266, ref-correct 107/79 sc.) | 0.142 (n=266, ref-correct 106/78 sc.) | -0.029 | [-0.110, +0.050] | [-0.137, +0.075] | 20000/20000 | assessed |
| C4c | Fog vs clean | common support (ids paired without error under both) | 0.181 (n=266, ref-correct 116/84 sc.) | 0.142 (n=266, ref-correct 106/78 sc.) | +0.040 | [-0.052, +0.132] | [-0.081, +0.161] | 20000/20000 | assessed |

Descriptive, outside the family:

| Id | Contrast | Population | F_h A | F_h B | Δ | 95% raw | 99% Bonferroni | Valid draws | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| views | 1 view vs 6 views | clean | 0.109 (n=133, ref-correct 55/51 sc.) | 0.176 (n=133, ref-correct 51/47 sc.) | -0.067 | [-0.199, +0.063] | n/a | 20000/20000 | descriptive (no decision) |

- descriptive only: in the final split every behavior MCQ has six views and every perception MCQ has one, so Δ_views = −Δ_task in every draw

`*` = the Bonferroni-adjusted interval excludes zero. Draws 20000, seed 0; support rule: ≥ 20 reference-correct pairs from ≥ 10 scenes per side, valid draws ≥ 50%.

C4 common support, ids dropped per input file: C4a {'clean/bf16': 0, 'clean/int4': 0, 'NoImage/bf16': 0, 'NoImage/int4': 0}; C4b {'clean/bf16': 0, 'clean/int4': 0, 'LowLight/bf16': 0, 'LowLight/int4': 0}; C4c {'clean/bf16': 0, 'clean/int4': 0, 'Fog/bf16': 0, 'Fog/int4': 0}

### T4. Breakdowns by cell, views and margin

MCQ, by cell:

| Group | n | Scenes | Acc BF16 / INT4 | Δacc [95%] | Harmful | Beneficial | Neutral | F_h [95%] | Change [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| behavior/MCQ | 532 | 116 | 0.357 / 0.327 | -0.030 [-0.061, +0.002] | 34 | 18 | 33 | 0.179 [0.125, 0.243] | 0.160 [0.125, 0.200] |
| perception/MCQ | 532 | 116 | 0.440 / 0.406 | -0.034 [-0.057, -0.013] | 21 | 3 | 22 | 0.090 [0.048, 0.140] | 0.086 [0.059, 0.117] |

MCQ, by n_views:

| Group | n | Scenes | Acc BF16 / INT4 | Δacc [95%] | Harmful | Beneficial | Neutral | F_h [95%] | Change [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 532 | 116 | 0.440 / 0.406 | -0.034 [-0.057, -0.013] | 21 | 3 | 22 | 0.090 [0.048, 0.140] | 0.086 [0.059, 0.117] |
| 6 | 532 | 116 | 0.357 / 0.327 | -0.030 [-0.061, +0.002] | 34 | 18 | 33 | 0.179 [0.125, 0.243] | 0.160 [0.125, 0.200] |

MCQ, by margin:

| Group | n | Scenes | Acc BF16 / INT4 | Δacc [95%] | Harmful | Beneficial | Neutral | F_h [95%] | Change [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [0,0.25) | 524 | 112 | 0.418 / 0.357 | -0.061 [-0.098, -0.024] | 53 | 21 | 55 | 0.242 [0.181, 0.307] | 0.246 [0.205, 0.289] |
| [0.25,0.5) | 333 | 101 | 0.486 / 0.480 | -0.006 [-0.015, +0.000] | 2 | 0 | 0 | 0.012 [0.000, 0.030] | 0.006 [0.000, 0.015] |
| [0.5,1] | 207 | 61 | 0.208 / 0.208 | +0.000 [+0.000, +0.000] | 0 | 0 | 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### T5. Stability floor (self-disagreement across repeats)

| Precision | Condition | Model | Repeats | MCQ n | Letter change | Correctness change | Text change | Margin range max | VQA n | VQA correctness change | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | clean | qwen2.5-vl-7b-bf16 | 3 | 25 | 0.000 | 0.000 | 0.000 | 0.039 | n/a | n/a | 0 |
| int4 | clean | qwen2.5-vl-7b-awq | 3 | 25 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0 |

VQA columns come from judged repeats when available (plan section 7); text change alone is not a correctness measure.

### T6. Deployment profile (pooled over sessions, per-session rows below)

| Precision | Sessions | Requests | Errors | TTFT p50 / p95 / p99 (s) | Latency p50 / p95 / p99 (s) | Decode p50 (tok/s) | Peak memory (MiB) | Host |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **bf16** (pooled) | 3 | 150 | 0 | 0.11 / 0.39 / 0.41 | 2.10 / 2.39 / 2.41 | 63.3 | 19400 | 658b5a53146a / NVIDIA GeForce RTX 4090 |
| bf16 s1 | 1 | 50 | 0 | 0.12 / 0.39 / 0.41 | 2.11 / 2.40 / 2.42 | 63.3 | 19400 |  |
| bf16 s2 | 1 | 50 | 0 | 0.11 / 0.39 / 0.39 | 2.10 / 2.39 / 2.40 | 63.3 | 19400 |  |
| bf16 s3 | 1 | 50 | 0 | 0.11 / 0.40 / 0.41 | 2.10 / 2.39 / 2.41 | 63.3 | 19400 |  |
| **int4** (pooled) | 3 | 150 | 0 | 0.14 / 0.42 / 0.43 | 0.88 / 1.17 / 1.18 | 168.4 | 10292 | 658b5a53146a / NVIDIA GeForce RTX 4090 |
| int4 s1 | 1 | 50 | 0 | 0.14 / 0.42 / 0.46 | 0.88 / 1.18 / 1.19 | 168.3 | 10292 |  |
| int4 s2 | 1 | 50 | 0 | 0.15 / 0.42 / 0.43 | 0.89 / 1.18 / 1.18 | 168.4 | 10292 |  |
| int4 s3 | 1 | 50 | 0 | 0.14 / 0.41 / 0.42 | 0.87 / 1.15 / 1.17 | 168.4 | 10292 |  |
| bf16 s1 (reference, unpooled: qwen2.5-vl-7b-bf16) | 1 | 50 | 0 | 0.11 / 0.34 / 0.39 | 2.10 / 2.33 / 2.35 | 63.3 | 19296 |  |
| int4 s1 (reference, unpooled: qwen2.5-vl-7b-awq) | 1 | 50 | 0 | 0.12 / 0.34 / 0.38 | 0.86 / 1.09 / 1.11 | 168.4 | 10168 |  |

Budgets (proposal 5.2): TTFT p95 ≤ 2 s {bf16: pass, int4: pass}; decode p50 ≥ 15 tok/s {bf16: pass, int4: pass}; INT4 peak memory reduction 46.9% (≥ 25%: pass). Pooled percentiles are over the concatenated requests of sessions sharing checkpoint, serve settings and host.

### T7. Harmful examples (55)

| Question | Condition | Cell | Views | BF16 margin | BF16 → INT4 |
| --- | --- | --- | --- | --- | --- |
| 1d4db80d-bd063ae8-b7da0bc714 | Fog | behavior/MCQ | 6 | 0.13 | B → A |
| 240d1548-e154fa0b-1bc4fccc56 | Fog | behavior/MCQ | 6 | 0.00 | B → A |
| 3032b651-6e349f86-1871b4106d | Fog | behavior/MCQ | 6 | 0.14 | B → C |
| 3032b651-c1446529-141aa38fac | Fog | behavior/MCQ | 6 | 0.14 | D → A |
| 3469d30a-011b8e75-4acece530a | Fog | behavior/MCQ | 6 | 0.05 | C → D |
| 37284f81-fcafd394-fd175ec751 | Fog | behavior/MCQ | 6 | 0.04 | C → B |
| 4ed62829-c16bf051-fb93e76780 | Fog | behavior/MCQ | 6 | 0.00 | A → C |
| 7e2d9f38-042f4c4b-d27bffe655 | Fog | behavior/MCQ | 6 | 0.09 | D → A |
| ab8bf18d-e2143daf-343405b5a3 | Fog | behavior/MCQ | 6 | 0.08 | B → A |
| c34206e6-10a4d313-6849916919 | Fog | behavior/MCQ | 6 | 0.04 | C → A |
| e323a27a-3c730292-62520a731f | Fog | behavior/MCQ | 6 | 0.11 | B → D |
| ef5f9458-cddb14d9-509e2d8e72 | Fog | behavior/MCQ | 6 | 0.04 | D → C |
| fb3aaad9-80153dcc-31375f4724 | Fog | behavior/MCQ | 6 | 0.11 | D → A |
| 0e247ba6-2700a5b2-61f2d7e302 | Fog | perception/MCQ | 1 | 0.06 | B → C |
| 4284a8d9-81bb5113-0a02cf102e | Fog | perception/MCQ | 1 | 0.09 | A → C |
| 78c5f7d7-4c4e1e94-7c5b6c0951 | Fog | perception/MCQ | 1 | 0.00 | B → C |
| 9f5f2211-3e1d882d-f26a2baf80 | Fog | perception/MCQ | 1 | 0.20 | B → C |
| b79d940d-dae47bb9-f793ae6b05 | Fog | perception/MCQ | 1 | 0.08 | A → C |
| bc4fd5a0-f317f943-9bffebf0bf | Fog | perception/MCQ | 1 | 0.04 | A → C |
| dcdea563-0f6432a5-67d468e53a | Fog | perception/MCQ | 1 | 0.05 | A → C |
| e15260c4-7a639c70-6a24bed67f | Fog | perception/MCQ | 1 | 0.05 | A → C |
| 37284f81-fcafd394-fd175ec751 | LowLight | behavior/MCQ | 6 | 0.07 | C → B |
| 73030fb6-8de7ec06-24e410c705 | LowLight | behavior/MCQ | 6 | 0.00 | C → D |
| c2d5a5e7-7a8188fd-1bc89a72e7 | LowLight | behavior/MCQ | 6 | 0.04 | C → B |
| e5a3df5f-474e5ee2-a6c3f000be | LowLight | behavior/MCQ | 6 | 0.00 | A → D |
| eac3102e-ee4ed593-439f888048 | LowLight | behavior/MCQ | 6 | 0.09 | B → A |
| 240d1548-e154fa0b-d49b35d24f | LowLight | perception/MCQ | 1 | 0.05 | B → C |
| 4284a8d9-81bb5113-0a02cf102e | LowLight | perception/MCQ | 1 | 0.04 | A → C |
| 57f5503b-27ec52e7-d84762f12f | LowLight | perception/MCQ | 1 | 0.04 | B → A |
| 78c5f7d7-4c4e1e94-7c5b6c0951 | LowLight | perception/MCQ | 1 | 0.16 | B → C |
| bc4fd5a0-f317f943-9bffebf0bf | LowLight | perception/MCQ | 1 | 0.08 | A → C |
| c6da5920-378b6469-df35ab0211 | LowLight | perception/MCQ | 1 | 0.16 | B → C |
| d0b2d916-f1999ab8-59ac357ab5 | LowLight | perception/MCQ | 1 | 0.00 | A → C |
| 16c90eed-9ffb0eec-d7caaa08c0 | NoImage | behavior/MCQ | 6 | 0.37 | C → D |
| 19640fe4-c217fc76-42fe2a9f2a | NoImage | behavior/MCQ | 6 | 0.14 | B → A |
| 3032b651-6e349f86-1871b4106d | NoImage | behavior/MCQ | 6 | 0.15 | B → A |
| 45275e70-a02918af-505a702f17 | NoImage | behavior/MCQ | 6 | 0.20 | B → A |
| 9ac6dff0-7f32be25-618b2e25be | NoImage | behavior/MCQ | 6 | 0.00 | A → D |
| e323a27a-3c730292-62520a731f | NoImage | behavior/MCQ | 6 | 0.25 | B → D |
| f9e460f0-8e8e071e-adeb3c3088 | NoImage | behavior/MCQ | 6 | 0.13 | B → A |
| 10dfdae8-2e3ff290-4cbef9750e | clean | behavior/MCQ | 6 | 0.04 | D → C |
| 4ed62829-c16bf051-fb93e76780 | clean | behavior/MCQ | 6 | 0.09 | A → C |
| 79770033-fbc07f14-d659520519 | clean | behavior/MCQ | 6 | 0.00 | B → D |
| befb9609-82828760-02a753b780 | clean | behavior/MCQ | 6 | 0.05 | A → D |
| c2d5a5e7-7a8188fd-1bc89a72e7 | clean | behavior/MCQ | 6 | 0.09 | C → B |
| c34206e6-10a4d313-6849916919 | clean | behavior/MCQ | 6 | 0.08 | C → A |
| eac3102e-ee4ed593-439f888048 | clean | behavior/MCQ | 6 | 0.14 | B → A |
| fb3aaad9-80153dcc-31375f4724 | clean | behavior/MCQ | 6 | 0.06 | D → A |
| fd2d419d-3c8a9403-5da3755847 | clean | behavior/MCQ | 6 | 0.05 | C → A |
| 1fbcc26e-03d203bc-e4cdeb1b70 | clean | perception/MCQ | 1 | 0.00 | A → C |
| 57f5503b-27ec52e7-d84762f12f | clean | perception/MCQ | 1 | 0.00 | B → C |
| 9f5f2211-3e1d882d-f26a2baf80 | clean | perception/MCQ | 1 | 0.05 | B → C |
| befb9609-82828760-6ea5d1e557 | clean | perception/MCQ | 1 | 0.05 | A → C |
| d90b94e8-786371ca-0a2bbe9b63 | clean | perception/MCQ | 1 | 0.00 | B → C |
| dcdea563-0f6432a5-67d468e53a | clean | perception/MCQ | 1 | 0.05 | A → C |
