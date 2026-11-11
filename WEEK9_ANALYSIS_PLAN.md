# Week-9 build plan, v2: integrity fixes, contrasts, two-stage freeze, measurements, judge, tables

**Date:** November 4, 2026 (v1 November 4, reviewed twice the same day). **Author:** Ronak. **Status:** v2 for approval. Implementation and the freeze wait until this version is accepted.

**Order:** code fixes and tests → core freeze → GPU measurements on frozen settings → analysis. The freeze locks decisions; completed measurements gate analysis, not the freeze.

## What changed from v1

The v1 review found that the November 3 pilot slices drew from the final holdout, that two of the four prespecified contrasts are the same comparison, that the freeze as designed did not bind the experiment that actually runs, that pairing accepts mismatched conditions, that the deployment measurements used the official checkpoint rather than the selected one, that the table spec contradicted itself, and that every judge model named in v1 is retired or retiring. v2 fixes each, adds provenance fingerprints and population checks, makes the statistics fully prespecified, splits the freeze into a core freeze and a later judge-protocol freeze, and lists acceptance criteria per item.

---

## 1. Holdout exposure: exclusion list and population counts

**Finding.** `03_smoke_drivebench.py` defaulted to `--split all`; the November 3 slices contain 16 questions from 15 final scenes (7 of 8 MCQ), including the scene of the harmful example cited in v1. Stability and profile slices were development-only.

**Remedy.**
- `manifests/scene_split.json` is kept unchanged (the random split is not regenerated). New file `manifests/scene_exclusions.json`: the 15 exposed scene tokens, the result files that exposed them, the date, and the reason. `qflip.split.final_scenes()` returns final minus exclusions; `filter_split(..., "final")` applies it. `filter_split(..., "final_exposed")` returns the 15 for exploratory use only.
- Confirmatory population (verified): 116 scenes, 838 eligible questions: behavior/MCQ 133, perception/MCQ 133, perception/VQA 133, planning/VQA 399, prediction/VQA 40 (37 scenes). 266 MCQ, 572 VQA. Judge calls for the 8-run matrix: 4,576.
- `03` default becomes `--split development`. The margin-bin choice is no longer justified by the exposed flips; the bins stay at the October 19 default (0.25, 0.5).
- PROTOCOL section 5 gets the exclusion row and a change-log entry.

**Acceptance.** `filter_split(qs, "final")` yields 116 scenes / 838 questions; a test asserts that no excluded scene is in the final set and that every excluded token is a final-split token; `03` without `--split` writes development-scene records only.

## 2. Provenance and population integrity

### 2.1 Per-run metadata sidecar

Every result file `results/<run>/<precision>_<condition>.jsonl` gets `<same>.meta.json`, written by `03`/`06` before the first record:

```
split, split_file_sha256, exclusions_sha256, annotations_sha256,
config_sha256 (config/pilot.yaml), prompt_sha256 (prompt.txt),
code_revision (git HEAD; dirty flag), precision, condition,
served_model_name, model_path_or_id, model_revision,
checkpoint_fingerprint (sha256 of config.json + sha256 of every safetensors file, or the HF commit for hub models),
serve_manifest (copy of manifests/serve_<precision>.txt at run start), vllm_version, created
```

**Resume rule.** `03`/`06` refuse to append to an existing result file when the sidecar is missing or any field other than `created` differs from the current run. One population per file; records outside the declared population are rejected, never filtered.

### 2.2 Final-run guard

- `run_matrix.sh` with `SPLIT=final`, and `03 --split final` directly, refuse unless `manifests/config_frozen.yaml` exists and `freeze_config.py --check` passes.
- After the server is up, the driver compares `manifests/serve_<precision>.txt` (model path or id, revision, quantization method, kv bytes, max_pixels, max_model_len, served name) and the checkpoint fingerprint to the frozen values and aborts on mismatch.
- `01_serve.sh bf16` passes `--revision` from config. The INT4 default stays the official checkpoint for pilots; the frozen config names the own checkpoint, and the guard enforces it for final runs.
- Server ownership: the driver requires the port to be free before launch, waits on `/health`, then checks `/v1/models` returns the expected served name and that the serving process is its own child; an already-running server fails the run.

### 2.3 Analysis-side validation (`05`)

- Reads each file's sidecar; refuses files without one, files whose sidecars disagree on split, exclusions, annotations, config, prompt, or checkpoint fingerprint (ref and quant may differ only in precision fields), and files whose records contain a scene outside the declared split minus exclusions.
- For final-split files: requires the frozen config and checks its own effective settings (margin edges, group and contrast draw counts, coverage minimum, support minimum, judge threshold if VQA) against it; a CLI override of a frozen value is an error, not a warning.

**Acceptance.** Tests: sidecar written and checked; append refused on mismatch; `05` refuses a file with an out-of-population scene; `05` refuses a final-split file without freeze; `run_matrix.sh` dry run shows the guard; a unit test for the serve-manifest comparison.

## 3. Pairing

`pair_records` and `pair_vqa_records` raise `PairingError` (never `assert`) when, for a question id present on both sides, any of these differ: condition, `mcq_gt`/`gt`, question text, scene token, frame token, cell, selected views, or when `precision` is not (`bf16`, `int4`). Legacy records without question text use the annotation lookup already in `08`, recorded in the pair as `question_source: annotations`.

**C4 common support.** Each condition contrast uses the question ids present, without error, in all four files (clean BF16, clean INT4, condition BF16, condition INT4). The counts of ids dropped per file are written to the report.

**Acceptance.** Tests: each mismatch raises with the field named; the clean-vs-NoImage misuse that produced F_h 0.4 now raises; C4 support counts match a hand-built fixture.

## 4. Statistics: five confirmatory contrasts

**Family.** Five MCQ harmful-rate (F_h) contrasts, fixed here before results from the remaining confirmatory population exist:

| # | Contrast | A | B | Population |
|---|---|---|---|---|
| C1 | margin | BF16 margin in `[0, 0.25)` | all other bins | clean condition, pairs with a margin |
| C2 | task | behavior/MCQ | perception/MCQ | clean condition |
| C4a | NoImage vs clean | NoImage pairs | clean pairs | common support (section 3) |
| C4b | LowLight vs clean | LowLight pairs | clean pairs | common support |
| C4c | Fog vs clean | Fog pairs | clean pairs | common support |

C3 (1 vs 6 views) is dropped from the family: in the final split every behavior MCQ has six views and every perception MCQ has one, so Δ_views = −Δ_task in every draw. It is reported as descriptive with that sentence. Clean-only scope for C1 and C2 is a protocol amendment (the proposal's clean-primary language did not specify contrast scope). Change rate, accuracy delta, VQA contrasts, pooled-condition versions of C1/C2, and all breakdowns are exploratory.

**Statistic.** Δ = F_h(A) − F_h(B).

**Resampling.** Scene-level paired bootstrap: draw scenes with replacement; every pair of a drawn scene enters on its side. One `random.Random(seed)` per contrast, seed 0. Draws: 20,000 for contrasts, 2,000 for group intervals, stored separately as `n_boot_contrast` and `n_boot_group`.

**Intervals.** Raw percentile 95% and Bonferroni-adjusted 99% (α/5), both stored as `(lo, hi)`. Confirmatory rule: the adjusted interval excludes zero. The family stays five whatever happens to any member.

**Support rule (prespecified).** A contrast is assessed only if all hold: ordinary coverage on both sides (≥ 30 questions, ≥ 10 scenes); reference-correct support on both sides (≥ 20 reference-correct pairs from ≥ 10 scenes); valid draws ≥ 50% of `n_boot_contrast` (a draw is invalid when either side has zero reference-correct pairs). Otherwise `assessed: false` with `reason` naming the failed condition, and the point estimate still printed.

**Schema** (`contrasts[]` in the `05` JSON):

```
name, a_label, b_label, population,
a: {n, scenes, ref_correct, ref_correct_scenes, harmful, harmful_rate},
b: {same},
delta, ci95, ci99_bonferroni, excludes_zero_raw, excludes_zero_adjusted,
n_boot_contrast, valid_draws, assessed, reason, support_dropped (C4 only)
```

**Acceptance.** Tests: planted effect (A F_h 0.5, B 0.0, 20 scenes) gives Δ 0.5 with both intervals excluding zero; null gives intervals containing zero; same seed reproduces; B with no reference-correct pairs gives `assessed: false`, `reason: "support"`, `valid_draws: 0`, and the family length stays five; views contrast absent from the family and present under `descriptive`. End to end on development-scene files.

## 5. Two-stage freeze

### 5.1 Core freeze (after sections 1–4 and 8 land and tests pass)

| PROTOCOL item | Frozen value |
|---|---|
| 1 INT4 | own AWQ W4A16-asym g128, fingerprint from `manifests/quantize_qwen2.5-vl-7b-awq-w4a16-g128.json` plus the checkpoint sha256s |
| 1 base revision | `cc594898…`, passed with `--revision` for BF16 |
| 2 versions | vLLM 0.33.0, torch 2.14.0+cu130, transformers 5.20.2 (`requirements_frozen.txt`); llm-compressor 0.15.0 |
| 2 serving | `max_pixels` 200704, `max_model_len` 4096, kv 2 GiB, `max_num_seqs` 1, seed 0, prefix caching on |
| 3 prompt/images | originals with token cap, MCQ suffix on, NoImage black 224×224 |
| 4 decoding | greedy, 128 tokens, logprobs top-5 |
| 5 population | final minus `scene_exclusions.json`: 116 scenes, 838 questions; coverage 30 questions / 10 scenes; support 20 reference-correct / 10 scenes |
| 6 conditions | clean, NoImage, LowLight, Fog |
| 7 VQA | **unassessed in the core study.** VQA scoring, if it happens, is governed by the judge protocol (5.2) |
| 8 contrasts | the five in section 4; margin edges 0.25, 0.5; draws 2,000 / 20,000 |
| 10 stability | 60 development questions × 3 repeats, clean, own checkpoint |
| 11 profile | three sessions per precision on the pinned host, own checkpoint, pooled per section 6 |
| 12 mitigation | unchanged, conditional on week 12 |

`scripts/freeze_config.py` writes `manifests/config_frozen.yaml` with `frozen: {date, git_commit, protocol_sha256, split_sha256, exclusions_sha256, annotations_sha256, prompt_sha256, checkpoint_fingerprint}`; refuses if `PROTOCOL.md` contains `TO DECIDE` outside the change log and the judge section, or if the working tree is dirty. `--check` compares live config and the fingerprints to the frozen file and exits non-zero on any difference. Wired into `run_matrix.sh`, `03 --split final`, and `05` (section 2).

### 5.2 Judge protocol freeze (separate, later)

Frozen only after development-scene validation (section 7): judge model and request settings, template sha, correctness threshold with its human-label validation, noise-floor numbers, resume key. Written as `manifests/judge_frozen.yaml` with its own date and change-log entry. No final-scene VQA judging before it exists. If it never happens, VQA stays unassessed and the 100-pair human-scored sample is the VQA evidence.

**Acceptance.** Tests for `freeze_config.py` on a temp repo: writes the header; `--check` passes unchanged, fails on a changed value, fails on a changed split file or checkpoint; refuses with a `TO DECIDE` remaining outside the allowed sections.

## 6. Measurements after the core freeze (one GPU session)

All on the pinned host (m:11851), own checkpoint, frozen settings:

1. Stability, both precisions (own INT4), 60 development questions × 3.
2. Profile sessions 1–3, both precisions. The November 3 official-checkpoint profile and stability files are kept as reference under `results/reference_official/` and excluded from pooling.
3. Matrix: `run_matrix.sh` on the final population, 8 runs.
4. Judge development trial (section 7) if an API key and budget exist by then.

**Pooling rule (`07 --report --out`).** Pool requests across sessions only when the sessions share precision, checkpoint fingerprint, serve settings, and host; reject duplicate session numbers; pooled percentiles are computed over the concatenated requests; per-session rows and the session count are kept in the output. `09` formats that JSON and computes nothing.

**Acceptance.** Three sessions per precision present and pooled; stability summaries for the own checkpoint; matrix sidecars pass `05` validation; freeze check passes on every run.

## 7. VQA judge: cost, compatibility, noise, threshold

**Facts (verified November 4 on OpenAI's pages).** `gpt-3.5-turbo` shut down October 23, 2026; `gpt-5-mini` shuts down December 11, 2026; `gpt-4o-mini` January 14, 2027. OpenAI names `gpt-5.6-terra` as the replacement. `gpt-5-mini` rejects requests that carry `temperature`; GPT-5.4 accepts it only with `reasoning_effort: none`; GPT-5.6 is not documented, so a compatibility call is required. Exact reproduction of the released judge is no longer available; a qualified comparison remains possible by rescoring baseline predictions with the same replacement judge, if those predictions can be obtained.

**Code changes (`08`, `qflip.judge`).**
- `temperature`, `top_p`, `reasoning_effort` optional in config; omitted when null. A `--compat-check` flag sends one call and reports accepted parameters and usage.
- Record carries `input_sha256` over the complete judge input and settings (template sha, question, ground truth, object description, prediction, model, temperature, reasoning effort, max tokens) and a `replicate` id (default 0). Resume key = question id + replicate + `input_sha256`; a changed answer, model, or template is a new key. Replicates 1..k are separate rows, never deduplicated.
- `--limit 20` trial writes measured `usage` tokens; the cost note uses measured numbers, not the chars÷4 estimate.

**Development validation, before the judge freeze.**
1. Judge noise on fixed answers: 40 development VQA answers judged 3 times each (replicates); report score spread and correctness flips at the candidate threshold.
2. Judge scores across repeated model answers: the three stability repeats' VQA answers (BF16 showed text variation in 6 of 35, INT4 in 1 of 35) judged once each; report correctness changes across repeats. This is the VQA noise floor; text variation alone is not.
3. Threshold: 100 development answers scored blind by two team members on the rubric's correctness criterion; threshold chosen to maximize agreement with the human labels; the proposal's 100-pair double review covers the final-scene check.

**Cost note (`manifests/judge_cost.md`).** 4,576 calls for the matrix (572 VQA × 8 runs), 2,288 more with mitigation; token counts from the trial; prices from the provider price page on the day, URL and date recorded; totals per candidate; recommendation. Team decides budget and model; the judge freeze records it.

**Acceptance.** Tests: optional parameters omitted when null; `input_sha256` changes when any input changes; replicate rows coexist; resume skips only exact-key matches. Trial output has measured usage.

## 8. Producer schemas and the report table generator

- `05` adds `coverage` (per cell: eligible in population, present on each side, errored, unpaired, assessed), `harmful_examples` (id, condition, cell, views, margin, BF16 letter, INT4 letter), `descriptive` (the views comparison), and `contrasts` (section 4), plus `provenance` (the validated sidecar fields).
- `07 --report --out` writes the pooled JSON (section 6).
- `09_tables.py` reads those and writes markdown tables T1 coverage, T2 headline per condition, T3 contrasts (both intervals, assessed, reason), T4 breakdowns, T5 stability (MCQ and VQA-judged when available), T6 deployment (pooled, with per-session rows), T7 harmful examples. It formats only.

**Acceptance.** Synthetic fixtures produce all seven tables; UNASSESSED rows keep their n and reason; end to end on development-scene files.

## 9. Corrections to `manifests/session1_2026-11-03.md`

- The flip paragraph mixes 24 pairs (F_h 0.083, three November conditions) with 32 pairs (F_h 0.0625, adding October LowLight on the old host). Report the 24-pair numbers; keep LowLight separate.
- "Zero self-changes" applies to the 25 MCQ. VQA text varied across repeats in 6 of 35 BF16 and 1 of 35 INT4 questions; added, with the note that correctness-level noise is measured in section 7.
- The profile and stability numbers are labeled official-checkpoint reference, not session 1 of 3.
- The exposed-scene finding and the exclusion list are recorded.

## 10. Build order and acceptance summary

| Step | Items | Done when |
|---|---|---|
| 1 | exclusions, 03 default, pairing errors, sidecars, final guard, serve revision | tests pass; `05` on November 3 files refuses the final-scene records and the mixed-condition pairing |
| 2 | contrasts, schema, support rule | tests pass; `05 --contrasts` on development files prints five rows plus the descriptive views row |
| 3 | 07 pooling, 05 coverage/examples, 09 tables | fixtures and development files produce tables. Done November 6 (section 12) |
| 4 | judge code changes, cost note skeleton | tests pass; `--compat-check` and `--limit 20` ready to run. Done November 5 (section 12) |
| 5 | `freeze_config.py`, PROTOCOL rows, session-note corrections | `freeze_config.py` writes the frozen file; `--check` passes; no TO DECIDE outside judge section and change log. Done November 6 offline (section 12); the host run writes the real freeze |
| 6 | core freeze | frozen file committed with fingerprints |
| 7 | GPU session | stability, profile ×3, matrix, judge trial; all sidecars validate |
| 8 | analysis | `05` on the matrix with the frozen settings; `09` tables |

## 11. Amendments recorded after review 3 (November 4)

1. **Compatibility versus history fields.** The sidecar's compatibility set is explicit (`qflip.provenance.COMPAT_FIELDS`: split and input hashes, config and prompt hashes, precision, condition, served name, model path or id, revision, checkpoint fingerprint, quantization, dtype, max_pixels, max_model_len, kv args, vLLM version). The serve manifest copy, its timestamp, code revision, dirty flag, and host are history only. An unchanged server restarted later resumes; a changed scientific setting refuses, naming the field. Across precisions, the precision fields are validated each against its own frozen model.
2. **Independent freezes.** The core check hashes `manifests/PROTOCOL_frozen.md` (an immutable snapshot taken at the core freeze), not the live `PROTOCOL.md`, and compares only the core config sections (never `judge`). Judge changes edit the live protocol and `manifests/judge_frozen.yaml` without touching the core check. `05 --vqa` on final-split files takes its threshold from `judge_frozen.yaml` and refuses a differing CLI value.
3. **Same validation at both entry points.** `qflip.guard.prepare_run` is called by `03` and `06` directly: live-server check (`/v1/models` must list the expected served name), and for `--split final` the freeze guard (frozen config present, config and inputs unchanged, serve manifest matching the frozen model, revision, quantization method, and serving settings). The matrix driver adds a port-free check so an already-running server cannot be reused. Tested: an official-checkpoint server is refused when the freeze names the own checkpoint.
4. **Judge record identity (step 4 scope).** The score loader will key on question id + replicate + `input_sha256`; production scoring consumes replicate 0 whose hash matches the frozen judge and the current answer; noise replicates stay separate. The hash covers the serialized request (model, all optional settings, template, question, ground truth, description, prediction).
5. **Counts.** After exclusions the raw annotation count is 971 (including 133 dropped prediction MCQ); 838 is the eligible count. The test asserts 116 scenes, 971 raw, 838 eligible.
6. **TO DECIDE scan.** `freeze_config.py` parses table rows and reads the status column; prose and the change log are ignored. Rows in section 7 (judge) and section 12 (mitigation, deferred under its frozen selection rule) do not block the core freeze.
7. **Open questions closed.** Support rule stays 20 reference-correct pairs from 10 scenes and is never lowered to rescue a contrast. Judge noise pilot: 40 answers × 3 replicates sampled across VQA cells and both precisions; cannot establish zero variability. Views comparison stays descriptive with the Δ_views = −Δ_task annotation and no confirmatory flag.

## 12. Step-1 audit (November 4, review 4) and status

The reviewer's targeted checks on the step-1 implementation found eight gaps; all were confirmed against the code and fixed with regression tests.

| # | Gap | Fix | Test |
|---|---|---|---|
| 1 | `05` on final files compared only the live config to the freeze; sidecars recording max_pixels 401408 or max_tokens 256 passed | `check_meta_against_frozen` runs per file: input hashes, core config hash, every serving setting, model identity | `test_final_analysis_checks_each_sidecar_against_freeze` |
| 2 | `--no-provenance` skipped validation for any file | refuses any file whose sidecar declares `final`; output labeled `exploratory`, printed as EXPLORATORY | `test_no_provenance_refuses_final_sidecars` |
| 3 | pairing accepted (int4, bf16), swapping harmful and beneficial | `check_pair` requires the order (reference=bf16, quantized=int4) | `test_pairing_requires_reference_then_quantized_order` |
| 4 | guard checked the config endpoint while the client honored `QFLIP_BASE_URL`; alias did not establish checkpoint identity; no process ownership | guard uses the client's effective endpoint; `/v1/models` `root` compared to the manifest model when the server reports it; `01_serve.sh` records pid and port; guard requires that pid alive with a command line naming the model and port, and the port equal to the client's; matrix driver checks manifest pid equals the started server and the listener is that process | `test_check_server_uses_model_root_not_only_alias`, `test_prepare_run_binds_manifest_to_process_and_client_endpoint` |
| 5 | SEED and MAX_NUM_SEQS overridable without record; prefix caching implicit | manifest records seed, max_num_seqs, prefix_caching; `01_serve.sh` passes the prefix-caching flag explicitly; all three are compatibility fields and freeze-checked | item-1 test, seed/max_num_seqs/prefix_caching block |
| 6 | resume checked metadata only | existing records must carry the sidecar's precision and condition and a scene inside the population | `test_resume_inspects_existing_records` |
| 7 | full-file config hash broke resume on judge-only edits | `core_config_sha256` over the core sections is the compatibility field; the full-file hash is history | `test_judge_only_config_change_keeps_core_compatibility` |
| 8 | custom `--snapshot` not fingerprinted; null hash passed `--check` | snapshot path stored in the freeze, hash required at build, checked by `--check` | `test_freeze_requires_and_checks_the_protocol_snapshot`, `test_freeze_config_cli_custom_snapshot_is_fingerprinted` |

Follow-up audits (same day) found three remaining defects, then an address gap in F1, a too-broad skip in the socket test harness (exits other than a permission error now fail with stderr), and finally silently dropped invisible socket owners plus family-blind wildcards in F1; all confirmed and fixed (52 tests):

| # | Gap | Fix | Test |
|---|---|---|---|
| F1 | pid check proved arguments, not listener ownership; port matched as a substring (81 matched 8100); missing manifest port skipped the check; driver continued without a listener pid | guard parses the endpoint port exactly, requires pid and port lines, finds the process holding the LISTEN socket serving the endpoint's address and port through `/proc/net/tcp*` and `/proc/<pid>/fd` (wildcards by family: `0.0.0.0` for IPv4, `::` for both; a matching socket with no visible owner refuses; non-loopback endpoints refused; several owners refused as ambiguous), and requires it to be the manifest pid or its descendant; unverifiable ownership refuses; driver fails without a listener pid | `test_prepare_run_requires_listener_ownership` (real listener passes; same-argv sleeper refused), `test_listener_ownership_matches_endpoint_address` (`/proc` fixture: two pids on one port at different addresses, wildcard, IPv6, nonlocal host, ambiguity), `test_listener_pid_finds_the_owning_process`, `test_url_port_is_parsed_exactly` |
| F2 | BF16-labelled server with `--dtype float16` passed both checks as confirmatory | freeze records dtype per model (bfloat16 / float16); serve-manifest check and per-file analysis check enforce it | `test_reference_dtype_is_frozen_and_enforced` |
| F3 | resume validated deduplicated records; analysis did not check rows against the sidecar | `check_records_against_meta` validates every raw row (precision, condition, scene) before deduplication; shared by `ensure_meta` and `05` | `test_raw_rows_validated_before_dedup_on_resume_and_analysis` |

Open point for the next GPU session: the `root` field of vLLM's model card is the loaded model path in the versions used so far, but this has not been observed on vLLM 0.33.0 on the new host. If it is absent the guard falls back to the pid binding alone; if it is present with a different form (for example a resolved cache path for a hub id) the comparison must be adjusted before the final run. Record the observed card in the session note.

Step 1 was committed after the reviewer's sign-off. Step 2 (contrasts) is implemented: `qflip.contrasts` (`bootstrap_contrast`, `prespecified_contrasts`, `common_support`, support rule), `05 --contrasts` and `--n-boot-contrast`, the schema of section 4 plus `reason_detail`, the descriptive views row with its note, `n_boot_contrast` and the support minimums in the effective settings checked against the freeze on final files; PROTOCOL sections 8 and 9 rewritten. Tests: planted effect, null, seed reproducibility, support (reason `support`, `valid_draws` 0, family length five), coverage, valid-draws rule, common support with dropped counts, views absent from the family. End to end on the pilot files (all five unassessed for coverage at n = 8, as expected).

Step-2 review (November 4) found three correctness issues and two reporting corrections; all fixed with tests (61 tests): duplicated inputs (the same file twice, or the same (condition, question) in two file pairs) are refused before support is computed; the bootstrap seed now lives in config `analysis.seed`, is in the effective settings, and is checked against the freeze on final files together with the contrast draw count and support minimums; a contrast below half valid draws reports its intervals and exclusion flags as unavailable; draw counts must be positive integers; the descriptive views row carries no Bonferroni interval, no adjusted decision flag, and no decision marker; the C4 common-support audit counts ids dropped per input file (all four), taking each file's identity from its validated sidecar or, for legacy files, from any record including errored ones, so a file whose MCQs all errored keeps an empty id set; the interval rule is applied inside `bootstrap_contrast` by valid draws alone, so intervals are unavailable below half valid draws whichever support condition is reported first (62 tests). Protocol wording changed to "before results from the remaining confirmatory population exist".

Step 2 was committed after the reviewer's sign-off. Step 4 (judge code, November 5) is implemented: `qflip.judge.judge_settings` drops null optional settings (`temperature`, `top_p`, `reasoning_effort`, `seed`) so they are omitted from the request, with the length parameter name in `judge.max_tokens_param`; `request_body` is the request as sent and `input_sha256` hashes its canonical serialization; every record carries `replicate`, `input_sha256`, `judge_settings` and the provider's measured `usage`; `load_scores` keys on (question_id, replicate, input_sha256) with last-wins only within a key; `production_scores` selects, per current answer, the replicate-0 record whose hash matches under the configured or frozen settings and names the rest stale (changed answer, settings or template, replicate 0 missing, pre-step-4 record); `05 --vqa` uses only those, taking the settings from the `judge` block of `judge_frozen.yaml` on final files; `08 --compat-check` sends the configured request once and, if rejected, probes the baseline (`max_tokens`, then `max_completion_tokens`) and each optional setting alone; a 4xx is not retried; `--replicate` writes separate rows. `config/pilot.yaml` sets `judge.model` to null until the team decides. Tests: null settings omitted, hash changes for each of twelve input or setting variations, replicate rows coexist and resume skips only the exact key, production selection, measured usage, compat probing (67 tests). `manifests/judge_cost.md`: call counts (4,576 matrix; 7,330 with mitigation and validation), provisional token budget from the dry run (1,100 prompt, 250 completion; to be replaced by the trial's measured usage), prices from the provider page read November 5 for seven candidates, compatibility table and recommendation left for the trial.

Step-4 review (November 6) found four correctness issues, all confirmed and fixed with tests (73 tests): (1) `08` judged final-split files from the live config, with `--model` honored and no judge freeze; it now runs the same provenance and population validation as `05` before any call (`qflip.guard.prepare_judging`; `--no-provenance` only for sidecar-less pilot files), and final-split files are judged only with the frozen judge, any differing CLI or config value refused. (2) The judge freeze did not bind the rubric: `build_judge_frozen` / `check_judge_frozen` (and `scripts/freeze_judge.py`) record and verify the sha of the system line and of every template besides the canonical settings and threshold; `08` and `05 --vqa` refuse final files when the live code differs. (3) `production_scores` tried every description stored in historical rows and took the first matching hash, so with two rows for the same answer the selected score depended on dictionary order; it now derives the one expected description from the description file, whose sha256 mismatch is an error rather than a warning, and matches that exact input. (4) The trial summary aggregated usage over every historical key of the replicate; usage is now reported for the invocation's own calls (`invocation_usage`) and the score summary covers the current keys only. Cost note corrected: gpt-4.1-mini also shuts down January 14, 2027 (deprecations page).

Follow-up review (same day) found four remaining gaps, all confirmed and fixed with tests (73 tests): (1) `--no-provenance` accepted files with development sidecars and a `split: all` sidecar bypassed the judge freeze; the legacy option now covers sidecar-less files only, and every non-final path (legacy, `all`, development) checks the actual rows against the unexposed holdout scene set in both `08` and `05` (`qflip.guard.holdout_check`). (2) `judge.base_url` was recorded but not enforced and a null value let the SDK read `OPENAI_BASE_URL`; the effective endpoint is now part of the settings (default `https://api.openai.com/v1`), passed to the client explicitly, compared by the freeze check and hashed into `input_sha256`. (3) A rejected configured request on a final file was followed by fallback probes carrying the final answer with altered settings; final-split probes now stop at the first rejection (`fallback=False`), fallback probing runs only on development or synthetic prompts. (4) The description-file digest was absent from the freeze, so updating the file and the recorded constant together passed; `visual_description_sha256` is now a frozen fingerprint.

Step 4 was committed after the reviewer's sign-off. Step 3 (November 6) is implemented: `qflip.report.coverage_by_cell` (per condition and cell: eligible in the declared population or n/a, present and errored per side, paired, unpaired, scenes, coverage verdict) and `harmful_examples` (id, scene, condition, cell, views, BF16 margin, both option letters or both judge scores), both written by `05 --out` with `reason` on every unassessed group; `Pair` carries the two strict-rule letters. `07` writes a provenance block (`build_run_meta` fields) and the host into every session file, and `07 --report --out` pools through `qflip.report.pool_profiles`: same precision, checkpoint fingerprint, every compatibility field and host, distinct session numbers, percentiles over the concatenated requests, per-session rows and count kept, memory reduction from the pooled maxima, the three budgets evaluated; files without a provenance block are refused unless passed as `--reference`, which lists them unpooled. `qflip.tables` renders T1 coverage, T2 headline per condition (MCQ and judged VQA), T3 contrasts (both intervals, status, reason with detail, descriptive rows, C4 audit), T4 breakdowns, T5 stability (VQA columns n/a until judged repeats exist), T6 deployment (pooled and per-session rows, reference rows, budgets), T7 harmful examples; `scripts/09_tables.py` is the CLI and computes nothing. Tests: producers from records, pooling (concatenated percentiles, refusals for duplicate session, other fingerprint, other host, other serve setting, missing provenance; reference listed unpooled), all seven tables from fixtures with unassessed rows keeping n and reason, degradation without inputs (78 tests). End to end: the pilot files through `05 --contrasts --out` and `09` (T1 with n/a eligibility, one harmful example `D → A` under NoImage); synthetic sessions through `07 --report --out` and `09`.

Step-3 review (November 6) found five issues, two visible in the pilot tables; all confirmed and fixed with tests (80 tests): (1) coverage identified a file pair's condition through question ids, which repeat across conditions, so every condition collapsed into clean and a condition with all INT4 answers errored could show paired questions; `coverage_report` now keys conditions by the validated file identity (the pair's two files must carry the same condition) and uses that condition's pairs. (2) `05` wrote per-condition groups only with `--by condition` and the formatter read their absence as one condition; `05` now always writes them and T2 reports their absence instead. (3) `07` measured first and copied the local manifest afterwards; it now runs `qflip.guard.prepare_measurement` before the warm-up requests: the same server binding as a run (alias at the client's endpoint, listener ownership, loaded model) and the core freeze for study sessions, `--exploratory` skipping only the freeze and marking the file. (4) Pooling accepted placeholder provenance and mixed workloads, and compared precisions measured with different KV caches or hosts; sessions now need every compatibility field, a complete host block and the workload (split, condition, slice size, views, warm-up, the question ids measured), exploratory sessions are refused, and the cross-precision comparison is refused unless every non-precision setting, host and workload agree. (5) An all-error session crashed the report in numeric formatting; percentiles render as n/a, counts stay, and budget verdicts are unavailable rather than false. Corrections: T3 takes family size and adjusted alpha from the producer's `contrast_settings`; T6 lists reference rows when the input holds references only. End to end: the pilot files now give four T1 conditions and four T2 rows; a Fog file with every INT4 MCQ errored gives present 4/0, paired 0.

Follow-up review (same day) found three more: (1) the completeness check refused the producer's own INT4 metadata, whose `model_revision` is null because a local checkpoint is identified by its sha256 fingerprint; the revision is now optional for `sha256:` fingerprints and required for Hub models, tested with metadata from the real producer. (2) Sessions without the study flag (files from the previous collector) entered the pool; `study: true` is now required, unmarked and exploratory sessions go through `--reference`. (3) A condition with zero pairs vanished from T2, and `05` exited before writing coverage when no pair existed at all; `condition_rows` adds an n = 0 unassessed row with its reason for every validated input condition, and `05` continues with empty pairs. T6 labels references neutrally, adding the model identity when the file carries one. A last gap: the VQA section was written only when a usable pair existed, so with every judge record stale the headline had no VQA rows; `05` now writes the section whenever `--vqa` is given, with the n = 0 rows, covered by an integration test that runs `05` on the pilot answers with twelve stale judge records per side.

Step 3 was committed after the reviewer's sign-off. Step 5 (November 6, offline part): every PROTOCOL row outside sections 7 and 12 is FROZEN with the values of section 5.1 (`max_pixels` 200704, prefix caching on, originals with the token cap, coverage 30 / 10, versions as captured, `max_model_len` 4096, MCQ suffix, `max_tokens` 128, the own INT4 checkpoint, conditions, stability and profile designs); team items 1 (conditions) and 5 (image handling) adopt this plan's reviewed defaults, recorded in the change log as adopted defaults rather than an explicit team approval, so the team can still object before the host freeze. The TO DECIDE scan now reports only the section-7 VQA row. The session note carries the four corrections of section 9 (24-pair flip numbers with LowLight separate, VQA text variation in the stability repeats, official-checkpoint reference labels, the exposed-scene finding and the exclusion list) and an updated host checklist. `freeze_config.py` was dry-run against a stub checkpoint in a scratch directory: it writes the frozen file and the snapshot, `--check` passes, and `--check --serve-manifest` against the November 3 official-checkpoint manifest fails as it must (that manifest predates the seed and max_num_seqs lines, so every study server is started by the current `01_serve.sh`). The real freeze is written on the host, where the own checkpoint lives, as the first step of the GPU session.

Step-5 review (November 6) found five issues, all confirmed and fixed (83 tests): (1) the protocol promised version checks that no code performed; the freeze now records the pins of vllm, torch, transformers and compressed-tensors from `manifests/requirements_frozen.txt` with the file's hash, `01_serve.sh` records all four in the manifest, every sidecar carries them as compatibility fields (resume, analysis and pooling compare them), the freeze check requires every one reported and equal to the pin (the first fix had covered only vLLM and transformers; the follow-up review caught torch and compressed-tensors, with per-package tests at the manifest, sidecar, final-analysis and pooling boundaries), and the quantization environment is recorded from the quantize manifest (`check_versions`, `pinned_versions`). A further follow-up: the host reports torch `2.14.0+cu130` while the pin says `2.14.0`, so an exact comparison to the pin rejected the documented runtime; the freeze now records the exact installed builds (`builds`, from the serving venv, a serve manifest, or the pins for dry runs), requires each build's public version to equal its pin, and compares every manifest and sidecar to the build exactly, so `2.14.0+cu130` passes while `2.14.0`, `2.14.0+cu128` and `2.14.1+cu130` fail. Last gap: `--runtime auto` read the distribution metadata (2.14.0) while the serve script read the module (2.14.0+cu130); both now go through one collector, `qflip.versions` (module `__version__`), the serve script calling `scripts/runtime_versions.py`, verified in an isolated stub environment where the metadata lacks the tag the module reports: the manifest lines and `--runtime auto` agree and the auto-written freeze accepts the manifest (85 tests). (2) `freeze_config.py` verified the git state, wrote the snapshot, and then `build_frozen` looked again and recorded `git_dirty: true`; the verified pre-write state is now passed in. (3) The November 3 profile and stability files were tracked, so moving them would have dirtied the tree, and `results/reference_official/` was ignored; the move is now a tracked change and the directory is allowed; the host checklist sets `INT4_MODEL` explicitly for every INT4 server. (4) The frozen INT4 serving row named `awq_marlin`; it now names `compressed-tensors`. (5) The coverage row claimed prediction/VQA fails the rule; it clears 30 / 10 and is unassessed only because VQA scoring waits on the judge protocol. README's November 3 bullet carries the corrected 24-pair and reference wording.

GPU session 2 (November 7–8, `manifests/session2_2026-11-08.md`) completed steps 6 and 7 and ran step 8's analysis: freeze written and committed on the host; `/v1/models` `root` observed as the loaded model path for both servers (open point closed); stability floor zero for both precisions with the own checkpoint; three profile sessions per precision pooled under the rule of section 6 (all budgets met, memory −46.9%); the 8-run matrix with zero errors; `05 --contrasts` confirmatory: C1 excludes zero after adjustment (Δ +0.319, 99% [+0.140, +0.512]), C2 and C4a–c do not, every contrast assessed with full support. Judge trial on 40 development answers: three candidates scored 40 of 40 each. The local gpt-oss-20b judge exposed two protocol facts recorded in section 7 of the PROTOCOL: a reasoning judge needs a larger token budget and reduced effort, and the released score regex misses markdown, so extraction now has recorded fallbacks. DeepSeek V4 flash and pro through the team's proxy (after a first attempt failed on the upstream balance) need `reasoning_effort: none` (thinking off, verified through the proxy; the default and `low` return no score within 2048 tokens); projected matrix cost from measured tokens $2.97 and $9.99 unweighted ($2.66 and $8.82 weighted by the matrix cell mix); flash and pro differ by 5–6.5 points on average, the local judge is lower than either by 5–12 points on average but not on every answer. Numbers in `manifests/judge_cost.md`; the decision is the team's. The reviewer's pass on the session (November 8) found the released score regex reading the first term of an unbolded sum, the extractor unbound by the judge freeze, and no backend identity in the records; fixed with tests; a second pass (November 11) found the parser still accepting empty sides, trailing operators, unbalanced parentheses and truncated decimals, backend metadata recorded but never compared at freeze, resume or analysis, local identity checked against a text file only (and IPv6 loopback not recognised as local), a stored score surviving an empty reply, and regex flags missing from the extractor digest; all fixed: extractor version 4 accepts a complete expression or nothing, one predicate (`record_backend_status`) is shared by resume, freeze evidence and production scoring, a local judge's running server is checked like the study servers, no reply means no score, and the digest covers every regex's pattern and flags. A third pass found resume and freeze evidence still trusting the stored score field and a limited re-judging summary mixing backends; `usable_score` (backend, then the reply re-read under the current extractor) is now the single criterion in resume, evidence, summary and analysis, the summary reports stale and pending rows separately, and `scripts/host_pull.sh` replaces the host checklist (93 tests).

Remaining: judge decision and freeze (40 × 3 noise test, 100-answer human agreement, threshold), VQA judging of the matrix, the report from `results/final/tables.md`, mitigation if time permits.
