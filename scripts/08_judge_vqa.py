"""Score VQA answers with the DriveBench rubric judge and write a sidecar file
next to each result file:

    results/<run>/bf16_clean.jsonl  ->  results/<run>/bf16_clean.judge.jsonl

    export JUDGE_API_KEY=...                        # per session; never stored (OPENAI_API_KEY also accepted)
    python scripts/08_judge_vqa.py --compat-check --model deepseek-chat --base-url https://api.deepseek.com/v1
    python scripts/08_judge_vqa.py --compat-check --model judge-local --base-url http://localhost:8001/v1   # scripts/serve_judge.sh
    python scripts/08_judge_vqa.py --compat-check                                   # one probe call: accepted settings, usage
    python scripts/08_judge_vqa.py results/dev/bf16_clean.jsonl --dry-run           # count, prompts, no calls
    python scripts/08_judge_vqa.py results/dev/bf16_clean.jsonl --limit 20          # cost trial: measured usage per call
    python scripts/08_judge_vqa.py results/dev/bf16_clean.jsonl results/dev/int4_clean.jsonl
    python scripts/08_judge_vqa.py results/dev/*_clean.jsonl --replicate 1          # noise study: separate rows
    python scripts/08_judge_vqa.py results/final/*_*.jsonl --workers 8

Judge model and request settings come from config `judge:` (temperature,
top_p, reasoning_effort, seed are sent only when non-null; the length
parameter name is `judge.max_tokens_param`). Every record carries the model,
the template hash, the settings, `input_sha256` over the serialized request,
and a `replicate` id. Resumable: a record whose (question_id, replicate,
input_sha256) key already has a score is skipped; a changed answer, model,
template or setting is a new key; errored keys are retried.

Prompts are the released templates verbatim (src/qflip/judge_prompts.py); the
planning template needs the object visual description from the toolkit's
data/visual_description.json, fetched on first use.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.config import load_config  # noqa: E402
from qflip.drivebench import load_questions  # noqa: E402
from qflip.freeze import FreezeError  # noqa: E402
from qflip.guard import check_judge_server, prepare_judging  # noqa: E402
from qflip.judge import (  # noqa: E402
    backend_description,
    build_prompt,
    compat_check,
    ensure_visual_description,
    expected_description,
    EXTRACTOR_VERSION,
    input_sha256,
    invocation_usage,
    is_local_endpoint,
    JudgeInputError,
    JudgeSettingsError,
    load_scores,
    load_visual_description,
    prompt_template,
    record_backend_status,
    request_body,
    request_judge,
    rescore,
    sidecar_path,
    template_sha,
    usable_score,
    VQA_CELLS,
)
from qflip.provenance import ProvenanceError  # noqa: E402
from qflip.records import append_record, read_records  # noqa: E402

PROBE_PROMPT = build_prompt("prediction/VQA", "Would the ego vehicle stop?", "Yes, the ego vehicle stops for the pedestrian.", "Yes.", None)


def build_jobs(records: list[dict], desc_data: dict | None, cells: tuple[str, ...], settings: dict, replicate: int, question_text: dict[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """Return (jobs, skipped). A job is a record plus its request body, hash and
    key; skipped records carry a reason (wrong cell, errored answer,
    description missing). question_text maps question_id -> question for
    records written before the record schema carried the question."""
    jobs, skipped = [], []
    for r in records:
        cell = r.get("cell")
        if cell not in cells:
            continue
        if r.get("error"):
            skipped.append({"question_id": r["question_id"], "reason": "answer errored"})
            continue
        if not r.get("question"):
            if not question_text or r["question_id"] not in question_text:
                skipped.append({"question_id": r["question_id"], "reason": "question text missing"})
                continue
            r = {**r, "question": question_text[r["question_id"]]}
        try:
            desc = expected_description(desc_data, r, r["question"])
        except (KeyError, ValueError, JudgeInputError) as e:
            skipped.append({"question_id": r["question_id"], "reason": f"description lookup: {e}"})
            continue
        body = request_body(settings, build_prompt(cell, r["question"], r["gt"], r["pred"], desc))
        h = input_sha256(body, settings["base_url"])
        jobs.append({"rec": r, "body": body, "desc": desc, "input_sha256": h, "key": (r["question_id"], replicate, h)})
    return jobs, skipped


def judge_record(job: dict, settings: dict, replicate: int, res: dict, backend: dict | None = None) -> dict:
    r = job["rec"]
    return {
        "question_id": r["question_id"],
        "scene_token": r["scene_token"],
        "cell": r["cell"],
        "condition": r.get("condition"),
        "precision": r.get("precision"),
        "replicate": replicate,
        "input_sha256": job["input_sha256"],
        "judge_model": settings["model"],
        "judge_settings": settings,
        "judge_backend": backend or {},
        "judge_template": prompt_template(r["cell"])[0],
        "judge_template_sha": template_sha(r["cell"]),
        "desc": job["desc"],
        **res,
    }


def reparse_file(path: Path) -> dict:
    """Re-read every stored reply of the file's sidecar under the current
    extractor, in place, without any call: scores, rules, extractor version
    and error are rewritten, replies and everything else kept."""
    out_path = sidecar_path(path)
    if not out_path.exists():
        print(f"{path}: no sidecar {out_path}")
        return {"file": str(path), "records": 0}
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    new = [rescore(r) for r in rows]
    changed = sum(1 for a, b in zip(rows, new) if a.get("score") != b.get("score") or a.get("score_rule") != b.get("score_rule"))
    gained = sum(1 for a, b in zip(rows, new) if a.get("score") is None and b.get("score") is not None)
    lost = sum(1 for a, b in zip(rows, new) if a.get("score") is not None and b.get("score") is None)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new))
    print(f"{out_path}: {len(rows)} records re-read under extractor {EXTRACTOR_VERSION}: {changed} changed (score or rule), {gained} newly scored, {lost} no longer scored")
    return {"file": str(path), "records": len(rows), "changed": changed, "gained": gained, "lost": lost}


def judge_file(path: Path, settings: dict, client, desc_data, cells, workers: int, limit: int | None, dry_run: bool, replicate: int, question_text: dict[str, str], backend: dict | None = None, expected_backend: dict | None = None) -> dict:
    records = read_records(path)
    out_path = sidecar_path(path)
    existing = load_scores(out_path)
    # resume counts a record as done only when it counts as a score now: written under the
    # expected backend (description, and the frozen response model on final files) with a
    # reply the current extractor parses; the same test the judge freeze and production
    # scoring apply (usable_score). The stored score field is not the criterion.
    done = {k for k, r in existing.items() if usable_score(r, expected_backend)[1] is None}
    jobs, skipped = build_jobs(records, desc_data, cells, settings, replicate, question_text)
    todo = [j for j in jobs if j["key"] not in done]
    scored_now = [j for j in jobs if j["key"] in done]
    other_keys = {k[0] for k in done} - {j["key"][0] for j in scored_now}
    if limit:
        todo = todo[:limit]
    print(f"{path}: {len(jobs)} judgeable, {len(scored_now)} scored under the current key, {len(todo)} to judge, {len(skipped)} skipped -> {out_path}")
    if other_keys:
        print(f"  {len(other_keys)} question(s) have scores only under other keys (other replicate, answer, model or settings); kept, not reused")
    for s in skipped[:5]:
        print(f"  skip {s['question_id']}: {s['reason']}")
    if dry_run:
        if todo:
            j = todo[0]
            prompt = j["body"]["messages"][0]["content"]
            print(f"  example prompt ({j['rec']['cell']}, {len(prompt)} chars, key {j['key'][0]}/{j['key'][1]}/{j['key'][2][:12]}):\n" + "\n".join("    " + line for line in prompt.splitlines()[-8:]))
        est = sum(len(j["body"]["messages"][0]["content"]) for j in todo) // 4
        print(f"  dry run: ~{est} prompt tokens for {len(todo)} calls (4 chars/token estimate; the cost note uses measured usage from --limit runs)")
        return {"file": str(path), "to_judge": len(todo), "skipped": len(skipped)}

    def one(j):
        res = request_judge(client, j["body"])
        rec = judge_record(j, settings, replicate, res, backend)
        why = record_backend_status(rec, expected_backend) if res.get("reply") else None
        if why:
            # final files: a reply from another backend than the frozen one is kept but never
            # scored, now or under any later --reparse
            rec.update(score=None, score_rule=None, error=why, rejected=why)
        return rec

    written = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in tqdm(ex.map(one, todo), total=len(todo)):
            append_record(out_path, rec)
            written.append(rec)

    # Score summary over the current keys only (this replicate, these answers,
    # these settings), each record passed through the same validity test as
    # resume, freeze evidence and production scoring: a record under the current
    # key that is not usable (other backend, rejected, unparseable or empty
    # reply) is stale, a key with no record yet (--limit) is pending. Usage is
    # over this invocation's calls only. Records under other keys stay in the
    # file but enter neither.
    scores = load_scores(out_path)
    by_cell, stale, pending = {}, {}, 0
    for j in jobs:
        rec = scores.get(j["key"])
        if rec is None:
            pending += 1
            continue
        live, why = usable_score(rec, expected_backend)
        if why:
            stale[why] = stale.get(why, 0) + 1
        else:
            by_cell.setdefault(rec["cell"], []).append(live["score"])
    usage = invocation_usage(written)
    seen = sorted({str(r.get("response_model")) for r in written if r.get("reply")})
    print(f"  current key (replicate {replicate}, {settings['model']}): scored {sum(len(v) for v in by_cell.values())}, stale {sum(stale.values())}, pending {pending}; response model(s) this invocation: {', '.join(seen) or 'none'}")
    for why, n in sorted(stale.items()):
        print(f"    stale {n}: {why}")
    for cell, v in sorted(by_cell.items()):
        print(f"  {cell:15s} n={len(v):4d} mean={sum(v)/len(v):6.1f}")
    if usage["with_usage"]:
        print(f"  this invocation: {usage['calls']} calls, {usage['errors']} errors; measured usage per call (mean over {usage['with_usage']}): prompt {usage['prompt_tokens_mean']}, completion {usage['completion_tokens_mean']}, reasoning {usage['reasoning_tokens']} total")
    return {"file": str(path), "scored": sum(len(v) for v in by_cell.values()), "stale": sum(stale.values()), "pending": pending, "usage": usage}


def run_compat_check(client, settings: dict, prompt: str, fallback: bool) -> int:
    print(f"compat check for {settings['model']} at {settings['base_url']}: {json.dumps({k: v for k, v in settings.items() if k not in ('model', 'base_url')})}" + ("" if fallback else "  (final-split prompt: no fallback probes)"))
    rows = compat_check(client, settings, prompt, fallback=fallback)
    for r in rows:
        status = "accepted" if r["accepted"] else "REJECTED"
        extra = f" usage={json.dumps(r['usage'])} latency={r['latency_s']}s score={r['score']}" if r["accepted"] else f" error={r['error']}"
        print(f"  {status:8s} {r['params']}{extra}")
    print(json.dumps({"model": settings["model"], "settings": settings, "attempts": rows}, indent=1))
    return 0 if rows[0]["accepted"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="result .jsonl files from 03/06")
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--model", help="override judge.model from config")
    ap.add_argument("--base-url", help="override judge.base_url from config (any OpenAI-compatible endpoint; refused on final-split files when it differs from the frozen judge)")
    ap.add_argument("--desc", default="data/visual_description.json")
    ap.add_argument("--cells", nargs="*", default=list(VQA_CELLS), help="cells to judge (default: the three VQA cells)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="judge at most N new questions per file (cost trial; usage is measured)")
    ap.add_argument("--replicate", type=int, default=0, help="replicate id written into each record (noise study: 1, 2, ...); rows never deduplicate across replicates")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and count; no API calls, no key needed")
    ap.add_argument("--compat-check", action="store_true", help="send one probe request with the configured settings and report which are accepted, with usage")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--exclusions-file", default="manifests/scene_exclusions.json")
    ap.add_argument("--no-provenance", action="store_true", help="pilot files written before sidecars existed; never for final-split files")
    ap.add_argument("--reparse", action="store_true", help="re-read every stored reply in the files' sidecars under the current extractor, in place; no call, no key")
    ap.add_argument("--judge-manifest", default="manifests/serve_judge.txt", help="serve manifest of a local judge (checked against the frozen revision on final files)")
    args = ap.parse_args()

    if args.reparse:
        if not args.files:
            sys.exit("--reparse needs result files")
        for f in args.files:
            reparse_file(Path(f))
        return

    cfg = load_config(args.config)
    if args.base_url:
        cfg["judge"]["base_url"] = args.base_url
    if args.replicate < 0:
        sys.exit("--replicate must be 0 or a positive integer")
    if not args.files and not args.compat_check:
        sys.exit("no result files given (or use --compat-check)")
    # Provenance, population and the judge settings the files may be judged with,
    # before any call: final-split files need the core freeze and the judge
    # freeze, and refuse --model or config values that differ from it.
    try:
        prep = prepare_judging(args.files, cfg, cli_model=args.model, no_provenance=args.no_provenance, split_file=args.split_file, exclusions_file=args.exclusions_file)
    except (ProvenanceError, FreezeError) as e:
        sys.exit(str(e))
    except JudgeSettingsError as e:
        sys.exit(f"{e} (team item 2: pass --model or set config judge.model)")
    settings, prov, backend = prep["settings"], prep["provenance"], prep["backend"]
    if args.files:
        print(f"provenance ok: split={prov['split']} ({prov['analysis_class']}), {len(prov['files'])} files" + (f", judge frozen {prep['judge_frozen']['date']}" if prep["judge_frozen"] else ""))
    # resume, freeze evidence and production scoring share one predicate: a record counts
    # only under the expected backend description; on final files also the frozen response model
    expected_backend = dict(backend)
    if prov["split"] == "final" and is_local_endpoint(settings["base_url"]) and not args.dry_run:
        # a local judge is bound to its running server (manifest revision, served alias,
        # loaded model, listener owner, --revision on the command line), not to a text file
        try:
            check_judge_server(settings, backend, args.judge_manifest)
        except (FreezeError, ProvenanceError) as e:
            sys.exit(f"final-split judging refused; local judge server: {e}")
    record_backend = backend_description(backend)

    client = None
    if not args.dry_run:
        # one key variable for whatever provider the endpoint is; a local vLLM judge accepts any value
        key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("JUDGE_API_KEY (or OPENAI_API_KEY) is not set; export it for this session (never write it to a file in the repo; use EMPTY for a local vLLM judge)")
        from openai import OpenAI

        # endpoint from the (frozen or configured) settings, passed explicitly: OPENAI_BASE_URL in the environment is ignored
        client = OpenAI(api_key=key, base_url=settings["base_url"], timeout=cfg["judge"].get("timeout_s", 120), max_retries=0)
        print(f"judge model: {settings['model']} at {settings['base_url']}  settings: " + ", ".join(f"{k}={v}" for k, v in settings.items() if k not in ("model", "base_url")))

    desc_data = question_text = None
    if args.files:
        try:
            desc_data = load_visual_description(ensure_visual_description(args.desc))
        except JudgeInputError as e:
            sys.exit(str(e))
        question_text = {q.question_id: q.question for q in load_questions(cfg["data"]["annotations"])}

    if args.compat_check:
        if args.dry_run:
            sys.exit("--compat-check needs a live call; drop --dry-run")
        prompt, fallback = PROBE_PROMPT, True
        if args.files:
            jobs, _ = build_jobs(read_records(Path(args.files[0])), desc_data, tuple(args.cells), settings, args.replicate, question_text)
            if jobs:
                prompt = jobs[0]["body"]["messages"][0]["content"].split("\n", 1)[1]
                # a final-split answer is sent only as the frozen request; no probing with altered settings
                fallback = prov["split"] != "final"
        sys.exit(run_compat_check(client, settings, prompt, fallback))

    for f in args.files:
        judge_file(Path(f), settings, client, desc_data, tuple(args.cells), args.workers, args.limit, args.dry_run, args.replicate, question_text, record_backend, expected_backend)


if __name__ == "__main__":
    main()
