"""Score VQA answers with the DriveBench rubric judge and write a sidecar file
next to each result file:

    results/<run>/bf16_clean.jsonl  ->  results/<run>/bf16_clean.judge.jsonl

    export OPENAI_API_KEY=...                       # per session; never stored
    python scripts/08_judge_vqa.py results/pilot/bf16_LowLight.jsonl --dry-run   # count, prompts, no calls
    python scripts/08_judge_vqa.py results/pilot/bf16_LowLight.jsonl results/pilot/int4_LowLight.jsonl
    python scripts/08_judge_vqa.py results/final/*_*.jsonl --workers 8

Judge model, temperature, max_tokens, seed come from config `judge:`; the
model name and the template hash are written into every record so a changed
judge can never be mixed with an old one silently. Resumable: questions with a
score in the sidecar are skipped; errored ones are retried.

Prompts are the released templates verbatim (src/qflip/judge_prompts.py); the
planning template needs the object visual description from the toolkit's
data/visual_description.json, fetched on first use.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.config import load_config  # noqa: E402
from qflip.drivebench import load_questions  # noqa: E402
from qflip.judge import (  # noqa: E402
    VQA_CELLS,
    build_prompt,
    ensure_visual_description,
    load_scores,
    load_visual_description,
    needs_description,
    prompt_template,
    request_judge,
    scored_ids,
    sidecar_path,
    template_sha,
    visual_description,
)
from qflip.records import append_record, read_records  # noqa: E402


def build_jobs(records: list[dict], desc_data: dict | None, cells: tuple[str, ...], question_text: dict[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """Return (jobs, skipped). A job is a record plus its filled prompt; skipped
    records carry a reason (wrong cell, errored answer, description missing).
    question_text maps question_id -> question for records written before the
    record schema carried the question."""
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
        desc = None
        if needs_description(cell):
            if desc_data is None:
                skipped.append({"question_id": r["question_id"], "reason": "no visual description file"})
                continue
            try:
                desc = visual_description(desc_data, r["scene_token"], r["frame_token"], r["question"])
            except (KeyError, ValueError) as e:
                skipped.append({"question_id": r["question_id"], "reason": f"description lookup: {e}"})
                continue
        jobs.append({"rec": r, "prompt": build_prompt(cell, r["question"], r["gt"], r["pred"], desc), "desc": desc})
    return jobs, skipped


def judge_file(path: Path, cfg: dict, client, desc_data, cells, workers: int, limit: int | None, dry_run: bool, question_text: dict[str, str]) -> dict:
    jcfg = cfg["judge"]
    records = read_records(path)
    out_path = sidecar_path(path)
    done = scored_ids(load_scores(out_path))
    jobs, skipped = build_jobs(records, desc_data, cells, question_text)
    todo = [j for j in jobs if j["rec"]["question_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{path}: {len(jobs)} judgeable, {len(done)} scored, {len(todo)} to judge, {len(skipped)} skipped -> {out_path}")
    for s in skipped[:5]:
        print(f"  skip {s['question_id']}: {s['reason']}")
    if dry_run:
        if todo:
            j = todo[0]
            print(f"  example prompt ({j['rec']['cell']}, {len(j['prompt'])} chars):\n" + "\n".join("    " + line for line in j["prompt"].splitlines()[-8:]))
        est = sum(len(j["prompt"]) for j in todo) // 4
        print(f"  dry run: ~{est} prompt tokens for {len(todo)} calls (4 chars/token estimate)")
        return {"file": str(path), "to_judge": len(todo), "skipped": len(skipped)}

    def one(j):
        r = j["rec"]
        res = request_judge(client, jcfg["model"], j["prompt"], jcfg["temperature"], jcfg["max_tokens"], jcfg.get("seed"))
        return {
            "question_id": r["question_id"],
            "scene_token": r["scene_token"],
            "cell": r["cell"],
            "condition": r.get("condition"),
            "precision": r.get("precision"),
            "judge_model": jcfg["model"],
            "judge_template": prompt_template(r["cell"])[0],
            "judge_template_sha": template_sha(r["cell"]),
            "desc": j["desc"],
            **res,
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in tqdm(ex.map(one, todo), total=len(todo)):
            append_record(out_path, rec)

    scores = load_scores(out_path)
    by_cell = {}
    for r in scores.values():
        if r.get("score") is not None:
            by_cell.setdefault(r["cell"], []).append(r["score"])
    errors = sum(1 for r in scores.values() if r.get("score") is None)
    print(f"  scored {sum(len(v) for v in by_cell.values())}, errors {errors}")
    for cell, v in sorted(by_cell.items()):
        print(f"  {cell:15s} n={len(v):4d} mean={sum(v)/len(v):6.1f}")
    return {"file": str(path), "scored": sum(len(v) for v in by_cell.values()), "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="result .jsonl files from 03/06")
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--model", help="override judge.model from config")
    ap.add_argument("--desc", default="data/visual_description.json")
    ap.add_argument("--cells", nargs="*", default=list(VQA_CELLS), help="cells to judge (default: the three VQA cells)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="judge at most N new questions per file (cost check)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and count; no API calls, no key needed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["judge"]["model"] = args.model
    if not cfg["judge"].get("model"):
        sys.exit("judge.model is not set in config (team item 2); pass --model or set it")

    desc_data = load_visual_description(ensure_visual_description(args.desc))
    question_text = {q.question_id: q.question for q in load_questions(cfg["data"]["annotations"])}
    client = None
    if not args.dry_run:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("OPENAI_API_KEY is not set; export it for this session (never write it to a file in the repo)")
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=cfg["judge"].get("base_url") or None, timeout=cfg["judge"].get("timeout_s", 120))
        print(f"judge model: {cfg['judge']['model']}  temperature={cfg['judge']['temperature']}  max_tokens={cfg['judge']['max_tokens']}")

    for f in args.files:
        judge_file(Path(f), cfg, client, desc_data, tuple(args.cells), args.workers, args.limit, args.dry_run, question_text)


if __name__ == "__main__":
    main()
