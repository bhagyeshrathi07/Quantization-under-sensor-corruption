"""Exploratory side study: a language model labels the blind sheet under the
same guide as the three human labellers, into its own copy, which is never
merged into the master sheet (the master has no column for it; 12 refuses).

    # a copy for the model, then one prompt file per batch of items
    python scripts/14_llm_label.py --labeller fable --sheet manifests/judge_label_sheet_2026-11-17.csv --emit --out-dir results/dev_vqa512/llm_fable
    # ... each batch answered by the model (a Claude agent given the prompt file) into <batch>.reply.txt ...
    python scripts/14_llm_label.py --labeller fable --sheet manifests/judge_label_sheet_2026-11-17.csv --ingest results/dev_vqa512/llm_fable/batch_*.reply.txt

    # Codex end to end (emit, run `codex exec` per batch from an empty folder, ingest)
    python scripts/14_llm_label.py --labeller astra --sheet manifests/judge_label_sheet_2026-11-17.csv --backend codex --effort high --out-dir results/dev_vqa512/llm_astra

The prompt holds the guide verbatim and the batch's items (item id, cell,
question, reference answer, model answer): exactly what a human sees, nothing
from the key or the judge. The model is told to open no file. Labels are
written into the copy in sheet order through the labelling tool's own writer
(scripts/13_label_ui.py: backed-up atomic rewrite, order enforced), one reply
at a time; a reply whose items are not the next unlabelled ones, or whose
labels are not 0 or 1, is refused whole. Reasons and ambiguity flags go to
<out-dir>/labels.json, never onto the sheet. Nothing here touches the master
sheet, the human copies, the key or the judge records.

Protocol: the model labels are an exploratory comparison, reported only after
the human consensus exists (manifests/judge_decisions_2026-11-13.md, side
study of November 19); they are not calibration truth and are not shown to
the human labellers before their own copies are committed.
"""
import argparse
import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qflip.calibrate import BLIND_COLUMNS, CalibrationError, label_column  # noqa: E402

_spec = importlib.util.spec_from_file_location("label_ui", Path(__file__).with_name("13_label_ui.py"))
label_ui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(label_ui)

DEFAULT_GUIDE = "manifests/labelling_guide.md"

PROMPT_HEAD = """You are one of several independent labellers in a research study. Below is the labelling guide, verbatim, and a batch of items from a blind sheet. For every item, decide whether the MODEL ANSWER is correct (1) or incorrect (0) when read against the REFERENCE ANSWER, by the guide's rules for that item's cell. Apply the guide as written; where it leaves a case open, choose the label you lean to and set "ambiguous": true.

Rules for you:
- Use only what is in this message. Do not open any file, run any command, search anything, or use knowledge of the dataset's images or scores. You see exactly what the human labellers see.
- Label every item in the batch, in the order given, each exactly once.
- Reply with one JSON array and nothing else (no prose before or after, no code fence), one object per item:
  {{"item": "<item id>", "label": 0 or 1, "reason": "<one sentence>", "ambiguous": true or false}}

===== LABELLING GUIDE (verbatim) =====
{guide}
===== END OF GUIDE =====

===== ITEMS ({n} in this batch) =====
{items}
===== END OF ITEMS =====

Reply now with the JSON array for these {n} items, in order."""


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        return [r for r in reader if any((v or "").strip() for v in r.values())], cols


def make_copy(sheet: Path, labeller: str) -> Path:
    """The model's own copy: the master's blind columns plus one label column, like 10 --copies."""
    dest = label_ui.copy_path(sheet, labeller)
    rows, cols = read_rows(sheet)
    for c in BLIND_COLUMNS:
        if c not in cols:
            sys.exit(f"{sheet}: column {c} missing")
    if label_column(labeller) in cols:
        sys.exit(f"{sheet}: the master sheet already has a column for {labeller}; a model labeller must not be a human column")
    if dest.exists():
        return dest
    with dest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[*BLIND_COLUMNS, label_column(labeller)])
        w.writeheader()
        w.writerows([{**{c: r.get(c, "") for c in BLIND_COLUMNS}, label_column(labeller): ""} for r in rows])
    return dest


def item_block(r: dict) -> str:
    return (f"--- ITEM {r['item']} ---\nCELL: {r['cell']}\nQUESTION: {r['question']}\nREFERENCE ANSWER: {r['ground_truth']}\nMODEL ANSWER:\n{r['answer']}\n")


def emit(copy: Path, guide: Path, out_dir: Path, batch: int) -> list[Path]:
    rows, _ = read_rows(copy)
    guide_text = guide.read_text(encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for b, start in enumerate(range(0, len(rows), batch), 1):
        chunk = rows[start:start + batch]
        p = out_dir / f"batch_{b:02d}.prompt.txt"
        p.write_text(PROMPT_HEAD.format(guide=guide_text, n=len(chunk), items="\n".join(item_block(r) for r in chunk)), encoding="utf-8")
        (out_dir / f"batch_{b:02d}.items.json").write_text(json.dumps([r["item"] for r in chunk]) + "\n")
        paths.append(p)
    return paths


def parse_reply(text: str) -> list[dict]:
    """The first JSON array in the reply (a code fence or stray prose around it is tolerated, nothing else)."""
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if not m:
        raise CalibrationError("no JSON array in the reply")
    data = json.loads(m.group(0))
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise CalibrationError("the reply is not a JSON array of objects")
    return data


def ingest_reply(copy_obj: label_ui.Copy, reply_path: Path, expected: list[str] | None, record: dict) -> int:
    """Write one reply's labels into the copy, in order, all or nothing checked
    before the first write: the items must be exactly the expected batch (when
    known) and the next unlabelled rows, every label 0 or 1."""
    data = parse_reply(reply_path.read_text(encoding="utf-8"))
    ids = [str(x.get("item")) for x in data]
    if expected is not None and ids != list(expected):
        raise CalibrationError(f"{reply_path}: items {ids[:3]}... differ from the batch's {list(expected)[:3]}... (count {len(ids)} vs {len(expected)})")
    start = copy_obj.n_labelled()
    want = [r["item"] for r in copy_obj.rows[start:start + len(ids)]]
    if ids != want:
        raise CalibrationError(f"{reply_path}: items are not the next unlabelled rows of the copy (reply starts {ids[:2]}, copy expects {want[:2]})")
    for x in data:
        if x.get("label") not in (0, 1) or isinstance(x.get("label"), bool):
            raise CalibrationError(f"{reply_path}: item {x.get('item')}: label {x.get('label')!r} is not 0 or 1")
    for i, x in enumerate(data):
        copy_obj.set_label(start + i, int(x["label"]))
        record["items"][x["item"]] = {"label": int(x["label"]), "reason": str(x.get("reason", ""))[:500], "ambiguous": bool(x.get("ambiguous", False)), "reply": reply_path.name}
    return len(data)


def run_codex(prompt_path: Path, reply_path: Path, effort: str, codex_cmd: list[str], timeout_s: int, model: str | None = None) -> None:
    """One `codex exec` turn from an empty temporary folder (no repository in
    sight), read-only sandbox, the reply written by Codex's -o to reply_path;
    the JSON event stream is kept beside it (`<batch>.events.jsonl`) as the
    execution record, and the model is passed explicitly when given."""
    reply_path = reply_path.resolve()  # codex runs elsewhere, so the reply path must not depend on its working folder
    with tempfile.TemporaryDirectory() as empty:
        cmd = [*codex_cmd, "exec", "--json", "--skip-git-repo-check", "-s", "read-only", "-o", str(reply_path), "-c", f'model_reasoning_effort="{effort}"']
        if model:
            cmd += ["-m", model]
        cmd.append("-")
        proc = subprocess.run(cmd, input=prompt_path.read_text(encoding="utf-8"), capture_output=True, text=True, cwd=empty, timeout=timeout_s)
    reply_path.with_name(reply_path.name.replace(".reply.txt", ".events.jsonl")).write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0 or not reply_path.exists():
        raise RuntimeError(f"{prompt_path.name}: codex exit {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeller", required=True, help="the model's column name, e.g. fable or astra (must not be a human column)")
    ap.add_argument("--sheet", required=True, help="the master sheet; the model's copy is <sheet>.<labeller>.csv")
    ap.add_argument("--guide", default=DEFAULT_GUIDE)
    ap.add_argument("--out-dir", help="prompts, replies and labels.json (default results/dev_vqa512/llm_<labeller>)")
    ap.add_argument("--batch", type=int, default=15, help="items per prompt")
    ap.add_argument("--emit", action="store_true", help="write the copy and the batch prompts; no model call")
    ap.add_argument("--ingest", nargs="*", help="reply files to write into the copy (each must answer the next unlabelled batch)")
    ap.add_argument("--backend", choices=["codex"], help="run the batches through this backend, then ingest")
    ap.add_argument("--effort", default="high", help="reasoning effort for the backend")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per batch for the backend")
    ap.add_argument("--codex-cmd", default="codex", help="the codex executable (tests substitute a fake)")
    ap.add_argument("--model", help="model id passed to the backend explicitly and recorded (codex: -m); without it the backend's configured default is used and recorded as 'configured default'")
    ap.add_argument("--record", nargs="*", default=[], metavar="KEY=VALUE", help="operator-reported facts recorded in labels.json (e.g. backend=claude-agent model=claude-fable-5-1 effort=medium) for replies produced outside this script")
    args = ap.parse_args()
    sheet = Path(args.sheet)
    out_dir = Path(args.out_dir or f"results/dev_vqa512/llm_{args.labeller}")
    copy = make_copy(sheet, args.labeller)
    guide = Path(args.guide)
    record_path = out_dir / "labels.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {"labeller": args.labeller, "sheet": str(sheet), "guide": str(guide), "guide_sha256": label_ui.hashlib.sha256(guide.read_bytes()).hexdigest(), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "backend": None, "model": None, "effort": None, "items": {}}
    for kv in args.record:
        k, _, v = kv.partition("=")
        if not k or not v:
            sys.exit(f"--record takes KEY=VALUE, got {kv!r}")
        record[k] = v
        record.setdefault("operator_reported", []).append(k)

    if args.emit or args.backend:
        prompts = emit(copy, guide, out_dir, args.batch)
        print(f"{copy}: {len(prompts)} prompt(s) of up to {args.batch} items in {out_dir}")
        if args.emit and not args.backend:
            return

    replies: list[Path] = [Path(p) for p in (args.ingest or [])]
    if args.backend == "codex":
        record["backend"], record["effort"] = "codex", args.effort
        record["model"] = args.model or "configured default (~/.codex/config.toml)"
        prompts = sorted(out_dir.glob("batch_*.prompt.txt"))
        todo = [p for p in prompts if not p.with_name(p.name.replace(".prompt.txt", ".reply.txt")).exists()]
        print(f"codex: {len(todo)} batch(es) to run, {len(prompts) - len(todo)} replied already")
        failures = []

        def one(p):
            try:
                run_codex(p, p.with_name(p.name.replace(".prompt.txt", ".reply.txt")), args.effort, args.codex_cmd.split(), args.timeout, args.model)
                print(f"  {p.name}: replied")
            except Exception as e:  # noqa: BLE001 - report and continue with the other batches
                failures.append(str(e))
                print(f"  {p.name}: FAILED {e}")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            list(ex.map(one, todo))
        replies = sorted(p for p in out_dir.glob("batch_*.reply.txt"))
        if failures and not replies:
            sys.exit(f"codex: every batch failed; first error: {failures[0]}")

    if not replies:
        sys.exit("nothing to ingest (give reply files, or --emit / --backend)")
    c = label_ui.Copy(copy, args.labeller)
    try:
        done_replies = {v["reply"] for v in record["items"].values()}
        n = 0
        for rp in sorted(replies):
            if rp.name in done_replies:
                continue
            items_file = rp.with_name(rp.name.replace(".reply.txt", ".items.json"))
            expected = json.loads(items_file.read_text()) if items_file.exists() else None
            n += ingest_reply(c, rp, expected, record)
            print(f"  {rp.name}: {len(expected) if expected else '?'} item(s) written")
        out_dir.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"{copy}: {c.n_labelled()} / {len(c.rows)} labelled ({n} this run); reasons in {record_path}")
        if args.backend == "codex" and failures:
            sys.exit(f"codex: {len(failures)} batch(es) failed; rerun to retry them. First error: {failures[0]}")
    except (CalibrationError, label_ui.CopyError) as e:
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        sys.exit(str(e))
    finally:
        c.close()


if __name__ == "__main__":
    main()
