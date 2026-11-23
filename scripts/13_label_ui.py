"""Local browser tool for labelling one person's copy of the blind sheet.

    python scripts/13_label_ui.py --labeller ronak

Opens http://127.0.0.1:8765 in the browser: one item per screen (cell, the
cell's section of the labelling guide, question, reference answer and model
answer side by side), keys 1 and 0 or two buttons, left and right arrows to
revisit. Every label is written to that person's copy
(`manifests/judge_label_sheet_2026-11-17.<name>.csv`) before the screen
advances: the copy is rewritten through a temporary file in the same folder,
the previous version kept as `<copy>.bak`, and the written file is read back
and compared to what was intended before the request succeeds. One instance
per copy (a lock file beside it, which stays in place; only its lock is
released on exit); a write is refused when the file on disk is not the one
this instance last wrote. A held key labels nothing, and a label
is taken only after the item has been on screen for a moment. Only the
`label_<name>` column changes; the blind columns are written back byte for
byte. Labels can be given only in sheet order (the next unlabelled item, or a
revision of an item already labelled), so the copy always satisfies the rule
`scripts/11_calibrate_judge.py` checks. Reopening resumes at the first
unlabelled item.

The tool reads nothing but the copy and the guide: no key, no judge records,
no other person's column. Standard library only; `--check` validates the copy
and prints the progress without starting a server.

Consensus mode (the discussion among the three humans, after every block is
committed):

    python scripts/13_label_ui.py --consensus

opens the master sheet instead: every item with its human label (and who gave
it, from the assignment), the model labellers' labels and one-sentence
reasons (`manifests/judge_label_sheet_<date>.<model>.csv`,
`results/dev_vqa512/llm_<model>/labels.json`), the consensus and a comment
box. The team settles each item: 1 or 0 as the consensus, a comment where it
differs from the human label (written to `consensus_reason`). "Disagreements
only" filters to the items where a model differs from the human; "accept the
human label on every agreed item" fills the rest in one step, as decided on
November 19. Only the `consensus` and `consensus_reason` columns are written;
every other column is verified unchanged on read-back. Same lock, backup and
on-disk checks as labelling.
"""
import argparse
import csv
import fcntl
import hashlib
import html
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qflip.calibrate import BLIND_COLUMNS, CalibrationError, label_column, load_assignment, parse_label  # noqa: E402

DEFAULT_SHEET = "manifests/judge_label_sheet_2026-11-17.csv"


def default_assignment(sheet: Path) -> Path | None:
    """The sheet's assignment file (`<stem>.assignment.json` beside it) when one exists."""
    p = sheet.with_name(f"{sheet.stem}.assignment.json")
    return p if p.exists() else None
DEFAULT_GUIDE = "manifests/labelling_guide.md"


class CopyError(Exception):
    pass


def copy_path(sheet: Path, name: str) -> Path:
    return sheet.with_name(f"{sheet.stem}.{name}{sheet.suffix}")


def read_raw(path: Path) -> str:
    """The file's text with its line endings untouched."""
    with path.open(encoding="utf-8", newline="") as fh:
        return fh.read()


def serialize(rows: list[dict], cols: list[str]) -> str:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\r\n")
    w.writeheader()
    w.writerows([{c: r.get(c, "") for c in cols} for r in rows])
    return buf.getvalue()


def blind_digest(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r.get(c, "") for c in BLIND_COLUMNS], ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class Copy:
    """One labeller's copy: rows in sheet order, their labels, safe rewrites."""

    def __init__(self, path: Path, labeller: str, assignment: Path | None = None):
        """`assignment` (scripts/15_label_assignment.py) restricts the labeller
        to their assigned items: only those are shown and labelled, in sheet
        order; a label on any other row is refused at load."""
        self.path, self.labeller, self.col = path, labeller, label_column(labeller)
        self.lock = threading.Lock()
        # one tool per copy: a second instance would hold stale labels and write them over the first one's
        self.lock_path = path.with_name(path.name + ".lock")
        self.lock_fh = self.lock_path.open("a")
        try:
            fcntl.flock(self.lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_fh.close()
            raise CopyError(f"{path} is open in another instance of this tool (lock {self.lock_path}); close it first, two instances would overwrite each other's labels")
        text = read_raw(path)
        self.disk_text = text
        reader = csv.DictReader(io.StringIO(text, newline=""))
        self.cols = list(reader.fieldnames or [])
        if self.cols != [*BLIND_COLUMNS, self.col]:
            raise CopyError(f"{path}: columns {self.cols} are not {[*BLIND_COLUMNS, self.col]}; this is not {labeller}'s copy written by 10 --copies")
        self.rows = [r for r in reader if any((v or "").strip() for v in r.values())]
        if not self.rows:
            raise CopyError(f"{path}: no items")
        seen = set()
        for r in self.rows:
            if not (r.get("item") or "").strip() or r["item"] in seen:
                raise CopyError(f"{path}: item id missing or repeated ({r.get('item')!r})")
            seen.add(r["item"])
        self.labels = [parse_label(r.get(self.col), f"{path} {r['item']} {labeller}") for r in self.rows]
        # the labeller's rows, in sheet order: all of them, or the assignment's
        self.assignment_path = assignment
        if assignment is not None:
            assigned = load_assignment(assignment, self.rows, [labeller], require_all=False)
            mine = assigned.get(labeller) or []
            if not mine:
                raise CopyError(f"{assignment}: {labeller} has no assigned items")
            pos = {r["item"]: i for i, r in enumerate(self.rows)}
            self.idx = [pos[it] for it in mine]
            outside = [self.rows[i]["item"] for i, l in enumerate(self.labels) if l is not None and i not in set(self.idx)]
            if outside:
                raise CopyError(f"{path}: {outside[0]} is labelled but not in {labeller}'s assignment ({assignment}); fix the copy by hand first")
        else:
            self.idx = list(range(len(self.rows)))
        self.check_prefix()
        self.digest = blind_digest(self.rows)
        # the file must round-trip: what we would write for the current labels is what is on disk (modulo line endings)
        if serialize(self.rows, self.cols).replace("\r\n", "\n") != text.replace("\r\n", "\n"):
            raise CopyError(f"{path}: the file does not round-trip through the CSV writer (edited by another program?); label it by hand or restore it from git")

    def my_labels(self) -> list:
        return [self.labels[i] for i in self.idx]

    def check_prefix(self):
        mine = self.my_labels()
        k = sum(1 for l in mine if l is not None)
        if any(l is None for l in mine[:k]):
            first_blank = next(i for i, l in enumerate(mine) if l is None)
            raise CopyError(f"{self.path}: {self.rows[self.idx[first_blank]]['item']} is blank but a later item is labelled; items are labelled in sheet order. Fix the copy by hand first.")

    def n_labelled(self) -> int:
        """Labelled rows among the labeller's own (assigned) rows."""
        return sum(1 for l in self.my_labels() if l is not None)

    def n_items(self) -> int:
        return len(self.idx)

    def state(self) -> dict:
        """The labeller's items only, indexed by position within them."""
        return {"labeller": self.labeller, "copy": str(self.path), "n": len(self.idx), "labelled": self.n_labelled(), "next": self.n_labelled(),
                "assignment": str(self.assignment_path) if self.assignment_path else None,
                "items": [{"index": p, "item": self.rows[i]["item"], "cell": self.rows[i]["cell"], "question": self.rows[i]["question"], "ground_truth": self.rows[i]["ground_truth"], "answer": self.rows[i]["answer"], "label": self.labels[i]}
                          for p, i in enumerate(self.idx)]}

    def set_label(self, index: int, label: int) -> dict:
        """`index` is the position within the labeller's items."""
        with self.lock:
            if not (0 <= index < len(self.idx)):
                raise CopyError(f"no item at index {index}")
            if label not in (0, 1):
                raise CopyError(f"label {label!r} is not 0 or 1")
            k = self.n_labelled()
            if index > k:
                raise CopyError(f"item {self.rows[self.idx[index]]['item']} comes after the first unlabelled item {self.rows[self.idx[k]]['item']}; items are labelled in sheet order")
            row = self.idx[index]
            old = self.labels[row]
            self.labels[row] = label
            try:
                self.write()
            except Exception:
                self.labels[row] = old
                raise
            return {"item": self.rows[row]["item"], "label": label, "labelled": self.n_labelled(), "n": len(self.idx), "changed": old is not None and old != label}

    def write(self):
        if read_raw(self.path) != self.disk_text:
            raise CopyError(f"{self.path} changed on disk since this tool last wrote it (another program or instance); nothing written. Restart the tool to pick up the file as it is now.")
        for r, l in zip(self.rows, self.labels):
            r[self.col] = "" if l is None else str(l)
        text = serialize(self.rows, self.cols)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        bak = self.path.with_name(self.path.name + ".bak")
        shutil.copy2(self.path, bak)
        os.replace(tmp, self.path)
        back = read_raw(self.path)
        reader = csv.DictReader(io.StringIO(back, newline=""))
        rows = list(reader)
        if back != text or blind_digest(rows) != self.digest or [parse_label(r[self.col], "") for r in rows] != self.labels:
            shutil.copy2(bak, self.path)
            raise CopyError(f"{self.path}: the written file does not read back as intended; restored from {bak}")
        self.disk_text = text

    def close(self):
        """Release the lock and its descriptor. The lock file itself stays:
        deleting it would let a second instance lock the old inode while a
        third creates and locks a new one (review of November 16)."""
        if self.lock_fh.closed:
            return
        try:
            fcntl.flock(self.lock_fh, fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass
        self.lock_fh.close()


class Master:
    """The master sheet in consensus mode: every item with its human label,
    the model labellers' labels and reasons; writes the consensus columns only."""

    def __init__(self, path: Path, assignment: Path | None, models: list[str], records_dir: Path):
        self.path, self.models = path, models
        self.lock = threading.Lock()
        self.lock_path = path.with_name(path.name + ".lock")
        self.lock_fh = self.lock_path.open("a")
        try:
            fcntl.flock(self.lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_fh.close()
            raise CopyError(f"{path} is open in another instance of this tool (lock {self.lock_path}); one person records the consensus")
        text = read_raw(path)
        self.disk_text = text
        reader = csv.DictReader(io.StringIO(text, newline=""))
        self.cols = list(reader.fieldnames or [])
        for c in (*BLIND_COLUMNS, "consensus", "consensus_reason"):
            if c not in self.cols:
                raise CopyError(f"{path}: column {c} missing; this is not the master sheet")
        self.labellers = [c[len("label_"):] for c in self.cols if c.startswith("label_")]
        self.rows = [r for r in reader if any((v or "").strip() for v in r.values())]
        if not self.rows:
            raise CopyError(f"{path}: no items")
        pos = {}
        for i, r in enumerate(self.rows):
            if not (r.get("item") or "").strip() or r["item"] in pos:
                raise CopyError(f"{path}: item id missing or repeated ({r.get('item')!r})")
            pos[r["item"]] = i
        if serialize(self.rows, self.cols).replace("\r\n", "\n") != text.replace("\r\n", "\n"):
            raise CopyError(f"{path}: the file does not round-trip through the CSV writer (edited by another program?)")
        # who labelled each item: the assignment when given, else the one filled label column
        who = {}
        if assignment is not None:
            for n, items in load_assignment(assignment, self.rows, self.labellers).items():
                for it in items:
                    who.setdefault(it, []).append(n)
        self.human = []
        unfinished = []
        for r in self.rows:
            names = who.get(r["item"]) or [n for n in self.labellers if (r.get(label_column(n)) or "").strip()]
            labels = {n: parse_label(r.get(label_column(n)), f"{path} {r['item']} {n}") for n in names}
            if not names or any(l is None for l in labels.values()):
                unfinished.append(r["item"])
            self.human.append({n: l for n, l in labels.items() if l is not None})
        if unfinished:
            # the model labels and reasons are shown here, so no human may still be labelling
            raise CopyError(f"{path}: {len(unfinished)} item(s) without every assigned human label (first {unfinished[0]}); consensus mode opens only when every block is fully labelled, so that nobody sees a model label before their own")
        self.consensus = [parse_label(r.get("consensus"), f"{path} {r['item']} consensus") for r in self.rows]
        self.reasons = [(r.get("consensus_reason") or "") for r in self.rows]
        # the model labellers: their copies (same items, same text) and their reason records
        self.model_labels = {}
        for m in models:
            mp = copy_path(path, m)
            if not mp.exists():
                raise CopyError(f"{mp} missing: the model labeller {m} has no copy")
            mrows, mcols = [], []
            with mp.open(newline="", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                mcols = list(rd.fieldnames or [])
                mrows = [x for x in rd if any((v or "").strip() for v in x.values())]
            if [x["item"] for x in mrows] != [r["item"] for r in self.rows] or blind_digest(mrows) != blind_digest(self.rows):
                raise CopyError(f"{mp}: items or displayed text differ from the master sheet")
            rec_path = records_dir / f"llm_{m}" / "labels.json"
            recs = json.loads(rec_path.read_text())["items"] if rec_path.exists() else {}
            self.model_labels[m] = [{"label": parse_label(x.get(label_column(m)), f"{mp} {x['item']} {m}"), "reason": recs.get(x["item"], {}).get("reason", ""), "ambiguous": bool(recs.get(x["item"], {}).get("ambiguous", False))} for x in mrows]
        self.fixed_digest = self._fixed_digest(self.rows)

    def _fixed_digest(self, rows):
        h = hashlib.sha256()
        for r in rows:
            h.update(json.dumps([r.get(c, "") for c in self.cols if c not in ("consensus", "consensus_reason")], ensure_ascii=False).encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def n_filled(self) -> int:
        return sum(1 for c in self.consensus if c is not None)

    def item(self, i: int) -> dict:
        r = self.rows[i]
        hl = self.human[i]
        human_label = next(iter(hl.values())) if len(hl) == 1 else None
        models = {m: self.model_labels[m][i] for m in self.models}
        disagreement = any(v["label"] != human_label for v in models.values()) if human_label is not None else True
        return {"index": i, "item": r["item"], "cell": r["cell"], "question": r["question"], "ground_truth": r["ground_truth"], "answer": r["answer"],
                "human": hl, "human_label": human_label, "models": models, "consensus": self.consensus[i], "reason": self.reasons[i], "disagreement": disagreement}

    def state(self) -> dict:
        return {"mode": "consensus", "sheet": str(self.path), "n": len(self.rows), "filled": self.n_filled(), "models": self.models, "labellers": self.labellers,
                "items": [self.item(i) for i in range(len(self.rows))]}

    def set_consensus(self, index: int, label: int | None, reason: str | None) -> dict:
        """Set the consensus (1, 0, or None to clear) and/or the comment of one item."""
        with self.lock:
            if not (0 <= index < len(self.rows)):
                raise CopyError(f"no item at index {index}")
            if label not in (0, 1, None):
                raise CopyError(f"consensus {label!r} is not 0, 1 or empty")
            old = (self.consensus[index], self.reasons[index])
            new_reason = reason.replace("\r", " ").replace("\n", " ").strip()[:1000] if reason is not None else self.reasons[index]
            human_label = self.item(index)["human_label"]
            if label is not None and human_label is not None and label != human_label and not new_reason:
                raise CopyError(f"item {self.rows[index]['item']}: the consensus differs from the human label ({human_label}); a comment with the reason is required")
            self.consensus[index] = label
            self.reasons[index] = new_reason
            try:
                self.write()
            except Exception:
                self.consensus[index], self.reasons[index] = old
                raise
            return {"item": self.rows[index]["item"], "consensus": label, "reason": self.reasons[index], "filled": self.n_filled(), "n": len(self.rows)}

    def accept_agreed(self) -> dict:
        """Every item without a consensus where the human and every model agree takes the human label (decision of November 19)."""
        with self.lock:
            done = []
            for i in range(len(self.rows)):
                it = self.item(i)
                if self.consensus[i] is None and it["human_label"] is not None and not it["disagreement"]:
                    self.consensus[i] = it["human_label"]
                    done.append(self.rows[i]["item"])
            if done:
                try:
                    self.write()
                except Exception:
                    for i in range(len(self.rows)):
                        if self.rows[i]["item"] in set(done):
                            self.consensus[i] = None
                    raise
            return {"accepted": len(done), "filled": self.n_filled(), "n": len(self.rows)}

    def write(self):
        if read_raw(self.path) != self.disk_text:
            raise CopyError(f"{self.path} changed on disk since this tool last wrote it; nothing written. Restart the tool.")
        for r, c, why in zip(self.rows, self.consensus, self.reasons):
            r["consensus"] = "" if c is None else str(c)
            r["consensus_reason"] = why
        text = serialize(self.rows, self.cols)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        bak = self.path.with_name(self.path.name + ".bak")
        shutil.copy2(self.path, bak)
        os.replace(tmp, self.path)
        back = read_raw(self.path)
        rows = list(csv.DictReader(io.StringIO(back, newline="")))
        if back != text or self._fixed_digest(rows) != self.fixed_digest or [parse_label(r["consensus"], "") for r in rows] != self.consensus:
            shutil.copy2(bak, self.path)
            raise CopyError(f"{self.path}: the written file does not read back as intended; restored from {bak}")
        self.disk_text = text

    def close(self):
        if self.lock_fh.closed:
            return
        try:
            fcntl.flock(self.lock_fh, fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass
        self.lock_fh.close()


def guide_sections(guide: Path) -> dict:
    """{heading: text} for every '## ' section of the guide, plus 'full'."""
    if not guide.exists():
        return {"full": f"(guide {guide} not found)"}
    text = guide.read_text(encoding="utf-8")
    sections, current, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current, buf = line[3:].strip(), []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    sections["full"] = text
    return sections


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label sheet: __LABELLER__</title>
<style>
:root { --bg:#f6f4ef; --fg:#1d1c1a; --muted:#6b675f; --panel:#fffdf8; --line:#d9d4c8; --ok:#1f6f43; --no:#9b2c2c; --accent:#2a4d7a; }
@media (prefers-color-scheme: dark) { :root { --bg:#17191c; --fg:#e8e6e1; --muted:#9a978f; --panel:#1f2226; --line:#3a3e45; --ok:#4cbf7a; --no:#e06c6c; --accent:#8ab4e8; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { display:flex; flex-wrap:wrap; gap:12px 24px; align-items:baseline; padding:12px 24px; border-bottom:1px solid var(--line); background:var(--panel); position:sticky; top:0; }
header b { font-size:18px; }
.muted { color:var(--muted); }
main { max-width: 1400px; margin: 0 auto; padding: 16px 24px 120px; }
.cell { font: 600 13px/1 ui-monospace, monospace; letter-spacing:.04em; text-transform:uppercase; color:var(--accent); }
.q { font-size:18px; margin: 8px 0 16px; white-space: pre-wrap; }
.cols { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
@media (max-width: 900px) { .cols { grid-template-columns: 1fr; } }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; min-width:0; }
.panel h3 { margin:0 0 8px; font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
.text { white-space: pre-wrap; overflow-wrap: anywhere; }
details { margin: 12px 0; }
.md { font-size:15px; overflow-wrap: anywhere; }
details .md { max-width: 80ch; }
.md h3, .md h4, .md h5, .md h6 { margin: 14px 0 6px; font-size: 15px; color: var(--accent); }
.md p { margin: 6px 0; } .md ul, .md ol { margin: 6px 0; padding-left: 1.4em; } .md li { margin: 3px 0; }
.md code { font: 13px ui-monospace, monospace; background: var(--bg); border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; }
summary { cursor: pointer; font-weight: 600; }
mark { background: #ffe58a; color: inherit; padding: 0 2px; border-radius: 2px; }
table.votes { border-collapse: collapse; width: 100%; font-size: 14px; } table.votes th, table.votes td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
table.votes td.lab { font-weight: 700; } .consensus-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
.consensus-row textarea { flex: 1 1 320px; font: 14px system-ui, sans-serif; padding: 6px; border: 1px solid var(--line); border-radius: 6px; background: var(--bg); color: var(--fg); }
.consensus-row label { flex-basis: 100%; font-size: 13px; color: var(--muted); }
@media (prefers-color-scheme: dark) { mark { background: #7a5c00; } }
.bar { position:fixed; left:0; right:0; bottom:0; background:var(--panel); border-top:1px solid var(--line); padding:12px 24px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:center; }
button { font: 600 18px system-ui, sans-serif; padding: 12px 28px; border-radius:8px; border:2px solid var(--line); background:var(--bg); color:var(--fg); cursor:pointer; }
button.ok { border-color: var(--ok); } button.no { border-color: var(--no); }
button.ok.on { background: var(--ok); color:#fff; } button.no.on { background: var(--no); color:#fff; }
button.nav { font-weight:400; padding: 12px 16px; }
button:disabled { opacity:.4; cursor:default; }
#status { min-width: 10em; text-align:center; }
#status.err { color: var(--no); font-weight:600; }
progress { width: 220px; height: 10px; }
kbd { font: 12px ui-monospace, monospace; border:1px solid var(--line); border-radius:4px; padding:0 5px; color:var(--muted); }
</style>
<header>
  <b>__LABELLER__</b>
  <span id="progress-text" class="muted"></span>
  <progress id="progress" max="1" value="0"></progress>
  <span class="muted">keys: <kbd>1</kbd> correct <kbd>0</kbd> incorrect <kbd>&larr;</kbd> <kbd>&rarr;</kbd> move</span>
  <label class="muted"><input type="checkbox" id="render" checked> render markdown</label>
  <label class="muted"><input type="checkbox" id="hilite" checked> mark reference words in the model answer</label>
  <label class="muted" id="filter-wrap" hidden><input type="checkbox" id="filter" checked> disagreements only</label>
</header>
<main>
  <div class="cell" id="cell"></div>
  <div class="muted" id="item-id"></div>
  <div class="q" id="question"></div>
  <div class="cols">
    <div class="panel"><h3>Reference answer (ground truth)</h3><div class="text" id="gt"></div></div>
    <div class="panel"><h3>Model answer</h3><div class="text" id="answer"></div></div>
  </div>
  <div class="panel" id="votes" hidden>
    <h3>Labels</h3>
    <table class="votes"><thead><tr><th>who</th><th>label</th><th>reason</th></tr></thead><tbody id="votes-body"></tbody></table>
    <div class="consensus-row">
      <label>Comment / reason for the consensus (needed where it differs from the human label):</label>
      <textarea id="reason" rows="2" placeholder="one line"></textarea>
      <button class="nav" id="save-reason">save comment</button>
      <button class="nav" id="clear-consensus" title="remove the consensus of this item">clear consensus</button>
      <button class="nav" id="accept-agreed" title="every item without a consensus where the human and both models agree takes the human label">accept the human label on every agreed item</button>
    </div>
  </div>
  <details open><summary>Criterion for this cell</summary><div class="md" id="guide-cell"></div></details>
  <details><summary>Full labelling guide</summary><div class="md" id="guide-full"></div></details>
</main>
<div class="bar">
  <button class="nav" id="prev" title="previous item">&larr; back</button>
  <button class="ok" id="b1">1 &nbsp; correct</button>
  <button class="no" id="b0">0 &nbsp; incorrect</button>
  <button class="nav" id="next" title="next item">next &rarr;</button>
  <span id="status" class="muted"></span>
</div>
<script>
const GUIDE = __GUIDE__;
let S = null, cur = 0, busy = false, shownAt = 0, view = [];  // view: indices of the items shown (consensus mode filters)
const consensusMode = () => S && S.mode === "consensus";
function rebuildView() { view = consensusMode() && $("filter").checked ? S.items.map((it, i) => it.disagreement ? i : -1).filter(i => i >= 0) : S.items.map((_, i) => i); if (!view.length) view = S.items.map((_, i) => i); }
const ARM_MS = 400;  // an item must have been on screen this long before a label is taken
const $ = id => document.getElementById(id);
const esc = t => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function inline(t) {
  return esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}
function md(text) {
  // headings, nested bullet and numbered lists, paragraphs, `code`, **bold**; everything escaped
  const out = [], stack = []; let para = [];
  const flushPara = () => { if (para.length) { out.push("<p>" + inline(para.join(" ")) + "</p>"); para = []; } };
  const closeList = () => { const l = stack.pop(); out.push((l.li ? "</li>" : "") + `</${l.t}>`); };
  const closeAll = () => { while (stack.length) closeList(); };
  const item = (t, n, txt, num) => {
    flushPara();
    while (stack.length && stack[stack.length - 1].n > n) closeList();
    if (stack.length && stack[stack.length - 1].n === n && stack[stack.length - 1].t !== t) closeList();
    if (!stack.length || stack[stack.length - 1].n < n) { out.push(`<${t}>`); stack.push({t, n, li: false}); }
    const top = stack[stack.length - 1];
    if (top.li) out.push("</li>");
    out.push((num ? `<li value="${num}">` : "<li>") + inline(txt)); top.li = true;  // the model's own numbering is shown, never renumbered
  };
  for (const raw of text.split("\n")) {
    const line = raw.replace(/\t/g, "    ").trimEnd();
    const indent = line.length - line.trimStart().length;
    let m;
    if (!line.trim()) { flushPara(); continue; }
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); closeAll(); const h = Math.min(m[1].length + 2, 6); out.push(`<h${h}>${inline(m[2])}</h${h}>`); continue; }
    if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) { item("ul", indent, m[1]); continue; }
    if ((m = line.match(/^\s*(\d+)[.)]\s+(.*)$/))) { item("ol", indent, m[2], m[1]); continue; }
    if ((m = line.match(/^\s*[-*•]\s*$/))) { item("ul", indent, ""); continue; }  // a cut-off bullet
    if (stack.length && indent > 0) { out.push(" " + inline(line.trim())); continue; }  // continuation of the open item
    closeAll(); para.push(line.trim());
  }
  flushPara(); closeAll();
  return out.join("\n");
}
function renderVotes(it) {
  const body = $("votes-body"); body.innerHTML = "";
  const row = (who, label, reason) => { const tr = document.createElement("tr"); const a = document.createElement("td"); a.textContent = who; const b = document.createElement("td"); b.className = "lab"; b.textContent = label === null || label === undefined ? "-" : String(label); const c = document.createElement("td"); c.textContent = reason || ""; tr.append(a, b, c); body.appendChild(tr); };
  for (const [who, label] of Object.entries(it.human)) row(who + " (human)", label, "");
  for (const [m, v] of Object.entries(it.models)) row(m, v.label, (v.ambiguous ? "[flagged ambiguous] " : "") + (v.reason || ""));
  row("consensus", it.consensus, it.reason);
  $("reason").value = it.reason || "";
}
function render() {
  if (consensusMode()) {
    const it = S.items[view[cur]];
    const dis = S.items.filter(x => x.disagreement).length, disDone = S.items.filter(x => x.disagreement && x.consensus !== null).length;
    $("progress-text").textContent = `consensus ${S.filled} / ${S.n} (disagreements ${disDone} / ${dis}), showing ${cur + 1} of ${view.length}`;
    $("progress").max = S.n; $("progress").value = S.filled;
    $("cell").textContent = it.cell; $("item-id").textContent = it.item + (it.disagreement ? "  (model disagrees with the human)" : "  (all agree)");
    $("question").textContent = it.question; $("gt").textContent = it.ground_truth;
    if ($("render").checked) { $("answer").className = "md"; $("answer").innerHTML = md(it.answer); } else { $("answer").className = "text"; $("answer").textContent = it.answer; }
    if ($("hilite").checked) highlight($("answer"), it.ground_truth);
    $("guide-cell").innerHTML = md(GUIDE[it.cell] || "");
    renderVotes(it);
    $("b1").classList.toggle("on", it.consensus === 1); $("b0").classList.toggle("on", it.consensus === 0);
    $("prev").disabled = cur === 0; $("next").disabled = cur + 1 >= view.length;
    shownAt = performance.now(); window.scrollTo(0, 0);
    return;
  }
  const it = S.items[cur];
  $("progress-text").textContent = `${S.labelled} / ${S.n} labelled, showing item ${cur + 1} of your ${S.n}`;
  $("progress").max = S.n; $("progress").value = S.labelled;
  $("cell").textContent = it.cell; $("item-id").textContent = it.item;
  $("question").textContent = it.question; $("gt").textContent = it.ground_truth;
  if ($("render").checked) { $("answer").className = "md"; $("answer").innerHTML = md(it.answer); } else { $("answer").className = "text"; $("answer").textContent = it.answer; }
  if ($("hilite").checked) highlight($("answer"), it.ground_truth);
  $("guide-cell").innerHTML = md(GUIDE[it.cell] || "(no section for this cell in the guide)");
  $("b1").classList.toggle("on", it.label === 1); $("b0").classList.toggle("on", it.label === 0);
  $("prev").disabled = cur === 0;
  $("next").disabled = cur >= S.labelled || cur + 1 >= S.n;
  shownAt = performance.now();
  window.scrollTo(0, 0);
}
// Reading aid only: words of the reference answer (4+ letters, not function words, simple plural/-ing/-ed stems) are marked
// wherever they occur in the model answer, so the eye lands on candidate matches. It decides nothing.
const STOP = new Set(["that","this","with","from","there","their","they","them","then","than","will","would","should","could","have","has","been","being","into","onto","also","very","about","which","what","when","where","while","because","ego","vehicle","object","objects","action","actions","these","those","such","some","same","other","only","more","most","much","many","each","both","after","before","through","over","under","next","first","second","third","firstly","secondly","thirdly","notice","noticed","camera","side","probability","reason","high","low","the","and","are","is","to","of","in","on","at","it","its","for","as","by","an","or","be","so","no","not","none"]);
function stem(w) { const m = w.match(/^(.{4,})(ing|ed|es|s)$/); return m ? m[1] : w; }  // keep at least four letters of the stem ("speed" stays "speed")
function refTerms(gt) {
  const out = new Map();
  for (const w of (gt.toLowerCase().match(/[a-z]{4,}/g) || [])) { if (!STOP.has(w)) { const st = stem(w); if (st.length >= 3) out.set(st, w); } }
  return [...out.keys()];
}
function highlight(root, gt) {
  const stems = refTerms(gt);
  if (!stems.length) return;
  const re = new RegExp("\\b(" + stems.join("|") + ")[a-z]*\\b", "gi");  // stems are letters only, nothing to escape
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = []; let n; while ((n = walker.nextNode())) nodes.push(n);
  for (const node of nodes) {
    const text = node.nodeValue; if (!re.test(text)) { re.lastIndex = 0; continue; } re.lastIndex = 0;
    const frag = document.createDocumentFragment(); let last = 0, m;
    while ((m = re.exec(text))) { frag.appendChild(document.createTextNode(text.slice(last, m.index))); const mk = document.createElement("mark"); mk.textContent = m[0]; frag.appendChild(mk); last = m.index + m[0].length; }
    frag.appendChild(document.createTextNode(text.slice(last)));
    node.parentNode.replaceChild(frag, node);
  }
}
function status(msg, err) { const s = $("status"); s.textContent = msg; s.className = err ? "err" : "muted"; }
async function load() {
  const r = await fetch("/api/state"); S = await r.json();
  $("guide-full").innerHTML = md(GUIDE.full || "");
  if (consensusMode()) {
    $("votes").hidden = false; $("filter-wrap").hidden = false;
    document.querySelector("header b").textContent = "Consensus";
    $("b1").innerHTML = "1 &nbsp; consensus: correct"; $("b0").innerHTML = "0 &nbsp; consensus: incorrect";
    rebuildView(); cur = Math.max(0, view.findIndex(i => S.items[i].consensus === null));
    render(); return;
  }
  cur = Math.min(S.next, S.n - 1);
  render();
  if (S.labelled >= S.n) status("all items labelled");
}
async function post(path, body) {
  const r = await fetch(path, {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(body)});
  const j = await r.json(); if (!r.ok) throw new Error(j.error || "request failed"); return j;
}
async function setConsensus(v) {
  if (busy || !S) return;
  const i = view[cur], it = S.items[i];
  busy = true; status("saving…");
  try {
    const j = await post("/api/consensus", {index: i, item: it.item, label: v, reason: $("reason").value});
    it.consensus = j.consensus; it.reason = j.reason; S.filled = j.filled;
    status(v === null ? "consensus cleared" : "saved");
    if (v !== null && cur + 1 < view.length) cur += 1;
    render();
  } catch (e) { status(e.message, true); }
  busy = false;
}
async function saveReason() {
  if (busy || !S) return;
  const i = view[cur], it = S.items[i];
  busy = true; status("saving…");
  try { const j = await post("/api/consensus", {index: i, item: it.item, label: it.consensus, reason: $("reason").value}); it.reason = j.reason; status("comment saved"); renderVotes(it); }
  catch (e) { status(e.message, true); }
  busy = false;
}
async function acceptAgreed() {
  if (busy || !S) return;
  busy = true; status("filling…");
  try { const j = await post("/api/accept-agreed", {}); const r = await fetch("/api/state"); S = await r.json(); rebuildView(); cur = Math.min(cur, view.length - 1); render(); status(`accepted the human label on ${j.accepted} item(s); consensus ${j.filled} / ${j.n}`); }
  catch (e) { status(e.message, true); }
  busy = false;
}
async function label(v) {
  if (consensusMode()) return setConsensus(v);
  if (busy || !S) return;
  if (performance.now() - shownAt < ARM_MS) { status("read the item first"); return; }
  const it = S.items[cur];
  if (cur > S.labelled) { status("label the earlier items first", true); return; }
  const idx = cur;  // the item being labelled; navigation is blocked until the save returns, and the advance is from this index
  busy = true; status("saving…");
  try {
    const r = await fetch("/api/label", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({index: idx, item: it.item, label: v})});
    const j = await r.json();
    if (!r.ok) { status(j.error || "save failed", true); busy = false; return; }
    it.label = v; S.labelled = j.labelled;
    status(j.changed ? "saved (changed)" : "saved");
    cur = Math.min(idx + 1, S.n - 1);
    render();
    if (S.labelled >= S.n && idx + 1 >= S.n) status("saved; all items labelled");
  } catch (e) { status("save failed: " + e, true); }
  busy = false;
}
function move(d) { if (!S || busy) return; const n = cur + d; if (consensusMode()) { if (n < 0 || n >= view.length) return; cur = n; status(""); render(); return; } if (n < 0 || n >= S.n || n > S.labelled) return; cur = n; status(""); render(); }
$("b1").onclick = () => label(1); $("b0").onclick = () => label(0);
$("prev").onclick = () => move(-1); $("next").onclick = () => move(1);
$("render").onchange = () => { if (S) render(); };
$("filter").onchange = () => { if (S) { rebuildView(); cur = 0; render(); } };
$("save-reason").onclick = saveReason; $("clear-consensus").onclick = () => setConsensus(null); $("accept-agreed").onclick = acceptAgreed;
$("hilite").onchange = () => { if (S) render(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.repeat) return;  // a held key labels nothing: one key press, one label
  if (e.key === "1") label(1); else if (e.key === "0") label(0);
  else if (e.key === "ArrowLeft") move(-1); else if (e.key === "ArrowRight") move(1);
});
load();
</script>
"""


def make_handler(copy: Copy, page: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                with copy.lock:
                    self._json(200, copy.state())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            if isinstance(copy, Master):
                try:
                    if self.path == "/api/accept-agreed":
                        return self._json(200, copy.accept_agreed())
                    if self.path != "/api/consensus":
                        return self._json(404, {"error": "not found"})
                    index, label, item, reason = body.get("index"), body.get("label"), body.get("item"), body.get("reason")
                    if not isinstance(index, int) or isinstance(index, bool) or (label is not None and (not isinstance(label, int) or isinstance(label, bool))):
                        return self._json(400, {"error": "index must be an integer and label 0, 1 or null"})
                    if item is not None and (not 0 <= index < len(copy.rows) or copy.rows[index]["item"] != item):
                        return self._json(409, {"error": f"item {item} is not at index {index}; reload the page"})
                    return self._json(200, copy.set_consensus(index, label, reason if isinstance(reason, str) else None))
                except (CopyError, CalibrationError) as e:
                    return self._json(409, {"error": str(e)})
            if self.path != "/api/label":
                return self._json(404, {"error": "not found"})
            try:
                index, label, item = body.get("index"), body.get("label"), body.get("item")
                if not isinstance(index, int) or isinstance(index, bool) or not isinstance(label, int) or isinstance(label, bool):
                    return self._json(400, {"error": "index and label must be integers"})
                if item is not None and (not 0 <= index < copy.n_items() or copy.rows[copy.idx[index]]["item"] != item):
                    return self._json(409, {"error": f"item {item} is not at index {index}; reload the page"})
                self._json(200, copy.set_label(index, label))
            except (CopyError, CalibrationError) as e:
                self._json(409, {"error": str(e)})
    return Handler


def build_page(labeller: str, guide: Path) -> str:
    return PAGE.replace("__LABELLER__", html.escape(labeller)).replace("__GUIDE__", json.dumps(guide_sections(guide), ensure_ascii=False).replace("</", "<\\/"))


def serve(copy, guide: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(copy, build_page(getattr(copy, "labeller", "consensus"), guide)))


def open_browser(url: str) -> None:
    try:
        if webbrowser.open(url):
            return
    except Exception:
        pass
    proc = Path("/proc/version")
    if proc.exists() and "microsoft" in proc.read_text().lower():  # WSL: hand the URL to Windows
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", url.replace("&", "^&")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    print(f"open {url} in your browser")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeller", help="your name as on the sheet, e.g. ronak (labelling mode)")
    ap.add_argument("--consensus", action="store_true", help="discussion mode on the master sheet: human and model labels side by side, consensus and comment per item")
    ap.add_argument("--models", nargs="*", default=["fable", "astra"], help="model labellers shown in consensus mode (their copies and records must exist)")
    ap.add_argument("--records-dir", default="results/dev_vqa512", help="where llm_<model>/labels.json live")
    ap.add_argument("--sheet", default=DEFAULT_SHEET, help="the master sheet; the copy is <sheet>.<labeller>.csv")
    ap.add_argument("--copy", help="the copy to label (default: next to --sheet)")
    ap.add_argument("--assignment", help="who labels which items (scripts/15_label_assignment.py); default: the sheet's <stem>.assignment.json when it exists; 'none' to label every item")
    ap.add_argument("--guide", default=DEFAULT_GUIDE)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--check", action="store_true", help="validate the copy and print the progress; no server")
    args = ap.parse_args()
    if args.consensus:
        return consensus_main(args)
    if not args.labeller:
        sys.exit("--labeller <name> (labelling) or --consensus (discussion) is required")
    path = Path(args.copy) if args.copy else copy_path(Path(args.sheet), args.labeller)
    if not path.exists():
        sys.exit(f"{path} not found; your copy is written by scripts/10_build_label_sheet.py --copies and committed with the sheet")
    assignment = default_assignment(Path(args.sheet)) if args.assignment is None else (None if args.assignment == "none" else Path(args.assignment))
    try:
        copy = Copy(path, args.labeller, assignment)
    except (CopyError, CalibrationError) as e:
        sys.exit(str(e))
    first, last = copy.rows[copy.idx[0]]["item"], copy.rows[copy.idx[-1]]["item"]
    print(f"{path}: {copy.n_items()} items for {args.labeller} ({first} to {last}" + (f", assignment {assignment}" if assignment else ", every item") + f"), {copy.n_labelled()} labelled"
          + ("" if copy.n_labelled() >= copy.n_items() else f", next {copy.rows[copy.idx[copy.n_labelled()]]['item']}"))
    if args.check:
        copy.close()
        return
    guide = Path(args.guide)
    if not guide.exists():
        print(f"warning: {guide} not found; the criterion panel will be empty")
    try:
        server = serve(copy, guide, port=args.port)
    except OSError as e:
        sys.exit(f"cannot listen on port {args.port}: {e} (another copy of the tool running? use --port)")
    url = f"http://127.0.0.1:{args.port}/"
    print(f"labelling {copy.labeller}'s copy at {url} (Ctrl+C to stop; every label is already saved when the screen advances)")
    if not args.no_browser:
        open_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        copy.close()
        print(f"\n{path}: {copy.n_labelled()} / {copy.n_items()} labelled")


def consensus_main(args):
    sheet = Path(args.sheet)
    if not sheet.exists():
        sys.exit(f"{sheet} not found")
    assignment = default_assignment(sheet) if args.assignment is None else (None if args.assignment == "none" else Path(args.assignment))
    try:
        master = Master(sheet, assignment, list(args.models), Path(args.records_dir))
    except (CopyError, CalibrationError) as e:
        sys.exit(str(e))
    items = [master.item(i) for i in range(len(master.rows))]
    dis = sum(1 for it in items if it["disagreement"])
    print(f"{sheet}: {len(items)} items, consensus on {master.n_filled()}, {dis} item(s) where a model disagrees with the human (models {', '.join(args.models)})")
    if args.check:
        master.close()
        return
    guide = Path(args.guide)
    try:
        server = serve(master, guide, port=args.port)
    except OSError as e:
        master.close()
        sys.exit(f"cannot listen on port {args.port}: {e} (another copy of the tool running? use --port)")
    url = f"http://127.0.0.1:{args.port}/"
    print(f"consensus mode at {url} (Ctrl+C to stop; every consensus and comment is saved when you press it)")
    if not args.no_browser:
        open_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        master.close()
        print(f"\n{sheet}: consensus on {master.n_filled()} / {len(master.rows)}")


if __name__ == "__main__":
    main()
