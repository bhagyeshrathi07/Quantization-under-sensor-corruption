"""Local browser tool for labelling one person's copy of the blind sheet.

    python scripts/13_label_ui.py --labeller ronak

Opens http://127.0.0.1:8765 in the browser: one item per screen (cell, the
cell's section of the labelling guide, question, reference answer and model
answer side by side), keys 1 and 0 or two buttons, left and right arrows to
revisit. Every label is written to that person's copy
(`manifests/judge_label_sheet_2026-11-14.<name>.csv`) before the screen
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
from qflip.calibrate import BLIND_COLUMNS, CalibrationError, label_column, parse_label  # noqa: E402

DEFAULT_SHEET = "manifests/judge_label_sheet_2026-11-14.csv"
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

    def __init__(self, path: Path, labeller: str):
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
        self.check_prefix()
        self.digest = blind_digest(self.rows)
        # the file must round-trip: what we would write for the current labels is what is on disk (modulo line endings)
        if serialize(self.rows, self.cols).replace("\r\n", "\n") != text.replace("\r\n", "\n"):
            raise CopyError(f"{path}: the file does not round-trip through the CSV writer (edited by another program?); label it by hand or restore it from git")

    def check_prefix(self):
        k = self.n_labelled()
        if any(l is None for l in self.labels[:k]):
            first_blank = next(i for i, l in enumerate(self.labels) if l is None)
            raise CopyError(f"{self.path}: {self.rows[first_blank]['item']} is blank but a later item is labelled; items are labelled in sheet order. Fix the copy by hand first.")

    def n_labelled(self) -> int:
        return sum(1 for l in self.labels if l is not None)

    def state(self) -> dict:
        return {"labeller": self.labeller, "copy": str(self.path), "n": len(self.rows), "labelled": self.n_labelled(), "next": self.n_labelled(),
                "items": [{"index": i, "item": r["item"], "cell": r["cell"], "question": r["question"], "ground_truth": r["ground_truth"], "answer": r["answer"], "label": l}
                          for i, (r, l) in enumerate(zip(self.rows, self.labels))]}

    def set_label(self, index: int, label: int) -> dict:
        with self.lock:
            if not (0 <= index < len(self.rows)):
                raise CopyError(f"no item at index {index}")
            if label not in (0, 1):
                raise CopyError(f"label {label!r} is not 0 or 1")
            k = self.n_labelled()
            if index > k:
                raise CopyError(f"item {self.rows[index]['item']} comes after the first unlabelled item {self.rows[k]['item']}; items are labelled in sheet order")
            old = self.labels[index]
            self.labels[index] = label
            try:
                self.write()
            except Exception:
                self.labels[index] = old
                raise
            return {"item": self.rows[index]["item"], "label": label, "labelled": self.n_labelled(), "n": len(self.rows), "changed": old is not None and old != label}

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
  <label class="muted"><input type="checkbox" id="render" checked> render the model answer's markdown</label>
</header>
<main>
  <div class="cell" id="cell"></div>
  <div class="muted" id="item-id"></div>
  <div class="q" id="question"></div>
  <div class="cols">
    <div class="panel"><h3>Reference answer (ground truth)</h3><div class="text" id="gt"></div></div>
    <div class="panel"><h3>Model answer</h3><div class="text" id="answer"></div></div>
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
let S = null, cur = 0, busy = false, shownAt = 0;
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
function render() {
  const it = S.items[cur];
  $("progress-text").textContent = `${S.labelled} / ${S.n} labelled, showing item ${cur + 1}`;
  $("progress").max = S.n; $("progress").value = S.labelled;
  $("cell").textContent = it.cell; $("item-id").textContent = it.item;
  $("question").textContent = it.question; $("gt").textContent = it.ground_truth;
  if ($("render").checked) { $("answer").className = "md"; $("answer").innerHTML = md(it.answer); } else { $("answer").className = "text"; $("answer").textContent = it.answer; }
  $("guide-cell").innerHTML = md(GUIDE[it.cell] || "(no section for this cell in the guide)");
  $("b1").classList.toggle("on", it.label === 1); $("b0").classList.toggle("on", it.label === 0);
  $("prev").disabled = cur === 0;
  $("next").disabled = cur >= S.labelled || cur + 1 >= S.n;
  shownAt = performance.now();
  window.scrollTo(0, 0);
}
function status(msg, err) { const s = $("status"); s.textContent = msg; s.className = err ? "err" : "muted"; }
async function load() {
  const r = await fetch("/api/state"); S = await r.json();
  cur = Math.min(S.next, S.n - 1);
  $("guide-full").innerHTML = md(GUIDE.full || "");
  render();
  if (S.labelled >= S.n) status("all items labelled");
}
async function label(v) {
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
function move(d) { if (!S || busy) return; const n = cur + d; if (n < 0 || n >= S.n || n > S.labelled) return; cur = n; status(""); render(); }
$("b1").onclick = () => label(1); $("b0").onclick = () => label(0);
$("prev").onclick = () => move(-1); $("next").onclick = () => move(1);
$("render").onchange = () => { if (S) render(); };
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
            if self.path != "/api/label":
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                index, label, item = body.get("index"), body.get("label"), body.get("item")
                if not isinstance(index, int) or isinstance(index, bool) or not isinstance(label, int) or isinstance(label, bool):
                    return self._json(400, {"error": "index and label must be integers"})
                if item is not None and (not 0 <= index < len(copy.rows) or copy.rows[index]["item"] != item):
                    return self._json(409, {"error": f"item {item} is not at index {index}; reload the page"})
                self._json(200, copy.set_label(index, label))
            except (CopyError, CalibrationError) as e:
                self._json(409, {"error": str(e)})
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})
    return Handler


def build_page(labeller: str, guide: Path) -> str:
    return PAGE.replace("__LABELLER__", html.escape(labeller)).replace("__GUIDE__", json.dumps(guide_sections(guide), ensure_ascii=False).replace("</", "<\\/"))


def serve(copy: Copy, guide: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(copy, build_page(copy.labeller, guide)))


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
    ap.add_argument("--labeller", required=True, help="your name as on the sheet, e.g. ronak")
    ap.add_argument("--sheet", default=DEFAULT_SHEET, help="the master sheet; the copy is <sheet>.<labeller>.csv")
    ap.add_argument("--copy", help="the copy to label (default: next to --sheet)")
    ap.add_argument("--guide", default=DEFAULT_GUIDE)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--check", action="store_true", help="validate the copy and print the progress; no server")
    args = ap.parse_args()
    path = Path(args.copy) if args.copy else copy_path(Path(args.sheet), args.labeller)
    if not path.exists():
        sys.exit(f"{path} not found; your copy is written by scripts/10_build_label_sheet.py --copies and committed with the sheet")
    try:
        copy = Copy(path, args.labeller)
    except (CopyError, CalibrationError) as e:
        sys.exit(str(e))
    print(f"{path}: {len(copy.rows)} items, {copy.n_labelled()} labelled" + ("" if copy.n_labelled() >= len(copy.rows) else f", next {copy.rows[copy.n_labelled()]['item']}"))
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
        print(f"\n{path}: {copy.n_labelled()} / {len(copy.rows)} labelled")


if __name__ == "__main__":
    main()
