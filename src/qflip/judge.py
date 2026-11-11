"""VQA rubric judge, benchmark-compatible.

Reproduces what the released DriveBench evaluator does with `--eval-gpt`
(evaluate/eval.py + request.py), with the parts that make it unusable as
shipped replaced:

- The released request.py calls `call_chatgpt(prompt, max_tokens=512)` but
  `call_chatgpt` takes no such argument, so every call raises and the retry
  loop never ends. Here the request is made with the openai client.
- The judge model is hard-coded to gpt-3.5-turbo upstream (shut down October
  23, 2026); here it comes from config (`judge.model`) and is written into
  every record, together with every request setting.
- A reply without "Total Score:" raises upstream; here it is recorded as an
  error so a rerun can retry it.

What is kept verbatim: the prompt templates (judge_prompts.py), the system
line, the visual-description lookup, and the score regex. Temperature 0 is
the released setting but is optional here, because current judge models
reject or ignore it (plan section 7); every optional setting is omitted from
the request when it is null in config.

Record identity. Every judge record carries `input_sha256`, the hash of the
serialized request actually sent (model, every optional setting, the
max-tokens parameter, and the single user message, which contains the system
line, the filled template, question, ground truth, object description, and
prediction), plus a `replicate` id (default 0). The resume key is
(question_id, replicate, input_sha256): a changed answer, model, template,
or setting is a new key and is judged again; replicates 1..k are separate
rows and are never deduplicated. Production scoring (05 --vqa) consumes
replicate 0 whose hash matches the current answer and the current (or
frozen) judge settings; anything else is reported as stale and not used.

The API key comes from JUDGE_API_KEY (or OPENAI_API_KEY) only; it is never read from a file.
"""

import hashlib
import inspect
import ipaddress
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from qflip import judge_prompts

# Released toolkit file, not part of the HF dataset. Downloaded on demand.
VISUAL_DESCRIPTION_URL = "https://raw.githubusercontent.com/drive-bench/toolkit/main/data/visual_description.json"
VISUAL_DESCRIPTION_SHA256 = "4cf48342f2a325144ed1153516ad63f70c8bd2a3bc3be9af2afc22698271fd58"  # main, Oct 20, 2026

# upstream request.py prepare_chatgpt_message
SYSTEM_LINE = "You are an evaluator who rates answers based on their closeness to the correct answer."

# upstream eval.py forward(): cell -> prompt template
PROMPT_BY_CELL = {
    "perception/MCQ": ("PERCEPTION_MCQ_PROMPT", judge_prompts.PERCEPTION_MCQ_PROMPT),
    "perception/VQA": ("PERCEPTION_VQA_PROMPT", judge_prompts.PERCEPTION_VQA_PROMPT),
    "prediction/VQA": ("PREDICTION_VQA_PROMPT", judge_prompts.PREDICTION_VQA_PROMPT),
    "planning/VQA": ("PLANNING_VQA_PROMPT", judge_prompts.PLANNING_VQA_PROMPT),
    "behavior/MCQ": ("BEHAVIOR_MCQ_PROMPT", judge_prompts.BEHAVIOR_MCQ_PROMPT),
}
VQA_CELLS = ("perception/VQA", "prediction/VQA", "planning/VQA")

# Request settings that are sent only when set (non-null) in config `judge:`.
OPTIONAL_PARAMS = ("temperature", "top_p", "reasoning_effort", "seed")
# The provider endpoint when config `judge.base_url` is null. Always passed to
# the client explicitly, so the SDK's OPENAI_BASE_URL environment variable
# can never redirect a judge run silently.
DEFAULT_BASE_URL = "https://api.openai.com/v1"
# Name of the completion-length parameter; GPT-5 models reject `max_tokens`.
TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")
# Non-secret description of what stands behind the endpoint (config `judge.backend`):
# the provider or proxy, the upstream model id the proxy routes the alias to (copied
# from its routing table by hand; the request itself names only the alias), the
# checkpoint revision of a local judge, and how thinking is controlled. Recorded in
# every record and bound by the judge freeze. Not a request field, so not in the hash.
BACKEND_KEYS = ("provider", "upstream_model", "revision", "thinking_control")
# Score extraction version. 1: the released regex only. 2: markdown-tolerant
# fallbacks (November 8). 3: the Total Score line parsed as a whole (November 8
# review). 4: complete grammar only: an equation or parenthesised sum must be
# consumed entirely, delimiters balanced, no trailing operator, decimals
# rejected rather than truncated (November 11 review). Bound by the judge freeze
# with a digest of the extractor source and regexes; every record names the
# version it was scored with; production scoring re-reads the stored reply.
EXTRACTOR_VERSION = "4"

# upstream eval.py extract_visual_description workaround
_NO_DESC_QUESTION = "In this scenario, what are safe actions to take for the ego vehicle?"
_NO_DESC_TEXT = "No visual description needed for this question."
_SCORE_RE = re.compile(r"Total Score:\s*(\d{1,3})\b")

ScoreKey = tuple[str, int, str | None]  # (question_id, replicate, input_sha256)


class JudgeSettingsError(ValueError):
    pass


class JudgeInputError(RuntimeError):
    """A judge input file (the object visual descriptions) is not the recorded one."""


def ensure_visual_description(path: str | Path) -> Path:
    """Download data/visual_description.json from the released toolkit if absent."""
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"[judge] fetching {VISUAL_DESCRIPTION_URL} -> {p}")
        with urllib.request.urlopen(VISUAL_DESCRIPTION_URL, timeout=60) as r:  # noqa: S310 - fixed https URL
            p.write_bytes(r.read())
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != VISUAL_DESCRIPTION_SHA256:
        raise JudgeInputError(f"{p} sha256 {digest[:12]} differs from the recorded {VISUAL_DESCRIPTION_SHA256[:12]}; the object descriptions are judge input and must be the recorded file")
    return p


def load_visual_description(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def visual_description(desc_data: dict, scene_token: str, frame_token: str, question: str) -> str | None:
    """Upstream extract_visual_description, unchanged in behavior: the object
    tag `<c1,CAM_FRONT,x,y>` in the question selects the key object by its
    `<c1,CAM_FRONT` prefix; no tag -> None; tag without a match -> ValueError."""
    if question == _NO_DESC_QUESTION:
        return _NO_DESC_TEXT
    key_object = desc_data[scene_token]["key_frames"][frame_token]["key_object_infos"]
    m = re.search(r"<(.*)>", question)
    if not m:
        return None
    prefix = ",".join(m.group(1).split(",")[:2])
    key = next((k for k in key_object if k.startswith(f"<{prefix}")), None)
    if key is None:
        raise ValueError(f"No matching object found for prefix: {prefix}")
    return key_object[key]["Visual_description"].lower()


def prompt_template(cell: str) -> tuple[str, str]:
    """(template name, template text) for a cell; KeyError for cells the released scorer does not judge."""
    return PROMPT_BY_CELL[cell]


def template_sha(cell: str) -> str:
    return hashlib.sha1(prompt_template(cell)[1].encode()).hexdigest()[:12]


def build_prompt(cell: str, question: str, gt: str, pred: str, desc: str | None) -> str:
    """Upstream request.py forward(): the template is filled with DESC only when
    a description exists; templates without a {DESC} placeholder ignore it."""
    _, template = prompt_template(cell)
    if not desc:
        return template.format(GT=gt, PRED=pred, QUESTION=question)
    return template.format(GT=gt, PRED=pred, QUESTION=question, DESC=desc)


def needs_description(cell: str) -> bool:
    return "{DESC}" in prompt_template(cell)[1]


def expected_description(desc_data: dict | None, rec: dict, question: str) -> str | None:
    """The one description the judge input for this record must carry: looked
    up in the hash-checked description file when the cell's template uses it,
    None otherwise. KeyError / ValueError from the lookup propagate."""
    if not needs_description(rec["cell"]):
        return None
    if desc_data is None:
        raise JudgeInputError("no visual description file")
    return visual_description(desc_data, rec["scene_token"], rec["frame_token"], question)


def judge_fingerprints() -> dict:
    """What the judge freeze binds besides the request settings: the system
    line, every template the released scorer routes to, the recorded digest of
    the object-description file (judge input), and the score extractor."""
    return {
        "system_line_sha256": hashlib.sha256(SYSTEM_LINE.encode()).hexdigest(),
        "templates": {cell: template_sha(cell) for cell in PROMPT_BY_CELL},
        "visual_description_sha256": VISUAL_DESCRIPTION_SHA256,
        "extractor": extractor_fingerprint(),
    }


_MARKDOWN_CHARS = re.compile(r"[*_#|`]")
_TOTAL_LINE_RE = re.compile(r"Total Score\s*:?\s*([^\n]*)")
_OUT_OF_100 = re.compile(r"(?:/|out of)\s*100\b", re.IGNORECASE)
_COMMENT_PARENS = re.compile(r"\([^()+=]*\)")  # a balanced parenthesis with no operator inside is commentary
_DECIMAL = re.compile(r"\d[.,]\d")
_TERM = r"\d{1,3}"
_SUM = rf"{_TERM}(?:\s*\+\s*{_TERM})+"
_NUM_OR_SUM = rf"(?:{_SUM}|{_TERM})"
# the whole arithmetic forms accepted: `a + b = n`, `n = a + b`, `n (a + b)`; nothing but digit-free prose may follow
_ARITH_RE = re.compile(rf"^\s*(?P<a>{_NUM_OR_SUM})\s*(?P<mid>[^\d+=()]*?)\s*(?:=\s*(?P<b>{_NUM_OR_SUM})|\(\s*(?P<c>{_SUM})\s*\))\s*(?P<rest>[^\d+=()]*)$", re.S)
_PLAIN_RE = re.compile(rf"^\s*(?P<n>{_TERM})(?![.,]?\d)\b")


def _eval_sum(expr: str) -> int:
    return sum(int(t) for t in re.findall(_TERM, expr))


def _parse_total_segment(seg: str) -> tuple[int | None, bool]:
    """(value, arithmetic) for the text after `Total Score`. `/100`, `out of
    100` and balanced operator-free parentheses are commentary. A decimal
    attached to the number (`99.5`) is no score (the rubric is integer; nothing
    is truncated). Without `+`, `=` or parentheses the leading number is the
    score. Otherwise the whole segment must be one complete form, `a + b = n`,
    `n = a + b` or `n (a + b)`, with nothing but digit-free prose between or
    after the parts; both sides must evaluate to the same value. A sum without a stated total, an
    empty side, a trailing operator, or an unbalanced delimiter is no score."""
    seg = _COMMENT_PARENS.sub(" ", _OUT_OF_100.sub(" ", seg))
    if _DECIMAL.search(seg):
        return None, False
    if not re.search(r"[+=()]", seg):
        m = _PLAIN_RE.match(seg)
        return (int(m.group("n")) if m else None), False
    m = _ARITH_RE.match(seg)
    if not m:
        return None, True
    left = _eval_sum(m.group("a"))
    right = _eval_sum(m.group("b") or m.group("c"))
    if left != right:
        return None, True
    return left, True


def extract_score_rule(reply: str) -> tuple[int | None, str | None]:
    """(score, rule). The first `Total Score` line is read after markdown
    characters are stripped (`**Total Score:** 15/100`, a table cell, a sum
    `0 + 5 + 8 = **13**`). Rule `released` when the upstream regex
    `Total Score:\\s*(\\d+)` on the raw reply gives the same value, `markdown`
    when only the stripped line yields it, `arithmetic` when the line is a
    complete equation or parenthesised sum (_parse_total_segment). A value
    outside 0..100, a decimal, an incomplete or inconsistent expression is not
    a score."""
    reply = reply or ""
    text = _MARKDOWN_CHARS.sub(" ", reply.replace("\u202f", " ").replace("\u00a0", " "))
    m = _TOTAL_LINE_RE.search(text)
    if not m:
        return None, None
    val, arithmetic = _parse_total_segment(m.group(1))
    if val is None or not 0 <= val <= 100:
        return None, None
    if arithmetic:
        return val, "arithmetic"
    rm = _SCORE_RE.search(reply)
    return val, ("released" if rm and int(rm.group(1)) == val else "markdown")


def extract_score(reply: str) -> int | None:
    """Score under the current extractor (extract_score_rule)."""
    return extract_score_rule(reply)[0]


def extractor_fingerprint() -> dict:
    """Version and source digest of the score extractor, bound by the judge
    freeze: a parser change after the freeze is detected like a template edit."""
    src = "\n".join(inspect.getsource(f) for f in (_eval_sum, _parse_total_segment, extract_score_rule))
    # pattern and flags of every regex the extractor uses (IGNORECASE alone changes what parses)
    src += "\n".join(f"{p.pattern}\t{p.flags}" for p in (_SCORE_RE, _MARKDOWN_CHARS, _TOTAL_LINE_RE, _OUT_OF_100, _COMMENT_PARENS, _DECIMAL, _ARITH_RE, _PLAIN_RE))
    return {"version": EXTRACTOR_VERSION, "sha256": hashlib.sha256(src.encode()).hexdigest()}


def rescore(rec: dict) -> dict:
    """The record re-read under the current extractor: `score`, `score_rule`,
    `extractor_version` and `error` derived again from the stored `reply`,
    which is never changed. No reply (transport error, empty answer) means no
    score, whatever number the record carried. A record rejected when it was
    written (`rejected`: a reply from another backend than the frozen one)
    stays rejected: its reply is kept, its score is never restored."""
    out = dict(rec, extractor_version=EXTRACTOR_VERSION)
    if rec.get("rejected"):
        out.update(score=None, score_rule=None, error=rec["rejected"])
        return out
    if not rec.get("reply"):
        out.update(score=None, score_rule=None, error=rec.get("error") or "empty reply")
        return out
    score, rule = extract_score_rule(rec["reply"])
    out.update(score=score, score_rule=rule, error=None if score is not None else "no Total Score in reply")
    return out


def judge_messages(prompt: str) -> list[dict]:
    """Upstream sends system line and prompt joined by a newline as ONE user message."""
    return [{"role": "user", "content": "\n".join([SYSTEM_LINE, prompt])}]


# ---------------------------------------------------------------- settings and identity


def judge_base_url(jcfg: dict) -> str:
    return (jcfg.get("base_url") or DEFAULT_BASE_URL).rstrip("/")


def judge_settings(jcfg: dict) -> dict:
    """Request settings from a config `judge:` section (or the `judge` block of
    manifests/judge_frozen.yaml): the provider endpoint (`base_url`, the
    default when null), the model, the length parameter, and the optional
    settings. Null optional settings are dropped, so they are omitted from the
    request and absent from the hash."""
    model = jcfg.get("model")
    if not model:
        raise JudgeSettingsError("judge.model is not set")
    if jcfg.get("max_tokens") is None:
        raise JudgeSettingsError("judge.max_tokens is not set")
    param = jcfg.get("max_tokens_param") or "max_tokens"
    if param not in TOKEN_PARAMS:
        raise JudgeSettingsError(f"judge.max_tokens_param must be one of {TOKEN_PARAMS}, got {param!r}")
    out = {"base_url": judge_base_url(jcfg), "model": model, "max_tokens": int(jcfg["max_tokens"]), "max_tokens_param": param}
    for k in OPTIONAL_PARAMS:
        if jcfg.get(k) is not None:
            out[k] = jcfg[k]
    return out


def judge_backend(jcfg: dict) -> dict:
    """The non-secret backend description from config `judge.backend` (keys
    BACKEND_KEYS, null values dropped). Written into every record as
    `judge_backend` and into the judge freeze as `backend`; a final-split run
    refuses when the live block differs from the frozen one."""
    b = jcfg.get("backend") or {}
    unknown = sorted(set(b) - set(BACKEND_KEYS))
    if unknown:
        raise JudgeSettingsError(f"judge.backend has unknown keys {unknown}; allowed: {list(BACKEND_KEYS)}")
    return {k: b[k] for k in BACKEND_KEYS if b.get(k) is not None}


def is_local_endpoint(base_url: str) -> bool:
    """True for `localhost` and every loopback address, IPv6 included
    (`http://[::1]:8001/v1`, `http://127.0.0.2:8001`)."""
    host = urlsplit(base_url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def backend_description(backend: dict | None) -> dict:
    """The config-controlled part of a backend block (BACKEND_KEYS, null dropped)."""
    return {k: v for k, v in (backend or {}).items() if k in BACKEND_KEYS and v is not None}


def record_backend_status(rec: dict, backend: dict | None) -> str | None:
    """The one validity predicate for a judge record against an expected
    backend, shared by resume (08), the judge freeze (validation evidence) and
    production scoring (05): None when the record was written under the same
    backend description and, when `backend` names a `response_model`, the
    provider reported that model; otherwise the reason. A record written before
    the description existed carries none, which equals an empty description:
    earlier trials stay valid for development and no identity is invented."""
    if backend is None:
        return None
    want, have = backend_description(backend), backend_description(rec.get("judge_backend"))
    if want != have:
        return f"backend description differs: record {have or '{}'}, expected {want or '{}'}"
    if backend.get("response_model") is not None and rec.get("response_model") != backend["response_model"]:
        return f"backend identity: response model {rec.get('response_model')!r} differs from the frozen {backend['response_model']!r}"
    return None


def request_body(settings: dict, prompt: str) -> dict:
    """The chat-completions request exactly as sent. Optional settings appear
    only when present in `settings`; the endpoint is not a request field."""
    body = {"model": settings["model"], "messages": judge_messages(prompt), settings["max_tokens_param"]: settings["max_tokens"]}
    for k in OPTIONAL_PARAMS:
        if k in settings:
            body[k] = settings[k]
    return body


def input_sha256(body: dict, base_url: str) -> str:
    """Hash of the provider endpoint and the serialized request: model, every
    setting sent, and the message (system line, template, question, ground
    truth, description, prediction). The same request to another provider is
    another judge."""
    payload = {"endpoint": base_url, "request": body}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def judge_input_hash(cell: str, question: str, gt: str, pred: str, desc: str | None, settings: dict) -> str:
    return input_sha256(request_body(settings, build_prompt(cell, question, gt, pred, desc)), settings["base_url"])


def score_key(rec: dict) -> ScoreKey:
    """Resume key. Records written before step 4 carry neither field and get
    (question_id, 0, None): they never match a current hash."""
    return (rec["question_id"], int(rec.get("replicate") or 0), rec.get("input_sha256"))


# ---------------------------------------------------------------- requests


def _usage_dict(usage) -> dict | None:
    if usage is None:
        return None
    out = {"prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None), "total_tokens": getattr(usage, "total_tokens", None)}
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
    if reasoning is not None:
        out["reasoning_tokens"] = reasoning
    return out


def request_judge(client, body: dict, retries: int = 5) -> dict:
    """One judge call with the given request body. Retries transport errors
    with backoff; never loops forever. Measured usage is recorded as sent back
    by the provider (the cost note uses these numbers)."""
    err = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**body)
            reply = (resp.choices[0].message.content or "").strip()
            score, rule = extract_score_rule(reply)
            return {
                "reply": reply,
                "score": score,
                "score_rule": rule,
                "extractor_version": EXTRACTOR_VERSION,
                "usage": _usage_dict(getattr(resp, "usage", None)),
                "latency_s": round(time.perf_counter() - t0, 3),
                # what the provider says answered: the model id it returns (an alias
                # for a proxy, the served name for vLLM), its response id and
                # fingerprint, and when; the judge freeze binds the response model
                "response_model": getattr(resp, "model", None),
                "response_id": getattr(resp, "id", None),
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "created": getattr(resp, "created", None),
                "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "error": None if score is not None else "no Total Score in reply",
            }
        except Exception as e:  # noqa: BLE001 - record, back off, retry
            err = repr(e)
            if _is_request_error(e):
                break  # a rejected parameter or model does not get better with retries
            time.sleep(min(2**attempt, 30))
    return {"reply": "", "score": None, "score_rule": None, "extractor_version": EXTRACTOR_VERSION, "usage": None, "latency_s": None, "response_model": None, "response_id": None, "system_fingerprint": None, "created": None, "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": err}


def _is_request_error(e: Exception) -> bool:
    """4xx from the provider (unsupported parameter, unknown model, bad key)."""
    status = getattr(e, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 429


def compat_check(client, settings: dict, prompt: str, fallback: bool = True) -> list[dict]:
    """Send the configured request once; if it is rejected and `fallback` is
    on, probe the baseline and each optional setting on its own so the
    rejected ones are named. Fallback probes change the request, so they are
    only for development or synthetic prompts: a probe carrying a final-split
    answer runs with `fallback=False` and stops at the first rejection.
    Returns one row per attempt: {params, accepted, error, usage, latency_s, score}."""

    def attempt(label: str, s: dict) -> dict:
        res = request_judge(client, request_body(s, prompt), retries=1)
        return {"params": label, "accepted": res["error"] is None or res["error"] == "no Total Score in reply", "error": res["error"], "usage": res["usage"], "latency_s": res["latency_s"], "score": res["score"]}

    optional = [k for k in OPTIONAL_PARAMS if k in settings]
    full = attempt("configured: " + ", ".join([settings["max_tokens_param"], *optional]), settings)
    rows = [full]
    if full["accepted"] or not fallback:
        return rows
    base = {k: v for k, v in settings.items() if k not in OPTIONAL_PARAMS}
    rows.append(attempt(f"baseline: {settings['max_tokens_param']} only", base))
    if not rows[-1]["accepted"] and settings["max_tokens_param"] == "max_tokens":
        rows.append(attempt("baseline: max_completion_tokens only", {**base, "max_tokens_param": "max_completion_tokens"}))
    base_ok = next((r for r in rows[1:] if r["accepted"]), None)
    if base_ok is None:
        return rows
    base_param = "max_completion_tokens" if "max_completion_tokens" in base_ok["params"] else "max_tokens"
    for k in optional:
        rows.append(attempt(f"{base_param} + {k}={settings[k]!r}", {**base, "max_tokens_param": base_param, k: settings[k]}))
    return rows


# ---------------------------------------------------------------- sidecars


def load_scores(path: str | Path) -> dict[ScoreKey, dict]:
    """Judge sidecar -> {(question_id, replicate, input_sha256): record}. The
    last record per key wins (a retry of the same request); different keys
    coexist (replicates, re-judged answers, changed settings). Errored records
    are kept so the caller can see them but are not treated as scored."""
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[ScoreKey, dict] = {}
    with open(p) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[score_key(rec)] = rec
    return out


def scored_keys(scores: dict[ScoreKey, dict]) -> set[ScoreKey]:
    return {k for k, r in scores.items() if r.get("score") is not None}


def scored_ids(scores: dict[ScoreKey, dict]) -> set[str]:
    """Question ids with at least one scored record of any key (reporting only;
    production scoring uses production_scores)."""
    return {k[0] for k in scored_keys(scores)}


def usable_score(rec: dict, backend: dict | None) -> tuple[dict | None, str | None]:
    """(record re-read under the current extractor, None) when the record counts
    as a score under `backend`, else (None, reason). The one test of a judge
    record's validity: written under the expected backend
    (record_backend_status) and a stored reply that parses now (rescore). The
    stored number is never the criterion. Used by resume (08), the judge freeze
    (validation evidence), the 08 summary and production scoring (05)."""
    why = record_backend_status(rec, backend)
    if why:
        return None, why
    live = rescore(rec)
    if live.get("score") is None:
        if rec.get("reply") and not rec.get("rejected"):
            return None, "stored reply yields no score under the current extractor"
        return None, f"no usable reply ({live.get('error')})"
    if live["score"] != rec.get("score"):
        live["score_recorded"] = rec.get("score")
    return live, None


def production_scores(scores: dict[ScoreKey, dict], records: list[dict], settings: dict, desc_data: dict | None, question_text: dict[str, str] | None = None, backend: dict | None = None) -> tuple[dict[str, dict], dict[str, str]]:
    """Select, per answer record, the replicate-0 judge record whose input hash
    equals the hash of the current answer under `settings`, with the object
    description derived from the hash-checked description file (never from
    stored judge rows, so the selection cannot depend on which rows exist or
    their order). The selected record's score is re-read from its stored reply
    under the current extractor (the stored number is never trusted; when it
    differs it is kept as `score_recorded`; no reply, no score), and when
    `backend` is given (the frozen backend: description and response model)
    the record must satisfy record_backend_status.
    Returns ({question_id: judge record}, {question_id: reason}) where the
    second map names answers that have judge records but no usable one (stale
    answer, changed settings or template, replicate 0 missing, pre-step-4
    record, description lookup failure, reply without a parseable total,
    other backend)."""
    by_qid: dict[str, list[dict]] = {}
    for rec in scores.values():
        by_qid.setdefault(rec["question_id"], []).append(rec)
    selected, stale = {}, {}
    for r in records:
        qid = r["question_id"]
        cands = by_qid.get(qid)
        if not cands or r.get("error") or r.get("cell") not in PROMPT_BY_CELL:
            continue
        question = r.get("question") or (question_text or {}).get(qid)
        if question is None:
            stale[qid] = "question text missing"
            continue
        if all(c.get("input_sha256") is None for c in cands):
            stale[qid] = "pre-step-4 record without input_sha256"
            continue
        try:
            desc = expected_description(desc_data, r, question)
        except (KeyError, ValueError, JudgeInputError) as e:
            stale[qid] = f"description lookup: {e}"
            continue
        rec = scores.get((qid, 0, judge_input_hash(r["cell"], question, r["gt"], r["pred"], desc, settings)))
        if rec is None:
            if not any(int(c.get("replicate") or 0) == 0 for c in cands):
                stale[qid] = "replicate 0 missing"
            else:
                stale[qid] = "no replicate-0 record matches the current answer and judge settings"
            continue
        live, why = usable_score(rec, backend)
        if why:
            stale[qid] = why
            continue
        selected[qid] = live
    return selected, stale


def invocation_usage(written: list[dict]) -> dict:
    """Measured usage of the records one invocation just wrote (the cost note
    uses these; historical records under other keys never enter)."""
    out = {"calls": len(written), "with_usage": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    for r in written:
        if r.get("score") is None:
            out["errors"] += 1
        u = r.get("usage") or {}
        if u.get("prompt_tokens") is not None:
            out["with_usage"] += 1
            out["prompt_tokens"] += u["prompt_tokens"]
            out["completion_tokens"] += u.get("completion_tokens") or 0
            out["reasoning_tokens"] += u.get("reasoning_tokens") or 0
    n = out["with_usage"]
    out["prompt_tokens_mean"] = round(out["prompt_tokens"] / n, 1) if n else None
    out["completion_tokens_mean"] = round(out["completion_tokens"] / n, 1) if n else None
    return out


def sidecar_path(results_file: str | Path) -> Path:
    """results/<run>/bf16_clean.jsonl -> results/<run>/bf16_clean.judge.jsonl"""
    p = Path(results_file)
    return p.with_name(p.stem + ".judge.jsonl")
