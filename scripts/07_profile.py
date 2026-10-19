"""Deployment profile for one served precision: time to first token, latency,
decode throughput, and peak GPU memory, over a fixed question slice.

    bash scripts/01_serve.sh bf16            # KV_BYTES must be identical across precisions
    python scripts/07_profile.py --precision bf16 --session 1
    python scripts/07_profile.py --precision bf16 --session 2   # separate invocation = separate session
    python scripts/07_profile.py --precision int4 --session 1
    python scripts/07_profile.py --report results/profile/*.json

Each session writes results/profile/<precision>_<condition>_s<k>.json with
per-request timings, p50/p95/p99 summaries, the peak GPU memory seen while the
slice ran, and the server manifest from 01_serve.sh, so a session is
self-describing. Three sessions per precision; the report shows all three so
host variance is visible. Warm-up requests are excluded. The slice is the
first N six-view questions of the development split (fixed order, so every
session and precision sees the same prompts).

Budgets (proposal 5.2): p95 TTFT <= 2 s, decode >= 15 tok/s, INT4 peak memory
at least 25% below BF16 at equal KV-cache bytes.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.client import chat_stream, make_client  # noqa: E402
from qflip.config import load_config  # noqa: E402
from qflip.drivebench import load_questions  # noqa: E402
from qflip.gpu import PeakMemorySampler, query_gpu_info  # noqa: E402
from qflip.runner import messages_for, read_system_prompt  # noqa: E402
from qflip.split import filter_split  # noqa: E402
from qflip.stats import summarize_latencies  # noqa: E402


def profile_summary(requests: list[dict]) -> dict:
    ok = [r for r in requests if not r["error"] and r["ttft_s"] is not None]
    tok_s = [r["decode_tok_s"] for r in ok if r["decode_tok_s"] is not None]
    return {
        "requests": len(requests),
        "errors": len(requests) - len(ok),
        "ttft_s": summarize_latencies([r["ttft_s"] for r in ok]),
        "latency_s": summarize_latencies([r["latency_s"] for r in ok]),
        "decode_tok_s": summarize_latencies(tok_s),
        "prompt_tokens_mean": sum(r["prompt_tokens"] or 0 for r in ok) / len(ok) if ok else None,
        "completion_tokens_mean": sum(r["completion_tokens"] or 0 for r in ok) / len(ok) if ok else None,
    }


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def print_report(paths: list[str]) -> None:
    rows = [json.loads(Path(p).read_text()) for p in paths]
    print(f"{'file':34s} {'peak MiB':>9s} {'TTFT p50':>9s} {'TTFT p95':>9s} {'lat p50':>8s} {'lat p95':>8s} {'tok/s p50':>10s} {'kv_bytes':>12s}")
    for r in rows:
        s = r["summary"]
        print(
            f"{Path(r['file']).name:34s} {str(r['gpu_peak_mib']):>9s} {s['ttft_s']['p50']:9.3f} {s['ttft_s']['p95']:9.3f} "
            f"{s['latency_s']['p50']:8.2f} {s['latency_s']['p95']:8.2f} {s['decode_tok_s']['p50']:10.1f} "
            f"{r['serve_manifest'].get('kv_args', '?').split()[-1] if r['serve_manifest'].get('kv_args') else '?':>12s}"
        )
    by_prec = {}
    for r in rows:
        by_prec.setdefault(r["precision"], []).append(r["gpu_peak_mib"])
    if {"bf16", "int4"} <= set(by_prec):
        b = max(x for x in by_prec["bf16"] if x is not None)
        i = max(x for x in by_prec["int4"] if x is not None)
        print(f"\npeak memory: bf16 {b} MiB, int4 {i} MiB, reduction {(1 - i / b) * 100:.1f}% (budget >= 25%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "int4"])
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--views", type=int, default=6, help="only questions with this many views (0 = any)")
    ap.add_argument("--split", default="development")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--out-dir", default="results/profile")
    ap.add_argument("--report", nargs="*", help="summarize existing profile JSON files instead of running")
    args = ap.parse_args()

    if args.report is not None:
        print_report(args.report)
        return
    if not args.precision:
        sys.exit("--precision required unless --report")

    cfg = load_config(args.config)
    model = cfg["models"][args.precision]
    client = make_client(cfg)
    questions = [q for q in filter_split(load_questions(cfg["data"]["annotations"]), args.split, args.split_file)
                 if q.cell is not None and (args.views == 0 or q.n_views == args.views)]
    questions.sort(key=lambda q: q.question_id)
    slice_ = questions[: args.n]
    if len(slice_) < args.n:
        print(f"warning: only {len(slice_)} questions match (wanted {args.n})")
    system_prompt = read_system_prompt(cfg)
    gen = cfg["generation"]

    for q in slice_[: args.warmup]:
        chat_stream(client, model, messages_for(q, cfg, args.condition, system_prompt), gen)

    requests = []
    with PeakMemorySampler(interval_s=0.2) as mem:
        for q in tqdm(slice_, desc=f"{args.precision} s{args.session}"):
            c = chat_stream(client, model, messages_for(q, cfg, args.condition, system_prompt), gen)
            requests.append({
                "question_id": q.question_id, "cell": q.cell, "n_views": q.n_views,
                "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens,
                "ttft_s": c.ttft_s, "latency_s": round(c.latency_s, 4), "decode_tok_s": c.decode_tok_s,
                "error": c.error,
            })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.precision}_{args.condition}_s{args.session}.json"
    result = {
        "file": str(out), "precision": args.precision, "condition": args.condition, "model": model,
        "session": args.session, "n": len(slice_), "views": args.views, "split": args.split, "warmup": args.warmup,
        "generation": gen, "gpu": query_gpu_info(), "gpu_baseline_mib": mem.baseline_mib, "gpu_peak_mib": mem.peak_mib,
        "serve_manifest": read_manifest(Path("manifests") / f"serve_{args.precision}.txt"),
        "summary": profile_summary(requests), "requests": requests,
    }
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}, indent=2))
    s = result["summary"]
    print(f"\nbudgets: TTFT p95 {s['ttft_s']['p95']} s (<= 2), decode p50 {s['decode_tok_s']['p50']} tok/s (>= 15), peak {mem.peak_mib} MiB")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
