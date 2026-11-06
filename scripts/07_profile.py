"""Deployment profile for one served precision: time to first token, latency,
decode throughput, and peak GPU memory, over a fixed question slice.

    bash scripts/01_serve.sh bf16            # KV_BYTES must be identical across precisions
    python scripts/07_profile.py --precision bf16 --session 1
    python scripts/07_profile.py --precision bf16 --session 2   # separate invocation = separate session
    python scripts/07_profile.py --precision int4 --session 1
    python scripts/07_profile.py --report results/profile/*.json --out results/profile/pooled.json

Each session writes results/profile/<precision>_<condition>_s<k>.json with
per-request timings, p50/p95/p99 summaries, the peak GPU memory seen while the
slice ran, the server manifest from 01_serve.sh, a provenance block (the same
compatibility fields as a result sidecar: checkpoint fingerprint, serve
settings, inputs) and the host, so a session is self-describing. Three
sessions per precision. --report pools sessions that share precision,
checkpoint, serve settings and host (plan section 6): percentiles over the
concatenated requests, per-session rows kept, duplicate session numbers
refused; files without a provenance block (the November 3 official-checkpoint
reference) are listed unpooled via --reference. Warm-up requests are
excluded. The slice is the first N six-view questions of the development
split (fixed order, so every session and precision sees the same prompts).

Budgets (proposal 5.2): p95 TTFT <= 2 s, decode >= 15 tok/s, INT4 peak memory
at least 25% below BF16 at equal KV-cache bytes.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.client import chat_stream, make_client  # noqa: E402
from qflip.config import load_config  # noqa: E402
from qflip.drivebench import load_questions  # noqa: E402
from qflip.freeze import FreezeError  # noqa: E402
from qflip.gpu import PeakMemorySampler, query_gpu_info  # noqa: E402
from qflip.guard import prepare_measurement  # noqa: E402
from qflip.provenance import ProvenanceError  # noqa: E402
from qflip.report import ProfileError, fmt, pool_profiles, profile_summary  # noqa: E402,F401
from qflip.runner import messages_for, read_system_prompt  # noqa: E402
from qflip.split import filter_split  # noqa: E402


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def print_report(paths: list[str], reference: list[str] = (), out: str | None = None) -> None:
    rows = [json.loads(Path(p).read_text()) for p in paths]
    refs = [json.loads(Path(p).read_text()) for p in reference]
    try:
        pooled = pool_profiles(rows, refs)
    except ProfileError as e:
        sys.exit(f"pooling refused: {e}")
    for prec, p in pooled["precisions"].items():
        s = p["pooled"]
        print(f"{prec}: {p['n_sessions']} session(s) pooled, {s['requests']} requests, {s['errors']} errors; TTFT p50 {fmt(s['ttft_s']['p50'])} p95 {fmt(s['ttft_s']['p95'])}; "
              f"latency p50 {fmt(s['latency_s']['p50'], 2)} p95 {fmt(s['latency_s']['p95'], 2)}; decode p50 {fmt(s['decode_tok_s']['p50'], 1)} tok/s; peak {s['gpu_peak_mib_max']} MiB (per session {s['gpu_peak_mib_per_session']})")
    if pooled["memory_reduction"] is not None:
        print(f"peak memory reduction (pooled max): {pooled['memory_reduction'] * 100:.1f}% (budget >= 25%)")
    print(f"budgets: {json.dumps(pooled['budgets'])}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(pooled, indent=2))
        print(f"wrote {out}")
    print(f"\n{'file':34s} {'peak MiB':>9s} {'TTFT p50':>9s} {'TTFT p95':>9s} {'lat p50':>8s} {'lat p95':>8s} {'tok/s p50':>10s} {'kv_bytes':>12s}")
    for r in rows + refs:
        s = r["summary"]
        kv = (r.get("serve_manifest") or {}).get("kv_args")
        print(
            f"{Path(r['file']).name:34s} {str(r.get('gpu_peak_mib')):>9s} {fmt(s['ttft_s']['p50']):>9s} {fmt(s['ttft_s']['p95']):>9s} "
            f"{fmt(s['latency_s']['p50'], 2):>8s} {fmt(s['latency_s']['p95'], 2):>8s} {fmt(s['decode_tok_s']['p50'], 1):>10s} "
            f"{kv.split()[-1] if kv else '?':>12s}" + ("  (reference, unpooled)" if r in refs else "")
        )


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
    ap.add_argument("--exclusions-file", default="manifests/scene_exclusions.json")
    ap.add_argument("--out-dir", default="results/profile")
    ap.add_argument("--report", nargs="*", help="pool existing profile JSON files instead of running")
    ap.add_argument("--reference", nargs="*", default=[], help="with --report: sessions listed unpooled (official-checkpoint reference, files without a provenance block)")
    ap.add_argument("--out", help="with --report: write the pooled JSON here (input of 09_tables.py)")
    ap.add_argument("--exploratory", action="store_true", help="skip the core-freeze check (pilot measurement); the file is marked and is never pooled into the study")
    args = ap.parse_args()

    if args.report is not None:
        print_report(args.report, args.reference, args.out)
        return
    if not args.precision:
        sys.exit("--precision required unless --report")

    cfg = load_config(args.config)
    model = cfg["models"][args.precision]
    # Before any request: the served alias at the client's endpoint, the listener owned by
    # the manifest's process, the loaded model, and (study sessions) the core freeze.
    try:
        provenance = prepare_measurement(cfg, study=not args.exploratory, config_path=args.config, precision=args.precision, condition=args.condition,
                                         split=args.split, split_file=args.split_file, exclusions_file=args.exclusions_file)
    except (ProvenanceError, FreezeError) as e:
        sys.exit(f"profile refused: {e}")
    print(f"server ok: {provenance['served_model_name']} = {provenance['model_path_or_id']} ({provenance['checkpoint_fingerprint']}); {'study session (freeze checked)' if not args.exploratory else 'EXPLORATORY session (no freeze check; never pooled)'}")
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
        "provenance": provenance, "study": not args.exploratory,
        "host": {"hostname": platform.node(), "gpu_name": query_gpu_info().get("name")},
        "summary": profile_summary(requests), "requests": requests,
    }
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}, indent=2))
    s = result["summary"]
    print(f"\nbudgets: TTFT p95 {s['ttft_s']['p95']} s (<= 2), decode p50 {s['decode_tok_s']['p50']} tok/s (>= 15), peak {mem.peak_mib} MiB")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
