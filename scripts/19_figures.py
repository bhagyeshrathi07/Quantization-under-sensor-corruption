"""Figures for the report as SVG, from 05's JSON. No plotting library.

    python scripts/19_figures.py --compare results/final/compare.json --out-dir results/final/figures

F1 `fh_by_margin.svg`: the harmful-flip rate per BF16 margin bin, pooled over
the conditions the comparison was run on (exploratory, table T4), with the
95% scene-bootstrap interval of each bin and the harmful / BF16-correct
counts on the bars. An interval of [0, 0] means no harmful flip was observed
in that bin, not a zero population risk; the figure says so.
"""
import argparse
import json
from pathlib import Path


def fh_by_margin_svg(groups: list[dict]) -> str:
    bins = []
    for g in groups:
        n_ref_correct = round(g["ref_acc"] * g["n"])
        lo, hi = g["ci"]["harmful_rate"]
        bins.append((g["label"], g["n"], g["harmful"], n_ref_correct, g["harmful_rate"], lo, hi))
    W, H, L, B, T, R = 560, 380, 70, 70, 56, 20
    ph, pw = H - T - B, W - L - R
    ymax = max(0.35, max(hi for *_, hi in bins) * 1.15)

    def y(v):
        return T + ph - v / ymax * ph

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="18" fill="#222" font-weight="600">Harmful-flip rate by BF16 first-token margin bin</text>',
         f'<text x="{L}" y="36" fill="#555" font-size="12">exploratory, pooled over conditions (table T4); bars with 95% scene-bootstrap intervals; labels: harmful / BF16-correct</text>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="#444"/>', f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#444"/>']
    tick = 0.0
    while tick <= ymax + 1e-9:
        s.append(f'<line x1="{L-4}" y1="{y(tick):.1f}" x2="{L+pw}" y2="{y(tick):.1f}" stroke="#ddd"/>')
        s.append(f'<text x="{L-8}" y="{y(tick)+4:.1f}" text-anchor="end" fill="#444">{tick:.1f}</text>')
        tick += 0.1
    bw = pw / len(bins) * 0.5
    for i, (label, n, harmful, nrc, fh, lo, hi) in enumerate(bins):
        cx = L + pw * (i + 0.5) / len(bins)
        s.append(f'<rect x="{cx-bw/2:.1f}" y="{y(fh):.1f}" width="{bw:.1f}" height="{T+ph-y(fh):.1f}" fill="#2a4d7a"/>')
        s.append(f'<line x1="{cx:.1f}" y1="{y(lo):.1f}" x2="{cx:.1f}" y2="{y(hi):.1f}" stroke="#111" stroke-width="2"/>')
        for v in (lo, hi):
            s.append(f'<line x1="{cx-6:.1f}" y1="{y(v):.1f}" x2="{cx+6:.1f}" y2="{y(v):.1f}" stroke="#111" stroke-width="2"/>')
        s.append(f'<text x="{cx:.1f}" y="{T+ph+18}" text-anchor="middle" fill="#222">{label}</text>')
        s.append(f'<text x="{cx:.1f}" y="{T+ph+34}" text-anchor="middle" fill="#666">{n} pairs</text>')
        s.append(f'<text x="{cx:.1f}" y="{y(hi)-8:.1f}" text-anchor="middle" fill="#222">{fh:.3f} ({harmful}/{nrc})</text>')
    s.append(f'<text x="{L+pw/2:.1f}" y="{H-26}" text-anchor="middle" fill="#222">BF16 margin bin (probability gap between the top two options at the first token)</text>')
    s.append(f'<text x="{L+pw/2:.1f}" y="{H-8}" text-anchor="middle" fill="#666" font-size="12">an interval of [0, 0] means no harmful flip was observed in that bin, not a zero population risk</text>')
    s.append(f'<text x="16" y="{T+ph/2:.1f}" text-anchor="middle" fill="#222" transform="rotate(-90 16 {T+ph/2:.1f})">harmful-flip rate F_h</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True, help="JSON written by 05 --out, run with --by margin among its factors")
    ap.add_argument("--out-dir", default="results/final/figures")
    args = ap.parse_args()
    comp = json.loads(Path(args.compare).read_text())
    groups = comp["mcq"]["groups"].get("margin")
    if not groups:
        raise SystemExit(f"{args.compare}: no margin groups (run 05 with --by margin)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "fh_by_margin.svg"
    p.write_text(fh_by_margin_svg(groups), encoding="utf-8")
    print(f"wrote {p}: " + ", ".join(f"{g['label']} {g['harmful']}/{round(g['ref_acc'] * g['n'])}" for g in groups))


if __name__ == "__main__":
    main()
