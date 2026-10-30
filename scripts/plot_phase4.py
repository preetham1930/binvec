#!/usr/bin/env python3
"""Recall vs single-query latency, binvec vs FAISS (Deep 10M). Writes results/phase4-10M.png.

Each line is the best latency reachable at or above each recall level (the Pareto frontier of
that system's parameter sweep), from results/phase4-binvec-10M.csv and phase4-faiss-10M.csv.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = sys.argv[1] if len(sys.argv) > 1 else "results"


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def frontier(pts):
    # pts: (recall, latency, label). Keep points not dominated (higher recall and lower latency).
    pts = sorted(pts, key=lambda p: (-p[0], p[1]))
    out, best = [], float("inf")
    for p in pts:
        if p[1] < best:
            out.append(p)
            best = p[1]
    return sorted(out)


series = {}
series["binvec (1-bit + 4-bit in RAM, rerank from file)"] = frontier(
    [(float(r["recall10"]), float(r["p50_ms"]), r) for r in rows(f"{res}/phase4-binvec-10M.csv")])
names = {
    "ivf-pq4fs+refine-flat": "FAISS IVF-PQ 4-bit FastScan + full-vector rerank",
    "ivf-rabitq-fs+refine-flat": "FAISS IVF-RaBitQ FastScan + full-vector rerank",
    "ivf-flat": "FAISS IVF-Flat (exact within lists)",
}
fr = rows(f"{res}/phase4-faiss-10M.csv")
for cfg, name in names.items():
    series[name] = frontier([(float(r["recall10"]), float(r["p50_ms"]), r) for r in fr if r["config"] == cfg])

INK, INK2, GRID, SURF = "#1f1f1e", "#62615b", "#e4e3de", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for (name, pts), c, m in zip(series.items(), COLORS, MARKERS):
    pts = [p for p in pts if p[0] >= 0.95]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, color=c, lw=2, marker=m, ms=6, mec=SURF, mew=1.5, label=name, zorder=3)
ax.set_yscale("log")
ax.set_xlim(0.95, 1.0)
ax.set_yticks([0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
ax.set_xlabel("recall@10", color=INK2)
ax.set_ylabel("p50 latency, one query on one thread (ms, log scale)", color=INK2)
ax.grid(True, which="major", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)
ax.set_title("Deep 10M, same laptop: recall vs latency (lower right is better)",
             loc="left", color=INK, fontsize=11.5, pad=12)
leg = ax.legend(loc="upper left", frameon=False, fontsize=8.8, labelcolor=INK)
fig.text(0.01, 0.012, "Best p50 at each recall over each system's parameter sweep; FAISS 1.15.1.\n"
         "Index RAM: binvec 0.8 GB (full vectors read from the file); FAISS IVF-PQ / RaBitQ + rerank "
         "4.1-4.2 GB; IVF-Flat 3.9 GB.", fontsize=7.5, color=INK2, linespacing=1.5)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(f"{res}/phase4-10M.png", facecolor=SURF)
print("wrote", f"{res}/phase4-10M.png")
for name, pts in series.items():
    print(name)
    for p in pts:
        if p[0] >= 0.95:
            r = p[2]
            extra = f"np={r['nprobe']} " + (f"rk={r['refine_k']} rr={r['rerank']}" if 'refine_k' in r else f"kf={r['k_factor']}")
            print(f"   {p[0]:.4f} p50 {p[1]:.3f} p99 {float(r['p99_ms']):.3f} qps {r['qps']}  {extra}")
