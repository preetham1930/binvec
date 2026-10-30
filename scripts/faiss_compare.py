#!/usr/bin/env python3
"""FAISS on the same data, same machine, same measurements as `binvec bench`.

Builds several FAISS indexes over the base vectors, sweeps their search parameters, and writes
one CSV row per setting: recall@10 against the same ground truth, single-thread latency
(one query per call), batch throughput on all cores, index size and build time.

Usage:
  faiss_compare.py --base base.fbin --queries query.fbin --gt gt.bin --nlist 4096 \
                   --csv results/phase4-faiss-10M.csv --label "deep10M laptop"
"""
import argparse
import csv
import os
import time

import faiss
import numpy as np


def read_fbin(path, limit=None):
    n, d = np.fromfile(path, dtype=np.uint32, count=2)
    n = int(n) if limit is None else min(int(n), limit)
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, int(d)))


def read_gt(path, k):
    nq, kk = np.fromfile(path, dtype=np.uint32, count=2)
    ids = np.fromfile(path, dtype=np.uint32, count=int(nq) * int(kk), offset=8)
    return ids.reshape(int(nq), int(kk))[:, :k]


def recall_at(found, truth, k):
    return np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth)])


# (name, factory string, uses k_factor)
CONFIGS = [
    ("ivf-flat", "IVF{nlist},Flat", False),
    ("ivf-pq4fs+refine-flat", "IVF{nlist},PQ{m}x4fs,RFlat", True),
    ("ivf-rabitq-fs+refine-flat", "IVF{nlist},RaBitQfs,RFlat", True),
    ("ivf-rabitq4-fs+refine-flat", "IVF{nlist},RaBitQfs4,RFlat", True),
    ("ivf-rabitq-fs+refine-sq8", "IVF{nlist},RaBitQfs,Refine(SQ8)", True),
]


def index_gb(index, tmp_dir):
    # Size on disk = size in RAM, near enough. Writing streams the index to the file; serialising
    # it to bytes would need a second in-memory copy (4 GB at 10M with full vectors).
    path = os.path.join(tmp_dir, f"faiss-size-{os.getpid()}.index")
    faiss.write_index(index, path)
    size = os.path.getsize(path) / 1e9
    os.remove(path)
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--train-n", type=int, default=500_000)
    ap.add_argument("--nprobe", default="32,64,128,256")
    ap.add_argument("--k-factor", default="10,20,50,100,200")
    ap.add_argument("--configs", default=",".join(c[0] for c in CONFIGS))
    ap.add_argument("--nq-latency", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--tmp-dir", default="/tmp")
    a = ap.parse_args()

    k = 10
    base = read_fbin(a.base, a.limit)
    n, d = base.shape
    xq = np.ascontiguousarray(read_fbin(a.queries))
    gt = read_gt(a.gt, k)
    nq = min(len(xq), len(gt))
    xq, gt = xq[:nq], gt[:nq]
    nl = min(a.nq_latency, nq)
    rng = np.random.default_rng(1)
    train = np.ascontiguousarray(base[np.sort(rng.choice(n, min(a.train_n, n), replace=False))])
    print(f"faiss {faiss.__version__}, n={n} d={d} nq={nq}, threads={a.threads}", flush=True)

    header = ["label", "system", "config", "n", "nlist", "nprobe", "k_factor", "recall10",
              "p50_ms", "p99_ms", "mean_ms", "qps", "threads", "index_gb", "build_s"]
    new = not os.path.exists(a.csv)
    out = open(a.csv, "a", newline="")
    w = csv.writer(out)
    if new:
        w.writerow(header)
    print(f"{'config':28} {'nprobe':>6} {'kfac':>4} {'recall@10':>9} {'p50 ms':>8} {'p99 ms':>8} "
          f"{'QPS':>7} {'index GB':>8} {'build s':>7}", flush=True)

    m = d // 2  # PQ sub-quantizers of 2 dims, 4 bits each: d/4 bytes per vector
    wanted = a.configs.split(",")
    for name, fac, uses_kf in CONFIGS:
        if name not in wanted:
            continue
        faiss.omp_set_num_threads(a.threads)
        t0 = time.time()
        index = faiss.index_factory(d, fac.format(nlist=a.nlist, m=m))
        index.train(train)
        step = 1_000_000
        for i in range(0, n, step):
            index.add(np.ascontiguousarray(base[i:i + step]))
        build_s = time.time() - t0
        size_gb = index_gb(index, a.tmp_dir)
        ivf = faiss.extract_index_ivf(index)
        for np_ in map(int, a.nprobe.split(",")):
            for kf in (map(int, a.k_factor.split(",")) if uses_kf else [0]):
                ivf.nprobe = np_
                if uses_kf:
                    index.k_factor = kf  # rerank k_factor * k candidates with the refine codes
                # Latency: one query per call, one thread, as binvec measures it.
                faiss.omp_set_num_threads(1)
                lat = []
                for i in range(nl):
                    t = time.perf_counter()
                    index.search(xq[i:i + 1], k)
                    lat.append((time.perf_counter() - t) * 1e3)
                lat.sort()
                p50, p99 = lat[int(0.5 * nl + 0.5) - 1], lat[min(nl - 1, int(np.ceil(0.99 * nl)) - 1)]
                mean = sum(lat) / nl
                # Throughput and recall: all queries in one batch, all threads.
                faiss.omp_set_num_threads(a.threads)
                t = time.perf_counter()
                _, I = index.search(xq, k)
                qps = nq / (time.perf_counter() - t)
                rec = recall_at(I, gt, k)
                print(f"{name:28} {np_:>6} {kf:>4} {rec:>9.4f} {p50:>8.3f} {p99:>8.3f} {qps:>7.0f} "
                      f"{size_gb:>8.2f} {build_s:>7.0f}", flush=True)
                w.writerow([a.label, f"faiss {faiss.__version__}", name, n, a.nlist, np_, kf,
                            f"{rec:.4f}", f"{p50:.3f}", f"{p99:.3f}", f"{mean:.3f}", f"{qps:.0f}",
                            a.threads, f"{size_gb:.3f}", f"{build_s:.1f}"])
                out.flush()
        del index


if __name__ == "__main__":
    main()
