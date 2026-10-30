# binvec

Approximate nearest-neighbour search over large vector sets on a single machine, written in Rust.

> **Deep 100M on a laptop: 0.980 recall@10 at 56 ms p99**, from a cold page cache, with a 2.4 GB
> in-RAM index and the 38 GB of full vectors read from SSD. Or 0.960 recall@10 at 24 ms p99.
> Deep 10M: 0.975 recall@10 at 2.4 ms p99.

## Results

Every number comes from `results/*.csv`, produced by `binvec bench`, with ground truth from the
[NeurIPS'21 Big-ANN benchmark](https://big-ann-benchmarks.com/neurips21.html). Latency is one
query at a time on one thread; QPS uses all 16 threads. Machine: Dell Inspiron 16 7610,
i7-11800H, 16 GB RAM, WSL2 (Kali) with 11 GB given to Linux, on mains power.

**Deep 10M, v0.1** (nlist = 4096, index RAM 0.24 GB; full vectors 3.8 GB memory-mapped)

| nprobe | rerank | recall@10 | p50 ms | p99 ms | QPS |
|---|---|---|---|---|---|
| 8 | 500 | 0.811 | 0.47 | 0.77 | 13,489 |
| 32 | 500 | 0.914 | 1.21 | 2.01 | 2,748 |
| 64 | 500 | 0.927 | 1.61 | 2.47 | 2,683 |
| 128 | 500 | 0.931 | 2.98 | 4.08 | 2,321 |
| 128 | 0 (bits only) | 0.332 | 2.68 | 4.03 | 1,970 |

**What this shows:** doubling nprobe from 64 to 128 barely raises recall, and bits-only recall is
stuck near 0.33 at every nprobe. So the lost recall isn't from clustering; it's from the 1-bit
estimate ranking true neighbours too low to make the rerank list. v0.2 targets exactly that.

**Deep 10M, v0.2: which fix mattered** (nprobe = 64; `results/ablation.csv`)

| Variant | recall@10, rerank 100 | recall@10, rerank 500 | p99 ms (rerank 500) |
|---|---|---|---|
| v0.1: least-squares scale, no rotation | 0.757 | 0.927 | 2.50 |
| + random rotation only | 0.756 | 0.928 | 2.32 |
| + unbiased (RaBitQ) scale only | **0.871** | **0.975** | 2.93 |
| + both | 0.869 | 0.975 | 2.42 |

The unbiased estimator does all the work. Same codes, same memory, same speed: only the one float
stored per vector changes. Rotation made no measurable difference on Deep; it stays as an option.

**Deep 10M, v0.2 vs the ceiling.** The ceiling is the recall if every vector in the probed
clusters were scored exactly, i.e. the share of true neighbours that sit in those clusters.
Bits + rerank can't beat it.

| nprobe | recall@10 (rerank 500) | ceiling | p50 ms | p99 ms | QPS |
|---|---|---|---|---|---|
| 16 | 0.909 | 0.913 | 0.63 | 0.95 | 11,906 |
| 32 | 0.955 | 0.962 | 1.00 | 1.52 | 7,238 |
| 64 | 0.975 | 0.986 | 1.64 | 2.42 | 3,775 |
| 128 | — | 0.995 | — | — | — |

After v0.2 the codes lose at most 1.1 points against the ceiling; the remaining gap is clustering.
Laptop timings vary roughly ±20% between runs (heat, background load), so compare recall
columns with more confidence than QPS.

**Deep 100M, cold cache** (nlist = 16,384, index RAM 2.41 GB; 38 GB of vectors on SSD, 11 GB of
RAM for Linux). Every setting runs as its own process after emptying the Linux page cache, so
nothing is pre-loaded. Latency: 500 queries, one thread. Recall and QPS: 5,000 queries, 16
threads. `results/results-100M-cold.csv`

| nprobe | rerank | recall@10 | ceiling | p50 ms | p99 ms | p99 ms, no prefetch | QPS |
|---|---|---|---|---|---|---|---|
| 32 | 500 | 0.924 | 0.942 | 8.0 | 16.0 | 146.0 | 201 |
| 32 | 1000 | 0.937 | 0.942 | 25.1 | 45.0 | 232.0 | 60 |
| 64 | 500 | 0.950 | 0.975 | 18.8 | 38.0 | 418.5 | 124 |
| 64 | 1000 | 0.967 | 0.975 | 56.1 | 98.2 | 289.8 | 36 |
| 128 | 500 | 0.960 | 0.991 | 13.4 | 23.9 | 117.2 | 100 |
| 128 | 1000 | **0.980** | 0.991 | 35.3 | **56.3** | 251.4 | 49 |

- **Reranking reads from disk, and prefetch is what makes it fast.** Handing all of a query's
  rerank reads to the SSD at once cut p99 by 3–11×. Without it, every setting with rerank 500 or more
  misses 100 ms.
- **Rerank depth has to grow with the data.** At 100M each probed cluster holds 2.5× more
  vectors than at 10M, so rerank 500 leaves 3 points on the table at nprobe 128. Rerank 1000
  brings the gap back to ~1 point, as at 10M.
- **Single runs are noisy.** Some rows can't all be right: nprobe 128 / rerank 500 came out
  faster than nprobe 64 / rerank 500, despite scanning twice as many codes. Laptop timings under
  WSL2 swing with Windows background work, heat and SSD state. `scripts/repeat_100m.sh` re-runs
  the headline settings 3× and reports the median and range.
- The first 100M run (`results/results-100M.csv`) is superseded: its warm-up re-used the timed
  queries and all settings shared one process, so later settings found vectors already cached.

**Build at 100M:** 53 minutes in total. Data check (one pass over 38 GB) 105 s, k-means on 1M
points 632 s, assigning 100M vectors to 16,384 centroids 2,210 s (70% of the build), encoding
319 s.

## How it works

A query runs in three stages:

1. **Pick clusters.** k-means splits the vectors into `nlist` clusters. The query is compared
   with every cluster centre and only the `nprobe` nearest clusters are searched.
2. **Scan compressed codes in RAM.** Each vector is stored as its offset from its cluster centre,
   compressed to **1 bit per dimension** (the sign) plus two floats: 20 bytes instead of 384 for a
   96-dim vector. Distances are estimated from the bits with a per-byte lookup table: 12 table
   lookups per vector, no multiplications.
3. **Rerank exactly.** The best few hundred estimates are rescored with the full-precision vectors,
   which stay on disk (memory-mapped) and are read only for those candidates.

Stages 1–2 throw away most of the precision to be fast; stage 3 buys the accuracy back.
This follows the approach Exa describes in
[How we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db),
with these changes: binarising the **residual** (vector minus its cluster centre) rather than the raw
vector, using **8-dimension** lookup tables (256 entries) instead of 4, and (v0.2) a **random
rotation** plus the **unbiased estimator** from
[RaBitQ (Gao & Long, SIGMOD 2024)](https://arxiv.org/abs/2405.12497).

## Reproduce

Needs Linux (WSL2 is fine), Rust, ~5 GB disk for 10M or ~45 GB for 100M.

```bash
scripts/download_deep.sh 10        # Deep1B: first 10M vectors, queries, ground truth
scripts/run_10m.sh                 # verify data, build index, benchmark
scripts/ablation_10m.sh            # v0.2 comparison: rotation and estimator on/off

scripts/download_deep.sh 100       # first 100M vectors (~38 GB; reuses the 10M file)
scripts/build_100m.sh              # verify + build (unattended, ~30-45 min)
scripts/bench_100m.sh              # benchmark from a cold page cache (asks for sudo once)
scripts/repeat_100m.sh             # headline settings, 3 cold runs each, median + range
```

Synthetic data, no download:

```bash
cargo build --release
B=target/release/binvec
$B gen --n 1000000 --out-dir data/syn1m
$B build --base data/syn1m/base.fbin --nlist 1024 --out data/syn1m/idx.bin
$B bench --index data/syn1m/idx.bin --base data/syn1m/base.fbin \
         --queries data/syn1m/query.fbin --gt data/syn1m/gt.bin
```

## Roadmap

- [x] k-means IVF, residual binary codes, byte lookup-table scan, exact rerank
- [x] Benchmark harness: recall@10, p50/p99 latency, QPS, RAM, CSV output
- [x] Deep 10M results (v0.1)
- [x] v0.2 ablation: rotation × estimator, plus recall ceiling of the probed clusters
- [x] Build path for 100M: batched centroid assignment, sequential reads, one-pass brute force
- [x] Deep 100M first run, with and without parallel rerank prefetch
- [x] Deep 100M clean cold-cache latency, rerank up to 1000
- [ ] Median of 3 cold runs for the headline settings
- [ ] Close the 3-point gap to the ceiling at 100M: a second, finer in-RAM code tier before the disk rerank
- [ ] Faster build: approximate centroid assignment (70% of build time)
- [ ] SIMD scan kernel (quantised lookup tables with AVX2/AVX-512 shuffles)
- [ ] Filtered search with inverted indexes
- [ ] Head-to-head with FAISS IVF-PQ on the same machine
- [ ] Write-up: what broke between 10M and 100M

## Layout

| File | What it does |
|---|---|
| `src/ivf.rs` | The index: build, search, save/load |
| `src/kmeans.rs` | k-means clustering |
| `src/dist.rs` | Distance kernels |
| `src/io.rs` | Big-ANN file formats, memory-mapped base vectors |
| `src/topk.rs` | Bounded heap for the k best results |
| `src/bin/binvec.rs` | CLI: `gen`, `verify`, `build`, `bench`, `ceiling` |
| `results/` | Raw CSV output of every benchmark run quoted above |
