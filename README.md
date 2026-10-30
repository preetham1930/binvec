# binvec

Approximate nearest-neighbour search over large vector sets on a single machine, written in Rust.

> **Deep 100M on a laptop: 0.995 recall@10 at 10 ms p99** (median of 3 cold-cache runs), reading
> only 50 full vectors from SSD per query. In RAM: a 2.4 GB 1-bit index plus a 5.6 GB 4-bit
> refine tier; the 38 GB of full vectors stay on disk.
> Deep 10M: 0.995 recall@10 at 2.0 ms p99.

Every number below comes from a CSV in `results/`, written by `binvec bench`, with ground truth
from the [NeurIPS'21 Big-ANN benchmark](https://big-ann-benchmarks.com/neurips21.html).
Machine: Dell Inspiron 16 7610, i7-11800H (8 cores), 16 GB RAM, NVMe SSD, WSL2 (Kali) with 11 GB
given to Linux, on mains power. Latency is one query at a time on one thread; QPS uses 16 threads.

## Results (v0.5)

### Deep 100M, cold cache

"Cold" = the Linux page cache is emptied and a fresh process starts before every run, so the
full vectors really come from disk. The ceiling is the best recall possible at that nprobe: the
share of true neighbours inside the probed clusters.

**3 runs each, alternating order** (recall over 2,000 queries, latency over 500)

| nprobe | refine | rerank (disk reads) | recall@10 | ceiling | median p50 ms | median p99 ms | p99 range ms | QPS |
|---|---|---|---|---|---|---|---|---|
| 64 | 2,000 | 20 | 0.971 | 0.975 | 3.4 | 5.4 | 3.4 – 5.9 | 1,911 |
| 128 | 5,000 | 50 | 0.989 | 0.991 | 4.9 | 7.5 | 5.9 – 7.8 | 1,007 |
| 256 | 5,000 | 50 | **0.995** | 0.997 | 7.0 | **9.9** | 9.6 – 10.6 | 682 |
| 256 | 10,000 | 50 | 0.995 | 0.997 | 7.3 | 10.6 | 10.1 – 11.8 | 603 |
| 512 | 10,000 | 50 | 0.997 | 0.999 | 12.7 | 19.0 | one run | 346 |

(`results/phase2-100M-repeats.csv` for nprobe 64; `results/phase2b-repeats.csv` for the rest.)

**v0.4 → v0.5, same settings, same session, alternating runs** (`results/phase2-100M-repeats.csv`)

| nprobe / refine / rerank | recall@10 | p50 ms | p99 ms | scan ms | QPS |
|---|---|---|---|---|---|
| 64 / 2,000 / 20 | 0.971 | 6.7 → **3.4** | 11.9 → **5.4** | 5.6 → 1.8 | 694 → 1,911 |
| 128 / 5,000 / 50 | 0.989 | 13.1 → **5.2** | 21.4 → **9.1** | 10.7 → 3.0 | 297 → 863 |

The old path is v0.4's byte-table scan and heap, run from the v0.5 binary (so it already has the
faster refine stage). v0.4 as originally measured was 16.5 and 28.9 ms median p99
(`results/phase1b-repeats.csv`). Recall is identical to four decimals at every setting tested.

One run in the grid (`results/phase2-100M-grid.csv`, nprobe 256 / refine 5,000) had a 116 ms
p99; three repeats of the same setting gave 9.6–10.6 ms. It is reported, not used.

### Deep 10M, data in RAM

(`results/phase2b-10M.csv`; nlist 4,096; base file pre-loaded into the page cache)

| nprobe | refine | rerank | recall@10 | ceiling | p50 ms | p99 ms | QPS |
|---|---|---|---|---|---|---|---|
| 64 | 2,000 | 50 | 0.985 | 0.986 | 0.77 | 1.05 | 6,247 |
| 128 | 2,000 | 50 | 0.994 | 0.996 | 1.19 | 1.57 | 3,846 |
| 128 | 5,000 | 50 | **0.995** | 0.996 | 1.48 | **1.98** | 3,090 |

## How it works

A query runs in four stages; each throws away fewer candidates than the last and spends more
per candidate.

1. **Pick clusters.** k-means splits the vectors into `nlist` clusters (16,384 at 100M). The
   query is compared with every centre; only the `nprobe` nearest clusters are searched.
2. **1-bit scan, in RAM.** Each vector is stored as its offset from its cluster centre, randomly
   rotated and reduced to **1 bit per dimension** plus two floats: 20 bytes instead of 384.
   The scale uses the unbiased estimator from
   [RaBitQ (Gao & Long, SIGMOD 2024)](https://arxiv.org/abs/2405.12497).
   Codes are scored 32 at a time with AVX2 shuffles (FastScan, below), and the best few
   thousand are kept in a reservoir rather than a heap.
3. **4-bit refine, in RAM.** The best few thousand are re-scored from a **4-bit** copy of the
   same offset (56 bytes per vector), which is far more accurate than the bits. Candidates are
   visited in memory order, with the next few prefetched.
4. **Exact rerank, from SSD.** The best 20–50 are read from the full-precision file (memory-mapped)
   and scored exactly. All reads are issued at once (`madvise(WILLNEED)`), so the SSD serves them
   in parallel.

| Tier | Where | Bytes/vector | At 100M |
|---|---|---|---|
| 1-bit codes + ids + 2 floats | RAM | 24 | 2.4 GB |
| 4-bit refine codes + 2 floats | RAM | 56 | 5.6 GB |
| Full float32 vectors | SSD | 384 | 38.4 GB |

The design starts from Exa's
[How we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db)
(clustering, 1-bit codes, lookup-table scan, rerank) and changes it in measured steps (below).

## What changed and why

**v0.1 → v0.2: the estimator, not the rotation** (Deep 10M, nprobe 64; `results/ablation.csv`).
v0.1's bits-only recall was stuck at ~0.33 at every nprobe, so the loss was in ranking, not
clustering. An ablation of the two candidate fixes:

| Variant | recall@10, rerank 100 | recall@10, rerank 500 |
|---|---|---|
| v0.1: least-squares scale, no rotation | 0.757 | 0.927 |
| + random rotation only | 0.756 | 0.928 |
| + unbiased (RaBitQ) scale only | **0.871** | **0.975** |
| + both | 0.869 | 0.975 |

Same codes, same memory, same speed; only the one float stored per vector changed.

**v0.3: 100M on a laptop, with the full vectors on SSD** (`results/results-100M-cold.csv`,
`results/repeats-100M.csv`).
- **Parallel reads made disk rerank viable.** Issuing all of a query's reads at once cut p99
  3–11×. Without it, rerank 500 missed 100 ms at every nprobe.
- **Rerank depth has to scale with cluster size.** Clusters are 2.5× bigger at 100M, so rerank 500
  left 3 points below the ceiling; rerank 1000 got 0.979 at 55 ms median p99.
- **A benchmarking bug, caught and fixed.** The first 100M run warmed up on the same queries it
  timed, and all settings shared one process, so later settings found vectors already cached
  (`results/results-100M.csv`, superseded). Every 100M number since is cold.
- **A puzzle, resolved in v0.4.** nprobe 128 kept beating nprobe 64 on latency. With per-stage
  timing and alternating run order (`results/phase1-100M-stages.csv`), 128 is slower as it should
  be (scan 11 ms vs 6 ms); the first run after a pause is slow (18 ms disk vs 9 ms), and 64
  always ran first.

**v0.4: a 4-bit tier, and loading done right** (`results/phase1*.csv`).
- **The refine tier cuts disk reads 20×.** At 100M, 50 reads (after 4-bit refine) beat 1,000
  reads (after bits alone): 0.989 vs 0.979 recall, 29 vs 55 ms median p99.
- **How a 5.6 GB array is loaded mattered 10×.** Reading the index and refine files into owned
  memory gave p99 251 ms; memory-mapping them gave 23 ms, same setting
  (`results/phase1b-load.csv`). The scan stage alone went from 49 ms to 9 ms. Linux recorded
  **zero** swap and zero major faults in both cases, which ruled out the first hypothesis
  (swapping). The likely cause is outside Linux: the copy briefly needs the data twice, filling
  the VM's 11 GB, and Windows (16 GB total) appears to page the VM's own memory. Not proven;
  a native-Linux run would settle it.
- **Build 2.4× faster with approximate assignment.** Comparing each vector only with the
  centroids in its 8 nearest groups (of 128) instead of all 16,384: build 53 → 22 minutes
  (assignment 2,210 → 570 s), 98.7% of vectors land in the same cluster, ceiling unchanged
  (0.9914 vs 0.9910 at nprobe 128). At 10M (4,096 clusters) it saves almost nothing.

**v0.5: a faster scan** (`results/phase2*.csv`). The scan was 70–80% of query time. Three changes,
measured separately on Deep 10M (nprobe 128 / refine 5,000 / rerank 50, `results/phase2b-10M.csv`):

| Code layout | Candidate collector | scan ms | p50 ms | p99 ms | QPS |
|---|---|---|---|---|---|
| byte tables (v0.4) | heap (v0.4) | 4.06 | 4.28 | 6.80 | 1,729 |
| byte tables | reservoir | 3.29 | 3.56 | 4.81 | 1,768 |
| FastScan | heap | 1.89 | 2.20 | 3.25 | 2,958 |
| FastScan | reservoir | **1.17** | **1.48** | **1.98** | **3,090** |

- **FastScan SIMD kernel** (`src/fastscan.rs`; André et al., VLDB 2015, as in FAISS). The 96 bits
  are split into 24 groups of 4. Each group's 16 possible partial scores are rounded to bytes, so
  one group's table fits in half a 256-bit register. Codes are stored in blocks of 32 vectors,
  so one `vpshufb` instruction looks up 32 table entries at once. Byte rounding is checked
  in the tests (error ≤ 12 steps); recall did not change at any setting.
- **Reservoir instead of a heap** (`src/topk.rs`). Keeping 5,000 candidates in a heap costs a
  log-time insert with an unpredictable branch for most vectors. A reservoir appends anything
  under the threshold and, when its buffer of 2k fills, keeps the best k with one
  `select_nth_unstable`. This only mattered once the kernel was fast: with FastScan it cut the
  scan stage 38%.
- **Refine stage: 8 independent sums and memory order** (`src/refine.rs`). One running sum made
  every add wait for the last; eight lanes let the compiler use SIMD. Candidates are sorted by
  position before refining, so reads into the 5.6 GB array go forwards, list by list.
- **Converting needs no rebuild.** `binvec convert` repacks an index's codes into FastScan
  blocks in 7 s at 100M; vector order is unchanged, so the refine file stays valid.
- **The cheaper scan moved the best setting.** nprobe 256 now fits in 10 ms p99, which takes the
  recall ceiling from 0.991 to 0.997. At nprobe 512 the ceiling is 0.999, but the scan
  (9.7 ms) dominates again.

## Reproduce

Needs Linux (WSL2 is fine) and Rust. Disk: ~5 GB for 10M, ~55 GB for 100M with all tiers.

```bash
scripts/download_deep.sh 10        # Deep1B: first 10M vectors, queries, ground truth
scripts/run_10m.sh                 # verify data, build index, benchmark
scripts/ablation_10m.sh            # v0.2: rotation and estimator on/off

scripts/download_deep.sh 100       # first 100M vectors (~38 GB; reuses the 10M file)
scripts/build_100m.sh              # verify + build (~22 min with --assign-probe 8, else ~53)
scripts/bench_100m.sh              # v0.3 cold-cache benchmark (asks for sudo once)
scripts/phase1.sh                  # v0.4: refine tier, stage timing, two-level build
scripts/phase1b.sh                 # v0.4: memory-mapped loading, grid, repeats
scripts/phase2.sh                  # v0.5: FastScan conversion, ablation, old vs new path
scripts/phase2b.sh                 # v0.5: repeats at nprobe 128 / 256, nprobe 512
```

Synthetic data, no download:

```bash
cargo build --release
B=target/release/binvec
$B gen --n 1000000 --out-dir data/syn1m
$B build --base data/syn1m/base.fbin --nlist 1024 --out data/syn1m/idx.bin   # FastScan layout by default
$B refine --index data/syn1m/idx.bin --base data/syn1m/base.fbin --out data/syn1m/refine.bin
$B bench --index data/syn1m/idx.bin --base data/syn1m/base.fbin \
         --queries data/syn1m/query.fbin --gt data/syn1m/gt.bin \
         --refine data/syn1m/refine.bin --refine-k 1000 --rerank 30
```

## Roadmap

- [x] k-means IVF, residual 1-bit codes, lookup-table scan, exact rerank
- [x] Benchmark harness: recall, p50/p99, QPS, per-stage time, page faults, CSV output
- [x] Unbiased (RaBitQ) estimator; ablation against rotation
- [x] Deep 100M with full vectors on SSD; parallel rerank reads
- [x] Cold-cache method; medians of repeated runs
- [x] 4-bit refine tier in RAM; memory-mapped index files
- [x] Two-level approximate centroid assignment (2.4× faster 100M build)
- [x] SIMD (FastScan) scan kernel, reservoir collector, vectorised refine
- [ ] Filtered search with inverted indexes
- [ ] Head-to-head with FAISS on the same machine
- [ ] Write-up

## Layout

| File | What it does |
|---|---|
| `src/ivf.rs` | The index: build, search (4 stages), save/load |
| `src/fastscan.rs` | FastScan block layout and AVX2 scan kernel (with a scalar fallback) |
| `src/refine.rs` | 4-bit refine tier |
| `src/kmeans.rs` | k-means; exact and two-level centroid assignment |
| `src/buf.rs` | Arrays that are either owned or windows into a memory-mapped file |
| `src/dist.rs` | Distance kernels |
| `src/io.rs` | Big-ANN file formats; memory-mapped full vectors, prefetch |
| `src/topk.rs` | Heap and reservoir collectors for the k best candidates |
| `src/bin/binvec.rs` | CLI: `gen`, `verify`, `build`, `convert`, `refine`, `bench`, `ceiling` |
| `results/` | Raw output of every benchmark run quoted above |
