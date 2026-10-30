# binvec: design notes

How a vector index for 100M vectors ended up fitting on a 16 GB laptop at 0.995 recall@10 and
10 ms p99, what each step was supposed to fix, what the measurements said, and what I'd do next.
All numbers are from CSVs in `results/`; the [README](../README.md) has the full tables.

## 1. The problem and the constraint

Given a query vector, return its 10 nearest neighbours among N vectors, quickly and nearly
always correctly: measured as recall@10 against exact ground truth, and as p50/p99 latency for one
query on one thread.

The constraint shaped everything. The machine is a laptop: 8 cores, 16 GB RAM (11–12 GB of it
given to Linux under WSL2), one NVMe SSD. Deep 100M (from the Big-ANN benchmark, 96 dimensions)
is 38.4 GB of float32 vectors, so the full vectors cannot live in memory. The question becomes:
**how few bytes per vector can stay in RAM, and how few full vectors must be read from SSD per
query, without giving up recall?**

The starting point was Exa's post
[How we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db):
cluster the vectors, store compressed codes, scan the codes of a few clusters with lookup tables,
rerank the best candidates with full precision.

## 2. The design

```
query ─► 1. nearest clusters ─► 2. 1-bit scan ─► 3. 4-bit refine ─► 4. exact rerank ─► top 10
          (16,384 centroids)     (~1.6M codes)     (5,000 codes)      (50 vectors, SSD)
             RAM                   RAM 24 B/vec      RAM 56 B/vec       SSD 384 B/vec
```

Each stage throws away fewer candidates than the last and spends more per candidate.

| Tier | Where | Bytes/vector | At 100M |
|---|---|---|---|
| Centroids, 1-bit codes, ids, 2 floats | RAM | 24 | 2.4 GB |
| 4-bit codes + 2 floats | RAM | 56 | 5.6 GB |
| Full float32 vectors | SSD (memory-mapped) | 384 | 38.4 GB |

Where the time goes at the headline setting (Deep 100M, cold, nprobe 256 / refine 5,000 /
rerank 50; medians of 3 runs, `results/phase2b-repeats.csv`):

| Stage | ms |
|---|---|
| 1. nearest clusters | ~0.5 |
| 2. 1-bit scan of ~1.6M codes | 4.6 |
| 3. 4-bit refine of 5,000 | 0.7 |
| 4. 50 reads from SSD + exact distances | 1.2 |
| **p50 / p99** | **7.0 / 9.9** |

**1-bit codes.** Each vector is stored as its offset from its cluster centre, randomly rotated,
reduced to its signs (96 bits = 12 bytes), plus the offset's norm and a scale. Distances are
estimated from the signs. The scale uses the unbiased estimator from RaBitQ (Gao & Long, SIGMOD
2024).

**FastScan.** The 96 bits are split into 24 groups of 4. Each group's 16 possible partial scores
are rounded to bytes and held in a SIMD register; codes are stored in blocks of 32 vectors, so one
AVX2 `vpshufb` scores a group for 32 vectors. The tables never touch memory.

**4-bit refine.** The best few thousand by 1-bit score are re-scored from a 4-bit copy of the
same offset (per-vector min/max), far more accurate than the signs. This is what lets the SSD
stage read 50 vectors instead of 1,000.

**Rerank from SSD.** The base file is memory-mapped. All 50 reads for a query are issued at
once with `madvise(WILLNEED)`, so the SSD serves them in parallel, then scored exactly.

## 3. How it got here: each decision and its measurement

Each version changed one thing and was measured against the last. The pattern repeated:
measure, find the dominant cost, fix it, measure again, find the next one.

**v0.1 → v0.2: the estimator, not the rotation.** v0.1 (signs + least-squares scale) had
bits-only recall stuck near 0.33 at every nprobe. That ruled out clustering (more clusters
didn't help) and pointed at ranking. Two candidate fixes, ablated on Deep 10M
(`results/ablation.csv`): a random rotation alone changed nothing (0.757 → 0.756 at rerank 100);
the unbiased scale alone took it to 0.871. One float per vector, same memory and speed.

**v0.3: 100M with the vectors on SSD.** Reading rerank candidates one after another made disk
the bottleneck; issuing all of a query's reads at once cut p99 3–11×. Clusters at 100M are 2.5×
larger than at 10M, so the rerank depth had to grow with them: 1,000 reads gave 0.979 recall at
55 ms median p99. Correct, but 1,000 random SSD reads per query is a lot.

**v0.4: a 4-bit tier, and loading done right.** The 4-bit refine tier cut disk reads 20×:
50 reads after refine beat 1,000 after bits alone (0.989 vs 0.979 recall, 29 vs 55 ms p99).
Then a surprise: reading the 8 GB of index files into owned memory gave p99 251 ms; memory-mapping
the same files gave 23 ms. I expected swapping and added swap and page-fault counters to prove
it: both read **zero**. Hypothesis refuted. The likely cause sits outside Linux (the copy briefly
holds the data twice, and Windows appears to page the VM's own memory), but I couldn't prove it
from inside the VM, so the README says so. Mapping is right either way: the data lives in RAM
once. Also in v0.4: assigning vectors to clusters by searching only the 8 nearest of 128 centroid
groups cut the 100M build from 53 to 22 minutes, with 98.7% of vectors landing in the same
cluster and no change in recall ceiling.

**v0.5: making the scan cheap.** Per-stage timing showed the 1-bit scan at 70–80% of query time.
FastScan cut it 3–4×. That exposed the next cost: the binary heap holding 5,000 candidates.
A reservoir (append below a threshold; when a 2k buffer fills, `select_nth_unstable` back to k)
cut the scan another 38% with FastScan, but only 19% on the old kernel: fixing the biggest cost
had exposed the next one. The refine stage had one float accumulator, so every add waited for
the last; eight accumulators let the compiler vectorise it (3–4× on a 1M test). Net, same settings and recall:
p99 21.4 → 9.1 ms at 100M. The cheaper scan also moved the best setting: nprobe 256 now fits in
10 ms, raising the recall ceiling from 0.991 to 0.997.

**v0.6: filtered search.** "Nearest neighbours among vectors with label L" is where naive
designs break: searching first and filtering after keeps ~5 of 50 candidates for a 10% label
(recall 0.49) and none for rare ones (0.001 at 0.01%). Two strategies, each good at one end:

- *In-scan filter*: skip non-matching vectors inside the scan; keep probing clusters (nearest
  first, skipping clusters with no match) until a budget of matching vectors is covered. Cost
  grows as the label gets rarer: 97 ms at 0.01%.
- *Inverted index*: label → sorted list of vectors; score every match from its 4-bit code, rerank
  50. Cost grows with the number of matches: 94 ms at 1%.

They cross between 1% and 0.1%, which is the case for a planner. Per-cluster label counts price
both before scoring anything: vectors the scan would walk vs. matches × c. I guessed c = 10;
stage timings said one 4-bit score costs ~30 FastScan scores (95 ns vs 3 ns), and with 30 the
planner picks the faster strategy at every selectivity tested. Result at 100M: recall ≥ 0.99
for labels on 50% down to 0.01% of vectors, p99 ≤ 21 ms, including labels correlated with
position (each label a slab of space, so filtered neighbours are far from the query).

**v0.7: against FAISS.** Same machine, data, queries and measurement rules, FAISS 1.15.1 swept
over nprobe and rerank depth. On Deep 10M, per-query latency is level with FAISS's fastest IVF
index (IVF-PQ 4-bit FastScan + full-vector rerank) from 0.985 to 0.999 recall, using 5.2× less
RAM (0.80 vs 4.16 GB). FAISS has 1.25–1.8× higher throughput through its batch API. The RAM gap is
the point: FAISS's 0.99-recall configurations keep every full vector in memory, ~41 GB at 100M.

## 4. Measuring honestly

Most of the work was making the numbers trustworthy.

- **Ground truth** from the Big-ANN benchmark, re-verified by brute force; filtered ground truth
  computed exactly (one pass over the base file, each vector compared only with the queries
  filtered on its label) and cross-checked against numpy.
- **Cold cache** at 100M: page cache dropped and a fresh process for every run, so the full
  vectors really come from SSD. Warm-up queries come from the end of the query file, never the
  timed ones. The first 100M run broke both rules; its numbers are kept, marked superseded.
- **Repeats, alternated.** A puzzle (nprobe 128 faster than 64) turned out to be run order: the
  first run after a pause is slow, and 64 always ran first. Headline numbers are medians of 3
  alternating runs, with the p99 range shown.
- **A ceiling metric**: the share of true neighbours inside the probed clusters. It separates
  "the clustering missed it" from "the ranking missed it" and costs no disk reads, so it's
  instant even at 100M. It pointed at the estimator in v0.2.
- **Counters that can prove me wrong**: major page faults and swap in every benchmark row. They
  refuted my swap hypothesis in v0.4 and again cleared Linux of an unexplained slow run in v0.6.
- **Stale binaries.** One benchmark ran a binary that `cargo test` hadn't rebuilt. Since then,
  `cargo build --release` runs before every benchmark script.

## 5. Limits

- One machine, under WSL2. Two slow runs (v0.4 loading, one v0.6 run) were outside what Linux
  could see; a native-Linux run would settle them.
- One dataset: Deep, 96 dimensions. Production text embeddings are wider; every tier scales with
  d, and the 1-bit estimator's error shrinks as d grows, but I haven't measured it.
- Filter labels are synthetic, with planted frequencies, to control selectivity exactly. Real
  filters (domains, dates) have skewed distributions and are often correlated with content; the
  correlated variant is a stand-in for that.
- The index is static: no inserts, deletes or updates.
- Throughput is one query per thread; FAISS's batch API is faster there.

## 6. What I'd do next

**Batch queries per cluster.** Several queries probing the same cluster can share one pass over
its codes (the FastScan kernel already processes 32 vectors per step; adding queries reuses the
loaded block). That is the most likely source of FAISS's throughput lead.

**Date-range filters.** Sort each cluster's vectors by date at build time: a date range then
becomes one contiguous run per cluster, scanned with the same kernel, with no per-vector test.

**Updates.** Append new vectors to their cluster's tail (codes are per-vector, so no retraining),
mark deletes with a bitmap checked in the scan, and rebuild clusters in the background when they
drift past a size or quality threshold.

**A billion vectors.** At 1B × 96 dimensions, the tiers would be 24 GB (1-bit), 56 GB (4-bit)
and 384 GB (full vectors). The 1-bit tier fits one large machine; the 4-bit tier could move to
SSD as well (5,000 reads of 56 bytes per query, the same parallel-read trick as stage 4), or the
index could be sharded by cluster across machines, each query going to the shards that own its
nearest clusters. Each option is a measurement away from a decision; this project's method
would apply unchanged.
