//! Command-line entry point.
//!
//!   binvec gen     — make a small synthetic dataset (for testing without downloads)
//!   binvec verify  — brute-force a few queries and compare with the published ground truth
//!   binvec build   — build an index from an .fbin file
//!   binvec bench   — measure recall@10, latency and throughput for several settings

use binvec::dist::l2sq;
use binvec::io::{write_fbin, write_gt, Fbin, GroundTruth};
use binvec::ivf::{BuildParams, Estimator, IvfIndex, SearchParams, StageTimes};
use binvec::refine::Refine;
use binvec::rng::Rng;
use binvec::topk::{Hit, TopK};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "binvec", about = "Billion-scale vector search on one machine")]
struct Cli {
    /// Worker threads for parallel steps (default: all cores)
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate a synthetic clustered dataset with exact ground truth
    Gen {
        #[arg(long, default_value_t = 1_000_000)]
        n: usize,
        #[arg(long, default_value_t = 96)]
        d: usize,
        #[arg(long, default_value_t = 1000)]
        nq: usize,
        #[arg(long, default_value_t = 1000)]
        clusters: usize,
        #[arg(long, default_value_t = 0.6)]
        sigma: f32,
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Brute-force the first queries and check the published ground truth agrees
    Verify {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        gt: PathBuf,
        #[arg(long, default_value_t = 50)]
        nq: usize,
    },
    /// Build an index
    Build {
        #[arg(long)]
        base: PathBuf,
        /// Use only the first N vectors of the file
        #[arg(long)]
        limit: Option<usize>,
        /// Number of k-means lists (rule of thumb: about sqrt(N), e.g. 4096 for 10M)
        #[arg(long)]
        nlist: usize,
        /// Points used to train k-means
        #[arg(long, default_value_t = 500_000)]
        train_n: usize,
        #[arg(long, default_value_t = 10)]
        iters: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Skip the random rotation (v0.1 behaviour, for comparison)
        #[arg(long)]
        no_rotate: bool,
        /// How to scale the bit score: "unbiased" (RaBitQ, default) or "ls" (v0.1)
        #[arg(long, default_value = "unbiased")]
        estimator: String,
        /// 0 = exact centroid assignment. N > 0 = two-level approximate assignment that checks
        /// only the centroids in the N nearest groups (much faster at large nlist; try 8).
        #[arg(long, default_value_t = 0)]
        assign_probe: usize,
        /// Code layout: "fastscan" (SIMD, default) or "bytes" (v0.4 byte-LUT scan)
        #[arg(long, default_value = "fastscan")]
        layout: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Convert an index to the FastScan layout (no rebuild, no base vectors needed)
    Convert {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Build the 4-bit refine tier for an existing index (reads the base file once)
    Refine {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Recall ceiling of the clustering: the fraction of true top-k neighbours that sit in the
    /// probed lists. Equals the recall of scoring every probed vector exactly, but needs no disk
    /// reads, so it is instant even at 100M.
    Ceiling {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        gt: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value = "8,16,32,64,128,256")]
        nprobe: String,
    },
    /// Benchmark recall / latency / throughput
    Bench {
        #[arg(long)]
        index: PathBuf,
        /// Base vectors for exact reranking
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        gt: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Comma-separated nprobe values to try
        #[arg(long, default_value = "8,16,32,64")]
        nprobe: String,
        /// Comma-separated rerank sizes (0 = no rerank)
        #[arg(long, default_value = "0,100,300")]
        rerank: String,
        /// 4-bit refine tier file (from `binvec refine`); enables --refine-k
        #[arg(long)]
        refine: Option<PathBuf>,
        /// Comma-separated: candidates kept from the 1-bit scan for the refine tier (0 = off)
        #[arg(long, default_value = "0")]
        refine_k: String,
        /// Read index and refine arrays into owned memory instead of memory-mapping them
        /// (the v0.3 behaviour; doubles peak memory while loading)
        #[arg(long)]
        copy_load: bool,
        /// Collect scan candidates in a binary heap (the v0.4 behaviour) instead of the
        /// reservoir. For before/after comparisons.
        #[arg(long)]
        heap: bool,
        /// Queries timed one by one on a single thread for latency percentiles
        #[arg(long, default_value_t = 2000)]
        nq_latency: usize,
        /// Append results to this CSV
        #[arg(long, default_value = "results/results.csv")]
        csv: PathBuf,
        /// Free-text label stored with each CSV row (dataset, machine, notes)
        #[arg(long, default_value = "")]
        label: String,
        /// Prefetch rerank candidates from disk in parallel (matters when data exceeds RAM)
        #[arg(long)]
        prefetch: bool,
        /// Use only the first N queries for the throughput/recall pass (default: all)
        #[arg(long)]
        nq: Option<usize>,
        /// Untimed queries before timing. They are taken from the END of the query file, so
        /// they never pre-load the vectors of the queries being timed. Use 0 for cold-cache runs.
        #[arg(long, default_value_t = 200)]
        warmup: usize,
    },
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse().expect("bad number in list"))
        .collect()
}

/// Exact top-k for many queries in ONE pass over `base`.
///
/// The loop order matters when `base` is bigger than RAM: we split the base into blocks (in
/// parallel) and score every query against a block while it is in cache. Looping over queries
/// first would stream the whole file from disk once per query.
fn exact_topk_many(base: &[f32], d: usize, queries: &[f32], k: usize) -> Vec<Vec<Hit>> {
    const BLOCK: usize = 1 << 16; // vectors per block (24 MB at d = 96)
    let nq = queries.len() / d;
    base.par_chunks(BLOCK * d)
        .enumerate()
        .map(|(bi, blk)| {
            let off = (bi * BLOCK) as u32;
            queries
                .chunks_exact(d)
                .map(|q| {
                    let mut t = TopK::new(k);
                    for (i, x) in blk.chunks_exact(d).enumerate() {
                        let dist = l2sq(q, x);
                        if dist < t.threshold() {
                            t.push(dist, off + i as u32);
                        }
                    }
                    t.into_sorted()
                })
                .collect::<Vec<_>>()
        })
        .reduce(
            || vec![Vec::new(); nq],
            |a, b| {
                a.into_iter()
                    .zip(b)
                    .map(|(x, y)| {
                        let mut t = TopK::new(k);
                        for h in x.into_iter().chain(y) {
                            t.push(h.dist, h.id);
                        }
                        t.into_sorted()
                    })
                    .collect()
            },
        )
}

fn recall_at(found: &[u32], truth: &[u32], k: usize) -> f64 {
    let truth = &truth[..k.min(truth.len())];
    found.iter().take(k).filter(|id| truth.contains(id)).count() as f64 / truth.len() as f64
}

/// This process's major page faults so far (each one waited for a disk read).
/// Field 12 of /proc/self/stat; 0 where unavailable.
fn major_faults() -> u64 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            let after = s.rsplit_once(')')?.1;
            after.split_whitespace().nth(9)?.parse().ok()
        })
        .unwrap_or(0)
}

/// System-wide pages swapped in and out so far (/proc/vmstat); (0, 0) where unavailable.
fn swap_pages() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/vmstat").unwrap_or_default();
    let get = |key: &str| {
        s.lines()
            .find_map(|l| l.strip_prefix(key)?.trim().parse().ok())
            .unwrap_or(0)
    };
    (get("pswpin "), get("pswpout "))
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    let i = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[i]
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .unwrap();
    }

    match cli.cmd {
        Cmd::Gen {
            n,
            d,
            nq,
            clusters,
            sigma,
            out_dir,
        } => {
            std::fs::create_dir_all(&out_dir)?;
            let mut rng = Rng::new(7);
            let centers: Vec<f32> = (0..clusters * d).map(|_| rng.normal()).collect();
            let sample = |rng: &mut Rng| -> Vec<f32> {
                let c = rng.below(clusters);
                let mut v: Vec<f32> = (0..d)
                    .map(|j| centers[c * d + j] + sigma * rng.normal())
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            };
            let base: Vec<f32> = (0..n).flat_map(|_| sample(&mut rng)).collect();
            let queries: Vec<f32> = (0..nq).flat_map(|_| sample(&mut rng)).collect();
            let k = 100;
            eprintln!("[gen] computing exact ground truth for {nq} queries over {n} vectors...");
            let rows = exact_topk_many(&base, d, &queries, k);
            let ids: Vec<u32> = rows.iter().flat_map(|r| r.iter().map(|h| h.id)).collect();
            let dists: Vec<f32> = rows.iter().flat_map(|r| r.iter().map(|h| h.dist)).collect();
            write_fbin(&out_dir.join("base.fbin"), n, d, &base)?;
            write_fbin(&out_dir.join("query.fbin"), nq, d, &queries)?;
            write_gt(&out_dir.join("gt.bin"), nq, k, &ids, &dists)?;
            eprintln!(
                "[gen] wrote base.fbin, query.fbin, gt.bin to {}",
                out_dir.display()
            );
        }

        Cmd::Verify {
            base,
            limit,
            queries,
            gt,
            nq,
        } => {
            let base = Fbin::open(&base, limit)?;
            base.advise_sequential();
            let qs = Fbin::open(&queries, None)?;
            let gt = GroundTruth::read(&gt)?;
            let nq = nq.min(qs.n).min(gt.nq);
            eprintln!(
                "[verify] brute force {nq} queries over {} vectors (d={})",
                base.n, base.d
            );
            let t = Instant::now();
            let rows = exact_topk_many(base.data(), base.d, &qs.data()[..nq * qs.d], 10);
            let recalls: Vec<f64> = rows
                .iter()
                .enumerate()
                .map(|(i, hits)| {
                    let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
                    recall_at(&ids, gt.row(i), 10)
                })
                .collect();
            let r = recalls.iter().sum::<f64>() / nq as f64;
            println!(
                "brute-force recall@10 vs ground truth: {r:.4}  ({:.1}s)",
                t.elapsed().as_secs_f64()
            );
            println!("expected ~1.000 — anything much lower means the data or ground-truth file is wrong");
        }

        Cmd::Build {
            base,
            limit,
            nlist,
            train_n,
            iters,
            seed,
            no_rotate,
            estimator,
            assign_probe,
            layout,
            out,
        } => {
            let base = Fbin::open(&base, limit)?;
            let t = Instant::now();
            let mut idx = IvfIndex::build(
                &base,
                &BuildParams {
                    nlist,
                    train_n,
                    iters,
                    seed,
                    rotate: !no_rotate,
                    estimator: match estimator.as_str() {
                        "ls" => Estimator::LeastSquares,
                        "unbiased" => Estimator::Unbiased,
                        other => panic!("unknown --estimator {other} (use ls or unbiased)"),
                    },
                    assign_probe,
                },
            );
            match layout.as_str() {
                "fastscan" => idx.to_fastscan(),
                "bytes" => {}
                other => panic!("unknown --layout {other} (use fastscan or bytes)"),
            }
            eprintln!(
                "[build] total {:.1}s, saving to {}",
                t.elapsed().as_secs_f64(),
                out.display()
            );
            idx.save(&out)?;
        }

        Cmd::Convert { index, out } => {
            let t = Instant::now();
            let mut idx = IvfIndex::load(&index)?;
            idx.to_fastscan();
            idx.save(&out)?;
            eprintln!(
                "[convert] {} -> {} ({}) in {:.1}s",
                index.display(),
                out.display(),
                idx.variant(),
                t.elapsed().as_secs_f64()
            );
        }

        Cmd::Refine {
            index,
            base,
            limit,
            out,
        } => {
            let idx = IvfIndex::load(&index)?;
            let base = Fbin::open(&base, limit)?;
            let rf = Refine::build(&idx, &base);
            rf.save(&out)?;
            eprintln!("[refine] saved to {}", out.display());
        }

        Cmd::Ceiling {
            index,
            queries,
            gt,
            k,
            nprobe,
        } => {
            let idx = IvfIndex::load(&index)?;
            idx.print_balance();
            let list_of = idx.list_of_ids();
            let qs = Fbin::open(&queries, None)?;
            let gt = GroundTruth::read(&gt)?;
            let nq = qs.n.min(gt.nq);
            println!("{:>7} {:>13}", "nprobe", "ceiling@10");
            for &np in &parse_list(&nprobe) {
                let total: f64 = (0..nq)
                    .into_par_iter()
                    .map(|i| {
                        let probed = idx.probed_lists(qs.vector(i), np);
                        let truth = &gt.row(i)[..k];
                        truth
                            .iter()
                            .filter(|&&id| probed.contains(&list_of[id as usize]))
                            .count() as f64
                            / k as f64
                    })
                    .sum();
                println!("{np:>7} {:>13.4}", total / nq as f64);
            }
        }

        Cmd::Bench {
            index,
            base,
            limit,
            queries,
            gt,
            k,
            nprobe,
            rerank,
            refine,
            refine_k,
            copy_load,
            heap,
            nq_latency,
            csv,
            label,
            prefetch,
            nq: nq_limit,
            warmup,
        } => {
            let t = Instant::now();
            let idx = IvfIndex::load_with(&index, copy_load)?;
            eprintln!("[bench] index loaded in {:.1}s", t.elapsed().as_secs_f64());
            idx.print_balance();
            let rf = match &refine {
                Some(p) => {
                    let t = Instant::now();
                    let rf = Refine::load_with(p, copy_load)?;
                    assert_eq!(rf.n, idx.n, "refine file does not match index");
                    eprintln!(
                        "[bench] refine tier loaded in {:.1}s ({:.2} GB)",
                        t.elapsed().as_secs_f64(),
                        rf.ram_bytes() as f64 / 1e9
                    );
                    Some(rf)
                }
                None => None,
            };
            let base = Fbin::open(&base, limit)?;
            assert_eq!(
                base.n, idx.n,
                "base vector count does not match index (check --limit)"
            );
            let qs = Fbin::open(&queries, None)?;
            let gt = GroundTruth::read(&gt)?;
            let nq = qs.n.min(gt.nq).min(nq_limit.unwrap_or(usize::MAX));
            let nl = nq_latency.min(nq);
            let threads = rayon::current_num_threads();
            let ram_gb = (idx.ram_bytes() + rf.as_ref().map_or(0, |r| r.ram_bytes())) as f64 / 1e9;

            const HEADER: &str = "label,n,nlist,nprobe,refine_k,rerank,recall10,p50_ms,p99_ms,mean_ms,qps,threads,ram_gb,t_centroids_ms,t_scan_ms,t_refine_ms,t_rerank_ms,major_faults,swap_in_pages,swap_out_pages";
            if let Some(dir) = csv.parent() {
                std::fs::create_dir_all(dir)?;
            }
            if csv.exists() {
                let first = std::fs::read_to_string(&csv)?
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string();
                if first != HEADER {
                    return Err(std::io::Error::other(format!(
                        "{} has an older column layout; pass a new --csv file name",
                        csv.display()
                    )));
                }
            }
            let new_file = !csv.exists();
            let mut out = OpenOptions::new().create(true).append(true).open(&csv)?;
            if new_file {
                writeln!(out, "{HEADER}")?;
            }

            println!(
                "{:>6} {:>6} {:>6} {:>9} {:>8} {:>8} {:>8} {:>7}  | {:>6} {:>6} {:>6} {:>6} | {:>8} {:>7} {:>7}",
                "nprobe",
                "refine",
                "rerank",
                "recall@10",
                "p50 ms",
                "p99 ms",
                "mean ms",
                "QPS",
                "cent",
                "scan",
                "refine",
                "rerank",
                "majflt",
                "swapin",
                "swapout"
            );
            let refine_list = if rf.is_some() {
                parse_list(&refine_k)
            } else {
                vec![0]
            };
            for &np in &parse_list(&nprobe) {
                for &rk in &refine_list {
                    for &rr in &parse_list(&rerank) {
                        let p = SearchParams {
                            k,
                            nprobe: np,
                            refine_k: rk,
                            rerank: rr,
                            prefetch,
                            heap,
                        };
                        // Warm-up uses queries the timed pass never sees (from the end of the file).
                        for i in (0..qs.n).rev().take(warmup.min(qs.n.saturating_sub(nl))) {
                            idx.search(qs.vector(i), &p, rf.as_ref(), Some(&base), None);
                        }
                        // Latency: one query at a time, one thread, with a per-stage breakdown.
                        let mut lat = Vec::with_capacity(nl);
                        let mut st = StageTimes::default();
                        let (f0, (si0, so0)) = (major_faults(), swap_pages());
                        for i in 0..nl {
                            let t = Instant::now();
                            let hits = idx.search(
                                qs.vector(i),
                                &p,
                                rf.as_ref(),
                                Some(&base),
                                Some(&mut st),
                            );
                            lat.push(t.elapsed().as_secs_f64() * 1e3);
                            std::hint::black_box(hits);
                        }
                        let (f1, (si1, so1)) = (major_faults(), swap_pages());
                        let (flt, sin, sout) = (f1 - f0, si1 - si0, so1 - so0);
                        let mean = lat.iter().sum::<f64>() / nl as f64;
                        lat.sort_by(|a, b| a.total_cmp(b));
                        let ms = |x: f64| x * 1e3 / nl as f64;
                        let (tc, ts, tr, tk) =
                            (ms(st.centroids), ms(st.scan), ms(st.refine), ms(st.rerank));
                        // Throughput and recall: all queries, all threads.
                        let t = Instant::now();
                        let total: f64 = (0..nq)
                            .into_par_iter()
                            .map(|i| {
                                let hits =
                                    idx.search(qs.vector(i), &p, rf.as_ref(), Some(&base), None);
                                let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
                                recall_at(&ids, gt.row(i), k)
                            })
                            .sum();
                        let qps = nq as f64 / t.elapsed().as_secs_f64();
                        let recall = total / nq as f64;
                        let (p50, p99) = (pct(&lat, 0.50), pct(&lat, 0.99));
                        println!(
                            "{np:>6} {rk:>6} {rr:>6} {recall:>9.4} {p50:>8.3} {p99:>8.3} {mean:>8.3} {qps:>7.0}  | {tc:>6.2} {ts:>6.2} {tr:>6.2} {tk:>6.2} | {flt:>8} {sin:>7} {sout:>7}"
                        );
                        writeln!(
                            out,
                            "\"{label} {}{}\",{},{},{np},{rk},{rr},{recall:.4},{p50:.3},{p99:.3},{mean:.3},{qps:.0},{threads},{ram_gb:.3},{tc:.3},{ts:.3},{tr:.3},{tk:.3},{flt},{sin},{sout}",
                            idx.variant(),
                            if prefetch { " prefetch" } else { "" },
                            idx.n,
                            idx.nlist,
                        )?;
                    }
                }
            }
            eprintln!("[bench] results appended to {}", csv.display());
        }
    }
    Ok(())
}
