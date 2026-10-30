//! The index: IVF (k-means lists) + residual binary codes + exact rerank.
//!
//! ## What is stored per vector (20 bytes for d = 96, vs 384 bytes for the raw floats)
//! - `code`  : d bits, the signs of e' = P·(x − centroid). P is a random rotation (or identity).
//! - `scale` : one float that turns the bit score into an estimate of r·e (see "estimators").
//! - `rnorm` : ||e||², stored exactly.
//!
//! ## Estimated distance (no raw vector needed)
//! With r = q − centroid:  ||q − x||² = ||r − e||² = ||r||² + ||e||² − 2·(r·e).
//! A rotation keeps dot products, so r·e = r'·e' with r' = P r, e' = P e.
//! We approximate r'·e' ≈ scale · (r'·s), where s ∈ {−1,+1}^d are the stored signs.
//!
//! ## Estimators (choose with `--estimator`)
//! - `ls`       : scale = ||e'||₁ / d. The least-squares fit e' ≈ scale·s. Simple, but biased:
//!   it shrinks the cross term, more for some vectors than others, which distorts
//!   the ranking. This was v0.1.
//! - `unbiased` : scale = ||e||² / ||e'||₁. The RaBitQ estimator (Gao & Long, SIGMOD 2024).
//!   Combined with a random rotation its error is unbiased and concentrated.
//!
//! ## Why rotate?
//! Real embeddings put very different amounts of variance in different dimensions. One sign
//! bit per dimension wastes bits on low-variance dimensions and loses detail on high-variance
//! ones. A random orthogonal rotation spreads the variance evenly across dimensions, so each
//! bit carries about the same information. It costs one d×d matrix-vector product per query.
//!
//! ## Computing r'·s fast: a lookup table per byte
//! One code byte covers 8 dimensions and has only 256 possible values. For each byte position
//! we precompute all 256 partial sums of ±r' over those 8 dims, once per (query, list). Scoring
//! a vector is then d/8 lookups and adds (12 for d = 96), no multiplications.

use crate::buf::{map_file, Buf};
use crate::dist::{dot, l2sq};
use crate::fastscan;
use crate::filter::Labels;
use crate::io::Fbin;
use crate::kmeans;
use crate::refine::Refine;
use crate::rng::Rng;
use crate::topk::{Collector, Hit, TopK};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::path::Path;
use std::time::Instant;

const MAGIC: &[u8; 8] = b"BINVEC02"; // codes as d/8 bytes per vector (byte-LUT scan)
const MAGIC_FS: &[u8; 8] = b"BINVEC03"; // codes in FastScan blocks of 32

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Estimator {
    LeastSquares,
    Unbiased,
}

impl Estimator {
    pub fn name(self) -> &'static str {
        match self {
            Estimator::LeastSquares => "ls",
            Estimator::Unbiased => "unbiased",
        }
    }
}

pub struct IvfIndex {
    pub d: usize,
    pub nlist: usize,
    pub n: usize,
    pub rotate: bool,
    pub estimator: Estimator,
    /// d×d row-major rotation (identity when `rotate` is false).
    rot: Vec<f32>,
    /// Centroids in the rotated space (P·c).
    pub centroids: Vec<f32>,
    cnorms: Vec<f32>,
    /// Vectors of list c are at positions offsets[c]..offsets[c+1].
    pub offsets: Vec<u64>,
    pub ids: Buf<u32>,
    pub codes: Buf<u8>,
    pub scale: Buf<f32>,
    pub rnorm: Buf<f32>,
    /// FastScan layout (see `fastscan.rs`): list c's blocks are
    /// fs_blocks[fs_offsets[c]·bb .. fs_offsets[c+1]·bb]. Empty when `codes` is used instead.
    pub fs_offsets: Vec<u64>,
    pub fs_blocks: Buf<u8>,
}

pub struct BuildParams {
    pub nlist: usize,
    pub train_n: usize,
    pub iters: usize,
    pub seed: u64,
    pub rotate: bool,
    pub estimator: Estimator,
    /// 0 = exact centroid assignment; > 0 = two-level approximate assignment probing this many
    /// groups of centroids (see `kmeans::TwoLevel`). Much faster for large nlist.
    pub assign_probe: usize,
}

/// Per-query search settings.
#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    pub k: usize,
    pub nprobe: usize,
    /// Candidates kept from the 1-bit scan for the 4-bit refine tier (0 = no refine tier).
    pub refine_k: usize,
    /// Candidates rescored exactly from the full vectors on disk (0 = no rerank).
    pub rerank: usize,
    /// Start all rerank reads before computing any.
    pub prefetch: bool,
    /// Collect scan candidates in a binary heap (v0.4) instead of the reservoir. For benchmarks.
    pub heap: bool,
}

/// Time spent in each stage of one search, in seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimes {
    pub centroids: f64,
    pub scan: f64,
    pub refine: f64,
    pub rerank: f64,
}

impl StageTimes {
    pub fn add(&mut self, o: &StageTimes) {
        self.centroids += o.centroids;
        self.scan += o.scan;
        self.refine += o.refine;
        self.rerank += o.rerank;
    }
}

/// How a filtered query is answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    /// Pick per query from the label's statistics (see `Filter`).
    Auto,
    /// In-scan filtering: skip non-matching vectors inside the 1-bit scan, and keep probing
    /// lists until enough matching vectors have been covered.
    Scan,
    /// Inverted index: score exactly the vectors carrying the label, from their 4-bit codes.
    Postings,
    /// Baseline: search unfiltered, then drop non-matching results.
    Post,
}

impl FilterMode {
    pub fn name(self) -> &'static str {
        match self {
            FilterMode::Auto => "auto",
            FilterMode::Scan => "scan",
            FilterMode::Postings => "postings",
            FilterMode::Post => "post",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "auto" => FilterMode::Auto,
            "scan" => FilterMode::Scan,
            "postings" => FilterMode::Postings,
            "post" => FilterMode::Post,
            _ => return None,
        })
    }
}

/// A label filter for `search_filtered`.
///
/// The planner (`Auto`) compares two costs, both in "vectors touched":
/// - in-scan: the total size of the lists the scan would walk (nearest first, lists without the
///   label skipped) until `budget` matching vectors are covered;
/// - inverted index: the number of vectors with the label × `postings_cost` (a 4-bit distance
///   costs this many 1-bit FastScan scores).
///
/// Neither needs the query to be scored first: the per-list label counts answer both.
pub struct Filter<'a> {
    pub labels: &'a Labels,
    pub label: u16,
    pub mode: FilterMode,
    /// Matching vectors the in-scan strategy should cover before it stops probing.
    pub budget: usize,
    pub postings_cost: f64,
}

/// What a filtered search did.
#[derive(Clone, Debug, Default)]
pub struct FilterStats {
    pub used: Option<FilterMode>,
    /// Lists scanned (0 for the inverted index).
    pub lists: usize,
    /// Vectors scored in the first stage (1-bit for scans, 4-bit for postings).
    pub scanned: usize,
    /// The lists scanned, for measuring the filtered ceiling.
    pub probed: Vec<u32>,
}

/// Raw pointer we can share across threads; each thread writes disjoint slots.
#[derive(Clone, Copy)]
pub(crate) struct SyncPtr<T>(pub(crate) *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

/// Random orthogonal d×d matrix: Gaussian entries, then Gram–Schmidt on the rows (in f64).
pub fn random_rotation(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut m: Vec<f64> = (0..d * d).map(|_| rng.normal() as f64).collect();
    for i in 0..d {
        for j in 0..i {
            let p: f64 = (0..d).map(|t| m[i * d + t] * m[j * d + t]).sum();
            for t in 0..d {
                m[i * d + t] -= p * m[j * d + t];
            }
        }
        let norm: f64 = (0..d).map(|t| m[i * d + t].powi(2)).sum::<f64>().sqrt();
        for t in 0..d {
            m[i * d + t] /= norm;
        }
    }
    m.into_iter().map(|x| x as f32).collect()
}

pub fn identity(d: usize) -> Vec<f32> {
    let mut m = vec![0f32; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

/// out = P · x
#[inline]
pub fn matvec(p: &[f32], x: &[f32], out: &mut [f32]) {
    let d = x.len();
    for (i, o) in out.iter_mut().enumerate() {
        *o = dot(&p[i * d..(i + 1) * d], x);
    }
}

impl IvfIndex {
    #[inline]
    pub fn code_bytes(&self) -> usize {
        self.d / 8
    }

    pub fn variant(&self) -> String {
        format!(
            "{}+{}{}",
            if self.rotate { "rot" } else { "norot" },
            self.estimator.name(),
            if self.is_fastscan() { "+fastscan" } else { "" }
        )
    }

    pub fn is_fastscan(&self) -> bool {
        !self.fs_offsets.is_empty()
    }

    /// Re-pack the 1-bit codes into the FastScan block layout (and drop the old layout).
    pub fn to_fastscan(&mut self) {
        if self.is_fastscan() {
            return;
        }
        let cb = self.code_bytes();
        let per_list: Vec<Vec<u8>> = (0..self.nlist)
            .into_par_iter()
            .map(|c| {
                let (lo, hi) = (self.offsets[c] as usize, self.offsets[c + 1] as usize);
                let mut out = Vec::new();
                fastscan::pack(&self.codes[lo * cb..hi * cb], cb, &mut out);
                out
            })
            .collect();
        let bb = fastscan::block_bytes(2 * cb);
        let mut fs_offsets = vec![0u64; self.nlist + 1];
        let mut blocks = Vec::with_capacity(per_list.iter().map(|v| v.len()).sum());
        for (c, v) in per_list.into_iter().enumerate() {
            fs_offsets[c + 1] = fs_offsets[c] + (v.len() / bb) as u64;
            blocks.extend_from_slice(&v);
        }
        self.fs_offsets = fs_offsets;
        self.fs_blocks = blocks.into();
        self.codes = Vec::new().into();
    }

    pub fn build(base: &Fbin, p: &BuildParams) -> Self {
        let (n, d) = (base.n, base.d);
        assert!(d % 8 == 0, "dimension must be a multiple of 8 (got {d})");
        let cb = d / 8;
        let data = base.data();
        base.advise_sequential(); // full scans below; see Fbin::advise_sequential

        // 1. Train k-means on an evenly spaced sample (in the original space).
        let t = Instant::now();
        let train_n = p.train_n.min(n).max(p.nlist);
        let step = n / train_n;
        let mut sample = Vec::with_capacity(train_n * d);
        for i in 0..train_n {
            sample.extend_from_slice(base.vector(i * step));
        }
        eprintln!(
            "[build] training k-means: {train_n} points -> {} lists",
            p.nlist
        );
        let centroids = kmeans::train(&sample, d, p.nlist, p.iters, p.seed, p.assign_probe);
        let cnorms = kmeans::norms(&centroids, d);
        eprintln!("[build] k-means done in {:.1}s", t.elapsed().as_secs_f64());

        // 2. Assign every vector to its nearest centroid.
        let t = Instant::now();
        let assign = if p.assign_probe > 0 {
            let two = kmeans::TwoLevel::new(&centroids, d, p.assign_probe, p.seed);
            let check_n = (train_n.min(20_000)) * d;
            eprintln!(
                "[build] two-level assignment (probe {}): agrees with exact on {:.2}% of a sample",
                p.assign_probe,
                100.0 * two.agreement(&sample[..check_n], &centroids)
            );
            two.assign_all(data)
        } else {
            kmeans::assign_all(data, d, &centroids, &cnorms)
        };
        drop(sample);
        eprintln!(
            "[build] assigned {n} vectors in {:.1}s",
            t.elapsed().as_secs_f64()
        );

        // 3. Counting sort: where does each vector go in list order?
        let mut offsets = vec![0u64; p.nlist + 1];
        for &c in &assign {
            offsets[c as usize + 1] += 1;
        }
        for c in 0..p.nlist {
            offsets[c + 1] += offsets[c];
        }
        let mut cursor = offsets.clone();
        // u32 is enough for 4 billion vectors and saves 400 MB at 100M compared with u64.
        let pos: Vec<u32> = assign
            .iter()
            .map(|&c| {
                let p = cursor[c as usize];
                cursor[c as usize] += 1;
                p as u32
            })
            .collect();

        // 4. Rotation, and centroids moved into the rotated space.
        let rot = if p.rotate {
            random_rotation(d, p.seed ^ 0xA5A5)
        } else {
            identity(d)
        };
        let mut rcent = vec![0f32; p.nlist * d];
        for c in 0..p.nlist {
            matvec(
                &rot,
                &centroids[c * d..(c + 1) * d],
                &mut rcent[c * d..(c + 1) * d],
            );
        }

        // 5. Encode residuals straight into their final slots, in parallel.
        let t = Instant::now();
        let mut ids = vec![0u32; n];
        let mut codes = vec![0u8; n * cb];
        let mut scale = vec![0f32; n];
        let mut rnorm = vec![0f32; n];
        let (pi, pc, ps, pr) = (
            SyncPtr(ids.as_mut_ptr()),
            SyncPtr(codes.as_mut_ptr()),
            SyncPtr(scale.as_mut_ptr()),
            SyncPtr(rnorm.as_mut_ptr()),
        );
        let (rotate, est) = (p.rotate, p.estimator);
        (0..n).into_par_iter().for_each_init(
            || (vec![0f32; d], vec![0f32; d]),
            |(e, er), i| {
                let (pi, pc, ps, pr) = (pi, pc, ps, pr);
                let x = &data[i * d..(i + 1) * d];
                let c = assign[i] as usize;
                let cen = &centroids[c * d..(c + 1) * d];
                for j in 0..d {
                    e[j] = x[j] - cen[j];
                }
                let er: &[f32] = if rotate {
                    matvec(&rot, e, er);
                    er
                } else {
                    e
                };
                let l2 = dot(er, er);
                let l1: f32 = er.iter().map(|v| v.abs()).sum();
                let s = match est {
                    Estimator::LeastSquares => l1 / d as f32,
                    Estimator::Unbiased if l1 > 0.0 => l2 / l1,
                    Estimator::Unbiased => 0.0,
                };
                let slot = pos[i] as usize;
                // SAFETY: `pos` is a permutation of 0..n, so every thread writes distinct slots.
                unsafe {
                    let code = std::slice::from_raw_parts_mut(pc.0.add(slot * cb), cb);
                    for j in 0..cb {
                        let mut byte = 0u8;
                        for t in 0..8 {
                            if er[8 * j + t] > 0.0 {
                                byte |= 1 << t;
                            }
                        }
                        code[j] = byte;
                    }
                    *pi.0.add(slot) = i as u32;
                    *ps.0.add(slot) = s;
                    *pr.0.add(slot) = l2;
                }
            },
        );
        eprintln!("[build] encoded in {:.1}s", t.elapsed().as_secs_f64());

        let cnorms = kmeans::norms(&rcent, d);
        let idx = IvfIndex {
            d,
            nlist: p.nlist,
            n,
            rotate: p.rotate,
            estimator: p.estimator,
            rot,
            centroids: rcent,
            cnorms,
            offsets,
            ids: ids.into(),
            codes: codes.into(),
            scale: scale.into(),
            rnorm: rnorm.into(),
            fs_offsets: Vec::new(),
            fs_blocks: Vec::new().into(),
        };
        idx.print_balance();
        idx
    }

    /// The d×d rotation matrix (identity when `rotate` is false).
    pub fn rotation(&self) -> &[f32] {
        &self.rot
    }

    pub fn print_balance(&self) {
        let sizes: Vec<u64> = (0..self.nlist)
            .map(|c| self.offsets[c + 1] - self.offsets[c])
            .collect();
        let max = *sizes.iter().max().unwrap();
        let min = *sizes.iter().min().unwrap();
        let mean = self.n as f64 / self.nlist as f64;
        eprintln!(
            "[index] {} vectors, {} lists ({}), list size min {min} / mean {mean:.0} / max {max}, RAM {:.2} GB",
            self.n,
            self.nlist,
            self.variant(),
            self.ram_bytes() as f64 / 1e9
        );
    }

    /// Bytes this index keeps in RAM (the full vectors stay on disk).
    pub fn ram_bytes(&self) -> usize {
        self.rot.len() * 4
            + self.centroids.len() * 4
            + self.offsets.len() * 8
            + self.ids.len() * 4
            + self.codes.len()
            + self.fs_offsets.len() * 8
            + self.fs_blocks.len()
            + self.scale.len() * 4
            + self.rnorm.len() * 4
    }

    /// The `nprobe` lists nearest to an already-rotated query, nearest first.
    /// ||c||² − 2 q·c orders centroids by distance to q (||q||² is the same for all).
    fn nearest_lists(&self, qr: &[f32], nprobe: usize) -> Vec<(f32, u32)> {
        let d = self.d;
        let mut cd: Vec<(f32, u32)> = self
            .centroids
            .chunks_exact(d)
            .enumerate()
            .map(|(c, cen)| (self.cnorms[c] - 2.0 * dot(qr, cen), c as u32))
            .collect();
        if nprobe < cd.len() {
            cd.select_nth_unstable_by(nprobe - 1, |a, b| a.0.total_cmp(&b.0));
            cd.truncate(nprobe);
        }
        // Nearest list first: good candidates early make the threshold tight sooner.
        cd.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        cd
    }

    /// Which list each vector id was assigned to.
    pub fn list_of_ids(&self) -> Vec<u32> {
        let mut out = vec![0u32; self.n];
        for c in 0..self.nlist {
            for slot in self.offsets[c] as usize..self.offsets[c + 1] as usize {
                out[self.ids[slot] as usize] = c as u32;
            }
        }
        out
    }

    /// Ids of the lists a query would probe (for measuring the clustering ceiling).
    pub fn probed_lists(&self, q: &[f32], nprobe: usize) -> Vec<u32> {
        let mut qr = vec![0f32; self.d];
        if self.rotate {
            matvec(&self.rot, q, &mut qr);
        } else {
            qr.copy_from_slice(q);
        }
        self.nearest_lists(&qr, nprobe)
            .into_iter()
            .map(|(_, c)| c)
            .collect()
    }

    /// Search one query. Stages:
    /// 1. pick the `nprobe` nearest lists;
    /// 2. score every vector in them from its 1-bit code, keep the best `refine_k` (or `rerank`);
    /// 3. (optional) rescore those from the 4-bit `refine` codes in RAM, keep the best `rerank`;
    /// 4. (optional) rescore those exactly from the full vectors in `base` (on disk), keep `k`.
    ///
    /// Pass `times` to get how long each stage took.
    pub fn search(
        &self,
        q: &[f32],
        p: &SearchParams,
        refine: Option<&Refine>,
        base: Option<&Fbin>,
        times: Option<&mut StageTimes>,
    ) -> Vec<Hit> {
        self.search_impl(q, p, refine, base, times, None).0
    }

    /// Search among the vectors carrying one label. See `Filter` for the strategies.
    pub fn search_filtered(
        &self,
        q: &[f32],
        p: &SearchParams,
        refine: Option<&Refine>,
        base: Option<&Fbin>,
        times: Option<&mut StageTimes>,
        f: &Filter,
    ) -> (Vec<Hit>, FilterStats) {
        self.search_impl(q, p, refine, base, times, Some(f))
    }

    /// Lists to scan for a filtered query: walk lists nearest first, skipping lists with no
    /// match, until at least `nprobe` lists have been passed *and* `budget` matching vectors
    /// are covered. Centroids are ordered lazily, in growing chunks, since the walk may stop
    /// after a few hundred lists or need all of them. Returns `None` as soon as the lists would
    /// hold more than `give_up` vectors (the planner then uses the inverted index instead).
    fn filtered_lists(
        &self,
        qr: &[f32],
        nprobe: usize,
        f: &Filter,
        budget: usize,
        give_up: f64,
    ) -> Option<(Vec<(f32, u32)>, usize)> {
        let d = self.d;
        let mut all: Vec<(f32, u32)> = self
            .centroids
            .chunks_exact(d)
            .enumerate()
            .map(|(c, cen)| (self.cnorms[c] - 2.0 * dot(qr, cen), c as u32))
            .collect();
        let cmp = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0);
        let (mut sorted, mut chunk) = (0usize, nprobe.max(64));
        let (mut chosen, mut passing, mut scanned) = (Vec::new(), 0usize, 0usize);
        let mut i = 0;
        while i < all.len() {
            if i == sorted {
                let end = (sorted + chunk).min(all.len());
                let rest = &mut all[sorted..];
                if end - sorted < rest.len() {
                    rest.select_nth_unstable_by(end - sorted - 1, cmp);
                }
                all[sorted..end].sort_unstable_by(cmp);
                sorted = end;
                chunk *= 2;
            }
            let (dist, c) = all[i];
            i += 1;
            let cnt = f.labels.count(c as usize, f.label) as usize;
            if cnt > 0 {
                chosen.push((dist, c));
                passing += cnt;
                scanned += (self.offsets[c as usize + 1] - self.offsets[c as usize]) as usize;
                if scanned as f64 > give_up {
                    return None;
                }
            }
            if i >= nprobe && passing >= budget {
                break;
            }
        }
        Some((chosen, scanned))
    }

    fn search_impl(
        &self,
        q: &[f32],
        p: &SearchParams,
        refine: Option<&Refine>,
        base: Option<&Fbin>,
        times: Option<&mut StageTimes>,
        filter: Option<&Filter>,
    ) -> (Vec<Hit>, FilterStats) {
        let d = self.d;
        let cb = self.code_bytes();
        let nprobe = p.nprobe.min(self.nlist);
        let k = p.k;
        let t0 = Instant::now();
        let mut stats = FilterStats::default();

        // Rotate the query once; distances to rotated centroids equal distances to originals.
        let mut qr = vec![0f32; d];
        if self.rotate {
            matvec(&self.rot, q, &mut qr);
        } else {
            qr.copy_from_slice(q);
        }

        // Stage 1: which lists (or, for the inverted-index strategy, no lists at all).
        let mode = match filter {
            None => None,
            Some(f) if f.mode == FilterMode::Postings && refine.is_none() => Some(FilterMode::Scan),
            Some(f) => Some(f.mode),
        };
        let mut mode_used = mode;
        let cd = match (filter, mode) {
            (Some(f), Some(FilterMode::Scan | FilterMode::Auto)) => {
                let budget = f.budget.min(f.labels.frequency(f.label));
                // Auto: stop walking as soon as scanning would cost more than the postings.
                let give_up = if mode == Some(FilterMode::Auto) && refine.is_some() {
                    f.labels.frequency(f.label) as f64 * f.postings_cost
                } else {
                    f64::INFINITY
                };
                match self.filtered_lists(&qr, nprobe, f, budget, give_up) {
                    Some((lists, scanned)) => {
                        mode_used = Some(FilterMode::Scan);
                        stats.scanned = scanned;
                        lists
                    }
                    None => {
                        mode_used = Some(FilterMode::Postings);
                        Vec::new()
                    }
                }
            }
            (Some(_), Some(FilterMode::Postings)) => Vec::new(),
            _ => {
                let cd = self.nearest_lists(&qr, nprobe);
                stats.scanned = cd
                    .iter()
                    .map(|&(_, c)| {
                        (self.offsets[c as usize + 1] - self.offsets[c as usize]) as usize
                    })
                    .sum();
                cd
            }
        };
        stats.used = mode_used;
        stats.lists = cd.len();
        if mode_used != Some(FilterMode::Postings) && filter.is_some() {
            stats.probed = cd.iter().map(|&(_, c)| c).collect();
        }
        let in_scan = match (filter, mode_used) {
            (Some(f), Some(FilterMode::Scan)) => Some(f),
            _ => None,
        };
        let t1 = Instant::now();

        let use_refine = refine.is_some() && p.refine_k > 0;
        let want = if p.rerank > 0 { p.rerank.max(k) } else { k };
        let mut r = vec![0f32; d];

        let dense = in_scan.is_some_and(|f| {
            let m = f.labels.frequency(f.label);
            4 * m >= self.n && 4 * m <= 3 * self.n
        });
        let mut cand: Vec<Hit>;
        let t2;
        if let (Some(f), Some(FilterMode::Postings), Some(rf)) = (filter, mode_used, refine) {
            // Stage 2+3, inverted-index strategy: score exactly the matching vectors, from their
            // 4-bit codes (no 1-bit pass: there is nothing to prune). Postings are in slot order,
            // so lists come one after another, as in the refine stage.
            let posts = f.labels.postings(f.label);
            stats.scanned = posts.len();
            let mut best = Collector::new(want, p.heap);
            self.refine_slots(&qr, posts.iter().copied(), rf, &mut r, &mut best);
            cand = best.into_sorted();
            t2 = Instant::now();
        } else {
            // Stage 2: scan the 1-bit codes of those lists. Candidates are kept by *slot*
            // (position in list order), which is where the refine codes live; ids are looked up
            // at the end.
            let keep = if use_refine {
                p.refine_k.max(want)
            } else {
                want
            };
            let mut col = Collector::new(keep, p.heap);
            let fast = self.is_fastscan();
            let mut lut = vec![0f32; if fast { 0 } else { cb * 256 }];
            let mut qlut = vec![0u8; if fast { 2 * cb * 16 } else { 0 }];
            let mut sums: Vec<u16> = Vec::new();
            for &(_, c) in &cd {
                let c = c as usize;
                let cen = &self.centroids[c * d..(c + 1) * d];
                for j in 0..d {
                    r[j] = qr[j] - cen[j];
                }
                let rr = dot(&r, &r);
                let (lo, hi) = (self.offsets[c] as usize, self.offsets[c + 1] as usize);

                if fast {
                    // FastScan: integer scores for 32 vectors per SIMD step, then one pass to
                    // turn them into distance estimates.
                    let (delta, bias) = fastscan::build_qlut(&r, &mut qlut);
                    let bb = fastscan::block_bytes(2 * cb);
                    let (b0, b1) = (self.fs_offsets[c] as usize, self.fs_offsets[c + 1] as usize);
                    sums.resize((b1 - b0) * fastscan::BLOCK, 0);
                    fastscan::scan(&self.fs_blocks[b0 * bb..b1 * bb], 2 * cb, &qlut, &mut sums);
                    let mut thr = col.threshold();
                    let (rn, sc) = (&self.rnorm[lo..hi], &self.scale[lo..hi]);
                    let it = sums.iter().zip(rn).zip(sc).enumerate();
                    if let Some(f) = in_scan {
                        let lab = f.labels.labels(lo, hi);
                        if dense {
                            // Near 50% selectivity the label test is a coin flip the CPU
                            // mispredicts half the time; `&` instead of `&&` folds it into the
                            // threshold test, a branch that is almost always false.
                            for (v, ((&sum, &rn), &sc)) in it {
                                let est = rr + rn - 2.0 * sc * (delta * sum as f32 + bias);
                                if (est < thr) & (lab[v] == f.label) {
                                    col.push(est, (lo + v) as u32);
                                    thr = col.threshold();
                                }
                            }
                        } else {
                            // Rare or common labels: the label test is predictable, and skipping
                            // the arithmetic for non-matching vectors is the cheaper choice.
                            for (v, ((&sum, &rn), &sc)) in it {
                                if lab[v] != f.label {
                                    continue;
                                }
                                let est = rr + rn - 2.0 * sc * (delta * sum as f32 + bias);
                                if est < thr {
                                    col.push(est, (lo + v) as u32);
                                    thr = col.threshold();
                                }
                            }
                        }
                    } else {
                        for (v, ((&sum, &rn), &sc)) in it {
                            let est = rr + rn - 2.0 * sc * (delta * sum as f32 + bias);
                            if est < thr {
                                col.push(est, (lo + v) as u32);
                                thr = col.threshold();
                            }
                        }
                    }
                    continue;
                }

                build_lut(&r, &mut lut);
                let codes = &self.codes[lo * cb..hi * cb];
                for (v, code) in codes.chunks_exact(cb).enumerate() {
                    let slot = lo + v;
                    if let Some(f) = in_scan {
                        if f.labels.label(slot) != f.label {
                            continue;
                        }
                    }
                    let mut s = 0.0f32;
                    for (j, &b) in code.iter().enumerate() {
                        s += lut[j * 256 + b as usize];
                    }
                    let est = rr + self.rnorm[slot] - 2.0 * self.scale[slot] * s;
                    if est < col.threshold() {
                        col.push(est, slot as u32);
                    }
                }
            }
            cand = match (use_refine, refine) {
                // The refine stage doesn't need distance order. Slot order instead makes its
                // reads into the 5.6 GB refine array run forwards, list by list.
                (true, Some(_)) => {
                    let mut v = col.into_unsorted();
                    v.sort_unstable_by_key(|h| h.id);
                    v
                }
                _ => col.into_sorted(),
            };
            t2 = Instant::now();

            // Stage 3: 4-bit refine in RAM.
            if let (true, Some(rf)) = (use_refine, refine) {
                let mut best = Collector::new(want, true);
                self.refine_slots(&qr, cand.iter().map(|h| h.id), rf, &mut r, &mut best);
                cand = best.into_sorted();
            }
        }
        // Post-filtering: the unfiltered search ran; drop what doesn't match before the disk.
        if let (Some(f), Some(FilterMode::Post)) = (filter, mode_used) {
            cand.retain(|h| f.labels.label(h.id as usize) == f.label);
        }
        let t3 = Instant::now();

        // Stage 4: exact rescoring from the full (unrotated) vectors on disk.
        let out = match (p.rerank > 0, base) {
            (true, Some(base)) => {
                if p.prefetch {
                    for h in &cand {
                        base.prefetch(self.ids[h.id as usize] as usize);
                    }
                }
                let mut out = TopK::new(k);
                for h in &cand {
                    let id = self.ids[h.id as usize];
                    out.push(l2sq(q, base.vector(id as usize)), id);
                }
                out.into_sorted()
            }
            _ => {
                cand.truncate(k);
                cand.iter()
                    .map(|h| Hit {
                        dist: h.dist,
                        id: self.ids[h.id as usize],
                    })
                    .collect()
            }
        };
        if let Some(t) = times {
            let t4 = Instant::now();
            t.centroids += (t1 - t0).as_secs_f64();
            t.scan += (t2 - t1).as_secs_f64();
            t.refine += (t3 - t2).as_secs_f64();
            t.rerank += (t4 - t3).as_secs_f64();
        }
        (out, stats)
    }

    /// 4-bit distances for slots given in ascending order: each list's query residual is
    /// computed once, and codes are prefetched a few slots ahead.
    fn refine_slots(
        &self,
        qr: &[f32],
        slots: impl Iterator<Item = u32> + Clone,
        rf: &Refine,
        r: &mut [f32],
        out: &mut Collector,
    ) {
        let d = self.d;
        // Far enough ahead to keep ~16 cache misses in flight (the inverted-index path touches
        // vectors spread over the whole 5.6 GB array, so nearly every one is a miss).
        const AHEAD: usize = 16;
        let mut ahead = slots.clone().skip(AHEAD);
        for s in slots.clone().take(AHEAD) {
            rf.prefetch(s as usize);
        }
        let mut list_end = 0usize;
        for s in slots {
            if let Some(n) = ahead.next() {
                rf.prefetch(n as usize);
            }
            let slot = s as usize;
            if slot >= list_end {
                let c = self.offsets.partition_point(|&o| o as usize <= slot) - 1;
                list_end = self.offsets[c + 1] as usize;
                let cen = &self.centroids[c * d..(c + 1) * d];
                for j in 0..d {
                    r[j] = qr[j] - cen[j];
                }
            }
            out.push(rf.dist(slot, r), s);
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(path)?);
        w.write_all(if self.is_fastscan() { MAGIC_FS } else { MAGIC })?;
        let flags =
            (self.rotate as usize) | ((self.estimator == Estimator::Unbiased) as usize) << 1;
        for v in [self.d, self.nlist, self.n, flags] {
            w.write_all(&(v as u64).to_le_bytes())?;
        }
        write_pod(&mut w, &self.rot)?;
        write_pod(&mut w, &self.centroids)?;
        write_pod(&mut w, &self.offsets)?;
        if self.is_fastscan() {
            write_pod(&mut w, &self.fs_offsets)?;
        }
        write_pod(&mut w, &self.ids[..])?;
        if self.is_fastscan() {
            write_pod(&mut w, &self.fs_blocks[..])?;
        } else {
            write_pod(&mut w, &self.codes[..])?;
        }
        write_pod(&mut w, &self.scale[..])?;
        write_pod(&mut w, &self.rnorm[..])?;
        w.flush()
    }

    /// Load an index. The big per-vector arrays are memory-mapped (see `buf.rs`) and pre-read,
    /// so they live in RAM once. `copy = true` reads them into owned memory instead (v0.3
    /// behaviour, kept to measure the difference).
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with(path, false)
    }

    pub fn load_with(path: &Path, copy: bool) -> Result<Self> {
        let map = map_file(path, true)?;
        let fast = map.len() >= 8 && &map[0..8] == MAGIC_FS;
        if map.len() < 40 || !(fast || &map[0..8] == MAGIC) {
            return Err(std::io::Error::other(
                "not a binvec v0.2/v0.3 index file (rebuild it with `binvec build`)",
            ));
        }
        let word =
            |i: usize| u64::from_le_bytes(map[8 + 8 * i..16 + 8 * i].try_into().unwrap()) as usize;
        let (d, nlist, n, flags) = (word(0), word(1), word(2), word(3));
        let cb = d / 8;
        let mut off = 40;
        // Small arrays: always copied. Big arrays: mapped unless `copy`.
        let rot: Buf<f32> = Buf::copied(&map, off, d * d);
        off += d * d * 4;
        let centroids: Buf<f32> = Buf::copied(&map, off, nlist * d);
        off += nlist * d * 4;
        let offsets: Buf<u64> = Buf::copied(&map, off, nlist + 1);
        off += (nlist + 1) * 8;
        let fs_offsets: Vec<u64> = if fast {
            let v = Buf::<u64>::copied(&map, off, nlist + 1).to_vec();
            off += (nlist + 1) * 8;
            v
        } else {
            Vec::new()
        };
        // Bytes of the code section: FastScan blocks, or plain codes.
        let code_len = if fast {
            fs_offsets[nlist] as usize * fastscan::block_bytes(2 * cb)
        } else {
            n * cb
        };
        let big =
            |off: usize, len: usize, size: usize| -> (usize, usize) { (off, off + len * size) };
        let (o_ids, next) = big(off, n, 4);
        let (o_codes, next) = big(next, code_len, 1);
        let (o_scale, next) = big(next, n, 4);
        let (o_rnorm, _) = big(next, n, 4);
        let (ids, codes, scale, rnorm): (Buf<u32>, Buf<u8>, Buf<f32>, Buf<f32>) = if copy {
            (
                Buf::copied(&map, o_ids, n),
                Buf::copied(&map, o_codes, code_len),
                Buf::copied(&map, o_scale, n),
                Buf::copied(&map, o_rnorm, n),
            )
        } else {
            (
                Buf::mapped(&map, o_ids, n),
                Buf::mapped(&map, o_codes, code_len),
                Buf::mapped(&map, o_scale, n),
                Buf::mapped(&map, o_rnorm, n),
            )
        };
        let (plain, fs_blocks): (Buf<u8>, Buf<u8>) = if fast {
            (Vec::new().into(), codes)
        } else {
            (codes, Vec::new().into())
        };
        let centroids = centroids.to_vec();
        let cnorms = kmeans::norms(&centroids, d);
        Ok(IvfIndex {
            d,
            nlist,
            n,
            rotate: flags & 1 == 1,
            estimator: if flags & 2 == 2 {
                Estimator::Unbiased
            } else {
                Estimator::LeastSquares
            },
            rot: rot.to_vec(),
            centroids,
            cnorms,
            offsets: offsets.to_vec(),
            ids,
            codes: plain,
            scale,
            rnorm,
            fs_offsets,
            fs_blocks,
        })
    }
}

/// lut[j*256 + b] = Σ_{t=0..8} (bit t of b ? +r[8j+t] : −r[8j+t]).
/// Built incrementally: entry b differs from entry (b with its lowest set bit cleared) by
/// flipping one dimension from −r to +r, i.e. adding 2·r. So each entry costs one add.
#[inline]
pub fn build_lut(r: &[f32], lut: &mut [f32]) {
    let cb = r.len() / 8;
    for j in 0..cb {
        let rj = &r[8 * j..8 * j + 8];
        let t = &mut lut[j * 256..(j + 1) * 256];
        t[0] = -rj.iter().sum::<f32>();
        for b in 1..256usize {
            let low = b.trailing_zeros() as usize;
            t[b] = t[b & (b - 1)] + 2.0 * rj[low];
        }
    }
}

/// Plain-old-data types we can write as raw little-endian bytes (x86/ARM are little-endian).
pub trait Pod: Copy + Default {}
impl Pod for u8 {}
impl Pod for u16 {}
impl Pod for u32 {}
impl Pod for u64 {}
impl Pod for f32 {}

pub(crate) fn write_pod<T: Pod, W: Write>(w: &mut W, v: &[T]) -> Result<()> {
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    w.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_direct_sum() {
        let r: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut lut = vec![0f32; 2 * 256];
        build_lut(&r, &mut lut);
        for j in 0..2 {
            for b in 0..256usize {
                let direct: f32 = (0..8)
                    .map(|t| {
                        if b >> t & 1 == 1 {
                            r[8 * j + t]
                        } else {
                            -r[8 * j + t]
                        }
                    })
                    .sum();
                assert!((lut[j * 256 + b] - direct).abs() < 1e-4, "j={j} b={b}");
            }
        }
    }

    #[test]
    fn rotation_is_orthogonal_and_keeps_distances() {
        let d = 96;
        let p = random_rotation(d, 3);
        for i in 0..d {
            for j in 0..d {
                let g = dot(&p[i * d..(i + 1) * d], &p[j * d..(j + 1) * d]);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (g - want).abs() < 1e-4,
                    "P P^T not identity at ({i},{j}): {g}"
                );
            }
        }
        let mut rng = Rng::new(9);
        let a: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let b: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let (mut pa, mut pb) = (vec![0f32; d], vec![0f32; d]);
        matvec(&p, &a, &mut pa);
        matvec(&p, &b, &mut pb);
        assert!((l2sq(&a, &b) - l2sq(&pa, &pb)).abs() < 1e-3);
    }

    /// On anisotropic data (very uneven variance per dimension), the unbiased estimator with
    /// rotation should estimate r·e with much lower average error than v0.1 (ls, no rotation).
    #[test]
    fn rotation_plus_unbiased_reduces_error() {
        let d = 96;
        let mut rng = Rng::new(5);
        let p = random_rotation(d, 11);
        let err = |rotate: bool, est: Estimator, rng: &mut Rng| -> f64 {
            let mut total = 0.0f64;
            for _ in 0..2000 {
                // Variance falls off steeply with the dimension index, like real embeddings.
                let e: Vec<f32> = (0..d)
                    .map(|j| rng.normal() * (-(j as f32) / 12.0).exp())
                    .collect();
                let r: Vec<f32> = (0..d)
                    .map(|j| rng.normal() * (-(j as f32) / 12.0).exp())
                    .collect();
                let (mut er, mut rr) = (e.clone(), r.clone());
                if rotate {
                    matvec(&p, &e, &mut er);
                    matvec(&p, &r, &mut rr);
                }
                let l1: f32 = er.iter().map(|v| v.abs()).sum();
                let scale = match est {
                    Estimator::LeastSquares => l1 / d as f32,
                    Estimator::Unbiased => dot(&er, &er) / l1,
                };
                let rs: f32 = er
                    .iter()
                    .zip(&rr)
                    .map(|(a, b)| if *a > 0.0 { *b } else { -*b })
                    .sum();
                total += ((scale * rs - dot(&r, &e)) as f64).abs();
            }
            total / 2000.0
        };
        let base = err(false, Estimator::LeastSquares, &mut rng);
        let new = err(true, Estimator::Unbiased, &mut rng);
        assert!(
            new < 0.8 * base,
            "expected lower error: v0.1 {base:.4} vs v0.2 {new:.4}"
        );
    }
}
