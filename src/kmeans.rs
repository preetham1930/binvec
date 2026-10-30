//! k-means clustering: splits the space into `k` regions so a query only has to look inside
//! the few regions nearest to it (the "IVF" idea — inverted file).
//!
//! Trick used for speed: ||x − c||² = ||x||² − 2·x·c + ||c||². For a fixed x, ||x||² is the same
//! for every centroid, so the nearest centroid is the one minimising ||c||² − 2·x·c.
//! That turns each comparison into one dot product.
//!
//! ## Why assignment is done in batches
//! Finding the nearest of k centroids for n vectors is n·k dot products: the most expensive
//! step of the build (100M × 16384 ≈ 1.6 trillion). Done one vector at a time, every centroid
//! is pulled from L2/L3 cache (16384 × 96 floats = 6 MB doesn't fit in L1) for every vector,
//! so the CPU spends its time waiting on memory. In batches of `BATCH` vectors, each centroid is
//! loaded once and compared with all of them while it sits in registers/L1, which cuts memory
//! traffic by BATCH× and lets the FMA units stay busy.

use crate::dist::dot;
use crate::rng::Rng;
use rayon::prelude::*;

const BATCH: usize = 8;
const LANES: usize = 16;

/// Index of the nearest centroid, given precomputed centroid norms.
#[inline]
pub fn nearest(x: &[f32], centroids: &[f32], norms: &[f32], d: usize) -> usize {
    let mut best = f32::INFINITY;
    let mut best_i = 0;
    for (i, c) in centroids.chunks_exact(d).enumerate() {
        let s = norms[i] - 2.0 * dot(x, c);
        if s < best {
            best = s;
            best_i = i;
        }
    }
    best_i
}

/// Dot products of one centroid `c` with four vectors at once: `c` is read once per chunk and
/// multiplied into four accumulators, instead of being read four times. Written with
/// `chunks_exact` + fixed-size arrays so there are no bounds checks in the loop and the compiler
/// can keep all four accumulators in SIMD registers.
#[inline(always)]
fn dots4(c: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    let mut a0 = [0f32; LANES];
    let mut a1 = [0f32; LANES];
    let mut a2 = [0f32; LANES];
    let mut a3 = [0f32; LANES];
    let it = c
        .chunks_exact(LANES)
        .zip(x[0].chunks_exact(LANES))
        .zip(x[1].chunks_exact(LANES))
        .zip(x[2].chunks_exact(LANES))
        .zip(x[3].chunks_exact(LANES));
    for ((((cc, y0), y1), y2), y3) in it {
        let cc: &[f32; LANES] = cc.try_into().unwrap();
        let y0: &[f32; LANES] = y0.try_into().unwrap();
        let y1: &[f32; LANES] = y1.try_into().unwrap();
        let y2: &[f32; LANES] = y2.try_into().unwrap();
        let y3: &[f32; LANES] = y3.try_into().unwrap();
        for l in 0..LANES {
            a0[l] = cc[l].mul_add(y0[l], a0[l]);
            a1[l] = cc[l].mul_add(y1[l], a1[l]);
            a2[l] = cc[l].mul_add(y2[l], a2[l]);
            a3[l] = cc[l].mul_add(y3[l], a3[l]);
        }
    }
    let mut s = [
        a0.iter().sum::<f32>(),
        a1.iter().sum::<f32>(),
        a2.iter().sum::<f32>(),
        a3.iter().sum::<f32>(),
    ];
    let full = c.len() / LANES * LANES;
    for j in full..c.len() {
        for (k, xk) in x.iter().enumerate() {
            s[k] += c[j] * xk[j];
        }
    }
    s
}

/// Nearest centroid for up to BATCH vectors (`xs` holds m·d floats, m ≤ BATCH).
fn nearest_batch(xs: &[f32], centroids: &[f32], norms: &[f32], d: usize, out: &mut [u32]) {
    let m = xs.len() / d;
    if m < BATCH {
        for (b, o) in out.iter_mut().enumerate().take(m) {
            *o = nearest(&xs[b * d..(b + 1) * d], centroids, norms, d) as u32;
        }
        return;
    }
    let v = |b: usize| &xs[b * d..(b + 1) * d];
    let lo_x = [v(0), v(1), v(2), v(3)];
    let hi_x = [v(4), v(5), v(6), v(7)];
    let mut best = [f32::INFINITY; BATCH];
    let mut best_i = [0u32; BATCH];
    for (i, (c, &cn)) in centroids.chunks_exact(d).zip(norms).enumerate() {
        let lo = dots4(c, lo_x);
        let hi = dots4(c, hi_x);
        let p = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
        for b in 0..BATCH {
            let s = cn - 2.0 * p[b];
            if s < best[b] {
                best[b] = s;
                best_i[b] = i as u32;
            }
        }
    }
    out[..BATCH].copy_from_slice(&best_i);
}

/// Nearest centroid for every vector in `data`, in parallel and in batches.
pub fn assign_all(data: &[f32], d: usize, centroids: &[f32], norms: &[f32]) -> Vec<u32> {
    let n = data.len() / d;
    let mut out = vec![0u32; n];
    out.par_chunks_mut(BATCH)
        .zip(data.par_chunks(d * BATCH))
        .with_min_len(64)
        .for_each(|(o, xs)| nearest_batch(xs, centroids, norms, d, o));
    out
}

pub fn norms(centroids: &[f32], d: usize) -> Vec<f32> {
    centroids.chunks_exact(d).map(|c| dot(c, c)).collect()
}

/// Lloyd's algorithm on `sample` (n_s × d). Returns k·d centroids.
/// `probe` > 0 uses the two-level approximate assignment (see `TwoLevel`) in each iteration.
pub fn train(
    sample: &[f32],
    d: usize,
    k: usize,
    iters: usize,
    seed: u64,
    probe: usize,
) -> Vec<f32> {
    train_inner(sample, d, k, iters, seed, probe, true)
}

fn train_inner(
    sample: &[f32],
    d: usize,
    k: usize,
    iters: usize,
    seed: u64,
    probe: usize,
    verbose: bool,
) -> Vec<f32> {
    let n = sample.len() / d;
    assert!(n >= k, "need at least k sample points ({n} < {k})");
    let mut rng = Rng::new(seed);

    // Initialise with k distinct random sample points.
    let mut picked = std::collections::HashSet::with_capacity(k);
    let mut centroids = Vec::with_capacity(k * d);
    while picked.len() < k {
        let i = rng.below(n);
        if picked.insert(i) {
            centroids.extend_from_slice(&sample[i * d..(i + 1) * d]);
        }
    }

    for it in 0..iters {
        // Assignment is the expensive part and runs in parallel; summing is cheap and serial.
        let assign = if probe > 0 {
            TwoLevel::new(&centroids, d, probe, seed ^ it as u64).assign_all(sample)
        } else {
            let cn = norms(&centroids, d);
            assign_all(sample, d, &centroids, &cn)
        };
        let mut sums = vec![0.0f64; k * d];
        let mut counts = vec![0u64; k];
        let mut inertia = 0.0f64;
        for (x, &c) in sample.chunks_exact(d).zip(&assign) {
            let c = c as usize;
            counts[c] += 1;
            let s = &mut sums[c * d..(c + 1) * d];
            for j in 0..d {
                s[j] += x[j] as f64;
            }
            inertia += crate::dist::l2sq(x, &centroids[c * d..(c + 1) * d]) as f64;
        }

        let mut empty = 0;
        for c in 0..k {
            if counts[c] == 0 {
                // Empty cluster: restart it on a random sample point.
                empty += 1;
                let i = rng.below(n);
                centroids[c * d..(c + 1) * d].copy_from_slice(&sample[i * d..(i + 1) * d]);
            } else {
                for j in 0..d {
                    centroids[c * d + j] = (sums[c * d + j] / counts[c] as f64) as f32;
                }
            }
        }
        if verbose {
            eprintln!(
                "  kmeans iter {:>2}/{iters}: mean sq dist {:.5}, empty clusters {empty}",
                it + 1,
                inertia / n as f64
            );
        }
    }
    centroids
}

/// Two-level approximate nearest-centroid search, for building large indexes.
///
/// Exact assignment compares each vector with all k centroids (16,384 at 100M: 70% of the build).
/// Here the centroids themselves are clustered into ~sqrt(k) groups. A vector is compared with
/// the group centres, then only with the centroids in its `probe` nearest groups:
/// sqrt(k) + probe·sqrt(k) dot products instead of k (≈1,150 instead of 16,384 for probe 8).
/// The price is that a vector near a group boundary can land in its second-nearest centroid;
/// `agreement` measures how often that happens.
pub struct TwoLevel {
    d: usize,
    probe: usize,
    coarse: Vec<f32>,
    coarse_norms: Vec<f32>,
    /// Group g's centroids are grouped[starts[g]..starts[g+1]] (rows of d floats).
    starts: Vec<usize>,
    grouped: Vec<f32>,
    grouped_norms: Vec<f32>,
    /// Original index of each grouped centroid.
    members: Vec<u32>,
}

impl TwoLevel {
    pub fn new(centroids: &[f32], d: usize, probe: usize, seed: u64) -> Self {
        let k = centroids.len() / d;
        let ck = ((k as f64).sqrt().round() as usize).clamp(1, k);
        let coarse = train_inner(centroids, d, ck, 10, seed ^ 0xC0A5, 0, false);
        let coarse_norms = norms(&coarse, d);
        let owner = assign_all(centroids, d, &coarse, &coarse_norms);
        let mut starts = vec![0usize; ck + 1];
        for &g in &owner {
            starts[g as usize + 1] += 1;
        }
        for g in 0..ck {
            starts[g + 1] += starts[g];
        }
        let mut cursor = starts.clone();
        let mut grouped = vec![0f32; k * d];
        let mut members = vec![0u32; k];
        for (c, &g) in owner.iter().enumerate() {
            let m = cursor[g as usize];
            cursor[g as usize] += 1;
            grouped[m * d..(m + 1) * d].copy_from_slice(&centroids[c * d..(c + 1) * d]);
            members[m] = c as u32;
        }
        let grouped_norms = norms(&grouped, d);
        TwoLevel {
            d,
            probe: probe.clamp(1, ck),
            coarse,
            coarse_norms,
            starts,
            grouped,
            grouped_norms,
            members,
        }
    }

    #[inline]
    fn nearest(&self, x: &[f32], scratch: &mut Vec<(f32, u32)>) -> u32 {
        let d = self.d;
        scratch.clear();
        for (g, c) in self.coarse.chunks_exact(d).enumerate() {
            scratch.push((self.coarse_norms[g] - 2.0 * dot(x, c), g as u32));
        }
        let p = self.probe;
        if p < scratch.len() {
            scratch.select_nth_unstable_by(p - 1, |a, b| a.0.total_cmp(&b.0));
        }
        let mut best = f32::INFINITY;
        let mut best_m = 0usize;
        for &(_, g) in scratch.iter().take(p) {
            let (lo, hi) = (self.starts[g as usize], self.starts[g as usize + 1]);
            for m in lo..hi {
                let s = self.grouped_norms[m] - 2.0 * dot(x, &self.grouped[m * d..(m + 1) * d]);
                if s < best {
                    best = s;
                    best_m = m;
                }
            }
        }
        self.members[best_m]
    }

    pub fn assign_all(&self, data: &[f32]) -> Vec<u32> {
        data.par_chunks(self.d)
            .map_init(Vec::new, |scratch, x| self.nearest(x, scratch))
            .collect()
    }

    /// Fraction of `sample` vectors assigned to the same centroid as exact search would pick.
    pub fn agreement(&self, sample: &[f32], centroids: &[f32]) -> f64 {
        let cn = norms(centroids, self.d);
        let approx = self.assign_all(sample);
        let exact = assign_all(sample, self.d, centroids, &cn);
        approx.iter().zip(&exact).filter(|(a, b)| a == b).count() as f64 / approx.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_level_with_full_probe_is_exact() {
        let (d, k, n) = (32, 400, 500);
        let mut rng = Rng::new(2);
        let cents: Vec<f32> = (0..k * d).map(|_| rng.normal()).collect();
        let data: Vec<f32> = (0..n * d).map(|_| rng.normal()).collect();
        // Probing every group must give exactly the exact answer.
        let t = TwoLevel::new(&cents, d, usize::MAX, 3);
        assert_eq!(t.agreement(&data, &cents), 1.0);
        // A small probe is approximate but should still mostly agree.
        let t = TwoLevel::new(&cents, d, 4, 3);
        assert!(t.agreement(&data, &cents) > 0.5);
    }

    #[test]
    fn batched_matches_single() {
        let (d, k, n) = (96, 300, 203); // n not a multiple of BATCH on purpose
        let mut rng = Rng::new(1);
        let cents: Vec<f32> = (0..k * d).map(|_| rng.normal()).collect();
        let data: Vec<f32> = (0..n * d).map(|_| rng.normal()).collect();
        let cn = norms(&cents, d);
        let got = assign_all(&data, d, &cents, &cn);
        for i in 0..n {
            assert_eq!(
                got[i] as usize,
                nearest(&data[i * d..(i + 1) * d], &cents, &cn, d),
                "vector {i}"
            );
        }
    }
}
