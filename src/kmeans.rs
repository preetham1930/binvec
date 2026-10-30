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
pub fn train(sample: &[f32], d: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
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
        let cn = norms(&centroids, d);
        // Assignment is the expensive part and runs in parallel; summing is cheap and serial.
        let assign = assign_all(sample, d, &centroids, &cn);
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
        eprintln!(
            "  kmeans iter {:>2}/{iters}: mean sq dist {:.5}, empty clusters {empty}",
            it + 1,
            inertia / n as f64
        );
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

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
