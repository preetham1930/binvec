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
//!                it shrinks the cross term, more for some vectors than others, which distorts
//!                the ranking. This was v0.1.
//! - `unbiased` : scale = ||e||² / ||e'||₁. The RaBitQ estimator (Gao & Long, SIGMOD 2024).
//!                Combined with a random rotation its error is unbiased and concentrated.
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

use crate::dist::{dot, l2sq};
use crate::io::Fbin;
use crate::kmeans;
use crate::rng::Rng;
use crate::topk::{Hit, TopK};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Result, Write};
use std::path::Path;
use std::time::Instant;

const MAGIC: &[u8; 8] = b"BINVEC02";

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
    pub ids: Vec<u32>,
    pub codes: Vec<u8>,
    pub scale: Vec<f32>,
    pub rnorm: Vec<f32>,
}

pub struct BuildParams {
    pub nlist: usize,
    pub train_n: usize,
    pub iters: usize,
    pub seed: u64,
    pub rotate: bool,
    pub estimator: Estimator,
}

/// Raw pointer we can share across threads; each thread writes disjoint slots.
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
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
fn matvec(p: &[f32], x: &[f32], out: &mut [f32]) {
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
            "{}+{}",
            if self.rotate { "rot" } else { "norot" },
            self.estimator.name()
        )
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
        let centroids = kmeans::train(&sample, d, p.nlist, p.iters, p.seed);
        drop(sample);
        let cnorms = kmeans::norms(&centroids, d);
        eprintln!("[build] k-means done in {:.1}s", t.elapsed().as_secs_f64());

        // 2. Assign every vector to its nearest centroid.
        let t = Instant::now();
        let assign = kmeans::assign_all(data, d, &centroids, &cnorms);
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
            ids,
            codes,
            scale,
            rnorm,
        };
        idx.print_balance();
        idx
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

    /// Search one query.
    /// - `nprobe`: how many nearest lists to scan (more = better recall, slower)
    /// - `rerank`: how many best code-scored candidates to rescore exactly (0 = no rerank)
    /// - `prefetch`: start all rerank reads before computing any (helps when the full vectors
    ///   are on disk rather than in the page cache)
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        nprobe: usize,
        rerank: usize,
        base: Option<&Fbin>,
        prefetch: bool,
    ) -> Vec<Hit> {
        let d = self.d;
        let cb = self.code_bytes();
        let nprobe = nprobe.min(self.nlist);

        // Rotate the query once; distances to rotated centroids equal distances to originals.
        let mut qr = vec![0f32; d];
        if self.rotate {
            matvec(&self.rot, q, &mut qr);
        } else {
            qr.copy_from_slice(q);
        }

        // Step 1: nearest lists.
        let cd = self.nearest_lists(&qr, nprobe);

        // Step 2: scan the codes of those lists.
        let keep = if rerank > 0 { rerank.max(k) } else { k };
        let mut cand = TopK::new(keep);
        let mut r = vec![0f32; d];
        let mut lut = vec![0f32; cb * 256];
        for &(_, c) in &cd {
            let c = c as usize;
            let cen = &self.centroids[c * d..(c + 1) * d];
            for j in 0..d {
                r[j] = qr[j] - cen[j];
            }
            let rr = dot(&r, &r);
            build_lut(&r, &mut lut);

            let (lo, hi) = (self.offsets[c] as usize, self.offsets[c + 1] as usize);
            let codes = &self.codes[lo * cb..hi * cb];
            for (v, code) in codes.chunks_exact(cb).enumerate() {
                let mut s = 0.0f32;
                for (j, &b) in code.iter().enumerate() {
                    s += lut[j * 256 + b as usize];
                }
                let slot = lo + v;
                let est = rr + self.rnorm[slot] - 2.0 * self.scale[slot] * s;
                if est < cand.threshold() {
                    cand.push(est, self.ids[slot]);
                }
            }
        }

        // Step 3: exact rescoring from the full (unrotated) vectors on disk.
        match (rerank > 0, base) {
            (true, Some(base)) => {
                let cand = cand.into_sorted();
                if prefetch {
                    for h in &cand {
                        base.prefetch(h.id as usize);
                    }
                }
                let mut out = TopK::new(k);
                for h in &cand {
                    out.push(l2sq(q, base.vector(h.id as usize)), h.id);
                }
                out.into_sorted()
            }
            _ => {
                let mut v = cand.into_sorted();
                v.truncate(k);
                v
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(path)?);
        w.write_all(MAGIC)?;
        let flags =
            (self.rotate as usize) | ((self.estimator == Estimator::Unbiased) as usize) << 1;
        for v in [self.d, self.nlist, self.n, flags] {
            w.write_all(&(v as u64).to_le_bytes())?;
        }
        write_pod(&mut w, &self.rot)?;
        write_pod(&mut w, &self.centroids)?;
        write_pod(&mut w, &self.offsets)?;
        write_pod(&mut w, &self.ids)?;
        write_pod(&mut w, &self.codes)?;
        write_pod(&mut w, &self.scale)?;
        write_pod(&mut w, &self.rnorm)?;
        w.flush()
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::with_capacity(1 << 22, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::other(
                "not a binvec v0.2 index file (rebuild it with `binvec build`)",
            ));
        }
        let mut u = [0u8; 8];
        let mut next = || -> Result<usize> {
            r.read_exact(&mut u)?;
            Ok(u64::from_le_bytes(u) as usize)
        };
        let (d, nlist, n, flags) = (next()?, next()?, next()?, next()?);
        let cb = d / 8;
        let rot: Vec<f32> = read_pod(&mut r, d * d)?;
        let centroids: Vec<f32> = read_pod(&mut r, nlist * d)?;
        let offsets: Vec<u64> = read_pod(&mut r, nlist + 1)?;
        let ids: Vec<u32> = read_pod(&mut r, n)?;
        let codes: Vec<u8> = read_pod(&mut r, n * cb)?;
        let scale: Vec<f32> = read_pod(&mut r, n)?;
        let rnorm: Vec<f32> = read_pod(&mut r, n)?;
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
            rot,
            centroids,
            cnorms,
            offsets,
            ids,
            codes,
            scale,
            rnorm,
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
trait Pod: Copy + Default {}
impl Pod for u8 {}
impl Pod for u32 {}
impl Pod for u64 {}
impl Pod for f32 {}

fn write_pod<T: Pod, W: Write>(w: &mut W, v: &[T]) -> Result<()> {
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    w.write_all(bytes)
}

fn read_pod<T: Pod, R: Read>(r: &mut R, len: usize) -> Result<Vec<T>> {
    let mut v = vec![T::default(); len];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * std::mem::size_of::<T>())
    };
    r.read_exact(bytes)?;
    Ok(v)
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
