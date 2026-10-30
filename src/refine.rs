//! Second in-RAM tier: a 4-bit copy of every vector's (rotated) residual.
//!
//! Why: at 100M, the 1-bit scan has to hand the SSD rerank about 1,000 candidates to reach
//! ~0.98 recall, and each one is a random disk read. A 4-bit code is much more accurate than the
//! 1-bit one, so it can re-sort a few thousand bit-scan survivors in RAM and pass only the best
//! few hundred to the disk. Fewer reads, better candidates.
//!
//! Per vector: 4 bits × d (48 bytes for d = 96) + two floats (`lo`, `step`) = 56 bytes.
//! 100M vectors ≈ 5.6 GB. Stored in slot order (the same order as the 1-bit codes), so a
//! candidate's slot finds its refine code directly.
//!
//! Quantisation: e ≈ lo + step·q with q ∈ {0..15} per dimension; lo = min(e), step = (max−min)/15.
//! The estimate ||r − ê||² is computed directly: d multiply-adds per candidate, no lookup table.
//! A few thousand candidates cost well under a millisecond.

use crate::buf::{map_file, Buf};
use crate::io::Fbin;
use crate::ivf::{matvec, write_pod, IvfIndex, SyncPtr};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::path::Path;
use std::time::Instant;

const MAGIC: &[u8; 8] = b"BINVREF1";

pub struct Refine {
    pub n: usize,
    pub d: usize,
    codes: Buf<u8>,
    lo: Buf<f32>,
    step: Buf<f32>,
}

impl Refine {
    pub fn ram_bytes(&self) -> usize {
        self.codes.len() + self.lo.len() * 4 + self.step.len() * 4
    }

    /// Quantise one residual into `code` (d/2 bytes); returns (lo, step).
    pub fn encode(e: &[f32], code: &mut [u8]) -> (f32, f32) {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in e {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let step = if hi > lo { (hi - lo) / 15.0 } else { 1.0 };
        let inv = 1.0 / step;
        for (j, b) in code.iter_mut().enumerate() {
            let q0 = ((e[2 * j] - lo) * inv).round().clamp(0.0, 15.0) as u8;
            let q1 = ((e[2 * j + 1] - lo) * inv).round().clamp(0.0, 15.0) as u8;
            *b = q0 | (q1 << 4);
        }
        (lo, step)
    }

    /// Estimated ||r − e||² for the vector at `slot`, where r = rotated query − its centroid.
    #[inline]
    pub fn dist(&self, slot: usize, r: &[f32]) -> f32 {
        let h = self.d / 2;
        let code = &self.codes[slot * h..(slot + 1) * h];
        let (lo, step) = (self.lo[slot], self.step[slot]);
        let mut s = 0.0f32;
        for (j, &b) in code.iter().enumerate() {
            let a = r[2 * j] - (lo + step * (b & 15) as f32);
            let c = r[2 * j + 1] - (lo + step * (b >> 4) as f32);
            s += a * a + c * c;
        }
        s
    }

    /// Build from an index and its base vectors. Reads `base` front to back once.
    pub fn build(idx: &IvfIndex, base: &Fbin) -> Self {
        let (n, d) = (idx.n, idx.d);
        assert_eq!(base.n, n, "base vector count does not match index");
        assert!(d % 2 == 0);
        let h = d / 2;
        let t = Instant::now();

        // id -> slot (inverse of idx.ids).
        let mut slot_of = vec![0u32; n];
        for (slot, &id) in idx.ids.iter().enumerate() {
            slot_of[id as usize] = slot as u32;
        }

        let mut codes = vec![0u8; n * h];
        let mut lo = vec![0f32; n];
        let mut step = vec![0f32; n];
        let (pc, pl, ps) = (
            SyncPtr(codes.as_mut_ptr()),
            SyncPtr(lo.as_mut_ptr()),
            SyncPtr(step.as_mut_ptr()),
        );
        base.advise_sequential();
        (0..n).into_par_iter().with_min_len(4096).for_each_init(
            || (vec![0f32; d], vec![0f32; d]),
            |(px, e), i| {
                let (pc, pl, ps) = (pc, pl, ps);
                let x = base.vector(i);
                let slot = slot_of[i] as usize;
                let c = idx.offsets.partition_point(|&o| o as usize <= slot) - 1;
                if idx.rotate {
                    matvec(idx.rotation(), x, px);
                } else {
                    px.copy_from_slice(x);
                }
                let cen = &idx.centroids[c * d..(c + 1) * d];
                for j in 0..d {
                    e[j] = px[j] - cen[j];
                }
                // SAFETY: slot_of is a permutation, so every thread writes distinct slots.
                unsafe {
                    let code = std::slice::from_raw_parts_mut(pc.0.add(slot * h), h);
                    let (l, s) = Refine::encode(e, code);
                    *pl.0.add(slot) = l;
                    *ps.0.add(slot) = s;
                }
            },
        );
        eprintln!(
            "[refine] encoded {n} vectors in {:.1}s ({:.2} GB)",
            t.elapsed().as_secs_f64(),
            (codes.len() + 8 * n) as f64 / 1e9
        );
        Refine {
            n,
            d,
            codes: codes.into(),
            lo: lo.into(),
            step: step.into(),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(path)?);
        w.write_all(MAGIC)?;
        for v in [self.n, self.d] {
            w.write_all(&(v as u64).to_le_bytes())?;
        }
        write_pod(&mut w, &self.codes[..])?;
        write_pod(&mut w, &self.lo[..])?;
        write_pod(&mut w, &self.step[..])?;
        w.flush()
    }

    /// Load, memory-mapped and pre-read (see `buf.rs`); `copy = true` reads into owned memory.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with(path, false)
    }

    pub fn load_with(path: &Path, copy: bool) -> Result<Self> {
        let map = map_file(path, true)?;
        if map.len() < 24 || &map[0..8] != MAGIC {
            return Err(std::io::Error::other("not a binvec refine file"));
        }
        let n = u64::from_le_bytes(map[8..16].try_into().unwrap()) as usize;
        let d = u64::from_le_bytes(map[16..24].try_into().unwrap()) as usize;
        let (o_codes, o_lo) = (24, 24 + n * d / 2);
        let o_step = o_lo + n * 4;
        let rf = if copy {
            Refine {
                n,
                d,
                codes: Buf::copied(&map, o_codes, n * d / 2),
                lo: Buf::copied(&map, o_lo, n),
                step: Buf::copied(&map, o_step, n),
            }
        } else {
            Refine {
                n,
                d,
                codes: Buf::mapped(&map, o_codes, n * d / 2),
                lo: Buf::mapped(&map, o_lo, n),
                step: Buf::mapped(&map, o_step, n),
            }
        };
        Ok(rf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn four_bit_estimate_is_close() {
        let d = 96;
        let mut rng = Rng::new(4);
        let e: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let r: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let mut code = vec![0u8; d / 2];
        let (lo, step) = Refine::encode(&e, &mut code);
        let rf = Refine {
            n: 1,
            d,
            codes: code.into(),
            lo: vec![lo].into(),
            step: vec![step].into(),
        };
        let exact: f32 = r.iter().zip(&e).map(|(a, b)| (a - b) * (a - b)).sum();
        let est = rf.dist(0, &r);
        // 16 levels over the range: error per dim ≤ step/2, so the estimate should be within a
        // few percent of the true squared distance.
        assert!(
            (est - exact).abs() / exact < 0.05,
            "exact {exact}, est {est}"
        );
    }
}
