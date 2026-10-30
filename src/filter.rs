//! Filtered search: "the nearest neighbours of q among vectors with label L".
//!
//! ## The problem
//! A web index is searched with filters (a domain, a date range, a category). If you search
//! first and filter after, a filter that keeps 1% of vectors throws away ~99% of the results and
//! recall collapses. If you filter first and search after, you need an index per filter.
//!
//! ## What this module provides
//! - Labels: one `u16` label per vector, with **planted frequencies** so selectivity is exact:
//!   label 0 is on 50% of vectors, label 1 on 10%, 2 on 1%, 3 on 0.1%, 4 on 0.01%; the remaining
//!   38.89% is spread over 1,000 background labels.
//!   - `uniform`: labels are independent of position (the easy case for an IVF index).
//!   - `correlated`: labels follow a random direction in space, so each label occupies a slab.
//!     A query's filtered neighbours are then often far from its own clusters (the hard case).
//! - `Labels`, built once per index from the per-id labels:
//!   - the label of every slot, read inside the scan to skip non-matching vectors;
//!   - an **inverted index**: for each label, the slots that carry it (sorted);
//!   - per-list label counts, used to skip lists without matches and to plan each query.
//!
//! The strategies that use these are in `IvfIndex::search_filtered`.

use crate::buf::Buf;
use crate::io::Fbin;
use crate::ivf::{write_pod, IvfIndex};
use crate::rng::Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Result, Write};
use std::path::Path;

/// Planted selectivities of labels 0..5.
pub const TIERS: [f64; 5] = [0.5, 0.1, 0.01, 0.001, 0.0001];
/// Background labels sharing the rest.
pub const BACKGROUND: usize = 1000;
pub const NLABELS: usize = TIERS.len() + BACKGROUND;

const MAGIC: &[u8; 8] = b"BINVLAB1";

/// A well-mixed 64-bit hash (splitmix64): independent pseudo-random values per id, in parallel.
#[inline]
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Map a position u in [0, 1) to a label: the first 61.11% of the line is split into the tiers,
/// the rest goes to a background label chosen by `h`.
#[inline]
pub fn label_at(u: f64, h: u64) -> u16 {
    let mut acc = 0.0;
    for (t, &s) in TIERS.iter().enumerate() {
        acc += s;
        if u < acc {
            return t as u16;
        }
    }
    (TIERS.len() + (h % BACKGROUND as u64) as usize) as u16
}

/// Labels independent of the vectors.
pub fn uniform_labels(n: usize, seed: u64) -> Vec<u16> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let h = mix(seed ^ (i as u64).wrapping_mul(0x9E37_79B9));
            label_at(unit(h), mix(h))
        })
        .collect()
}

/// Labels that follow a random direction: a vector's position on the line is the rank of its
/// projection, so each tier is a slab of space. Ranks come from a sorted sample of projections.
pub fn correlated_labels(base: &Fbin, seed: u64) -> Vec<u16> {
    let d = base.d;
    let mut rng = Rng::new(seed);
    let mut dir: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
    let norm = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
    dir.iter_mut().for_each(|x| *x /= norm);
    let proj = |v: &[f32]| v.iter().zip(&dir).map(|(a, b)| a * b).sum::<f32>();
    let step = (base.n / 1_000_000).max(1);
    let mut sample: Vec<f32> = (0..base.n)
        .step_by(step)
        .map(|i| proj(base.vector(i)))
        .collect();
    sample.sort_unstable_by(|a, b| a.total_cmp(b));
    let len = sample.len() as f64;
    base.advise_sequential();
    base.data()
        .par_chunks(d)
        .enumerate()
        .map(|(i, v)| {
            let p = proj(v);
            let u = sample.partition_point(|&s| s < p) as f64 / len;
            label_at(u.min(0.999_999_999), mix(seed ^ i as u64))
        })
        .collect()
}

pub fn write_labels(path: &Path, labels: &[u16]) -> Result<()> {
    let mut w = BufWriter::with_capacity(1 << 22, File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&(labels.len() as u64).to_le_bytes())?;
    w.write_all(&(NLABELS as u32).to_le_bytes())?;
    write_pod(&mut w, labels)?;
    w.flush()
}

pub fn read_labels(path: &Path) -> Result<Vec<u16>> {
    let mut r = BufReader::with_capacity(1 << 22, File::open(path)?);
    let mut m = [0u8; 8];
    r.read_exact(&mut m)?;
    if &m != MAGIC {
        return Err(std::io::Error::other("not a binvec labels file"));
    }
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8)?;
    let n = u64::from_le_bytes(b8) as usize;
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    let mut bytes = vec![0u8; n * 2];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Per-index label structures, all keyed by slot (position in list order).
pub struct Labels {
    pub nlabels: usize,
    nlist: usize,
    slot_label: Buf<u16>,
    post_off: Vec<u64>,
    post: Buf<u32>,
    counts: Vec<u32>,
}

impl Labels {
    pub fn build(idx: &IvfIndex, by_id: &[u16]) -> Self {
        assert_eq!(by_id.len(), idx.n, "labels file does not match the index");
        let nl = NLABELS;
        let slot_label: Vec<u16> = idx.ids.par_iter().map(|&id| by_id[id as usize]).collect();
        // Per-list counts, one row per list.
        let counts: Vec<u32> = (0..idx.nlist)
            .into_par_iter()
            .flat_map_iter(|c| {
                let mut row = vec![0u32; nl];
                for &l in &slot_label[idx.offsets[c] as usize..idx.offsets[c + 1] as usize] {
                    row[l as usize] += 1;
                }
                row
            })
            .collect();
        // Inverted index by counting sort; walking slots in order keeps each posting list sorted.
        let mut post_off = vec![0u64; nl + 1];
        for &l in &slot_label {
            post_off[l as usize + 1] += 1;
        }
        for l in 0..nl {
            post_off[l + 1] += post_off[l];
        }
        let mut fill: Vec<u64> = post_off[..nl].to_vec();
        let mut post = vec![0u32; idx.n];
        for (slot, &l) in slot_label.iter().enumerate() {
            post[fill[l as usize] as usize] = slot as u32;
            fill[l as usize] += 1;
        }
        Labels {
            nlabels: nl,
            nlist: idx.nlist,
            slot_label: slot_label.into(),
            post_off,
            post: post.into(),
            counts,
        }
    }

    #[inline]
    pub fn label(&self, slot: usize) -> u16 {
        self.slot_label[slot]
    }

    /// Slot labels for a range of slots (one list).
    #[inline]
    pub fn labels(&self, lo: usize, hi: usize) -> &[u16] {
        &self.slot_label[lo..hi]
    }

    /// How many vectors in list `c` carry label `l`.
    #[inline]
    pub fn count(&self, c: usize, l: u16) -> u32 {
        debug_assert!(c < self.nlist);
        self.counts[c * self.nlabels + l as usize]
    }

    /// Slots carrying label `l`, ascending.
    #[inline]
    pub fn postings(&self, l: u16) -> &[u32] {
        let l = l as usize;
        &self.post[self.post_off[l] as usize..self.post_off[l + 1] as usize]
    }

    pub fn frequency(&self, l: u16) -> usize {
        self.postings(l).len()
    }

    pub fn ram_bytes(&self) -> usize {
        self.slot_label.len() * 2 + self.post.len() * 4 + self.counts.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planted_frequencies_hold() {
        let n = 2_000_000;
        let l = uniform_labels(n, 7);
        let mut c = vec![0usize; NLABELS];
        for &x in &l {
            c[x as usize] += 1;
        }
        for (t, &s) in TIERS.iter().enumerate() {
            let got = c[t] as f64 / n as f64;
            // within 5 standard deviations
            let sd = (s * (1.0 - s) / n as f64).sqrt();
            assert!((got - s).abs() < 5.0 * sd, "tier {t}: {got} vs {s}");
        }
    }
}
