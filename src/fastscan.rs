//! FastScan: score 32 vectors at a time with SIMD shuffles instead of scalar table lookups.
//!
//! ## Why
//! The byte-LUT scan (`ivf::build_lut`) does 12 dependent loads from 12 KB of f32 tables per
//! vector. At 100M it is 70–80% of query time. FastScan (André et al., "Cache locality is not
//! enough", VLDB 2015; the 4-bit "fast scan" kernels in FAISS) keeps the tables *in registers*.
//!
//! ## How
//! - Split the 96 sign bits into 24 groups of 4 dimensions ("nibbles"). Group g, bit t ↔ dim 4g+t.
//!   Each group has a 16-entry table T_g[c] = Σ_t (bit t of c ? +r : −r)[4g+t].
//! - Quantise every table to u8 with one shared step δ and a per-table offset:
//!   T_g[c] ≈ min_g + δ·q_g[c], q ∈ 0..=255. A 16-entry u8 table is one 128-bit register.
//! - Store codes in blocks of 32 vectors. For each group, 16 bytes: byte i holds the nibble of
//!   vector i (low 4 bits) and of vector i+16 (high 4 bits). One AVX2 `vpshufb` then looks up
//!   16 vectors in two tables at once.
//! - Sum the 24 u8 lookups per vector in u16 (max 24·255 = 6,120, no overflow), then
//!   r·s ≈ δ·sum + Σ_g min_g.
//!
//! The u8 rounding costs a little accuracy (at most 12·δ on the sum); the 4-bit refine tier
//! after the scan absorbs it.

pub const BLOCK: usize = 32;

/// Bytes per block for a code of `groups` nibbles per vector.
#[inline]
pub fn block_bytes(groups: usize) -> usize {
    groups * 16
}

/// Nibble g of a vector's bit code (cb = d/8 bytes): dims 4g..4g+3.
#[inline]
fn nibble(code: &[u8], g: usize) -> u8 {
    let b = code[g / 2];
    if g.is_multiple_of(2) {
        b & 15
    } else {
        b >> 4
    }
}

/// Append the blocked layout of `n` bit codes (each `cb` bytes) to `out`. The last block is
/// padded with zero codes; callers skip the padding by list length.
pub fn pack(codes: &[u8], cb: usize, out: &mut Vec<u8>) {
    let n = codes.len() / cb;
    let groups = cb * 2;
    let nb = n.div_ceil(BLOCK);
    let start = out.len();
    out.resize(start + nb * block_bytes(groups), 0);
    for b in 0..nb {
        let blk = &mut out[start + b * block_bytes(groups)..start + (b + 1) * block_bytes(groups)];
        for g in 0..groups {
            for i in 0..16 {
                let v0 = b * BLOCK + i;
                let v1 = v0 + 16;
                let lo = if v0 < n {
                    nibble(&codes[v0 * cb..(v0 + 1) * cb], g)
                } else {
                    0
                };
                let hi = if v1 < n {
                    nibble(&codes[v1 * cb..(v1 + 1) * cb], g)
                } else {
                    0
                };
                blk[g * 16 + i] = lo | (hi << 4);
            }
        }
    }
}

/// Quantised tables for one (query, list): `qlut` gets groups·16 bytes.
/// Returns (δ, Σ min_g) so that r·s ≈ δ·sum + bias.
pub fn build_qlut(r: &[f32], qlut: &mut [u8]) -> (f32, f32) {
    let groups = r.len() / 4;
    let mut t = [[0f32; 16]; 64];
    assert!(
        groups <= 64,
        "dimension too large for the fixed table buffer"
    );
    let mut bias = 0.0f32;
    let mut span = 0.0f32;
    for g in 0..groups {
        let rg = &r[4 * g..4 * g + 4];
        let tg = &mut t[g];
        tg[0] = -(rg[0] + rg[1] + rg[2] + rg[3]);
        for c in 1..16usize {
            let low = c.trailing_zeros() as usize;
            tg[c] = tg[c & (c - 1)] + 2.0 * rg[low];
        }
        let (mn, mx) = tg
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        bias += mn;
        span = span.max(mx - mn);
        // stash min in slot for the second pass
        tg.iter_mut().for_each(|v| *v -= mn);
    }
    let delta = if span > 0.0 { span / 255.0 } else { 1.0 };
    let inv = 1.0 / delta;
    for g in 0..groups {
        for c in 0..16 {
            qlut[g * 16 + c] = (t[g][c] * inv).round().clamp(0.0, 255.0) as u8;
        }
    }
    (delta, bias)
}

/// Integer sums for every vector of `nb` blocks (scalar reference implementation).
pub fn scan_scalar(blocks: &[u8], groups: usize, qlut: &[u8], sums: &mut [u16]) {
    let bb = block_bytes(groups);
    for (b, blk) in blocks.chunks_exact(bb).enumerate() {
        let out = &mut sums[b * BLOCK..(b + 1) * BLOCK];
        out.iter_mut().for_each(|s| *s = 0);
        for g in 0..groups {
            let tab = &qlut[g * 16..g * 16 + 16];
            for i in 0..16 {
                let byte = blk[g * 16 + i];
                out[i] += tab[(byte & 15) as usize] as u16;
                out[i + 16] += tab[(byte >> 4) as usize] as u16;
            }
        }
    }
}

/// Integer sums using AVX2 if the CPU has it, otherwise the scalar version.
/// Results are bit-identical either way.
pub fn scan(blocks: &[u8], groups: usize, qlut: &[u8], sums: &mut [u16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if groups.is_multiple_of(2) && groups <= 64 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support checked at runtime just above.
            unsafe { scan_avx2(blocks, groups, qlut, sums) };
            return;
        }
    }
    scan_scalar(blocks, groups, qlut, sums);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(blocks: &[u8], groups: usize, qlut: &[u8], sums: &mut [u16]) {
    use std::arch::x86_64::*;
    let bb = block_bytes(groups);
    let pairs = groups / 2;
    let nb = blocks.len() / bb;
    assert!(sums.len() >= nb * BLOCK);
    // Tables for groups (2p, 2p+1) side by side in one 256-bit register: lane 0 / lane 1.
    let mut luts = [_mm256_setzero_si256(); 32];
    for (p, l) in luts.iter_mut().enumerate().take(pairs) {
        *l = _mm256_loadu_si256(qlut.as_ptr().add(p * 32) as *const __m256i);
    }
    let low4 = _mm256_set1_epi8(0x0f);
    for b in 0..nb {
        let blk = blocks.as_ptr().add(b * bb);
        let mut acc_a = _mm256_setzero_si256(); // vectors 0..15
        let mut acc_b = _mm256_setzero_si256(); // vectors 16..31
        for (p, &lut) in luts.iter().enumerate().take(pairs) {
            // 32 bytes = group 2p (lane 0) and group 2p+1 (lane 1), same layout as the tables.
            let data = _mm256_loadu_si256(blk.add(p * 32) as *const __m256i);
            let lo = _mm256_and_si256(data, low4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(data, 4), low4);
            let ra = _mm256_shuffle_epi8(lut, lo);
            let rb = _mm256_shuffle_epi8(lut, hi);
            // Widen each 16-byte lane to 16 × u16 and add both groups into the accumulators.
            acc_a = _mm256_add_epi16(acc_a, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ra)));
            acc_a = _mm256_add_epi16(acc_a, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(ra, 1)));
            acc_b = _mm256_add_epi16(acc_b, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rb)));
            acc_b = _mm256_add_epi16(acc_b, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(rb, 1)));
        }
        let out = sums.as_mut_ptr().add(b * BLOCK) as *mut __m256i;
        _mm256_storeu_si256(out, acc_a);
        _mm256_storeu_si256(out.add(1), acc_b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivf::build_lut;
    use crate::rng::Rng;

    fn random_codes(n: usize, cb: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        (0..n * cb).map(|_| rng.below(256) as u8).collect()
    }

    #[test]
    fn simd_matches_scalar_exactly() {
        let (d, n) = (96, 77); // not a multiple of 32: exercises padding
        let cb = d / 8;
        let codes = random_codes(n, cb, 1);
        let mut blocks = Vec::new();
        pack(&codes, cb, &mut blocks);
        let mut rng = Rng::new(2);
        let r: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let mut q = vec![0u8; (d / 4) * 16];
        build_qlut(&r, &mut q);
        let nb = n.div_ceil(BLOCK);
        let (mut s1, mut s2) = (vec![0u16; nb * BLOCK], vec![0u16; nb * BLOCK]);
        scan_scalar(&blocks, d / 4, &q, &mut s1);
        scan(&blocks, d / 4, &q, &mut s2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn quantised_sum_tracks_float_lut() {
        // δ·sum + bias should be close to the exact f32 byte-LUT score for every vector.
        let (d, n) = (96, 64);
        let cb = d / 8;
        let codes = random_codes(n, cb, 3);
        let mut blocks = Vec::new();
        pack(&codes, cb, &mut blocks);
        let mut rng = Rng::new(4);
        let r: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let mut q = vec![0u8; (d / 4) * 16];
        let (delta, bias) = build_qlut(&r, &mut q);
        let mut sums = vec![0u16; n];
        scan(&blocks, d / 4, &q, &mut sums);
        let mut lut = vec![0f32; cb * 256];
        build_lut(&r, &mut lut);
        for v in 0..n {
            let exact: f32 = (0..cb)
                .map(|j| lut[j * 256 + codes[v * cb + j] as usize])
                .sum();
            let approx = delta * sums[v] as f32 + bias;
            // Each of 24 tables is rounded by at most δ/2.
            assert!(
                (approx - exact).abs() <= 12.0 * delta + 1e-3,
                "v={v}: {approx} vs {exact}"
            );
        }
    }
}
