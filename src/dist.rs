//! Distance kernels.
//!
//! Floating-point addition is not associative, so the compiler will not reorder a plain
//! `sum` into SIMD lanes on its own. Keeping 16 independent accumulators gives it permission:
//! each lane adds its own column, and with `target-cpu=native` this becomes AVX2/AVX-512 code.
//! `mul_add` becomes a single fused multiply-add instruction on CPUs with FMA.

const LANES: usize = 16; // one AVX-512 register of f32 (two AVX2 registers)

/// Squared Euclidean distance.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let (ac, bc) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let (ar, br) = (ac.remainder(), bc.remainder());
    for (x, y) in ac.zip(bc) {
        for l in 0..LANES {
            let t = x[l] - y[l];
            acc[l] = t.mul_add(t, acc[l]);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ar.iter().zip(br) {
        let t = x - y;
        s += t * t;
    }
    s
}

/// Dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let (ac, bc) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let (ar, br) = (ac.remainder(), bc.remainder());
    for (x, y) in ac.zip(bc) {
        for l in 0..LANES {
            acc[l] = x[l].mul_add(y[l], acc[l]);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive() {
        let a: Vec<f32> = (0..101).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..101).map(|i| (i as f32 * 0.11).cos()).collect();
        let naive_l2: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let naive_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((l2sq(&a, &b) - naive_l2).abs() < 1e-3);
        assert!((dot(&a, &b) - naive_dot).abs() < 1e-3);
    }
}
