//! x86-64 AVX2 reduction kernels.
//!
//! `vhaddpd` sums adjacent pairs but interleaves the two source
//! registers across the 128-bit lane boundary: from `[x0 x1 x2 x3]` and
//! `[x4 x5 x6 x7]` it produces `[x0+x1, x4+x5, x2+x3, x6+x7]`. A single
//! `vpermpd` with the selector `(0, 2, 1, 3)` puts that back in index
//! order. Four tree outputs per iteration, and each output is still one
//! IEEE-754 addition of the correct pair, so the result is
//! bit-identical to the scalar loop.
//!
//! Every function here is `#[target_feature(enable = "avx2")]` and must
//! only be reached through [`super::active_backend`], which probes
//! `CPUID`.

#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Number of `f64` lanes per 256-bit register.
const LANES: usize = 4;

/// Selector for `_mm256_permute4x64_pd`: result lane `i` takes source
/// lane `(IDX >> 2i) & 3`, giving `(0, 2, 1, 3)`.
const DEINTERLEAVE: i32 = 0b11_01_10_00;

/// See [`super::scalar::collapse`].
#[inline]
pub(super) fn collapse(buf: &mut [f64], half: usize) {
    // SAFETY: reached only when `active_backend()` reported AVX2.
    unsafe { collapse_avx2(buf, half) }
}

#[target_feature(enable = "avx2")]
unsafe fn collapse_avx2(buf: &mut [f64], half: usize) {
    let mut k = 0;
    let p = buf.as_mut_ptr();
    while k + LANES <= half {
        // SAFETY: reads cover `2k .. 2k+8 <= 2*half <= buf.len()`;
        // writes cover `k .. k+4 <= half`. `k + 4 < 2k + 8` always, so
        // the store never clobbers an unread input.
        let a = _mm256_loadu_pd(p.add(2 * k));
        let b = _mm256_loadu_pd(p.add(2 * k + 4));
        let h = _mm256_hadd_pd(a, b);
        _mm256_storeu_pd(p.add(k), _mm256_permute4x64_pd(h, DEINTERLEAVE));
        k += LANES;
    }
    for k in k..half {
        buf[k] = buf[2 * k] + buf[2 * k + 1];
    }
}

/// See [`super::scalar::sq_dev_collapse`].
#[inline]
pub(super) fn sq_dev_collapse(xs: &[f64], m: f64, out: &mut [f64]) {
    // SAFETY: reached only when `active_backend()` reported AVX2.
    unsafe { sq_dev_collapse_avx2(xs, m, out) }
}

#[target_feature(enable = "avx2")]
unsafe fn sq_dev_collapse_avx2(xs: &[f64], m: f64, out: &mut [f64]) {
    let half = out.len();
    let mut k = 0;
    let src = xs.as_ptr();
    let dst = out.as_mut_ptr();
    let mv = _mm256_set1_pd(m);
    while k + LANES <= half {
        // Separate subtract and multiply: `_mm256_fmadd_pd` would
        // round once instead of twice and diverge from the reference.
        let d0 = _mm256_sub_pd(_mm256_loadu_pd(src.add(2 * k)), mv);
        let d1 = _mm256_sub_pd(_mm256_loadu_pd(src.add(2 * k + 4)), mv);
        let s0 = _mm256_mul_pd(d0, d0);
        let s1 = _mm256_mul_pd(d1, d1);
        let h = _mm256_hadd_pd(s0, s1);
        _mm256_storeu_pd(dst.add(k), _mm256_permute4x64_pd(h, DEINTERLEAVE));
        k += LANES;
    }
    for k in k..half {
        let a = xs[2 * k] - m;
        let b = xs[2 * k + 1] - m;
        out[k] = a * a + b * b;
    }
}

/// See [`super::scalar::dot_collapse`].
#[inline]
pub(super) fn dot_collapse(a: &[f64], b: &[f64], out: &mut [f64]) {
    // SAFETY: reached only when `active_backend()` reported AVX2.
    unsafe { dot_collapse_avx2(a, b, out) }
}

#[target_feature(enable = "avx2")]
unsafe fn dot_collapse_avx2(a: &[f64], b: &[f64], out: &mut [f64]) {
    let half = out.len();
    let mut k = 0;
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let dst = out.as_mut_ptr();
    while k + LANES <= half {
        let p0 = _mm256_mul_pd(
            _mm256_loadu_pd(pa.add(2 * k)),
            _mm256_loadu_pd(pb.add(2 * k)),
        );
        let p1 = _mm256_mul_pd(
            _mm256_loadu_pd(pa.add(2 * k + 4)),
            _mm256_loadu_pd(pb.add(2 * k + 4)),
        );
        let h = _mm256_hadd_pd(p0, p1);
        _mm256_storeu_pd(dst.add(k), _mm256_permute4x64_pd(h, DEINTERLEAVE));
        k += LANES;
    }
    for k in k..half {
        out[k] = a[2 * k] * b[2 * k] + a[2 * k + 1] * b[2 * k + 1];
    }
}
