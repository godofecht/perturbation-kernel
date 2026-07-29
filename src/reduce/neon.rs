//! AArch64 Advanced SIMD reduction kernels.
//!
//! `vpaddq_f64` is exactly the primitive this reduction shape wants:
//! given `[a0 a1]` and `[a2 a3]` it returns `[a0+a1, a2+a3]`, which is
//! one tree level for two outputs in one instruction. Each lane
//! performs the same single IEEE-754 addition the scalar loop performs,
//! so the output is bit-identical.
//!
//! NEON is part of the AArch64 architectural baseline, so no runtime
//! feature probe is needed and these functions are safe to call
//! unconditionally on this target.

#![allow(unsafe_code)]

use std::arch::aarch64::*;

/// Number of `f64` lanes per vector register.
const LANES: usize = 2;

/// See [`super::scalar::collapse`].
#[inline]
pub(super) fn collapse(buf: &mut [f64], half: usize) {
    // Two outputs per iteration consume four inputs. The store to
    // `[k, k+1]` never overlaps the not-yet-read window `[2k+4, ..)`
    // because `k + 1 < 2k + 4` for all `k >= 0`.
    let mut k = 0;
    let p = buf.as_mut_ptr();
    while k + LANES <= half {
        // SAFETY: reads cover `2k .. 2k+4 <= 2*half <= buf.len()`;
        // writes cover `k .. k+2 <= half <= buf.len()`.
        unsafe {
            let v0 = vld1q_f64(p.add(2 * k));
            let v1 = vld1q_f64(p.add(2 * k + 2));
            vst1q_f64(p.add(k), vpaddq_f64(v0, v1));
        }
        k += LANES;
    }
    for k in k..half {
        buf[k] = buf[2 * k] + buf[2 * k + 1];
    }
}

/// See [`super::scalar::sq_dev_collapse`].
#[inline]
pub(super) fn sq_dev_collapse(xs: &[f64], m: f64, out: &mut [f64]) {
    let half = out.len();
    let mut k = 0;
    let src = xs.as_ptr();
    let dst = out.as_mut_ptr();
    // SAFETY: `out.len() == half` and the caller guarantees
    // `xs.len() >= 2 * half`.
    unsafe {
        let mv = vdupq_n_f64(m);
        while k + LANES <= half {
            // Subtract then multiply as two separate instructions:
            // `vfmaq` would round once instead of twice and diverge
            // from the scalar reference.
            let d0 = vsubq_f64(vld1q_f64(src.add(2 * k)), mv);
            let d1 = vsubq_f64(vld1q_f64(src.add(2 * k + 2)), mv);
            let s0 = vmulq_f64(d0, d0);
            let s1 = vmulq_f64(d1, d1);
            vst1q_f64(dst.add(k), vpaddq_f64(s0, s1));
            k += LANES;
        }
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
    let half = out.len();
    let mut k = 0;
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let dst = out.as_mut_ptr();
    // SAFETY: the caller guarantees `a.len() == b.len() >= 2 * half`.
    unsafe {
        while k + LANES <= half {
            let p0 = vmulq_f64(vld1q_f64(pa.add(2 * k)), vld1q_f64(pb.add(2 * k)));
            let p1 = vmulq_f64(vld1q_f64(pa.add(2 * k + 2)), vld1q_f64(pb.add(2 * k + 2)));
            vst1q_f64(dst.add(k), vpaddq_f64(p0, p1));
            k += LANES;
        }
    }
    for k in k..half {
        out[k] = a[2 * k] * b[2 * k] + a[2 * k + 1] * b[2 * k + 1];
    }
}
