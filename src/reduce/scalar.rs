//! Portable scalar reduction kernels.
//!
//! These are the reference semantics. Every vectorised kernel in this
//! directory must produce bit-identical output to the function of the
//! same name here.

/// `buf[k] = buf[2k] + buf[2k+1]` for `k < half`, in place.
///
/// Writing `buf[k]` is safe against the reads at `buf[2k]`,
/// `buf[2k+1]` because both are loaded before the store and
/// `k <= 2k` for all `k >= 0`.
#[inline]
pub(super) fn collapse(buf: &mut [f64], half: usize) {
    for k in 0..half {
        buf[k] = buf[2 * k] + buf[2 * k + 1];
    }
}

/// `out[k] = (xs[2k] - m)^2 + (xs[2k+1] - m)^2` for `k < out.len()`.
///
/// Written as subtract-then-multiply so the expression cannot be
/// contracted into a fused multiply-add.
#[inline]
pub(super) fn sq_dev_collapse(xs: &[f64], m: f64, out: &mut [f64]) {
    for k in 0..out.len() {
        let a = xs[2 * k] - m;
        let b = xs[2 * k + 1] - m;
        out[k] = a * a + b * b;
    }
}

/// `out[k] = a[2k]*b[2k] + a[2k+1]*b[2k+1]` for `k < out.len()`.
#[inline]
pub(super) fn dot_collapse(a: &[f64], b: &[f64], out: &mut [f64]) {
    for k in 0..out.len() {
        out[k] = a[2 * k] * b[2 * k] + a[2 * k + 1] * b[2 * k + 1];
    }
}
