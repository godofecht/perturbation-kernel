//! Deterministic reductions (SCHEMA §8 D3).
//!
//! Every aggregation the crate performs goes through this module. The
//! contract is stronger than "fast": the reduction tree has a *fixed
//! shape* determined by the index order, so the result is a pure
//! function of the input slice and nothing else. The number of threads,
//! the CPU feature set, and the vector width are all invisible in the
//! output bits.
//!
//! # Reduction shape
//!
//! One level collapses `buf[k] = buf[2k] + buf[2k+1]` for
//! `k < len/2`, and carries an odd tail element up unchanged:
//!
//! ```text
//! [a b c d e]  ->  [a+b  c+d  e]  ->  [a+b+c+d  e]  ->  [a+b+c+d+e]
//! ```
//!
//! This is the `order = "tree"`, `leaf_order = "index"` policy of
//! SCHEMA §5, and it is what the v1.0.0 reference implementation
//! computed. All backends here reproduce it exactly.
//!
//! # Why SIMD is exact here
//!
//! Each output of a level is *one* IEEE-754 addition of one specific
//! pair of inputs. A vector unit performing four of those additions at
//! once performs the same four additions, so the result is
//! bit-identical to the scalar loop. No reassociation, no FMA
//! contraction, no horizontal accumulator. `tests/reduce.rs` asserts
//! this on every length from 0 to 4096 and under proptest.
//!
//! The SIMD paths are selected once per process by
//! [`active_backend`], which reads the CPU feature bits on `x86_64`
//! and uses the architectural NEON baseline on `aarch64`.

use crate::config::Backend;

mod scalar;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod neon;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2;

/// Which vector path the reductions are actually taking.
///
/// Reported by [`active_backend`] for provenance and used by the
/// cross-backend tests. It never changes the value of a reduction,
/// only how long it takes to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdPath {
    /// Portable scalar loop.
    Scalar,
    /// AArch64 Advanced SIMD (`vpaddq_f64`), 2 lanes.
    Neon,
    /// x86-64 AVX2 (`vhaddpd` + `vpermpd`), 4 lanes.
    Avx2,
}

impl SimdPath {
    /// Human-readable tag used in [`crate::report::Report`] provenance.
    pub fn as_str(self) -> &'static str {
        match self {
            SimdPath::Scalar => "scalar",
            SimdPath::Neon => "neon",
            SimdPath::Avx2 => "avx2",
        }
    }
}

/// The vector path this process will use for [`Backend::Auto`].
///
/// On `aarch64` NEON is part of the architectural baseline, so it is
/// always available. On `x86_64` AVX2 is probed with
/// `is_x86_feature_detected!`, which reads `CPUID` once and caches the
/// answer.
pub fn active_backend() -> SimdPath {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        SimdPath::Neon
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            SimdPath::Avx2
        } else {
            SimdPath::Scalar
        }
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64"))))]
    {
        SimdPath::Scalar
    }
}

/// Resolve a configured [`Backend`] to the vector path to use.
///
/// [`Backend::Gpu`] resolves to the host vector path here: the GPU
/// backend performs its own device-side reduction and only falls back
/// to this module for the host-side tail.
pub(crate) fn path_for(backend: Backend) -> SimdPath {
    match backend {
        Backend::Scalar => SimdPath::Scalar,
        Backend::Simd | Backend::Auto | Backend::Gpu | Backend::GpuF32 => active_backend(),
    }
}

/// Collapse one reduction level of `buf[..len]` in place.
///
/// Writes `buf[k] = buf[2k] + buf[2k+1]` for `k < len/2` and carries an
/// odd tail element to `buf[len/2]`. Returns the new length.
#[inline]
fn collapse(path: SimdPath, buf: &mut [f64], len: usize) -> usize {
    let half = len / 2;
    match path {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        SimdPath::Neon => neon::collapse(buf, half),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        SimdPath::Avx2 => avx2::collapse(buf, half),
        _ => scalar::collapse(buf, half),
    }
    if len & 1 == 1 {
        buf[half] = buf[len - 1];
        half + 1
    } else {
        half
    }
}

/// Deterministic pairwise sum of `xs` (SCHEMA §8 D3).
///
/// Bit-identical to the v1.0.0 reference reduction for every input, and
/// identical across all vector paths. Allocates one scratch buffer;
/// use [`tree_sum_into`] in a loop to reuse it.
pub fn tree_sum(xs: &[f64]) -> f64 {
    let mut scratch = Vec::new();
    tree_sum_into(xs, &mut scratch)
}

/// [`tree_sum`] reusing a caller-owned scratch buffer.
///
/// The buffer is overwritten, not read. This is the form the invariance
/// functionals use so that a `d`-dimensional ensemble costs one
/// allocation rather than `d`.
pub fn tree_sum_into(xs: &[f64], scratch: &mut Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    if xs.len() == 1 {
        return xs[0];
    }
    scratch.clear();
    scratch.extend_from_slice(xs);
    reduce_in_place(path_for(Backend::Auto), scratch)
}

/// Collapse `buf` to a single value, consuming it as scratch.
fn reduce_in_place(path: SimdPath, buf: &mut [f64]) -> f64 {
    let mut len = buf.len();
    if len == 0 {
        return 0.0;
    }
    while len > 1 {
        len = collapse(path, buf, len);
    }
    buf[0]
}

/// Deterministic mean, `tree_sum(xs) / xs.len()`.
///
/// Returns `0.0` for an empty slice, matching the "empty ensemble
/// yields a zero report" convention of the example functionals.
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    tree_sum(xs) / xs.len() as f64
}

/// [`mean`] reusing a caller-owned scratch buffer.
pub fn mean_into(xs: &[f64], scratch: &mut Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    tree_sum_into(xs, scratch) / xs.len() as f64
}

/// Deterministic sum of squared deviations from `m`.
///
/// Equal, bit for bit, to `tree_sum(&xs.map(|x| (x - m) * (x - m)))`.
/// The squaring is fused into the first reduction level so the
/// intermediate array of squares is never materialised, and it is
/// written as an explicit subtract-then-multiply so no compiler or
/// vector unit can contract it into an FMA (which would round
/// differently).
pub fn sum_sq_dev(xs: &[f64], m: f64) -> f64 {
    let mut scratch = Vec::new();
    sum_sq_dev_into(xs, m, &mut scratch)
}

/// [`sum_sq_dev`] reusing a caller-owned scratch buffer.
pub fn sum_sq_dev_into(xs: &[f64], m: f64, scratch: &mut Vec<f64>) -> f64 {
    sum_sq_dev_on(path_for(Backend::Auto), xs, m, scratch)
}

/// [`sum_sq_dev_into`] on an explicitly chosen vector path.
fn sum_sq_dev_on(path: SimdPath, xs: &[f64], m: f64, scratch: &mut Vec<f64>) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        let d = xs[0] - m;
        return d * d;
    }
    let half = n / 2;
    scratch.clear();
    scratch.resize(half + (n & 1), 0.0);

    match path {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        SimdPath::Neon => neon::sq_dev_collapse(xs, m, &mut scratch[..half]),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        SimdPath::Avx2 => avx2::sq_dev_collapse(xs, m, &mut scratch[..half]),
        _ => scalar::sq_dev_collapse(xs, m, &mut scratch[..half]),
    }
    if n & 1 == 1 {
        let d = xs[n - 1] - m;
        scratch[half] = d * d;
    }
    reduce_in_place(path, scratch)
}

/// Deterministic inner product under the same tree shape.
///
/// Equal, bit for bit, to `tree_sum(&a.zip(b).map(|(x, y)| x * y))`.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    dot_on(path_for(Backend::Auto), a, b)
}

/// [`dot`] on an explicitly chosen vector path.
fn dot_on(path: SimdPath, a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return a[0] * b[0];
    }
    let half = n / 2;
    let mut scratch = vec![0.0; half + (n & 1)];
    match path {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        SimdPath::Neon => neon::dot_collapse(a, b, &mut scratch[..half]),
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        SimdPath::Avx2 => avx2::dot_collapse(a, b, &mut scratch[..half]),
        _ => scalar::dot_collapse(a, b, &mut scratch[..half]),
    }
    if n & 1 == 1 {
        scratch[half] = a[n - 1] * b[n - 1];
    }
    reduce_in_place(path, &mut scratch)
}

/// Reference implementations, kept public to the crate so the tests can
/// assert the vector paths against them.
#[doc(hidden)]
pub mod reference {
    /// The literal v1.0.0 reduction: allocate a fresh level each time.
    pub fn tree_sum(xs: &[f64]) -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        let mut buf: Vec<f64> = xs.to_vec();
        while buf.len() > 1 {
            let n = buf.len();
            let half = n / 2;
            let mut next = Vec::with_capacity(half + (n & 1));
            for k in 0..half {
                next.push(buf[2 * k] + buf[2 * k + 1]);
            }
            if n & 1 == 1 {
                next.push(buf[n - 1]);
            }
            buf = next;
        }
        buf[0]
    }

    /// The literal v1.0.0 centred second moment: materialise the
    /// squares, then reduce.
    pub fn sum_sq_dev(xs: &[f64], m: f64) -> f64 {
        let centred: Vec<f64> = xs.iter().map(|x| (x - m) * (x - m)).collect();
        tree_sum(&centred)
    }

    /// Reference inner product under the same tree shape.
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        let prods: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
        tree_sum(&prods)
    }

    /// Run a reduction on an explicitly chosen vector path, for the
    /// cross-path equivalence tests.
    pub fn tree_sum_on(path: super::SimdPath, xs: &[f64]) -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        let mut buf = xs.to_vec();
        let mut len = buf.len();
        while len > 1 {
            len = super::collapse(path, &mut buf, len);
        }
        buf[0]
    }

    /// [`super::sum_sq_dev`] on an explicitly chosen vector path.
    pub fn sum_sq_dev_on(path: super::SimdPath, xs: &[f64], m: f64) -> f64 {
        super::sum_sq_dev_on(path, xs, m, &mut Vec::new())
    }

    /// [`super::dot`] on an explicitly chosen vector path.
    pub fn dot_on(path: super::SimdPath, a: &[f64], b: &[f64]) -> f64 {
        super::dot_on(path, a, b)
    }

    /// Every vector path this build can actually execute.
    ///
    /// Always contains [`super::SimdPath::Scalar`]; contains the
    /// vector path as well when the `simd` feature is on and the CPU
    /// supports it.
    pub fn available_paths() -> Vec<super::SimdPath> {
        let mut v = vec![super::SimdPath::Scalar];
        let active = super::active_backend();
        if active != super::SimdPath::Scalar {
            v.push(active);
        }
        v
    }
}
