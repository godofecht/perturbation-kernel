//! Reduction equivalence across vector paths (SCHEMA §8 D3).
//!
//! The claim these tests defend is narrow and total: *the vector path
//! makes no arithmetic difference*. Not "close", not "within
//! tolerance" -- the same bits, for every input and every length.
//!
//! That is only defensible because of how the reduction is shaped.
//! Each output of a tree level is one IEEE-754 addition of one fixed
//! pair of inputs, so a vector unit doing four of them at once does the
//! same four additions. If any kernel here ever reassociated, used a
//! horizontal accumulator, or let a subtract-multiply contract into an
//! FMA, these tests would fail.

use perturbation_kernel::reduce::{self, reference, SimdPath};
use proptest::prelude::*;

/// Deterministic spread-magnitude data. Mixing exponents is what makes
/// a reassociated sum diverge, so the fixtures deliberately do it.
fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut z = seed | 1;
    (0..n)
        .map(|i| {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            let mantissa = (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64) - 0.5;
            // Exponents from 2^-20 to 2^20, cycling.
            mantissa * 2f64.powi((i % 41) as i32 - 20)
        })
        .collect()
}

fn bits(x: f64) -> u64 {
    x.to_bits()
}

// ---------------------------------------------------------------------
// The vector paths agree with the v1.0.0 reference on every length.
// ---------------------------------------------------------------------

#[test]
fn tree_sum_matches_v1_reference_on_every_small_length() {
    for n in 0..=4_096usize {
        let xs = data(n, 0xA5A5_1234);
        let want = reference::tree_sum(&xs);
        let got = reduce::tree_sum(&xs);
        assert_eq!(
            bits(want),
            bits(got),
            "tree_sum diverged from the v1.0.0 reference at n = {n}"
        );
    }
}

#[test]
fn sum_sq_dev_matches_v1_reference_on_every_small_length() {
    for n in 0..=4_096usize {
        let xs = data(n, 0x1357_9BDF);
        let m = if xs.is_empty() {
            0.0
        } else {
            reduce::mean(&xs)
        };
        assert_eq!(
            bits(reference::sum_sq_dev(&xs, m)),
            bits(reduce::sum_sq_dev(&xs, m)),
            "sum_sq_dev diverged from the v1.0.0 reference at n = {n}"
        );
    }
}

#[test]
fn dot_matches_v1_reference_on_every_small_length() {
    for n in 0..=2_048usize {
        let a = data(n, 0x2468_ACE0);
        let b = data(n, 0xFDB9_7531);
        assert_eq!(
            bits(reference::dot(&a, &b)),
            bits(reduce::dot(&a, &b)),
            "dot diverged from the v1.0.0 reference at n = {n}"
        );
    }
}

// ---------------------------------------------------------------------
// Every available vector path agrees with every other.
// ---------------------------------------------------------------------

#[test]
fn all_vector_paths_agree_bit_for_bit() {
    let paths = reference::available_paths();
    // On a machine with no vector path this degenerates to a
    // self-comparison; say so rather than passing silently.
    eprintln!("vector paths under test: {paths:?}");

    for n in [
        0usize, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1_000, 4_097, 65_537,
    ] {
        let xs = data(n, 0xDEAD_BEEF);
        let m = reduce::mean(&xs);
        let ys = data(n, 0x0BAD_F00D);

        let sums: Vec<u64> = paths
            .iter()
            .map(|p| bits(reference::tree_sum_on(*p, &xs)))
            .collect();
        assert!(
            sums.windows(2).all(|w| w[0] == w[1]),
            "tree_sum disagreed across {paths:?} at n = {n}: {sums:x?}"
        );

        let devs: Vec<u64> = paths
            .iter()
            .map(|p| bits(reference::sum_sq_dev_on(*p, &xs, m)))
            .collect();
        assert!(
            devs.windows(2).all(|w| w[0] == w[1]),
            "sum_sq_dev disagreed across {paths:?} at n = {n}: {devs:x?}"
        );

        let dots: Vec<u64> = paths
            .iter()
            .map(|p| bits(reference::dot_on(*p, &xs, &ys)))
            .collect();
        assert!(
            dots.windows(2).all(|w| w[0] == w[1]),
            "dot disagreed across {paths:?} at n = {n}: {dots:x?}"
        );
    }
}

#[test]
fn the_vector_path_is_actually_being_exercised() {
    // Every other test in this file compares vector paths against the
    // scalar reference. If the dispatch quietly falls back to scalar,
    // they all still pass while testing nothing.
    //
    // Falling back is legitimate: `simd` may be off, the target may be
    // neither aarch64 nor x86-64, and an x86-64 CPU without AVX2 really
    // should use the scalar loop. Rosetta is one such CPU. So the
    // requirement is opt-in: CI sets `PK_REQUIRE_SIMD=1` on the runners
    // where a vector path is known to exist, which is where the AVX2
    // kernels actually get executed.
    let active = reduce::active_backend();
    eprintln!("active vector path: {}", active.as_str());
    eprintln!("paths under test:   {:?}", reference::available_paths());

    if std::env::var_os("PK_REQUIRE_SIMD").is_some() {
        assert_ne!(
            active,
            SimdPath::Scalar,
            "PK_REQUIRE_SIMD is set, but the reductions fell back to the \
             scalar path, so the vector kernels were never executed"
        );
    }
    assert_eq!(SimdPath::Scalar.as_str(), "scalar");
}

// ---------------------------------------------------------------------
// Non-finite inputs must propagate identically, not be special-cased.
// ---------------------------------------------------------------------

#[test]
fn non_finite_inputs_propagate_identically() {
    for n in [2usize, 3, 8, 9, 33] {
        for pos in [0usize, 1, n - 1] {
            for special in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
                let mut xs = data(n, 0x5EED);
                xs[pos] = special;
                let want = reference::tree_sum(&xs);
                let got = reduce::tree_sum(&xs);
                // NaN != NaN, so compare the classification instead.
                assert_eq!(
                    want.is_nan(),
                    got.is_nan(),
                    "NaN-ness diverged at n = {n}, pos = {pos}"
                );
                if !want.is_nan() {
                    assert_eq!(bits(want), bits(got), "diverged at n = {n}, pos = {pos}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Contract properties.
// ---------------------------------------------------------------------

#[test]
fn empty_and_singleton_are_defined() {
    assert_eq!(bits(reduce::tree_sum(&[])), bits(0.0));
    assert_eq!(bits(reduce::mean(&[])), bits(0.0));
    assert_eq!(bits(reduce::sum_sq_dev(&[], 3.0)), bits(0.0));
    assert_eq!(bits(reduce::dot(&[], &[])), bits(0.0));

    assert_eq!(bits(reduce::tree_sum(&[7.5])), bits(7.5));
    assert_eq!(bits(reduce::mean(&[7.5])), bits(7.5));
    assert_eq!(bits(reduce::sum_sq_dev(&[7.5], 5.5)), bits(4.0));
}

#[test]
fn scratch_reuse_does_not_change_the_answer() {
    let mut scratch = Vec::new();
    for n in [1usize, 2, 17, 100, 1_023] {
        let xs = data(n, 0xC0FFEE);
        let fresh = reduce::tree_sum(&xs);
        let reused = reduce::tree_sum_into(&xs, &mut scratch);
        assert_eq!(bits(fresh), bits(reused), "scratch reuse changed n = {n}");
    }
}

#[test]
#[should_panic(expected = "length mismatch")]
fn dot_rejects_mismatched_lengths() {
    reduce::dot(&[1.0, 2.0], &[1.0]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn prop_tree_sum_matches_reference(xs in prop::collection::vec(-1e12f64..1e12, 0..2000)) {
        prop_assert_eq!(bits(reference::tree_sum(&xs)), bits(reduce::tree_sum(&xs)));
    }

    #[test]
    fn prop_sum_sq_dev_matches_reference(
        xs in prop::collection::vec(-1e6f64..1e6, 0..2000),
        m in -1e6f64..1e6,
    ) {
        prop_assert_eq!(
            bits(reference::sum_sq_dev(&xs, m)),
            bits(reduce::sum_sq_dev(&xs, m))
        );
    }

    #[test]
    fn prop_all_paths_agree(xs in prop::collection::vec(-1e9f64..1e9, 0..1500)) {
        let m = reduce::mean(&xs);
        for p in reference::available_paths() {
            prop_assert_eq!(
                bits(reference::tree_sum_on(p, &xs)),
                bits(reference::tree_sum_on(SimdPath::Scalar, &xs))
            );
            prop_assert_eq!(
                bits(reference::sum_sq_dev_on(p, &xs, m)),
                bits(reference::sum_sq_dev_on(SimdPath::Scalar, &xs, m))
            );
        }
    }

    /// A tree sum of `n` copies of `x` is exactly `n * x` when `n` is a
    /// power of two: every partial sum is exactly representable.
    #[test]
    fn prop_power_of_two_sum_is_exact(x in -1e6f64..1e6, k in 0u32..12) {
        let n = 1usize << k;
        let xs = vec![x; n];
        prop_assert_eq!(bits(reduce::tree_sum(&xs)), bits(x * n as f64));
    }
}
