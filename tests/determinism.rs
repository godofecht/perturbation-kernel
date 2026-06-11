//! Determinism / reproducibility (SCHEMA §8; Paper Thm 8.2).
//!
//! D1: seed totality -- the 64-bit seed fully determines the stream.
//! D2: per-index substreams -- changing one index does not perturb
//!     the others.
//! D3: deterministic reduction order -- tree+index.
//! D4: bit-for-bit reproducibility under repeated runs.

use perturbation_kernel::config::{Config, Intensity, Lipschitz, Reduction};
use perturbation_kernel::engine::{fork_rng, Engine};
use perturbation_kernel::examples::{gaussian, markov, Vector};
use serde_json::json;

fn cfg_for(seed: u64, n: u64) -> Config {
    Config {
        schema_version: "1.0.0".into(),
        n,
        seed,
        intensity: Intensity {
            kind: "uniform_interval".into(),
            params: json!({ "low": 0.0, "high": 0.3 }),
            null_parameter: json!(0.0),
        },
        reduction: Reduction::default(),
        lipschitz: Lipschitz {
            forward_l: Some(1.0),
            invariance_lambda: Some(1.0),
        },
        accuracy: None,
    }
}

// ---------------------------------------------------------------------
// D4: bit-for-bit equality on repeated runs (SCHEMA §8 D4).
// ---------------------------------------------------------------------

#[test]
fn d4_same_seed_same_value_gaussian() {
    let cfg = cfg_for(12345, 4_096);
    let base: Vector = vec![0.0, 0.0].into_boxed_slice();
    let fam = gaussian::GaussianShift {
        sigma_max: 0.3,
        d: 2,
    };
    let r1 = Engine::run(&base, &fam, &gaussian::Identity { d: 2 }, &gaussian::NegDispersion, &cfg)
        .unwrap();
    let r2 = Engine::run(&base, &fam, &gaussian::Identity { d: 2 }, &gaussian::NegDispersion, &cfg)
        .unwrap();
    assert_eq!(r1.value.to_bits(), r2.value.to_bits());
}

#[test]
fn d4_same_seed_same_value_markov() {
    let cfg = cfg_for(98765, 4_096);
    let base = markov::Label { i: 0 };
    let fam = markov::UniformMixing {
        k: 5,
        theta_max: 0.3,
    };
    let r1 = Engine::run(
        &base,
        &fam,
        &markov::BaseIndicator { base_label: 0 },
        &markov::Survival,
        &cfg,
    )
    .unwrap();
    let r2 = Engine::run(
        &base,
        &fam,
        &markov::BaseIndicator { base_label: 0 },
        &markov::Survival,
        &cfg,
    )
    .unwrap();
    assert_eq!(r1.value.to_bits(), r2.value.to_bits());
}

// ---------------------------------------------------------------------
// D4: byte-for-byte equality of the serialised Report.
// ---------------------------------------------------------------------

#[test]
fn d4_byte_for_byte_report() {
    let cfg = cfg_for(20260610, 2_048);
    let base: Vector = vec![1.0].into_boxed_slice();
    let fam = gaussian::GaussianShift {
        sigma_max: 0.2,
        d: 1,
    };
    let r1 = Engine::run(&base, &fam, &gaussian::Identity { d: 1 }, &gaussian::NegDispersion, &cfg)
        .unwrap();
    let r2 = Engine::run(&base, &fam, &gaussian::Identity { d: 1 }, &gaussian::NegDispersion, &cfg)
        .unwrap();
    let j1 = r1.to_json().unwrap();
    let j2 = r2.to_json().unwrap();
    assert_eq!(j1.as_bytes(), j2.as_bytes());
}

// ---------------------------------------------------------------------
// D1: changing the seed changes the value (otherwise the seed isn't
// total).
// ---------------------------------------------------------------------

#[test]
fn d1_different_seed_changes_value() {
    let base: Vector = vec![0.0].into_boxed_slice();
    let fam = gaussian::GaussianShift {
        sigma_max: 0.3,
        d: 1,
    };
    let r_a = Engine::run(
        &base,
        &fam,
        &gaussian::Identity { d: 1 },
        &gaussian::NegDispersion,
        &cfg_for(1, 4_096),
    )
    .unwrap();
    let r_b = Engine::run(
        &base,
        &fam,
        &gaussian::Identity { d: 1 },
        &gaussian::NegDispersion,
        &cfg_for(2, 4_096),
    )
    .unwrap();
    assert_ne!(r_a.value.to_bits(), r_b.value.to_bits());
}

// ---------------------------------------------------------------------
// D2: per-index substreams are pairwise independent of each other.
// Equivalently: fork(seed, i) != fork(seed, j) as RNG streams.
// ---------------------------------------------------------------------

#[test]
fn d2_per_index_substreams_distinct() {
    use rand::Rng as _;
    let seed = 0xCAFEBABE;
    let mut a = fork_rng(seed, 0);
    let mut b = fork_rng(seed, 1);
    let xa: u64 = a.gen();
    let xb: u64 = b.gen();
    assert_ne!(xa, xb);
}

#[test]
fn d2_per_index_substream_reproducible() {
    use rand::Rng as _;
    let seed = 0xCAFEBABE;
    let mut a1 = fork_rng(seed, 17);
    let mut a2 = fork_rng(seed, 17);
    let xa1: u64 = a1.gen();
    let xa2: u64 = a2.gen();
    assert_eq!(xa1, xa2);
}
