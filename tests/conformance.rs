//! Conformance test suite: the SCHEMA §3 C1-C6 / §11 informative checklist.
//!
//! These tests target the published examples in
//! [`perturbation_kernel::examples`] because they're the surface the
//! schema actually constrains; the checklist is a chain of MUST
//! clauses on the engine + components.

use perturbation_kernel::config::{
    Accuracy, Config, Intensity, Lipschitz, Reduction,
};
use perturbation_kernel::engine::Engine;
use perturbation_kernel::examples::{bistable, gaussian, markov, Vector};
use perturbation_kernel::forward::ForwardModel;
use perturbation_kernel::invariance::Invariance;
use perturbation_kernel::Rng;
use proptest::prelude::*;
use rand::SeedableRng;
use serde_json::json;

// ---------------------------------------------------------------------
// C1: typed spaces, metric d_O properties (SCHEMA §3 C1).
// ---------------------------------------------------------------------

/// Euclidean distance: the canonical metric used by the Gaussian
/// example.
fn d_euclid(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

proptest! {
    /// Identity: d(x, x) == 0.
    #[test]
    fn c1_metric_identity(x in proptest::collection::vec(-10.0f64..10.0, 3)) {
        prop_assert!(d_euclid(&x, &x).abs() < 1e-12);
    }

    /// Symmetry: d(x, y) == d(y, x).
    #[test]
    fn c1_metric_symmetry(
        x in proptest::collection::vec(-10.0f64..10.0, 3),
        y in proptest::collection::vec(-10.0f64..10.0, 3),
    ) {
        prop_assert!((d_euclid(&x, &y) - d_euclid(&y, &x)).abs() < 1e-9);
    }

    /// Triangle: d(x,z) <= d(x,y) + d(y,z) up to fp tolerance.
    #[test]
    fn c1_metric_triangle(
        x in proptest::collection::vec(-5.0f64..5.0, 3),
        y in proptest::collection::vec(-5.0f64..5.0, 3),
        z in proptest::collection::vec(-5.0f64..5.0, 3),
    ) {
        let xz = d_euclid(&x, &z);
        let xy = d_euclid(&x, &y);
        let yz = d_euclid(&y, &z);
        prop_assert!(xz <= xy + yz + 1e-9);
    }
}

// ---------------------------------------------------------------------
// C2: identity recovery (SCHEMA §3 C2, §11 item 1).
// With intensity collapsed to null_parameter, value MUST equal
// Phi(delta_{F(base)}) to fp tolerance.
// ---------------------------------------------------------------------

fn cfg_gaussian(n: u64, low: f64, high: f64) -> Config {
    Config {
        schema_version: "1.0.0".into(),
        n,
        seed: 0xDEADBEEF,
        intensity: Intensity {
            kind: "uniform_interval".into(),
            params: json!({ "low": low, "high": high }),
            null_parameter: json!(0.0),
        },
        reduction: Reduction::default(),
        lipschitz: Lipschitz {
            forward_l: Some(1.0),
            invariance_lambda: None,
        },
        accuracy: None,
    }
}

#[test]
fn c2_identity_recovery_gaussian() {
    // Collapse rho to {0}: high == 0.
    let cfg = cfg_gaussian(2_000, 0.0, 0.0);
    let base: Vector = vec![1.0, -2.0, 3.0].into_boxed_slice();
    let fam = gaussian::GaussianShift {
        sigma_max: 0.0,
        d: 3,
    };
    let model = gaussian::Identity { d: 3 };
    let inv = gaussian::NegDispersion;
    let report = Engine::run(&base, &fam, &model, &inv, &cfg).unwrap();
    // With sigma == 0 every S_i == base; F is identity; variance is 0.
    assert!(report.value.abs() < 1e-12, "value={}", report.value);
    assert_eq!(report.functional, "negative_dispersion");
    assert_eq!(report.n_effective, 2_000);
}

#[test]
fn c2_identity_recovery_bistable() {
    let cfg = Config {
        schema_version: "1.0.0".into(),
        n: 500,
        seed: 1,
        intensity: Intensity {
            kind: "uniform_interval".into(),
            params: json!({ "low": 0.0, "high": 0.0 }),
            null_parameter: json!(0.0),
        },
        reduction: Reduction::default(),
        lipschitz: Lipschitz::default(),
        accuracy: None,
    };
    // Start the marble at the +1 well; with theta == 0 the
    // deterministic drift takes it back to +1 every step.
    let base = bistable::Marble { x: 1.0 };
    let fam = bistable::Langevin {
        dt: 0.01,
        theta_max: 0.0,
    };
    let model = bistable::WellOccupancy;
    let inv = bistable::Polarisation;
    let report = Engine::run(&base, &fam, &model, &inv, &cfg).unwrap();
    assert!((report.value - 1.0).abs() < 1e-12, "value={}", report.value);
}

#[test]
fn c2_identity_recovery_markov() {
    let cfg = Config {
        schema_version: "1.0.0".into(),
        n: 1_000,
        seed: 42,
        intensity: Intensity {
            kind: "uniform_interval".into(),
            params: json!({ "low": 0.0, "high": 0.0 }),
            null_parameter: json!(0.0),
        },
        reduction: Reduction::default(),
        lipschitz: Lipschitz::default(),
        accuracy: None,
    };
    let base = markov::Label { i: 2 };
    let fam = markov::UniformMixing {
        k: 5,
        theta_max: 0.0,
    };
    let model = markov::BaseIndicator { base_label: 2 };
    let inv = markov::Survival;
    let report = Engine::run(&base, &fam, &model, &inv, &cfg).unwrap();
    // theta == 0 means we never mix; the indicator is 1 every time.
    assert!((report.value - 1.0).abs() < 1e-12, "value={}", report.value);
}

#[test]
fn c2_null_parameter_mismatch_rejected() {
    // Same family, but lie about the null in the config.
    let mut cfg = cfg_gaussian(10, 0.0, 0.0);
    cfg.intensity.null_parameter = json!(0.25);
    let base: Vector = vec![0.0].into_boxed_slice();
    let fam = gaussian::GaussianShift {
        sigma_max: 0.0,
        d: 1,
    };
    let res = Engine::run(&base, &fam, &gaussian::Identity { d: 1 }, &gaussian::NegDispersion, &cfg);
    assert!(matches!(res, Err(perturbation_kernel::Error::NullParameterMismatch { .. })));
}

// ---------------------------------------------------------------------
// C3 / C4: declared Lipschitz constants surface into error_bound
// (SCHEMA §3 C3, C4, §6 row error_bound).
// ---------------------------------------------------------------------

#[test]
fn c3_c4_forward_and_invariance_lipschitz_declared() {
    let fam = gaussian::GaussianShift {
        sigma_max: 0.1,
        d: 1,
    };
    let model = gaussian::Identity { d: 1 };
    // Identity Lipschitz is 1.
    assert_eq!(model.lipschitz(), Some(1.0));
    // Survival example: Lambda = 1.
    assert_eq!(markov::Survival.lipschitz_w1(), Some(1.0));
    // NegDispersion not declared globally Lipschitz on R.
    assert_eq!(gaussian::NegDispersion.lipschitz_w1(), None);
    // Drop unused.
    let _ = fam;
}

// ---------------------------------------------------------------------
// C5 / C6: engine wiring + accuracy claim activates the error bound.
// ---------------------------------------------------------------------

#[test]
fn c5_c6_engine_with_accuracy_emits_error_bound() {
    let cfg = Config {
        schema_version: "1.0.0".into(),
        n: 400_000,
        seed: 7,
        intensity: Intensity {
            kind: "uniform_interval".into(),
            params: json!({ "low": 0.0, "high": 0.2 }),
            null_parameter: json!(0.0),
        },
        reduction: Reduction::default(),
        lipschitz: Lipschitz {
            forward_l: Some(1.0),
            invariance_lambda: Some(1.0),
        },
        accuracy: Some(Accuracy {
            epsilon: 0.05,
            eta: 0.05,
            observation_diameter: 2.0,
            obs_dim: 1,
        }),
    };
    let base = markov::Label { i: 0 };
    let fam = markov::UniformMixing {
        k: 4,
        theta_max: 0.2,
    };
    let model = markov::BaseIndicator { base_label: 0 };
    let inv = markov::Survival;
    let report = Engine::run(&base, &fam, &model, &inv, &cfg).unwrap();
    assert!(report.error_bound.available);
    assert!(report.error_bound.epsilon > 0.0);
    assert!(report.error_bound.epsilon < 1.0);
    assert!(report.stability_modulus.is_some());
    // 0 <= value <= 1 for an indicator average.
    assert!(report.value >= 0.0 && report.value <= 1.0);
}

// ---------------------------------------------------------------------
// SCHEMA §11 item 2: order invariance of Invariance::measure.
// ---------------------------------------------------------------------

#[test]
fn s11_order_invariance_of_measure() {
    let mut rng = Rng::from_seed([3u8; 32]);
    use rand::Rng as _;
    let ys: Vec<f64> = (0..256).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let inv = markov::Survival;
    let r1 = inv.measure(&ys);
    let mut shuffled = ys.clone();
    // Reverse and rotate -- two distinct permutations.
    shuffled.reverse();
    let r2 = inv.measure(&shuffled);
    shuffled.rotate_left(7);
    let r3 = inv.measure(&shuffled);
    // Mean is permutation-invariant up to last-bit tree-reduction
    // wobble; bound at 1e-12 is conservative for tree_sum at this N.
    assert!((r1.value - r2.value).abs() < 1e-12);
    assert!((r1.value - r3.value).abs() < 1e-12);
}

// ---------------------------------------------------------------------
// SCHEMA §11 item 5: dishonest accuracy claim must be rejected.
// ---------------------------------------------------------------------

#[test]
fn s11_sample_floor_rejects_underpowered_accuracy_claim() {
    // Tiny n with tight epsilon/eta -> floor violated.
    let cfg = Config {
        schema_version: "1.0.0".into(),
        n: 10,
        seed: 1,
        intensity: Intensity {
            kind: "uniform_interval".into(),
            params: json!({ "low": 0.0, "high": 0.1 }),
            null_parameter: json!(0.0),
        },
        reduction: Reduction::default(),
        lipschitz: Lipschitz {
            forward_l: Some(1.0),
            invariance_lambda: Some(1.0),
        },
        accuracy: Some(Accuracy {
            epsilon: 0.001,
            eta: 0.001,
            observation_diameter: 2.0,
            obs_dim: 1,
        }),
    };
    let base = markov::Label { i: 0 };
    let fam = markov::UniformMixing {
        k: 4,
        theta_max: 0.1,
    };
    let res = Engine::run(
        &base,
        &fam,
        &markov::BaseIndicator { base_label: 0 },
        &markov::Survival,
        &cfg,
    );
    assert!(matches!(
        res,
        Err(perturbation_kernel::Error::SampleFloor { .. })
    ));
}

// ---------------------------------------------------------------------
// SCHEMA §10 versioning: wrong major rejected.
// ---------------------------------------------------------------------

#[test]
fn s10_major_version_rejected() {
    let mut cfg = cfg_gaussian(10, 0.0, 0.0);
    cfg.schema_version = "2.0.0".into();
    let base: Vector = vec![0.0].into_boxed_slice();
    let res = Engine::run(
        &base,
        &gaussian::GaussianShift {
            sigma_max: 0.0,
            d: 1,
        },
        &gaussian::Identity { d: 1 },
        &gaussian::NegDispersion,
        &cfg,
    );
    assert!(matches!(
        res,
        Err(perturbation_kernel::Error::SchemaVersion { .. })
    ));
}
