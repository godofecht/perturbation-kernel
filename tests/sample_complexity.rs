//! Sample-complexity floor (SCHEMA §7; Paper Thm 7.3).
//!
//! Two empirical checks:
//!
//! 1. Variance across seeds shrinks as `N` grows (1/sqrt(N) rate).
//! 2. The Thm 7.3 epsilon, with the declared `Lambda` and `D`, bounds
//!    the observed deviation from the population mean with at least
//!    the asserted probability `1 - eta`.

use perturbation_kernel::config::{
    Accuracy, Config, Intensity, Lipschitz, Reduction,
};
use perturbation_kernel::engine::Engine;
use perturbation_kernel::examples::markov;
use serde_json::json;

fn run_with(seed: u64, n: u64, with_accuracy: bool) -> perturbation_kernel::report::Report {
    let cfg = Config {
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
        accuracy: if with_accuracy {
            Some(Accuracy {
                epsilon: 0.1,
                eta: 0.1,
                observation_diameter: 1.0, // indicator in {0,1}
                obs_dim: 1,
            })
        } else {
            None
        },
    };
    let base = markov::Label { i: 0 };
    let fam = markov::UniformMixing {
        k: 4,
        theta_max: 0.3,
    };
    Engine::run(
        &base,
        &fam,
        &markov::BaseIndicator { base_label: 0 },
        &markov::Survival,
        &cfg,
    )
    .unwrap()
}

fn variance_across_seeds(n: u64, trials: usize) -> f64 {
    let xs: Vec<f64> = (0..trials)
        .map(|t| run_with(0x100 + t as u64, n, false).value)
        .collect();
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64
}

#[test]
fn variance_decreases_with_n() {
    // Small N -> big variance, big N -> small variance. 1/N rate;
    // the ratio should be far above 1 with safety margin against
    // sampling noise.
    let v_small = variance_across_seeds(64, 16);
    let v_large = variance_across_seeds(4_096, 16);
    assert!(
        v_small > v_large,
        "v_small={} should exceed v_large={}",
        v_small,
        v_large
    );
    // With 64x more samples, variance should shrink by roughly 64x.
    // Demand at least 10x to absorb finite-trial noise.
    assert!(
        v_small / v_large > 10.0,
        "ratio={} too small",
        v_small / v_large
    );
}

#[test]
fn thm_7_3_bound_covers_observed_deviation() {
    // Estimate the population mean with a giant N, then check that
    // smaller-N runs land within the reported epsilon (Paper Thm
    // 7.3(c)) with at least the asserted probability.
    let pop = run_with(0xFFFF_FFFF, 400_000, false).value;
    let n = 100_000;
    let trials = 24;
    let mut covered = 0usize;
    let mut last_eps = 0.0_f64;
    for t in 0..trials {
        let r = run_with(t as u64, n, true);
        last_eps = r.error_bound.epsilon;
        if (r.value - pop).abs() <= r.error_bound.epsilon {
            covered += 1;
        }
    }
    // The asserted (eps=0.1, eta=0.1) bound promises coverage >= 90%.
    // Demand at least 0.75 to absorb finite-trial noise.
    let coverage = covered as f64 / trials as f64;
    assert!(
        coverage >= 0.75,
        "coverage={} < 0.75 (epsilon~{}, pop={})",
        coverage,
        last_eps,
        pop
    );
}

#[test]
fn sample_floor_formula_is_monotone_in_eps() {
    use perturbation_kernel::config::sample_floor;
    let big = sample_floor(1.0, 1.0, 0.01, 0.05, 1);
    let small = sample_floor(1.0, 1.0, 0.1, 0.05, 1);
    assert!(big > small, "tighter epsilon must require more samples");
}

#[test]
fn sample_floor_formula_is_monotone_in_eta() {
    use perturbation_kernel::config::sample_floor;
    // Use a regime (loose eps, high obs_dim) where the stochastic
    // McDiarmid term dominates the Fournier-Guillin bias term, so
    // tightening eta moves the floor (Paper Thm 7.3(a) carries the
    // eta dependence; Thm 7.3(b) does not).
    let strict = sample_floor(1.0, 10.0, 0.5, 0.001, 3);
    let loose = sample_floor(1.0, 10.0, 0.5, 0.5, 3);
    assert!(
        strict > loose,
        "stricter eta must require more samples: strict={}, loose={}",
        strict,
        loose
    );
}
