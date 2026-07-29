//! Cross-backend and cross-entry-point equivalence.
//!
//! The crate offers four ways to compute the same estimator: the
//! generic trait engine (threaded or sequential), and the built-in
//! [`Family`] path (host or device). The host ones must all agree bit
//! for bit; a user choosing a backend for speed must never be choosing
//! a different number.
//!
//! These tests deliberately straddle [`PARALLEL_MIN`], because that is
//! where the engine switches from a sequential loop to a thread pool
//! and is the one place a scheduling-dependent bug could hide.

use perturbation_kernel::config::{Backend, Config, Intensity, Lipschitz, Reduction};
use perturbation_kernel::engine::{Engine, PARALLEL_MIN};
use perturbation_kernel::examples::{bistable, gaussian, markov, Vector};
use perturbation_kernel::family::Family;
use perturbation_kernel::Error;
use serde_json::json;

fn cfg(n: u64, seed: u64, backend: Backend) -> Config {
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
        backend,
    }
}

/// Sizes chosen to bracket the sequential/threaded switch and to
/// exercise odd tails in the reduction tree.
fn sizes() -> Vec<u64> {
    vec![
        1,
        2,
        3,
        7,
        PARALLEL_MIN - 1,
        PARALLEL_MIN,
        PARALLEL_MIN + 1,
        PARALLEL_MIN * 4 + 3,
    ]
}

fn run_gaussian(n: u64, seed: u64, backend: Backend) -> f64 {
    let base: Vector = vec![0.5, -1.25, 3.0].into_boxed_slice();
    Engine::run(
        &base,
        &gaussian::GaussianShift {
            sigma_max: 0.3,
            d: 3,
        },
        &gaussian::Identity { d: 3 },
        &gaussian::NegDispersion,
        &cfg(n, seed, backend),
    )
    .unwrap()
    .value
}

fn run_bistable(n: u64, seed: u64, backend: Backend) -> f64 {
    Engine::run(
        &bistable::Marble { x: 0.9 },
        &bistable::Langevin {
            dt: 0.01,
            theta_max: 0.5,
        },
        &bistable::WellOccupancy,
        &bistable::Polarisation,
        &cfg(n, seed, backend),
    )
    .unwrap()
    .value
}

fn run_markov(n: u64, seed: u64, backend: Backend) -> f64 {
    Engine::run(
        &markov::Label { i: 0 },
        &markov::UniformMixing {
            k: 5,
            theta_max: 0.3,
        },
        &markov::BaseIndicator { base_label: 0 },
        &markov::Survival,
        &cfg(n, seed, backend),
    )
    .unwrap()
    .value
}

// ---------------------------------------------------------------------
// Backend::Scalar == Backend::Simd == Backend::Auto.
// ---------------------------------------------------------------------

#[test]
fn host_backends_agree_bit_for_bit() {
    for n in sizes() {
        for seed in [1u64, 20260610] {
            for (name, f) in [
                ("gaussian", run_gaussian as fn(u64, u64, Backend) -> f64),
                ("bistable", run_bistable),
                ("markov", run_markov),
            ] {
                let scalar = f(n, seed, Backend::Scalar);
                let simd = f(n, seed, Backend::Simd);
                let auto = f(n, seed, Backend::Auto);
                assert_eq!(
                    scalar.to_bits(),
                    simd.to_bits(),
                    "{name}: scalar != simd at n = {n}, seed = {seed}"
                );
                assert_eq!(
                    scalar.to_bits(),
                    auto.to_bits(),
                    "{name}: scalar != auto at n = {n}, seed = {seed}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// Threading is invisible: the threaded run equals the sequential one.
// ---------------------------------------------------------------------

#[test]
fn threaded_run_equals_sequential_run() {
    let base: Vector = vec![0.5, -1.25, 3.0].into_boxed_slice();
    for n in sizes() {
        let c = cfg(n, 4242, Backend::Auto);
        let threaded = Engine::run(
            &base,
            &gaussian::GaussianShift {
                sigma_max: 0.3,
                d: 3,
            },
            &gaussian::Identity { d: 3 },
            &gaussian::NegDispersion,
            &c,
        )
        .unwrap();
        let sequential = Engine::run_sequential(
            &base,
            &gaussian::GaussianShift {
                sigma_max: 0.3,
                d: 3,
            },
            &gaussian::Identity { d: 3 },
            &gaussian::NegDispersion,
            &c,
        )
        .unwrap();
        assert_eq!(
            threaded.value.to_bits(),
            sequential.value.to_bits(),
            "threading changed the value at n = {n}"
        );
    }
}

/// Repeating a threaded run many times would expose any dependence on
/// how `rayon` happened to split the range on a given day.
#[test]
fn repeated_threaded_runs_are_stable() {
    let n = PARALLEL_MIN * 8 + 1;
    let first = run_gaussian(n, 777, Backend::Auto);
    for i in 0..32 {
        assert_eq!(
            first.to_bits(),
            run_gaussian(n, 777, Backend::Auto).to_bits(),
            "threaded run {i} disagreed with run 0"
        );
    }
}

// ---------------------------------------------------------------------
// Family::run == Engine::run on the corresponding trait impls.
// ---------------------------------------------------------------------

#[test]
fn family_path_matches_trait_path_bit_for_bit() {
    for n in sizes() {
        for seed in [1u64, 20260610] {
            for backend in [Backend::Scalar, Backend::Auto] {
                let c = cfg(n, seed, backend);

                let fam = Family::Gaussian {
                    base: vec![0.5, -1.25, 3.0],
                    sigma_max: 0.3,
                };
                assert_eq!(
                    fam.run(&c).unwrap().value.to_bits(),
                    run_gaussian(n, seed, backend).to_bits(),
                    "gaussian family != trait path at n = {n}, seed = {seed}, {backend:?}"
                );

                let fam = Family::Bistable {
                    x0: 0.9,
                    dt: 0.01,
                    theta_max: 0.5,
                };
                assert_eq!(
                    fam.run(&c).unwrap().value.to_bits(),
                    run_bistable(n, seed, backend).to_bits(),
                    "bistable family != trait path at n = {n}, seed = {seed}, {backend:?}"
                );

                let fam = Family::Markov {
                    k: 5,
                    start: 0,
                    base_label: 0,
                    theta_max: 0.3,
                };
                assert_eq!(
                    fam.run(&c).unwrap().value.to_bits(),
                    run_markov(n, seed, backend).to_bits(),
                    "markov family != trait path at n = {n}, seed = {seed}, {backend:?}"
                );
            }
        }
    }
}

#[test]
fn family_reports_carry_the_right_functional_tag() {
    let c = cfg(1024, 1, Backend::Auto);
    for (fam, tag) in [
        (
            Family::Gaussian {
                base: vec![1.0],
                sigma_max: 0.1,
            },
            "negative_dispersion",
        ),
        (
            Family::Bistable {
                x0: 0.5,
                dt: 0.01,
                theta_max: 0.2,
            },
            "polarisation",
        ),
        (
            Family::Markov {
                k: 3,
                start: 0,
                base_label: 0,
                theta_max: 0.2,
            },
            "tail_survival",
        ),
    ] {
        assert_eq!(fam.run(&c).unwrap().functional, tag);
    }
}

// ---------------------------------------------------------------------
// Provenance and error paths.
// ---------------------------------------------------------------------

#[test]
fn execution_block_records_the_backend_and_strips_cleanly() {
    let r = Family::Markov {
        k: 3,
        start: 0,
        base_label: 0,
        theta_max: 0.2,
    }
    .run(&cfg(PARALLEL_MIN * 2, 1, Backend::Auto))
    .unwrap();

    let exec = r.execution.as_ref().expect("host runs record provenance");
    assert_eq!(exec.backend, "auto");
    assert_eq!(exec.precision, "f64");
    assert!(exec.device.is_none());
    assert_eq!(exec.threaded, cfg!(feature = "parallel"));

    // The v1.0.0 payload has no `execution` key at all.
    let v1: serde_json::Value = serde_json::from_str(&r.to_json_v1().unwrap()).unwrap();
    assert!(v1.get("execution").is_none());
    let full: serde_json::Value = serde_json::from_str(&r.to_json().unwrap()).unwrap();
    assert!(full.get("execution").is_some());
}

#[test]
fn a_default_config_serialises_without_the_backend_key() {
    // Backend is additive to SCHEMA §5, so a default config must still
    // produce exactly the v1.0.0 wire form.
    let c = cfg(10, 1, Backend::Auto);
    let v: serde_json::Value = serde_json::from_str(&c.to_json().unwrap()).unwrap();
    assert!(v.get("backend").is_none());

    let c = cfg(10, 1, Backend::Simd);
    let v: serde_json::Value = serde_json::from_str(&c.to_json().unwrap()).unwrap();
    assert_eq!(v.get("backend").unwrap(), "simd");

    // And it round-trips.
    let back = Config::from_json(&c.to_json().unwrap()).unwrap();
    assert_eq!(back.backend, Backend::Simd);
}

#[test]
fn generic_engine_refuses_the_gpu_backend() {
    let base: Vector = vec![1.0].into_boxed_slice();
    let err = Engine::run(
        &base,
        &gaussian::GaussianShift {
            sigma_max: 0.1,
            d: 1,
        },
        &gaussian::Identity { d: 1 },
        &gaussian::NegDispersion,
        &cfg(64, 1, Backend::Gpu),
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::UnsupportedBackend { backend: "gpu", .. }),
        "expected UnsupportedBackend, got {err:?}"
    );
}

#[test]
fn families_reject_out_of_domain_hyperparameters() {
    let c = cfg(16, 1, Backend::Auto);
    let cases: Vec<Family> = vec![
        Family::Gaussian {
            base: vec![],
            sigma_max: 0.1,
        },
        Family::Gaussian {
            base: vec![1.0],
            sigma_max: -1.0,
        },
        Family::Gaussian {
            base: vec![f64::NAN],
            sigma_max: 0.1,
        },
        Family::Bistable {
            x0: 0.0,
            dt: 0.0,
            theta_max: 0.1,
        },
        Family::Markov {
            k: 0,
            start: 0,
            base_label: 0,
            theta_max: 0.1,
        },
        Family::Markov {
            k: 3,
            start: 5,
            base_label: 0,
            theta_max: 0.1,
        },
        Family::Markov {
            k: 3,
            start: 0,
            base_label: 0,
            theta_max: 1.5,
        },
    ];
    for fam in cases {
        let err = fam.run(&c).unwrap_err();
        assert!(
            matches!(err, Error::InvalidFamily(_)),
            "expected InvalidFamily for {fam:?}, got {err:?}"
        );
    }
}

#[test]
fn families_enforce_the_same_config_contract_as_the_engine() {
    let fam = Family::Markov {
        k: 3,
        start: 0,
        base_label: 0,
        theta_max: 0.2,
    };

    // n = 0 (SCHEMA §5).
    assert!(matches!(
        fam.run(&cfg(0, 1, Backend::Auto)).unwrap_err(),
        Error::EmptyEnsemble
    ));

    // Wrong major version (SCHEMA §10).
    let mut c = cfg(16, 1, Backend::Auto);
    c.schema_version = "2.0.0".into();
    assert!(matches!(
        fam.run(&c).unwrap_err(),
        Error::SchemaVersion { .. }
    ));

    // Null-parameter mismatch (SCHEMA §5, C2).
    let mut c = cfg(16, 1, Backend::Auto);
    c.intensity.null_parameter = json!(0.5);
    assert!(matches!(
        fam.run(&c).unwrap_err(),
        Error::NullParameterMismatch { .. }
    ));
}

#[test]
fn family_round_trips_through_json() {
    for fam in [
        Family::Gaussian {
            base: vec![1.0, 2.0],
            sigma_max: 0.3,
        },
        Family::Bistable {
            x0: 0.9,
            dt: 0.01,
            theta_max: 0.5,
        },
        Family::Markov {
            k: 5,
            start: 1,
            base_label: 0,
            theta_max: 0.3,
        },
    ] {
        let s = serde_json::to_string(&fam).unwrap();
        let back: Family = serde_json::from_str(&s).unwrap();
        assert_eq!(fam, back, "round trip changed {fam:?} (json: {s})");
    }
}
