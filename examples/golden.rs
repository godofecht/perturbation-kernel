//! Dumps Report::value bit patterns for a fixed matrix of configs.
//! Used to prove the optimised paths stay bit-identical to v1.0.0.
use perturbation_kernel::config::{Config, Intensity, Lipschitz, Reduction};
use perturbation_kernel::engine::Engine;
use perturbation_kernel::examples::{bistable, gaussian, markov, Vector};
use serde_json::json;

fn cfg(seed: u64, n: u64) -> Config {
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
        backend: Default::default(),
    }
}

fn main() {
    for &n in &[1u64, 2, 3, 7, 1000, 4096, 65537] {
        for &seed in &[1u64, 20260610] {
            let c = cfg(seed, n);

            let base: Vector = vec![0.5, -1.25, 3.0].into_boxed_slice();
            let g = Engine::run(
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

            let b = Engine::run(
                &bistable::Marble { x: 0.9 },
                &bistable::Langevin {
                    dt: 0.01,
                    theta_max: 0.5,
                },
                &bistable::WellOccupancy,
                &bistable::Polarisation,
                &c,
            )
            .unwrap();

            let m = Engine::run(
                &markov::Label { i: 0 },
                &markov::UniformMixing {
                    k: 5,
                    theta_max: 0.3,
                },
                &markov::BaseIndicator { base_label: 0 },
                &markov::Survival,
                &c,
            )
            .unwrap();

            println!("{} {} gaussian {:016x}", n, seed, g.value.to_bits());
            println!("{} {} bistable {:016x}", n, seed, b.value.to_bits());
            println!("{} {} markov   {:016x}", n, seed, m.value.to_bits());
        }
    }
}
