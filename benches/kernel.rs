//! Throughput benchmarks.
//!
//! Every group compares an optimised path against the v1.0.0 reference
//! path computing the same number, so the speedups are measured rather
//! than asserted:
//!
//! * `reduce/*` compares the vectorised reductions against
//!   [`perturbation_kernel::reduce::reference`], which is the literal
//!   v1.0.0 code.
//! * `engine/*` compares [`Backend::Auto`] (threaded draws, vectorised
//!   reduction) against [`Backend::Scalar`] (one thread, scalar loops).
//! * `family/*` compares the flat-storage
//!   [`perturbation_kernel::family::Family`] path against the generic
//!   trait path, which must allocate one observation per draw.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use perturbation_kernel::config::{Backend, Config, Intensity, Lipschitz, Reduction};
use perturbation_kernel::engine::Engine;
use perturbation_kernel::examples::{bistable, gaussian, markov, Vector};
use perturbation_kernel::family::Family;
use perturbation_kernel::reduce::{self, reference, SimdPath};
use serde_json::json;
use std::hint::black_box;

fn cfg(n: u64, backend: Backend) -> Config {
    Config {
        schema_version: "1.0.0".into(),
        n,
        seed: 20260610,
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

/// Deterministic pseudo-data, independent of the crate's RNG so the
/// reduction benches measure only the reduction.
fn data(n: usize) -> Vec<f64> {
    let mut z = 0x243F_6A88_85A3_08D3_u64;
    (0..n)
        .map(|_| {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64) - 0.5
        })
        .collect()
}

fn bench_reductions(c: &mut Criterion) {
    let mut g = c.benchmark_group("reduce");
    for &n in &[1_024usize, 65_536, 1_048_576] {
        let xs = data(n);
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(
            BenchmarkId::new("tree_sum/v1_reference", n),
            &xs,
            |b, xs| b.iter(|| black_box(reference::tree_sum(black_box(xs)))),
        );
        g.bench_with_input(BenchmarkId::new("tree_sum/scalar", n), &xs, |b, xs| {
            b.iter(|| black_box(reference::tree_sum_on(SimdPath::Scalar, black_box(xs))))
        });
        g.bench_with_input(BenchmarkId::new("tree_sum/simd", n), &xs, |b, xs| {
            b.iter(|| black_box(reduce::tree_sum(black_box(xs))))
        });

        g.bench_with_input(
            BenchmarkId::new("sum_sq_dev/v1_reference", n),
            &xs,
            |b, xs| b.iter(|| black_box(reference::sum_sq_dev(black_box(xs), 0.25))),
        );
        g.bench_with_input(BenchmarkId::new("sum_sq_dev/simd", n), &xs, |b, xs| {
            b.iter(|| black_box(reduce::sum_sq_dev(black_box(xs), 0.25)))
        });
    }
    g.finish();
}

fn bench_engine(c: &mut Criterion) {
    let mut g = c.benchmark_group("engine");
    g.sample_size(20);
    for &n in &[16_384u64, 262_144] {
        g.throughput(Throughput::Elements(n));
        for (tag, backend) in [("scalar", Backend::Scalar), ("auto", Backend::Auto)] {
            let cfg = cfg(n, backend);

            let base: Vector = vec![0.5, -1.25, 3.0].into_boxed_slice();
            g.bench_with_input(
                BenchmarkId::new(format!("gaussian_d3/{tag}"), n),
                &cfg,
                |b, cfg| {
                    b.iter(|| {
                        Engine::run(
                            black_box(&base),
                            &gaussian::GaussianShift {
                                sigma_max: 0.3,
                                d: 3,
                            },
                            &gaussian::Identity { d: 3 },
                            &gaussian::NegDispersion,
                            cfg,
                        )
                        .unwrap()
                    })
                },
            );

            g.bench_with_input(
                BenchmarkId::new(format!("bistable/{tag}"), n),
                &cfg,
                |b, cfg| {
                    b.iter(|| {
                        Engine::run(
                            &bistable::Marble { x: 0.9 },
                            &bistable::Langevin {
                                dt: 0.01,
                                theta_max: 0.5,
                            },
                            &bistable::WellOccupancy,
                            &bistable::Polarisation,
                            cfg,
                        )
                        .unwrap()
                    })
                },
            );

            g.bench_with_input(
                BenchmarkId::new(format!("markov/{tag}"), n),
                &cfg,
                |b, cfg| {
                    b.iter(|| {
                        Engine::run(
                            &markov::Label { i: 0 },
                            &markov::UniformMixing {
                                k: 5,
                                theta_max: 0.3,
                            },
                            &markov::BaseIndicator { base_label: 0 },
                            &markov::Survival,
                            cfg,
                        )
                        .unwrap()
                    })
                },
            );
        }
    }
    g.finish();
}

fn bench_family(c: &mut Criterion) {
    let mut g = c.benchmark_group("family");
    g.sample_size(20);
    let fam = Family::Gaussian {
        base: vec![0.5, -1.25, 3.0],
        sigma_max: 0.3,
    };
    {
        let n = 262_144u64;
        g.throughput(Throughput::Elements(n));
        let cfg = cfg(n, Backend::Auto);
        g.bench_with_input(BenchmarkId::new("gaussian_d3/trait", n), &cfg, |b, cfg| {
            let base: Vector = vec![0.5, -1.25, 3.0].into_boxed_slice();
            b.iter(|| {
                Engine::run(
                    &base,
                    &gaussian::GaussianShift {
                        sigma_max: 0.3,
                        d: 3,
                    },
                    &gaussian::Identity { d: 3 },
                    &gaussian::NegDispersion,
                    cfg,
                )
                .unwrap()
            })
        });
        g.bench_with_input(BenchmarkId::new("gaussian_d3/family", n), &cfg, |b, cfg| {
            b.iter(|| fam.run(cfg).unwrap())
        });
    }
    g.finish();
}

criterion_group!(benches, bench_reductions, bench_engine, bench_family);
criterion_main!(benches);
