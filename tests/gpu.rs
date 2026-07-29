//! GPU backend tests.
//!
//! There are two device backends and they make different promises, so
//! they get different tests.
//!
//! [`Backend::Gpu`] claims **bit-identity with the host**. That is
//! tested the only way such a claim can honestly be tested: by
//! comparing `to_bits()` across a spread of parameters, ensemble sizes
//! and seeds, and by pinning each layer underneath it separately (the
//! keystream, the emulated-`f64` uniform, the host-side parameter
//! derivation) so a failure says which layer broke.
//!
//! [`Backend::GpuF32`] claims only statistical agreement, and is tested
//! against the Monte Carlo error rather than against the bits.
//!
//! Every test skips, loudly, when no adapter is available. Set
//! `PK_REQUIRE_GPU=1` to turn the skip into a failure; CI does that on
//! the runner where a software Vulkan device is installed on purpose,
//! so a broken shader cannot pass as a skipped suite.

#![cfg(feature = "gpu")]

use perturbation_kernel::config::{Backend, Config, Intensity, Lipschitz, Reduction};
use perturbation_kernel::engine::fork_rng;
use perturbation_kernel::family::Family;
use perturbation_kernel::gpu;
use perturbation_kernel::Error;
use serde_json::json;

macro_rules! device_or_skip {
    () => {
        match gpu::context() {
            Ok(c) => c,
            Err(e) => {
                if std::env::var_os("PK_REQUIRE_GPU").is_some() {
                    panic!("PK_REQUIRE_GPU is set but no compute device was found: {e}");
                }
                eprintln!("SKIP: no compute device available ({e})");
                return;
            }
        }
    };
}

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

fn markov(k: u32, start: u32, base_label: u32, theta_max: f64) -> Family {
    Family::Markov {
        k,
        start,
        base_label,
        theta_max,
    }
}

// =====================================================================
// Backend::Gpu -- bit-identical to the host
// =====================================================================

/// The headline claim, tested across the parameter space rather than at
/// one convenient point.
#[test]
fn gpu_markov_is_bit_identical_to_the_host() {
    let _ctx = device_or_skip!();

    // Alphabet sizes chosen to cover the shapes that change control
    // flow: powers of two (which give the widest rejection zone in
    // rand's integer sampler), primes, k = 1 where every draw survives,
    // and a large k. Intensities cover the null perturbation, the full
    // range, and awkward mantissas.
    let families = [
        markov(5, 0, 0, 0.3),
        markov(2, 0, 1, 0.5),
        markov(4, 1, 0, 1.0),
        markov(17, 3, 3, 0.85),
        markov(256, 7, 7, 0.123_456_789),
        markov(3, 2, 1, 0.999),
        markov(1, 0, 0, 1.0),
        markov(1_000_003, 5, 5, 0.77),
        markov(5, 0, 0, 0.0),
    ];
    // Sizes that straddle the workgroup size (64) and the host's
    // sequential/threaded switch (4096).
    let sizes = [
        1u64, 2, 63, 64, 65, 127, 1_000, 4_095, 4_096, 4_097, 100_000,
    ];

    let mut checked = 0usize;
    for fam in &families {
        for &n in &sizes {
            for &seed in &[1u64, 20_260_610, 0] {
                let host = fam.run(&cfg(n, seed, Backend::Scalar)).unwrap().value;
                let dev = fam.run(&cfg(n, seed, Backend::Gpu)).unwrap().value;
                assert_eq!(
                    host.to_bits(),
                    dev.to_bits(),
                    "{fam:?} at n = {n}, seed = {seed}: host {host:.17} ({:016x}) \
                     but device {dev:.17} ({:016x})",
                    host.to_bits(),
                    dev.to_bits(),
                );
                checked += 1;
            }
        }
    }
    eprintln!("{checked} host/device pairs agreed bit for bit");
}

/// The device must agree with `Backend::Auto` too, not just the
/// reference path. This is trivially implied by the host backends
/// agreeing with each other, and asserting it directly means a
/// regression in either place is caught here.
#[test]
fn gpu_markov_agrees_with_the_threaded_host_path() {
    let _ctx = device_or_skip!();
    let fam = markov(7, 0, 0, 0.45);
    for &n in &[4_096u64, 50_000, 262_144] {
        let auto = fam.run(&cfg(n, 99, Backend::Auto)).unwrap().value;
        let dev = fam.run(&cfg(n, 99, Backend::Gpu)).unwrap().value;
        assert_eq!(auto.to_bits(), dev.to_bits(), "n = {n}");
    }
}

/// Layer 1: the device reads the same ChaCha20 keystream as the host.
///
/// If this fails, nothing above it can be trusted, so it is worth
/// asserting on its own rather than only through a derived value.
#[test]
fn device_keystream_matches_the_host_word_for_word() {
    use rand::RngCore;
    let ctx = device_or_skip!();
    let n = 512u32;
    let dev = ctx.debug_stream_words(7, n, 0.3).unwrap();
    for i in 0..n as u64 {
        let mut r = fork_rng(7, i);
        let host: [u32; 8] = std::array::from_fn(|_| r.next_u32());
        assert_eq!(
            host, dev[i as usize],
            "keystream diverged at draw {i}: host {host:08x?}, device {:08x?}",
            dev[i as usize]
        );
    }
}

/// Layer 2: the host-side transcription of `rand`'s `UniformFloat`
/// construction produces the same samples as `rand` itself.
///
/// The fields of `rand`'s `UniformFloat` are private, so the device is
/// handed a `(low, scale)` pair this crate derives independently. This
/// pins that derivation against the real thing.
#[test]
fn uniform_parameter_derivation_matches_rand() {
    use rand::distributions::{Distribution, Uniform};
    use rand::Rng as _;

    for &theta_max in &[0.0f64, 0.3, 0.5, 0.85, 1.0, 0.123_456_789, 1e-9] {
        let (low, scale) = gpu::uniform_inclusive_params(0.0, theta_max);
        let dist = Uniform::new_inclusive(0.0, theta_max);
        for i in 0..5_000u64 {
            let mut a = fork_rng(7, i);
            let want = dist.sample(&mut a);

            let mut b = fork_rng(7, i);
            let bits: u64 = b.gen();
            // rand's UniformFloat<f64>::sample, written out.
            let value1_2 = f64::from_bits((bits >> 12) | (1023u64 << 52));
            let got = (value1_2 - 1.0) * scale + low;

            assert_eq!(
                want.to_bits(),
                got.to_bits(),
                "theta_max = {theta_max}, draw {i}: rand gave {want:.17}, \
                 the transcription gave {got:.17}"
            );
        }
    }
}

#[test]
fn gpu_refuses_families_it_cannot_compute_exactly() {
    let _ctx = device_or_skip!();
    for fam in [
        Family::Gaussian {
            base: vec![0.5, -1.0],
            sigma_max: 0.3,
        },
        Family::Bistable {
            x0: 0.0,
            dt: 0.01,
            theta_max: 0.5,
        },
    ] {
        let err = fam.run(&cfg(1_024, 1, Backend::Gpu)).unwrap_err();
        match &err {
            Error::UnsupportedBackend { backend, reason } => {
                assert_eq!(*backend, "gpu");
                // The message has to name the way out, or it is just a
                // refusal.
                assert!(
                    reason.contains("gpu_f32"),
                    "the error should point at the approximate backend: {reason}"
                );
            }
            other => panic!("expected UnsupportedBackend, got {other:?}"),
        }
    }
}

#[test]
fn gpu_reports_declare_double_precision() {
    let ctx = device_or_skip!();
    let r = markov(4, 0, 0, 0.25)
        .run(&cfg(4_096, 7, Backend::Gpu))
        .unwrap();
    let exec = r.execution.as_ref().expect("device runs record provenance");
    assert_eq!(exec.backend, "gpu");
    // The exact path really is double precision: the device returns an
    // integer count and the host divides in f64.
    assert_eq!(exec.precision, "f64");
    assert_eq!(exec.device.as_deref(), Some(ctx.name.as_str()));
    eprintln!("device: {}", ctx.name);
}

// =====================================================================
// Backend::GpuF32 -- statistical agreement only
// =====================================================================

fn all_families() -> Vec<Family> {
    vec![
        Family::Gaussian {
            base: vec![0.5, -1.25, 3.0],
            sigma_max: 0.3,
        },
        // On the ridge between the wells, so the perturbation actually
        // decides the readout. Deep inside a well the drift wins and
        // polarisation is identically 1, which would make the
        // host/device comparison vacuous.
        Family::Bistable {
            x0: 0.0,
            dt: 0.01,
            theta_max: 0.5,
        },
        markov(5, 0, 0, 0.3),
    ]
}

#[test]
fn gpu_f32_runs_are_bit_reproducible() {
    let _ctx = device_or_skip!();
    for fam in all_families() {
        let c = cfg(50_000, 20_260_610, Backend::GpuF32);
        let first = fam.run(&c).unwrap().value;
        for i in 1..5 {
            let again = fam.run(&c).unwrap().value;
            assert_eq!(
                first.to_bits(),
                again.to_bits(),
                "{}: device run {i} disagreed with run 0 ({first} vs {again})",
                fam.name()
            );
        }
    }
}

#[test]
fn gpu_f32_seed_is_total() {
    let _ctx = device_or_skip!();
    let fam = markov(5, 0, 0, 0.3);
    let a = fam.run(&cfg(50_000, 1, Backend::GpuF32)).unwrap().value;
    let b = fam.run(&cfg(50_000, 2, Backend::GpuF32)).unwrap().value;
    assert_ne!(a, b, "the seed did not reach the device");
}

/// Single precision plus Box-Muller means these are two estimates of
/// the same quantity, not one number computed twice. The tolerance is a
/// generous multiple of the Monte Carlo standard error at this `n`, not
/// a value tuned to the observed gap.
#[test]
fn gpu_f32_agrees_with_host_within_monte_carlo_error() {
    let _ctx = device_or_skip!();
    let n = 200_000u64;
    let tol = 10.0 / (n as f64).sqrt();

    for fam in all_families() {
        let host = fam.run(&cfg(n, 20_260_610, Backend::Auto)).unwrap().value;
        let dev = fam.run(&cfg(n, 20_260_610, Backend::GpuF32)).unwrap().value;
        let gap = (host - dev).abs();
        eprintln!(
            "{:>9}: host {host:+.9}  device {dev:+.9}  |gap| {gap:.3e}  tol {tol:.3e}",
            fam.name()
        );
        assert!(
            gap < tol,
            "{}: device {dev} and host {host} differ by {gap}, beyond the \
             Monte Carlo tolerance {tol}",
            fam.name()
        );
    }
}

#[test]
fn gpu_f32_reports_declare_single_precision() {
    let _ctx = device_or_skip!();
    let r = markov(4, 0, 0, 0.25)
        .run(&cfg(4_096, 7, Backend::GpuF32))
        .unwrap();
    let exec = r.execution.as_ref().unwrap();
    assert_eq!(exec.backend, "gpu_f32");
    assert_eq!(exec.precision, "f32");
}

// =====================================================================
// The device reduction, which is exactly specified in either mode
// =====================================================================

/// The host reduction tree, run in single precision. This is the exact
/// value the device kernel must produce.
fn host_f32_tree(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut buf = xs.to_vec();
    let mut len = buf.len();
    while len > 1 {
        let half = len / 2;
        for k in 0..half {
            buf[k] = buf[2 * k] + buf[2 * k + 1];
        }
        if len & 1 == 1 {
            buf[half] = buf[len - 1];
            len = half + 1;
        } else {
            len = half;
        }
    }
    buf[0]
}

fn data_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut z = seed | 1;
    (0..n)
        .map(|i| {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            let m = (z >> 40) as f32 * (1.0 / 16_777_216.0) - 0.5;
            m * 2f32.powi((i % 17) as i32 - 8)
        })
        .collect()
}

#[test]
fn gpu_reduction_matches_host_f32_tree() {
    let ctx = device_or_skip!();
    for n in [
        1usize, 2, 3, 5, 7, 8, 9, 63, 64, 65, 127, 1_000, 4_096, 65_537,
    ] {
        let xs = data_f32(n, 0xBEEF_1234);
        let want = host_f32_tree(&xs);
        let got = ctx.reduce_f32(&xs).unwrap();
        assert_eq!(
            want.to_bits(),
            got.to_bits(),
            "device reduction diverged from the host f32 tree at n = {n}"
        );
    }
}

#[test]
fn gpu_reduction_is_exact_on_representable_sums() {
    let ctx = device_or_skip!();
    for k in 0..12u32 {
        let n = 1usize << k;
        let xs = vec![0.5f32; n];
        assert_eq!(ctx.reduce_f32(&xs).unwrap(), 0.5 * n as f32, "n = {n}");
    }
}

// =====================================================================
// Contract invariants that hold on either device backend
// =====================================================================

#[test]
fn gpu_respects_the_estimator_range() {
    let _ctx = device_or_skip!();
    let n = 20_000;

    let surv = markov(4, 0, 0, 0.25)
        .run(&cfg(n, 3, Backend::Gpu))
        .unwrap()
        .value;
    assert!((0.0..=1.0).contains(&surv), "survival out of range: {surv}");

    let pol = Family::Bistable {
        x0: 0.0,
        dt: 0.01,
        theta_max: 0.5,
    }
    .run(&cfg(n, 3, Backend::GpuF32))
    .unwrap()
    .value;
    assert!(
        (-1.0..=1.0).contains(&pol),
        "polarisation out of range: {pol}"
    );

    let disp = Family::Gaussian {
        base: vec![0.0, 0.0],
        sigma_max: 0.3,
    }
    .run(&cfg(n, 3, Backend::GpuF32))
    .unwrap()
    .value;
    assert!(
        disp <= 0.0,
        "negative dispersion should be <= 0, got {disp}"
    );
}

#[test]
fn gpu_zero_intensity_recovers_the_base_state() {
    let _ctx = device_or_skip!();
    // C2 (SCHEMA §3): at null intensity the perturbation is the
    // identity, so survival of the start label is exactly 1.
    for backend in [Backend::Gpu, Backend::GpuF32] {
        let surv = markov(5, 2, 2, 0.0)
            .run(&cfg(10_000, 5, backend))
            .unwrap()
            .value;
        assert_eq!(
            surv, 1.0,
            "{backend:?}: null intensity must recover the base label"
        );
    }

    let disp = Family::Gaussian {
        base: vec![1.5, -2.0],
        sigma_max: 0.0,
    }
    .run(&cfg(10_000, 5, Backend::GpuF32))
    .unwrap()
    .value;
    assert_eq!(disp, 0.0, "null intensity must give zero dispersion");
}

#[test]
fn gpu_handles_degenerate_ensemble_sizes() {
    let _ctx = device_or_skip!();
    for n in [1u64, 2, 3, 5, 65] {
        for fam in all_families() {
            let r = fam.run(&cfg(n, 11, Backend::GpuF32)).unwrap();
            assert!(
                r.value.is_finite(),
                "{} produced {} at n = {n}",
                fam.name(),
                r.value
            );
        }
    }
}
