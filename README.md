# perturbation-kernel

Reference Rust implementation of the perturbation-kernel object defined in
[`../SCHEMA.md`](../SCHEMA.md) (v1.0.0). The mathematics it implements is in
[`../paper.tex`](../paper.tex), *A Measure-Theoretic Schema for Perturbation
Kernels*. Where the two disagree, the paper governs the mathematics and the
schema governs the wire formats.

## Build

```
cargo build
cargo test
```

`rustc 1.75+` is sufficient. `cargo build --release` produces an `rlib`, a
`cdylib`, and a `staticlib`; the latter two carry the C-ABI surface from
[`src/abi.rs`](src/abi.rs) (SCHEMA §9).

## Conformance status

The crate satisfies every MUST of SCHEMA §§3-7. The C1-C6 checklist (SCHEMA §3):

| Point | Requirement | Where |
|-------|-------------|-------|
| C1 | Typed spaces with a metric `d_O` | type parameters on the traits; metric properties exercised by `tests/conformance.rs::c1_*` (proptest) |
| C2 | `Perturbation<S>` with null-parameter identity | [`src/perturbation.rs`](src/perturbation.rs); identity recovery checked in `tests/conformance.rs::c2_identity_recovery_*` |
| C3 | `ForwardModel<S,O>` with optional declared `L` | [`src/forward.rs`](src/forward.rs) |
| C4 | `Invariance<O>` with optional declared `Lambda` | [`src/invariance.rs`](src/invariance.rs); order invariance in `tests/conformance.rs::s11_order_invariance_of_measure` |
| C5 | `Config` carrying `rho`, `N`, seed, reduction | [`src/config.rs`](src/config.rs); JSON round-trip via `serde` |
| C6 | Engine running `Phi-hat_N(s)` with seeded reduction | [`src/engine.rs`](src/engine.rs); per-index fork in `fork_rng` |

SCHEMA §7 sample-complexity floor and §8 determinism (D1-D4) are exercised by
`tests/sample_complexity.rs` and `tests/determinism.rs` respectively. A
dishonest accuracy claim (N below the Thm 7.3(c) floor) is rejected with
`Error::SampleFloor`.

## Usage

```rust
use perturbation_kernel::{
    config::{Config, Intensity, Lipschitz, Reduction},
    engine::Engine,
    examples::markov,
};
use serde_json::json;

let cfg = Config {
    schema_version: "1.0.0".into(),
    n: 100_000,
    seed: 20260610,
    intensity: Intensity {
        kind: "uniform_interval".into(),
        params: json!({ "low": 0.0, "high": 0.3 }),
        null_parameter: json!(0.0),
    },
    reduction: Reduction::default(),
    lipschitz: Lipschitz { forward_l: Some(1.0), invariance_lambda: Some(1.0) },
    accuracy: None,
};

let base = markov::Label { i: 0 };
let fam  = markov::UniformMixing { k: 4, theta_max: 0.3 };
let model = markov::BaseIndicator { base_label: 0 };
let inv  = markov::Survival;

let report = Engine::run(&base, &fam, &model, &inv, &cfg).unwrap();
println!("{}", report.to_json_pretty().unwrap());
```

Three worked examples ship in [`src/examples.rs`](src/examples.rs):

1. **Gaussian shift in `R^d`** -- vector state, Gaussian perturbation,
   identity forward model, negative empirical variance as the invariance.
2. **Bistable double-well marble** -- scalar state under a one-step
   Euler-Maruyama Langevin perturbation, sign-of-well forward readout,
   polarisation as the invariance.
3. **Finite-state Markov chain** -- discrete state under an
   epsilon-uniform-mixing perturbation, indicator forward map, tail
   survival as the invariance.

## C-ABI

The minimal projection (SCHEMA §9) lives in [`src/abi.rs`](src/abi.rs):

```c
double      pk_run(/* ... */);   // returns opaque pk_report*
double      pk_report_value(const pk_report*);
const char* pk_report_json(const pk_report*);
void        pk_free_report(pk_report*);
```

A C/C++ consumer supplies three vtables of `extern "C"` function pointers
implementing `Perturbation<f64>`, `ForwardModel<f64, f64>`, and
`Invariance<f64>`; the engine remains the single source of truth (Paper
§"Why the kernel wants a compiled core", SCHEMA §9).

## Determinism

`ChaCha20Rng` is keyed by `cfg.seed` and forked per draw index `i` via a
SplitMix64-style mix of `(seed, i)`. All sampling threads `&mut Rng`
through `Perturbation::sample_theta` and `Perturbation::apply`; nothing
in the crate calls `thread_rng()`. Two runs with the same `Config`
return bit-identical `Report::value` and byte-identical
`Report::to_json()` output (Paper Thm 8.2; verified by
`tests/determinism.rs::d4_*`).

## License

MIT. See [`LICENSE`](LICENSE).
