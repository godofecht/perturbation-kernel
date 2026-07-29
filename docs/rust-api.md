# Rust API

```toml
[dependencies]
perturbation-kernel = "2"
```

Full rustdoc is published with each release. This page covers the shape
of the API and the decisions you have to make.

## Two entry points

**`Family::run`** takes one of the three built-in families as data. This
is the path Python drives, the path a GPU can execute, and the path with
the optimised flat storage.

```rust
use perturbation_kernel::config::{Backend, Config};
use perturbation_kernel::family::Family;

let cfg = Config {
    n: 262_144,
    seed: 20_260_610,
    backend: Backend::Auto,
    ..Default::default()
};

let report = Family::Markov { k: 5, start: 0, base_label: 0, theta_max: 0.3 }
    .run(&cfg)?;
```

**`Engine::run`** takes three trait implementations. This is the
extension point, and it is where your own model goes.

```rust
let report = Engine::run(&base, &perturbation, &forward, &invariance, &cfg)?;
```

The two agree bit for bit on the built-in families;
`tests/backends.rs` asserts it across ensemble sizes, seeds and
backends.

## Implementing your own model

Three traits. A complete worked version is
`examples/custom_family.rs`, which asks whether a fitted regression
slope keeps its sign under measurement error.

```rust
pub trait Perturbation<S> {
    type Theta: Serialize + DeserializeOwned + Clone;
    fn null(&self) -> Self::Theta;
    fn sample_theta(&self, rng: &mut Rng) -> Self::Theta;
    fn apply(&self, s: &S, theta: &Self::Theta, rng: &mut Rng) -> S;
}

pub trait ForwardModel<S, O> {
    fn eval(&self, s: &S) -> O;
    fn lipschitz(&self) -> Option<f64> { None }
}

pub trait Invariance<O> {
    fn measure(&self, ensemble: &[O]) -> Report;
    fn lipschitz_w1(&self) -> Option<f64> { None }
    fn name(&self) -> &str { "unspecified" }
}
```

Three rules that the engine relies on and cannot check for you:

**All randomness goes through the `rng` argument.** Nothing may read
global state or call `thread_rng()`. The engine hands each draw its own
substream, and that is what makes the loop safe to parallelise.

**`null()` must be a genuine identity.** `apply(s, null(), rng)` has to
return `s` in distribution. The engine cross-checks the value against
the config's `null_parameter` and refuses to run on a mismatch, but it
cannot check that your `apply` actually honours it.

**`measure` must be permutation-invariant** and must reduce through
`reduce::mean`, `reduce::tree_sum` or `reduce::sum_sq_dev`. A plain
`.iter().sum()` gives up bit-reproducibility across thread counts,
because floating-point addition is not associative.

## Threading and the `Send`/`Sync` bounds

`Engine::run` requires `S: Sync`, `O: Send`, `P: Sync`, `F: Sync`, so
the draw loop can use a thread pool. Any implementation that honours the
purity requirement satisfies these, because a pure function of
`(s, theta, rng)` holds no thread-affine state.

If yours genuinely cannot cross threads, use `Engine::run_sequential`,
which has the v1.0.0 signature without the bounds and returns a
bit-identical report. The C ABI uses it, because a foreign vtable holds
an opaque `void*` and nothing on the Rust side can know whether it is
safe to share.

## Reductions

```rust
use perturbation_kernel::reduce;

reduce::tree_sum(&xs);              // deterministic pairwise sum
reduce::mean(&xs);
reduce::sum_sq_dev(&xs, mean);      // centred second moment
reduce::dot(&a, &b);

reduce::tree_sum_into(&xs, &mut scratch);       // reuse a buffer
reduce::sum_sq_dev_into(&xs, mean, &mut scratch);

reduce::active_backend();           // SimdPath::{Scalar, Neon, Avx2}
```

The `_into` variants take a caller-owned scratch buffer, so reducing `d`
columns costs one allocation rather than `d`. All of them are
bit-identical to the v1.0.0 reduction and to each other across vector
paths.

## The GPU

```rust
use perturbation_kernel::gpu;

gpu::available();       // bool
gpu::context()?.name;   // "Apple M4 Max (Metal, IntegratedGpu)"
```

Requires the `gpu` feature. `Engine::run` returns
`Error::UnsupportedBackend` for `Backend::Gpu`, because a device cannot
invoke user trait implementations; use `Family::run`.

## Errors

```rust
pub enum Error {
    Json(serde_json::Error),
    SchemaVersion { got: String, want_major: u64 },
    NullParameterMismatch { config: String, perturbation: String },
    EmptyEnsemble,
    SampleFloor { epsilon: f64, eta: f64, floor: u64, n: u64 },
    UnsupportedBackend { backend: &'static str, reason: String },
    Gpu(String),
    InvalidFamily(String),
}
```

## Migrating from 1.0

The wire formats are unchanged and `SCHEMA_VERSION` is still `1.0.0`.
Values are bit-identical, enforced by the golden check in CI. Three
source-level changes:

- `Config` gained a `backend` field. Use `..Default::default()` in
  struct literals, or add `backend: Backend::Auto`.
- `Engine::run` gained `Send`/`Sync` bounds. `Engine::run_sequential`
  keeps the old signature.
- `Report` gained an optional `execution` block. `to_json_v1()` emits
  the strict v1.0.0 field set.

`engine::tree_sum` is still there, re-exported from `reduce`.
