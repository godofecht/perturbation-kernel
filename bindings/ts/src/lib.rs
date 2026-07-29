//! WebAssembly bindings for perturbation-kernel.
//!
//! Compiled for `wasm32-unknown-unknown` and wrapped by
//! `wasm-bindgen`, so the same estimator runs in a browser, in Node and
//! in Deno without a native toolchain on the consumer's machine.
//!
//! Only the scalar backend exists here. Threads need shared memory that
//! `wasm32-unknown-unknown` does not provide by default, and the vector
//! kernels are architecture intrinsics with no wasm equivalent. That
//! costs nothing in correctness: the scalar path *is* the reference
//! path, so every value this module returns is bit-identical to the
//! native library.

use perturbation_kernel::config::Config;
use perturbation_kernel::family::Family;
use wasm_bindgen::prelude::*;

/// Run a family against a config, both as JSON, returning the report as
/// JSON (SCHEMA §6).
///
/// Taking JSON on both sides keeps the wasm boundary free of struct
/// layout agreements, exactly as the C ABI does.
#[wasm_bindgen]
pub fn run_family(family_json: &str, config_json: &str) -> Result<String, JsError> {
    let family: Family = serde_json::from_str(family_json)
        .map_err(|e| JsError::new(&format!("bad family descriptor: {e}")))?;
    let cfg =
        Config::from_json(config_json).map_err(|e| JsError::new(&format!("bad config: {e}")))?;
    let report = family.run(&cfg).map_err(|e| JsError::new(&e.to_string()))?;
    report
        .to_json()
        .map_err(|e| JsError::new(&format!("could not serialise the report: {e}")))
}

/// Deterministic pairwise sum (SCHEMA §8 D3).
///
/// Exposed because reproducing the engine's reduction order is the only
/// way to check an externally computed ensemble against a report;
/// ordinary summation will not match.
#[wasm_bindgen]
pub fn tree_sum(xs: &[f64]) -> f64 {
    perturbation_kernel::reduce::tree_sum(xs)
}

/// Theorem 7.3(c) sample floor for an accuracy target.
#[wasm_bindgen]
pub fn sample_floor(
    invariance_lambda: f64,
    observation_diameter: f64,
    epsilon: f64,
    eta: f64,
    obs_dim: u32,
) -> u64 {
    perturbation_kernel::config::sample_floor(
        invariance_lambda,
        observation_diameter,
        epsilon,
        eta,
        obs_dim,
    )
}

/// Crate version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Schema version this build implements (SCHEMA §10).
#[wasm_bindgen]
pub fn schema_version() -> String {
    perturbation_kernel::SCHEMA_VERSION.to_string()
}
