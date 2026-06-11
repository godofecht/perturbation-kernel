//! Reference implementation of the perturbation-kernel object
//! (SCHEMA.md v1.0.0; companion to `paper.tex`, "A Measure-Theoretic Schema
//! for Perturbation Kernels").
//!
//! The crate exposes the four canonical traits
//! ([`Perturbation`](perturbation::Perturbation),
//! [`ForwardModel`](forward::ForwardModel),
//! [`Invariance`](invariance::Invariance)) plus the
//! [`Engine`](engine::Engine) implementing the plug-in estimator
//! `Phi-hat_N(s)` of Paper Def. 7.1 / SCHEMA §2.
//!
//! Conformance is reached by satisfying every MUST of SCHEMA §§3-7; see
//! the C1-C6 checklist (SCHEMA §3) and `tests/conformance.rs` for the
//! verification harness. The C-ABI projection lives in [`abi`] per
//! SCHEMA §9.
//!
//! All randomness flows through an explicitly-threaded
//! [`rand_chacha::ChaCha20Rng`] keyed by the 64-bit `seed` of
//! [`config::Config`], so two runs with the same config produce
//! bit-identical [`report::Report`]s (SCHEMA §8 / Paper Thm 8.2).

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod abi;
pub mod config;
pub mod engine;
pub mod examples;
pub mod forward;
pub mod invariance;
pub mod perturbation;
pub mod report;

/// Crate-wide RNG type alias.
///
/// The reference engine uses `ChaCha20` because it is reproducible, fast,
/// and has a counter-based [`SeedableRng`] which makes the per-index
/// substream fork of SCHEMA §8 D2 cheap. The type alias is intentionally
/// exposed: external [`Perturbation`](perturbation::Perturbation) /
/// [`ForwardModel`](forward::ForwardModel) implementations
/// take `&mut Rng` and so must agree on the concrete type to satisfy the
/// determinism contract (SCHEMA §8 D1).
pub type Rng = rand_chacha::ChaCha20Rng;

/// Schema version this implementation advertises (SCHEMA §10).
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Errors returned by the engine and the C-ABI surface.
///
/// The variants correspond to the MUST clauses of SCHEMA §§3-8 that a
/// caller can violate at runtime: a malformed config, a mismatched
/// null parameter (SCHEMA §5 last paragraph / C2), an incompatible
/// major schema version (SCHEMA §10), or an empty ensemble.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Wrapper for `serde_json` decode failures on [`config::Config`] or
    /// [`report::Report`] payloads (SCHEMA §§5-6).
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// `schema_version` major component differs from this crate's
    /// `SCHEMA_VERSION` (SCHEMA §10).
    #[error("incompatible schema major version: got {got}, expected major {want_major}")]
    SchemaVersion {
        /// Version provided by the caller.
        got: String,
        /// Major component this crate accepts.
        want_major: u64,
    },

    /// The `intensity.null_parameter` in [`config::Config`] does not equal
    /// `Perturbation::null()` (SCHEMA §5, C2).
    #[error("null_parameter mismatch: config {config} vs perturbation {perturbation}")]
    NullParameterMismatch {
        /// Serialised value from the config.
        config: String,
        /// Serialised value from the perturbation.
        perturbation: String,
    },

    /// `n` was zero (SCHEMA §5 row `n`).
    #[error("n must be >= 1, got 0")]
    EmptyEnsemble,

    /// Sample-complexity floor (SCHEMA §7) violated for an accuracy
    /// claim.
    #[error("sample-complexity floor: requested ({epsilon},{eta}) needs N >= {floor}, got {n}")]
    SampleFloor {
        /// Target additive error.
        epsilon: f64,
        /// Target failure probability.
        eta: f64,
        /// Minimum `N` derived from Paper Thm 7.3(c).
        floor: u64,
        /// Actual `n` supplied.
        n: u64,
    },
}

/// Convenience alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;
