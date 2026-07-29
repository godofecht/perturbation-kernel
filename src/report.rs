//! `Report` wire format (SCHEMA §6).
//!
//! [`Report`] is the only object the caller needs for the verdict on
//! a perturbation experiment: the scalar `Phi-hat_N(s)`, the
//! provenance `(seed, reduction)`, and an OPTIONAL non-asymptotic
//! error bound derived from Paper Thm 7.3.

use serde::{Deserialize, Serialize};

use crate::config::Reduction;
use crate::SCHEMA_VERSION;

/// Top-level report object (SCHEMA §6).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Report {
    /// Schema version (SCHEMA §10).
    pub schema_version: String,
    /// The scalar invariance value `Phi-hat_N(s)` (SCHEMA §6 row `value`,
    /// Paper Def. 7.1).
    pub value: f64,
    /// Tag identifying which `Phi` produced `value` (SCHEMA §6).
    pub functional: String,
    /// Sample size actually used. Equals `cfg.n` for a complete run.
    pub n_effective: u64,
    /// Provenance: RNG seed (SCHEMA §8 D4).
    pub seed: u64,
    /// Provenance: reduction policy (SCHEMA §8 D3).
    pub reduction: Reduction,
    /// OPTIONAL Paper Thm 7.3 error bound block (SCHEMA §6).
    pub error_bound: ErrorBound,
    /// OPTIONAL Paper Thm 5.4 stability modulus `Lambda * L * C`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stability_modulus: Option<f64>,
    /// OPTIONAL execution provenance. Additive to SCHEMA §6: it
    /// records *how* the value was computed, which matters because
    /// [`crate::config::Backend::Gpu`] is a different numerical path
    /// from the host backends.
    ///
    /// Use [`Report::to_json_v1`] to serialise without this block when
    /// a strict v1.0.0 payload is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<Execution>,
}

/// Execution provenance (additive to SCHEMA §6).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Execution {
    /// Requested backend: `"auto"`, `"scalar"`, `"simd"`, or `"gpu"`.
    pub backend: String,
    /// Vector path actually taken on the host: `"scalar"`, `"neon"`,
    /// or `"avx2"`.
    pub simd_path: String,
    /// `true` when the draw loop was spread across a thread pool.
    /// Does not affect [`Report::value`].
    pub threaded: bool,
    /// Device description when the run touched a GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Working precision of the ensemble: `"f64"` on the host paths,
    /// `"f32"` on the GPU path.
    pub precision: String,
}

impl Execution {
    /// Provenance for a run that stayed on the host.
    pub fn host(backend: crate::config::Backend, n: u64) -> Self {
        Self {
            backend: backend.as_str().to_string(),
            simd_path: crate::reduce::path_for(backend).as_str().to_string(),
            threaded: cfg!(feature = "parallel")
                && n >= crate::engine::PARALLEL_MIN
                && backend != crate::config::Backend::Scalar
                && !backend.is_device(),
            device: None,
            precision: "f64".to_string(),
        }
    }
}

/// Non-asymptotic error bound block (SCHEMA §6).
///
/// `available` is `true` iff `Lambda, D, obs_dim` were all supplied and
/// the bound was actually computed; otherwise `false` (SCHEMA §6 last
/// paragraph).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErrorBound {
    /// `true` iff the bound was actually derived.
    pub available: bool,
    /// Achieved `epsilon` under Paper Thm 7.3 with the supplied
    /// `Lambda`, `D`, `N`, and `eta`. `0.0` when `available == false`.
    pub epsilon: f64,
    /// Failure probability used in the bound.
    pub eta: f64,
    /// Free-text basis tag, e.g. `"mcdiarmid+fournier_guillin"`.
    pub basis: String,
    /// Mirror of the input constants for provenance.
    pub constants: BoundConstants,
}

/// Constants that fix the bound (SCHEMA §6 `error_bound.constants`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct BoundConstants {
    /// `Lambda` (Paper Assumption 5.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lambda: Option<f64>,
    /// Observation diameter `D`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d: Option<f64>,
    /// Observation dimension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub obs_dim: Option<u32>,
}

impl Report {
    /// Build a report with no bound, no stability modulus, just the value
    /// and provenance. The invariance trait calls this; the engine
    /// later enriches the bound and stability fields.
    pub fn raw(value: f64, functional: &str, n: u64, seed: u64, reduction: Reduction) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            value,
            functional: functional.to_string(),
            n_effective: n,
            seed,
            reduction,
            error_bound: ErrorBound {
                available: false,
                epsilon: 0.0,
                eta: 0.0,
                basis: "none".to_string(),
                constants: BoundConstants::default(),
            },
            stability_modulus: None,
            execution: None,
        }
    }

    /// Serialise to canonical JSON (SCHEMA §6).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialise without the additive [`Execution`] block, giving a
    /// payload with exactly the SCHEMA v1.0.0 field set.
    pub fn to_json_v1(&self) -> Result<String, serde_json::Error> {
        let mut stripped = self.clone();
        stripped.execution = None;
        serde_json::to_string(&stripped)
    }

    /// Pretty-printed JSON for human inspection.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
