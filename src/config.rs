//! `Config` wire format (SCHEMA §5).
//!
//! [`Config`] carries everything in `(rho, N, seed)` plus the reduction
//! policy. The canonical JSON form is the one in SCHEMA §5; the
//! in-memory form is the obvious struct, with `serde` derive providing
//! the JSON codec.

use serde::{Deserialize, Serialize};

use crate::{Error, Result, SCHEMA_VERSION};

/// Top-level configuration object (SCHEMA §5).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    /// Schema version of this payload (SCHEMA §10).
    pub schema_version: String,
    /// Sample size `N` (SCHEMA §5 row `n`, Paper Def. 7.1).
    pub n: u64,
    /// 64-bit RNG seed (SCHEMA §5 row `seed`, §8 D1).
    pub seed: u64,
    /// Intensity sampler `rho` descriptor (SCHEMA §5 row `intensity`).
    pub intensity: Intensity,
    /// Reduction policy (SCHEMA §5 row `reduction`, §8 D3).
    pub reduction: Reduction,
    /// Declared Lipschitz constants (SCHEMA §5 row `lipschitz`,
    /// Paper Assumption 5.1).
    pub lipschitz: Lipschitz,
    /// OPTIONAL accuracy target `(epsilon, eta)` (SCHEMA §7).
    /// Asserting this enables the sample-complexity check in
    /// [`crate::engine::Engine::run`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<Accuracy>,
    /// Execution backend. Additive to SCHEMA §5 and omitted from the
    /// wire form when [`Backend::Auto`], so a default config still
    /// serialises to the exact v1.0.0 payload.
    ///
    /// This selects *how* the estimator is evaluated, not *what* it
    /// evaluates: [`Backend::Scalar`] and [`Backend::Simd`] are
    /// bit-identical to each other and to v1.0.0.
    /// [`Backend::Gpu`] is a distinct numerical path; see
    /// [`crate::gpu`] for its determinism contract.
    #[serde(default, skip_serializing_if = "Backend::is_auto")]
    pub backend: Backend,
}

/// Execution backend selector (additive to SCHEMA §5).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Pick the fastest host path available: vectorised reductions,
    /// and multi-threaded ensemble generation above
    /// [`crate::engine::PARALLEL_MIN`]. Bit-identical to
    /// [`Backend::Scalar`].
    #[default]
    Auto,
    /// Force the portable scalar path. This is the reference
    /// implementation and the arbiter when two paths disagree.
    Scalar,
    /// Force the vectorised host path even on inputs too small to
    /// amortise it. Bit-identical to [`Backend::Scalar`].
    Simd,
    /// Run on a compute device through `wgpu`, **bit-identically to
    /// the host**.
    ///
    /// Available only for families the device can reproduce exactly.
    /// Requesting it for one it cannot is an error rather than a silent
    /// change of answer; see [`crate::gpu`] for which families qualify
    /// and why.
    Gpu,
    /// Run on a compute device in single precision.
    ///
    /// Available for every built-in family and considerably faster than
    /// [`Backend::Gpu`], but the ensemble is carried in `f32` and
    /// normal deviates come from Box-Muller rather than the ziggurat,
    /// so the result agrees with the host statistically rather than bit
    /// for bit. Opt in deliberately, and read
    /// [`crate::report::Execution`] on anything it produces.
    #[serde(rename = "gpu_f32")]
    GpuF32,
}

impl Backend {
    /// `true` for [`Backend::Auto`]; used to keep the default out of
    /// the serialised config.
    pub fn is_auto(&self) -> bool {
        matches!(self, Backend::Auto)
    }

    /// Lowercase tag matching the serialised form.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Scalar => "scalar",
            Backend::Simd => "simd",
            Backend::Gpu => "gpu",
            Backend::GpuF32 => "gpu_f32",
        }
    }

    /// `true` for backends that dispatch to a compute device.
    pub fn is_device(self) -> bool {
        matches!(self, Backend::Gpu | Backend::GpuF32)
    }
}

impl Default for Config {
    /// A minimal valid config: `n = 1024`, seed `0`, a Dirac intensity
    /// with null parameter `0.0`, tree/index reduction, no declared
    /// Lipschitz constants, no accuracy claim.
    ///
    /// Intended for `..Default::default()` so that additive fields do
    /// not break struct-literal construction.
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            n: 1024,
            seed: 0,
            intensity: Intensity {
                kind: "dirac".to_string(),
                params: serde_json::json!({}),
                null_parameter: serde_json::json!(0.0),
            },
            reduction: Reduction::default(),
            lipschitz: Lipschitz::default(),
            accuracy: None,
            backend: Backend::Auto,
        }
    }
}

/// Intensity descriptor for `rho` (SCHEMA §5).
///
/// The actual sampler is implemented by the
/// [`Perturbation`](crate::perturbation::Perturbation) type; this
/// descriptor exists for cross-checking and provenance only. The
/// engine enforces that `null_parameter` matches
/// `Perturbation::null()` (SCHEMA §5 last paragraph; C2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Intensity {
    /// Sampler family name (e.g. `"uniform_interval"`,
    /// `"gaussian"`, `"dirac"`).
    pub kind: String,
    /// Sampler hyperparameters, opaque JSON.
    pub params: serde_json::Value,
    /// Null parameter `theta_0` (Paper Def. 3.1; SCHEMA §3 C2).
    pub null_parameter: serde_json::Value,
}

/// Reduction policy (SCHEMA §5 row `reduction`, §8 D3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Reduction {
    /// `"tree"` (REQUIRED for reproducible parallel) or `"sequential"`.
    pub order: String,
    /// `"index"` (REQUIRED for cross-implementation agreement).
    pub leaf_order: String,
}

impl Default for Reduction {
    fn default() -> Self {
        Self {
            order: "tree".to_string(),
            leaf_order: "index".to_string(),
        }
    }
}

/// Declared Lipschitz constants (SCHEMA §5 / Paper Assumption 5.1).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct Lipschitz {
    /// `L` for the forward model (Paper Assumption 5.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_l: Option<f64>,
    /// `Lambda` for the invariance functional (Paper Assumption 5.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invariance_lambda: Option<f64>,
}

/// OPTIONAL accuracy target (SCHEMA §7).
///
/// When present, the engine enforces the sample-complexity floor of
/// Paper Thm 7.3(c).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Accuracy {
    /// Target additive error `epsilon > 0`.
    pub epsilon: f64,
    /// Target failure probability `eta in (0,1)`.
    pub eta: f64,
    /// Observation diameter `D` (Paper Thm 7.3).
    pub observation_diameter: f64,
    /// Observation dimension `d_obs` (Paper Thm 7.3(b), Fournier-Guillin).
    pub obs_dim: u32,
}

impl Config {
    /// Decode a JSON payload (SCHEMA §5).
    pub fn from_json(s: &str) -> Result<Self> {
        let cfg: Config = serde_json::from_str(s)?;
        cfg.validate_version()?;
        Ok(cfg)
    }

    /// Encode to JSON (SCHEMA §5).
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Reject configs with a different MAJOR `schema_version` (SCHEMA §10).
    pub fn validate_version(&self) -> Result<()> {
        let want_major = major(SCHEMA_VERSION);
        let got_major = major(&self.schema_version);
        if got_major != want_major {
            return Err(Error::SchemaVersion {
                got: self.schema_version.clone(),
                want_major,
            });
        }
        Ok(())
    }

    /// Return `self` with the execution backend replaced.
    #[must_use]
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Required `N` floor under Paper Thm 7.3(c) for the asserted
    /// `(epsilon, eta)` and the declared `Lambda` (SCHEMA §7).
    ///
    /// Returns `None` if no accuracy was asserted or `Lambda` is
    /// missing.
    pub fn sample_floor(&self) -> Option<u64> {
        let acc = self.accuracy?;
        let lambda = self.lipschitz.invariance_lambda?;
        Some(sample_floor(
            lambda,
            acc.observation_diameter,
            acc.epsilon,
            acc.eta,
            acc.obs_dim,
        ))
    }
}

/// Do two `null_parameter` values denote the same number?
///
/// Compared numerically rather than by JSON value identity, because
/// most JSON producers cannot express the difference. JavaScript has a
/// single number type, so `JSON.stringify(0.0)` is `0`; strict
/// `serde_json::Value` equality would reject that against the `0.0` a
/// Rust implementation reports, for no reason a caller could act on.
///
/// Non-numeric parameters still compare structurally, so a family whose
/// `theta` is a vector or a record keeps exact matching.
pub fn null_parameters_agree(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

/// Extract the SemVer major component (SCHEMA §10).
fn major(v: &str) -> u64 {
    v.split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Sample-complexity floor of Paper Thm 7.3(c) (SCHEMA §7).
///
/// Returns `max(N_stochastic, N_bias)` where
/// * `N_stochastic = ceil( Lambda^2 D^2 / (2 eps^2) * ln(2/eta) )` (McDiarmid),
/// * `N_bias` is the smallest `N` whose Fournier-Guillin rate
///   majorises `eps/2`: `O(N^{-1/d_obs})` for `d_obs > 2`,
///   `O(N^{-1/2} log N)` for `d_obs <= 2`.
///
/// The constants hidden in `O(.)` are absorbed into a unit prefactor
/// here; this is intentionally a *floor*, not a tight estimate, and
/// matches the conservative reading the schema demands.
pub fn sample_floor(lambda: f64, d: f64, eps: f64, eta: f64, obs_dim: u32) -> u64 {
    let stoch = (lambda * lambda * d * d) / (2.0 * eps * eps) * (2.0 / eta).ln();
    let bias = if obs_dim <= 2 {
        // O(N^{-1/2} log N) <= eps/2 -- solve numerically.
        let target = eps / 2.0;
        let mut n = 1.0_f64;
        // doubling search then bisection: cheap, monotone.
        while n.powf(-0.5) * (n + 1.0).ln() > target && n < 1e18 {
            n *= 2.0;
        }
        n
    } else {
        let target = eps / 2.0;
        target.powf(-(obs_dim as f64))
    };
    let raw = stoch.max(bias).ceil();
    if raw.is_finite() && raw > 0.0 {
        raw as u64
    } else {
        1
    }
}
