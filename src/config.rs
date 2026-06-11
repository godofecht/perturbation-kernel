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

/// Extract the SemVer major component (SCHEMA §10).
fn major(v: &str) -> u64 {
    v.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(0)
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
