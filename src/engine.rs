//! Engine (SCHEMA §4.4): runs the plug-in estimator of Paper Def. 7.1.
//!
//! The algorithm (SCHEMA §4.4 / §8):
//!
//! ```text
//! seed_rng = ChaCha20::from_seed(cfg.seed)
//! for i in 0..N:
//!     sub_rng_i = fork(seed_rng, i)        // counter-based substream, D2
//!     theta_i   = fam.sample_theta(sub_rng_i)
//!     S_i       = fam.apply(s, theta_i, sub_rng_i)
//!     Y_i       = model.eval(S_i)
//! report = inv.measure(&Y[..])             // reduced in cfg.reduction
//! ```
//!
//! `fork(seed, i)` is implemented by re-seeding a fresh `ChaCha20` from
//! the 64-bit value `mix64(seed, i)`. This is the per-index substream
//! required by SCHEMA §8 D2: each draw is independent of the others
//! and the result is invariant under the number of threads that actually
//! evaluated the loop body.

use rand::SeedableRng;

use crate::config::{Config, Reduction};
use crate::forward::ForwardModel;
use crate::invariance::Invariance;
use crate::perturbation::Perturbation;
use crate::report::{BoundConstants, ErrorBound, Report};
use crate::{Error, Result, Rng};

/// Engine: the plug-in estimator runner (SCHEMA §4.4).
///
/// This is a unit struct because every piece of state lives in the
/// arguments to [`Engine::run`]. Keeping it nominal lets us hang
/// future methods (sweeps, batch APIs) off it without a breaking
/// change.
pub struct Engine;

impl Engine {
    /// Run the plug-in estimator and return the [`Report`]
    /// (SCHEMA §4.4 / Paper Def. 7.1).
    ///
    /// Errors:
    /// * [`Error::SchemaVersion`] if `cfg.schema_version` has a different
    ///   MAJOR (SCHEMA §10).
    /// * [`Error::EmptyEnsemble`] if `cfg.n == 0` (SCHEMA §5).
    /// * [`Error::NullParameterMismatch`] if the config's
    ///   `null_parameter` doesn't agree with `fam.null()` (SCHEMA §5, C2).
    /// * [`Error::SampleFloor`] if `cfg.accuracy` is set and `cfg.n` is
    ///   below the Paper Thm 7.3(c) floor (SCHEMA §7).
    pub fn run<S, O, P, F, I>(
        base: &S,
        fam: &P,
        model: &F,
        inv: &I,
        cfg: &Config,
    ) -> Result<Report>
    where
        P: Perturbation<S>,
        F: ForwardModel<S, O>,
        I: Invariance<O>,
    {
        cfg.validate_version()?;
        if cfg.n == 0 {
            return Err(Error::EmptyEnsemble);
        }

        // C2: enforce that the null parameter declared in the wire
        // payload matches the implementation's (SCHEMA §5 last paragraph).
        let null_p = serde_json::to_value(fam.null())?;
        if null_p != cfg.intensity.null_parameter {
            return Err(Error::NullParameterMismatch {
                config: cfg.intensity.null_parameter.to_string(),
                perturbation: null_p.to_string(),
            });
        }

        // SCHEMA §7 sample-complexity floor.
        if let Some(floor) = cfg.sample_floor() {
            if cfg.n < floor {
                let acc = cfg.accuracy.expect("floor implies accuracy set");
                return Err(Error::SampleFloor {
                    epsilon: acc.epsilon,
                    eta: acc.eta,
                    floor,
                    n: cfg.n,
                });
            }
        }

        // SCHEMA §8 D1/D2: deterministic per-index substreams.
        let mut ys: Vec<O> = Vec::with_capacity(cfg.n as usize);
        for i in 0..cfg.n {
            let mut sub = fork_rng(cfg.seed, i);
            let theta = fam.sample_theta(&mut sub);
            let s_i = fam.apply(base, &theta, &mut sub);
            ys.push(model.eval(&s_i));
        }

        // SCHEMA §8 D3: deterministic reduction order. The
        // `Invariance::measure` impl owns the reduction; the engine
        // simply passes the ensemble in index order, which is the
        // "leaf_order = index" mandated by SCHEMA §5.
        validate_reduction(&cfg.reduction)?;
        let mut report = inv.measure(&ys);

        // Stitch provenance (SCHEMA §6).
        report.schema_version = crate::SCHEMA_VERSION.to_string();
        report.functional = inv.name().to_string();
        report.n_effective = cfg.n;
        report.seed = cfg.seed;
        report.reduction = cfg.reduction.clone();

        // Error bound (Paper Thm 7.3, SCHEMA §6).
        let (lambda, l, d_obs, diameter) = (
            cfg.lipschitz.invariance_lambda,
            cfg.lipschitz.forward_l.or_else(|| model.lipschitz()),
            cfg.accuracy.map(|a| a.obs_dim),
            cfg.accuracy.map(|a| a.observation_diameter),
        );
        if let (Some(lambda), Some(d), Some(d_obs)) = (lambda, diameter, d_obs) {
            // Solve Paper Thm 7.3(a): P(|hat - E[hat]| >= t) <= 2 exp(-2N t^2 / (Lambda^2 D^2))
            // gives t = sqrt( Lambda^2 D^2 / (2N) * ln(2/eta) ).
            let eta = cfg.accuracy.map(|a| a.eta).unwrap_or(0.05);
            let stoch =
                ((lambda * lambda * d * d) / (2.0 * cfg.n as f64) * (2.0 / eta).ln()).sqrt();
            // Bias part (b): Fournier-Guillin rate -- we surface the same
            // shape used by config::sample_floor.
            let bias = if d_obs <= 2 {
                let n = cfg.n as f64;
                lambda * n.powf(-0.5) * (n + 1.0).ln()
            } else {
                lambda * (cfg.n as f64).powf(-(1.0 / d_obs as f64))
            };
            let epsilon = stoch + bias;
            report.error_bound = ErrorBound {
                available: true,
                epsilon,
                eta,
                basis: "mcdiarmid+fournier_guillin".to_string(),
                constants: BoundConstants {
                    lambda: Some(lambda),
                    d: Some(d),
                    obs_dim: Some(d_obs),
                },
            };
        }

        // Stability modulus (Paper Thm 5.4): Lambda * L * C, with
        // C absorbed into the family's mixing constant; we expose
        // Lambda * L when both are declared, and let downstream
        // analyses multiply by their family-specific C.
        if let (Some(lambda), Some(l)) = (lambda, l) {
            report.stability_modulus = Some(lambda * l);
        }

        Ok(report)
    }
}

/// Per-index RNG fork (SCHEMA §8 D2).
///
/// `mix64` is a splittable-RNG-style finaliser (SplitMix64) of the
/// pair `(seed, i)`. The output reseeds a fresh `ChaCha20`. This is
/// counter-based by construction: drawing `theta_i` does not advance
/// the stream that produces `theta_{i+1}`.
pub fn fork_rng(seed: u64, i: u64) -> Rng {
    let mut z = seed
        .wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&z.to_le_bytes());
    seed_bytes[8..16].copy_from_slice(&seed.to_le_bytes());
    seed_bytes[16..24].copy_from_slice(&i.to_le_bytes());
    // last 8 bytes left zero; ChaCha20 keys on the full 32-byte seed.
    Rng::from_seed(seed_bytes)
}

fn validate_reduction(r: &Reduction) -> Result<()> {
    match (r.order.as_str(), r.leaf_order.as_str()) {
        ("tree", "index") | ("sequential", "index") => Ok(()),
        _ => Err(Error::Json(serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "unsupported reduction policy: order={:?}, leaf_order={:?}",
                r.order, r.leaf_order
            ),
        )))),
    }
}

/// Tree-order reduction of a slice of `f64` (SCHEMA §8 D3).
///
/// Used by the example invariance functionals. Pairwise summation is
/// permutation-tolerant only up to last-bit rounding because the tree
/// shape is fixed by the index order ("leaf_order = index"); two runs
/// with the same `N` therefore produce bit-identical sums regardless
/// of the number of threads.
pub fn tree_sum(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut buf: Vec<f64> = xs.to_vec();
    while buf.len() > 1 {
        let n = buf.len();
        let half = n / 2;
        let mut next = Vec::with_capacity(half + (n & 1));
        for k in 0..half {
            next.push(buf[2 * k] + buf[2 * k + 1]);
        }
        if n & 1 == 1 {
            next.push(buf[n - 1]);
        }
        buf = next;
    }
    buf[0]
}
