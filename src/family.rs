//! Built-in perturbation families as data (additive to SCHEMA §4).
//!
//! [`crate::engine::Engine::run`] takes three trait impls, which is the
//! right surface for a Rust caller extending the schema. It is the
//! wrong surface for two other callers:
//!
//! * a compute device, which cannot invoke Rust trait objects;
//! * a Python or C caller, which has no way to name a Rust type.
//!
//! [`Family`] closes that gap. It is a plain enum naming one of the
//! three worked examples of [`crate::examples`] together with its
//! hyperparameters, so it can be serialised, sent to a GPU, or built
//! from a Python `dict`. Running a [`Family`] on the host produces a
//! value bit-identical to running the corresponding trait impls
//! through [`crate::engine::Engine::run`]; `tests/family.rs` asserts
//! exactly that.
//!
//! The host implementation here is also the optimised one. Where
//! [`crate::engine::Engine::run`] must materialise a `Vec<O>` of
//! whatever type the forward model returns, [`Family::run`] knows the
//! observation is scalar or a fixed-width vector and writes it into a
//! flat buffer, so a `d`-dimensional ensemble of `N` draws costs one
//! allocation rather than `N`.

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::engine::fork_rng;
use crate::examples::{bistable, gaussian, markov};
use crate::forward::ForwardModel;
use crate::invariance::Invariance;
use crate::perturbation::Perturbation;
use crate::reduce;
use crate::report::{Execution, Report};
use crate::{Error, Result};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// One of the built-in worked examples, fully specified.
///
/// The `null_parameter` of all three is `0.0`, matching the C2
/// contract of SCHEMA §3: at zero intensity the state is returned
/// unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum Family {
    /// Gaussian shift in `R^d`: `S' = s + sigma * N(0, I)` with
    /// `sigma ~ U[0, sigma_max]`, identity forward model, negative
    /// empirical dispersion as the invariance.
    Gaussian {
        /// Base state `s`. Its length fixes `d`.
        base: Vec<f64>,
        /// Upper end of the uniform intensity `rho`.
        sigma_max: f64,
    },
    /// Bistable double-well marble: one Euler-Maruyama Langevin step
    /// in `V(x) = (x^2 - 1)^2`, sign-of-well readout, polarisation as
    /// the invariance.
    Bistable {
        /// Initial position.
        x0: f64,
        /// Euler-Maruyama step size.
        dt: f64,
        /// Upper end of the uniform intensity `rho`.
        theta_max: f64,
    },
    /// Finite-state Markov chain: with probability `theta` the label is
    /// replaced by a uniform draw on `0..k`, indicator forward map,
    /// tail survival as the invariance.
    Markov {
        /// Alphabet size.
        k: u32,
        /// Starting label.
        start: u32,
        /// Label whose survival is measured.
        base_label: u32,
        /// Upper end of the uniform intensity `rho`.
        theta_max: f64,
    },
}

impl Family {
    /// Stable tag for this family, matching the serde `family` field.
    pub fn name(&self) -> &'static str {
        match self {
            Family::Gaussian { .. } => "gaussian",
            Family::Bistable { .. } => "bistable",
            Family::Markov { .. } => "markov",
        }
    }

    /// Name of the invariance functional this family is paired with
    /// (SCHEMA §6 row `functional`).
    pub fn functional(&self) -> &'static str {
        match self {
            Family::Gaussian { .. } => "negative_dispersion",
            Family::Bistable { .. } => "polarisation",
            Family::Markov { .. } => "tail_survival",
        }
    }

    /// Null parameter `theta_0`, for the C2 cross-check against
    /// [`crate::config::Intensity::null_parameter`].
    pub fn null_parameter(&self) -> serde_json::Value {
        serde_json::json!(0.0)
    }

    /// Observation dimension: `d` for the Gaussian family, `1` for the
    /// two scalar families.
    pub fn obs_dim(&self) -> usize {
        match self {
            Family::Gaussian { base, .. } => base.len(),
            _ => 1,
        }
    }

    /// Validate the hyperparameters.
    ///
    /// Catches the cases the trait impls would otherwise turn into a
    /// panic inside `rand_distr`: a non-positive alphabet, a negative
    /// intensity bound, a mixing probability above one.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::InvalidFamily(m.to_string()));
        match self {
            Family::Gaussian { base, sigma_max } => {
                if base.is_empty() {
                    return bad("gaussian: base state must have at least one coordinate");
                }
                if !base.iter().all(|x| x.is_finite()) {
                    return bad("gaussian: base state must be finite");
                }
                if !(sigma_max.is_finite() && *sigma_max >= 0.0) {
                    return bad("gaussian: sigma_max must be finite and >= 0");
                }
            }
            Family::Bistable { x0, dt, theta_max } => {
                if !x0.is_finite() {
                    return bad("bistable: x0 must be finite");
                }
                if !(dt.is_finite() && *dt > 0.0) {
                    return bad("bistable: dt must be finite and > 0");
                }
                if !(theta_max.is_finite() && *theta_max >= 0.0) {
                    return bad("bistable: theta_max must be finite and >= 0");
                }
            }
            Family::Markov {
                k,
                start,
                base_label,
                theta_max,
            } => {
                if *k == 0 {
                    return bad("markov: alphabet size k must be >= 1");
                }
                if start >= k {
                    return bad("markov: start must be < k");
                }
                if base_label >= k {
                    return bad("markov: base_label must be < k");
                }
                if !(theta_max.is_finite() && (0.0..=1.0).contains(theta_max)) {
                    return bad("markov: theta_max must lie in [0, 1]");
                }
            }
        }
        Ok(())
    }

    /// Run the plug-in estimator for this family.
    ///
    /// Dispatches on `cfg.backend`: the host paths are bit-identical to
    /// [`crate::engine::Engine::run`] on the corresponding trait impls,
    /// and [`Backend::Gpu`](crate::config::Backend::Gpu) runs on a compute device under the
    /// looser contract documented in [`crate::gpu`].
    pub fn run(&self, cfg: &Config) -> Result<Report> {
        self.validate()?;
        cfg.validate_version()?;
        if cfg.n == 0 {
            return Err(Error::EmptyEnsemble);
        }
        if self.null_parameter() != cfg.intensity.null_parameter {
            return Err(Error::NullParameterMismatch {
                config: cfg.intensity.null_parameter.to_string(),
                perturbation: self.null_parameter().to_string(),
            });
        }
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

        if cfg.backend.is_device() {
            #[cfg(feature = "gpu")]
            {
                return crate::gpu::run_family(self, cfg);
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(Error::UnsupportedBackend {
                    backend: "gpu",
                    reason: "crate was built without the `gpu` feature".to_string(),
                });
            }
        }

        let (value, exec) = self.run_host(cfg);
        Ok(self.finish(value, cfg, exec))
    }

    /// Wrap a computed value in a fully-provenanced [`Report`].
    ///
    /// Public so the GPU backend can share the identical bound and
    /// provenance arithmetic.
    pub fn finish(&self, value: f64, cfg: &Config, exec: Execution) -> Report {
        let l = cfg.lipschitz.forward_l.or_else(|| self.forward_lipschitz());
        crate::engine::finish_report(
            Report::raw(
                value,
                self.functional(),
                cfg.n,
                cfg.seed,
                cfg.reduction.clone(),
            ),
            cfg,
            l,
            Some(exec),
        )
    }

    /// Host evaluation: flat ensemble storage, optional threading,
    /// vectorised reduction.
    fn run_host(&self, cfg: &Config) -> (f64, Execution) {
        let exec = Execution::host(cfg.backend, cfg.n);
        let value = match self {
            Family::Gaussian { base, sigma_max } => run_gaussian(base, *sigma_max, cfg),
            Family::Bistable { x0, dt, theta_max } => {
                let f = bistable::Langevin {
                    dt: *dt,
                    theta_max: *theta_max,
                };
                let s0 = bistable::Marble { x: *x0 };
                run_scalar(cfg, |rng| {
                    let theta = f.sample_theta(rng);
                    bistable::WellOccupancy.eval(&f.apply(&s0, &theta, rng))
                })
            }
            Family::Markov {
                k,
                start,
                base_label,
                theta_max,
            } => {
                let f = markov::UniformMixing {
                    k: *k,
                    theta_max: *theta_max,
                };
                let s0 = markov::Label { i: *start };
                let readout = markov::BaseIndicator {
                    base_label: *base_label,
                };
                run_scalar(cfg, |rng| {
                    let theta = f.sample_theta(rng);
                    readout.eval(&f.apply(&s0, &theta, rng))
                })
            }
        };
        (value, exec)
    }

    /// Declared `W_1`-Lipschitz constant of this family's invariance,
    /// where one exists.
    pub fn lipschitz_w1(&self) -> Option<f64> {
        match self {
            Family::Gaussian { .. } => gaussian::NegDispersion.lipschitz_w1(),
            Family::Bistable { .. } => bistable::Polarisation.lipschitz_w1(),
            Family::Markov { .. } => markov::Survival.lipschitz_w1(),
        }
    }

    /// Declared Lipschitz constant `L` of this family's forward model.
    pub fn forward_lipschitz(&self) -> Option<f64> {
        match self {
            Family::Gaussian { base, .. } => {
                ForwardModel::<_, crate::examples::Vector>::lipschitz(&gaussian::Identity {
                    d: base.len(),
                })
            }
            Family::Bistable { .. } => bistable::WellOccupancy.lipschitz(),
            Family::Markov { .. } => markov::BaseIndicator { base_label: 0 }.lipschitz(),
        }
    }
}

/// `true` when the draw loop for `cfg` should use the thread pool.
///
/// Only reachable with the `parallel` feature on; without it there is no
/// pool to dispatch to.
#[cfg(feature = "parallel")]
#[inline]
fn threaded(cfg: &Config) -> bool {
    cfg!(feature = "parallel")
        && cfg.n >= crate::engine::PARALLEL_MIN
        && cfg.backend != crate::config::Backend::Scalar
}

/// Gaussian family, `N x d` ensemble in one row-major allocation.
///
/// The body of [`gaussian::GaussianShift::apply`] is inlined so the
/// perturbed state is written straight into its row. The RNG is drawn
/// in exactly the order the trait impl draws it (one uniform for
/// `theta`, then `d` normals in coordinate order), which is what makes
/// the result bit-identical to
/// [`crate::engine::Engine::run`].
fn run_gaussian(base: &[f64], sigma_max: f64, cfg: &Config) -> f64 {
    use rand_distr::{Distribution, Normal, Uniform};

    let n = cfg.n as usize;
    let d = base.len();
    let mut flat = vec![0.0f64; n * d];
    let seed = cfg.seed;

    let fill = |i: usize, row: &mut [f64]| {
        let mut rng = fork_rng(seed, i as u64);
        let theta = Uniform::new_inclusive(0.0, sigma_max).sample(&mut rng);
        let normal = Normal::new(0.0, theta).expect("sigma >= 0");
        for (j, v) in base.iter().enumerate() {
            row[j] = v + normal.sample(&mut rng);
        }
    };

    #[cfg(feature = "parallel")]
    if threaded(cfg) {
        flat.par_chunks_mut(d)
            .enumerate()
            .for_each(|(i, row)| fill(i, row));
    } else {
        flat.chunks_mut(d)
            .enumerate()
            .for_each(|(i, row)| fill(i, row));
    }
    #[cfg(not(feature = "parallel"))]
    flat.chunks_mut(d)
        .enumerate()
        .for_each(|(i, row)| fill(i, row));

    // Negative sum over coordinates of the empirical variance,
    // accumulated in coordinate order so the outer `+=` sequence
    // matches `gaussian::NegDispersion::measure`.
    let mut col = vec![0.0f64; n];
    let mut scratch = Vec::with_capacity(n / 2 + 1);
    let mut total = 0.0f64;
    for k in 0..d {
        for i in 0..n {
            col[i] = flat[i * d + k];
        }
        let mean = reduce::mean_into(&col, &mut scratch);
        total += reduce::sum_sq_dev_into(&col, mean, &mut scratch) / n as f64;
    }
    -total
}

/// Scalar-observation families: flat `Vec<f64>` ensemble, mean
/// invariance.
fn run_scalar<D>(cfg: &Config, draw: D) -> f64
where
    D: Fn(&mut crate::Rng) -> f64 + Sync,
{
    let n = cfg.n as usize;
    let mut ys = vec![0.0f64; n];
    let seed = cfg.seed;
    let one = |i: usize, y: &mut f64| {
        let mut rng = fork_rng(seed, i as u64);
        *y = draw(&mut rng);
    };

    #[cfg(feature = "parallel")]
    if threaded(cfg) {
        ys.par_iter_mut().enumerate().for_each(|(i, y)| one(i, y));
    } else {
        ys.iter_mut().enumerate().for_each(|(i, y)| one(i, y));
    }
    #[cfg(not(feature = "parallel"))]
    ys.iter_mut().enumerate().for_each(|(i, y)| one(i, y));

    reduce::mean(&ys)
}
