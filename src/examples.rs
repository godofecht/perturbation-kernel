//! Worked examples of the perturbation-kernel object (SCHEMA §4, Paper §3.2).
//!
//! Three instances live here, each demonstrating a different shape of
//! state space:
//!
//! 1. **Gaussian shift in `R^d`** ([`gaussian`]). State is a fixed-size
//!    vector; perturbation adds a zero-mean Gaussian noise of intensity
//!    `theta = sigma`; the forward model is the identity; the invariance
//!    is the empirical variance, treated as `-dispersion` (Paper §5
//!    canonical functional).
//!
//! 2. **Bistable double-well marble** ([`bistable`]). State is the
//!    sign of the well the marble settles in after a noisy nudge.
//!    Demonstrates that recovery (the C2 contract) is meaningful
//!    even for highly non-linear dynamics: at intensity zero the
//!    marble stays put.
//!
//! 3. **Finite-state Markov chain** ([`markov`]). State is a discrete
//!    label; perturbation is an epsilon-mixing with the uniform
//!    distribution; the invariance is `1 - epsilon_eff` (a tail-survival
//!    proxy). Demonstrates the discrete-space case.
//!
//! All three are pure functions of `(s, theta, rng-stream)` and use no
//! global state, satisfying SCHEMA §8 D1.

use rand::Rng as _;
use rand_distr::{Distribution, Normal, Uniform};

use crate::engine::tree_sum;
use crate::forward::ForwardModel;
use crate::invariance::Invariance;
use crate::perturbation::Perturbation;
use crate::report::Report;
use crate::Rng;

// =====================================================================
// Example 1: Gaussian shift in R^d
// =====================================================================

/// State and observation for [`gaussian`]: a fixed-size `R^d` vector
/// (boxed slice so the dimension is data, not a generic).
pub type Vector = Box<[f64]>;

/// Module for the Gaussian-shift example (Paper Def. 3.1 instance).
pub mod gaussian {
    use super::*;

    /// Perturbation family: `S' = s + sigma * N(0, I)`. `theta = sigma`.
    /// `null = 0.0` (SCHEMA §3 C2).
    pub struct GaussianShift {
        /// Upper bound for the uniform `rho` on `[0, sigma_max]`.
        pub sigma_max: f64,
        /// Ambient dimension of `S = R^d`.
        pub d: usize,
    }

    impl Perturbation<Vector> for GaussianShift {
        type Theta = f64;
        fn null(&self) -> f64 {
            0.0
        }
        fn sample_theta(&self, rng: &mut Rng) -> f64 {
            Uniform::new_inclusive(0.0, self.sigma_max).sample(rng)
        }
        fn apply(&self, s: &Vector, theta: &f64, rng: &mut Rng) -> Vector {
            let n = Normal::new(0.0, *theta).expect("sigma >= 0");
            let mut out = vec![0.0; self.d].into_boxed_slice();
            for (i, v) in s.iter().enumerate() {
                out[i] = v + n.sample(rng);
            }
            out
        }
    }

    /// Forward model: identity `F(s) = s`. Lipschitz constant `L = 1`.
    pub struct Identity {
        /// Mirror of [`GaussianShift::d`] for declared semantics.
        pub d: usize,
    }
    impl ForwardModel<Vector, Vector> for Identity {
        fn eval(&self, s: &Vector) -> Vector {
            s.clone()
        }
        fn lipschitz(&self) -> Option<f64> {
            Some(1.0)
        }
    }

    /// Invariance functional: negative empirical variance summed over
    /// coordinates. By Paper §5 this is a dispersion-style functional;
    /// "negative" so larger means *more* invariant (Paper Def. 3.5
    /// order-reflecting property).
    pub struct NegDispersion;

    impl Invariance<Vector> for NegDispersion {
        fn measure(&self, ensemble: &[Vector]) -> Report {
            if ensemble.is_empty() {
                return Report::raw(0.0, self.name(), 0, 0, Default::default());
            }
            let d = ensemble[0].len();
            let n = ensemble.len() as f64;
            let mut total = 0.0_f64;
            for k in 0..d {
                let col: Vec<f64> = ensemble.iter().map(|v| v[k]).collect();
                let mean = tree_sum(&col) / n;
                let centred: Vec<f64> =
                    col.iter().map(|x| (x - mean) * (x - mean)).collect();
                total += tree_sum(&centred) / n;
            }
            Report::raw(-total, self.name(), ensemble.len() as u64, 0, Default::default())
        }
        fn lipschitz_w1(&self) -> Option<f64> {
            // Empirical variance is *not* globally W_1-Lipschitz on
            // unbounded R^d. The schema permits this: lipschitz_w1
            // returns `None` and `error_bound.available` ends up
            // `false` unless the caller declares the constants by hand.
            None
        }
        fn name(&self) -> &str {
            "negative_dispersion"
        }
    }
}

// =====================================================================
// Example 2: bistable double-well marble
// =====================================================================

/// Module for the bistable double-well example (Paper §3.2 instance).
pub mod bistable {
    use super::*;

    /// State: scalar position of the marble in a double-well potential
    /// `V(x) = (x^2 - 1)^2`. Wrapped in a struct so we can derive serde.
    #[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
    pub struct Marble {
        /// Position along the well axis. Wells at `+/- 1`.
        pub x: f64,
    }

    /// Perturbation family: a single Langevin step with noise
    /// intensity `theta`. `null = 0.0`.
    pub struct Langevin {
        /// Step size (Euler-Maruyama).
        pub dt: f64,
        /// Upper bound for the uniform `rho` on `[0, theta_max]`.
        pub theta_max: f64,
    }

    impl Perturbation<Marble> for Langevin {
        type Theta = f64;
        fn null(&self) -> f64 {
            0.0
        }
        fn sample_theta(&self, rng: &mut Rng) -> f64 {
            Uniform::new_inclusive(0.0, self.theta_max).sample(rng)
        }
        fn apply(&self, s: &Marble, theta: &f64, rng: &mut Rng) -> Marble {
            // dV/dx = 4 x (x^2 - 1). Single Euler-Maruyama step.
            let drift = -4.0 * s.x * (s.x * s.x - 1.0);
            let noise = if *theta == 0.0 {
                0.0
            } else {
                Normal::new(0.0, theta * self.dt.sqrt())
                    .expect("theta >= 0")
                    .sample(rng)
            };
            Marble {
                x: s.x + drift * self.dt + noise,
            }
        }
    }

    /// Forward model: `F(s) = sign(s.x)`, a binary readout of the
    /// occupied well. Lipschitz constant declared `None` (the sign
    /// function is not Lipschitz across `x = 0`).
    pub struct WellOccupancy;
    impl ForwardModel<Marble, f64> for WellOccupancy {
        fn eval(&self, s: &Marble) -> f64 {
            if s.x >= 0.0 {
                1.0
            } else {
                -1.0
            }
        }
        fn lipschitz(&self) -> Option<f64> {
            None
        }
    }

    /// Invariance functional: empirical *mean* of `F(S_i)`. With wells
    /// at `+/- 1` this lies in `[-1, 1]` and equals the polarisation;
    /// it is 1-Lipschitz in `W_1` on `{-1, +1}`.
    pub struct Polarisation;
    impl Invariance<f64> for Polarisation {
        fn measure(&self, ensemble: &[f64]) -> Report {
            if ensemble.is_empty() {
                return Report::raw(0.0, self.name(), 0, 0, Default::default());
            }
            let s = tree_sum(ensemble);
            let val = s / ensemble.len() as f64;
            Report::raw(val, self.name(), ensemble.len() as u64, 0, Default::default())
        }
        fn lipschitz_w1(&self) -> Option<f64> {
            Some(1.0)
        }
        fn name(&self) -> &str {
            "polarisation"
        }
    }
}

// =====================================================================
// Example 3: finite-state Markov chain
// =====================================================================

/// Module for the finite-state Markov chain example (Paper §3.2).
pub mod markov {
    use super::*;

    /// Discrete state: a label in `0..k`.
    #[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
    pub struct Label {
        /// Index into the state alphabet.
        pub i: u32,
    }

    /// Perturbation family: with probability `theta`, replace the
    /// current label by a uniform draw on `0..k`; otherwise keep it.
    /// `null = 0.0`.
    pub struct UniformMixing {
        /// Alphabet size `k`.
        pub k: u32,
        /// Upper bound for the uniform `rho` on `[0, theta_max] subset [0,1]`.
        pub theta_max: f64,
    }

    impl Perturbation<Label> for UniformMixing {
        type Theta = f64;
        fn null(&self) -> f64 {
            0.0
        }
        fn sample_theta(&self, rng: &mut Rng) -> f64 {
            Uniform::new_inclusive(0.0, self.theta_max).sample(rng)
        }
        fn apply(&self, s: &Label, theta: &f64, rng: &mut Rng) -> Label {
            let u: f64 = rng.gen();
            if u < *theta {
                Label {
                    i: rng.gen_range(0..self.k),
                }
            } else {
                *s
            }
        }
    }

    /// Forward model: `F(s) = (s.i == base_label) as f64`. Treats the
    /// base label as the privileged state. Lipschitz `L = 1` under
    /// the Hamming-style metric `d_O(a,b) = |a-b|`.
    pub struct BaseIndicator {
        /// Label whose survival we measure.
        pub base_label: u32,
    }
    impl ForwardModel<Label, f64> for BaseIndicator {
        fn eval(&self, s: &Label) -> f64 {
            if s.i == self.base_label {
                1.0
            } else {
                0.0
            }
        }
        fn lipschitz(&self) -> Option<f64> {
            Some(1.0)
        }
    }

    /// Invariance functional: empirical survival probability. The
    /// scalar in `[0, 1]` is `(1/N) sum_i F(S_i)`. 1-Lipschitz in
    /// `W_1` on `{0, 1}`.
    pub struct Survival;
    impl Invariance<f64> for Survival {
        fn measure(&self, ensemble: &[f64]) -> Report {
            if ensemble.is_empty() {
                return Report::raw(0.0, self.name(), 0, 0, Default::default());
            }
            let s = tree_sum(ensemble);
            let val = s / ensemble.len() as f64;
            Report::raw(val, self.name(), ensemble.len() as u64, 0, Default::default())
        }
        fn lipschitz_w1(&self) -> Option<f64> {
            Some(1.0)
        }
        fn name(&self) -> &str {
            "tail_survival"
        }
    }
}
