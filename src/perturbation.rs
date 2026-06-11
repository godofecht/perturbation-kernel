//! Perturbation family trait (SCHEMA §4.1; Paper Def. 3.1).
//!
//! A type implementing [`Perturbation<S>`] realises the Markov kernel
//! `P((s, theta), .)`, the intensity sampler `rho`, and the null
//! parameter `theta_0` that makes `apply(s, null(), rng) == s` in law
//! (SCHEMA §3 C2, Paper Def. 3.1).

use crate::Rng;

/// Perturbation family `P` and intensity `rho` (SCHEMA §4.1).
///
/// Implementations MUST be pure functions of their arguments: all
/// randomness flows through `rng`, which is owned by the engine and
/// forked per-index from the [`Config::seed`](crate::config::Config)
/// to satisfy the determinism contract (SCHEMA §8 D1-D2).
pub trait Perturbation<S> {
    /// Intensity parameter type. Serialised JSON for cross-checking
    /// against [`crate::config::Config::intensity::null_parameter`]
    /// (SCHEMA §5).
    type Theta: serde::Serialize + serde::de::DeserializeOwned + Clone;

    /// Null parameter `theta_0` (Paper Def. 3.1; SCHEMA §3 C2).
    ///
    /// Contract: `apply(s, &null(), rng) == s` in distribution. This
    /// is what makes "perturb then recover" a meaningful claim.
    fn null(&self) -> Self::Theta;

    /// Draw `theta ~ rho` (SCHEMA §4.1).
    ///
    /// MUST consume the `rng` deterministically: with a fixed seed and
    /// a fixed sequence of calls the same sequence of `theta`s is
    /// produced. No global state may be read.
    fn sample_theta(&self, rng: &mut Rng) -> Self::Theta;

    /// Sample `S' ~ P((s, theta), .)` (SCHEMA §4.1; Paper Def. 3.1).
    ///
    /// MUST be a pure function of `(s, theta, rng stream)`. The engine
    /// supplies a per-index substream (SCHEMA §8 D2) so the output is
    /// independent of evaluation order.
    fn apply(&self, s: &S, theta: &Self::Theta, rng: &mut Rng) -> S;
}
