//! Invariance functional trait (SCHEMA §4.3; Paper Def. 3.5).
//!
//! An [`Invariance<O>`] computes the scalar `Phi` on the empirical
//! observation measure carried by the ensemble of `Y_i = F(S_i)`.
//! It MUST be permutation-invariant up to the reduction policy's
//! last-bit floating-point rounding (SCHEMA §8 D3).

use crate::report::Report;

/// Invariance functional `Phi: M_1(O) -> R` (SCHEMA §4.3).
pub trait Invariance<O> {
    /// Apply `Phi` to the empirical measure carried by `ensemble`
    /// (SCHEMA §4.3 / Paper Def. 3.5).
    ///
    /// MUST be permutation-invariant: `measure` is a function of the
    /// multiset, not the sequence. Any aggregation MUST follow the
    /// reduction order declared in [`crate::config::Config`] (SCHEMA
    /// §8 D3).
    fn measure(&self, ensemble: &[O]) -> Report;

    /// Declared `W_1`-Lipschitz constant `Lambda`
    /// (Paper Assumption 5.1, SCHEMA §3 C4).
    fn lipschitz_w1(&self) -> Option<f64> {
        None
    }

    /// Tag identifying which `Phi` produced the value (SCHEMA §6
    /// row `functional`). The default `"unspecified"` MUST be
    /// overridden by every published functional.
    fn name(&self) -> &str {
        "unspecified"
    }
}
