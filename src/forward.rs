//! Forward-model trait (SCHEMA §4.2; Paper Def. 3.2).
//!
//! A [`ForwardModel<S, O>`] is the measurable map `F: S -> O` whose
//! pushforward `nu_s = F_# mu_s` is the induced observation measure
//! evaluated by the [`Invariance`](crate::invariance::Invariance)
//! functional.

/// Deterministic forward model `F: S -> O` (SCHEMA §4.2).
///
/// `eval` MUST be pure: stochastic readouts MUST be folded into
/// the state via the [`Perturbation`](crate::perturbation::Perturbation),
/// keeping `F` a measurable map (SCHEMA §4.2 Requirements).
pub trait ForwardModel<S, O> {
    /// Compute `F(s)` (SCHEMA §4.2).
    fn eval(&self, s: &S) -> O;

    /// Declared Lipschitz constant `L` w.r.t. the metric `d_O`
    /// (Paper Assumption 5.1, SCHEMA §3 C3).
    ///
    /// `None` means "not declared"; without it the engine cannot
    /// surface `error_bound.available = true` in the
    /// [`crate::report::Report`] (SCHEMA §6).
    fn lipschitz(&self) -> Option<f64> {
        None
    }
}
