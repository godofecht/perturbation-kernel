/-
Non-asymptotic error bound for the plug-in estimator — Thm 7.3 of the paper.

We state the sample-complexity inequality:

  N ≥ max( Λ² D² / (2 ε²) · ln(2/η),  N_bias(ε) )
    ⟹  ℙ( |Φ̂_N(s) − Φ(ν_s)| ≤ ε )  ≥  1 − η

with the Fournier–Guillin bias term `N_bias` parameterised by the
observation dimension `d_obs` (Thm 7.3(b)).

Both the empirical estimator and the probability law it lives on need
joint-product / iid machinery from `Mathlib.Probability` to be set up; we
keep that abstract here and write the *statement* of Thm 7.3(c) so the
file type-checks. Proof: `sorry`.
-/
import PerturbationKernel.Basic
import PerturbationKernel.Stability
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.Analysis.SpecialFunctions.Log.Basic

noncomputable section
open MeasureTheory Real
open scoped ENNReal NNReal

namespace PerturbationKernel

universe u v w

variable {S : Type u} {Θ : Type v} {O : Type w}
    [MeasurableSpace S] [MeasurableSpace Θ]
    [MeasurableSpace O] [TopologicalSpace O] [OpensMeasurableSpace O]
    [PseudoMetricSpace S] [PseudoMetricSpace Θ] [PseudoMetricSpace O]

/-! ### Empirical estimator (Def. 7.1).

We do not construct the joint product measure here; instead we treat the
estimator as a placeholder real-valued random variable indexed by
`(K, s, N, ω)` for `ω` an iid sample. -/

/-- Abstract plug-in estimator `Φ̂_N(s)` of Def. 7.1; lifted into the
"deviation event" formulation that Thm 7.3 controls. The precise
construction (product measure + empirical distribution + `Φ` applied)
belongs to a future `PerturbationKernel.Empirical` module. -/
opaque plugInDeviation
    (K : PerturbationKernel S Θ O) (s : S) (N : ℕ) (ε : ℝ) : ℝ

/-- Sample-complexity floor for the bias term (Thm 7.3(b), Fournier–Guillin).
For `d_obs > 2` this is `O(ε^(−d_obs))`; for `d_obs ≤ 2` it absorbs the
`log N` correction. Statement-only placeholder. -/
def biasFloor (d_obs : ℕ) (ε : ℝ) : ℕ := by
  classical
  exact if d_obs ≤ 2 then
    Nat.ceil (1 / ε ^ 2 * Real.log (1 / ε) ^ 2)
  else
    Nat.ceil (1 / ε ^ d_obs)

/-- Stochastic part of the sample-complexity floor (McDiarmid arm of
Thm 7.3(a) / (c)): `⌈ Λ² D² / (2 ε²) · ln(2/η) ⌉`. -/
def stochasticFloor (Λ D ε η : ℝ) : ℕ :=
  Nat.ceil ((Λ ^ 2 * D ^ 2 / (2 * ε ^ 2)) * Real.log (2 / η))

/-- **Theorem 7.3(c) (non-asymptotic sample-complexity floor).** Under
Assumption 5.1 and bounded observation diameter `D`, taking the sample size
above both the McDiarmid floor and the Fournier–Guillin bias floor delivers
the `(ε, η)` accuracy / confidence guarantee for the plug-in estimator.

The conclusion is stated as a bound on `plugInDeviation`; the actual
probability statement lives in `PerturbationKernel.Empirical` (TODO).
Body: `sorry`. -/
theorem sampleComplexity_plugIn
    (K : PerturbationKernel S Θ O)
    (H : StabilityHypotheses K)
    (s : S)
    (D : ℝ) (_hD_nonneg : 0 ≤ D)
    (_hDiam : True)   -- placeholder: `diam (support ν_s) ≤ D`
    (ε η : ℝ) (_hε_pos : 0 < ε) (_hη_pos : 0 < η) (_hη_lt : η < 1)
    (d_obs : ℕ)
    (N : ℕ)
    (_hN_stoch : stochasticFloor H.Λ D ε η ≤ N)
    (_hN_bias  : biasFloor d_obs ε ≤ N) :
    |plugInDeviation K s N ε| ≤ ε := by
  sorry

end PerturbationKernel
