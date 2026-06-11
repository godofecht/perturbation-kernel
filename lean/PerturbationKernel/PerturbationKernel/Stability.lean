/-
Second-order stability of the invariance value — Thm 5.4 of the paper.

Statement: under Assumption 5.1 (`L`-Lipschitz forward model, `Λ`-Wasserstein-
Lipschitz invariance functional, `C`-Lipschitz-in-parameter kernel), changing
the intensity measure `ρ ↦ ρ'` by Wasserstein distance `δ` moves the
invariance value by at most `Λ · L · C · δ`.

Mathlib gap: as of Lean 4.30.0-rc2 there is no canonical `W₁` Wasserstein-1
distance type in Mathlib's `MeasureTheory.Measure` namespace covering
arbitrary Polish metric spaces (the relevant material lives in
`MeasureTheory.Measure.Hausdorff` / `EReal.WassersteinDistance` only in
development branches). We therefore introduce a local placeholder
`Wass1 : Measure X → Measure X → ℝ≥0∞` and state the theorem against it.
When Mathlib lands a canonical Wasserstein distance, swap this alias for
the real definition and the statement is unchanged.
-/
import PerturbationKernel.Basic
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.Topology.MetricSpace.Lipschitz

noncomputable section
open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace PerturbationKernel

universe u v w

variable {S : Type u} {Θ : Type v} {O : Type w}
    [MeasurableSpace S] [MeasurableSpace Θ]
    [MeasurableSpace O] [TopologicalSpace O] [OpensMeasurableSpace O]
    [PseudoMetricSpace S] [PseudoMetricSpace Θ] [PseudoMetricSpace O]

/-- Local placeholder for the Wasserstein-1 distance between two measures
on a metric space. **Gap:** replace with the canonical Mathlib definition
when one exists (currently only sketched in PRs against `Mathlib/
MeasureTheory/Distance/Wasserstein` upstream). -/
opaque Wass1 {X : Type*} [PseudoMetricSpace X] [MeasurableSpace X]
    (μ ν : Measure X) : ℝ≥0∞

/-- Assumption 5.1 of the paper: declared Lipschitz data for stability. -/
structure StabilityHypotheses (K : PerturbationKernel S Θ O) where
  /-- Forward-model Lipschitz constant `L`. -/
  L : ℝ
  L_nonneg : 0 ≤ L
  forward_lipschitz : LipschitzWith ⟨L, L_nonneg⟩ K.forward.toFun
  /-- Invariance-functional Wasserstein-1 Lipschitz constant `Λ`. -/
  Λ : ℝ
  Λ_nonneg : 0 ≤ Λ
  invariance_w1_lipschitz :
    ∀ ν ν' : ProbabilityMeasure O,
      |K.invariance ν - K.invariance ν'|
        ≤ Λ * (Wass1 (ν : Measure O) (ν' : Measure O)).toReal
  /-- Kernel Lipschitz-in-parameter constant `C` (Lemma 5.3). -/
  C : ℝ
  C_nonneg : 0 ≤ C
  kernel_param_lipschitz :
    ∀ s : S, ∀ θ θ' : Θ,
      Wass1 (K.family.apply s θ) (K.family.apply s θ')
        ≤ ENNReal.ofReal (C * dist θ θ')

/-- **Theorem 5.4 (second-order stability).** Two perturbation kernels that
differ only in their intensity measures `ρ` and `ρ'` produce invariance
values whose difference is bounded by `Λ · L · C · W₁(ρ, ρ')`.

The hypothesis is exactly Assumption 5.1 of the paper. Proof: chain
Assumption 5.1 + Lemma 5.2 (pushforward contracts `W₁` under Lipschitz `F`)
+ Lemma 5.3 (mixing contracts `W₁`); statement-only here. -/
theorem stability_second_order
    (K K' : PerturbationKernel S Θ O)
    (H  : StabilityHypotheses K)
    (_H' : StabilityHypotheses K')
    (hSameFamily : K.family.toKernel = K'.family.toKernel)
    (_hSameNull  : K.family.null    = K'.family.null)
    (_hSameForward : K.forward = K'.forward)
    (_hSameInv : K.invariance = K'.invariance)
    (s : S)
    (hP  : ∀ s θ, IsProbabilityMeasure (K.family.apply s θ))
    (hP' : ∀ s θ, IsProbabilityMeasure (K'.family.apply s θ)) :
    |K.invarianceValue s hP - K'.invarianceValue s hP'|
      ≤ H.Λ * H.L * H.C * (Wass1 K.family.rho K'.family.rho).toReal := by
  sorry

end PerturbationKernel
