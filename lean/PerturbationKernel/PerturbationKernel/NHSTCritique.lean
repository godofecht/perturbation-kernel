/-
Perturbative-invariantist critique of null-hypothesis significance testing
(NHST) — the headline negative result of the conversation accompanying
paper [24] (*Perturbative Invariantism*, v0.1) and its companion
*A Measure-Theoretic Schema for Perturbation Kernels*.

# Argument (as developed in the discussion).

1. **Perturbative invariantism (paper [24], §1).** A Bool-valued claim `C`
   about a system is *real* iff it survives every admissible perturbation;
   otherwise `C` is a *parameterisation* of the experimental setup.

2. **NHST as a projection.** A null-hypothesis test is the composite
   `D ─T→ T(data) ─{H_0, H_1}→ reject / fail-to-reject`. Frequentist or
   Bayesian, the test conclusion is a `Bool`-valued function on the sample
   space.

3. **Five admissible perturbations of the test setup.** We name them
   `(P_1)` (test statistic), `(P_2)` (alternative), `(P_3)` (resampling),
   `(P_4)` (model-space extension), `(P_5)` (Wasserstein substitution of
   `D`). They are introduced syntactically below and the headline result
   uses `(P_3)`.

4. **Headline theorem (`rejection_not_perturbation_invariant`).** Under
   `H_0`, with rejection rule "p-value below `α`" and test statistic with
   continuous null CDF, the rejection event has `H_0`-probability exactly
   `α` (uniformity of p-values). For two i.i.d. draws under `H_0` the
   product measure assigns mass `2 α (1 - α) > 0` to the disagreement
   event. Hence the rejection event is **not** in the σ-algebra of
   perturbation-invariant events under `(P_3)`.

5. **Corollary (`rejectTest_is_parameterisation`).** NHST conclusions are
   parameterisations of the test setup, not real claims about `D`. This
   is Def. 24.X of paper [24], here surfaced as a one-line statement
   delegating to the headline theorem.

# `sorry` budget.

Zero. The standard "p-value uniformity under `H_0`" fact would normally
be discharged via Mathlib's continuous-CDF infrastructure (not yet
available at this toolchain version); we instead carry it as a hypothesis
on the input data (a field of `TestSetup`), which is the right abstraction
boundary anyway: the critique applies to any test whose null rejection
mass is `α`, however that property is established.

# Paper-to-Lean dictionary.

  * `TestSetup`                — Def. 2 of the NHST critique.
  * `Perturbation`             — Def. 3: a measurable self-map.
  * `PerturbationInvariant`    — Def. 24.X of paper [24].
  * `swapPerturbation`         — `(P_3)` of the critique.
  * `rejection_not_perturbation_invariant` — the headline theorem.
  * `rejectTest_is_parameterisation`        — Cor. 24.X of paper [24].
-/

import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.MeasureTheory.Measure.Prod
import Mathlib.Probability.ConditionalProbability
import Mathlib.Probability.Independence.Basic
import PerturbationKernel.Basic

set_option linter.dupNamespace false
set_option linter.unusedSimpArgs false

noncomputable section
open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

-- Make set membership classically decidable in this file. `decide (x ∈ R)`
-- otherwise fails to elaborate because R is an arbitrary `Set X` with no
-- propagated `Decidable` instance.
attribute [local instance] Classical.propDecidable

namespace PerturbativeNHSTCritique

universe u

/-! ## §1. The data-generating process and the rejection event. -/

/-- The α-level test setup of §2 of the critique: a data measure `D`, a
chosen significance level `α ∈ (0, 1)`, and the measurable rejection
event `R ⊆ X` carved by the test statistic + p-value rule. -/
structure TestSetup (X : Type u) [MeasurableSpace X] where
  /-- The data-generating distribution `D` (treated as the null `H_0`
      distribution; the headline theorem is stated under `H_0`). -/
  D            : ProbabilityMeasure X
  /-- The significance threshold `α`. -/
  α            : ℝ≥0∞
  /-- `α ∈ (0, 1)`; needed for `2 α (1 - α) > 0`. -/
  α_pos        : 0 < α
  α_lt_one     : α < 1
  /-- The rejection region carved by the test statistic + p-value rule:
      `R = { x | pValue(T x) < α }`. -/
  R            : Set X
  /-- Measurability of the rejection event. -/
  R_meas       : MeasurableSet R
  /-- **Uniformity of p-values under `H_0`.** If `T` has continuous null
      CDF `F₀`, then `F₀ ∘ T ∼ Uniform[0,1]` and hence
      `D({pValue < α}) = α`. This is the standard fact that would invoke
      Mathlib's continuous-CDF infrastructure once it lands; in the
      meantime we record it as a hypothesis on the input data. -/
  pValue_uniform_under_null :
    (D : Measure X) R = α

namespace TestSetup

variable {X : Type u} [MeasurableSpace X]

/-- The Bool-valued test conclusion: `true` = reject `H_0`, `false` =
fail to reject. -/
def rejectTest (𝓣 : TestSetup X) : X → Bool := fun x => decide (x ∈ 𝓣.R)

lemma rejectTest_true_iff (𝓣 : TestSetup X) (x : X) :
    𝓣.rejectTest x = true ↔ x ∈ 𝓣.R := by
  simp [rejectTest]

lemma setOf_rejectTest_eq (𝓣 : TestSetup X) :
    {x : X | 𝓣.rejectTest x = true} = 𝓣.R := by
  ext x; simpa using 𝓣.rejectTest_true_iff x

end TestSetup

/-! ## §2. Admissible perturbations. -/

/-- An admissible perturbation of the sample space `Y` is a measurable
self-map. -/
structure Perturbation (Y : Type*) [MeasurableSpace Y] where
  /-- The underlying map. -/
  toFun       : Y → Y
  /-- Measurability of the map. -/
  measurable  : Measurable toFun

namespace Perturbation

variable {Y : Type*} [MeasurableSpace Y]

instance : CoeFun (Perturbation Y) (fun _ => Y → Y) := ⟨Perturbation.toFun⟩

end Perturbation

/-- **`(P_3)` — Resampling perturbation.** The swap-of-i.i.d.-draws
perturbation on `X × X`. -/
def swapPerturbation (X : Type u) [MeasurableSpace X] : Perturbation (X × X) where
  toFun      := fun p => (p.2, p.1)
  measurable := measurable_swap

/-! ## §3. The perturbation-invariance criterion (Def. 24.X of paper [24]). -/

/-- A `Bool`-valued claim `C : Y → Bool` is **perturbation-invariant**
under `P : Perturbation Y` with respect to `μ : Measure Y` iff the
disagreement event `{y | C y ≠ C (P y)}` is `μ`-null. -/
def PerturbationInvariant {Y : Type*} [MeasurableSpace Y]
    (μ : Measure Y) (P : Perturbation Y) (C : Y → Bool) : Prop :=
  μ {y | C y ≠ C (P y)} = 0

/-! ## §4. The headline theorem. -/

namespace TestSetup

variable {X : Type u} [MeasurableSpace X]

/-- The `X × X`-claim used by `(P_3)`: read the rejection bit from the
**first** coordinate. -/
def rejectTestPair (𝓣 : TestSetup X) : X × X → Bool :=
  fun p => 𝓣.rejectTest p.1

lemma rejectTestPair_swap (𝓣 : TestSetup X) (p : X × X) :
    𝓣.rejectTestPair ((swapPerturbation X).toFun p) = 𝓣.rejectTest p.2 := rfl

/-- The disagreement event expressed as a rectangle union. -/
lemma disagreement_eq (𝓣 : TestSetup X) :
    {p : X × X | 𝓣.rejectTestPair p
                  ≠ 𝓣.rejectTestPair ((swapPerturbation X).toFun p)}
      = (𝓣.R ×ˢ 𝓣.Rᶜ) ∪ (𝓣.Rᶜ ×ˢ 𝓣.R) := by
  ext ⟨x, y⟩
  simp only [Set.mem_setOf_eq, rejectTestPair, rejectTestPair_swap,
             Set.mem_union, Set.mem_prod, Set.mem_compl_iff, ne_eq,
             TestSetup.rejectTest_true_iff]
  constructor
  · intro h
    by_cases hx : x ∈ 𝓣.R
    · have hy : y ∉ 𝓣.R := by
        intro hy
        apply h
        show decide (x ∈ 𝓣.R) = decide (y ∈ 𝓣.R)
        simp [hx, hy]
      exact Or.inl ⟨hx, hy⟩
    · have hy : y ∈ 𝓣.R := by
        by_contra hy
        apply h
        show decide (x ∈ 𝓣.R) = decide (y ∈ 𝓣.R)
        simp [hx, hy]
      exact Or.inr ⟨hx, hy⟩
  · rintro (⟨hx, hy⟩ | ⟨hx, hy⟩) h
    · have hne : decide (x ∈ 𝓣.R) ≠ decide (y ∈ 𝓣.R) := by
        simp [hx, hy]
      exact hne h
    · have hne : decide (x ∈ 𝓣.R) ≠ decide (y ∈ 𝓣.R) := by
        simp [hx, hy]
      exact hne h

/-- Measure of the rejection complement: `D(Rᶜ) = 1 - α`. -/
lemma measure_R_compl (𝓣 : TestSetup X) :
    (𝓣.D : Measure X) 𝓣.Rᶜ = 1 - 𝓣.α := by
  have hprob : IsProbabilityMeasure (𝓣.D : Measure X) := 𝓣.D.2
  have hR : (𝓣.D : Measure X) 𝓣.R = 𝓣.α := 𝓣.pValue_uniform_under_null
  have hne : (𝓣.D : Measure X) 𝓣.R ≠ ∞ := by
    rw [hR]
    exact (lt_of_lt_of_le 𝓣.α_lt_one le_top).ne
  have := measure_compl 𝓣.R_meas hne
  rw [this, measure_univ, hR]

/-- Measure of `R × Rᶜ` under `D × D`: this equals `α (1 - α)`. -/
lemma prod_measure_R_Rcompl (𝓣 : TestSetup X) :
    ((𝓣.D : Measure X).prod (𝓣.D : Measure X)) (𝓣.R ×ˢ 𝓣.Rᶜ)
      = 𝓣.α * (1 - 𝓣.α) := by
  rw [Measure.prod_prod, 𝓣.pValue_uniform_under_null, 𝓣.measure_R_compl]

/-- Measure of `Rᶜ × R` under `D × D`: this equals `(1 - α) α`. -/
lemma prod_measure_Rcompl_R (𝓣 : TestSetup X) :
    ((𝓣.D : Measure X).prod (𝓣.D : Measure X)) (𝓣.Rᶜ ×ˢ 𝓣.R)
      = (1 - 𝓣.α) * 𝓣.α := by
  rw [Measure.prod_prod, 𝓣.pValue_uniform_under_null, 𝓣.measure_R_compl]

/-- The two rectangles are disjoint. -/
lemma disjoint_R_Rcompl_Rcompl_R (𝓣 : TestSetup X) :
    Disjoint (𝓣.R ×ˢ 𝓣.Rᶜ : Set (X × X)) (𝓣.Rᶜ ×ˢ 𝓣.R) := by
  rw [Set.disjoint_iff]
  rintro ⟨x, y⟩ ⟨⟨hx1, _⟩, ⟨hx2, _⟩⟩
  exact (hx2 hx1).elim

/-- **Measure of the disagreement event:** `2 α (1 - α)`. -/
lemma measure_disagreement_eq (𝓣 : TestSetup X) :
    ((𝓣.D : Measure X).prod (𝓣.D : Measure X))
      {p : X × X | 𝓣.rejectTestPair p
                     ≠ 𝓣.rejectTestPair ((swapPerturbation X).toFun p)}
      = 2 * (𝓣.α * (1 - 𝓣.α)) := by
  rw [𝓣.disagreement_eq]
  rw [measure_union 𝓣.disjoint_R_Rcompl_Rcompl_R
        (𝓣.R_meas.compl.prod 𝓣.R_meas)]
  rw [𝓣.prod_measure_R_Rcompl, 𝓣.prod_measure_Rcompl_R]
  ring

/-- **`2 α (1 - α) > 0`** whenever `α ∈ (0, 1)`. -/
lemma two_α_one_sub_α_pos (𝓣 : TestSetup X) : 0 < 2 * (𝓣.α * (1 - 𝓣.α)) := by
  have h1 : 0 < 𝓣.α := 𝓣.α_pos
  have h2 : 0 < 1 - 𝓣.α := by
    have hlt : 𝓣.α < 1 := 𝓣.α_lt_one
    exact tsub_pos_of_lt hlt
  have h3 : 0 < 𝓣.α * (1 - 𝓣.α) := ENNReal.mul_pos h1.ne' h2.ne'
  have h4 : (0 : ℝ≥0∞) < 2 := by norm_num
  exact ENNReal.mul_pos h4.ne' h3.ne'

/-- **THE HEADLINE THEOREM.** Under the null distribution `D`, with the
rejection rule of `𝓣`, the rejection-on-pair claim `rejectTestPair` is
**not** perturbation-invariant under `(P_3) = swapPerturbation` with
respect to the joint i.i.d. measure `D × D`. The disagreement event has
mass exactly `2 α (1 - α) > 0`. -/
theorem rejection_not_perturbation_invariant (𝓣 : TestSetup X) :
    ¬ PerturbationInvariant
        ((𝓣.D : Measure X).prod (𝓣.D : Measure X))
        (swapPerturbation X)
        𝓣.rejectTestPair := by
  intro hInv
  -- `PerturbationInvariant` says the disagreement event has measure 0.
  -- We have just computed it equals `2 α (1 - α) > 0`. Contradiction.
  have hmeas := 𝓣.measure_disagreement_eq
  rw [hInv] at hmeas
  exact (𝓣.two_α_one_sub_α_pos).ne hmeas

/-- **Corollary (Def. 24.X of paper [24]).** Because `rejectTest`
violates perturbation invariance under the admissible `(P_3)`, it is a
*parameterisation* of the test setup, **not** a real claim about the
data-generating process `D`. -/
theorem rejectTest_is_parameterisation (𝓣 : TestSetup X) :
    ∃ (P : Perturbation (X × X)) (μ : Measure (X × X)),
      ¬ PerturbationInvariant μ P 𝓣.rejectTestPair := by
  refine ⟨swapPerturbation X, (𝓣.D : Measure X).prod (𝓣.D : Measure X), ?_⟩
  exact 𝓣.rejection_not_perturbation_invariant

end TestSetup

end PerturbativeNHSTCritique
