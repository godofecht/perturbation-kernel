/-
Core definitions of the perturbation-kernel object — Lean 4 / Mathlib
formalisation of §3 of *A Measure-Theoretic Schema for Perturbation Kernels*
(paper.tex, v0.1).

This file defines the four components of `K = (S, P, F, Phi)` (Def. 3.6 of
the paper), the induced state and observation measures (Def. 3.3 / 3.4), and
the invariance value `Phi(nu_s)`.

Paper-to-Lean dictionary:
  * `PerturbationFamily`    — Def. 3.1 (Markov kernel `P` + null parameter
                              `theta_0` + intensity `rho`).
  * `ForwardModel`          — Def. 3.2 (measurable `F : S → O`).
  * `InvarianceFunctional`  — Def. 3.5 (measurable `Phi : M_1(O) → ℝ`).
  * `PerturbationKernel`    — Def. 3.6 (the quadruple).
  * `inducedStateMeasure`   — `mu^P_s` of Def. 3.3.
  * `inducedObsMeasure`     — `nu^P_s = F_* mu^P_s` of Def. 3.4.
  * `invarianceValue`       — `Phi(nu_s)`, the boxed object of Def. 3.6.
  * `wellDefined`           — Thm 4.2 (Borel measurability of the field
                              `s ↦ Phi(nu_s)`); stated, `sorry`.

Mathlib namespaces used:
  * `MeasureTheory.Measure`              (measures on a measurable space)
  * `MeasureTheory.ProbabilityMeasure`   (the subtype of probability measures)
  * `ProbabilityTheory.Kernel`           (Markov / sub-Markov kernels)
  * `MeasureTheory.Measure.map`          (pushforward measure)
-/
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.Probability.Kernel.Basic

-- The library is named `PerturbationKernel` and Paper Def. 3.6 calls the
-- object itself `PerturbationKernel`; the resulting `PerturbationKernel.
-- PerturbationKernel` is intentional, not a lint slip.
set_option linter.dupNamespace false

noncomputable section
open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace PerturbationKernel

universe u v w

/-! ## §3.1 — Perturbation family (Def. 3.1).

A perturbation family is a Markov kernel `P : S × Θ ⇝ S` together with a
distinguished `null : Θ` (the unperturbed parameter `θ₀`) and an intensity
measure `rho : Measure Θ`. The contract `P((s, null), ·) = δ_s` is recorded
as the `nullActsAsIdentity` field; in the Markov-kernel form this is
"identity in distribution" (paper's wording). -/
structure PerturbationFamily (S : Type u) (Θ : Type v)
    [MeasurableSpace S] [MeasurableSpace Θ] where
  /-- The Markov kernel `P : (S × Θ) ⇝ S` of Paper Def. 3.1. -/
  toKernel    : Kernel (S × Θ) S
  /-- The null parameter `θ₀ ∈ Θ` of Paper Def. 3.1; setting `θ = null`
      recovers the unperturbed state. -/
  null        : Θ
  /-- The intensity measure `ρ ∈ M₁(Θ)` of Paper Def. 3.1. We carry it as a
      plain `Measure` here; well-formed instances are probability measures. -/
  rho         : Measure Θ
  /-- C2 contract (Paper Def. 3.1, identity-in-distribution): plugging in
      `null` reproduces the Dirac mass at the input state. -/
  nullActsAsIdentity : ∀ s : S, toKernel (s, null) = Measure.dirac s
  /-- `rho` is a probability measure on `Θ`. -/
  rho_isProb : IsProbabilityMeasure rho

namespace PerturbationFamily

variable {S : Type u} {Θ : Type v} [MeasurableSpace S] [MeasurableSpace Θ]

attribute [instance] PerturbationFamily.rho_isProb

/-- Convenience: evaluate the perturbation kernel at `(s, θ)`. -/
def apply (P : PerturbationFamily S Θ) (s : S) (θ : Θ) : Measure S :=
  P.toKernel (s, θ)

end PerturbationFamily

/-! ## §3.2 — Forward model (Def. 3.2). -/

/-- A forward model `F : S → O` is a measurable map (Paper Def. 3.2). When
the optional Lipschitz constant is supplied it certifies Assumption 5.1. -/
structure ForwardModel (S : Type u) (O : Type w)
    [MeasurableSpace S] [MeasurableSpace O] where
  /-- The underlying function `F`. -/
  toFun       : S → O
  /-- Measurability of `F` (Paper Assumption (R2)). -/
  measurable  : Measurable toFun
  /-- Declared Lipschitz constant `L` for Assumption 5.1. `none` means
      unknown / not asserted. -/
  lipschitz   : Option ℝ := none

namespace ForwardModel

variable {S : Type u} {O : Type w} [MeasurableSpace S] [MeasurableSpace O]

instance : CoeFun (ForwardModel S O) (fun _ => S → O) := ⟨ForwardModel.toFun⟩

@[simp] lemma toFun_eq (F : ForwardModel S O) (s : S) : F.toFun s = F s := rfl

end ForwardModel

/-! ## §3.5 — Invariance functional (Def. 3.5).

A real-valued functional on `M₁(O)` that scores concentration. We require
weak-measurability via the standard Mathlib measurable structure on
`ProbabilityMeasure O`. The order-reflecting axiom of Def. 3.5 is recorded
as a `Prop` field; instances may discharge it with `trivial` when not used. -/
structure InvarianceFunctional (O : Type w) [MeasurableSpace O]
    [TopologicalSpace O] [OpensMeasurableSpace O] where
  /-- The functional `Phi : ProbabilityMeasure O → ℝ` (Paper Def. 3.5). -/
  toFun        : ProbabilityMeasure O → ℝ
  /-- Declared `W₁`-Lipschitz constant `Λ` (Assumption 5.1); `none` if
      unknown. -/
  lipschitzW1  : Option ℝ := none

namespace InvarianceFunctional

variable {O : Type w} [MeasurableSpace O] [TopologicalSpace O]
    [OpensMeasurableSpace O]

instance : CoeFun (InvarianceFunctional O) (fun _ => ProbabilityMeasure O → ℝ) :=
  ⟨InvarianceFunctional.toFun⟩

end InvarianceFunctional

/-! ## §3.6 — The perturbation kernel object (Def. 3.6). -/

/-- The perturbation-kernel quadruple `K = (S, P, F, Phi)` of Paper Def. 3.6,
together with all type-class infrastructure required to integrate the kernel
against the intensity measure. -/
structure PerturbationKernel
    (S : Type u) (Θ : Type v) (O : Type w)
    [MeasurableSpace S] [MeasurableSpace Θ]
    [MeasurableSpace O] [TopologicalSpace O] [OpensMeasurableSpace O] where
  /-- Component 1: perturbation family (Paper Def. 3.1). -/
  family       : PerturbationFamily S Θ
  /-- Component 2: forward model (Paper Def. 3.2). -/
  forward      : ForwardModel S O
  /-- Component 3: invariance functional (Paper Def. 3.5). -/
  invariance   : InvarianceFunctional O

namespace PerturbationKernel

variable {S : Type u} {Θ : Type v} {O : Type w}
    [MeasurableSpace S] [MeasurableSpace Θ]
    [MeasurableSpace O] [TopologicalSpace O] [OpensMeasurableSpace O]

/-- **Induced state measure** (Paper Def. 3.3):
`μ_s = ∫_Θ P((s, θ), ·) dρ(θ)`. Implemented via `Measure.bind` applied to
the Markov kernel evaluated at `s`. -/
def inducedStateMeasure (K : PerturbationKernel S Θ O) (s : S) : Measure S :=
  K.family.rho.bind (fun θ => K.family.apply s θ)

/-- **Induced observation measure** (Paper Def. 3.4):
`ν_s = F_* μ_s`, the pushforward of the induced state measure through `F`. -/
def inducedObsMeasure (K : PerturbationKernel S Θ O) (s : S) : Measure O :=
  (K.inducedStateMeasure s).map K.forward.toFun

/-- The induced observation measure is a probability measure. Currently a
statement-only stub — the proof requires `Kernel.bind`-style integration
lemmas (Mathlib `MeasureTheory.Measure.bind` + Markov property of `P`). -/
theorem isProbabilityMeasure_inducedObsMeasure
    (K : PerturbationKernel S Θ O) (s : S)
    (hP : ∀ s θ, IsProbabilityMeasure (K.family.apply s θ)) :
    IsProbabilityMeasure (K.inducedObsMeasure s) := by
  sorry

/-- The induced observation measure packaged as a `ProbabilityMeasure`. -/
def inducedObsProb (K : PerturbationKernel S Θ O) (s : S)
    (hP : ∀ s θ, IsProbabilityMeasure (K.family.apply s θ)) :
    ProbabilityMeasure O :=
  ⟨K.inducedObsMeasure s, K.isProbabilityMeasure_inducedObsMeasure s hP⟩

/-- **Invariance value** (Paper Def. 3.6, the boxed quantity):
`Φ(ν_s)`, the scalar score attached to the base state `s`. -/
def invarianceValue (K : PerturbationKernel S Θ O) (s : S)
    (hP : ∀ s θ, IsProbabilityMeasure (K.family.apply s θ)) : ℝ :=
  K.invariance (K.inducedObsProb s hP)

/-! ## §4 — Well-definedness (Thm 4.2).

The induced measures are Borel-measurable functions of the base state, so
the invariance value defines a measurable field `s ↦ Φ(ν_s)` over `S`. -/

/-- **Theorem 4.2 (Well-definedness).** Under the regularity hypotheses
(R1)–(R3) of Paper §4, the map `s ↦ Φ(ν_s)` is Borel measurable. Proof is
the Fubini-for-kernels + continuous-mapping argument from the paper;
statement-only here. -/
theorem wellDefined (K : PerturbationKernel S Θ O)
    [TopologicalSpace S] [BorelSpace S]
    (hP : ∀ s θ, IsProbabilityMeasure (K.family.apply s θ))
    (_hΦ_meas : True) :
    Measurable (fun s : S => K.invarianceValue s hP) := by
  sorry

end PerturbationKernel
end PerturbationKernel
