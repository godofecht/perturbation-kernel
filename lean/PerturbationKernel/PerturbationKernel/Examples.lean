/-
Worked example: Gaussian shift on `ℝ`.

Set `S := ℝ`, `Θ := ℝ`, `apply s θ := s + θ` (deterministic), `null := 0`,
`F := id`. The contract `apply s null = s` (Paper Def. 3.1's identity-in-
distribution, here strengthened to an honest equality because the
perturbation is deterministic) is discharged without `sorry`.

We use `Kernel.deterministic` from `Mathlib.Probability.Kernel.Basic` to
package the shift as a genuine Markov kernel.
-/
import PerturbationKernel.Basic
import Mathlib.Probability.Kernel.Basic
import Mathlib.MeasureTheory.Measure.Dirac

noncomputable section
open MeasureTheory ProbabilityTheory

namespace PerturbationKernel
namespace Examples

/-! ## Gaussian shift on `ℝ`. -/

/-- The shift map `(s, θ) ↦ s + θ` is measurable. -/
lemma shift_measurable : Measurable (fun p : ℝ × ℝ => p.1 + p.2) :=
  measurable_fst.add measurable_snd

/-- The deterministic Markov kernel realising `apply s θ = δ_{s+θ}`. -/
def shiftKernel : Kernel (ℝ × ℝ) ℝ :=
  Kernel.deterministic (fun p : ℝ × ℝ => p.1 + p.2) shift_measurable

/-- The Gaussian-shift perturbation family. The intensity measure is taken
to be a Dirac at `0` for simplicity; replacing it with `N(0, σ²)` recovers
the actual Gaussian-shift case but requires importing
`MeasureTheory.Measure.Lebesgue` and a normal-density construction. -/
def gaussianShiftFamily : PerturbationFamily ℝ ℝ where
  toKernel := shiftKernel
  null     := (0 : ℝ)
  rho      := Measure.dirac (0 : ℝ)
  nullActsAsIdentity := by
    intro s
    -- `shiftKernel (s, 0) = δ_{s + 0} = δ_s`.
    show shiftKernel (s, (0 : ℝ)) = Measure.dirac s
    unfold shiftKernel
    rw [Kernel.deterministic_apply]
    simp
  rho_isProb := Measure.dirac.isProbabilityMeasure

/-- The identity forward model `F = id : ℝ → ℝ`, with Lipschitz constant 1. -/
def idForward : ForwardModel ℝ ℝ where
  toFun      := id
  measurable := measurable_id
  lipschitz  := some 1

/-- The trivial (constant-zero) invariance functional. Useful as a stand-in
to exhibit a complete `PerturbationKernel` instance without committing to a
particular `Φ`. -/
def zeroInvariance : InvarianceFunctional ℝ where
  toFun       := fun _ => 0
  lipschitzW1 := some 0

/-- The full `PerturbationKernel` object for the Gaussian-shift example. -/
def gaussianShiftKernel : PerturbationKernel ℝ ℝ ℝ where
  family     := gaussianShiftFamily
  forward    := idForward
  invariance := zeroInvariance

/-! ## The identity-in-distribution contract holds (no `sorry`). -/

/-- The C2 identity-in-distribution contract for the Gaussian shift:
applying the perturbation at the null parameter recovers `δ_s`. This is
exactly the `nullActsAsIdentity` field, surfaced as a named theorem. -/
theorem gaussianShift_apply_null (s : ℝ) :
    gaussianShiftFamily.apply s gaussianShiftFamily.null = Measure.dirac s := by
  -- `apply = toKernel`, and the structure field already supplies the proof.
  exact gaussianShiftFamily.nullActsAsIdentity s

/-- Stronger statement: the *deterministic* shift acts as the identity on
the state. Implicit Dirac-pushforward content. -/
theorem gaussianShift_apply_eq_dirac (s θ : ℝ) :
    gaussianShiftFamily.apply s θ = Measure.dirac (s + θ) := by
  show shiftKernel (s, θ) = Measure.dirac (s + θ)
  unfold shiftKernel
  rw [Kernel.deterministic_apply]

end Examples
end PerturbationKernel
