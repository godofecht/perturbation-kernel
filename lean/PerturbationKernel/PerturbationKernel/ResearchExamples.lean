/-
Exact, deterministic statistical examples.

The finite-state perturbation model resamples its label uniformly with
probability theta; theta itself is uniform on [0, theta_max]. When the
initial and reported labels agree, expected survival is
1 - theta_max * (1 - 1 / k) / 2. The result here proves the algebraic
sensitivity of this model, not a Monte Carlo concentration theorem.
-/
import PerturbationKernel.Basic
import Mathlib.Tactic

namespace PerturbationKernel
namespace ResearchExamples

/-- Expected survival under a uniformly distributed perturbation intensity.
The argument `returnProbability` is the probability that resampling
returns the original label. -/
def labelSurvival (returnProbability thetaMax : ℝ) : ℝ :=
  1 - thetaMax * (1 - returnProbability) / 2

/-- A change in maximum intensity has an *exact* linear effect on the mean. -/
theorem survival_difference (r theta delta : ℝ) :
    labelSurvival r theta - labelSurvival r (theta + delta)
      = delta * (1 - r) / 2 := by
  unfold labelSurvival
  ring

/-- More resampling cannot increase expected survival if return probability ≤ 1. -/
theorem survival_antitone (r theta delta : ℝ)
    (hr : r ≤ 1) (hdelta : 0 ≤ delta) :
    labelSurvival r (theta + delta) ≤ labelSurvival r theta := by
  have hfactor : 0 ≤ (1 - r) / 2 := by linarith
  have hnonneg : 0 ≤ delta * ((1 - r) / 2) :=
    mul_nonneg hdelta hfactor
  have heq := survival_difference r theta delta
  linarith

/-- For five classes, the model predicts 80% survival at intensity 0.5. -/
example : labelSurvival (1 / 5) (1 / 2) = (4 / 5 : ℝ) := by
  norm_num [labelSurvival]

/-- At intensity 0.3, the exact expected survival is 0.88. -/
example : labelSurvival (1 / 5) (3 / 10) = (22 / 25 : ℝ) := by
  norm_num [labelSurvival]

/-- Doubling the uniform intensity range from 0.25 to 0.5
costs exactly 10 percentage points of expected survival (five classes). -/
example :
    labelSurvival (1 / 5) (1 / 4) -
      labelSurvival (1 / 5) (1 / 2) = (1 / 10 : ℝ) := by
  norm_num [labelSurvival]

end ResearchExamples
end PerturbationKernel
