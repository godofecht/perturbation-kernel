# PerturbationKernel — Lean 4 / Mathlib formalisation

Mathlib-style formalisation of the perturbation-kernel object defined in
*A Measure-Theoretic Schema for Perturbation Kernels* (`../../paper.tex`)
and governed by `../../SCHEMA.md`.

The mathematical object is the quadruple `K = (S, P, F, Φ)` of Paper
Def. 3.6: a state space `S`, a Markov-kernel perturbation family `P` on
`S × Θ`, a measurable forward model `F : S → O`, and an invariance
functional `Φ : M₁(O) → ℝ`.

## Build

```sh
cd perturbation_kernel/lean/PerturbationKernel
lake update     # fetch Mathlib (slow first time)
lake build
```

Toolchain: `leanprover/lean4:v4.30.0-rc2` (see `lean-toolchain`).

## Module layout

```
PerturbationKernel.lean                # root, re-exports
PerturbationKernel/Basic.lean          # §3:  the four components + K
PerturbationKernel/Stability.lean      # §5:  Thm 5.4 statement
PerturbationKernel/SampleComplexity.lean # §7: Thm 7.3 statement
PerturbationKernel/Conformance.lean    # §8:  Def 8.1 + Thm 8.2
PerturbationKernel/Examples.lean       # Gaussian shift on ℝ
```

## What is proved vs what is stated

| Paper item            | Lean identifier                                       | Status         |
|-----------------------|-------------------------------------------------------|----------------|
| Def. 3.1              | `PerturbationKernel.PerturbationFamily`               | defined        |
| Def. 3.2              | `PerturbationKernel.ForwardModel`                     | defined        |
| Def. 3.5              | `PerturbationKernel.InvarianceFunctional`             | defined        |
| Def. 3.6              | `PerturbationKernel.PerturbationKernel`               | defined        |
| Def. 3.3 (`μ_s`)      | `PerturbationKernel.inducedStateMeasure`              | defined        |
| Def. 3.4 (`ν_s`)      | `PerturbationKernel.inducedObsMeasure`                | defined        |
| Def. 3.6 (`Φ(ν_s)`)   | `PerturbationKernel.invarianceValue`                  | defined        |
| Thm 4.2               | `PerturbationKernel.wellDefined`                      | stated, `sorry`|
| Thm 5.4               | `PerturbationKernel.stability_second_order`           | stated, `sorry`|
| Thm 7.3(c)            | `PerturbationKernel.sampleComplexity_plugIn`          | stated, `sorry`|
| Def. 8.1              | `PerturbationKernel.ConformingImpl`                   | defined        |
| Thm 8.2               | `PerturbationKernel.governance`                       | stated, `sorry`|
| Gaussian-shift C2     | `PerturbationKernel.Examples.gaussianShift_apply_null`| **proved**     |
| Gaussian-shift Dirac  | `PerturbationKernel.Examples.gaussianShift_apply_eq_dirac` | **proved** |

All theorems compile (no syntactic gaps). Every statement matches the
paper hypotheses; only the proofs are deferred.

## Mathlib namespaces used

* `MeasureTheory.Measure`              — measures and `Measure.bind`
* `MeasureTheory.ProbabilityMeasure`   — the `{μ // IsProbabilityMeasure μ}` subtype
* `ProbabilityTheory.Kernel`           — Markov kernels (`Mathlib.Probability.Kernel.Basic`)
* `MeasureTheory.Measure.dirac`        — Dirac measures
* `MeasureTheory.Constructions.BorelSpace` — Borel σ-algebras

### Wasserstein gap

Lean 4.30.0-rc2 / Mathlib does not yet expose a canonical `W₁` distance on
`Measure X` for arbitrary Polish metric spaces. `Stability.lean` therefore
introduces a local `opaque Wass1` placeholder and states Thm 5.4 against
it. When Mathlib lands a canonical Wasserstein-1 distance (in
`Mathlib.MeasureTheory.Distance.Wasserstein` or similar), replacing
`Wass1` with the real definition leaves the theorem statement intact.

## Next steps for full formalisation

1. Replace the `Wass1` placeholder with the Mathlib Wasserstein-1 distance
   once it lands (or formalise it from Kantorovich–Rubinstein duality,
   currently available via `MeasureTheory.lintegral` over couplings).
2. Discharge `wellDefined` (Thm 4.2) using `Measure.bind` + the kernel
   Fubini theorem (`Kernel.lintegral_lintegral`).
3. Build a `PerturbationKernel.Empirical` module that constructs the iid
   product measure and the empirical-measure-valued random variable, then
   prove Thm 7.3(a) via `Mathlib`'s `McDiarmid` (or via a hand-rolled
   bounded-differences argument) and Thm 7.3(b) by importing the
   Fournier–Guillin rate (currently not in Mathlib; would need to be
   stated as an axiom or proved from scratch).
4. Sharpen `Examples.lean` to a genuine Gaussian-shift family with
   `rho := normal 0 σ²` once `Mathlib.Probability.Distributions.Gaussian`
   is in scope.
