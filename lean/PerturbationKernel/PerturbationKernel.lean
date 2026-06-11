/-
Root module of the `PerturbationKernel` library. Re-exports the core
definitions, the stability theorem (Thm 5.4), the sample-complexity bound
(Thm 7.3), the conformance contract (Def 8.1 / Thm 8.2), and a worked
Gaussian-shift example. See `PerturbationKernel/Basic.lean` for the
substantive content.
-/
import PerturbationKernel.Basic
import PerturbationKernel.Stability
import PerturbationKernel.SampleComplexity
import PerturbationKernel.Conformance
import PerturbationKernel.Examples
