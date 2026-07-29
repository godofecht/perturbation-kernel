# perturbation-kernel

Reproducible perturbation-kernel estimators with SIMD and GPU backends.

A perturbation kernel answers one question: how much does a result move
when you nudge the thing that produced it?

You supply four objects. A base state. A family of random perturbations
with an intensity you control. A forward map to whatever you actually
observe. A scalar functional of the resulting ensemble. The estimator
returns that scalar and a non-asymptotic error bound.

The answer is a pure function of `(family, n, seed)`. Same three inputs,
same bits, on any machine, any thread count, any CPU vector width.

```python
import perturbation_kernel as pk

cfg = pk.Config(n=262_144, seed=20260610, invariance_lambda=1.0)
report = pk.Markov(k=5, theta_max=0.3).run(cfg)

report.value        # 0.8802871704101562
report.functional   # 'tail_survival'
report.execution    # {'backend': 'auto', 'simd_path': 'neon', ...}
```

## Why this exists

The usual way to ask whether a result is real is to compute a p-value
against a null hypothesis nobody believes, using a threshold somebody
picked in 1925. The perturbative alternative asks the direct question
instead: perturb the inputs by the error you actually measured, re-run
the pipeline, and see whether the result comes back.

That gives you a number with units. "The reported category survives up
to input noise of 0.5" is a claim a reader can check against their own
instrument. It does not depend on a convention, and it fails loudly
when the effect is not there.

The mathematics is in *A Measure-Theoretic Schema for Perturbation
Kernels*; the wire formats are in `SCHEMA.md` v1.0.0. This repository is
the reference implementation, in Rust, with a Lean 4 formalisation of
the headline theorems alongside it.

## What is here

| | |
|---|---|
| **Rust crate** | The engine, three worked families, a C ABI, and the SIMD and GPU backends |
| **Python package** | `pip install perturbation-kernel`, abi3 wheels for CPython 3.8+ |
| **Lean 4 library** | The theorem statements, with the Gaussian-shift case fully proven |

## Three guarantees

**Bit-identical across host backends.** `scalar`, `simd` and `auto`
compute the same reduction tree with the same IEEE-754 operations. Your
CPU's vector width and your machine's core count are not observable in
the output. Verified against golden values captured from v1.0.0, on
every length from 0 to 4096 and under property tests.

**Bit-identical on the GPU too, or an error.** `backend="gpu"` emulates
binary64 in `u32` pairs and transcribes `rand`'s integer sampler, so a
device run returns the same bits as the host. Families that need
transcendental functions WGSL does not specify exactly are refused
rather than silently approximated; `backend="gpu_f32"` runs those,
faster and approximately, and says so in every report it produces.

**Refuses claims it cannot support.** Assert an accuracy target below
the Theorem 7.3(c) sample floor and the run is rejected, with the
required `n` in the error message. The bound is not a decoration you can
quietly drop.

## Next

- [Install](install.md)
- [Quickstart](quickstart.md)
- [What a perturbation kernel is](concepts.md)
- [Choosing a backend](backends.md)
