# perturbation-kernel

Reproducible perturbation-kernel estimators with SIMD and GPU backends.

[![CI](https://github.com/godofecht/perturbation-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/godofecht/perturbation-kernel/actions/workflows/ci.yml)
[![Docs](https://github.com/godofecht/perturbation-kernel/actions/workflows/docs.yml/badge.svg)](https://godofecht.github.io/perturbation-kernel/)
[![PyPI](https://img.shields.io/pypi/v/perturbation-kernel)](https://pypi.org/project/perturbation-kernel/)
[![crates.io](https://img.shields.io/crates/v/perturbation-kernel)](https://crates.io/crates/perturbation-kernel)

A perturbation kernel answers one question: how much does a result move
when you nudge the thing that produced it?

You supply a base state, a family of random perturbations with an
intensity you control, a forward map to whatever you actually observe,
and a scalar functional of the resulting ensemble. You get back that
scalar and a non-asymptotic error bound. The answer is a pure function
of `(family, n, seed)`, so the same three inputs give the same bits on
any machine, any thread count, any CPU vector width.

**[Documentation](https://godofecht.github.io/perturbation-kernel/)**

```python
import perturbation_kernel as pk

cfg = pk.Config(n=262_144, seed=20260610, invariance_lambda=1.0)
report = pk.Markov(k=5, theta_max=0.3).run(cfg)

report.value        # 0.8802871704101562
report.execution    # {'backend': 'auto', 'simd_path': 'neon', ...}
```

Reference implementation of the perturbation-kernel object defined in
`SCHEMA.md` v1.0.0. The mathematics is in *A Measure-Theoretic Schema
for Perturbation Kernels*. Where the two disagree, the paper governs the
mathematics and the schema governs the wire formats.

| Implementation | Path | Status |
|---|---|---|
| **Rust** · engine, C ABI, SIMD and GPU backends | `src/`, `tests/` | 61 tests, `clippy` clean |
| **Python** · bindings, abi3 wheels for CPython 3.8+ | `python/` | 94 tests |
| **Lean 4 / Mathlib** · formalised statements | `lean/PerturbationKernel/` | `lake build` green; 5 theorems stated, Gaussian-shift example fully proven, four headline theorems `sorry`'d at the statement layer |

## Install

```bash
pip install perturbation-kernel
```

```toml
[dependencies]
perturbation-kernel = "2"
```

## Backends

| Backend | Where | Agreement with the reference |
|---|---|---|
| `scalar` | one thread, portable loops | it *is* the reference |
| `simd` | NEON on aarch64, AVX2 on x86-64 | bit-identical |
| `auto` | vectorised, threaded above 4096 draws | bit-identical |
| `gpu` | `wgpu` compute on Metal, Vulkan or DX12 | **bit-identical** |
| `gpu_f32` | the same, in single precision | statistically equivalent |

The three host backends compute the same reduction tree with the same
IEEE-754 operations, so your CPU's vector width and your core count are
not observable in the output.

`gpu` is bit-identical too. That is possible because the Markov family's
arithmetic is a short list of exactly-specified operations: `f64.wgsl`
emulates binary64 multiplication in `u32` pairs, `rand`'s integer
sampler is transcribed rather than approximated, and the observation is
an indicator so the reduction is integer addition. Families that draw
normal deviates need `ln` and `exp`, which WGSL does not specify
exactly, so `gpu` refuses them rather than quietly returning a different
number. `gpu_f32` runs those, faster and approximately, and says so in
`report.execution`.

<!-- BENCH:START -->

| | speedup | against |
|---|---|---|
| Reductions | 1.8x to 11.5x | the v1.0.0 reduction |
| Engine | 2.4x to 3.2x | one thread, scalar loops |

Measured by CI on AMD EPYC 7763 64-Core Processor (4 cores). Full tables in
[BENCHMARKS.md](BENCHMARKS.md).

<!-- BENCH:END -->

## Bit-identity

`GOLDEN.txt` holds 42 `Report.value` bit patterns produced by the
original v1.0.0 implementation. CI regenerates and diffs them on Linux,
macOS and Windows on every push:

```bash
cargo run --release --example golden > golden.actual
diff GOLDEN.txt golden.actual
```

The file is never regenerated to make a build pass. Every optimisation
here, the vectorised reductions, the thread pool, the flat ensemble
storage, was accepted only after that diff came back empty.

## Build

```bash
cargo test --release              # 61 tests
cargo test --release --features gpu
cargo bench --bench kernel
cargo clippy --all-targets --all-features -- -D warnings
```

```bash
cd python
maturin develop --release
python -m pytest tests
```

```bash
cd lean/PerturbationKernel && lake build
```

Rust 1.85 or later; 1.87 with the `gpu` feature, which depends on `wgpu`.

## Conformance

Every MUST of SCHEMA §§3-7 is satisfied. The C1-C6 checklist:

| Point | Requirement | Where |
|-------|-------------|-------|
| C1 | Typed spaces with a metric `d_O` | type parameters on the traits; exercised by `tests/conformance.rs::c1_*` |
| C2 | `Perturbation<S>` with null-parameter identity | `src/perturbation.rs`; `tests/conformance.rs::c2_identity_recovery_*` |
| C3 | `ForwardModel<S,O>` with optional declared `L` | `src/forward.rs` |
| C4 | `Invariance<O>` with optional declared `Lambda` | `src/invariance.rs`; `tests/conformance.rs::s11_order_invariance_of_measure` |
| C5 | `Config` carrying `rho`, `N`, seed, reduction | `src/config.rs` |
| C6 | Engine running `Phi-hat_N(s)` with seeded reduction | `src/engine.rs`; per-index fork in `fork_rng` |

SCHEMA §7 sample-complexity and §8 determinism are exercised by
`tests/sample_complexity.rs` and `tests/determinism.rs`. Cross-backend
equivalence is in `tests/backends.rs` and `tests/reduce.rs`; the device
contract is in `tests/gpu.rs`.

## Examples

```bash
python python/examples/01_does_my_finding_survive.py
python python/examples/02_where_does_it_tip.py
python python/examples/03_how_many_draws.py
python python/examples/04_gpu_sweep.py
python python/examples/05_reproducibility_receipt.py

cargo run --release --example custom_family
```

Output and commentary for all of them is in
[Examples](https://godofecht.github.io/perturbation-kernel/examples/).

## License

MIT. See [LICENSE](LICENSE).
