# perturbation-kernel

Perturbative stability estimation with a reproducibility guarantee at
the bit level. Rust core, five language bindings.

[![CI](https://github.com/godofecht/perturbation-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/godofecht/perturbation-kernel/actions/workflows/ci.yml)
[![Docs](https://github.com/godofecht/perturbation-kernel/actions/workflows/docs.yml/badge.svg)](https://godofecht.github.io/perturbation-kernel/)
[![PyPI](https://img.shields.io/pypi/v/perturbation-kernel?logo=pypi&logoColor=white)](https://pypi.org/project/perturbation-kernel/)
[![crates.io](https://img.shields.io/crates/v/perturbation-kernel?logo=rust&logoColor=white)](https://crates.io/crates/perturbation-kernel)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)](https://pypi.org/project/perturbation-kernel/)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust&logoColor=white)](https://github.com/godofecht/perturbation-kernel/blob/main/Cargo.toml)
[![License](https://img.shields.io/pypi/l/perturbation-kernel)](LICENSE)

Given a base state `s`, a parametrised Markov kernel `P((s, θ), ·)`, an
intensity law `ρ` over `θ`, a measurable forward map `F: S → O`, and a
functional `Φ: M₁(O) → R`, the engine evaluates the plug-in estimator

```
Φ̂_N(s) = Φ( (1/N) Σᵢ δ_{F(Sᵢ)} ),    Sᵢ ~ P((s, θᵢ), ·),  θᵢ ~ ρ
```

and returns it with a non-asymptotic error bound. The estimate is a pure
function of `(family, N, seed)`: no ambient entropy, no thread-count
dependence, no accumulation order that varies with hardware.

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

| Surface | Path | Mechanism |
|---|---|---|
| **Rust** · engine, traits, backends | `src/`, `tests/` | native |
| **Python** · abi3 wheels, CPython 3.8+ | `python/` | PyO3 |
| **C / C++** · header-only C++20 wrapper | `bindings/cpp/` | C ABI |
| **Zig** · `@cImport` module | `bindings/zig/` | C ABI |
| **Julia** · `ccall` module | `bindings/julia/` | C ABI |
| **TypeScript** · Node, Deno, browser | `bindings/ts/` | wasm32 |
| **Lean 4 / Mathlib** · formalised statements | `lean/` | `lake build` green; 5 theorems stated, Gaussian-shift example fully proven, four headline theorems `sorry`'d at the statement layer |

Every binding routes through the same engine. CI runs a cross-language
agreement job comparing raw `f64` bit patterns rather than each binding
against its own constant:

```
language       value                  bits
----------------------------------------------------------
rust           0.8802871704101562     3fec2b5000000000
python         0.8802871704101562     3fec2b5000000000
c++            0.8802871704101562     3fec2b5000000000
julia          0.8802871704101562     3fec2b5000000000
typescript     0.8802871704101562     3fec2b5000000000
```

## Install

```bash
pip install perturbation-kernel          # Python
cargo add perturbation-kernel            # Rust
```

The other four are built from this repository rather than pulled from a
registry. C++, Zig and Julia link the C ABI produced by
`cargo build --release`; TypeScript builds a wasm module with
`wasm-pack`. See `bindings/` for the per-language build and the
conformance test each one ships.

## Backends

| Backend | Where | Agreement with the reference |
|---|---|---|
| `scalar` | one thread, portable loops | it *is* the reference |
| `simd` | NEON on aarch64, AVX2 on x86-64 | bit-identical |
| `auto` | vectorised, threaded above 4096 draws | bit-identical |
| `gpu` | `wgpu` compute on Metal, Vulkan or DX12 | **bit-identical** |
| `gpu_f32` | the same, in single precision | statistically equivalent |

Vectorisation is exact because each tree-level output is one IEEE-754
addition of one fixed operand pair; four lanes perform the same four
additions. No lane-crossing accumulator, no reassociation, and the
centring step is an explicit subtract-then-multiply so it cannot
contract into an FMA, which rounds once where the reference rounds
twice.

`gpu` is exact for `Markov` because that family's arithmetic reduces to
a short, fully-specified list: `f64.wgsl` emulates binary64
multiplication in `u32` pairs with round-to-nearest-even, rand's Lemire
rejection sampler is transcribed rather than approximated, and the
indicator observation makes the reduction integer addition, which is
associative and therefore scheduling-invariant. Families drawing normal
deviates require `ln` and `exp`, whose accuracy WGSL leaves to the
driver, so `gpu` rejects them rather than silently returning a different
number. `gpu_f32` runs those, faster and approximately, and records
`precision: "f32"` in `report.execution`.

<!-- BENCH:START -->

| | speedup | against |
|---|---|---|
| Reductions | 1.6x to 10.9x | the v1.0.0 reduction |
| Engine | 2.4x to 3.2x | one thread, scalar loops |

Measured by CI on AMD EPYC 7763 64-Core Processor (4 cores). Full tables in
[BENCHMARKS.md](BENCHMARKS.md).

<!-- BENCH:END -->

## Bit-identity

`GOLDEN.txt` holds 42 `Report.value` bit patterns produced by the
original v1.0.0 implementation. CI regenerates and diffs them on every
push across six native OS/architecture combinations, Linux, macOS and
Windows on both x86-64 and aarch64, under every feature combination:

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
