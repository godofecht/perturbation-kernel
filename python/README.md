# perturbation-kernel

Perturbative stability estimation with a reproducibility guarantee at
the bit level.

Given a base state `s`, a parametrised Markov kernel `P((s, θ), ·)`, an
intensity law `ρ` over `θ`, a measurable forward map `F: S → O`, and a
functional `Φ: M₁(O) → R`, the engine evaluates the plug-in estimator

```
Φ̂_N(s) = Φ( (1/N) Σᵢ δ_{F(Sᵢ)} ),    Sᵢ ~ P((s, θᵢ), ·),  θᵢ ~ ρ
```

and returns it with a non-asymptotic error bound. The estimate is a pure
function of `(family, N, seed)`: no ambient entropy, no thread-count
dependence, no accumulation order that varies with hardware.

Reference implementation of `SCHEMA.md` v1.0.0.

## Install

```
pip install perturbation-kernel
```

abi3 wheels for CPython 3.8+ on Linux, macOS and Windows, both
architectures. No Rust toolchain required.

## Use

```python
import perturbation_kernel as pk

cfg = pk.Config(n=262_144, seed=20260610, invariance_lambda=1.0)
r = pk.Markov(k=5, theta_max=0.3).run(cfg)

r.value       # 0.8802871704101562
r.functional  # 'tail_survival'
r.execution   # {'backend': 'auto', 'simd_path': 'neon', 'precision': 'f64', ...}
```

## Built-in families

| Family | Kernel `P((s, θ), ·)` | Functional `Φ` | Codomain |
|---|---|---|---|
| `Gaussian(base, sigma_max)` | `s + θ·N(0, I)`, `θ ~ U[0, σ_max]` | negative summed coordinate variance | `(-∞, 0]` |
| `Bistable(x0, dt, theta_max)` | one Euler-Maruyama step in `V(x) = (x²-1)²` | polarisation `E[sgn X]` | `[-1, 1]` |
| `Markov(k, theta_max, start, base_label)` | mix to `U{0..k-1}` w.p. `θ` | survival `P(X = base_label)` | `[0, 1]` |

All three satisfy the C2 identity contract: `θ₀ = 0` and
`P((s, θ₀), ·) = δ_s`, so the null perturbation is exactly recoverable
rather than approximately so.

```python
pk.Markov(k=5, theta_max=0.0, start=2, base_label=2).run(cfg).value  # exactly 1.0
pk.Gaussian(base=[1.5, -2.0], sigma_max=0.0).run(cfg).value          # exactly 0.0
```

## Determinism

Draw `i` consumes the substream `ChaCha20(mix64(seed, i))`, so the
ensemble is index-addressable and order-free. Reduction is a fixed-shape
pairwise tree keyed to the index order, not to the thread count, which
is what keeps a non-associative float sum reproducible.

```python
a = pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=50_000, seed=7))
b = pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=50_000, seed=7, backend="scalar"))
a.value.hex() == b.value.hex()   # True — identical mantissa, not "close"
```

CI verifies this against golden values captured from v1.0.0, across four
native OS/architecture combinations and every feature configuration.

## Backends

```python
pk.available_backends()   # ['auto', 'scalar', 'simd', 'gpu', 'gpu_f32']
pk.simd_path()            # 'neon' | 'avx2' | 'scalar'
pk.gpu_device()           # 'Apple M4 Max (Metal, IntegratedGpu)' or None
```

| Backend | Execution | Agreement |
|---|---|---|
| `scalar` | single-threaded, portable | reference |
| `simd` | NEON `vpaddq_f64` / AVX2 `vhaddpd`+`vpermpd` | bit-identical |
| `auto` | vectorised, `rayon` above 4096 draws | bit-identical |
| `gpu` | `wgpu` compute, emulated binary64 | bit-identical |
| `gpu_f32` | `wgpu` compute, single precision | statistically equivalent |

Vectorisation is exact because each tree-level output is one IEEE-754
addition of one fixed operand pair; four lanes perform the same four
additions. No lane-crossing accumulator, no reassociation, and the
centring step is an explicit subtract-then-multiply so it cannot
contract into an FMA (which rounds once where the reference rounds
twice).

`gpu` is exact for `Markov` because that family's arithmetic reduces to
a short, fully-specified list: `f64.wgsl` emulates binary64
multiplication in `u32` pairs with round-to-nearest-even, rand's Lemire
rejection sampler is transcribed rather than approximated, and the
indicator observation makes the reduction integer addition — associative,
hence scheduling-invariant. Families drawing normal deviates require
`ln`/`exp`, whose accuracy WGSL leaves to the driver, so `gpu` rejects
them rather than silently returning a different number:

```python
pk.Gaussian(base=[0.5], sigma_max=0.3).run(pk.Config(n=1024, backend="gpu"))
# RuntimeError: ... use backend "gpu_f32" to accept a single-precision result
```

## Error bounds

Declare `Λ` (Wasserstein-1 Lipschitz constant of `Φ`), the observation
diameter `D`, and `d_obs`, and the report carries the Theorem 7.3 bound:
a McDiarmid concentration term `√(Λ²D²/2N · ln(2/η))` plus a
Fournier-Guillin bias term for the empirical-measure convergence rate.

An accuracy claim below the Theorem 7.3(c) sample floor is rejected at
run time rather than reported optimistically:

```python
pk.sample_floor(invariance_lambda=1.0, observation_diameter=1.0,
                epsilon=0.05, eta=0.05, obs_dim=1)          # 262144

pk.Markov(k=5, theta_max=0.3).run(pk.Config(
    n=1000, seed=1, invariance_lambda=1.0,
    epsilon=0.05, eta=0.05, observation_diameter=1.0, obs_dim=1))
# ValueError: sample-complexity floor: requested (0.05,0.05) needs N >= 262144
```

The floor is dominated by the bias term, not the variance term: the
Wasserstein rate converges more slowly than the concentration does, so
tightening `ε` by 10× costs 256× the compute.

## Other languages

The same engine, the same bits, through the C ABI (and wasm for
TypeScript): **Rust**, **C/C++**, **Zig**, **Julia**, **TypeScript**.
CI runs a cross-language agreement job that compares raw `f64` bit
patterns across all of them rather than each against a constant.

## Documentation

<https://godofecht.github.io/perturbation-kernel/>

## License

MIT.
