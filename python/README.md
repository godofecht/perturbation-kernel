# perturbation-kernel

Reproducible perturbation-kernel estimators with SIMD and GPU backends.

A perturbation kernel answers one question: how much does a result move
when you nudge the thing that produced it? You supply a base state, a
family of random perturbations, a forward map to what you actually
observe, and a scalar functional of the resulting ensemble. You get back
that scalar and a non-asymptotic error bound.

The estimator is a pure function of `(family, n, seed)`. Same three
inputs, same bits, on any machine, any thread count, any CPU vector
width.

Python bindings over the Rust reference implementation of
[SCHEMA.md v1.0.0](https://github.com/godofecht/perturbation-kernel).

## Install

```
pip install perturbation-kernel
```

Wheels ship for Linux, macOS and Windows on CPython 3.8+ (stable ABI, so
one wheel per platform covers every version).

## Use

```python
import perturbation_kernel as pk

cfg = pk.Config(n=100_000, seed=20260610, invariance_lambda=1.0)
report = pk.Markov(k=5, theta_max=0.3).run(cfg)

print(report.value)        # 0.880235
print(report.functional)   # tail_survival
print(report.execution)    # {'backend': 'auto', 'simd_path': 'neon', ...}
```

Three families ship built in:

| Family | Perturbation | Invariance | Range |
|---|---|---|---|
| `Gaussian(base, sigma_max)` | `s + sigma * N(0, I)` | negative dispersion | `<= 0` |
| `Bistable(x0, dt, theta_max)` | one Langevin step in a double well | polarisation | `[-1, 1]` |
| `Markov(k, theta_max, start, base_label)` | mix to uniform w.p. `theta` | tail survival | `[0, 1]` |

## Error bounds

Declare the Lipschitz constants and an accuracy target, and the report
carries the Theorem 7.3 bound. Claiming an accuracy that `n` cannot
support is an error, not a footnote:

```python
cfg = pk.Config(
    n=1_000_000, seed=1,
    invariance_lambda=1.0, forward_l=1.0,
    epsilon=0.05, eta=0.05, observation_diameter=1.0, obs_dim=1,
)
r = pk.Markov(k=5, theta_max=0.3).run(cfg)
print(r.error_bound)   # {'epsilon': 0.0158..., 'eta': 0.05, ...}

pk.Markov(k=5, theta_max=0.3).run(pk.Config(
    n=100, seed=1, invariance_lambda=1.0,
    epsilon=0.05, eta=0.05, observation_diameter=1.0, obs_dim=1))
# ValueError: sample-complexity floor: requested (0.05,0.05) needs N >= 262144, got 100
```

## Backends

```python
pk.available_backends()   # ['auto', 'scalar', 'simd', 'gpu']
pk.simd_path()            # 'neon'
pk.gpu_device()           # 'Apple M4 Max (Metal, IntegratedGpu)'
```

`auto`, `scalar` and `simd` are bit-identical to each other. They differ
only in how long they take.

`gpu` is a different numerical path. The device carries the ensemble in
single precision and draws normals by Box-Muller rather than the
ziggurat, so it agrees with the host statistically rather than bit for
bit. It draws from the same ChaCha20 substreams, which keeps the
agreement tight in practice (~1e-4 at n=200k). `report.execution`
always records which path ran, so a device result can never be mistaken
for a host one.

```python
host = pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=200_000, seed=1))
dev  = pk.Markov(k=5, theta_max=0.3).run(
    pk.Config(n=200_000, seed=1, backend="gpu")
)
abs(host.value - dev.value)   # 3e-4
```

## Reproducibility

```python
a = pk.Config(n=50_000, seed=7)
b = pk.Config(n=50_000, seed=7, backend="scalar")
pk.Markov(k=5, theta_max=0.3).run(a).value \
    == pk.Markov(k=5, theta_max=0.3).run(b).value   # True, exactly
```

Randomness flows through a ChaCha20 stream keyed by `seed` and forked
per draw index, so draw `i` never depends on how many other draws ran or
in what order. `report.to_json(v1=True)` gives the strict SCHEMA v1.0.0
payload if you need to hand it to another implementation.

## License

MIT.
