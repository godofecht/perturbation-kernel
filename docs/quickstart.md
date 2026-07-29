# Quickstart

## Your first run

```python
import perturbation_kernel as pk

cfg = pk.Config(n=262_144, seed=20260610)
report = pk.Markov(k=5, theta_max=0.3).run(cfg)

print(report.value)   # 0.8802871704101562
```

A pipeline sorts samples into one of five categories. At intensity
`theta` a sample is reassigned uniformly at random; otherwise it keeps
its label. `theta` itself is drawn uniformly from `[0, theta_max]`. The
value is the fraction of samples still in category 0 afterwards.

Run it again with the same seed and you get the same bits. Run it on a
different machine, with a different core count, on a CPU with a
different vector width, and you still get the same bits.

## The three built-in families

| Family | Perturbation | Invariance | Range |
|---|---|---|---|
| `Gaussian(base, sigma_max)` | `s + sigma * N(0, I)` | negative dispersion | `<= 0` |
| `Bistable(x0, dt, theta_max)` | one Langevin step in a double well | polarisation | `[-1, 1]` |
| `Markov(k, theta_max, start, base_label)` | mix to uniform with probability `theta` | tail survival | `[0, 1]` |

All three have null parameter `0.0`: at zero intensity the state comes
back unchanged, so the value is exactly the unperturbed one.

```python
pk.Markov(k=5, theta_max=0.0, start=2, base_label=2).run(cfg).value
# 1.0, exactly

pk.Gaussian(base=[1.5, -2.0], sigma_max=0.0).run(cfg).value
# 0.0, exactly
```

## Sweeping the intensity

The single most useful thing to do with a perturbation kernel is to walk
the intensity up and watch where the result stops holding.

```python
cfg = pk.Config(n=262_144, seed=1)
for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
    v = pk.Markov(k=5, theta_max=theta).run(cfg).value
    print(f"{theta:.2f}  {v:.4f}")
```

```
0.00  1.0000
0.25  0.8999
0.50  0.8011
0.75  0.7010
1.00  0.6004
```

Survival falls linearly, and the slope is `(1 - 1/k)/2`, which for
`k = 5` is `0.4`. The closed form is `1 - (theta/2)(1 - 1/k)`, so this
table can be checked by hand. Worked through properly in
[example 01](examples.md).

## Asking for an error bound

Declare the Lipschitz constants and an accuracy target and the report
carries the Theorem 7.3 bound:

```python
cfg = pk.Config(
    n=262_144, seed=1,
    invariance_lambda=1.0, forward_l=1.0,
    epsilon=0.05, eta=0.05,
    observation_diameter=1.0, obs_dim=1,
)
r = pk.Markov(k=5, theta_max=0.3).run(cfg)

r.error_bound
# {'epsilon': 0.0270..., 'eta': 0.05, 'basis': 'mcdiarmid+fournier_guillin', ...}
```

Ask for more precision than `n` supports and the run is refused:

```python
pk.Markov(k=5, theta_max=0.3).run(pk.Config(
    n=1000, seed=1, invariance_lambda=1.0,
    epsilon=0.05, eta=0.05, observation_diameter=1.0, obs_dim=1,
))
# ValueError: sample-complexity floor: requested (0.05,0.05)
#             needs N >= 262144, got 1000
```

Work the other way and size the run from the claim:

```python
pk.sample_floor(invariance_lambda=1.0, observation_diameter=1.0,
                epsilon=0.05, eta=0.05, obs_dim=1)
# 262144
```

See [Error bounds](error-bounds.md).

## Picking a backend

```python
pk.Config(n=1_000_000, seed=1, backend="gpu")
```

`auto` is the default and is the right answer almost always. See
[Backends](backends.md) for what each one guarantees, and for the
measured crossover where a device starts to win.

## The same thing in Rust

```rust
use perturbation_kernel::config::{Backend, Config};
use perturbation_kernel::family::Family;

let cfg = Config {
    n: 262_144,
    seed: 20_260_610,
    backend: Backend::Auto,
    ..Default::default()
};

let report = Family::Markov {
    k: 5,
    start: 0,
    base_label: 0,
    theta_max: 0.3,
}
.run(&cfg)?;

println!("{}", report.value);
```

To plug in your own perturbation, forward model and functional, see the
[Rust API](rust-api.md).

## Where to go next

- [Concepts](concepts.md) if you want to know what the four objects are
  and why the schema insists on them.
- [Examples](examples.md) for five worked problems with output.
- [Reproducibility](reproducibility.md) if you are publishing a number.
