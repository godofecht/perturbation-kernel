# Python API

```python
import perturbation_kernel as pk
```

## Config

```python
pk.Config(
    n,                          # ensemble size, required
    seed=0,                     # 64-bit RNG seed
    *,
    backend="auto",             # 'auto' | 'scalar' | 'simd' | 'gpu'
    forward_l=None,             # Lipschitz constant L of the forward model
    invariance_lambda=None,     # W_1 Lipschitz constant Lambda
    epsilon=None,               # target additive error
    eta=None,                   # failure probability, in (0, 1)
    observation_diameter=None,  # diameter D of the observation space
    obs_dim=None,               # dimension of the observation space
    intensity_kind="uniform_interval",
    schema_version=None,        # defaults to the crate's SCHEMA_VERSION
)
```

The four accuracy fields are all-or-nothing. Supplying some but not all
raises `ValueError`, because a partial claim would silently disable the
sample-complexity floor rather than enforce a weaker version of it.

| Member | |
|---|---|
| `.n`, `.seed`, `.schema_version` | read-only |
| `.backend` | readable and writable |
| `.to_json()` | the SCHEMA §5 payload |
| `Config.from_json(s)` | static, validates the major version |
| `.sample_floor()` | required `n` for the asserted accuracy, or `None` |

A config with `backend="auto"` serialises without a `backend` key, so
the default payload is byte-identical to SCHEMA v1.0.0.

## Families

```python
pk.Gaussian(base, sigma_max)
pk.Bistable(x0, dt, theta_max)
pk.Markov(k, theta_max, start=0, base_label=0)
```

Frozen dataclasses. Each has `.run(config) -> Report`, `.to_dict()`
giving the JSON descriptor, and `.name`.

Hyperparameters are validated at run time: an empty base state, a
non-positive step, an alphabet of size zero, a start label outside the
alphabet, or a mixing probability outside `[0, 1]` all raise
`ValueError` rather than panicking somewhere inside the sampler.

## Report

| Member | |
|---|---|
| `.value` | the estimate `Phi-hat_N(s)` |
| `.functional` | which functional produced it |
| `.n_effective`, `.seed`, `.schema_version` | provenance |
| `.error_bound` | dict, or `None` when the constants were not declared |
| `.stability_modulus` | `Lambda * L`, or `None` |
| `.execution` | dict: backend, simd path, threading, device, precision |
| `.to_json(*, pretty=False, v1=False)` | `v1=True` strips `execution` |
| `float(report)` | the value |

## Module functions

```python
pk.run(config, family)          # what Family.run dispatches to
pk.sweep(config, families)       # list of reports, one per family

pk.available_backends()          # e.g. ['auto', 'scalar', 'simd', 'gpu']
pk.simd_path()                   # 'scalar' | 'neon' | 'avx2'
pk.gpu_device()                  # device description, or None

pk.tree_sum(xs)                  # the engine's reduction, exposed
pk.sample_floor(invariance_lambda, observation_diameter,
                epsilon, eta, obs_dim)

pk.SCHEMA_VERSION                # '1.0.0'
pk.__version__
```

`available_backends()` is a capability probe rather than a compile-time
list: `'gpu'` appears only when the wheel has GPU support and a device
is actually usable.

`tree_sum` is exposed because reproducing the engine's reduction order
is the only way to check an externally computed ensemble against a
report. Ordinary summation will not match.

## Errors

| Exception | When |
|---|---|
| `ValueError` | schema violations and domain errors: `n = 0`, incompatible major version, null-parameter mismatch, an accuracy claim below the sample floor, out-of-domain family parameters, an unknown backend name |
| `RuntimeError` | `backend="gpu"` requested but no device is usable, or the wheel was built without GPU support |
| `TypeError` | `pk.run` given something that is not a `Family` |

The split is deliberate. A `ValueError` is a caller mistake you can fix
by passing different arguments. A `RuntimeError` is missing hardware.

## Working with numpy

There is no numpy dependency, and the API takes and returns plain
Python floats and lists. `Gaussian(base=...)` accepts any sequence, so
a numpy array works directly:

```python
import numpy as np

base = np.array([0.5, -1.25, 3.0])
pk.Gaussian(base=base, sigma_max=0.3).run(cfg)
```

For a sweep, collect the values and build the array yourself:

```python
thetas = np.linspace(0, 1, 21)
values = np.array([
    pk.Markov(k=5, theta_max=float(t)).run(cfg).value for t in thetas
])
```
