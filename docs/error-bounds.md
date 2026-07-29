# Error bounds

The estimator can tell you how far it might be from the quantity it is
estimating. It will also refuse to run when you ask for a precision your
ensemble size cannot deliver.

## Getting one

Four numbers have to be declared before a bound can be computed:

| Field | Meaning |
|---|---|
| `invariance_lambda` | Wasserstein-1 Lipschitz constant of the functional |
| `observation_diameter` | diameter `D` of the observation space |
| `obs_dim` | dimension of the observation space |
| `eta` | failure probability, so `1 - eta` is the confidence |

```python
cfg = pk.Config(
    n=262_144, seed=1,
    invariance_lambda=1.0,
    epsilon=0.05, eta=0.05,
    observation_diameter=1.0, obs_dim=1,
)
r = pk.Markov(k=5, theta_max=0.3).run(cfg)

r.error_bound
# {'epsilon': 0.027021, 'eta': 0.05,
#  'basis': 'mcdiarmid+fournier_guillin', ...}
```

For an indicator readout these are easy: the observations live in
`{0, 1}`, so `D = 1` and `obs_dim = 1`, and the mean is 1-Lipschitz in
`W_1`, so `Lambda = 1`.

Leave them out and `error_bound` is `None`. That is the honest outcome,
and it is what the built-in `Gaussian` family does by default: the
empirical variance is not globally `W_1`-Lipschitz on an unbounded
space, so there is no constant to declare and no bound to report.

## What the bound is made of

Two terms, added.

**Concentration.** McDiarmid's inequality on the plug-in estimator gives

```
t = sqrt( Lambda^2 D^2 / (2N) * ln(2/eta) )
```

This is the familiar `1/sqrt(N)` behaviour.

**Bias.** The empirical measure is not the true measure, and the
functional is evaluated at the wrong measure by an amount governed by
the Wasserstein convergence rate. The Fournier-Guillin rate is
`O(N^{-1/d})` for `d > 2` and `O(N^{-1/2} log N)` at or below two
dimensions.

The bias term is the one that hurts. It converges more slowly, so past
a certain point buying precision means paying its rate rather than the
concentration rate:

| target `+/-` | draws needed |
|---|---|
| 0.50 | 1,024 |
| 0.20 | 8,192 |
| 0.10 | 65,536 |
| 0.05 | 262,144 |
| 0.02 | 4,194,304 |
| 0.01 | 16,777,216 |

Tightening from `+/-0.1` to `+/-0.01` costs 256 times the compute.
Halving your error bar is not a four-fold job.

## The sample floor

Asserting an accuracy target you cannot support is an error:

```python
pk.Markov(k=5, theta_max=0.3).run(pk.Config(
    n=1000, seed=1, invariance_lambda=1.0,
    epsilon=0.05, eta=0.05, observation_diameter=1.0, obs_dim=1,
))
# ValueError: sample-complexity floor: requested (0.05,0.05)
#             needs N >= 262144, got 1000
```

This is deliberate. A bound that can be quietly dropped when it comes
back inconvenient is not doing any work. Here the claim is part of the
config, and a config that cannot be honoured does not run.

Size the ensemble from the claim instead:

```python
n = pk.sample_floor(
    invariance_lambda=1.0, observation_diameter=1.0,
    epsilon=0.05, eta=0.05, obs_dim=1,
)   # 262144
```

## It is a bound, not an estimate

The constants are deliberately conservative. Measured against a family
with a closed form, twenty independent seeds at the floor for a
`+/-0.05` claim:

```
closed form                0.880000
8M-draw estimate           0.879945
worst error seen           0.001216
claimed bound              0.05
claims violated            0/20
```

The bound held every time with 41x margin on the worst case. Read that
as the bound doing its job rather than as it being useless: McDiarmid
plus Fournier-Guillin is a worst-case guarantee over all functionals
satisfying the declared constants, and your particular functional is
usually much better behaved than the worst one.

If you want a tight interval, use the bound to size the run and then
report the empirical spread across seeds. If you want a guarantee,
report the bound. They are different claims.

[Example 03](examples.md) works this through end to end, including the
empirical check above.

## The stability modulus

When both Lipschitz constants are declared, the report also carries

```python
r.stability_modulus   # Lambda * L
```

This is the Theorem 5.4 modulus with the family-specific mixing constant
`C` factored out, so downstream analyses multiply by their own `C`. It
bounds how much the invariance can move per unit of perturbation
intensity, which is the quantity to compare against when you are
deciding whether an observed change is larger than the theory permits.
