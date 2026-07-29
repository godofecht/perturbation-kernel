# Concepts

## The question

You have a pipeline. It takes inputs and produces a number. You would
like to know whether that number is a property of the world or a
property of the particular inputs you happened to get.

The perturbative answer is to nudge the inputs and look. Not once, but
across a family of nudges with an intensity you control, so that the
output is a curve rather than a single verdict: *the result holds up to
this much noise, and stops holding beyond it.*

That curve has units. It can be compared against the error bars on your
instrument. It does not require choosing a null hypothesis or a
significance threshold.

## The four objects

The schema insists on exactly four things. Each one is a trait in Rust
and a field in the wire format.

### The state space and its metric

Whatever your pipeline operates on: a vector, a scalar, a label, a data
frame. The only requirement is a metric `d_O` on the *observation*
space, because the error bounds are stated in the Wasserstein-1 distance
and that needs a ground metric.

### The perturbation family

A Markov kernel `P((s, theta), .)` that takes a state and an intensity
and returns a perturbed state, together with a sampler `rho` for the
intensity itself.

The load-bearing part is the **null parameter** `theta_0`. The contract
is that `apply(s, theta_0)` returns `s` in distribution. Without it,
"perturb and recover" is not a meaningful phrase, because there is no
baseline to recover to. The engine cross-checks the null parameter
declared in the config against the one the implementation reports, and
refuses to run if they disagree.

```python
# theta_max = 0 is the null perturbation, so this is exactly 1.0.
pk.Markov(k=5, theta_max=0.0, start=2, base_label=2).run(cfg).value
```

### The forward model

A measurable map `F: S -> O` from the state to what you actually
observe. It must be a pure function. If your readout is itself
stochastic, that randomness belongs in the perturbation, not here, so
that `F` stays a map and its pushforward stays well defined.

Optionally it declares a Lipschitz constant `L`. Declaring it is what
lets the engine surface an error bound.

### The invariance functional

A scalar `Phi` on the empirical measure of the ensemble. It must be
permutation-invariant, because it is a function of the multiset of
observations rather than the sequence.

Optionally it declares a Wasserstein-1 Lipschitz constant `Lambda`. Not
every useful functional has one. The empirical variance, for instance,
is not globally `W_1`-Lipschitz on an unbounded space, and the built-in
`Gaussian` family reports `None` rather than inventing a constant. The
consequence is honest: no declared constant means no error bound.

## The estimator

With those four objects the engine computes the plug-in estimator:

```
for i in 0..N:
    rng_i   = fork(seed, i)
    theta_i = rho.sample(rng_i)
    S_i     = P(s, theta_i, rng_i)
    Y_i     = F(S_i)

value = Phi(empirical measure of Y)
```

Two details in there carry the whole reproducibility story.

**`fork(seed, i)`** gives draw `i` its own RNG substream, derived from
`(seed, i)` alone. Draw 5 does not depend on whether draws 0 through 4
have run, or in what order, or on how many threads. That is what makes
the loop safe to parallelise without changing the answer.

**`Phi` reduces in a fixed tree order.** Floating-point addition is not
associative, so a sum whose grouping depends on the thread count
produces different bits on different machines. The reduction here always
collapses `buf[k] = buf[2k] + buf[2k+1]` with an odd tail carried up, a
shape fixed by the index order and nothing else.

## The report

```json
{
  "schema_version": "1.0.0",
  "value": 0.8802871704101562,
  "functional": "tail_survival",
  "n_effective": 262144,
  "seed": 20260610,
  "reduction": { "order": "tree", "leaf_order": "index" },
  "error_bound": {
    "available": true,
    "epsilon": 0.027021005040057862,
    "eta": 0.05,
    "basis": "mcdiarmid+fournier_guillin",
    "constants": { "lambda": 1.0, "d": 1.0, "obs_dim": 1 }
  },
  "stability_modulus": 1.0
}
```

Everything needed to re-run it, and everything needed to say how much to
trust it. `execution` is added on top of this by default and records
which backend produced the value; `to_json(v1=True)` strips it for a
strict v1.0.0 payload.

## What this is not

It is not a significance test. There is no null hypothesis and no
threshold. If you want a binary verdict you have to supply the tolerance
yourself, in the units of your problem, and that is the point.

It is not a sensitivity analysis in the derivative sense. A gradient
tells you about an infinitesimal neighbourhood of the operating point.
This tells you about a neighbourhood of the size you actually care
about, which for a nonlinear system is a different question with a
different answer. [Example 02](examples.md) shows a case where the two
diverge.

It is not a bootstrap. A bootstrap resamples the data you have, so it
estimates sampling variability under the empirical distribution. This
perturbs the data by a noise model you specify, so it estimates
robustness under the error process you believe in. They answer different
questions and you may well want both.
