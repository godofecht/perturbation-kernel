# Backends

There is one estimator and five ways to evaluate it. Four of them
produce identical bits. One does not, and you have to ask for it by
name.

```python
pk.Config(n=1_000_000, seed=1, backend="auto")     # default
pk.Config(n=1_000_000, seed=1, backend="scalar")
pk.Config(n=1_000_000, seed=1, backend="simd")
pk.Config(n=1_000_000, seed=1, backend="gpu")      # exact, or an error
pk.Config(n=1_000_000, seed=1, backend="gpu_f32")  # fast, approximate
```

| Backend | Where it runs | Agreement with the reference |
|---|---|---|
| `scalar` | one thread, portable loops | it *is* the reference |
| `simd` | NEON or AVX2 reductions | bit-identical |
| `auto` | vectorised, threaded above 4096 draws | bit-identical |
| `gpu` | `wgpu` compute, emulated `f64` | bit-identical |
| `gpu_f32` | `wgpu` compute, single precision | statistically equivalent |

`auto` is the default and is almost always the right answer.

## Why SIMD is exact

This is the part people reasonably disbelieve, so here is the argument.

The reduction collapses one level at a time:

```
[a b c d e]  ->  [a+b  c+d  e]  ->  [a+b+c+d  e]  ->  [a+b+c+d+e]
```

Each output of a level is **one IEEE-754 addition of one specific pair
of inputs**. A vector unit that performs four of those additions
simultaneously performs the same four additions. There is no
accumulator carried across lanes, no reassociation, no horizontal sum.

On aarch64 the primitive is `vpaddq_f64`, which takes `[a0 a1]` and
`[a2 a3]` and returns `[a0+a1, a2+a3]`. That is exactly one tree level
for two outputs in one instruction. On x86-64, `vhaddpd` does the same
thing but interleaves across the 128-bit lane boundary, so a single
`vpermpd` puts the results back in index order.

The centring step in the variance path is written as an explicit
subtract followed by a multiply rather than a fused multiply-add,
because an FMA rounds once where the reference rounds twice.

`tests/reduce.rs` asserts bit-equality against the v1.0.0 reduction on
every length from 0 to 4096, on every available vector path, plus
property tests over random inputs and explicit coverage of infinities
and NaN.

## Why threading is invisible

Draw `i` uses the RNG substream `fork(seed, i)`, which depends on
`(seed, i)` and nothing else. Results are collected in index order.
A run split across fourteen cores and a run on one core visit the same
substreams and reduce them in the same tree, so they produce the same
bits.

The switch happens at 4096 draws (`engine::PARALLEL_MIN`). Below that
the pool's fork/join overhead costs more than the work. The threshold
changes when the work happens and never what it computes;
`tests/backends.rs` runs the equivalence checks at 4095, 4096 and 4097
specifically to pin that boundary.

## Why the GPU can be exact

WGSL has no `f64`, which is the usual reason a device result differs
from a host one. It is not an insurmountable reason.

A family qualifies for exact device evaluation when its arithmetic is a
short list of exactly-specified operations. `Markov` qualifies. It draws
one uniform intensity, one uniform variate and one bounded integer, and
its observation is an indicator. Three things then line up:

**Emulated binary64.** `src/gpu/shaders/f64.wgsl` implements IEEE-754
doubles in `u32` pairs, with round-to-nearest-even multiplication. The
uniform path performs exactly one rounded multiply, so one emulated
operation closes the gap. The rest of the conversion is exact by
construction because the integers involved fit in 53 bits.

**A transcribed integer sampler.** `rand`'s `gen_range` is Lemire's
multiply-shift with rejection: pure integer arithmetic, exact anywhere.
It is transcribed rather than reimplemented, down to using the
conservative `(range << range.leading_zeros()) - 1` rejection zone that
`rand` uses for 32-bit types. Using the modulus form instead, which
`rand` reserves for `i8` and `i16`, is *statistically identical* and
bit-wise different. That bug survived every aggregate check and was
caught only by comparing bits.

**An integer reduction.** The observation is `0` or `1`, so the ensemble
is a count. Integer addition is associative, so the total does not
depend on how the device schedules anything. The host's `f64` tree sum
of zeros and ones is that same integer, exactly.

The result: `to_bits()` equality across 297 combinations of alphabet
size, intensity, ensemble size and seed, checked in `tests/gpu.rs`. Each
layer underneath is pinned separately, so a failure names the layer that
broke: the keystream against the host word for word, and the
`(low, scale)` derivation against `rand` itself.

### What does not qualify

`Gaussian` and `Bistable` draw normal deviates. The ziggurat in
`rand_distr` calls `ln` and `exp` on its rejection paths, and WGSL
specifies the accuracy of neither. Emulating them in software `f64`
would be a libm port and would run slower than the CPU it is meant to
beat.

So `backend="gpu"` refuses them, rather than quietly returning a
different number:

```python
pk.Gaussian(base=[0.5], sigma_max=0.3).run(
    pk.Config(n=1024, seed=1, backend="gpu")
)
# RuntimeError: backend gpu unavailable: the gaussian family draws normal
# deviates, which need transcendental functions WGSL does not specify
# exactly; use backend "gpu_f32" to accept a single-precision result, or
# a host backend for an exact one
```

## `gpu_f32`: fast and approximate

`gpu_f32` runs every family, carries the ensemble in `f32`, and draws
normals by Box-Muller. It is the faster of the two device backends and
the less reproducible one.

| Property | Holds? |
|---|---|
| Same device, same `(seed, n)`, repeated runs | yes, bit-identical |
| Same device, different workgroup scheduling | yes, bit-identical |
| Different devices, `Markov` family | yes |
| Different devices, `Gaussian` / `Bistable` | last-ulp differences possible |
| Device vs. host | statistically equivalent |

The device draws from the same ChaCha20 substreams as the host and only
the float transforms differ, which keeps the agreement much tighter than
two independent runs would be. At 200,000 draws:

```
 gaussian: host -0.089898693  device -0.090028523  |gap| 1.298e-4
 bistable: host +0.001880000  device -0.002010000  |gap| 3.890e-3
   markov: host +0.880235000  device +0.880535000  |gap| 3.000e-4
```

The Monte Carlo standard error at that `n` is about `2.2e-3`, so the
gaps are inside the noise of the estimate itself.

The reduction is exact here too. Every output is one IEEE-754 `f32`
addition of one fixed pair, and the levels ping-pong between two buffers
so no invocation can read a slot another has overwritten.
`tests/gpu.rs` checks it against a host `f32` tree, bit for bit,
including at odd lengths and across workgroup boundaries.

### Which is faster

Measured on an Apple M4 Max, `Markov`, best of three:

| draws | host | `gpu` | | `gpu_f32` | |
|---|---|---|---|---|---|
| 10,000 | 0.42 ms | 0.31 ms | 1.4x | 1.11 ms | 0.4x |
| 50,000 | 2.16 ms | 0.34 ms | 6.3x | 0.96 ms | 2.2x |
| 200,000 | 7.26 ms | 0.66 ms | 11.0x | 1.01 ms | 7.2x |
| 1,000,000 | 31.08 ms | 2.40 ms | 12.9x | 1.29 ms | 24.1x |
| 4,000,000 | 112.65 ms | 3.44 ms | 32.7x | 2.61 ms | 43.1x |

The ordering is not the one you would guess. Below about a million
draws the exact backend is the *faster* of the two, because its
reduction is a single atomic counter while `gpu_f32` allocates buffers
and runs a dispatch per tree level. The emulated double-precision
multiply costs less than that. Above a million the fixed cost stops
mattering and single precision pulls ahead on raw throughput.

So the usual advice inverts: reach for `gpu` first. It is exact, and up
to about a million draws it is also quicker. Reach for `gpu_f32` when
you are running very large ensembles, or a family the exact path
refuses.

Reproduce the table on your own hardware with
[example 04](examples.md).

### What the GPU will not do

`Engine::run` with `backend = "gpu"` is an error, not a silent fallback:

```
UnsupportedBackend { backend: "gpu", reason: "a device cannot invoke
user Rust trait impls; use family::Family for GPU execution" }
```

A compute device cannot call back into arbitrary Rust trait
implementations. GPU execution is available for the three built-in
families, which exist as data precisely so they can be sent to a device.
Custom models run on the host.

## Provenance

Every report records how it was produced:

```python
r = pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=8192, seed=1))
r.execution
# {'backend': 'auto', 'simd_path': 'neon', 'threaded': True,
#  'device': None, 'precision': 'f64'}
```

A device result carries `precision: 'f32'` and the device name, so it
can never be mistaken for a host result when someone finds the JSON
six months later.

## Turning things off

```toml
perturbation-kernel = { version = "2", default-features = false }
```

drops `rayon` and the intrinsics. The build gets smaller and slower and
computes exactly the same numbers. The feature matrix in CI runs the
full test suite under every combination for that reason.
