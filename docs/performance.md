# Performance

Everything on this page is measured, not asserted. Reproduce it with:

```bash
cargo bench --bench kernel
```

Numbers below are from an Apple M4 Max (14 CPU cores, 32 GPU cores),
`rustc` 1.92, release profile with thin LTO. Treat them as
order-of-magnitude guidance: they were taken on a laptop, and the
parallel figures in particular move by tens of percent between runs
depending on thermal state.

Every benchmark compares an optimised path against a reference path
that computes the same number, so the ratios mean something.

## Reductions

Against `reduce::reference`, which is the literal v1.0.0 code:

| | N | v1.0.0 | vectorised | |
|---|---|---|---|---|
| `tree_sum` | 1,024 | 0.8 us | 0.3 us | 2.97x |
| `tree_sum` | 65,536 | 114.3 us | 27.5 us | 4.15x |
| `tree_sum` | 1,048,576 | 1787.1 us | 474.0 us | 3.77x |
| `sum_sq_dev` | 1,024 | 0.9 us | 0.3 us | 3.08x |
| `sum_sq_dev` | 65,536 | 117.2 us | 17.4 us | 6.73x |
| `sum_sq_dev` | 1,048,576 | 1621.6 us | 408.2 us | 3.97x |

Two changes are stacked there. Isolating them, against an
allocation-free scalar loop with the same tree shape:

| | N | scalar | NEON | |
|---|---|---|---|---|
| `tree_sum` | 1,024 | 0.5 us | 0.3 us | 1.83x |
| `tree_sum` | 65,536 | 49.1 us | 27.5 us | 1.78x |
| `tree_sum` | 1,048,576 | 581.4 us | 474.0 us | 1.23x |

So roughly half the win is dropping the allocations and half is the
vector unit. The v1.0.0 reduction allocated a fresh `Vec` for every
level of the tree, which is `log2(N)` allocations per reduction; the
current one allocates once and collapses in place.

The NEON advantage narrows at a megabyte because the reduction becomes
memory-bound. Two doubles per instruction does not help when you are
waiting on DRAM. `sum_sq_dev` keeps more of its advantage because the
squaring is fused into the first tree level, so the array of squares is
never materialised and the pass reads half as much.

## Engine

`Backend::Scalar` (one thread, scalar loops) against `Backend::Auto`
(threaded draws, vectorised reduction):

| family | N | scalar | auto | |
|---|---|---|---|---|
| gaussian (d=3) | 16,384 | 9.0 ms | 1.8 ms | 5.14x |
| bistable | 16,384 | 5.7 ms | 2.9 ms | 1.97x |
| markov | 16,384 | 5.6 ms | 3.5 ms | 1.57x |
| gaussian (d=3) | 262,144 | 158.0 ms | 54.6 ms | 2.90x |
| bistable | 262,144 | 163.8 ms | 47.2 ms | 3.47x |
| markov | 262,144 | 149.1 ms | 37.3 ms | 4.00x |

Three to five times on fourteen cores is well short of linear, and it is
worth being clear about why rather than quoting the best row.

The draw loop is dominated by re-keying a `ChaCha20` generator for every
index. That is the price of the per-draw substream, and it is the thing
that makes the loop reproducible under any thread count. It is also
memory- and instruction-heavy enough that fourteen threads contend
rather than scale.

The generic `Engine::run` path additionally allocates one observation
per draw, because it cannot know what type the forward model returns.
For a boxed slice that is a call into the allocator per draw, and the
allocator is a shared resource.

## Flat storage

The `Family` path knows the observation is scalar or fixed-width, so it
writes into one flat buffer:

| N | trait path | `Family` path | |
|---|---|---|---|
| 262,144 | 52.1 ms | 41.8 ms | 1.25x |

Modest, and it is the reason the built-in families exist as data rather
than only as trait implementations. The larger payoff is that the same
representation is what a GPU can consume.

## GPU

Measured through the Python bindings, `Markov`, best of three:

| draws | host | `gpu` | | `gpu_f32` | |
|---|---|---|---|---|---|
| 10,000 | 0.42 ms | 0.31 ms | 1.4x | 1.11 ms | 0.4x |
| 50,000 | 2.16 ms | 0.34 ms | 6.3x | 0.96 ms | 2.2x |
| 200,000 | 7.26 ms | 0.66 ms | 11.0x | 1.01 ms | 7.2x |
| 1,000,000 | 31.08 ms | 2.40 ms | 12.9x | 1.29 ms | 24.1x |
| 4,000,000 | 112.65 ms | 3.44 ms | 32.7x | 2.61 ms | 43.1x |

Two things here are worth reading twice.

The exact backend is 32.7x faster than the host at four million draws
*and* returns the same bits. Exactness is not being paid for in speed
at that scale.

And below a million draws the exact backend beats the single-precision
one. Its reduction is a single atomic counter, where `gpu_f32` allocates
buffers and dispatches once per tree level; the emulated
double-precision multiply costs less than that overhead. The ordering
only inverts once the ensemble is large enough for throughput to
dominate the fixed cost.

A 25-cell sweep at 500,000 draws per cell, 12.5M draws total, completes
in 0.01 s on the exact backend.

Reproduce with `python python/examples/04_gpu_sweep.py`, which checks
the bits on `gpu` and the Monte Carlo error on `gpu_f32` at every size.

## Choosing

- Under about 10,000 draws: the host. The device's fixed cost is the
  whole runtime below that.
- From there to about a million: `gpu`. It is exact *and* the fastest
  of the three.
- Above a million, if you can accept single precision: `gpu_f32`.
- For a family `gpu` refuses, or on a machine with no device: `auto`,
  which is bit-identical to `scalar` and several times faster.

## What was not optimised

The `ChaCha20` re-key per draw is the dominant cost on the host and it
was left alone on purpose. A cheaper substream construction would be
faster and would change every value the crate has ever produced, which
is a trade the golden check exists to prevent anyone making by accident.
Changing it would be a major version bump with the divergence stated up
front.
