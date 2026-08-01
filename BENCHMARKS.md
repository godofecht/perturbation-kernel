# Benchmarks

Measured by CI, not by hand. Regenerated on every push that
touches `src/`, `benches/` or `Cargo.toml`, and monthly so the
numbers keep tracking toolchain drift.

- **Machine** AMD EPYC 7763 64-Core Processor, 4 cores
- **Toolchain** rustc 1.97.1 (8bab26f4f 2026-07-14)
- **Commit** `84f7f0ad7a28`
- **Run** [30688060980](../../actions/runs/30688060980)

A shared CI runner is a noisy place to measure. Treat these as
order-of-magnitude guidance; the parallel figures in particular
move between runs with whatever else the host is doing.

Every comparison is against a path that computes the *same
number*, so the ratios mean something.

## Reductions

Against `reduce::reference`, which is the literal v1.0.0 code.
Two changes stack here: the reduction no longer allocates a
`Vec` per tree level, and the levels are vectorised.

| | v1.0.0 | vectorised | speedup |
|---|---|---|---|
| `tree_sum`, N = 1,024 | 3.7 us | 410 ns | **9.00x** |
| `sum_sq_dev`, N = 1,024 | 3.9 us | 417 ns | **9.32x** |
| `tree_sum`, N = 65,536 | 60.2 us | 33.3 us | **1.81x** |
| `sum_sq_dev`, N = 65,536 | 74.0 us | 24.2 us | **3.06x** |
| `tree_sum`, N = 1,048,576 | 985.5 us | 635.7 us | **1.55x** |
| `sum_sq_dev`, N = 1,048,576 | 4.8 ms | 445.2 us | **10.86x** |

### Vectorisation alone

Against an allocation-free scalar loop with the same tree
shape, so this is the vector unit and nothing else.

| | scalar | vectorised | speedup |
|---|---|---|---|
| `tree_sum`, N = 1,024 | 868 ns | 410 ns | **2.12x** |
| `tree_sum`, N = 65,536 | 54.8 us | 33.3 us | **1.64x** |
| `tree_sum`, N = 1,048,576 | 951.8 us | 635.7 us | **1.50x** |

## Engine

`Backend::Scalar` (one thread, scalar loops) against
`Backend::Auto` (threaded draws, vectorised reduction).
Both produce identical bits.

| | scalar | auto | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 16,384 | 3.9 ms | 1.6 ms | **2.36x** |
| `bistable`, N = 16,384 | 3.0 ms | 945.4 us | **3.15x** |
| `markov`, N = 16,384 | 2.9 ms | 924.3 us | **3.10x** |
| `gaussian_d3`, N = 262,144 | 61.1 ms | 24.4 ms | **2.50x** |
| `bistable`, N = 262,144 | 47.7 ms | 15.0 ms | **3.17x** |
| `markov`, N = 262,144 | 46.1 ms | 14.5 ms | **3.17x** |

## Flat storage

The `Family` path knows the observation width, so it writes
into one buffer instead of allocating per draw.

| | trait path | `Family` path | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 262,144 | 24.1 ms | 17.6 ms | **1.37x** |
