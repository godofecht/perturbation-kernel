# Benchmarks

Measured by CI, not by hand. Regenerated on every push that
touches `src/`, `benches/` or `Cargo.toml`, and monthly so the
numbers keep tracking toolchain drift.

- **Machine** AMD EPYC 7763 64-Core Processor, 4 cores
- **Toolchain** rustc 1.97.1 (8bab26f4f 2026-07-14)
- **Commit** `6daea909fc2e`
- **Run** [30413098672](../../actions/runs/30413098672)

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
| `tree_sum`, N = 1,024 | 3.6 us | 398 ns | **9.13x** |
| `sum_sq_dev`, N = 1,024 | 3.8 us | 414 ns | **9.17x** |
| `tree_sum`, N = 65,536 | 63.9 us | 34.6 us | **1.85x** |
| `sum_sq_dev`, N = 65,536 | 81.1 us | 22.3 us | **3.63x** |
| `tree_sum`, N = 1,048,576 | 1.0 ms | 667.4 us | **1.53x** |
| `sum_sq_dev`, N = 1,048,576 | 4.3 ms | 467.8 us | **9.28x** |

### Vectorisation alone

Against an allocation-free scalar loop with the same tree
shape, so this is the vector unit and nothing else.

| | scalar | vectorised | speedup |
|---|---|---|---|
| `tree_sum`, N = 1,024 | 863 ns | 398 ns | **2.17x** |
| `tree_sum`, N = 65,536 | 57.6 us | 34.6 us | **1.67x** |
| `tree_sum`, N = 1,048,576 | 943.0 us | 667.4 us | **1.41x** |

## Engine

`Backend::Scalar` (one thread, scalar loops) against
`Backend::Auto` (threaded draws, vectorised reduction).
Both produce identical bits.

| | scalar | auto | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 16,384 | 3.9 ms | 1.6 ms | **2.36x** |
| `bistable`, N = 16,384 | 3.0 ms | 944.6 us | **3.13x** |
| `markov`, N = 16,384 | 2.9 ms | 919.7 us | **3.16x** |
| `gaussian_d3`, N = 262,144 | 60.9 ms | 23.9 ms | **2.55x** |
| `bistable`, N = 262,144 | 46.6 ms | 15.0 ms | **3.11x** |
| `markov`, N = 262,144 | 45.9 ms | 14.5 ms | **3.16x** |

## Flat storage

The `Family` path knows the observation width, so it writes
into one buffer instead of allocating per draw.

| | trait path | `Family` path | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 262,144 | 23.9 ms | 17.4 ms | **1.37x** |
