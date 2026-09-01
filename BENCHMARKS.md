# Benchmarks

Measured by CI, not by hand. Regenerated on every push that
touches `src/`, `benches/` or `Cargo.toml`, and monthly so the
numbers keep tracking toolchain drift.

- **Machine** AMD EPYC 7763 64-Core Processor, 4 cores
- **Toolchain** rustc 1.98.0 (88d9e12ae 2026-08-18)
- **Commit** `f1ad927a682b`
- **Run** [33491656237](../../actions/runs/33491656237)

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
| `tree_sum`, N = 1,024 | 3.7 us | 399 ns | **9.20x** |
| `sum_sq_dev`, N = 1,024 | 4.4 us | 415 ns | **10.54x** |
| `tree_sum`, N = 65,536 | 64.0 us | 34.8 us | **1.84x** |
| `sum_sq_dev`, N = 65,536 | 82.2 us | 22.5 us | **3.65x** |
| `tree_sum`, N = 1,048,576 | 1.1 ms | 671.0 us | **1.58x** |
| `sum_sq_dev`, N = 1,048,576 | 4.7 ms | 470.0 us | **9.97x** |

### Vectorisation alone

Against an allocation-free scalar loop with the same tree
shape, so this is the vector unit and nothing else.

| | scalar | vectorised | speedup |
|---|---|---|---|
| `tree_sum`, N = 1,024 | 841 ns | 399 ns | **2.11x** |
| `tree_sum`, N = 65,536 | 57.3 us | 34.8 us | **1.65x** |
| `tree_sum`, N = 1,048,576 | 948.7 us | 671.0 us | **1.41x** |

## Engine

`Backend::Scalar` (one thread, scalar loops) against
`Backend::Auto` (threaded draws, vectorised reduction).
Both produce identical bits.

| | scalar | auto | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 16,384 | 3.9 ms | 1.7 ms | **2.32x** |
| `bistable`, N = 16,384 | 3.0 ms | 943.7 us | **3.16x** |
| `markov`, N = 16,384 | 2.9 ms | 922.9 us | **3.13x** |
| `gaussian_d3`, N = 262,144 | 61.5 ms | 23.9 ms | **2.57x** |
| `bistable`, N = 262,144 | 47.2 ms | 15.0 ms | **3.15x** |
| `markov`, N = 262,144 | 45.6 ms | 14.6 ms | **3.13x** |

## Flat storage

The `Family` path knows the observation width, so it writes
into one buffer instead of allocating per draw.

| | trait path | `Family` path | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 262,144 | 24.0 ms | 17.2 ms | **1.40x** |
