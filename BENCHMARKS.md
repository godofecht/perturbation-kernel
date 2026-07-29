# Benchmarks

Measured by CI, not by hand. Regenerated on every push that
touches `src/`, `benches/` or `Cargo.toml`, and monthly so the
numbers keep tracking toolchain drift.

- **Machine** AMD EPYC 7763 64-Core Processor, 4 cores
- **Toolchain** rustc 1.97.1 (8bab26f4f 2026-07-14)
- **Commit** `b19796bebf34`
- **Run** [30443447957](../../actions/runs/30443447957)

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
| `tree_sum`, N = 1,024 | 4.0 us | 411 ns | **9.63x** |
| `sum_sq_dev`, N = 1,024 | 4.1 us | 417 ns | **9.82x** |
| `tree_sum`, N = 65,536 | 77.3 us | 33.3 us | **2.32x** |
| `sum_sq_dev`, N = 65,536 | 87.7 us | 24.1 us | **3.64x** |
| `tree_sum`, N = 1,048,576 | 966.8 us | 517.4 us | **1.87x** |
| `sum_sq_dev`, N = 1,048,576 | 4.7 ms | 397.8 us | **11.81x** |

### Vectorisation alone

Against an allocation-free scalar loop with the same tree
shape, so this is the vector unit and nothing else.

| | scalar | vectorised | speedup |
|---|---|---|---|
| `tree_sum`, N = 1,024 | 849 ns | 411 ns | **2.07x** |
| `tree_sum`, N = 65,536 | 54.9 us | 33.3 us | **1.65x** |
| `tree_sum`, N = 1,048,576 | 878.1 us | 517.4 us | **1.70x** |

## Engine

`Backend::Scalar` (one thread, scalar loops) against
`Backend::Auto` (threaded draws, vectorised reduction).
Both produce identical bits.

| | scalar | auto | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 16,384 | 3.9 ms | 1.6 ms | **2.37x** |
| `bistable`, N = 16,384 | 3.0 ms | 949.7 us | **3.12x** |
| `markov`, N = 16,384 | 2.9 ms | 927.7 us | **3.10x** |
| `gaussian_d3`, N = 262,144 | 61.0 ms | 24.0 ms | **2.54x** |
| `bistable`, N = 262,144 | 47.5 ms | 15.0 ms | **3.16x** |
| `markov`, N = 262,144 | 46.1 ms | 14.6 ms | **3.15x** |

## Flat storage

The `Family` path knows the observation width, so it writes
into one buffer instead of allocating per draw.

| | trait path | `Family` path | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 262,144 | 24.4 ms | 17.6 ms | **1.39x** |
