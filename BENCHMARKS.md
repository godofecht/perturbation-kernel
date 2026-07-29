# Benchmarks

Measured by CI, not by hand. Regenerated on every push that
touches `src/`, `benches/` or `Cargo.toml`, and monthly so the
numbers keep tracking toolchain drift.

- **Machine** AMD EPYC 7763 64-Core Processor, 4 cores
- **Toolchain** rustc 1.97.1 (8bab26f4f 2026-07-14)
- **Commit** `d4a308be8cff`
- **Run** [30414466330](../../actions/runs/30414466330)

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
| `tree_sum`, N = 1,024 | 3.6 us | 398 ns | **9.10x** |
| `sum_sq_dev`, N = 1,024 | 3.8 us | 420 ns | **9.05x** |
| `tree_sum`, N = 65,536 | 60.0 us | 31.8 us | **1.89x** |
| `sum_sq_dev`, N = 65,536 | 74.0 us | 21.9 us | **3.38x** |
| `tree_sum`, N = 1,048,576 | 979.8 us | 530.8 us | **1.85x** |
| `sum_sq_dev`, N = 1,048,576 | 4.6 ms | 401.1 us | **11.54x** |

### Vectorisation alone

Against an allocation-free scalar loop with the same tree
shape, so this is the vector unit and nothing else.

| | scalar | vectorised | speedup |
|---|---|---|---|
| `tree_sum`, N = 1,024 | 846 ns | 398 ns | **2.12x** |
| `tree_sum`, N = 65,536 | 54.6 us | 31.8 us | **1.72x** |
| `tree_sum`, N = 1,048,576 | 886.5 us | 530.8 us | **1.67x** |

## Engine

`Backend::Scalar` (one thread, scalar loops) against
`Backend::Auto` (threaded draws, vectorised reduction).
Both produce identical bits.

| | scalar | auto | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 16,384 | 3.9 ms | 1.7 ms | **2.37x** |
| `bistable`, N = 16,384 | 3.0 ms | 945.2 us | **3.15x** |
| `markov`, N = 16,384 | 2.9 ms | 923.0 us | **3.15x** |
| `gaussian_d3`, N = 262,144 | 62.3 ms | 24.2 ms | **2.57x** |
| `bistable`, N = 262,144 | 47.0 ms | 15.0 ms | **3.14x** |
| `markov`, N = 262,144 | 46.0 ms | 14.6 ms | **3.16x** |

## Flat storage

The `Family` path knows the observation width, so it writes
into one buffer instead of allocating per draw.

| | trait path | `Family` path | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 262,144 | 24.2 ms | 17.7 ms | **1.37x** |
