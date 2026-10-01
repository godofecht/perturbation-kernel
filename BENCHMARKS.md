# Benchmarks

Measured by CI, not by hand. Regenerated on every push that
touches `src/`, `benches/` or `Cargo.toml`, and monthly so the
numbers keep tracking toolchain drift.

- **Machine** AMD EPYC 7763 64-Core Processor, 4 cores
- **Toolchain** rustc 1.98.1 (48a229cea 2026-09-01)
- **Commit** `5a4661151a6d`
- **Run** [36851731783](../../actions/runs/36851731783)

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
| `tree_sum`, N = 1,024 | 3.6 us | 399 ns | **9.11x** |
| `sum_sq_dev`, N = 1,024 | 3.8 us | 415 ns | **9.17x** |
| `tree_sum`, N = 65,536 | 61.0 us | 32.4 us | **1.88x** |
| `sum_sq_dev`, N = 65,536 | 74.6 us | 22.3 us | **3.35x** |
| `tree_sum`, N = 1,048,576 | 1.0 ms | 545.4 us | **1.91x** |
| `sum_sq_dev`, N = 1,048,576 | 4.9 ms | 409.9 us | **11.94x** |

### Vectorisation alone

Against an allocation-free scalar loop with the same tree
shape, so this is the vector unit and nothing else.

| | scalar | vectorised | speedup |
|---|---|---|---|
| `tree_sum`, N = 1,024 | 842 ns | 399 ns | **2.11x** |
| `tree_sum`, N = 65,536 | 55.4 us | 32.4 us | **1.71x** |
| `tree_sum`, N = 1,048,576 | 902.3 us | 545.4 us | **1.65x** |

## Engine

`Backend::Scalar` (one thread, scalar loops) against
`Backend::Auto` (threaded draws, vectorised reduction).
Both produce identical bits.

| | scalar | auto | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 16,384 | 3.9 ms | 1.7 ms | **2.33x** |
| `bistable`, N = 16,384 | 3.0 ms | 945.5 us | **3.15x** |
| `markov`, N = 16,384 | 2.9 ms | 928.2 us | **3.13x** |
| `gaussian_d3`, N = 262,144 | 61.6 ms | 24.5 ms | **2.52x** |
| `bistable`, N = 262,144 | 47.1 ms | 15.1 ms | **3.12x** |
| `markov`, N = 262,144 | 45.9 ms | 14.7 ms | **3.13x** |

## Flat storage

The `Family` path knows the observation width, so it writes
into one buffer instead of allocating per draw.

| | trait path | `Family` path | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 262,144 | 25.1 ms | 17.6 ms | **1.42x** |
