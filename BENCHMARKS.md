# Benchmarks

Measured by CI, not by hand. Regenerated on every push that
touches `src/`, `benches/` or `Cargo.toml`, and monthly so the
numbers keep tracking toolchain drift.

- **Machine** AMD EPYC 7763 64-Core Processor, 4 cores
- **Toolchain** rustc 1.97.1 (8bab26f4f 2026-07-14)
- **Commit** `f67a39b74e05`
- **Run** [30414197015](../../actions/runs/30414197015)

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
| `tree_sum`, N = 1,024 | 3.8 us | 398 ns | **9.58x** |
| `sum_sq_dev`, N = 1,024 | 4.0 us | 415 ns | **9.61x** |
| `tree_sum`, N = 65,536 | 64.0 us | 34.5 us | **1.86x** |
| `sum_sq_dev`, N = 65,536 | 81.5 us | 22.3 us | **3.65x** |
| `tree_sum`, N = 1,048,576 | 1.1 ms | 685.4 us | **1.56x** |
| `sum_sq_dev`, N = 1,048,576 | 5.5 ms | 505.6 us | **10.78x** |

### Vectorisation alone

Against an allocation-free scalar loop with the same tree
shape, so this is the vector unit and nothing else.

| | scalar | vectorised | speedup |
|---|---|---|---|
| `tree_sum`, N = 1,024 | 867 ns | 398 ns | **2.18x** |
| `tree_sum`, N = 65,536 | 57.5 us | 34.5 us | **1.67x** |
| `tree_sum`, N = 1,048,576 | 978.4 us | 685.4 us | **1.43x** |

## Engine

`Backend::Scalar` (one thread, scalar loops) against
`Backend::Auto` (threaded draws, vectorised reduction).
Both produce identical bits.

| | scalar | auto | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 16,384 | 4.0 ms | 1.7 ms | **2.36x** |
| `bistable`, N = 16,384 | 3.0 ms | 950.6 us | **3.13x** |
| `markov`, N = 16,384 | 2.9 ms | 960.6 us | **3.03x** |
| `gaussian_d3`, N = 262,144 | 62.9 ms | 25.7 ms | **2.44x** |
| `bistable`, N = 262,144 | 47.1 ms | 15.3 ms | **3.07x** |
| `markov`, N = 262,144 | 46.1 ms | 15.3 ms | **3.01x** |

## Flat storage

The `Family` path knows the observation width, so it writes
into one buffer instead of allocating per draw.

| | trait path | `Family` path | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 262,144 | 26.2 ms | 18.2 ms | **1.44x** |
