# Benchmarks

Measured by CI, not by hand. Regenerated on every push that
touches `src/`, `benches/` or `Cargo.toml`, and monthly so the
numbers keep tracking toolchain drift.

- **Machine** INTEL(R) XEON(R) PLATINUM 8573C, 4 cores
- **Toolchain** rustc 1.99.1 (58cf8f9c3 2026-10-13)
- **Commit** `59921f6a9a0a`
- **Run** [41300755462](../../actions/runs/41300755462)

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
| `tree_sum`, N = 1,024 | 988 ns | 337 ns | **2.93x** |
| `sum_sq_dev`, N = 1,024 | 1.2 us | 314 ns | **3.68x** |
| `tree_sum`, N = 65,536 | 80.3 us | 30.7 us | **2.61x** |
| `sum_sq_dev`, N = 65,536 | 96.9 us | 21.5 us | **4.50x** |
| `tree_sum`, N = 1,048,576 | 1.8 ms | 1.2 ms | **1.46x** |
| `sum_sq_dev`, N = 1,048,576 | 9.4 ms | 842.1 us | **11.17x** |

### Vectorisation alone

Against an allocation-free scalar loop with the same tree
shape, so this is the vector unit and nothing else.

| | scalar | vectorised | speedup |
|---|---|---|---|
| `tree_sum`, N = 1,024 | 749 ns | 337 ns | **2.22x** |
| `tree_sum`, N = 65,536 | 52.2 us | 30.7 us | **1.70x** |
| `tree_sum`, N = 1,048,576 | 1.3 ms | 1.2 ms | **1.06x** |

## Engine

`Backend::Scalar` (one thread, scalar loops) against
`Backend::Auto` (threaded draws, vectorised reduction).
Both produce identical bits.

| | scalar | auto | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 16,384 | 3.8 ms | 1.8 ms | **2.16x** |
| `bistable`, N = 16,384 | 3.2 ms | 1.1 ms | **2.89x** |
| `markov`, N = 16,384 | 3.1 ms | 1.1 ms | **2.87x** |
| `gaussian_d3`, N = 262,144 | 69.2 ms | 27.8 ms | **2.49x** |
| `bistable`, N = 262,144 | 51.7 ms | 17.5 ms | **2.95x** |
| `markov`, N = 262,144 | 50.5 ms | 17.3 ms | **2.92x** |

## Flat storage

The `Family` path knows the observation width, so it writes
into one buffer instead of allocating per draw.

| | trait path | `Family` path | speedup |
|---|---|---|---|
| `gaussian_d3`, N = 262,144 | 27.8 ms | 20.5 ms | **1.36x** |
