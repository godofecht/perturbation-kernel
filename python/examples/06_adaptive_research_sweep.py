"""Adaptive sensitivity search for a statistical finding.

Run:
    python python/examples/06_adaptive_research_sweep.py

We ask how much random relabelling a five-class result tolerates before
its probability of surviving drops below 0.8. The kernel estimates that
probability by Monte Carlo. A binary search needs just 10 kernel
evaluations, versus 257 for a uniform grid of the same resolution.
The closed-form expectation supplies an independent validation target.

This example measures elapsed time locally. Evaluation-count reduction is
deterministic; runtime speedup depends on hardware and other workloads.
"""
from __future__ import annotations

import time

import perturbation_kernel as pk

K = 5
TARGET = 0.8
N = 32_768
SEED = 707
GRID_STEPS = 256
ITERATIONS = 8


def exact_survival(theta: float) -> float:
    """Expected agreement for uniform [0, theta] relabelling intensity."""
    return 1.0 - theta * (1.0 - 1.0 / K) / 2.0


def estimate(theta: float) -> float:
    config = pk.Config(n=N, seed=SEED, invariance_lambda=1.0)
    return pk.Markov(k=K, theta_max=theta).run(config).value


def grid_search() -> tuple[float, int, float]:
    started = time.perf_counter()
    crossing = 1.0
    evaluations = 0
    for i in range(GRID_STEPS + 1):
        theta = i / GRID_STEPS
        value = estimate(theta)
        evaluations += 1
        if value <= TARGET and crossing == 1.0:
            crossing = theta
    return crossing, evaluations, time.perf_counter() - started


def adaptive_search() -> tuple[float, int, float]:
    started = time.perf_counter()
    lo, hi = 0.0, 1.0
    assert estimate(lo) > TARGET
    assert estimate(hi) < TARGET
    for _ in range(ITERATIONS):
        mid = (lo + hi) / 2.0
        if estimate(mid) > TARGET:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0, ITERATIONS + 2, time.perf_counter() - started


def main() -> None:
    exact = 2 * (1.0 - TARGET) / (1.0 - 1.0 / K)
    grid, grid_calls, grid_seconds = grid_search()
    adaptive, adaptive_calls, adaptive_seconds = adaptive_search()
    assert abs(exact_survival(exact) - TARGET) < 1e-12
    assert abs(grid - exact) < 0.04, (grid, exact)
    assert abs(adaptive - exact) < 0.04, (adaptive, exact)
    assert estimate(0.3) == estimate(0.3), "fixed-seed determinism"

    print("Five-category result: minimum acceptable survival", TARGET)
    print(f"Exact population threshold: {exact:.5f}")
    print(f"Grid estimate:              {grid:.5f}")
    print(f"Adaptive estimate:          {adaptive:.5f}")
    print(f"Grid:     {grid_calls:3d} evaluations, {grid_seconds:.4f} s")
    print(f"Adaptive: {adaptive_calls:3d} evaluations, {adaptive_seconds:.4f} s")
    print(f"Kernel evaluations avoided: {grid_calls - adaptive_calls}")
    print(f"Evaluation-count reduction: {grid_calls / adaptive_calls:.1f}x")
    if adaptive_seconds > 0:
        print(f"Observed runtime ratio:     {grid_seconds / adaptive_seconds:.2f}x")
    print("Search tolerance: 0.04 intensity units (finite-sample Monte Carlo).")
    print("No sequentially valid confidence interval is claimed.")


if __name__ == "__main__":
    main()
