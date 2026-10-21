"""Trace the complete perturbation-kernel estimator on a tractable model.

    python python/examples/07_kernel_mechanics.py

S = {0,1,2,3,4}.  Theta is uniform on [0, theta_max].
P((s,theta), y) = (1-theta) * 1[y=s] + theta/5.
F(y) = 1[y=s].  Phi(nu) = integral F(y) d nu(y).
The kernel samples theta, samples y conditional on theta, applies F,
and averages the binary readout with a fixed reduction tree.
Its population expectation is 1 - theta_max*(1-1/5)/2.

This reproduces the statistical model, not the PRNG's private draw stream.
"""
from __future__ import annotations

import math

import perturbation_kernel as pk

K = 5
N = 65_536
SEED = 20261022


def conditional_survival(theta: float) -> float:
    """P(Y = s | theta) for the Markov family."""
    return 1.0 - theta + theta / K


def expected_survival(theta_max: float) -> float:
    """Integrate the conditional probability over uniform theta."""
    return 1.0 - (1.0 - 1.0 / K) * theta_max / 2.0


def main() -> None:
    cfg = pk.Config(n=N, seed=SEED, invariance_lambda=1.0)
    print("S = 0..4; s = 0; Theta ~ Uniform[0, theta_max]")
    print("P((s,theta), s) = 1-theta+theta/5; F(y) = 1[y=s]")
    print("Phi(nu) = nu({s}); estimate = mean(F(Y_i))")
    print(f"draw count = {N}, seed = {SEED}")
    print(f"{'theta_max':>10} {'E[theta]':>10} {'P(Y=s)':>10} {'estimate':>10} {'gap':>10} {'s.e.':>10}")
    for theta_max in (0.0, 0.3, 0.5, 1.0):
        family = pk.Markov(k=K, theta_max=theta_max)
        observed = family.run(cfg).value
        expected = expected_survival(theta_max)
        se = math.sqrt(expected * (1.0 - expected) / N)
        gap = abs(observed - expected)
        print(
            f"{theta_max:10.2f} {theta_max / 2:10.4f} "
            f"{expected:10.5f} {observed:10.5f} {gap:10.5f} {se:10.5f}"
        )
        assert gap < 0.02, (theta_max, observed, expected)
        assert family.run(cfg).value == observed
        if theta_max == 0:
            assert observed == 1.0, "the null perturbation must preserve the label"

    assert abs(conditional_survival(0.3) - 0.76) < 1e-12
    assert expected_survival(0.3) == 0.88 or abs(expected_survival(0.3) - 0.88) < 1e-12
    print("Verified null identity, population agreement and fixed-seed reproducibility.")
    print("Standard errors describe Bernoulli Monte Carlo variability, not CI guarantees.")


if __name__ == "__main__":
    main()
