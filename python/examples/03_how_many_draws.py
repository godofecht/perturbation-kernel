"""How many draws do I need to claim a number?

The situation
-------------
You want to publish "the survival probability is 0.88, plus or minus
0.01, at 95% confidence". Before you can write that sentence you have
to know how many draws it takes to earn it.

Guessing is the usual approach, and it is usually wrong in the
expensive direction: people pick a round number like a million,
discover afterwards that the bound they can actually defend is much
looser than the one they wrote down, and quietly drop the interval.

What this script does
---------------------
Works backwards from the claim to the ensemble size, using the
Theorem 7.3(c) floor. Then it checks the answer empirically, because a
sample-size formula you have never tested against the estimator it
governs is a formula you are trusting on faith.

Run:  python 03_how_many_draws.py
"""

import perturbation_kernel as pk

# The claim we want to make. The observations here are indicators, so
# they live in {0, 1}: the observation diameter is exactly 1 and the
# functional is 1-Lipschitz in the Wasserstein-1 metric.
ETA = 0.05  # 95% confidence
DIAMETER = 1.0
LAMBDA = 1.0
OBS_DIM = 1

FAMILY = pk.Markov(k=5, theta_max=0.3)


def main() -> None:
    print(__doc__.split("Run:")[0].strip())
    print()

    print("Required draws for a target additive error")
    print()
    print(f"{'target +/-':>12}  {'draws needed':>16}")
    print("-" * 32)
    for eps in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
        floor = pk.sample_floor(LAMBDA, DIAMETER, eps, ETA, OBS_DIM)
        print(f"{eps:12.2f}  {floor:16,}")

    print()
    print("Note how steeply this climbs. Tightening the claim from +/-0.1 to")
    print("+/-0.01 costs 256 times the compute, and it is the bias term, not")
    print("the variance term, that dominates: the Fournier-Guillin rate for")
    print("the empirical measure converges more slowly than the McDiarmid")
    print("concentration does. Halving your error bar is not a 4x job.")
    print()

    # ------------------------------------------------------------------
    # Now check the floor is real rather than decorative.
    # ------------------------------------------------------------------
    target = 0.05
    floor = pk.sample_floor(LAMBDA, DIAMETER, target, ETA, OBS_DIM)
    print(f"Checking the floor for a +/-{target} claim ({floor:,} draws)")
    print()

    # One draw short must be refused.
    try:
        FAMILY.run(
            pk.Config(
                n=floor - 1, seed=1, invariance_lambda=LAMBDA,
                epsilon=target, eta=ETA,
                observation_diameter=DIAMETER, obs_dim=OBS_DIM,
            )
        )
        print("  FAIL: an under-powered claim was accepted")
    except ValueError as e:
        print(f"  n = {floor - 1:,} is refused: {str(e).split(':')[0]}")

    cfg = pk.Config(
        n=floor, seed=1, invariance_lambda=LAMBDA,
        epsilon=target, eta=ETA,
        observation_diameter=DIAMETER, obs_dim=OBS_DIM,
    )
    r = FAMILY.run(cfg)
    print(f"  n = {floor:,} is accepted, bound +/-{r.error_bound['epsilon']:.4f}")
    print()

    # ------------------------------------------------------------------
    # Does the estimator actually land inside the interval it promises?
    # ------------------------------------------------------------------
    print("Does the estimator honour the interval?")
    print()

    # A high-n run is the stand-in for the truth. The exact answer for
    # this family is 1 - (theta_max / 2)(1 - 1/k), which we can check
    # against.
    exact = 1.0 - (0.3 / 2.0) * (1.0 - 1.0 / 5.0)
    reference = FAMILY.run(pk.Config(n=8_000_000, seed=999)).value
    print(f"  closed form                {exact:.6f}")
    print(f"  8M-draw estimate           {reference:.6f}")
    print(f"  gap                        {abs(exact - reference):.2e}")
    print()

    misses = 0
    trials = 20
    worst = 0.0
    for seed in range(trials):
        v = FAMILY.run(
            pk.Config(
                n=floor, seed=seed, invariance_lambda=LAMBDA,
                epsilon=target, eta=ETA,
                observation_diameter=DIAMETER, obs_dim=OBS_DIM,
            )
        ).value
        err = abs(v - exact)
        worst = max(worst, err)
        if err > target:
            misses += 1

    print(f"  {trials} independent seeds at n = {floor:,}")
    print(f"  worst error seen           {worst:.6f}")
    print(f"  claimed bound              {target}")
    print(f"  claims violated            {misses}/{trials}")
    print()
    if misses == 0:
        print(f"  The bound held every time, with {target / worst:.0f}x margin on the worst case.")
        print("  It is a bound, not an estimate, so being loose is the point:")
        print("  McDiarmid plus Fournier-Guillin is deliberately conservative.")
    else:
        print(f"  {misses} violations in {trials} trials against a {ETA:.0%} failure budget.")

    print()
    print("The takeaway: pick the claim first, derive n from it, and let the")
    print("library refuse the runs that cannot support it.")


if __name__ == "__main__":
    main()
