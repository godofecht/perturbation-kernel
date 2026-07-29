"""Does my finding survive the noise I already know about?

The situation
-------------
A pipeline assigns each sample to one of five categories, and reports
that the population sits in category 0. Before publishing that, you
want to know how fragile the assignment is: your inputs carry noise,
and noise occasionally pushes a sample into a different category.

The question is not "is p < 0.05". It is: *at what level of input
noise does my reported category stop coming back?* That is a property
of the pipeline, it has units, and it does not depend on a threshold
anyone chose by convention.

What this script does
---------------------
Sweeps the perturbation intensity and measures how often the reported
category survives. Then it locates the largest noise level at which
survival still clears a tolerance you set -- the honest robustness
limit of the claim.

Run:  python 01_does_my_finding_survive.py
"""

import perturbation_kernel as pk

CATEGORIES = 5
REPORTED = 0

# Large enough that the sampling error (~1/sqrt(N) = 0.002) is well
# below the resolution we care about.
N = 262_144
SEED = 20260610

# The tightest additive-error claim this N supports. Asking for less
# than this is rejected outright rather than reported optimistically;
# example 03 shows how to work out the number for a target of your own.
EPSILON = 0.05


# How much category churn you are willing to tolerate. At 0.80, one
# sample in five landing elsewhere is the point where you stop calling
# the assignment stable.
TOLERANCE = 0.80


def survival(theta_max: float) -> pk.Report:
    """Fraction of perturbed samples still in the reported category."""
    cfg = pk.Config(
        n=N,
        seed=SEED,
        invariance_lambda=1.0,
        forward_l=1.0,
        # Survival is an indicator, so it lives in [0, 1] and the
        # observation diameter is exactly 1.
        epsilon=EPSILON,
        eta=0.05,
        observation_diameter=1.0,
        obs_dim=1,
    )
    return pk.Markov(
        k=CATEGORIES, start=REPORTED, base_label=REPORTED, theta_max=theta_max
    ).run(cfg)


def main() -> None:
    print(__doc__.split("Run:")[0].strip())
    print()
    print(f"categories        {CATEGORIES}")
    print(f"reported category {REPORTED}")
    print(f"draws             {N:,}   seed {SEED}")
    print(f"accuracy claim    +/- {EPSILON} at 95%")
    print()

    print(f"{'noise theta':>12}  {'survival':>10}  {'+/- bound':>10}  verdict")
    print("-" * 56)

    curve = []
    for theta in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = survival(theta)
        eps = r.error_bound["epsilon"]
        curve.append((theta, r.value))
        if r.value >= TOLERANCE:
            verdict = "holds"
        elif r.value >= TOLERANCE - 0.1:
            verdict = "weakening"
        else:
            verdict = "GONE"
        print(f"{theta:12.2f}  {r.value:10.4f}  {eps:10.4f}  {verdict}")

    print()

    # Locate the tolerance crossing by bisection. Survival is monotone
    # decreasing in theta, so this is well posed.
    #
    # Note the family's own floor: mixing to uniform over k categories
    # leaves a 1/k chance of landing back where you started, so
    # survival bottoms out at 1 - (1 - 1/k)/2 and never reaches zero.
    # Comparing against that floor rather than against 0 is what keeps
    # the reading honest.
    floor = 1.0 - (1.0 - 1.0 / CATEGORIES) / 2.0
    print(f"This family cannot push survival below {floor:.4f}")
    print(f"(mixing over {CATEGORIES} categories returns to the start 1 in {CATEGORIES} times).")
    print()

    if survival(1.0).value >= TOLERANCE:
        print(f"Survival never falls below the {TOLERANCE:.2f} tolerance, even at")
        print("maximum intensity. The finding is robust across the whole family.")
        limit = None
    else:
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if survival(mid).value >= TOLERANCE:
                lo = mid
            else:
                hi = mid
        limit = 0.5 * (lo + hi)
        print(f"Survival crosses the {TOLERANCE:.2f} tolerance at theta = {limit:.4f}.")
        print()
        print("Read that as: the reported category is stable as long as your")
        print(f"input noise stays below theta = {limit:.3f}. Above it, more than")
        print(f"{(1 - TOLERANCE) * 100:.0f}% of samples land in a different category.")

    print()
    print("Compare that number to the noise you actually measured. If your")
    print("noise is above it, the finding is not there. No significance test")
    print("can rescue it, because the pipeline itself does not reproduce it.")


if __name__ == "__main__":
    main()
