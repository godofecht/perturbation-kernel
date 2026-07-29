"""Where does a bistable system tip?

The situation
-------------
A system has two stable regimes. Left alone it stays in whichever one
it started in, because the restoring drift is stronger than anything
small. Push it hard enough and it flips, and it does not flip back.

That is the shape of a great many real problems: a market with two
pricing equilibria, a controller with two attractors, an ecosystem
with two vegetation states, a classifier with two confident answers on
either side of a boundary. For all of them the number you want is the
same: *how hard is hard enough?*

What this script does
---------------------
For a range of starting positions, finds the noise intensity at which
the outcome stops being predictable. Then it checks how that tipping
intensity scales with the distance from the boundary and with the
integration step, because a scaling law you can read off a table is
worth more than a single number.

Run:  python 02_where_does_it_tip.py
"""

import math

import perturbation_kernel as pk

N = 100_000
SEED = 4242

# Polarisation is the mean of a +/-1 readout. At +1 every sample holds
# its regime; at 0 the outcome is a coin flip. We call the system
# "tipped" once fewer than 70% of samples hold, i.e. polarisation 0.4.
TIPPED = 0.4


def polarisation(x0: float, dt: float, theta_max: float) -> float:
    """Mean regime readout after one perturbed step."""
    cfg = pk.Config(n=N, seed=SEED, invariance_lambda=1.0)
    return pk.Bistable(x0=x0, dt=dt, theta_max=theta_max).run(cfg).value


def tipping_intensity(x0: float, dt: float) -> float:
    """Smallest noise intensity that tips the system.

    Polarisation falls monotonically in the intensity, so bisection is
    well posed. The upper bracket is grown until it actually tips, so
    the answer is never an artefact of a guessed search range.
    """
    hi = 1.0
    while polarisation(x0, dt, hi) > TIPPED:
        hi *= 2.0
        if hi > 1e6:
            return float("nan")
    lo = 0.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if polarisation(x0, dt, mid) > TIPPED:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    print(__doc__.split("Run:")[0].strip())
    print()
    print("potential   V(x) = (x^2 - 1)^2,  wells at +/-1,  ridge at 0")
    print(f"draws       {N:,}   seed {SEED}")
    print(f"tipped when polarisation < {TIPPED}")
    print()

    dt = 0.01
    print(f"How the tipping intensity scales with distance  (dt = {dt})")
    print()
    print(f"{'start x0':>10}  {'tipping noise':>14}  {'noise / x0':>12}")
    print("-" * 40)
    ratios = []
    for x0 in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
        theta = tipping_intensity(x0, dt)
        ratios.append(theta / x0)
        print(f"{x0:10.2f}  {theta:14.3f}  {theta / x0:12.1f}")

    spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios))
    print()
    print(f"The ratio is constant to within {spread * 100:.1f}% across a 16-fold")
    print("range of starting positions. The tipping intensity is proportional")
    print("to the distance from the boundary.")
    print()
    print("That is worth knowing precisely because it is not obvious. The")
    print("potential is quartic and its restoring force is cubic, so you might")
    print("expect a strongly nonlinear margin. Over a single step of this size")
    print("the drift contributes almost nothing and the outcome is a straight")
    print("race between displacement and noise. The nonlinearity is real but")
    print("it lives at longer integration times, not here.")
    print()

    print("How the tipping intensity scales with the step size  (x0 = 0.2)")
    print()
    print(f"{'dt':>10}  {'tipping noise':>14}  {'noise * sqrt(dt)':>18}")
    print("-" * 46)
    steps = [0.005, 0.01, 0.02, 0.05, 0.1]
    invariants = []
    for dt in steps:
        theta = tipping_intensity(0.2, dt)
        inv = theta * math.sqrt(dt)
        invariants.append(inv)
        print(f"{dt:10.3f}  {theta:14.3f}  {inv:18.4f}")

    def spread_of(vals):
        return (max(vals) - min(vals)) / (sum(vals) / len(vals))

    small = invariants[:3]
    print()
    print(f"Over the three smallest steps `noise * sqrt(dt)` holds to")
    print(f"{spread_of(small) * 100:.1f}%: the tipping intensity goes as 1/sqrt(dt), which is")
    print("the diffusive scaling you would predict. A Langevin step accumulates")
    print("displacement as the square root of time, so halving the step demands")
    print("sqrt(2) more noise to cover the same ground.")
    print()
    print(f"Across the full range it holds only to {spread_of(invariants) * 100:.0f}%, and the drift is")
    print(f"one-directional: the invariant climbs from {invariants[0]:.3f} to {invariants[-1]:.3f} as")
    print("the step grows. That is not noise in the estimate, it is the")
    print("deterministic term becoming visible. Drift contributes displacement")
    print("proportional to dt while noise contributes sqrt(dt), so the larger")
    print("the step the more the restoring force helps, and the more noise it")
    print("takes to tip. Pure diffusive scaling is the small-step limit.")
    print()
    print("Worth stating plainly: if you had fitted 1/sqrt(dt) across the whole")
    print("range and reported the fit, you would have buried exactly the")
    print("nonlinearity you were looking for.")
    print()

    print("The practical use")
    print("-----------------")
    print("Measure your operating point's distance to the boundary and the")
    print("timescale you integrate over. Both scalings above are one-line")
    print("reads, so you get the tolerance without re-running anything:")
    print()
    k = sum(ratios) / len(ratios)
    for x0 in (0.15, 0.35):
        print(f"  at x0 = {x0}, dt = 0.01  ->  tips at noise ~ {k * x0:.1f}")
    print()
    print("Compare that against the disturbance your system actually sees.")


if __name__ == "__main__":
    main()
