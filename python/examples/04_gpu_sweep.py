"""When is the GPU actually worth it?

The situation
-------------
You have a parameter sweep to run and a GPU sitting idle. The
temptation is to send everything to the device. The reality is that a
device has a fixed cost per run -- buffers to allocate, a pipeline to
dispatch, a result to copy back -- and below some ensemble size that
cost is the whole runtime.

What this script does
---------------------
Times all three backends across a range of ensemble sizes and checks
what each one actually promises:

* ``gpu`` is bit-identical to the host, so the check is on the bits;
* ``gpu_f32`` is single precision, so the check is against the Monte
  Carlo error.

Then it runs a real two-dimensional sweep on whichever backend the
measurement says is right.

Run:  python 04_gpu_sweep.py
"""

import math
import time

import perturbation_kernel as pk

FAMILY = pk.Markov(k=5, theta_max=0.3)
SEED = 20260610


def timed(family, n, backend, repeats=3):
    """Best-of-`repeats` wall time and the value produced."""
    cfg = pk.Config(n=n, seed=SEED, backend=backend)
    best = float("inf")
    value = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = family.run(cfg).value
        best = min(best, time.perf_counter() - t0)
    return best, value


def main() -> None:
    print(__doc__.split("Run:")[0].strip())
    print()

    print(f"backends available  {pk.available_backends()}")
    print(f"host vector path    {pk.simd_path()}")
    device = pk.gpu_device()
    print(f"device              {device or 'none'}")
    print()

    if device is None:
        print("No compute device on this machine, so there is nothing to")
        print("compare. The host backends are used automatically and every")
        print("other example in this directory runs unchanged.")
        return

    print(f"{'draws':>12}  {'host (ms)':>10}  {'gpu (ms)':>10}  {'gpu x':>7}  "
          f"{'exact?':>7}  {'f32 (ms)':>10}  {'f32 x':>7}  {'|gap|':>9}")
    print("-" * 84)

    crossover = None
    all_exact = True
    for n in [10_000, 50_000, 200_000, 1_000_000, 4_000_000]:
        t_host, v_host = timed(FAMILY, n, "auto")
        t_gpu, v_gpu = timed(FAMILY, n, "gpu")
        t_f32, v_f32 = timed(FAMILY, n, "gpu_f32")

        exact = v_host.hex() == v_gpu.hex()
        all_exact &= exact
        gap = abs(v_host - v_f32)
        if t_host / t_gpu > 1.0 and crossover is None:
            crossover = n
        print(f"{n:12,}  {t_host * 1e3:10.2f}  {t_gpu * 1e3:10.2f}  "
              f"{t_host / t_gpu:6.2f}x  {'yes' if exact else 'NO':>7}  "
              f"{t_f32 * 1e3:10.2f}  {t_host / t_f32:6.2f}x  {gap:9.2e}")

    print()
    if all_exact:
        print("The `exact?` column is the point of the `gpu` backend: every")
        print("value came back bit-identical to the host, not close to it.")
    else:
        print("A `gpu` result did NOT match the host. That is a bug; please")
        print("report it with the numbers above.")
    print()
    print("`gpu_f32` is faster still, because it skips the emulated")
    print("double-precision arithmetic and the atomic readback. It pays for")
    print("that with single precision, so its column is checked against the")
    print("Monte Carlo error rather than against the bits.")
    print()

    if crossover is None:
        print("The host wins at every size measured, so use it and keep the")
        print("bit-identical guarantee for free.")
        backend = "auto"
    else:
        print(f"The device starts winning at about {crossover:,} draws. Below that")
        print("the fixed cost of allocating buffers, dispatching and copying")
        print("the result back dominates.")
        backend = "gpu"
    print()

    # ------------------------------------------------------------------
    # A real sweep, on whichever backend won.
    # ------------------------------------------------------------------
    print(f"Two-dimensional sweep on backend '{backend}'")
    print()
    print("Survival probability against alphabet size and mixing intensity.")
    print()

    ks = [2, 4, 8, 16, 32]
    thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
    n = 500_000

    header = "  k \\ theta " + "".join(f"{t:>9.2f}" for t in thetas)
    print(header)
    print("-" * len(header))

    t0 = time.perf_counter()
    for k in ks:
        row = f"{k:>11}"
        for theta in thetas:
            cfg = pk.Config(n=n, seed=SEED, backend=backend)
            v = pk.Markov(k=k, theta_max=theta).run(cfg).value
            row += f"{v:>9.4f}"
        print(row)
    elapsed = time.perf_counter() - t0

    total = len(ks) * len(thetas) * n
    print()
    print(f"{len(ks) * len(thetas)} cells, {n:,} draws each, {total:,} draws total")
    print(f"in {elapsed:.2f} s -- {total / elapsed / 1e6:.1f}M draws/second")
    print()
    print("The exact answer is 1 - (theta/2)(1 - 1/k), so the table can be")
    print("checked by eye: survival falls linearly in theta, and the fall is")
    print("steeper for larger alphabets because there are more wrong places")
    print("to land.")


if __name__ == "__main__":
    main()
