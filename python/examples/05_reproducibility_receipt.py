"""Publishing a number someone else can check.

The situation
-------------
You are about to put a number in a paper, a report, or a dashboard.
Six months from now someone -- possibly you -- will want to know
whether it is still true, and whether the code still produces it.

"Available on request" does not survive contact with reality. What
does is a receipt: the config, the family, and the value, small enough
to paste into an appendix and complete enough to re-run.

What this script does
---------------------
Produces a receipt, then verifies one. The verification is the part
that matters: it re-runs from the receipt alone and checks the value
comes back bit for bit, so a receipt that has drifted from the code
fails loudly instead of quietly.

Run:  python 05_reproducibility_receipt.py
"""

from __future__ import annotations

import json

import perturbation_kernel as pk


def issue(family: pk.Family, config: pk.Config) -> dict:
    """Run, and return everything needed to reproduce the run."""
    report = family.run(config)
    return {
        "family": family.to_dict(),
        "config": json.loads(config.to_json()),
        # The strict SCHEMA v1.0.0 payload, so another implementation
        # of the schema can read it without knowing about this one.
        "report": json.loads(report.to_json(v1=True)),
        # The exact bits, because a decimal rendering of a float is a
        # lossy summary and "reproduces to 6 decimal places" is a much
        # weaker claim than the engine actually supports.
        "value_bits": report.value.hex(),
        "library": {
            "package": "perturbation-kernel",
            "version": pk.__version__,
            "schema_version": pk.SCHEMA_VERSION,
        },
    }


def verify(receipt: dict) -> tuple[bool, str]:
    """Re-run from a receipt and check the value comes back exactly."""
    fam_spec = dict(receipt["family"])
    kind = fam_spec.pop("family")
    family = {
        "gaussian": pk.Gaussian,
        "bistable": pk.Bistable,
        "markov": pk.Markov,
    }[kind](**fam_spec)

    config = pk.Config.from_json(json.dumps(receipt["config"]))
    value = family.run(config).value

    if value.hex() != receipt["value_bits"]:
        return False, (
            f"value changed: receipt {receipt['value_bits']} "
            f"({float.fromhex(receipt['value_bits'])!r}), "
            f"got {value.hex()} ({value!r})"
        )
    if pk.SCHEMA_VERSION.split(".")[0] != receipt["library"]["schema_version"].split(".")[0]:
        return False, "schema major version has moved on"
    return True, f"reproduced exactly: {value!r}"


def main() -> None:
    print(__doc__.split("Run:")[0].strip())
    print()

    family = pk.Markov(k=5, theta_max=0.3)
    config = pk.Config(
        n=262_144,
        seed=20260610,
        invariance_lambda=1.0,
        forward_l=1.0,
        epsilon=0.05,
        eta=0.05,
        observation_diameter=1.0,
        obs_dim=1,
    )

    receipt = issue(family, config)

    print("The receipt")
    print("-----------")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print()

    print("What you would write")
    print("--------------------")
    r = receipt["report"]
    print(
        f"  Survival probability {r['value']:.4f} "
        f"(+/-{r['error_bound']['epsilon']:.4f} at {1 - r['error_bound']['eta']:.0%}), "
    )
    print(f"  n = {r['n_effective']:,}, seed = {r['seed']}.")
    print()

    print("Verifying it")
    print("------------")
    ok, message = verify(receipt)
    print(f"  {'PASS' if ok else 'FAIL'}: {message}")
    print()

    # A receipt that has drifted must fail, or the check is theatre.
    print("Verifying a receipt that has been tampered with")
    print("-----------------------------------------------")
    tampered = json.loads(json.dumps(receipt))
    tampered["family"]["theta_max"] = 0.31
    ok, message = verify(tampered)
    print(f"  {'PASS' if ok else 'FAIL'}: {message}")
    print()

    print("Verifying the same run on a different backend")
    print("---------------------------------------------")
    for backend in ("scalar", "simd"):
        alt = json.loads(json.dumps(receipt))
        alt["config"]["backend"] = backend
        ok, message = verify(alt)
        print(f"  {backend:>7}: {'PASS' if ok else 'FAIL'} -- {message}")
    print()
    print("  The host backends are bit-identical, so a reviewer on different")
    print("  hardware with a different core count reproduces the same bits.")
    print()

    if pk.gpu_device() is not None:
        print("Verifying on the GPU")
        print("--------------------")
        alt = json.loads(json.dumps(receipt))
        alt["config"]["backend"] = "gpu"
        ok, message = verify(alt)
        print(f"  {'PASS' if ok else 'FAIL'} -- {message}")
        print()
        print("  Expected. The device is single precision and draws normals by")
        print("  a different algorithm, so it agrees statistically and not bit")
        print("  for bit. A receipt is a claim about an exact computation, so")
        print("  verify it on the backend that issued it -- which is why every")
        print("  report records its own `execution` block.")


if __name__ == "__main__":
    main()
