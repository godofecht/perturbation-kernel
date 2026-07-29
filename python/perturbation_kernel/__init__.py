"""Reproducible perturbation-kernel estimators.

A perturbation kernel answers one question: *how much does a result
move when you nudge the thing that produced it?* You supply a base
state, a family of random perturbations, a forward map to whatever you
actually observe, and a scalar functional of the resulting ensemble.
The estimator returns that scalar plus a non-asymptotic error bound.

The whole point is that the answer is reproducible. A run is a pure
function of ``(family, n, seed)``, so the same three inputs give the
same bits on any machine, any thread count, and any CPU vector width.

Quick start
-----------

>>> import perturbation_kernel as pk
>>> cfg = pk.Config(n=100_000, seed=20260610, invariance_lambda=1.0)
>>> report = pk.Markov(k=5, theta_max=0.3).run(cfg)
>>> round(report.value, 6)  # doctest: +SKIP
0.880235

Backends
--------

``Config(backend=...)`` selects how the estimator is evaluated:

``"auto"`` (default)
    Vectorised reductions and a thread pool. Bit-identical to
    ``"scalar"``.
``"scalar"``
    One thread, portable loops. The reference path.
``"simd"``
    Force vectorisation even on small inputs. Bit-identical to
    ``"scalar"``.
``"gpu"``
    Run on a compute device, **bit-identically to the host**. Available
    for the families a device can reproduce exactly, which today means
    :class:`Markov`; the others raise rather than quietly returning a
    different number.
``"gpu_f32"``
    Run on a compute device in single precision. Supports every family
    and is faster, but carries the ensemble in ``f32`` and draws
    normals by Box-Muller rather than the ziggurat, so it agrees with
    the host statistically rather than bit for bit.

``Report.execution`` always records which path produced a value, so a
device result can never be mistaken for a host one.

Call :func:`available_backends` to see what this build can actually
run; the device entries appear only when the wheel has GPU support
*and* a device is present.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence

from . import _perturbation_kernel as _ext
from ._perturbation_kernel import (
    Config,
    Report,
    SCHEMA_VERSION,
    available_backends,
    gpu_device,
    sample_floor,
    simd_path,
    tree_sum,
)

__version__ = _ext.__version__

__all__ = [
    "Config",
    "Report",
    "SCHEMA_VERSION",
    "__version__",
    "Family",
    "Gaussian",
    "Bistable",
    "Markov",
    "run",
    "sweep",
    "available_backends",
    "gpu_device",
    "sample_floor",
    "simd_path",
    "tree_sum",
]


class Family:
    """Base class for the built-in perturbation families.

    A family bundles a perturbation kernel, a forward model and an
    invariance functional -- the three objects the schema requires --
    into one value you can serialise, send to a GPU, or hand to
    :func:`run`.
    """

    #: Tag matching the ``family`` field of the JSON descriptor.
    name: str = ""

    def run(self, config: Config) -> Report:
        """Evaluate this family under ``config``."""
        return run(config, self)

    def to_dict(self) -> Dict[str, Any]:
        """The JSON descriptor for this family, as a dict."""
        d = asdict(self)  # type: ignore[call-overload]
        d["family"] = self.name
        return d


@dataclass(frozen=True)
class Gaussian(Family):
    """Gaussian shift in ``R^d``.

    The state is perturbed by ``sigma * N(0, I)`` with the intensity
    ``sigma`` drawn uniformly from ``[0, sigma_max]``. The forward
    model is the identity and the invariance is the negative empirical
    dispersion, so a *larger* value means a more stable result.

    Parameters
    ----------
    base:
        The unperturbed state. Its length fixes ``d``.
    sigma_max:
        Upper end of the intensity range. ``sigma_max = 0`` is the null
        perturbation and must return dispersion exactly ``0``.
    """

    base: Sequence[float]
    sigma_max: float
    name = "gaussian"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "gaussian",
            "base": list(self.base),
            "sigma_max": self.sigma_max,
        }


@dataclass(frozen=True)
class Bistable(Family):
    """Bistable double-well marble in ``V(x) = (x^2 - 1)^2``.

    One Euler-Maruyama Langevin step of size ``dt`` with noise
    intensity drawn uniformly from ``[0, theta_max]``, read out as the
    sign of the well the marble lands in. The invariance is the
    polarisation, i.e. the mean readout, which lies in ``[-1, 1]``.

    This is the example to reach for when the question is "does a
    strong enough nudge flip the outcome?"
    """

    x0: float
    dt: float
    theta_max: float
    name = "bistable"


@dataclass(frozen=True)
class Markov(Family):
    """Finite-state chain under epsilon-uniform mixing.

    With probability ``theta`` the label is replaced by a uniform draw
    on ``0..k``; otherwise it survives. The invariance is the empirical
    survival probability of ``base_label``, a number in ``[0, 1]``.

    Draws no normal deviates, so this is the family whose GPU path is
    pure integer arithmetic and tracks the host most closely.
    """

    k: int
    theta_max: float
    start: int = 0
    base_label: int = 0
    name = "markov"


def run(config: Config, family: Family) -> Report:
    """Run ``family`` under ``config`` and return the report.

    Raises
    ------
    ValueError
        The config or the family violates the schema: a zero ensemble,
        an incompatible major schema version, an accuracy claim below
        the Theorem 7.3(c) sample floor, or hyperparameters outside the
        family's domain.
    RuntimeError
        ``backend="gpu"`` was requested but no device is usable, or the
        wheel was built without GPU support.
    """
    if isinstance(family, Gaussian):
        return _ext.run_gaussian(config, list(family.base), family.sigma_max)
    if isinstance(family, Bistable):
        return _ext.run_bistable(config, family.x0, family.dt, family.theta_max)
    if isinstance(family, Markov):
        return _ext.run_markov(
            config, family.k, family.start, family.base_label, family.theta_max
        )
    raise TypeError(
        f"expected a Family (Gaussian, Bistable or Markov), got {type(family).__name__}"
    )


def sweep(config: Config, families: Iterable[Family]) -> List[Report]:
    """Run several families under one config.

    Each run is independent and keyed by the same seed, so the sweep is
    reproducible as a whole and comparable point to point.

    >>> import perturbation_kernel as pk
    >>> cfg = pk.Config(n=10_000, seed=1)
    >>> vals = [r.value for r in pk.sweep(
    ...     cfg, [pk.Markov(k=5, theta_max=t) for t in (0.0, 0.5)]
    ... )]
    >>> vals[0] > vals[1]
    True
    """
    return [run(config, f) for f in families]
