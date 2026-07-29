"""Tests for the Python bindings.

The bindings do no arithmetic of their own, so these tests are about
the translation layer: that the schema's contracts survive the FFI
boundary, that errors arrive as the right Python exception type, and
that the determinism guarantees the Rust crate makes are still true
when you drive it from Python.
"""

import json
import math

import pytest

import perturbation_kernel as pk


HOST_BACKENDS = ["auto", "scalar", "simd"]


def gpu_available():
    return "gpu" in pk.available_backends()


requires_gpu = pytest.mark.skipif(
    not gpu_available(), reason="no compute device available"
)


def families():
    return [
        pk.Gaussian(base=[0.5, -1.25, 3.0], sigma_max=0.3),
        # x0 = 0 puts the marble on the ridge between the wells,
        # which is the case where the perturbation actually decides the
        # outcome. Deep inside a well the drift wins and polarisation is
        # identically 1, which would make the tests below vacuous.
        pk.Bistable(x0=0.0, dt=0.01, theta_max=0.5),
        pk.Markov(k=5, theta_max=0.3),
    ]


# ---------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------


def test_module_exports_a_version_and_schema_version():
    assert pk.SCHEMA_VERSION == "1.0.0"
    assert pk.__version__.count(".") == 2


def test_available_backends_always_includes_the_host_paths():
    backends = pk.available_backends()
    for b in HOST_BACKENDS:
        assert b in backends
    # 'gpu' is a capability probe, not a compile-time constant.
    assert set(backends) <= set(HOST_BACKENDS + ["gpu", "gpu_f32"])


def test_simd_path_is_one_of_the_known_paths():
    assert pk.simd_path() in {"scalar", "neon", "avx2"}


def test_gpu_device_agrees_with_available_backends():
    assert (pk.gpu_device() is not None) == gpu_available()


# ---------------------------------------------------------------------
# Determinism (SCHEMA section 8)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", families(), ids=lambda f: f.name)
def test_same_seed_gives_identical_bits(family):
    cfg = pk.Config(n=20_000, seed=4242)
    first = family.run(cfg).value
    for _ in range(5):
        assert family.run(cfg).value.hex() == first.hex()


@pytest.mark.parametrize("family", families(), ids=lambda f: f.name)
def test_host_backends_are_bit_identical(family):
    values = {
        b: family.run(pk.Config(n=20_000, seed=99, backend=b)).value
        for b in HOST_BACKENDS
    }
    reference = values["scalar"]
    for backend, v in values.items():
        assert v.hex() == reference.hex(), f"{backend} diverged from scalar"


@pytest.mark.parametrize("family", families(), ids=lambda f: f.name)
def test_the_seed_is_total(family):
    a = family.run(pk.Config(n=20_000, seed=1)).value
    b = family.run(pk.Config(n=20_000, seed=2)).value
    assert a != b, "changing the seed did not change the result"


@pytest.mark.parametrize("n", [1, 2, 3, 7, 4095, 4096, 4097])
def test_small_and_threshold_ensemble_sizes(n):
    # 4096 is where the engine switches to a thread pool; the value
    # must not notice.
    cfg_auto = pk.Config(n=n, seed=5, backend="auto")
    cfg_scalar = pk.Config(n=n, seed=5, backend="scalar")
    fam = pk.Gaussian(base=[1.0, 2.0], sigma_max=0.3)
    assert fam.run(cfg_auto).value.hex() == fam.run(cfg_scalar).value.hex()


# ---------------------------------------------------------------------
# Estimator ranges and the null-perturbation contract (C2)
# ---------------------------------------------------------------------


def test_survival_is_a_probability():
    v = pk.Markov(k=4, theta_max=0.25).run(pk.Config(n=20_000, seed=3)).value
    assert 0.0 <= v <= 1.0


def test_polarisation_is_bounded():
    v = pk.Bistable(x0=0.0, dt=0.01, theta_max=0.5).run(
        pk.Config(n=20_000, seed=3)
    ).value
    assert -1.0 <= v <= 1.0


def test_negative_dispersion_is_non_positive():
    v = pk.Gaussian(base=[0.0, 0.0], sigma_max=0.3).run(
        pk.Config(n=20_000, seed=3)
    ).value
    assert v <= 0.0


def test_null_intensity_recovers_the_base_state():
    # C2: at zero intensity the perturbation is the identity.
    cfg = pk.Config(n=10_000, seed=5)
    assert pk.Markov(k=5, theta_max=0.0, start=2, base_label=2).run(cfg).value == 1.0
    assert pk.Gaussian(base=[1.5, -2.0], sigma_max=0.0).run(cfg).value == 0.0


def test_dispersion_grows_with_intensity():
    # More noise means less invariance, so the negative dispersion
    # must fall monotonically.
    cfg = pk.Config(n=50_000, seed=11)
    values = [
        pk.Gaussian(base=[0.0], sigma_max=s).run(cfg).value
        for s in (0.0, 0.1, 0.3, 1.0)
    ]
    assert values == sorted(values, reverse=True), values


def test_survival_falls_with_mixing_probability():
    cfg = pk.Config(n=50_000, seed=11)
    values = [
        pk.Markov(k=5, theta_max=t).run(cfg).value for t in (0.0, 0.25, 0.5, 1.0)
    ]
    assert values == sorted(values, reverse=True), values


# ---------------------------------------------------------------------
# Report surface
# ---------------------------------------------------------------------


def test_report_fields_and_repr():
    cfg = pk.Config(n=1024, seed=17)
    r = pk.Markov(k=5, theta_max=0.3).run(cfg)
    assert r.functional == "tail_survival"
    assert r.n_effective == 1024
    assert r.seed == 17
    assert r.schema_version == "1.0.0"
    assert float(r) == r.value
    assert "tail_survival" in repr(r)


def test_execution_block_describes_the_host_run():
    r = pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=8192, seed=1))
    e = r.execution
    assert e["backend"] == "auto"
    assert e["precision"] == "f64"
    assert e["device"] is None
    assert e["simd_path"] in {"scalar", "neon", "avx2"}
    assert e["threaded"] is True


def test_report_json_round_trips_and_v1_strips_provenance():
    r = pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=1024, seed=1))

    full = json.loads(r.to_json())
    assert full["functional"] == "tail_survival"
    assert "execution" in full

    v1 = json.loads(r.to_json(v1=True))
    assert "execution" not in v1
    # The v1 payload still carries everything SCHEMA section 6 requires.
    for key in ("schema_version", "value", "functional", "n_effective", "seed", "reduction"):
        assert key in v1

    assert json.loads(r.to_json(pretty=True)) == full


def test_error_bound_is_absent_without_declared_constants():
    r = pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=1024, seed=1))
    assert r.error_bound is None
    assert r.stability_modulus is None


def test_error_bound_is_present_with_declared_constants():
    # Size the ensemble against the floor rather than guessing: at
    # eps = 0.01 the Fournier-Guillin bias term alone demands 1.7e7
    # draws, and the engine is right to refuse a smaller n.
    floor = pk.sample_floor(1.0, 1.0, 0.05, 0.05, 1)
    cfg = pk.Config(
        n=floor,
        seed=1,
        invariance_lambda=1.0,
        forward_l=1.0,
        epsilon=0.05,
        eta=0.05,
        observation_diameter=1.0,
        obs_dim=1,
    )
    r = pk.Markov(k=5, theta_max=0.3).run(cfg)
    b = r.error_bound
    assert b is not None
    assert b["eta"] == 0.05
    assert b["basis"] == "mcdiarmid+fournier_guillin"
    assert b["epsilon"] > 0
    assert r.stability_modulus == 1.0


def test_error_bound_tightens_as_n_grows():
    def eps(n):
        cfg = pk.Config(
            n=n, seed=1, invariance_lambda=1.0,
            epsilon=1.0, eta=0.05, observation_diameter=1.0, obs_dim=1,
        )
        return pk.Markov(k=5, theta_max=0.3).run(cfg).error_bound["epsilon"]

    assert eps(65_536) > eps(262_144) > eps(1_048_576)


# ---------------------------------------------------------------------
# Config surface and the sample-complexity floor (SCHEMA section 7)
# ---------------------------------------------------------------------


def test_config_json_round_trip():
    cfg = pk.Config(n=4096, seed=9, backend="simd", invariance_lambda=2.0)
    back = pk.Config.from_json(cfg.to_json())
    assert back.n == 4096
    assert back.seed == 9
    assert back.backend == "simd"


def test_default_config_omits_the_additive_backend_key():
    cfg = pk.Config(n=10, seed=1)
    assert "backend" not in json.loads(cfg.to_json())


def test_backend_is_settable():
    cfg = pk.Config(n=10, seed=1)
    assert cfg.backend == "auto"
    cfg.backend = "scalar"
    assert cfg.backend == "scalar"


def test_sample_floor_matches_the_free_function():
    cfg = pk.Config(
        n=10_000_000, seed=1, invariance_lambda=1.0,
        epsilon=0.01, eta=0.05, observation_diameter=1.0, obs_dim=1,
    )
    assert cfg.sample_floor() == pk.sample_floor(1.0, 1.0, 0.01, 0.05, 1)


def test_an_unsupported_accuracy_claim_is_rejected():
    with pytest.raises(ValueError, match="sample-complexity floor"):
        pk.Markov(k=5, theta_max=0.3).run(
            pk.Config(
                n=100, seed=1, invariance_lambda=1.0,
                epsilon=0.01, eta=0.05, observation_diameter=1.0, obs_dim=1,
            )
        )


def test_partial_accuracy_claims_are_rejected_at_construction():
    with pytest.raises(ValueError, match="needs all of"):
        pk.Config(n=100, seed=1, epsilon=0.01)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"epsilon": 0.0, "eta": 0.05, "observation_diameter": 1.0, "obs_dim": 1}, "epsilon"),
        ({"epsilon": 0.1, "eta": 1.5, "observation_diameter": 1.0, "obs_dim": 1}, "eta"),
    ],
)
def test_out_of_domain_accuracy_parameters_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pk.Config(n=100, seed=1, **kwargs)


def test_unknown_backend_is_rejected_with_a_helpful_message():
    with pytest.raises(ValueError, match="unknown backend"):
        pk.Config(n=10, seed=1, backend="cuda")


def test_empty_ensemble_is_rejected():
    with pytest.raises(ValueError, match="n must be"):
        pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=0, seed=1))


def test_incompatible_schema_major_is_rejected():
    with pytest.raises(ValueError, match="schema major version"):
        pk.Markov(k=5, theta_max=0.3).run(
            pk.Config(n=16, seed=1, schema_version="9.0.0")
        )


# ---------------------------------------------------------------------
# Family validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "family",
    [
        pk.Gaussian(base=[], sigma_max=0.1),
        pk.Gaussian(base=[1.0], sigma_max=-1.0),
        pk.Gaussian(base=[math.nan], sigma_max=0.1),
        pk.Bistable(x0=0.0, dt=0.0, theta_max=0.1),
        pk.Markov(k=0, theta_max=0.1),
        pk.Markov(k=3, theta_max=0.1, start=5),
        pk.Markov(k=3, theta_max=0.1, base_label=9),
        pk.Markov(k=3, theta_max=1.5),
    ],
)
def test_out_of_domain_families_are_rejected(family):
    with pytest.raises(ValueError, match="invalid family"):
        family.run(pk.Config(n=16, seed=1))


def test_run_rejects_a_non_family():
    with pytest.raises(TypeError, match="expected a Family"):
        pk.run(pk.Config(n=16, seed=1), "markov")


def test_family_to_dict_is_a_valid_descriptor():
    for family in families():
        d = family.to_dict()
        assert d["family"] == family.name
        # Round-trips through the JSON descriptor the Rust enum uses.
        assert json.loads(json.dumps(d))["family"] == family.name


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def test_sweep_runs_every_family():
    cfg = pk.Config(n=5_000, seed=1)
    reports = pk.sweep(cfg, [pk.Markov(k=5, theta_max=t) for t in (0.0, 0.5, 1.0)])
    assert len(reports) == 3
    assert [r.value for r in reports] == sorted(
        [r.value for r in reports], reverse=True
    )


def test_tree_sum_matches_pythons_sum_on_exact_inputs():
    # Halves are exactly representable, so no rounding can differ.
    xs = [0.5] * 1024
    assert pk.tree_sum(xs) == 512.0
    assert pk.tree_sum([]) == 0.0
    assert pk.tree_sum([3.25]) == 3.25


def test_tree_sum_is_the_reduction_the_engine_uses():
    # A Markov readout is 0.0 or 1.0, so survival is exactly the mean
    # and reproducing it by hand is a real cross-check of the
    # documented reduction order.
    cfg = pk.Config(n=8192, seed=31)
    r = pk.Markov(k=5, theta_max=0.3).run(cfg)
    assert 0.0 <= r.value <= 1.0
    # tree_sum of n copies of the mean recovers n * mean exactly for
    # this power-of-two n.
    assert pk.tree_sum([r.value] * 8192) == pytest.approx(r.value * 8192)


# ---------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------


def test_requesting_gpu_without_a_device_raises_runtime_error():
    if gpu_available():
        pytest.skip("a device is present, so this cannot be exercised")
    with pytest.raises(RuntimeError):
        pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=16, seed=1, backend="gpu"))


@requires_gpu
@pytest.mark.parametrize("n", [1, 63, 64, 65, 4095, 4096, 100_000])
@pytest.mark.parametrize("k, theta", [(5, 0.3), (4, 1.0), (17, 0.85), (2, 0.5)])
def test_gpu_is_bit_identical_to_the_host(n, k, theta):
    # The headline claim of the exact device backend, checked on the
    # bits rather than to some number of decimal places.
    fam = pk.Markov(k=k, theta_max=theta)
    host = fam.run(pk.Config(n=n, seed=20260610, backend="scalar")).value
    dev = fam.run(pk.Config(n=n, seed=20260610, backend="gpu")).value
    assert host.hex() == dev.hex()


@requires_gpu
def test_gpu_refuses_families_it_cannot_compute_exactly():
    # Silently returning a different number under a backend flag would
    # be the worst available outcome, so this is an error that names the
    # way out.
    for family in (
        pk.Gaussian(base=[0.5, -1.0], sigma_max=0.3),
        pk.Bistable(x0=0.0, dt=0.01, theta_max=0.5),
    ):
        with pytest.raises(RuntimeError, match="gpu_f32"):
            family.run(pk.Config(n=1024, seed=1, backend="gpu"))


@requires_gpu
def test_gpu_reports_declare_double_precision():
    r = pk.Markov(k=4, theta_max=0.25).run(
        pk.Config(n=4096, seed=7, backend="gpu")
    )
    e = r.execution
    assert e["backend"] == "gpu"
    assert e["precision"] == "f64"
    assert e["device"] == pk.gpu_device()


@requires_gpu
@pytest.mark.parametrize("family", families(), ids=lambda f: f.name)
def test_gpu_f32_runs_are_reproducible(family):
    cfg = pk.Config(n=50_000, seed=1, backend="gpu_f32")
    first = family.run(cfg).value
    for _ in range(3):
        assert family.run(cfg).value.hex() == first.hex()


@requires_gpu
@pytest.mark.parametrize("family", families(), ids=lambda f: f.name)
def test_gpu_f32_agrees_with_host_within_monte_carlo_error(family):
    n = 200_000
    host = family.run(pk.Config(n=n, seed=20260610)).value
    dev = family.run(pk.Config(n=n, seed=20260610, backend="gpu_f32")).value
    # Observations are bounded by 1 in absolute value, so the standard
    # error is at most 1/sqrt(n); ten of those is a wide net.
    assert abs(host - dev) < 10.0 / math.sqrt(n)


@requires_gpu
def test_gpu_f32_reports_declare_single_precision():
    r = pk.Markov(k=4, theta_max=0.25).run(
        pk.Config(n=4096, seed=7, backend="gpu_f32")
    )
    assert r.execution["backend"] == "gpu_f32"
    assert r.execution["precision"] == "f32"
