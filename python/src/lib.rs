//! PyO3 bindings for the perturbation-kernel reference implementation.
//!
//! The Rust engine takes three trait impls, which a Python caller
//! cannot supply. This module therefore exposes
//! [`perturbation_kernel::family::Family`] -- the built-in worked
//! examples as data -- plus the full `Config` / `Report` wire types and
//! the backend selector.
//!
//! Everything here is a thin translation layer. No estimator
//! arithmetic happens in this crate: values come back from the core
//! crate unchanged, so the Python API inherits the same determinism
//! guarantees, the same bit-identity across host backends, and the same
//! documented GPU caveat.
//!
//! The extension is built against the stable ABI (`abi3-py38`), so one
//! wheel per platform serves every CPython from 3.8 onward.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use perturbation_kernel::config::{
    Accuracy, Backend, Config as CoreConfig, Intensity, Lipschitz, Reduction,
};
use perturbation_kernel::family::Family;
use perturbation_kernel::report::Report as CoreReport;
use perturbation_kernel::{reduce, Error, SCHEMA_VERSION};

/// Map a core error onto the closest Python exception.
///
/// Domain and contract violations are `ValueError`, because they are
/// caller mistakes recoverable by passing different arguments. Missing
/// hardware is `RuntimeError`, because it is not.
fn to_py(e: Error) -> PyErr {
    match e {
        Error::Gpu(_) | Error::UnsupportedBackend { .. } => PyRuntimeError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

fn parse_backend(name: &str) -> PyResult<Backend> {
    match name.to_ascii_lowercase().as_str() {
        "auto" => Ok(Backend::Auto),
        "scalar" => Ok(Backend::Scalar),
        "simd" => Ok(Backend::Simd),
        "gpu" => Ok(Backend::Gpu),
        "gpu_f32" => Ok(Backend::GpuF32),
        other => Err(PyValueError::new_err(format!(
            "unknown backend {other:?}; expected one of \
             'auto', 'scalar', 'simd', 'gpu', 'gpu_f32'"
        ))),
    }
}

// =====================================================================
// Config
// =====================================================================

/// Run configuration (SCHEMA §5).
#[pyclass(name = "Config", module = "perturbation_kernel")]
#[derive(Clone)]
pub struct PyConfig {
    inner: CoreConfig,
}

#[pymethods]
impl PyConfig {
    /// Build a config.
    ///
    /// `n` is the ensemble size and `seed` keys the RNG; together they
    /// determine the result completely. The Lipschitz constants and the
    /// accuracy target are optional, but asserting an accuracy without
    /// meeting the Theorem 7.3(c) sample floor is rejected at run time
    /// rather than silently reported.
    #[new]
    #[pyo3(signature = (
        n,
        seed = 0,
        *,
        backend = "auto",
        forward_l = None,
        invariance_lambda = None,
        epsilon = None,
        eta = None,
        observation_diameter = None,
        obs_dim = None,
        intensity_kind = "uniform_interval",
        schema_version = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: u64,
        seed: u64,
        backend: &str,
        forward_l: Option<f64>,
        invariance_lambda: Option<f64>,
        epsilon: Option<f64>,
        eta: Option<f64>,
        observation_diameter: Option<f64>,
        obs_dim: Option<u32>,
        intensity_kind: &str,
        schema_version: Option<String>,
    ) -> PyResult<Self> {
        // The accuracy block is all-or-nothing: a partial claim would
        // silently disable the sample-complexity floor.
        let accuracy = match (epsilon, eta, observation_diameter, obs_dim) {
            (None, None, None, None) => None,
            (Some(epsilon), Some(eta), Some(d), Some(obs_dim)) => {
                if !(epsilon > 0.0) {
                    return Err(PyValueError::new_err("epsilon must be > 0"));
                }
                if !(eta > 0.0 && eta < 1.0) {
                    return Err(PyValueError::new_err("eta must lie in (0, 1)"));
                }
                Some(Accuracy {
                    epsilon,
                    eta,
                    observation_diameter: d,
                    obs_dim,
                })
            }
            _ => {
                return Err(PyValueError::new_err(
                    "an accuracy claim needs all of epsilon, eta, \
                     observation_diameter and obs_dim",
                ))
            }
        };

        Ok(Self {
            inner: CoreConfig {
                schema_version: schema_version.unwrap_or_else(|| SCHEMA_VERSION.to_string()),
                n,
                seed,
                intensity: Intensity {
                    kind: intensity_kind.to_string(),
                    params: serde_json::json!({}),
                    null_parameter: serde_json::json!(0.0),
                },
                reduction: Reduction::default(),
                lipschitz: Lipschitz {
                    forward_l,
                    invariance_lambda,
                },
                accuracy,
                backend: parse_backend(backend)?,
            },
        })
    }

    /// Decode a config from its JSON wire form (SCHEMA §5).
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreConfig::from_json(s).map_err(to_py)?,
        })
    }

    /// Encode to the JSON wire form (SCHEMA §5).
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(to_py)
    }

    /// The Theorem 7.3(c) floor on `n` for the asserted accuracy, or
    /// `None` when no accuracy was claimed.
    fn sample_floor(&self) -> Option<u64> {
        self.inner.sample_floor()
    }

    #[getter]
    fn n(&self) -> u64 {
        self.inner.n
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn backend(&self) -> &'static str {
        self.inner.backend.as_str()
    }

    #[setter]
    fn set_backend(&mut self, backend: &str) -> PyResult<()> {
        self.inner.backend = parse_backend(backend)?;
        Ok(())
    }

    #[getter]
    fn schema_version(&self) -> &str {
        &self.inner.schema_version
    }

    fn __repr__(&self) -> String {
        format!(
            "Config(n={}, seed={}, backend='{}')",
            self.inner.n,
            self.inner.seed,
            self.inner.backend.as_str()
        )
    }
}

// =====================================================================
// Report
// =====================================================================

/// Result of a run (SCHEMA §6).
#[pyclass(name = "Report", module = "perturbation_kernel")]
#[derive(Clone)]
pub struct PyReport {
    inner: CoreReport,
}

#[pymethods]
impl PyReport {
    /// The estimate itself, `Phi-hat_N(s)`.
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    /// Tag naming the invariance functional that produced `value`.
    #[getter]
    fn functional(&self) -> &str {
        &self.inner.functional
    }

    /// Ensemble size actually used.
    #[getter]
    fn n_effective(&self) -> u64 {
        self.inner.n_effective
    }

    /// The seed this run was keyed by.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn schema_version(&self) -> &str {
        &self.inner.schema_version
    }

    /// Theorem 5.4 stability modulus, when both Lipschitz constants
    /// were declared.
    #[getter]
    fn stability_modulus(&self) -> Option<f64> {
        self.inner.stability_modulus
    }

    /// The Theorem 7.3 error bound as a dict, or `None` when the
    /// constants needed to derive it were not supplied.
    #[getter]
    fn error_bound<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if !self.inner.error_bound.available {
            return Ok(None);
        }
        let d = PyDict::new(py);
        d.set_item("epsilon", self.inner.error_bound.epsilon)?;
        d.set_item("eta", self.inner.error_bound.eta)?;
        d.set_item("basis", &self.inner.error_bound.basis)?;
        d.set_item("lambda", self.inner.error_bound.constants.lambda)?;
        d.set_item("observation_diameter", self.inner.error_bound.constants.d)?;
        d.set_item("obs_dim", self.inner.error_bound.constants.obs_dim)?;
        Ok(Some(d))
    }

    /// How the value was computed: backend, vector path, threading,
    /// device, and working precision.
    ///
    /// Worth checking before comparing two reports: a `'gpu'` backend
    /// carries `precision='f32'` and is not bit-comparable with a host
    /// result.
    #[getter]
    fn execution<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(e) = self.inner.execution.as_ref() else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("backend", &e.backend)?;
        d.set_item("simd_path", &e.simd_path)?;
        d.set_item("threaded", e.threaded)?;
        d.set_item("device", e.device.clone())?;
        d.set_item("precision", &e.precision)?;
        Ok(Some(d))
    }

    /// Serialise to JSON (SCHEMA §6).
    #[pyo3(signature = (*, pretty = false, v1 = false))]
    fn to_json(&self, pretty: bool, v1: bool) -> PyResult<String> {
        let r = if pretty {
            self.inner.to_json_pretty()
        } else if v1 {
            // Strict v1.0.0 field set: drops the additive `execution`
            // provenance block.
            self.inner.to_json_v1()
        } else {
            self.inner.to_json()
        };
        r.map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Report(value={:.12}, functional='{}', n_effective={}, seed={})",
            self.inner.value, self.inner.functional, self.inner.n_effective, self.inner.seed
        )
    }

    fn __float__(&self) -> f64 {
        self.inner.value
    }
}

// =====================================================================
// Families
// =====================================================================

/// Gaussian shift in `R^d` with negative empirical dispersion as the
/// invariance.
///
/// `base` is the unperturbed state; its length fixes `d`.
#[pyfunction]
#[pyo3(signature = (config, base, sigma_max))]
fn run_gaussian(config: &PyConfig, base: Vec<f64>, sigma_max: f64) -> PyResult<PyReport> {
    run(config, Family::Gaussian { base, sigma_max })
}

/// Bistable double-well marble with polarisation as the invariance.
#[pyfunction]
#[pyo3(signature = (config, x0, dt, theta_max))]
fn run_bistable(config: &PyConfig, x0: f64, dt: f64, theta_max: f64) -> PyResult<PyReport> {
    run(config, Family::Bistable { x0, dt, theta_max })
}

/// Finite-state Markov chain with tail survival as the invariance.
#[pyfunction]
#[pyo3(signature = (config, k, start = 0, base_label = 0, theta_max = 0.0))]
fn run_markov(
    config: &PyConfig,
    k: u32,
    start: u32,
    base_label: u32,
    theta_max: f64,
) -> PyResult<PyReport> {
    run(
        config,
        Family::Markov {
            k,
            start,
            base_label,
            theta_max,
        },
    )
}

fn run(config: &PyConfig, family: Family) -> PyResult<PyReport> {
    family
        .run(&config.inner)
        .map(|inner| PyReport { inner })
        .map_err(to_py)
}

/// Run a family given as a JSON descriptor.
///
/// The descriptor is the serde form of the Rust enum, e.g.
/// `{"family": "markov", "k": 5, "start": 0, "base_label": 0,
/// "theta_max": 0.3}`. Useful for driving a sweep from a config file.
#[pyfunction]
fn run_json(config: &PyConfig, family_json: &str) -> PyResult<PyReport> {
    let family: Family = serde_json::from_str(family_json)
        .map_err(|e| PyValueError::new_err(format!("bad family descriptor: {e}")))?;
    run(config, family)
}

// =====================================================================
// Introspection
// =====================================================================

/// Backends this build can actually run, in preference order.
///
/// The device entries appear only when the wheel was built with the
/// `gpu` feature *and* a compute device is present, so this is a real
/// capability probe rather than a compile-time list.
///
/// `'gpu'` is bit-identical to the host and supports the families the
/// device can reproduce exactly. `'gpu_f32'` is single precision,
/// supports every family, and is faster.
#[pyfunction]
fn available_backends() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut v = vec!["auto", "scalar", "simd"];
    #[cfg(feature = "gpu")]
    if perturbation_kernel::gpu::available() {
        v.push("gpu");
        v.push("gpu_f32");
    }
    v
}

/// The vector path the host backends are using: `'scalar'`, `'neon'`,
/// or `'avx2'`.
///
/// Informational only. It never changes a computed value.
#[pyfunction]
fn simd_path() -> &'static str {
    reduce::active_backend().as_str()
}

/// A description of the compute device, or `None` if none is usable.
#[pyfunction]
fn gpu_device() -> Option<String> {
    #[cfg(feature = "gpu")]
    {
        perturbation_kernel::gpu::context()
            .ok()
            .map(|c| c.name.clone())
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

/// Deterministic pairwise sum (SCHEMA §8 D3).
///
/// Exposed because reproducing the engine's reduction order is the only
/// way to check an externally computed ensemble against a `Report`.
#[pyfunction]
fn tree_sum(xs: Vec<f64>) -> f64 {
    reduce::tree_sum(&xs)
}

/// Theorem 7.3(c) sample floor for an accuracy target.
#[pyfunction]
#[pyo3(signature = (invariance_lambda, observation_diameter, epsilon, eta, obs_dim))]
fn sample_floor(
    invariance_lambda: f64,
    observation_diameter: f64,
    epsilon: f64,
    eta: f64,
    obs_dim: u32,
) -> u64 {
    perturbation_kernel::config::sample_floor(
        invariance_lambda,
        observation_diameter,
        epsilon,
        eta,
        obs_dim,
    )
}

#[pymodule]
fn _perturbation_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Rust extension backing the perturbation_kernel package.",
    )?;
    m.add("SCHEMA_VERSION", SCHEMA_VERSION)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyReport>()?;
    m.add_function(wrap_pyfunction!(run_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(run_bistable, m)?)?;
    m.add_function(wrap_pyfunction!(run_markov, m)?)?;
    m.add_function(wrap_pyfunction!(run_json, m)?)?;
    m.add_function(wrap_pyfunction!(available_backends, m)?)?;
    m.add_function(wrap_pyfunction!(simd_path, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_device, m)?)?;
    m.add_function(wrap_pyfunction!(tree_sum, m)?)?;
    m.add_function(wrap_pyfunction!(sample_floor, m)?)?;
    Ok(())
}
