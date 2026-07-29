//! Minimal C-ABI projection (SCHEMA §9).
//!
//! The schema's stance is that C/C++ consumers reach the kernel through
//! a stable C ABI exposed by the reference core; they MUST NOT be a
//! second implementation. This module provides exactly that surface:
//! one entry point ([`pk_run`]) that takes a JSON [`crate::config::Config`]
//! and three opaque component handles, plus the standard accessors and
//! a free function.
//!
//! Because Rust trait objects are not stable across the ABI boundary,
//! the canonical interop pattern used here is a *vtable struct of
//! `extern "C"` function pointers*. A C/C++ caller fills in those
//! pointers and registers an opaque `state*`. The Rust side then wraps
//! them as
//! [`Perturbation`] /
//! [`ForwardModel`] /
//! [`Invariance`] impls and drives the
//! engine.
//!
//! `pk_free_report` reclaims the [`Report`]
//! ownership; nothing here leaks. All `unsafe` in the crate is confined
//! to this file (SCHEMA §9, conventions in the prompt).

#![allow(unsafe_code)]
#![allow(missing_docs)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};

use crate::config::Config;
use crate::engine::Engine;
use crate::forward::ForwardModel;
use crate::invariance::Invariance;
use crate::perturbation::Perturbation;
use crate::report::Report;
use crate::Rng;

/// Error codes returned through the `out_err` out-parameter.
#[repr(C)]
pub enum PkErr {
    Ok = 0,
    InvalidConfig = 1,
    NullParameterMismatch = 2,
    SampleFloor = 3,
    EmptyEnsemble = 4,
    Panic = 5,
}

/// Opaque report handle.
pub struct PkReport {
    json: CString,
    value: f64,
}

/// Vtable for a foreign [`Perturbation<f64>`] over a scalar state and
/// scalar parameter -- the minimal universal projection that suffices
/// for the example workloads and for cross-language smoke tests.
#[repr(C)]
pub struct PkPerturbationVTable {
    pub state: *mut c_void,
    pub null: extern "C" fn(state: *mut c_void) -> f64,
    pub sample_theta: extern "C" fn(state: *mut c_void, seed_lo: u64, seed_hi: u64) -> f64,
    pub apply:
        extern "C" fn(state: *mut c_void, s: f64, theta: f64, seed_lo: u64, seed_hi: u64) -> f64,
}

/// Vtable for a foreign [`ForwardModel<f64, f64>`].
#[repr(C)]
pub struct PkForwardVTable {
    pub state: *mut c_void,
    pub eval: extern "C" fn(state: *mut c_void, s: f64) -> f64,
    pub lipschitz: f64, // negative means "not declared"
}

/// Vtable for a foreign [`Invariance<f64>`].
#[repr(C)]
pub struct PkInvarianceVTable {
    pub state: *mut c_void,
    /// Reduce by computing `mean(g(y_i))`; the C side supplies `g`.
    pub g: extern "C" fn(state: *mut c_void, y: f64) -> f64,
    pub lipschitz_w1: f64, // negative means "not declared"
    pub name: *const c_char,
}

struct CPerturb<'a>(&'a PkPerturbationVTable);
impl<'a> Perturbation<f64> for CPerturb<'a> {
    type Theta = f64;
    fn null(&self) -> f64 {
        (self.0.null)(self.0.state)
    }
    fn sample_theta(&self, rng: &mut Rng) -> f64 {
        // Hand the foreign sampler a 128-bit "seed" lifted from the
        // ChaCha stream so it can be deterministic if it wants to be.
        use rand::Rng as _;
        let lo: u64 = rng.gen();
        let hi: u64 = rng.gen();
        (self.0.sample_theta)(self.0.state, lo, hi)
    }
    fn apply(&self, s: &f64, theta: &f64, rng: &mut Rng) -> f64 {
        use rand::Rng as _;
        let lo: u64 = rng.gen();
        let hi: u64 = rng.gen();
        (self.0.apply)(self.0.state, *s, *theta, lo, hi)
    }
}

struct CForward<'a>(&'a PkForwardVTable);
impl<'a> ForwardModel<f64, f64> for CForward<'a> {
    fn eval(&self, s: &f64) -> f64 {
        (self.0.eval)(self.0.state, *s)
    }
    fn lipschitz(&self) -> Option<f64> {
        if self.0.lipschitz < 0.0 {
            None
        } else {
            Some(self.0.lipschitz)
        }
    }
}

struct CInvariance<'a>(&'a PkInvarianceVTable);
impl<'a> Invariance<f64> for CInvariance<'a> {
    fn measure(&self, ensemble: &[f64]) -> Report {
        let mut acc = 0.0_f64;
        for &y in ensemble {
            acc += (self.0.g)(self.0.state, y);
        }
        let val = if ensemble.is_empty() {
            0.0
        } else {
            acc / ensemble.len() as f64
        };
        Report::raw(
            val,
            self.name(),
            ensemble.len() as u64,
            0,
            Default::default(),
        )
    }
    fn lipschitz_w1(&self) -> Option<f64> {
        if self.0.lipschitz_w1 < 0.0 {
            None
        } else {
            Some(self.0.lipschitz_w1)
        }
    }
    fn name(&self) -> &str {
        if self.0.name.is_null() {
            return "c_extern";
        }
        // SAFETY: name was provided by the caller as a NUL-terminated C
        // string; we only read it.
        unsafe { CStr::from_ptr(self.0.name) }
            .to_str()
            .unwrap_or("c_extern")
    }
}

/// Entry point: run the engine and return an opaque [`PkReport`]
/// pointer (SCHEMA §9).
///
/// # Safety
///
/// All pointer arguments must be valid for the duration of the call:
/// `base_state` points to an `f64`; the three vtable pointers are
/// non-null and remain valid; `config_json` is a NUL-terminated C
/// string; `out_err` is a writable pointer to `c_int`.
#[no_mangle]
pub unsafe extern "C" fn pk_run(
    base_state: *const f64,
    perturbation: *const PkPerturbationVTable,
    forward_model: *const PkForwardVTable,
    invariance: *const PkInvarianceVTable,
    config_json: *const c_char,
    out_err: *mut c_int,
) -> *mut PkReport {
    let set_err = |code: PkErr| {
        if !out_err.is_null() {
            *out_err = code as c_int;
        }
    };

    if base_state.is_null()
        || perturbation.is_null()
        || forward_model.is_null()
        || invariance.is_null()
        || config_json.is_null()
    {
        set_err(PkErr::InvalidConfig);
        return std::ptr::null_mut();
    }

    let cfg_json = match CStr::from_ptr(config_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_err(PkErr::InvalidConfig);
            return std::ptr::null_mut();
        }
    };
    let cfg: Config = match Config::from_json(cfg_json) {
        Ok(c) => c,
        Err(_) => {
            set_err(PkErr::InvalidConfig);
            return std::ptr::null_mut();
        }
    };

    let base = *base_state;
    let p = CPerturb(&*perturbation);
    let f = CForward(&*forward_model);
    let i = CInvariance(&*invariance);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Engine::run_sequential(&base, &p, &f, &i, &cfg)
    }));
    match result {
        Ok(Ok(report)) => {
            let json = report.to_json().unwrap_or_default();
            let cjson = match CString::new(json) {
                Ok(s) => s,
                Err(_) => {
                    set_err(PkErr::Panic);
                    return std::ptr::null_mut();
                }
            };
            let boxed = Box::new(PkReport {
                json: cjson,
                value: report.value,
            });
            set_err(PkErr::Ok);
            Box::into_raw(boxed)
        }
        Ok(Err(crate::Error::NullParameterMismatch { .. })) => {
            set_err(PkErr::NullParameterMismatch);
            std::ptr::null_mut()
        }
        Ok(Err(crate::Error::SampleFloor { .. })) => {
            set_err(PkErr::SampleFloor);
            std::ptr::null_mut()
        }
        Ok(Err(crate::Error::EmptyEnsemble)) => {
            set_err(PkErr::EmptyEnsemble);
            std::ptr::null_mut()
        }
        Ok(Err(_)) => {
            set_err(PkErr::InvalidConfig);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_err(PkErr::Panic);
            std::ptr::null_mut()
        }
    }
}

/// Accessor: scalar value `Phi-hat_N(s)` (SCHEMA §9, §6 row `value`).
///
/// # Safety
///
/// `r` must be a pointer returned by [`pk_run`] and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn pk_report_value(r: *const PkReport) -> f64 {
    if r.is_null() {
        return f64::NAN;
    }
    (*r).value
}

/// Accessor: the full [`Report`] as a JSON C string (SCHEMA §9, §6).
///
/// # Safety
///
/// `r` must be a pointer returned by [`pk_run`] and not yet freed. The
/// returned C string is owned by the report; it lives until
/// [`pk_free_report`] is called.
#[no_mangle]
pub unsafe extern "C" fn pk_report_json(r: *const PkReport) -> *const c_char {
    if r.is_null() {
        return std::ptr::null();
    }
    (*r).json.as_ptr()
}

/// Reclaim ownership and drop the report (SCHEMA §9).
///
/// # Safety
///
/// `r` must be a pointer returned by [`pk_run`] and not yet freed.
/// After this call the pointer must not be dereferenced.
#[no_mangle]
pub unsafe extern "C" fn pk_free_report(r: *mut PkReport) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

// =====================================================================
// Family projection (SCHEMA §9, additive)
// =====================================================================
//
// The vtable surface above is the general one: a foreign caller supplies
// its own perturbation, forward model and functional. That is the right
// shape for extending the schema from C, and the wrong shape for simply
// *using* it, which is what a binding in another language almost always
// wants.
//
// `pk_run_family` closes that gap. Both arguments are JSON, so a binding
// needs no struct layout agreement, no function pointers and no
// lifetime discipline: it formats two strings, gets a handle back, and
// frees it. Every non-Rust binding in this repository is built on this
// one function.

/// Run one of the built-in families (SCHEMA §9, additive).
///
/// `family_json` is the serde form of [`crate::family::Family`], for
/// example:
///
/// ```json
/// {"family":"markov","k":5,"start":0,"base_label":0,"theta_max":0.3}
/// ```
///
/// `config_json` is a [`Config`] payload (SCHEMA §5). Returns an opaque
/// report handle to be freed with [`pk_free_report`], or null on error
/// with a code written to `out_err`.
///
/// # Safety
///
/// `family_json` and `config_json` must be valid NUL-terminated C
/// strings for the duration of the call. `out_err` must be null or
/// point to a writable `c_int`.
#[no_mangle]
pub unsafe extern "C" fn pk_run_family(
    family_json: *const c_char,
    config_json: *const c_char,
    out_err: *mut c_int,
) -> *mut PkReport {
    let set_err = |code: PkErr| {
        if !out_err.is_null() {
            *out_err = code as c_int;
        }
    };
    set_err(PkErr::Ok);

    if family_json.is_null() || config_json.is_null() {
        set_err(PkErr::InvalidConfig);
        return std::ptr::null_mut();
    }

    // Catch panics at the boundary: unwinding into a foreign frame is
    // undefined behaviour.
    let result = std::panic::catch_unwind(|| {
        let fam_s = CStr::from_ptr(family_json).to_str().ok()?;
        let cfg_s = CStr::from_ptr(config_json).to_str().ok()?;
        let family: crate::family::Family = serde_json::from_str(fam_s).ok()?;
        let cfg = Config::from_json(cfg_s).ok()?;
        Some((family, cfg))
    });

    let (family, cfg) = match result {
        Ok(Some(v)) => v,
        Ok(None) => {
            set_err(PkErr::InvalidConfig);
            return std::ptr::null_mut();
        }
        Err(_) => {
            set_err(PkErr::Panic);
            return std::ptr::null_mut();
        }
    };

    match std::panic::catch_unwind(move || family.run(&cfg)) {
        Ok(Ok(report)) => {
            let json = report.to_json().unwrap_or_default();
            Box::into_raw(Box::new(PkReport {
                json: CString::new(json).unwrap_or_default(),
                value: report.value,
            }))
        }
        Ok(Err(e)) => {
            set_err(match e {
                crate::Error::NullParameterMismatch { .. } => PkErr::NullParameterMismatch,
                crate::Error::SampleFloor { .. } => PkErr::SampleFloor,
                crate::Error::EmptyEnsemble => PkErr::EmptyEnsemble,
                _ => PkErr::InvalidConfig,
            });
            std::ptr::null_mut()
        }
        Err(_) => {
            set_err(PkErr::Panic);
            std::ptr::null_mut()
        }
    }
}

/// Crate version as a static NUL-terminated string, e.g. `"2.1.0"`.
#[no_mangle]
pub extern "C" fn pk_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Schema version this build implements, e.g. `"1.0.0"` (SCHEMA §10).
#[no_mangle]
pub extern "C" fn pk_schema_version() -> *const c_char {
    concat!("1.0.0", "\0").as_ptr() as *const c_char
}

/// Host vector path in use: `"scalar"`, `"neon"` or `"avx2"`.
///
/// Informational. It never changes a computed value.
#[no_mangle]
pub extern "C" fn pk_simd_path() -> *const c_char {
    match crate::reduce::active_backend() {
        crate::reduce::SimdPath::Scalar => c"scalar".as_ptr(),
        crate::reduce::SimdPath::Neon => c"neon".as_ptr(),
        crate::reduce::SimdPath::Avx2 => c"avx2".as_ptr(),
    }
}

/// `1` when a compute device is usable, `0` otherwise.
#[no_mangle]
pub extern "C" fn pk_gpu_available() -> c_int {
    #[cfg(feature = "gpu")]
    {
        crate::gpu::available() as c_int
    }
    #[cfg(not(feature = "gpu"))]
    {
        0
    }
}
