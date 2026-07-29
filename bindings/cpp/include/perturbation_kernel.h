/* perturbation-kernel: C ABI (SCHEMA section 9).
 *
 * Link against libperturbation_kernel.{a,so,dylib}, produced by
 * `cargo build --release`.
 *
 * Two surfaces are exposed. `pk_run_family` takes JSON and runs one of
 * the built-in families; it is what every non-Rust binding in this
 * repository uses, because JSON requires no agreement on struct layout.
 * The vtable surface below it lets a C caller supply its own
 * perturbation, forward model and functional, which is how you extend
 * the schema rather than merely use it.
 *
 * Determinism: a run is a pure function of (family, n, seed). Nothing
 * here reads OS randomness or ambient state. Results are bit-identical
 * across host backends, operating systems and CPU architectures.
 */

#ifndef PERTURBATION_KERNEL_H
#define PERTURBATION_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque result handle. Free exactly once with pk_free_report. */
typedef struct pk_report pk_report;

/* Error codes written through the out_err parameter. */
typedef enum {
  PK_OK = 0,
  PK_INVALID_CONFIG = 1,
  PK_NULL_PARAMETER_MISMATCH = 2,
  PK_SAMPLE_FLOOR = 3,   /* accuracy claim below the Thm 7.3(c) floor */
  PK_EMPTY_ENSEMBLE = 4,
  PK_PANIC = 5
} pk_err;

/* ---- family surface -------------------------------------------------
 *
 * family_json is the serde form of the built-in family enum, e.g.
 *   {"family":"markov","k":5,"start":0,"base_label":0,"theta_max":0.3}
 *   {"family":"gaussian","base":[0.5,-1.25],"sigma_max":0.3}
 *   {"family":"bistable","x0":0.0,"dt":0.01,"theta_max":0.5}
 *
 * config_json is a SCHEMA section 5 payload. Returns NULL on error with
 * a pk_err written to out_err.
 */
pk_report *pk_run_family(const char *family_json, const char *config_json,
                         int *out_err);

/* ---- results -------------------------------------------------------- */

/* The scalar estimate. */
double pk_report_value(const pk_report *r);

/* The full report as JSON (SCHEMA section 6). Owned by the report;
 * invalid after pk_free_report. */
const char *pk_report_json(const pk_report *r);

void pk_free_report(pk_report *r);

/* ---- introspection --------------------------------------------------
 * All return static strings; do not free. */

const char *pk_version(void);        /* crate version   */
const char *pk_schema_version(void); /* schema version  */
const char *pk_simd_path(void);      /* "scalar" | "neon" | "avx2" */
int pk_gpu_available(void);          /* 1 if a device is usable */

/* ---- vtable surface -------------------------------------------------
 *
 * Supply your own model over a scalar state. sample_theta and apply
 * receive a 128-bit value lifted from the engine's ChaCha20 stream;
 * derive your randomness from it and your implementation inherits the
 * determinism contract. Read a global RNG instead and you give it up.
 */

typedef struct {
  void *state;
  double (*null_theta)(void *state);
  double (*sample_theta)(void *state, uint64_t seed_lo, uint64_t seed_hi);
  double (*apply)(void *state, double s, double theta, uint64_t seed_lo,
                  uint64_t seed_hi);
} pk_perturbation_vtable;

typedef struct {
  void *state;
  double (*eval)(void *state, double s);
  double lipschitz; /* negative means "not declared" */
} pk_forward_vtable;

typedef struct {
  void *state;
  /* Per-observation function; the engine averages g over the ensemble
   * on the Rust side, so the reduction order stays guaranteed. */
  double (*g)(void *state, double y);
  double lipschitz_w1; /* negative means "not declared" */
  const char *name;
} pk_invariance_vtable;

pk_report *pk_run(const double *base_state,
                  const pk_perturbation_vtable *perturbation,
                  const pk_forward_vtable *forward_model,
                  const pk_invariance_vtable *invariance,
                  const char *config_json, int *out_err);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PERTURBATION_KERNEL_H */
