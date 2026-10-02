#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct pk_report pk_report;
pk_report *pk_run_family(const char *, const char *, int *);
double pk_report_value(const pk_report *);
const char *pk_report_json(const pk_report *);
void pk_free_report(pk_report *);
const char *pk_version(void);
const char *pk_schema_version(void);
const char *pk_simd_path(void);
int pk_gpu_available(void);

static SEXP scalar_string(const char *x)
{
    if (x == NULL) return ScalarString(NA_STRING);
    return mkString(x);
}

SEXP C_pk_run_family(SEXP family_json, SEXP config_json)
{
    int err = 0;
    pk_report *r = pk_run_family(CHAR(asChar(family_json)), CHAR(asChar(config_json)), &err);
    if (r == NULL) error("perturbation-kernel error %d", err);

    SEXP out = PROTECT(allocVector(VECSXP, 2));
    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_VECTOR_ELT(out, 0, ScalarReal(pk_report_value(r)));
    SET_VECTOR_ELT(out, 1, scalar_string(pk_report_json(r)));
    SET_STRING_ELT(names, 0, mkChar("value"));
    SET_STRING_ELT(names, 1, mkChar("json"));
    setAttrib(out, R_NamesSymbol, names);

    pk_free_report(r);
    UNPROTECT(2);
    return out;
}

SEXP C_pk_version(void) { return scalar_string(pk_version()); }
SEXP C_pk_schema_version(void) { return scalar_string(pk_schema_version()); }
SEXP C_pk_simd_path(void) { return scalar_string(pk_simd_path()); }
SEXP C_pk_gpu_available(void) { return ScalarLogical(pk_gpu_available() ? 1 : 0); }

static const R_CallMethodDef call_methods[] = {
    {"C_pk_run_family", (DL_FUNC) &C_pk_run_family, 2},
    {"C_pk_version", (DL_FUNC) &C_pk_version, 0},
    {"C_pk_schema_version", (DL_FUNC) &C_pk_schema_version, 0},
    {"C_pk_simd_path", (DL_FUNC) &C_pk_simd_path, 0},
    {"C_pk_gpu_available", (DL_FUNC) &C_pk_gpu_available, 0},
    {NULL, NULL, 0}
};

void R_init_perturbationkernel(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
