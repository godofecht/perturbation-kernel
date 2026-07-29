# C API

The schema's position is that C and C++ consumers reach the kernel
through a stable C ABI exposed by the reference core, rather than
becoming a second implementation of it. This is that surface.

```bash
cargo build --release
# target/release/libperturbation_kernel.{so,dylib}   shared
# target/release/libperturbation_kernel.a            static
```

## The surface

```c
typedef struct pk_report pk_report;

pk_report*  pk_run(const double* base_state,
                   const pk_perturbation_vtable* perturbation,
                   const pk_forward_vtable* forward_model,
                   const pk_invariance_vtable* invariance,
                   const char* config_json,
                   int* out_err);

double      pk_report_value(const pk_report*);
const char* pk_report_json(const pk_report*);
void        pk_free_report(pk_report*);
```

Error codes returned through `out_err`:

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | invalid config |
| 2 | null-parameter mismatch |
| 3 | sample-complexity floor violated |
| 4 | empty ensemble |
| 5 | a panic was caught at the boundary |

## Supplying the model

Rust trait objects are not stable across an ABI boundary, so the
interop pattern is a vtable of `extern "C"` function pointers plus an
opaque `state` pointer you own.

```c
typedef struct {
    void*  state;
    double (*null)(void* state);
    double (*sample_theta)(void* state, uint64_t seed_lo, uint64_t seed_hi);
    double (*apply)(void* state, double s, double theta,
                    uint64_t seed_lo, uint64_t seed_hi);
} pk_perturbation_vtable;

typedef struct {
    void*  state;
    double (*eval)(void* state, double s);
    double lipschitz;        /* negative means "not declared" */
} pk_forward_vtable;

typedef struct {
    void*  state;
    double (*g)(void* state, double y);
    double lipschitz_w1;     /* negative means "not declared" */
    const char* name;
} pk_invariance_vtable;
```

The projection is scalar: state, intensity and observation are all
`double`. That is enough for the worked examples and for
cross-language conformance testing, and it keeps the ABI small enough
to be stable.

The invariance vtable takes a per-observation function `g` rather than a
reduction, and the engine averages `g(y)` over the ensemble on the Rust
side. This is deliberate: the reduction order is the part that has to be
right for reproducibility, so it stays where it can be guaranteed.

## Determinism across the boundary

`sample_theta` and `apply` receive a 128-bit value lifted from the
engine's ChaCha20 stream rather than being handed a generator. Derive
your randomness from it and your implementation inherits the crate's
determinism contract. Read a global RNG instead and you give it up, and
nothing on the Rust side can stop you.

## A complete call

```c
#include <stdio.h>
#include <stdint.h>

extern double survives(void* st, double y) { return y > 0.5 ? 1.0 : 0.0; }
extern double identity(void* st, double s) { return s; }
extern double null_theta(void* st)        { return 0.0; }

static uint64_t mix(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static double sample_theta(void* st, uint64_t lo, uint64_t hi) {
    return (double)(mix(lo) >> 11) * (1.0 / 9007199254740992.0) * 0.3;
}

static double apply(void* st, double s, double theta,
                    uint64_t lo, uint64_t hi) {
    double u = (double)(mix(hi) >> 11) * (1.0 / 9007199254740992.0);
    return u < theta ? 1.0 - s : s;
}

int main(void) {
    pk_perturbation_vtable p = { NULL, null_theta, sample_theta, apply };
    pk_forward_vtable      f = { NULL, identity, 1.0 };
    pk_invariance_vtable   i = { NULL, survives, 1.0, "survival" };

    const char* cfg =
        "{\"schema_version\":\"1.0.0\",\"n\":100000,\"seed\":20260610,"
        "\"intensity\":{\"kind\":\"uniform_interval\",\"params\":{},"
        "\"null_parameter\":0.0},"
        "\"reduction\":{\"order\":\"tree\",\"leaf_order\":\"index\"},"
        "\"lipschitz\":{\"forward_l\":1.0,\"invariance_lambda\":1.0}}";

    double base = 1.0;
    int err = 0;
    pk_report* r = pk_run(&base, &p, &f, &i, cfg, &err);
    if (err != 0) { fprintf(stderr, "pk_run failed: %d\n", err); return 1; }

    printf("value = %.12f\n", pk_report_value(r));
    printf("json  = %s\n",    pk_report_json(r));
    pk_free_report(r);
    return 0;
}
```

Link against the static library:

```bash
cc example.c target/release/libperturbation_kernel.a -lm -lpthread -ldl -o example
```

## Memory and threads

`pk_run` returns ownership. Call `pk_free_report` exactly once; the JSON
string from `pk_report_json` is owned by the report and is invalid after
the free.

The C path runs single-threaded. Your vtables carry an opaque `void*`
and nothing on the Rust side can determine whether it is safe to share
across threads, so the ABI drives `Engine::run_sequential`. The value is
bit-identical to the threaded path; only the wall time differs.

Panics are caught at the boundary and reported as code 5 rather than
unwinding into your C frame.
