# perturbationkernel for R / RStudio

Thin R binding over the same C ABI used by the C++, Zig and Julia bindings.
All computation remains in the Rust engine, so the determinism and
bit-identity contract is unchanged.

## Build

From the repository root:

```bash
cargo build --release
R CMD INSTALL bindings/r
```

On macOS/Linux, if the library is elsewhere:

```bash
R CMD INSTALL bindings/r --configure-vars='PK_LIB=/absolute/path/to/lib'
```

## RStudio quickstart

```r
library(perturbationkernel)

cfg <- pk_config(
  n = 262144L,
  seed = 20260610L,
  invariance_lambda = 1.0
)

report <- pk_run(
  pk_markov(k = 5L, theta_max = 0.3),
  cfg
)

report$value
# 0.8802871704101562

report$json
```

Built-in families:

```r
pk_markov(k = 5L, theta_max = 0.3)
pk_gaussian(base = c(0.5, -1.25), sigma_max = 0.3)
pk_bistable(x0 = 0, dt = 0.01, theta_max = 0.5)
```

Introspection:

```r
pk_version()
pk_schema_version()
pk_simd_path()
pk_gpu_available()
```
