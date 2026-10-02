library(perturbationkernel)

cfg <- pk_config(n = 262144L, seed = 20260610L, invariance_lambda = 1.0)
r <- pk_run(pk_markov(k = 5L, theta_max = 0.3), cfg)

stopifnot(identical(r$value, 0.8802871704101562))
stopifnot(is.character(r$json), length(r$json) == 1L)
stopifnot(pk_schema_version() == "1.0.0")
