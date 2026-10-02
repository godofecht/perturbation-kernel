#' perturbation-kernel R binding
#'
#' Thin R/RStudio surface over the perturbation-kernel C ABI.
#' @name perturbationkernel
NULL

.pk_escape <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", x)
  gsub('"', '\\"', x, fixed = TRUE)
}

.pk_json_string <- function(x) paste0('"', .pk_escape(x), '"')
.pk_json_num <- function(x) {
  if (length(x) != 1L || !is.finite(x)) stop("expected one finite numeric value")
  format(x, scientific = FALSE, trim = TRUE, digits = 17)
}
.pk_json_int <- function(x) {
  if (length(x) != 1L || is.na(x) || x != as.integer(x)) stop("expected one integer value")
  as.character(as.integer(x))
}
.pk_field <- function(name, value) paste0(.pk_json_string(name), ":", value)

#' Construct a perturbation-kernel configuration
pk_config <- function(n = 1024L, seed = 0L, backend = "auto",
                      forward_l = NULL, invariance_lambda = NULL,
                      epsilon = NULL, eta = NULL,
                      observation_diameter = NULL, obs_dim = NULL) {
  structure(list(
    n = as.integer(n), seed = as.integer(seed), backend = backend,
    forward_l = forward_l, invariance_lambda = invariance_lambda,
    epsilon = epsilon, eta = eta,
    observation_diameter = observation_diameter, obs_dim = obs_dim
  ), class = "pk_config")
}

.pk_config_json <- function(x) {
  parts <- c(
    .pk_field("schema_version", .pk_json_string("1.0.0")),
    .pk_field("n", .pk_json_int(x$n)),
    .pk_field("seed", .pk_json_int(x$seed)),
    .pk_field("intensity", paste0("{",
      .pk_field("kind", .pk_json_string("uniform_interval")), ",",
      .pk_field("params", "{}"), ",",
      .pk_field("null_parameter", "0.0"), "}")),
    .pk_field("reduction", paste0("{",
      .pk_field("order", .pk_json_string("tree")), ",",
      .pk_field("leaf_order", .pk_json_string("index")), "}"))
  )

  lips <- character()
  if (!is.null(x$forward_l)) lips <- c(lips, .pk_field("forward_l", .pk_json_num(x$forward_l)))
  if (!is.null(x$invariance_lambda)) {
    lips <- c(lips, .pk_field("invariance_lambda", .pk_json_num(x$invariance_lambda)))
  }
  parts <- c(parts, .pk_field("lipschitz", paste0("{", paste(lips, collapse = ","), "}")))

  accuracy <- c(x$epsilon, x$eta, x$observation_diameter, x$obs_dim)
  supplied <- c(!is.null(x$epsilon), !is.null(x$eta), !is.null(x$observation_diameter), !is.null(x$obs_dim))
  if (any(supplied) && !all(supplied)) stop("accuracy fields are all-or-nothing")
  if (all(supplied)) {
    parts <- c(parts, .pk_field("accuracy", paste0("{",
      .pk_field("epsilon", .pk_json_num(x$epsilon)), ",",
      .pk_field("eta", .pk_json_num(x$eta)), ",",
      .pk_field("observation_diameter", .pk_json_num(x$observation_diameter)), ",",
      .pk_field("obs_dim", .pk_json_int(x$obs_dim)), "}")))
  }

  if (!identical(x$backend, "auto")) {
    parts <- c(parts, .pk_field("backend", .pk_json_string(x$backend)))
  }
  paste0("{", paste(parts, collapse = ","), "}")
}

#' Gaussian perturbation family
pk_gaussian <- function(base, sigma_max = 0) {
  structure(list(base = as.numeric(base), sigma_max = sigma_max), class = c("pk_gaussian", "pk_family"))
}

#' Bistable perturbation family
pk_bistable <- function(x0 = 0, dt = 0.01, theta_max = 0) {
  structure(list(x0 = x0, dt = dt, theta_max = theta_max), class = c("pk_bistable", "pk_family"))
}

#' Markov perturbation family
pk_markov <- function(k = 2L, theta_max = 0, start = 0L, base_label = 0L) {
  structure(list(k = as.integer(k), theta_max = theta_max,
                 start = as.integer(start), base_label = as.integer(base_label)),
            class = c("pk_markov", "pk_family"))
}

.pk_family_json <- function(x) {
  if (inherits(x, "pk_gaussian")) {
    base <- paste(vapply(x$base, .pk_json_num, character(1)), collapse = ",")
    return(paste0("{",
      .pk_field("family", .pk_json_string("gaussian")), ",",
      .pk_field("base", paste0("[", base, "]")), ",",
      .pk_field("sigma_max", .pk_json_num(x$sigma_max)), "}"))
  }
  if (inherits(x, "pk_bistable")) {
    return(paste0("{",
      .pk_field("family", .pk_json_string("bistable")), ",",
      .pk_field("x0", .pk_json_num(x$x0)), ",",
      .pk_field("dt", .pk_json_num(x$dt)), ",",
      .pk_field("theta_max", .pk_json_num(x$theta_max)), "}"))
  }
  if (inherits(x, "pk_markov")) {
    return(paste0("{",
      .pk_field("family", .pk_json_string("markov")), ",",
      .pk_field("k", .pk_json_int(x$k)), ",",
      .pk_field("start", .pk_json_int(x$start)), ",",
      .pk_field("base_label", .pk_json_int(x$base_label)), ",",
      .pk_field("theta_max", .pk_json_num(x$theta_max)), "}"))
  }
  stop("unknown perturbation-kernel family")
}

#' Run a perturbation family
#'
#' @return A list with numeric \code{value} and raw report JSON.
pk_run <- function(family, config = pk_config()) {
  if (!inherits(family, "pk_family")) stop("family must be created by pk_gaussian(), pk_bistable(), or pk_markov()")
  if (!inherits(config, "pk_config")) stop("config must be created by pk_config()")
  .Call(C_pk_run_family, .pk_family_json(family), .pk_config_json(config))
}

pk_version <- function() .Call(C_pk_version)
pk_schema_version <- function() .Call(C_pk_schema_version)
pk_simd_path <- function() .Call(C_pk_simd_path)
pk_gpu_available <- function() .Call(C_pk_gpu_available)
