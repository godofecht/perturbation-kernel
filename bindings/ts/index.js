// perturbation-kernel: TypeScript / JavaScript API over the wasm build.
//
// The wasm module takes and returns JSON, mirroring the C ABI, so this
// layer only formats descriptors and parses reports. Nothing here
// computes: every value comes back from the same Rust engine, so it is
// bit-identical to the native library.

const wasm = require("./pkg/perturbation_kernel_wasm.js");

/** Thrown for any error the engine reports. */
class PerturbationKernelError extends Error {
  constructor(message) {
    super(message);
    this.name = "PerturbationKernelError";
  }
}

/**
 * Build a SCHEMA section 5 config.
 *
 * The four accuracy fields are all-or-nothing: supplying some but not
 * all is rejected, because a partial claim would quietly disable the
 * sample-complexity floor rather than enforce a weaker one.
 *
 * Only the scalar backend exists in wasm, so `backend` is not accepted
 * here. The value is the same either way.
 */
function config({
  n = 1024,
  seed = 0,
  forwardL,
  invarianceLambda,
  epsilon,
  eta,
  observationDiameter,
  obsDim,
} = {}) {
  const lipschitz = {};
  if (forwardL !== undefined) lipschitz.forward_l = forwardL;
  if (invarianceLambda !== undefined) lipschitz.invariance_lambda = invarianceLambda;

  const cfg = {
    schema_version: "1.0.0",
    n,
    seed,
    intensity: { kind: "uniform_interval", params: {}, null_parameter: 0.0 },
    reduction: { order: "tree", leaf_order: "index" },
    lipschitz,
  };

  const accuracy = [epsilon, eta, observationDiameter, obsDim];
  if (accuracy.some((v) => v !== undefined)) {
    if (accuracy.some((v) => v === undefined)) {
      throw new PerturbationKernelError(
        "an accuracy claim needs all of epsilon, eta, observationDiameter and obsDim",
      );
    }
    cfg.accuracy = {
      epsilon,
      eta,
      observation_diameter: observationDiameter,
      obs_dim: obsDim,
    };
  }
  return cfg;
}

function run(family, cfg) {
  let json;
  try {
    json = wasm.run_family(JSON.stringify(family), JSON.stringify(cfg));
  } catch (e) {
    throw new PerturbationKernelError(String(e.message ?? e));
  }
  return JSON.parse(json);
}

/** Gaussian shift in R^d; invariance is the negative empirical dispersion. */
const gaussian = (base, sigmaMax) => ({ family: "gaussian", base, sigma_max: sigmaMax });

/** Bistable double-well marble; invariance is the polarisation in [-1, 1]. */
const bistable = (x0, dt, thetaMax) => ({
  family: "bistable",
  x0,
  dt,
  theta_max: thetaMax,
});

/** Finite-state chain; invariance is the survival probability in [0, 1]. */
const markov = (k, thetaMax, start = 0, baseLabel = 0) => ({
  family: "markov",
  k,
  start,
  base_label: baseLabel,
  theta_max: thetaMax,
});

module.exports = {
  PerturbationKernelError,
  config,
  run,
  gaussian,
  bistable,
  markov,
  /** Deterministic pairwise sum, the reduction the engine uses. */
  treeSum: (xs) => wasm.tree_sum(Float64Array.from(xs)),
  /** Theorem 7.3(c) sample floor for an accuracy target. */
  sampleFloor: (lambda, diameter, epsilon, eta, obsDim) =>
    wasm.sample_floor(lambda, diameter, epsilon, eta, obsDim),
  version: () => wasm.version(),
  schemaVersion: () => wasm.schema_version(),
};
