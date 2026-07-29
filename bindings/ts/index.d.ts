// Type definitions for perturbation-kernel.

export declare class PerturbationKernelError extends Error {}

export interface ConfigOptions {
  /** Ensemble size. */
  n?: number;
  /** 64-bit RNG seed; together with `n` and the family it determines the result. */
  seed?: number;
  /** Declared Lipschitz constant of the forward model. */
  forwardL?: number;
  /** Declared Wasserstein-1 Lipschitz constant of the functional. */
  invarianceLambda?: number;
  /** Target additive error. Requires the other three accuracy fields. */
  epsilon?: number;
  /** Failure probability in (0, 1). */
  eta?: number;
  /** Diameter of the observation space. */
  observationDiameter?: number;
  /** Dimension of the observation space. */
  obsDim?: number;
}

export interface Config {
  schema_version: string;
  n: number;
  seed: number;
  [key: string]: unknown;
}

export interface ErrorBound {
  available: boolean;
  epsilon: number;
  eta: number;
  basis: string;
  constants: { lambda?: number; d?: number; obs_dim?: number };
}

export interface Report {
  schema_version: string;
  /** The estimate. */
  value: number;
  /** Which functional produced `value`. */
  functional: string;
  n_effective: number;
  seed: number;
  reduction: { order: string; leaf_order: string };
  error_bound: ErrorBound;
  stability_modulus?: number;
  execution?: {
    backend: string;
    simd_path: string;
    threaded: boolean;
    device?: string | null;
    precision: string;
  };
}

export type Family =
  | { family: "gaussian"; base: number[]; sigma_max: number }
  | { family: "bistable"; x0: number; dt: number; theta_max: number }
  | {
      family: "markov";
      k: number;
      start: number;
      base_label: number;
      theta_max: number;
    };

export declare function config(options?: ConfigOptions): Config;
export declare function run(family: Family, cfg: Config): Report;

export declare function gaussian(base: number[], sigmaMax: number): Family;
export declare function bistable(x0: number, dt: number, thetaMax: number): Family;
export declare function markov(
  k: number,
  thetaMax: number,
  start?: number,
  baseLabel?: number,
): Family;

/** Deterministic pairwise sum; the reduction order the engine uses. */
export declare function treeSum(xs: number[] | Float64Array): number;
/** Required ensemble size for an accuracy target (Theorem 7.3(c)). */
export declare function sampleFloor(
  invarianceLambda: number,
  observationDiameter: number,
  epsilon: number,
  eta: number,
  obsDim: number,
): bigint;
export declare function version(): string;
export declare function schemaVersion(): string;
