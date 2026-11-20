# perturbation-kernel

Reproducible perturbation-kernel estimators for Node.js. The Rust engine
is compiled to WebAssembly, so nothing native is built on install.

```bash
npm install perturbation-kernel
```

```js
const pk = require("perturbation-kernel");

const cfg = pk.config({ n: 262144, seed: 20260610 });
const report = pk.run(pk.markov(5, 0.3), cfg);

report.value; // 0.8802871704101562
```

The estimate is a pure function of `(family, n, seed)`. It matches the
Rust, Python, C++, Zig and Julia builds bit for bit, because this
package runs the same scalar reference path they are checked against.

## API

| Function | Returns |
|---|---|
| `config({ n, seed, forwardL, invarianceLambda, epsilon, eta, observationDiameter, obsDim })` | A config. The four accuracy fields are all or nothing. |
| `run(family, config)` | A `Report` with `value`, `functional` and `error_bound`. |
| `gaussian(base, sigmaMax)` | Gaussian shift in R^d. Invariance is the negative empirical dispersion. |
| `bistable(x0, dt, thetaMax)` | Double-well marble. Invariance is the polarisation in [-1, 1]. |
| `markov(k, thetaMax, start = 0, baseLabel = 0)` | Finite-state chain. Invariance is the survival probability. |
| `treeSum(xs)` | The deterministic pairwise sum the engine uses. |
| `sampleFloor(lambda, diameter, epsilon, eta, obsDim)` | The Theorem 7.3(c) sample floor. |

Errors from the engine are thrown as `PerturbationKernelError`. Types
ship in `index.d.ts`.

This build is scalar only. Threads and SIMD are not available in
`wasm32-unknown-unknown`, and the scalar path is the reference, so the
values are the same. For speed, use the Rust crate or the Python
package.

## Links

- Documentation: https://godofecht.github.io/perturbation-kernel/
- Source: https://github.com/godofecht/perturbation-kernel
- Python: https://pypi.org/project/perturbation-kernel/
- Rust: https://crates.io/crates/perturbation-kernel

MIT licence.
