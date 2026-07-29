// Conformance test for the TypeScript/wasm binding.
//
// The binding does no arithmetic, so what matters is that values cross
// the wasm boundary unchanged. Expected numbers are the ones the Rust,
// Python, C++ and Zig suites assert on.

import assert from "node:assert/strict";
import { createRequire } from "node:module";
const pk = createRequire(import.meta.url)("../index.js");

let failures = 0;
const check = (name, fn) => {
  try {
    fn();
    console.log(`  ${name.padEnd(58)} ok`);
  } catch (e) {
    console.log(`  ${name.padEnd(58)} FAILED\n      ${e.message}`);
    failures++;
  }
};

console.log(
  `perturbation-kernel ${pk.version()} (schema ${pk.schemaVersion()}), wasm\n`,
);

check("markov matches the reference value exactly", () => {
  const r = pk.run(pk.markov(5, 0.3), pk.config({ n: 262144, seed: 20260610 }));
  assert.equal(r.value, 0.8802871704101562);
  assert.equal(r.functional, "tail_survival");
});

check("the seed is total", () => {
  const a = pk.run(pk.markov(5, 0.3), pk.config({ n: 20000, seed: 1 })).value;
  const b = pk.run(pk.markov(5, 0.3), pk.config({ n: 20000, seed: 2 })).value;
  assert.notEqual(a, b);
});

check("repeated runs are bit-identical", () => {
  const cfg = pk.config({ n: 50000, seed: 7 });
  const first = pk.run(pk.markov(5, 0.3), cfg).value;
  for (let i = 0; i < 3; i++) {
    assert.equal(pk.run(pk.markov(5, 0.3), cfg).value, first);
  }
});

check("null intensity recovers the base state", () => {
  const cfg = pk.config({ n: 10000, seed: 5 });
  assert.equal(pk.run(pk.markov(5, 0.0, 2, 2), cfg).value, 1.0);
  // The dispersion is negated, so zero comes back as -0. IEEE equality
  // treats that as zero; node:assert/strict uses Object.is, which does
  // not. `===` is the right comparison here.
  assert.ok(pk.run(pk.gaussian([1.5, -2.0], 0.0), cfg).value === 0.0);
});

check("estimator ranges hold", () => {
  const cfg = pk.config({ n: 20000, seed: 3 });
  const pol = pk.run(pk.bistable(0.0, 0.01, 0.5), cfg).value;
  assert.ok(pol >= -1 && pol <= 1);
  assert.ok(pk.run(pk.gaussian([0, 0], 0.3), cfg).value <= 0);
});

check("survival falls with the mixing probability", () => {
  const cfg = pk.config({ n: 50000, seed: 11 });
  const vals = [0.0, 0.25, 0.5, 1.0].map(
    (t) => pk.run(pk.markov(5, t), cfg).value,
  );
  assert.deepEqual(vals, [...vals].sort((a, b) => b - a));
});

check("an unsupported accuracy claim is rejected", () => {
  assert.throws(
    () =>
      pk.run(
        pk.markov(5, 0.3),
        pk.config({
          n: 1000,
          seed: 1,
          invarianceLambda: 1.0,
          epsilon: 0.05,
          eta: 0.05,
          observationDiameter: 1.0,
          obsDim: 1,
        }),
      ),
    /sample-complexity floor/,
  );
});

check("an empty ensemble is rejected", () => {
  assert.throws(() => pk.run(pk.markov(5, 0.3), pk.config({ n: 0 })), /n must be/);
});

check("an out-of-domain family is rejected", () => {
  assert.throws(() => pk.run(pk.markov(0, 0.3), pk.config({ n: 16 })), /invalid family/);
});

check("a partial accuracy claim is rejected before it reaches wasm", () => {
  assert.throws(() => pk.config({ n: 100, epsilon: 0.05 }), /needs all of/);
});

check("tree_sum is exact on representable inputs", () => {
  assert.equal(pk.treeSum(new Array(1024).fill(0.5)), 512);
  assert.equal(pk.treeSum([]), 0);
  assert.equal(pk.treeSum([3.25]), 3.25);
});

check("sample_floor matches the documented table", () => {
  assert.equal(pk.sampleFloor(1, 1, 0.05, 0.05, 1), 262144n);
  assert.equal(pk.sampleFloor(1, 1, 0.1, 0.05, 1), 65536n);
});

check("the error bound tightens as n grows", () => {
  const eps = (n) =>
    pk.run(
      pk.markov(5, 0.3),
      pk.config({
        n,
        seed: 1,
        invarianceLambda: 1.0,
        epsilon: 1.0,
        eta: 0.05,
        observationDiameter: 1.0,
        obsDim: 1,
      }),
    ).error_bound.epsilon;
  assert.ok(eps(65536) > eps(262144));
});

console.log(`\n${failures ? "FAILURES" : "all checks passed"}`);
process.exit(failures ? 1 : 0);
