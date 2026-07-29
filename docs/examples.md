# Examples

All of these are runnable files in the repository, and every number
below is real output.

## Python

In `python/examples/`. Install the package, then run them directly.

### 01. Does my finding survive?

```bash
python python/examples/01_does_my_finding_survive.py
```

A pipeline sorts samples into five categories and reports category 0.
How much input noise before that stops coming back?

```
 noise theta    survival   +/- bound  verdict
--------------------------------------------------------
        0.00      1.0000      0.0270  holds
        0.10      0.9606      0.0270  holds
        0.20      0.9202      0.0270  holds
        0.40      0.8401      0.0270  holds
        0.60      0.7603      0.0270  weakening
        0.80      0.6810      0.0270  GONE
        1.00      0.6012      0.0270  GONE

Survival crosses the 0.80 tolerance at theta = 0.4993.
```

The closed form puts the crossing at exactly `0.5`, so the estimate is
right to a tenth of a percent. The example also states the family's own
floor: mixing over `k` categories leaves a `1/k` chance of landing back
where you started, so survival bottoms out at `1 - (1 - 1/k)/2` and
never reaches zero. Comparing against that floor rather than against
zero is what keeps the reading honest.

### 02. Where does it tip?

```bash
python python/examples/02_where_does_it_tip.py
```

A bistable system with two stable regimes. How hard a push does it take
to flip, and how does that scale?

```
  start x0   tipping noise    noise / x0
----------------------------------------
      0.05           2.838          56.8
      0.10           5.673          56.7
      0.20          11.334          56.7
      0.30          16.968          56.6
      0.50          28.105          56.2
      0.80          44.287          55.4
```

The ratio is constant to 2.5% across a sixteen-fold range, so the
tipping intensity is proportional to the distance from the boundary.
That is worth knowing precisely because the potential is quartic and you
might have expected otherwise; over a single step of this size the drift
contributes almost nothing.

The second half sweeps the integration step and finds `noise * sqrt(dt)`
holding to 5% over the three smallest steps and drifting one-directionally
to 32% across the full range. The drift is not noise in the estimate, it
is the deterministic term becoming visible: drift contributes
displacement proportional to `dt` while noise contributes `sqrt(dt)`.
Fitting `1/sqrt(dt)` across the whole range would have buried exactly
the nonlinearity you were looking for.

### 03. How many draws do I need?

```bash
python python/examples/03_how_many_draws.py
```

Works backwards from a claim to an ensemble size, then checks the answer
empirically.

```
  n = 262,143 is refused: sample-complexity floor
  n = 262,144 is accepted, bound +/-0.0270

  closed form                0.880000
  8M-draw estimate           0.879945
  gap                        5.54e-05

  20 independent seeds at n = 262,144
  worst error seen           0.001216
  claimed bound              0.05
  claims violated            0/20
```

### 04. When is the GPU worth it?

```bash
python python/examples/04_gpu_sweep.py
```

Times host against device, finds the crossover, and checks agreement at
every size. Then runs a 25-cell sweep on whichever won. On an M4 Max:

```
       draws    host (ms)   device (ms)    speedup       |gap|        tol
--------------------------------------------------------------------------
      10,000         0.44          0.63      0.71x    4.20e-03   5.00e-02
      50,000         1.56          0.66      2.35x    6.60e-04   2.24e-02
     200,000         6.32          0.94      6.70x    3.00e-04   1.12e-02
   1,000,000        27.30          2.63     10.37x    3.90e-04   5.00e-03
   4,000,000       108.48          2.60     41.76x    8.78e-05   2.50e-03

25 cells, 500,000 draws each, 12,500,000 draws total
in 0.02 s -- 591.2M draws/second
```

Skips the comparison cleanly and explains why on a machine with no
device.

### 05. Publishing a number someone else can check

```bash
python python/examples/05_reproducibility_receipt.py
```

Issues a receipt, verifies it, then verifies three that should fail or
should not:

```
Verifying it
  PASS: reproduced exactly: 0.8802871704101562

Verifying a receipt that has been tampered with
  FAIL: value changed: receipt 0x1.c2b5000000000p-1, got 0x1.c0b8800000000p-1

Verifying the same run on a different backend
   scalar: PASS -- reproduced exactly: 0.8802871704101562
     simd: PASS -- reproduced exactly: 0.8802871704101562

Verifying on the GPU
  FAIL -- value changed: receipt 0x1.c2b5000000000p-1, got 0x1.c300800000000p-1
```

The last one is expected and is the point. A receipt is a claim about an
exact computation, so verify it on the backend that issued it. That is
why every report records its own `execution` block.

## Rust

In `examples/`.

### Your own model

```bash
cargo run --release --example custom_family
```

Implements all three traits for a real question: a fitted regression
slope is positive, does it stay positive under measurement error?

```
observations   12
fitted slope   +0.11836 per step

   error sigma   sign survives     +/- bound  verdict
------------------------------------------------------------
          0.00          1.0000       0.00112  solid
          0.25          1.0000       0.00112  solid
          0.50          0.9998       0.00112  solid
          1.00          0.9833       0.00112  holds
          2.00          0.9065       0.00112  shaky
          4.00          0.7965       0.00112  GONE

Sign survival drops below 95% at sigma = 2, which is
1.6x the full spread of the data (1.26).
```

The ladder deliberately runs past the spread of the data itself.
Stopping where the answer is still "solid" would only show that the
question had not been pushed hard enough to have an answer.

Read this one if you are writing your own `Perturbation`,
`ForwardModel` and `Invariance`. It is commented against the C2, C3 and
C4 conformance points.

### Golden values

```bash
cargo run --release --example golden
```

Prints the 42 `Report.value` bit patterns that CI diffs against
`GOLDEN.txt` on every platform. This is the check that keeps every
optimisation in the crate honest.
