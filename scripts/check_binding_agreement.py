#!/usr/bin/env python3
"""Check that every language binding returns the same bits.

Each binding has its own conformance test asserting a hardcoded
reference value. That catches a binding that broke, but not a reference
value that was wrong in the first place, and it says nothing about the
bindings *relative to each other*.

This does the complementary check: run the same family and config
through every language available on the machine, and compare the raw
`f64` bit patterns. Any disagreement is a boundary that lost precision,
which is the failure mode a decimal comparison would hide.

Run by `.github/workflows/bindings.yml`. Languages that are not
installed are reported and skipped rather than failing the run, so the
same script is useful on a workstation.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# One fixed problem, chosen because every binding supports it and the
# closed form is known: 1 - (theta/2)(1 - 1/k) = 0.88 for these values.
N = 262_144
SEED = 20_260_610
FAMILY = {"family": "markov", "k": 5, "start": 0, "base_label": 0, "theta_max": 0.3}
CONFIG = {
    "schema_version": "1.0.0",
    "n": N,
    "seed": SEED,
    "intensity": {
        "kind": "uniform_interval",
        "params": {},
        "null_parameter": 0.0,
    },
    "reduction": {"order": "tree", "leaf_order": "index"},
    "lipschitz": {},
}


def bits(x: float) -> str:
    """The exact bit pattern, because decimals hide the differences."""
    return struct.pack(">d", x).hex()


def run(cmd, **kw) -> str:
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, cwd=ROOT, **kw
    ).stdout.strip()


results: dict[str, float] = {}
skipped: list[str] = []


# ---- Rust -----------------------------------------------------------
src = textwrap.dedent(
    """
    use perturbation_kernel::config::Config;
    use perturbation_kernel::family::Family;
    fn main() {
        let f = Family::Markov { k: 5, start: 0, base_label: 0, theta_max: 0.3 };
        let c = Config { n: %d, seed: %d, ..Default::default() };
        println!("{}", f.run(&c).unwrap().value);
    }
    """
    % (N, SEED)
)
(ROOT / "examples" / "_agreement.rs").write_text(src)
try:
    results["rust"] = float(run(["cargo", "run", "-q", "--release", "--example", "_agreement"]))
finally:
    (ROOT / "examples" / "_agreement.rs").unlink(missing_ok=True)


# ---- Python ---------------------------------------------------------
try:
    import perturbation_kernel as pk  # noqa: F401

    results["python"] = float(
        run(
            [
                sys.executable,
                "-c",
                "import perturbation_kernel as pk;"
                f"print(pk.Markov(k=5, theta_max=0.3).run(pk.Config(n={N}, seed={SEED})).value)",
            ]
        )
    )
except ImportError:
    skipped.append("python (package not installed)")


# ---- C++ ------------------------------------------------------------
cpp = ROOT / "build" / "test_bindings"
if cpp.exists():
    # The C++ test prints its own checks; re-run the value through a
    # tiny program instead so the comparison is on one number.
    prog = ROOT / "build" / "_agreement.cpp"
    prog.write_text(
        textwrap.dedent(
            f"""
            #include <perturbation_kernel.hpp>
            #include <cstdio>
            int main() {{
              namespace pk = perturbation_kernel;
              auto r = pk::Markov{{.k = 5, .theta_max = 0.3}}
                           .run(pk::Config{{.n = {N}, .seed = {SEED}}});
              std::printf("%.17g\\n", r.value());
            }}
            """
        )
    )
    lib = next((ROOT / "target" / "release").glob("libperturbation_kernel.a"), None)
    if lib:
        out = ROOT / "build" / "_agreement"
        # The Rust staticlib drags in the platform's system libraries;
        # wgpu adds the graphics frameworks on Apple targets.
        if sys.platform == "darwin":
            sys_libs = [
                "-framework", "CoreFoundation", "-framework", "Foundation",
                "-framework", "Metal", "-framework", "QuartzCore",
                "-framework", "CoreGraphics", "-framework", "IOKit",
                "-framework", "IOSurface", "-framework", "AppKit",
                "-lobjc", "-liconv",
            ]
        else:
            sys_libs = ["-lpthread", "-ldl", "-lm"]
        subprocess.run(
            ["c++", "-std=c++20", "-I", str(ROOT / "bindings/cpp/include"),
             str(prog), str(lib), *sys_libs, "-o", str(out)],
            check=True, cwd=ROOT,
        )
        results["c++"] = float(run([str(out)]))
    else:
        skipped.append("c++ (no static library)")
else:
    skipped.append("c++ (not built)")


# ---- TypeScript / wasm ----------------------------------------------
if (ROOT / "bindings/ts/pkg").exists() and shutil.which("node"):
    results["typescript"] = float(
        run(
            ["node", "-e",
             "const pk=require('./bindings/ts/index.js');"
             f"console.log(pk.run(pk.markov(5,0.3),pk.config({{n:{N},seed:{SEED}}})).value)"]
        )
    )
else:
    skipped.append("typescript (wasm package not built)")


# ---- Julia ----------------------------------------------------------
if shutil.which("julia"):
    lib = next(
        (p for p in (ROOT / "target" / "release").glob("libperturbation_kernel.*")
         if p.suffix in {".so", ".dylib"}),
        None,
    )
    if lib:
        results["julia"] = float(
            run(
                ["julia", "--project=bindings/julia", "-e",
                 "using PerturbationKernel;"
                 f"println(run(Markov(k=5, theta_max=0.3), Config(n={N}, seed={SEED})).value)"],
                env={**__import__("os").environ, "PK_LIBRARY": str(lib)},
            )
        )
    else:
        skipped.append("julia (no shared library)")
else:
    skipped.append("julia (not installed)")


# ---- Zig ------------------------------------------------------------
if shutil.which("zig"):
    skipped.append("zig (covered by its own conformance job)")
else:
    skipped.append("zig (not installed)")


# ---- verdict --------------------------------------------------------
print(f"family {json.dumps(FAMILY)}")
print(f"n = {N:,}  seed = {SEED}\n")
print(f"{'language':<14} {'value':<22} bits")
print("-" * 58)
for name, value in results.items():
    print(f"{name:<14} {value!r:<22} {bits(value)}")

if skipped:
    print("\nskipped:")
    for s in skipped:
        print(f"  {s}")

distinct = {bits(v) for v in results.values()}
print()
if len(results) < 2:
    print("FAIL: fewer than two bindings ran, so nothing was compared")
    sys.exit(1)
if len(distinct) != 1:
    print(f"FAIL: {len(distinct)} distinct bit patterns across {len(results)} bindings")
    sys.exit(1)
print(f"all {len(results)} bindings returned the same bits: {distinct.pop()}")
