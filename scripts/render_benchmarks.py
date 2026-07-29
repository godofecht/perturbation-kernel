#!/usr/bin/env python3
"""Turn criterion output into BENCHMARKS.md and the README table.

Run by `.github/workflows/bench.yml` so the numbers committed to the
repository are always ones CI measured, on a machine it names, rather
than ones a person typed in once.

Usage:  python3 scripts/render_benchmarks.py bench.raw
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}

# Criterion prints the benchmark id on its own line, then a `time:`
# line holding [lower estimate upper]. The estimate is the middle one.
ID_RE = re.compile(r"^((?:reduce|engine|family)/\S+)\s*$", re.M)
TIME_RE = re.compile(
    r"time:\s+\[[\d.]+ \S+ ([\d.]+) (\S+) [\d.]+ \S+\]"
)

README_START = "<!-- BENCH:START -->"
README_END = "<!-- BENCH:END -->"


def parse(raw: str) -> dict[str, float]:
    ids = ID_RE.findall(raw)
    times = TIME_RE.findall(raw)
    if len(ids) != len(times):
        print(
            f"warning: {len(ids)} ids but {len(times)} timings; "
            "pairing the shorter of the two",
            file=sys.stderr,
        )
    return {
        name: float(value) * UNITS[unit]
        for name, (value, unit) in zip(ids, times)
    }


def us(seconds: float) -> str:
    micros = seconds * 1e6
    if micros >= 1000:
        return f"{micros / 1000:,.1f} ms"
    if micros >= 1:
        return f"{micros:,.1f} us"
    return f"{micros * 1000:,.0f} ns"


def ratio(res: dict[str, float], slow: str, fast: str) -> str | None:
    if slow not in res or fast not in res:
        return None
    return f"{res[slow] / res[fast]:.2f}x"


def row(res, label, slow, fast):
    r = ratio(res, slow, fast)
    if r is None:
        return None
    return f"| {label} | {us(res[slow])} | {us(res[fast])} | **{r}** |"


def build(res: dict[str, float]) -> tuple[str, str]:
    """Return (full BENCHMARKS.md body, short README table)."""
    cpu = os.environ.get("BENCH_CPU", "unknown CPU")
    cores = os.environ.get("BENCH_CORES", "?")
    rustc = os.environ.get("BENCH_RUSTC", "unknown rustc")
    run = os.environ.get("BENCH_RUN", "")
    sha = os.environ.get("BENCH_SHA", "")[:12]

    header = [
        "# Benchmarks",
        "",
        "Measured by CI, not by hand. Regenerated on every push that",
        "touches `src/`, `benches/` or `Cargo.toml`, and monthly so the",
        "numbers keep tracking toolchain drift.",
        "",
        f"- **Machine** {cpu}, {cores} cores",
        f"- **Toolchain** {rustc}",
        f"- **Commit** `{sha}`",
    ]
    if run:
        header.append(f"- **Run** [{run}](../../actions/runs/{run})")
    header += [
        "",
        "A shared CI runner is a noisy place to measure. Treat these as",
        "order-of-magnitude guidance; the parallel figures in particular",
        "move between runs with whatever else the host is doing.",
        "",
        "Every comparison is against a path that computes the *same",
        "number*, so the ratios mean something.",
        "",
    ]

    sections: list[str] = []

    # ---- reductions ----
    rows = []
    for n in ("1024", "65536", "1048576"):
        rows.append(
            row(
                res,
                f"`tree_sum`, N = {int(n):,}",
                f"reduce/tree_sum/v1_reference/{n}",
                f"reduce/tree_sum/simd/{n}",
            )
        )
        rows.append(
            row(
                res,
                f"`sum_sq_dev`, N = {int(n):,}",
                f"reduce/sum_sq_dev/v1_reference/{n}",
                f"reduce/sum_sq_dev/simd/{n}",
            )
        )
    rows = [r for r in rows if r]
    if rows:
        sections += [
            "## Reductions",
            "",
            "Against `reduce::reference`, which is the literal v1.0.0 code.",
            "Two changes stack here: the reduction no longer allocates a",
            "`Vec` per tree level, and the levels are vectorised.",
            "",
            "| | v1.0.0 | vectorised | speedup |",
            "|---|---|---|---|",
            *rows,
            "",
        ]

    # Isolate the vectorisation from the allocation change.
    rows = [
        r
        for r in (
            row(
                res,
                f"`tree_sum`, N = {int(n):,}",
                f"reduce/tree_sum/scalar/{n}",
                f"reduce/tree_sum/simd/{n}",
            )
            for n in ("1024", "65536", "1048576")
        )
        if r
    ]
    if rows:
        sections += [
            "### Vectorisation alone",
            "",
            "Against an allocation-free scalar loop with the same tree",
            "shape, so this is the vector unit and nothing else.",
            "",
            "| | scalar | vectorised | speedup |",
            "|---|---|---|---|",
            *rows,
            "",
        ]

    # ---- engine ----
    rows = []
    for n in ("16384", "262144"):
        for fam in ("gaussian_d3", "bistable", "markov"):
            rows.append(
                row(
                    res,
                    f"`{fam}`, N = {int(n):,}",
                    f"engine/{fam}/scalar/{n}",
                    f"engine/{fam}/auto/{n}",
                )
            )
    rows = [r for r in rows if r]
    if rows:
        sections += [
            "## Engine",
            "",
            "`Backend::Scalar` (one thread, scalar loops) against",
            "`Backend::Auto` (threaded draws, vectorised reduction).",
            "Both produce identical bits.",
            "",
            "| | scalar | auto | speedup |",
            "|---|---|---|---|",
            *rows,
            "",
        ]

    # ---- family ----
    r = row(
        res,
        "`gaussian_d3`, N = 262,144",
        "family/gaussian_d3/trait/262144",
        "family/gaussian_d3/family/262144",
    )
    if r:
        sections += [
            "## Flat storage",
            "",
            "The `Family` path knows the observation width, so it writes",
            "into one buffer instead of allocating per draw.",
            "",
            "| | trait path | `Family` path | speedup |",
            "|---|---|---|---|",
            r,
            "",
        ]

    full = "\n".join(header + sections).rstrip() + "\n"

    # ---- short README table ----
    def best(prefix_slow, prefix_fast, keys):
        vals = []
        for k in keys:
            a, b = prefix_slow.format(k), prefix_fast.format(k)
            if a in res and b in res:
                vals.append(res[a] / res[b])
        return vals

    red = best(
        "reduce/tree_sum/v1_reference/{}",
        "reduce/tree_sum/simd/{}",
        ("1024", "65536", "1048576"),
    ) + best(
        "reduce/sum_sq_dev/v1_reference/{}",
        "reduce/sum_sq_dev/simd/{}",
        ("1024", "65536", "1048576"),
    )
    eng = []
    for n in ("16384", "262144"):
        for fam in ("gaussian_d3", "bistable", "markov"):
            eng += best(
                f"engine/{fam}/scalar/{n}" + "{}",
                f"engine/{fam}/auto/{n}" + "{}",
                ("",),
            )

    short = [
        README_START,
        "",
        "| | speedup | against |",
        "|---|---|---|",
    ]
    if red:
        short.append(
            f"| Reductions | {min(red):.1f}x to {max(red):.1f}x | the v1.0.0 reduction |"
        )
    if eng:
        short.append(
            f"| Engine | {min(eng):.1f}x to {max(eng):.1f}x | one thread, scalar loops |"
        )
    short += [
        "",
        f"Measured by CI on {cpu} ({cores} cores). Full tables in",
        "[BENCHMARKS.md](BENCHMARKS.md).",
        "",
        README_END,
    ]
    return full, "\n".join(short)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    raw = pathlib.Path(sys.argv[1]).read_text(errors="ignore")
    res = parse(raw)
    if not res:
        print("no benchmark results parsed; leaving files alone", file=sys.stderr)
        return 1
    print(f"parsed {len(res)} benchmark results")

    full, short = build(res)
    pathlib.Path("BENCHMARKS.md").write_text(full)

    readme = pathlib.Path("README.md")
    text = readme.read_text()
    if README_START in text and README_END in text:
        pre = text.split(README_START)[0]
        post = text.split(README_END)[1]
        readme.write_text(pre + short + post)
        print("updated the README table")
    else:
        print(
            f"README has no {README_START} block; wrote BENCHMARKS.md only",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
