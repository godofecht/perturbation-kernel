# Language bindings

Six surfaces, one engine. Rust and Python are published to registries;
the other four are built from this repository.

| Language | Mechanism | Path |
|---|---|---|
| Rust | native | `src/` |
| Python | PyO3, abi3 wheels | `python/` |
| C / C++ | C ABI, header-only C++20 | `bindings/cpp/` |
| Zig | `@cImport` over the C ABI | `bindings/zig/` |
| Julia | `ccall` over the C ABI | `bindings/julia/` |
| TypeScript | wasm32 via `wasm-bindgen` | `bindings/ts/` |

## They return the same bits

Each binding ships a conformance test asserting the same reference
value. That catches a binding that broke, but says nothing about the
bindings relative to each other, so CI also runs every language on one
runner and compares raw `f64` bit patterns:

```
family {"family": "markov", "k": 5, "start": 0, "base_label": 0, "theta_max": 0.3}
n = 262,144  seed = 20260610

language       value                  bits
----------------------------------------------------------
rust           0.8802871704101562     3fec2b5000000000
python         0.8802871704101562     3fec2b5000000000
c++            0.8802871704101562     3fec2b5000000000
julia          0.8802871704101562     3fec2b5000000000
typescript     0.8802871704101562     3fec2b5000000000
```

Comparing bits rather than decimals is deliberate. A boundary that
silently narrows a `double` to a `float` still prints the same six
decimal places.

## The ABI they share

Four of the six go through one C function:

```c
pk_report *pk_run_family(const char *family_json, const char *config_json,
                         int *out_err);
```

JSON on both sides, so a binding needs no struct layout agreement, no
function pointers and no lifetime discipline. It formats two strings,
gets a handle, reads a value, frees the handle.

The older vtable surface is still there for callers who want to supply
their own perturbation, forward model and functional. That is how you
*extend* the schema from C; `pk_run_family` is how you *use* it. See
the [C API reference](c-api.md).

## C++

Header-only, C++20 for designated initialisers. RAII on the report
handle and exceptions carrying the error code.

```cpp
#include <perturbation_kernel.hpp>
namespace pk = perturbation_kernel;

auto r = pk::Markov{.k = 5, .theta_max = 0.3}
             .run(pk::Config{.n = 262144, .seed = 20260610});
std::cout << r.value() << '\n';
```

```bash
cargo build --release
cmake -S bindings/cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Zig

```zig
const pk = @import("perturbation_kernel.zig");

var r = try (pk.Markov{ .k = 5, .theta_max = 0.3 })
    .run(allocator, .{ .n = 262144, .seed = 20260610 });
defer r.deinit();
```

```bash
cargo build --release
cd bindings/zig && zig build check
```

Zig 0.14's Mach-O reader rejects archive members produced by current
Apple toolchains, so on macOS the link goes through the system linker
instead. `build.zig` documents the two-step form, and it is what the
macOS CI job runs.

## Julia

No dependencies. Julia's standard library has no JSON module and a
binding should not pull one in to emit six fields.

```julia
using PerturbationKernel

r = run(Markov(k = 5, theta_max = 0.3), Config(n = 262144, seed = 20260610))
r.value
```

```bash
cargo build --release
PK_LIBRARY=$PWD/target/release/libperturbation_kernel.dylib \
  julia --project=bindings/julia -e 'using Pkg; Pkg.test()'
```

`PK_LIBRARY` points at the shared library; without it the module looks
for `libperturbation_kernel` on the default search path.

## TypeScript

Compiled to `wasm32-unknown-unknown`, so it runs in Node, Deno and the
browser with no native toolchain on the consumer's machine. Ships
`.d.ts` types.

```typescript
import * as pk from "perturbation-kernel";

const r = pk.run(pk.markov(5, 0.3), pk.config({ n: 262144, seed: 20260610 }));
r.value;
```

```bash
cd bindings/ts
wasm-pack build --target nodejs --out-dir pkg --release
node tests/test.mjs
```

Scalar backend only. Threads need shared memory that
`wasm32-unknown-unknown` does not provide by default, and the vector
kernels are architecture intrinsics with no wasm equivalent. Neither
costs anything in correctness: the scalar path *is* the reference path,
so the values are unchanged. The agreement table above includes the
wasm build for exactly that reason.

## Two bugs the bindings found

Adding callers in four more languages surfaced two defects that Rust
and Python had never exercised.

**Null parameters were compared by strict JSON value identity.**
JavaScript has one number type, so `JSON.stringify(0.0)` is `0`, and
`serde_json::Value` integer `0` did not equal float `0.0`. Every
JavaScript caller was rejected with a null-parameter mismatch. They are
now compared numerically when both sides are numbers, structurally
otherwise, so a family whose `theta` is a vector keeps exact matching.

**`getrandom` blocked the wasm32 target.** It refuses to build there
without being told which host API to use. The crate never calls it,
since every random bit comes from a ChaCha20 stream keyed by the config
seed, but it still has to link. Dropping `rand`'s `std` feature to
remove it would have routed `rand_distr`'s ziggurat through `libm`
rather than the platform math library, which could move the values and
break bit-identity with v1.0.0, so the shim is scoped to wasm32 alone.
