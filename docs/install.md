# Install

## Python

```bash
pip install perturbation-kernel
```

Wheels ship for Linux, macOS and Windows. They are built against the
stable ABI, so one wheel per platform serves every CPython from 3.8
onward and you do not need a Rust toolchain.

The GPU backend is compiled into the published wheels. `wgpu` loads its
driver lazily and reports no adapter on a machine without one, so the
wheel installs and runs identically on a headless box. Check what you
actually got:

```python
import perturbation_kernel as pk

pk.available_backends()   # ['auto', 'scalar', 'simd', 'gpu']
pk.simd_path()            # 'neon' | 'avx2' | 'scalar'
pk.gpu_device()           # 'Apple M4 Max (Metal, IntegratedGpu)' or None
```

### From source

```bash
git clone https://github.com/godofecht/perturbation-kernel
cd perturbation-kernel/python
pip install maturin
maturin develop --release
```

## Rust

```toml
[dependencies]
perturbation-kernel = "2"
```

Requires Rust 1.85 or later, or 1.87 with the `gpu` feature, which
depends on `wgpu`. Default features are `parallel` and `simd`, both of
which are bit-identical to the reference path.

```toml
# Portable scalar only: no rayon, no intrinsics.
perturbation-kernel = { version = "2", default-features = false }

# With the wgpu compute backend.
perturbation-kernel = { version = "2", features = ["gpu"] }
```

| Feature | Default | Effect |
|---|---|---|
| `parallel` | yes | Spreads the draw loop across a `rayon` pool above 4096 draws |
| `simd` | yes | NEON on aarch64, AVX2 on x86-64, for the reductions |
| `gpu` | no | Adds `wgpu`, enabling `Backend::Gpu` for the built-in families |

Disabling either default feature changes how long a run takes and
nothing else. There is no configuration under which the numbers move.

## C and C++

The crate builds a `cdylib` and a `staticlib` carrying the C ABI:

```bash
cargo build --release
# target/release/libperturbation_kernel.{dylib,so,a}
```

See the [C API reference](c-api.md).

## Lean 4

```bash
cd lean/PerturbationKernel
lake build
```

## Verifying the install

The fastest meaningful check is that two backends agree exactly:

```python
import perturbation_kernel as pk

a = pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=50_000, seed=7))
b = pk.Markov(k=5, theta_max=0.3).run(pk.Config(n=50_000, seed=7, backend="scalar"))
assert a.value.hex() == b.value.hex()
```

If that fails, the build is broken in a way that matters. Please open an
issue with the output of `pk.simd_path()` and your platform.
