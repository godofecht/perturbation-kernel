//! GPU execution backend (`wgpu`, Metal / Vulkan / DX12).
//!
//! # What runs on the device
//!
//! Both halves of the estimator: the draw loop of SCHEMA §4.4 (one
//! invocation per index `i`) and the reduction of SCHEMA §8 D3 (one
//! dispatch per tree level, ping-ponged between two buffers so no
//! invocation reads a slot another has overwritten). Only the final
//! `d` scalars cross back over PCIe.
//!
//! The per-index substream fork is the same one the host uses. The
//! device recomputes `mix64(seed, i)` in emulated 64-bit arithmetic,
//! builds the identical 32-byte ChaCha20 key, and reads the identical
//! keystream. Two draws with the same `(seed, i)` therefore consume the
//! same random bits on the host and on the device.
//!
//! # Two device backends
//!
//! [`Backend::Gpu`](crate::config::Backend::Gpu) is **bit-identical to
//! the host**. [`Backend::GpuF32`](crate::config::Backend::GpuF32) is
//! single precision and agrees only statistically. They are separate
//! flags rather than one flag with a caveat, because the difference
//! between "the same number" and "a number that close" is exactly the
//! thing a caller needs to opt into on purpose.
//!
//! ## The exact path
//!
//! Bit-identity is achievable when a family's arithmetic is a short
//! list of exactly-specified operations. The Markov family qualifies:
//! it draws one uniform intensity, one uniform variate and one bounded
//! integer, and its observation is an indicator.
//!
//! Three things make that reproducible on a device with no `f64` type:
//!
//! * `f64.wgsl` emulates IEEE-754 binary64 in `u32` pairs, with
//!   round-to-nearest-even multiplication. The uniform path performs
//!   exactly one rounded multiply, so emulating that one operation is
//!   enough.
//! * `rand`'s integer sampler is Lemire's multiply-shift with
//!   rejection, which is pure integer arithmetic and therefore exact
//!   anywhere. It is transcribed rather than approximated, down to
//!   using the conservative `(range << range.leading_zeros()) - 1`
//!   rejection zone that `rand` uses for 32-bit types.
//! * The observation is `0` or `1`, so the ensemble is a count and the
//!   reduction is integer addition. Integer addition is associative,
//!   so the total does not depend on how the device schedules anything.
//!   The host's `f64` tree sum of zeros and ones is the same integer.
//!
//! `tests/gpu.rs` pins each layer separately, so a failure names the
//! layer: the keystream against the host word for word, the
//! `(low, scale)` derivation against `rand` itself, and finally the
//! reports against each other on `to_bits()` across 297 combinations of
//! alphabet size, intensity, ensemble size and seed.
//!
//! Families that draw normal deviates do not qualify and are refused
//! rather than silently approximated. The ziggurat of `rand_distr`
//! calls `ln` and `exp` on its rejection paths, and WGSL specifies
//! neither exactly; emulating them in software `f64` would be both
//! large and slower than the CPU it is meant to beat.
//!
//! ## The single-precision path
//!
//! [`Backend::GpuF32`](crate::config::Backend::GpuF32) runs every
//! family, carries the ensemble in `f32`, and draws normals by
//! Box-Muller. It is considerably faster and considerably less
//! reproducible:
//!
//! | Property | Holds? |
//! |---|---|
//! | Same device, same `(seed, N)`, repeated runs | yes, bit-identical |
//! | Same device, different workgroup scheduling | yes, bit-identical |
//! | Different devices, Markov family | yes |
//! | Different devices, Gaussian / Bistable | last-ulp differences possible |
//! | Device vs. host | statistically equivalent |
//!
//! Box-Muller calls `log`, `sqrt` and `cos`, whose accuracy WGSL leaves
//! to the driver; that is the only cross-device wobble, and it is
//! confined to the families that draw normals.
//!
//! The *reduction* is exactly specified on every conforming device even
//! here: each output is one IEEE-754 `f32` addition of one fixed pair,
//! and the levels ping-pong between two buffers so no invocation reads
//! a slot another has overwritten. `tests/gpu.rs` checks it against a
//! host `f32` tree, bit for bit.
//!
//! Which backend produced a value is recorded in
//! [`crate::report::Execution`], so a report can never be mistaken for
//! a host-computed one.

use std::sync::{Arc, OnceLock};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::config::Config;
use crate::family::Family;
use crate::report::{Execution, Report};
use crate::{Error, Result};

/// Invocations per workgroup. Matches `@workgroup_size(64)` in both
/// shaders.
const WORKGROUP: u32 = 64;

/// Maximum workgroups along the x axis of a dispatch. Anything larger
/// is folded into the y axis; the default device limit is 65535.
const MAX_GRID_X: u32 = 32_768;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawParams {
    n: u32,
    seed_lo: u32,
    seed_hi: u32,
    family: u32,
    d: u32,
    k: u32,
    start: u32,
    base_label: u32,
    grid_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    p_intensity: f32,
    p_dt: f32,
    p_x0: f32,
    pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MarkovParams {
    n: u32,
    seed_lo: u32,
    seed_hi: u32,
    k: u32,
    start: u32,
    base_label: u32,
    grid_x: u32,
    pad0: u32,
    scale_hi: u32,
    scale_lo: u32,
    low_hi: u32,
    low_lo: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReduceParams {
    len: u32,
    cols: u32,
    stride: u32,
    grid_x: u32,
    mode: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A device plus the two compiled pipelines.
///
/// Acquiring an adapter costs tens of milliseconds, so one context is
/// created per process and shared. It is `Send + Sync`.
pub struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    draw: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    markov_exact: wgpu::ComputePipeline,
    debug_uniform: wgpu::ComputePipeline,
    /// Human-readable device description, surfaced in
    /// [`Execution::device`].
    pub name: String,
    max_binding: u64,
}

static CONTEXT: OnceLock<std::result::Result<Arc<Context>, String>> = OnceLock::new();

/// The process-wide device context, initialising it on first call.
///
/// Returns [`Error::Gpu`] when no adapter is available. Callers that
/// want to *skip* rather than fail (tests on a headless runner, for
/// instance) should match on that variant.
pub fn context() -> Result<Arc<Context>> {
    CONTEXT
        .get_or_init(|| Context::new().map(Arc::new).map_err(|e| e.to_string()))
        .clone()
        .map_err(Error::Gpu)
}

/// `true` when a compute device could be acquired.
///
/// Intended for tests and for feature probing at runtime.
pub fn available() -> bool {
    context().is_ok()
}

impl Context {
    fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| Error::Gpu(format!("no compute adapter: {e}")))?;

        let info = adapter.get_info();
        let name = format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type);
        let limits = adapter.limits();
        let max_binding = limits.max_storage_buffer_binding_size as u64;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("perturbation-kernel"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| Error::Gpu(format!("device request failed: {e}")))?;

        // A validation error inside a dispatch is otherwise reported
        // asynchronously and lost; make it panic loudly instead.
        device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu validation: {e}")));

        let draw = compile(
            &device,
            "draw",
            concat!(
                include_str!("shaders/common.wgsl"),
                include_str!("shaders/draw.wgsl")
            ),
        );
        let reduce = compile(
            &device,
            "collapse",
            concat!(
                include_str!("shaders/common.wgsl"),
                include_str!("shaders/reduce.wgsl")
            ),
        );

        let markov_exact = compile(
            &device,
            "draw_markov",
            concat!(
                include_str!("shaders/f64.wgsl"),
                include_str!("shaders/common.wgsl"),
                include_str!("shaders/markov_exact.wgsl")
            ),
        );

        let debug_uniform = compile(
            &device,
            "debug_uniform",
            concat!(
                include_str!("shaders/f64.wgsl"),
                include_str!("shaders/common.wgsl"),
                include_str!("shaders/markov_exact.wgsl"),
                include_str!("shaders/debug_uniform.wgsl")
            ),
        );

        Ok(Self {
            device,
            queue,
            draw,
            reduce,
            markov_exact,
            debug_uniform,
            name,
            max_binding,
        })
    }
}

fn compile(device: &wgpu::Device, entry: &str, src: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Workgroup grid covering `threads` invocations.
///
/// Returns `(x, y)` workgroup counts; the shaders fold `(x, y)` back to
/// a linear index using `x` and drop the overshoot with a bounds check.
fn grid(threads: u32) -> (u32, u32) {
    let wg = threads.div_ceil(WORKGROUP).max(1);
    let x = wg.min(MAX_GRID_X);
    (x, wg.div_ceil(x))
}

/// Run a built-in family on a compute device.
///
/// Called by [`crate::family::Family::run`] when
/// [`crate::config::Backend::Gpu`] is selected; the config has already
/// been validated at that point.
pub fn run_family(family: &Family, cfg: &Config) -> Result<Report> {
    let ctx = context()?;

    if cfg.backend == crate::config::Backend::Gpu {
        // The exact path, or nothing. Silently returning a different
        // number under a backend flag would be the worst outcome
        // available, so an unsupported family is an error that names
        // the alternative.
        let Family::Markov {
            k,
            start,
            base_label,
            theta_max,
        } = family
        else {
            return Err(Error::UnsupportedBackend {
                backend: "gpu",
                reason: format!(
                    "the {} family draws normal deviates, which need transcendental \
                     functions WGSL does not specify exactly; use backend \"gpu_f32\" \
                     to accept a single-precision result, or a host backend for an \
                     exact one",
                    family.name()
                ),
            });
        };
        let value = ctx.evaluate_markov_exact(*k, *start, *base_label, *theta_max, cfg)?;
        return Ok(family.finish(
            value,
            cfg,
            Execution {
                backend: "gpu".to_string(),
                simd_path: "device".to_string(),
                threaded: true,
                device: Some(ctx.name.clone()),
                precision: "f64".to_string(),
            },
        ));
    }

    let value = ctx.evaluate(family, cfg)?;
    let exec = Execution {
        backend: "gpu_f32".to_string(),
        simd_path: "device".to_string(),
        threaded: true,
        device: Some(ctx.name.clone()),
        precision: "f32".to_string(),
    };
    Ok(family.finish(value, cfg, exec))
}

/// `rand`'s `UniformFloat<f64>::new_inclusive`, reproduced so the device
/// can be handed the same `(low, scale)` pair the host samples with.
///
/// The fields of `rand`'s `UniformFloat` are private, so this is a
/// transcription rather than a query. `tests/gpu.rs` pins it by drawing
/// from both this and `rand::distributions::Uniform` and asserting the
/// samples are bit-identical.
pub fn uniform_inclusive_params(low: f64, high: f64) -> (f64, f64) {
    // (u64::MAX >> 12).into_float_with_exponent(0) - 1.0
    let max_rand = f64::from_bits((u64::MAX >> 12) | (1023u64 << 52)) - 1.0;
    let mut scale = (high - low) / max_rand;
    // Walk `scale` down until the largest representable draw no longer
    // overshoots `high`. In practice this runs zero or one times.
    while scale * max_rand + low > high {
        scale = f64::from_bits(scale.to_bits() - 1);
    }
    (low, scale)
}

impl Context {
    /// Bit-exact Markov evaluation.
    ///
    /// The device counts how many draws kept `base_label` and returns
    /// that count as an integer. The host then divides by `n` in `f64`,
    /// exactly as `reduce::mean` does: a tree sum of values that are
    /// each `0.0` or `1.0` is an exact integer in `f64`, so the two
    /// reductions agree by construction and the only thing that has to
    /// match is the count.
    fn evaluate_markov_exact(
        &self,
        k: u32,
        start: u32,
        base_label: u32,
        theta_max: f64,
        cfg: &Config,
    ) -> Result<f64> {
        let n = u32::try_from(cfg.n)
            .map_err(|_| Error::Gpu(format!("n = {} exceeds the 2^32 device limit", cfg.n)))?;

        let (low, scale) = uniform_inclusive_params(0.0, theta_max);
        let params = MarkovParams {
            n,
            seed_lo: cfg.seed as u32,
            seed_hi: (cfg.seed >> 32) as u32,
            k,
            start,
            base_label,
            grid_x: grid(n).0,
            pad0: 0,
            scale_hi: (scale.to_bits() >> 32) as u32,
            scale_lo: scale.to_bits() as u32,
            low_hi: (low.to_bits() >> 32) as u32,
            low_lo: low.to_bits() as u32,
        };

        let count = self.count_hits(&params, n)?;
        Ok(count as f64 / cfg.n as f64)
    }

    /// Dispatch the exact Markov kernel and read back the hit count.
    fn count_hits(&self, params: &MarkovParams, n: u32) -> Result<u32> {
        let pbuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("markov-params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let total = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("markov-total"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("markov-exact"),
            layout: &self.markov_exact.get_bind_group_layout(0),
            entries: &[entry(0, &pbuf), entry(1, &total)],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("markov-exact"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("markov-exact"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.markov_exact);
            pass.set_bind_group(0, &bind, &[]);
            let (x, y) = grid(n);
            pass.dispatch_workgroups(x, y, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("markov-readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&total, 0, &staging, 0, 4);
        self.queue.submit(Some(enc.finish()));

        let words = self.read_back_u32(&staging, 1)?;
        Ok(words[0])
    }

    fn evaluate(&self, family: &Family, cfg: &Config) -> Result<f64> {
        let n = u32::try_from(cfg.n)
            .map_err(|_| Error::Gpu(format!("n = {} exceeds the 2^32 device limit", cfg.n)))?;
        let d = family.obs_dim() as u32;
        let bytes = (n as u64) * (d as u64) * 4;
        if bytes > self.max_binding {
            return Err(Error::Gpu(format!(
                "ensemble of {n} x {d} f32 needs {bytes} B, device caps a storage binding at {} B",
                self.max_binding
            )));
        }

        let ensemble = self.draw_ensemble(family, cfg, n, d)?;
        let sums = self.reduce_columns(&ensemble, n, d, 0, None)?;

        match family {
            Family::Gaussian { .. } => {
                // Two-pass variance, mirroring the host: column means
                // first, then the centred second moment.
                let means: Vec<f32> = sums.iter().map(|s| s / n as f32).collect();
                let ssd = self.reduce_columns(&ensemble, n, d, 1, Some(&means))?;
                // Accumulate in coordinate order in f64, as
                // `NegDispersion::measure` does.
                let mut total = 0.0f64;
                for s in &ssd {
                    total += *s as f64 / cfg.n as f64;
                }
                Ok(-total)
            }
            _ => Ok(sums[0] as f64 / cfg.n as f64),
        }
    }

    /// Dispatch the draw kernel, returning the column-major ensemble
    /// buffer.
    fn draw_ensemble(&self, family: &Family, cfg: &Config, n: u32, d: u32) -> Result<wgpu::Buffer> {
        let (base, params) = describe(family, cfg, n, d);

        let base_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("base"),
                contents: bytemuck::cast_slice(&base),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let param_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("draw-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let ensemble = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ensemble"),
            size: (n as u64) * (d as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw"),
            layout: &self.draw.get_bind_group_layout(0),
            entries: &[
                entry(0, &param_buf),
                entry(1, &base_buf),
                entry(2, &ensemble),
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("draw"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("draw"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.draw);
            pass.set_bind_group(0, &bind, &[]);
            let (x, y) = grid(n);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.queue.submit(Some(enc.finish()));
        Ok(ensemble)
    }

    /// Reduce every column of `src` to a single scalar, level by level.
    ///
    /// `mode = 1` centres and squares against `means` in the first
    /// level, giving the sum of squared deviations. `src` is never
    /// written, so the same ensemble can be reduced twice.
    fn reduce_columns(
        &self,
        src: &wgpu::Buffer,
        n: u32,
        d: u32,
        mode: u32,
        means: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let stride = n as u64;
        let size = stride * d as u64 * 4;

        let scratch: [wgpu::Buffer; 2] = std::array::from_fn(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce-scratch"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });

        let mean_data: Vec<f32> = means
            .map(|m| m.to_vec())
            .unwrap_or_else(|| vec![0.0; d as usize]);
        let mean_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("means"),
                contents: bytemuck::cast_slice(&mean_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reduce"),
            });

        // Keep the per-level parameter buffers alive until submission.
        let mut keep = Vec::new();
        let mut len = n;
        let mut level = 0usize;
        while len > 1 {
            let out_len = len / 2 + (len & 1);
            let (x, y) = grid(out_len * d);
            let params = ReduceParams {
                len,
                cols: d,
                stride: n,
                grid_x: x,
                // Only the first level centres; later levels are plain
                // sums of the already-squared partials.
                mode: if level == 0 { mode } else { 0 },
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            let pbuf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("reduce-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let read: &wgpu::Buffer = if level == 0 {
                src
            } else {
                &scratch[(level - 1) % 2]
            };
            let write = &scratch[level % 2];

            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reduce"),
                layout: &self.reduce.get_bind_group_layout(0),
                entries: &[
                    entry(0, &pbuf),
                    entry(1, read),
                    entry(2, write),
                    entry(3, &mean_buf),
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("collapse"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.reduce);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
            keep.push((pbuf, bind));
            len = out_len;
            level += 1;
        }

        // Column `c`'s total sits at offset `c * stride`. Pull back the
        // `d` scalars rather than the whole buffer.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (d as u64 * 4).max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let final_buf: &wgpu::Buffer = if level == 0 {
            src
        } else {
            &scratch[(level - 1) % 2]
        };
        for c in 0..d as u64 {
            enc.copy_buffer_to_buffer(final_buf, c * stride * 4, &staging, c * 4, 4);
        }
        self.queue.submit(Some(enc.finish()));

        let out = self.read_back(&staging, d as usize)?;

        // `mode = 1` with `n == 1` never runs a level, so the centring
        // has to happen on the host for that degenerate case.
        if mode == 1 && level == 0 {
            let m = means.map(|m| m.to_vec()).unwrap_or_default();
            return Ok(out
                .iter()
                .enumerate()
                .map(|(c, x)| {
                    let dv = x - m.get(c).copied().unwrap_or(0.0);
                    dv * dv
                })
                .collect());
        }
        Ok(out)
    }

    fn read_back_u32(&self, staging: &wgpu::Buffer, count: usize) -> Result<Vec<u32>> {
        let bytes = self.map_and_read(staging)?;
        Ok(bytemuck::cast_slice::<u8, u32>(&bytes)[..count].to_vec())
    }

    fn read_back(&self, staging: &wgpu::Buffer, count: usize) -> Result<Vec<f32>> {
        let bytes = self.map_and_read(staging)?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)[..count].to_vec())
    }

    /// Block until the queue drains, then copy the mapped bytes out.
    fn map_and_read(&self, staging: &wgpu::Buffer) -> Result<Vec<u8>> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("device poll failed: {e}")))?;
        rx.recv()
            .map_err(|_| Error::Gpu("readback channel closed".to_string()))?
            .map_err(|e| Error::Gpu(format!("buffer map failed: {e}")))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("mapped range unavailable: {e}")))?;
        let out = data.to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Test hook: the first eight ChaCha20 keystream words the device
    /// reads for each draw index in `0..n`.
    ///
    /// Comparing raw words against the host locates a divergence at the
    /// stream level instead of inferring it from an aggregate.
    pub fn debug_stream_words(&self, seed: u64, n: u32, theta_max: f64) -> Result<Vec<[u32; 8]>> {
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct DebugParams {
            n: u32,
            seed_lo: u32,
            seed_hi: u32,
            grid_x: u32,
            scale_hi: u32,
            scale_lo: u32,
            pad0: u32,
            pad1: u32,
        }
        let (_low, scale) = uniform_inclusive_params(0.0, theta_max);
        let params = DebugParams {
            n,
            seed_lo: seed as u32,
            seed_hi: (seed >> 32) as u32,
            grid_x: grid(n).0,
            scale_hi: (scale.to_bits() >> 32) as u32,
            scale_lo: scale.to_bits() as u32,
            pad0: 0,
            pad1: 0,
        };
        let pbuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let size = (n as u64) * 32;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug-out"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug-readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug"),
            layout: &self.debug_uniform.get_bind_group_layout(0),
            entries: &[entry(0, &pbuf), entry(1, &out)],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("debug"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("debug"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.debug_uniform);
            pass.set_bind_group(0, &bind, &[]);
            let (x, y) = grid(n);
            pass.dispatch_workgroups(x, y, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));
        let words = self.read_back_u32(&staging, n as usize * 8)?;
        Ok((0..n as usize)
            .map(|i| {
                let mut row = [0u32; 8];
                row.copy_from_slice(&words[i * 8..i * 8 + 8]);
                row
            })
            .collect())
    }

    /// Reduce a host-supplied `f32` slice on the device.
    ///
    /// Exposed for `tests/gpu.rs`, which uses it to assert that the
    /// device reduction agrees bit for bit with a host `f32` tree
    /// reduction of the same input.
    pub fn reduce_f32(&self, xs: &[f32]) -> Result<f32> {
        if xs.is_empty() {
            return Ok(0.0);
        }
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("host-input"),
                contents: bytemuck::cast_slice(xs),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        Ok(self.reduce_columns(&buf, xs.len() as u32, 1, 0, None)?[0])
    }
}

fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

/// Flatten a [`Family`] into the device parameter block.
fn describe(family: &Family, cfg: &Config, n: u32, d: u32) -> (Vec<f32>, DrawParams) {
    let mut p = DrawParams {
        n,
        seed_lo: cfg.seed as u32,
        seed_hi: (cfg.seed >> 32) as u32,
        family: 0,
        d,
        k: 1,
        start: 0,
        base_label: 0,
        grid_x: grid(n).0,
        pad0: 0,
        pad1: 0,
        pad2: 0,
        p_intensity: 0.0,
        p_dt: 0.0,
        p_x0: 0.0,
        pad3: 0.0,
    };
    let base = match family {
        Family::Gaussian { base, sigma_max } => {
            p.family = 0;
            p.p_intensity = *sigma_max as f32;
            base.iter().map(|x| *x as f32).collect()
        }
        Family::Bistable { x0, dt, theta_max } => {
            p.family = 1;
            p.p_intensity = *theta_max as f32;
            p.p_dt = *dt as f32;
            p.p_x0 = *x0 as f32;
            vec![0.0]
        }
        Family::Markov {
            k,
            start,
            base_label,
            theta_max,
        } => {
            p.family = 2;
            p.k = *k;
            p.start = *start;
            p.base_label = *base_label;
            p.p_intensity = *theta_max as f32;
            vec![0.0]
        }
    };
    (base, p)
}
