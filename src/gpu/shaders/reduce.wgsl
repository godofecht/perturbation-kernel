// Device reduction with the host tree shape (SCHEMA §8 D3).
//
// One dispatch collapses one level: `dst[k] = src[2k] + src[2k+1]`,
// with an odd tail element carried up unchanged. Source and
// destination are separate buffers and swapped between levels, so no
// invocation can read a slot another invocation has already
// overwritten; the result is a pure function of the input and
// independent of scheduling.
//
// Because every output is a single IEEE-754 `f32` addition of one
// fixed pair, this reproduces `reduce::tree_sum` run in single
// precision *exactly*, on any conforming device.
// `tests/gpu.rs::gpu_reduction_matches_host_f32_tree` asserts it.

struct ReduceParams {
    len: u32,      // live length of each column in `src`
    cols: u32,     // number of independent columns
    stride: u32,   // distance between columns in both buffers
    grid_x: u32,   // workgroups along x, for the 2D dispatch fold
    mode: u32,     // 0 = plain sum, 1 = squared deviation from mean
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> params: ReduceParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
// One mean per column, used only when `mode == 1`.
@group(0) @binding(3) var<storage, read> means: array<f32>;

const WG: u32 = 64u;

@compute @workgroup_size(64)
fn collapse(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let half = params.len / 2u;
    let out_len = half + (params.len & 1u);
    let t = (wid.y * params.grid_x + wid.x) * WG + lid.x;
    if (t >= out_len * params.cols) {
        return;
    }

    let col = t / out_len;
    let k = t - col * out_len;
    let off = col * params.stride;

    // `mode == 1` folds the centring and squaring into the first
    // level, matching `reduce::sum_sq_dev_into` on the host.
    var a: f32;
    var b: f32;
    if (params.mode == 1u) {
        let m = means[col];
        if (k < half) {
            let da = src[off + 2u * k] - m;
            let db = src[off + 2u * k + 1u] - m;
            a = da * da;
            b = db * db;
        } else {
            let dt = src[off + params.len - 1u] - m;
            dst[off + k] = dt * dt;
            return;
        }
    } else {
        if (k < half) {
            a = src[off + 2u * k];
            b = src[off + 2u * k + 1u];
        } else {
            dst[off + k] = src[off + params.len - 1u];
            return;
        }
    }
    dst[off + k] = a + b;
}
