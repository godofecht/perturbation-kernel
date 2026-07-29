// Test-only: dump the device's intermediate values for draw `i`.
//
// Used by `tests/gpu.rs` to pin the emulated-f64 uniform path against
// the host directly, rather than inferring a mismatch from an aggregate
// that happens to differ. Each index writes four words: the bits of
// `theta`, then the bits of `u`.

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

@group(0) @binding(0) var<storage, read> dparams: DebugParams;
@group(0) @binding(1) var<storage, read_write> dout: array<u32>;

const DWG: u32 = 64u;

@compute @workgroup_size(64)
fn debug_uniform(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * dparams.grid_x + wid.x) * DWG + lid.x;
    if (i >= dparams.n) {
        return;
    }
    var rng = stream_new(U64(dparams.seed_lo, dparams.seed_hi), U64(i, 0u));
    // Raw keystream words, so a divergence can be located precisely
    // rather than inferred from a derived value.
    for (var w = 0u; w < 8u; w = w + 1u) {
        dout[i * 8u + w] = next_u32(&rng);
    }
}
