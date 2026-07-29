// Bit-exact device evaluation of the finite-state Markov family.
//
// This kernel reproduces the host arithmetic exactly, so the value it
// returns is bit-identical to `Backend::Scalar`. Three things make that
// possible where the single-precision path cannot manage it:
//
//   * the family draws no normal deviates, so nothing here needs `log`,
//     `sqrt` or `cos`, whose accuracy WGSL leaves to the driver;
//   * the two floating-point values involved are produced by exactly
//     one rounded operation each, which `f64.wgsl` emulates;
//   * the observation is an indicator, so the ensemble is a count and
//     the reduction is integer addition, which is associative and
//     therefore immune to scheduling entirely.
//
// The RNG stream is already shared with the host: `common.wgsl` builds
// the same 32-byte ChaCha20 key from the same SplitMix64 mix.

struct MarkovParams {
    n: u32,
    seed_lo: u32,
    seed_hi: u32,
    k: u32,
    start: u32,
    base_label: u32,
    grid_x: u32,
    pad0: u32,
    // `UniformFloat<f64> { low, scale }` for the intensity, computed on
    // the host by `rand` itself and uploaded as raw bits. The
    // construction involves a division and a correction loop; doing it
    // host-side means the device never has to divide.
    scale_hi: u32,
    scale_lo: u32,
    low_hi: u32,
    low_lo: u32,
}

@group(0) @binding(0) var<storage, read> params: MarkovParams;
@group(0) @binding(1) var<storage, read_write> total: atomic<u32>;

const WG: u32 = 64u;

var<workgroup> partial: array<u32, 64>;

// `rand::distributions::Standard` for f64:
//   scale = 1 / 2^53;  value = next_u64() >> 11;  scale * (value as f64)
// The multiply is by a power of two and the integer fits in 53 bits, so
// the whole thing is exact.
fn standard_f64(s: ptr<function, Stream>) -> F64 {
    let lo = next_u32(s);
    let hi = next_u32(s);
    // value = (hi:lo) >> 11
    let v_lo = (lo >> 11u) | (hi << 21u);
    let v_hi = hi >> 11u;
    return f64_from_scaled_int(v_hi, v_lo, -53);
}

// `rand::distributions::uniform::UniformFloat<f64>::sample`:
//   value1_2 = (next_u64() >> 12).into_float_with_exponent(0)
//   value0_1 = value1_2 - 1.0
//   value0_1 * scale + low
//
// `value1_2` lies in [1, 2) and `value1_2 - 1.0` is exact by Sterbenz,
// so `value0_1` is just the 52-bit mantissa scaled by 2^-52. `low` is
// zero for every built-in family, so the trailing add is the identity
// and is skipped.
fn uniform_f64(s: ptr<function, Stream>, scale: F64) -> F64 {
    let lo = next_u32(s);
    let hi = next_u32(s);
    let m_lo = (lo >> 12u) | (hi << 20u);
    let m_hi = hi >> 12u;
    let value0_1 = f64_from_scaled_int(m_hi, m_lo, -52);
    return f64_mul(value0_1, scale);
}

// `rand::Rng::gen_range(0..k)` for u32, which is
// `UniformInt::sample_single`: Lemire's multiply-shift with rejection.
// Pure integer arithmetic, so it is exact by construction.
fn gen_range_u32(s: ptr<function, Stream>, k: u32) -> u32 {
    if (k == 0u) {
        return next_u32(s);
    }
    // `gen_range(0..k)` becomes `sample_single_inclusive(0, k-1)`, whose
    // range is `k`. For 32-bit types rand takes the cheap conservative
    // zone rather than the modulus one:
    //
    //     zone = (range << range.leading_zeros()) - 1
    //
    // The modulus form is used only for i8 and i16. Getting this wrong
    // is invisible in aggregate -- both versions are unbiased -- and
    // shows up only as a bit-level mismatch, which is exactly what the
    // exactness test is for.
    let zone = (k << countLeadingZeros(k)) - 1u;
    // Assigned in the loop rather than returned from it: naga does not
    // treat a `loop` whose only exits are `return` as diverging.
    var result = 0u;
    loop {
        let v = next_u32(s);
        let prod = mul32(v, k);
        // wmul gives (lo, hi); Lemire rejects on lo and keeps hi.
        if (prod.x <= zone) {
            result = prod.y;
            break;
        }
    }
    return result;
}

@compute @workgroup_size(64)
fn draw_markov(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * params.grid_x + wid.x) * WG + lid.x;

    var hit = 0u;
    if (i < params.n) {
        var rng = stream_new(U64(params.seed_lo, params.seed_hi), U64(i, 0u));

        let scale = F64(params.scale_hi, params.scale_lo);
        let theta = uniform_f64(&rng, scale);
        let u = standard_f64(&rng);

        var label = params.start;
        if (f64_lt(u, theta)) {
            label = gen_range_u32(&rng, params.k);
        }
        if (label == params.base_label) {
            hit = 1u;
        }
    }

    // Integer tree reduction inside the workgroup, then one atomic per
    // workgroup. Addition on u32 is associative, so the total does not
    // depend on how the device schedules anything.
    partial[lid.x] = hit;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        atomicAdd(&total, partial[0]);
    }
}
