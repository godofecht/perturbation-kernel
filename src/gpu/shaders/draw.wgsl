// Ensemble generation: one invocation per draw index `i`.
//
// Writes the observation `Y_i = F(P(s, theta_i))` into `out`, stored
// column-major so each observation coordinate is a contiguous run of
// `n` floats and the reduction kernel can treat it as one column.

struct DrawParams {
    n: u32,
    seed_lo: u32,
    seed_hi: u32,
    family: u32,      // 0 = gaussian, 1 = bistable, 2 = markov
    d: u32,           // observation dimension
    k: u32,           // markov alphabet size
    start: u32,       // markov start label
    base_label: u32,  // markov measured label
    grid_x: u32,      // workgroups along x, for the 2D dispatch fold
    pad0: u32,
    pad1: u32,
    pad2: u32,
    p_intensity: f32, // sigma_max / theta_max
    p_dt: f32,        // bistable Euler-Maruyama step
    p_x0: f32,        // bistable initial position
    pad3: f32,
}

@group(0) @binding(0) var<storage, read> params: DrawParams;
@group(0) @binding(1) var<storage, read> base: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const WG: u32 = 64u;

@compute @workgroup_size(64)
fn draw(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Fold the 2D workgroup grid back to a linear draw index. A 1D
    // dispatch would cap out at 65535 workgroups, i.e. ~4.2M draws.
    let i = (wid.y * params.grid_x + wid.x) * WG + lid.x;
    if (i >= params.n) {
        return;
    }

    var rng = stream_new(U64(params.seed_lo, params.seed_hi), U64(i, 0u));
    let theta = next_f32(&rng) * params.p_intensity;

    switch params.family {
        // Gaussian shift: S' = s + theta * N(0, I), identity readout.
        case 0u: {
            for (var j = 0u; j < params.d; j = j + 1u) {
                out[j * params.n + i] = base[j] + theta * next_normal(&rng);
            }
        }
        // Bistable marble: one Euler-Maruyama step in V(x) = (x^2-1)^2,
        // sign-of-well readout.
        case 1u: {
            let x = params.p_x0;
            let drift = -4.0 * x * (x * x - 1.0);
            var noise = 0.0;
            if (theta != 0.0) {
                noise = next_normal(&rng) * theta * sqrt(params.p_dt);
            }
            let xn = x + drift * params.p_dt + noise;
            out[i] = select(-1.0, 1.0, xn >= 0.0);
        }
        // Markov chain: mix to uniform with probability theta,
        // indicator readout on `base_label`.
        default: {
            let u = next_f32(&rng);
            var label = params.start;
            if (u < theta) {
                label = min(u32(next_f32(&rng) * f32(params.k)), params.k - 1u);
            }
            out[i] = select(0.0, 1.0, label == params.base_label);
        }
    }
}
