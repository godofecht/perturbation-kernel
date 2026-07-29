// 64-bit integer emulation and ChaCha20, shared by the device kernels.
//
// WGSL has no 64-bit integer type, but the per-index substream fork of
// SCHEMA §8 D2 is defined by 64-bit SplitMix64 arithmetic. Emulating it
// on a u32 pair is exact: every operation below is wrapping unsigned
// arithmetic, which WGSL defines, so the device derives byte-identical
// ChaCha20 keys to `engine::fork_rng` on the host.

struct U64 {
    lo: u32,
    hi: u32,
}

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(lo, a.hi + b.hi + carry);
}

fn u64_xor(a: U64, b: U64) -> U64 {
    return U64(a.lo ^ b.lo, a.hi ^ b.hi);
}

// Logical right shift by `n` in 1..=31.
fn u64_shr(a: U64, n: u32) -> U64 {
    return U64((a.lo >> n) | (a.hi << (32u - n)), a.hi >> n);
}

// Low 64 bits of the product. The 32x32 -> 64 core is computed on
// 16-bit halves; each partial product and each carry-in sum is bounded
// below 2^32, so no intermediate wraps unintentionally.
fn u64_mul(a: U64, b: U64) -> U64 {
    let a0 = a.lo & 0xFFFFu;
    let a1 = a.lo >> 16u;
    let b0 = b.lo & 0xFFFFu;
    let b1 = b.lo >> 16u;

    let lo_lo = a0 * b0;
    let mid1 = a1 * b0 + (lo_lo >> 16u);
    let mid2 = a0 * b1 + (mid1 & 0xFFFFu);

    let lo = (mid2 << 16u) | (lo_lo & 0xFFFFu);
    let hi = a1 * b1 + (mid1 >> 16u) + (mid2 >> 16u) + a.lo * b.hi + a.hi * b.lo;
    return U64(lo, hi);
}

// SplitMix64 finaliser of (seed, i), matching `engine::fork_rng`.
fn mix64(seed: U64, i: U64) -> U64 {
    var z = u64_add(seed, u64_mul(i, U64(0x7F4A7C15u, 0x9E3779B9u)));
    z = u64_mul(u64_xor(z, u64_shr(z, 30u)), U64(0x1CE4E5B9u, 0xBF58476Du));
    z = u64_mul(u64_xor(z, u64_shr(z, 27u)), U64(0x133111EBu, 0x94D049BBu));
    return u64_xor(z, u64_shr(z, 31u));
}

// ---------------------------------------------------------------------
// ChaCha20
// ---------------------------------------------------------------------

fn rotl(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

// A counter-based ChaCha20 keystream reader.
//
// `key` is the 32-byte seed as eight little-endian words, exactly the
// layout `Rng::from_seed` receives on the host: mix64 output, then the
// config seed, then the draw index, then zero. `word` is the index into
// the keystream; blocks are generated on demand and cached.
struct Stream {
    key: array<u32, 8>,
    block: array<u32, 16>,
    block_index: u32,
    word: u32,
    loaded: u32,
}

fn stream_new(seed: U64, index: U64) -> Stream {
    let z = mix64(seed, index);
    var s: Stream;
    s.key[0] = z.lo;
    s.key[1] = z.hi;
    s.key[2] = seed.lo;
    s.key[3] = seed.hi;
    s.key[4] = index.lo;
    s.key[5] = index.hi;
    s.key[6] = 0u;
    s.key[7] = 0u;
    s.block_index = 0u;
    s.word = 0u;
    s.loaded = 0u;
    return s;
}

fn chacha_block(s: ptr<function, Stream>, counter: u32) {
    // Every index into `x` and `w` below is a literal, deliberately.
    //
    // These are function-local arrays, which HLSL wants to keep in
    // registers, and registers are not addressable. FXC -- the compiler
    // wgpu's DX12 backend uses on Windows -- refuses a dynamically
    // indexed write to one: "array reference cannot be used as an
    // l-value; not natively addressable". Writing the loops out is the
    // portable answer, and costs nothing since the trip counts are
    // fixed anyway.
    var x: array<u32, 16>;
    x[0] = 0x61707865u;
    x[1] = 0x3320646eu;
    x[2] = 0x79622d32u;
    x[3] = 0x6b206574u;
    x[4] = (*s).key[0];
    x[5] = (*s).key[1];
    x[6] = (*s).key[2];
    x[7] = (*s).key[3];
    x[8] = (*s).key[4];
    x[9] = (*s).key[5];
    x[10] = (*s).key[6];
    x[11] = (*s).key[7];
    x[12] = counter;
    x[13] = 0u;
    x[14] = 0u;
    x[15] = 0u;

    var w = x;
    for (var round = 0u; round < 10u; round = round + 1u) {
        // Column rounds.
        w[0] = w[0] + w[4]; w[12] = rotl(w[12] ^ w[0], 16u);
        w[8] = w[8] + w[12]; w[4] = rotl(w[4] ^ w[8], 12u);
        w[0] = w[0] + w[4]; w[12] = rotl(w[12] ^ w[0], 8u);
        w[8] = w[8] + w[12]; w[4] = rotl(w[4] ^ w[8], 7u);

        w[1] = w[1] + w[5]; w[13] = rotl(w[13] ^ w[1], 16u);
        w[9] = w[9] + w[13]; w[5] = rotl(w[5] ^ w[9], 12u);
        w[1] = w[1] + w[5]; w[13] = rotl(w[13] ^ w[1], 8u);
        w[9] = w[9] + w[13]; w[5] = rotl(w[5] ^ w[9], 7u);

        w[2] = w[2] + w[6]; w[14] = rotl(w[14] ^ w[2], 16u);
        w[10] = w[10] + w[14]; w[6] = rotl(w[6] ^ w[10], 12u);
        w[2] = w[2] + w[6]; w[14] = rotl(w[14] ^ w[2], 8u);
        w[10] = w[10] + w[14]; w[6] = rotl(w[6] ^ w[10], 7u);

        w[3] = w[3] + w[7]; w[15] = rotl(w[15] ^ w[3], 16u);
        w[11] = w[11] + w[15]; w[7] = rotl(w[7] ^ w[11], 12u);
        w[3] = w[3] + w[7]; w[15] = rotl(w[15] ^ w[3], 8u);
        w[11] = w[11] + w[15]; w[7] = rotl(w[7] ^ w[11], 7u);

        // Diagonal rounds.
        w[0] = w[0] + w[5]; w[15] = rotl(w[15] ^ w[0], 16u);
        w[10] = w[10] + w[15]; w[5] = rotl(w[5] ^ w[10], 12u);
        w[0] = w[0] + w[5]; w[15] = rotl(w[15] ^ w[0], 8u);
        w[10] = w[10] + w[15]; w[5] = rotl(w[5] ^ w[10], 7u);

        w[1] = w[1] + w[6]; w[12] = rotl(w[12] ^ w[1], 16u);
        w[11] = w[11] + w[12]; w[6] = rotl(w[6] ^ w[11], 12u);
        w[1] = w[1] + w[6]; w[12] = rotl(w[12] ^ w[1], 8u);
        w[11] = w[11] + w[12]; w[6] = rotl(w[6] ^ w[11], 7u);

        w[2] = w[2] + w[7]; w[13] = rotl(w[13] ^ w[2], 16u);
        w[8] = w[8] + w[13]; w[7] = rotl(w[7] ^ w[8], 12u);
        w[2] = w[2] + w[7]; w[13] = rotl(w[13] ^ w[2], 8u);
        w[8] = w[8] + w[13]; w[7] = rotl(w[7] ^ w[8], 7u);

        w[3] = w[3] + w[4]; w[14] = rotl(w[14] ^ w[3], 16u);
        w[9] = w[9] + w[14]; w[4] = rotl(w[4] ^ w[9], 12u);
        w[3] = w[3] + w[4]; w[14] = rotl(w[14] ^ w[3], 8u);
        w[9] = w[9] + w[14]; w[4] = rotl(w[4] ^ w[9], 7u);
    }

    (*s).block[0] = w[0] + x[0];
    (*s).block[1] = w[1] + x[1];
    (*s).block[2] = w[2] + x[2];
    (*s).block[3] = w[3] + x[3];
    (*s).block[4] = w[4] + x[4];
    (*s).block[5] = w[5] + x[5];
    (*s).block[6] = w[6] + x[6];
    (*s).block[7] = w[7] + x[7];
    (*s).block[8] = w[8] + x[8];
    (*s).block[9] = w[9] + x[9];
    (*s).block[10] = w[10] + x[10];
    (*s).block[11] = w[11] + x[11];
    (*s).block[12] = w[12] + x[12];
    (*s).block[13] = w[13] + x[13];
    (*s).block[14] = w[14] + x[14];
    (*s).block[15] = w[15] + x[15];
    (*s).block_index = counter;
    (*s).loaded = 1u;
}

fn next_u32(s: ptr<function, Stream>) -> u32 {
    let counter = (*s).word / 16u;
    if ((*s).loaded == 0u || (*s).block_index != counter) {
        chacha_block(s, counter);
    }
    // A dynamic *read* of a function-local array, which FXC handles by
    // unrolling into a select chain. Written as an explicit chain so
    // the cost is visible and the compiler has nothing to refuse.
    let i = (*s).word % 16u;
    var out = (*s).block[0];
    if (i == 1u) { out = (*s).block[1]; }
    else if (i == 2u) { out = (*s).block[2]; }
    else if (i == 3u) { out = (*s).block[3]; }
    else if (i == 4u) { out = (*s).block[4]; }
    else if (i == 5u) { out = (*s).block[5]; }
    else if (i == 6u) { out = (*s).block[6]; }
    else if (i == 7u) { out = (*s).block[7]; }
    else if (i == 8u) { out = (*s).block[8]; }
    else if (i == 9u) { out = (*s).block[9]; }
    else if (i == 10u) { out = (*s).block[10]; }
    else if (i == 11u) { out = (*s).block[11]; }
    else if (i == 12u) { out = (*s).block[12]; }
    else if (i == 13u) { out = (*s).block[13]; }
    else if (i == 14u) { out = (*s).block[14]; }
    else if (i == 15u) { out = (*s).block[15]; }
    (*s).word = (*s).word + 1u;
    return out;
}

// Uniform on [0, 1) with 24 bits of precision -- the most an f32
// mantissa can carry without gaps.
fn next_f32(s: ptr<function, Stream>) -> f32 {
    return f32(next_u32(s) >> 8u) * (1.0 / 16777216.0);
}

// Standard normal by Box-Muller. Uses `log`, `sqrt` and `cos`, whose
// precision WGSL leaves to the driver; this is the one place where two
// different devices may disagree in the last ulp.
fn next_normal(s: ptr<function, Stream>) -> f32 {
    var u1 = next_f32(s);
    let u2 = next_f32(s);
    // Guard the log: u1 == 0 has probability 2^-24 but is not
    // impossible, and log(0) is -inf.
    u1 = max(u1, 1.0 / 16777216.0);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
