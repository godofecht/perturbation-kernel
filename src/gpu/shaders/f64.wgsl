// Software IEEE-754 binary64, enough of it to reproduce `rand` exactly.
//
// WGSL has no `f64`, which is the whole reason the single-precision
// device path cannot match the host bit for bit. For the families that
// draw only uniforms and integers, though, the host's arithmetic is a
// very short list: convert a 53-bit integer to a double, multiply two
// doubles once, compare. Emulating exactly those three operations in
// u32 pairs closes the gap completely.
//
// Everything here is exact. `f64_mul` rounds to nearest with ties to
// even, which is what the host does, and the conversion helpers are
// exact by construction because their inputs fit in 53 bits.
//
// The domain is deliberately narrow: finite, non-negative values in the
// normal range. No infinities, no NaN, no subnormals. Every value that
// reaches these functions is a uniform variate in [0, 1) or a scale
// factor derived from a finite intensity bound, so none of those cases
// can arise. `family::Family::validate` is what guarantees it.

// IEEE-754 binary64, split as the high and low halves of its bit
// pattern. `hi` carries the sign, the 11-bit exponent and the top 20
// mantissa bits.
struct F64 {
    hi: u32,
    lo: u32,
}

fn f64_zero() -> F64 {
    return F64(0u, 0u);
}

fn f64_is_zero(a: F64) -> bool {
    return ((a.hi & 0x7FFFFFFFu) | a.lo) == 0u;
}

// Ordering for non-negative finite values. Their bit patterns are
// monotone in the value, so an unsigned lexicographic compare on
// (hi, lo) is the same as comparing the numbers.
fn f64_lt(a: F64, b: F64) -> bool {
    if (a.hi != b.hi) {
        return a.hi < b.hi;
    }
    return a.lo < b.lo;
}

// Exact 32x32 -> 64 bit unsigned product.
//
// Computed on 16-bit halves. Each partial product is at most
// (2^16 - 1)^2, and each carry-in sum is bounded below 2^32, so no
// intermediate wraps.
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;

    let p00 = a0 * b0;
    let mid1 = a1 * b0 + (p00 >> 16u);
    let mid2 = a0 * b1 + (mid1 & 0xFFFFu);

    let lo = (mid2 << 16u) | (p00 & 0xFFFFu);
    let hi = a1 * b1 + (mid1 >> 16u) + (mid2 >> 16u);
    return vec2<u32>(lo, hi);
}

// Index of the highest set bit of a 64-bit value held as (hi, lo).
// Returns 0 when the value is zero; callers check for zero first.
fn msb64(hi: u32, lo: u32) -> u32 {
    if (hi != 0u) {
        return 63u - countLeadingZeros(hi);
    }
    return 31u - countLeadingZeros(lo);
}

// Exact conversion of `v * 2^e`, where `v = hi * 2^32 + lo` is a
// non-negative integer below 2^53.
//
// This is the one conversion `rand` performs on the uniform path:
// `(bits >> 11) as f64 * 2^-53` for `Standard<f64>`, and the mantissa
// of `value1_2 - 1.0` for `UniformFloat<f64>`. Both fit in 53 bits, so
// no rounding happens and the result is exact.
fn f64_from_scaled_int(v_hi: u32, v_lo: u32, e: i32) -> F64 {
    if ((v_hi | v_lo) == 0u) {
        return f64_zero();
    }
    let p = msb64(v_hi, v_lo);

    // Shift the value left so its leading one lands at bit 52, then
    // drop that leading one: it is the implicit bit.
    let s = 52u - p;
    var m_hi = v_hi;
    var m_lo = v_lo;
    if (s >= 32u) {
        m_hi = m_lo << (s - 32u);
        m_lo = 0u;
    } else if (s > 0u) {
        m_hi = (m_hi << s) | (m_lo >> (32u - s));
        m_lo = m_lo << s;
    }

    let exponent = u32(i32(p) + e + 1023);
    return F64((exponent << 20u) | (m_hi & 0x000FFFFFu), m_lo);
}

// Round-to-nearest-even product of two non-negative normal doubles.
//
// The 53-bit mantissas multiply to a 106-bit product, held here in four
// u32 limbs. Because both operands are normalised the product lands in
// [2^104, 2^106), so the result keeps either the top 53 bits starting
// at bit 104 or at bit 105, and the bits below feed the rounding
// decision.
fn f64_mul(a: F64, b: F64) -> F64 {
    if (f64_is_zero(a) || f64_is_zero(b)) {
        return f64_zero();
    }

    let ea = (a.hi >> 20u) & 0x7FFu;
    let eb = (b.hi >> 20u) & 0x7FFu;

    // Mantissas with the implicit leading one restored: 53 bits each,
    // split as a 21-bit high part and a 32-bit low part.
    let ah = (a.hi & 0x000FFFFFu) | 0x00100000u;
    let al = a.lo;
    let bh = (b.hi & 0x000FFFFFu) | 0x00100000u;
    let bl = b.lo;

    // a*b = ah*bh << 64 + (ah*bl + al*bh) << 32 + al*bl
    let t_ll = mul32(al, bl);
    let t_lh = mul32(al, bh);
    let t_hl = mul32(ah, bl);
    let t_hh = mul32(ah, bh);

    var p0 = t_ll.x;
    var p1 = t_ll.y;
    var p2 = 0u;
    var p3 = 0u;

    // Accumulate the two middle products into limb 1, carrying up.
    var carry = 0u;
    var s = p1 + t_lh.x;
    if (s < p1) { carry = carry + 1u; }
    p1 = s;
    s = p1 + t_hl.x;
    if (s < p1) { carry = carry + 1u; }
    p1 = s;

    p2 = t_lh.y + t_hl.y;
    var carry2 = 0u;
    if (p2 < t_lh.y) { carry2 = carry2 + 1u; }
    s = p2 + carry;
    if (s < p2) { carry2 = carry2 + 1u; }
    p2 = s;
    s = p2 + t_hh.x;
    if (s < p2) { carry2 = carry2 + 1u; }
    p2 = s;

    p3 = t_hh.y + carry2;

    // Bit 105 of the product is bit 9 of limb 3.
    var shift = 52u;
    if ((p3 & 0x00000200u) != 0u) {
        shift = 53u;
    }

    // Extract the 53-bit result mantissa, the guard bit just below it,
    // and a sticky bit summarising everything below that.
    var m_hi = 0u;
    var m_lo = 0u;
    var guard = 0u;
    var sticky = 0u;

    // `shift` is 52 or 53, so every shift distance below lands strictly
    // inside 1..31 and no shift-by-32 edge case arises.
    let sh = shift;
    m_lo = (p1 >> (sh - 32u)) | (p2 << (64u - sh));
    m_hi = (p2 >> (sh - 32u)) | (p3 << (64u - sh));

    let gpos = sh - 1u;               // 51 or 52, always inside limb 1
    guard = (p1 >> (gpos - 32u)) & 1u;

    // Sticky: any set bit strictly below the guard bit.
    let below = gpos - 32u;           // 19 or 20 bits of limb 1
    let mask = (1u << below) - 1u;
    if (((p1 & mask) | p0) != 0u) {
        sticky = 1u;
    }

    var exponent = i32(ea) + i32(eb) - 1023 + i32(shift) - 52;

    // Round to nearest, ties to even.
    if (guard == 1u && (sticky == 1u || (m_lo & 1u) == 1u)) {
        let old = m_lo;
        m_lo = m_lo + 1u;
        if (m_lo < old) {
            m_hi = m_hi + 1u;
        }
        // Carrying out of bit 52 means the mantissa became 2^53.
        if ((m_hi & 0x00200000u) != 0u) {
            m_lo = (m_lo >> 1u) | (m_hi << 31u);
            m_hi = m_hi >> 1u;
            exponent = exponent + 1;
        }
    }

    return F64((u32(exponent) << 20u) | (m_hi & 0x000FFFFFu), m_lo);
}
