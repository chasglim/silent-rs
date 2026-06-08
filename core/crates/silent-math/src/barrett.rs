//! PQC-specific Barrett and Montgomery reduction helpers.
//!
//! Provides pre-computed constants for the small moduli
//! used in ML-KEM (q = 3329) and ML-DSA (q = 8380417).
//!
//! These operate on narrow types and are designed for
//! constant-time use — no secret-dependent branching.
//!
//! Based on NIST FIPS 203 (ML-KEM) and FIPS 204 (ML-DSA).

// ── ML-KEM (q = 3329) ──────────────────────────────────────────────────────

/// Barrett reduction for q = 3329 (ML-KEM).
///
/// Based on FIPS 203, Algorithm 5:
/// ```text
/// t  = (value * 20159 + (1 << 25)) >> 26
/// result = value - t * 3329
/// ```
/// where 20159 = ceil(2^26 / 3329).
///
/// Returns result in [-3328, 3328].
#[inline]
pub fn barrett_reduce_mlkem(value: i16) -> i16 {
    let v = value as i32;
    let t = ((v.wrapping_mul(20159i32).wrapping_add(1i32 << 25)) >> 26) as i16;
    value.wrapping_sub(t.wrapping_mul(3329))
}

/// Montgomery reduction for q = 3329 (ML-KEM).
///
/// Based on FIPS 203, Algorithm 6:
/// ```text
/// k = (value * 62209) mod 2^16
/// result = (value - k * 3329) / 2^16
/// ```
/// where 62209 = 3329^(-1) mod 2^16.
///
/// Input `value` should be the product of two Montgomery-domain
/// elements (each up to 3329 * R in magnitude, giving a product
/// up to ~3329^2 * R^2).
#[inline]
pub fn montgomery_reduce_mlkem(value: i32) -> i16 {
    let qinv = 62209i32;
    let k = (value.wrapping_mul(qinv)) as u16 as i32; // k = value * qinv mod 2^16
    let t = k.wrapping_mul(3329) >> 16;
    (value >> 16) as i16 - t as i16
}

/// Centered reduction for ML-KEM: maps value into [0, 3328].
///
/// Equivalent to `value mod 3329` stored as unsigned representative.
#[inline]
pub fn csubq_mlkem(value: i16) -> i16 {
    // Branchless: if value < 0, add 3329.
    // Arithmetic right shift of negative value produces -1 (all 1s).
    value + ((value >> 15) & 3329)
}

// ── ML-DSA (q = 8380417) ───────────────────────────────────────────────────

/// Montgomery multiplication for ML-DSA.
///
/// q = 8380417, R = 2^32, qinv = 58728449 (q^(-1) mod 2^32).
///
/// Computes `(a * b) * R^(-1) mod 8380417`.
#[inline]
pub fn montgomery_mul_mldsa(a: i64, b: i64) -> i32 {
    let m = a.wrapping_mul(b);
    let k = m.wrapping_mul(58728449i64) as u32 as i64; // m * qinv mod 2^32
    let t = k.wrapping_mul(8380417) >> 32;
    (m >> 32) as i32 - t as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barrett_mlkem_zero() {
        assert_eq!(barrett_reduce_mlkem(0), 0);
    }

    #[test]
    fn barrett_mlkem_small_positive() {
        // Barrett reduction returns a value r such that r ≡ a (mod 3329)
        // and |r| ≤ 3328. The exact value may differ from the "naive"
        // remainder when the input is near multiples of q.
        assert_eq!(barrett_reduce_mlkem(0), 0);
        assert_eq!(barrett_reduce_mlkem(100), 100);
        // -3328 ≡ 1 (mod 3329), Barrett returns the centered remainder: 1
        assert_eq!(barrett_reduce_mlkem(-3328), 1);
        // 3328 now reduces to -1 (correctly centered)
        assert_eq!(barrett_reduce_mlkem(3328), -1);
    }

    #[test]
    fn barrett_then_csubq_gives_correct() {
        assert_eq!(csubq_mlkem(barrett_reduce_mlkem(3328)), 3328);
        assert_eq!(csubq_mlkem(barrett_reduce_mlkem(3329)), 0);
        assert_eq!(csubq_mlkem(-1), 3328);
        assert_eq!(csubq_mlkem(100), 100);
        assert_eq!(csubq_mlkem(3328), 3328);
    }

    #[test]
    fn barrett_mlkem_wraps_at_q() {
        // Barrett may give negative results for values >= q - epsilon.
        // Verify equivalence modulo q.
        let r = barrett_reduce_mlkem(3329 + 100);
        assert_eq!(csubq_mlkem(r), 100);
        let r2 = barrett_reduce_mlkem(2 * 3329 + 500);
        assert_eq!(csubq_mlkem(r2), 500);
    }

    #[test]
    fn csubq_mlkem_unsigned() {
        // csubq maps to unsigned [0, 3328].
        assert_eq!(csubq_mlkem(0), 0);
        assert_eq!(csubq_mlkem(100), 100);
        assert_eq!(csubq_mlkem(-1), 3328);
        assert_eq!(csubq_mlkem(-100), 3229);
        assert_eq!(csubq_mlkem(3328), 3328);
    }
}
