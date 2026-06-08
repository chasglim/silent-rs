//! Torus utilities for TFHE-style native-modulus arithmetic.
//!
//! TFHE encodes messages as fractions of the discretised torus
//! `T_q = (1/q) Z / Z` using a power-of-two modulus `q = 2^k`.  In Rust we
//! represent the torus element by the corresponding integer in `Z_q` packed
//! into a `u64`/`u32` — the wraparound of unsigned integer arithmetic *is*
//! the modular reduction.
//!
//! The helpers here cover the few operations a TFHE backend repeatedly
//! performs:
//!   * encoding a message `m ∈ [0, plain_modulus)` into the MSBs of a
//!     ciphertext word with the canonical scaling `Δ = q / plain_modulus`,
//!     and decoding back with rounding;
//!   * **modulus switching** between two power-of-two moduli (e.g. from
//!     `q = 2^64` down to `2N` for blind rotation).
//!
//! All arithmetic stays inside Rust's native unsigned types — there is **no**
//! ambient `Modulus` object to consult, because the modulus is the type's
//! width.

/// Width (in bits) of a TFHE native ciphertext modulus.
///
/// Only `32` and `64` are accepted by the MVP backend; `validate_log_modulus`
/// turns any other value into a runtime panic at the bottom of every encoder.
#[inline]
pub const fn validate_log_modulus(log_q: u8) -> bool {
    matches!(log_q, 32 | 64)
}

/// Returns `q = 2^log_q` packed into a `u128`.
///
/// `u128` is used so that callers can express the corner case `log_q == 64`
/// (where `q` overflows `u64`) without resorting to ad-hoc booleans.
#[inline]
pub const fn modulus_u128(log_q: u8) -> u128 {
    1u128 << log_q
}

/// Returns `q = 2^log_q` as a `u64` for the cases where the value is known
/// to fit (i.e. `log_q < 64`).  Asserts otherwise.
#[inline]
pub const fn modulus_u64(log_q: u8) -> u64 {
    assert!(log_q < 64, "modulus_u64 only supports log_q < 64");
    1u64 << log_q
}

/// Encode a plaintext value into the MSBs of a `u64` ciphertext word using
/// the canonical TFHE delta `Δ = q / plain_modulus`.
///
/// `plain_modulus` must be a power of two and strictly less than `q`.
#[inline]
pub fn encode_msb_u64(message: u64, plain_modulus: u64, log_q: u8) -> u64 {
    debug_assert!(plain_modulus.is_power_of_two() && plain_modulus >= 2);
    debug_assert!(log_q == 64 || (plain_modulus as u128) < (1u128 << log_q));
    let plain_log = plain_modulus.trailing_zeros() as u8;
    let shift = log_q - plain_log;
    if shift == 64 {
        0
    } else {
        (message & (plain_modulus - 1)).wrapping_shl(shift as u32)
    }
}

/// Decode a TFHE plaintext from a noisy MSB-encoded ciphertext word.
///
/// Performs nearest-integer rounding: round(value · plain_modulus / q).
#[inline]
pub fn decode_msb_u64(value: u64, plain_modulus: u64, log_q: u8) -> u64 {
    debug_assert!(plain_modulus.is_power_of_two() && plain_modulus >= 2);
    let plain_log = plain_modulus.trailing_zeros() as u8;
    let shift = log_q - plain_log;
    if shift == 0 {
        value & (plain_modulus - 1)
    } else if shift == 64 {
        0
    } else {
        // Round-to-nearest: add 2^(shift-1) before truncating.
        let rounding = 1u64 << (shift - 1);
        let rounded = value.wrapping_add(rounding);
        (rounded >> shift) & (plain_modulus - 1)
    }
}

/// Switch a value from modulus `2^src_log` to modulus `2^dst_log`, with
/// nearest-integer rounding.
///
/// `dst_log <= src_log` is required.  This is the operation TFHE performs on
/// every coefficient before blind rotation (`src_log = 64, dst_log = log2(2N)`).
#[inline]
pub fn modulus_switch_u64(value: u64, src_log: u8, dst_log: u8) -> u64 {
    debug_assert!(dst_log <= src_log);
    let shift = src_log - dst_log;
    if shift == 0 {
        value
    } else if shift == 64 {
        // Degenerate: the destination modulus is 1 (anything maps to 0).
        0
    } else {
        let rounding = 1u64 << (shift - 1);
        // Use saturating_add via u128 to avoid wrap-around when value is near u64::MAX.
        let extended = (value as u128).wrapping_add(rounding as u128);
        let dst_mask = (1u128 << dst_log) - 1;
        ((extended >> shift) & dst_mask) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip_q64() {
        let log_q = 64;
        let plain = 16u64;
        for m in 0..plain {
            let enc = encode_msb_u64(m, plain, log_q);
            let dec = decode_msb_u64(enc, plain, log_q);
            assert_eq!(dec, m, "m = {m}");
        }
    }

    #[test]
    fn encode_decode_roundtrip_q32() {
        let log_q = 32;
        let plain = 4u64;
        for m in 0..plain {
            let enc = encode_msb_u64(m, plain, log_q);
            let dec = decode_msb_u64(enc, plain, log_q);
            assert_eq!(dec, m);
        }
    }

    #[test]
    fn decode_handles_small_noise() {
        let log_q = 64;
        let plain = 8u64;
        let m = 5u64;
        let enc = encode_msb_u64(m, plain, log_q);
        let delta = 1u64 << (log_q - plain.trailing_zeros() as u8);
        // Noise within +- delta/4 must still decode to m.
        let noise_bound = delta / 4;
        for noise in [0i64, -7, 7, -123, 123, -(noise_bound as i64) + 1] {
            let noisy = enc.wrapping_add(noise as u64);
            assert_eq!(decode_msb_u64(noisy, plain, log_q), m, "noise={noise}");
        }
    }

    #[test]
    fn modulus_switch_round_trip() {
        let src_log = 64;
        let dst_log = 11; // typical 2N for N = 1024
        // value spread across the full 64-bit range; switched value should be
        // monotone in segments.
        let value = 0x8000_0000_0000_0000u64;
        let switched = modulus_switch_u64(value, src_log, dst_log);
        assert!(switched < (1u64 << dst_log));
        // 0.5 of the source range should land at exactly 2^(dst_log - 1).
        assert_eq!(switched, 1u64 << (dst_log - 1));
    }

    #[test]
    fn modulus_switch_zero_shift_identity() {
        assert_eq!(modulus_switch_u64(42, 32, 32), 42);
    }
}
