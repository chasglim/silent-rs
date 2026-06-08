//! Torus-to-integer decoding (small-message exact variant).
//!
//! This is the "route B" from b2t.md: discrete exact decoding for small
//! message spaces, suitable for correctness testing.

use crate::bridge::{BridgeError, BridgeParams};

/// Torus-to-integer decoding for small message spaces.
///
/// Given a torus phase `phase = Δ_T · m + noise (mod q_TFHE)`,
/// recover the full TFHE plaintext slot `m ∈ [0, M)` where
/// `M = message_modulus * carry_modulus = 2^message_bits` and
/// `Δ_T = q_TFHE / (2M)`.
///
/// This uses deterministic rounding with a noise margin assumption.
/// It is exact when the noise is less than `Δ_T / 2`.
pub fn decode_torus_exact(phase: u64, params: &BridgeParams) -> u64 {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let padded_bits = params.message_bits + 1;
    let shift = if log_q as usize >= padded_bits {
        log_q as u32 - padded_bits as u32
    } else {
        0
    };

    let delta = if shift >= 64 { 0 } else { 1u64 << shift };

    let rounded = if delta == 0 {
        0
    } else {
        (phase.wrapping_add(delta >> 1)) >> shift
    };

    let padded_mask = params.padded_message_modulus() - 1;
    (rounded & padded_mask) % params.message_modulus()
}

/// Decode an exact torus-native torus-native BFV message.
///
/// The input phase is interpreted as a noisy representative of `z / p in T`,
/// where `p` is the BFV plaintext modulus. This returns the nearest `z mod p`.
pub fn decode_torus_native_exact(phase: u64, params: &BridgeParams) -> Result<u64, BridgeError> {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let q_t = 1u128
        .checked_shl(log_q as u32)
        .ok_or_else(|| BridgeError::IncompatibleParameters("torus modulus exceeds u128".into()))?;
    let p = params.bfv_params.plain_modulus() as u128;
    if p == 0 {
        return Err(BridgeError::IncompatibleParameters(
            "BFV plaintext modulus must be non-zero".into(),
        ));
    }
    let rounded = ((phase as u128) * p + (q_t / 2)) / q_t;
    Ok((rounded % p) as u64)
}

/// Encode an integer message to torus representation.
pub fn encode_torus(message: u64, params: &BridgeParams) -> u64 {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let padded_bits = params.message_bits + 1;
    let shift = if log_q as usize >= padded_bits {
        log_q as u32 - padded_bits as u32
    } else {
        0
    };
    let m = message & (params.message_modulus() - 1);
    if shift >= 64 { 0 } else { m << shift }
}

pub(crate) fn center_decode_scalar(params: &BridgeParams) -> Result<u64, BridgeError> {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    if log_q >= 64 {
        return Err(BridgeError::IncompatibleParameters(
            "center-only Fmod decode scalar requires TFHE torus modulus to fit in u64".into(),
        ));
    }

    let q_t = 1u64 << log_q;
    let delta = q_t / params.padded_message_modulus();
    silent_math::numth::mod_inverse(
        delta % params.bfv_params.plain_modulus(),
        params.bfv_params.plain_modulus(),
    )
    .ok_or_else(|| {
        BridgeError::IncompatibleParameters(format!(
            "TFHE delta {} is not invertible modulo BFV plaintext modulus {}",
            delta,
            params.bfv_params.plain_modulus()
        ))
    })
}

/// Decode a signed message from torus.
/// Maps `m ∈ [0, M)` to `[-M/2, M/2)`.
pub fn decode_torus_signed(phase: u64, params: &BridgeParams) -> i64 {
    let m = decode_torus_exact(phase, params);
    let half = params.message_modulus() >> 1;
    if m >= half {
        m as i64 - params.message_modulus() as i64
    } else {
        m as i64
    }
}
