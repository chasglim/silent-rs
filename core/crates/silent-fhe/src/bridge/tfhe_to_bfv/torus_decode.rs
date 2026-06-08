use super::bfv_helpers::{
    add_constant_bfv, scalar_mul_bfv, shift_ciphertext_by_monomial, trivial_constant_bfv,
};
use super::fmod::{decode_torus_exact, decode_torus_native_exact};
use crate::bridge::noise::{NoiseBound, check_torus_decode_margin};
use crate::bridge::{BridgeError, BridgeKeys, BridgeParams};
use crate::core::evaluator::HeEvaluator;
use crate::schemes::bfv::ops::BfvEvaluator;
use silent_rlwe::{Ciphertext, LweCiphertext};

/// Center-only Fmod / TorusDecode.
///
/// This backend is exact only when each encrypted phase is already on a TFHE
/// encoding center `Δ_T * m`. It does not implement interval rounding for
/// noisy TFHE phases. The full public converter uses a separate exact LWE1
/// zero-noise backend and rejects all other TFHE parameter sets.
pub fn bfv_fmod_torus_decode(
    phase_ct: &Ciphertext,
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    ensure_torus_decode_preconditions(params)?;

    if keys.fmod_keys.coeffs.is_empty() {
        return Err(BridgeError::IncompatibleParameters(
            "BFV Fmod keys are empty; decoding polynomial must be precomputed".into(),
        ));
    }

    let scalar = keys.fmod_keys.center_decode_scalar.ok_or_else(|| {
        BridgeError::IncompatibleParameters(
            "BFV Fmod center decode scalar is missing; Fmod keys were not generated for these parameters"
                .into(),
        )
    })?;
    Ok(scalar_mul_bfv(phase_ct, scalar, params))
}

/// Exact zero-noise decoder for one-dimensional LWE inputs.
///
/// For `n=1`, the decoded message is the public Boolean affine function
/// `F(s) = F(0) + (F(1)-F(0))*s`, where
/// `F(s) = TorusDecode(b - a*s mod q_T)`. The bridge key already contains a
/// BFV encryption of the secret bit `s`, so this backend is linear and avoids
/// the high-coefficient periodic Fmod polynomial.
pub fn bfv_decode_lwe1_zero_noise_to_coefficients(
    tfhe_cts: &[LweCiphertext],
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    ensure_lwe1_direct_decode_preconditions(params)?;
    if tfhe_cts.is_empty() {
        return Err(BridgeError::IncompatibleParameters(
            "LWE1 TFHE->BFV decode requires at least one TFHE ciphertext".into(),
        ));
    }
    if tfhe_cts.len() > params.num_slots {
        return Err(BridgeError::IncompatibleParameters(format!(
            "LWE1 TFHE->BFV decode received {} ciphertexts but bridge num_slots is {}",
            tfhe_cts.len(),
            params.num_slots
        )));
    }
    let secret_bit_ct = keys.tfhe_to_bfv.rk.first().ok_or_else(|| {
        BridgeError::IncompatibleParameters("TFHE->BFV repacking key is empty".into())
    })?;

    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let q_mask = (1u64 << log_q) - 1;
    let t = params.bfv_params.plain_modulus();
    let evaluator = BfvEvaluator::new(params.bridge_bfv_eval_params());
    let mut packed = trivial_constant_bfv(params, 0);
    let mut has_packed = false;

    for (index, lwe) in tfhe_cts.iter().enumerate() {
        if lwe.dimension().0 != 1 {
            return Err(BridgeError::IncompatibleParameters(format!(
                "LWE1 TFHE->BFV decode expected LWE dimension 1, got {}",
                lwe.dimension().0
            )));
        }
        if lwe.log_modulus() != log_q {
            return Err(BridgeError::IncompatibleParameters(format!(
                "TFHE ciphertext has log modulus {}, expected {}",
                lwe.log_modulus(),
                log_q
            )));
        }

        let body = lwe.body() & q_mask;
        let mask = lwe.mask()[0] & q_mask;
        let decode_if_zero = decode_torus_exact(body, params) % t;
        let phase_if_one = body.wrapping_sub(mask) & q_mask;
        let decode_if_one = decode_torus_exact(phase_if_one, params) % t;
        let slope = if decode_if_one >= decode_if_zero {
            decode_if_one - decode_if_zero
        } else {
            t - (decode_if_zero - decode_if_one)
        };

        let mut decoded = scalar_mul_bfv(secret_bit_ct, slope, params);
        if decode_if_zero != 0 {
            decoded = add_constant_bfv(&decoded, decode_if_zero, params);
        }
        let shifted = shift_ciphertext_by_monomial(&decoded, index, params);
        if has_packed {
            packed = evaluator.add(&packed, &shifted);
        } else {
            packed = shifted;
            has_packed = true;
        }
    }

    Ok(packed)
}

/// Exact LWE1 decoder for torus-native `1/p` torus-native inputs.
///
/// This is a deliberately narrow public baseline for TFHE -> BFV. It assumes
/// each TLWE phase is already normalized by TFHE PBS/functional bootstrap to
/// an exact BFV plaintext lattice point `z/p in T`; it then evaluates the
/// one-bit affine decryption table under BFV and packs the resulting `z mod p`
/// coefficients.
pub fn bfv_decode_lwe1_torus_native_to_coefficients(
    tfhe_cts: &[LweCiphertext],
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    ensure_lwe1_torus_native_preconditions(params)?;
    if tfhe_cts.is_empty() {
        return Err(BridgeError::IncompatibleParameters(
            "LWE1 torus-native TFHE->BFV decode requires at least one TFHE ciphertext".into(),
        ));
    }
    if tfhe_cts.len() > params.num_slots {
        return Err(BridgeError::IncompatibleParameters(format!(
            "LWE1 torus-native TFHE->BFV decode received {} ciphertexts but bridge num_slots is {}",
            tfhe_cts.len(),
            params.num_slots
        )));
    }
    let secret_bit_ct = keys.tfhe_to_bfv.rk.first().ok_or_else(|| {
        BridgeError::IncompatibleParameters("TFHE->BFV repacking key is empty".into())
    })?;

    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let evaluator = BfvEvaluator::new(params.bridge_bfv_eval_params());
    let mut packed = trivial_constant_bfv(params, 0);
    let mut has_packed = false;

    for (index, lwe) in tfhe_cts.iter().enumerate() {
        if lwe.dimension().0 != 1 {
            return Err(BridgeError::IncompatibleParameters(format!(
                "LWE1 torus-native TFHE->BFV decode expected LWE dimension 1, got {}",
                lwe.dimension().0
            )));
        }
        if lwe.log_modulus() != log_q {
            return Err(BridgeError::IncompatibleParameters(format!(
                "TFHE ciphertext has log modulus {}, expected {}",
                lwe.log_modulus(),
                log_q
            )));
        }

        let body = reduce_torus(lwe.body(), log_q);
        let mask = reduce_torus(lwe.mask()[0], log_q);
        let decode_if_zero = decode_torus_native_exact(body, params)?;
        let phase_if_one = torus_sub(body, mask, log_q);
        let decode_if_one = decode_torus_native_exact(phase_if_one, params)?;
        let t = params.bfv_params.plain_modulus();
        let slope = if decode_if_one >= decode_if_zero {
            decode_if_one - decode_if_zero
        } else {
            t - (decode_if_zero - decode_if_one)
        };

        let mut decoded = scalar_mul_bfv(secret_bit_ct, slope, params);
        if decode_if_zero != 0 {
            decoded = add_constant_bfv(&decoded, decode_if_zero, params);
        }
        let shifted = shift_ciphertext_by_monomial(&decoded, index, params);
        if has_packed {
            packed = evaluator.add(&packed, &shifted);
        } else {
            packed = shifted;
            has_packed = true;
        }
    }

    Ok(packed)
}

pub fn ensure_lwe1_direct_decode_preconditions(params: &BridgeParams) -> Result<(), BridgeError> {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    if log_q >= 64 {
        return Err(BridgeError::IncompatibleParameters(
            "direct LWE1 TFHE->BFV decode requires TFHE torus modulus to fit in u64".into(),
        ));
    }
    if params.tfhe_params.lwe_dimension().0 != 1 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "direct exact TFHE->BFV decode currently requires TFHE LWE dimension 1, got {}",
            params.tfhe_params.lwe_dimension().0
        )));
    }
    if params.bfv_params.plain_modulus() <= params.message_modulus() {
        return Err(BridgeError::IncompatibleParameters(format!(
            "direct LWE1 TFHE->BFV decode requires BFV plaintext modulus > message modulus {}, got {}",
            params.message_modulus(),
            params.bfv_params.plain_modulus()
        )));
    }
    Ok(())
}

pub fn ensure_lwe1_torus_native_preconditions(params: &BridgeParams) -> Result<(), BridgeError> {
    if params.tfhe_params.ciphertext_modulus_log().0 > 64 {
        return Err(BridgeError::IncompatibleParameters(
            "torus-native LWE1 TFHE->BFV decode requires TFHE torus modulus to fit in u64".into(),
        ));
    }
    if params.tfhe_params.lwe_dimension().0 != 1 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "torus-native exact TFHE->BFV decode currently requires TFHE LWE dimension 1, got {}. Higher-dimensional inputs require SILENT functional switching with TRGSW keys",
            params.tfhe_params.lwe_dimension().0
        )));
    }
    if params.bfv_params.plain_modulus() <= 1 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "torus-native LWE1 TFHE->BFV decode requires BFV plaintext modulus p > 1, got {}",
            params.bfv_params.plain_modulus()
        )));
    }
    Ok(())
}

fn reduce_torus(value: u64, log_q: u8) -> u64 {
    if log_q == 64 {
        value
    } else {
        value & ((1u64 << log_q) - 1)
    }
}

fn torus_sub(lhs: u64, rhs: u64, log_q: u8) -> u64 {
    reduce_torus(lhs.wrapping_sub(rhs), log_q)
}

pub fn ensure_torus_decode_preconditions(params: &BridgeParams) -> Result<(), BridgeError> {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0 as u32;
    if log_q >= 64 {
        return Err(BridgeError::IncompatibleParameters(
            "BFV Fmod/TorusDecode for q=2^64 requires CRT/plaintext lifting".into(),
        ));
    }
    let q_t = 1u128
        .checked_shl(log_q)
        .ok_or_else(|| BridgeError::IncompatibleParameters("torus modulus exceeds u128".into()))?;
    let plain_modulus = params.bfv_params.plain_modulus() as u128;
    if plain_modulus <= q_t {
        return Err(BridgeError::IncompatibleParameters(format!(
            "BFV plaintext modulus {} is too small to hold torus phases modulo 2^{}; a CRT/plaintext-lifting backend is required",
            plain_modulus, log_q
        )));
    }
    if !silent_math::numth::is_prime(params.bfv_params.plain_modulus()) {
        return Err(BridgeError::IncompatibleParameters(format!(
            "center-only BFV Fmod requires prime BFV plaintext modulus, got {}",
            params.bfv_params.plain_modulus()
        )));
    }
    if params.message_bits > 12 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "center-only BFV Fmod is limited to message_bits <= 12, got {}",
            params.message_bits
        )));
    }
    check_torus_decode_margin(params, &NoiseBound::new(0))?;
    Ok(())
}

pub fn decode_torus_scalar_for_test(phase: u64, params: &BridgeParams) -> Result<u64, BridgeError> {
    ensure_torus_decode_preconditions_for_scalar(params)?;
    Ok(decode_torus_exact(phase, params))
}

fn ensure_torus_decode_preconditions_for_scalar(params: &BridgeParams) -> Result<(), BridgeError> {
    if params.message_modulus() == 0 {
        return Err(BridgeError::IncompatibleParameters(
            "message modulus must be non-zero".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemes::bfv::params::BfvParameters;
    use crate::schemes::tfhe::params::TfheParameters;
    use silent_params::presets;

    #[test]
    fn torus_decode_preconditions_reject_small_plain_modulus() {
        let bfv = BfvParameters::new(presets::toy::toy_bfv_1024()).unwrap();
        let tfhe = TfheParameters::new(presets::toy::toy_tfhe_n512()).unwrap();
        let params = crate::bridge::BridgeParams::new(
            bfv,
            tfhe,
            4,
            8,
            crate::bridge::BridgeLayout::Coefficient,
        )
        .unwrap();
        let err = ensure_torus_decode_preconditions(&params).unwrap_err();
        assert!(matches!(err, BridgeError::IncompatibleParameters(_)));
    }
}
