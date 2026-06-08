use num_bigint::BigUint;
use num_traits::{FromPrimitive, ToPrimitive};
use silent_math::rns::RnsContext;
use silent_params::CiphertextModulusLog;
use silent_rlwe::LweCiphertext;

use crate::bridge::BridgeError;
use crate::bridge::BridgeParams;
use crate::bridge::bfv_to_tfhe::extract::ExtractedBfvLwe;

/// Modulus switch BFV extracted LWE from `Q_BFV` to TFHE torus modulus `q_T`.
pub fn modswitch_bfv_lwe_to_tfhe(
    lwe_q: &ExtractedBfvLwe,
    params: &BridgeParams,
) -> Result<LweCiphertext, BridgeError> {
    let base_q = params.bfv_params.ring().rns().clone();
    let rns_ctx = RnsContext::new(base_q);
    let q_bfv = rns_ctx.base_prod_u128().ok_or_else(|| {
        BridgeError::IncompatibleParameters(
            "BFV->TFHE modswitch requires ciphertext modulus product to fit in u128".into(),
        )
    })?;
    let log_q_t = params.tfhe_params.ciphertext_modulus_log().0;
    let q_t = 1u128.checked_shl(log_q_t as u32).ok_or_else(|| {
        BridgeError::IncompatibleParameters("TFHE ciphertext modulus exceeds u128".into())
    })?;
    let padded_plaintext_modulus = params
        .tfhe_params
        .total_message_modulus()
        .checked_mul(2)
        .ok_or_else(|| {
            BridgeError::IncompatibleParameters(
                "TFHE padded plaintext modulus overflows u64".into(),
            )
        })?;
    let plain_modulus = params.bfv_params.plain_modulus() as u128;
    if plain_modulus < params.message_modulus() as u128 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "BFV->TFHE requires BFV plaintext modulus t={} to cover message modulus {}",
            params.bfv_params.plain_modulus(),
            params.message_modulus()
        )));
    }
    let count = lwe_q.dimension() + 1;
    let size_q = lwe_q.residues_q().len() / count;
    let mut data = vec![0u64; count];
    let q_bfv_big = BigUint::from_u128(q_bfv).unwrap();
    let q_t_big = BigUint::from_u128(q_t).unwrap();
    let plain_modulus_big = BigUint::from_u128(plain_modulus).unwrap();
    let denominator = &q_bfv_big * BigUint::from_u128(padded_plaintext_modulus.into()).unwrap();
    let numerator_scale = &q_t_big * &plain_modulus_big;
    let half_denominator = &denominator >> 1;

    for coeff in 0..count {
        let mut residues = vec![0u64; size_q];
        for limb in 0..size_q {
            residues[limb] = lwe_q.residues_q()[limb * count + coeff];
        }
        let lifted = rns_ctx.compose_u128(&residues).map_err(|e| {
            BridgeError::IncompatibleParameters(format!(
                "BFV->TFHE CRT reconstruction failed: {e:?}"
            ))
        })?;

        let numerator = BigUint::from_u128(lifted).unwrap() * &numerator_scale + &half_denominator;
        let quotient: BigUint = numerator / &denominator;
        let mut result_coeff = quotient.to_u64().ok_or_else(|| {
            BridgeError::IncompatibleParameters(
                "BFV->TFHE modswitch quotient does not fit in u64".into(),
            )
        })?;

        // Modulo q_t reduction. For q_t = 2^64, every u64 is already canonical.
        if log_q_t < 64 {
            let modulus = 1u64 << log_q_t;
            result_coeff %= modulus;
        }
        data[coeff] = result_coeff;
    }

    let mut out = LweCiphertext::from_data(
        data,
        CiphertextModulusLog(params.tfhe_params.ciphertext_modulus_log().0),
    );
    out.reduce();
    Ok(out)
}

/// Modulus switch a BFV extracted LWE into torus-native torus-native TFHE.
///
/// This keeps the BFV plaintext contract `z in Z_p` as the torus point `z/p`
/// instead of re-encoding it into SILENT TFHE's padded integer encoder. If
/// the BFV extracted phase is close to `(Q_BFV / p) * z`, the returned LWE phase
/// is close to `(q_T / p) * z`.
pub fn modswitch_bfv_lwe_to_torus_native_tfhe(
    lwe_q: &ExtractedBfvLwe,
    params: &BridgeParams,
) -> Result<LweCiphertext, BridgeError> {
    let base_q = params.bfv_params.ring().rns().clone();
    let rns_ctx = RnsContext::new(base_q);
    let q_bfv = rns_ctx.base_prod_u128().ok_or_else(|| {
        BridgeError::IncompatibleParameters(
            "BFV->TFHE torus-native modswitch requires ciphertext modulus product to fit in u128"
                .into(),
        )
    })?;
    let log_q_t = params.tfhe_params.ciphertext_modulus_log().0;
    let q_t = 1u128.checked_shl(log_q_t as u32).ok_or_else(|| {
        BridgeError::IncompatibleParameters("TFHE ciphertext modulus exceeds u128".into())
    })?;

    let count = lwe_q.dimension() + 1;
    let size_q = lwe_q.residues_q().len() / count;
    let mut data = vec![0u64; count];
    let q_bfv_big = BigUint::from_u128(q_bfv).unwrap();
    let q_t_big = BigUint::from_u128(q_t).unwrap();
    let half_q_bfv = &q_bfv_big >> 1;

    for coeff in 0..count {
        let mut residues = vec![0u64; size_q];
        for limb in 0..size_q {
            residues[limb] = lwe_q.residues_q()[limb * count + coeff];
        }
        let lifted = rns_ctx.compose_u128(&residues).map_err(|e| {
            BridgeError::IncompatibleParameters(format!(
                "BFV->TFHE torus-native CRT reconstruction failed: {e:?}"
            ))
        })?;

        let numerator = BigUint::from_u128(lifted).unwrap() * &q_t_big + &half_q_bfv;
        let quotient: BigUint = numerator / &q_bfv_big;
        let mut result_coeff = quotient.to_u64().ok_or_else(|| {
            BridgeError::IncompatibleParameters(
                "BFV->TFHE torus-native modswitch quotient does not fit in u64".into(),
            )
        })?;

        if log_q_t < 64 {
            let modulus = 1u64 << log_q_t;
            result_coeff %= modulus;
        }
        data[coeff] = result_coeff;
    }

    let mut out = LweCiphertext::from_data(
        data,
        CiphertextModulusLog(params.tfhe_params.ciphertext_modulus_log().0),
    );
    out.reduce();
    Ok(out)
}
