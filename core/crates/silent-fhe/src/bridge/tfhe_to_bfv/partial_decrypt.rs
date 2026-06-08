use super::bfv_helpers::{scalar_mul_bfv, trivial_constant_bfv};
use crate::bridge::keys::TfheToBfvKey;
use crate::bridge::{BridgeError, BridgeParams};
use crate::core::evaluator::HeEvaluator;
use crate::schemes::bfv::ops::BfvEvaluator;
use silent_rlwe::{Ciphertext, LweCiphertext};

/// Homomorphic Partial Decryption.
pub fn partial_decrypt(
    tfhe_cts: &[LweCiphertext],
    key: &TfheToBfvKey,
    params: &BridgeParams,
) -> Result<Vec<Ciphertext>, BridgeError> {
    ensure_phase_repack_preconditions(params)?;

    let degree = params.bfv_params.degree();
    let evaluator = BfvEvaluator::new(params.bridge_bfv_eval_params());
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let q_tfhe_mask = if log_q == 64 {
        u64::MAX
    } else {
        (1u64 << log_q) - 1
    };

    if key.rk.len() < params.tfhe_params.lwe_dimension().0 {
        return Err(BridgeError::IncompatibleParameters(
            "repacking key is smaller than TFHE LWE dimension".into(),
        ));
    }

    Ok(tfhe_cts
        .iter()
        .map(|lwe| {
            if lwe.dimension().0 != params.tfhe_params.lwe_dimension().0 {
                return Err(BridgeError::IncompatibleParameters(format!(
                    "TFHE ciphertext has LWE dimension {}, expected {}",
                    lwe.dimension().0,
                    params.tfhe_params.lwe_dimension().0
                )));
            }
            if lwe.log_modulus() != log_q {
                return Err(BridgeError::IncompatibleParameters(format!(
                    "TFHE ciphertext has log modulus {}, expected {}",
                    lwe.log_modulus(),
                    log_q
                )));
            }
            if lwe.dimension().0 > degree {
                return Err(BridgeError::IncompatibleParameters(format!(
                    "TFHE LWE dimension {} exceeds BFV ring degree {} for naive partial decrypt",
                    lwe.dimension().0,
                    degree
                )));
            }

            let body = lwe.body() & q_tfhe_mask;
            let mut accum = trivial_constant_bfv(params, body);
            for (j, &a_j) in lwe.mask().iter().enumerate() {
                let a = a_j & q_tfhe_mask;
                if a == 0 {
                    continue;
                }
                let scaled = scalar_mul_bfv(&key.rk[j], torus_negate(a, log_q), params);
                accum = evaluator.add(&accum, &scaled);
            }
            Ok(accum)
        })
        .collect::<Result<Vec<_>, _>>()?)
}

pub fn ensure_phase_repack_preconditions(params: &BridgeParams) -> Result<(), BridgeError> {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    if log_q == 64 {
        return Err(BridgeError::IncompatibleParameters(
            "TFHE->BFV phase repacking cannot lift q_T=2^64 phases into a single BFV plaintext modulus; use CRT/plaintext lifting"
                .into(),
        ));
    }

    let q_tfhe = 1u64.checked_shl(log_q as u32).ok_or_else(|| {
        BridgeError::IncompatibleParameters("TFHE ciphertext modulus exceeds u64".into())
    })?;

    if params.tfhe_params.lwe_dimension().0 > params.bfv_params.degree() {
        return Err(BridgeError::IncompatibleParameters(format!(
            "TFHE LWE dimension {} exceeds BFV ring degree {} for phase repacking",
            params.tfhe_params.lwe_dimension().0,
            params.bfv_params.degree()
        )));
    }

    if params.bfv_params.plain_modulus() == q_tfhe {
        return Ok(());
    }

    super::fmod_precompute::ensure_periodic_center_preconditions(params)
}

pub fn ensure_native_phase_repack_preconditions(params: &BridgeParams) -> Result<(), BridgeError> {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    if log_q == 64 {
        return Err(BridgeError::IncompatibleParameters(
            "native TFHE->BFV phase repacking cannot use BFV plaintext modulus 2^64; use CRT/plaintext lifting"
                .into(),
        ));
    }

    let q_tfhe = 1u64.checked_shl(log_q as u32).ok_or_else(|| {
        BridgeError::IncompatibleParameters("TFHE ciphertext modulus exceeds u64".into())
    })?;
    if params.bfv_params.plain_modulus() != q_tfhe {
        return Err(BridgeError::IncompatibleParameters(format!(
            "naive TFHE->BFV phase repacking requires BFV plaintext modulus t={} to equal TFHE torus modulus q_T=2^{}={}; CRT/plaintext lifting is required otherwise",
            params.bfv_params.plain_modulus(),
            log_q,
            q_tfhe
        )));
    }

    if params.tfhe_params.lwe_dimension().0 > params.bfv_params.degree() {
        return Err(BridgeError::IncompatibleParameters(format!(
            "TFHE LWE dimension {} exceeds BFV ring degree {} for native phase repacking",
            params.tfhe_params.lwe_dimension().0,
            params.bfv_params.degree()
        )));
    }

    Ok(())
}

fn torus_negate(value: u64, log_q: u8) -> u64 {
    if value == 0 {
        return 0;
    }
    if log_q == 64 {
        value.wrapping_neg()
    } else {
        (1u64 << log_q) - value
    }
}
