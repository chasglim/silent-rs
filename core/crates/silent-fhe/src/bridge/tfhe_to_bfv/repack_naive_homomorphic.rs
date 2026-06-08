//! TFHE -> BFV helper paths.
//!
//! This module now contains:
//! * `repack_naive`: a genuine homomorphic baseline that performs naive
//!   partial decryption under BFV and packs the resulting phases into BFV
//!   coefficient layout;
//! * `repack_with_secret`: an explicit trusted oracle kept only for tests.

#[cfg(test)]
use crate::schemes::bfv::crypto::BfvEncryptor;
#[cfg(test)]
use crate::schemes::tfhe::crypto::TfheDecryptor;
#[cfg(test)]
use crate::schemes::tfhe::encoding::TfheEncoder;
use silent_rlwe::{Ciphertext, LweCiphertext};
#[cfg(test)]
use silent_rlwe::{Plaintext, SecretKey};
#[cfg(test)]
use silent_utils::rng::SecureRng;

use super::bfv_helpers::shift_ciphertext_by_monomial;
use super::partial_decrypt::partial_decrypt;
use crate::bridge::keys::TfheToBfvKey;
use crate::bridge::{BridgeError, BridgeParams};
use crate::core::evaluator::HeEvaluator;
use crate::schemes::bfv::ops::BfvEvaluator;

pub fn repack_naive(
    tfhe_cts: &[LweCiphertext],
    key: &TfheToBfvKey,
    params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    if tfhe_cts.is_empty() {
        return Err(BridgeError::IncompatibleParameters(
            "repack_naive requires at least one TFHE ciphertext".into(),
        ));
    }
    if tfhe_cts.len() > params.num_slots {
        return Err(BridgeError::IncompatibleParameters(format!(
            "repack_naive received {} TFHE ciphertexts but bridge num_slots is {}",
            tfhe_cts.len(),
            params.num_slots
        )));
    }

    let phases = partial_decrypt(tfhe_cts, key, params)?;

    let evaluator = BfvEvaluator::new(params.bridge_bfv_eval_params());
    let mut packed = shift_ciphertext_by_monomial(&phases[0], 0, params);
    for (index, ct) in phases
        .iter()
        .enumerate()
        .skip(1)
        .take(params.num_slots.saturating_sub(1))
    {
        let shifted = shift_ciphertext_by_monomial(ct, index, params);
        packed = evaluator.add(&packed, &shifted);
    }
    Ok(packed)
}

#[cfg(test)]
pub fn repack_with_secret(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
    tfhe_dec: &TfheDecryptor,
    bfv_sk: &SecretKey,
    rng: SecureRng,
) -> Result<Ciphertext, BridgeError> {
    let mut encryptor = BfvEncryptor::new(params.bfv_params.clone(), rng);
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let degree = params.bfv_params.degree();
    let t = params.bfv_params.plain_modulus();

    let mut poly = silent_ring::Poly::new(degree, 1);
    for (index, ct) in tfhe_cts.iter().take(params.num_slots).enumerate() {
        let message = tfhe_dec.decrypt_full(ct, &encoder) % t;
        poly.limb_mut(0)[index] = message;
    }

    Ok(encryptor.encrypt_symmetric(bfv_sk, &Plaintext { value: poly }))
}
