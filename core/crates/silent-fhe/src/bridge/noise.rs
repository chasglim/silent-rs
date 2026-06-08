//! Noise margin management for the bridge.

use crate::bridge::{BridgeError, BridgeParams};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoiseBound {
    pub max_abs: u128,
}

impl NoiseBound {
    pub fn new(max_abs: u128) -> Self {
        Self { max_abs }
    }
}

pub fn check_torus_decode_margin(
    params: &BridgeParams,
    noise: &NoiseBound,
) -> Result<(), BridgeError> {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0 as u32;
    let q_t = 1u128
        .checked_shl(log_q)
        .ok_or_else(|| BridgeError::IncompatibleParameters("torus modulus exceeds u128".into()))?;
    let message_modulus = params.message_modulus() as u128;
    let conservative_bound = q_t / (4 * message_modulus.max(1));
    if noise.max_abs > conservative_bound {
        return Err(BridgeError::IncompatibleParameters(format!(
            "torus decode noise bound {} exceeds conservative limit {}",
            noise.max_abs, conservative_bound
        )));
    }
    Ok(())
}
