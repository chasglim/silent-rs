//! Bridge-side TFHE operations applied after BFV→TFHE conversion.
//!
//! These are thin wrappers around [`crate::schemes::tfhe::ops::TfheEvaluator`]
//! that run a PBS with a specific lookup table.

pub mod compare;
pub mod lut;
pub mod minmax;
pub mod mux;
pub mod relu;

pub use lut::{Lut, eval_lut as apply_lut_batch};

use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::params::TfheParameters;
use silent_math::fft64::FftMul;
use silent_rlwe::{LweBootstrapKey, LweCiphertext, LweKeyswitchKey};
use thiserror::Error;

/// Errors specific to bridge evaluation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BridgeOpsError {
    #[error("PBS not available: missing bootstrap or keyswitch key")]
    MissingPbsKeys,
    #[error("MUX requires a full TFHE plaintext modulus of at least 4")]
    UnsupportedMuxPlaintextSpace,
    #[error("TFHE evaluation error: {0}")]
    TfheEval(#[from] crate::schemes::tfhe::ops::TfheEvalError),
}

/// Apply an identity LUT via PBS (useful for testing the bridge round-trip).
///
/// This refreshes the noise but leaves the message unchanged.
pub fn eval_identity(
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bsk: &LweBootstrapKey,
    ksk: &LweKeyswitchKey,
    ct: &LweCiphertext,
) -> Result<LweCiphertext, BridgeOpsError> {
    eval_lut(params, encoder, bsk, ksk, ct, |m| m)
}

/// Apply a ReLU LUT: `f(m) = max(m, 0)`.
///
/// The message must be encoded as a **signed** integer in the range
/// `[-message_modulus/2, message_modulus/2)` for this to behave as true
/// ReLU.  For unsigned messages ReLU is simply identity.
pub fn eval_relu(
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bsk: &LweBootstrapKey,
    ksk: &LweKeyswitchKey,
    ct: &LweCiphertext,
) -> Result<LweCiphertext, BridgeOpsError> {
    let message_modulus = encoder.full_plaintext_modulus();
    eval_lut(params, encoder, bsk, ksk, ct, |m| {
        let signed = if m >= message_modulus / 2 {
            m as i64 - message_modulus as i64
        } else {
            m as i64
        };
        let relu = signed.max(0);
        relu as u64 % message_modulus
    })
}

/// Generic LUT evaluation via PBS.
pub fn eval_lut<F: FnMut(u64) -> u64>(
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bsk: &LweBootstrapKey,
    ksk: &LweKeyswitchKey,
    ct: &LweCiphertext,
    mut f: F,
) -> Result<LweCiphertext, BridgeOpsError> {
    let evaluator = crate::schemes::tfhe::ops::TfheEvaluator::new(params.clone());
    let backend = silent_math::fft64::SchoolbookMul;
    evaluator
        .apply_lookup_table_with(encoder, bsk, ksk, ct, |m| f(m), &backend)
        .map_err(BridgeOpsError::TfheEval)
}

/// LUT evaluation using the cached FFT bootstrap key (faster for repeated calls).
pub fn eval_lut_fft<F: FnMut(u64) -> u64>(
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bsk: &LweBootstrapKey,
    ksk: &LweKeyswitchKey,
    bsk_fft: &crate::schemes::tfhe::bootstrap_fft::LweBootstrapKeyFft,
    fft: &FftMul,
    ct: &LweCiphertext,
    mut f: F,
) -> Result<LweCiphertext, BridgeOpsError> {
    let evaluator = crate::schemes::tfhe::ops::TfheEvaluator::new(params.clone());
    evaluator
        .apply_lookup_table_fft(encoder, bsk, ksk, bsk_fft, fft, ct, |m| f(m))
        .map_err(BridgeOpsError::TfheEval)
}
