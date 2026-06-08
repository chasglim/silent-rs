use super::{BridgeOpsError, eval_lut};
use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::ops::TfheEvaluator;
use crate::schemes::tfhe::params::TfheParameters;
use silent_rlwe::{LweBootstrapKey, LweCiphertext, LweKeyswitchKey};

/// Compare two TLWE ciphertexts: `cmp(x, y) = 1[x >= y]`.
pub fn eval_compare(
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bsk: &LweBootstrapKey,
    ksk: &LweKeyswitchKey,
    x: &LweCiphertext,
    y: &LweCiphertext,
) -> Result<LweCiphertext, BridgeOpsError> {
    let evaluator = TfheEvaluator::new(params.clone());
    let d = evaluator.sub(x, y).map_err(BridgeOpsError::TfheEval)?;

    let message_modulus = encoder.full_plaintext_modulus();
    eval_lut(params, encoder, bsk, ksk, &d, |m| {
        if m < message_modulus / 2 { 1 } else { 0 }
    })
}
