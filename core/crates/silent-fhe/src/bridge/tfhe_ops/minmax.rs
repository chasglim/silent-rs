use super::BridgeOpsError;
use super::compare::eval_compare;
use super::mux::eval_mux;
use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::params::TfheParameters;
use silent_rlwe::{LweBootstrapKey, LweCiphertext, LweKeyswitchKey};

/// Minimum over bounded unsigned values in the lower half of the full TFHE
/// plaintext domain: `min(x, y) = mux(cmp(y, x), x, y)`.
pub fn eval_min(
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bsk: &LweBootstrapKey,
    ksk: &LweKeyswitchKey,
    x: &LweCiphertext,
    y: &LweCiphertext,
) -> Result<LweCiphertext, BridgeOpsError> {
    let cmp = eval_compare(params, encoder, bsk, ksk, y, x)?;
    eval_mux(params, encoder, bsk, ksk, &cmp, x, y)
}

/// Maximum over bounded unsigned values in the lower half of the full TFHE
/// plaintext domain: `max(x, y) = mux(cmp(x, y), x, y)`.
pub fn eval_max(
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bsk: &LweBootstrapKey,
    ksk: &LweKeyswitchKey,
    x: &LweCiphertext,
    y: &LweCiphertext,
) -> Result<LweCiphertext, BridgeOpsError> {
    let cmp = eval_compare(params, encoder, bsk, ksk, x, y)?;
    eval_mux(params, encoder, bsk, ksk, &cmp, x, y)
}
