use super::BridgeOpsError;
use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::ops::TfheEvaluator;
use crate::schemes::tfhe::params::TfheParameters;
use silent_rlwe::{LweBootstrapKey, LweCiphertext, LweKeyswitchKey};

/// Multiplexing: `mux(b, x, y) = b * x + (1 - b) * y`.
///
/// `b` must encrypt `0` or `1`.  `x` and `y` must decrypt into the lower half
/// of the full TFHE plaintext domain; this is the same bounded-integer domain
/// used by [`super::compare::eval_compare`].
pub fn eval_mux(
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bsk: &LweBootstrapKey,
    ksk: &LweKeyswitchKey,
    b: &LweCiphertext,
    x: &LweCiphertext,
    y: &LweCiphertext,
) -> Result<LweCiphertext, BridgeOpsError> {
    let evaluator = TfheEvaluator::new(params.clone());

    let x_part = eval_select_product(params, encoder, bsk, ksk, b, x, true)?;
    let y_part = eval_select_product(params, encoder, bsk, ksk, b, y, false)?;
    evaluator
        .add(&x_part, &y_part)
        .map_err(BridgeOpsError::TfheEval)
}

fn eval_select_product(
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bsk: &LweBootstrapKey,
    ksk: &LweKeyswitchKey,
    selector: &LweCiphertext,
    value: &LweCiphertext,
    choose_when_one: bool,
) -> Result<LweCiphertext, BridgeOpsError> {
    let pack_factor = encoder.full_plaintext_modulus() / 2;
    if pack_factor < 2 {
        return Err(BridgeOpsError::UnsupportedMuxPlaintextSpace);
    }

    let evaluator = TfheEvaluator::new(params.clone());
    let scaled_selector = evaluator
        .scalar_mul(selector, pack_factor)
        .map_err(BridgeOpsError::TfheEval)?;
    let packed = evaluator
        .add(&scaled_selector, value)
        .map_err(BridgeOpsError::TfheEval)?;

    super::eval_lut(params, encoder, bsk, ksk, &packed, |packed_message| {
        let selector_bit = packed_message / pack_factor;
        let selected = selector_bit != 0;
        let should_keep = if choose_when_one { selected } else { !selected };
        if should_keep {
            packed_message % pack_factor
        } else {
            0
        }
    })
}
