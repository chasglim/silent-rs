use super::linear_transform::bsgs_linear_transform;
use crate::bridge::{BridgeError, BridgeKeys, BridgeParams};
use silent_rlwe::Ciphertext;

/// Convert from Coefficient layout to Slot layout.
pub fn coeff_to_slot(
    ct: &Ciphertext,
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    // BSGS acts in the batching slot domain. For coefficient-layout plaintext
    // x, the visible slots are NTT(x); to output slots x we apply INTT.
    let matrix = keys.slot_to_coeff_matrix.as_ref().ok_or_else(|| {
        BridgeError::IncompatibleParameters(
            "CoeffToSlot inverse-NTT matrix missing; check if BFV plaintext modulus is NTT-friendly"
                .into(),
        )
    })?;

    let n = params.bfv_params.degree();
    let bias = vec![0u64; n];

    bsgs_linear_transform(ct, matrix, &bias, keys, params)
}
