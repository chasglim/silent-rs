use crate::bridge::tfhe_to_bfv::linear_transform::galois_linear_transform;
use crate::bridge::{BridgeError, BridgeKeys, BridgeParams};
use silent_rlwe::Ciphertext;

/// Convert BFV batching layout slots to coefficient layout.
///
/// Pegasus converts a packed RLWE ciphertext to LWE ciphertexts by first
/// evaluating SlotsToCoefficients in the RLWE/BFV domain and then extracting
/// coefficients. In SILENT's `BatchEncoder`, decoding slots is a plaintext
/// forward NTT. Therefore the slot-domain transform that makes the output
/// coefficient vector equal the input slot vector is the forward-NTT matrix.
pub fn slot_to_coeff(
    ct: &Ciphertext,
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    let matrix = keys.coeff_to_slot_matrix.as_ref().ok_or_else(|| {
        BridgeError::IncompatibleParameters(
            "SlotToCoeff forward-NTT matrix missing; check if BFV plaintext modulus is NTT-friendly"
                .into(),
        )
    })?;

    let n = params.bfv_params.degree();
    let bias = vec![0u64; n];
    galois_linear_transform(ct, matrix, &bias, keys, params)
}
