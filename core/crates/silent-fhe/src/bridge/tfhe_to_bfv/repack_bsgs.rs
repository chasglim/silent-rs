use crate::bridge::keys::TfheToBfvKey;
use crate::bridge::{BridgeError, BridgeParams};
use silent_rlwe::{Ciphertext, LweCiphertext};

pub fn repack_bsgs(
    _tfhe_cts: &[LweCiphertext],
    _key: &TfheToBfvKey,
    _params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    Err(BridgeError::IncompatibleParameters(
        "TFHE->BFV BSGS repacking requires packed BFV accumulator rotations and a proven noise budget; use the naive homomorphic path or provide the BSGS backend"
            .into(),
    ))
}
