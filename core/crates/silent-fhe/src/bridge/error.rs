//! Error types produced by the BFV ↔ TFHE bridge.

use thiserror::Error;

/// Result alias used throughout the bridge module.
pub type BridgeResult<T> = Result<T, BridgeError>;

/// Errors that can arise during scheme conversion.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BridgeError {
    #[error("slot index {0} out of range for BFV batch size {1}")]
    SlotIndexOutOfRange(usize, usize),

    #[error("batch size {0} exceeds BFV polynomial degree {1}")]
    BatchTooLarge(usize, usize),

    #[error("plaintext modulus mismatch: BFV t={0}, TFHE message_modulus={1}")]
    PlaintextModulusMismatch(u64, u64),

    #[error(
        "ciphertext modulus mismatch: cannot convert between schemes with different power-of-two moduli"
    )]
    CiphertextModulusMismatch,

    #[error("BFV ciphertext must be decryptable (expected 2 elements, got {0})")]
    InvalidBfvCiphertext(usize),

    #[error("TFHE ciphertext body does not fit into BFV plaintext modulus")]
    TfheBodyOverflow,

    #[error("bridge parameters are incompatible: {0}")]
    IncompatibleParameters(String),
}
