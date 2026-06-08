//! Errors returned by the TFHE [`super::ShortintServerKey`] / [`super::ShortintClientKey`] APIs.

use thiserror::Error;

use super::types::{Degree, MaxDegree, MaxNoiseLevel, NoiseLevel};

/// Shortint homomorphic operation or encoding error.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ShortintError {
    #[error("plaintext message {message} is out of range [0, {max}]")]
    MessageOutOfRange { message: u64, max: u64 },
    #[error("TFHE parameters do not match the ciphertext (dimension / modulus / shortint moduli)")]
    ParameterMismatch,
    #[error("homomorphic carry capacity exceeded (degree {degree} > max {max_degree})")]
    CarryFull {
        degree: Degree,
        max_degree: MaxDegree,
    },
    #[error("noise budget exceeded after operation (noise {noise:?} > max {max_noise:?})")]
    NoiseTooBig {
        noise: NoiseLevel,
        max_noise: MaxNoiseLevel,
    },
    #[error("bivariate PBS: rhs degree {0:?} must be strictly less than pack factor {1}")]
    UnscaledScaledOverlap(Degree, u64),
    #[error("message_modulus must be ≥ 2 for shortint noise-degree formulas")]
    MessageModulusTooSmall,
    #[error("bivariate PBS preconditions not met: {0}")]
    BivariatePbsPrecondition(&'static str),
}
