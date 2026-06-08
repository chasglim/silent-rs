use crate::distribution::DistributionType;
use crate::security::SecurityLevel;
use silent_math::modulus::ModulusError;
use silent_math::rns::RnsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParamError {
    #[error("ring dimension must be non-zero")]
    ZeroRingDim,
    #[error("ring dimension {0} must be a power of two")]
    RingDimNotPowerOfTwo(usize),
    #[error("log_n {log_n} does not match ring_dim {ring_dim}")]
    RingLogMismatch { log_n: u8, ring_dim: usize },
    #[error("coefficient modulus chain cannot be empty")]
    EmptyCoefficientModulusChain,
    #[error("unsupported modulus bit width {0}; SILENT currently supports 2..=61-bit NTT moduli")]
    InvalidModulusBits(u16),
    #[error(
        "HE Standard lookup is not available yet for distribution {distribution:?} and security {security:?}"
    )]
    UnsupportedStandardCombination {
        distribution: DistributionType,
        security: SecurityLevel,
    },
    #[error(
        "ciphertext modulus budget {required} exceeds HE Standard max {max} for ring_dim {ring_dim} and security {security:?}"
    )]
    SecurityBudgetExceeded {
        required: u16,
        max: u16,
        ring_dim: usize,
        security: SecurityLevel,
    },
    #[error("plaintext modulus must be at least 2")]
    PlaintextModulusTooSmall,
    #[error("default scale bits must be non-zero")]
    ZeroScaleBits,
    #[error("share modulus bits {share_bits} must cover plaintext modulus bits {plain_bits}")]
    ShareModulusTooSmall { share_bits: u16, plain_bits: u16 },
    #[error(
        "reconstruction bound bits {reconstruction_bits} exceed share modulus bits {share_bits}"
    )]
    ReconstructionBoundExceedsShareModulus {
        reconstruction_bits: u16,
        share_bits: u16,
    },
    #[error(
        "fixed-point scale bits {scale_bits} exceed reconstruction bound bits {reconstruction_bits}"
    )]
    ScaleExceedsReconstructionBound {
        scale_bits: u8,
        reconstruction_bits: u16,
    },
    #[error("required HSS budget {required} exceeds ciphertext modulus budget {available}")]
    HssBudgetExceeded { required: u16, available: u16 },
    #[error("max_linear_terms must be non-zero")]
    ZeroMaxLinearTerms,
    #[error("PQC size field `{field}` must be non-zero")]
    InvalidPqcSize { field: &'static str },
    #[error("failed to generate an NTT-friendly prime with {bits} bits for ring_dim {ring_dim}")]
    PrimeGenerationFailed { bits: u16, ring_dim: usize },
    #[error(
        "runtime ring dimension {actual} does not match parameter set ring dimension {expected}"
    )]
    RuntimeRingDimensionMismatch { expected: usize, actual: usize },
    #[error(
        "runtime plaintext modulus {actual} does not match parameter set plaintext modulus {expected}"
    )]
    RuntimePlaintextModulusMismatch { expected: u64, actual: u64 },
    #[error(
        "runtime {label} modulus chain bits {actual:?} do not match parameter set {expected:?}"
    )]
    RuntimeModulusChainMismatch {
        label: &'static str,
        expected: Vec<u16>,
        actual: Vec<u16>,
    },
    #[error("runtime key-switch modulus bits {actual:?} do not match parameter set {expected:?}")]
    RuntimeKeySwitchModulusMismatch {
        expected: Option<u16>,
        actual: Option<u16>,
    },
    #[error("invalid modulus value: {0:?}")]
    Modulus(ModulusError),
    #[error("RNS configuration error: {0:?}")]
    Rns(RnsError),

    // ----- TFHE-specific -------------------------------------------------
    #[error("TFHE only supports a 32- or 64-bit native ciphertext modulus, got 2^{0}")]
    TfheUnsupportedCiphertextModulusLog(u8),
    #[error("TFHE {which}_dimension must be non-zero")]
    TfheZeroDimension { which: &'static str },
    #[error("TFHE polynomial_size {0} must be a non-zero power of two")]
    TfhePolynomialSizeNotPowerOfTwo(usize),
    #[error("TFHE {which}_modulus {value} must be a non-zero power of two")]
    TfheModulusNotPowerOfTwo { which: &'static str, value: u64 },
    #[error(
        "TFHE total plaintext space {total} must be strictly smaller than the ciphertext modulus 2^{ciphertext_modulus_log}"
    )]
    TfheMessageSpaceTooLarge {
        total: u128,
        ciphertext_modulus_log: u8,
    },
    #[error(
        "TFHE {which} decomposition (base_log={base_log}, level={level}) does not fit in the ciphertext modulus 2^{ciphertext_modulus_log}"
    )]
    TfheDecompositionBudgetExceeded {
        which: &'static str,
        base_log: u8,
        level: u8,
        ciphertext_modulus_log: u8,
    },
    #[error(
        "TFHE LWE dimension {lwe_dimension} exceeds the equivalent big-LWE dimension {big_lwe_dimension} (k*N); after-PBS key-switch would be impossible"
    )]
    TfheKeyChainInconsistent {
        lwe_dimension: usize,
        big_lwe_dimension: usize,
    },
}

impl From<ModulusError> for ParamError {
    fn from(value: ModulusError) -> Self {
        Self::Modulus(value)
    }
}

impl From<RnsError> for ParamError {
    fn from(value: RnsError) -> Self {
        Self::Rns(value)
    }
}
