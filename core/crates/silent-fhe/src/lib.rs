pub mod bridge;
pub mod core;
pub mod io_impls;
pub mod schemes;

// Convenience re-exports
pub use core::encoder::HeEncoder;
pub use core::evaluator::HeEvaluator;
pub use core::keys::{KeyGenerator, PublicKeyGen};
pub use core::scheme::HeScheme;
pub use schemes::bfv::crypto::{BfvDecryptor, BfvEncryptor};
pub use schemes::bfv::encoding::BatchEncoder;
pub use schemes::bfv::ops::BfvEvaluator;
pub use schemes::bfv::params::BfvParameters;
pub use schemes::bfv::scheme::BfvScheme;
pub use schemes::bgv::crypto::{BgvDecryptor, BgvEncryptor};
pub use schemes::bgv::encoding::BgvBatchEncoder;
pub use schemes::bgv::keys::BgvKeyGenerator;
pub use schemes::bgv::ops::BgvEvaluator;
pub use schemes::bgv::params::BgvParameters;
pub use schemes::bgv::scheme::BgvScheme;
pub use schemes::shortint::{
    BivariateLookupTable, CiphertextNoiseDegree, Degree, MaxDegree, MaxNoiseLevel, NoiseLevel,
    ShortintCiphertext, ShortintClientKey, ShortintError, ShortintEvaluator, ShortintKeyGenerator,
    ShortintKeys, ShortintScheme, ShortintServerKey, generate_shortint_keys,
};
pub use schemes::tfhe::bootstrap::TfheBootstrapper;
pub use schemes::tfhe::bootstrap_fft::{LweBootstrapKeyFft, blind_rotate_assign_fft};
pub use schemes::tfhe::crypto::{TfheDecryptor, TfheEncryptor};
pub use schemes::tfhe::encoding::{TfheEncoder, TfhePlaintext};
pub use schemes::tfhe::keys::TfheKeyGenerator;
pub use schemes::tfhe::ops::{TfheEvalError, TfheEvaluator};
pub use schemes::tfhe::params::TfheParameters;
pub use schemes::tfhe::scheme::TfheScheme;
pub use silent_rlwe::LwePublicKey;

// Re-export the negacyclic-multiplier traits so callers can pick a TFHE
// backend without depending on `silent-math` directly.
pub use silent_math::fft64::{
    FftBuckets, FftMul, FftPlan, FftPoly, KaratsubaMul, NegacyclicMul, SchoolbookMul, with_fft_mul,
};
