//! TFHE (CGGI) scheme implementation, scheduled to be the second FHE family
//! that lives alongside BFV in `silent-fhe`.
//!
//! Layout (mirrors the BFV submodule):
//!   * [`params`]    — runtime parameter wrapper implementing [`HeContext`]
//!   * [`scheme`]    — the [`HeScheme`] marker
//!   * [`encoding`]  — message → LWE-plaintext encoder/decoder
//!   * [`keys`]      — secret-key / bootstrap-key / KS-key generators
//!   * [`crypto`]    — symmetric LWE encryption & decryption
//!   * [`ops`]       — homomorphic linear operations + `apply_lookup_table`
//!   * [`keyswitch`] — LWE→LWE key switching (M5)
//!   * [`bootstrap`] — programmable bootstrapping pipeline (M5)
//!
//! [`HeContext`]: crate::core::context::HeContext
//! [`HeScheme`]: crate::core::scheme::HeScheme

pub mod crypto;
pub mod encoding;
pub mod keys;
pub mod ops;
pub mod params;
pub mod sampling;
pub mod scheme;

pub mod bootstrap;
pub mod bootstrap_fft;
pub mod keyswitch;

pub use bootstrap::{TfheBootstrapper, sample_extract};
pub use bootstrap_fft::{LweBootstrapKeyFft, blind_rotate_assign_fft};
pub use crypto::{TfheDecryptor, TfheEncryptor, encrypt_lwe_under_sk};
pub use encoding::{TfheEncoder, TfhePlaintext};
pub use keys::TfheKeyGenerator;
pub use keyswitch::{LweKeyswitcher, build_lwe_keyswitch_key};
pub use ops::{TfheEvalError, TfheEvaluator};
pub use params::TfheParameters;
pub use scheme::TfheScheme;
