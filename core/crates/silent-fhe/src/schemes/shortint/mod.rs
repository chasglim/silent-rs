//! Single-block TFHE short integers (`message_modulus × carry_modulus`): degree
//! and noise metadata, bivariate PBS packing, and carry-cleaning bootstraps.
//!
//! This module implements the usual “shortint” evaluation surface on top of the
//! crate’s TFHE PBS core—not a binding to any particular external library.

mod ciphertext;
mod client_key;
mod error;
mod evaluator;
mod keygen;
mod scheme;
mod server_key;
mod types;

pub use ciphertext::ShortintCiphertext;
pub use client_key::ShortintClientKey;
pub use error::ShortintError;
pub use evaluator::ShortintEvaluator;
pub use keygen::{ShortintKeyGenerator, ShortintKeys};
pub use scheme::ShortintScheme;
pub use server_key::{BivariateLookupTable, ShortintServerKey};
pub use types::{CiphertextNoiseDegree, Degree, MaxDegree, MaxNoiseLevel, NoiseLevel};

use rand::RngCore;
use rand_core::SeedableRng;
use silent_utils::rng::SecureRng;

use crate::schemes::tfhe::bootstrap::generate_pbs_keys;
use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::params::TfheParameters;

/// Generate matching [`ShortintClientKey`] and [`ShortintServerKey`] for `params`.
///
/// Uses the same PBS keygen path as [`generate_pbs_keys`].  Client encryption RNG
/// is derived from `rng` via a fresh 32-byte seed so it stays independent from
/// bootstrap / keyswitch sampling.
pub fn generate_shortint_keys(
    params: TfheParameters,
    rng: &mut SecureRng,
) -> Result<(ShortintClientKey, ShortintServerKey), ShortintError> {
    let (sk_small, _sk_glwe, bsk, ksk) = generate_pbs_keys(params.clone(), rng);
    let encoder = TfheEncoder::new(params.clone());
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    let client_rng = SecureRng::from_seed(seed);
    let client = ShortintClientKey::new(params.clone(), sk_small, encoder.clone(), client_rng);
    let server = ShortintServerKey::try_new(params, bsk, ksk, encoder)?;
    Ok((client, server))
}
