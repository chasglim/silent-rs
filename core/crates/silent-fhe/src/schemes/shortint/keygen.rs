//! Key ceremony for shortint: one shot client + server material (mirrors BFV’s
//! `BfvKeyGenerator` ergonomics, but returns both encryptor and eval keys).

use rand_core::SeedableRng;
use silent_utils::rng::SecureRng;

use crate::core::keys::KeyGenerator;
use crate::schemes::tfhe::params::TfheParameters;

use super::client_key::ShortintClientKey;
use super::error::ShortintError;
use super::server_key::ShortintServerKey;

/// Client encrypt/decrypt + server homomorphic / PBS keys from one generation run.
#[derive(Clone)]
pub struct ShortintKeys {
    pub client: ShortintClientKey,
    pub server: ShortintServerKey,
}

/// Sample [`ShortintKeys`] for validated [`TfheParameters`].
///
/// Mirrors [`crate::schemes::bfv::keys::BfvKeyGenerator`]: own parameters and RNG,
/// then call [`Self::try_generate`] or the infallible [`Self::generate`] when
/// parameters are known-good (e.g. presets).
pub struct ShortintKeyGenerator {
    params: TfheParameters,
    rng: SecureRng,
}

impl ShortintKeyGenerator {
    pub fn new(params: TfheParameters) -> Self {
        Self::with_rng(params, SecureRng::from_entropy())
    }

    pub fn with_rng(params: TfheParameters, rng: SecureRng) -> Self {
        Self { params, rng }
    }

    #[inline]
    pub fn params(&self) -> &TfheParameters {
        &self.params
    }

    /// Fallible generation (preferred when `TfheParameters` comes from user input).
    pub fn try_generate(&mut self) -> Result<ShortintKeys, ShortintError> {
        let (client, server) = super::generate_shortint_keys(self.params.clone(), &mut self.rng)?;
        Ok(ShortintKeys { client, server })
    }

    /// Same as [`Self::try_generate`] but panics on parameter / shortint layout errors.
    pub fn generate(&mut self) -> ShortintKeys {
        self.try_generate().expect(
            "shortint key generation failed; check TfheParameters (e.g. message_modulus ≥ 2)",
        )
    }
}

impl KeyGenerator for ShortintKeyGenerator {
    /// Full shortint material: client secret + server evaluation key.
    type SecretKey = ShortintKeys;

    fn generate_secret_key(&mut self) -> Self::SecretKey {
        self.generate()
    }
}
