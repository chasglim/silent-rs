//! Shortint client key: small LWE secret + encrypt / decrypt.

use silent_rlwe::LweSecretKey;
use silent_utils::rng::SecureRng;

use super::ciphertext::ShortintCiphertext;
use super::error::ShortintError;
use crate::schemes::tfhe::crypto::{TfheDecryptor, TfheEncryptor};
use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::params::TfheParameters;

/// Client-held secret material for short integers under TFHE.
#[derive(Clone)]
pub struct ShortintClientKey {
    params: TfheParameters,
    sk: LweSecretKey,
    encoder: TfheEncoder,
    encryptor: TfheEncryptor,
}

impl ShortintClientKey {
    pub(crate) fn new(
        params: TfheParameters,
        sk: LweSecretKey,
        encoder: TfheEncoder,
        rng: SecureRng,
    ) -> Self {
        let encryptor = TfheEncryptor::new(params.clone(), rng);
        Self {
            params,
            sk,
            encoder,
            encryptor,
        }
    }

    #[inline]
    pub fn parameters(&self) -> &TfheParameters {
        &self.params
    }

    #[inline]
    pub fn encoder(&self) -> &TfheEncoder {
        &self.encoder
    }

    /// Encrypt `message ∈ [0, message_modulus)`.
    pub fn encrypt(&mut self, message: u64) -> Result<ShortintCiphertext, ShortintError> {
        let mm = self.params.message_modulus().0;
        if message >= mm {
            return Err(ShortintError::MessageOutOfRange {
                message,
                max: mm.saturating_sub(1),
            });
        }
        let ct = self
            .encryptor
            .encrypt_message(&self.sk, &self.encoder, message);
        Ok(ShortintCiphertext::new_fresh(ct, &self.params))
    }

    /// Decrypt to the message slot (carry stripped).
    pub fn decrypt(&self, ct: &ShortintCiphertext) -> Result<u64, ShortintError> {
        ct.ensure_params(&self.params)?;
        let dec = TfheDecryptor::new(self.params.clone(), self.sk.clone());
        Ok(dec.decrypt_message(ct.lwe_ciphertext(), &self.encoder))
    }

    /// Decrypt the full `(message, carry)` slot.
    pub fn decrypt_full(&self, ct: &ShortintCiphertext) -> Result<u64, ShortintError> {
        ct.ensure_params(&self.params)?;
        let dec = TfheDecryptor::new(self.params.clone(), self.sk.clone());
        Ok(dec.decrypt_full(ct.lwe_ciphertext(), &self.encoder))
    }
}
