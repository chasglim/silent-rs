//! TFHE key generators.
//!
//! M4 only ships the secret-key paths.  Bootstrap-key and KSK construction
//! land alongside the PBS pipeline in M5.

use rand_core::SeedableRng;
use silent_ring::NativePoly;
use silent_rlwe::{GlweSecretKey, LweCiphertext, LwePublicKey, LweSecretKey};
use silent_utils::rng::SecureRng;

use crate::core::keys::{KeyGenerator, PublicKeyGen};

use super::crypto::encrypt_lwe_under_sk;
use super::params::TfheParameters;
use super::sampling::fill_binary;

/// Default number of LWE zero encryptions stored in a public key.  We pick
/// `lwe_dimension + 64` as a conservative leftover-hash-lemma margin: the
/// random binary subset selecting from `count` ciphertexts must have at
/// least `n + ω(log n)` "extra" entries to guarantee uniformity of the
/// resulting LWE mask.  Sixty-four extra rows give a comfortable
/// statistical security margin for the parameter sets we ship.
pub const DEFAULT_PUBLIC_KEY_ZERO_ENCRYPTION_COUNT_OVERHEAD: usize = 64;

/// Owns the entropy source used for all TFHE key sampling.  Mirrors the role
/// played by `BfvKeyGenerator` for BFV.
pub struct TfheKeyGenerator {
    params: TfheParameters,
    rng: SecureRng,
}

impl TfheKeyGenerator {
    pub fn new(params: TfheParameters) -> Self {
        Self::with_rng(params, SecureRng::from_entropy())
    }

    pub fn with_rng(params: TfheParameters, rng: SecureRng) -> Self {
        Self { params, rng }
    }

    pub fn params(&self) -> &TfheParameters {
        &self.params
    }

    /// Sample a fresh small (LWE) secret key.  Distribution: uniform binary.
    pub fn generate_lwe_secret_key(&mut self) -> LweSecretKey {
        let mut data = vec![0u64; self.params.lwe_dimension().0];
        fill_binary(&mut self.rng, &mut data);
        LweSecretKey::from_data(data, self.params.ciphertext_modulus_log())
    }

    /// Sample a fresh GLWE secret key.  Distribution: uniform binary.
    pub fn generate_glwe_secret_key(&mut self) -> GlweSecretKey {
        let log_q = self.params.ciphertext_modulus_log().0;
        let n = self.params.polynomial_size().0;
        let polys = (0..self.params.glwe_dimension().0)
            .map(|_| {
                let mut buf = vec![0u64; n];
                fill_binary(&mut self.rng, &mut buf);
                NativePoly::from_u64(&buf, log_q)
            })
            .collect();
        GlweSecretKey::from_polys(polys, self.params.ciphertext_modulus_log())
    }

    /// Returns the equivalent flat LWE secret key for a GLWE secret key.
    /// Useful when encrypting under the "big" key after sample extraction.
    pub fn lwe_secret_from_glwe(&self, glwe_sk: &GlweSecretKey) -> LweSecretKey {
        LweSecretKey::from_data(glwe_sk.flatten(), self.params.ciphertext_modulus_log())
    }

    /// Direct mutable access to the inner RNG for callers that need to
    /// generate multiple correlated keys (e.g. PBS in M5 will reuse this RNG
    /// for every key matrix it samples).
    #[allow(dead_code)]
    pub(crate) fn rng_mut(&mut self) -> &mut SecureRng {
        &mut self.rng
    }

    /// Generate a TFHE LWE public key from a secret key, using the default
    /// zero-encryption count (`lwe_dimension + 64`).
    ///
    /// The public key is just `count` independent symmetric encryptions of
    /// the zero plaintext under `sk`.  Encryption with this key uses a
    /// random binary subset-sum (see
    /// [`super::crypto::TfheEncryptor::encrypt_with_public_key`]).
    pub fn generate_lwe_public_key(&mut self, sk: &LweSecretKey) -> LwePublicKey {
        let count =
            self.params.lwe_dimension().0 + DEFAULT_PUBLIC_KEY_ZERO_ENCRYPTION_COUNT_OVERHEAD;
        self.generate_lwe_public_key_with_count(sk, count)
    }

    /// Generate an LWE public key with an explicit number of zero
    /// encryptions.  Larger `count` increases mask randomness per
    /// ciphertext at the cost of a bigger key.  Must be ≥ `lwe_dimension`.
    pub fn generate_lwe_public_key_with_count(
        &mut self,
        sk: &LweSecretKey,
        count: usize,
    ) -> LwePublicKey {
        debug_assert_eq!(sk.dimension().0, self.params.lwe_dimension().0);
        debug_assert!(
            count >= self.params.lwe_dimension().0,
            "public-key zero-encryption count {count} must be ≥ lwe_dimension {}",
            self.params.lwe_dimension().0
        );

        let log_q = self.params.ciphertext_modulus_log();
        let noise = self.params.lwe_noise();
        let zeros: Vec<LweCiphertext> = (0..count)
            .map(|_| encrypt_lwe_under_sk(&mut self.rng, sk, 0, noise, log_q))
            .collect();
        LwePublicKey::from_zero_encryptions(zeros, self.params.lwe_dimension(), log_q)
    }
}

impl KeyGenerator for TfheKeyGenerator {
    type SecretKey = LweSecretKey;

    /// The "default" TFHE secret key is the *small* LWE key — the one under
    /// which user-facing ciphertexts live.  GLWE secret keys (used for
    /// bootstrap-key encryption) are obtained via
    /// [`TfheKeyGenerator::generate_glwe_secret_key`] separately because they
    /// have a different structural type.
    fn generate_secret_key(&mut self) -> Self::SecretKey {
        self.generate_lwe_secret_key()
    }
}

impl PublicKeyGen for TfheKeyGenerator {
    type PublicKey = LwePublicKey;

    fn generate_public_key(&mut self, sk: &Self::SecretKey) -> Self::PublicKey {
        self.generate_lwe_public_key(sk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use silent_params::presets;

    #[test]
    fn lwe_key_has_correct_dimension() {
        let params = TfheParameters::new(presets::toy::toy_tfhe_n512()).unwrap();
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([7u8; 32]));
        let sk = keygen.generate_lwe_secret_key();
        assert_eq!(sk.dimension().0, params.lwe_dimension().0);
        assert!(sk.data().iter().all(|&v| v == 0 || v == 1));
    }

    #[test]
    fn glwe_key_shape_matches_params() {
        let params = TfheParameters::new(presets::toy::toy_tfhe_n512()).unwrap();
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([7u8; 32]));
        let sk = keygen.generate_glwe_secret_key();
        assert_eq!(sk.glwe_dimension().0, params.glwe_dimension().0);
        assert_eq!(sk.polynomial_size().0, params.polynomial_size().0);
        for poly in sk.polys() {
            assert!(poly.coeffs().iter().all(|&c| c == 0 || c == 1));
        }
    }

    #[test]
    fn lwe_public_key_has_expected_shape() {
        let params = TfheParameters::new(presets::toy::toy_tfhe_n512()).unwrap();
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([7u8; 32]));
        let sk = keygen.generate_lwe_secret_key();
        let pk = keygen.generate_lwe_public_key(&sk);
        assert_eq!(pk.dimension().0, params.lwe_dimension().0);
        assert_eq!(pk.log_modulus(), params.ciphertext_modulus_log().0);
        assert_eq!(
            pk.count(),
            params.lwe_dimension().0 + DEFAULT_PUBLIC_KEY_ZERO_ENCRYPTION_COUNT_OVERHEAD
        );
        for ct in pk.ciphertexts() {
            assert_eq!(ct.dimension().0, params.lwe_dimension().0);
        }
    }

    #[test]
    fn flatten_glwe_key_matches_concatenation() {
        let params = TfheParameters::new(presets::toy::toy_tfhe_n512()).unwrap();
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([7u8; 32]));
        let glwe_sk = keygen.generate_glwe_secret_key();
        let flat = keygen.lwe_secret_from_glwe(&glwe_sk);
        assert_eq!(
            flat.dimension().0,
            params.glwe_dimension().0 * params.polynomial_size().0
        );
    }
}
