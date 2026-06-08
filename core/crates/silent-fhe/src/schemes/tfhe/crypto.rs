//! Symmetric LWE encryption / decryption for TFHE.
//!
//! Algorithms:
//!
//! ```text
//! Enc(s, μ):
//!     a ←$ U(Z_q^n)
//!     e ← noise
//!     b = ⟨a, s⟩ + Δ·μ + e   (mod q)
//!     return (a, b)
//!
//! Dec(s, (a, b)):
//!     m̃ = b − ⟨a, s⟩         (mod q)
//!     return round(m̃ · plain / q)
//! ```
//!
//! `q = 2^k` is power-of-two, so all "mod q" operations are wrapping
//! arithmetic on `u64`/`u32`.

use rand::RngCore;
use silent_params::{CiphertextModulusLog, NoiseDistribution};
use silent_rlwe::{LweCiphertext, LwePublicKey, LweSecretKey};
use silent_utils::rng::SecureRng;

use super::encoding::{TfheEncoder, TfhePlaintext};
use super::params::TfheParameters;
use super::sampling::{fill_uniform_mod, sample_noise};

/// Symmetric LWE encryption under an *arbitrary* secret key.
///
/// Used both by [`TfheEncryptor::encrypt_symmetric`] (which fixes the small
/// LWE secret key) and by the M5 key-switch / bootstrap key generators
/// (which encrypt under the big sample-extracted GLWE key, the small key, or
/// any custom intermediate key).
///
/// The plaintext is interpreted as an already-scaled value (i.e. `Δ·μ + e`'s
/// leading term).  Use the encoder to obtain the right magnitude.
pub fn encrypt_lwe_under_sk(
    rng: &mut SecureRng,
    sk: &LweSecretKey,
    plaintext_value: u64,
    noise: NoiseDistribution,
    log_modulus: CiphertextModulusLog,
) -> LweCiphertext {
    let n = sk.dimension().0;
    let mut data = vec![0u64; n + 1];

    fill_uniform_mod(rng, log_modulus.0, &mut data[..n]);

    let mut inner: u64 = 0;
    for (a, s) in data[..n].iter().zip(sk.data().iter()) {
        inner = inner.wrapping_add(a.wrapping_mul(*s));
    }

    let e = sample_noise(rng, noise, log_modulus.0);
    data[n] = inner.wrapping_add(plaintext_value).wrapping_add(e);

    let mut ct = LweCiphertext::from_data(data, log_modulus);
    ct.reduce();
    ct
}

/// Symmetric LWE encryptor.
#[derive(Clone)]
pub struct TfheEncryptor {
    params: TfheParameters,
    rng: SecureRng,
}

impl TfheEncryptor {
    pub fn new(params: TfheParameters, rng: SecureRng) -> Self {
        Self { params, rng }
    }

    pub fn params(&self) -> &TfheParameters {
        &self.params
    }

    /// Encrypt a TFHE plaintext under the LWE secret key `sk` (small key).
    pub fn encrypt_symmetric(
        &mut self,
        sk: &LweSecretKey,
        plaintext: TfhePlaintext,
    ) -> LweCiphertext {
        debug_assert_eq!(sk.dimension().0, self.params.lwe_dimension().0);
        debug_assert_eq!(
            plaintext.log_modulus,
            self.params.ciphertext_modulus_log().0
        );

        encrypt_lwe_under_sk(
            &mut self.rng,
            sk,
            plaintext.value,
            self.params.lwe_noise(),
            self.params.ciphertext_modulus_log(),
        )
    }

    /// Convenience overload that encrypts a *message* slot directly.
    pub fn encrypt_message(
        &mut self,
        sk: &LweSecretKey,
        encoder: &TfheEncoder,
        message: u64,
    ) -> LweCiphertext {
        let plain = encoder.encode_message(message);
        self.encrypt_symmetric(sk, plain)
    }

    /// Asymmetric LWE encryption under a public key.
    ///
    /// Algorithm (binary-subset-sum a.k.a. "regev pubkey encryption"):
    ///
    /// ```text
    /// r ← {0, 1}^count   (uniform binary)
    /// ct = Σᵢ rᵢ · pk[i] + (0, …, 0, μ̃)   (mod q)
    /// ```
    ///
    /// where `μ̃ = plaintext.value` is the encoded plaintext.  Wraps on
    /// overflow because the modulus is power-of-two.
    pub fn encrypt_with_public_key(
        &mut self,
        pk: &LwePublicKey,
        plaintext: TfhePlaintext,
    ) -> LweCiphertext {
        debug_assert_eq!(pk.dimension().0, self.params.lwe_dimension().0);
        debug_assert_eq!(
            plaintext.log_modulus,
            self.params.ciphertext_modulus_log().0
        );

        let count = pk.count();
        let n = self.params.lwe_dimension().0;
        let mut data = vec![0u64; n + 1];

        // Sample a uniform binary mask one buffered u64 at a time.
        let mut bit_buffer: u64 = 0;
        let mut bits_left: u32 = 0;
        for i in 0..count {
            if bits_left == 0 {
                bit_buffer = self.rng.next_u64();
                bits_left = 64;
            }
            let select = bit_buffer & 1;
            bit_buffer >>= 1;
            bits_left -= 1;

            if select == 0 {
                continue;
            }

            let zero_ct = pk.ciphertext(i).as_slice();
            for (acc, &v) in data.iter_mut().zip(zero_ct.iter()) {
                *acc = acc.wrapping_add(v);
            }
        }

        // Add the encoded plaintext to the body.
        data[n] = data[n].wrapping_add(plaintext.value);

        let mut ct = LweCiphertext::from_data(data, self.params.ciphertext_modulus_log());
        ct.reduce();
        ct
    }

    /// Convenience: encrypt a *message* (M4-style) under the given public
    /// key.  Internally calls [`Self::encrypt_with_public_key`] after the
    /// encoder maps `message` into the padded plaintext space.
    pub fn encrypt_message_with_public_key(
        &mut self,
        pk: &LwePublicKey,
        encoder: &TfheEncoder,
        message: u64,
    ) -> LweCiphertext {
        let plain = encoder.encode_message(message);
        self.encrypt_with_public_key(pk, plain)
    }
}

/// Symmetric LWE decryptor.
#[derive(Clone, Debug)]
pub struct TfheDecryptor {
    params: TfheParameters,
    sk: LweSecretKey,
}

impl TfheDecryptor {
    pub fn new(params: TfheParameters, sk: LweSecretKey) -> Self {
        Self { params, sk }
    }

    pub fn params(&self) -> &TfheParameters {
        &self.params
    }

    pub fn secret_key(&self) -> &LweSecretKey {
        &self.sk
    }

    /// Recover the noisy plaintext word `b - <a, s>`.  Caller is expected to
    /// pass it through the encoder to recover the message.
    pub fn decrypt_raw(&self, ct: &LweCiphertext) -> u64 {
        debug_assert_eq!(ct.dimension().0, self.params.lwe_dimension().0);
        debug_assert_eq!(ct.log_modulus(), self.params.ciphertext_modulus_log().0);

        let mut inner: u64 = 0;
        for (a, s) in ct.mask().iter().zip(self.sk.data().iter()) {
            inner = inner.wrapping_add(a.wrapping_mul(*s));
        }
        let mut value = ct.body().wrapping_sub(inner);
        if ct.log_modulus() < 64 {
            let mask = (1u64 << ct.log_modulus()) - 1;
            value &= mask;
        }
        value
    }

    /// Decrypt + decode in one step.  Returns the *message* slot value.
    pub fn decrypt_message(&self, ct: &LweCiphertext, encoder: &TfheEncoder) -> u64 {
        let raw = self.decrypt_raw(ct);
        encoder.decode_message(TfhePlaintext::new(raw, ct.log_modulus()))
    }

    /// Decrypt + decode the full plaintext slot (message + carry).
    pub fn decrypt_full(&self, ct: &LweCiphertext, encoder: &TfheEncoder) -> u64 {
        let raw = self.decrypt_raw(ct);
        encoder.decode_full(TfhePlaintext::new(raw, ct.log_modulus()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemes::tfhe::keys::TfheKeyGenerator;
    use rand_core::SeedableRng;
    use silent_params::presets;

    fn fresh_params() -> TfheParameters {
        TfheParameters::new(presets::toy::toy_tfhe_n512()).unwrap()
    }

    #[test]
    fn encrypt_decrypt_zero_recovers_zero() {
        let params = fresh_params();
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([3u8; 32]));
        let sk = keygen.generate_lwe_secret_key();
        let encoder = TfheEncoder::new(params.clone());
        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([4u8; 32]));
        let dec = TfheDecryptor::new(params, sk.clone());

        let ct = enc.encrypt_message(&sk, &encoder, 0);
        assert_eq!(dec.decrypt_message(&ct, &encoder), 0);
    }

    #[test]
    fn encrypt_decrypt_via_public_key_recovers_each_message() {
        let params = fresh_params();
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([10u8; 32]));
        let sk = keygen.generate_lwe_secret_key();
        let pk = keygen.generate_lwe_public_key(&sk);
        let encoder = TfheEncoder::new(params.clone());
        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([11u8; 32]));
        let dec = TfheDecryptor::new(params, sk);

        for m in 0..encoder.params().message_modulus().0 {
            let ct = enc.encrypt_message_with_public_key(&pk, &encoder, m);
            let recovered = dec.decrypt_message(&ct, &encoder);
            assert_eq!(recovered, m, "pubkey round trip failed for {}", m);
        }
    }

    #[test]
    fn encrypt_decrypt_each_message() {
        let params = fresh_params();
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([5u8; 32]));
        let sk = keygen.generate_lwe_secret_key();
        let encoder = TfheEncoder::new(params.clone());
        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([6u8; 32]));
        let dec = TfheDecryptor::new(params, sk.clone());

        for m in 0..encoder.params().message_modulus().0 {
            let ct = enc.encrypt_message(&sk, &encoder, m);
            let recovered = dec.decrypt_message(&ct, &encoder);
            assert_eq!(recovered, m, "round trip failed for {}", m);
        }
    }
}
