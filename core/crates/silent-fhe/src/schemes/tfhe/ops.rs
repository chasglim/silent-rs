//! Homomorphic linear operations on TFHE LWE ciphertexts.
//!
//! M4 ships only the closed-form linear primitives (no PBS):
//!
//! ```text
//!   ct_a + ct_b      Add(LWE, LWE) -> LWE
//!   ct_a - ct_b      Sub(LWE, LWE) -> LWE
//!   - ct             Neg(LWE)       -> LWE
//!   k * ct (small k) ScalarMul(u64, LWE) -> LWE
//!   ct + Δ·μ         AddPlain(LWE, plaintext) -> LWE
//! ```
//!
//! Programmable bootstrap (which would let us implement
//! [`HeEvaluator::mul`]) lands in M5.  Until then the BFV-style
//! `mul(ct_a, ct_b)` is intentionally *not* implemented for TFHE — the
//! evaluator instead exposes a `mul_unsupported_in_mvp` helper that returns
//! an error so callers cannot accidentally use the wrong primitive.

use silent_math::fft64::{FftMul, NegacyclicMul};
use silent_rlwe::{LweBootstrapKey, LweCiphertext, LweKeyswitchKey};
use thiserror::Error;

use super::bootstrap::TfheBootstrapper;
use super::bootstrap_fft::LweBootstrapKeyFft;
use super::encoding::{TfheEncoder, TfhePlaintext};
use super::params::TfheParameters;

/// Errors produced by the TFHE linear evaluator.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TfheEvalError {
    #[error("ciphertexts must share the TFHE evaluator's modulus and dimension")]
    ShapeMismatch,
    #[error(
        "homomorphic ciphertext-ciphertext multiplication requires a programmable bootstrap; use TfheEvaluator::apply_lookup_table once available in M5"
    )]
    CiphertextMulRequiresPbs,
}

/// Linear-algebra evaluator for TFHE LWE ciphertexts.
#[derive(Clone, Debug)]
pub struct TfheEvaluator {
    params: TfheParameters,
}

impl TfheEvaluator {
    pub fn new(params: TfheParameters) -> Self {
        Self { params }
    }

    pub fn params(&self) -> &TfheParameters {
        &self.params
    }

    fn ensure_shape(&self, ct: &LweCiphertext) -> Result<(), TfheEvalError> {
        if ct.dimension().0 != self.params.lwe_dimension().0
            || ct.log_modulus() != self.params.ciphertext_modulus_log().0
        {
            return Err(TfheEvalError::ShapeMismatch);
        }
        Ok(())
    }

    /// Sum of two LWE ciphertexts.  Adds noises additively.
    pub fn add(
        &self,
        a: &LweCiphertext,
        b: &LweCiphertext,
    ) -> Result<LweCiphertext, TfheEvalError> {
        self.ensure_shape(a)?;
        self.ensure_shape(b)?;
        let mut out = a.clone();
        out.add_assign(b);
        Ok(out)
    }

    /// In-place sum.
    pub fn add_assign(
        &self,
        a: &mut LweCiphertext,
        b: &LweCiphertext,
    ) -> Result<(), TfheEvalError> {
        self.ensure_shape(a)?;
        self.ensure_shape(b)?;
        a.add_assign(b);
        Ok(())
    }

    /// Difference of two LWE ciphertexts.
    pub fn sub(
        &self,
        a: &LweCiphertext,
        b: &LweCiphertext,
    ) -> Result<LweCiphertext, TfheEvalError> {
        self.ensure_shape(a)?;
        self.ensure_shape(b)?;
        let mut out = a.clone();
        out.sub_assign(b);
        Ok(out)
    }

    /// Negate a ciphertext (in-place).
    pub fn negate(&self, a: &mut LweCiphertext) -> Result<(), TfheEvalError> {
        self.ensure_shape(a)?;
        a.negate_assign();
        Ok(())
    }

    /// Multiply every coefficient by a scalar.  The plaintext value scales
    /// the same way (provided the scaled message stays within the carry
    /// buffer); the noise scales linearly so callers must keep an eye on
    /// [`TfheParameters::carry_modulus`].
    pub fn scalar_mul(
        &self,
        a: &LweCiphertext,
        scalar: u64,
    ) -> Result<LweCiphertext, TfheEvalError> {
        self.ensure_shape(a)?;
        let mut out = a.clone();
        out.scalar_mul_assign(scalar);
        Ok(out)
    }

    /// Add a constant plaintext to a ciphertext (only the body is mutated).
    pub fn add_plain(
        &self,
        a: &LweCiphertext,
        plain: TfhePlaintext,
    ) -> Result<LweCiphertext, TfheEvalError> {
        self.ensure_shape(a)?;
        if plain.log_modulus != self.params.ciphertext_modulus_log().0 {
            return Err(TfheEvalError::ShapeMismatch);
        }
        let mut out = a.clone();
        out.body_add_assign(plain.value);
        Ok(out)
    }

    /// Reserved entry point for ciphertext-ciphertext multiplication.  TFHE
    /// realises ct·ct as `apply_lookup_table(c1 + c2 · message_modulus, f)`
    /// where `f(packed) = msg(packed) · carry(packed)`; the higher-level
    /// scheme caller is expected to compose [`Self::apply_lookup_table`] with
    /// scalar additions itself.  We leave the trait error here so that any
    /// premature call surfaces clearly.
    pub fn mul(
        &self,
        _a: &LweCiphertext,
        _b: &LweCiphertext,
    ) -> Result<LweCiphertext, TfheEvalError> {
        Err(TfheEvalError::CiphertextMulRequiresPbs)
    }

    /// Apply a lookup table `f` to an LWE ciphertext via TFHE programmable
    /// bootstrapping.  Requires the bootstrap key (`bsk`) and the LWE→LWE
    /// key-switch key (`ksk`) generated under the same parameters.
    ///
    /// Uses the [`silent_math::fft64::FftMul`] backend (see
    /// [`super::bootstrap::TfheBootstrapper::apply_lookup_table`]).  Use
    /// [`Self::apply_lookup_table_with`] to supply [`SchoolbookMul`],
    /// [`silent_math::fft64::KaratsubaMul`], or another [`NegacyclicMul`].
    ///
    /// For workloads that run many PBS operations on the same key material,
    /// build a [`LweBootstrapKeyFft`] once ([`TfheBootstrapper::build_fft_bsk_cache`])
    /// and call [`Self::apply_lookup_table_fft`] to skip redundant BSK forward FFTs.
    pub fn apply_lookup_table<F: FnMut(u64) -> u64>(
        &self,
        encoder: &TfheEncoder,
        bsk: &LweBootstrapKey,
        ksk: &LweKeyswitchKey,
        ct: &LweCiphertext,
        f: F,
    ) -> Result<LweCiphertext, TfheEvalError> {
        self.ensure_shape(ct)?;
        let bootstrapper = TfheBootstrapper::new(self.params.clone(), bsk, ksk);
        Ok(bootstrapper.apply_lookup_table(encoder, ct, f))
    }

    /// Backend-pluggable variant of [`Self::apply_lookup_table`].
    pub fn apply_lookup_table_with<F, M>(
        &self,
        encoder: &TfheEncoder,
        bsk: &LweBootstrapKey,
        ksk: &LweKeyswitchKey,
        ct: &LweCiphertext,
        f: F,
        backend: &M,
    ) -> Result<LweCiphertext, TfheEvalError>
    where
        F: FnMut(u64) -> u64,
        M: NegacyclicMul,
    {
        self.ensure_shape(ct)?;
        let bootstrapper = TfheBootstrapper::new(self.params.clone(), bsk, ksk);
        Ok(bootstrapper.apply_lookup_table_with(encoder, ct, f, backend))
    }

    /// Programmable bootstrap using a pre-built [`LweBootstrapKeyFft`].
    ///
    /// Bit-exact with [`Self::apply_lookup_table`] / [`TfheBootstrapper::apply_lookup_table_with`]
    /// when passed the same [`FftMul`] that was used to construct `bsk_fft`.
    /// The `bsk` argument must be the same [`LweBootstrapKey`] that `bsk_fft`
    /// was derived from (it is only used to build the internal
    /// [`TfheBootstrapper`] for parameter / keyswitch consistency).
    pub fn apply_lookup_table_fft<F: FnMut(u64) -> u64>(
        &self,
        encoder: &TfheEncoder,
        bsk: &LweBootstrapKey,
        ksk: &LweKeyswitchKey,
        bsk_fft: &LweBootstrapKeyFft,
        fft: &FftMul,
        ct: &LweCiphertext,
        f: F,
    ) -> Result<LweCiphertext, TfheEvalError> {
        self.ensure_shape(ct)?;
        let bootstrapper = TfheBootstrapper::new(self.params.clone(), bsk, ksk);
        Ok(bootstrapper.apply_lookup_table_fft(encoder, bsk_fft, fft, ct, f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemes::tfhe::{
        crypto::{TfheDecryptor, TfheEncryptor},
        encoding::TfheEncoder,
        keys::TfheKeyGenerator,
    };
    use rand_core::SeedableRng;
    use silent_params::presets;
    use silent_utils::rng::SecureRng;

    fn setup() -> (TfheParameters, TfheEncoder, TfheKeyGenerator) {
        let params = TfheParameters::new(presets::toy::toy_tfhe_n512()).unwrap();
        let encoder = TfheEncoder::new(params.clone());
        let keygen = TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([10u8; 32]));
        (params, encoder, keygen)
    }

    #[test]
    fn shape_errors_are_caught() {
        let (params, _enc, _kg) = setup();
        let evaluator = TfheEvaluator::new(params.clone());
        let small = LweCiphertext::zeros(
            silent_params::LweDimension(8),
            silent_params::CiphertextModulusLog(64),
        );
        assert_eq!(
            evaluator.add(&small, &small),
            Err(TfheEvalError::ShapeMismatch)
        );
    }

    #[test]
    fn mul_returns_pbs_required_error() {
        let (params, _enc, _kg) = setup();
        let evaluator = TfheEvaluator::new(params.clone());
        let z = LweCiphertext::zeros(params.lwe_dimension(), params.ciphertext_modulus_log());
        assert_eq!(
            evaluator.mul(&z, &z),
            Err(TfheEvalError::CiphertextMulRequiresPbs)
        );
    }

    #[test]
    fn homomorphic_add_recovers_sum() {
        let (params, encoder, mut keygen) = setup();
        let sk = keygen.generate_lwe_secret_key();
        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([11u8; 32]));
        let dec = TfheDecryptor::new(params.clone(), sk.clone());
        let evaluator = TfheEvaluator::new(params.clone());

        let m1 = 1u64;
        let m2 = 2u64;
        let ct1 = enc.encrypt_message(&sk, &encoder, m1);
        let ct2 = enc.encrypt_message(&sk, &encoder, m2);
        let sum = evaluator.add(&ct1, &ct2).unwrap();
        // Carry buffer absorbs the addition, so decode at the full slot first
        // to confirm the carry is in fact set, then via the message slot.
        assert_eq!(dec.decrypt_full(&sum, &encoder), m1 + m2);
        assert_eq!(
            dec.decrypt_message(&sum, &encoder),
            (m1 + m2) % encoder.params().message_modulus().0
        );
    }

    #[test]
    fn homomorphic_sub_handles_wraparound() {
        let (params, encoder, mut keygen) = setup();
        let sk = keygen.generate_lwe_secret_key();
        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([12u8; 32]));
        let dec = TfheDecryptor::new(params.clone(), sk.clone());
        let evaluator = TfheEvaluator::new(params.clone());

        // Use the *full* slot (message + carry) so subtraction stays inside the
        // unsigned plaintext range and we get an unambiguous integer result.
        let p1 = encoder.encode_full(7);
        let p2 = encoder.encode_full(5);
        let ct1 = enc.encrypt_symmetric(&sk, p1);
        let ct2 = enc.encrypt_symmetric(&sk, p2);
        let diff = evaluator.sub(&ct1, &ct2).unwrap();
        assert_eq!(dec.decrypt_full(&diff, &encoder), 2);
    }

    #[test]
    fn negate_then_add_is_subtract() {
        let (params, encoder, mut keygen) = setup();
        let sk = keygen.generate_lwe_secret_key();
        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([13u8; 32]));
        let dec = TfheDecryptor::new(params.clone(), sk.clone());
        let evaluator = TfheEvaluator::new(params.clone());

        let ct1 = enc.encrypt_message(&sk, &encoder, 3);
        let mut ct2 = enc.encrypt_message(&sk, &encoder, 1);
        evaluator.negate(&mut ct2).unwrap();
        let sum = evaluator.add(&ct1, &ct2).unwrap();
        // 3 + (-1) mod message_modulus = 2.
        assert_eq!(dec.decrypt_message(&sum, &encoder), 2);
    }

    #[test]
    fn scalar_mul_inside_carry_buffer() {
        let (params, encoder, mut keygen) = setup();
        let sk = keygen.generate_lwe_secret_key();
        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([14u8; 32]));
        let dec = TfheDecryptor::new(params.clone(), sk.clone());
        let evaluator = TfheEvaluator::new(params.clone());

        let ct = enc.encrypt_message(&sk, &encoder, 1);
        let scaled = evaluator.scalar_mul(&ct, 3).unwrap();
        // 1 * 3 = 3 fits in the carry buffer of toy_tfhe (carry_modulus = 4).
        assert_eq!(dec.decrypt_full(&scaled, &encoder), 3);
    }

    #[test]
    fn add_plain_only_touches_body() {
        let (params, encoder, mut keygen) = setup();
        let sk = keygen.generate_lwe_secret_key();
        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([15u8; 32]));
        let dec = TfheDecryptor::new(params.clone(), sk.clone());
        let evaluator = TfheEvaluator::new(params.clone());

        let ct = enc.encrypt_message(&sk, &encoder, 1);
        let constant = encoder.encode_message(2);
        let combined = evaluator.add_plain(&ct, constant).unwrap();
        assert_eq!(dec.decrypt_message(&combined, &encoder), 3);
    }
}
