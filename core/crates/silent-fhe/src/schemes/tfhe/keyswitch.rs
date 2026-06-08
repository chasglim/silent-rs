//! LWE-to-LWE key switching for TFHE.
//!
//! Convention (matches tfhe-rs and OpenFHE):
//! `KSK[i][j]` is an LWE encryption of `−s_in[i] · 2^(log_q − (j+1)·base_log)`
//! under `sk_out`.  The key-switch then reduces to:
//!
//! ```text
//!   c_out := (0, …, 0, b_in)
//!   for each input coordinate i:
//!     d_{i, .} ← gadget.decompose(a_in[i])
//!     for each level j:
//!       c_out += d_{i, j} · KSK[i][j]      (signed-scalar LWE multiply)
//!   return c_out
//! ```
//!
//! Plugging in `KSK[i][j] ≈ Enc_{sk_out}(−s_in[i] · scale_j)` and observing
//! `Σ_j d_{i, j} · scale_j ≈ a_in[i]` (the gadget identity), one recovers
//! `c_out ≈ Enc_{sk_out}(b_in − ⟨a_in, s_in⟩)` modulo a controlled noise
//! growth bounded by `level · n_in · base² · σ_KSK / 12`.

use silent_params::{
    CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount, NoiseDistribution,
};
use silent_ring::GadgetDecomposition;
use silent_rlwe::{LweCiphertext, LweKeyswitchKey, LweSecretKey};
use silent_utils::rng::SecureRng;

use super::crypto::encrypt_lwe_under_sk;

/// Generate an LWE→LWE key-switch key.
///
/// `sk_in` is the source secret key (typically the big sample-extracted key);
/// `sk_out` is the destination (the small LWE key).
///
/// Both keys must already share the same `log_modulus`.
pub fn build_lwe_keyswitch_key(
    rng: &mut SecureRng,
    sk_in: &LweSecretKey,
    sk_out: &LweSecretKey,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    noise: NoiseDistribution,
    log_modulus: CiphertextModulusLog,
) -> LweKeyswitchKey {
    assert_eq!(sk_in.log_modulus(), log_modulus.0);
    assert_eq!(sk_out.log_modulus(), log_modulus.0);

    let mut ksk = LweKeyswitchKey::new(
        sk_in.dimension(),
        sk_out.dimension(),
        base_log.0,
        level.0,
        log_modulus,
    );

    for (i, s_i) in sk_in.data().iter().enumerate() {
        let block = ksk.block_mut(i);
        for j in 0..level.0 as usize {
            // Encrypted message: -s_i * 2^(log_q - (j+1) * base_log).
            // For log_q == 64 + (j+1)*base_log == 64 the shift saturates to
            // zero, in which case we just encrypt zero — the corresponding
            // level cannot influence the recomposition.
            let shift = (log_modulus.0 as u32)
                .checked_sub((j as u32 + 1) * base_log.0 as u32)
                .expect("validate() rules out base*level > log_q");
            let scaled = if shift >= 64 {
                0
            } else {
                s_i.wrapping_neg().wrapping_shl(shift)
            };
            block[j] = encrypt_lwe_under_sk(rng, sk_out, scaled, noise, log_modulus);
        }
    }
    ksk
}

/// Stateless key switcher: applies an [`LweKeyswitchKey`] to an LWE
/// ciphertext.
#[derive(Clone, Debug)]
pub struct LweKeyswitcher<'a> {
    ksk: &'a LweKeyswitchKey,
    gadget: GadgetDecomposition,
}

impl<'a> LweKeyswitcher<'a> {
    pub fn new(ksk: &'a LweKeyswitchKey) -> Self {
        let gadget = GadgetDecomposition::new(ksk.log_modulus(), ksk.base_log(), ksk.level());
        Self { ksk, gadget }
    }

    /// Apply key switch: returns an LWE ciphertext under `sk_out`.
    pub fn keyswitch(&self, input: &LweCiphertext) -> LweCiphertext {
        debug_assert_eq!(input.dimension().0, self.ksk.input_dimension().0);
        debug_assert_eq!(input.log_modulus(), self.ksk.log_modulus());

        let log_modulus = CiphertextModulusLog(self.ksk.log_modulus());
        let mut out = LweCiphertext::zeros(self.ksk.output_dimension(), log_modulus);
        // Carry the body of the input directly into the output body.
        *out.body_mut() = input.body();

        let level = self.gadget.level() as usize;
        let mut digits = vec![0u64; level];

        for (i, &a_i) in input.mask().iter().enumerate() {
            self.gadget.decompose_into(a_i, &mut digits);
            let block = self.ksk.block(i);
            for (j, digit) in digits.iter().enumerate() {
                if *digit == 0 {
                    continue;
                }
                let mut term = block[j].clone();
                term.scalar_mul_assign(*digit);
                out.add_assign(&term);
            }
        }

        out.reduce();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemes::tfhe::{
        crypto::TfheDecryptor, encoding::TfheEncoder, keys::TfheKeyGenerator,
        params::TfheParameters,
    };
    use rand_core::SeedableRng;
    use silent_params::{LweDimension, presets};
    use silent_utils::rng::SecureRng;

    fn small_params() -> TfheParameters {
        // Keep the dimensions tiny so the key-switch test stays fast on the
        // schoolbook FFT backend.
        let mut p = presets::toy::toy_tfhe_n512();
        p.lwe_dimension = LweDimension(64);
        p.polynomial_size = silent_params::PolynomialSize(64);
        p.glwe_dimension = silent_params::GlweDimension(1);
        TfheParameters::new(p).unwrap()
    }

    #[test]
    fn keyswitch_roundtrip_recovers_message() {
        let params = small_params();
        let encoder = TfheEncoder::new(params.clone());
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([42u8; 32]));
        let mut key_rng = SecureRng::from_seed([43u8; 32]);

        // sk_in: a *bigger* LWE key (acts as if it were the big GLWE-flat key).
        // For the M5 key-switch test we just take any larger random binary key.
        let big_dim = LweDimension(params.lwe_dimension().0 * 2);
        let mut big_data = vec![0u64; big_dim.0];
        crate::schemes::tfhe::sampling::fill_binary(&mut key_rng, &mut big_data);
        let sk_big = LweSecretKey::from_data(big_data, params.ciphertext_modulus_log());

        // sk_out: the small LWE key (size = params.lwe_dimension).
        let sk_small = keygen.generate_lwe_secret_key();

        // KSK from big -> small.
        let ksk = build_lwe_keyswitch_key(
            &mut key_rng,
            &sk_big,
            &sk_small,
            params.ks_base_log(),
            params.ks_level(),
            params.lwe_noise(),
            params.ciphertext_modulus_log(),
        );

        // Encrypt a message under sk_big.
        let mut enc_rng = SecureRng::from_seed([44u8; 32]);
        let plaintext = encoder.encode_message(2);
        let ct_big = encrypt_lwe_under_sk(
            &mut enc_rng,
            &sk_big,
            plaintext.value,
            params.lwe_noise(),
            params.ciphertext_modulus_log(),
        );

        // Apply KS.
        let ks = LweKeyswitcher::new(&ksk);
        let ct_small = ks.keyswitch(&ct_big);

        // Decrypt under sk_small.
        let dec = TfheDecryptor::new(params.clone(), sk_small.clone());
        assert_eq!(dec.decrypt_message(&ct_small, &encoder), 2);
    }

    #[test]
    fn keyswitch_preserves_zero() {
        let params = small_params();
        let encoder = TfheEncoder::new(params.clone());
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([1u8; 32]));
        let mut key_rng = SecureRng::from_seed([2u8; 32]);

        let big_dim = LweDimension(params.lwe_dimension().0 * 2);
        let mut big_data = vec![0u64; big_dim.0];
        crate::schemes::tfhe::sampling::fill_binary(&mut key_rng, &mut big_data);
        let sk_big = LweSecretKey::from_data(big_data, params.ciphertext_modulus_log());
        let sk_small = keygen.generate_lwe_secret_key();
        let ksk = build_lwe_keyswitch_key(
            &mut key_rng,
            &sk_big,
            &sk_small,
            params.ks_base_log(),
            params.ks_level(),
            params.lwe_noise(),
            params.ciphertext_modulus_log(),
        );

        let mut enc_rng = SecureRng::from_seed([3u8; 32]);
        let ct_big = encrypt_lwe_under_sk(
            &mut enc_rng,
            &sk_big,
            0,
            params.lwe_noise(),
            params.ciphertext_modulus_log(),
        );
        let ks = LweKeyswitcher::new(&ksk);
        let ct_small = ks.keyswitch(&ct_big);
        let dec = TfheDecryptor::new(params.clone(), sk_small.clone());
        let _ = encoder; // silence unused if tests change
        assert_eq!(dec.decrypt_full(&ct_small, &TfheEncoder::new(params)), 0);
    }
}
