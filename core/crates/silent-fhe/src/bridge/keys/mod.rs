//! Bridge key material.
//!
//! Bridge key material.

pub mod b2t_ksk;
pub mod fmod_keys;
pub mod t2b_repack_key;

pub use b2t_ksk::BfvToTfheKsk as BfvToTfheKey;
pub use fmod_keys::BfvFmodKeys;
pub use t2b_repack_key::TfheToBfvRepackKey as TfheToBfvKey;

use crate::bridge::tfhe_to_bfv::fmod;
use crate::bridge::tfhe_to_bfv::fmod_precompute;
use crate::bridge::tfhe_to_bfv::matrix;
use crate::bridge::tfhe_to_bfv::torus_decode;
use crate::bridge::tfhe_to_bfv::{TfheToBfvFunctionalKey, generate_tfhe_to_bfv_functional_key};
use crate::schemes::bfv::crypto::BfvEncryptor;
use crate::schemes::bfv::keys::BfvKeyGenerator;
use crate::schemes::tfhe::keyswitch::build_lwe_keyswitch_key;
use rand_core::SeedableRng;
use silent_rlwe::{LweSecretKey, SecretKey};
use silent_utils::rng::SecureRng;

use super::params::BridgeParams;

#[derive(Clone, Debug)]
pub struct BridgeKeys {
    pub bfv_to_tfhe: BfvToTfheKey,
    pub tfhe_to_bfv: TfheToBfvKey,
    pub fmod_keys: BfvFmodKeys,
    pub bfv_relin_key: Option<silent_rlwe::EvaluationKey>,
    pub bfv_galois_keys: Option<silent_rlwe::GaloisKey>,
    pub coeff_to_slot_matrix: Option<Vec<Vec<u64>>>,
    pub slot_to_coeff_matrix: Option<Vec<Vec<u64>>>,
}

pub struct BridgeKeyGenerator {
    params: BridgeParams,
    rng: SecureRng,
}

impl BridgeKeyGenerator {
    pub fn new(params: BridgeParams) -> Self {
        Self::with_rng(params, SecureRng::from_entropy())
    }

    pub fn with_rng(params: BridgeParams, rng: SecureRng) -> Self {
        Self { params, rng }
    }

    pub fn params(&self) -> &BridgeParams {
        &self.params
    }

    pub fn generate_bfv_to_tfhe_key(
        &mut self,
        bfv_sk: &SecretKey,
        tfhe_sk: &LweSecretKey,
    ) -> BfvToTfheKey {
        let sk_bfv_lwe = bfv_secret_key_as_lwe(bfv_sk, &self.params);
        let log_q = self.params.tfhe_params.ciphertext_modulus_log();
        let ksk = build_lwe_keyswitch_key(
            &mut self.rng,
            &sk_bfv_lwe,
            tfhe_sk,
            self.params.tfhe_params.ks_base_log(),
            self.params.tfhe_params.ks_level(),
            self.params.tfhe_params.lwe_noise(),
            log_q,
        );
        BfvToTfheKey { ksk }
    }

    pub fn generate_tfhe_to_bfv_key(
        &mut self,
        bfv_sk: &SecretKey,
        tfhe_sk: &LweSecretKey,
    ) -> TfheToBfvKey {
        let degree = self.params.bfv_params.degree();
        let mut encryptor = BfvEncryptor::new(self.params.bfv_params.clone(), self.rng.clone());
        let mut rk = Vec::with_capacity(tfhe_sk.data().len());
        for &coeff in tfhe_sk.data() {
            let mut poly = silent_ring::Poly::new(degree, 1);
            poly.limb_mut(0)[0] = coeff % self.params.bfv_params.plain_modulus();
            rk.push(encryptor.encrypt_symmetric(bfv_sk, &silent_rlwe::Plaintext { value: poly }));
        }
        TfheToBfvKey { rk }
    }

    pub fn generate_tfhe_to_bfv_functional_key(
        &mut self,
        bfv_sk: &SecretKey,
        tfhe_sk: &LweSecretKey,
    ) -> Result<TfheToBfvFunctionalKey, super::BridgeError> {
        generate_tfhe_to_bfv_functional_key(&mut self.rng, &self.params, bfv_sk, tfhe_sk)
    }

    pub fn generate(&mut self, bfv_sk: &SecretKey, tfhe_sk: &LweSecretKey) -> BridgeKeys {
        let mut bfv_keygen =
            BfvKeyGenerator::with_rng(self.params.bridge_bfv_eval_params(), self.rng.clone());
        let bfv_relin_key = bfv_keygen.relinearization_key(bfv_sk).ok();

        // Slot-domain bridge transforms use the actual BatchEncoder Galois
        // permutations. Generate the full odd automorphism set for the
        // correctness-first S2C/C2S path; optimized BSGS subsets can be added
        // once their permutation model is audited.
        let n = self.params.bfv_params.degree();
        let indices = slot_transform_galois_elements(n);
        let bfv_galois_keys = Some(
            bfv_keygen
                .galois_keys(bfv_sk, &indices)
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to generate BFV Galois keys for bridge slot transforms (n={n}, indices_len={}): {err:?}",
                        indices.len()
                    )
                }),
        );

        let t = self.params.bfv_params.plain_modulus();
        let coeff_to_slot_matrix = matrix::generate_coeff_to_slot_matrix(n, t);
        let slot_to_coeff_matrix = matrix::generate_slot_to_coeff_matrix(n, t);

        let (fmod_coeffs, center_decode_scalar) =
            if torus_decode::ensure_torus_decode_preconditions(&self.params).is_ok() {
                let coeffs = fmod_precompute::precompute_fmod_lagrange(&self.params)
                    .unwrap_or_else(|err| {
                        panic!("failed to precompute BFV Fmod coefficients for bridge: {err}")
                    });
                let scalar = fmod::center_decode_scalar(&self.params).unwrap_or_else(|err| {
                    panic!("failed to precompute BFV Fmod center decode scalar for bridge: {err}")
                });
                (coeffs, Some(scalar))
            } else {
                (Vec::new(), None)
            };
        let fmod_keys = BfvFmodKeys {
            coeffs: fmod_coeffs,
            center_decode_scalar,
            periodic_center_coeffs: Vec::new(),
        };

        BridgeKeys {
            bfv_to_tfhe: self.generate_bfv_to_tfhe_key(bfv_sk, tfhe_sk),
            tfhe_to_bfv: self.generate_tfhe_to_bfv_key(bfv_sk, tfhe_sk),
            fmod_keys,
            bfv_relin_key,
            bfv_galois_keys,
            coeff_to_slot_matrix,
            slot_to_coeff_matrix,
        }
    }
}

fn slot_transform_galois_elements(n: usize) -> Vec<u32> {
    (1..(2 * n) as u32).step_by(2).filter(|&g| g != 1).collect()
}

/// Interpret the BFV secret-key coefficients as an LWE secret key in `Z_2^k`.
pub fn bfv_secret_key_as_lwe(bfv_sk: &SecretKey, params: &BridgeParams) -> LweSecretKey {
    // BFV secret key is generated in coefficient domain and then NTT-forwarded.
    // We need to inverse it back to get coefficient domain ternary values.
    let mut sk_coeffs = bfv_sk.value.clone();
    sk_coeffs.ntt_inverse(params.bfv_params.ring());

    let _log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let q0 = params.bfv_params.ring().rns().moduli()[0].value();
    let lwe_data = sk_coeffs
        .limb(0)
        .iter()
        .take(params.bfv_params.degree())
        .map(|&coeff| {
            let centered = if coeff > q0 / 2 {
                -((q0 - coeff) as i64)
            } else {
                coeff as i64
            };
            centered.wrapping_neg() as u64
        })
        .collect::<Vec<u64>>();

    LweSecretKey::from_data(lwe_data, params.tfhe_params.ciphertext_modulus_log())
}
