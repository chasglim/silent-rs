use crate::core::keys::{KeyGenerator, PublicKeyGen};
use crate::schemes::bfv::params::{BfvMulMethod, BfvParameters};
use rand_core::SeedableRng;
use silent_rlwe::KeyGenerator as RlweKeyGen;
use silent_rlwe::{PublicKey, SecretKey};
use silent_utils::rng::SecureRng;

pub struct BfvKeyGenerator {
    inner: RlweKeyGen,
    mul_method: BfvMulMethod,
}

impl BfvKeyGenerator {
    pub fn new(params: BfvParameters) -> Self {
        Self::with_rng(params, SecureRng::from_entropy())
    }

    pub fn with_rng(params: BfvParameters, rng: SecureRng) -> Self {
        Self {
            inner: RlweKeyGen::new(params.runtime_params().clone(), rng),
            mul_method: params.mul_method(),
        }
    }

    fn resolve_mul_method(&self) -> BfvMulMethod {
        match std::env::var("SILENT_BFV_MUL_METHOD") {
            Ok(value) => match value.to_ascii_lowercase().as_str() {
                "behz" => BfvMulMethod::Behz,
                "hps" => BfvMulMethod::Hps,
                "hpspoverq" | "hps_poverq" => BfvMulMethod::HpsPoverq,
                "hpspoverqleveled" | "hps_poverq_leveled" | "hpspoverq_leveled" => {
                    BfvMulMethod::HpsPoverqLeveled
                }
                _ => self.mul_method,
            },
            Err(_) => self.mul_method,
        }
    }

    pub fn relinearization_key(
        &mut self,
        sk: &SecretKey,
    ) -> Result<silent_rlwe::EvaluationKey, silent_math::rns::RnsError> {
        if matches!(self.resolve_mul_method(), BfvMulMethod::Hps) {
            self.inner.relinearization_key_hps(sk)
        } else {
            self.inner.relinearization_key(sk)
        }
    }

    pub fn galois_keys(
        &mut self,
        sk: &SecretKey,
        substitution_indices: &[u32],
    ) -> Result<silent_rlwe::GaloisKey, silent_math::rns::RnsError> {
        if matches!(self.resolve_mul_method(), BfvMulMethod::Hps) {
            self.inner.galois_keys_hps(sk, substitution_indices)
        } else {
            self.inner.galois_keys(sk, substitution_indices)
        }
    }
}

impl KeyGenerator for BfvKeyGenerator {
    type SecretKey = SecretKey;

    fn generate_secret_key(&mut self) -> SecretKey {
        self.inner.secret_key()
    }
}

impl PublicKeyGen for BfvKeyGenerator {
    type PublicKey = PublicKey;

    fn generate_public_key(&mut self, sk: &SecretKey) -> PublicKey {
        self.inner.public_key(sk)
    }
}
