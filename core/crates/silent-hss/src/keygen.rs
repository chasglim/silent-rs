use crate::HssContext;
use crate::HssEvalKey;
use crate::utils::sample_uniform_poly;
use rand::RngCore;
use silent_rlwe::{KeyGenerator, PublicKey, SecretKey};
use silent_utils::rng::SecureRng;

pub struct HssKeyGenerator {
    rlwe_keygen: KeyGenerator,
    context: HssContext,
    rng: SecureRng,
}

impl HssKeyGenerator {
    pub fn new(context: HssContext, rng: SecureRng) -> Self {
        Self {
            rlwe_keygen: KeyGenerator::new(context.runtime_params().clone(), rng.clone()),
            context,
            rng,
        }
    }

    /// Generates a secret key share.
    pub fn generate_secret_key(&mut self) -> SecretKey {
        self.rlwe_keygen.secret_key()
    }

    /// Generates a public key from the secret key.
    pub fn generate_public_key(&mut self, sk: &SecretKey) -> PublicKey {
        self.rlwe_keygen.public_key(sk)
    }

    /// Splits a secret key into two additive shares for HSS.
    /// Returns (share1, share2).
    ///
    /// Split `(1, s)` into two additive shares in NTT domain:
    /// `(b0, s0) + (b1, s1) = (1, s)`.
    pub fn split_secret_key(&mut self, sk: &SecretKey) -> (crate::HssShare, crate::HssShare) {
        let ring = &self.context.runtime_params().ring;
        let s_ntt = &sk.value;
        let degree = s_ntt.degree();
        let num_moduli = s_ntt.num_moduli();
        let moduli = ring.rns().moduli();

        // Random additive sharing of the first component (1 in NTT domain).
        let mut b0 = sample_uniform_poly(&mut self.rng, degree, num_moduli, moduli);
        b0.ntt_forward(ring);
        let mut b1 = Self::constant_poly(1, degree, num_moduli);
        b1.sub_assign(&b0, ring);

        // Random additive sharing of the second component (s in NTT domain).
        let mut s0 = sample_uniform_poly(&mut self.rng, degree, num_moduli, moduli);
        s0.ntt_forward(ring);
        let mut s1 = s_ntt.clone();
        s1.sub_assign(&s0, ring);

        let share1 = crate::HssShare::new(b0, s0);
        let share2 = crate::HssShare::new(b1, s1);

        (share1, share2)
    }

    /// Generates two paper-style evaluation keys `(ek_0, ek_1)`.
    /// Each key contains one secret-key share and shared PRF key `K`.
    pub fn generate_eval_keys(&mut self, sk: &SecretKey) -> (HssEvalKey, HssEvalKey) {
        let (share0, share1) = self.split_secret_key(sk);
        let mut prf_key = [0u8; 32];
        self.rng.fill_bytes(&mut prf_key);

        (
            HssEvalKey::new(share0, 0, prf_key),
            HssEvalKey::new(share1, 1, prf_key),
        )
    }

    fn constant_poly(val: u64, degree: usize, num_moduli: usize) -> silent_ring::Poly {
        let mut p = unsafe { silent_ring::Poly::new_uninit(degree, num_moduli) };
        for i in 0..num_moduli {
            let limb = p.limb_mut(i);
            limb.fill(val);
        }
        p
    }
}
