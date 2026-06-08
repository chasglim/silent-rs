use crate::HssCiphertext;
use crate::HssContext;
use crate::utils::{sample_cbd_poly, sample_ternary_poly};
use silent_math::rns::RnsError;
use silent_ring::Poly;
use silent_rlwe::{Ciphertext, PublicKey};
use silent_utils::rng::SecureRng;

pub struct HssEncryptor {
    context: HssContext,
    pk: PublicKey,
    delta: Vec<u64>,
    rng: SecureRng,
}

impl HssEncryptor {
    pub fn new(context: HssContext, pk: PublicKey, rng: SecureRng) -> Result<Self, RnsError> {
        let params = context.runtime_params();
        let plain_modulus = context.plain_modulus();
        let base_q = params.ring.rns();
        let q_big = base_q.base_prod();
        let (delta_big, _) = q_big.div_mod_u64(plain_modulus);

        let moduli = base_q.moduli();
        let mut delta = Vec::with_capacity(moduli.len());
        for modulus in moduli {
            let val = delta_big.mod_u64(modulus.value());
            delta.push(val);
        }

        Ok(Self {
            context,
            pk,
            delta,
            rng,
        })
    }

    /// Encrypts input message m into HSS Ciphertext.
    /// HSS.Enc(pk, m) = (Enc(m), Enc_OKDM(m)) = (Enc(m), Enc(m*s))
    pub fn encrypt(&mut self, m: u64) -> Result<HssCiphertext, RnsError> {
        // 1. Encrypt m (Standard LPR) -> enc_m
        let enc_m = self.encrypt_asymmetric(m, false)?;

        // 2. Encrypt m*s (OKDM) -> enc_m_s
        // OKDM: Encrypt 0, but add Delta*m to component 1 (which multiplies s during decryption).
        let enc_m_s = self.encrypt_asymmetric(m, true)?;

        Ok(HssCiphertext {
            enc_m,
            enc_m_times_s: enc_m_s,
        })
    }

    /// Encrypts a plaintext polynomial (e.g. SIMD encoded message).
    /// poly_p is a polynomial in R_p (Inverse NTT form, i.e. Coeffs).
    pub fn encrypt_input_poly(&mut self, poly_p: &Poly) -> Result<HssCiphertext, RnsError> {
        // Same logic but with polynomial message

        // 1. Standard LPR Enc(poly_p)
        let enc_m = self.encrypt_poly_asymmetric(poly_p, false)?;

        // 2. OKDM Enc(poly_p * s)
        let enc_m_s = self.encrypt_poly_asymmetric(poly_p, true)?;

        Ok(HssCiphertext {
            enc_m,
            enc_m_times_s: enc_m_s,
        })
    }

    /// Standard LPR Encryption or OKDM Encryption.
    /// if okdm_mode is false: Encrypts m. Result decrypts to m. (m added to c0).
    /// if okdm_mode is true: Encrypts m*s. Result decrypts to m*s. (m added to c1).
    fn encrypt_asymmetric(&mut self, m: u64, okdm_mode: bool) -> Result<Ciphertext, RnsError> {
        let params = self.context.runtime_params();
        let ring = &params.ring;
        let moduli = ring.rns().moduli();
        let degree = ring.degree();
        let num_moduli = moduli.len();

        // 1. Sample u, e1, e2
        let mut u = sample_ternary_poly(&mut self.rng, degree, num_moduli, moduli);
        let mut e1 = sample_cbd_poly(&mut self.rng, degree, num_moduli, moduli);
        let mut e2 = sample_cbd_poly(&mut self.rng, degree, num_moduli, moduli);

        // 2. Transfrom to NTT
        u.ntt_forward(ring);
        e1.ntt_forward(ring);
        e2.ntt_forward(ring);

        // 3. pk = (b, a)
        // c0 = b*u + e1 + Delta*m (if !okdm)
        // c1 = a*u + e2 + Delta*m (if okdm)

        let pk_ct = &self.pk.pk;
        let b = &pk_ct.data[0];
        let a = &pk_ct.data[1];

        // c0 calculation
        let mut c0 = b.clone();
        c0.mul_assign(&u, ring);
        c0.add_assign(&e1, ring);

        // c1 calculation
        let mut c1 = a.clone();
        c1.mul_assign(&u, ring);
        c1.add_assign(&e2, ring);

        // Add message
        let m_poly = self.create_message_poly(m);

        if !okdm_mode {
            // Standard Enc: add to c0
            c0.add_assign(&m_poly, ring);
        } else {
            // OKDM Enc (Enc of m*s): add to c1
            c1.add_assign(&m_poly, ring);
        }

        Ok(Ciphertext::new(vec![c0, c1], params.clone(), true))
    }

    fn encrypt_poly_asymmetric(
        &mut self,
        poly_p: &Poly,
        okdm_mode: bool,
    ) -> Result<Ciphertext, RnsError> {
        let params = self.context.runtime_params();
        let ring = &params.ring;
        let moduli = ring.rns().moduli();
        let degree = ring.degree();
        let num_moduli = moduli.len();

        // 1. Sample u, e1, e2
        let mut u = sample_ternary_poly(&mut self.rng, degree, num_moduli, moduli);
        let mut e1 = sample_cbd_poly(&mut self.rng, degree, num_moduli, moduli);
        let mut e2 = sample_cbd_poly(&mut self.rng, degree, num_moduli, moduli);

        // 2. Transfrom to NTT
        u.ntt_forward(ring);
        e1.ntt_forward(ring);
        e2.ntt_forward(ring);

        // 3. pk = (b, a)
        let pk_ct = &self.pk.pk;
        let b = &pk_ct.data[0];
        let a = &pk_ct.data[1];

        // c0 calculation
        let mut c0 = b.clone();
        c0.mul_assign(&u, ring);
        c0.add_assign(&e1, ring);

        // c1 calculation
        let mut c1 = a.clone();
        c1.mul_assign(&u, ring);
        c1.add_assign(&e2, ring);

        // Add message (Lift poly_p to Delta * poly_p in NTT)
        // Note: poly_p is in Coeff domain (mod p).
        let m_poly = self.lift_message_poly(poly_p)?;

        if !okdm_mode {
            // Standard Enc: add to c0
            c0.add_assign(&m_poly, ring);
        } else {
            // OKDM Enc: add to c1
            c1.add_assign(&m_poly, ring);
        }

        Ok(Ciphertext::new(vec![c0, c1], params.clone(), true))
    }

    fn create_message_poly(&self, m: u64) -> Poly {
        let ring = &self.context.runtime_params().ring;
        let moduli = ring.rns().moduli();
        let degree = ring.degree();

        // Constant poly Delta * m
        let mut p = unsafe { Poly::new_uninit(degree, moduli.len()) };

        // Pre-compute scalar Delta*m mod q_i
        for (i, modulus) in moduli.iter().enumerate() {
            let m_mod = m % modulus.value();
            let delta_mod = self.delta[i];
            let val = silent_math::arith::mul_mod_u64(delta_mod, m_mod, modulus.value());
            p.limb_mut(i).fill(val);
        }
        p
    }

    fn lift_message_poly(&self, poly_p: &Poly) -> Result<Poly, RnsError> {
        let ring = &self.context.runtime_params().ring;
        let degree = ring.degree();
        let moduli = ring.rns().moduli();
        let num_moduli = moduli.len();

        let mut p_lifted = unsafe { Poly::new_uninit(degree, num_moduli) };
        let p_limb = poly_p.limb(0);

        for i in 0..degree {
            let m_val = p_limb[i];
            for j in 0..num_moduli {
                let q = moduli[j].value();
                let delta = self.delta[j];
                let val = silent_math::arith::mul_mod_u64(m_val, delta, q);
                p_lifted.limb_mut(j)[i] = val;
            }
        }
        p_lifted.ntt_forward(ring);
        Ok(p_lifted)
    }
}
