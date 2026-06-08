use crate::schemes::bgv::ciphertext::{BgvCiphertext, BgvPlaintextFactor};
use crate::schemes::bgv::params::{BgvLevelError, BgvParameters};
use crate::schemes::bgv::rns_big;
use num_bigint::BigUint;
use num_traits::{Signed, ToPrimitive, Zero};
use silent_math::arith;
use silent_ring::Poly;
use silent_rlwe::{Ciphertext, Encryptor, Plaintext, PublicKey, SecretKey};
use silent_utils::rng::SecureRng;

pub struct BgvEncryptor {
    params: BgvParameters,
    encryptor: Encryptor,
}

impl BgvEncryptor {
    pub fn new(params: BgvParameters, rng: SecureRng) -> Self {
        let encryptor = Encryptor::new(params.runtime_params().clone(), rng);
        Self { params, encryptor }
    }

    pub fn encrypt_symmetric(&mut self, sk: &SecretKey, plaintext: &Plaintext) -> Ciphertext {
        let mut ct = self
            .encryptor
            .encrypt_zero_symmetric_scaled_error(sk, self.params.plain_modulus());
        self.add_plaintext_to_c0_ntt(&mut ct, plaintext);

        for poly in ct.data.iter_mut() {
            poly.ntt_inverse(self.params.ring());
        }
        ct.is_ntt = false;
        ct
    }

    pub fn encrypt_symmetric_bgv(
        &mut self,
        sk: &SecretKey,
        plaintext: &Plaintext,
    ) -> BgvCiphertext {
        let raw = self.encrypt_symmetric(sk, plaintext);
        BgvCiphertext::new(self.params.clone(), raw).expect("fresh BGV ciphertext shape is valid")
    }

    pub fn encrypt_public(&mut self, pk: &PublicKey, plaintext: &Plaintext) -> Ciphertext {
        let mut ct = self
            .encryptor
            .encrypt_zero_public_scaled_error(pk, self.params.plain_modulus());
        self.add_plaintext_to_c0_ntt(&mut ct, plaintext);

        for poly in ct.data.iter_mut() {
            poly.ntt_inverse(self.params.ring());
        }
        ct.is_ntt = false;
        ct
    }

    pub fn encrypt_public_bgv(&mut self, pk: &PublicKey, plaintext: &Plaintext) -> BgvCiphertext {
        let raw = self.encrypt_public(pk, plaintext);
        BgvCiphertext::new(self.params.clone(), raw).expect("fresh BGV ciphertext shape is valid")
    }

    pub fn encrypt_public_with_error_scalar(
        &mut self,
        pk: &PublicKey,
        plaintext: &Plaintext,
        error_scalar: u64,
    ) -> Result<Ciphertext, BgvLevelError> {
        if error_scalar % self.params.plain_modulus() != 0 {
            return Err(
                BgvLevelError::BootstrapFusionErrorScaleNotPlaintextMultiple {
                    error_scalar,
                    plain_modulus: self.params.plain_modulus(),
                },
            );
        }

        let mut ct = self
            .encryptor
            .encrypt_zero_public_scaled_error(pk, error_scalar);
        self.add_plaintext_to_c0_ntt(&mut ct, plaintext);

        for poly in ct.data.iter_mut() {
            poly.ntt_inverse(self.params.ring());
        }
        ct.is_ntt = false;
        Ok(ct)
    }

    pub fn encrypt_public_bgv_with_error_scalar(
        &mut self,
        pk: &PublicKey,
        plaintext: &Plaintext,
        error_scalar: u64,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        let raw = self.encrypt_public_with_error_scalar(pk, plaintext, error_scalar)?;
        BgvCiphertext::new(self.params.clone(), raw)
    }

    fn add_plaintext_to_c0_ntt(&self, ct: &mut Ciphertext, plaintext: &Plaintext) {
        let ring = self.params.ring();
        let degree = ring.degree();
        let t = self.params.plain_modulus();
        assert_eq!(plaintext.value.degree(), degree);
        assert_eq!(plaintext.value.num_moduli(), 1);

        let message = plaintext.value.limb(0);
        let c0 = &mut ct.data[0];
        for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
            let q = modulus.value();
            let mut embedded = vec![0u64; degree];
            for (out, &value) in embedded.iter_mut().zip(message.iter()) {
                *out = (value % t) % q;
            }
            silent_math::ntt::ntt_forward(&mut embedded, &ring.ntt_tables()[limb_idx]);

            let c0_limb = c0.limb_mut(limb_idx);
            for (dst, &value) in c0_limb.iter_mut().zip(embedded.iter()) {
                *dst = arith::add_mod(*dst, value, q);
            }
        }
    }
}

pub struct BgvDecryptor {
    params: BgvParameters,
    sk: SecretKey,
}

impl BgvDecryptor {
    pub fn new(params: BgvParameters, sk: SecretKey) -> Self {
        Self { params, sk }
    }

    pub fn decrypt(&self, ciphertext: &Ciphertext) -> Plaintext {
        let phase = self.phase_coefficients(ciphertext);
        decode_bgv_phase(&self.params, &phase, BgvPlaintextFactor::one())
            .expect("BGV decrypt phase is produced under compatible parameters")
    }

    pub fn decrypt_bgv(&self, ciphertext: &BgvCiphertext) -> Plaintext {
        assert_eq!(self.params.degree(), ciphertext.params().degree());
        assert_eq!(
            self.params.plain_modulus(),
            ciphertext.params().plain_modulus()
        );

        let phase = if self.params.q_modulus_count() == ciphertext.q_modulus_count() {
            self.phase_coefficients(ciphertext.raw())
        } else {
            let level_decryptor = BgvDecryptor::new(ciphertext.params().clone(), self.sk.clone());
            level_decryptor.phase_coefficients(ciphertext.raw())
        };

        decode_bgv_phase(ciphertext.params(), &phase, ciphertext.int_factor())
            .expect("BGV decrypt phase is produced under compatible parameters")
    }

    pub fn invariant_noise_budget(&self, ciphertext: &Ciphertext) -> usize {
        self.invariant_noise_estimate(ciphertext).budget_bits
    }

    pub fn invariant_noise_budget_bgv(&self, ciphertext: &BgvCiphertext) -> usize {
        if self.params.q_modulus_count() == ciphertext.q_modulus_count() {
            self.invariant_noise_budget(ciphertext.raw())
        } else {
            let level_decryptor = BgvDecryptor::new(ciphertext.params().clone(), self.sk.clone());
            level_decryptor.invariant_noise_budget(ciphertext.raw())
        }
    }

    pub fn invariant_noise_max(&self, ciphertext: &Ciphertext) -> u128 {
        self.invariant_noise_estimate(ciphertext)
            .max_noise
            .to_u128()
            .unwrap_or(u128::MAX)
    }

    pub fn invariant_noise_max_bgv(&self, ciphertext: &BgvCiphertext) -> u128 {
        if self.params.q_modulus_count() == ciphertext.q_modulus_count() {
            self.invariant_noise_max(ciphertext.raw())
        } else {
            let level_decryptor = BgvDecryptor::new(ciphertext.params().clone(), self.sk.clone());
            level_decryptor.invariant_noise_max(ciphertext.raw())
        }
    }

    pub fn invariant_noise_max_bits(&self, ciphertext: &Ciphertext) -> usize {
        self.invariant_noise_estimate(ciphertext).max_noise_bits()
    }

    pub fn invariant_noise_max_bits_bgv(&self, ciphertext: &BgvCiphertext) -> usize {
        if self.params.q_modulus_count() == ciphertext.q_modulus_count() {
            self.invariant_noise_max_bits(ciphertext.raw())
        } else {
            let level_decryptor = BgvDecryptor::new(ciphertext.params().clone(), self.sk.clone());
            level_decryptor.invariant_noise_max_bits(ciphertext.raw())
        }
    }

    fn phase_coefficients(&self, ciphertext: &Ciphertext) -> Poly {
        let ring = self.params.ring();
        let degree = ring.degree();
        let ct_is_ntt = ciphertext.is_ntt;

        let mut phase = ciphertext.data[0].clone();
        if !ct_is_ntt {
            phase.ntt_forward(ring);
        }

        let ct_moduli_count = phase.num_moduli();
        let sk_prepared = if self.sk.value.num_moduli() > ct_moduli_count {
            let mut reduced = Poly::new(degree, ct_moduli_count);
            reduced
                .data_mut()
                .copy_from_slice(&self.sk.value.data()[..degree * ct_moduli_count]);
            reduced
        } else {
            self.sk.value.clone()
        };

        let mut s_pow = sk_prepared.clone();
        for i in 1..ciphertext.data.len() {
            let mut term = ciphertext.data[i].clone();
            if !ct_is_ntt {
                term.ntt_forward(ring);
            }
            term.mul_assign(&s_pow, ring);
            phase.add_assign(&term, ring);

            if i < ciphertext.data.len() - 1 {
                s_pow.mul_assign(&sk_prepared, ring);
            }
        }

        phase.ntt_inverse(ring);
        phase
    }

    fn invariant_noise_estimate(&self, ciphertext: &Ciphertext) -> BgvNoiseEstimate {
        let phase = self.phase_coefficients(ciphertext);
        let ring = self.params.ring();
        let degree = ring.degree();
        let size_q = ring.rns().len();
        let t = self.params.plain_modulus();
        let q = rns_big::ciphertext_modulus(ring);
        let q_half = &q >> 1usize;

        let mut residues = vec![0u64; size_q];
        let mut max_noise = BigUint::zero();
        for coeff_idx in 0..degree {
            for (limb_idx, residue) in residues.iter_mut().enumerate() {
                *residue = phase.limb(limb_idx)[coeff_idx];
            }
            let phase_centered = rns_big::compose_centered(&residues, ring, &q);
            let message = rns_big::signed_mod_u64(&phase_centered, t);
            let noise = phase_centered - num_bigint::BigInt::from(message);
            let abs_noise = noise
                .abs()
                .to_biguint()
                .expect("absolute BigInt converts to BigUint");
            max_noise = max_noise.max(abs_noise);
        }

        let budget_bits = if max_noise.is_zero() {
            rns_big::bit_width(&q_half)
        } else {
            rns_big::bit_width(&q_half).saturating_sub(rns_big::bit_width(&max_noise))
        };

        BgvNoiseEstimate {
            max_noise,
            budget_bits,
        }
    }
}

pub fn decode_bgv_phase(
    params: &BgvParameters,
    phase: &Poly,
    plaintext_factor: BgvPlaintextFactor,
) -> Result<Plaintext, BgvLevelError> {
    let ring = params.ring();
    let degree = ring.degree();
    let size_q = ring.rns().len();
    if phase.degree() != degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: degree,
            actual: phase.degree(),
        });
    }
    if phase.num_moduli() != size_q {
        return Err(BgvLevelError::CiphertextLimbMismatch {
            expected: size_q,
            actual: phase.num_moduli(),
        });
    }

    let t = params.plain_modulus();
    let factor_inv = plaintext_factor.inverse_mod(t)?;
    let q = rns_big::ciphertext_modulus(ring);
    let mut out = Poly::new(degree, 1);
    let mut residues = vec![0u64; size_q];
    for coeff_idx in 0..degree {
        for (limb_idx, residue) in residues.iter_mut().enumerate() {
            *residue = phase.limb(limb_idx)[coeff_idx];
        }
        let centered = rns_big::compose_centered(&residues, ring, &q);
        let mut decoded = rns_big::signed_mod_u64(&centered, t);
        if factor_inv != 1 {
            decoded = ((u128::from(decoded) * u128::from(factor_inv)) % u128::from(t)) as u64;
        }
        out.limb_mut(0)[coeff_idx] = decoded;
    }

    Ok(Plaintext { value: out })
}

struct BgvNoiseEstimate {
    max_noise: BigUint,
    budget_bits: usize,
}

impl BgvNoiseEstimate {
    fn max_noise_bits(&self) -> usize {
        rns_big::bit_width(&self.max_noise)
    }
}
