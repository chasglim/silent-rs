use crate::schemes::bfv::params::BfvParameters;
use silent_rlwe::{Ciphertext, Encryptor, Plaintext, SecretKey};

use silent_math::arith;
use silent_math::numth;
use silent_utils::rng::SecureRng;

pub struct BfvEncryptor {
    params: BfvParameters,
    encryptor: Encryptor,
    // Precomputed Delta (Q/t) in RNS basis of Q
    delta: Vec<u64>,
}

impl BfvEncryptor {
    pub fn new(params: BfvParameters, rng: SecureRng) -> Self {
        let encryptor = Encryptor::new(params.runtime_params().clone(), rng);
        let delta = Self::compute_delta(&params);

        Self {
            params,
            encryptor,
            delta,
        }
    }

    fn compute_delta(params: &BfvParameters) -> Vec<u64> {
        let q_moduli = params.ring().rns().moduli();
        let t = params.plain_modulus();

        // 1. Compute Q mod t
        let mut q_mod_t = 1u64;
        for modulus in q_moduli {
            let qi_mod_t = modulus.value() % t;
            q_mod_t = (q_mod_t as u128 * qi_mod_t as u128 % (t as u128)) as u64;
        }

        // 2. Compute Delta_i = (-Q_mod_t) * t^(-1) mod q_i
        let mut delta = Vec::with_capacity(q_moduli.len());
        for modulus in q_moduli {
            let qi = modulus.value();

            // t^(-1) mod qi
            let t_inv = numth::mod_inverse(t % qi, qi).expect("t must be coprime to qi");

            // -Q_mod_t mod qi
            // We know Q_mod_t < t < qi usually, so just (qi - Q_mod_t) % qi
            let neg_q_mod_t = if q_mod_t == 0 { 0 } else { qi - (q_mod_t % qi) };

            let delta_i = arith::mul_mod_u64(neg_q_mod_t, t_inv, qi);
            delta.push(delta_i);
        }

        delta
    }

    pub fn encrypt_symmetric(&mut self, sk: &SecretKey, plaintext: &Plaintext) -> Ciphertext {
        // 1. Encrypt Zero -> (c0, c1)
        let mut ct = self.encryptor.encrypt_zero_symmetric(sk);

        // 2. Add Delta * m to c0
        // c0 is ct.data[0]
        // m is plaintext.value
        let c0 = &mut ct.data[0];
        let m = &plaintext.value;
        let delta = &self.delta;

        // Assuming m is in coefficient domain for scaling?
        // NO, BFV multiplication m * Delta is done in the same domain as c0.
        // c0 is typically in NTT domain for efficiency, OR strictly Coefficient.
        // rlwe encryption (encrypt_zero) usually outputs in NTT or Coeff?
        // `encrypt_zero_symmetric` in `lib.rs`:
        // Ciphertext::new(vec![b_poly, a_poly], ..., true) -> true usually means is_ntt.
        // If c0 is in NTT, we must transform Delta*m to NTT or do addition in Coeff.
        // Usually m is encoded as Poly (coeffs). Delta is scalar.
        // So Delta*m is just scaling coeffs.
        // Then we transform (Delta*m) to NTT to add to c0.

        let ring = self.params.ring();
        let moduli = ring.rns().moduli();

        // We need a temporary poly for Delta*m
        // Plaintext poly `m` usually has 1 modulus (dummy) or coefficients fit in u64.
        // We treat `m` coefficients as values to be scaled up to Q.

        // Iterate over Q limbs
        for (i, modulus) in moduli.iter().enumerate() {
            let qi = modulus.value();
            let delta_i = delta[i];

            // c0 limb for modulus q_i
            let c0_limb = c0.limb_mut(i);

            // m coeffs (assuming m is small, coeffs < t)
            // If m has multiple limbs (unlikely for plaintext), take first.
            let m_limb = m.limb(0); // Plaintext should have coefficients

            // We need to add Delta_i * m_coeff to c0_coeff (element-wise)
            // BUT c0 is in NTT domain!
            // So we must:
            // 1. Create a scalar poly P_i = Delta_i * m
            // 2. NTT(P_i)
            // 3. Add to c0

            // Optimization: Can we do this lazily?
            // For now, explicit loop.

            let mut scaled_m = vec![0u64; ring.degree()];
            // Scaled m = m * delta mod qi
            for (j, &m_val) in m_limb.iter().enumerate() {
                scaled_m[j] = arith::mul_mod_u64(m_val % qi, delta_i, qi);
            }

            // NTT forward on scaled_m
            // Use low-level NTT from math crate
            silent_math::ntt::ntt_forward(&mut scaled_m, &ring.ntt_tables()[i]);

            // Add to c0
            for (j, &val) in scaled_m.iter().enumerate() {
                c0_limb[j] = arith::add_mod(c0_limb[j], val, qi);
            }
        }

        // Return BFV ciphertext in coefficient domain (SEAL-style).
        ct.data[0].ntt_inverse(ring);
        ct.data[1].ntt_inverse(ring);
        ct.is_ntt = false;

        ct
    }
}

pub struct BfvDecryptor {
    params: BfvParameters,
    sk: SecretKey,
}

impl BfvDecryptor {
    pub fn new(params: BfvParameters, sk: SecretKey) -> Self {
        Self { params, sk }
    }

    pub fn decrypt(&self, ciphertext: &Ciphertext) -> Plaintext {
        // 1. Generic decrypt to get noisy M = c0 + c1*s
        // We can reuse Evaluator logic or manually dot product.
        // Manual simple dot product for now.

        let ring = self.params.ring();
        let _moduli = ring.rns().moduli(); // unused but might be useful for debug
        let degree = ring.degree();
        let ct_is_ntt = ciphertext.is_ntt;

        // M initialized with c0 (convert to NTT if needed)
        let mut m_poly = ciphertext.data[0].clone();
        if !ct_is_ntt {
            m_poly.ntt_forward(ring);
        }

        // Prepare secret key power s^1
        // Resize s to match ciphertext context (usually Q, but s might be QP)
        // Prepare secret key (resized if necessary)
        let ct_moduli_count = m_poly.num_moduli();
        let sk_prepared = if self.sk.value.num_moduli() > ct_moduli_count {
            let mut s_reduced = silent_ring::Poly::new(degree, ct_moduli_count);
            s_reduced
                .data_mut()
                .copy_from_slice(&self.sk.value.data()[..degree * ct_moduli_count]);
            s_reduced
        } else {
            self.sk.value.clone()
        };

        // Add c_i * s^i for i >= 1
        let mut s_pow = sk_prepared.clone();

        for i in 1..ciphertext.data.len() {
            let c_i = &ciphertext.data[i];
            let mut term = c_i.clone();

            // term = c_i * s^i
            if !ct_is_ntt {
                term.ntt_forward(ring);
            }
            term.mul_assign(&s_pow, ring);
            m_poly.add_assign(&term, ring);

            // Prepare next power of s: s^{i+1} = s^i * s
            if i < ciphertext.data.len() - 1 {
                s_pow.mul_assign(&sk_prepared, ring);
            }
        }
        // 2. Inverse NTT to get coefficients of M
        m_poly.ntt_inverse(ring);

        // 3. Scale and Round: m = round(t/Q * M)
        // Use RnsTool
        let rns_tool = self.params.rns_tool();

        // Use `scale_and_round` which takes Q inputs and outputs t inputs.
        // But `scale_and_round` in RnsTool typically expects input in `[u64]` flat buffer of all limbs.
        // `Poly` structure is limb-separated (Vec<Vec<u64>> effectively if we view it that way, but data is presumably flat or limb-by-limb?
        // Wait, `Poly` in `ring` is `data: Vec<u64>`.
        // Layout: RNS major? (Limb 0 full, Limb 1 full...)
        // `rns_tool` functions like `scale_and_round` usually expect (Limb 0, Limb 1...) contiguous?
        // Let's check `Poly` layout. `ring/src/poly.rs`: "data: Vec<u64>" of size `degree * num_moduli`.
        // Usually "Limb 0 (all coeffs), Limb 1 (all coeffs)".
        // `rns_tool` usually processes "coeff-wise scaling" which iterates over limbs?
        // Let's check `rns_tool.rs` signature.
        // `scale_and_round(input: &[u64], count: usize)`
        // `count` is usually number of coefficients.
        // `input` length is `count * size_q`.
        // Input layout in openfhe `DCRTPoly`: limb-major?
        // `rns_tool.rs`:
        // Loop `coeff` in 0..count:
        //   Loop `i` in 0..size_q:
        //     `input[i * count + coeff]`
        // This means Input is stored as: Limb 0 (all coeffs), Limb 1 (all coeffs)...
        // i.e., Structure of Arrays (SoA).
        // This matches `Poly` if `Poly` stores Limb 0 then Limb 1.

        // Perform scale and round using fast float-based approximation
        // Output is `Vec<u64>` of size `count` (1 limb, mod t).
        let raw_m = if std::env::var("SILENT_BFV_DECRYPT_EXACT").is_ok() {
            rns_tool
                .scale_and_round(&m_poly.data(), degree)
                .expect("Scale and round failed")
        } else {
            rns_tool
                .scale_and_round_float(&m_poly.data(), degree)
                .expect("Scale and round failed")
        };

        // 4. Wrap in Plaintext
        use silent_ring::Poly as RingPoly;
        let mut res_poly = RingPoly::new(degree, 1);
        res_poly.limb_mut(0).copy_from_slice(&raw_m);

        Plaintext { value: res_poly }
    }
}
