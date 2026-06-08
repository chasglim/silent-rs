use crate::HssCiphertext;
use crate::HssContext;
use crate::HssEvalKey;
use crate::HssShare;

use rand::SeedableRng;
use silent_math::bigint::BigUint;
use silent_math::modulus::Modulus;
use silent_math::rns::RnsError;
use silent_math::rns_tool::RnsTool;
use silent_rlwe::EncryptionParams;
use silent_utils::rng::{SecureRng, blake2xb_fill};

pub struct HssEvaluator;

impl HssEvaluator {
    /// Context-first BKS19 Fig.2 "Load an input into memory".
    pub fn load(
        context: &HssContext,
        eval_key: &HssEvalKey,
        ciphertext: &HssCiphertext,
        instruction_id: u64,
    ) -> Result<HssShare, RnsError> {
        Self::hss_load(
            eval_key,
            ciphertext,
            instruction_id,
            context.runtime_params(),
            context.plain_modulus(),
        )
    }

    /// Context-first BKS19 Fig.2 "Add values in memory".
    pub fn add_mem(
        context: &HssContext,
        eval_key: &HssEvalKey,
        left: &HssShare,
        right: &HssShare,
        instruction_id: u64,
    ) -> HssShare {
        Self::hss_add_mem(
            eval_key,
            left,
            right,
            instruction_id,
            context.runtime_params(),
        )
    }

    /// Context-first BKS19 Fig.2 "Add input values".
    pub fn add_input(
        context: &HssContext,
        left: &HssCiphertext,
        right: &HssCiphertext,
    ) -> HssCiphertext {
        Self::hss_add_input(left, right, context.runtime_params())
    }

    /// Context-first BKS19 Fig.2 "Multiply memory value by input".
    pub fn mult_mem(
        context: &HssContext,
        eval_key: &HssEvalKey,
        share_in: &HssShare,
        ct_in: &HssCiphertext,
        instruction_id: u64,
    ) -> Result<HssShare, RnsError> {
        Self::hss_mult_mem(
            eval_key,
            share_in,
            ct_in,
            instruction_id,
            context.runtime_params(),
            context.plain_modulus(),
        )
    }

    pub fn dec_share_pair(
        context: &HssContext,
        key_share_0: &HssShare,
        key_share_1: &HssShare,
        ciphertext: &HssCiphertext,
        rng: &mut SecureRng,
    ) -> Result<(HssShare, HssShare), RnsError> {
        Self::hss_dec_share_pair(
            key_share_0,
            key_share_1,
            ciphertext,
            context.runtime_params(),
            context.plain_modulus(),
            rng,
        )
    }

    pub fn mult(
        context: &HssContext,
        share_in: &HssShare,
        ct_in: &HssCiphertext,
    ) -> Result<HssShare, RnsError> {
        Self::hss_mult(
            share_in,
            ct_in,
            context.runtime_params(),
            context.plain_modulus(),
        )
    }

    pub fn mult_pair_exact(
        context: &HssContext,
        share_0: &HssShare,
        share_1: &HssShare,
        ct_in: &HssCiphertext,
        key_share_0: &HssShare,
        key_share_1: &HssShare,
        rng: &mut SecureRng,
    ) -> Result<(HssShare, HssShare), RnsError> {
        Self::hss_mult_pair_exact(
            share_0,
            share_1,
            ct_in,
            key_share_0,
            key_share_1,
            context.runtime_params(),
            context.plain_modulus(),
            rng,
        )
    }

    pub fn share_plain_coeffs_exact(
        context: &HssContext,
        coeffs_t: &[u64],
        key_share_0: &HssShare,
        key_share_1: &HssShare,
        rng: &mut SecureRng,
    ) -> Result<(HssShare, HssShare), RnsError> {
        Self::hss_share_plain_coeffs_exact(
            coeffs_t,
            key_share_0,
            key_share_1,
            context.runtime_params(),
            context.plain_modulus(),
            rng,
        )
    }

    pub fn dec_share(
        context: &HssContext,
        share: &HssShare,
        ciphertext: &HssCiphertext,
    ) -> Result<HssShare, RnsError> {
        Self::hss_dec_share(
            share,
            ciphertext,
            context.runtime_params(),
            context.plain_modulus(),
        )
    }

    pub fn reconstruct(context: &HssContext, shares: &[HssShare]) -> Result<Vec<u64>, RnsError> {
        Self::hss_reconstruct(shares, context.runtime_params(), context.plain_modulus())
    }

    pub fn add(context: &HssContext, ct1: &HssCiphertext, ct2: &HssCiphertext) -> HssCiphertext {
        Self::hss_add(ct1, ct2, context.runtime_params())
    }

    pub fn mul_plain(
        context: &HssContext,
        ct: &HssCiphertext,
        p_poly: &silent_ring::Poly,
    ) -> HssCiphertext {
        Self::hss_mul_plain(ct, p_poly, context.runtime_params())
    }

    pub fn ddec(
        context: &HssContext,
        share: &HssShare,
        ciphertext: &HssCiphertext,
    ) -> Result<silent_ring::Poly, RnsError> {
        Self::hss_ddec(
            share,
            ciphertext,
            context.runtime_params(),
            context.plain_modulus(),
        )
    }

    /// Output modulus is evaluation-specific, so it remains an explicit argument.
    pub fn output_mem(
        context: &HssContext,
        share: &HssShare,
        output_modulus: u64,
    ) -> Result<Vec<u64>, RnsError> {
        Self::hss_output_mem(share, context.runtime_params(), output_modulus)
    }

    /// BKS19 Fig.2 "Load an input into memory":
    /// `t_b^x = DDec(b, s_b, C^x) + (1-2b) * PRF(K,id) mod q`.
    pub fn hss_load(
        eval_key: &HssEvalKey,
        ciphertext: &HssCiphertext,
        instruction_id: u64,
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<HssShare, RnsError> {
        Self::validate_party_index(eval_key.party_index)?;
        Self::validate_share_in_ring(&eval_key.share, &params.ring)?;
        Self::validate_ciphertext_in_ring(ciphertext, &params.ring)?;
        if plain_modulus < 2 {
            return Err(RnsError::InvalidResidues);
        }
        let mut out = Self::hss_dec_share(&eval_key.share, ciphertext, params, plain_modulus)?;
        Self::apply_prf_mask_in_place(&mut out, eval_key, instruction_id, params);
        Ok(out)
    }

    /// BKS19 Fig.2 "Add values in memory":
    /// `t_b^{x+x'} = t_b^x + t_b^{x'} + (1-2b) * PRF(K,id) mod q`.
    pub fn hss_add_mem(
        eval_key: &HssEvalKey,
        left: &HssShare,
        right: &HssShare,
        instruction_id: u64,
        params: &EncryptionParams,
    ) -> HssShare {
        let ring = &params.ring;
        let mut out = left.clone();
        out.elements[0].add_assign(&right.elements[0], ring);
        out.elements[1].add_assign(&right.elements[1], ring);
        Self::apply_prf_mask_in_place(&mut out, eval_key, instruction_id, params);
        out
    }

    /// BKS19 Fig.2 "Multiply memory value by input":
    /// `t_b^{x*x'} = DDec(b, t_b^x, C^{x'}) + (1-2b) * PRF(K,id) mod q`.
    pub fn hss_mult_mem(
        eval_key: &HssEvalKey,
        share_in: &HssShare,
        ct_in: &HssCiphertext,
        instruction_id: u64,
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<HssShare, RnsError> {
        Self::validate_party_index(eval_key.party_index)?;
        if plain_modulus < 2 {
            return Err(RnsError::InvalidResidues);
        }
        let mut out = Self::hss_mult(share_in, ct_in, params, plain_modulus)?;
        Self::apply_prf_mask_in_place(&mut out, eval_key, instruction_id, params);
        Ok(out)
    }

    /// Computes a pair of decryption shares for the two key shares and re-randomizes
    /// them into exact additive shares of (x, x*s) over R_q.
    /// This is the strict path expected before feeding shares into `hss_mult`.
    pub fn hss_dec_share_pair(
        key_share_0: &HssShare,
        key_share_1: &HssShare,
        ciphertext: &HssCiphertext,
        params: &EncryptionParams,
        plain_modulus: u64,
        rng: &mut SecureRng,
    ) -> Result<(HssShare, HssShare), RnsError> {
        Self::validate_share_in_ring(key_share_0, &params.ring)?;
        Self::validate_share_in_ring(key_share_1, &params.ring)?;
        Self::validate_ciphertext_in_ring(ciphertext, &params.ring)?;
        if plain_modulus < 2 {
            return Err(RnsError::InvalidResidues);
        }
        let raw0 = Self::hss_dec_share(key_share_0, ciphertext, params, plain_modulus)?;
        let raw1 = Self::hss_dec_share(key_share_1, ciphertext, params, plain_modulus)?;
        let x_coeffs = Self::hss_reconstruct(&[raw0, raw1], params, plain_modulus)?;

        Self::hss_share_plain_coeffs_exact(
            &x_coeffs,
            key_share_0,
            key_share_1,
            params,
            plain_modulus,
            rng,
        )
    }

    pub fn hss_mult(
        share_in: &HssShare,
        ct_in: &HssCiphertext,
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<HssShare, RnsError> {
        Self::validate_share_in_ring(share_in, &params.ring)?;
        Self::validate_ciphertext_in_ring(ct_in, &params.ring)?;
        if plain_modulus < 2 {
            return Err(RnsError::InvalidResidues);
        }
        let ring = &params.ring;

        // BKS19 Mult:
        // 1) convert share to NTT domain for dot product with ciphertext
        // 2) compute dot products against Enc(x) and Enc(x*s)
        // 3) Round(Q->p) and Lift(p->Q) on both components to keep the memory-share invariant
        let mut share0_ntt = share_in.elements[0].clone();
        share0_ntt.ntt_forward(ring);
        let mut share1_ntt = share_in.elements[1].clone();
        share1_ntt.ntt_forward(ring);
        let share_ntt = HssShare {
            elements: [share0_ntt, share1_ntt],
        };

        let t_q = Self::hss_dot_prod(&share_ntt, ct_in, ring)?;

        let mut t0 = t_q[0].clone();
        t0.ntt_inverse(ring);
        let t0_rescaled = Self::rescale_poly(&t0, params, plain_modulus)?;

        let mut t1 = t_q[1].clone();
        t1.ntt_inverse(ring);
        let t1_rescaled = Self::rescale_poly(&t1, params, plain_modulus)?;

        Ok(HssShare {
            elements: [t0_rescaled, t1_rescaled],
        })
    }

    /// Computes `hss_mult` for a full additive share pair and re-shares it back to an exact
    /// `(m, m*s)` share pair.
    ///
    /// This helper is intended for chained multiplications: raw `hss_mult` outputs are valid for
    /// reconstruction, but re-sharing avoids invariant drift before feeding the pair into another
    /// multiplication round.
    pub fn hss_mult_pair_exact(
        share_0: &HssShare,
        share_1: &HssShare,
        ct_in: &HssCiphertext,
        key_share_0: &HssShare,
        key_share_1: &HssShare,
        params: &EncryptionParams,
        plain_modulus: u64,
        rng: &mut SecureRng,
    ) -> Result<(HssShare, HssShare), RnsError> {
        Self::validate_share_in_ring(key_share_0, &params.ring)?;
        Self::validate_share_in_ring(key_share_1, &params.ring)?;
        let out0 = Self::hss_mult(share_0, ct_in, params, plain_modulus)?;
        let out1 = Self::hss_mult(share_1, ct_in, params, plain_modulus)?;
        let coeffs = Self::hss_reconstruct(&[out0, out1], params, plain_modulus)?;
        Self::hss_share_plain_coeffs_exact(
            &coeffs,
            key_share_0,
            key_share_1,
            params,
            plain_modulus,
            rng,
        )
    }

    /// Builds exact additive shares of (x, x*s) in R_q from plaintext coefficients in R_p.
    pub fn hss_share_plain_coeffs_exact(
        coeffs_t: &[u64],
        key_share_0: &HssShare,
        key_share_1: &HssShare,
        params: &EncryptionParams,
        plain_modulus: u64,
        rng: &mut SecureRng,
    ) -> Result<(HssShare, HssShare), RnsError> {
        if plain_modulus < 2 {
            return Err(RnsError::InvalidResidues);
        }
        Self::validate_share_in_ring(key_share_0, &params.ring)?;
        Self::validate_share_in_ring(key_share_1, &params.ring)?;
        let ring = &params.ring;
        let degree = ring.degree();
        let moduli = ring.rns().moduli();
        let num_moduli = moduli.len();

        let x_q = Self::lift_centered_coeffs_to_rq(coeffs_t, params, plain_modulus);

        let mut x_ntt = x_q.clone();
        x_ntt.ntt_forward(ring);

        // sk share invariant: key_share_0[1] + key_share_1[1] = s (NTT domain)
        let mut s_ntt = key_share_0.elements[1].clone();
        s_ntt.add_assign(&key_share_1.elements[1], ring);

        let mut xs_ntt = x_ntt.clone();
        xs_ntt.mul_assign(&s_ntt, ring);
        let mut xs_q = xs_ntt;
        xs_q.ntt_inverse(ring);

        // Uniform masks give fresh additive shares over R_q.
        let mask0 = crate::utils::sample_uniform_poly(rng, degree, num_moduli, moduli);
        let mask1 = crate::utils::sample_uniform_poly(rng, degree, num_moduli, moduli);

        let share0 = HssShare {
            elements: [mask0.clone(), mask1.clone()],
        };

        let mut share1_e0 = x_q;
        share1_e0.sub_assign(&mask0, ring);
        let mut share1_e1 = xs_q;
        share1_e1.sub_assign(&mask1, ring);
        let share1 = HssShare {
            elements: [share1_e0, share1_e1],
        };

        Ok((share0, share1))
    }

    fn lift_centered_coeffs_to_rq(
        coeffs_t: &[u64],
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> silent_ring::Poly {
        let degree = params.ring.degree();
        let moduli = params.ring.rns().moduli();
        let num_moduli = moduli.len();
        let p_half = plain_modulus / 2;

        let mut lifted = unsafe { silent_ring::Poly::new_uninit(degree, num_moduli) };

        for i in 0..degree {
            let limited = if i < coeffs_t.len() {
                coeffs_t[i] % plain_modulus
            } else {
                0
            };
            for (j, modulus) in moduli.iter().enumerate() {
                let q = modulus.value();
                if limited > p_half {
                    let abs = plain_modulus - limited;
                    let abs_mod_q = abs % q;
                    lifted.limb_mut(j)[i] = if abs_mod_q == 0 { 0 } else { q - abs_mod_q };
                } else {
                    lifted.limb_mut(j)[i] = limited % q;
                }
            }
        }

        lifted
    }

    fn apply_prf_mask_in_place(
        share: &mut HssShare,
        eval_key: &HssEvalKey,
        instruction_id: u64,
        params: &EncryptionParams,
    ) {
        let ring = &params.ring;
        let mask = Self::prf_mask(eval_key, instruction_id, params);

        if eval_key.is_party_zero() {
            share.elements[0].add_assign(&mask.elements[0], ring);
            share.elements[1].add_assign(&mask.elements[1], ring);
        } else {
            share.elements[0].sub_assign(&mask.elements[0], ring);
            share.elements[1].sub_assign(&mask.elements[1], ring);
        }
    }

    fn prf_mask(eval_key: &HssEvalKey, instruction_id: u64, params: &EncryptionParams) -> HssShare {
        let ring = &params.ring;
        let degree = ring.degree();
        let moduli = ring.rns().moduli();
        let num_moduli = moduli.len();

        let seed0 = Self::derive_prf_seed(&eval_key.prf_key, instruction_id, 0);
        let seed1 = Self::derive_prf_seed(&eval_key.prf_key, instruction_id, 1);

        let mut rng0 = SecureRng::from_seed(seed0);
        let mut rng1 = SecureRng::from_seed(seed1);

        let e0 = crate::utils::sample_uniform_poly(&mut rng0, degree, num_moduli, moduli);
        let e1 = crate::utils::sample_uniform_poly(&mut rng1, degree, num_moduli, moduli);
        HssShare { elements: [e0, e1] }
    }

    fn derive_prf_seed(prf_key: &[u8; 32], instruction_id: u64, component: u8) -> [u8; 32] {
        let mut key64 = [0u8; 64];
        key64[..32].copy_from_slice(prf_key);
        key64[32..40].copy_from_slice(&instruction_id.to_le_bytes());
        key64[40] = component;

        let mut out = [0u8; 32];
        blake2xb_fill(&mut out, &key64, 0);
        out
    }

    /// Computes the dot product between the HSS ciphertext and the secret share.
    /// Returns [Poly; 2] in **NTT Domain** (mod Q).
    pub fn hss_dot_prod(
        share: &HssShare,
        ciphertext: &HssCiphertext,
        ring: &silent_ring::RingContext,
    ) -> Result<[silent_ring::Poly; 2], RnsError> {
        Self::validate_share_in_ring(share, ring)?;
        Self::validate_ciphertext_in_ring(ciphertext, ring)?;
        let degree = ring.degree();
        let num_moduli = ring.rns().moduli().len();

        let mut t0 = unsafe { silent_ring::Poly::new_uninit(degree, num_moduli) };
        let mut t1 = unsafe { silent_ring::Poly::new_uninit(degree, num_moduli) };
        t0.data_mut().fill(0);
        t1.data_mut().fill(0);

        let c0 = &ciphertext.enc_m.data[0];
        let c1 = &ciphertext.enc_m.data[1];

        // Share elements are already in NTT (from hss_dec_share)
        let e0 = &share.elements[0];
        let e1 = &share.elements[1];

        // t0 = c0 * e0 + c1 * e1 (dot product with enc_m)
        {
            let mut tmp = c0.clone();
            tmp.mul_assign(e0, ring);
            t0.add_assign(&tmp, ring);

            let mut tmp2 = c1.clone();
            tmp2.mul_assign(e1, ring);
            t0.add_assign(&tmp2, ring);
        }

        // t1 = c0s * e0 + c1s * e1 (dot product with enc_m_times_s)
        {
            let c0s = &ciphertext.enc_m_times_s.data[0];
            let c1s = &ciphertext.enc_m_times_s.data[1];

            let mut tmp = c0s.clone();
            tmp.mul_assign(e0, ring);
            t1.add_assign(&tmp, ring);

            let mut tmp2 = c1s.clone();
            tmp2.mul_assign(e1, ring);
            t1.add_assign(&tmp2, ring);
        }

        Ok([t0, t1])
    }

    /// Scales and Rounds a polynomial from Q to p.
    pub fn scale_and_round(
        poly_q: &silent_ring::Poly,
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<silent_ring::Poly, RnsError> {
        let degree = params.ring.degree();
        let rounded = Self::round_q_to_t_no_bigint(poly_q, params, plain_modulus)?;

        let mut res_poly = unsafe { silent_ring::Poly::new_uninit(degree, 1) };
        let out_limb = res_poly.limb_mut(0);
        out_limb.copy_from_slice(&rounded);

        Ok(res_poly)
    }

    pub fn hss_dec_share(
        share: &HssShare,
        ciphertext: &HssCiphertext,
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<HssShare, RnsError> {
        if plain_modulus < 2 {
            return Err(RnsError::InvalidResidues);
        }
        let t_q = Self::hss_dot_prod(share, ciphertext, &params.ring)?;
        let ring = &params.ring;

        // Rescale t0: Delta y -> y.
        let mut t0 = t_q[0].clone();
        t0.ntt_inverse(ring);
        let t0_rescaled = Self::rescale_poly(&t0, params, plain_modulus)?;

        // Rescale t1: Delta y s -> y s.
        let mut t1 = t_q[1].clone();
        t1.ntt_inverse(ring);
        let t1_rescaled = Self::rescale_poly(&t1, params, plain_modulus)?;

        Ok(HssShare {
            elements: [t0_rescaled, t1_rescaled],
        })
    }

    pub fn rescale_poly(
        poly_q: &silent_ring::Poly,
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<silent_ring::Poly, RnsError> {
        let ring = &params.ring;
        let degree = ring.degree();
        let num_moduli = poly_q.num_moduli();
        let mut res_poly = unsafe { silent_ring::Poly::new_uninit(degree, num_moduli) };
        let moduli = ring.rns().moduli();
        let rounded = Self::round_q_to_t_no_bigint(poly_q, params, plain_modulus)?;

        let p_half = plain_modulus / 2;

        for i in 0..degree {
            // Convert to symmetric representation [-P/2, P/2) and store in each RNS limb
            // If limited >= P/2, store as negative: qi - (P - limited)
            // If limited < P/2, store as positive: limited
            let limited = rounded[i];
            for j in 0..num_moduli {
                let qi = moduli[j].value();
                if limited > p_half {
                    // Negative value: limited - P in symmetric rep.
                    let abs = plain_modulus - limited;
                    let abs_mod_q = abs % qi;
                    res_poly.limb_mut(j)[i] = if abs_mod_q == 0 { 0 } else { qi - abs_mod_q };
                } else {
                    // Positive value: just store limited
                    res_poly.limb_mut(j)[i] = limited % qi;
                }
            }
        }
        Ok(res_poly)
    }

    pub fn hss_reconstruct(
        shares: &[HssShare],
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<Vec<u64>, RnsError> {
        if plain_modulus < 2 {
            return Err(RnsError::InvalidResidues);
        }
        if shares.is_empty() {
            return Ok(Vec::new());
        }
        let ring = &params.ring;
        let degree = ring.degree();
        let num_moduli = ring.rns().moduli().len();
        let base_q = ring.rns();

        // Sum shares in Q (element 0)
        let mut sum_poly = unsafe { silent_ring::Poly::new_uninit(degree, num_moduli) };
        for i in 0..num_moduli {
            sum_poly.limb_mut(i).fill(0);
        }

        for share in shares {
            Self::validate_share_in_ring(share, ring)?;
            sum_poly.add_assign(&share.elements[0], ring);
        }
        let _ = (degree, num_moduli, base_q);
        Self::centered_reduce_mod_t_no_bigint(&sum_poly, params, plain_modulus)
    }

    pub fn hss_add(
        ct1: &HssCiphertext,
        ct2: &HssCiphertext,
        params: &EncryptionParams,
    ) -> HssCiphertext {
        let mut res = ct1.clone();
        let ring = &params.ring;

        res.enc_m.data[0].add_assign(&ct2.enc_m.data[0], ring);
        res.enc_m.data[1].add_assign(&ct2.enc_m.data[1], ring);

        res.enc_m_times_s.data[0].add_assign(&ct2.enc_m_times_s.data[0], ring);
        res.enc_m_times_s.data[1].add_assign(&ct2.enc_m_times_s.data[1], ring);

        res
    }

    /// BKS19 Fig.2 "Add input values":
    /// `C^{x+x'} = C^x + C^{x'} mod q`.
    pub fn hss_add_input(
        left: &HssCiphertext,
        right: &HssCiphertext,
        params: &EncryptionParams,
    ) -> HssCiphertext {
        Self::hss_add(left, right, params)
    }

    pub fn hss_mul_plain(
        ct: &HssCiphertext,
        p_poly: &silent_ring::Poly,
        params: &EncryptionParams,
    ) -> HssCiphertext {
        let mut res = ct.clone();
        let ring = &params.ring;

        res.enc_m.data[0].mul_assign(p_poly, ring);
        res.enc_m.data[1].mul_assign(p_poly, ring);

        res.enc_m_times_s.data[0].mul_assign(p_poly, ring);
        res.enc_m_times_s.data[1].mul_assign(p_poly, ring);

        res
    }

    /// Distributed Decryption (BKS19 PKE.DDec).
    /// Computes Round(<share, ct>) -> Poly in R_p.
    pub fn hss_ddec(
        share: &HssShare,
        ciphertext: &HssCiphertext,
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<silent_ring::Poly, RnsError> {
        if plain_modulus < 2 {
            return Err(RnsError::InvalidResidues);
        }
        let t_q = Self::hss_dot_prod(share, ciphertext, &params.ring)?;

        let mut t0 = t_q[0].clone();
        t0.ntt_inverse(&params.ring);

        // Scale and Round t0
        Self::rescale_poly(&t0, params, plain_modulus)
    }

    /// BKS19 Fig.2 "Output from memory":
    /// parse `t_b^x = (x_b, \hat t_b)` and return `x_b mod r`.
    pub fn hss_output_mem(
        share: &HssShare,
        params: &EncryptionParams,
        output_modulus: u64,
    ) -> Result<Vec<u64>, RnsError> {
        if output_modulus < 2 {
            return Err(RnsError::InvalidResidues);
        }
        Self::validate_share_in_ring(share, &params.ring)?;
        Self::centered_reduce_mod_t_no_bigint(&share.elements[0], params, output_modulus)
    }

    fn validate_party_index(party_index: u8) -> Result<(), RnsError> {
        if party_index > 1 {
            return Err(RnsError::InvalidBase);
        }
        Ok(())
    }

    fn validate_share_in_ring(
        share: &HssShare,
        ring: &silent_ring::RingContext,
    ) -> Result<(), RnsError> {
        Self::validate_poly_in_ring(&share.elements[0], ring)?;
        Self::validate_poly_in_ring(&share.elements[1], ring)?;
        Ok(())
    }

    fn validate_ciphertext_in_ring(
        ciphertext: &HssCiphertext,
        ring: &silent_ring::RingContext,
    ) -> Result<(), RnsError> {
        Self::validate_rlwe_ciphertext_component(&ciphertext.enc_m, ring)?;
        Self::validate_rlwe_ciphertext_component(&ciphertext.enc_m_times_s, ring)?;
        Ok(())
    }

    fn validate_rlwe_ciphertext_component(
        ciphertext: &silent_rlwe::Ciphertext,
        ring: &silent_ring::RingContext,
    ) -> Result<(), RnsError> {
        if ciphertext.data.len() != 2 || !ciphertext.is_ntt {
            return Err(RnsError::InvalidBase);
        }
        if !Self::same_ring_shape(ciphertext.params.ring.as_ref(), ring) {
            return Err(RnsError::InvalidBase);
        }
        for poly in &ciphertext.data {
            Self::validate_poly_in_ring(poly, ring)?;
        }
        Ok(())
    }

    fn same_ring_shape(
        left: &silent_ring::RingContext,
        right: &silent_ring::RingContext,
    ) -> bool {
        if left.degree() != right.degree() {
            return false;
        }
        let left_moduli = left.rns().moduli();
        let right_moduli = right.rns().moduli();
        if left_moduli.len() != right_moduli.len() {
            return false;
        }
        left_moduli
            .iter()
            .zip(right_moduli.iter())
            .all(|(l, r)| l.value() == r.value())
    }

    fn validate_poly_in_ring(
        poly: &silent_ring::Poly,
        ring: &silent_ring::RingContext,
    ) -> Result<(), RnsError> {
        if poly.degree() != ring.degree() || poly.num_moduli() != ring.rns().len() {
            return Err(RnsError::InvalidBase);
        }
        let moduli = ring.rns().moduli();
        for (limb_idx, modulus) in moduli.iter().enumerate() {
            let q = modulus.value();
            if poly.limb(limb_idx).iter().any(|&coeff| coeff >= q) {
                return Err(RnsError::InvalidResidues);
            }
        }
        Ok(())
    }

    fn round_q_to_t_no_bigint(
        poly_q: &silent_ring::Poly,
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<Vec<u64>, RnsError> {
        let degree = params.ring.degree();
        let base_q = params.ring.rns();

        // Strict path: always use exact RnsRounder-based scaling/rounding.
        // This avoids floating-point approximation drift on large modulus chains.
        if params.rns_tool.base_t().value() == plain_modulus {
            return params.rns_tool.scale_and_round(poly_q.data(), degree);
        }

        let tool = RnsTool::new(base_q.clone(), Modulus::new(plain_modulus)?, None)?;
        tool.scale_and_round(poly_q.data(), degree)
    }

    fn centered_reduce_mod_t_no_bigint(
        poly_q: &silent_ring::Poly,
        params: &EncryptionParams,
        plain_modulus: u64,
    ) -> Result<Vec<u64>, RnsError> {
        let base_q = params.ring.rns();
        let degree = params.ring.degree();
        let size_q = base_q.len();

        // Exact no-bigint path when Q fits u128.
        if let Some(q) = base_q.base_prod_u128() {
            let q_half = q >> 1;
            let p = u128::from(plain_modulus);
            let moduli = base_q.moduli();
            let invs = base_q.inv_punctured_prod_mod_base();
            let invs_shoup = base_q.inv_punctured_prod_mod_base_shoup();

            let mut out = vec![0u64; degree];
            for coeff in 0..degree {
                let mut crt_sum = 0u128;
                for i in 0..size_q {
                    let qi = moduli[i].value();
                    let xi = poly_q.limb(i)[coeff];
                    let term_residue =
                        silent_math::arith::mul_mod_shoup(xi, invs[i], invs_shoup[i], qi);
                    let q_hat = q / u128::from(qi);
                    let term = u128::from(term_residue) * q_hat;
                    debug_assert!(term < q);

                    // crt_sum = (crt_sum + term) mod q, overflow-safe.
                    if crt_sum >= q - term {
                        crt_sum = crt_sum + term - q;
                    } else {
                        crt_sum += term;
                    }
                }

                let reduced = if crt_sum > q_half {
                    let abs_neg = q - crt_sum;
                    let m = (abs_neg % p) as u64;
                    if m == 0 { 0 } else { plain_modulus - m }
                } else {
                    (crt_sum % p) as u64
                };
                out[coeff] = reduced;
            }
            return Ok(out);
        }

        // Exact big-int path for large modulus chains.
        let moduli = base_q.moduli();
        let invs = base_q.inv_punctured_prod_mod_base();
        let punctured = base_q.punctured_prod();
        let q = base_q.base_prod().clone();
        let (q_half, _) = q.div_mod_u64(2);

        let mut out = vec![0u64; degree];
        for coeff in 0..degree {
            let mut value = BigUint::zero();
            for i in 0..size_q {
                let qi = moduli[i].value();
                let xi = poly_q.limb(i)[coeff];
                let term_residue = silent_math::arith::mul_mod_u64(xi, invs[i], qi);
                let term = punctured[i].mul_u64(term_residue);
                value.add_assign(&term);
            }
            let value = Self::mod_big_small(&value, &q, size_q as u64);

            let reduced = if value > q_half {
                let mut abs_neg = q.clone();
                abs_neg.sub_assign(&value);
                let m = abs_neg.mod_u64(plain_modulus);
                if m == 0 { 0 } else { plain_modulus - m }
            } else {
                value.mod_u64(plain_modulus)
            };
            out[coeff] = reduced;
        }
        Ok(out)
    }

    fn div_big_by_big_small(numerator: &BigUint, denominator: &BigUint, max_quot: u64) -> u64 {
        let mut low = 0u64;
        let mut high = max_quot;
        let mut best = 0u64;

        while low <= high {
            let mid = low + ((high - low) >> 1);
            let prod = denominator.mul_u64(mid);
            match prod.cmp(numerator) {
                std::cmp::Ordering::Greater => {
                    if mid == 0 {
                        break;
                    }
                    high = mid - 1;
                }
                _ => {
                    best = mid;
                    low = mid + 1;
                }
            }
        }

        best
    }

    fn mod_big_small(value: &BigUint, modulus: &BigUint, max_quot: u64) -> BigUint {
        let q = Self::div_big_by_big_small(value, modulus, max_quot);
        let mut remainder = value.clone();
        let prod = modulus.mul_u64(q);
        remainder.sub_assign(&prod);
        remainder
    }
}
