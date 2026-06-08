use crate::core::encoder::HeEncoder;
use crate::core::evaluator::HeEvaluator;
use crate::schemes::bgv::ciphertext::{BgvCiphertext, BgvPlaintextFactor};
use crate::schemes::bgv::encoding::BgvBatchEncoder;
use crate::schemes::bgv::keys::{
    BGV_KSW_BASE_LOG, BgvSlotMatrixAxis, bgv_canonical_galois_element_for_params,
    bgv_diagonal_bsgs_assignments, bgv_full_slot_sum_galois_elements, bgv_ksw_levels,
    bgv_slot_linear_transform_diagonal_elements, bgv_slot_matrix_axis_sum_diagonal_elements,
    bgv_slot_merge_galois_elements, bgv_slot_rotation_assignments,
    bgv_slot_zero_to_slot_galois_element, validate_slot_matrix_dimensions,
};
use crate::schemes::bgv::params::{BgvLevelError, BgvModSwitchStep, BgvParameters};
use crate::schemes::bgv::rns_big;
use num_bigint::{BigInt, Sign};
use num_traits::{One, ToPrimitive, Zero};
use silent_math::arith;
use silent_ring::{Poly, RingContext};
use silent_rlwe::{Ciphertext, EvaluationKey, GaloisKey, Plaintext};
use std::cmp::Reverse;
use std::collections::HashMap;

pub struct BgvEvaluator {
    params: BgvParameters,
}

#[derive(Clone, Debug)]
pub struct BgvFastRotationPrecomputation {
    params: BgvParameters,
    coeff: Ciphertext,
    c1_digits: Vec<Poly>,
}

#[derive(Clone, Debug)]
struct BgvDiagonalBsgsTerm {
    baby: u32,
    diagonal: Plaintext,
}

#[derive(Clone, Debug)]
struct BgvDiagonalBsgsGroup {
    giant: u32,
    giant_inverse: u32,
    terms: Vec<BgvDiagonalBsgsTerm>,
}

impl BgvFastRotationPrecomputation {
    pub fn params(&self) -> &BgvParameters {
        &self.params
    }

    pub fn q_modulus_count(&self) -> usize {
        self.params.q_modulus_count()
    }
}

impl HeEvaluator for BgvEvaluator {
    type Context = BgvParameters;

    fn new(context: BgvParameters) -> Self {
        Self { params: context }
    }

    fn add(&self, c1: &Ciphertext, c2: &Ciphertext) -> Ciphertext {
        binary_linear_op(
            c1,
            c2,
            self.params.ring(),
            LinearOp::Add,
            self.params.runtime_params(),
        )
    }

    fn sub(&self, c1: &Ciphertext, c2: &Ciphertext) -> Ciphertext {
        binary_linear_op(
            c1,
            c2,
            self.params.ring(),
            LinearOp::Sub,
            self.params.runtime_params(),
        )
    }

    fn mul(&self, c1: &Ciphertext, c2: &Ciphertext) -> Ciphertext {
        let ring = self.params.ring();
        let mut lhs = c1.data.clone();
        let mut rhs = c2.data.clone();

        if !c1.is_ntt {
            for poly in lhs.iter_mut() {
                poly.ntt_forward(ring);
            }
        }
        if !c2.is_ntt {
            for poly in rhs.iter_mut() {
                poly.ntt_forward(ring);
            }
        }

        let degree = ring.degree();
        let size_q = ring.rns().len();
        let mut result = vec![Poly::new(degree, size_q); lhs.len() + rhs.len() - 1];

        for (i, left) in lhs.iter().enumerate() {
            for (j, right) in rhs.iter().enumerate() {
                let mut product = left.clone();
                product.mul_assign(right, ring);
                result[i + j].add_assign(&product, ring);
            }
        }

        for poly in result.iter_mut() {
            poly.ntt_inverse(ring);
        }

        Ciphertext::new(result, self.params.runtime_params().clone(), false)
    }
}

impl BgvEvaluator {
    /// Drop ciphertext limbs from the top of the BGV Q chain.
    ///
    /// This is level truncation for LSB-embedded BGV ciphertexts: it preserves
    /// the low-order RNS limbs and returns a matching lower-level parameter
    /// context. It does not rescale the ciphertext and is intentionally named
    /// separately from modulus switching.
    pub fn drop_level(
        &self,
        ct: &Ciphertext,
        levels: usize,
    ) -> Result<(BgvParameters, Ciphertext), BgvLevelError> {
        let current = self.params.q_modulus_count();
        if levels == 0 {
            return Ok((self.params.clone(), ct.clone()));
        }
        if levels >= current {
            return Err(BgvLevelError::TooManyCiphertextModuli {
                requested: current.saturating_sub(levels),
                available: current,
            });
        }
        self.drop_to_q_limbs(ct, current - levels)
    }

    /// Truncate a ciphertext to the first `q_limbs` RNS limbs.
    pub fn drop_to_q_limbs(
        &self,
        ct: &Ciphertext,
        q_limbs: usize,
    ) -> Result<(BgvParameters, Ciphertext), BgvLevelError> {
        if q_limbs == 0 {
            return Err(BgvLevelError::EmptyCiphertextModulus);
        }
        let current = self.params.q_modulus_count();
        if q_limbs > current {
            return Err(BgvLevelError::TooManyCiphertextModuli {
                requested: q_limbs,
                available: current,
            });
        }

        self.ensure_full_level_ciphertext(ct)?;

        let target_params = self.params.with_q_prefix(q_limbs)?;
        let data = ct
            .data
            .iter()
            .map(|poly| truncate_poly_q_prefix(poly, q_limbs))
            .collect();

        Ok((
            target_params.clone(),
            Ciphertext::new(data, target_params.runtime_params().clone(), ct.is_ntt),
        ))
    }

    /// Modulus-switch a ciphertext to a lower prefix of the BGV Q chain.
    ///
    /// Raw BGV modulus switching divides out dropped ciphertext primes. The
    /// plaintext residue is therefore multiplied by the inverse dropped-prime
    /// factor modulo `t`; callers that need transparent plaintext recovery
    /// should prefer `mod_switch_down_bgv`, which tracks that factor.
    pub fn mod_switch_down(
        &self,
        ct: &Ciphertext,
        levels: usize,
    ) -> Result<(BgvParameters, Ciphertext), BgvLevelError> {
        let current = self.params.q_modulus_count();
        if levels == 0 {
            return Ok((self.params.clone(), ct.clone()));
        }
        if levels >= current {
            return Err(BgvLevelError::TooManyCiphertextModuli {
                requested: current.saturating_sub(levels),
                available: current,
            });
        }
        self.mod_switch_to_q_limbs(ct, current - levels)
    }

    /// Modulus-switch a ciphertext to the first `q_limbs` RNS limbs.
    pub fn mod_switch_to_q_limbs(
        &self,
        ct: &Ciphertext,
        q_limbs: usize,
    ) -> Result<(BgvParameters, Ciphertext), BgvLevelError> {
        if q_limbs == 0 {
            return Err(BgvLevelError::EmptyCiphertextModulus);
        }
        let current = self.params.q_modulus_count();
        if q_limbs > current {
            return Err(BgvLevelError::TooManyCiphertextModuli {
                requested: q_limbs,
                available: current,
            });
        }
        if q_limbs == current {
            return Ok((self.params.clone(), ct.clone()));
        }

        self.ensure_full_level_ciphertext(ct)?;
        let mut current_params = self.params.clone();
        let mut current_ct = to_coeff_domain(ct, self.params.ring());

        while current_params.q_modulus_count() > q_limbs {
            let step = current_params
                .mod_switch_step_for_source_limbs(current_params.q_modulus_count())?;
            let next_params = current_params.with_q_prefix(current_params.q_modulus_count() - 1)?;
            let data = current_ct
                .data
                .iter()
                .map(|poly| {
                    mod_switch_poly_down_one(
                        poly,
                        next_params.ring(),
                        current_params.plain_modulus(),
                        step,
                    )
                })
                .collect();

            current_ct = Ciphertext::new(data, next_params.runtime_params().clone(), false);
            current_params = next_params;
        }

        Ok((current_params, current_ct))
    }

    pub fn add_bgv(
        &self,
        c1: &BgvCiphertext,
        c2: &BgvCiphertext,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        let (c1, c2) = self.adjust_levels_depth_and_factors_bgv(c1, c2)?;
        let params = c1.params().clone();
        let factor = c1.int_factor();

        let level_evaluator = BgvEvaluator::new(params.clone());
        let raw = level_evaluator.add(c1.raw(), c2.raw());
        let noise_scale_degree = c1.noise_scale_degree().max(c2.noise_scale_degree());
        BgvCiphertext::with_metadata(params, raw, factor, noise_scale_degree)
    }

    pub fn sub_bgv(
        &self,
        c1: &BgvCiphertext,
        c2: &BgvCiphertext,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        let (c1, c2) = self.adjust_levels_depth_and_factors_bgv(c1, c2)?;
        let params = c1.params().clone();
        let factor = c1.int_factor();

        let level_evaluator = BgvEvaluator::new(params.clone());
        let raw = level_evaluator.sub(c1.raw(), c2.raw());
        let noise_scale_degree = c1.noise_scale_degree().max(c2.noise_scale_degree());
        BgvCiphertext::with_metadata(params, raw, factor, noise_scale_degree)
    }

    pub fn mul_bgv(
        &self,
        c1: &BgvCiphertext,
        c2: &BgvCiphertext,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        let (params, c1, c2) = self.align_bgv_levels(c1, c2)?;

        let level_evaluator = BgvEvaluator::new(params.clone());
        let raw = level_evaluator.mul(c1.raw(), c2.raw());
        let factor = c1
            .int_factor()
            .mul_mod(c2.int_factor(), params.plain_modulus())?;
        let noise_scale_degree = c1.noise_scale_degree() + c2.noise_scale_degree();
        BgvCiphertext::with_metadata(params, raw, factor, noise_scale_degree)
    }

    pub fn adjust_levels_and_factors_bgv(
        &self,
        c1: &BgvCiphertext,
        c2: &BgvCiphertext,
    ) -> Result<(BgvCiphertext, BgvCiphertext), BgvLevelError> {
        self.adjust_levels_depth_and_factors_bgv(c1, c2)
    }

    pub fn adjust_levels_depth_and_factors_bgv(
        &self,
        c1: &BgvCiphertext,
        c2: &BgvCiphertext,
    ) -> Result<(BgvCiphertext, BgvCiphertext), BgvLevelError> {
        let (params, c1, c2) = self.align_bgv_levels(c1, c2)?;
        let f1 = c1.int_factor();
        let f2 = c2.int_factor();
        let (left, right) = if f1 == f2 {
            (c1, c2)
        } else {
            let t = params.plain_modulus();
            let factor = f1.mul_mod(f2, t)?;
            let left = scale_bgv_ciphertext_by_factor(&c1, f2)?;
            let right = scale_bgv_ciphertext_by_factor(&c2, f1)?;
            debug_assert_eq!(left.int_factor(), factor);
            debug_assert_eq!(right.int_factor(), factor);
            (left, right)
        };

        align_bgv_noise_scale_degree(left, right)
    }

    pub fn relinearize_bgv(
        &self,
        ct: &BgvCiphertext,
        evk: &EvaluationKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let level_evk = self.evaluation_key_for_params(evk, ct.params())?;
        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        let raw = level_evaluator.relinearize(ct.raw(), &level_evk);
        BgvCiphertext::with_metadata(
            ct.params().clone(),
            raw,
            ct.int_factor(),
            ct.noise_scale_degree(),
        )
    }

    pub fn proxy_reencrypt_bgv(
        &self,
        ct: &BgvCiphertext,
        rekey: &EvaluationKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let level_rekey = self.evaluation_key_for_params(rekey, ct.params())?;
        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        let raw = level_evaluator.proxy_reencrypt(ct.raw(), &level_rekey)?;
        BgvCiphertext::with_metadata(
            ct.params().clone(),
            raw,
            ct.int_factor(),
            ct.noise_scale_degree(),
        )
    }

    pub fn mul_relinearized_bgv(
        &self,
        c1: &BgvCiphertext,
        c2: &BgvCiphertext,
        evk: &EvaluationKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        let product = self.mul_bgv(c1, c2)?;
        self.relinearize_bgv(&product, evk)
    }

    pub fn mul_many_relinearized_bgv(
        &self,
        ciphertexts: &[BgvCiphertext],
        evk: &EvaluationKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        if ciphertexts.is_empty() {
            return Err(BgvLevelError::EmptyMulManyInput);
        }
        for ct in ciphertexts {
            self.ensure_compatible_bgv_operand(ct)?;
        }

        let mut work = ciphertexts.to_vec();
        while work.len() > 1 {
            work.sort_by_key(|ct| (Reverse(ct.q_modulus_count()), ct.noise_scale_degree()));
            let mut next = Vec::with_capacity((work.len() + 1) / 2);
            let mut iter = work.into_iter();
            while let Some(left) = iter.next() {
                if let Some(right) = iter.next() {
                    next.push(self.mul_relinearized_bgv(&left, &right, evk)?);
                } else {
                    next.push(left);
                }
            }
            work = next;
        }

        Ok(work
            .pop()
            .expect("BGV multi-ciphertext multiplication has a non-empty work queue"))
    }

    pub fn rotate_bgv(
        &self,
        ct: &BgvCiphertext,
        k: u32,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        if k == 0 {
            return Ok(ct.clone());
        }

        let switch_key = gk
            .keys
            .get(&k)
            .ok_or(BgvLevelError::MissingGaloisKey { index: k })?;
        let level_key = self.evaluation_key_for_params(switch_key, ct.params())?;
        let mut keys = HashMap::new();
        keys.insert(k, level_key);
        let level_gk = GaloisKey { keys };

        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        let raw = level_evaluator.rotate(ct.raw(), k, &level_gk);
        BgvCiphertext::with_metadata(
            ct.params().clone(),
            raw,
            ct.int_factor(),
            ct.noise_scale_degree(),
        )
    }

    pub fn rotate_slots_bgv(
        &self,
        ct: &BgvCiphertext,
        rotation: i32,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let assignments = bgv_slot_rotation_assignments(ct.params(), rotation)?;
        if assignments.is_empty() {
            return Ok(ct.clone());
        }

        for assignment in &assignments {
            if assignment.galois_element != 1 && !gk.keys.contains_key(&assignment.galois_element) {
                return Err(BgvLevelError::MissingGaloisKey {
                    index: assignment.galois_element,
                });
            }
        }

        let precomp = self.fast_rotation_precompute_bgv(ct)?;
        let mut result = None;
        for assignment in assignments {
            let rotated = if assignment.galois_element == 1 {
                ct.clone()
            } else {
                self.fast_rotate_bgv(ct, &precomp, assignment.galois_element, gk)?
            };
            let mask = slot_mask_plaintext_many(ct.params(), &assignment.destination_slots);
            let term = self.mul_plain_bgv(&rotated, &mask)?;
            result = Some(match result {
                Some(acc) => self.add_bgv(&acc, &term)?,
                None => term,
            });
        }

        result.ok_or(BgvLevelError::SlotRotationUnsupported {
            rotation,
            slots: ct.params().degree(),
        })
    }

    pub fn shift_slots_bgv(
        &self,
        ct: &BgvCiphertext,
        shift: i32,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        if shift == 0 {
            return Ok(ct.clone());
        }

        let slots = ct.params().degree();
        let distance = shift.unsigned_abs() as usize;
        if distance >= slots {
            return self.mul_plain_bgv(ct, &slot_mask_plaintext_many(ct.params(), &[]));
        }

        let rotated = self.rotate_slots_bgv(ct, shift, gk)?;
        let kept_slots = if shift > 0 {
            (distance..slots).collect::<Vec<_>>()
        } else {
            (0..slots - distance).collect::<Vec<_>>()
        };
        let mask = slot_mask_plaintext_many(ct.params(), &kept_slots);
        self.mul_plain_bgv(&rotated, &mask)
    }

    pub fn fast_rotation_precompute_bgv(
        &self,
        ct: &BgvCiphertext,
    ) -> Result<BgvFastRotationPrecomputation, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        level_evaluator.fast_rotation_precompute(ct.raw())
    }

    pub fn fast_rotate_bgv(
        &self,
        ct: &BgvCiphertext,
        precomp: &BgvFastRotationPrecomputation,
        k: u32,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        ensure_bgv_precomp_matches_ciphertext(precomp, ct)?;
        if k == 0 {
            return Ok(ct.clone());
        }

        let switch_key = gk
            .keys
            .get(&k)
            .ok_or(BgvLevelError::MissingGaloisKey { index: k })?;
        let level_key = self.evaluation_key_for_params(switch_key, ct.params())?;
        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        let raw = level_evaluator.fast_rotate_with_key(precomp, k, &level_key)?;
        BgvCiphertext::with_metadata(
            ct.params().clone(),
            raw,
            ct.int_factor(),
            ct.noise_scale_degree(),
        )
    }

    pub fn sum_slots_bgv(
        &self,
        ct: &BgvCiphertext,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let precomp = self.fast_rotation_precompute_bgv(ct)?;
        let mut acc = ct.clone();
        for k in bgv_full_slot_sum_galois_elements(ct.params().degree()) {
            let rotated = self.fast_rotate_bgv(ct, &precomp, k, gk)?;
            acc = self.add_bgv(&acc, &rotated)?;
        }
        Ok(acc)
    }

    pub fn replicate_slot_bgv(
        &self,
        ct: &BgvCiphertext,
        slot: usize,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        if slot >= ct.params().degree() {
            return Err(BgvLevelError::SlotIndexOutOfRange {
                slot,
                slots: ct.params().degree(),
            });
        }

        let selected = self.mul_plain_bgv(ct, &slot_mask_plaintext(ct.params(), slot))?;
        self.sum_slots_bgv(&selected, gk)
    }

    pub fn replicate_all_slots_bgv(
        &self,
        ct: &BgvCiphertext,
        gk: &GaloisKey,
    ) -> Result<Vec<BgvCiphertext>, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        (0..ct.params().degree())
            .map(|slot| self.replicate_slot_bgv(ct, slot, gk))
            .collect()
    }

    pub fn inner_product_bgv(
        &self,
        c1: &BgvCiphertext,
        c2: &BgvCiphertext,
        active_slots: usize,
        evk: &EvaluationKey,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        let product = self.mul_relinearized_bgv(c1, c2, evk)?;
        self.sum_active_slots_bgv(&product, active_slots, gk)
    }

    pub fn inner_product_plain_bgv(
        &self,
        ct: &BgvCiphertext,
        plaintext: &Plaintext,
        active_slots: usize,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        ensure_plaintext_compatible(ct.params(), plaintext)?;
        let product = self.mul_plain_bgv(ct, plaintext)?;
        self.sum_active_slots_bgv(&product, active_slots, gk)
    }

    pub fn linear_weighted_sum_plain_bgv(
        &self,
        ciphertexts: &[BgvCiphertext],
        weights: &[Plaintext],
    ) -> Result<BgvCiphertext, BgvLevelError> {
        if ciphertexts.is_empty() {
            return Err(BgvLevelError::EmptyLinearCombinationInput);
        }
        if ciphertexts.len() != weights.len() {
            return Err(BgvLevelError::LinearCombinationLengthMismatch {
                ciphertexts: ciphertexts.len(),
                plaintexts: weights.len(),
            });
        }

        let mut result = None;
        for (ct, weight) in ciphertexts.iter().zip(weights.iter()) {
            self.ensure_compatible_bgv_operand(ct)?;
            ensure_plaintext_compatible(ct.params(), weight)?;
            let term = self.mul_plain_bgv(ct, weight)?;
            result = Some(match result {
                Some(acc) => self.add_bgv(&acc, &term)?,
                None => term,
            });
        }

        result.ok_or(BgvLevelError::EmptyLinearCombinationInput)
    }

    pub fn evaluate_polynomial_plain_bgv(
        &self,
        ct: &BgvCiphertext,
        coefficients: &[Plaintext],
        evk: &EvaluationKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        if coefficients.is_empty() {
            return Err(BgvLevelError::EmptyPolynomialEvaluation);
        }
        self.ensure_compatible_bgv_operand(ct)?;
        for coefficient in coefficients {
            ensure_plaintext_compatible(ct.params(), coefficient)?;
        }

        let zero = zero_plaintext(ct.params());
        let mut result = self.mul_plain_bgv(ct, &zero)?;
        result = self.add_plain_bgv(
            &result,
            coefficients
                .last()
                .expect("BGV polynomial coefficient list is non-empty"),
        )?;

        for coefficient in coefficients[..coefficients.len() - 1].iter().rev() {
            result = self.mul_relinearized_bgv(&result, ct, evk)?;
            result = self.add_plain_bgv(&result, coefficient)?;
        }

        Ok(result)
    }

    pub fn evaluate_polynomials_plain_bgv(
        &self,
        ct: &BgvCiphertext,
        polynomials: &[Vec<Plaintext>],
        evk: &EvaluationKey,
    ) -> Result<Vec<BgvCiphertext>, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        if polynomials.is_empty() {
            return Ok(Vec::new());
        }

        let mut max_degree = 0usize;
        for polynomial in polynomials {
            if polynomial.is_empty() {
                return Err(BgvLevelError::EmptyPolynomialEvaluation);
            }
            max_degree = max_degree.max(polynomial.len() - 1);
            for coefficient in polynomial {
                ensure_plaintext_compatible(ct.params(), coefficient)?;
            }
        }

        let zero = zero_plaintext(ct.params());
        let mut powers = Vec::with_capacity(max_degree);
        if max_degree > 0 {
            powers.push(ct.clone());
            while powers.len() < max_degree {
                let next = self.mul_relinearized_bgv(
                    powers
                        .last()
                        .expect("BGV power basis has at least x before extension"),
                    ct,
                    evk,
                )?;
                powers.push(next);
            }
        }

        let mut outputs = Vec::with_capacity(polynomials.len());
        for polynomial in polynomials {
            let mut result = self.mul_plain_bgv(ct, &zero)?;
            result = self.add_plain_bgv(&result, &polynomial[0])?;
            for (power_idx, coefficient) in polynomial.iter().enumerate().skip(1) {
                let term = self.mul_plain_bgv(&powers[power_idx - 1], coefficient)?;
                result = self.add_bgv(&result, &term)?;
            }
            outputs.push(result);
        }

        Ok(outputs)
    }

    pub fn slot_linear_transform_plain_bgv(
        &self,
        ct: &BgvCiphertext,
        matrix: &[Vec<u64>],
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let diagonals = slot_matrix_diagonal_plaintexts(ct.params(), matrix)?;
        self.diagonal_linear_transform_bgv(ct, &diagonals, gk)
    }

    pub fn slot_linear_transform_plain_bsgs_bgv(
        &self,
        ct: &BgvCiphertext,
        matrix: &[Vec<u64>],
        baby_galois_elements: &[u32],
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let diagonals = slot_matrix_diagonal_plaintexts(ct.params(), matrix)?;
        self.diagonal_linear_transform_bsgs_bgv(ct, &diagonals, baby_galois_elements, gk)
    }

    pub fn slot_linear_transform_plain_many_bgv(
        &self,
        ct: &BgvCiphertext,
        matrices: &[Vec<Vec<u64>>],
        gk: &GaloisKey,
    ) -> Result<Vec<BgvCiphertext>, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let transforms = slot_matrix_many_diagonal_plaintexts(ct.params(), matrices)?;
        self.diagonal_linear_transform_many_bgv(ct, &transforms, gk)
    }

    pub fn sum_slot_matrix_rows_bgv(
        &self,
        ct: &BgvCiphertext,
        rows: usize,
        columns: usize,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let diagonals =
            slot_matrix_axis_sum_plaintexts(ct.params(), rows, columns, BgvSlotMatrixAxis::Rows)?;
        self.diagonal_linear_transform_bgv(ct, &diagonals, gk)
    }

    pub fn sum_slot_matrix_columns_bgv(
        &self,
        ct: &BgvCiphertext,
        rows: usize,
        columns: usize,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let diagonals = slot_matrix_axis_sum_plaintexts(
            ct.params(),
            rows,
            columns,
            BgvSlotMatrixAxis::Columns,
        )?;
        self.diagonal_linear_transform_bgv(ct, &diagonals, gk)
    }

    pub fn sum_slot_matrix_rows_bsgs_bgv(
        &self,
        ct: &BgvCiphertext,
        rows: usize,
        columns: usize,
        baby_galois_elements: &[u32],
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let diagonals =
            slot_matrix_axis_sum_plaintexts(ct.params(), rows, columns, BgvSlotMatrixAxis::Rows)?;
        self.diagonal_linear_transform_bsgs_bgv(ct, &diagonals, baby_galois_elements, gk)
    }

    pub fn sum_slot_matrix_columns_bsgs_bgv(
        &self,
        ct: &BgvCiphertext,
        rows: usize,
        columns: usize,
        baby_galois_elements: &[u32],
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let diagonals = slot_matrix_axis_sum_plaintexts(
            ct.params(),
            rows,
            columns,
            BgvSlotMatrixAxis::Columns,
        )?;
        self.diagonal_linear_transform_bsgs_bgv(ct, &diagonals, baby_galois_elements, gk)
    }

    pub fn diagonal_linear_transform_bgv(
        &self,
        ct: &BgvCiphertext,
        diagonals: &[(u32, Plaintext)],
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        if diagonals.is_empty() {
            return Err(BgvLevelError::EmptyDiagonalTransform);
        }
        self.ensure_compatible_bgv_operand(ct)?;
        for (_, diagonal) in diagonals {
            ensure_plaintext_compatible(ct.params(), diagonal)?;
        }

        let precomp = self.fast_rotation_precompute_bgv(ct)?;
        let mut result = None;
        for (element, diagonal) in diagonals {
            if !is_identity_galois_element(*element) && !gk.keys.contains_key(element) {
                return Err(BgvLevelError::MissingGaloisKey { index: *element });
            }

            let rotated = if is_identity_galois_element(*element) {
                ct.clone()
            } else {
                self.fast_rotate_bgv(ct, &precomp, *element, gk)?
            };
            let term = self.mul_plain_bgv(&rotated, diagonal)?;
            result = Some(match result {
                Some(acc) => self.add_bgv(&acc, &term)?,
                None => term,
            });
        }

        result.ok_or(BgvLevelError::EmptyDiagonalTransform)
    }

    pub fn diagonal_linear_transform_many_bgv(
        &self,
        ct: &BgvCiphertext,
        transforms: &[Vec<(u32, Plaintext)>],
        gk: &GaloisKey,
    ) -> Result<Vec<BgvCiphertext>, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        if transforms.is_empty() {
            return Ok(Vec::new());
        }

        let mut needed_rotations = Vec::new();
        for transform in transforms {
            if transform.is_empty() {
                return Err(BgvLevelError::EmptyDiagonalTransform);
            }
            for (element, diagonal) in transform {
                ensure_plaintext_compatible(ct.params(), diagonal)?;
                let element = canonical_galois_element(*element);
                if element == 1 {
                    continue;
                }
                if !gk.keys.contains_key(&element) {
                    return Err(BgvLevelError::MissingGaloisKey { index: element });
                }
                if !needed_rotations.contains(&element) {
                    needed_rotations.push(element);
                }
            }
        }

        let precomp = self.fast_rotation_precompute_bgv(ct)?;
        let mut rotations = HashMap::new();
        rotations.insert(1u32, ct.clone());
        for element in needed_rotations {
            rotations.insert(element, self.fast_rotate_bgv(ct, &precomp, element, gk)?);
        }

        let mut outputs = Vec::with_capacity(transforms.len());
        for transform in transforms {
            let mut result: Option<BgvCiphertext> = None;
            for (element, diagonal) in transform {
                let element = canonical_galois_element(*element);
                let rotated = rotations
                    .get(&element)
                    .expect("BGV diagonal rotation was precomputed");
                let term = self.mul_plain_bgv(rotated, diagonal)?;
                result = Some(match result {
                    Some(acc) => self.add_bgv(&acc, &term)?,
                    None => term,
                });
            }
            outputs.push(result.ok_or(BgvLevelError::EmptyDiagonalTransform)?);
        }

        Ok(outputs)
    }

    pub fn diagonal_linear_transform_bsgs_bgv(
        &self,
        ct: &BgvCiphertext,
        diagonals: &[(u32, Plaintext)],
        baby_galois_elements: &[u32],
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        if diagonals.is_empty() {
            return Err(BgvLevelError::EmptyDiagonalTransform);
        }
        self.ensure_compatible_bgv_operand(ct)?;
        for (_, diagonal) in diagonals {
            ensure_plaintext_compatible(ct.params(), diagonal)?;
        }

        let groups = build_diagonal_bsgs_groups(ct.params(), diagonals, baby_galois_elements)?;
        ensure_bsgs_galois_keys(&groups, gk)?;

        let precomp = self.fast_rotation_precompute_bgv(ct)?;
        let mut baby_rotations = HashMap::new();
        baby_rotations.insert(1u32, ct.clone());
        for baby in bsgs_unique_babies(&groups) {
            if baby != 1 {
                baby_rotations.insert(baby, self.fast_rotate_bgv(ct, &precomp, baby, gk)?);
            }
        }

        let mut result = None;
        for group in groups {
            let mut inner = None;
            for term in group.terms {
                let adjusted =
                    automorph_plaintext_coeff(&term.diagonal, group.giant_inverse, ct.params())?;
                let baby_ct = baby_rotations
                    .get(&term.baby)
                    .expect("BGV baby rotation was precomputed");
                let product = self.mul_plain_bgv(baby_ct, &adjusted)?;
                inner = Some(match inner {
                    Some(acc) => self.add_bgv(&acc, &product)?,
                    None => product,
                });
            }

            let inner = inner.ok_or(BgvLevelError::EmptyDiagonalTransform)?;
            let outer = if group.giant == 1 {
                inner
            } else {
                self.rotate_bgv(&inner, group.giant, gk)?
            };
            result = Some(match result {
                Some(acc) => self.add_bgv(&acc, &outer)?,
                None => outer,
            });
        }

        result.ok_or(BgvLevelError::EmptyDiagonalTransform)
    }

    pub fn merge_slots_bgv(
        &self,
        ciphertexts: &[BgvCiphertext],
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        if ciphertexts.is_empty() {
            return Err(BgvLevelError::EmptySlotMergeInput);
        }
        if ciphertexts.len() > self.params.degree() {
            return Err(BgvLevelError::SlotMergeInputCountExceeded {
                requested: ciphertexts.len(),
                available: self.params.degree(),
            });
        }
        for ct in ciphertexts {
            self.ensure_compatible_bgv_operand(ct)?;
        }
        for k in bgv_slot_merge_galois_elements(&self.params, ciphertexts.len()) {
            if !gk.keys.contains_key(&k) {
                return Err(BgvLevelError::MissingGaloisKey { index: k });
            }
        }

        let selector = slot_mask_plaintext(&self.params, 0);
        let mut merged = self.mul_plain_bgv(&ciphertexts[0], &selector)?;
        for (slot, ct) in ciphertexts.iter().enumerate().skip(1) {
            let k = bgv_slot_zero_to_slot_galois_element(ct.params(), slot).ok_or(
                BgvLevelError::SlotMergeInputCountExceeded {
                    requested: slot + 1,
                    available: ct.params().degree(),
                },
            )?;
            let masked = self.mul_plain_bgv(ct, &selector)?;
            let rotated = self.rotate_bgv(&masked, k, gk)?;
            merged = self.add_bgv(&merged, &rotated)?;
        }

        Ok(merged)
    }

    pub fn add_plain_bgv(
        &self,
        ct: &BgvCiphertext,
        plaintext: &Plaintext,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let encoded =
            scale_plaintext_by_factor(plaintext, ct.int_factor(), ct.params().plain_modulus())?;
        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        let raw = level_evaluator.add_plain(ct.raw(), &encoded);
        BgvCiphertext::with_metadata(
            ct.params().clone(),
            raw,
            ct.int_factor(),
            ct.noise_scale_degree(),
        )
    }

    pub fn sub_plain_bgv(
        &self,
        ct: &BgvCiphertext,
        plaintext: &Plaintext,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let encoded =
            scale_plaintext_by_factor(plaintext, ct.int_factor(), ct.params().plain_modulus())?;
        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        let raw = level_evaluator.sub_plain(ct.raw(), &encoded);
        BgvCiphertext::with_metadata(
            ct.params().clone(),
            raw,
            ct.int_factor(),
            ct.noise_scale_degree(),
        )
    }

    pub fn mul_plain_bgv(
        &self,
        ct: &BgvCiphertext,
        plaintext: &Plaintext,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        let raw = level_evaluator.mul_plain(ct.raw(), plaintext);
        BgvCiphertext::with_metadata(
            ct.params().clone(),
            raw,
            ct.int_factor(),
            ct.noise_scale_degree(),
        )
    }

    pub fn drop_level_bgv(
        &self,
        ct: &BgvCiphertext,
        levels: usize,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        let (params, raw) = level_evaluator.drop_level(ct.raw(), levels)?;
        BgvCiphertext::with_metadata(params, raw, ct.int_factor(), ct.noise_scale_degree())
    }

    pub fn mod_switch_down_bgv(
        &self,
        ct: &BgvCiphertext,
        levels: usize,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        if levels == 0 {
            return Ok(ct.clone());
        }
        if levels >= ct.q_modulus_count() {
            return Err(BgvLevelError::TooManyCiphertextModuli {
                requested: ct.q_modulus_count().saturating_sub(levels),
                available: ct.q_modulus_count(),
            });
        }
        self.mod_switch_to_q_limbs_bgv(ct, ct.q_modulus_count() - levels)
    }

    pub fn mod_switch_to_q_limbs_bgv(
        &self,
        ct: &BgvCiphertext,
        q_limbs: usize,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        if q_limbs == ct.q_modulus_count() {
            return Ok(ct.clone());
        }

        let level_evaluator = BgvEvaluator::new(ct.params().clone());
        let (params, raw) = level_evaluator.mod_switch_to_q_limbs(ct.raw(), q_limbs)?;
        let factor = mod_switch_plaintext_factor(ct.int_factor(), ct.params(), q_limbs)?;
        let levels = ct.q_modulus_count() - q_limbs;
        let noise_scale_degree = ct.noise_scale_degree().saturating_sub(levels);
        BgvCiphertext::with_metadata(params, raw, factor, noise_scale_degree)
    }

    pub fn compress_bgv(
        &self,
        ct: &BgvCiphertext,
        q_limbs: usize,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.mod_switch_to_q_limbs_bgv(ct, q_limbs)
    }

    pub fn scale_plaintext_factor_bgv(
        &self,
        ct: &BgvCiphertext,
        factor: u64,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        let factor = BgvPlaintextFactor::new(factor, ct.params().plain_modulus())?;
        scale_bgv_ciphertext_by_factor(ct, factor)
    }

    pub fn sum_active_slots_bgv(
        &self,
        ct: &BgvCiphertext,
        active_slots: usize,
        gk: &GaloisKey,
    ) -> Result<BgvCiphertext, BgvLevelError> {
        self.ensure_compatible_bgv_operand(ct)?;
        if active_slots == 0 || active_slots > ct.params().degree() {
            return Err(BgvLevelError::InvalidActiveSlotCount {
                requested: active_slots,
                available: ct.params().degree(),
            });
        }

        let selected_slots = (0..active_slots).collect::<Vec<_>>();
        let selected =
            self.mul_plain_bgv(ct, &slot_mask_plaintext_many(ct.params(), &selected_slots))?;
        self.sum_slots_bgv(&selected, gk)
    }

    pub fn relinearize(&self, ct: &Ciphertext, evk: &EvaluationKey) -> Ciphertext {
        if ct.data.len() != 3 {
            return ct.clone();
        }

        let ring = self.params.ring();
        let mut c0 = ct.data[0].clone();
        let mut c1 = ct.data[1].clone();
        let mut c2 = ct.data[2].clone();
        if ct.is_ntt {
            c0.ntt_inverse(ring);
            c1.ntt_inverse(ring);
            c2.ntt_inverse(ring);
        }

        let (sw0, sw1) = self.switch_key_gadget(&c2, evk);
        c0.add_assign(&sw0, ring);
        c1.add_assign(&sw1, ring);

        Ciphertext::new(vec![c0, c1], self.params.runtime_params().clone(), false)
    }

    pub fn proxy_reencrypt(
        &self,
        ct: &Ciphertext,
        rekey: &EvaluationKey,
    ) -> Result<Ciphertext, BgvLevelError> {
        self.ensure_full_level_ciphertext(ct)?;
        if ct.data.len() != 2 {
            return Err(BgvLevelError::ProxyReencryptionCiphertextSize {
                actual: ct.data.len(),
            });
        }
        if rekey.elements.len() != bgv_ksw_levels(&self.params) {
            return Err(BgvLevelError::EvaluationKeyLevelMismatch {
                expected: bgv_ksw_levels(&self.params),
                actual: rekey.elements.len(),
            });
        }

        let ring = self.params.ring();
        let mut c0 = ct.data[0].clone();
        let mut c1 = ct.data[1].clone();
        if ct.is_ntt {
            c0.ntt_inverse(ring);
            c1.ntt_inverse(ring);
        }

        let (sw0, sw1) = self.switch_key_gadget(&c1, rekey);
        c0.add_assign(&sw0, ring);

        Ok(Ciphertext::new(
            vec![c0, sw1],
            self.params.runtime_params().clone(),
            false,
        ))
    }

    pub fn mul_relinearized(
        &self,
        c1: &Ciphertext,
        c2: &Ciphertext,
        evk: &EvaluationKey,
    ) -> Ciphertext {
        let product = self.mul(c1, c2);
        self.relinearize(&product, evk)
    }

    pub fn sum_slots(&self, ct: &Ciphertext, gk: &GaloisKey) -> Ciphertext {
        let precomp = self
            .fast_rotation_precompute(ct)
            .expect("BGV fast rotation precomputation failed");
        let mut acc = ct.clone();
        for k in bgv_full_slot_sum_galois_elements(self.params.degree()) {
            let rotated = self
                .fast_rotate(&precomp, k, gk)
                .expect("BGV fast rotation failed");
            acc = self.add(&acc, &rotated);
        }
        acc
    }

    pub fn diagonal_linear_transform(
        &self,
        ct: &Ciphertext,
        diagonals: &[(u32, Plaintext)],
        gk: &GaloisKey,
    ) -> Result<Ciphertext, BgvLevelError> {
        if diagonals.is_empty() {
            return Err(BgvLevelError::EmptyDiagonalTransform);
        }
        self.ensure_full_level_ciphertext(ct)?;
        for (_, diagonal) in diagonals {
            ensure_plaintext_compatible(&self.params, diagonal)?;
        }

        let precomp = self.fast_rotation_precompute(ct)?;
        let mut result = None;
        for (element, diagonal) in diagonals {
            if !is_identity_galois_element(*element) && !gk.keys.contains_key(element) {
                return Err(BgvLevelError::MissingGaloisKey { index: *element });
            }

            let rotated = if is_identity_galois_element(*element) {
                precomp.coeff.clone()
            } else {
                self.fast_rotate(&precomp, *element, gk)?
            };
            let term = self.mul_plain(&rotated, diagonal);
            result = Some(match result {
                Some(acc) => self.add(&acc, &term),
                None => term,
            });
        }

        result.ok_or(BgvLevelError::EmptyDiagonalTransform)
    }

    pub fn diagonal_linear_transform_many(
        &self,
        ct: &Ciphertext,
        transforms: &[Vec<(u32, Plaintext)>],
        gk: &GaloisKey,
    ) -> Result<Vec<Ciphertext>, BgvLevelError> {
        self.ensure_full_level_ciphertext(ct)?;
        if transforms.is_empty() {
            return Ok(Vec::new());
        }

        let mut needed_rotations = Vec::new();
        for transform in transforms {
            if transform.is_empty() {
                return Err(BgvLevelError::EmptyDiagonalTransform);
            }
            for (element, diagonal) in transform {
                ensure_plaintext_compatible(&self.params, diagonal)?;
                let element = canonical_galois_element(*element);
                if element == 1 {
                    continue;
                }
                if !gk.keys.contains_key(&element) {
                    return Err(BgvLevelError::MissingGaloisKey { index: element });
                }
                if !needed_rotations.contains(&element) {
                    needed_rotations.push(element);
                }
            }
        }

        let precomp = self.fast_rotation_precompute(ct)?;
        let mut rotations = HashMap::new();
        rotations.insert(1u32, precomp.coeff.clone());
        for element in needed_rotations {
            rotations.insert(element, self.fast_rotate(&precomp, element, gk)?);
        }

        let mut outputs = Vec::with_capacity(transforms.len());
        for transform in transforms {
            let mut result: Option<Ciphertext> = None;
            for (element, diagonal) in transform {
                let element = canonical_galois_element(*element);
                let rotated = rotations
                    .get(&element)
                    .expect("BGV diagonal rotation was precomputed");
                let term = self.mul_plain(rotated, diagonal);
                result = Some(match result {
                    Some(acc) => self.add(&acc, &term),
                    None => term,
                });
            }
            outputs.push(result.ok_or(BgvLevelError::EmptyDiagonalTransform)?);
        }

        Ok(outputs)
    }

    pub fn diagonal_linear_transform_bsgs(
        &self,
        ct: &Ciphertext,
        diagonals: &[(u32, Plaintext)],
        baby_galois_elements: &[u32],
        gk: &GaloisKey,
    ) -> Result<Ciphertext, BgvLevelError> {
        if diagonals.is_empty() {
            return Err(BgvLevelError::EmptyDiagonalTransform);
        }
        self.ensure_full_level_ciphertext(ct)?;
        for (_, diagonal) in diagonals {
            ensure_plaintext_compatible(&self.params, diagonal)?;
        }

        let groups = build_diagonal_bsgs_groups(&self.params, diagonals, baby_galois_elements)?;
        ensure_bsgs_galois_keys(&groups, gk)?;

        let precomp = self.fast_rotation_precompute(ct)?;
        let mut baby_rotations = HashMap::new();
        baby_rotations.insert(1u32, precomp.coeff.clone());
        for baby in bsgs_unique_babies(&groups) {
            if baby != 1 {
                baby_rotations.insert(baby, self.fast_rotate(&precomp, baby, gk)?);
            }
        }

        let mut result = None;
        for group in groups {
            let mut inner = None;
            for term in group.terms {
                let adjusted =
                    automorph_plaintext_coeff(&term.diagonal, group.giant_inverse, &self.params)?;
                let baby_ct = baby_rotations
                    .get(&term.baby)
                    .expect("BGV baby rotation was precomputed");
                let product = self.mul_plain(baby_ct, &adjusted);
                inner = Some(match inner {
                    Some(acc) => self.add(&acc, &product),
                    None => product,
                });
            }

            let inner = inner.ok_or(BgvLevelError::EmptyDiagonalTransform)?;
            let outer = if group.giant == 1 {
                inner
            } else {
                self.rotate(&inner, group.giant, gk)
            };
            result = Some(match result {
                Some(acc) => self.add(&acc, &outer),
                None => outer,
            });
        }

        result.ok_or(BgvLevelError::EmptyDiagonalTransform)
    }

    pub fn merge_slots(
        &self,
        ciphertexts: &[Ciphertext],
        gk: &GaloisKey,
    ) -> Result<Ciphertext, BgvLevelError> {
        if ciphertexts.is_empty() {
            return Err(BgvLevelError::EmptySlotMergeInput);
        }
        if ciphertexts.len() > self.params.degree() {
            return Err(BgvLevelError::SlotMergeInputCountExceeded {
                requested: ciphertexts.len(),
                available: self.params.degree(),
            });
        }

        for ct in ciphertexts {
            self.ensure_full_level_ciphertext(ct)?;
        }
        for k in bgv_slot_merge_galois_elements(&self.params, ciphertexts.len()) {
            if !gk.keys.contains_key(&k) {
                return Err(BgvLevelError::MissingGaloisKey { index: k });
            }
        }

        let selector = slot_mask_plaintext(&self.params, 0);
        let mut merged = self.mul_plain(&ciphertexts[0], &selector);
        for (slot, ct) in ciphertexts.iter().enumerate().skip(1) {
            let k = bgv_slot_zero_to_slot_galois_element(&self.params, slot).ok_or(
                BgvLevelError::SlotMergeInputCountExceeded {
                    requested: slot + 1,
                    available: self.params.degree(),
                },
            )?;
            if !gk.keys.contains_key(&k) {
                return Err(BgvLevelError::MissingGaloisKey { index: k });
            }
            let masked = self.mul_plain(ct, &selector);
            let rotated = self.rotate(&masked, k, gk);
            merged = self.add(&merged, &rotated);
        }

        Ok(merged)
    }

    pub fn fast_rotation_precompute(
        &self,
        ct: &Ciphertext,
    ) -> Result<BgvFastRotationPrecomputation, BgvLevelError> {
        self.ensure_full_level_ciphertext(ct)?;
        let coeff = to_coeff_domain(ct, self.params.ring());
        let c1_digits = if coeff.data.len() == 2 {
            decompose_gadget_digits(
                &coeff.data[1],
                self.params.ring(),
                bgv_ksw_levels(&self.params),
            )
        } else {
            Vec::new()
        };

        Ok(BgvFastRotationPrecomputation {
            params: self.params.clone(),
            coeff,
            c1_digits,
        })
    }

    pub fn fast_rotate(
        &self,
        precomp: &BgvFastRotationPrecomputation,
        k: u32,
        gk: &GaloisKey,
    ) -> Result<Ciphertext, BgvLevelError> {
        if k == 0 {
            return Ok(precomp.coeff.clone());
        }
        let switch_key = gk
            .keys
            .get(&k)
            .ok_or(BgvLevelError::MissingGaloisKey { index: k })?;
        self.fast_rotate_with_key(precomp, k, switch_key)
    }

    pub fn add_plain(&self, ct: &Ciphertext, plaintext: &Plaintext) -> Ciphertext {
        let mut out = to_coeff_domain(ct, self.params.ring());
        add_plain_to_c0(
            &mut out.data[0],
            plaintext,
            self.params.ring(),
            self.params.plain_modulus(),
            LinearOp::Add,
        );
        out
    }

    pub fn sub_plain(&self, ct: &Ciphertext, plaintext: &Plaintext) -> Ciphertext {
        let mut out = to_coeff_domain(ct, self.params.ring());
        add_plain_to_c0(
            &mut out.data[0],
            plaintext,
            self.params.ring(),
            self.params.plain_modulus(),
            LinearOp::Sub,
        );
        out
    }

    pub fn mul_plain(&self, ct: &Ciphertext, plaintext: &Plaintext) -> Ciphertext {
        let ring = self.params.ring();
        let plain_ntt = lift_plaintext_ntt(plaintext, ring, self.params.plain_modulus());
        let mut out = ct.data.clone();

        if !ct.is_ntt {
            for poly in out.iter_mut() {
                poly.ntt_forward(ring);
            }
        }

        for poly in out.iter_mut() {
            poly.mul_assign(&plain_ntt, ring);
            poly.ntt_inverse(ring);
        }

        Ciphertext::new(out, self.params.runtime_params().clone(), false)
    }

    pub fn rotate(&self, ct: &Ciphertext, k: u32, gk: &GaloisKey) -> Ciphertext {
        if ct.data.len() != 2 {
            return ct.clone();
        }
        let Some(switch_key) = gk.keys.get(&k) else {
            if k == 0 {
                return ct.clone();
            }
            panic!("Rotation key for k={k} not found");
        };

        let ring = self.params.ring();
        let degree = ring.degree();
        let mut c0_rot = ct.data[0].clone();
        let mut c1_rot = ct.data[1].clone();
        if ct.is_ntt {
            c0_rot.ntt_inverse(ring);
            c1_rot.ntt_inverse(ring);
        }

        let mut scratch = vec![0u64; degree];
        apply_galois_coeff_inplace(&mut c0_rot, k, ring, &mut scratch);
        apply_galois_coeff_inplace(&mut c1_rot, k, ring, &mut scratch);

        let (sw0, sw1) = self.switch_key_gadget(&c1_rot, switch_key);
        c0_rot.add_assign(&sw0, ring);

        Ciphertext::new(
            vec![c0_rot, sw1],
            self.params.runtime_params().clone(),
            false,
        )
    }

    fn fast_rotate_with_key(
        &self,
        precomp: &BgvFastRotationPrecomputation,
        k: u32,
        switch_key: &EvaluationKey,
    ) -> Result<Ciphertext, BgvLevelError> {
        self.ensure_compatible_fast_rotation_precomp(precomp)?;
        if precomp.coeff.data.len() != 2 || k == 0 {
            return Ok(precomp.coeff.clone());
        }

        let ring = self.params.ring();
        let degree = ring.degree();
        let mut scratch = vec![0u64; degree];
        let mut c0_rot = precomp.coeff.data[0].clone();
        apply_galois_coeff_inplace(&mut c0_rot, k, ring, &mut scratch);

        let rotated_digits = precomp
            .c1_digits
            .iter()
            .map(|digit| {
                let mut rotated = digit.clone();
                apply_galois_coeff_inplace(&mut rotated, k, ring, &mut scratch);
                rotated
            })
            .collect::<Vec<_>>();
        let (sw0, sw1) = self.switch_key_gadget_from_digits(rotated_digits, switch_key);
        c0_rot.add_assign(&sw0, ring);

        Ok(Ciphertext::new(
            vec![c0_rot, sw1],
            self.params.runtime_params().clone(),
            false,
        ))
    }

    fn switch_key_gadget(&self, poly_coeff: &Poly, evk: &EvaluationKey) -> (Poly, Poly) {
        let ring = self.params.ring();
        let digits = decompose_gadget_digits(poly_coeff, ring, evk.elements.len());
        self.switch_key_gadget_from_digits(digits, evk)
    }

    fn switch_key_gadget_from_digits(
        &self,
        digits: Vec<Poly>,
        evk: &EvaluationKey,
    ) -> (Poly, Poly) {
        let ring = self.params.ring();
        let degree = ring.degree();
        let size_q = ring.rns().len();
        let expected_levels = bgv_ksw_levels(&self.params);
        assert_eq!(
            evk.elements.len(),
            expected_levels,
            "BGV key-switch key level count mismatch"
        );
        assert_eq!(
            digits.len(),
            expected_levels,
            "BGV key-switch digit count mismatch"
        );

        let mut acc0 = Poly::new(degree, size_q);
        let mut acc1 = Poly::new(degree, size_q);

        for (digit, key_ct) in digits.into_iter().zip(evk.elements.iter()) {
            let mut digit_ntt = digit;
            digit_ntt.ntt_forward(ring);

            let mut term0 = digit_ntt.clone();
            term0.mul_assign(&key_ct.data[0], ring);
            acc0.add_assign(&term0, ring);

            let mut term1 = digit_ntt;
            term1.mul_assign(&key_ct.data[1], ring);
            acc1.add_assign(&term1, ring);
        }

        acc0.ntt_inverse(ring);
        acc1.ntt_inverse(ring);
        (acc0, acc1)
    }

    fn ensure_compatible_fast_rotation_precomp(
        &self,
        precomp: &BgvFastRotationPrecomputation,
    ) -> Result<(), BgvLevelError> {
        if precomp.params.degree() != self.params.degree() {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.params.degree(),
                actual: precomp.params.degree(),
            });
        }
        if precomp.params.plain_modulus() != self.params.plain_modulus() {
            return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
                expected: self.params.plain_modulus(),
                actual: precomp.params.plain_modulus(),
            });
        }
        if !same_modulus_chain(precomp.params.ring(), self.params.ring()) {
            return Err(BgvLevelError::CiphertextModulusChainMismatch);
        }
        Ok(())
    }

    fn ensure_full_level_ciphertext(&self, ct: &Ciphertext) -> Result<(), BgvLevelError> {
        let expected_degree = self.params.degree();
        let expected_limbs = self.params.q_modulus_count();
        for poly in ct.data.iter() {
            if poly.degree() != expected_degree {
                return Err(BgvLevelError::CiphertextDegreeMismatch {
                    expected: expected_degree,
                    actual: poly.degree(),
                });
            }
            if poly.num_moduli() != expected_limbs {
                return Err(BgvLevelError::CiphertextLimbMismatch {
                    expected: expected_limbs,
                    actual: poly.num_moduli(),
                });
            }
        }
        Ok(())
    }

    fn align_bgv_levels(
        &self,
        c1: &BgvCiphertext,
        c2: &BgvCiphertext,
    ) -> Result<(BgvParameters, BgvCiphertext, BgvCiphertext), BgvLevelError> {
        self.ensure_compatible_bgv_operand(c1)?;
        self.ensure_compatible_bgv_operand(c2)?;

        let target_limbs = c1.q_modulus_count().min(c2.q_modulus_count());
        let left = mod_switch_bgv_to_q_limbs(c1, target_limbs)?;
        let right = mod_switch_bgv_to_q_limbs(c2, target_limbs)?;
        ensure_same_bgv_modulus_chain(&left, &right)?;

        Ok((left.params().clone(), left, right))
    }

    fn ensure_compatible_bgv_operand(&self, ct: &BgvCiphertext) -> Result<(), BgvLevelError> {
        if ct.params().degree() != self.params.degree() {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.params.degree(),
                actual: ct.params().degree(),
            });
        }
        if ct.params().plain_modulus() != self.params.plain_modulus() {
            return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
                expected: self.params.plain_modulus(),
                actual: ct.params().plain_modulus(),
            });
        }
        Ok(())
    }

    fn evaluation_key_for_params(
        &self,
        evk: &EvaluationKey,
        params: &BgvParameters,
    ) -> Result<EvaluationKey, BgvLevelError> {
        if !is_prefix_modulus_chain(self.params.ring(), params.ring()) {
            return Err(BgvLevelError::CiphertextModulusChainMismatch);
        }

        let expected_levels = bgv_ksw_levels(params);
        if evk.elements.len() < expected_levels {
            return Err(BgvLevelError::EvaluationKeyLevelMismatch {
                expected: expected_levels,
                actual: evk.elements.len(),
            });
        }

        let elements = evk
            .elements
            .iter()
            .take(expected_levels)
            .map(|key_ct| {
                if params.q_modulus_count() == self.params.q_modulus_count() {
                    Ok(key_ct.clone())
                } else {
                    let (_, raw) = self.drop_to_q_limbs(key_ct, params.q_modulus_count())?;
                    Ok(raw)
                }
            })
            .collect::<Result<Vec<_>, BgvLevelError>>()?;

        Ok(EvaluationKey { elements })
    }
}

#[derive(Clone, Copy)]
enum LinearOp {
    Add,
    Sub,
}

fn binary_linear_op(
    c1: &Ciphertext,
    c2: &Ciphertext,
    ring: &RingContext,
    op: LinearOp,
    params: &silent_rlwe::EncryptionParams,
) -> Ciphertext {
    let left = to_coeff_domain(c1, ring);
    let right = to_coeff_domain(c2, ring);
    let degree = ring.degree();
    let size_q = ring.rns().len();
    let size = left.data.len().max(right.data.len());
    let mut result = Vec::with_capacity(size);

    for i in 0..size {
        match (left.data.get(i), right.data.get(i)) {
            (Some(a), Some(b)) => {
                let mut poly = a.clone();
                match op {
                    LinearOp::Add => poly.add_assign(b, ring),
                    LinearOp::Sub => poly.sub_assign(b, ring),
                }
                result.push(poly);
            }
            (Some(a), None) => result.push(a.clone()),
            (None, Some(b)) => {
                let mut poly = b.clone();
                match op {
                    LinearOp::Add => {}
                    LinearOp::Sub => negate_poly_inplace(&mut poly, ring),
                }
                result.push(poly);
            }
            (None, None) => result.push(Poly::new(degree, size_q)),
        }
    }

    Ciphertext::new(result, params.clone(), false)
}

fn add_plain_to_c0(c0: &mut Poly, plaintext: &Plaintext, ring: &RingContext, t: u64, op: LinearOp) {
    assert_eq!(plaintext.value.degree(), ring.degree());
    assert_eq!(plaintext.value.num_moduli(), 1);
    let plain_limb = plaintext.value.limb(0);

    for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        let c0_limb = c0.limb_mut(limb_idx);
        for (dst, &plain_coeff) in c0_limb.iter_mut().zip(plain_limb.iter()) {
            let value = (plain_coeff % t) % q;
            *dst = match op {
                LinearOp::Add => arith::add_mod(*dst, value, q),
                LinearOp::Sub => arith::sub_mod(*dst, value, q),
            };
        }
    }
}

fn lift_plaintext_ntt(plaintext: &Plaintext, ring: &RingContext, t: u64) -> Poly {
    assert_eq!(plaintext.value.degree(), ring.degree());
    assert_eq!(plaintext.value.num_moduli(), 1);

    let degree = ring.degree();
    let size_q = ring.rns().len();
    let plain_limb = plaintext.value.limb(0);
    let mut out = Poly::new(degree, size_q);

    for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        let limb = out.limb_mut(limb_idx);
        for (dst, &plain_coeff) in limb.iter_mut().zip(plain_limb.iter()) {
            *dst = (plain_coeff % t) % q;
        }
    }

    out.ntt_forward(ring);
    out
}

fn to_coeff_domain(ct: &Ciphertext, ring: &RingContext) -> Ciphertext {
    let mut out = ct.clone();
    if out.is_ntt {
        for poly in out.data.iter_mut() {
            poly.ntt_inverse(ring);
        }
        out.is_ntt = false;
    }
    out
}

fn negate_poly_inplace(poly: &mut Poly, ring: &RingContext) {
    for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        for coeff in poly.limb_mut(limb_idx) {
            *coeff = arith::neg_mod(*coeff, q);
        }
    }
}

fn decompose_gadget_digits(poly: &Poly, ring: &RingContext, levels: usize) -> Vec<Poly> {
    let degree = ring.degree();
    let size_q = ring.rns().len();
    let q = rns_big::ciphertext_modulus(ring);
    let q_half = &q >> 1usize;
    let mut digits = vec![Poly::new(degree, size_q); levels];
    let mut residues = vec![0u64; size_q];

    for coeff in 0..degree {
        for (limb_idx, residue) in residues.iter_mut().enumerate() {
            *residue = poly.limb(limb_idx)[coeff];
        }
        let value = rns_big::compose_residues(&residues, ring, &q);
        let mut signed = if value > q_half {
            BigInt::from_biguint(Sign::Plus, value) - BigInt::from_biguint(Sign::Plus, q.clone())
        } else {
            BigInt::from_biguint(Sign::Plus, value)
        };

        for digit_poly in digits.iter_mut().take(levels) {
            let digit = take_balanced_digit_big(&mut signed);
            for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
                digit_poly.limb_mut(limb_idx)[coeff] = signed_digit_to_mod(digit, modulus.value());
            }
        }
    }

    digits
}

fn take_balanced_digit_big(value: &mut BigInt) -> i64 {
    let base = BigInt::one() << BGV_KSW_BASE_LOG;
    let half = &base >> 1usize;
    let mut digit = value.clone() % &base;
    if digit < BigInt::zero() {
        digit += &base;
    }
    if digit > half {
        digit -= &base;
    }
    *value = (value.clone() - &digit) / &base;
    digit.to_i64().expect("balanced gadget digit fits in i64")
}

fn signed_digit_to_mod(digit: i64, modulus: u64) -> u64 {
    if digit >= 0 {
        (digit as u64) % modulus
    } else {
        let abs = digit.unsigned_abs() % modulus;
        if abs == 0 { 0 } else { modulus - abs }
    }
}

fn truncate_poly_q_prefix(poly: &Poly, q_limbs: usize) -> Poly {
    let degree = poly.degree();
    let mut data = Vec::with_capacity(degree * q_limbs);
    for limb_idx in 0..q_limbs {
        data.extend_from_slice(poly.limb(limb_idx));
    }
    Poly::from_vec(data, degree, q_limbs)
}

fn mod_switch_poly_down_one(
    poly: &Poly,
    target_ring: &RingContext,
    plain_modulus: u64,
    step: &BgvModSwitchStep,
) -> Poly {
    let target_limbs = target_ring.rns().len();
    assert_eq!(poly.num_moduli(), step.source_q_limbs);
    assert_eq!(target_limbs, step.target_q_limbs);

    let degree = target_ring.degree();
    let mut out = Poly::new(degree, target_limbs);

    for coeff_idx in 0..degree {
        let dropped_residue = poly.limb(target_limbs)[coeff_idx];
        let neg_dropped_residue = if dropped_residue == 0 {
            0
        } else {
            step.dropped_modulus - dropped_residue
        };
        let correction_mod_dropped = ((u128::from(neg_dropped_residue)
            * u128::from(step.t_inv_mod_dropped))
            % u128::from(step.dropped_modulus)) as u64;

        for (limb_idx, modulus) in target_ring.rns().moduli().iter().enumerate() {
            let q = modulus.value();
            let correction_mod_q =
                centered_residue_mod_modulus(correction_mod_dropped, step.dropped_modulus, q);
            let adjusted = (u128::from(poly.limb(limb_idx)[coeff_idx])
                + u128::from(plain_modulus) * u128::from(correction_mod_q))
                % u128::from(q);
            out.limb_mut(limb_idx)[coeff_idx] = ((adjusted
                * u128::from(step.dropped_inv_mod_targets[limb_idx]))
                % u128::from(q)) as u64;
        }
    }

    out
}

fn centered_residue_mod_modulus(residue: u64, source_modulus: u64, target_modulus: u64) -> u64 {
    if residue > source_modulus / 2 {
        let abs = (source_modulus - residue) % target_modulus;
        if abs == 0 { 0 } else { target_modulus - abs }
    } else {
        residue % target_modulus
    }
}

fn ensure_same_bgv_level(c1: &BgvCiphertext, c2: &BgvCiphertext) -> Result<(), BgvLevelError> {
    if c1.q_modulus_count() != c2.q_modulus_count() {
        return Err(BgvLevelError::CiphertextLevelMismatch {
            left: c1.q_modulus_count(),
            right: c2.q_modulus_count(),
        });
    }
    Ok(())
}

fn ensure_same_bgv_modulus_chain(
    c1: &BgvCiphertext,
    c2: &BgvCiphertext,
) -> Result<(), BgvLevelError> {
    ensure_same_bgv_level(c1, c2)?;
    if !same_modulus_chain(c1.params().ring(), c2.params().ring()) {
        return Err(BgvLevelError::CiphertextModulusChainMismatch);
    }
    Ok(())
}

fn ensure_bgv_precomp_matches_ciphertext(
    precomp: &BgvFastRotationPrecomputation,
    ct: &BgvCiphertext,
) -> Result<(), BgvLevelError> {
    if precomp.params().degree() != ct.params().degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: ct.params().degree(),
            actual: precomp.params().degree(),
        });
    }
    if precomp.params().plain_modulus() != ct.params().plain_modulus() {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: ct.params().plain_modulus(),
            actual: precomp.params().plain_modulus(),
        });
    }
    if !same_modulus_chain(precomp.params().ring(), ct.params().ring()) {
        return Err(BgvLevelError::CiphertextModulusChainMismatch);
    }
    Ok(())
}

fn ensure_plaintext_compatible(
    params: &BgvParameters,
    plaintext: &Plaintext,
) -> Result<(), BgvLevelError> {
    if plaintext.value.degree() != params.degree() {
        return Err(BgvLevelError::PlaintextDegreeMismatch {
            expected: params.degree(),
            actual: plaintext.value.degree(),
        });
    }
    if plaintext.value.num_moduli() != 1 {
        return Err(BgvLevelError::PlaintextLimbMismatch {
            expected: 1,
            actual: plaintext.value.num_moduli(),
        });
    }
    Ok(())
}

fn is_identity_galois_element(element: u32) -> bool {
    element == 0 || element == 1
}

fn canonical_galois_element(element: u32) -> u32 {
    if is_identity_galois_element(element) {
        1
    } else {
        element
    }
}

fn build_diagonal_bsgs_groups(
    params: &BgvParameters,
    diagonals: &[(u32, Plaintext)],
    baby_galois_elements: &[u32],
) -> Result<Vec<BgvDiagonalBsgsGroup>, BgvLevelError> {
    let diagonal_elements = diagonals
        .iter()
        .map(|(element, _)| *element)
        .collect::<Vec<_>>();
    let assignments =
        bgv_diagonal_bsgs_assignments(params, &diagonal_elements, baby_galois_elements)?;
    let mut groups: Vec<BgvDiagonalBsgsGroup> = Vec::new();
    for assignment in assignments {
        let diagonal = diagonals[assignment.diagonal_index].1.clone();
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.giant == assignment.giant)
        {
            group.terms.push(BgvDiagonalBsgsTerm {
                baby: assignment.baby,
                diagonal,
            });
        } else {
            groups.push(BgvDiagonalBsgsGroup {
                giant: assignment.giant,
                giant_inverse: assignment.giant_inverse,
                terms: vec![BgvDiagonalBsgsTerm {
                    baby: assignment.baby,
                    diagonal,
                }],
            });
        }
    }

    Ok(groups)
}

fn ensure_bsgs_galois_keys(
    groups: &[BgvDiagonalBsgsGroup],
    gk: &GaloisKey,
) -> Result<(), BgvLevelError> {
    for group in groups {
        if group.giant != 1 && !gk.keys.contains_key(&group.giant) {
            return Err(BgvLevelError::MissingGaloisKey { index: group.giant });
        }
        for term in group.terms.iter() {
            if term.baby != 1 && !gk.keys.contains_key(&term.baby) {
                return Err(BgvLevelError::MissingGaloisKey { index: term.baby });
            }
        }
    }
    Ok(())
}

fn bsgs_unique_babies(groups: &[BgvDiagonalBsgsGroup]) -> Vec<u32> {
    let mut babies = Vec::new();
    for group in groups {
        for term in group.terms.iter() {
            if !babies.contains(&term.baby) {
                babies.push(term.baby);
            }
        }
    }
    babies
}

fn automorph_plaintext_coeff(
    plaintext: &Plaintext,
    element: u32,
    params: &BgvParameters,
) -> Result<Plaintext, BgvLevelError> {
    ensure_plaintext_compatible(params, plaintext)?;
    let element = bgv_canonical_galois_element_for_params(params, element)?;
    if element == 1 {
        return Ok(plaintext.clone());
    }

    let degree = params.degree();
    let modulus = params.plain_modulus();
    let map = params.ring().coeff_galois_map(element as u64);
    let src = plaintext.value.limb(0);
    let mut value = Poly::new(degree, 1);
    let dst = value.limb_mut(0);

    for i in 0..degree {
        let index = map[i] as usize;
        let coeff = src[i] % modulus;
        if index < degree {
            dst[index] = coeff;
        } else {
            let dest = index - degree;
            dst[dest] = if coeff == 0 { 0 } else { modulus - coeff };
        }
    }

    Ok(Plaintext { value })
}

fn same_modulus_chain(left: &RingContext, right: &RingContext) -> bool {
    left.rns().moduli() == right.rns().moduli()
}

fn is_prefix_modulus_chain(parent: &RingContext, child: &RingContext) -> bool {
    let child_len = child.rns().len();
    child_len <= parent.rns().len() && child.rns().moduli() == &parent.rns().moduli()[..child_len]
}

fn mod_switch_bgv_to_q_limbs(
    ct: &BgvCiphertext,
    q_limbs: usize,
) -> Result<BgvCiphertext, BgvLevelError> {
    if q_limbs == ct.q_modulus_count() {
        return Ok(ct.clone());
    }
    if q_limbs > ct.q_modulus_count() {
        return Err(BgvLevelError::TooManyCiphertextModuli {
            requested: q_limbs,
            available: ct.q_modulus_count(),
        });
    }

    let evaluator = BgvEvaluator::new(ct.params().clone());
    evaluator.mod_switch_to_q_limbs_bgv(ct, q_limbs)
}

fn mod_switch_plaintext_factor(
    factor: BgvPlaintextFactor,
    params: &BgvParameters,
    target_limbs: usize,
) -> Result<BgvPlaintextFactor, BgvLevelError> {
    let plain_modulus = params.plain_modulus();
    let dropped_factor_inv = params.mod_switch_factor_inv_to_q_limbs(target_limbs)?;
    factor.mul_mod(
        BgvPlaintextFactor::new(dropped_factor_inv, plain_modulus)?,
        plain_modulus,
    )
}

fn scale_bgv_ciphertext_by_factor(
    ct: &BgvCiphertext,
    factor: BgvPlaintextFactor,
) -> Result<BgvCiphertext, BgvLevelError> {
    let int_factor = ct
        .int_factor()
        .mul_mod(factor, ct.params().plain_modulus())?;
    if factor.value() == 1 {
        return BgvCiphertext::with_metadata(
            ct.params().clone(),
            ct.raw().clone(),
            int_factor,
            ct.noise_scale_degree(),
        );
    }

    let raw = scale_raw_ciphertext_by_factor(ct.raw(), factor, ct.params().ring());
    BgvCiphertext::with_metadata(
        ct.params().clone(),
        raw,
        int_factor,
        ct.noise_scale_degree() + 1,
    )
}

fn align_bgv_noise_scale_degree(
    left: BgvCiphertext,
    right: BgvCiphertext,
) -> Result<(BgvCiphertext, BgvCiphertext), BgvLevelError> {
    let target_degree = left.noise_scale_degree().max(right.noise_scale_degree());
    Ok((
        set_bgv_noise_scale_degree(left, target_degree)?,
        set_bgv_noise_scale_degree(right, target_degree)?,
    ))
}

fn set_bgv_noise_scale_degree(
    ct: BgvCiphertext,
    noise_scale_degree: usize,
) -> Result<BgvCiphertext, BgvLevelError> {
    if ct.noise_scale_degree() == noise_scale_degree {
        return Ok(ct);
    }
    BgvCiphertext::with_metadata(
        ct.params().clone(),
        ct.raw().clone(),
        ct.int_factor(),
        noise_scale_degree,
    )
}

fn scale_raw_ciphertext_by_factor(
    ct: &Ciphertext,
    factor: BgvPlaintextFactor,
    ring: &RingContext,
) -> Ciphertext {
    let mut out = ct.clone();
    for poly in out.data.iter_mut() {
        scale_poly_by_scalar(poly, factor.value(), ring);
    }
    out
}

fn scale_plaintext_by_factor(
    plaintext: &Plaintext,
    factor: BgvPlaintextFactor,
    plain_modulus: u64,
) -> Result<Plaintext, BgvLevelError> {
    factor.inverse_mod(plain_modulus)?;
    if factor.value() == 1 {
        return Ok(plaintext.clone());
    }

    let mut value = plaintext.value.clone();
    let scalar = factor.value() % plain_modulus;
    for coeff in value.limb_mut(0) {
        *coeff = ((u128::from(*coeff) * u128::from(scalar)) % u128::from(plain_modulus)) as u64;
    }
    Ok(Plaintext { value })
}

fn slot_matrix_diagonal_plaintexts(
    params: &BgvParameters,
    matrix: &[Vec<u64>],
) -> Result<Vec<(u32, Plaintext)>, BgvLevelError> {
    let degree = params.degree();
    let plain_modulus = params.plain_modulus();
    let encoder = BgvBatchEncoder::new(params.clone());
    bgv_slot_linear_transform_diagonal_elements(params, matrix)?
        .into_iter()
        .map(|element| {
            let map = params.ring().gen_automorphism_map(u64::from(element));
            let diagonal = (0..degree)
                .map(|row| matrix[row][map[row]] % plain_modulus)
                .collect::<Vec<_>>();
            Ok((element, encoder.encode(&diagonal)))
        })
        .collect()
}

fn slot_matrix_many_diagonal_plaintexts(
    params: &BgvParameters,
    matrices: &[Vec<Vec<u64>>],
) -> Result<Vec<Vec<(u32, Plaintext)>>, BgvLevelError> {
    matrices
        .iter()
        .map(|matrix| slot_matrix_diagonal_plaintexts(params, matrix))
        .collect()
}

fn zero_plaintext(params: &BgvParameters) -> Plaintext {
    BgvBatchEncoder::new(params.clone()).encode(&vec![0; params.degree()])
}

fn slot_matrix_axis_sum_plaintexts(
    params: &BgvParameters,
    rows: usize,
    columns: usize,
    axis: BgvSlotMatrixAxis,
) -> Result<Vec<(u32, Plaintext)>, BgvLevelError> {
    let active_slots = validate_slot_matrix_dimensions(params, rows, columns)?;
    let degree = params.degree();
    let encoder = BgvBatchEncoder::new(params.clone());

    bgv_slot_matrix_axis_sum_diagonal_elements(params, rows, columns, axis)?
        .into_iter()
        .map(|element| {
            let map = params.ring().gen_automorphism_map(u64::from(element));
            let diagonal = (0..degree)
                .map(|destination| {
                    let source = map[destination];
                    if destination < active_slots
                        && source < active_slots
                        && match axis {
                            BgvSlotMatrixAxis::Rows => source / columns == destination / columns,
                            BgvSlotMatrixAxis::Columns => source % columns == destination % columns,
                        }
                    {
                        1
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>();
            Ok((element, encoder.encode(&diagonal)))
        })
        .collect()
}

fn slot_mask_plaintext(params: &BgvParameters, slot: usize) -> Plaintext {
    slot_mask_plaintext_many(params, &[slot])
}

fn slot_mask_plaintext_many(params: &BgvParameters, slots: &[usize]) -> Plaintext {
    let mut mask = vec![0u64; params.degree()];
    for &slot in slots {
        if slot < mask.len() {
            mask[slot] = 1;
        }
    }
    BgvBatchEncoder::new(params.clone()).encode(&mask)
}

fn scale_poly_by_scalar(poly: &mut Poly, scalar: u64, ring: &RingContext) {
    for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        let scalar = scalar % q;
        for coeff in poly.limb_mut(limb_idx) {
            *coeff = ((u128::from(*coeff) * u128::from(scalar)) % u128::from(q)) as u64;
        }
    }
}

fn apply_galois_coeff_inplace(
    poly: &mut Poly,
    galois_elt: u32,
    ring: &RingContext,
    scratch: &mut [u64],
) {
    let degree = ring.degree();
    let map = ring.coeff_galois_map(galois_elt as u64);
    for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        let limb = poly.limb_mut(limb_idx);
        for i in 0..degree {
            let index = map[i] as usize;
            let value = limb[i];
            if index < degree {
                scratch[index] = value;
            } else {
                let dest = index - degree;
                scratch[dest] = if value == 0 { 0 } else { q - value };
            }
        }
        limb.copy_from_slice(&scratch[..degree]);
    }
}
