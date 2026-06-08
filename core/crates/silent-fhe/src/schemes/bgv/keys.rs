use crate::core::keys::{KeyGenerator, PublicKeyGen};
use crate::schemes::bgv::params::{BgvLevelError, BgvParameters};
use rand_core::{RngCore, SeedableRng};
use silent_math::{arith, numth};
use silent_ring::Poly;
use silent_rlwe::KeyGenerator as RlweKeyGen;
use silent_rlwe::{Ciphertext, Encryptor, EvaluationKey, GaloisKey, PublicKey, SecretKey};
use silent_utils::rng::SecureRng;
use std::collections::HashMap;

pub(crate) const BGV_KSW_BASE_LOG: usize = 8;

pub struct BgvKeyGenerator {
    params: BgvParameters,
    inner: RlweKeyGen,
    rng: SecureRng,
    error_scalar: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct BgvDiagonalBsgsAssignment {
    pub diagonal_index: usize,
    pub baby: u32,
    pub giant: u32,
    pub giant_inverse: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct BgvSlotRotationAssignment {
    pub galois_element: u32,
    pub destination_slots: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BgvSlotMatrixAxis {
    Rows,
    Columns,
}

impl BgvKeyGenerator {
    pub fn new(params: BgvParameters) -> Self {
        Self::with_rng(params, SecureRng::from_entropy())
    }

    pub fn with_rng(params: BgvParameters, rng: SecureRng) -> Self {
        Self::with_rng_and_error_scalar(params.clone(), rng, params.plain_modulus())
            .expect("BGV plaintext modulus is a valid default error scale")
    }

    pub fn with_rng_and_error_scalar(
        params: BgvParameters,
        mut rng: SecureRng,
        error_scalar: u64,
    ) -> Result<Self, BgvLevelError> {
        if error_scalar % params.plain_modulus() != 0 {
            return Err(
                BgvLevelError::BootstrapFusionErrorScaleNotPlaintextMultiple {
                    error_scalar,
                    plain_modulus: params.plain_modulus(),
                },
            );
        }

        let mut inner_seed = [0u8; 32];
        let mut bgv_seed = [0u8; 32];
        rng.fill_bytes(&mut inner_seed);
        rng.fill_bytes(&mut bgv_seed);
        Ok(Self {
            inner: RlweKeyGen::new(
                params.runtime_params().clone(),
                SecureRng::from_seed(inner_seed),
            ),
            rng: SecureRng::from_seed(bgv_seed),
            error_scalar,
            params,
        })
    }

    fn child_rng(&mut self) -> SecureRng {
        let mut seed = [0u8; 32];
        self.rng.fill_bytes(&mut seed);
        SecureRng::from_seed(seed)
    }

    pub fn relinearization_key(
        &mut self,
        sk: &SecretKey,
    ) -> Result<silent_rlwe::EvaluationKey, silent_math::rns::RnsError> {
        let ring = self.params.ring();
        let levels = bgv_ksw_levels(&self.params);
        let mut s2 = sk.value.clone();
        s2.mul_assign(&sk.value, ring);

        let mut elements = Vec::with_capacity(levels);
        for level in 0..levels {
            elements.push(self.encrypt_key_component(sk, &s2, level));
        }
        Ok(EvaluationKey { elements })
    }

    /// Generate a public-key switching hint from `sk_from` to `pk_to`.
    ///
    /// The returned key has the same BV gadget layout as BGV relinearization
    /// and Galois keys, but each gadget component encrypts a power-of-two
    /// multiple of `sk_from` under the target public key.  Evaluators consume it
    /// through `proxy_reencrypt_bgv`, so this stays on the shared BGV
    /// key-switching path.
    pub fn proxy_reencryption_key(
        &mut self,
        sk_from: &SecretKey,
        pk_to: &PublicKey,
    ) -> Result<silent_rlwe::EvaluationKey, BgvLevelError> {
        self.validate_secret_key_shape(sk_from)?;
        self.public_key_switching_key_for_ntt_poly(&sk_from.value, pk_to)
    }

    /// Generate a public-key switching hint for an NTT-domain key polynomial.
    ///
    /// This is the low-level primitive behind PRE, collective Galois keys, and
    /// threshold relinearization protocols: callers derive the source-key
    /// polynomial contribution themselves, while this method owns the shared BGV
    /// BV-gadget encryption layout.
    pub fn public_key_switching_key_for_ntt_poly(
        &mut self,
        key_poly_ntt: &Poly,
        pk_to: &PublicKey,
    ) -> Result<silent_rlwe::EvaluationKey, BgvLevelError> {
        self.validate_key_poly_shape(key_poly_ntt)?;
        self.validate_public_key_shape(pk_to)?;

        let levels = bgv_ksw_levels(&self.params);
        let mut elements = Vec::with_capacity(levels);
        for level in 0..levels {
            elements.push(self.encrypt_public_key_component(pk_to, key_poly_ntt, level));
        }
        Ok(EvaluationKey { elements })
    }

    pub fn galois_keys(
        &mut self,
        sk: &SecretKey,
        substitution_indices: &[u32],
    ) -> Result<silent_rlwe::GaloisKey, silent_math::rns::RnsError> {
        let levels = bgv_ksw_levels(&self.params);
        let mut keys = HashMap::new();

        for &k in substitution_indices {
            let map = self.params.ring().automorphism_map(k as u64);
            let mut sk_rot = sk.value.clone();
            sk_rot.apply_automorphism(&map);

            let mut elements = Vec::with_capacity(levels);
            for level in 0..levels {
                elements.push(self.encrypt_key_component(sk, &sk_rot, level));
            }
            keys.insert(k, EvaluationKey { elements });
        }

        Ok(GaloisKey { keys })
    }

    pub fn full_slot_sum_galois_keys(
        &mut self,
        sk: &SecretKey,
    ) -> Result<silent_rlwe::GaloisKey, silent_math::rns::RnsError> {
        self.galois_keys(sk, &bgv_full_slot_sum_galois_elements(self.params.degree()))
    }

    pub fn slot_merge_galois_keys(
        &mut self,
        sk: &SecretKey,
        slot_count: usize,
    ) -> Result<silent_rlwe::GaloisKey, silent_math::rns::RnsError> {
        self.galois_keys(
            sk,
            &bgv_slot_merge_galois_elements(&self.params, slot_count),
        )
    }

    pub fn slot_rotation_galois_elements(
        &self,
        rotations: &[i32],
    ) -> Result<Vec<u32>, BgvLevelError> {
        bgv_slot_rotation_galois_elements(&self.params, rotations)
    }

    pub fn slot_rotation_galois_keys(
        &mut self,
        sk: &SecretKey,
        rotations: &[i32],
    ) -> Result<silent_rlwe::GaloisKey, BgvLevelError> {
        let elements = self.slot_rotation_galois_elements(rotations)?;
        self.galois_keys(sk, &elements).map_err(BgvLevelError::from)
    }

    pub fn diagonal_transform_galois_keys(
        &mut self,
        sk: &SecretKey,
        galois_elements: &[u32],
    ) -> Result<silent_rlwe::GaloisKey, silent_math::rns::RnsError> {
        let mut unique = Vec::new();
        for &element in galois_elements {
            if (element == 0 || element == 1) || unique.contains(&element) {
                continue;
            }
            unique.push(element);
        }
        self.galois_keys(sk, &unique)
    }

    pub fn diagonal_bsgs_galois_elements(
        &self,
        diagonal_galois_elements: &[u32],
        baby_galois_elements: &[u32],
    ) -> Result<Vec<u32>, BgvLevelError> {
        bgv_diagonal_bsgs_galois_elements(
            &self.params,
            diagonal_galois_elements,
            baby_galois_elements,
        )
    }

    pub fn diagonal_bsgs_galois_keys(
        &mut self,
        sk: &SecretKey,
        diagonal_galois_elements: &[u32],
        baby_galois_elements: &[u32],
    ) -> Result<silent_rlwe::GaloisKey, BgvLevelError> {
        let elements =
            self.diagonal_bsgs_galois_elements(diagonal_galois_elements, baby_galois_elements)?;
        self.galois_keys(sk, &elements).map_err(BgvLevelError::from)
    }

    pub fn slot_linear_transform_galois_elements(
        &self,
        matrix: &[Vec<u64>],
    ) -> Result<Vec<u32>, BgvLevelError> {
        bgv_slot_linear_transform_galois_elements(&self.params, matrix)
    }

    pub fn slot_linear_transform_galois_keys(
        &mut self,
        sk: &SecretKey,
        matrix: &[Vec<u64>],
    ) -> Result<silent_rlwe::GaloisKey, BgvLevelError> {
        let elements = self.slot_linear_transform_galois_elements(matrix)?;
        self.galois_keys(sk, &elements).map_err(BgvLevelError::from)
    }

    pub fn slot_linear_transform_many_galois_elements(
        &self,
        matrices: &[Vec<Vec<u64>>],
    ) -> Result<Vec<u32>, BgvLevelError> {
        let mut elements = Vec::new();
        for matrix in matrices {
            for element in bgv_slot_linear_transform_galois_elements(&self.params, matrix)? {
                push_unique_non_identity(&mut elements, element);
            }
        }
        Ok(elements)
    }

    pub fn slot_linear_transform_many_galois_keys(
        &mut self,
        sk: &SecretKey,
        matrices: &[Vec<Vec<u64>>],
    ) -> Result<silent_rlwe::GaloisKey, BgvLevelError> {
        let elements = self.slot_linear_transform_many_galois_elements(matrices)?;
        self.galois_keys(sk, &elements).map_err(BgvLevelError::from)
    }

    pub fn slot_linear_transform_bsgs_galois_elements(
        &self,
        matrix: &[Vec<u64>],
        baby_galois_elements: &[u32],
    ) -> Result<Vec<u32>, BgvLevelError> {
        let diagonal_elements = bgv_slot_linear_transform_diagonal_elements(&self.params, matrix)?;
        bgv_diagonal_bsgs_galois_elements(&self.params, &diagonal_elements, baby_galois_elements)
    }

    pub fn slot_linear_transform_bsgs_galois_keys(
        &mut self,
        sk: &SecretKey,
        matrix: &[Vec<u64>],
        baby_galois_elements: &[u32],
    ) -> Result<silent_rlwe::GaloisKey, BgvLevelError> {
        let elements =
            self.slot_linear_transform_bsgs_galois_elements(matrix, baby_galois_elements)?;
        self.galois_keys(sk, &elements).map_err(BgvLevelError::from)
    }

    pub fn slot_matrix_row_sum_galois_elements(
        &self,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<u32>, BgvLevelError> {
        bgv_slot_matrix_axis_sum_galois_elements(
            &self.params,
            rows,
            columns,
            BgvSlotMatrixAxis::Rows,
        )
    }

    pub fn slot_matrix_column_sum_galois_elements(
        &self,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<u32>, BgvLevelError> {
        bgv_slot_matrix_axis_sum_galois_elements(
            &self.params,
            rows,
            columns,
            BgvSlotMatrixAxis::Columns,
        )
    }

    pub fn slot_matrix_row_sum_galois_keys(
        &mut self,
        sk: &SecretKey,
        rows: usize,
        columns: usize,
    ) -> Result<silent_rlwe::GaloisKey, BgvLevelError> {
        let elements = self.slot_matrix_row_sum_galois_elements(rows, columns)?;
        self.galois_keys(sk, &elements).map_err(BgvLevelError::from)
    }

    pub fn slot_matrix_column_sum_galois_keys(
        &mut self,
        sk: &SecretKey,
        rows: usize,
        columns: usize,
    ) -> Result<silent_rlwe::GaloisKey, BgvLevelError> {
        let elements = self.slot_matrix_column_sum_galois_elements(rows, columns)?;
        self.galois_keys(sk, &elements).map_err(BgvLevelError::from)
    }

    pub fn slot_matrix_row_sum_bsgs_galois_elements(
        &self,
        rows: usize,
        columns: usize,
        baby_galois_elements: &[u32],
    ) -> Result<Vec<u32>, BgvLevelError> {
        let diagonal_elements = bgv_slot_matrix_axis_sum_diagonal_elements(
            &self.params,
            rows,
            columns,
            BgvSlotMatrixAxis::Rows,
        )?;
        bgv_diagonal_bsgs_galois_elements(&self.params, &diagonal_elements, baby_galois_elements)
    }

    pub fn slot_matrix_column_sum_bsgs_galois_elements(
        &self,
        rows: usize,
        columns: usize,
        baby_galois_elements: &[u32],
    ) -> Result<Vec<u32>, BgvLevelError> {
        let diagonal_elements = bgv_slot_matrix_axis_sum_diagonal_elements(
            &self.params,
            rows,
            columns,
            BgvSlotMatrixAxis::Columns,
        )?;
        bgv_diagonal_bsgs_galois_elements(&self.params, &diagonal_elements, baby_galois_elements)
    }

    pub fn slot_matrix_row_sum_bsgs_galois_keys(
        &mut self,
        sk: &SecretKey,
        rows: usize,
        columns: usize,
        baby_galois_elements: &[u32],
    ) -> Result<silent_rlwe::GaloisKey, BgvLevelError> {
        let elements =
            self.slot_matrix_row_sum_bsgs_galois_elements(rows, columns, baby_galois_elements)?;
        self.galois_keys(sk, &elements).map_err(BgvLevelError::from)
    }

    pub fn slot_matrix_column_sum_bsgs_galois_keys(
        &mut self,
        sk: &SecretKey,
        rows: usize,
        columns: usize,
        baby_galois_elements: &[u32],
    ) -> Result<silent_rlwe::GaloisKey, BgvLevelError> {
        let elements =
            self.slot_matrix_column_sum_bsgs_galois_elements(rows, columns, baby_galois_elements)?;
        self.galois_keys(sk, &elements).map_err(BgvLevelError::from)
    }

    fn encrypt_key_component(
        &mut self,
        sk: &SecretKey,
        key_poly_ntt: &Poly,
        level: usize,
    ) -> Ciphertext {
        let mut encryptor = Encryptor::new(self.params.runtime_params().clone(), self.child_rng());
        let mut ct = encryptor.encrypt_zero_symmetric_scaled_error(sk, self.error_scalar);
        let ring = self.params.ring();
        let exp = (level * BGV_KSW_BASE_LOG) as u64;

        for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
            let qi = modulus.value();
            let scalar = numth::mod_pow(2, exp, qi);
            let c0_limb = ct.data[0].limb_mut(limb_idx);
            let key_limb = key_poly_ntt.limb(limb_idx);
            for i in 0..ring.degree() {
                let term = arith::mul_mod_u64(key_limb[i], scalar, qi);
                c0_limb[i] = arith::add_mod(c0_limb[i], term, qi);
            }
        }

        ct
    }

    fn encrypt_public_key_component(
        &mut self,
        pk: &PublicKey,
        key_poly_ntt: &Poly,
        level: usize,
    ) -> Ciphertext {
        let mut encryptor = Encryptor::new(self.params.runtime_params().clone(), self.child_rng());
        let mut ct = encryptor.encrypt_zero_public_scaled_error(pk, self.error_scalar);
        let ring = self.params.ring();
        let exp = (level * BGV_KSW_BASE_LOG) as u64;

        for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
            let qi = modulus.value();
            let scalar = numth::mod_pow(2, exp, qi);
            let c0_limb = ct.data[0].limb_mut(limb_idx);
            let key_limb = key_poly_ntt.limb(limb_idx);
            for i in 0..ring.degree() {
                let term = arith::mul_mod_u64(key_limb[i], scalar, qi);
                c0_limb[i] = arith::add_mod(c0_limb[i], term, qi);
            }
        }

        ct
    }

    fn validate_secret_key_shape(&self, sk: &SecretKey) -> Result<(), BgvLevelError> {
        self.validate_key_poly_shape(&sk.value)
    }

    fn validate_key_poly_shape(&self, key_poly_ntt: &Poly) -> Result<(), BgvLevelError> {
        let expected_degree = self.params.degree();
        let expected_limbs = self.params.q_modulus_count();
        if key_poly_ntt.degree() != expected_degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: expected_degree,
                actual: key_poly_ntt.degree(),
            });
        }
        if key_poly_ntt.num_moduli() < expected_limbs {
            return Err(BgvLevelError::CiphertextLimbMismatch {
                expected: expected_limbs,
                actual: key_poly_ntt.num_moduli(),
            });
        }
        Ok(())
    }

    fn validate_public_key_shape(&self, pk: &PublicKey) -> Result<(), BgvLevelError> {
        if pk.pk.data.len() != 2 {
            return Err(BgvLevelError::PublicKeySizeMismatch {
                actual: pk.pk.data.len(),
            });
        }

        let expected_degree = self.params.degree();
        let expected_limbs = self.params.q_modulus_count();
        for poly in &pk.pk.data {
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
}

pub(crate) fn bgv_ksw_levels(params: &BgvParameters) -> usize {
    let q_bits = params
        .ring()
        .rns()
        .base_prod_u128()
        .map(|q| (u128::BITS - q.leading_zeros()) as usize)
        .unwrap_or_else(|| {
            params
                .ring()
                .rns()
                .moduli()
                .iter()
                .map(|modulus| modulus.bit_count() as usize)
                .sum()
        });
    q_bits.div_ceil(BGV_KSW_BASE_LOG) + 1
}

pub(crate) fn bgv_diagonal_bsgs_galois_elements(
    params: &BgvParameters,
    diagonal_galois_elements: &[u32],
    baby_galois_elements: &[u32],
) -> Result<Vec<u32>, BgvLevelError> {
    let assignments =
        bgv_diagonal_bsgs_assignments(params, diagonal_galois_elements, baby_galois_elements)?;
    let mut elements = Vec::new();
    for assignment in assignments {
        push_unique_non_identity(&mut elements, assignment.baby);
        push_unique_non_identity(&mut elements, assignment.giant);
    }
    Ok(elements)
}

pub(crate) fn bgv_slot_linear_transform_diagonal_elements(
    params: &BgvParameters,
    matrix: &[Vec<u64>],
) -> Result<Vec<u32>, BgvLevelError> {
    validate_slot_linear_transform_matrix(params, matrix)?;
    let degree = params.degree();
    let plain_modulus = params.plain_modulus();
    let mut elements = Vec::new();

    for element in bgv_all_slot_linear_transform_elements(params) {
        let map = params.ring().gen_automorphism_map(u64::from(element));
        let nonzero = (0..degree).any(|row| matrix[row][map[row]] % plain_modulus != 0);
        if nonzero {
            elements.push(element);
        }
    }

    if elements.is_empty() {
        elements.push(1);
    }
    Ok(elements)
}

pub(crate) fn bgv_slot_linear_transform_galois_elements(
    params: &BgvParameters,
    matrix: &[Vec<u64>],
) -> Result<Vec<u32>, BgvLevelError> {
    let mut elements = Vec::new();
    for element in bgv_slot_linear_transform_diagonal_elements(params, matrix)? {
        push_unique_non_identity(&mut elements, element);
    }
    Ok(elements)
}

pub(crate) fn bgv_slot_matrix_axis_sum_diagonal_elements(
    params: &BgvParameters,
    rows: usize,
    columns: usize,
    axis: BgvSlotMatrixAxis,
) -> Result<Vec<u32>, BgvLevelError> {
    let active_slots = validate_slot_matrix_dimensions(params, rows, columns)?;
    let mut elements = Vec::new();

    for element in bgv_all_slot_linear_transform_elements(params) {
        let map = params.ring().gen_automorphism_map(u64::from(element));
        let nonzero = (0..active_slots).any(|destination| {
            let source = map[destination];
            source < active_slots
                && match axis {
                    BgvSlotMatrixAxis::Rows => source / columns == destination / columns,
                    BgvSlotMatrixAxis::Columns => source % columns == destination % columns,
                }
        });
        if nonzero {
            elements.push(element);
        }
    }

    if elements.is_empty() {
        elements.push(1);
    }
    Ok(elements)
}

pub(crate) fn bgv_slot_matrix_axis_sum_galois_elements(
    params: &BgvParameters,
    rows: usize,
    columns: usize,
    axis: BgvSlotMatrixAxis,
) -> Result<Vec<u32>, BgvLevelError> {
    let mut elements = Vec::new();
    for element in bgv_slot_matrix_axis_sum_diagonal_elements(params, rows, columns, axis)? {
        push_unique_non_identity(&mut elements, element);
    }
    Ok(elements)
}

pub(crate) fn bgv_diagonal_bsgs_assignments(
    params: &BgvParameters,
    diagonal_galois_elements: &[u32],
    baby_galois_elements: &[u32],
) -> Result<Vec<BgvDiagonalBsgsAssignment>, BgvLevelError> {
    let babies = normalized_bsgs_babies(params, diagonal_galois_elements, baby_galois_elements)?;
    let baby_inverses = babies
        .iter()
        .map(|&baby| Ok((baby, bgv_inverse_galois_element(params, baby)?)))
        .collect::<Result<Vec<_>, BgvLevelError>>()?;

    let mut group_scores = HashMap::new();
    for &element in diagonal_galois_elements {
        let element = bgv_canonical_galois_element_for_params(params, element)?;
        for (_, baby_inv) in baby_inverses.iter() {
            let giant = bgv_multiply_galois_elements(params, element, *baby_inv);
            *group_scores.entry(giant).or_insert(0usize) += 1;
        }
    }

    let mut assignments = Vec::with_capacity(diagonal_galois_elements.len());
    for (diagonal_index, &element) in diagonal_galois_elements.iter().enumerate() {
        let element = bgv_canonical_galois_element_for_params(params, element)?;
        let mut best_baby = 1u32;
        let mut best_giant = element;
        let mut best_score = 0usize;

        for (baby, baby_inv) in baby_inverses.iter() {
            let giant = bgv_multiply_galois_elements(params, element, *baby_inv);
            let score = *group_scores.get(&giant).unwrap_or(&0);
            let improves_score = score > best_score;
            let improves_tie =
                score == best_score && (giant == 1 || (best_giant != 1 && *baby < best_baby));
            if improves_score || improves_tie {
                best_baby = *baby;
                best_giant = giant;
                best_score = score;
            }
        }

        assignments.push(BgvDiagonalBsgsAssignment {
            diagonal_index,
            baby: best_baby,
            giant: best_giant,
            giant_inverse: bgv_inverse_galois_element(params, best_giant)?,
        });
    }

    Ok(assignments)
}

pub(crate) fn bgv_canonical_galois_element_for_params(
    params: &BgvParameters,
    element: u32,
) -> Result<u32, BgvLevelError> {
    if element == 0 {
        return Ok(1);
    }
    let modulus = bgv_galois_element_modulus(params);
    let reduced = u64::from(element) % modulus;
    if reduced == 0 || numth::mod_inverse(reduced, modulus).is_none() {
        return Err(BgvLevelError::InvalidGaloisElement { element, modulus });
    }
    Ok(reduced as u32)
}

fn normalized_bsgs_babies(
    params: &BgvParameters,
    diagonal_galois_elements: &[u32],
    baby_galois_elements: &[u32],
) -> Result<Vec<u32>, BgvLevelError> {
    let mut babies = vec![1u32];
    if baby_galois_elements.is_empty() {
        let target = bsgs_default_baby_count(diagonal_galois_elements.len());
        for &element in diagonal_galois_elements {
            let element = bgv_canonical_galois_element_for_params(params, element)?;
            if !babies.contains(&element) {
                babies.push(element);
            }
            if babies.len() >= target {
                break;
            }
        }
    } else {
        for &element in baby_galois_elements {
            let element = bgv_canonical_galois_element_for_params(params, element)?;
            if !babies.contains(&element) {
                babies.push(element);
            }
        }
    }
    Ok(babies)
}

fn bsgs_default_baby_count(term_count: usize) -> usize {
    let term_count = term_count.max(1);
    let mut count = 1usize;
    while count.saturating_mul(count) < term_count {
        count += 1;
    }
    count
}

fn validate_slot_linear_transform_matrix(
    params: &BgvParameters,
    matrix: &[Vec<u64>],
) -> Result<(), BgvLevelError> {
    if matrix.is_empty() {
        return Err(BgvLevelError::EmptySlotLinearTransformMatrix);
    }

    let degree = params.degree();
    if matrix.len() != degree {
        return Err(BgvLevelError::SlotLinearTransformRowCountMismatch {
            expected: degree,
            actual: matrix.len(),
        });
    }
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != degree {
            return Err(BgvLevelError::SlotLinearTransformColumnCountMismatch {
                row: row_idx,
                expected: degree,
                actual: row.len(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_slot_matrix_dimensions(
    params: &BgvParameters,
    rows: usize,
    columns: usize,
) -> Result<usize, BgvLevelError> {
    let slots = params.degree();
    let active_slots =
        rows.checked_mul(columns)
            .ok_or(BgvLevelError::InvalidSlotMatrixDimensions {
                rows,
                columns,
                slots,
            })?;
    if rows == 0 || columns == 0 || active_slots > slots {
        return Err(BgvLevelError::InvalidSlotMatrixDimensions {
            rows,
            columns,
            slots,
        });
    }
    Ok(active_slots)
}

fn bgv_all_slot_linear_transform_elements(params: &BgvParameters) -> Vec<u32> {
    let modulus = params
        .degree()
        .checked_mul(2)
        .and_then(|value| u32::try_from(value).ok())
        .expect("BGV ring degree too large for u32 Galois indices");
    (1..modulus).step_by(2).collect()
}

fn bgv_inverse_galois_element(params: &BgvParameters, element: u32) -> Result<u32, BgvLevelError> {
    let element = bgv_canonical_galois_element_for_params(params, element)?;
    let modulus = bgv_galois_element_modulus(params);
    numth::mod_inverse(u64::from(element), modulus)
        .map(|inverse| inverse as u32)
        .ok_or(BgvLevelError::InvalidGaloisElement { element, modulus })
}

fn bgv_multiply_galois_elements(params: &BgvParameters, left: u32, right: u32) -> u32 {
    let modulus = bgv_galois_element_modulus(params);
    ((u128::from(left) * u128::from(right)) % u128::from(modulus)) as u32
}

fn bgv_galois_element_modulus(params: &BgvParameters) -> u64 {
    (params.degree() as u64) * 2
}

fn push_unique_non_identity(elements: &mut Vec<u32>, element: u32) {
    if element != 1 && !elements.contains(&element) {
        elements.push(element);
    }
}

pub(crate) fn bgv_full_slot_sum_galois_elements(degree: usize) -> Vec<u32> {
    let cyclotomic_order = degree
        .checked_mul(2)
        .expect("BGV ring degree too large for cyclotomic order");
    assert!(
        cyclotomic_order <= u32::MAX as usize,
        "BGV ring degree too large for u32 Galois indices"
    );
    (3..cyclotomic_order)
        .step_by(2)
        .map(|index| index as u32)
        .collect()
}

pub(crate) fn bgv_slot_merge_galois_elements(
    params: &BgvParameters,
    slot_count: usize,
) -> Vec<u32> {
    (1..slot_count.min(params.degree()))
        .filter_map(|slot| bgv_slot_zero_to_slot_galois_element(params, slot))
        .collect()
}

pub(crate) fn bgv_slot_rotation_galois_elements(
    params: &BgvParameters,
    rotations: &[i32],
) -> Result<Vec<u32>, BgvLevelError> {
    let mut elements = Vec::new();
    for &rotation in rotations {
        for assignment in bgv_slot_rotation_assignments(params, rotation)? {
            push_unique_non_identity(&mut elements, assignment.galois_element);
        }
    }
    Ok(elements)
}

pub(crate) fn bgv_slot_rotation_assignments(
    params: &BgvParameters,
    rotation: i32,
) -> Result<Vec<BgvSlotRotationAssignment>, BgvLevelError> {
    let slots = params.degree();
    let shift = normalize_slot_rotation(rotation, slots)?;
    if shift == 0 {
        return Ok(Vec::new());
    }

    let cyclotomic_order = slots
        .checked_mul(2)
        .ok_or(BgvLevelError::SlotRotationUnsupported { rotation, slots })?;
    if cyclotomic_order > u32::MAX as usize {
        return Err(BgvLevelError::SlotRotationUnsupported { rotation, slots });
    }

    let mut groups = Vec::<BgvSlotRotationAssignment>::new();
    for destination in 0..slots {
        let source = (destination + slots - shift) % slots;
        let element = (1..cyclotomic_order)
            .step_by(2)
            .find(|element| automorphism_source_slot(slots, *element, destination) == source)
            .map(|element| element as u32)
            .ok_or(BgvLevelError::SlotRotationUnsupported { rotation, slots })?;

        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.galois_element == element)
        {
            group.destination_slots.push(destination);
        } else {
            groups.push(BgvSlotRotationAssignment {
                galois_element: element,
                destination_slots: vec![destination],
            });
        }
    }

    Ok(groups)
}

pub(crate) fn bgv_slot_zero_to_slot_galois_element(
    params: &BgvParameters,
    target_slot: usize,
) -> Option<u32> {
    if target_slot == 0 {
        return Some(1);
    }
    if target_slot >= params.degree() {
        return None;
    }

    let cyclotomic_order = params.degree().checked_mul(2)?;
    if cyclotomic_order > u32::MAX as usize {
        return None;
    }

    (3..cyclotomic_order)
        .step_by(2)
        .find(|index| params.ring().gen_automorphism_map(*index as u64)[target_slot] == 0)
        .map(|index| index as u32)
}

fn normalize_slot_rotation(rotation: i32, slots: usize) -> Result<usize, BgvLevelError> {
    if slots == 0 {
        return Err(BgvLevelError::SlotRotationUnsupported { rotation, slots });
    }
    Ok((i64::from(rotation)).rem_euclid(slots as i64) as usize)
}

fn automorphism_source_slot(degree: usize, element: usize, destination: usize) -> usize {
    let log_n = degree.trailing_zeros();
    let reversed = reverse_low_bits(degree + destination, log_n + 1);
    let index_raw = ((element * reversed) >> 1) & (degree - 1);
    reverse_low_bits(index_raw, log_n)
}

fn reverse_low_bits(value: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        value.reverse_bits() >> (usize::BITS - bits)
    }
}

impl KeyGenerator for BgvKeyGenerator {
    type SecretKey = SecretKey;

    fn generate_secret_key(&mut self) -> SecretKey {
        self.inner.secret_key()
    }
}

impl PublicKeyGen for BgvKeyGenerator {
    type PublicKey = PublicKey;

    fn generate_public_key(&mut self, sk: &SecretKey) -> PublicKey {
        let mut encryptor = Encryptor::new(self.params.runtime_params().clone(), self.child_rng());
        PublicKey {
            pk: encryptor.encrypt_zero_symmetric_scaled_error(sk, self.error_scalar),
        }
    }
}
