use rand::{RngCore, SeedableRng};

use silent_hss::{
    HssCiphertext, HssContext, HssEncryptor, HssEvaluator, HssKeyGenerator, HssShare,
};
use silent_math::modulus::Modulus;
use silent_math::rns::RnsBase;
use silent_math::rns_tool::RnsToolConfig;
use silent_rlwe::{EncryptionParams, SecretKey};
use silent_utils::rng::SecureRng;

use crate::hss_bridge::context_from_runtime;
use crate::modular::error::ModularError;
use crate::modular::matrix::ModVector;
use crate::modular::ring_matrix_mul::{
    PolyElem, PolyMatrix, RingMatrixMulCrs, RingMatrixMulPublicA, RingMatrixMulPublicB,
    RingMatrixMulStateA, RingMatrixMulStateB, decode_a_ring, decode_b_ring, encode_a_ring,
    encode_b_raw_ring, ring_matrix_mul_setup,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Party {
    A,
    B,
}

#[derive(Clone, Debug)]
pub struct PointFunctionParams {
    pub ell: usize,
    pub m: usize,
    pub matrix_rows: usize,
    pub gadget_cols: usize,
    pub q: u64,
    pub p: u64,
    pub encode_a_noise: i64,
    pub encode_b_noise: i64,
    /// If true, use Section 5.1 Approach II carry emulation over 2m columns.
    pub carry_emulation: bool,
}

impl PointFunctionParams {
    pub fn domain_size(&self) -> usize {
        self.ell * self.m
    }

    pub fn validate(&self) -> Result<(), ModularError> {
        if self.ell == 0 || self.m == 0 {
            return Err(ModularError::InvalidParams("ell and m must be positive"));
        }
        if self.matrix_rows == 0 || self.gadget_cols == 0 {
            return Err(ModularError::InvalidParams(
                "matrix_rows and gadget_cols must be positive",
            ));
        }
        if self.q < 2 || self.p < 2 {
            return Err(ModularError::InvalidParams("q and p must be >= 2"));
        }
        if self.q <= self.p {
            return Err(ModularError::InvalidParams(
                "q must be strictly larger than p for strict private point-function/HSS alignment",
            ));
        }
        if self.q % 2 == 0 {
            return Err(ModularError::InvalidParams(
                "q must be odd to support NTT-friendly ring moduli",
            ));
        }
        if self.encode_a_noise < 0 || self.encode_b_noise < 0 {
            return Err(ModularError::InvalidParams("noise bounds must be >= 0"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PointFunctionCrs {
    pub matrix_crs: RingMatrixMulCrs,
}

#[derive(Clone, Debug)]
pub struct PartyAPublicKey {
    pub matrix_public: RingMatrixMulPublicA,
}

#[derive(Clone, Debug)]
pub struct PartyASecretKey {
    pub matrix_state: RingMatrixMulStateA,
}

#[derive(Clone, Debug)]
pub struct PartyBPublicKey {
    pub matrix_public: RingMatrixMulPublicB,
    pub selection_cts: Vec<HssCiphertext>,
}

#[derive(Clone, Debug)]
pub struct PartyBSecretKey {
    pub matrix_state: RingMatrixMulStateB,
    pub selection_cts: Vec<HssCiphertext>,
}

/// Lemma 2 composition helper: Party A material for full-payload private point-function.
/// `half_ab`: A is the payload-chosen side (role A).
/// `half_ba`: A is the zero-payload side in the swapped half instance (role B).
#[derive(Clone, Debug)]
pub struct FullPartyAPublicKey {
    pub half_ab: PartyAPublicKey,
    pub half_ba: PartyBPublicKey,
}

#[derive(Clone, Debug)]
pub struct FullPartyASecretKey {
    pub half_ab: PartyASecretKey,
    pub half_ba: PartyBSecretKey,
}

/// Lemma 2 composition helper: Party B material for full-payload private point-function.
/// `half_ab`: B is the zero-payload side (role B).
/// `half_ba`: B is the payload-chosen side in the swapped half instance (role A).
#[derive(Clone, Debug)]
pub struct FullPartyBPublicKey {
    pub half_ab: PartyBPublicKey,
    pub half_ba: PartyAPublicKey,
}

#[derive(Clone, Debug)]
pub struct FullPartyBSecretKey {
    pub half_ab: PartyBSecretKey,
    pub half_ba: PartyASecretKey,
}

#[derive(Clone, Debug)]
pub struct EvaluationKey {
    pub party: Party,
    pub t_share: PolyMatrix,
    pub selection_cts: Vec<HssCiphertext>,
    pub selector_is_trivial_one: bool,
}

#[derive(Clone, Debug)]
pub struct FullEvaluationKey {
    pub party: Party,
    pub half_ab: EvaluationKey,
    pub half_ba: EvaluationKey,
}

#[derive(Clone, Debug)]
pub struct EvalAllOutput {
    pub party: Party,
    pub rows: usize,
    pub cols: usize,
    pub shares: Vec<HssShare>,
}

pub struct PointFunctionEvaluator {
    params: PointFunctionParams,
    hss_context: HssContext,
    hss_params: EncryptionParams,
}

impl PointFunctionEvaluator {
    pub fn new(params: PointFunctionParams) -> Result<Self, ModularError> {
        params.validate()?;
        let target_slots = params.ell
            * if params.carry_emulation {
                2 * params.m
            } else {
                params.m
            };
        let hss_params = Self::build_hss_params(params.p, params.q, target_slots)?;
        let hss_context =
            context_from_runtime("operator-point-function-hss", hss_params.clone(), params.p)
                .map_err(|err| ModularError::Backend(err.to_string()))?;
        Ok(Self {
            params,
            hss_context,
            hss_params,
        })
    }

    fn build_hss_params(
        plain_modulus: u64,
        ciphertext_modulus: u64,
        target_slots: usize,
    ) -> Result<EncryptionParams, ModularError> {
        let mut degree = Self::choose_hss_degree(plain_modulus, target_slots)?;
        loop {
            let base_q = RnsBase::from_values(vec![ciphertext_modulus])
                .map_err(|e| ModularError::Backend(format!("{e:?}")))?;
            let base_t =
                Modulus::new(plain_modulus).map_err(|e| ModularError::Backend(format!("{e:?}")))?;
            let config = RnsToolConfig::new(base_q, base_t);
            match EncryptionParams::from_rns_config(degree, config) {
                Ok(params) => return Ok(params),
                Err(err) if degree > 1 => {
                    degree >>= 1;
                    if degree == 0 {
                        return Err(ModularError::Backend(format!("{err:?}")));
                    }
                }
                Err(err) => return Err(ModularError::Backend(format!("{err:?}"))),
            }
        }
    }

    fn choose_hss_degree(plain_modulus: u64, target_slots: usize) -> Result<usize, ModularError> {
        if plain_modulus <= 2 {
            return Err(ModularError::InvalidParams("plain modulus must be > 2"));
        }
        let mut max_degree = 1usize;
        loop {
            let Some(next) = max_degree.checked_mul(2) else {
                break;
            };
            let two_n = (next as u128) << 1;
            if (plain_modulus as u128 - 1) % two_n != 0 {
                break;
            }
            max_degree = next;
        }
        let mut target_degree = 1usize;
        let desired = target_slots.max(1);
        while target_degree < desired {
            target_degree <<= 1;
        }
        Ok(target_degree.min(max_degree).max(1))
    }

    pub fn hss_degree(&self) -> usize {
        self.hss_params.ring.degree()
    }

    fn derive_secure_rng<R: RngCore + ?Sized>(rng: &mut R) -> SecureRng {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        SecureRng::from_seed(seed)
    }

    fn encoded_cols(&self) -> usize {
        if self.params.carry_emulation {
            2 * self.params.m
        } else {
            self.params.m
        }
    }

    fn memory_ring_cols(&self) -> usize {
        2 * self.encoded_cols()
    }

    fn validate_index(&self, idx: (usize, usize)) -> Result<(), ModularError> {
        if idx.0 >= self.params.ell {
            return Err(ModularError::IndexOutOfBounds {
                index: idx.0,
                upper_bound: self.params.ell,
                context: "index row",
            });
        }
        if idx.1 >= self.params.m {
            return Err(ModularError::IndexOutOfBounds {
                index: idx.1,
                upper_bound: self.params.m,
                context: "index column",
            });
        }
        Ok(())
    }

    pub fn setup<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<PointFunctionCrs, ModularError> {
        let matrix_crs = ring_matrix_mul_setup(
            self.params.matrix_rows,
            self.encoded_cols(),
            self.params.gadget_cols,
            self.hss_degree(),
            self.params.q,
            rng,
        )?;
        Ok(PointFunctionCrs { matrix_crs })
    }

    /// private point-function genA (half-chosen payload side).
    pub fn gen_a<R: RngCore + ?Sized>(
        &self,
        crs: &PointFunctionCrs,
        t_a: (usize, usize),
        payload: u64,
        rng: &mut R,
    ) -> Result<(PartyAPublicKey, PartyASecretKey), ModularError> {
        self.validate_index(t_a)?;

        let encoded_cols = self.encoded_cols();
        let degree = self.hss_degree();
        let mut a = PolyMatrix::zeros(self.params.ell, encoded_cols, degree, self.params.q)?;
        a.set_constant(t_a.0, t_a.1, payload % self.params.p)?;

        let (matrix_public, matrix_state) =
            encode_a_ring(&crs.matrix_crs, &a, self.params.encode_a_noise, rng)?;
        Ok((
            PartyAPublicKey { matrix_public },
            PartyASecretKey { matrix_state },
        ))
    }

    fn build_cyclic_shift_matrix(&self, shift: usize) -> Result<PolyMatrix, ModularError> {
        let size = self.encoded_cols();
        let degree = self.hss_degree();
        let mut s = PolyMatrix::zeros(size, size, degree, self.params.q)?;
        for row in 0..size {
            let col = (row + shift) % size;
            s.set_constant(row, col, 1)?;
        }
        Ok(s)
    }

    /// private point-function genB (half-chosen, payload fixed to 0).
    pub fn gen_b<R: RngCore + ?Sized>(
        &self,
        crs: &PointFunctionCrs,
        t_b: (usize, usize),
        rng: &mut R,
    ) -> Result<(PartyBPublicKey, PartyBSecretKey), ModularError> {
        self.validate_index(t_b)?;
        let encoded_cols = self.encoded_cols();

        // Build real hss key material and input shares [[e_{i_B,j}]].
        let hss_rng = Self::derive_secure_rng(rng);
        let mut keygen = HssKeyGenerator::new(self.hss_context.clone(), hss_rng.clone());
        let sk = keygen.generate_secret_key();
        let sk_coeffs = self.extract_hss_secret_coeffs(&sk)?;
        let pk = keygen.generate_public_key(&sk);
        let mut encryptor = HssEncryptor::new(self.hss_context.clone(), pk, hss_rng.clone())
            .map_err(|e| ModularError::Backend(format!("{e:?}")))?;

        let mut selection_cts = Vec::with_capacity(self.params.ell);
        for row in 0..self.params.ell {
            let bit = if row == t_b.0 { 1 } else { 0 };
            let ct = encryptor
                .encrypt(bit)
                .map_err(|e| ModularError::Backend(format!("{e:?}")))?;
            selection_cts.push(ct);
        }

        // Fig.4, ring form: E = [S_{j_B} | sk * S_{j_B}].
        let shift = self.build_cyclic_shift_matrix(t_b.1)?;
        let degree = self.hss_degree();
        let mut e =
            PolyMatrix::zeros(encoded_cols, self.memory_ring_cols(), degree, self.params.q)?;
        let sk_poly = PolyElem::from_coeffs(sk_coeffs, self.params.q)?;
        for row in 0..encoded_cols {
            for col in 0..encoded_cols {
                let s_entry = shift.get(row, col)?.clone();
                e.set(row, col, s_entry.clone())?;
                e.set(row, encoded_cols + col, s_entry.mul_negacyclic(&sk_poly)?)?;
            }
        }

        let (matrix_public, matrix_state) =
            encode_b_raw_ring(&crs.matrix_crs, &e, self.params.encode_b_noise, rng)?;
        Ok((
            PartyBPublicKey {
                matrix_public: matrix_public.clone(),
                selection_cts: selection_cts.clone(),
            },
            PartyBSecretKey {
                matrix_state,
                selection_cts,
            },
        ))
    }

    /// Lemma 2 composition (half-chosen -> full): Party A local generation.
    pub fn gen_a_full<R: RngCore + ?Sized>(
        &self,
        crs: &PointFunctionCrs,
        t_a: (usize, usize),
        payload_a: u64,
        rng: &mut R,
    ) -> Result<(FullPartyAPublicKey, FullPartyASecretKey), ModularError> {
        let (half_ab_pk, half_ab_sk) = self.gen_a(crs, t_a, payload_a, rng)?;
        let (half_ba_pk, half_ba_sk) = self.gen_b(crs, t_a, rng)?;
        Ok((
            FullPartyAPublicKey {
                half_ab: half_ab_pk,
                half_ba: half_ba_pk,
            },
            FullPartyASecretKey {
                half_ab: half_ab_sk,
                half_ba: half_ba_sk,
            },
        ))
    }

    /// Lemma 2 composition (half-chosen -> full): Party B local generation.
    pub fn gen_b_full<R: RngCore + ?Sized>(
        &self,
        crs: &PointFunctionCrs,
        t_b: (usize, usize),
        payload_b: u64,
        rng: &mut R,
    ) -> Result<(FullPartyBPublicKey, FullPartyBSecretKey), ModularError> {
        let (half_ab_pk, half_ab_sk) = self.gen_b(crs, t_b, rng)?;
        let (half_ba_pk, half_ba_sk) = self.gen_a(crs, t_b, payload_b, rng)?;
        Ok((
            FullPartyBPublicKey {
                half_ab: half_ab_pk,
                half_ba: half_ba_pk,
            },
            FullPartyBSecretKey {
                half_ab: half_ab_sk,
                half_ba: half_ba_sk,
            },
        ))
    }

    fn extract_hss_secret_coeffs(&self, sk: &SecretKey) -> Result<Vec<u64>, ModularError> {
        let mut s_coeff = sk.value.clone();
        s_coeff.ntt_inverse(&self.hss_params.ring);
        if s_coeff.num_moduli() == 0 {
            return Err(ModularError::InvalidParams("empty HSS secret key"));
        }
        Ok(s_coeff.limb(0).to_vec())
    }

    fn decode_and_extract_t_share(
        &self,
        decoded_q: &PolyMatrix,
    ) -> Result<PolyMatrix, ModularError> {
        let expected_cols = self.memory_ring_cols();
        if decoded_q.cols() < expected_cols {
            return Err(ModularError::DimensionMismatch {
                lhs: (decoded_q.rows(), decoded_q.cols()),
                rhs: (decoded_q.rows(), expected_cols),
                context: "decode_and_extract_t_share",
            });
        }
        decoded_q.slice_cols(0, expected_cols)
    }

    /// private point-function key derivation_A
    pub fn key_der_a(
        &self,
        crs: &PointFunctionCrs,
        pk_b: &PartyBPublicKey,
        sk_a: &PartyASecretKey,
    ) -> Result<EvaluationKey, ModularError> {
        let decoded = decode_a_ring(&crs.matrix_crs, &pk_b.matrix_public, &sk_a.matrix_state)?;
        let t_share = self.decode_and_extract_t_share(&decoded)?;
        Ok(EvaluationKey {
            party: Party::A,
            t_share,
            selection_cts: pk_b.selection_cts.clone(),
            selector_is_trivial_one: self.params.ell == 1,
        })
    }

    /// private point-function key derivation_B
    pub fn key_der_b(
        &self,
        crs: &PointFunctionCrs,
        pk_a: &PartyAPublicKey,
        sk_b: &PartyBSecretKey,
    ) -> Result<EvaluationKey, ModularError> {
        let decoded = decode_b_ring(&crs.matrix_crs, &pk_a.matrix_public, &sk_b.matrix_state)?;
        let t_share = self.decode_and_extract_t_share(&decoded)?;
        Ok(EvaluationKey {
            party: Party::B,
            t_share,
            selection_cts: sk_b.selection_cts.clone(),
            selector_is_trivial_one: self.params.ell == 1,
        })
    }

    /// Lemma 2 composition: derive Party A full evaluation key from Party B full public key.
    pub fn key_der_a_full(
        &self,
        crs: &PointFunctionCrs,
        pk_b: &FullPartyBPublicKey,
        sk_a: &FullPartyASecretKey,
    ) -> Result<FullEvaluationKey, ModularError> {
        let half_ab = self.key_der_a(crs, &pk_b.half_ab, &sk_a.half_ab)?;
        let half_ba = self.key_der_b(crs, &pk_b.half_ba, &sk_a.half_ba)?;
        Ok(FullEvaluationKey {
            party: Party::A,
            half_ab,
            half_ba,
        })
    }

    /// Lemma 2 composition: derive Party B full evaluation key from Party A full public key.
    pub fn key_der_b_full(
        &self,
        crs: &PointFunctionCrs,
        pk_a: &FullPartyAPublicKey,
        sk_b: &FullPartyBSecretKey,
    ) -> Result<FullEvaluationKey, ModularError> {
        let half_ab = self.key_der_b(crs, &pk_a.half_ab, &sk_b.half_ab)?;
        let half_ba = self.key_der_a(crs, &pk_a.half_ba, &sk_b.half_ba)?;
        Ok(FullEvaluationKey {
            party: Party::B,
            half_ab,
            half_ba,
        })
    }

    /// Shift function from Fig.4: cyclic row shift down by `shift`.
    pub fn shift_rows_down(
        &self,
        matrix: &PolyMatrix,
        shift: usize,
    ) -> Result<PolyMatrix, ModularError> {
        if matrix.rows() == 0 {
            return Err(ModularError::InvalidParams("cannot shift empty matrix"));
        }
        let mut out = PolyMatrix::zeros(
            matrix.rows(),
            matrix.cols(),
            matrix.degree(),
            matrix.modulus(),
        )?;
        let s = shift % matrix.rows();
        for src_row in 0..matrix.rows() {
            let dst_row = (src_row + s) % matrix.rows();
            for col in 0..matrix.cols() {
                out.set(dst_row, col, matrix.get(src_row, col)?.clone())?;
            }
        }
        Ok(out)
    }

    /// Mat2Vec helper from Fig.4.
    pub fn mat2vec(&self, matrix: &PolyMatrix) -> Result<ModVector, ModularError> {
        let mut values = Vec::with_capacity(matrix.rows() * matrix.cols());
        for row in 0..matrix.rows() {
            for col in 0..matrix.cols() {
                values.push(matrix.get(row, col)?.coeff(0)?);
            }
        }
        ModVector::from_values(values, matrix.modulus())
    }

    fn ring_elem_to_hss_poly(&self, elem: &PolyElem) -> Result<silent_ring::Poly, ModularError> {
        let degree = self.hss_params.ring.degree();
        if elem.degree() != degree {
            return Err(ModularError::InvalidParams(
                "ring_elem_to_hss_poly: degree mismatch",
            ));
        }
        let moduli = self.hss_params.ring.rns().moduli();
        let num_moduli = moduli.len();
        let mut out = silent_ring::Poly::new(degree, num_moduli);
        for (limb_idx, modulus) in moduli.iter().enumerate() {
            let mod_value = modulus.value();
            for (coeff_idx, coeff) in elem.coeffs().iter().enumerate().take(degree) {
                out.limb_mut(limb_idx)[coeff_idx] = *coeff % mod_value;
            }
        }
        Ok(out)
    }

    fn poly_pair_to_memory_share(
        &self,
        y: &PolyElem,
        y_times_s: &PolyElem,
    ) -> Result<HssShare, ModularError> {
        Ok(HssShare {
            elements: [
                self.ring_elem_to_hss_poly(y)?,
                self.ring_elem_to_hss_poly(y_times_s)?,
            ],
        })
    }

    fn zero_hss_share(&self) -> HssShare {
        let degree = self.hss_params.ring.degree();
        let num_moduli = self.hss_params.ring.rns().moduli().len();
        HssShare {
            elements: [
                silent_ring::Poly::new(degree, num_moduli),
                silent_ring::Poly::new(degree, num_moduli),
            ],
        }
    }

    fn add_hss_share_assign(&self, lhs: &mut HssShare, rhs: &HssShare) {
        lhs.elements[0].add_assign(&rhs.elements[0], &self.hss_params.ring);
        lhs.elements[1].add_assign(&rhs.elements[1], &self.hss_params.ring);
    }

    /// private point-function eval-all using real hss multiplication backend.
    pub fn eval_all(&self, key: &EvaluationKey) -> Result<EvalAllOutput, ModularError> {
        let ell = self.params.ell;
        let encoded_cols = self.encoded_cols();
        let memory_cols = self.memory_ring_cols();
        if key.t_share.rows() != ell || key.t_share.cols() != memory_cols {
            return Err(ModularError::DimensionMismatch {
                lhs: (key.t_share.rows(), key.t_share.cols()),
                rhs: (ell, memory_cols),
                context: "eval_all expects t_share dimensions (ell, 2*encoded_cols)",
            });
        }
        if key.selection_cts.len() != ell {
            return Err(ModularError::VectorLengthMismatch {
                lhs: key.selection_cts.len(),
                rhs: ell,
                context: "eval_all selection length",
            });
        }

        let mut selected = vec![self.zero_hss_share(); ell * encoded_cols];

        if key.selector_is_trivial_one {
            if ell != 1 {
                return Err(ModularError::InvalidParams(
                    "trivial selector optimization requires ell == 1",
                ));
            }
            for col in 0..encoded_cols {
                let mut y = key.t_share.get(0, col)?.clone();
                let mut y_times_s = key.t_share.get(0, encoded_cols + col)?.clone();
                if key.party == Party::B {
                    y.negate_assign();
                    y_times_s.negate_assign();
                }
                selected[col] = self.poly_pair_to_memory_share(&y, &y_times_s)?;
            }
        } else {
            for shift in 0..ell {
                let shifted_t = self.shift_rows_down(&key.t_share, shift)?;
                let input_ct = &key.selection_cts[shift];

                for row in 0..ell {
                    for col in 0..encoded_cols {
                        let mut y = shifted_t.get(row, col)?.clone();
                        let mut y_times_s = shifted_t.get(row, encoded_cols + col)?.clone();
                        if key.party == Party::B {
                            y.negate_assign();
                            y_times_s.negate_assign();
                        }
                        let mem_share = self.poly_pair_to_memory_share(&y, &y_times_s)?;
                        let out_share = HssEvaluator::hss_mult(
                            &mem_share,
                            input_ct,
                            &self.hss_params,
                            self.params.p,
                        )
                        .map_err(|e| ModularError::Backend(format!("{e:?}")))?;
                        let idx = row * encoded_cols + col;
                        self.add_hss_share_assign(&mut selected[idx], &out_share);
                    }
                }
            }
        }

        let final_shares = if self.params.carry_emulation {
            let mut out = vec![self.zero_hss_share(); ell * self.params.m];
            for row in 0..ell {
                for col in 0..self.params.m {
                    let left_idx = row * encoded_cols + col;
                    let src_right_row = (row + ell - 1) % ell;
                    let right_idx = src_right_row * encoded_cols + (self.params.m + col);
                    let dst_idx = row * self.params.m + col;

                    out[dst_idx] = selected[left_idx].clone();
                    self.add_hss_share_assign(&mut out[dst_idx], &selected[right_idx]);
                }
            }
            out
        } else {
            let mut out = vec![self.zero_hss_share(); ell * self.params.m];
            for row in 0..ell {
                for col in 0..self.params.m {
                    out[row * self.params.m + col] = selected[row * encoded_cols + col].clone();
                }
            }
            out
        };
        Ok(EvalAllOutput {
            party: key.party,
            rows: ell,
            cols: self.params.m,
            shares: final_shares,
        })
    }

    /// Lemma 2 composition evaluator: run both half instances and sum local shares.
    pub fn eval_all_full(&self, key: &FullEvaluationKey) -> Result<EvalAllOutput, ModularError> {
        match key.party {
            Party::A => {
                if key.half_ab.party != Party::A || key.half_ba.party != Party::B {
                    return Err(ModularError::InvalidParams(
                        "invalid full key party layout for A",
                    ));
                }
            }
            Party::B => {
                if key.half_ab.party != Party::B || key.half_ba.party != Party::A {
                    return Err(ModularError::InvalidParams(
                        "invalid full key party layout for B",
                    ));
                }
            }
        }

        let out_ab = self.eval_all(&key.half_ab)?;
        let out_ba = self.eval_all(&key.half_ba)?;
        if out_ab.rows != out_ba.rows || out_ab.cols != out_ba.cols {
            return Err(ModularError::DimensionMismatch {
                lhs: (out_ab.rows, out_ab.cols),
                rhs: (out_ba.rows, out_ba.cols),
                context: "eval_all_full",
            });
        }
        if out_ab.shares.len() != out_ba.shares.len() {
            return Err(ModularError::VectorLengthMismatch {
                lhs: out_ab.shares.len(),
                rhs: out_ba.shares.len(),
                context: "eval_all_full shares",
            });
        }

        let mut shares = out_ab.shares;
        for (dst, src) in shares.iter_mut().zip(out_ba.shares.iter()) {
            self.add_hss_share_assign(dst, src);
        }

        Ok(EvalAllOutput {
            party: key.party,
            rows: out_ab.rows,
            cols: out_ab.cols,
            shares,
        })
    }

    pub fn reconstruct_eval_all(
        &self,
        out_a: &EvalAllOutput,
        out_b: &EvalAllOutput,
    ) -> Result<ModVector, ModularError> {
        if out_a.rows != out_b.rows || out_a.cols != out_b.cols {
            return Err(ModularError::DimensionMismatch {
                lhs: (out_a.rows, out_a.cols),
                rhs: (out_b.rows, out_b.cols),
                context: "reconstruct_eval_all",
            });
        }
        if out_a.party != Party::A || out_b.party != Party::B {
            return Err(ModularError::InvalidParams(
                "reconstruct_eval_all expects outputs ordered as (A, B)",
            ));
        }
        let total = out_a.rows * out_a.cols;
        if out_a.shares.len() != total || out_b.shares.len() != total {
            return Err(ModularError::InvalidParams(
                "reconstruct_eval_all share vector length mismatch",
            ));
        }

        let mut values = Vec::with_capacity(total);
        for idx in 0..total {
            let rec = HssEvaluator::hss_reconstruct(
                &[out_a.shares[idx].clone(), out_b.shares[idx].clone()],
                &self.hss_params,
                self.params.p,
            )
            .map_err(|e| ModularError::Backend(format!("{e:?}")))?;
            let val = rec
                .first()
                .copied()
                .ok_or(ModularError::InvalidParams("empty reconstruction output"))?;
            values.push(val);
        }
        ModVector::from_values(values, self.params.p)
    }

    /// Convert one party's HSS memory output from `eval_all` into arithmetic
    /// scalar shares.
    ///
    /// This is the local half of the private point-function/LUT interface used by example's
    /// nonlinear stack: each party can reduce its own HSS memory shares to
    /// arithmetic shares without reconstructing the selected value.
    pub fn output_eval_all_share(
        &self,
        out: &EvalAllOutput,
        output_modulus: u64,
    ) -> Result<ModVector, ModularError> {
        if output_modulus < 2 {
            return Err(ModularError::InvalidParams("output modulus must be >= 2"));
        }
        if out.rows != self.params.ell || out.cols != self.params.m {
            return Err(ModularError::DimensionMismatch {
                lhs: (out.rows, out.cols),
                rhs: (self.params.ell, self.params.m),
                context: "output_eval_all_share",
            });
        }
        let total = out.rows * out.cols;
        if out.shares.len() != total {
            return Err(ModularError::InvalidParams(
                "output_eval_all_share share vector length mismatch",
            ));
        }

        let mut values = Vec::with_capacity(total);
        for share in &out.shares {
            let coeffs = HssEvaluator::hss_output_mem(share, &self.hss_params, output_modulus)
                .map_err(|e| ModularError::Backend(format!("{e:?}")))?;
            let val = coeffs
                .first()
                .copied()
                .ok_or(ModularError::InvalidParams("empty HSS output share"))?;
            values.push(val % output_modulus);
        }
        ModVector::from_values(values, output_modulus)
    }
}

/// Reconstruct subtractive vector shares: share_A - share_B (mod p).
pub fn reconstruct_sub_vector(a: &ModVector, b: &ModVector) -> Result<ModVector, ModularError> {
    if a.modulus() != b.modulus() {
        return Err(ModularError::InvalidParams(
            "reconstruct_sub_vector requires equal moduli",
        ));
    }
    if a.len() != b.len() {
        return Err(ModularError::VectorLengthMismatch {
            lhs: a.len(),
            rhs: b.len(),
            context: "reconstruct_sub_vector",
        });
    }
    let mut out = a.clone();
    out.sub_assign(b)?;
    Ok(out)
}

/// Expected target index for direct (i,j) decomposition with carry handling:
/// row = i_A + i_B + carry(j_A + j_B), col = j_A + j_B mod m.
pub fn expected_hot_index_with_carry(
    ell: usize,
    m: usize,
    t_a: (usize, usize),
    t_b: (usize, usize),
) -> usize {
    let col_sum = t_a.1 + t_b.1;
    let col = col_sum % m;
    let carry = col_sum / m;
    let row = (t_a.0 + t_b.0 + carry) % ell;
    row * m + col
}
