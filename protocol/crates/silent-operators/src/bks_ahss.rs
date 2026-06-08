use rand::{RngCore, SeedableRng};
use silent_hss::{HssCiphertext, HssContext, HssEncryptor, HssEvalKey, HssEvaluator, HssShare};
use silent_rlwe::PublicKey;
use silent_utils::rng::SecureRng;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ahss::{
    AhssBackend, AhssEvalStats, AhssFullState, AhssMemoryShare, DistributedDecrypt, LinearOpPlan,
    OkdmMaterial, OkdmOracle, RestrictedMulPlan,
};
use crate::coefficient_cert::{CoeffCert, PrimitiveBound};
use crate::direct_reload::{
    ApproxShareResult, BridgeRound, Fh2aApprox, Fh2aExact, MaterializationAuthority,
    ReloadMaterializer, ReloadMode,
};
use crate::error::OperatorError;
use crate::he_bridge::{BridgeState, HeBridgeBackend, PublicLinearOp};

/// Concrete BKS-style AHSS engine backed by `silent-hss`.
///
/// This is the production-facing implementation for the protocol layer: it
/// calls the core BKS-HSS load, distributed decryption, multiplication, and
/// exact re-sharing routines instead of relying on test-only arithmetic mocks.
pub struct BksAhssEngine {
    context: HssContext,
    public_key: PublicKey,
    eval_keys: [HssEvalKey; 2],
    rng: Mutex<SecureRng>,
    instruction_counter: AtomicU64,
    paired_load_instruction: Mutex<Option<u64>>,
}

pub struct BksRoundedFh2a<'a> {
    engine: &'a BksAhssEngine,
    rounding_bits: usize,
    latency_ms: f64,
}

impl<'a> BksRoundedFh2a<'a> {
    pub fn new(engine: &'a BksAhssEngine, rounding_bits: usize) -> Self {
        Self {
            engine,
            rounding_bits,
            latency_ms: 0.0,
        }
    }

    pub fn with_latency_ms(mut self, latency_ms: f64) -> Self {
        self.latency_ms = latency_ms;
        self
    }
}

pub struct BksBridgeRounder<'a> {
    engine: &'a BksAhssEngine,
    latency_ms: f64,
}

impl<'a> BksBridgeRounder<'a> {
    pub fn new(engine: &'a BksAhssEngine) -> Self {
        Self {
            engine,
            latency_ms: 0.0,
        }
    }

    pub fn with_latency_ms(mut self, latency_ms: f64) -> Self {
        self.latency_ms = latency_ms;
        self
    }
}

impl BksAhssEngine {
    pub fn new(
        context: HssContext,
        public_key: PublicKey,
        eval_key0: HssEvalKey,
        eval_key1: HssEvalKey,
        rng: SecureRng,
    ) -> Result<Self, OperatorError> {
        if eval_key0.party_index != 0 || eval_key1.party_index != 1 {
            return Err(OperatorError::InvalidParams(
                "BKS AHSS engine requires eval keys ordered as parties 0 and 1",
            ));
        }
        Ok(Self {
            context,
            public_key,
            eval_keys: [eval_key0, eval_key1],
            rng: Mutex::new(rng),
            instruction_counter: AtomicU64::new(1),
            paired_load_instruction: Mutex::new(None),
        })
    }

    pub fn context(&self) -> &HssContext {
        &self.context
    }

    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }

    pub fn eval_key(&self, party_id: usize) -> Result<&HssEvalKey, OperatorError> {
        self.eval_keys
            .get(party_id)
            .ok_or(OperatorError::InvalidParams(
                "BKS AHSS party id must be 0 or 1",
            ))
    }

    pub fn encrypt_coeffs(&self, coeffs: &[u64]) -> Result<HssCiphertext, OperatorError> {
        let pt = self.poly_from_coeffs(coeffs)?;
        let mut encryptor = HssEncryptor::new(
            self.context.clone(),
            self.public_key.clone(),
            self.fresh_rng()?,
        )?;
        encryptor
            .encrypt_input_poly(&pt)
            .map_err(OperatorError::from)
    }

    pub fn input_material_from_coeffs(
        &self,
        coeffs: &[u64],
    ) -> Result<OkdmMaterial<HssCiphertext>, OperatorError> {
        Ok(OkdmMaterial {
            coordinates: vec![self.encrypt_coeffs(coeffs)?],
            coordinate_tags: vec!["bks_input_material".to_string()],
            noise_bound: 0.0,
        })
    }

    /// Exact pairwise load with a single BKS instruction id for both parties.
    pub fn load_full(
        &self,
        material: OkdmMaterial<HssCiphertext>,
        cert: CoeffCert,
    ) -> Result<AhssFullState<HssCiphertext, HssShare>, OperatorError> {
        material.validate()?;
        cert.validate()?;
        let ct = material
            .coordinates
            .first()
            .ok_or(OperatorError::InvalidParams(
                "BKS AHSS load requires one full HSS ciphertext material",
            ))?;
        let instruction_id = self.next_instruction_id();
        let party0 = HssEvaluator::load(&self.context, &self.eval_keys[0], ct, instruction_id)?;
        let party1 = HssEvaluator::load(&self.context, &self.eval_keys[1], ct, instruction_id)?;
        Ok(AhssFullState {
            input_material: material,
            memory_shares: [
                AhssMemoryShare {
                    party_id: 0,
                    value_times_secret_share: party0,
                },
                AhssMemoryShare {
                    party_id: 1,
                    value_times_secret_share: party1,
                },
            ],
            cert,
        })
    }

    /// Exact restricted multiplication for a share pair.
    ///
    /// It uses `silent-hss::HssEvaluator::mult_pair_exact`, which reconstructs
    /// the product and re-shares it as a fresh exact `(m, m*s)` AHSS memory pair,
    /// avoiding invariant drift across chained restricted multiplications.
    pub fn restricted_mul_pair_exact(
        &self,
        input: &AhssFullState<HssCiphertext, HssShare>,
        memory: &[AhssMemoryShare<HssShare>; 2],
    ) -> Result<([AhssMemoryShare<HssShare>; 2], AhssEvalStats), OperatorError> {
        input.validate()?;
        let ct = input
            .input_material
            .coordinates
            .first()
            .ok_or(OperatorError::InvalidParams(
                "BKS AHSS restricted multiplication requires input material",
            ))?;
        let (party0, party1) = HssEvaluator::mult_pair_exact(
            &self.context,
            &memory[0].value_times_secret_share,
            &memory[1].value_times_secret_share,
            ct,
            &self.eval_keys[0].share,
            &self.eval_keys[1].share,
            &mut self.fresh_rng()?,
        )?;
        Ok((
            [
                AhssMemoryShare {
                    party_id: 0,
                    value_times_secret_share: party0,
                },
                AhssMemoryShare {
                    party_id: 1,
                    value_times_secret_share: party1,
                },
            ],
            AhssEvalStats {
                okdm_count: input.input_material.coordinate_count(),
                ddec_count: 2,
                ring_add_count: 0,
                ring_mul_count: 2,
                communication_bytes: self.share_pair_size_bytes(),
                elapsed_ms: 0.0,
            },
        ))
    }

    pub fn reconstruct_coeffs(
        &self,
        shares: &[AhssMemoryShare<HssShare>; 2],
    ) -> Result<Vec<u64>, OperatorError> {
        HssEvaluator::reconstruct(
            &self.context,
            &[
                shares[0].value_times_secret_share.clone(),
                shares[1].value_times_secret_share.clone(),
            ],
        )
        .map_err(OperatorError::from)
    }

    pub fn decrypt_ciphertext_coeffs(
        &self,
        ciphertext: &HssCiphertext,
    ) -> Result<Vec<u64>, OperatorError> {
        let raw0 = HssEvaluator::dec_share(&self.context, &self.eval_keys[0].share, ciphertext)?;
        let raw1 = HssEvaluator::dec_share(&self.context, &self.eval_keys[1].share, ciphertext)?;
        HssEvaluator::reconstruct(&self.context, &[raw0, raw1]).map_err(OperatorError::from)
    }

    pub fn eval_public_poly_linear(
        &self,
        state: &AhssFullState<HssCiphertext, HssShare>,
        public_coeffs: &[u64],
        op_name: &str,
        output_bound: u128,
    ) -> Result<(AhssFullState<HssCiphertext, HssShare>, AhssEvalStats), OperatorError> {
        state.validate()?;
        if op_name.is_empty() {
            return Err(OperatorError::InvalidParams(
                "BKS AHSS public polynomial linear op name must not be empty",
            ));
        }
        if public_coeffs.len() > self.context.degree() {
            return Err(OperatorError::InvalidParams(
                "BKS AHSS public polynomial exceeds ring degree",
            ));
        }
        let public_poly_ntt = self.plain_ntt_poly_from_coeffs(public_coeffs)?;
        let input_material = OkdmMaterial {
            coordinates: state
                .input_material
                .coordinates
                .iter()
                .map(|ct| HssEvaluator::mul_plain(&self.context, ct, &public_poly_ntt))
                .collect(),
            coordinate_tags: vec![op_name.to_string(); state.input_material.coordinate_count()],
            noise_bound: state.input_material.noise_bound,
        };
        let memory_shares = [
            AhssMemoryShare {
                party_id: state.memory_shares[0].party_id,
                value_times_secret_share: self.mul_share_plain_ntt(
                    &state.memory_shares[0].value_times_secret_share,
                    &public_poly_ntt,
                ),
            },
            AhssMemoryShare {
                party_id: state.memory_shares[1].party_id,
                value_times_secret_share: self.mul_share_plain_ntt(
                    &state.memory_shares[1].value_times_secret_share,
                    &public_poly_ntt,
                ),
            },
        ];
        let mut cert = state.cert.with_bound(output_bound, op_name);
        cert.noise_bound += state.input_material.noise_bound;
        let stats = AhssEvalStats {
            okdm_count: 0,
            ddec_count: 0,
            ring_add_count: 0,
            ring_mul_count: input_material.coordinate_count() + 4,
            communication_bytes: 0,
            elapsed_ms: 0.0,
        };
        Ok((
            AhssFullState {
                input_material,
                memory_shares,
                cert,
            },
            stats,
        ))
    }

    fn next_instruction_id(&self) -> u64 {
        self.instruction_counter.fetch_add(1, Ordering::Relaxed)
    }

    fn fresh_rng(&self) -> Result<SecureRng, OperatorError> {
        let mut seed = [0u8; 32];
        let mut guard = self
            .rng
            .lock()
            .map_err(|_| OperatorError::Backend("BKS AHSS RNG mutex poisoned".to_string()))?;
        guard.fill_bytes(&mut seed);
        Ok(SecureRng::from_seed(seed))
    }

    fn poly_from_coeffs(&self, coeffs: &[u64]) -> Result<silent_ring::Poly, OperatorError> {
        if coeffs.len() > self.context.degree() {
            return Err(OperatorError::InvalidParams(
                "BKS AHSS plaintext coefficient vector exceeds ring degree",
            ));
        }
        let mut poly = silent_ring::Poly::new(self.context.degree(), 1);
        let p = self.context.plain_modulus();
        for (idx, &coeff) in coeffs.iter().enumerate() {
            poly.limb_mut(0)[idx] = coeff % p;
        }
        Ok(poly)
    }

    fn plain_ntt_poly_from_coeffs(
        &self,
        coeffs: &[u64],
    ) -> Result<silent_ring::Poly, OperatorError> {
        let ring = &self.context.runtime_params().ring;
        let degree = ring.degree();
        if coeffs.len() > degree {
            return Err(OperatorError::InvalidParams(
                "BKS AHSS plaintext polynomial exceeds ring degree",
            ));
        }
        let p = self.context.plain_modulus();
        let p_half = p / 2;
        let mut poly = silent_ring::Poly::new(degree, ring.rns().moduli().len());
        for idx in 0..degree {
            let coeff = coeffs.get(idx).copied().unwrap_or(0) % p;
            for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
                let q = modulus.value();
                poly.limb_mut(limb_idx)[idx] = if coeff > p_half {
                    let abs = p - coeff;
                    let abs_mod_q = abs % q;
                    if abs_mod_q == 0 { 0 } else { q - abs_mod_q }
                } else {
                    coeff % q
                };
            }
        }
        poly.ntt_forward(ring);
        Ok(poly)
    }

    fn scalar_plain_poly(&self, scalar: u128) -> Result<silent_ring::Poly, OperatorError> {
        self.plain_ntt_poly_from_coeffs(&[(scalar % self.context.plain_modulus() as u128) as u64])
    }

    fn scale_share(&self, share: &HssShare, scalar: u128) -> Result<HssShare, OperatorError> {
        let scalar_poly = self.scalar_plain_poly(scalar)?;
        Ok(self.mul_share_plain_ntt(share, &scalar_poly))
    }

    fn mul_share_plain_ntt(
        &self,
        share: &HssShare,
        public_poly_ntt: &silent_ring::Poly,
    ) -> HssShare {
        let ring = &self.context.runtime_params().ring;
        let mut out = share.clone();
        out.elements[0].ntt_forward(ring);
        out.elements[0].mul_assign(public_poly_ntt, ring);
        out.elements[0].ntt_inverse(ring);
        out.elements[1].ntt_forward(ring);
        out.elements[1].mul_assign(public_poly_ntt, ring);
        out.elements[1].ntt_inverse(ring);
        out
    }

    fn add_plain_vecs(&self, lhs: &[u64], rhs: &[u64]) -> Result<Vec<u64>, OperatorError> {
        if lhs.len() != rhs.len() {
            return Err(OperatorError::InvalidParams(
                "BKS AHSS plaintext vector lengths must match",
            ));
        }
        let p = self.context.plain_modulus();
        Ok(lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&a, &b)| ((a % p) + (b % p)) % p)
            .collect())
    }

    fn coeff_norm_centered(&self, coeffs: &[u64]) -> u128 {
        let p = self.context.plain_modulus();
        coeffs
            .iter()
            .map(|&coeff| {
                let value = coeff % p;
                value.min(p - value) as u128
            })
            .max()
            .unwrap_or(0)
    }

    fn round_coeffs(&self, coeffs: &[u64], rounding_bits: usize) -> Vec<u64> {
        let p = self.context.plain_modulus();
        if rounding_bits == 0 {
            return coeffs.iter().map(|&value| value % p).collect();
        }
        let step = if rounding_bits >= 63 {
            u64::MAX
        } else {
            1u64 << rounding_bits
        };
        if step <= 1 {
            return coeffs.iter().map(|&value| value % p).collect();
        }
        coeffs
            .iter()
            .map(|&value| {
                let centered = self.centered_i128(value);
                let rounded = round_i128_to_multiple(centered, step as i128);
                rounded.rem_euclid(p as i128) as u64
            })
            .collect()
    }

    fn centered_i128(&self, value: u64) -> i128 {
        let p = self.context.plain_modulus();
        let value = value % p;
        if value > p / 2 {
            -((p - value) as i128)
        } else {
            value as i128
        }
    }

    fn max_centered_error(&self, exact: &[u64], approx: &[u64]) -> i128 {
        let p = self.context.plain_modulus();
        exact
            .iter()
            .zip(approx.iter())
            .map(|(&a, &b)| {
                let a = a % p;
                let b = b % p;
                let diff = if a >= b { a - b } else { b - a };
                diff.min(p - diff) as i128
            })
            .max()
            .unwrap_or(0)
    }

    fn share_pair_size_bytes(&self) -> usize {
        let degree = self.context.degree();
        let limbs = self.context.runtime_params().ring.rns().moduli().len();
        2 * 2 * degree * limbs * core::mem::size_of::<u64>()
    }

    fn ciphertext_size_bytes(&self, ct: &HssCiphertext) -> usize {
        ct.enc_m
            .data
            .iter()
            .chain(ct.enc_m_times_s.data.iter())
            .map(|poly| poly.data().len() * core::mem::size_of::<u64>())
            .sum()
    }

    fn paired_load_instruction_id(&self, party_id: usize) -> Result<u64, OperatorError> {
        let mut guard = self.paired_load_instruction.lock().map_err(|_| {
            OperatorError::Backend("BKS AHSS paired load mutex poisoned".to_string())
        })?;
        match party_id {
            0 => {
                let id = self.next_instruction_id();
                *guard = Some(id);
                Ok(id)
            }
            1 => guard.take().ok_or(OperatorError::Protocol(
                "BKS AHSS party 1 load observed before party 0 load",
            )),
            _ => Err(OperatorError::InvalidParams(
                "BKS AHSS party id must be 0 or 1",
            )),
        }
    }
}

impl Fh2aApprox<HssCiphertext, Vec<u64>> for BksRoundedFh2a<'_> {
    fn share_approx(
        &self,
        ciphertext: &HssCiphertext,
    ) -> Result<ApproxShareResult<Vec<u64>>, OperatorError> {
        let exact = self.engine.decrypt_ciphertext_coeffs(ciphertext)?;
        let reconstructed = self.engine.round_coeffs(&exact, self.rounding_bits);
        let (party0, party1) = HssEvaluator::share_plain_coeffs_exact(
            &self.engine.context,
            &reconstructed,
            &self.engine.eval_keys[0].share,
            &self.engine.eval_keys[1].share,
            &mut self.engine.fresh_rng()?,
        )?;
        let shares = [
            HssEvaluator::output_mem(
                &self.engine.context,
                &party0,
                self.engine.context.plain_modulus(),
            )?,
            HssEvaluator::output_mem(
                &self.engine.context,
                &party1,
                self.engine.context.plain_modulus(),
            )?,
        ];
        Ok(ApproxShareResult {
            shares,
            max_abs_error: self.engine.max_centered_error(&exact, &reconstructed),
            reconstructed,
        })
    }

    fn latency_ms(&self) -> f64 {
        self.latency_ms
    }
}

impl BridgeRound<HssCiphertext> for BksBridgeRounder<'_> {
    fn round_bridge(
        &self,
        ciphertext: &HssCiphertext,
        rounding_bits: usize,
    ) -> Result<HssCiphertext, OperatorError> {
        let exact = self.engine.decrypt_ciphertext_coeffs(ciphertext)?;
        let rounded = self.engine.round_coeffs(&exact, rounding_bits);
        self.engine.encrypt_coeffs(&rounded)
    }

    fn max_abs_error(
        &self,
        original: &HssCiphertext,
        rounded: &HssCiphertext,
    ) -> Result<i128, OperatorError> {
        let exact = self.engine.decrypt_ciphertext_coeffs(original)?;
        let approx = self.engine.decrypt_ciphertext_coeffs(rounded)?;
        Ok(self.engine.max_centered_error(&exact, &approx))
    }

    fn latency_ms(&self) -> f64 {
        self.latency_ms
    }
}

fn round_i128_to_multiple(value: i128, step: i128) -> i128 {
    if step <= 1 {
        return value;
    }
    let half = step / 2;
    if value >= 0 {
        ((value + half) / step) * step
    } else {
        -(((-value + half) / step) * step)
    }
}

impl OkdmOracle<HssCiphertext, Vec<u64>> for BksAhssEngine {
    fn okdm(
        &self,
        value: &Vec<u64>,
        _coordinate_index: usize,
    ) -> Result<HssCiphertext, OperatorError> {
        self.encrypt_coeffs(value)
    }
}

impl DistributedDecrypt<HssCiphertext, HssShare, Vec<u64>> for BksAhssEngine {
    fn ddec(
        &self,
        _party_id: usize,
        key_share: &HssShare,
        ct: &HssCiphertext,
    ) -> Result<Vec<u64>, OperatorError> {
        let poly = HssEvaluator::ddec(&self.context, key_share, ct)?;
        Ok(poly.limb(0).to_vec())
    }
}

impl AhssBackend<HssCiphertext, Vec<u64>, HssShare> for BksAhssEngine {
    fn load_share(
        &self,
        party_id: usize,
        material: &OkdmMaterial<HssCiphertext>,
        cert: &CoeffCert,
    ) -> Result<HssShare, OperatorError> {
        cert.validate()?;
        let ct = material
            .coordinates
            .first()
            .ok_or(OperatorError::InvalidParams(
                "BKS AHSS load requires one full HSS ciphertext material",
            ))?;
        let instruction_id = self.paired_load_instruction_id(party_id)?;
        HssEvaluator::load(&self.context, self.eval_key(party_id)?, ct, instruction_id)
            .map_err(OperatorError::from)
    }

    fn eval_linear_material(
        &self,
        material: &OkdmMaterial<HssCiphertext>,
        op: &LinearOpPlan,
    ) -> Result<OkdmMaterial<HssCiphertext>, OperatorError> {
        op.validate()?;
        let coordinates = match op.primitive {
            PrimitiveBound::PlainMul { l1_coeff_norm, .. } => {
                let scalar_poly = self.scalar_plain_poly(l1_coeff_norm)?;
                material
                    .coordinates
                    .iter()
                    .map(|ct| HssEvaluator::mul_plain(&self.context, ct, &scalar_poly))
                    .collect()
            }
            PrimitiveBound::Rotate { .. } | PrimitiveBound::Add { .. } => {
                material.coordinates.clone()
            }
            _ => {
                return Err(OperatorError::Protocol(
                    "BKS AHSS concrete linear material currently supports add/rotate metadata and scalar plaintext multiplication",
                ));
            }
        };
        Ok(OkdmMaterial {
            coordinates,
            coordinate_tags: op.coordinate_tags.clone(),
            noise_bound: material.noise_bound,
        })
    }

    fn eval_linear_share(
        &self,
        share: &AhssMemoryShare<HssShare>,
        op: &LinearOpPlan,
    ) -> Result<AhssMemoryShare<HssShare>, OperatorError> {
        let value_times_secret_share = match op.primitive {
            PrimitiveBound::PlainMul { l1_coeff_norm, .. } => {
                self.scale_share(&share.value_times_secret_share, l1_coeff_norm)?
            }
            PrimitiveBound::Rotate { .. } | PrimitiveBound::Add { .. } => {
                share.value_times_secret_share.clone()
            }
            _ => {
                return Err(OperatorError::Protocol(
                    "BKS AHSS concrete linear share currently supports add/rotate metadata and scalar plaintext multiplication",
                ));
            }
        };
        Ok(AhssMemoryShare {
            party_id: share.party_id,
            value_times_secret_share,
        })
    }

    fn restricted_mul_share(
        &self,
        input_material: &OkdmMaterial<HssCiphertext>,
        memory_share: &AhssMemoryShare<HssShare>,
        _plan: &RestrictedMulPlan,
    ) -> Result<AhssMemoryShare<HssShare>, OperatorError> {
        let ct = input_material
            .coordinates
            .first()
            .ok_or(OperatorError::InvalidParams(
                "BKS AHSS restricted multiplication requires input material",
            ))?;
        Ok(AhssMemoryShare {
            party_id: memory_share.party_id,
            value_times_secret_share: HssEvaluator::mult(
                &self.context,
                &memory_share.value_times_secret_share,
                ct,
            )?,
        })
    }

    fn add_ct(&self, a: &HssCiphertext, b: &HssCiphertext) -> Result<HssCiphertext, OperatorError> {
        Ok(HssEvaluator::add(&self.context, a, b))
    }

    fn add_pt(&self, a: &Vec<u64>, b: &Vec<u64>) -> Result<Vec<u64>, OperatorError> {
        self.add_plain_vecs(a, b)
    }
}

impl HeBridgeBackend<HssCiphertext, Vec<u64>> for BksAhssEngine {
    fn encrypt_bridge(
        &self,
        x: &Vec<u64>,
        cert: CoeffCert,
    ) -> Result<BridgeState<HssCiphertext>, OperatorError> {
        Ok(BridgeState {
            ciphertext: self.encrypt_coeffs(x)?,
            cert,
            compatible_with_bks_coordinate_1: true,
            he_key_id: "bks-ahss-default-key".to_string(),
        })
    }

    fn public_linear(
        &self,
        state: &BridgeState<HssCiphertext>,
        op: &PublicLinearOp,
    ) -> Result<BridgeState<HssCiphertext>, OperatorError> {
        op.validate()?;
        let ciphertext = match op.primitive {
            PrimitiveBound::PlainMul { l1_coeff_norm, .. } => {
                let scalar_poly = self.scalar_plain_poly(l1_coeff_norm)?;
                HssEvaluator::mul_plain(&self.context, &state.ciphertext, &scalar_poly)
            }
            PrimitiveBound::Add { .. } | PrimitiveBound::Rotate { .. } => state.ciphertext.clone(),
            _ => {
                return Err(OperatorError::Protocol(
                    "BKS AHSS bridge currently supports add/rotate metadata and scalar plaintext multiplication",
                ));
            }
        };
        Ok(BridgeState {
            ciphertext,
            cert: state.cert.clone(),
            compatible_with_bks_coordinate_1: state.compatible_with_bks_coordinate_1,
            he_key_id: op
                .output_key_id
                .clone()
                .unwrap_or_else(|| state.he_key_id.clone()),
        })
    }

    fn size_bytes(&self, ct: &HssCiphertext) -> usize {
        self.ciphertext_size_bytes(ct)
    }
}

impl Fh2aExact<HssCiphertext, Vec<u64>> for BksAhssEngine {
    fn share_exact(&self, ciphertext: &HssCiphertext) -> Result<[Vec<u64>; 2], OperatorError> {
        let raw0 = HssEvaluator::dec_share(&self.context, &self.eval_keys[0].share, ciphertext)?;
        let raw1 = HssEvaluator::dec_share(&self.context, &self.eval_keys[1].share, ciphertext)?;
        let z = HssEvaluator::reconstruct(&self.context, &[raw0, raw1])?;
        let p = self.context.plain_modulus();
        let mut rng = self.fresh_rng()?;
        let mut z0 = Vec::with_capacity(z.len());
        let mut z1 = Vec::with_capacity(z.len());
        for value in z {
            let mask = rng.next_u64() % p;
            z0.push(mask);
            z1.push((value + p - mask) % p);
        }
        Ok([z0, z1])
    }
}

impl Fh2aApprox<HssCiphertext, Vec<u64>> for BksAhssEngine {
    fn share_approx(
        &self,
        ciphertext: &HssCiphertext,
    ) -> Result<ApproxShareResult<Vec<u64>>, OperatorError> {
        let shares = self.share_exact(ciphertext)?;
        let reconstructed = self.combine_plain(&shares)?;
        Ok(ApproxShareResult {
            shares,
            reconstructed,
            max_abs_error: 0,
        })
    }
}

impl ReloadMaterializer<HssCiphertext, Vec<u64>, HssShare> for BksAhssEngine {
    fn combine_plain(&self, shares: &[Vec<u64>; 2]) -> Result<Vec<u64>, OperatorError> {
        self.add_plain_vecs(&shares[0], &shares[1])
    }

    fn materialization_authority(&self) -> MaterializationAuthority {
        MaterializationAuthority::KeyOwnerService
    }

    fn synthesize_material(
        &self,
        value: &Vec<u64>,
        bridge: &BridgeState<HssCiphertext>,
        mode: ReloadMode,
    ) -> Result<AhssFullState<HssCiphertext, HssShare>, OperatorError> {
        let material_ct =
            if mode == ReloadMode::ExactFastPath && bridge.compatible_with_bks_coordinate_1 {
                bridge.ciphertext.clone()
            } else {
                self.encrypt_coeffs(value)?
            };
        let (party0, party1) = HssEvaluator::share_plain_coeffs_exact(
            &self.context,
            value,
            &self.eval_keys[0].share,
            &self.eval_keys[1].share,
            &mut self.fresh_rng()?,
        )?;
        Ok(AhssFullState {
            input_material: OkdmMaterial {
                coordinates: vec![material_ct],
                coordinate_tags: vec!["bks_direct_reload_material".to_string()],
                noise_bound: 0.0,
            },
            memory_shares: [
                AhssMemoryShare {
                    party_id: 0,
                    value_times_secret_share: party0,
                },
                AhssMemoryShare {
                    party_id: 1,
                    value_times_secret_share: party1,
                },
            ],
            cert: bridge.cert.clone(),
        })
    }

    fn coeff_norm(&self, value: &Vec<u64>) -> u128 {
        self.coeff_norm_centered(value)
    }

    fn communication_bytes(&self, value: &Vec<u64>, _mode: ReloadMode) -> usize {
        value.len() * core::mem::size_of::<u64>() * 2
    }

    fn okdm_synthesis_ms(&self, _value: &Vec<u64>, _mode: ReloadMode) -> f64 {
        0.0
    }

    fn material_add_ms(&self, _mode: ReloadMode) -> f64 {
        0.0
    }

    fn load_latency_ms(&self, _mode: ReloadMode) -> f64 {
        0.0
    }
}
