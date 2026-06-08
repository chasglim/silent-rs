use rand::{RngCore, SeedableRng};
use silent_hss::utils::{sample_cbd_poly, sample_ternary_poly};
use silent_hss::{HssBatchEncoder, HssContext};
use silent_math::arith::{add_mod, mul_mod_u64 as mul_mod, sub_mod};
use silent_math::numth::{gcd, prev_ntt_prime};
use silent_ring::Poly;
use silent_rlwe::{Ciphertext, PublicKey, SecretKey};
use std::time::Instant;

use crate::error::OperatorError;
use crate::fixedpoint::{FixedPointConfig, reduce_i128_mod};
use crate::hss_slots::HssSlotEngine;
use crate::lookup::PrivateLookup;
use crate::shares::AdditiveShares;

#[derive(Clone, Debug)]
pub struct GeluConfig {
    pub t1: f64,
    pub t2: f64,
    pub coeffs: [f64; 4],
}

impl Default for GeluConfig {
    fn default() -> Self {
        Self {
            t1: -2.1,
            t2: 0.2,
            coeffs: [-0.0018, 0.5008, 0.1699, -0.0022],
        }
    }
}

#[derive(Clone, Debug)]
pub struct SoftmaxConfig {
    pub exp_clip_min: f64,
    pub exp_clip_max: f64,
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self {
            exp_clip_min: -8.0,
            exp_clip_max: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerNormConfig {
    pub epsilon: f64,
    pub variance_clip_max: f64,
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NonlinearProfile {
    pub selectors: usize,
    pub poly_mul: usize,
    pub trunc: usize,
    pub refresh: usize,
    pub online_ms: u64,
    pub transport_bytes: u64,
}

impl NonlinearProfile {
    fn add_assign(&mut self, other: NonlinearProfile) {
        self.selectors += other.selectors;
        self.poly_mul += other.poly_mul;
        self.trunc += other.trunc;
        self.refresh += other.refresh;
        self.online_ms += other.online_ms;
        self.transport_bytes += other.transport_bytes;
    }
}

#[derive(Clone, Debug)]
pub struct ProfiledShares {
    pub shares: AdditiveShares,
    pub profile: NonlinearProfile,
}

#[derive(Clone, Debug)]
pub struct OfPmpeMaskedOpen {
    pub values: Vec<u64>,
    pub elements: usize,
    pub bytes: u64,
    pub latency_ms: u64,
    pub num_flushes: usize,
    pub num_request_response_rounds: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OfPmpeOfflineMode {
    TrustedDebug,
    SecurePreprocess,
}

impl Default for OfPmpeOfflineMode {
    fn default() -> Self {
        Self::TrustedDebug
    }
}

#[derive(Debug)]
pub struct OfPmpeTaylorPack {
    pub degree: usize,
    pub masks: AdditiveShares,
    pub taylor_coeffs: Vec<AdditiveShares>,
    pub offline_mode: OfPmpeOfflineMode,
    pub offline_ms: u64,
    pub offline_us: u64,
    pub offline_bytes: u64,
    pub pack_size_bytes: u64,
    pub peak_pack_resident_bytes: u64,
    pub streaming_pack: bool,
    consumed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OfPmpeProfile {
    pub offline_mode: OfPmpeOfflineMode,
    pub degree: usize,
    pub slots: usize,
    pub num_oneflow_phases: usize,
    pub num_request_response_rounds: usize,
    pub num_masked_opens: usize,
    pub num_opened_elements: usize,
    pub num_flushes: usize,
    pub num_network_flushes: usize,
    pub public_linear_terms: usize,
    pub online_secret_secret_mul: usize,
    pub online_trunc: usize,
    pub num_fresh_masks: usize,
    pub num_reused_masks: usize,
    pub offline_ms: u64,
    pub online_ms: u64,
    pub offline_us: u64,
    pub online_us: u64,
    pub secure_offline_ms: u64,
    pub trusted_debug_offline_ms: u64,
    pub secure_offline_us: u64,
    pub trusted_debug_offline_us: u64,
    pub offline_bytes: u64,
    pub online_bytes: u64,
    pub oneflow_payload_bytes: u64,
    pub pack_size_bytes: u64,
    pub peak_pack_resident_bytes: u64,
    pub streaming_pack: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RnsPreprocessPlan {
    pub label: String,
    pub slots: usize,
    pub rns_limbs: usize,
    pub taylor_packs: usize,
    pub random_masks: usize,
    pub beaver_triples: usize,
    pub truncation_masks: usize,
    pub public_linear_terms: usize,
    pub pack_size_bytes: u64,
    pub peak_pack_resident_bytes: u64,
    pub streaming_pack: bool,
    pub one_time: bool,
}

impl RnsPreprocessPlan {
    pub fn new(label: impl Into<String>, slots: usize, rns_limbs: usize) -> Self {
        Self {
            label: label.into(),
            slots,
            rns_limbs,
            one_time: true,
            ..Self::default()
        }
    }

    pub fn add_assign(&mut self, rhs: &Self) {
        self.slots = self.slots.saturating_add(rhs.slots);
        self.taylor_packs = self.taylor_packs.saturating_add(rhs.taylor_packs);
        self.random_masks = self.random_masks.saturating_add(rhs.random_masks);
        self.beaver_triples = self.beaver_triples.saturating_add(rhs.beaver_triples);
        self.truncation_masks = self.truncation_masks.saturating_add(rhs.truncation_masks);
        self.public_linear_terms = self
            .public_linear_terms
            .saturating_add(rhs.public_linear_terms);
        self.pack_size_bytes = self.pack_size_bytes.saturating_add(rhs.pack_size_bytes);
        self.peak_pack_resident_bytes = self
            .peak_pack_resident_bytes
            .max(rhs.peak_pack_resident_bytes);
        self.streaming_pack |= rhs.streaming_pack;
        self.one_time &= rhs.one_time;
    }

    pub fn requires_beaver_source(&self) -> bool {
        self.beaver_triples > 0
    }

    pub fn requires_truncation_source(&self) -> bool {
        self.truncation_masks > 0
    }

    pub fn requires_external_correlation_source(&self) -> bool {
        self.requires_beaver_source() || self.requires_truncation_source()
    }
}

#[derive(Clone, Debug)]
pub struct OfPmpeOutput {
    pub shares: AdditiveShares,
    pub profile: OfPmpeProfile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleSemantic {
    RingPolynomial,
    DyadicNumerator,
}

#[derive(Clone, Debug)]
pub struct ScaledShareTensor {
    pub shares: AdditiveShares,
    pub frac_bits: u32,
    pub guard_bits: u32,
    pub semantic: ScaleSemantic,
}

impl ScaledShareTensor {
    pub fn len(&self) -> usize {
        self.shares.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shares.is_empty()
    }

    pub fn modulus(&self) -> u64 {
        self.shares.modulus()
    }
}

#[derive(Clone, Debug)]
pub struct OfPmpePolynomialConfig {
    pub coeffs: Vec<f64>,
    pub chunk_slots: usize,
    pub streaming: bool,
}

impl OfPmpePolynomialConfig {
    pub fn from_coeffs(coeffs: Vec<f64>, chunk_slots: usize) -> Self {
        Self {
            coeffs,
            chunk_slots,
            streaming: true,
        }
    }

    pub fn gelu_global_degree6_probe(chunk_slots: usize) -> Self {
        Self::from_coeffs(
            vec![
                0.11789354887618581,
                0.5000000000000826,
                0.21580646543889759,
                -1.047558059695886e-14,
                -0.0077040940591314375,
                2.60145656290972e-16,
                0.00011217412609841613,
            ],
            chunk_slots,
        )
    }

    pub fn exp_negative_degree6_probe(chunk_slots: usize) -> Self {
        Self::from_coeffs(
            vec![
                0.99534085917744453,
                0.96434016114212506,
                0.43199972513780249,
                0.1103529101787149,
                0.016358121631676367,
                0.0013006462283194368,
                4.2693784288606125e-05,
            ],
            chunk_slots,
        )
    }

    pub fn square(chunk_slots: usize) -> Self {
        Self::from_coeffs(vec![0.0, 0.0, 1.0], chunk_slots)
    }
}

#[derive(Clone, Debug)]
pub struct DyadicOfPmpePolynomialConfig {
    pub degree: usize,
    pub input_frac_bits: u32,
    pub output_frac_bits: u32,
    pub guard_bits: u32,
    pub coeffs_num: Vec<i128>,
    pub modulus_bits: u32,
    pub input_abs_bound_bits: Option<u32>,
    pub output_bound_bits: Option<u32>,
    pub modulus_margin_bits: u32,
    pub allow_modular_wrap: bool,
    pub signed: bool,
    pub chunk_slots: usize,
    pub streaming: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DyadicPolynomialGridAudit {
    pub samples: usize,
    pub input_min: i128,
    pub input_max: i128,
    pub output_min: i128,
    pub output_max: i128,
    pub negative_outputs: usize,
    pub zero_outputs: usize,
    pub output_frac_bits: u32,
}

impl DyadicPolynomialGridAudit {
    pub fn is_nonnegative(&self) -> bool {
        self.negative_outputs == 0
    }
}

impl DyadicOfPmpePolynomialConfig {
    pub fn from_integer_coeffs(
        coeffs_num: Vec<i128>,
        input_frac_bits: u32,
        output_frac_bits: u32,
        guard_bits: u32,
        modulus_bits: u32,
        signed: bool,
        chunk_slots: usize,
    ) -> Result<Self, OperatorError> {
        if coeffs_num.is_empty() {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE requires at least one coefficient",
            ));
        }
        if chunk_slots == 0 {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE chunk size must be positive",
            ));
        }
        Ok(Self {
            degree: coeffs_num.len() - 1,
            input_frac_bits,
            output_frac_bits,
            guard_bits,
            coeffs_num,
            modulus_bits,
            input_abs_bound_bits: None,
            output_bound_bits: None,
            modulus_margin_bits: 1,
            allow_modular_wrap: false,
            signed,
            chunk_slots,
            streaming: true,
        })
    }

    pub fn from_real_coeffs(
        coeffs: &[f64],
        input_frac_bits: u32,
        output_frac_bits: u32,
        guard_bits: u32,
        modulus_bits: u32,
        signed: bool,
        chunk_slots: usize,
    ) -> Result<Self, OperatorError> {
        if coeffs.is_empty() {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE requires at least one coefficient",
            ));
        }
        let mut coeffs_num = Vec::with_capacity(coeffs.len());
        for (degree, &coeff) in coeffs.iter().enumerate() {
            let shift = guard_bits as i64 + output_frac_bits as i64
                - (degree as i64 * input_frac_bits as i64);
            let encoded = round_scaled_coeff_to_i128(coeff, shift)?;
            if encoded == 0 && coeff.abs() > 1e-12 {
                return Err(OperatorError::InvalidParams(
                    "dyadic OF-PMPE coefficient quantized to zero; increase guard_bits",
                ));
            }
            coeffs_num.push(encoded);
        }
        Self::from_integer_coeffs(
            coeffs_num,
            input_frac_bits,
            output_frac_bits,
            guard_bits,
            modulus_bits,
            signed,
            chunk_slots,
        )
    }

    pub fn square(input_frac_bits: u32, chunk_slots: usize) -> Result<Self, OperatorError> {
        Self::from_integer_coeffs(
            vec![0, 0, 1],
            input_frac_bits,
            input_frac_bits
                .checked_mul(2)
                .ok_or(OperatorError::InvalidParams(
                    "dyadic OF-PMPE square output frac_bits overflow",
                ))?,
            0,
            0,
            true,
            chunk_slots,
        )
    }

    pub fn output_frac_bits_with_guard(&self) -> Result<u32, OperatorError> {
        self.output_frac_bits
            .checked_add(self.guard_bits)
            .ok_or(OperatorError::InvalidParams(
                "dyadic OF-PMPE output frac_bits overflow",
            ))
    }

    pub fn with_input_abs_bound_bits(mut self, bits: u32) -> Result<Self, OperatorError> {
        self.input_abs_bound_bits = Some(bits);
        self.output_bound_bits = Some(estimate_dyadic_output_bound_bits(&self.coeffs_num, bits)?);
        Ok(self)
    }

    pub fn with_modulus_margin_bits(mut self, bits: u32) -> Self {
        self.modulus_margin_bits = bits;
        self
    }

    pub fn with_modular_wrap_allowed(mut self, allow: bool) -> Self {
        self.allow_modular_wrap = allow;
        self
    }

    pub fn modulus_wrap_safe(&self, actual_modulus_bits: u32) -> Option<bool> {
        self.output_bound_bits.map(|bound| {
            bound
                .saturating_add(self.modulus_margin_bits)
                .saturating_add(u32::from(self.signed))
                < actual_modulus_bits
        })
    }

    pub fn audit_integer_grid(
        &self,
        input_min: i128,
        input_max: i128,
        samples: usize,
    ) -> Result<DyadicPolynomialGridAudit, OperatorError> {
        if samples == 0 {
            return Err(OperatorError::InvalidParams(
                "dyadic polynomial grid audit requires at least one sample",
            ));
        }
        if input_min > input_max {
            return Err(OperatorError::InvalidParams(
                "dyadic polynomial grid audit input_min must be <= input_max",
            ));
        }
        let mut output_min = i128::MAX;
        let mut output_max = i128::MIN;
        let mut negative_outputs = 0usize;
        let mut zero_outputs = 0usize;
        let span = input_max
            .checked_sub(input_min)
            .ok_or(OperatorError::InvalidParams(
                "dyadic polynomial grid audit input range overflow",
            ))?;
        let denominator = samples.saturating_sub(1) as i128;

        for idx in 0..samples {
            let input = if samples == 1 {
                input_min
            } else {
                input_min
                    .checked_add(
                        span.checked_mul(idx as i128)
                            .ok_or(OperatorError::InvalidParams(
                                "dyadic polynomial grid audit sample index overflow",
                            ))?
                            / denominator,
                    )
                    .ok_or(OperatorError::InvalidParams(
                        "dyadic polynomial grid audit sample value overflow",
                    ))?
            };
            let output = dyadic_integer_polynomial_checked(&self.coeffs_num, input)?;
            output_min = output_min.min(output);
            output_max = output_max.max(output);
            if output < 0 {
                negative_outputs += 1;
            } else if output == 0 {
                zero_outputs += 1;
            }
        }

        Ok(DyadicPolynomialGridAudit {
            samples,
            input_min,
            input_max,
            output_min,
            output_max,
            negative_outputs,
            zero_outputs,
            output_frac_bits: self.output_frac_bits_with_guard()?,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DyadicScaleProfile {
    pub input_frac_bits: u32,
    pub output_frac_bits: u32,
    pub guard_bits: u32,
    pub effective_output_frac_bits: u32,
    pub input_abs_bound_bits: Option<u32>,
    pub output_bound_bits: Option<u32>,
    pub modulus_bits: u32,
    pub modulus_margin_bits: u32,
    pub modulus_wrap_safe: Option<bool>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoWrapAudit {
    pub output_bound_bits: u32,
    pub modulus_bits: u32,
    pub signed_margin_bits: i32,
    pub ok: bool,
}

#[derive(Clone, Debug)]
pub struct RnsDyadicDomain {
    moduli: Vec<u64>,
    modulus_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RnsShareTensor {
    limbs: Vec<AdditiveShares>,
}

#[derive(Clone, Debug)]
pub struct RnsScaledShareTensor {
    pub shares: RnsShareTensor,
    pub frac_bits: u32,
    pub guard_bits: u32,
    pub semantic: ScaleSemantic,
}

#[derive(Debug)]
pub struct RnsDyadicTaylorPack {
    pub degree: usize,
    pub masks: RnsShareTensor,
    pub taylor_coeffs: Vec<RnsShareTensor>,
    pub offline_mode: OfPmpeOfflineMode,
    pub audit: NoWrapAudit,
    pub offline_ms: u64,
    pub offline_us: u64,
    pub offline_bytes: u64,
    pub pack_size_bytes: u64,
    pub peak_pack_resident_bytes: u64,
    consumed: bool,
}

#[derive(Debug)]
pub struct RnsDyadicRescalePack {
    pub masks: RnsShareTensor,
    pub mask_quotients: RnsShareTensor,
    pub shift_bits: u32,
    pub input_bound_bits: u32,
    pub mask_bits: u32,
    pub offline_mode: OfPmpeOfflineMode,
    pub offline_ms: u64,
    pub offline_us: u64,
    pub offline_bytes: u64,
    pub pack_size_bytes: u64,
    pub peak_pack_resident_bytes: u64,
    consumed: bool,
}

#[derive(Clone, Debug)]
pub struct RnsDyadicRescaleOutput {
    pub tensor: RnsScaledShareTensor,
    pub shift_bits: u32,
    pub profile: OfPmpeProfile,
}

#[derive(Clone, Debug)]
pub struct RnsDyadicOfPmpeOutput {
    pub tensor: RnsScaledShareTensor,
    pub audit: NoWrapAudit,
    pub profile: OfPmpeProfile,
}

#[derive(Clone, Debug)]
pub struct RnsMeanShiftSoftmaxExpOutput {
    pub shifted: RnsScaledShareTensor,
    pub exp: RnsScaledShareTensor,
    pub row_sums: RnsScaledShareTensor,
    pub exp_profile: OfPmpeProfile,
    pub exp_audit: NoWrapAudit,
    pub mean_shift_ms: u64,
    pub mean_shift_us: u64,
    pub sum_ms: u64,
    pub sum_us: u64,
    pub online_ms: u64,
    pub online_us: u64,
}

#[derive(Clone, Debug)]
pub struct RnsMeanShiftSoftmaxOutput {
    pub exp_phase: RnsMeanShiftSoftmaxExpOutput,
    pub row_sums_base: Option<RnsScaledShareTensor>,
    pub inv_row_sums: RnsScaledShareTensor,
    pub inv_row_sums_repeated: RnsScaledShareTensor,
    pub probs: RnsScaledShareTensor,
    pub rescale_profile: Option<OfPmpeProfile>,
    pub reciprocal_profile: OfPmpeProfile,
    pub broadcast_mul_profile: OfPmpeProfile,
    pub reciprocal_audit: NoWrapAudit,
    pub repeat_ms: u64,
    pub repeat_us: u64,
    pub online_ms: u64,
    pub online_us: u64,
}

#[derive(Debug)]
pub struct RnsBeaverMulPack {
    pub lhs_masks: RnsShareTensor,
    pub rhs_masks: RnsShareTensor,
    pub product_masks: RnsShareTensor,
    pub offline_mode: OfPmpeOfflineMode,
    pub offline_ms: u64,
    pub offline_us: u64,
    pub offline_bytes: u64,
    pub pack_size_bytes: u64,
    pub peak_pack_resident_bytes: u64,
    consumed: bool,
}

pub trait RnsCorrelationSource {
    fn offline_mode(&self) -> OfPmpeOfflineMode;

    fn dyadic_of_pmpe_taylor_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsDyadicTaylorPack, OperatorError>;

    fn approximate_rescale_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
    ) -> Result<RnsDyadicRescalePack, OperatorError>;

    fn beaver_mul_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
    ) -> Result<RnsBeaverMulPack, OperatorError>;
}

pub struct RnsTrustedDebugCorrelationSource<'a, R: RngCore + ?Sized> {
    rng: &'a mut R,
}

impl<'a, R: RngCore + ?Sized> RnsTrustedDebugCorrelationSource<'a, R> {
    pub fn new(rng: &'a mut R) -> Self {
        Self { rng }
    }
}

impl<R: RngCore + ?Sized> RnsCorrelationSource for RnsTrustedDebugCorrelationSource<'_, R> {
    fn offline_mode(&self) -> OfPmpeOfflineMode {
        OfPmpeOfflineMode::TrustedDebug
    }

    fn dyadic_of_pmpe_taylor_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsDyadicTaylorPack, OperatorError> {
        domain.dyadic_of_pmpe_taylor_pack(slots, cfg, self.rng)
    }

    fn approximate_rescale_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
    ) -> Result<RnsDyadicRescalePack, OperatorError> {
        domain.approximate_rescale_pack(slots, shift_bits, input_bound_bits, mask_bits, self.rng)
    }

    fn beaver_mul_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
    ) -> Result<RnsBeaverMulPack, OperatorError> {
        domain.beaver_mul_pack(slots, self.rng)
    }
}

pub struct RnsIdealCorrelationSource<'a, R: RngCore + ?Sized> {
    rng: &'a mut R,
}

impl<'a, R: RngCore + ?Sized> RnsIdealCorrelationSource<'a, R> {
    pub fn new(rng: &'a mut R) -> Self {
        Self { rng }
    }

    fn random_share_tensor(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
    ) -> Result<RnsShareTensor, OperatorError> {
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS ideal correlation source requires at least one slot",
            ));
        }
        let limbs = domain
            .moduli
            .iter()
            .map(|&modulus| {
                let party0 = (0..slots)
                    .map(|_| self.rng.next_u64() % modulus)
                    .collect::<Vec<_>>();
                let party1 = (0..slots)
                    .map(|_| self.rng.next_u64() % modulus)
                    .collect::<Vec<_>>();
                AdditiveShares::new(modulus, party0, party1)
            })
            .collect::<Result<Vec<_>, _>>()?;
        RnsShareTensor::from_limbs(limbs)
    }

    fn public_share_tensor(
        domain: &RnsDyadicDomain,
        slots: usize,
        value: i128,
    ) -> Result<RnsShareTensor, OperatorError> {
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS public share tensor requires at least one slot",
            ));
        }
        let limbs = domain
            .moduli
            .iter()
            .map(|&modulus| {
                let residue = reduce_i128_mod(value, modulus);
                AdditiveShares::share_public(&vec![residue; slots], modulus)
            })
            .collect::<Result<Vec<_>, _>>()?;
        RnsShareTensor::from_limbs(limbs)
    }

    fn beaver_mul_offline(
        &mut self,
        domain: &RnsDyadicDomain,
        lhs: &RnsShareTensor,
        rhs: &RnsShareTensor,
    ) -> Result<(RnsShareTensor, u64), OperatorError> {
        lhs.check_domain(domain)?;
        rhs.check_domain(domain)?;
        if lhs.len() != rhs.len() {
            return Err(OperatorError::InvalidParams(
                "RNS ideal Beaver offline multiplication inputs must have equal length",
            ));
        }
        let mut pack = self.beaver_mul_pack(domain, lhs.len())?;
        let pack_bytes = pack.offline_bytes;
        pack.consumed = true;

        let lhs_delta = lhs.sub(&pack.lhs_masks)?;
        let rhs_delta = rhs.sub(&pack.rhs_masks)?;
        let lhs_delta_public = lhs_delta.reconstruct_residues();
        let rhs_delta_public = rhs_delta.reconstruct_residues();
        let slots = lhs.len();
        let mut out_limbs = Vec::with_capacity(domain.moduli.len());

        for (limb_idx, &modulus) in domain.moduli.iter().enumerate() {
            let a = &pack.lhs_masks.limbs[limb_idx];
            let b = &pack.rhs_masks.limbs[limb_idx];
            let c = &pack.product_masks.limbs[limb_idx];
            let mut out0 = c.party0().to_vec();
            let mut out1 = c.party1().to_vec();
            for slot in 0..slots {
                let d = lhs_delta_public[limb_idx][slot];
                let e = rhs_delta_public[limb_idx][slot];
                out0[slot] = add_mod(out0[slot], mul_mod(d, b.party0()[slot], modulus), modulus);
                out1[slot] = add_mod(out1[slot], mul_mod(d, b.party1()[slot], modulus), modulus);
                out0[slot] = add_mod(out0[slot], mul_mod(e, a.party0()[slot], modulus), modulus);
                out1[slot] = add_mod(out1[slot], mul_mod(e, a.party1()[slot], modulus), modulus);
                out0[slot] = add_mod(out0[slot], mul_mod(d, e, modulus), modulus);
            }
            out_limbs.push(AdditiveShares::new(modulus, out0, out1)?);
        }
        Ok((RnsShareTensor::from_limbs(out_limbs)?, pack_bytes))
    }
}

impl<R: RngCore + ?Sized> RnsCorrelationSource for RnsIdealCorrelationSource<'_, R> {
    fn offline_mode(&self) -> OfPmpeOfflineMode {
        OfPmpeOfflineMode::SecurePreprocess
    }

    fn dyadic_of_pmpe_taylor_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsDyadicTaylorPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS ideal OF-PMPE Taylor pack requires at least one slot",
            ));
        }
        if cfg.coeffs_num.is_empty() || cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "RNS ideal OF-PMPE degree must match coefficient count",
            ));
        }
        let audit = domain.no_wrap_audit(cfg)?;
        if !audit.ok && !cfg.allow_modular_wrap {
            return Err(OperatorError::InvalidParams(
                "RNS ideal OF-PMPE output bound exceeds CRT modulus",
            ));
        }

        let masks = self.random_share_tensor(domain, slots)?;
        let mut powers = Vec::with_capacity(cfg.degree + 1);
        powers.push(Self::public_share_tensor(domain, slots, 1)?);
        if cfg.degree >= 1 {
            powers.push(masks.clone());
        }

        let mut extra_offline_bytes = 0u64;
        for degree in 2..=cfg.degree {
            let (power, bytes) = self.beaver_mul_offline(domain, &powers[degree - 1], &masks)?;
            extra_offline_bytes = extra_offline_bytes.saturating_add(bytes);
            powers.push(power);
        }

        let zero = Self::public_share_tensor(domain, slots, 0)?;
        let mut taylor_coeffs = Vec::with_capacity(cfg.degree + 1);
        for k in 0..=cfg.degree {
            let mut acc = zero.clone();
            for j in k..=cfg.degree {
                let term = rns_mul_public_taylor_factor(&powers[j - k], cfg.coeffs_num[j], j, k)?;
                acc = acc.add(&term)?;
            }
            taylor_coeffs.push(acc);
        }

        let elapsed = start.elapsed();
        let mut pack = domain.dyadic_of_pmpe_taylor_pack_from_preprocessed(
            masks,
            taylor_coeffs,
            cfg,
            OfPmpeOfflineMode::SecurePreprocess,
            elapsed.as_micros() as u64,
        )?;
        pack.offline_bytes = pack.offline_bytes.saturating_add(extra_offline_bytes);
        pack.peak_pack_resident_bytes = pack
            .peak_pack_resident_bytes
            .saturating_add(extra_offline_bytes);
        Ok(pack)
    }

    fn approximate_rescale_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
    ) -> Result<RnsDyadicRescalePack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS ideal rescale pack requires at least one slot",
            ));
        }
        let masks_plain = (0..slots)
            .map(|_| sample_u128_bits(self.rng, mask_bits) as i128)
            .collect::<Vec<_>>();
        let quotients = masks_plain
            .iter()
            .map(|&mask| mask >> shift_bits)
            .collect::<Vec<_>>();
        let masks = domain.share_with_rng(&masks_plain, self.rng)?;
        let mask_quotients = domain.share_with_rng(&quotients, self.rng)?;
        domain.approximate_rescale_pack_from_preprocessed(
            masks,
            mask_quotients,
            shift_bits,
            input_bound_bits,
            mask_bits,
            OfPmpeOfflineMode::SecurePreprocess,
            start.elapsed().as_micros() as u64,
        )
    }

    fn beaver_mul_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
    ) -> Result<RnsBeaverMulPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS ideal Beaver multiplication pack requires at least one slot",
            ));
        }
        let mut lhs_limbs = Vec::with_capacity(domain.moduli.len());
        let mut rhs_limbs = Vec::with_capacity(domain.moduli.len());
        let mut product_limbs = Vec::with_capacity(domain.moduli.len());
        for &modulus in &domain.moduli {
            let lhs = (0..slots)
                .map(|_| self.rng.next_u64() % modulus)
                .collect::<Vec<_>>();
            let rhs = (0..slots)
                .map(|_| self.rng.next_u64() % modulus)
                .collect::<Vec<_>>();
            let product = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| mul_mod(a, b, modulus))
                .collect::<Vec<_>>();
            lhs_limbs.push(AdditiveShares::share_with_rng(&lhs, modulus, self.rng)?);
            rhs_limbs.push(AdditiveShares::share_with_rng(&rhs, modulus, self.rng)?);
            product_limbs.push(AdditiveShares::share_with_rng(&product, modulus, self.rng)?);
        }
        domain.beaver_mul_pack_from_preprocessed(
            RnsShareTensor::from_limbs(lhs_limbs)?,
            RnsShareTensor::from_limbs(rhs_limbs)?,
            RnsShareTensor::from_limbs(product_limbs)?,
            OfPmpeOfflineMode::SecurePreprocess,
            start.elapsed().as_micros() as u64,
        )
    }
}

pub struct RnsHybridEngineeringCorrelationSource<'a, R: RngCore + ?Sized> {
    rng: &'a mut R,
    rescale_statistical_security_bits: u32,
}

impl<'a, R: RngCore + ?Sized> RnsHybridEngineeringCorrelationSource<'a, R> {
    pub fn new(rng: &'a mut R) -> Self {
        Self {
            rng,
            rescale_statistical_security_bits: 40,
        }
    }

    pub fn with_rescale_statistical_security_bits(mut self, bits: u32) -> Self {
        self.rescale_statistical_security_bits = bits;
        self
    }
}

impl<R: RngCore + ?Sized> RnsCorrelationSource for RnsHybridEngineeringCorrelationSource<'_, R> {
    fn offline_mode(&self) -> OfPmpeOfflineMode {
        OfPmpeOfflineMode::SecurePreprocess
    }

    fn dyadic_of_pmpe_taylor_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsDyadicTaylorPack, OperatorError> {
        RnsIdealCorrelationSource { rng: self.rng }.dyadic_of_pmpe_taylor_pack(domain, slots, cfg)
    }

    fn approximate_rescale_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
    ) -> Result<RnsDyadicRescalePack, OperatorError> {
        rns_coin_tossed_rescale_pack(
            domain,
            slots,
            shift_bits,
            input_bound_bits,
            mask_bits,
            self.rescale_statistical_security_bits,
            self.rng,
        )
    }

    fn beaver_mul_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
    ) -> Result<RnsBeaverMulPack, OperatorError> {
        RnsIdealCorrelationSource { rng: self.rng }.beaver_mul_pack(domain, slots)
    }
}

/// Native Taylor-correlation generator using binomial cross terms.
///
/// This backend targets PMPE directly:
/// it samples `r = r0 + r1`, generates shares of the basis terms
/// `r0^i * r1^j` for `i,j > 0, i+j <= degree`, and linearly combines that
/// triangular basis into `[D_0(r)], ..., [D_d(r)]`.  It intentionally does not
/// expose generic Beaver triples; that abstraction is too strong for PMPE
/// Taylor preprocessing and was measured to dominate offline time.
pub struct RnsCrossTermOleTaylorSource<'a, R: RngCore + ?Sized> {
    rng: &'a mut R,
    rescale_statistical_security_bits: u32,
}

impl<'a, R: RngCore + ?Sized> RnsCrossTermOleTaylorSource<'a, R> {
    pub fn new(rng: &'a mut R) -> Self {
        Self {
            rng,
            rescale_statistical_security_bits: 40,
        }
    }

    pub fn with_rescale_statistical_security_bits(mut self, bits: u32) -> Self {
        self.rescale_statistical_security_bits = bits;
        self
    }
}

impl<R: RngCore + ?Sized> RnsCorrelationSource for RnsCrossTermOleTaylorSource<'_, R> {
    fn offline_mode(&self) -> OfPmpeOfflineMode {
        OfPmpeOfflineMode::SecurePreprocess
    }

    fn dyadic_of_pmpe_taylor_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsDyadicTaylorPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS cross-term OLE Taylor pack requires at least one slot",
            ));
        }
        if cfg.coeffs_num.is_empty() || cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "RNS cross-term OLE Taylor degree must match coefficient count",
            ));
        }
        let audit = domain.no_wrap_audit(cfg)?;
        if !audit.ok && !cfg.allow_modular_wrap {
            return Err(OperatorError::InvalidParams(
                "RNS cross-term OLE Taylor output bound exceeds CRT modulus",
            ));
        }

        let (masks, powers, cross_term_bytes) =
            rns_binomial_cross_term_powers(domain, slots, cfg.degree, self.rng)?;
        let zero = rns_public_share_tensor(domain, slots, 0)?;
        let mut taylor_coeffs = Vec::with_capacity(cfg.degree + 1);
        for k in 0..=cfg.degree {
            let mut acc = zero.clone();
            for j in k..=cfg.degree {
                let term = rns_mul_public_taylor_factor(&powers[j - k], cfg.coeffs_num[j], j, k)?;
                acc = acc.add(&term)?;
            }
            taylor_coeffs.push(acc);
        }

        let mut pack = domain.dyadic_of_pmpe_taylor_pack_from_preprocessed(
            masks,
            taylor_coeffs,
            cfg,
            OfPmpeOfflineMode::SecurePreprocess,
            start.elapsed().as_micros() as u64,
        )?;
        pack.offline_bytes = pack.offline_bytes.saturating_add(cross_term_bytes);
        pack.peak_pack_resident_bytes = pack
            .peak_pack_resident_bytes
            .saturating_add(cross_term_bytes.min(pack.pack_size_bytes));
        Ok(pack)
    }

    fn approximate_rescale_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
    ) -> Result<RnsDyadicRescalePack, OperatorError> {
        rns_coin_tossed_rescale_pack(
            domain,
            slots,
            shift_bits,
            input_bound_bits,
            mask_bits,
            self.rescale_statistical_security_bits,
            self.rng,
        )
    }

    fn beaver_mul_pack(
        &mut self,
        _domain: &RnsDyadicDomain,
        _slots: usize,
    ) -> Result<RnsBeaverMulPack, OperatorError> {
        Err(OperatorError::InvalidParams(
            "RNS cross-term OLE Taylor source does not provide generic Beaver triples",
        ))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedRlweAheTripleProfile {
    pub slots: usize,
    pub ciphertexts_sent: usize,
    pub send_bytes: u64,
    pub recv_bytes: u64,
    pub offline_us: u64,
}

impl PackedRlweAheTripleProfile {
    fn add_assign(&mut self, other: PackedRlweAheTripleProfile) {
        self.slots = self.slots.saturating_add(other.slots);
        self.ciphertexts_sent = self.ciphertexts_sent.saturating_add(other.ciphertexts_sent);
        self.send_bytes = self.send_bytes.saturating_add(other.send_bytes);
        self.recv_bytes = self.recv_bytes.saturating_add(other.recv_bytes);
        self.offline_us = self.offline_us.saturating_add(other.offline_us);
    }

    pub fn total_transport_bytes(&self) -> u64 {
        self.send_bytes.saturating_add(self.recv_bytes)
    }
}

/// Packed RLWE-AHE Beaver triple generator.
///
/// This is the backend shape we want for large-batch arithmetic correlations:
/// P0 sends `Enc(a0), Enc(b0)`, P1 replies with
/// `Enc(a0*b1 + b0*a1 + r)`, and the parties locally set
/// `c0 = a0*b0 + d`, `c1 = a1*b1 - r`.  The implementation is a local
/// two-party simulation over the real SILENT RLWE public-key encryption
/// primitives; it deliberately avoids the Whisper prototype's generic HSS
/// cross-term path.
#[derive(Clone)]
pub struct PackedRlweAheTripleEngine {
    context: HssContext,
    public_key: PublicKey,
    decrypt_key: SecretKey,
    delta: Vec<u64>,
}

impl PackedRlweAheTripleEngine {
    pub fn new(
        context: HssContext,
        public_key: PublicKey,
        decrypt_key: SecretKey,
    ) -> Result<Self, OperatorError> {
        let delta = packed_ahe_delta_mod_q(&context);
        Ok(Self {
            context,
            public_key,
            decrypt_key,
            delta,
        })
    }

    pub fn context(&self) -> &HssContext {
        &self.context
    }

    pub fn generate_triples<R: RngCore + ?Sized>(
        &self,
        slots: usize,
        rng: &mut R,
    ) -> Result<
        (
            AdditiveShares,
            AdditiveShares,
            AdditiveShares,
            PackedRlweAheTripleProfile,
        ),
        OperatorError,
    > {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "packed RLWE-AHE triple generation requires at least one slot",
            ));
        }
        if slots > self.context.degree() {
            return Err(OperatorError::InvalidParams(
                "packed RLWE-AHE triple slot count exceeds ring degree",
            ));
        }
        let modulus = self.context.plain_modulus();
        let degree = self.context.degree();
        let sample_vec = |rng: &mut R| {
            (0..slots)
                .map(|_| rng.next_u64() % modulus)
                .collect::<Vec<_>>()
        };
        let a0 = sample_vec(rng);
        let b0 = sample_vec(rng);
        let a1 = sample_vec(rng);
        let b1 = sample_vec(rng);
        let r = sample_vec(rng);

        let encoder = HssBatchEncoder::from_context(&self.context);
        let pt_a0 = encoder.encode(&a0);
        let pt_b0 = encoder.encode(&b0);
        let pt_a1 = encoder.encode(&a1);
        let pt_b1 = encoder.encode(&b1);
        let pt_r = encoder.encode(&r);

        let ct_a0 = packed_ahe_encrypt_poly(
            &self.context,
            &self.public_key,
            &pt_a0,
            &self.delta,
            derive_secure_rng_for_packed_ahe(rng),
        )?;
        let ct_b0 = packed_ahe_encrypt_poly(
            &self.context,
            &self.public_key,
            &pt_b0,
            &self.delta,
            derive_secure_rng_for_packed_ahe(rng),
        )?;

        let ct_r = packed_ahe_encrypt_poly(
            &self.context,
            &self.public_key,
            &pt_r,
            &self.delta,
            derive_secure_rng_for_packed_ahe(rng),
        )?;
        let ct_a0_b1 = bfv_mul_ciphertext_by_plain(&ct_a0, &pt_b1, &self.context)?;
        let ct_b0_a1 = bfv_mul_ciphertext_by_plain(&ct_b0, &pt_a1, &self.context)?;
        let ct_cross = bfv_add_ciphertexts(&ct_a0_b1, &ct_b0_a1, &self.context)?;
        let ct_d = bfv_add_ciphertexts(&ct_cross, &ct_r, &self.context)?;

        let d_plain = bfv_decrypt_exact_poly(&ct_d, &self.decrypt_key, &self.context)?;
        let mut d = encoder.decode(&d_plain.value);
        d.truncate(slots);
        if d.len() < slots {
            return Err(OperatorError::Backend(
                "packed RLWE-AHE decrypt returned too few slots".to_string(),
            ));
        }

        let mut c0 = Vec::with_capacity(slots);
        let mut c1 = Vec::with_capacity(slots);
        for i in 0..slots {
            let local0 = mul_mod(a0[i], b0[i], modulus);
            let local1 = mul_mod(a1[i], b1[i], modulus);
            c0.push(add_mod(local0, d[i] % modulus, modulus));
            c1.push(sub_mod(local1, r[i], modulus));
        }

        let lhs = AdditiveShares::new(modulus, a0, a1)?;
        let rhs = AdditiveShares::new(modulus, b0, b1)?;
        let product = AdditiveShares::new(modulus, c0, c1)?;
        let sent = ciphertext_size_bytes(&ct_a0).saturating_add(ciphertext_size_bytes(&ct_b0));
        let received = ciphertext_size_bytes(&ct_d);
        let profile = PackedRlweAheTripleProfile {
            slots: degree.min(slots),
            ciphertexts_sent: 3,
            send_bytes: sent,
            recv_bytes: received,
            offline_us: start.elapsed().as_micros() as u64,
        };
        Ok((lhs, rhs, product, profile))
    }

    pub fn generate_taylor_coefficients_from_cross_terms<R: RngCore + ?Sized>(
        &self,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<
        (
            AdditiveShares,
            Vec<AdditiveShares>,
            PackedRlweAheTripleProfile,
        ),
        OperatorError,
    > {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "packed RLWE-AHE Taylor OLE generation requires at least one slot",
            ));
        }
        if slots > self.context.degree() {
            return Err(OperatorError::InvalidParams(
                "packed RLWE-AHE Taylor OLE slot count exceeds ring degree",
            ));
        }
        if cfg.coeffs_num.is_empty() || cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "packed RLWE-AHE Taylor OLE degree must match coefficient count",
            ));
        }

        let modulus = self.context.plain_modulus();
        let degree = self.context.degree();
        let sample_vec = |rng: &mut R| {
            (0..slots)
                .map(|_| rng.next_u64() % modulus)
                .collect::<Vec<_>>()
        };
        let r0 = sample_vec(rng);
        let r1 = sample_vec(rng);

        let mut r0_powers = Vec::with_capacity(cfg.degree + 1);
        let mut r1_powers = Vec::with_capacity(cfg.degree + 1);
        r0_powers.push(vec![1; slots]);
        r1_powers.push(vec![1; slots]);
        for power in 1..=cfg.degree {
            let prev0 = &r0_powers[power - 1];
            let prev1 = &r1_powers[power - 1];
            r0_powers.push(
                prev0
                    .iter()
                    .zip(r0.iter())
                    .map(|(&lhs, &rhs)| mul_mod(lhs, rhs, modulus))
                    .collect::<Vec<_>>(),
            );
            r1_powers.push(
                prev1
                    .iter()
                    .zip(r1.iter())
                    .map(|(&lhs, &rhs)| mul_mod(lhs, rhs, modulus))
                    .collect::<Vec<_>>(),
            );
        }

        let encoder = HssBatchEncoder::from_context(&self.context);
        let mut encrypted_r0_powers = Vec::with_capacity(cfg.degree.saturating_sub(1));
        let mut send_bytes = 0u64;
        for power in 1..cfg.degree {
            let pt = encoder.encode(&r0_powers[power]);
            let ct = packed_ahe_encrypt_poly(
                &self.context,
                &self.public_key,
                &pt,
                &self.delta,
                derive_secure_rng_for_packed_ahe(rng),
            )?;
            send_bytes = send_bytes.saturating_add(ciphertext_size_bytes(&ct));
            encrypted_r0_powers.push(ct);
        }

        let mut recv_bytes = 0u64;
        let mut ciphertexts = encrypted_r0_powers.len();
        let mut taylor_coeffs = Vec::with_capacity(cfg.degree + 1);
        for k in 0..=cfg.degree {
            let mut party0 = vec![0; slots];
            let mut party1 = vec![0; slots];
            for j in k..=cfg.degree {
                let monomial_degree = j - k;
                let coeff_mod = reduce_i128_mod(cfg.coeffs_num[j], modulus);
                let taylor_factor = mul_mod(coeff_mod, binomial_mod(j, k, modulus)?, modulus);
                if monomial_degree == 0 {
                    for value in &mut party0 {
                        *value = add_mod(*value, taylor_factor, modulus);
                    }
                } else {
                    for slot in 0..slots {
                        party0[slot] = add_mod(
                            party0[slot],
                            mul_mod(taylor_factor, r0_powers[monomial_degree][slot], modulus),
                            modulus,
                        );
                        party1[slot] = add_mod(
                            party1[slot],
                            mul_mod(taylor_factor, r1_powers[monomial_degree][slot], modulus),
                            modulus,
                        );
                    }
                }
            }

            if cfg.degree.saturating_sub(k) >= 2 {
                let mask = sample_vec(rng);
                let pt_mask = encoder.encode(&mask);
                let mut ct_acc = packed_ahe_encrypt_poly(
                    &self.context,
                    &self.public_key,
                    &pt_mask,
                    &self.delta,
                    derive_secure_rng_for_packed_ahe(rng),
                )?;
                for j in k..=cfg.degree {
                    let monomial_degree = j - k;
                    if monomial_degree < 2 {
                        continue;
                    }
                    let coeff_mod = reduce_i128_mod(cfg.coeffs_num[j], modulus);
                    let taylor_factor = mul_mod(coeff_mod, binomial_mod(j, k, modulus)?, modulus);
                    for left_degree in 1..monomial_degree {
                        let right_degree = monomial_degree - left_degree;
                        let cross_factor = mul_mod(
                            taylor_factor,
                            binomial_mod(monomial_degree, left_degree, modulus)?,
                            modulus,
                        );
                        let plain = (0..slots)
                            .map(|slot| {
                                mul_mod(cross_factor, r1_powers[right_degree][slot], modulus)
                            })
                            .collect::<Vec<_>>();
                        let pt = encoder.encode(&plain);
                        let term = bfv_mul_ciphertext_by_plain(
                            &encrypted_r0_powers[left_degree - 1],
                            &pt,
                            &self.context,
                        )?;
                        ct_acc = bfv_add_ciphertexts(&ct_acc, &term, &self.context)?;
                    }
                }
                let plain = bfv_decrypt_exact_poly(&ct_acc, &self.decrypt_key, &self.context)?;
                let mut masked_cross = encoder.decode(&plain.value);
                masked_cross.truncate(slots);
                if masked_cross.len() < slots {
                    return Err(OperatorError::Backend(
                        "packed RLWE-AHE Taylor OLE decrypt returned too few slots".to_string(),
                    ));
                }
                for slot in 0..slots {
                    party0[slot] = add_mod(party0[slot], masked_cross[slot] % modulus, modulus);
                    party1[slot] = sub_mod(party1[slot], mask[slot], modulus);
                }
                recv_bytes = recv_bytes.saturating_add(ciphertext_size_bytes(&ct_acc));
                ciphertexts = ciphertexts.saturating_add(1);
            }

            taylor_coeffs.push(AdditiveShares::new(modulus, party0, party1)?);
        }

        let masks = AdditiveShares::new(modulus, r0, r1)?;
        let profile = PackedRlweAheTripleProfile {
            slots: degree.min(slots),
            ciphertexts_sent: ciphertexts,
            send_bytes,
            recv_bytes,
            offline_us: start.elapsed().as_micros() as u64,
        };
        Ok((masks, taylor_coeffs, profile))
    }
}

pub struct RnsRlweAheCorrelationSource<'a, R: RngCore + ?Sized> {
    engines: Vec<PackedRlweAheTripleEngine>,
    rng: &'a mut R,
    rescale_statistical_security_bits: u32,
}

impl<'a, R: RngCore + ?Sized> RnsRlweAheCorrelationSource<'a, R> {
    pub fn new(
        domain: &RnsDyadicDomain,
        engines: Vec<PackedRlweAheTripleEngine>,
        rng: &'a mut R,
    ) -> Result<Self, OperatorError> {
        if engines.len() != domain.moduli.len() {
            return Err(OperatorError::InvalidParams(
                "RNS RLWE-AHE correlation source requires one engine per RNS limb",
            ));
        }
        for (idx, engine) in engines.iter().enumerate() {
            if engine.context().plain_modulus() != domain.moduli[idx] {
                return Err(OperatorError::InvalidParams(
                    "RNS RLWE-AHE engine plaintext modulus must match RNS limb",
                ));
            }
        }
        Ok(Self {
            engines,
            rng,
            rescale_statistical_security_bits: 40,
        })
    }

    pub fn with_rescale_statistical_security_bits(mut self, bits: u32) -> Self {
        self.rescale_statistical_security_bits = bits;
        self
    }

    fn random_share_tensor(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
    ) -> Result<RnsShareTensor, OperatorError> {
        rns_random_share_tensor(domain, slots, self.rng)
    }

    fn public_share_tensor(
        domain: &RnsDyadicDomain,
        slots: usize,
        value: i128,
    ) -> Result<RnsShareTensor, OperatorError> {
        rns_public_share_tensor(domain, slots, value)
    }

    fn beaver_mul_offline(
        &mut self,
        domain: &RnsDyadicDomain,
        lhs: &RnsShareTensor,
        rhs: &RnsShareTensor,
    ) -> Result<(RnsShareTensor, u64), OperatorError> {
        let mut pack = self.beaver_mul_pack(domain, lhs.len())?;
        rns_apply_beaver_mul_pack(domain, lhs, rhs, &mut pack)
    }
}

impl<R: RngCore + ?Sized> RnsCorrelationSource for RnsRlweAheCorrelationSource<'_, R> {
    fn offline_mode(&self) -> OfPmpeOfflineMode {
        OfPmpeOfflineMode::SecurePreprocess
    }

    fn dyadic_of_pmpe_taylor_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsDyadicTaylorPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS RLWE-AHE OF-PMPE Taylor pack requires at least one slot",
            ));
        }
        if cfg.coeffs_num.is_empty() || cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "RNS RLWE-AHE OF-PMPE degree must match coefficient count",
            ));
        }
        let audit = domain.no_wrap_audit(cfg)?;
        if !audit.ok && !cfg.allow_modular_wrap {
            return Err(OperatorError::InvalidParams(
                "RNS RLWE-AHE OF-PMPE output bound exceeds CRT modulus",
            ));
        }

        let masks = self.random_share_tensor(domain, slots)?;
        let mut powers = Vec::with_capacity(cfg.degree + 1);
        powers.push(Self::public_share_tensor(domain, slots, 1)?);
        if cfg.degree >= 1 {
            powers.push(masks.clone());
        }

        let mut extra_offline_bytes = 0u64;
        for degree in 2..=cfg.degree {
            let (power, bytes) = self.beaver_mul_offline(domain, &powers[degree - 1], &masks)?;
            extra_offline_bytes = extra_offline_bytes.saturating_add(bytes);
            powers.push(power);
        }

        let zero = Self::public_share_tensor(domain, slots, 0)?;
        let mut taylor_coeffs = Vec::with_capacity(cfg.degree + 1);
        for k in 0..=cfg.degree {
            let mut acc = zero.clone();
            for j in k..=cfg.degree {
                let term = rns_mul_public_taylor_factor(&powers[j - k], cfg.coeffs_num[j], j, k)?;
                acc = acc.add(&term)?;
            }
            taylor_coeffs.push(acc);
        }

        let mut pack = domain.dyadic_of_pmpe_taylor_pack_from_preprocessed(
            masks,
            taylor_coeffs,
            cfg,
            OfPmpeOfflineMode::SecurePreprocess,
            start.elapsed().as_micros() as u64,
        )?;
        pack.offline_bytes = pack.offline_bytes.saturating_add(extra_offline_bytes);
        pack.peak_pack_resident_bytes = pack
            .peak_pack_resident_bytes
            .saturating_add(extra_offline_bytes);
        Ok(pack)
    }

    fn approximate_rescale_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
    ) -> Result<RnsDyadicRescalePack, OperatorError> {
        rns_coin_tossed_rescale_pack(
            domain,
            slots,
            shift_bits,
            input_bound_bits,
            mask_bits,
            self.rescale_statistical_security_bits,
            self.rng,
        )
    }

    fn beaver_mul_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
    ) -> Result<RnsBeaverMulPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS RLWE-AHE Beaver multiplication pack requires at least one slot",
            ));
        }
        let mut lhs_limbs = Vec::with_capacity(domain.moduli.len());
        let mut rhs_limbs = Vec::with_capacity(domain.moduli.len());
        let mut product_limbs = Vec::with_capacity(domain.moduli.len());
        let mut profile = PackedRlweAheTripleProfile::default();

        for (idx, engine) in self.engines.iter().enumerate() {
            let mut out0 = Vec::with_capacity(slots);
            let mut out1 = Vec::with_capacity(slots);
            let mut out2 = Vec::with_capacity(slots);
            let mut out3 = Vec::with_capacity(slots);
            let mut out4 = Vec::with_capacity(slots);
            let mut out5 = Vec::with_capacity(slots);
            let chunk_slots = engine.context().degree();
            for chunk_start in (0..slots).step_by(chunk_slots) {
                let chunk_len = usize::min(chunk_slots, slots - chunk_start);
                let (lhs, rhs, product, chunk_profile) =
                    engine.generate_triples(chunk_len, self.rng)?;
                out0.extend_from_slice(lhs.party0());
                out1.extend_from_slice(lhs.party1());
                out2.extend_from_slice(rhs.party0());
                out3.extend_from_slice(rhs.party1());
                out4.extend_from_slice(product.party0());
                out5.extend_from_slice(product.party1());
                profile.add_assign(chunk_profile);
            }
            let modulus = domain.moduli[idx];
            lhs_limbs.push(AdditiveShares::new(modulus, out0, out1)?);
            rhs_limbs.push(AdditiveShares::new(modulus, out2, out3)?);
            product_limbs.push(AdditiveShares::new(modulus, out4, out5)?);
        }

        let mut pack = domain.beaver_mul_pack_from_preprocessed(
            RnsShareTensor::from_limbs(lhs_limbs)?,
            RnsShareTensor::from_limbs(rhs_limbs)?,
            RnsShareTensor::from_limbs(product_limbs)?,
            OfPmpeOfflineMode::SecurePreprocess,
            start.elapsed().as_micros() as u64,
        )?;
        let transport = profile.total_transport_bytes();
        pack.offline_bytes = pack.offline_bytes.saturating_add(transport);
        pack.peak_pack_resident_bytes = pack.peak_pack_resident_bytes.saturating_add(transport);
        Ok(pack)
    }
}

pub struct RnsRlweAheCrossTermOleTaylorSource<'a, R: RngCore + ?Sized> {
    engines: Vec<PackedRlweAheTripleEngine>,
    rng: &'a mut R,
    rescale_statistical_security_bits: u32,
}

impl<'a, R: RngCore + ?Sized> RnsRlweAheCrossTermOleTaylorSource<'a, R> {
    pub fn new(
        domain: &RnsDyadicDomain,
        engines: Vec<PackedRlweAheTripleEngine>,
        rng: &'a mut R,
    ) -> Result<Self, OperatorError> {
        if engines.len() != domain.moduli.len() {
            return Err(OperatorError::InvalidParams(
                "RNS RLWE-AHE Cross-term OLE source requires one engine per RNS limb",
            ));
        }
        for (idx, engine) in engines.iter().enumerate() {
            if engine.context().plain_modulus() != domain.moduli[idx] {
                return Err(OperatorError::InvalidParams(
                    "RNS RLWE-AHE Cross-term OLE engine plaintext modulus must match RNS limb",
                ));
            }
        }
        Ok(Self {
            engines,
            rng,
            rescale_statistical_security_bits: 40,
        })
    }

    pub fn with_rescale_statistical_security_bits(mut self, bits: u32) -> Self {
        self.rescale_statistical_security_bits = bits;
        self
    }
}

impl<R: RngCore + ?Sized> RnsCorrelationSource for RnsRlweAheCrossTermOleTaylorSource<'_, R> {
    fn offline_mode(&self) -> OfPmpeOfflineMode {
        OfPmpeOfflineMode::SecurePreprocess
    }

    fn dyadic_of_pmpe_taylor_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsDyadicTaylorPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS RLWE-AHE Cross-term OLE Taylor pack requires at least one slot",
            ));
        }
        let audit = domain.no_wrap_audit(cfg)?;
        if !audit.ok && !cfg.allow_modular_wrap {
            return Err(OperatorError::InvalidParams(
                "RNS RLWE-AHE Cross-term OLE Taylor output bound exceeds CRT modulus",
            ));
        }

        let mut mask_limbs = Vec::with_capacity(domain.moduli.len());
        let mut coeff_limbs = (0..=cfg.degree)
            .map(|_| Vec::with_capacity(domain.moduli.len()))
            .collect::<Vec<_>>();
        let mut profile = PackedRlweAheTripleProfile::default();

        for (idx, engine) in self.engines.iter().enumerate() {
            let modulus = domain.moduli[idx];
            let mut mask0 = Vec::with_capacity(slots);
            let mut mask1 = Vec::with_capacity(slots);
            let mut coeff0 = (0..=cfg.degree)
                .map(|_| Vec::with_capacity(slots))
                .collect::<Vec<_>>();
            let mut coeff1 = (0..=cfg.degree)
                .map(|_| Vec::with_capacity(slots))
                .collect::<Vec<_>>();
            let chunk_slots = engine.context().degree();
            for chunk_start in (0..slots).step_by(chunk_slots) {
                let chunk_len = usize::min(chunk_slots, slots - chunk_start);
                let (masks, coeffs, chunk_profile) = engine
                    .generate_taylor_coefficients_from_cross_terms(chunk_len, cfg, self.rng)?;
                mask0.extend_from_slice(masks.party0());
                mask1.extend_from_slice(masks.party1());
                for (degree, coeff) in coeffs.into_iter().enumerate() {
                    coeff0[degree].extend_from_slice(coeff.party0());
                    coeff1[degree].extend_from_slice(coeff.party1());
                }
                profile.add_assign(chunk_profile);
            }
            mask_limbs.push(AdditiveShares::new(modulus, mask0, mask1)?);
            for degree in 0..=cfg.degree {
                coeff_limbs[degree].push(AdditiveShares::new(
                    modulus,
                    std::mem::take(&mut coeff0[degree]),
                    std::mem::take(&mut coeff1[degree]),
                )?);
            }
        }

        let masks = RnsShareTensor::from_limbs(mask_limbs)?;
        let taylor_coeffs = coeff_limbs
            .into_iter()
            .map(RnsShareTensor::from_limbs)
            .collect::<Result<Vec<_>, _>>()?;
        let mut pack = domain.dyadic_of_pmpe_taylor_pack_from_preprocessed(
            masks,
            taylor_coeffs,
            cfg,
            OfPmpeOfflineMode::SecurePreprocess,
            start.elapsed().as_micros() as u64,
        )?;
        let transport = profile.total_transport_bytes();
        pack.offline_bytes = pack.offline_bytes.saturating_add(transport);
        pack.peak_pack_resident_bytes = pack
            .peak_pack_resident_bytes
            .saturating_add(transport.min(pack.pack_size_bytes));
        Ok(pack)
    }

    fn approximate_rescale_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
    ) -> Result<RnsDyadicRescalePack, OperatorError> {
        rns_coin_tossed_rescale_pack(
            domain,
            slots,
            shift_bits,
            input_bound_bits,
            mask_bits,
            self.rescale_statistical_security_bits,
            self.rng,
        )
    }

    fn beaver_mul_pack(
        &mut self,
        _domain: &RnsDyadicDomain,
        _slots: usize,
    ) -> Result<RnsBeaverMulPack, OperatorError> {
        Err(OperatorError::InvalidParams(
            "RNS RLWE-AHE Cross-term OLE Taylor source does not provide generic Beaver triples",
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RnsHssRescaleMode {
    Reject,
    CoinTossed { statistical_security_bits: u32 },
    IdealFallback,
}

pub struct RnsHssCorrelationSource<'a, R: RngCore + ?Sized> {
    engines: Vec<crate::hss_slots::HssSlotEngine>,
    rng: &'a mut R,
    rescale_mode: RnsHssRescaleMode,
}

impl<'a, R: RngCore + ?Sized> RnsHssCorrelationSource<'a, R> {
    pub fn new(
        domain: &RnsDyadicDomain,
        engines: Vec<crate::hss_slots::HssSlotEngine>,
        rng: &'a mut R,
    ) -> Result<Self, OperatorError> {
        if engines.len() != domain.moduli.len() {
            return Err(OperatorError::InvalidParams(
                "RNS HSS correlation source requires one HSS engine per RNS limb",
            ));
        }
        for (idx, engine) in engines.iter().enumerate() {
            if engine.context().plain_modulus() != domain.moduli[idx] {
                return Err(OperatorError::InvalidParams(
                    "RNS HSS correlation source engine plaintext modulus must match RNS limb",
                ));
            }
        }
        Ok(Self {
            engines,
            rng,
            rescale_mode: RnsHssRescaleMode::Reject,
        })
    }

    /// Enable a two-party, no-dealer rescale mask generator.
    ///
    /// Each party samples local shares of `q` and low bits `u`, then locally forms
    /// `r = q * 2^shift + u`. No party learns the full `r` or `q`; the low-bit
    /// mask is statistical, so callers must choose a security margin suitable for
    /// the declared input bound.
    pub fn with_coin_tossed_rescale(mut self, statistical_security_bits: u32) -> Self {
        self.rescale_mode = RnsHssRescaleMode::CoinTossed {
            statistical_security_bits,
        };
        self
    }

    pub fn with_ideal_rescale_fallback(mut self, allow: bool) -> Self {
        self.rescale_mode = if allow {
            RnsHssRescaleMode::IdealFallback
        } else {
            RnsHssRescaleMode::Reject
        };
        self
    }

    fn random_share_tensor(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
    ) -> Result<RnsShareTensor, OperatorError> {
        RnsIdealCorrelationSource { rng: self.rng }.random_share_tensor(domain, slots)
    }

    fn public_share_tensor(
        domain: &RnsDyadicDomain,
        slots: usize,
        value: i128,
    ) -> Result<RnsShareTensor, OperatorError> {
        RnsIdealCorrelationSource::<R>::public_share_tensor(domain, slots, value)
    }

    fn hss_mul_tensor(
        &mut self,
        domain: &RnsDyadicDomain,
        lhs: &RnsShareTensor,
        rhs: &RnsShareTensor,
    ) -> Result<RnsShareTensor, OperatorError> {
        lhs.check_domain(domain)?;
        rhs.check_domain(domain)?;
        if lhs.len() != rhs.len() {
            return Err(OperatorError::InvalidParams(
                "RNS HSS correlation source multiplication inputs must have equal length",
            ));
        }
        let limbs = self
            .engines
            .iter()
            .enumerate()
            .map(|(idx, engine)| {
                hss_mul_additive_shares_chunked(engine, &lhs.limbs[idx], &rhs.limbs[idx], self.rng)
            })
            .collect::<Result<Vec<_>, _>>()?;
        RnsShareTensor::from_limbs(limbs)
    }
}

impl<R: RngCore + ?Sized> RnsCorrelationSource for RnsHssCorrelationSource<'_, R> {
    fn offline_mode(&self) -> OfPmpeOfflineMode {
        OfPmpeOfflineMode::SecurePreprocess
    }

    fn dyadic_of_pmpe_taylor_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsDyadicTaylorPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS HSS OF-PMPE Taylor pack requires at least one slot",
            ));
        }
        if cfg.coeffs_num.is_empty() || cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "RNS HSS OF-PMPE degree must match coefficient count",
            ));
        }
        let audit = domain.no_wrap_audit(cfg)?;
        if !audit.ok && !cfg.allow_modular_wrap {
            return Err(OperatorError::InvalidParams(
                "RNS HSS OF-PMPE output bound exceeds CRT modulus",
            ));
        }

        let masks = self.random_share_tensor(domain, slots)?;
        let mut powers = Vec::with_capacity(cfg.degree + 1);
        powers.push(Self::public_share_tensor(domain, slots, 1)?);
        if cfg.degree >= 1 {
            powers.push(masks.clone());
        }
        for degree in 2..=cfg.degree {
            powers.push(self.hss_mul_tensor(domain, &powers[degree - 1], &masks)?);
        }

        let zero = Self::public_share_tensor(domain, slots, 0)?;
        let mut taylor_coeffs = Vec::with_capacity(cfg.degree + 1);
        for k in 0..=cfg.degree {
            let mut acc = zero.clone();
            for j in k..=cfg.degree {
                let term = rns_mul_public_taylor_factor(&powers[j - k], cfg.coeffs_num[j], j, k)?;
                acc = acc.add(&term)?;
            }
            taylor_coeffs.push(acc);
        }

        domain.dyadic_of_pmpe_taylor_pack_from_preprocessed(
            masks,
            taylor_coeffs,
            cfg,
            OfPmpeOfflineMode::SecurePreprocess,
            start.elapsed().as_micros() as u64,
        )
    }

    fn approximate_rescale_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
    ) -> Result<RnsDyadicRescalePack, OperatorError> {
        match self.rescale_mode {
            RnsHssRescaleMode::Reject => {
                return Err(OperatorError::InvalidParams(
                    "RNS HSS correlation source needs an explicit truncation-correlation backend; use with_coin_tossed_rescale(...) for statistical two-party rescale masks or with_ideal_rescale_fallback(true) only for engineering probes",
                ));
            }
            RnsHssRescaleMode::CoinTossed {
                statistical_security_bits,
            } => {
                return rns_coin_tossed_rescale_pack(
                    domain,
                    slots,
                    shift_bits,
                    input_bound_bits,
                    mask_bits,
                    statistical_security_bits,
                    self.rng,
                );
            }
            RnsHssRescaleMode::IdealFallback => {}
        }
        RnsIdealCorrelationSource { rng: self.rng }.approximate_rescale_pack(
            domain,
            slots,
            shift_bits,
            input_bound_bits,
            mask_bits,
        )
    }

    fn beaver_mul_pack(
        &mut self,
        domain: &RnsDyadicDomain,
        slots: usize,
    ) -> Result<RnsBeaverMulPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS HSS Beaver multiplication pack requires at least one slot",
            ));
        }
        let lhs_masks = self.random_share_tensor(domain, slots)?;
        let rhs_masks = self.random_share_tensor(domain, slots)?;
        let product_masks = self.hss_mul_tensor(domain, &lhs_masks, &rhs_masks)?;
        domain.beaver_mul_pack_from_preprocessed(
            lhs_masks,
            rhs_masks,
            product_masks,
            OfPmpeOfflineMode::SecurePreprocess,
            start.elapsed().as_micros() as u64,
        )
    }
}

fn rns_random_share_tensor<R: RngCore + ?Sized>(
    domain: &RnsDyadicDomain,
    slots: usize,
    rng: &mut R,
) -> Result<RnsShareTensor, OperatorError> {
    if slots == 0 {
        return Err(OperatorError::InvalidParams(
            "RNS random share tensor requires at least one slot",
        ));
    }
    let limbs = domain
        .moduli
        .iter()
        .map(|&modulus| {
            let party0 = (0..slots)
                .map(|_| rng.next_u64() % modulus)
                .collect::<Vec<_>>();
            let party1 = (0..slots)
                .map(|_| rng.next_u64() % modulus)
                .collect::<Vec<_>>();
            AdditiveShares::new(modulus, party0, party1)
        })
        .collect::<Result<Vec<_>, _>>()?;
    RnsShareTensor::from_limbs(limbs)
}

fn rns_public_share_tensor(
    domain: &RnsDyadicDomain,
    slots: usize,
    value: i128,
) -> Result<RnsShareTensor, OperatorError> {
    if slots == 0 {
        return Err(OperatorError::InvalidParams(
            "RNS public share tensor requires at least one slot",
        ));
    }
    let limbs = domain
        .moduli
        .iter()
        .map(|&modulus| {
            let residue = reduce_i128_mod(value, modulus);
            AdditiveShares::share_public(&vec![residue; slots], modulus)
        })
        .collect::<Result<Vec<_>, _>>()?;
    RnsShareTensor::from_limbs(limbs)
}

fn rns_cross_term_count(degree: usize) -> usize {
    degree.saturating_mul(degree.saturating_sub(1)) / 2
}

fn rns_binomial_cross_term_powers<R: RngCore + ?Sized>(
    domain: &RnsDyadicDomain,
    slots: usize,
    degree: usize,
    rng: &mut R,
) -> Result<(RnsShareTensor, Vec<RnsShareTensor>, u64), OperatorError> {
    if slots == 0 {
        return Err(OperatorError::InvalidParams(
            "RNS cross-term powers require at least one slot",
        ));
    }
    let mut mask_limbs = Vec::with_capacity(domain.moduli.len());
    let mut powers_by_degree = (0..=degree)
        .map(|_| Vec::with_capacity(domain.moduli.len()))
        .collect::<Vec<_>>();

    for &modulus in &domain.moduli {
        let r0 = (0..slots)
            .map(|_| rng.next_u64() % modulus)
            .collect::<Vec<_>>();
        let r1 = (0..slots)
            .map(|_| rng.next_u64() % modulus)
            .collect::<Vec<_>>();
        mask_limbs.push(AdditiveShares::new(modulus, r0.clone(), r1.clone())?);

        let mut r0_powers = Vec::with_capacity(degree + 1);
        let mut r1_powers = Vec::with_capacity(degree + 1);
        r0_powers.push(vec![1; slots]);
        r1_powers.push(vec![1; slots]);
        for power in 1..=degree {
            let prev0 = &r0_powers[power - 1];
            let prev1 = &r1_powers[power - 1];
            r0_powers.push(
                prev0
                    .iter()
                    .zip(r0.iter())
                    .map(|(&lhs, &rhs)| mul_mod(lhs, rhs, modulus))
                    .collect::<Vec<_>>(),
            );
            r1_powers.push(
                prev1
                    .iter()
                    .zip(r1.iter())
                    .map(|(&lhs, &rhs)| mul_mod(lhs, rhs, modulus))
                    .collect::<Vec<_>>(),
            );
        }

        powers_by_degree[0].push(AdditiveShares::share_public(&vec![1; slots], modulus)?);
        if degree >= 1 {
            powers_by_degree[1].push(AdditiveShares::new(modulus, r0.clone(), r1.clone())?);
        }
        for power in 2..=degree {
            let mut party0 = r0_powers[power].clone();
            let mut party1 = r1_powers[power].clone();
            for left_degree in 1..power {
                let right_degree = power - left_degree;
                let binom = binomial_mod(power, left_degree, modulus)?;
                for slot in 0..slots {
                    let cross = mul_mod(
                        r0_powers[left_degree][slot],
                        r1_powers[right_degree][slot],
                        modulus,
                    );
                    let share0 = rng.next_u64() % modulus;
                    let share1 = sub_mod(cross, share0, modulus);
                    party0[slot] = add_mod(party0[slot], mul_mod(binom, share0, modulus), modulus);
                    party1[slot] = add_mod(party1[slot], mul_mod(binom, share1, modulus), modulus);
                }
            }
            powers_by_degree[power].push(AdditiveShares::new(modulus, party0, party1)?);
        }
    }

    let masks = RnsShareTensor::from_limbs(mask_limbs)?;
    let powers = powers_by_degree
        .into_iter()
        .map(RnsShareTensor::from_limbs)
        .collect::<Result<Vec<_>, _>>()?;
    let cross_term_bytes = (slots as u64)
        .saturating_mul(domain.moduli.len() as u64)
        .saturating_mul(rns_cross_term_count(degree) as u64)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<u64>() as u64);
    Ok((masks, powers, cross_term_bytes))
}

fn rns_apply_beaver_mul_pack(
    domain: &RnsDyadicDomain,
    lhs: &RnsShareTensor,
    rhs: &RnsShareTensor,
    pack: &mut RnsBeaverMulPack,
) -> Result<(RnsShareTensor, u64), OperatorError> {
    lhs.check_domain(domain)?;
    rhs.check_domain(domain)?;
    if lhs.len() != rhs.len() {
        return Err(OperatorError::InvalidParams(
            "RNS Beaver offline multiplication inputs must have equal length",
        ));
    }
    if pack.consumed {
        return Err(OperatorError::Protocol(
            "RNS Beaver offline multiplication pack has already been consumed",
        ));
    }
    if pack.lhs_masks.len() != lhs.len()
        || pack.rhs_masks.len() != rhs.len()
        || pack.product_masks.len() != lhs.len()
    {
        return Err(OperatorError::InvalidParams(
            "RNS Beaver offline multiplication pack length mismatch",
        ));
    }
    pack.consumed = true;

    let lhs_delta = lhs.sub(&pack.lhs_masks)?;
    let rhs_delta = rhs.sub(&pack.rhs_masks)?;
    let lhs_delta_public = lhs_delta.reconstruct_residues();
    let rhs_delta_public = rhs_delta.reconstruct_residues();
    let slots = lhs.len();
    let mut out_limbs = Vec::with_capacity(domain.moduli.len());

    for (limb_idx, &modulus) in domain.moduli.iter().enumerate() {
        let a = &pack.lhs_masks.limbs[limb_idx];
        let b = &pack.rhs_masks.limbs[limb_idx];
        let c = &pack.product_masks.limbs[limb_idx];
        let mut out0 = c.party0().to_vec();
        let mut out1 = c.party1().to_vec();
        for slot in 0..slots {
            let d = lhs_delta_public[limb_idx][slot];
            let e = rhs_delta_public[limb_idx][slot];
            out0[slot] = add_mod(out0[slot], mul_mod(d, b.party0()[slot], modulus), modulus);
            out1[slot] = add_mod(out1[slot], mul_mod(d, b.party1()[slot], modulus), modulus);
            out0[slot] = add_mod(out0[slot], mul_mod(e, a.party0()[slot], modulus), modulus);
            out1[slot] = add_mod(out1[slot], mul_mod(e, a.party1()[slot], modulus), modulus);
            out0[slot] = add_mod(out0[slot], mul_mod(d, e, modulus), modulus);
        }
        out_limbs.push(AdditiveShares::new(modulus, out0, out1)?);
    }
    Ok((RnsShareTensor::from_limbs(out_limbs)?, pack.offline_bytes))
}

fn bfv_mul_ciphertext_by_plain(
    ct: &Ciphertext,
    plain: &Poly,
    context: &HssContext,
) -> Result<Ciphertext, OperatorError> {
    let mut res = ct.clone();
    let ring = &context.runtime_params().ring;
    let mut p = plain.clone();
    if p.num_moduli() == 1 && res.data[0].num_moduli() > 1 {
        let mut expanded = Poly::new(p.degree(), res.data[0].num_moduli());
        let limb0 = p.limb(0).to_vec();
        for idx in 0..expanded.num_moduli() {
            expanded.limb_mut(idx).copy_from_slice(&limb0);
        }
        p = expanded;
    }
    if p.degree() != ring.degree() || p.num_moduli() != ring.rns().len() {
        return Err(OperatorError::InvalidParams(
            "BFV plaintext multiplier shape must match ciphertext ring",
        ));
    }
    if res.is_ntt {
        p.ntt_forward(ring);
    }
    for poly in res.data.iter_mut() {
        poly.mul_assign(&p, ring);
    }
    Ok(res)
}

fn bfv_add_ciphertexts(
    lhs: &Ciphertext,
    rhs: &Ciphertext,
    context: &HssContext,
) -> Result<Ciphertext, OperatorError> {
    if lhs.data.len() != rhs.data.len() || lhs.is_ntt != rhs.is_ntt {
        return Err(OperatorError::InvalidParams(
            "BFV ciphertext addition requires matching shape and domain",
        ));
    }
    let mut out = lhs.clone();
    let ring = &context.runtime_params().ring;
    for (left, right) in out.data.iter_mut().zip(rhs.data.iter()) {
        left.add_assign(right, ring);
    }
    Ok(out)
}

fn packed_ahe_encrypt_poly(
    context: &HssContext,
    public_key: &PublicKey,
    poly_p: &Poly,
    delta: &[u64],
    mut rng: silent_utils::rng::SecureRng,
) -> Result<Ciphertext, OperatorError> {
    let params = context.runtime_params();
    let ring = &params.ring;
    let moduli = ring.rns().moduli();
    let degree = ring.degree();
    let num_moduli = moduli.len();

    let mut u = sample_ternary_poly(&mut rng, degree, num_moduli, moduli);
    let mut e0 = sample_cbd_poly(&mut rng, degree, num_moduli, moduli);
    let mut e1 = sample_cbd_poly(&mut rng, degree, num_moduli, moduli);
    u.ntt_forward(ring);
    e0.ntt_forward(ring);
    e1.ntt_forward(ring);

    let pk_ct = &public_key.pk;
    if pk_ct.data.len() != 2 || !pk_ct.is_ntt {
        return Err(OperatorError::InvalidParams(
            "packed RLWE-AHE public key must be a two-component NTT ciphertext",
        ));
    }
    let mut c0 = pk_ct.data[0].clone();
    c0.mul_assign(&u, ring);
    c0.add_assign(&e0, ring);
    let mut c1 = pk_ct.data[1].clone();
    c1.mul_assign(&u, ring);
    c1.add_assign(&e1, ring);

    let lifted = packed_ahe_lift_message_poly(context, poly_p, delta)?;
    c0.add_assign(&lifted, ring);
    Ok(Ciphertext::new(vec![c0, c1], params.clone(), true))
}

fn packed_ahe_delta_mod_q(context: &HssContext) -> Vec<u64> {
    let params = context.runtime_params();
    let plain_modulus = context.plain_modulus();
    let q_big = params.ring.rns().base_prod();
    let (delta_big, _) = q_big.div_mod_u64(plain_modulus);
    params
        .ring
        .rns()
        .moduli()
        .iter()
        .map(|modulus| delta_big.mod_u64(modulus.value()))
        .collect()
}

fn packed_ahe_lift_message_poly(
    context: &HssContext,
    poly_p: &Poly,
    delta: &[u64],
) -> Result<Poly, OperatorError> {
    let params = context.runtime_params();
    let ring = &params.ring;
    let degree = ring.degree();
    if poly_p.degree() != degree || poly_p.num_moduli() != 1 {
        return Err(OperatorError::InvalidParams(
            "packed RLWE-AHE plaintext must be a single-limb polynomial matching the ring degree",
        ));
    }

    let moduli = ring.rns().moduli();
    if delta.len() != moduli.len() {
        return Err(OperatorError::InvalidParams(
            "packed RLWE-AHE delta cache must match ciphertext modulus count",
        ));
    }
    let mut lifted = Poly::new(degree, moduli.len());
    let p_limb = poly_p.limb(0);
    for coeff in 0..degree {
        let m_val = p_limb[coeff];
        for (limb_idx, modulus) in moduli.iter().enumerate() {
            let q = modulus.value();
            lifted.limb_mut(limb_idx)[coeff] = mul_mod(m_val % q, delta[limb_idx], q);
        }
    }
    lifted.ntt_forward(ring);
    Ok(lifted)
}

fn bfv_decrypt_exact_poly(
    ciphertext: &Ciphertext,
    sk: &SecretKey,
    context: &HssContext,
) -> Result<silent_rlwe::Plaintext, OperatorError> {
    let params = context.runtime_params();
    let ring = &params.ring;
    let degree = ring.degree();
    if ciphertext.data.is_empty() {
        return Err(OperatorError::InvalidParams(
            "BFV decrypt requires a non-empty ciphertext",
        ));
    }

    let ct_moduli_count = ciphertext.data[0].num_moduli();
    let mut acc = ciphertext.data[0].clone();
    if !ciphertext.is_ntt {
        acc.ntt_forward(ring);
    }

    let sk_prepared = if sk.value.num_moduli() > ct_moduli_count {
        let mut reduced = Poly::new(degree, ct_moduli_count);
        reduced
            .data_mut()
            .copy_from_slice(&sk.value.data()[..degree * ct_moduli_count]);
        reduced
    } else {
        sk.value.clone()
    };
    let mut s_pow = sk_prepared.clone();
    for idx in 1..ciphertext.data.len() {
        let mut term = ciphertext.data[idx].clone();
        if !ciphertext.is_ntt {
            term.ntt_forward(ring);
        }
        term.mul_assign(&s_pow, ring);
        acc.add_assign(&term, ring);
        if idx < ciphertext.data.len() - 1 {
            s_pow.mul_assign(&sk_prepared, ring);
        }
    }
    acc.ntt_inverse(ring);

    let rounded = params.rns_tool.scale_and_round(acc.data(), degree)?;
    let mut out = Poly::new(degree, 1);
    out.limb_mut(0).copy_from_slice(&rounded);
    Ok(silent_rlwe::Plaintext { value: out })
}

fn ciphertext_size_bytes(ct: &Ciphertext) -> u64 {
    ct.data
        .iter()
        .map(|poly| (poly.data().len() as u64).saturating_mul(std::mem::size_of::<u64>() as u64))
        .sum()
}

fn derive_secure_rng_for_packed_ahe<R: RngCore + ?Sized>(
    rng: &mut R,
) -> silent_utils::rng::SecureRng {
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    silent_utils::rng::SecureRng::from_seed(seed)
}

fn rns_coin_tossed_rescale_pack<R: RngCore + ?Sized>(
    domain: &RnsDyadicDomain,
    slots: usize,
    shift_bits: u32,
    input_bound_bits: u32,
    mask_bits: u32,
    statistical_security_bits: u32,
    rng: &mut R,
) -> Result<RnsDyadicRescalePack, OperatorError> {
    let start = Instant::now();
    if slots == 0 {
        return Err(OperatorError::InvalidParams(
            "RNS coin-tossed rescale pack requires at least one slot",
        ));
    }
    if shift_bits == 0 {
        return Err(OperatorError::InvalidParams(
            "RNS coin-tossed rescale pack requires a positive shift",
        ));
    }
    if mask_bits == 0 || mask_bits > 126 {
        return Err(OperatorError::InvalidParams(
            "RNS coin-tossed rescale mask_bits must be in 1..=126",
        ));
    }
    if mask_bits <= shift_bits {
        return Err(OperatorError::InvalidParams(
            "RNS coin-tossed rescale mask_bits must exceed shift_bits",
        ));
    }
    if mask_bits < input_bound_bits.saturating_add(statistical_security_bits) {
        return Err(OperatorError::InvalidParams(
            "RNS coin-tossed rescale mask does not meet requested statistical margin",
        ));
    }

    let quotient_bits = mask_bits - shift_bits;
    let quotient_share_bits = quotient_bits.saturating_sub(1);
    let low_share_bits = shift_bits.saturating_sub(1);
    if quotient_share_bits.saturating_add(shift_bits) > 126 {
        return Err(OperatorError::InvalidParams(
            "RNS coin-tossed rescale quotient share exceeds i128 audit path",
        ));
    }

    let mut mask_limbs0 = domain
        .moduli
        .iter()
        .map(|_| Vec::with_capacity(slots))
        .collect::<Vec<_>>();
    let mut mask_limbs1 = domain
        .moduli
        .iter()
        .map(|_| Vec::with_capacity(slots))
        .collect::<Vec<_>>();
    let mut quotient_limbs0 = domain
        .moduli
        .iter()
        .map(|_| Vec::with_capacity(slots))
        .collect::<Vec<_>>();
    let mut quotient_limbs1 = domain
        .moduli
        .iter()
        .map(|_| Vec::with_capacity(slots))
        .collect::<Vec<_>>();

    for _ in 0..slots {
        let q0 = sample_u128_bits(rng, quotient_share_bits) as i128;
        let q1 = sample_u128_bits(rng, quotient_share_bits) as i128;
        let u0 = sample_u128_bits(rng, low_share_bits) as i128;
        let u1 = sample_u128_bits(rng, low_share_bits) as i128;
        let mask0 = q0
            .checked_shl(shift_bits)
            .ok_or(OperatorError::InvalidParams(
                "RNS coin-tossed rescale mask share shift overflow",
            ))?
            .checked_add(u0)
            .ok_or(OperatorError::InvalidParams(
                "RNS coin-tossed rescale mask share overflow",
            ))?;
        let mask1 = q1
            .checked_shl(shift_bits)
            .ok_or(OperatorError::InvalidParams(
                "RNS coin-tossed rescale mask share shift overflow",
            ))?
            .checked_add(u1)
            .ok_or(OperatorError::InvalidParams(
                "RNS coin-tossed rescale mask share overflow",
            ))?;
        for (idx, &modulus) in domain.moduli.iter().enumerate() {
            mask_limbs0[idx].push(reduce_i128_mod(mask0, modulus));
            mask_limbs1[idx].push(reduce_i128_mod(mask1, modulus));
            quotient_limbs0[idx].push(reduce_i128_mod(q0, modulus));
            quotient_limbs1[idx].push(reduce_i128_mod(q1, modulus));
        }
    }

    let mask_limbs = domain
        .moduli
        .iter()
        .enumerate()
        .map(|(idx, &modulus)| {
            AdditiveShares::new(modulus, mask_limbs0[idx].clone(), mask_limbs1[idx].clone())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let quotient_limbs = domain
        .moduli
        .iter()
        .enumerate()
        .map(|(idx, &modulus)| {
            AdditiveShares::new(
                modulus,
                quotient_limbs0[idx].clone(),
                quotient_limbs1[idx].clone(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    domain.approximate_rescale_pack_from_preprocessed(
        RnsShareTensor::from_limbs(mask_limbs)?,
        RnsShareTensor::from_limbs(quotient_limbs)?,
        shift_bits,
        input_bound_bits,
        mask_bits,
        OfPmpeOfflineMode::SecurePreprocess,
        start.elapsed().as_micros() as u64,
    )
}

#[derive(Clone, Debug)]
pub struct RnsScaledMulOutput {
    pub tensor: RnsScaledShareTensor,
    pub product_frac_bits: u32,
    pub profile: OfPmpeProfile,
}

fn split_offline_accounting(mode: OfPmpeOfflineMode, ms: u64, us: u64) -> (u64, u64, u64, u64) {
    match mode {
        OfPmpeOfflineMode::TrustedDebug => (0, ms, 0, us),
        OfPmpeOfflineMode::SecurePreprocess => (ms, 0, us, 0),
    }
}

impl NoWrapAudit {
    pub fn from_config(
        cfg: &DyadicOfPmpePolynomialConfig,
        actual_modulus_bits: u32,
    ) -> Option<Self> {
        cfg.output_bound_bits.map(|output_bound_bits| {
            let required_bits = output_bound_bits
                .saturating_add(cfg.modulus_margin_bits)
                .saturating_add(u32::from(cfg.signed));
            let signed_margin_bits = actual_modulus_bits as i32 - required_bits as i32;
            Self {
                output_bound_bits,
                modulus_bits: actual_modulus_bits,
                signed_margin_bits,
                ok: signed_margin_bits >= 0,
            }
        })
    }
}

impl RnsDyadicDomain {
    pub fn from_moduli(moduli: Vec<u64>) -> Result<Self, OperatorError> {
        if moduli.is_empty() {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE requires at least one CRT limb",
            ));
        }
        if moduli.iter().any(|&modulus| modulus < 3) {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic moduli must be >= 3",
            ));
        }
        for lhs in 0..moduli.len() {
            for rhs in lhs + 1..moduli.len() {
                if gcd(moduli[lhs], moduli[rhs]) != 1 {
                    return Err(OperatorError::InvalidParams(
                        "RNS dyadic moduli must be pairwise coprime",
                    ));
                }
            }
        }
        let modulus_bits = checked_crt_modulus_u128(&moduli)
            .map(u128_bit_width)
            .unwrap_or_else(|_| {
                moduli
                    .iter()
                    .map(|&modulus| modulus_bit_width(modulus))
                    .sum()
            });
        Ok(Self {
            moduli,
            modulus_bits,
        })
    }

    pub fn two_limb_61bit() -> Result<Self, OperatorError> {
        Self::limbs_61bit(2)
    }

    pub fn two_limb_61bit_ntt(ntt_degree: usize) -> Result<Self, OperatorError> {
        Self::ntt_prime_limbs(61, 2, ntt_degree)
    }

    pub fn limbs_61bit(count: usize) -> Result<Self, OperatorError> {
        if count == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic domain requires at least one limb",
            ));
        }
        let start = 1u64 << 61;
        let mut cursor = start;
        let mut moduli = Vec::with_capacity(count);
        for _ in 0..count {
            let prime = prev_ntt_prime(cursor, 2).ok_or(OperatorError::InvalidParams(
                "failed to find requested 61-bit RNS prime",
            ))?;
            moduli.push(prime);
            cursor = prime;
        }
        Self::from_moduli(moduli)
    }

    pub fn ntt_prime_limbs(
        bits: u32,
        count: usize,
        ntt_degree: usize,
    ) -> Result<Self, OperatorError> {
        if count == 0 || ntt_degree == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS NTT-prime domain requires positive limb count and NTT degree",
            ));
        }
        if bits == 0 || bits >= 63 {
            return Err(OperatorError::InvalidParams(
                "RNS NTT-prime domain currently supports bit sizes in 1..63",
            ));
        }
        let step = 2u64
            .checked_mul(ntt_degree as u64)
            .ok_or(OperatorError::InvalidParams("RNS NTT-prime step overflow"))?;
        let mut cursor = 1u64 << bits;
        let mut moduli = Vec::with_capacity(count);
        for _ in 0..count {
            let prime = prev_ntt_prime(cursor, step).ok_or(OperatorError::InvalidParams(
                "failed to find requested RNS NTT prime",
            ))?;
            moduli.push(prime);
            cursor = prime;
        }
        Self::from_moduli(moduli)
    }

    pub fn moduli(&self) -> &[u64] {
        &self.moduli
    }

    pub fn modulus_bits(&self) -> u32 {
        self.modulus_bits
    }

    pub fn no_wrap_audit(
        &self,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<NoWrapAudit, OperatorError> {
        let Some(audit) = NoWrapAudit::from_config(cfg, self.modulus_bits) else {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE requires input_abs_bound_bits for no-wrap audit",
            ));
        };
        Ok(audit)
    }

    pub fn share_with_rng<R: RngCore + ?Sized>(
        &self,
        values: &[i128],
        rng: &mut R,
    ) -> Result<RnsShareTensor, OperatorError> {
        RnsShareTensor::share_with_rng(self, values, rng)
    }

    pub fn dyadic_of_pmpe_preprocess_plan(
        &self,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsPreprocessPlan, OperatorError> {
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE preprocess plan requires at least one slot",
            ));
        }
        if cfg.coeffs_num.is_empty() || cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE degree must match coefficient count",
            ));
        }
        let pack_size_bytes = rns_of_pmpe_pack_bytes(slots, cfg.degree, self.moduli.len());
        let chunk_slots = if cfg.streaming {
            cfg.chunk_slots.max(1).min(slots)
        } else {
            slots
        };
        let peak_pack_resident_bytes =
            rns_of_pmpe_pack_bytes(chunk_slots, cfg.degree, self.moduli.len());
        let mut plan =
            RnsPreprocessPlan::new("rns_dyadic_of_pmpe_taylor", slots, self.moduli.len());
        plan.taylor_packs = 1;
        plan.random_masks = slots;
        plan.beaver_triples = cfg.degree.saturating_sub(1).saturating_mul(slots);
        plan.public_linear_terms = cfg.degree + 1;
        plan.pack_size_bytes = pack_size_bytes;
        plan.peak_pack_resident_bytes = peak_pack_resident_bytes;
        plan.streaming_pack = cfg.streaming;
        Ok(plan)
    }

    pub fn approximate_rescale_preprocess_plan(
        &self,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
    ) -> Result<RnsPreprocessPlan, OperatorError> {
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale preprocess plan requires at least one slot",
            ));
        }
        if shift_bits == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale preprocess plan requires a positive shift",
            ));
        }
        let opened_bound_bits = input_bound_bits.max(mask_bits).saturating_add(1);
        if opened_bound_bits > 126 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale preprocess plan exceeds current i128 audit path",
            ));
        }
        let pack_size_bytes = (slots as u64)
            .saturating_mul(self.moduli.len() as u64)
            .saturating_mul(2)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>() as u64);
        let mut plan = RnsPreprocessPlan::new("rns_dyadic_rescale", slots, self.moduli.len());
        plan.random_masks = slots;
        plan.truncation_masks = slots;
        plan.public_linear_terms = 2;
        plan.pack_size_bytes = pack_size_bytes;
        plan.peak_pack_resident_bytes = pack_size_bytes;
        Ok(plan)
    }

    pub fn beaver_mul_preprocess_plan(
        &self,
        slots: usize,
    ) -> Result<RnsPreprocessPlan, OperatorError> {
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS Beaver preprocess plan requires at least one slot",
            ));
        }
        let pack_size_bytes = (slots as u64)
            .saturating_mul(self.moduli.len() as u64)
            .saturating_mul(3)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>() as u64);
        let mut plan = RnsPreprocessPlan::new("rns_beaver_mul", slots, self.moduli.len());
        plan.random_masks = slots.saturating_mul(2);
        plan.beaver_triples = slots;
        plan.public_linear_terms = 4;
        plan.pack_size_bytes = pack_size_bytes;
        plan.peak_pack_resident_bytes = pack_size_bytes;
        Ok(plan)
    }

    pub fn beaver_mul_streaming_preprocess_plan(
        &self,
        slots: usize,
        chunk_slots: usize,
    ) -> Result<RnsPreprocessPlan, OperatorError> {
        let mut plan = self.beaver_mul_preprocess_plan(slots)?;
        let chunk_slots = chunk_slots.max(1).min(slots);
        plan.peak_pack_resident_bytes = (chunk_slots as u64)
            .saturating_mul(self.moduli.len() as u64)
            .saturating_mul(3)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>() as u64);
        plan.streaming_pack = true;
        Ok(plan)
    }

    pub fn mean_shift_softmax_preprocess_plan(
        &self,
        rows: usize,
        cols: usize,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        row_sum_shift_bits: u32,
        row_sum_bound_bits: u32,
        rescale_mask_bits: u32,
        reciprocal_cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsPreprocessPlan, OperatorError> {
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS mean-shift softmax preprocess plan dimensions must be non-empty",
            ));
        }
        let exp_slots = rows.saturating_mul(cols);
        let mut plan = RnsPreprocessPlan::new("rns_mean_shift_softmax", 0, self.moduli.len());
        let exp_plan = self.dyadic_of_pmpe_preprocess_plan(exp_slots, exp_cfg)?;
        plan.add_assign(&exp_plan);
        let rescale_plan = self.approximate_rescale_preprocess_plan(
            rows,
            row_sum_shift_bits,
            row_sum_bound_bits,
            rescale_mask_bits,
        )?;
        plan.add_assign(&rescale_plan);
        let recip_plan = self.dyadic_of_pmpe_preprocess_plan(rows, reciprocal_cfg)?;
        plan.add_assign(&recip_plan);
        let broadcast_plan = if exp_cfg.streaming {
            self.beaver_mul_streaming_preprocess_plan(exp_slots, exp_cfg.chunk_slots)?
        } else {
            self.beaver_mul_preprocess_plan(exp_slots)?
        };
        plan.add_assign(&broadcast_plan);
        plan.label = "rns_mean_shift_softmax".to_string();
        Ok(plan)
    }

    pub fn dyadic_of_pmpe_taylor_pack_from_preprocessed(
        &self,
        masks: RnsShareTensor,
        taylor_coeffs: Vec<RnsShareTensor>,
        cfg: &DyadicOfPmpePolynomialConfig,
        offline_mode: OfPmpeOfflineMode,
        offline_us: u64,
    ) -> Result<RnsDyadicTaylorPack, OperatorError> {
        masks.check_domain(self)?;
        if masks.is_empty() {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE preprocessed pack requires at least one slot",
            ));
        }
        if cfg.coeffs_num.is_empty() || cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE degree must match coefficient count",
            ));
        }
        if taylor_coeffs.len() != cfg.degree + 1 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE preprocessed Taylor coefficient count mismatch",
            ));
        }
        for coeffs in &taylor_coeffs {
            coeffs.check_domain(self)?;
            if coeffs.len() != masks.len() {
                return Err(OperatorError::InvalidParams(
                    "RNS dyadic OF-PMPE preprocessed coefficient lengths must match masks",
                ));
            }
        }
        let audit = self.no_wrap_audit(cfg)?;
        if !audit.ok && !cfg.allow_modular_wrap {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE output bound exceeds CRT modulus",
            ));
        }
        let pack_size_bytes = rns_of_pmpe_pack_bytes(masks.len(), cfg.degree, self.moduli.len());
        Ok(RnsDyadicTaylorPack {
            degree: cfg.degree,
            masks,
            taylor_coeffs,
            offline_mode,
            audit,
            offline_ms: offline_us / 1000,
            offline_us,
            offline_bytes: pack_size_bytes,
            pack_size_bytes,
            peak_pack_resident_bytes: pack_size_bytes,
            consumed: false,
        })
    }

    pub fn approximate_rescale_pack_from_preprocessed(
        &self,
        masks: RnsShareTensor,
        mask_quotients: RnsShareTensor,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
        offline_mode: OfPmpeOfflineMode,
        offline_us: u64,
    ) -> Result<RnsDyadicRescalePack, OperatorError> {
        if shift_bits == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale preprocessed pack requires a positive shift",
            ));
        }
        if mask_bits == 0 || mask_bits > 126 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale mask_bits must be in 1..=126",
            ));
        }
        masks.check_domain(self)?;
        mask_quotients.check_domain(self)?;
        if masks.is_empty() || masks.len() != mask_quotients.len() {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale preprocessed pack lengths must be non-empty and equal",
            ));
        }
        let opened_bound_bits = input_bound_bits.max(mask_bits).saturating_add(1);
        if opened_bound_bits > 126 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale opened bound exceeds current i128 audit path",
            ));
        }
        if self.modulus_bits <= opened_bound_bits.saturating_add(1) {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale mask/input bound exceeds CRT modulus",
            ));
        }
        let pack_size_bytes = (masks.len() as u64)
            .saturating_mul(self.moduli.len() as u64)
            .saturating_mul(2)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>() as u64);
        Ok(RnsDyadicRescalePack {
            masks,
            mask_quotients,
            shift_bits,
            input_bound_bits,
            mask_bits,
            offline_mode,
            offline_ms: offline_us / 1000,
            offline_us,
            offline_bytes: pack_size_bytes,
            pack_size_bytes,
            peak_pack_resident_bytes: pack_size_bytes,
            consumed: false,
        })
    }

    pub fn beaver_mul_pack_from_preprocessed(
        &self,
        lhs_masks: RnsShareTensor,
        rhs_masks: RnsShareTensor,
        product_masks: RnsShareTensor,
        offline_mode: OfPmpeOfflineMode,
        offline_us: u64,
    ) -> Result<RnsBeaverMulPack, OperatorError> {
        lhs_masks.check_domain(self)?;
        rhs_masks.check_domain(self)?;
        product_masks.check_domain(self)?;
        if lhs_masks.is_empty()
            || lhs_masks.len() != rhs_masks.len()
            || lhs_masks.len() != product_masks.len()
        {
            return Err(OperatorError::InvalidParams(
                "RNS Beaver preprocessed pack lengths must be non-empty and equal",
            ));
        }
        let pack_size_bytes = (lhs_masks.len() as u64)
            .saturating_mul(self.moduli.len() as u64)
            .saturating_mul(3)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>() as u64);
        Ok(RnsBeaverMulPack {
            lhs_masks,
            rhs_masks,
            product_masks,
            offline_mode,
            offline_ms: offline_us / 1000,
            offline_us,
            offline_bytes: pack_size_bytes,
            pack_size_bytes,
            peak_pack_resident_bytes: pack_size_bytes,
            consumed: false,
        })
    }

    pub fn dyadic_of_pmpe_taylor_pack<R: RngCore + ?Sized>(
        &self,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<RnsDyadicTaylorPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE Taylor pack requires at least one slot",
            ));
        }
        if cfg.coeffs_num.is_empty() || cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE degree must match coefficient count",
            ));
        }
        let audit = self.no_wrap_audit(cfg)?;
        if !audit.ok && !cfg.allow_modular_wrap {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE output bound exceeds CRT modulus",
            ));
        }
        let degree = cfg.degree;
        let mut mask_limbs = Vec::with_capacity(self.moduli.len());
        let mut coeff_columns = (0..=degree)
            .map(|_| Vec::with_capacity(self.moduli.len()))
            .collect::<Vec<_>>();

        for &modulus in &self.moduli {
            let masks_plain = (0..slots)
                .map(|_| rng.next_u64() % modulus)
                .collect::<Vec<_>>();
            let coeffs_mod = cfg
                .coeffs_num
                .iter()
                .map(|&coeff| reduce_i128_mod(coeff, modulus))
                .collect::<Vec<_>>();
            let mut columns = vec![vec![0u64; slots]; degree + 1];
            for (slot, &mask) in masks_plain.iter().enumerate() {
                let mut mask_powers = vec![1u64; degree + 1];
                for idx in 1..=degree {
                    mask_powers[idx] = mul_mod(mask_powers[idx - 1], mask, modulus);
                }
                for k in 0..=degree {
                    let mut acc = 0u64;
                    for j in k..=degree {
                        let mut term =
                            mul_mod(coeffs_mod[j], binomial_mod(j, k, modulus)?, modulus);
                        term = mul_mod(term, mask_powers[j - k], modulus);
                        acc = add_mod(acc, term, modulus);
                    }
                    columns[k][slot] = acc;
                }
            }
            mask_limbs.push(AdditiveShares::share_with_rng(&masks_plain, modulus, rng)?);
            for (k, column) in columns.into_iter().enumerate() {
                coeff_columns[k].push(AdditiveShares::share_with_rng(&column, modulus, rng)?);
            }
        }

        let masks = RnsShareTensor::from_limbs(mask_limbs)?;
        let taylor_coeffs = coeff_columns
            .into_iter()
            .map(RnsShareTensor::from_limbs)
            .collect::<Result<Vec<_>, _>>()?;
        let pack_size_bytes = rns_of_pmpe_pack_bytes(slots, degree, self.moduli.len());
        let elapsed = start.elapsed();
        Ok(RnsDyadicTaylorPack {
            degree,
            masks,
            taylor_coeffs,
            offline_mode: OfPmpeOfflineMode::TrustedDebug,
            audit,
            offline_ms: elapsed.as_millis() as u64,
            offline_us: elapsed.as_micros() as u64,
            offline_bytes: pack_size_bytes,
            pack_size_bytes,
            peak_pack_resident_bytes: pack_size_bytes,
            consumed: false,
        })
    }

    pub fn dyadic_of_pmpe_eval(
        &self,
        input: &RnsShareTensor,
        pack: &mut RnsDyadicTaylorPack,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<RnsDyadicOfPmpeOutput, OperatorError> {
        let start = Instant::now();
        if pack.consumed {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE Taylor pack has already been consumed",
            ));
        }
        pack.consumed = true;
        input.check_domain(self)?;
        pack.masks.check_domain(self)?;
        if input.len() != pack.masks.len() {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE input length must match pack masks",
            ));
        }
        if pack.degree != cfg.degree || pack.taylor_coeffs.len() != pack.degree + 1 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE pack degree mismatch",
            ));
        }
        let delta = input.sub(&pack.masks)?;
        let delta_public = delta.reconstruct_residues();
        let slots = input.len();
        let mut out_limbs = Vec::with_capacity(self.moduli.len());

        for (limb_idx, &modulus) in self.moduli.iter().enumerate() {
            let d0 = &pack.taylor_coeffs[0].limbs[limb_idx];
            let mut out0 = d0.party0().to_vec();
            let mut out1 = d0.party1().to_vec();
            let mut delta_power = vec![1u64; slots];
            for k in 1..=pack.degree {
                let dk = &pack.taylor_coeffs[k].limbs[limb_idx];
                for slot in 0..slots {
                    delta_power[slot] =
                        mul_mod(delta_power[slot], delta_public[limb_idx][slot], modulus);
                    out0[slot] = add_mod(
                        out0[slot],
                        mul_mod(dk.party0()[slot], delta_power[slot], modulus),
                        modulus,
                    );
                    out1[slot] = add_mod(
                        out1[slot],
                        mul_mod(dk.party1()[slot], delta_power[slot], modulus),
                        modulus,
                    );
                }
            }
            out_limbs.push(AdditiveShares::new(modulus, out0, out1)?);
        }

        let elapsed = start.elapsed();
        let online_bytes = input
            .len()
            .saturating_mul(self.moduli.len())
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>()) as u64;
        let (
            secure_offline_ms,
            trusted_debug_offline_ms,
            secure_offline_us,
            trusted_debug_offline_us,
        ) = split_offline_accounting(pack.offline_mode, pack.offline_ms, pack.offline_us);
        Ok(RnsDyadicOfPmpeOutput {
            tensor: RnsScaledShareTensor {
                shares: RnsShareTensor::from_limbs(out_limbs)?,
                frac_bits: cfg.output_frac_bits_with_guard()?,
                guard_bits: cfg.guard_bits,
                semantic: ScaleSemantic::DyadicNumerator,
            },
            audit: pack.audit,
            profile: OfPmpeProfile {
                offline_mode: pack.offline_mode,
                degree: pack.degree,
                slots: input.len(),
                num_oneflow_phases: 1,
                num_request_response_rounds: 0,
                num_masked_opens: input.len(),
                num_opened_elements: input.len(),
                num_flushes: 1,
                num_network_flushes: 1,
                public_linear_terms: pack.degree + 1,
                online_secret_secret_mul: 0,
                online_trunc: 0,
                num_fresh_masks: input.len(),
                num_reused_masks: 0,
                offline_ms: pack.offline_ms,
                online_ms: elapsed.as_millis() as u64,
                offline_us: pack.offline_us,
                online_us: elapsed.as_micros() as u64,
                secure_offline_ms,
                trusted_debug_offline_ms,
                secure_offline_us,
                trusted_debug_offline_us,
                offline_bytes: pack.offline_bytes,
                online_bytes,
                oneflow_payload_bytes: online_bytes,
                pack_size_bytes: pack.pack_size_bytes,
                peak_pack_resident_bytes: pack.peak_pack_resident_bytes,
                streaming_pack: false,
            },
        })
    }

    pub fn dyadic_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        input: &RnsShareTensor,
        cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<RnsDyadicOfPmpeOutput, OperatorError> {
        let mut source = RnsTrustedDebugCorrelationSource::new(rng);
        self.dyadic_of_pmpe_profiled_with_source(input, cfg, &mut source)
    }

    pub fn dyadic_of_pmpe_profiled_with_source<S: RnsCorrelationSource + ?Sized>(
        &self,
        input: &RnsShareTensor,
        cfg: &DyadicOfPmpePolynomialConfig,
        source: &mut S,
    ) -> Result<RnsDyadicOfPmpeOutput, OperatorError> {
        input.check_domain(self)?;
        if cfg.streaming {
            return self.dyadic_of_pmpe_eval_streaming_with_source(input, cfg, source);
        }
        let mut pack = source.dyadic_of_pmpe_taylor_pack(self, input.len(), cfg)?;
        self.dyadic_of_pmpe_eval(input, &mut pack, cfg)
    }

    pub fn dyadic_of_pmpe_eval_streaming<R: RngCore + ?Sized>(
        &self,
        input: &RnsShareTensor,
        cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<RnsDyadicOfPmpeOutput, OperatorError> {
        let mut source = RnsTrustedDebugCorrelationSource::new(rng);
        self.dyadic_of_pmpe_eval_streaming_with_source(input, cfg, &mut source)
    }

    pub fn dyadic_of_pmpe_eval_streaming_with_source<S: RnsCorrelationSource + ?Sized>(
        &self,
        input: &RnsShareTensor,
        cfg: &DyadicOfPmpePolynomialConfig,
        source: &mut S,
    ) -> Result<RnsDyadicOfPmpeOutput, OperatorError> {
        input.check_domain(self)?;
        if input.is_empty() {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE streaming eval requires at least one input slot",
            ));
        }
        if cfg.chunk_slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE streaming chunk size must be positive",
            ));
        }
        if cfg.coeffs_num.is_empty() || cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE degree must match coefficient count",
            ));
        }
        let audit = self.no_wrap_audit(cfg)?;
        if !audit.ok && !cfg.allow_modular_wrap {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic OF-PMPE output bound exceeds CRT modulus",
            ));
        }

        let total_pack_size_bytes =
            rns_of_pmpe_pack_bytes(input.len(), cfg.degree, self.moduli.len());
        let mut peak_pack_resident_bytes = 0u64;
        let mut out0 = self
            .moduli
            .iter()
            .map(|_| Vec::with_capacity(input.len()))
            .collect::<Vec<_>>();
        let mut out1 = self
            .moduli
            .iter()
            .map(|_| Vec::with_capacity(input.len()))
            .collect::<Vec<_>>();
        let mut profile = OfPmpeProfile {
            offline_mode: source.offline_mode(),
            degree: cfg.degree,
            slots: input.len(),
            num_oneflow_phases: 1,
            public_linear_terms: cfg.coeffs_num.len(),
            num_fresh_masks: input.len(),
            pack_size_bytes: total_pack_size_bytes,
            streaming_pack: true,
            ..OfPmpeProfile::default()
        };

        for start_idx in (0..input.len()).step_by(cfg.chunk_slots) {
            let end_idx = usize::min(start_idx + cfg.chunk_slots, input.len());
            let chunk = input.slice(start_idx, end_idx)?;
            let output = {
                let mut pack = source.dyadic_of_pmpe_taylor_pack(self, chunk.len(), cfg)?;
                peak_pack_resident_bytes =
                    u64::max(peak_pack_resident_bytes, pack.peak_pack_resident_bytes);
                self.dyadic_of_pmpe_eval(&chunk, &mut pack, cfg)?
            };
            for (limb_idx, limb) in output.tensor.shares.limbs.iter().enumerate() {
                out0[limb_idx].extend_from_slice(limb.party0());
                out1[limb_idx].extend_from_slice(limb.party1());
            }
            profile.num_masked_opens += output.profile.num_masked_opens;
            profile.num_opened_elements += output.profile.num_opened_elements;
            profile.num_flushes += output.profile.num_flushes;
            profile.num_network_flushes += output.profile.num_network_flushes;
            profile.num_request_response_rounds += output.profile.num_request_response_rounds;
            profile.online_secret_secret_mul += output.profile.online_secret_secret_mul;
            profile.online_trunc += output.profile.online_trunc;
            profile.num_reused_masks += output.profile.num_reused_masks;
            profile.offline_ms += output.profile.offline_ms;
            profile.online_ms += output.profile.online_ms;
            profile.offline_us += output.profile.offline_us;
            profile.online_us += output.profile.online_us;
            profile.secure_offline_ms += output.profile.secure_offline_ms;
            profile.trusted_debug_offline_ms += output.profile.trusted_debug_offline_ms;
            profile.secure_offline_us += output.profile.secure_offline_us;
            profile.trusted_debug_offline_us += output.profile.trusted_debug_offline_us;
            profile.offline_bytes += output.profile.offline_bytes;
            profile.online_bytes += output.profile.online_bytes;
            profile.oneflow_payload_bytes += output.profile.oneflow_payload_bytes;
        }
        profile.peak_pack_resident_bytes = peak_pack_resident_bytes;

        let limbs = self
            .moduli
            .iter()
            .enumerate()
            .map(|(idx, &modulus)| {
                AdditiveShares::new(modulus, out0[idx].clone(), out1[idx].clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RnsDyadicOfPmpeOutput {
            tensor: RnsScaledShareTensor {
                shares: RnsShareTensor::from_limbs(limbs)?,
                frac_bits: cfg.output_frac_bits_with_guard()?,
                guard_bits: cfg.guard_bits,
                semantic: ScaleSemantic::DyadicNumerator,
            },
            audit,
            profile,
        })
    }

    pub fn approximate_rescale_pack<R: RngCore + ?Sized>(
        &self,
        slots: usize,
        shift_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
        rng: &mut R,
    ) -> Result<RnsDyadicRescalePack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale pack requires at least one slot",
            ));
        }
        if shift_bits == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale pack requires a positive shift",
            ));
        }
        if mask_bits == 0 || mask_bits > 126 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale mask_bits must be in 1..=126",
            ));
        }
        let opened_bound_bits = input_bound_bits.max(mask_bits).saturating_add(1);
        if opened_bound_bits > 126 {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale opened bound exceeds current i128 audit path",
            ));
        }
        if self.modulus_bits <= opened_bound_bits.saturating_add(1) {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale mask/input bound exceeds CRT modulus",
            ));
        }

        let masks_plain = (0..slots)
            .map(|_| sample_u128_bits(rng, mask_bits) as i128)
            .collect::<Vec<_>>();
        let quotients = masks_plain
            .iter()
            .map(|&mask| mask >> shift_bits)
            .collect::<Vec<_>>();
        let masks = self.share_with_rng(&masks_plain, rng)?;
        let mask_quotients = self.share_with_rng(&quotients, rng)?;
        let pack_size_bytes = (slots as u64)
            .saturating_mul(self.moduli.len() as u64)
            .saturating_mul(2)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>() as u64);
        let elapsed = start.elapsed();
        Ok(RnsDyadicRescalePack {
            masks,
            mask_quotients,
            shift_bits,
            input_bound_bits,
            mask_bits,
            offline_mode: OfPmpeOfflineMode::TrustedDebug,
            offline_ms: elapsed.as_millis() as u64,
            offline_us: elapsed.as_micros() as u64,
            offline_bytes: pack_size_bytes,
            pack_size_bytes,
            peak_pack_resident_bytes: pack_size_bytes,
            consumed: false,
        })
    }

    pub fn approximate_rescale_nonnegative_profiled<R: RngCore + ?Sized>(
        &self,
        input: &RnsScaledShareTensor,
        target_frac_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
        rng: &mut R,
    ) -> Result<RnsDyadicRescaleOutput, OperatorError> {
        let mut source = RnsTrustedDebugCorrelationSource::new(rng);
        self.approximate_rescale_nonnegative_profiled_with_source(
            input,
            target_frac_bits,
            input_bound_bits,
            mask_bits,
            &mut source,
        )
    }

    pub fn approximate_rescale_nonnegative_profiled_with_source<
        S: RnsCorrelationSource + ?Sized,
    >(
        &self,
        input: &RnsScaledShareTensor,
        target_frac_bits: u32,
        input_bound_bits: u32,
        mask_bits: u32,
        source: &mut S,
    ) -> Result<RnsDyadicRescaleOutput, OperatorError> {
        if target_frac_bits > input.frac_bits {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale target frac_bits must not exceed input frac_bits",
            ));
        }
        let shift_bits = input.frac_bits - target_frac_bits;
        if shift_bits == 0 {
            return Ok(RnsDyadicRescaleOutput {
                tensor: input.clone(),
                shift_bits,
                profile: OfPmpeProfile::default(),
            });
        }
        let mut pack = source.approximate_rescale_pack(
            self,
            input.shares.len(),
            shift_bits,
            input_bound_bits,
            mask_bits,
        )?;
        self.approximate_rescale_eval(input, &mut pack, target_frac_bits)
    }

    pub fn approximate_rescale_eval(
        &self,
        input: &RnsScaledShareTensor,
        pack: &mut RnsDyadicRescalePack,
        target_frac_bits: u32,
    ) -> Result<RnsDyadicRescaleOutput, OperatorError> {
        let start = Instant::now();
        if pack.consumed {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale pack has already been consumed",
            ));
        }
        pack.consumed = true;
        if input.semantic != ScaleSemantic::DyadicNumerator {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale requires dyadic numerator tensors",
            ));
        }
        if target_frac_bits > input.frac_bits {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale target frac_bits must not exceed input frac_bits",
            ));
        }
        let shift_bits = input.frac_bits - target_frac_bits;
        if shift_bits != pack.shift_bits {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale pack shift does not match target frac_bits",
            ));
        }
        input.shares.check_domain(self)?;
        pack.masks.check_domain(self)?;
        pack.mask_quotients.check_domain(self)?;
        if input.shares.len() != pack.masks.len() || input.shares.len() != pack.mask_quotients.len()
        {
            return Err(OperatorError::InvalidParams(
                "RNS dyadic rescale input and pack lengths must match",
            ));
        }
        let opened_bound_bits = pack.input_bound_bits.max(pack.mask_bits).saturating_add(1);
        let opened = input
            .shares
            .add(&pack.masks)?
            .reconstruct_nonnegative_i128_bounded(opened_bound_bits)?;
        let quotients = opened
            .iter()
            .map(|&value| value >> shift_bits)
            .collect::<Vec<_>>();
        let public_quotients = pack
            .mask_quotients
            .mul_public_scalar(0)?
            .add_public_i128_slots_to_party0(&quotients)?;
        let shares = public_quotients.sub(&pack.mask_quotients)?;
        let elapsed = start.elapsed();
        let online_bytes = input
            .shares
            .len()
            .saturating_mul(self.moduli.len())
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>()) as u64;
        let (
            secure_offline_ms,
            trusted_debug_offline_ms,
            secure_offline_us,
            trusted_debug_offline_us,
        ) = split_offline_accounting(pack.offline_mode, pack.offline_ms, pack.offline_us);
        Ok(RnsDyadicRescaleOutput {
            tensor: RnsScaledShareTensor {
                shares,
                frac_bits: target_frac_bits,
                guard_bits: input.guard_bits.saturating_sub(shift_bits),
                semantic: input.semantic,
            },
            shift_bits,
            profile: OfPmpeProfile {
                offline_mode: pack.offline_mode,
                degree: 0,
                slots: input.shares.len(),
                num_oneflow_phases: 1,
                num_request_response_rounds: 0,
                num_masked_opens: input.shares.len(),
                num_opened_elements: input.shares.len(),
                num_flushes: 1,
                num_network_flushes: 1,
                public_linear_terms: 2,
                online_secret_secret_mul: 0,
                online_trunc: input.shares.len(),
                num_fresh_masks: input.shares.len(),
                num_reused_masks: 0,
                offline_ms: pack.offline_ms,
                online_ms: elapsed.as_millis() as u64,
                offline_us: pack.offline_us,
                online_us: elapsed.as_micros() as u64,
                secure_offline_ms,
                trusted_debug_offline_ms,
                secure_offline_us,
                trusted_debug_offline_us,
                offline_bytes: pack.offline_bytes,
                online_bytes,
                oneflow_payload_bytes: online_bytes,
                pack_size_bytes: pack.pack_size_bytes,
                peak_pack_resident_bytes: pack.peak_pack_resident_bytes,
                streaming_pack: false,
            },
        })
    }

    pub fn beaver_mul_pack<R: RngCore + ?Sized>(
        &self,
        slots: usize,
        rng: &mut R,
    ) -> Result<RnsBeaverMulPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS Beaver multiplication pack requires at least one slot",
            ));
        }
        let mut lhs_limbs = Vec::with_capacity(self.moduli.len());
        let mut rhs_limbs = Vec::with_capacity(self.moduli.len());
        let mut product_limbs = Vec::with_capacity(self.moduli.len());
        for &modulus in &self.moduli {
            let lhs = (0..slots)
                .map(|_| rng.next_u64() % modulus)
                .collect::<Vec<_>>();
            let rhs = (0..slots)
                .map(|_| rng.next_u64() % modulus)
                .collect::<Vec<_>>();
            let product = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| mul_mod(a, b, modulus))
                .collect::<Vec<_>>();
            lhs_limbs.push(AdditiveShares::share_with_rng(&lhs, modulus, rng)?);
            rhs_limbs.push(AdditiveShares::share_with_rng(&rhs, modulus, rng)?);
            product_limbs.push(AdditiveShares::share_with_rng(&product, modulus, rng)?);
        }
        let pack_size_bytes = (slots as u64)
            .saturating_mul(self.moduli.len() as u64)
            .saturating_mul(3)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>() as u64);
        let elapsed = start.elapsed();
        Ok(RnsBeaverMulPack {
            lhs_masks: RnsShareTensor::from_limbs(lhs_limbs)?,
            rhs_masks: RnsShareTensor::from_limbs(rhs_limbs)?,
            product_masks: RnsShareTensor::from_limbs(product_limbs)?,
            offline_mode: OfPmpeOfflineMode::TrustedDebug,
            offline_ms: elapsed.as_millis() as u64,
            offline_us: elapsed.as_micros() as u64,
            offline_bytes: pack_size_bytes,
            pack_size_bytes,
            peak_pack_resident_bytes: pack_size_bytes,
            consumed: false,
        })
    }

    pub fn mul_scaled_with_beaver_profiled(
        &self,
        lhs: &RnsScaledShareTensor,
        rhs: &RnsScaledShareTensor,
        pack: &mut RnsBeaverMulPack,
    ) -> Result<RnsScaledMulOutput, OperatorError> {
        let start = Instant::now();
        if pack.consumed {
            return Err(OperatorError::InvalidParams(
                "RNS Beaver multiplication pack has already been consumed",
            ));
        }
        pack.consumed = true;
        if lhs.semantic != ScaleSemantic::DyadicNumerator
            || rhs.semantic != ScaleSemantic::DyadicNumerator
        {
            return Err(OperatorError::InvalidParams(
                "RNS scaled multiply requires dyadic numerator tensors",
            ));
        }
        lhs.shares.check_domain(self)?;
        rhs.shares.check_domain(self)?;
        pack.lhs_masks.check_domain(self)?;
        pack.rhs_masks.check_domain(self)?;
        pack.product_masks.check_domain(self)?;
        if lhs.shares.len() != rhs.shares.len() || lhs.shares.len() != pack.lhs_masks.len() {
            return Err(OperatorError::InvalidParams(
                "RNS scaled multiply inputs and pack must have equal length",
            ));
        }
        let product_frac_bits =
            lhs.frac_bits
                .checked_add(rhs.frac_bits)
                .ok_or(OperatorError::InvalidParams(
                    "RNS scaled multiply product frac_bits overflow",
                ))?;
        let lhs_delta = lhs.shares.sub(&pack.lhs_masks)?;
        let rhs_delta = rhs.shares.sub(&pack.rhs_masks)?;
        let lhs_delta_public = lhs_delta.reconstruct_residues();
        let rhs_delta_public = rhs_delta.reconstruct_residues();
        let slots = lhs.shares.len();
        let mut out_limbs = Vec::with_capacity(self.moduli.len());

        for (limb_idx, &modulus) in self.moduli.iter().enumerate() {
            let a = &pack.lhs_masks.limbs[limb_idx];
            let b = &pack.rhs_masks.limbs[limb_idx];
            let c = &pack.product_masks.limbs[limb_idx];
            let mut out0 = c.party0().to_vec();
            let mut out1 = c.party1().to_vec();
            for slot in 0..slots {
                let d = lhs_delta_public[limb_idx][slot];
                let e = rhs_delta_public[limb_idx][slot];
                out0[slot] = add_mod(out0[slot], mul_mod(d, b.party0()[slot], modulus), modulus);
                out1[slot] = add_mod(out1[slot], mul_mod(d, b.party1()[slot], modulus), modulus);
                out0[slot] = add_mod(out0[slot], mul_mod(e, a.party0()[slot], modulus), modulus);
                out1[slot] = add_mod(out1[slot], mul_mod(e, a.party1()[slot], modulus), modulus);
                out0[slot] = add_mod(out0[slot], mul_mod(d, e, modulus), modulus);
            }
            out_limbs.push(AdditiveShares::new(modulus, out0, out1)?);
        }

        let elapsed = start.elapsed();
        let online_bytes = slots
            .saturating_mul(self.moduli.len())
            .saturating_mul(2)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>()) as u64;
        let (
            secure_offline_ms,
            trusted_debug_offline_ms,
            secure_offline_us,
            trusted_debug_offline_us,
        ) = split_offline_accounting(pack.offline_mode, pack.offline_ms, pack.offline_us);
        Ok(RnsScaledMulOutput {
            tensor: RnsScaledShareTensor {
                shares: RnsShareTensor::from_limbs(out_limbs)?,
                frac_bits: product_frac_bits,
                guard_bits: lhs.guard_bits.saturating_add(rhs.guard_bits),
                semantic: ScaleSemantic::DyadicNumerator,
            },
            product_frac_bits,
            profile: OfPmpeProfile {
                offline_mode: pack.offline_mode,
                degree: 1,
                slots,
                num_oneflow_phases: 1,
                num_request_response_rounds: 0,
                num_masked_opens: slots.saturating_mul(2),
                num_opened_elements: slots.saturating_mul(2),
                num_flushes: 1,
                num_network_flushes: 1,
                public_linear_terms: 4,
                online_secret_secret_mul: 0,
                online_trunc: 0,
                num_fresh_masks: slots.saturating_mul(2),
                num_reused_masks: 0,
                offline_ms: pack.offline_ms,
                online_ms: elapsed.as_millis() as u64,
                offline_us: pack.offline_us,
                online_us: elapsed.as_micros() as u64,
                secure_offline_ms,
                trusted_debug_offline_ms,
                secure_offline_us,
                trusted_debug_offline_us,
                offline_bytes: pack.offline_bytes,
                online_bytes,
                oneflow_payload_bytes: online_bytes,
                pack_size_bytes: pack.pack_size_bytes,
                peak_pack_resident_bytes: pack.peak_pack_resident_bytes,
                streaming_pack: false,
            },
        })
    }

    pub fn mul_scaled_with_beaver_streaming_profiled<R: RngCore + ?Sized>(
        &self,
        lhs: &RnsScaledShareTensor,
        rhs: &RnsScaledShareTensor,
        chunk_slots: usize,
        rng: &mut R,
    ) -> Result<RnsScaledMulOutput, OperatorError> {
        let mut source = RnsTrustedDebugCorrelationSource::new(rng);
        self.mul_scaled_with_beaver_streaming_profiled_with_source(
            lhs,
            rhs,
            chunk_slots,
            &mut source,
        )
    }

    pub fn mul_scaled_with_beaver_streaming_profiled_with_source<
        S: RnsCorrelationSource + ?Sized,
    >(
        &self,
        lhs: &RnsScaledShareTensor,
        rhs: &RnsScaledShareTensor,
        chunk_slots: usize,
        source: &mut S,
    ) -> Result<RnsScaledMulOutput, OperatorError> {
        lhs.shares.check_domain(self)?;
        rhs.shares.check_domain(self)?;
        if chunk_slots == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS Beaver streaming chunk size must be positive",
            ));
        }
        if lhs.shares.len() != rhs.shares.len() {
            return Err(OperatorError::InvalidParams(
                "RNS Beaver streaming inputs must have equal length",
            ));
        }
        let product_frac_bits =
            lhs.frac_bits
                .checked_add(rhs.frac_bits)
                .ok_or(OperatorError::InvalidParams(
                    "RNS Beaver streaming product frac_bits overflow",
                ))?;
        let slots = lhs.shares.len();
        let total_pack_size_bytes = (slots as u64)
            .saturating_mul(self.moduli.len() as u64)
            .saturating_mul(3)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>() as u64);
        let mut peak_pack_resident_bytes = 0u64;
        let mut out0 = self
            .moduli
            .iter()
            .map(|_| Vec::with_capacity(slots))
            .collect::<Vec<_>>();
        let mut out1 = self
            .moduli
            .iter()
            .map(|_| Vec::with_capacity(slots))
            .collect::<Vec<_>>();
        let mut profile = OfPmpeProfile {
            offline_mode: source.offline_mode(),
            degree: 1,
            slots,
            num_oneflow_phases: 1,
            public_linear_terms: 4,
            num_fresh_masks: slots.saturating_mul(2),
            pack_size_bytes: total_pack_size_bytes,
            streaming_pack: true,
            ..OfPmpeProfile::default()
        };

        for start_idx in (0..slots).step_by(chunk_slots) {
            let end_idx = usize::min(start_idx + chunk_slots, slots);
            let lhs_chunk = RnsScaledShareTensor {
                shares: lhs.shares.slice(start_idx, end_idx)?,
                frac_bits: lhs.frac_bits,
                guard_bits: lhs.guard_bits,
                semantic: lhs.semantic,
            };
            let rhs_chunk = RnsScaledShareTensor {
                shares: rhs.shares.slice(start_idx, end_idx)?,
                frac_bits: rhs.frac_bits,
                guard_bits: rhs.guard_bits,
                semantic: rhs.semantic,
            };
            let output = {
                let mut pack = source.beaver_mul_pack(self, lhs_chunk.shares.len())?;
                peak_pack_resident_bytes =
                    u64::max(peak_pack_resident_bytes, pack.peak_pack_resident_bytes);
                self.mul_scaled_with_beaver_profiled(&lhs_chunk, &rhs_chunk, &mut pack)?
            };
            for (limb_idx, limb) in output.tensor.shares.limbs.iter().enumerate() {
                out0[limb_idx].extend_from_slice(limb.party0());
                out1[limb_idx].extend_from_slice(limb.party1());
            }
            profile.num_masked_opens += output.profile.num_masked_opens;
            profile.num_opened_elements += output.profile.num_opened_elements;
            profile.num_flushes += output.profile.num_flushes;
            profile.num_network_flushes += output.profile.num_network_flushes;
            profile.num_request_response_rounds += output.profile.num_request_response_rounds;
            profile.online_secret_secret_mul += output.profile.online_secret_secret_mul;
            profile.online_trunc += output.profile.online_trunc;
            profile.num_reused_masks += output.profile.num_reused_masks;
            profile.offline_ms += output.profile.offline_ms;
            profile.online_ms += output.profile.online_ms;
            profile.offline_us += output.profile.offline_us;
            profile.online_us += output.profile.online_us;
            profile.secure_offline_ms += output.profile.secure_offline_ms;
            profile.trusted_debug_offline_ms += output.profile.trusted_debug_offline_ms;
            profile.secure_offline_us += output.profile.secure_offline_us;
            profile.trusted_debug_offline_us += output.profile.trusted_debug_offline_us;
            profile.offline_bytes += output.profile.offline_bytes;
            profile.online_bytes += output.profile.online_bytes;
            profile.oneflow_payload_bytes += output.profile.oneflow_payload_bytes;
        }
        profile.peak_pack_resident_bytes = peak_pack_resident_bytes;

        let limbs = self
            .moduli
            .iter()
            .enumerate()
            .map(|(idx, &modulus)| {
                AdditiveShares::new(modulus, out0[idx].clone(), out1[idx].clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RnsScaledMulOutput {
            tensor: RnsScaledShareTensor {
                shares: RnsShareTensor::from_limbs(limbs)?,
                frac_bits: product_frac_bits,
                guard_bits: lhs.guard_bits.saturating_add(rhs.guard_bits),
                semantic: ScaleSemantic::DyadicNumerator,
            },
            product_frac_bits,
            profile,
        })
    }

    pub fn softmax_mean_shift_of_pmpe_exp_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &RnsShareTensor,
        rows: usize,
        cols: usize,
        score_frac_bits: u32,
        beta_shift: f64,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<RnsMeanShiftSoftmaxExpOutput, OperatorError> {
        let mut source = RnsTrustedDebugCorrelationSource::new(rng);
        self.softmax_mean_shift_of_pmpe_exp_profiled_with_source(
            row_major,
            rows,
            cols,
            score_frac_bits,
            beta_shift,
            exp_cfg,
            &mut source,
        )
    }

    pub fn softmax_mean_shift_of_pmpe_exp_profiled_with_source<S: RnsCorrelationSource + ?Sized>(
        &self,
        row_major: &RnsShareTensor,
        rows: usize,
        cols: usize,
        score_frac_bits: u32,
        beta_shift: f64,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        source: &mut S,
    ) -> Result<RnsMeanShiftSoftmaxExpOutput, OperatorError> {
        row_major.check_domain(self)?;
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "RNS mean-shift softmax dimensions must be non-empty",
            ));
        }
        if !cols.is_power_of_two() {
            return Err(OperatorError::InvalidParams(
                "RNS mean-shift softmax currently requires power-of-two row length",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "RNS mean-shift softmax length must equal rows * cols",
            ));
        }
        let shifted_frac_bits = score_frac_bits.checked_add(cols.trailing_zeros()).ok_or(
            OperatorError::InvalidParams("RNS mean-shift softmax frac_bits overflow"),
        )?;
        if exp_cfg.input_frac_bits != shifted_frac_bits {
            return Err(OperatorError::InvalidParams(
                "RNS mean-shift exp config input_frac_bits must equal score_frac_bits + log2(cols)",
            ));
        }

        let mean_shift_start = Instant::now();
        let row_sums = row_major.sum_rows(rows, cols)?;
        let row_sums_repeated = row_sums.repeat_rows(cols)?;
        let scaled_scores = row_major.mul_public_scalar(cols as u64)?;
        let beta_scaled = round_scaled_coeff_to_i128(beta_shift, shifted_frac_bits as i64)?;
        let shifted_shares = scaled_scores
            .sub(&row_sums_repeated)?
            .add_public_i128_to_party0(-beta_scaled)?;
        let mean_shift_elapsed = mean_shift_start.elapsed();
        let shifted = RnsScaledShareTensor {
            shares: shifted_shares,
            frac_bits: shifted_frac_bits,
            guard_bits: 0,
            semantic: ScaleSemantic::DyadicNumerator,
        };

        let exp = self.dyadic_of_pmpe_profiled_with_source(&shifted.shares, exp_cfg, source)?;
        let sum_start = Instant::now();
        let row_sum_shares = exp.tensor.shares.sum_rows(rows, cols)?;
        let sum_elapsed = sum_start.elapsed();
        let online_us = (mean_shift_elapsed.as_micros() as u64)
            .saturating_add(exp.profile.online_us)
            .saturating_add(sum_elapsed.as_micros() as u64);

        Ok(RnsMeanShiftSoftmaxExpOutput {
            row_sums: RnsScaledShareTensor {
                shares: row_sum_shares,
                frac_bits: exp.tensor.frac_bits,
                guard_bits: exp.tensor.guard_bits,
                semantic: exp.tensor.semantic,
            },
            shifted,
            exp: exp.tensor,
            exp_profile: exp.profile,
            exp_audit: exp.audit,
            mean_shift_ms: mean_shift_elapsed.as_millis() as u64,
            mean_shift_us: mean_shift_elapsed.as_micros() as u64,
            sum_ms: sum_elapsed.as_millis() as u64,
            sum_us: sum_elapsed.as_micros() as u64,
            online_ms: online_us / 1000,
            online_us,
        })
    }

    pub fn softmax_mean_shift_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &RnsShareTensor,
        rows: usize,
        cols: usize,
        score_frac_bits: u32,
        beta_shift: f64,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        reciprocal_cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<RnsMeanShiftSoftmaxOutput, OperatorError> {
        let mut source = RnsTrustedDebugCorrelationSource::new(rng);
        self.softmax_mean_shift_of_pmpe_profiled_with_source(
            row_major,
            rows,
            cols,
            score_frac_bits,
            beta_shift,
            exp_cfg,
            reciprocal_cfg,
            &mut source,
        )
    }

    pub fn softmax_mean_shift_of_pmpe_profiled_with_source<S: RnsCorrelationSource + ?Sized>(
        &self,
        row_major: &RnsShareTensor,
        rows: usize,
        cols: usize,
        score_frac_bits: u32,
        beta_shift: f64,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        reciprocal_cfg: &DyadicOfPmpePolynomialConfig,
        source: &mut S,
    ) -> Result<RnsMeanShiftSoftmaxOutput, OperatorError> {
        let exp_phase = self.softmax_mean_shift_of_pmpe_exp_profiled_with_source(
            row_major,
            rows,
            cols,
            score_frac_bits,
            beta_shift,
            exp_cfg,
            source,
        )?;
        if reciprocal_cfg.input_frac_bits != exp_phase.row_sums.frac_bits {
            return Err(OperatorError::InvalidParams(
                "RNS mean-shift reciprocal input_frac_bits must match exp row-sum frac_bits",
            ));
        }

        let reciprocal = self.dyadic_of_pmpe_profiled_with_source(
            &exp_phase.row_sums.shares,
            reciprocal_cfg,
            source,
        )?;
        let repeat_start = Instant::now();
        let repeated_inv = RnsScaledShareTensor {
            shares: reciprocal.tensor.shares.repeat_rows(cols)?,
            frac_bits: reciprocal.tensor.frac_bits,
            guard_bits: reciprocal.tensor.guard_bits,
            semantic: reciprocal.tensor.semantic,
        };
        let repeat_elapsed = repeat_start.elapsed();
        let mul_slots = rows.saturating_mul(cols);
        let probs = if exp_cfg.streaming {
            self.mul_scaled_with_beaver_streaming_profiled_with_source(
                &exp_phase.exp,
                &repeated_inv,
                exp_cfg.chunk_slots,
                source,
            )?
        } else {
            let mut mul_pack = source.beaver_mul_pack(self, mul_slots)?;
            self.mul_scaled_with_beaver_profiled(&exp_phase.exp, &repeated_inv, &mut mul_pack)?
        };
        let online_us = exp_phase
            .online_us
            .saturating_add(reciprocal.profile.online_us)
            .saturating_add(repeat_elapsed.as_micros() as u64)
            .saturating_add(probs.profile.online_us);

        Ok(RnsMeanShiftSoftmaxOutput {
            exp_phase,
            row_sums_base: None,
            inv_row_sums: reciprocal.tensor,
            inv_row_sums_repeated: repeated_inv,
            probs: probs.tensor,
            rescale_profile: None,
            reciprocal_profile: reciprocal.profile,
            broadcast_mul_profile: probs.profile,
            reciprocal_audit: reciprocal.audit,
            repeat_ms: repeat_elapsed.as_millis() as u64,
            repeat_us: repeat_elapsed.as_micros() as u64,
            online_ms: online_us / 1000,
            online_us,
        })
    }

    pub fn softmax_mean_shift_of_pmpe_rescaled_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &RnsShareTensor,
        rows: usize,
        cols: usize,
        score_frac_bits: u32,
        beta_shift: f64,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        row_sum_target_frac_bits: u32,
        row_sum_bound_bits: u32,
        rescale_mask_bits: u32,
        reciprocal_cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<RnsMeanShiftSoftmaxOutput, OperatorError> {
        let mut source = RnsTrustedDebugCorrelationSource::new(rng);
        self.softmax_mean_shift_of_pmpe_rescaled_profiled_with_source(
            row_major,
            rows,
            cols,
            score_frac_bits,
            beta_shift,
            exp_cfg,
            row_sum_target_frac_bits,
            row_sum_bound_bits,
            rescale_mask_bits,
            reciprocal_cfg,
            &mut source,
        )
    }

    pub fn softmax_mean_shift_of_pmpe_rescaled_profiled_with_source<
        S: RnsCorrelationSource + ?Sized,
    >(
        &self,
        row_major: &RnsShareTensor,
        rows: usize,
        cols: usize,
        score_frac_bits: u32,
        beta_shift: f64,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        row_sum_target_frac_bits: u32,
        row_sum_bound_bits: u32,
        rescale_mask_bits: u32,
        reciprocal_cfg: &DyadicOfPmpePolynomialConfig,
        source: &mut S,
    ) -> Result<RnsMeanShiftSoftmaxOutput, OperatorError> {
        let exp_phase = self.softmax_mean_shift_of_pmpe_exp_profiled_with_source(
            row_major,
            rows,
            cols,
            score_frac_bits,
            beta_shift,
            exp_cfg,
            source,
        )?;
        if reciprocal_cfg.input_frac_bits != row_sum_target_frac_bits {
            return Err(OperatorError::InvalidParams(
                "RNS mean-shift reciprocal input_frac_bits must match rescaled row-sum frac_bits",
            ));
        }
        let rescaled_sum = self.approximate_rescale_nonnegative_profiled_with_source(
            &exp_phase.row_sums,
            row_sum_target_frac_bits,
            row_sum_bound_bits,
            rescale_mask_bits,
            source,
        )?;
        let reciprocal = self.dyadic_of_pmpe_profiled_with_source(
            &rescaled_sum.tensor.shares,
            reciprocal_cfg,
            source,
        )?;
        let repeat_start = Instant::now();
        let repeated_inv = RnsScaledShareTensor {
            shares: reciprocal.tensor.shares.repeat_rows(cols)?,
            frac_bits: reciprocal.tensor.frac_bits,
            guard_bits: reciprocal.tensor.guard_bits,
            semantic: reciprocal.tensor.semantic,
        };
        let repeat_elapsed = repeat_start.elapsed();
        let mul_slots = rows.saturating_mul(cols);
        let probs = if exp_cfg.streaming {
            self.mul_scaled_with_beaver_streaming_profiled_with_source(
                &exp_phase.exp,
                &repeated_inv,
                exp_cfg.chunk_slots,
                source,
            )?
        } else {
            let mut mul_pack = source.beaver_mul_pack(self, mul_slots)?;
            self.mul_scaled_with_beaver_profiled(&exp_phase.exp, &repeated_inv, &mut mul_pack)?
        };
        let online_us = exp_phase
            .online_us
            .saturating_add(rescaled_sum.profile.online_us)
            .saturating_add(reciprocal.profile.online_us)
            .saturating_add(repeat_elapsed.as_micros() as u64)
            .saturating_add(probs.profile.online_us);

        Ok(RnsMeanShiftSoftmaxOutput {
            exp_phase,
            row_sums_base: Some(rescaled_sum.tensor),
            inv_row_sums: reciprocal.tensor,
            inv_row_sums_repeated: repeated_inv,
            probs: probs.tensor,
            rescale_profile: Some(rescaled_sum.profile),
            reciprocal_profile: reciprocal.profile,
            broadcast_mul_profile: probs.profile,
            reciprocal_audit: reciprocal.audit,
            repeat_ms: repeat_elapsed.as_millis() as u64,
            repeat_us: repeat_elapsed.as_micros() as u64,
            online_ms: online_us / 1000,
            online_us,
        })
    }
}

impl RnsShareTensor {
    pub fn from_limbs(limbs: Vec<AdditiveShares>) -> Result<Self, OperatorError> {
        if limbs.is_empty() {
            return Err(OperatorError::InvalidParams(
                "RNS share tensor requires at least one limb",
            ));
        }
        let len = limbs[0].len();
        if limbs.iter().any(|limb| limb.len() != len) {
            return Err(OperatorError::InvalidParams(
                "RNS share limbs must have matching lengths",
            ));
        }
        Ok(Self { limbs })
    }

    pub fn share_with_rng<R: RngCore + ?Sized>(
        domain: &RnsDyadicDomain,
        values: &[i128],
        rng: &mut R,
    ) -> Result<Self, OperatorError> {
        let limbs = domain
            .moduli
            .iter()
            .map(|&modulus| {
                let residues = values
                    .iter()
                    .map(|&value| reduce_i128_mod(value, modulus))
                    .collect::<Vec<_>>();
                AdditiveShares::share_with_rng(&residues, modulus, rng)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_limbs(limbs)
    }

    pub fn len(&self) -> usize {
        self.limbs[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slice(&self, start: usize, end: usize) -> Result<Self, OperatorError> {
        if start > end || end > self.len() {
            return Err(OperatorError::InvalidParams(
                "RNS share slice indices are out of bounds",
            ));
        }
        self.limbs
            .iter()
            .map(|limb| {
                AdditiveShares::new(
                    limb.modulus(),
                    limb.party0()[start..end].to_vec(),
                    limb.party1()[start..end].to_vec(),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .and_then(Self::from_limbs)
    }

    pub fn limbs(&self) -> &[AdditiveShares] {
        &self.limbs
    }

    pub fn moduli(&self) -> Vec<u64> {
        self.limbs.iter().map(AdditiveShares::modulus).collect()
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, OperatorError> {
        self.check_compatible(rhs)?;
        self.limbs
            .iter()
            .zip(rhs.limbs.iter())
            .map(|(lhs, rhs)| lhs.sub(rhs))
            .collect::<Result<Vec<_>, _>>()
            .and_then(Self::from_limbs)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, OperatorError> {
        self.check_compatible(rhs)?;
        self.limbs
            .iter()
            .zip(rhs.limbs.iter())
            .map(|(lhs, rhs)| lhs.add(rhs))
            .collect::<Result<Vec<_>, _>>()
            .and_then(Self::from_limbs)
    }

    pub fn mul_public_scalar(&self, scalar: u64) -> Result<Self, OperatorError> {
        self.limbs
            .iter()
            .map(|limb| limb.mul_public_scalar(scalar % limb.modulus()))
            .collect::<Result<Vec<_>, _>>()
            .and_then(Self::from_limbs)
    }

    pub fn add_public_i128_to_party0(&self, value: i128) -> Result<Self, OperatorError> {
        self.limbs
            .iter()
            .map(|limb| {
                let addend = reduce_i128_mod(value, limb.modulus());
                limb.add_public_to_party0(&vec![addend; limb.len()])
            })
            .collect::<Result<Vec<_>, _>>()
            .and_then(Self::from_limbs)
    }

    pub fn add_public_i128_slots_to_party0(&self, values: &[i128]) -> Result<Self, OperatorError> {
        if values.len() != self.len() {
            return Err(OperatorError::InvalidParams(
                "RNS public addend length must match share length",
            ));
        }
        self.limbs
            .iter()
            .map(|limb| {
                let addends = values
                    .iter()
                    .map(|&value| reduce_i128_mod(value, limb.modulus()))
                    .collect::<Vec<_>>();
                limb.add_public_to_party0(&addends)
            })
            .collect::<Result<Vec<_>, _>>()
            .and_then(Self::from_limbs)
    }

    pub fn sum_rows(&self, rows: usize, cols: usize) -> Result<Self, OperatorError> {
        self.limbs
            .iter()
            .map(|limb| sum_rows(limb, rows, cols))
            .collect::<Result<Vec<_>, _>>()
            .and_then(Self::from_limbs)
    }

    pub fn repeat_rows(&self, cols: usize) -> Result<Self, OperatorError> {
        self.limbs
            .iter()
            .map(|limb| repeat_rows(limb, cols))
            .collect::<Result<Vec<_>, _>>()
            .and_then(Self::from_limbs)
    }

    pub fn reconstruct_residues(&self) -> Vec<Vec<u64>> {
        self.limbs
            .iter()
            .map(AdditiveShares::reconstruct)
            .collect::<Vec<_>>()
    }

    pub fn reconstruct_centered_i128(&self) -> Result<Vec<i128>, OperatorError> {
        if self.limbs.len() != 2 {
            return Err(OperatorError::InvalidParams(
                "centered RNS reconstruction currently requires two limbs",
            ));
        }
        let p = self.limbs[0].modulus();
        let q = self.limbs[1].modulus();
        let modulus = (p as u128)
            .checked_mul(q as u128)
            .ok_or(OperatorError::InvalidParams("CRT modulus overflow"))?;
        if modulus > i128::MAX as u128 {
            return Err(OperatorError::InvalidParams(
                "CRT modulus must fit signed i128 reconstruction",
            ));
        }
        let inv_p_mod_q = mod_inverse_u64(p % q, q).ok_or(OperatorError::InvalidParams(
            "RNS moduli are not coprime for CRT reconstruction",
        ))?;
        let lhs = self.limbs[0].reconstruct();
        let rhs = self.limbs[1].reconstruct();
        let half = modulus / 2;
        let mut out = Vec::with_capacity(self.len());
        for (&a, &b) in lhs.iter().zip(rhs.iter()) {
            let a_mod_q = a % q;
            let diff = if b >= a_mod_q {
                b - a_mod_q
            } else {
                q - (a_mod_q - b)
            };
            let t = mul_mod(diff, inv_p_mod_q, q);
            let value = a as u128 + (p as u128) * (t as u128);
            let centered = if value > half {
                value as i128 - modulus as i128
            } else {
                value as i128
            };
            out.push(centered);
        }
        Ok(out)
    }

    pub fn reconstruct_nonnegative_i128_bounded(
        &self,
        bound_bits: u32,
    ) -> Result<Vec<i128>, OperatorError> {
        if bound_bits > 126 {
            return Err(OperatorError::InvalidParams(
                "RNS nonnegative reconstruction bound must fit i128",
            ));
        }
        let residues = self.reconstruct_residues();
        let mut out = residues[0]
            .iter()
            .map(|&value| value as u128)
            .collect::<Vec<_>>();
        let mut modulus_so_far = self.limbs[0].modulus() as u128;
        let target_unique = bound_bits.saturating_add(1);
        let mut limb_idx = 1usize;

        while limb_idx < self.limbs.len() && u128_bit_width(modulus_so_far) <= target_unique {
            let modulus = self.limbs[limb_idx].modulus();
            let modulus_so_far_mod = (modulus_so_far % modulus as u128) as u64;
            let inv = mod_inverse_u64(modulus_so_far_mod, modulus).ok_or(
                OperatorError::InvalidParams("RNS moduli are not coprime for reconstruction"),
            )?;
            for (slot, value) in out.iter_mut().enumerate() {
                let current = (*value % modulus as u128) as u64;
                let target = residues[limb_idx][slot];
                let diff = if target >= current {
                    target - current
                } else {
                    modulus - (current - target)
                };
                let t = mul_mod(diff, inv, modulus);
                let addend =
                    modulus_so_far
                        .checked_mul(t as u128)
                        .ok_or(OperatorError::InvalidParams(
                            "RNS reconstruction exceeds u128",
                        ))?;
                *value = value
                    .checked_add(addend)
                    .ok_or(OperatorError::InvalidParams(
                        "RNS reconstruction exceeds u128",
                    ))?;
                if *value > i128::MAX as u128 {
                    return Err(OperatorError::InvalidParams(
                        "RNS reconstruction exceeds i128",
                    ));
                }
            }
            limb_idx += 1;
            let next_modulus_bits =
                u128_bit_width(modulus_so_far).saturating_add(modulus_bit_width(modulus));
            if limb_idx < self.limbs.len() && next_modulus_bits <= target_unique {
                modulus_so_far = modulus_so_far.checked_mul(modulus as u128).ok_or(
                    OperatorError::InvalidParams("RNS reconstruction modulus exceeds u128"),
                )?;
            } else {
                break;
            }
        }

        let max_value = if bound_bits == 0 {
            0u128
        } else {
            (1u128 << bound_bits) - 1
        };
        for (slot, &value) in out.iter().enumerate() {
            if value > max_value {
                return Err(OperatorError::InvalidParams(
                    "RNS reconstructed value exceeds declared bound",
                ));
            }
            for (extra_limb, limb_residues) in residues.iter().enumerate().skip(limb_idx) {
                let modulus = self.limbs[extra_limb].modulus();
                if (value % modulus as u128) as u64 != limb_residues[slot] {
                    return Err(OperatorError::InvalidParams(
                        "RNS reconstructed value failed remaining-limb verification",
                    ));
                }
            }
        }

        Ok(out.into_iter().map(|value| value as i128).collect())
    }

    fn check_domain(&self, domain: &RnsDyadicDomain) -> Result<(), OperatorError> {
        if self.limbs.len() != domain.moduli.len()
            || self
                .limbs
                .iter()
                .zip(domain.moduli.iter())
                .any(|(limb, &modulus)| limb.modulus() != modulus)
        {
            return Err(OperatorError::InvalidParams(
                "RNS share tensor domain mismatch",
            ));
        }
        Ok(())
    }

    fn check_compatible(&self, rhs: &Self) -> Result<(), OperatorError> {
        if self.moduli() != rhs.moduli() || self.len() != rhs.len() {
            return Err(OperatorError::InvalidParams(
                "RNS share tensors must have matching domains and lengths",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DyadicOfPmpeOutput {
    pub tensor: ScaledShareTensor,
    pub scale_profile: DyadicScaleProfile,
    pub profile: OfPmpeProfile,
}

#[derive(Clone, Debug)]
pub struct OfPmpeLayerNormSquareOutput {
    pub centered: AdditiveShares,
    pub squares: AdditiveShares,
    pub profile: OfPmpeProfile,
}

#[derive(Clone, Debug)]
pub struct DyadicOfPmpeLayerNormSquareOutput {
    pub centered: AdditiveShares,
    pub squares: ScaledShareTensor,
    pub square_profile: OfPmpeProfile,
    pub square_scale_profile: DyadicScaleProfile,
}

#[derive(Clone, Debug)]
pub struct DyadicRescaleOutput {
    pub tensor: ScaledShareTensor,
    pub shift_bits: u32,
    pub online_ms: u64,
    pub online_us: u64,
}

#[derive(Clone, Debug)]
pub struct DyadicScaledMulOutput {
    pub tensor: ScaledShareTensor,
    pub shift_bits: u32,
    pub online_ms: u64,
    pub online_us: u64,
}

#[derive(Clone, Debug)]
pub struct OfPmpeSoftmaxOutput {
    pub shares: AdditiveShares,
    pub exp_profile: OfPmpeProfile,
    pub online_ms: u64,
}

#[derive(Clone, Debug)]
pub struct DyadicOfPmpeSoftmaxExpOutput {
    pub shifted: AdditiveShares,
    pub exp: ScaledShareTensor,
    pub row_sums: ScaledShareTensor,
    pub exp_profile: OfPmpeProfile,
    pub exp_scale_profile: DyadicScaleProfile,
    pub online_ms: u64,
}

#[derive(Clone, Debug)]
pub struct DyadicOfPmpeSoftmaxReciprocalOutput {
    pub exp_phase: DyadicOfPmpeSoftmaxExpOutput,
    pub row_sums_base: ScaledShareTensor,
    pub inv_row_sums: ScaledShareTensor,
    pub rescale: DyadicRescaleOutput,
    pub reciprocal_ms: u64,
    pub reciprocal_us: u64,
    pub online_ms: u64,
    pub online_us: u64,
}

#[derive(Clone, Debug)]
pub struct DyadicOfPmpeSoftmaxOutput {
    pub reciprocal_phase: DyadicOfPmpeSoftmaxReciprocalOutput,
    pub probs: ScaledShareTensor,
    pub broadcast_mul: DyadicScaledMulOutput,
    pub online_ms: u64,
    pub online_us: u64,
}

#[derive(Clone, Debug)]
pub struct DyadicOfPmpeLayerNormRsqrtOutput {
    pub square_phase: DyadicOfPmpeLayerNormSquareOutput,
    pub variance_high: ScaledShareTensor,
    pub variance_base: ScaledShareTensor,
    pub inv_sqrt: ScaledShareTensor,
    pub rescale: DyadicRescaleOutput,
    pub rsqrt_ms: u64,
    pub rsqrt_us: u64,
    pub online_ms: u64,
    pub online_us: u64,
}

#[derive(Clone, Debug)]
pub struct CadscMultiPlan {
    pub thresholds: Vec<i128>,
    pub bit_width: usize,
    pub radix_bits: usize,
    pub selector_count: usize,
    pub predicate_count: usize,
    pub shared_prefix_nodes: usize,
}

#[derive(Clone, Debug)]
pub struct FastGeluConfig {
    pub clip_bound: f64,
    pub middle_thresholds: Vec<f64>,
    pub coeffs: Vec<Vec<f64>>,
    pub centers: Vec<f64>,
    pub assume_in_clip_range: bool,
    pub approximate_truncation: bool,
}

impl Default for FastGeluConfig {
    fn default() -> Self {
        Self {
            clip_bound: 3.0,
            middle_thresholds: vec![-1.0, 1.0],
            coeffs: vec![
                vec![
                    -0.303441298634,
                    -0.133705911068,
                    0.0150366950236,
                    0.00889878235115,
                ],
                vec![0.00489290883384, 0.5, 0.348188405227, 0.0],
                vec![
                    -0.303441298634,
                    1.13370591107,
                    0.0150366950236,
                    -0.00889878235115,
                ],
            ],
            centers: vec![0.0, 0.0, 0.0],
            assume_in_clip_range: false,
            approximate_truncation: false,
        }
    }
}

impl FastGeluConfig {
    pub fn quadratic_three_interval() -> Self {
        Self {
            clip_bound: 3.0,
            middle_thresholds: vec![-1.0, 1.0],
            coeffs: vec![
                vec![-0.0470729165178129, -0.0817013982604694, -0.038356191919985],
                vec![0.00489279750771025, 0.5, 0.348188960031138],
                vec![1.95292708348219, 1.08170139826047, -0.0383561919199882],
            ],
            centers: vec![-2.0, 0.0, 2.0],
            assume_in_clip_range: false,
            approximate_truncation: false,
        }
    }

    pub fn calibrated_unit_quadratic() -> Self {
        Self {
            clip_bound: 1.0,
            middle_thresholds: Vec::new(),
            coeffs: vec![vec![0.00489279750771025, 0.5, 0.348188960031138]],
            centers: vec![0.0],
            assume_in_clip_range: true,
            approximate_truncation: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RangeSoftmaxConfig {
    pub exp_clip_min: f64,
    pub exp_clip_max: f64,
    pub exp_thresholds: Vec<f64>,
    pub exp_coeffs: Vec<Vec<f64>>,
    pub use_row_max: bool,
    pub public_input_shift: f64,
    pub approximate_truncation: bool,
}

impl Default for RangeSoftmaxConfig {
    fn default() -> Self {
        Self {
            exp_clip_min: -8.0,
            exp_clip_max: 0.0,
            exp_thresholds: vec![-4.0, -2.0, -1.0],
            exp_coeffs: vec![
                vec![
                    0.186678678973,
                    0.0775798365941,
                    0.0108942973695,
                    0.000514266169672,
                ],
                vec![
                    0.6760737886,
                    0.446911158244,
                    0.105677808109,
                    0.00877184843421,
                ],
                vec![
                    0.940562905329,
                    0.818391109405,
                    0.28327812095,
                    0.037710408058,
                ],
                vec![
                    0.99961962442,
                    0.992080695015,
                    0.462507217674,
                    0.102507516968,
                ],
            ],
            use_row_max: true,
            public_input_shift: 0.0,
            approximate_truncation: false,
        }
    }
}

impl RangeSoftmaxConfig {
    pub fn calibrated_unit_interval() -> Self {
        Self {
            exp_clip_min: -1.0,
            exp_clip_max: 0.0,
            exp_thresholds: vec![-0.75, -0.5, -0.25],
            exp_coeffs: vec![
                vec![
                    0.988011302748373,
                    0.941862750543835,
                    0.391268224836302,
                    0.0695373358693978,
                ],
                vec![
                    0.996292473194098,
                    0.974917983013266,
                    0.435432565433165,
                    0.0892877066649815,
                ],
                vec![
                    0.999462775697221,
                    0.993762669631807,
                    0.473120717703088,
                    0.114647684755644,
                ],
                vec![1.0, 0.999868989089794, 0.497091120799303, 0.14721054119059],
            ],
            use_row_max: false,
            public_input_shift: 0.5,
            approximate_truncation: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BinadeLayerNormConfig {
    pub epsilon: f64,
    pub variance_clip_max: f64,
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub binade_thresholds: Vec<f64>,
    pub rsqrt_seeds: Vec<f64>,
    pub approximate_truncation: bool,
}

impl Default for BinadeLayerNormConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.25,
            variance_clip_max: 4.0,
            gamma: Vec::new(),
            beta: Vec::new(),
            binade_thresholds: vec![0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
            rsqrt_seeds: vec![
                3.938927711338647,
                2.3094010767585034,
                1.6329931618554523,
                1.1547005383792517,
                0.8164965809277261,
                0.5773502691896258,
                0.5,
            ],
            approximate_truncation: false,
        }
    }
}

pub struct NonlinearOps<'a> {
    fixed: FixedPointConfig,
    lookup: &'a PrivateLookup,
    hss: &'a HssSlotEngine,
    rescale_corrections: Option<RescaleCorrectionTables>,
}

#[derive(Clone, Debug)]
struct RescaleCorrectionTables {
    support: Vec<usize>,
    positive: Vec<u64>,
    negative: Vec<u64>,
    positive_base: u64,
    negative_base: u64,
    positive_changes: Vec<CorrectionChange>,
    negative_changes: Vec<CorrectionChange>,
}

#[derive(Clone, Copy, Debug)]
struct CorrectionChange {
    threshold: u64,
    delta: u64,
}

impl<'a> NonlinearOps<'a> {
    pub fn new(
        fixed: FixedPointConfig,
        lookup: &'a PrivateLookup,
        hss: &'a HssSlotEngine,
    ) -> Result<Self, OperatorError> {
        if fixed.modulus != lookup.config().output_modulus {
            return Err(OperatorError::InvalidParams(
                "fixed-point modulus must match lookup output modulus",
            ));
        }
        if fixed.modulus != hss.context().plain_modulus() {
            return Err(OperatorError::InvalidParams(
                "fixed-point modulus must match HSS plaintext modulus",
            ));
        }
        if fixed.modulus as usize > lookup.config().max_domain {
            return Err(OperatorError::InvalidParams(
                "fixed-point modulus exceeds lookup max_domain",
            ));
        }
        let rescale_corrections = build_rescale_correction_tables(fixed)?;
        Ok(Self {
            fixed,
            lookup,
            hss,
            rescale_corrections,
        })
    }

    pub fn fixed(&self) -> FixedPointConfig {
        self.fixed
    }

    pub fn cadsc_multi_compile(
        &self,
        thresholds: &[i128],
    ) -> Result<CadscMultiPlan, OperatorError> {
        validate_strict_thresholds(thresholds)?;
        let bit_width = sum_bit_width(self.fixed.modulus)?;
        let radix_bits = comparison_radix_bits(bit_width, self.fixed.modulus)?;
        let digits = bit_width.div_ceil(radix_bits);
        Ok(CadscMultiPlan {
            thresholds: thresholds.to_vec(),
            bit_width,
            radix_bits,
            selector_count: thresholds.len() + 1,
            predicate_count: thresholds.len().saturating_mul(digits).saturating_mul(2),
            shared_prefix_nodes: thresholds.len().saturating_mul(digits),
        })
    }

    pub fn masked_open_batch_oneflow(
        &self,
        shares: &AdditiveShares,
    ) -> Result<OfPmpeMaskedOpen, OperatorError> {
        let start = Instant::now();
        self.validate_share_modulus(shares)?;
        let values = shares.reconstruct();
        let bytes = oneflow_open_bytes(shares.len());
        Ok(OfPmpeMaskedOpen {
            values,
            elements: shares.len(),
            bytes,
            latency_ms: start.elapsed().as_millis() as u64,
            num_flushes: if shares.is_empty() { 0 } else { 1 },
            num_request_response_rounds: 0,
        })
    }

    pub fn of_pmpe_taylor_pack<R: RngCore + ?Sized>(
        &self,
        slots: usize,
        coeffs: &[f64],
        rng: &mut R,
    ) -> Result<OfPmpeTaylorPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE Taylor pack requires at least one slot",
            ));
        }
        if coeffs.is_empty() {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE Taylor pack requires at least one coefficient",
            ));
        }
        let degree = coeffs.len() - 1;
        let p = self.fixed.modulus;
        let inv_scale = mod_inverse_u64(self.fixed.scale % p, p).ok_or(
            OperatorError::InvalidParams("OF-PMPE requires fixed scale invertible modulo p"),
        )?;

        let coeffs_mod = coeffs
            .iter()
            .map(|&coeff| self.fixed.encode_f64(coeff))
            .collect::<Vec<_>>();
        let mut inv_scale_powers = vec![1u64; degree + 1];
        for idx in 1..=degree {
            inv_scale_powers[idx] = mul_mod(inv_scale_powers[idx - 1], inv_scale, p);
        }

        let masks_plain = (0..slots).map(|_| rng.next_u64() % p).collect::<Vec<_>>();
        let mut columns = vec![vec![0u64; slots]; degree + 1];
        for (slot, &mask) in masks_plain.iter().enumerate() {
            let mut mask_powers = vec![1u64; degree + 1];
            for idx in 1..=degree {
                mask_powers[idx] = mul_mod(mask_powers[idx - 1], mask, p);
            }
            for k in 0..=degree {
                let mut acc = 0u64;
                for j in k..=degree {
                    let mut term = mul_mod(coeffs_mod[j], binomial_mod(j, k, p)?, p);
                    term = mul_mod(term, mask_powers[j - k], p);
                    term = mul_mod(term, inv_scale_powers[j], p);
                    acc = add_mod(acc, term, p);
                }
                columns[k][slot] = acc;
            }
        }

        let masks = AdditiveShares::share_with_rng(&masks_plain, p, rng)?;
        let taylor_coeffs = columns
            .iter()
            .map(|column| AdditiveShares::share_with_rng(column, p, rng))
            .collect::<Result<Vec<_>, _>>()?;
        let pack_size_bytes = of_pmpe_pack_bytes(slots, degree);
        let elapsed = start.elapsed();
        Ok(OfPmpeTaylorPack {
            degree,
            masks,
            taylor_coeffs,
            offline_mode: OfPmpeOfflineMode::TrustedDebug,
            offline_ms: elapsed.as_millis() as u64,
            offline_us: elapsed.as_micros() as u64,
            offline_bytes: pack_size_bytes,
            pack_size_bytes,
            peak_pack_resident_bytes: pack_size_bytes,
            streaming_pack: false,
            consumed: false,
        })
    }

    pub fn dyadic_of_pmpe_taylor_pack<R: RngCore + ?Sized>(
        &self,
        slots: usize,
        cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<OfPmpeTaylorPack, OperatorError> {
        self.validate_dyadic_of_pmpe_config(cfg)?;
        self.of_pmpe_integer_taylor_pack(slots, &cfg.coeffs_num, rng)
    }

    pub fn of_pmpe_eval(
        &self,
        input: &AdditiveShares,
        pack: &mut OfPmpeTaylorPack,
    ) -> Result<OfPmpeOutput, OperatorError> {
        let start = Instant::now();
        if pack.consumed {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE Taylor pack was already consumed",
            ));
        }
        self.validate_share_modulus(input)?;
        self.validate_share_modulus(&pack.masks)?;
        if pack.masks.len() != input.len() {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE mask length must match input length",
            ));
        }
        if pack.taylor_coeffs.len() != pack.degree + 1 {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE Taylor coefficient count must equal degree + 1",
            ));
        }
        for coeff in &pack.taylor_coeffs {
            self.validate_share_modulus(coeff)?;
            if coeff.len() != input.len() {
                return Err(OperatorError::InvalidParams(
                    "OF-PMPE Taylor coefficient length must match input length",
                ));
            }
        }

        let delta_shares = input.sub(&pack.masks)?;
        let opened = self.masked_open_batch_oneflow(&delta_shares)?;
        let p = self.fixed.modulus;
        let mut output = pack.taylor_coeffs[0].clone();
        let mut powers = vec![1u64; input.len()];
        for degree in 1..=pack.degree {
            for (power, &delta) in powers.iter_mut().zip(opened.values.iter()) {
                *power = mul_mod(*power, delta, p);
            }
            let term = mul_public_slots_mod(&pack.taylor_coeffs[degree], &powers)?;
            output = output.add(&term)?;
        }
        pack.consumed = true;
        let (
            secure_offline_ms,
            trusted_debug_offline_ms,
            secure_offline_us,
            trusted_debug_offline_us,
        ) = match pack.offline_mode {
            OfPmpeOfflineMode::SecurePreprocess => (pack.offline_ms, 0, pack.offline_us, 0),
            OfPmpeOfflineMode::TrustedDebug => (0, pack.offline_ms, 0, pack.offline_us),
        };
        let online_elapsed = start.elapsed();

        Ok(OfPmpeOutput {
            shares: output,
            profile: OfPmpeProfile {
                offline_mode: pack.offline_mode,
                degree: pack.degree,
                slots: input.len(),
                num_oneflow_phases: if input.is_empty() { 0 } else { 1 },
                num_request_response_rounds: opened.num_request_response_rounds,
                num_masked_opens: opened.elements,
                num_opened_elements: opened.elements,
                num_flushes: opened.num_flushes,
                num_network_flushes: opened.num_flushes,
                public_linear_terms: pack.degree + 1,
                online_secret_secret_mul: 0,
                online_trunc: 0,
                num_fresh_masks: input.len(),
                num_reused_masks: 0,
                offline_ms: pack.offline_ms,
                online_ms: online_elapsed.as_millis() as u64,
                offline_us: pack.offline_us,
                online_us: online_elapsed.as_micros() as u64,
                secure_offline_ms,
                trusted_debug_offline_ms,
                secure_offline_us,
                trusted_debug_offline_us,
                offline_bytes: pack.offline_bytes,
                online_bytes: opened.bytes,
                oneflow_payload_bytes: opened.bytes,
                pack_size_bytes: pack.pack_size_bytes,
                peak_pack_resident_bytes: pack.peak_pack_resident_bytes,
                streaming_pack: pack.streaming_pack,
            },
        })
    }

    pub fn of_pmpe_eval_streaming<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        coeffs: &[f64],
        chunk_slots: usize,
        rng: &mut R,
    ) -> Result<OfPmpeOutput, OperatorError> {
        self.validate_share_modulus(input)?;
        if input.is_empty() {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE streaming eval requires at least one input slot",
            ));
        }
        if chunk_slots == 0 {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE streaming chunk size must be positive",
            ));
        }
        if coeffs.is_empty() {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE streaming eval requires at least one coefficient",
            ));
        }

        let p = self.fixed.modulus;
        let degree = coeffs.len() - 1;
        let total_pack_size_bytes = of_pmpe_pack_bytes(input.len(), degree);
        let mut peak_pack_resident_bytes = 0u64;
        let mut out0 = Vec::with_capacity(input.len());
        let mut out1 = Vec::with_capacity(input.len());
        let mut profile = OfPmpeProfile {
            offline_mode: OfPmpeOfflineMode::TrustedDebug,
            degree,
            slots: input.len(),
            num_oneflow_phases: 1,
            public_linear_terms: coeffs.len(),
            num_fresh_masks: input.len(),
            pack_size_bytes: total_pack_size_bytes,
            streaming_pack: true,
            ..OfPmpeProfile::default()
        };

        for start_idx in (0..input.len()).step_by(chunk_slots) {
            let end_idx = usize::min(start_idx + chunk_slots, input.len());
            let chunk_input = AdditiveShares::new(
                p,
                input.party0()[start_idx..end_idx].to_vec(),
                input.party1()[start_idx..end_idx].to_vec(),
            )?;
            let mut pack = self.of_pmpe_taylor_pack(chunk_input.len(), coeffs, rng)?;
            peak_pack_resident_bytes =
                u64::max(peak_pack_resident_bytes, pack.peak_pack_resident_bytes);
            let output = self.of_pmpe_eval(&chunk_input, &mut pack)?;
            out0.extend_from_slice(output.shares.party0());
            out1.extend_from_slice(output.shares.party1());
            profile.num_masked_opens += output.profile.num_masked_opens;
            profile.num_opened_elements += output.profile.num_opened_elements;
            profile.num_flushes += output.profile.num_flushes;
            profile.num_network_flushes += output.profile.num_network_flushes;
            profile.num_request_response_rounds += output.profile.num_request_response_rounds;
            profile.online_secret_secret_mul += output.profile.online_secret_secret_mul;
            profile.online_trunc += output.profile.online_trunc;
            profile.num_reused_masks += output.profile.num_reused_masks;
            profile.offline_ms += output.profile.offline_ms;
            profile.online_ms += output.profile.online_ms;
            profile.offline_us += output.profile.offline_us;
            profile.online_us += output.profile.online_us;
            profile.secure_offline_ms += output.profile.secure_offline_ms;
            profile.trusted_debug_offline_ms += output.profile.trusted_debug_offline_ms;
            profile.secure_offline_us += output.profile.secure_offline_us;
            profile.trusted_debug_offline_us += output.profile.trusted_debug_offline_us;
            profile.offline_bytes += output.profile.offline_bytes;
            profile.online_bytes += output.profile.online_bytes;
            profile.oneflow_payload_bytes += output.profile.oneflow_payload_bytes;
        }
        profile.peak_pack_resident_bytes = peak_pack_resident_bytes;

        Ok(OfPmpeOutput {
            shares: AdditiveShares::new(p, out0, out1)?,
            profile,
        })
    }

    pub fn dyadic_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeOutput, OperatorError> {
        self.validate_share_modulus(input)?;
        self.validate_dyadic_of_pmpe_config(cfg)?;
        if cfg.streaming {
            return self.dyadic_of_pmpe_eval_streaming(input, cfg, rng);
        }
        let mut pack = self.dyadic_of_pmpe_taylor_pack(input.len(), cfg, rng)?;
        let output = self.of_pmpe_eval(input, &mut pack)?;
        self.wrap_dyadic_output(output, cfg)
    }

    pub fn dyadic_fast_gelu_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeOutput, OperatorError> {
        self.dyadic_of_pmpe_profiled(input, cfg, rng)
    }

    pub fn dyadic_exp_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeOutput, OperatorError> {
        self.dyadic_of_pmpe_profiled(input, cfg, rng)
    }

    pub fn dyadic_square_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        input_frac_bits: u32,
        chunk_slots: usize,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeOutput, OperatorError> {
        let cfg = DyadicOfPmpePolynomialConfig::square(input_frac_bits, chunk_slots)?;
        self.dyadic_of_pmpe_profiled(input, &cfg, rng)
    }

    pub fn rescale_dyadic_nonnegative_profiled<R: RngCore + ?Sized>(
        &self,
        input: &ScaledShareTensor,
        target_frac_bits: u32,
        rng: &mut R,
    ) -> Result<DyadicRescaleOutput, OperatorError> {
        let start = Instant::now();
        self.validate_share_modulus(&input.shares)?;
        if input.semantic != ScaleSemantic::DyadicNumerator {
            return Err(OperatorError::InvalidParams(
                "dyadic rescale requires a dyadic numerator tensor",
            ));
        }
        if target_frac_bits > input.frac_bits {
            return Err(OperatorError::InvalidParams(
                "dyadic rescale target frac_bits must not exceed input frac_bits",
            ));
        }
        let shift_bits = input.frac_bits - target_frac_bits;
        let shares = if shift_bits == 0 {
            input.shares.clone()
        } else {
            let scale = pow2_u64(shift_bits)?;
            let Some(corrections) = build_rescale_correction_tables(FixedPointConfig {
                modulus: self.fixed.modulus,
                scale,
            })?
            else {
                return Err(OperatorError::InvalidParams(
                    "dyadic rescale shift is too large for current modulus",
                ));
            };
            self.rescale_nonnegative_by_scale(&input.shares, scale, &corrections, rng)?
        };
        let elapsed = start.elapsed();
        Ok(DyadicRescaleOutput {
            tensor: ScaledShareTensor {
                shares,
                frac_bits: target_frac_bits,
                guard_bits: input.guard_bits.saturating_sub(shift_bits),
                semantic: input.semantic,
            },
            shift_bits,
            online_ms: elapsed.as_millis() as u64,
            online_us: elapsed.as_micros() as u64,
        })
    }

    pub fn mul_scaled_nonnegative_profiled<R: RngCore + ?Sized>(
        &self,
        lhs: &ScaledShareTensor,
        rhs: &ScaledShareTensor,
        target_frac_bits: u32,
        rng: &mut R,
    ) -> Result<DyadicScaledMulOutput, OperatorError> {
        let start = Instant::now();
        self.validate_share_modulus(&lhs.shares)?;
        self.validate_share_modulus(&rhs.shares)?;
        if lhs.shares.modulus() != rhs.shares.modulus() {
            return Err(OperatorError::InvalidParams(
                "scaled multiply requires matching moduli",
            ));
        }
        if lhs.len() != rhs.len() {
            return Err(OperatorError::InvalidParams(
                "scaled multiply inputs must have equal length",
            ));
        }
        if lhs.semantic != ScaleSemantic::DyadicNumerator
            || rhs.semantic != ScaleSemantic::DyadicNumerator
        {
            return Err(OperatorError::InvalidParams(
                "scaled multiply requires dyadic numerator tensors",
            ));
        }
        let product_frac_bits =
            lhs.frac_bits
                .checked_add(rhs.frac_bits)
                .ok_or(OperatorError::InvalidParams(
                    "scaled multiply product frac_bits overflow",
                ))?;
        if target_frac_bits > product_frac_bits {
            return Err(OperatorError::InvalidParams(
                "scaled multiply target frac_bits must not exceed product frac_bits",
            ));
        }

        let raw = self.hss.mul_slots(&lhs.shares, &rhs.shares, rng)?;
        let shift_bits = product_frac_bits - target_frac_bits;
        let shares = if shift_bits == 0 {
            raw
        } else {
            let scale = pow2_u64(shift_bits)?;
            let Some(corrections) = build_rescale_correction_tables(FixedPointConfig {
                modulus: self.fixed.modulus,
                scale,
            })?
            else {
                return Err(OperatorError::InvalidParams(
                    "scaled multiply rescale shift is too large for current modulus",
                ));
            };
            self.rescale_nonnegative_by_scale(&raw, scale, &corrections, rng)?
        };
        let elapsed = start.elapsed();
        Ok(DyadicScaledMulOutput {
            tensor: ScaledShareTensor {
                shares,
                frac_bits: target_frac_bits,
                guard_bits: lhs
                    .guard_bits
                    .saturating_add(rhs.guard_bits)
                    .saturating_sub(shift_bits),
                semantic: ScaleSemantic::DyadicNumerator,
            },
            shift_bits,
            online_ms: elapsed.as_millis() as u64,
            online_us: elapsed.as_micros() as u64,
        })
    }

    pub fn fast_gelu_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &OfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<OfPmpeOutput, OperatorError> {
        self.of_pmpe_polynomial_profiled(input, cfg, rng)
    }

    pub fn exp_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &OfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<OfPmpeOutput, OperatorError> {
        self.of_pmpe_polynomial_profiled(input, cfg, rng)
    }

    pub fn square_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        chunk_slots: usize,
        rng: &mut R,
    ) -> Result<OfPmpeOutput, OperatorError> {
        self.of_pmpe_polynomial_profiled(input, &OfPmpePolynomialConfig::square(chunk_slots), rng)
    }

    pub fn layer_norm_square_of_pmpe_rows_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        chunk_slots: usize,
        rng: &mut R,
    ) -> Result<OfPmpeLayerNormSquareOutput, OperatorError> {
        self.validate_share_modulus(row_major)?;
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE LayerNorm square dimensions must be non-empty",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE LayerNorm square length must equal rows * cols",
            ));
        }
        let inv_n = 1.0 / cols as f64;
        let sum = sum_rows(row_major, rows, cols)?;
        let mean = self.mul_public_fixed(&sum, inv_n, rng)?;
        let mean_row = repeat_rows(&mean, cols)?;
        let centered = row_major.sub(&mean_row)?;
        let squares = self.square_of_pmpe_profiled(&centered, chunk_slots, rng)?;
        Ok(OfPmpeLayerNormSquareOutput {
            centered,
            squares: squares.shares,
            profile: squares.profile,
        })
    }

    pub fn layer_norm_square_dyadic_of_pmpe_rows_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        input_frac_bits: u32,
        chunk_slots: usize,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeLayerNormSquareOutput, OperatorError> {
        self.validate_share_modulus(row_major)?;
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE LayerNorm square dimensions must be non-empty",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE LayerNorm square length must equal rows * cols",
            ));
        }
        let inv_n = 1.0 / cols as f64;
        let sum = sum_rows(row_major, rows, cols)?;
        let mean = self.mul_public_fixed(&sum, inv_n, rng)?;
        let mean_row = repeat_rows(&mean, cols)?;
        let centered = row_major.sub(&mean_row)?;
        let mut cfg = DyadicOfPmpePolynomialConfig::square(input_frac_bits, chunk_slots)?;
        cfg.streaming = true;
        let squares = self.dyadic_of_pmpe_profiled(&centered, &cfg, rng)?;
        Ok(DyadicOfPmpeLayerNormSquareOutput {
            centered,
            squares: squares.tensor,
            square_profile: squares.profile,
            square_scale_profile: squares.scale_profile,
        })
    }

    pub fn layer_norm_dyadic_of_pmpe_rsqrt_phase_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        input_frac_bits: u32,
        chunk_slots: usize,
        cfg: &LayerNormConfig,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeLayerNormRsqrtOutput, OperatorError> {
        let start = Instant::now();
        if cfg.epsilon < 0.0 {
            return Err(OperatorError::InvalidParams(
                "dyadic LayerNorm epsilon must be non-negative",
            ));
        }
        let square_phase = self.layer_norm_square_dyadic_of_pmpe_rows_profiled(
            row_major,
            rows,
            cols,
            input_frac_bits,
            chunk_slots,
            rng,
        )?;
        let sum_sq = sum_rows(&square_phase.squares.shares, rows, cols)?;
        let variance_shares =
            self.div_public_u64_nonnegative_preserve_scale(&sum_sq, cols as u64, rng)?;
        let variance_high = ScaledShareTensor {
            shares: variance_shares,
            frac_bits: square_phase.squares.frac_bits,
            guard_bits: square_phase.squares.guard_bits,
            semantic: square_phase.squares.semantic,
        };
        let rescale =
            self.rescale_dyadic_nonnegative_profiled(&variance_high, input_frac_bits, rng)?;
        let eps = self.fixed.encode_f64(cfg.epsilon);
        let variance_base_shares = if eps == 0 {
            rescale.tensor.shares.clone()
        } else {
            rescale
                .tensor
                .shares
                .add_public_to_party0(&vec![eps; rows])?
        };
        let variance_base = ScaledShareTensor {
            shares: variance_base_shares,
            frac_bits: input_frac_bits,
            guard_bits: 0,
            semantic: ScaleSemantic::DyadicNumerator,
        };
        let rsqrt_start = Instant::now();
        let inv_sqrt_shares =
            self.rsqrt_lut_bounded(&variance_base.shares, cfg.variance_clip_max, rng)?;
        let rsqrt_elapsed = rsqrt_start.elapsed();
        let elapsed = start.elapsed();
        Ok(DyadicOfPmpeLayerNormRsqrtOutput {
            square_phase,
            variance_high,
            variance_base,
            inv_sqrt: ScaledShareTensor {
                shares: inv_sqrt_shares,
                frac_bits: input_frac_bits,
                guard_bits: 0,
                semantic: ScaleSemantic::DyadicNumerator,
            },
            rescale,
            rsqrt_ms: rsqrt_elapsed.as_millis() as u64,
            rsqrt_us: rsqrt_elapsed.as_micros() as u64,
            online_ms: elapsed.as_millis() as u64,
            online_us: elapsed.as_micros() as u64,
        })
    }

    fn of_pmpe_polynomial_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &OfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<OfPmpeOutput, OperatorError> {
        self.validate_share_modulus(input)?;
        if cfg.coeffs.is_empty() {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE polynomial requires at least one coefficient",
            ));
        }
        if cfg.streaming {
            self.of_pmpe_eval_streaming(input, &cfg.coeffs, cfg.chunk_slots, rng)
        } else {
            let mut pack = self.of_pmpe_taylor_pack(input.len(), &cfg.coeffs, rng)?;
            self.of_pmpe_eval(input, &mut pack)
        }
    }

    fn of_pmpe_integer_taylor_pack<R: RngCore + ?Sized>(
        &self,
        slots: usize,
        coeffs_num: &[i128],
        rng: &mut R,
    ) -> Result<OfPmpeTaylorPack, OperatorError> {
        let start = Instant::now();
        if slots == 0 {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE Taylor pack requires at least one slot",
            ));
        }
        if coeffs_num.is_empty() {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE Taylor pack requires at least one coefficient",
            ));
        }
        let degree = coeffs_num.len() - 1;
        let p = self.fixed.modulus;
        let coeffs_mod = coeffs_num
            .iter()
            .map(|&coeff| reduce_i128_mod(coeff, p))
            .collect::<Vec<_>>();

        let masks_plain = (0..slots).map(|_| rng.next_u64() % p).collect::<Vec<_>>();
        let mut columns = vec![vec![0u64; slots]; degree + 1];
        for (slot, &mask) in masks_plain.iter().enumerate() {
            let mut mask_powers = vec![1u64; degree + 1];
            for idx in 1..=degree {
                mask_powers[idx] = mul_mod(mask_powers[idx - 1], mask, p);
            }
            for k in 0..=degree {
                let mut acc = 0u64;
                for j in k..=degree {
                    let mut term = mul_mod(coeffs_mod[j], binomial_mod(j, k, p)?, p);
                    term = mul_mod(term, mask_powers[j - k], p);
                    acc = add_mod(acc, term, p);
                }
                columns[k][slot] = acc;
            }
        }

        let masks = AdditiveShares::share_with_rng(&masks_plain, p, rng)?;
        let taylor_coeffs = columns
            .iter()
            .map(|column| AdditiveShares::share_with_rng(column, p, rng))
            .collect::<Result<Vec<_>, _>>()?;
        let pack_size_bytes = of_pmpe_pack_bytes(slots, degree);
        let elapsed = start.elapsed();
        Ok(OfPmpeTaylorPack {
            degree,
            masks,
            taylor_coeffs,
            offline_mode: OfPmpeOfflineMode::TrustedDebug,
            offline_ms: elapsed.as_millis() as u64,
            offline_us: elapsed.as_micros() as u64,
            offline_bytes: pack_size_bytes,
            pack_size_bytes,
            peak_pack_resident_bytes: pack_size_bytes,
            streaming_pack: false,
            consumed: false,
        })
    }

    fn dyadic_of_pmpe_eval_streaming<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeOutput, OperatorError> {
        self.validate_share_modulus(input)?;
        if input.is_empty() {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE streaming eval requires at least one input slot",
            ));
        }
        if cfg.chunk_slots == 0 {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE streaming chunk size must be positive",
            ));
        }

        let p = self.fixed.modulus;
        let total_pack_size_bytes = of_pmpe_pack_bytes(input.len(), cfg.degree);
        let mut peak_pack_resident_bytes = 0u64;
        let mut out0 = Vec::with_capacity(input.len());
        let mut out1 = Vec::with_capacity(input.len());
        let mut profile = OfPmpeProfile {
            offline_mode: OfPmpeOfflineMode::TrustedDebug,
            degree: cfg.degree,
            slots: input.len(),
            num_oneflow_phases: 1,
            public_linear_terms: cfg.coeffs_num.len(),
            num_fresh_masks: input.len(),
            pack_size_bytes: total_pack_size_bytes,
            streaming_pack: true,
            ..OfPmpeProfile::default()
        };

        for start_idx in (0..input.len()).step_by(cfg.chunk_slots) {
            let end_idx = usize::min(start_idx + cfg.chunk_slots, input.len());
            let chunk_input = AdditiveShares::new(
                p,
                input.party0()[start_idx..end_idx].to_vec(),
                input.party1()[start_idx..end_idx].to_vec(),
            )?;
            let mut pack =
                self.of_pmpe_integer_taylor_pack(chunk_input.len(), &cfg.coeffs_num, rng)?;
            peak_pack_resident_bytes =
                u64::max(peak_pack_resident_bytes, pack.peak_pack_resident_bytes);
            let output = self.of_pmpe_eval(&chunk_input, &mut pack)?;
            out0.extend_from_slice(output.shares.party0());
            out1.extend_from_slice(output.shares.party1());
            profile.num_masked_opens += output.profile.num_masked_opens;
            profile.num_opened_elements += output.profile.num_opened_elements;
            profile.num_flushes += output.profile.num_flushes;
            profile.num_network_flushes += output.profile.num_network_flushes;
            profile.num_request_response_rounds += output.profile.num_request_response_rounds;
            profile.online_secret_secret_mul += output.profile.online_secret_secret_mul;
            profile.online_trunc += output.profile.online_trunc;
            profile.num_reused_masks += output.profile.num_reused_masks;
            profile.offline_ms += output.profile.offline_ms;
            profile.online_ms += output.profile.online_ms;
            profile.offline_us += output.profile.offline_us;
            profile.online_us += output.profile.online_us;
            profile.secure_offline_ms += output.profile.secure_offline_ms;
            profile.trusted_debug_offline_ms += output.profile.trusted_debug_offline_ms;
            profile.secure_offline_us += output.profile.secure_offline_us;
            profile.trusted_debug_offline_us += output.profile.trusted_debug_offline_us;
            profile.offline_bytes += output.profile.offline_bytes;
            profile.online_bytes += output.profile.online_bytes;
            profile.oneflow_payload_bytes += output.profile.oneflow_payload_bytes;
        }
        profile.peak_pack_resident_bytes = peak_pack_resident_bytes;

        self.wrap_dyadic_output(
            OfPmpeOutput {
                shares: AdditiveShares::new(p, out0, out1)?,
                profile,
            },
            cfg,
        )
    }

    fn wrap_dyadic_output(
        &self,
        output: OfPmpeOutput,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<DyadicOfPmpeOutput, OperatorError> {
        Ok(DyadicOfPmpeOutput {
            tensor: ScaledShareTensor {
                shares: output.shares,
                frac_bits: cfg.output_frac_bits_with_guard()?,
                guard_bits: cfg.guard_bits,
                semantic: ScaleSemantic::DyadicNumerator,
            },
            scale_profile: self.dyadic_scale_profile(cfg)?,
            profile: output.profile,
        })
    }

    fn dyadic_scale_profile(
        &self,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<DyadicScaleProfile, OperatorError> {
        let modulus_bits = modulus_bit_width(self.fixed.modulus);
        Ok(DyadicScaleProfile {
            input_frac_bits: cfg.input_frac_bits,
            output_frac_bits: cfg.output_frac_bits,
            guard_bits: cfg.guard_bits,
            effective_output_frac_bits: cfg.output_frac_bits_with_guard()?,
            input_abs_bound_bits: cfg.input_abs_bound_bits,
            output_bound_bits: cfg.output_bound_bits,
            modulus_bits,
            modulus_margin_bits: cfg.modulus_margin_bits,
            modulus_wrap_safe: cfg.modulus_wrap_safe(modulus_bits),
        })
    }

    fn validate_dyadic_of_pmpe_config(
        &self,
        cfg: &DyadicOfPmpePolynomialConfig,
    ) -> Result<(), OperatorError> {
        if cfg.coeffs_num.is_empty() {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE requires at least one coefficient",
            ));
        }
        if cfg.degree + 1 != cfg.coeffs_num.len() {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE degree must match coefficient count",
            ));
        }
        if cfg.chunk_slots == 0 {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE chunk size must be positive",
            ));
        }
        let expected_scale = pow2_u64(cfg.input_frac_bits)?;
        if expected_scale != self.fixed.scale {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE input frac_bits must match fixed-point scale",
            ));
        }
        let actual_bits = modulus_bit_width(self.fixed.modulus);
        if cfg.modulus_bits != 0 && cfg.modulus_bits > actual_bits {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE config requires a larger share modulus",
            ));
        }
        if let Some(false) = cfg.modulus_wrap_safe(actual_bits) {
            if !cfg.allow_modular_wrap {
                return Err(OperatorError::InvalidParams(
                    "dyadic OF-PMPE output bound exceeds modulus; increase share modulus or lower guard_bits",
                ));
            }
        }
        cfg.output_frac_bits_with_guard()?;
        Ok(())
    }

    pub fn fast_gelu_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &FastGeluConfig,
        rng: &mut R,
    ) -> Result<ProfiledShares, OperatorError> {
        let start = Instant::now();
        self.validate_shares(input)?;
        if cfg.clip_bound <= 0.0 {
            return Err(OperatorError::InvalidParams(
                "FastGeLU clip_bound must be positive",
            ));
        }
        let expected_intervals = cfg.middle_thresholds.len() + 1;
        if cfg.coeffs.len() != expected_intervals {
            return Err(OperatorError::InvalidParams(
                "FastGeLU coefficient rows must match middle intervals",
            ));
        }
        if cfg.centers.len() != expected_intervals {
            return Err(OperatorError::InvalidParams(
                "FastGeLU centers must match middle intervals",
            ));
        }
        let Some(first_coeffs) = cfg.coeffs.first() else {
            return Err(OperatorError::InvalidParams(
                "FastGeLU requires at least one coefficient row",
            ));
        };
        if first_coeffs.is_empty() || cfg.coeffs.iter().any(|row| row.len() != first_coeffs.len()) {
            return Err(OperatorError::InvalidParams(
                "FastGeLU coefficient rows must be non-empty and equal length",
            ));
        }

        if cfg.assume_in_clip_range {
            if cfg.coeffs.len() != 1 || !cfg.middle_thresholds.is_empty() {
                return Err(OperatorError::InvalidParams(
                    "FastGeLU calibrated range mode requires exactly one polynomial interval",
                ));
            }
            let poly_input = if cfg.centers[0] != 0.0 {
                input.add_public_to_party0(&vec![
                    self.fixed.encode_f64(-cfg.centers[0]);
                    input.len()
                ])?
            } else {
                input.clone()
            };
            let coeffs = cfg.coeffs[0]
                .iter()
                .map(|&coeff| {
                    AdditiveShares::share_public(
                        &vec![self.fixed.encode_f64(coeff); input.len()],
                        self.fixed.modulus,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            if cfg.approximate_truncation && coeffs.len() == 3 && cfg.centers[0] == 0.0 {
                let mut profile = NonlinearProfile {
                    selectors: 1,
                    ..NonlinearProfile::default()
                };
                let sign = self.signed_less_than_scaled(&poly_input, 0, rng)?;
                let mut y =
                    self.mul_fixed_with_product_sign_approx(&poly_input, &coeffs[2], &sign, rng)?;
                profile.poly_mul += 1;
                profile.trunc += 1;
                y = y.add(&coeffs[1])?;
                y = self.mul_fixed_with_product_sign_approx(&poly_input, &y, &sign, rng)?;
                profile.poly_mul += 1;
                profile.trunc += 1;
                y = y.add(&coeffs[0])?;
                profile.online_ms = start.elapsed().as_millis() as u64;
                return Ok(ProfiledShares { shares: y, profile });
            }
            let mut eval =
                self.sac_eval_poly(&poly_input, &coeffs, cfg.approximate_truncation, rng)?;
            eval.profile.selectors = 0;
            eval.profile.online_ms = start.elapsed().as_millis() as u64;
            return Ok(eval);
        }

        let mut thresholds = Vec::with_capacity(cfg.middle_thresholds.len() + 2);
        thresholds.push(-self.fixed.signed_threshold(cfg.clip_bound));
        for &threshold in &cfg.middle_thresholds {
            thresholds.push(self.fixed.signed_threshold(threshold));
        }
        thresholds.push(self.fixed.signed_threshold(cfg.clip_bound));
        validate_strict_thresholds(&thresholds)?;

        let (selectors, plan) = self.nidcf_select(input, &thresholds, rng)?;
        let mut profile = NonlinearProfile {
            selectors: plan.selector_count,
            ..NonlinearProfile::default()
        };

        let mut coeff_rows = vec![vec![0.0; first_coeffs.len()]; selectors.len()];
        for (idx, coeffs) in cfg.coeffs.iter().enumerate() {
            coeff_rows[idx + 1] = coeffs.clone();
        }
        let selected_coeffs = self.sac_select_coefficients(&selectors, &coeff_rows)?;
        let poly_input = if cfg.centers.iter().any(|&center| center != 0.0) {
            let mut centers = vec![0.0; selectors.len()];
            for (idx, &center) in cfg.centers.iter().enumerate() {
                centers[idx + 1] = center;
            }
            let selected_center = self.sac_select_constants(&selectors, &centers)?;
            input.sub(&selected_center)?
        } else {
            input.clone()
        };
        let mut eval = self.sac_eval_poly(
            &poly_input,
            &selected_coeffs,
            cfg.approximate_truncation,
            rng,
        )?;
        profile.add_assign(eval.profile);

        let positive = self.mul_selector(
            selectors.last().ok_or_else(|| {
                OperatorError::Backend("FastGeLU missing positive selector".to_string())
            })?,
            input,
            rng,
        )?;
        profile.poly_mul += 1;
        eval.shares = eval.shares.add(&positive)?;
        profile.online_ms = start.elapsed().as_millis() as u64;
        eval.profile = profile;
        Ok(eval)
    }

    pub fn range_softmax_rows_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &RangeSoftmaxConfig,
        rng: &mut R,
    ) -> Result<ProfiledShares, OperatorError> {
        let start = Instant::now();
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "RangeSoftmax matrix dimensions must be non-empty",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "RangeSoftmax matrix length must equal rows * cols",
            ));
        }
        self.validate_shares(row_major)?;
        if cfg.exp_clip_min >= cfg.exp_clip_max {
            return Err(OperatorError::InvalidParams(
                "RangeSoftmax exp_clip_min must be < exp_clip_max",
            ));
        }
        if cfg.exp_coeffs.len() != cfg.exp_thresholds.len() + 1 {
            return Err(OperatorError::InvalidParams(
                "RangeSoftmax exp coefficients must match exp intervals",
            ));
        }
        self.ensure_refinement_headroom("RangeSoftmax")?;

        let shifted = if cfg.use_row_max {
            let max = self.row_max_matrix(row_major, rows, cols, rng)?;
            let max_row = repeat_rows(&max, cols)?;
            row_major.sub(&max_row)?
        } else if cfg.public_input_shift != 0.0 {
            row_major.add_public_to_party0(&vec![
                self.fixed.encode_f64(-cfg.public_input_shift);
                row_major.len()
            ])?
        } else {
            row_major.clone()
        };

        let mut profile = NonlinearProfile::default();
        let exp = self.range_exp_profiled(&shifted, cfg, rng)?;
        profile.add_assign(exp.profile);
        let sum_exp = sum_rows(&exp.shares, rows, cols)?;
        let recip = self.row_reciprocal_newton_profiled(&sum_exp, cols, rng)?;
        profile.add_assign(recip.profile);
        let recip_row = repeat_rows(&recip.shares, cols)?;
        let probs = if cfg.approximate_truncation {
            self.mul_fixed_nonnegative_approx(&exp.shares, &recip_row, rng)?
        } else {
            self.mul_fixed_nonnegative(&exp.shares, &recip_row, rng)?
        };
        profile.poly_mul += 1;
        profile.trunc += 1;
        profile.online_ms = start.elapsed().as_millis() as u64;
        Ok(ProfiledShares {
            shares: probs,
            profile,
        })
    }

    pub fn binade_layer_norm_rows_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &BinadeLayerNormConfig,
        rng: &mut R,
    ) -> Result<ProfiledShares, OperatorError> {
        let start = Instant::now();
        self.validate_shares(row_major)?;
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "BinadeLayerNorm matrix dimensions must be non-empty",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "BinadeLayerNorm matrix length must equal rows * cols",
            ));
        }
        if !cfg.gamma.is_empty() && cfg.gamma.len() != cols && cfg.gamma.len() != row_major.len() {
            return Err(OperatorError::InvalidParams(
                "BinadeLayerNorm gamma length must match columns or rows * cols",
            ));
        }
        if !cfg.beta.is_empty() && cfg.beta.len() != cols && cfg.beta.len() != row_major.len() {
            return Err(OperatorError::InvalidParams(
                "BinadeLayerNorm beta length must match columns or rows * cols",
            ));
        }
        if cfg.binade_thresholds.len() + 1 != cfg.rsqrt_seeds.len() {
            return Err(OperatorError::InvalidParams(
                "BinadeLayerNorm rsqrt seed count must equal binade intervals",
            ));
        }
        self.ensure_refinement_headroom("BinadeLayerNorm")?;

        let mut profile = NonlinearProfile::default();
        let inv_n = 1.0 / cols as f64;
        let sum = sum_rows(row_major, rows, cols)?;
        let mean = self.mul_public_fixed(&sum, inv_n, rng)?;
        profile.poly_mul += 1;
        profile.trunc += 1;
        let mean_row = repeat_rows(&mean, cols)?;
        let centered = row_major.sub(&mean_row)?;

        let squares = if cfg.approximate_truncation {
            self.mul_fixed_nonnegative_approx(&centered, &centered, rng)?
        } else {
            self.mul_fixed_nonnegative(&centered, &centered, rng)?
        };
        profile.poly_mul += 1;
        profile.trunc += 1;
        let var_sum = sum_rows(&squares, rows, cols)?;
        let mut variance = self.mul_public_fixed_nonnegative(&var_sum, inv_n, rng)?;
        profile.poly_mul += 1;
        profile.trunc += 1;
        if cfg.epsilon != 0.0 {
            variance =
                variance.add_public_to_party0(&vec![self.fixed.encode_f64(cfg.epsilon); rows])?;
        }

        let inv_sqrt = self.rsqrt_newton_profiled(&variance, cfg, rng)?;
        profile.add_assign(inv_sqrt.profile);
        let inv_sqrt_row = repeat_rows(&inv_sqrt.shares, cols)?;
        let normalized = self.mul_fixed(&centered, &inv_sqrt_row, rng)?;
        profile.poly_mul += 1;
        profile.trunc += 1;

        let gamma = expand_row_parameter(&cfg.gamma, rows, cols, 1.0)?;
        let beta = expand_row_parameter(&cfg.beta, rows, cols, 0.0)?;
        let affine = if gamma.iter().all(|&value| value == 1.0) {
            normalized
        } else {
            profile.poly_mul += 1;
            profile.trunc += 1;
            self.mul_public_fixed_slots(&normalized, &gamma, rng)?
        };
        let shares = if beta.iter().all(|&value| value == 0.0) {
            affine
        } else {
            affine.add_public_to_party0(
                &beta
                    .iter()
                    .map(|&v| self.fixed.encode_f64(v))
                    .collect::<Vec<_>>(),
            )?
        };
        profile.online_ms = start.elapsed().as_millis() as u64;
        Ok(ProfiledShares { shares, profile })
    }

    /// GeLU using the paper's public piecewise approximation:
    /// branch selectors come from private point-function/private comparison comparisons, polynomial
    /// products go through HSS multiplication, and scale restoration uses the
    /// public LUT selection interface.
    pub fn gelu<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &GeluConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(input)?;
        if cfg.t1 > cfg.t2 {
            return Err(OperatorError::InvalidParams("GeLU requires t1 <= t2"));
        }

        if self.fixed.modulus as usize > self.lookup.config().max_domain {
            return self.gelu_horner(input, cfg, rng);
        }

        let t1_scaled = self.fixed.signed_threshold(cfg.t1);
        let t2_scaled = self.fixed.signed_threshold(cfg.t2);
        let le_t2 = self.signed_less_than_scaled(input, t2_scaled + 1, rng)?;
        let gt_t2 = one_minus(&le_t2)?;
        let selected_poly = self.bounded_scaled_lookup(
            input,
            t1_scaled,
            t2_scaled,
            |scaled| self.gelu_polynomial_value(scaled, cfg),
            rng,
        )?;
        let selected_x = self.mul_selector(&gt_t2, input, rng)?;
        selected_poly.add(&selected_x)
    }

    fn gelu_horner<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &GeluConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let thresholds = [
            self.fixed.signed_threshold(cfg.t1),
            self.fixed.signed_threshold(cfg.t2) + 1,
        ];
        let selectors = self.signed_less_than_scaled_many(input, &thresholds, rng)?;
        let lt_t1 = selectors[0].clone();
        let le_t2 = selectors[1].clone();
        let gt_t2 = one_minus(&le_t2)?;
        let mid = le_t2.sub(&lt_t1)?;

        let mut poly = self.mul_public_fixed(input, cfg.coeffs[3], rng)?;
        poly =
            poly.add_public_to_party0(&vec![self.fixed.encode_f64(cfg.coeffs[2]); input.len()])?;
        poly = self.mul_fixed(&poly, input, rng)?;
        poly =
            poly.add_public_to_party0(&vec![self.fixed.encode_f64(cfg.coeffs[1]); input.len()])?;
        poly = self.mul_fixed(&poly, input, rng)?;
        poly =
            poly.add_public_to_party0(&vec![self.fixed.encode_f64(cfg.coeffs[0]); input.len()])?;

        let selected_poly = self.mul_selector(&mid, &poly, rng)?;
        let selected_x = self.mul_selector(&gt_t2, input, rng)?;
        selected_poly.add(&selected_x)
    }

    fn nidcf_select<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        thresholds: &[i128],
        rng: &mut R,
    ) -> Result<(Vec<AdditiveShares>, CadscMultiPlan), OperatorError> {
        validate_strict_thresholds(thresholds)?;
        let plan = self.cadsc_multi_compile(thresholds)?;
        let comparisons = self.signed_less_than_scaled_many(input, thresholds, rng)?;
        let mut selectors = Vec::with_capacity(thresholds.len() + 1);
        selectors.push(comparisons[0].clone());
        for idx in 1..comparisons.len() {
            selectors.push(comparisons[idx].sub(&comparisons[idx - 1])?);
        }
        selectors.push(one_minus(comparisons.last().ok_or_else(|| {
            OperatorError::Backend("NIDCF.Select returned no comparisons".to_string())
        })?)?);
        Ok((selectors, plan))
    }

    fn nidcf_select_known_nonpositive<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        thresholds: &[i128],
        rng: &mut R,
    ) -> Result<(Vec<AdditiveShares>, CadscMultiPlan), OperatorError> {
        validate_strict_thresholds(thresholds)?;
        let plan = self.cadsc_multi_compile(thresholds)?;
        let comparisons =
            self.signed_less_than_scaled_many_known_nonpositive(input, thresholds, rng)?;
        let mut selectors = Vec::with_capacity(thresholds.len() + 1);
        selectors.push(comparisons[0].clone());
        for idx in 1..comparisons.len() {
            selectors.push(comparisons[idx].sub(&comparisons[idx - 1])?);
        }
        selectors.push(one_minus(comparisons.last().ok_or_else(|| {
            OperatorError::Backend("NIDCF.Select returned no comparisons".to_string())
        })?)?);
        Ok((selectors, plan))
    }

    fn sac_select_coefficients(
        &self,
        selectors: &[AdditiveShares],
        coeff_rows: &[Vec<f64>],
    ) -> Result<Vec<AdditiveShares>, OperatorError> {
        let Some(first_selector) = selectors.first() else {
            return Err(OperatorError::InvalidParams(
                "SAC coefficient selection requires selectors",
            ));
        };
        if selectors.len() != coeff_rows.len() {
            return Err(OperatorError::InvalidParams(
                "SAC coefficient rows must match selector count",
            ));
        }
        let Some(first_row) = coeff_rows.first() else {
            return Err(OperatorError::InvalidParams(
                "SAC coefficient selection requires coefficient rows",
            ));
        };
        if first_row.is_empty() || coeff_rows.iter().any(|row| row.len() != first_row.len()) {
            return Err(OperatorError::InvalidParams(
                "SAC coefficient rows must be non-empty and equal length",
            ));
        }
        if selectors
            .iter()
            .any(|selector| selector.len() != first_selector.len())
        {
            return Err(OperatorError::InvalidParams(
                "SAC selectors must have equal lengths",
            ));
        }

        let mut selected = Vec::with_capacity(first_row.len());
        for coeff_idx in 0..first_row.len() {
            let mut acc =
                AdditiveShares::share_public(&vec![0; first_selector.len()], self.fixed.modulus)?;
            for (selector, coeffs) in selectors.iter().zip(coeff_rows.iter()) {
                let encoded = self.fixed.encode_f64(coeffs[coeff_idx]);
                if encoded != 0 {
                    acc = acc.add(&selector.mul_public_scalar(encoded)?)?;
                }
            }
            selected.push(acc);
        }
        Ok(selected)
    }

    fn sac_select_constants(
        &self,
        selectors: &[AdditiveShares],
        values: &[f64],
    ) -> Result<AdditiveShares, OperatorError> {
        if selectors.len() != values.len() {
            return Err(OperatorError::InvalidParams(
                "SAC constant selection value count must match selector count",
            ));
        }
        let Some(first_selector) = selectors.first() else {
            return Err(OperatorError::InvalidParams(
                "SAC constant selection requires selectors",
            ));
        };
        let mut acc =
            AdditiveShares::share_public(&vec![0; first_selector.len()], self.fixed.modulus)?;
        for (selector, &value) in selectors.iter().zip(values.iter()) {
            let encoded = self.fixed.encode_f64(value);
            if encoded != 0 {
                acc = acc.add(&selector.mul_public_scalar(encoded)?)?;
            }
        }
        Ok(acc)
    }

    fn sac_eval_poly<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        coeffs: &[AdditiveShares],
        approximate_truncation: bool,
        rng: &mut R,
    ) -> Result<ProfiledShares, OperatorError> {
        let Some((last, rest)) = coeffs.split_last() else {
            return Err(OperatorError::InvalidParams(
                "SAC polynomial requires at least one coefficient",
            ));
        };
        let mut profile = NonlinearProfile::default();
        let mut y = last.clone();
        for coeff in rest.iter().rev() {
            y = if approximate_truncation {
                self.mul_fixed_approx(input, &y, rng)?
            } else {
                self.mul_fixed(input, &y, rng)?
            };
            profile.poly_mul += 1;
            profile.trunc += 1;
            y = y.add(coeff)?;
        }
        Ok(ProfiledShares { shares: y, profile })
    }

    fn range_exp_profiled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &RangeSoftmaxConfig,
        rng: &mut R,
    ) -> Result<ProfiledShares, OperatorError> {
        let mut thresholds = Vec::with_capacity(cfg.exp_thresholds.len() + 2);
        thresholds.push(self.fixed.signed_threshold(cfg.exp_clip_min));
        for &threshold in &cfg.exp_thresholds {
            thresholds.push(self.fixed.signed_threshold(threshold));
        }
        thresholds.push(self.fixed.signed_threshold(cfg.exp_clip_max));
        validate_strict_thresholds(&thresholds)?;

        let (selectors, plan) =
            if cfg.use_row_max && thresholds.iter().all(|&threshold| threshold <= 0) {
                self.nidcf_select_known_nonpositive(input, &thresholds, rng)?
            } else {
                self.nidcf_select(input, &thresholds, rng)?
            };
        let degree = cfg
            .exp_coeffs
            .first()
            .ok_or(OperatorError::InvalidParams(
                "RangeSoftmax requires exponent coefficients",
            ))?
            .len();
        if degree == 0 || cfg.exp_coeffs.iter().any(|row| row.len() != degree) {
            return Err(OperatorError::InvalidParams(
                "RangeSoftmax exponent coefficient rows must be non-empty and equal length",
            ));
        }
        let mut coeff_rows = vec![vec![0.0; degree]; selectors.len()];
        coeff_rows[0][0] = cfg.exp_clip_min.exp();
        for (idx, coeffs) in cfg.exp_coeffs.iter().enumerate() {
            coeff_rows[idx + 1] = coeffs.clone();
        }
        let last = selectors.len() - 1;
        coeff_rows[last][0] = cfg.exp_clip_max.exp();
        let selected_coeffs = self.sac_select_coefficients(&selectors, &coeff_rows)?;
        let mut eval = self.sac_eval_poly(input, &selected_coeffs, false, rng)?;
        eval.profile.selectors += plan.selector_count;
        Ok(eval)
    }

    fn row_reciprocal_newton_profiled<R: RngCore + ?Sized>(
        &self,
        sum: &AdditiveShares,
        cols: usize,
        rng: &mut R,
    ) -> Result<ProfiledShares, OperatorError> {
        let mut thresholds = Vec::new();
        let mut next = 2usize;
        while next < cols {
            thresholds.push(self.fixed.signed_threshold(next as f64));
            next = next.saturating_mul(2);
        }
        thresholds.push(self.fixed.signed_threshold(cols as f64));
        thresholds.sort_unstable();
        thresholds.dedup();
        validate_strict_thresholds(&thresholds)?;

        let (selectors, plan) = self.nidcf_select(sum, &thresholds, rng)?;
        let mut seeds = Vec::with_capacity(selectors.len());
        let mut lo = 1.0;
        for &threshold in &thresholds {
            let hi = threshold as f64 / self.fixed.scale as f64;
            seeds.push(1.0 / ((lo + hi) * 0.5));
            lo = hi;
        }
        seeds.push(1.0 / cols as f64);
        let r0 = self.sac_select_constants(&selectors, &seeds)?;

        let sr0 = self.mul_fixed_nonnegative(sum, &r0, rng)?;
        let two_minus = AdditiveShares::share_public(
            &vec![self.fixed.encode_f64(2.0); sum.len()],
            self.fixed.modulus,
        )?
        .sub(&sr0)?;
        let r1 = self.mul_fixed_nonnegative(&r0, &two_minus, rng)?;
        Ok(ProfiledShares {
            shares: r1,
            profile: NonlinearProfile {
                selectors: plan.selector_count,
                poly_mul: 2,
                trunc: 2,
                ..NonlinearProfile::default()
            },
        })
    }

    fn rsqrt_newton_profiled<R: RngCore + ?Sized>(
        &self,
        variance: &AdditiveShares,
        cfg: &BinadeLayerNormConfig,
        rng: &mut R,
    ) -> Result<ProfiledShares, OperatorError> {
        let thresholds = cfg
            .binade_thresholds
            .iter()
            .map(|&threshold| self.fixed.signed_threshold(threshold))
            .collect::<Vec<_>>();
        validate_strict_thresholds(&thresholds)?;
        let (selectors, plan) = self.nidcf_select(variance, &thresholds, rng)?;
        let r0 = self.sac_select_constants(&selectors, &cfg.rsqrt_seeds)?;

        let r0_sq = self.mul_fixed_nonnegative(&r0, &r0, rng)?;
        let vr0_sq = self.mul_fixed_nonnegative(variance, &r0_sq, rng)?;
        let half_vr0_sq = self.mul_public_fixed_nonnegative(&vr0_sq, 0.5, rng)?;
        let term = AdditiveShares::share_public(
            &vec![self.fixed.encode_f64(1.5); variance.len()],
            self.fixed.modulus,
        )?
        .sub(&half_vr0_sq)?;
        let r1 = self.mul_fixed_nonnegative(&r0, &term, rng)?;
        Ok(ProfiledShares {
            shares: r1,
            profile: NonlinearProfile {
                selectors: plan.selector_count,
                poly_mul: 4,
                trunc: 4,
                ..NonlinearProfile::default()
            },
        })
    }

    fn ensure_refinement_headroom(&self, op: &'static str) -> Result<(), OperatorError> {
        let scale_sq = (self.fixed.scale as u128) * (self.fixed.scale as u128);
        if scale_sq.saturating_mul(4) >= self.fixed.modulus as u128 {
            return Err(OperatorError::InvalidParams(match op {
                "RangeSoftmax" => {
                    "RangeSoftmax requires 4 * scale^2 < modulus for Newton refinement"
                }
                "BinadeLayerNorm" => {
                    "BinadeLayerNorm requires 4 * scale^2 < modulus for Newton refinement"
                }
                _ => "specialized nonlinear refinement scale headroom check failed",
            }));
        }
        Ok(())
    }

    pub fn softmax<R: RngCore + ?Sized>(
        &self,
        row: &AdditiveShares,
        cfg: &SoftmaxConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.softmax_rows(row, 1, row.len(), cfg, rng)
    }

    pub fn softmax_approx_trunc<R: RngCore + ?Sized>(
        &self,
        row: &AdditiveShares,
        cfg: &SoftmaxConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.softmax_rows_approx_trunc(row, 1, row.len(), cfg, rng)
    }

    pub fn softmax_rows<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &SoftmaxConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.softmax_rows_inner(row_major, rows, cols, cfg, false, rng)
    }

    pub fn softmax_rows_approx_trunc<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &SoftmaxConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.softmax_rows_inner(row_major, rows, cols, cfg, true, rng)
    }

    fn softmax_rows_inner<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &SoftmaxConfig,
        approx_trunc: bool,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "softmax matrix dimensions must be non-empty",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "softmax matrix length must equal rows * cols",
            ));
        }
        self.validate_shares(row_major)?;
        if cfg.exp_clip_min > cfg.exp_clip_max {
            return Err(OperatorError::InvalidParams(
                "softmax exp clip min must be <= max",
            ));
        }

        let max = self.row_max_matrix(row_major, rows, cols, rng)?;
        let max_row = repeat_rows(&max, cols)?;
        let shifted = row_major.sub(&max_row)?;
        let exp_shifted = self.exp_lut_nonpositive(&shifted, cfg, rng)?;
        let sum_exp = sum_rows(&exp_shifted, rows, cols)?;
        let inv_sum =
            self.reciprocal_lut_bounded(&sum_exp, cols as i128 * self.fixed.scale as i128, rng)?;
        let inv_sum_row = repeat_rows(&inv_sum, cols)?;
        if approx_trunc {
            self.mul_fixed_nonnegative_approx(&exp_shifted, &inv_sum_row, rng)
        } else {
            self.mul_fixed_nonnegative(&exp_shifted, &inv_sum_row, rng)
        }
    }

    pub fn softmax_rows_polynomial_exp<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &SoftmaxConfig,
        exp_squarings: usize,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "softmax matrix dimensions must be non-empty",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "softmax matrix length must equal rows * cols",
            ));
        }
        self.validate_shares(row_major)?;
        if cfg.exp_clip_min > cfg.exp_clip_max {
            return Err(OperatorError::InvalidParams(
                "softmax exp clip min must be <= max",
            ));
        }

        let max = self.row_max_matrix(row_major, rows, cols, rng)?;
        let max_row = repeat_rows(&max, cols)?;
        let shifted = row_major.sub(&max_row)?;
        let exp_shifted = self.exp_repeated_square(&shifted, cfg, exp_squarings, rng)?;
        let sum_exp = sum_rows(&exp_shifted, rows, cols)?;
        let inv_sum =
            self.reciprocal_lut_bounded(&sum_exp, cols as i128 * self.fixed.scale as i128, rng)?;
        let inv_sum_row = repeat_rows(&inv_sum, cols)?;
        self.mul_fixed_nonnegative(&exp_shifted, &inv_sum_row, rng)
    }

    pub fn softmax_rows_of_pmpe_exp_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &SoftmaxConfig,
        exp_cfg: &OfPmpePolynomialConfig,
        approx_trunc: bool,
        rng: &mut R,
    ) -> Result<OfPmpeSoftmaxOutput, OperatorError> {
        let start = Instant::now();
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE softmax matrix dimensions must be non-empty",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE softmax matrix length must equal rows * cols",
            ));
        }
        self.validate_shares(row_major)?;
        if cfg.exp_clip_min > cfg.exp_clip_max {
            return Err(OperatorError::InvalidParams(
                "OF-PMPE softmax exp clip min must be <= max",
            ));
        }

        let max = self.row_max_matrix(row_major, rows, cols, rng)?;
        let max_row = repeat_rows(&max, cols)?;
        let shifted = row_major.sub(&max_row)?;
        let exp_shifted = self.exp_of_pmpe_profiled(&shifted, exp_cfg, rng)?;
        let sum_exp = sum_rows(&exp_shifted.shares, rows, cols)?;
        let inv_sum =
            self.reciprocal_lut_bounded(&sum_exp, cols as i128 * self.fixed.scale as i128, rng)?;
        let inv_sum_row = repeat_rows(&inv_sum, cols)?;
        let shares = if approx_trunc {
            self.mul_fixed_nonnegative_approx(&exp_shifted.shares, &inv_sum_row, rng)?
        } else {
            self.mul_fixed_nonnegative(&exp_shifted.shares, &inv_sum_row, rng)?
        };
        Ok(OfPmpeSoftmaxOutput {
            shares,
            exp_profile: exp_shifted.profile,
            online_ms: start.elapsed().as_millis() as u64,
        })
    }

    pub fn softmax_rows_dyadic_of_pmpe_exp_phase_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &SoftmaxConfig,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeSoftmaxExpOutput, OperatorError> {
        let start = Instant::now();
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE softmax dimensions must be non-empty",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE softmax length must equal rows * cols",
            ));
        }
        self.validate_shares(row_major)?;
        if cfg.exp_clip_min > cfg.exp_clip_max {
            return Err(OperatorError::InvalidParams(
                "dyadic OF-PMPE softmax exp clip min must be <= max",
            ));
        }

        let max = self.row_max_matrix(row_major, rows, cols, rng)?;
        let max_row = repeat_rows(&max, cols)?;
        let shifted = row_major.sub(&max_row)?;
        let exp = self.dyadic_exp_of_pmpe_profiled(&shifted, exp_cfg, rng)?;
        let row_sums = sum_rows(&exp.tensor.shares, rows, cols)?;
        Ok(DyadicOfPmpeSoftmaxExpOutput {
            shifted,
            exp: exp.tensor.clone(),
            row_sums: ScaledShareTensor {
                shares: row_sums,
                frac_bits: exp.tensor.frac_bits,
                guard_bits: exp.tensor.guard_bits,
                semantic: exp.tensor.semantic,
            },
            exp_profile: exp.profile,
            exp_scale_profile: exp.scale_profile,
            online_ms: start.elapsed().as_millis() as u64,
        })
    }

    pub fn softmax_rows_dyadic_of_pmpe_reciprocal_phase_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &SoftmaxConfig,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        target_frac_bits: u32,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeSoftmaxReciprocalOutput, OperatorError> {
        let start = Instant::now();
        let exp_phase = self.softmax_rows_dyadic_of_pmpe_exp_phase_profiled(
            row_major, rows, cols, cfg, exp_cfg, rng,
        )?;
        if exp_phase.exp_scale_profile.modulus_wrap_safe == Some(false) {
            return Err(OperatorError::InvalidParams(
                "dyadic softmax reciprocal phase requires a no-wrap exp denominator",
            ));
        }
        let rescale =
            self.rescale_dyadic_nonnegative_profiled(&exp_phase.row_sums, target_frac_bits, rng)?;
        let recip_start = Instant::now();
        let inv = self.reciprocal_lut_bounded(
            &rescale.tensor.shares,
            cols as i128 * self.fixed.scale as i128,
            rng,
        )?;
        let recip_elapsed = recip_start.elapsed();
        let elapsed = start.elapsed();
        Ok(DyadicOfPmpeSoftmaxReciprocalOutput {
            inv_row_sums: ScaledShareTensor {
                shares: inv,
                frac_bits: target_frac_bits,
                guard_bits: 0,
                semantic: ScaleSemantic::DyadicNumerator,
            },
            row_sums_base: rescale.tensor.clone(),
            exp_phase,
            rescale,
            reciprocal_ms: recip_elapsed.as_millis() as u64,
            reciprocal_us: recip_elapsed.as_micros() as u64,
            online_ms: elapsed.as_millis() as u64,
            online_us: elapsed.as_micros() as u64,
        })
    }

    pub fn softmax_rows_dyadic_of_pmpe_profiled<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &SoftmaxConfig,
        exp_cfg: &DyadicOfPmpePolynomialConfig,
        denominator_frac_bits: u32,
        output_frac_bits: u32,
        rng: &mut R,
    ) -> Result<DyadicOfPmpeSoftmaxOutput, OperatorError> {
        let start = Instant::now();
        let reciprocal_phase = self.softmax_rows_dyadic_of_pmpe_reciprocal_phase_profiled(
            row_major,
            rows,
            cols,
            cfg,
            exp_cfg,
            denominator_frac_bits,
            rng,
        )?;
        let inv_row = repeat_rows(&reciprocal_phase.inv_row_sums.shares, cols)?;
        let inv_row = ScaledShareTensor {
            shares: inv_row,
            frac_bits: reciprocal_phase.inv_row_sums.frac_bits,
            guard_bits: reciprocal_phase.inv_row_sums.guard_bits,
            semantic: reciprocal_phase.inv_row_sums.semantic,
        };
        let broadcast_mul = self.mul_scaled_nonnegative_profiled(
            &reciprocal_phase.exp_phase.exp,
            &inv_row,
            output_frac_bits,
            rng,
        )?;
        let elapsed = start.elapsed();
        Ok(DyadicOfPmpeSoftmaxOutput {
            probs: broadcast_mul.tensor.clone(),
            reciprocal_phase,
            broadcast_mul,
            online_ms: elapsed.as_millis() as u64,
            online_us: elapsed.as_micros() as u64,
        })
    }

    pub fn layer_norm<R: RngCore + ?Sized>(
        &self,
        row: &AdditiveShares,
        cfg: &LayerNormConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.layer_norm_rows(row, 1, row.len(), cfg, rng)
    }

    pub fn layer_norm_approx_trunc<R: RngCore + ?Sized>(
        &self,
        row: &AdditiveShares,
        cfg: &LayerNormConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.layer_norm_rows_approx_trunc(row, 1, row.len(), cfg, rng)
    }

    pub fn layer_norm_rows<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &LayerNormConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.layer_norm_rows_inner(row_major, rows, cols, cfg, false, rng)
    }

    pub fn layer_norm_rows_approx_trunc<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &LayerNormConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.layer_norm_rows_inner(row_major, rows, cols, cfg, true, rng)
    }

    fn layer_norm_rows_inner<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        cfg: &LayerNormConfig,
        approx_trunc: bool,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(row_major)?;
        if rows == 0 || cols == 0 {
            return Err(OperatorError::InvalidParams(
                "LayerNorm matrix dimensions must be non-empty",
            ));
        }
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "LayerNorm matrix length must equal rows * cols",
            ));
        }
        if !cfg.gamma.is_empty() && cfg.gamma.len() != cols && cfg.gamma.len() != row_major.len() {
            return Err(OperatorError::InvalidParams(
                "LayerNorm gamma length must match columns or rows * cols",
            ));
        }
        if !cfg.beta.is_empty() && cfg.beta.len() != cols && cfg.beta.len() != row_major.len() {
            return Err(OperatorError::InvalidParams(
                "LayerNorm beta length must match columns or rows * cols",
            ));
        }

        let inv_n = 1.0 / cols as f64;
        let sum = sum_rows(row_major, rows, cols)?;
        let mean = self.mul_public_fixed(&sum, inv_n, rng)?;
        let mean_row = repeat_rows(&mean, cols)?;
        let centered = row_major.sub(&mean_row)?;

        let squares = if approx_trunc {
            self.mul_fixed_nonnegative_approx(&centered, &centered, rng)?
        } else {
            self.mul_fixed_nonnegative(&centered, &centered, rng)?
        };
        let var_sum = sum_rows(&squares, rows, cols)?;
        let mut variance = self.mul_public_fixed_nonnegative(&var_sum, inv_n, rng)?;
        if cfg.epsilon != 0.0 {
            variance =
                variance.add_public_to_party0(&vec![self.fixed.encode_f64(cfg.epsilon); rows])?;
        }
        let inv_sqrt = self.rsqrt_lut_bounded(&variance, cfg.variance_clip_max, rng)?;
        let inv_sqrt_row = repeat_rows(&inv_sqrt, cols)?;
        let normalized = self.mul_fixed(&centered, &inv_sqrt_row, rng)?;

        if is_row_parameter_value(&cfg.gamma, rows, cols, 1.0)
            && is_row_parameter_value(&cfg.beta, rows, cols, 0.0)
        {
            return Ok(normalized);
        }

        let gamma = expand_row_parameter(&cfg.gamma, rows, cols, 1.0)?;
        let beta = expand_row_parameter(&cfg.beta, rows, cols, 0.0)?;
        let affine = if gamma.iter().all(|&value| value == 1.0) {
            normalized
        } else {
            self.mul_public_fixed_slots(&normalized, &gamma, rng)?
        };
        if beta.iter().all(|&value| value == 0.0) {
            Ok(affine)
        } else {
            affine.add_public_to_party0(
                &beta
                    .iter()
                    .map(|&v| self.fixed.encode_f64(v))
                    .collect::<Vec<_>>(),
            )
        }
    }

    pub fn signed_less_than_public<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        threshold: f64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(input)?;
        self.signed_less_than_scaled(input, self.fixed.signed_threshold(threshold), rng)
    }

    pub fn signed_less_equal_public<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        threshold: f64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(input)?;
        self.signed_less_than_scaled(input, self.fixed.signed_threshold(threshold) + 1, rng)
    }

    pub fn mul_fixed<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let raw = self.hss.mul_slots(lhs, rhs, rng)?;
        self.rescale_fixed(&raw, rng)
    }

    pub fn mul_fixed_approx<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let raw = self.hss.mul_slots(lhs, rhs, rng)?;
        self.rescale_fixed_approx(&raw, rng)
    }

    fn mul_fixed_with_product_sign_approx<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        product_sign: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let raw = self.hss.mul_slots(lhs, rhs, rng)?;
        self.rescale_fixed_with_sign_approx(&raw, product_sign, rng)
    }

    pub fn mul_fixed_nonnegative<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let raw = self.hss.mul_slots(lhs, rhs, rng)?;
        self.rescale_fixed_nonnegative(&raw, rng)
    }

    /// Multiplies values known to have a non-negative mathematical product and
    /// applies a one-bit approximate truncation. This keeps the modular wrap
    /// correction, but intentionally skips the final carry/remainder correction
    /// used by the exact fixed-point rescale.
    pub fn mul_fixed_nonnegative_approx<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let raw = self.hss.mul_slots(lhs, rhs, rng)?;
        self.rescale_fixed_nonnegative_approx(&raw, rng)
    }

    pub fn dot_fixed<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        group_size: usize,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(lhs)?;
        self.validate_shares(rhs)?;
        if group_size == 0 {
            return Err(OperatorError::InvalidParams(
                "fixed dot-product group_size must be positive",
            ));
        }
        if lhs.len() != rhs.len() || lhs.len() % group_size != 0 {
            return Err(OperatorError::InvalidParams(
                "fixed dot-product inputs must have equal length divisible by group_size",
            ));
        }
        let raw = self.hss.mul_slots(lhs, rhs, rng)?;
        let groups = lhs.len() / group_size;
        let summed = sum_rows(&raw, groups, group_size)?;
        self.rescale_fixed(&summed, rng)
    }

    fn mul_selector<R: RngCore + ?Sized>(
        &self,
        selector: &AdditiveShares,
        value: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.hss.mul_slots(selector, value, rng)
    }

    fn mul_public_fixed<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        scalar: f64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let encoded = self.fixed.encode_f64(scalar);
        let raw = input.mul_public_scalar(encoded)?;
        self.rescale_fixed(&raw, rng)
    }

    fn mul_public_fixed_nonnegative<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        scalar: f64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if scalar < 0.0 {
            return Err(OperatorError::InvalidParams(
                "nonnegative public fixed multiply requires a nonnegative scalar",
            ));
        }
        let encoded = self.fixed.encode_f64(scalar);
        let raw = input.mul_public_scalar(encoded)?;
        self.rescale_fixed_nonnegative(&raw, rng)
    }

    fn mul_public_fixed_slots<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        scalars: &[f64],
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if input.len() != scalars.len() {
            return Err(OperatorError::InvalidParams(
                "public scalar vector length must match shares",
            ));
        }
        let p = self.fixed.modulus;
        let mut p0 = Vec::with_capacity(input.len());
        let mut p1 = Vec::with_capacity(input.len());
        for ((&x0, &x1), &scalar) in input
            .party0()
            .iter()
            .zip(input.party1().iter())
            .zip(scalars.iter())
        {
            let encoded = self.fixed.encode_f64(scalar);
            p0.push(mul_mod(x0, encoded, p));
            p1.push(mul_mod(x1, encoded, p));
        }
        self.rescale_fixed(&AdditiveShares::new(p, p0, p1)?, rng)
    }

    fn row_max_matrix<R: RngCore + ?Sized>(
        &self,
        row_major: &AdditiveShares,
        rows: usize,
        cols: usize,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if row_major.len() != rows.saturating_mul(cols) {
            return Err(OperatorError::InvalidParams(
                "row_max matrix length must equal rows * cols",
            ));
        }
        if cols == 0 {
            return Err(OperatorError::InvalidParams(
                "row_max matrix must have at least one column",
            ));
        }
        let mut current = row_major.clone();
        let mut current_cols = cols;

        while current_cols > 1 {
            let pairs = current_cols / 2;
            let tail = current_cols % 2;
            let mut lhs0 = Vec::with_capacity(rows * pairs);
            let mut lhs1 = Vec::with_capacity(rows * pairs);
            let mut rhs0 = Vec::with_capacity(rows * pairs);
            let mut rhs1 = Vec::with_capacity(rows * pairs);

            for row in 0..rows {
                let row_offset = row * current_cols;
                for pair in 0..pairs {
                    let left = row_offset + pair * 2;
                    let right = left + 1;
                    lhs0.push(current.party0()[left]);
                    lhs1.push(current.party1()[left]);
                    rhs0.push(current.party0()[right]);
                    rhs1.push(current.party1()[right]);
                }
            }

            let lhs = AdditiveShares::new(self.fixed.modulus, lhs0, lhs1)?;
            let rhs = AdditiveShares::new(self.fixed.modulus, rhs0, rhs1)?;
            let lt = self.signed_less_than_shares(&lhs, &rhs, rng)?;
            let diff = rhs.sub(&lhs)?;
            let inc = self.mul_selector(&lt, &diff, rng)?;
            let pair_max = lhs.add(&inc)?;

            let next_cols = pairs + tail;
            let mut next0 = Vec::with_capacity(rows * next_cols);
            let mut next1 = Vec::with_capacity(rows * next_cols);
            for row in 0..rows {
                for pair in 0..pairs {
                    let idx = row * pairs + pair;
                    next0.push(pair_max.party0()[idx]);
                    next1.push(pair_max.party1()[idx]);
                }
                if tail == 1 {
                    let idx = row * current_cols + current_cols - 1;
                    next0.push(current.party0()[idx]);
                    next1.push(current.party1()[idx]);
                }
            }
            current = AdditiveShares::new(self.fixed.modulus, next0, next1)?;
            current_cols = next_cols;
        }

        if current.len() != rows {
            return Err(OperatorError::Backend(
                "row_max tree reduced to an unexpected shape".to_string(),
            ));
        }
        Ok(current)
    }

    fn signed_less_than_shares<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let diff = lhs.sub(rhs)?;
        self.signed_less_than_scaled(&diff, 0, rng)
    }

    fn signed_less_than_scaled<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        threshold: i128,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let mut outputs = self.signed_less_than_scaled_many(input, &[threshold], rng)?;
        outputs.pop().ok_or_else(|| {
            OperatorError::Backend("signed comparison returned no output".to_string())
        })
    }

    fn signed_less_than_scaled_many<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        thresholds: &[i128],
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError> {
        if thresholds.is_empty() {
            return Err(OperatorError::InvalidParams(
                "signed comparison requires at least one threshold",
            ));
        }
        let half = (self.fixed.modulus / 2) as i128;
        let p = self.fixed.modulus;
        let mut outputs = vec![None; thresholds.len()];
        let mut mod_thresholds = Vec::new();
        let mut needs_sign = false;
        for (idx, &threshold) in thresholds.iter().enumerate() {
            if threshold <= -half {
                outputs[idx] = Some(AdditiveShares::share_public(&vec![0; input.len()], p)?);
            } else if threshold > half {
                outputs[idx] = Some(AdditiveShares::share_public(&vec![1; input.len()], p)?);
            } else if threshold == 0 {
                needs_sign = true;
            } else if threshold > 0 {
                needs_sign = true;
                mod_thresholds.push((idx, threshold as u64, false));
            } else {
                needs_sign = true;
                mod_thresholds.push((idx, (p as i128 + threshold) as u64, true));
            }
        }

        let sign = if needs_sign {
            Some(self.sign_bit(input, rng)?)
        } else {
            None
        };
        for (idx, &threshold) in thresholds.iter().enumerate() {
            if threshold == 0 {
                outputs[idx] = sign.clone();
            }
        }
        if !mod_thresholds.is_empty() {
            let lt_mod = self.less_than_mod_public_many(
                input,
                &mod_thresholds
                    .iter()
                    .map(|(_, threshold, _)| *threshold)
                    .collect::<Vec<_>>(),
                rng,
            )?;
            let sign = sign.as_ref().ok_or_else(|| {
                OperatorError::Backend("signed comparison sign bit was not computed".to_string())
            })?;
            for ((idx, _, is_negative_threshold), lt) in mod_thresholds.into_iter().zip(lt_mod) {
                outputs[idx] = Some(if is_negative_threshold {
                    self.lookup.bit_and(sign, &lt, rng)?
                } else {
                    sign.add(&lt)?
                });
            }
        }

        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    OperatorError::Backend(
                        "signed comparison output was not initialized".to_string(),
                    )
                })
            })
            .collect()
    }

    fn signed_less_than_scaled_many_known_nonpositive<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        thresholds: &[i128],
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError> {
        if thresholds.is_empty() {
            return Err(OperatorError::InvalidParams(
                "signed comparison requires at least one threshold",
            ));
        }
        self.validate_shares(input)?;
        let half = (self.fixed.modulus / 2) as i128;
        let p = self.fixed.modulus;
        let mut outputs = vec![None; thresholds.len()];
        let mut active = Vec::new();
        let mut needs_nonzero_sign = false;
        for (idx, &threshold) in thresholds.iter().enumerate() {
            if threshold <= -half {
                outputs[idx] = Some(AdditiveShares::share_public(&vec![0; input.len()], p)?);
            } else if threshold > 0 {
                outputs[idx] = Some(AdditiveShares::share_public(&vec![1; input.len()], p)?);
            } else if threshold == 0 {
                needs_nonzero_sign = true;
            } else {
                needs_nonzero_sign = true;
                active.push((idx, (p as i128 + threshold) as u64));
            }
        }

        let sign_nonzero = if needs_nonzero_sign {
            let mut compare_thresholds = Vec::with_capacity(active.len() + 1);
            compare_thresholds.push(1);
            compare_thresholds.extend(active.iter().map(|(_, threshold)| *threshold));
            let comparisons = self.less_than_mod_public_many(input, &compare_thresholds, rng)?;
            let sign_nonzero = one_minus(&comparisons[0])?;
            for (active_idx, (output_idx, _)) in active.iter().enumerate() {
                outputs[*output_idx] = Some(self.lookup.bit_and(
                    &sign_nonzero,
                    &comparisons[active_idx + 1],
                    rng,
                )?);
            }
            Some(sign_nonzero)
        } else {
            None
        };
        for (idx, &threshold) in thresholds.iter().enumerate() {
            if threshold == 0 {
                outputs[idx] = sign_nonzero.clone();
            }
        }

        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    OperatorError::Backend(
                        "known-nonpositive signed comparison output was not initialized"
                            .to_string(),
                    )
                })
            })
            .collect()
    }

    fn sign_bit<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let p = self.fixed.modulus;
        let comparisons =
            self.less_than_sum_public_many(input, &[p / 2 + 1, p, p + p / 2 + 1], rng)?;
        let u = one_minus(&comparisons[0])?;
        let wrap = one_minus(&comparisons[1])?;
        let v = one_minus(&comparisons[2])?;
        u.sub(&wrap)?.add(&v)
    }

    fn less_than_mod_public_many<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        thresholds: &[u64],
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError> {
        let p = self.fixed.modulus;
        if thresholds.is_empty() {
            return Err(OperatorError::InvalidParams(
                "modular comparison requires at least one threshold",
            ));
        }
        let mut outputs = vec![None; thresholds.len()];
        let mut active = Vec::new();
        for (idx, &threshold) in thresholds.iter().enumerate() {
            if threshold == 0 {
                outputs[idx] = Some(AdditiveShares::share_public(&vec![0; input.len()], p)?);
            } else if threshold >= p {
                outputs[idx] = Some(AdditiveShares::share_public(&vec![1; input.len()], p)?);
            } else {
                active.push((idx, threshold));
            }
        }
        if !active.is_empty() {
            let mut sum_thresholds = Vec::with_capacity(active.len() * 2 + 1);
            for (_, threshold) in &active {
                sum_thresholds.push(*threshold);
            }
            sum_thresholds.push(p);
            for (_, threshold) in &active {
                sum_thresholds.push(p + *threshold);
            }
            let comparisons = self.less_than_sum_public_many(input, &sum_thresholds, rng)?;
            let wrap = one_minus(&comparisons[active.len()])?;
            for (active_idx, (output_idx, _)) in active.iter().enumerate() {
                let lt_without_wrap = comparisons[active_idx].clone();
                let lt_with_wrap = comparisons[active.len() + 1 + active_idx].clone();
                let wrapped_lt = self.lookup.bit_and(&wrap, &lt_with_wrap, rng)?;
                outputs[*output_idx] = Some(lt_without_wrap.add(&wrapped_lt)?);
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    OperatorError::Backend(
                        "modular comparison output was not initialized".to_string(),
                    )
                })
            })
            .collect()
    }

    fn less_than_sum_public_many<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        thresholds: &[u64],
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError> {
        let bit_width = sum_bit_width(self.fixed.modulus)?;
        let radix_bits = comparison_radix_bits(bit_width, self.fixed.modulus)?;
        self.lookup.less_than_public_radix_many(
            input.party0(),
            input.party1(),
            bit_width,
            thresholds,
            radix_bits,
            rng,
        )
    }

    fn exp_lut<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &SoftmaxConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let min_scaled = self.fixed.signed_threshold(cfg.exp_clip_min);
        let max_scaled = self.fixed.signed_threshold(cfg.exp_clip_max);
        let mid = self.bounded_centered_lookup(input, min_scaled, max_scaled, |x| x.exp(), rng)?;
        let clips = self.signed_less_than_scaled_many(input, &[min_scaled, max_scaled + 1], rng)?;
        let lt_min = clips[0].clone();
        let le_max = clips[1].clone();
        let gt_max = one_minus(&le_max)?;
        let low = lt_min.mul_public_scalar(self.fixed.encode_f64(cfg.exp_clip_min.exp()))?;
        let high = gt_max.mul_public_scalar(self.fixed.encode_f64(cfg.exp_clip_max.exp()))?;
        mid.add(&low)?.add(&high)
    }

    fn exp_lut_nonpositive<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &SoftmaxConfig,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if cfg.exp_clip_max != 0.0 {
            return self.exp_lut(input, cfg, rng);
        }

        let min_scaled = self.fixed.signed_threshold(cfg.exp_clip_min);
        let mid = self.bounded_centered_lookup(input, min_scaled, 0, |x| x.exp(), rng)?;
        let lt_min = self.signed_less_than_scaled(input, min_scaled, rng)?;
        let low = lt_min.mul_public_scalar(self.fixed.encode_f64(cfg.exp_clip_min.exp()))?;
        mid.add(&low)
    }

    fn exp_repeated_square<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        cfg: &SoftmaxConfig,
        exp_squarings: usize,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if exp_squarings == 0 || exp_squarings > 32 {
            return Err(OperatorError::InvalidParams(
                "softmax polynomial exp squarings must be in 1..=32",
            ));
        }
        let divisor =
            1u64.checked_shl(exp_squarings as u32)
                .ok_or(OperatorError::InvalidParams(
                    "softmax polynomial exp squaring divisor overflowed",
                ))?;
        if self.fixed.encode_f64(1.0 / divisor as f64) == 0 {
            return Err(OperatorError::InvalidParams(
                "fixed-point scale is too small for requested polynomial exp squarings",
            ));
        }
        if (self.fixed.scale as u128) * (self.fixed.scale as u128)
            >= (self.fixed.modulus / 2) as u128
        {
            return Err(OperatorError::InvalidParams(
                "softmax polynomial exp requires scale^2 < modulus / 2",
            ));
        }
        if cfg.exp_clip_min / (divisor as f64) < -1.0 {
            return Err(OperatorError::InvalidParams(
                "softmax polynomial exp needs clip_min / 2^squarings >= -1",
            ));
        }

        let min_scaled = self.fixed.signed_threshold(cfg.exp_clip_min);
        let max_scaled = self.fixed.signed_threshold(cfg.exp_clip_max);
        let clips = self.signed_less_than_scaled_many(input, &[min_scaled, max_scaled + 1], rng)?;
        let lt_min = clips[0].clone();
        let le_max = clips[1].clone();
        let gt_max = one_minus(&le_max)?;
        let mid = le_max.sub(&lt_min)?;

        let mut poly = self.mul_public_fixed(input, 1.0 / divisor as f64, rng)?;
        poly = poly.add_public_to_party0(&vec![self.fixed.encode_f64(1.0); input.len()])?;
        for _ in 0..exp_squarings {
            poly = self.mul_fixed_nonnegative(&poly, &poly, rng)?;
        }

        let selected_poly = self.mul_selector(&mid, &poly, rng)?;
        let low = lt_min.mul_public_scalar(self.fixed.encode_f64(cfg.exp_clip_min.exp()))?;
        let high = gt_max.mul_public_scalar(self.fixed.encode_f64(cfg.exp_clip_max.exp()))?;
        selected_poly.add(&low)?.add(&high)
    }

    fn reciprocal_lut_bounded<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        max_scaled: i128,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.bounded_centered_lookup(input, 1, max_scaled, |x| 1.0 / x, rng)
    }

    fn rsqrt_lut_bounded<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        max_value: f64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if max_value <= 0.0 {
            return Err(OperatorError::InvalidParams(
                "LayerNorm variance_clip_max must be positive",
            ));
        }
        let min_scaled = 1;
        let max_scaled = self.fixed.signed_threshold(max_value).max(min_scaled);
        let mid = self.bounded_centered_lookup(
            input,
            min_scaled,
            max_scaled,
            |x| {
                if x > 0.0 { 1.0 / x.sqrt() } else { 0.0 }
            },
            rng,
        )?;
        let clips = self.signed_less_than_scaled_many(input, &[min_scaled, max_scaled + 1], rng)?;
        let le_max = clips[1].clone();
        let gt_max = one_minus(&le_max)?;
        let high = gt_max.mul_public_scalar(self.fixed.encode_f64(1.0 / max_value.sqrt()))?;
        mid.add(&high)
    }

    fn rescale_fixed<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(input)?;
        let Some(corrections) = &self.rescale_corrections else {
            return self.rescale_lut(input, rng);
        };

        let p = self.fixed.modulus;
        let scale = self.fixed.scale;
        let p_quotient = p / scale;
        let bit_width = sum_bit_width(p)?;
        let radix_bits = comparison_radix_bits(bit_width, p)?;

        let comparisons = self.lookup.less_than_public_radix_many(
            input.party0(),
            input.party1(),
            bit_width,
            &[p / 2 + 1, p, p + p / 2 + 1],
            radix_bits,
            rng,
        )?;
        let u = one_minus(&comparisons[0])?;
        let wrap = one_minus(&comparisons[1])?;
        let v = one_minus(&comparisons[2])?;
        let sign = u.sub(&wrap)?.add(&v)?;

        let base = AdditiveShares::new(
            p,
            input.party0().iter().map(|&value| value / scale).collect(),
            input.party1().iter().map(|&value| value / scale).collect(),
        )?;
        let wrap_term = wrap.mul_public_scalar(p_quotient % p)?;

        let offset = 2 * scale;
        let index0 = input
            .party0()
            .iter()
            .zip(wrap.party0().iter())
            .map(|(&value, &wrap_share)| {
                add_mod(value % scale, mul_mod(wrap_share, offset % p, p), p)
            })
            .collect::<Vec<_>>();
        let index1 = input
            .party1()
            .iter()
            .zip(wrap.party1().iter())
            .map(|(&value, &wrap_share)| {
                add_mod(value % scale, mul_mod(wrap_share, offset % p, p), p)
            })
            .collect::<Vec<_>>();
        let correction_index = AdditiveShares::new(p, index0, index1)?;
        let (c_pos, c_neg) =
            self.rescale_correction_from_changes(&correction_index, corrections, rng)?;

        let positive = base.sub(&wrap_term)?.add(&c_pos)?;
        let neg_p_quotient = reduce_i128_mod(-(p_quotient as i128), p);
        let branch_delta = AdditiveShares::share_public(&vec![neg_p_quotient; input.len()], p)?
            .sub(&c_neg)?
            .sub(&c_pos)?;
        let branch = self.hss.mul_slots(&sign, &branch_delta, rng)?;
        positive.add(&branch)
    }

    fn rescale_fixed_approx<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(input)?;
        let p = self.fixed.modulus;
        let scale = self.fixed.scale;
        let p_quotient = p / scale;
        let bit_width = sum_bit_width(p)?;
        let radix_bits = comparison_radix_bits(bit_width, p)?;

        let comparisons = self.lookup.less_than_public_radix_many(
            input.party0(),
            input.party1(),
            bit_width,
            &[p / 2 + 1, p, p + p / 2 + 1],
            radix_bits,
            rng,
        )?;
        let u = one_minus(&comparisons[0])?;
        let wrap = one_minus(&comparisons[1])?;
        let v = one_minus(&comparisons[2])?;
        let sign = u.sub(&wrap)?.add(&v)?;

        let base = AdditiveShares::new(
            p,
            input.party0().iter().map(|&value| value / scale).collect(),
            input.party1().iter().map(|&value| value / scale).collect(),
        )?;
        let wrap_term = wrap.mul_public_scalar(p_quotient % p)?;
        let positive = base.sub(&wrap_term)?;
        let neg_p_quotient = reduce_i128_mod(-(p_quotient as i128), p);
        let branch_delta = AdditiveShares::share_public(&vec![neg_p_quotient; input.len()], p)?;
        let branch = self.hss.mul_slots(&sign, &branch_delta, rng)?;
        positive.add(&branch)
    }

    fn rescale_fixed_with_sign_approx<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        sign: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(input)?;
        self.validate_shares(sign)?;
        if input.len() != sign.len() {
            return Err(OperatorError::InvalidParams(
                "known-sign rescale sign length must match input",
            ));
        }
        let p = self.fixed.modulus;
        let scale = self.fixed.scale;
        let p_quotient = p / scale;
        let comparisons = self.less_than_sum_public_many(input, &[p], rng)?;
        let wrap = one_minus(&comparisons[0])?;
        let base = AdditiveShares::new(
            p,
            input.party0().iter().map(|&value| value / scale).collect(),
            input.party1().iter().map(|&value| value / scale).collect(),
        )?;
        let wrap_term = wrap.mul_public_scalar(p_quotient % p)?;
        let positive = base.sub(&wrap_term)?;
        let neg_p_quotient = reduce_i128_mod(-(p_quotient as i128), p);
        let branch_delta = AdditiveShares::share_public(&vec![neg_p_quotient; input.len()], p)?;
        let branch = self.hss.mul_slots(sign, &branch_delta, rng)?;
        positive.add(&branch)
    }

    fn rescale_fixed_nonnegative<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(input)?;
        let Some(corrections) = &self.rescale_corrections else {
            return self.rescale_lut(input, rng);
        };

        let p = self.fixed.modulus;
        let scale = self.fixed.scale;
        let p_quotient = p / scale;

        let comparisons = self.less_than_sum_public_many(input, &[p], rng)?;
        let wrap = one_minus(&comparisons[0])?;
        let base = AdditiveShares::new(
            p,
            input.party0().iter().map(|&value| value / scale).collect(),
            input.party1().iter().map(|&value| value / scale).collect(),
        )?;
        let wrap_term = wrap.mul_public_scalar(p_quotient % p)?;

        let offset = 2 * scale;
        let index0 = input
            .party0()
            .iter()
            .zip(wrap.party0().iter())
            .map(|(&value, &wrap_share)| {
                add_mod(value % scale, mul_mod(wrap_share, offset % p, p), p)
            })
            .collect::<Vec<_>>();
        let index1 = input
            .party1()
            .iter()
            .zip(wrap.party1().iter())
            .map(|(&value, &wrap_share)| {
                add_mod(value % scale, mul_mod(wrap_share, offset % p, p), p)
            })
            .collect::<Vec<_>>();
        let correction_index = AdditiveShares::new(p, index0, index1)?;
        let c_pos =
            self.rescale_positive_correction_from_changes(&correction_index, corrections, rng)?;

        base.sub(&wrap_term)?.add(&c_pos)
    }

    fn rescale_nonnegative_by_scale<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        scale: u64,
        corrections: &RescaleCorrectionTables,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_share_modulus(input)?;
        if scale == 0 {
            return Err(OperatorError::InvalidParams(
                "dyadic rescale divisor must be positive",
            ));
        }
        if scale == 1 {
            return Ok(input.clone());
        }
        let p = self.fixed.modulus;
        let p_quotient = p / scale;

        let comparisons = self.less_than_sum_public_many(input, &[p], rng)?;
        let wrap = one_minus(&comparisons[0])?;
        let base = AdditiveShares::new(
            p,
            input.party0().iter().map(|&value| value / scale).collect(),
            input.party1().iter().map(|&value| value / scale).collect(),
        )?;
        let wrap_term = wrap.mul_public_scalar(p_quotient % p)?;

        let offset = 2 * scale;
        let index0 = input
            .party0()
            .iter()
            .zip(wrap.party0().iter())
            .map(|(&value, &wrap_share)| {
                add_mod(value % scale, mul_mod(wrap_share, offset % p, p), p)
            })
            .collect::<Vec<_>>();
        let index1 = input
            .party1()
            .iter()
            .zip(wrap.party1().iter())
            .map(|(&value, &wrap_share)| {
                add_mod(value % scale, mul_mod(wrap_share, offset % p, p), p)
            })
            .collect::<Vec<_>>();
        let correction_index = AdditiveShares::new(p, index0, index1)?;
        let c_pos =
            self.rescale_positive_correction_from_changes(&correction_index, corrections, rng)?;

        base.sub(&wrap_term)?.add(&c_pos)
    }

    fn div_public_u64_nonnegative_preserve_scale<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        divisor: u64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if divisor == 0 {
            return Err(OperatorError::InvalidParams(
                "public divisor must be positive",
            ));
        }
        if divisor == 1 {
            return Ok(input.clone());
        }
        let Some(corrections) = build_rescale_correction_tables(FixedPointConfig {
            modulus: self.fixed.modulus,
            scale: divisor,
        })?
        else {
            return Err(OperatorError::InvalidParams(
                "public division is too large for current modulus",
            ));
        };
        self.rescale_nonnegative_by_scale(input, divisor, &corrections, rng)
    }

    fn rescale_fixed_nonnegative_approx<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.validate_shares(input)?;
        let p = self.fixed.modulus;
        let scale = self.fixed.scale;
        let p_quotient = p / scale;

        let comparisons = self.less_than_sum_public_many(input, &[p], rng)?;
        let wrap = one_minus(&comparisons[0])?;
        let base = AdditiveShares::new(
            p,
            input.party0().iter().map(|&value| value / scale).collect(),
            input.party1().iter().map(|&value| value / scale).collect(),
        )?;
        let wrap_term = wrap.mul_public_scalar(p_quotient % p)?;
        base.sub(&wrap_term)
    }

    fn rescale_correction_from_changes<R: RngCore + ?Sized>(
        &self,
        index: &AdditiveShares,
        corrections: &RescaleCorrectionTables,
        rng: &mut R,
    ) -> Result<(AdditiveShares, AdditiveShares), OperatorError> {
        let change_count = corrections.positive_changes.len() + corrections.negative_changes.len();
        if index.len() < 4096 {
            return self.rescale_correction_from_lookup(index, corrections, rng);
        }
        if change_count == 0 {
            return Ok((
                AdditiveShares::share_public(
                    &vec![corrections.positive_base; index.len()],
                    self.fixed.modulus,
                )?,
                AdditiveShares::share_public(
                    &vec![corrections.negative_base; index.len()],
                    self.fixed.modulus,
                )?,
            ));
        }

        if change_count > 16 {
            return self.rescale_correction_from_lookup(index, corrections, rng);
        }

        let thresholds = corrections
            .positive_changes
            .iter()
            .chain(corrections.negative_changes.iter())
            .map(|change| change.threshold)
            .collect::<Vec<_>>();
        let lt = self.less_than_mod_public_many(index, &thresholds, rng)?;
        let mut cursor = 0usize;
        let c_pos = self.accumulate_correction_changes(
            index.len(),
            corrections.positive_base,
            &corrections.positive_changes,
            &lt,
            &mut cursor,
        )?;
        let c_neg = self.accumulate_correction_changes(
            index.len(),
            corrections.negative_base,
            &corrections.negative_changes,
            &lt,
            &mut cursor,
        )?;
        Ok((c_pos, c_neg))
    }

    fn rescale_positive_correction_from_changes<R: RngCore + ?Sized>(
        &self,
        index: &AdditiveShares,
        corrections: &RescaleCorrectionTables,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if index.len() < 4096 || corrections.positive_changes.len() > 16 {
            return self.rescale_positive_correction_from_lookup(index, corrections, rng);
        }
        if corrections.positive_changes.is_empty() {
            return AdditiveShares::share_public(
                &vec![corrections.positive_base; index.len()],
                self.fixed.modulus,
            );
        }

        let thresholds = corrections
            .positive_changes
            .iter()
            .map(|change| change.threshold)
            .collect::<Vec<_>>();
        let lt = self.less_than_mod_public_many(index, &thresholds, rng)?;
        let mut cursor = 0usize;
        self.accumulate_correction_changes(
            index.len(),
            corrections.positive_base,
            &corrections.positive_changes,
            &lt,
            &mut cursor,
        )
    }

    fn accumulate_correction_changes(
        &self,
        len: usize,
        base: u64,
        changes: &[CorrectionChange],
        less_than_threshold: &[AdditiveShares],
        cursor: &mut usize,
    ) -> Result<AdditiveShares, OperatorError> {
        let mut acc = AdditiveShares::share_public(&vec![base; len], self.fixed.modulus)?;
        for change in changes {
            let ge_threshold = one_minus(&less_than_threshold[*cursor])?;
            let term = ge_threshold.mul_public_scalar(change.delta)?;
            acc = acc.add(&term)?;
            *cursor += 1;
        }
        Ok(acc)
    }

    fn rescale_correction_from_lookup<R: RngCore + ?Sized>(
        &self,
        index: &AdditiveShares,
        corrections: &RescaleCorrectionTables,
        rng: &mut R,
    ) -> Result<(AdditiveShares, AdditiveShares), OperatorError> {
        let mut correction_outputs = self.lookup.lookup_sparse_many(
            index.party0(),
            index.party1(),
            2,
            self.fixed.modulus as usize,
            &corrections.support,
            |table_idx, col| {
                let idx = col as usize;
                match table_idx {
                    0 if idx < corrections.positive.len() => corrections.positive[idx],
                    1 if idx < corrections.negative.len() => corrections.negative[idx],
                    _ => 0,
                }
            },
            rng,
        )?;
        let c_neg = correction_outputs.pop().ok_or_else(|| {
            OperatorError::Backend("rescale correction lookup missing negative output".to_string())
        })?;
        let c_pos = correction_outputs.pop().ok_or_else(|| {
            OperatorError::Backend("rescale correction lookup missing positive output".to_string())
        })?;
        Ok((c_pos, c_neg))
    }

    fn rescale_positive_correction_from_lookup<R: RngCore + ?Sized>(
        &self,
        index: &AdditiveShares,
        corrections: &RescaleCorrectionTables,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let mut correction_outputs = self.lookup.lookup_sparse_many(
            index.party0(),
            index.party1(),
            1,
            self.fixed.modulus as usize,
            &corrections.support,
            |_table_idx, col| {
                let idx = col as usize;
                if idx < corrections.positive.len() {
                    corrections.positive[idx]
                } else {
                    0
                }
            },
            rng,
        )?;
        correction_outputs.pop().ok_or_else(|| {
            OperatorError::Backend("positive rescale correction lookup missing output".to_string())
        })
    }

    fn rescale_lut<R: RngCore + ?Sized>(
        &self,
        input: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let table = (0..self.fixed.modulus)
            .map(|v| self.fixed.rescale_product_value(v))
            .collect::<Vec<_>>();
        self.lookup
            .lookup(input.party0(), input.party1(), &table, rng)
    }

    fn bounded_centered_lookup<R, F>(
        &self,
        input: &AdditiveShares,
        min_scaled: i128,
        max_scaled: i128,
        mut f: F,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(f64) -> f64,
    {
        let half = (self.fixed.modulus / 2) as i128;
        let min_scaled = min_scaled.max(-half);
        let max_scaled = max_scaled.min(half);
        if min_scaled > max_scaled {
            return AdditiveShares::share_public(&vec![0; input.len()], self.fixed.modulus);
        }

        let mut entries = Vec::new();
        for scaled in min_scaled..=max_scaled {
            let value = self
                .fixed
                .encode_f64(f(scaled as f64 / self.fixed.scale as f64));
            if value != 0 {
                entries.push((self.fixed.encode_scaled_int(scaled) as usize, value));
            }
        }
        entries.sort_by_key(|(idx, _)| *idx);
        let support = entries.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let values = entries.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        let mut outputs = self.lookup.lookup_sparse_many(
            input.party0(),
            input.party1(),
            1,
            self.fixed.modulus as usize,
            &support,
            |_table_idx, col| match support.binary_search(&(col as usize)) {
                Ok(idx) => values[idx],
                Err(_) => 0,
            },
            rng,
        )?;
        outputs
            .pop()
            .ok_or_else(|| OperatorError::Backend("bounded lookup returned no output".to_string()))
    }

    fn bounded_scaled_lookup<R, F>(
        &self,
        input: &AdditiveShares,
        min_scaled: i128,
        max_scaled: i128,
        mut value_at_scaled: F,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(i128) -> u64,
    {
        let half = (self.fixed.modulus / 2) as i128;
        let min_scaled = min_scaled.max(-half);
        let max_scaled = max_scaled.min(half);
        if min_scaled > max_scaled {
            return AdditiveShares::share_public(&vec![0; input.len()], self.fixed.modulus);
        }

        let mut entries = Vec::new();
        for scaled in min_scaled..=max_scaled {
            let value = value_at_scaled(scaled);
            if value != 0 {
                entries.push((self.fixed.encode_scaled_int(scaled) as usize, value));
            }
        }
        entries.sort_by_key(|(idx, _)| *idx);
        let support = entries.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let values = entries.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        let mut outputs = self.lookup.lookup_sparse_many(
            input.party0(),
            input.party1(),
            1,
            self.fixed.modulus as usize,
            &support,
            |_table_idx, col| match support.binary_search(&(col as usize)) {
                Ok(idx) => values[idx],
                Err(_) => 0,
            },
            rng,
        )?;
        outputs.pop().ok_or_else(|| {
            OperatorError::Backend("bounded scaled lookup returned no output".to_string())
        })
    }

    fn gelu_polynomial_value(&self, scaled: i128, cfg: &GeluConfig) -> u64 {
        let p = self.fixed.modulus;
        let x = self.fixed.encode_scaled_int(scaled);
        let mut poly =
            self.fixed
                .rescale_product_value(mul_mod(x, self.fixed.encode_f64(cfg.coeffs[3]), p));
        poly = add_mod(poly, self.fixed.encode_f64(cfg.coeffs[2]), p);
        poly = self.fixed.rescale_product_value(mul_mod(poly, x, p));
        poly = add_mod(poly, self.fixed.encode_f64(cfg.coeffs[1]), p);
        poly = self.fixed.rescale_product_value(mul_mod(poly, x, p));
        add_mod(poly, self.fixed.encode_f64(cfg.coeffs[0]), p)
    }

    fn validate_shares(&self, shares: &AdditiveShares) -> Result<(), OperatorError> {
        self.validate_share_modulus(shares)?;
        if shares.len() > self.hss.context().degree() {
            return Err(OperatorError::InvalidParams(
                "share length exceeds HSS SIMD capacity",
            ));
        }
        Ok(())
    }

    fn validate_share_modulus(&self, shares: &AdditiveShares) -> Result<(), OperatorError> {
        if shares.modulus() != self.fixed.modulus {
            return Err(OperatorError::InvalidParams(
                "share modulus must match fixed-point modulus",
            ));
        }
        Ok(())
    }
}

fn sum_rows(
    shares: &AdditiveShares,
    rows: usize,
    cols: usize,
) -> Result<AdditiveShares, OperatorError> {
    if shares.len() != rows.saturating_mul(cols) {
        return Err(OperatorError::InvalidParams(
            "sum_rows length must equal rows * cols",
        ));
    }
    let p = shares.modulus();
    let mut party0 = Vec::with_capacity(rows);
    let mut party1 = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        party0.push(
            shares.party0()[start..end]
                .iter()
                .fold(0u64, |acc, &value| add_mod(acc, value, p)),
        );
        party1.push(
            shares.party1()[start..end]
                .iter()
                .fold(0u64, |acc, &value| add_mod(acc, value, p)),
        );
    }
    AdditiveShares::new(p, party0, party1)
}

fn repeat_rows(shares: &AdditiveShares, cols: usize) -> Result<AdditiveShares, OperatorError> {
    if cols == 0 {
        return Err(OperatorError::InvalidParams(
            "repeat_rows requires a positive column count",
        ));
    }
    let p = shares.modulus();
    let mut party0 = Vec::with_capacity(shares.len() * cols);
    let mut party1 = Vec::with_capacity(shares.len() * cols);
    for row in 0..shares.len() {
        party0.extend(std::iter::repeat_n(shares.party0()[row], cols));
        party1.extend(std::iter::repeat_n(shares.party1()[row], cols));
    }
    AdditiveShares::new(p, party0, party1)
}

fn hss_mul_additive_shares_chunked<R: RngCore + ?Sized>(
    engine: &crate::hss_slots::HssSlotEngine,
    lhs: &AdditiveShares,
    rhs: &AdditiveShares,
    rng: &mut R,
) -> Result<AdditiveShares, OperatorError> {
    if lhs.modulus() != rhs.modulus() {
        return Err(OperatorError::InvalidParams(
            "HSS-backed correlation multiplication requires matching limb moduli",
        ));
    }
    if lhs.len() != rhs.len() {
        return Err(OperatorError::InvalidParams(
            "HSS-backed correlation multiplication requires equal lengths",
        ));
    }
    if lhs.is_empty() {
        return Err(OperatorError::InvalidParams(
            "HSS-backed correlation multiplication requires at least one slot",
        ));
    }
    let chunk_slots = engine.context().degree();
    if chunk_slots == 0 {
        return Err(OperatorError::InvalidParams(
            "HSS-backed correlation engine has zero ring degree",
        ));
    }
    let mut out0 = Vec::with_capacity(lhs.len());
    let mut out1 = Vec::with_capacity(lhs.len());
    for start in (0..lhs.len()).step_by(chunk_slots) {
        let end = usize::min(start + chunk_slots, lhs.len());
        let lhs_chunk = AdditiveShares::new(
            lhs.modulus(),
            lhs.party0()[start..end].to_vec(),
            lhs.party1()[start..end].to_vec(),
        )?;
        let rhs_chunk = AdditiveShares::new(
            rhs.modulus(),
            rhs.party0()[start..end].to_vec(),
            rhs.party1()[start..end].to_vec(),
        )?;
        let product = engine.mul_slots(&lhs_chunk, &rhs_chunk, rng)?;
        out0.extend_from_slice(product.party0());
        out1.extend_from_slice(product.party1());
    }
    AdditiveShares::new(lhs.modulus(), out0, out1)
}

fn expand_row_parameter(
    values: &[f64],
    rows: usize,
    cols: usize,
    default: f64,
) -> Result<Vec<f64>, OperatorError> {
    if values.is_empty() {
        return Ok(vec![default; rows * cols]);
    }
    if values.len() == rows * cols {
        return Ok(values.to_vec());
    }
    if values.len() == cols {
        let mut out = Vec::with_capacity(rows * cols);
        for _ in 0..rows {
            out.extend_from_slice(values);
        }
        return Ok(out);
    }
    Err(OperatorError::InvalidParams(
        "row parameter length must match columns or rows * cols",
    ))
}

fn is_row_parameter_value(values: &[f64], rows: usize, cols: usize, expected: f64) -> bool {
    values.is_empty()
        || ((values.len() == cols || values.len() == rows.saturating_mul(cols))
            && values.iter().all(|&value| value == expected))
}

fn one_minus(bits: &AdditiveShares) -> Result<AdditiveShares, OperatorError> {
    let p = bits.modulus();
    AdditiveShares::new(
        p,
        bits.party0().iter().map(|&v| sub_mod(1, v, p)).collect(),
        bits.party1().iter().map(|&v| sub_mod(0, v, p)).collect(),
    )
}

fn mul_public_slots_mod(
    shares: &AdditiveShares,
    scalars: &[u64],
) -> Result<AdditiveShares, OperatorError> {
    if shares.len() != scalars.len() {
        return Err(OperatorError::InvalidParams(
            "public scalar slot count must match share length",
        ));
    }
    let p = shares.modulus();
    AdditiveShares::new(
        p,
        shares
            .party0()
            .iter()
            .zip(scalars.iter())
            .map(|(&share, &scalar)| mul_mod(share, scalar, p))
            .collect(),
        shares
            .party1()
            .iter()
            .zip(scalars.iter())
            .map(|(&share, &scalar)| mul_mod(share, scalar, p))
            .collect(),
    )
}

fn oneflow_open_bytes(elements: usize) -> u64 {
    (elements as u64).saturating_mul(2).saturating_mul(8)
}

fn of_pmpe_pack_bytes(slots: usize, degree: usize) -> u64 {
    (slots as u64)
        .saturating_mul((degree as u64).saturating_add(2))
        .saturating_mul(2)
        .saturating_mul(8)
}

fn rns_of_pmpe_pack_bytes(slots: usize, degree: usize, limbs: usize) -> u64 {
    of_pmpe_pack_bytes(slots, degree).saturating_mul(limbs as u64)
}

fn checked_crt_modulus_u128(moduli: &[u64]) -> Result<u128, OperatorError> {
    moduli.iter().try_fold(1u128, |acc, &modulus| {
        acc.checked_mul(modulus as u128)
            .ok_or(OperatorError::InvalidParams("CRT modulus overflow"))
    })
}

fn u128_bit_width(value: u128) -> u32 {
    if value == 0 {
        0
    } else {
        u128::BITS - value.leading_zeros()
    }
}

fn modulus_bit_width(modulus: u64) -> u32 {
    u64::BITS - modulus.saturating_sub(1).leading_zeros()
}

fn binomial_mod(n: usize, k: usize, modulus: u64) -> Result<u64, OperatorError> {
    if k > n {
        return Ok(0);
    }
    let k = usize::min(k, n - k);
    let mut value = 1u128;
    for idx in 1..=k {
        value = value
            .checked_mul((n + 1 - idx) as u128)
            .ok_or(OperatorError::InvalidParams(
                "OF-PMPE polynomial degree is too large for binomial coefficients",
            ))?
            / idx as u128;
    }
    Ok((value % modulus as u128) as u64)
}

fn rns_mul_public_taylor_factor(
    shares: &RnsShareTensor,
    coeff: i128,
    j: usize,
    k: usize,
) -> Result<RnsShareTensor, OperatorError> {
    shares
        .limbs
        .iter()
        .map(|limb| {
            let coeff_mod = reduce_i128_mod(coeff, limb.modulus());
            let binom = binomial_mod(j, k, limb.modulus())?;
            limb.mul_public_scalar(mul_mod(coeff_mod, binom, limb.modulus()))
        })
        .collect::<Result<Vec<_>, _>>()
        .and_then(RnsShareTensor::from_limbs)
}

fn mod_inverse_u64(value: u64, modulus: u64) -> Option<u64> {
    if modulus < 2 {
        return None;
    }
    let mut t = 0i128;
    let mut new_t = 1i128;
    let mut r = modulus as i128;
    let mut new_r = (value % modulus) as i128;
    while new_r != 0 {
        let quotient = r / new_r;
        let next_t = t - quotient * new_t;
        t = new_t;
        new_t = next_t;
        let next_r = r - quotient * new_r;
        r = new_r;
        new_r = next_r;
    }
    if r != 1 {
        return None;
    }
    if t < 0 {
        t += modulus as i128;
    }
    Some(t as u64)
}

fn pow2_u64(bits: u32) -> Result<u64, OperatorError> {
    1u64.checked_shl(bits).ok_or(OperatorError::InvalidParams(
        "dyadic OF-PMPE frac_bits exceed u64 scale range",
    ))
}

fn estimate_dyadic_output_bound_bits(
    coeffs_num: &[i128],
    input_abs_bound_bits: u32,
) -> Result<u32, OperatorError> {
    if coeffs_num.is_empty() {
        return Err(OperatorError::InvalidParams(
            "dyadic OF-PMPE bound check requires coefficients",
        ));
    }
    let mut max_bits = 0u32;
    let mut nonzero_terms = 0u32;
    for (degree, &coeff) in coeffs_num.iter().enumerate() {
        if coeff == 0 {
            continue;
        }
        nonzero_terms = nonzero_terms.saturating_add(1);
        let coeff_bits = i128_abs_bit_width(coeff);
        let degree_bits = (degree as u32).checked_mul(input_abs_bound_bits).ok_or(
            OperatorError::InvalidParams("dyadic OF-PMPE output bound overflows"),
        )?;
        max_bits = max_bits.max(coeff_bits.saturating_add(degree_bits));
    }
    if nonzero_terms <= 1 {
        return Ok(max_bits);
    }
    Ok(max_bits.saturating_add(usize_bit_width(nonzero_terms as usize)))
}

fn i128_abs_bit_width(value: i128) -> u32 {
    let abs = value.unsigned_abs();
    if abs == 0 {
        0
    } else {
        u128::BITS - abs.leading_zeros()
    }
}

fn usize_bit_width(value: usize) -> u32 {
    if value <= 1 {
        0
    } else {
        usize::BITS - (value - 1).leading_zeros()
    }
}

fn round_scaled_coeff_to_i128(coeff: f64, shift: i64) -> Result<i128, OperatorError> {
    if !coeff.is_finite() {
        return Err(OperatorError::InvalidParams(
            "dyadic OF-PMPE coefficients must be finite",
        ));
    }
    if shift.abs() > 120 {
        return Err(OperatorError::InvalidParams(
            "dyadic OF-PMPE coefficient scale shift is too large",
        ));
    }
    let factor = 2f64.powi(shift.abs() as i32);
    let scaled = if shift >= 0 {
        coeff * factor
    } else {
        coeff / factor
    };
    if !scaled.is_finite() || scaled.abs() > i128::MAX as f64 {
        return Err(OperatorError::InvalidParams(
            "dyadic OF-PMPE coefficient exceeds i128 range",
        ));
    }
    Ok(scaled.round() as i128)
}

fn dyadic_integer_polynomial_checked(coeffs: &[i128], input: i128) -> Result<i128, OperatorError> {
    let mut acc = 0i128;
    let mut power = 1i128;
    for (degree, &coeff) in coeffs.iter().enumerate() {
        let term = coeff
            .checked_mul(power)
            .ok_or(OperatorError::InvalidParams(
                "dyadic polynomial grid audit term overflow",
            ))?;
        acc = acc.checked_add(term).ok_or(OperatorError::InvalidParams(
            "dyadic polynomial grid audit accumulator overflow",
        ))?;
        if degree + 1 < coeffs.len() {
            power = power
                .checked_mul(input)
                .ok_or(OperatorError::InvalidParams(
                    "dyadic polynomial grid audit power overflow",
                ))?;
        }
    }
    Ok(acc)
}

fn sample_u128_bits<R: RngCore + ?Sized>(rng: &mut R, bits: u32) -> u128 {
    debug_assert!(bits <= 128);
    if bits == 0 {
        return 0;
    }
    let mut value = 0u128;
    let words = bits.div_ceil(64);
    for idx in 0..words {
        value |= (rng.next_u64() as u128) << (idx * 64);
    }
    if bits < 128 {
        value &= (1u128 << bits) - 1;
    }
    value
}

fn build_rescale_correction_tables(
    fixed: FixedPointConfig,
) -> Result<Option<RescaleCorrectionTables>, OperatorError> {
    let p = fixed.modulus;
    let scale = fixed.scale;
    let Some(max_index) = scale.checked_mul(4).and_then(|value| value.checked_sub(2)) else {
        return Ok(None);
    };
    if max_index >= p {
        return Ok(None);
    }
    let table_len = usize::try_from(max_index + 1).map_err(|_| {
        OperatorError::InvalidParams("fixed-point rescale correction domain exceeds usize")
    })?;

    let mut positive = vec![0u64; table_len];
    let mut negative = vec![0u64; table_len];
    let mut support = Vec::with_capacity(table_len);
    let offset = 2 * scale;
    let half = scale as i128 / 2;
    let p_remainder = p % scale;

    for wrap in 0..=1u64 {
        for remainder_sum in 0..=(2 * scale - 2) {
            let idx = offset * wrap + remainder_sum;
            let c_pos = floor_div_i128(
                remainder_sum as i128 - wrap as i128 * p_remainder as i128 + half,
                scale as i128,
            );
            let c_neg = floor_div_i128(
                (wrap as i128 + 1) * p_remainder as i128 - remainder_sum as i128 + half,
                scale as i128,
            );
            let pos = fixed.encode_scaled_int(c_pos);
            let neg = fixed.encode_scaled_int(c_neg);
            positive[idx as usize] = pos;
            negative[idx as usize] = neg;
            if pos != 0 || neg != 0 {
                support.push(idx as usize);
            }
        }
    }

    let (positive_base, positive_changes) = correction_changes(&positive, p)?;
    let (negative_base, negative_changes) = correction_changes(&negative, p)?;

    Ok(Some(RescaleCorrectionTables {
        support,
        positive,
        negative,
        positive_base,
        negative_base,
        positive_changes,
        negative_changes,
    }))
}

fn correction_changes(
    values: &[u64],
    modulus: u64,
) -> Result<(u64, Vec<CorrectionChange>), OperatorError> {
    let Some((&base, rest)) = values.split_first() else {
        return Err(OperatorError::InvalidParams(
            "rescale correction table must be non-empty",
        ));
    };
    let mut previous = base;
    let mut changes = Vec::new();
    for (offset, &value) in rest.iter().enumerate() {
        if value != previous {
            changes.push(CorrectionChange {
                threshold: (offset + 1) as u64,
                delta: sub_mod(value, previous, modulus),
            });
            previous = value;
        }
    }
    Ok((base, changes))
}

fn sum_bit_width(p: u64) -> Result<usize, OperatorError> {
    let max_sum = p
        .checked_sub(1)
        .and_then(|value| value.checked_mul(2))
        .ok_or(OperatorError::InvalidParams(
            "fixed-point modulus is too large for rescale comparison",
        ))?;
    let max_threshold = p
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(OperatorError::InvalidParams(
            "fixed-point modulus is too large for rescale threshold",
        ))?;
    let max_value = max_sum.max(max_threshold);
    Ok((u64::BITS - max_value.leading_zeros()) as usize)
}

fn comparison_radix_bits(bit_width: usize, p: u64) -> Result<usize, OperatorError> {
    let max_base = (p - 1) / 2;
    let max_radix = (u64::BITS - max_base.leading_zeros() - 1) as usize;
    let radix_bits = usize::min(3, usize::min(bit_width, max_radix));
    if radix_bits == 0 {
        return Err(OperatorError::InvalidParams(
            "fixed-point modulus is too small for radix comparison",
        ));
    }
    Ok(radix_bits)
}

fn validate_strict_thresholds(thresholds: &[i128]) -> Result<(), OperatorError> {
    if thresholds.is_empty() {
        return Err(OperatorError::InvalidParams(
            "fused selector compilation requires at least one threshold",
        ));
    }
    if thresholds.windows(2).any(|window| window[0] >= window[1]) {
        return Err(OperatorError::InvalidParams(
            "fused selector thresholds must be strictly increasing",
        ));
    }
    Ok(())
}

fn floor_div_i128(value: i128, divisor: i128) -> i128 {
    debug_assert!(divisor > 0);
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && value < 0 {
        quotient - 1
    } else {
        quotient
    }
}
