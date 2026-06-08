//! BGV bootstrapping/recryption building blocks and diagnostics.
//!
//! The recryption key material here is intentionally only one piece of a full
//! BGV bootstrap: the source secret-key polynomial encrypted under a target BGV
//! public key.  The audit helper records the exact correction that a future
//! homomorphic carry/centered-reduction circuit must reproduce before SILENT
//! can honestly expose full public BGV bootstrapping.

use crate::core::encoder::HeEncoder;
use crate::core::evaluator::HeEvaluator;
use crate::schemes::bgv::ciphertext::{BgvCiphertext, BgvPlaintextFactor};
use crate::schemes::bgv::crypto::{BgvDecryptor, BgvEncryptor};
use crate::schemes::bgv::encoding::BgvBatchEncoder;
use crate::schemes::bgv::ops::BgvEvaluator;
use crate::schemes::bgv::params::{BgvLevelError, BgvParameters};
use crate::schemes::bgv::rns_big;
use num_bigint::{BigInt, Sign};
use num_traits::{Signed, ToPrimitive, Zero};
use silent_math::{arith, numth};
use silent_ring::Poly;
use silent_rlwe::{Ciphertext, EvaluationKey, GaloisKey, Plaintext, PublicKey, SecretKey};
use silent_utils::rng::SecureRng;

#[derive(Clone, Debug)]
pub struct BgvRecryptionKey {
    source_degree: usize,
    source_plain_modulus: u64,
    encrypted_secret_key: BgvCiphertext,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapReductionAudit {
    pub raw_phase_coefficients: Vec<BigInt>,
    pub true_coefficients: Vec<u64>,
    pub component_reduced_coefficients: Vec<u64>,
    pub correction_coefficients: Vec<u64>,
    pub wrap_counts: Vec<i64>,
    pub wrap_correction_coefficients: Vec<u64>,
    pub nonzero_correction_count: usize,
    pub max_abs_wrap_count: u64,
    pub q_mod_plaintext: u64,
    pub factor_inverse: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapCarryPolynomial {
    pub plain_modulus: u64,
    pub max_abs_wrap_count: u64,
    pub q_mod_plaintext: u64,
    pub factor_inverse: u64,
    pub scalar_coefficients: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapPhaseTerm {
    pub secret_coefficient_index: usize,
    pub multiplier: BigInt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapPhaseCoefficientPlan {
    pub coefficient_index: usize,
    pub constant: BigInt,
    pub terms: Vec<BgvBootstrapPhaseTerm>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapPhaseLinearizationPlan {
    pub degree: usize,
    pub coefficients: Vec<BgvBootstrapPhaseCoefficientPlan>,
    pub max_abs_constant: BigInt,
    pub max_abs_multiplier: BigInt,
    pub ternary_secret_abs_phase_bound: BigInt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapCrtWrapReconstructionPlan {
    pub degree: usize,
    pub auxiliary_plain_moduli: Vec<u64>,
    pub modulus_product: BigInt,
    pub phase_bound: BigInt,
    pub max_abs_wrap_count: u64,
    pub wrap_interval_plan: BgvBootstrapWrapIntervalPlan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSingleModulusWrapClassifierPolynomial {
    pub source_degree: usize,
    pub classifier_plain_modulus: u64,
    pub phase_bound: BigInt,
    pub max_abs_wrap_count: u64,
    pub scalar_coefficients: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSameFieldCrtWrapTerm {
    pub residues: Vec<u64>,
    pub wrap_count: i64,
    pub wrap_scalar: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSameFieldCrtWrapClassifier {
    pub source_degree: usize,
    pub classifier_plain_modulus: u64,
    pub auxiliary_plain_moduli: Vec<u64>,
    pub max_abs_wrap_count: u64,
    pub tuple_count: u64,
    pub indicator_scalar_coefficients: Vec<Vec<Vec<u64>>>,
    pub nonzero_terms: Vec<BgvBootstrapSameFieldCrtWrapTerm>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionTerm {
    pub residues: Vec<u64>,
    pub correction_scalar: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionClassifier {
    pub source_degree: usize,
    pub classifier_plain_modulus: u64,
    pub source_plain_modulus: u64,
    pub target_plain_modulus: u64,
    pub source_factor_inverse: u64,
    pub auxiliary_plain_moduli: Vec<u64>,
    pub tuple_count: u64,
    pub indicator_scalar_coefficients: Vec<Vec<Vec<u64>>>,
    pub nonzero_terms: Vec<BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionTerm>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSameFieldCrtResidueBridge {
    pub source_degree: usize,
    pub fusion_plain_modulus: u64,
    pub phase_bound: BigInt,
    pub auxiliary_plain_moduli: Vec<u64>,
    pub residue_scalar_coefficients: Vec<Vec<u64>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSameFieldDecodePolynomial {
    pub source_degree: usize,
    pub fusion_plain_modulus: u64,
    pub source_plain_modulus: u64,
    pub factor_inverse: u64,
    pub phase_bound: BigInt,
    pub scalar_coefficients: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSameFieldDecodeLiftMismatch {
    pub raw_phase_residue: u64,
    pub target_mod_source: u64,
    pub centered_lift_mod_source: u64,
    pub correction_mod_source: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSameFieldDecodeLiftAudit {
    pub source_degree: usize,
    pub fusion_plain_modulus: u64,
    pub source_plain_modulus: u64,
    pub mismatch_count: usize,
    pub mismatches: Vec<BgvBootstrapSameFieldDecodeLiftMismatch>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSameFieldDecodeLiftCorrectionPolynomial {
    pub source_degree: usize,
    pub fusion_plain_modulus: u64,
    pub source_plain_modulus: u64,
    pub scalar_coefficients: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapFusionToSourceBridgePlan {
    pub source_degree: usize,
    pub fusion_plain_modulus: u64,
    pub source_plain_modulus: u64,
    pub same_ciphertext_modulus_chain: bool,
    pub plaintext_noise_vanishes_mod_source: bool,
    pub plaintext_factor_compatible: bool,
    pub can_reinterpret_without_noise_correction: bool,
    pub requires_noise_correction: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapFusionToSourceNoiseCorrectionAudit {
    pub source_degree: usize,
    pub fusion_plain_modulus: u64,
    pub source_plain_modulus: u64,
    pub target_coefficients: Vec<u64>,
    pub direct_coefficients: Vec<u64>,
    pub correction_coefficients: Vec<u64>,
    pub nonzero_correction_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapPlaintextModulusSwitchCorrectionAudit {
    pub source_degree: usize,
    pub source_plain_modulus: u64,
    pub target_plain_modulus: u64,
    pub source_factor_inverse: u64,
    pub source_plaintext_coefficients: Vec<u64>,
    pub direct_target_coefficients: Vec<u64>,
    pub correction_coefficients: Vec<u64>,
    pub nonzero_correction_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapFusionPlaintextModulusPlan {
    pub source_degree: usize,
    pub source_plain_modulus: u64,
    pub required_lookup_points: u64,
    pub selected_fusion_plain_modulus: u64,
    pub uses_source_plaintext_modulus: bool,
    pub plaintext_noise_vanishes_mod_source: bool,
    pub requires_small_plaintext_correction: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapSourceCompatibleFusionNoiseScalePlan {
    pub source_degree: usize,
    pub source_plain_modulus: u64,
    pub fusion_plain_modulus: u64,
    pub error_scalar: u64,
    pub uses_default_fusion_error_scalar: bool,
    pub bridges_without_noise_correction: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BgvBootstrapCircuitGap {
    HomomorphicFusionToSourcePlaintextBridge,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapCircuitReadiness {
    pub source_degree: usize,
    pub source_q_limbs: usize,
    pub target_q_limbs: usize,
    pub phase_bound_bits: u64,
    pub crt_modulus_product_bits: u64,
    pub auxiliary_plain_moduli: Vec<u64>,
    pub max_abs_wrap_count: u64,
    pub raw_phase_affine_input_layer: bool,
    pub encrypted_phase_mod_plaintext: bool,
    pub encrypted_crt_phase_residues: bool,
    pub crt_wrap_reconstruction_plan: bool,
    pub extracted_crt_residue_repack: bool,
    pub encrypted_coefficient_carry_correction: bool,
    pub corrected_recryption: bool,
    pub homomorphic_single_modulus_residue_classification: bool,
    pub homomorphic_same_field_crt_residue_bridge: bool,
    pub homomorphic_same_field_crt_residue_fusion: bool,
    pub homomorphic_same_field_plaintext_modulus_switch_correction: bool,
    pub homomorphic_same_field_plaintext_modulus_switch: bool,
    pub homomorphic_same_field_phase_decode: bool,
    pub homomorphic_same_field_fusion_refresh: bool,
    pub fusion_plaintext_modulus_planner: bool,
    pub source_compatible_fusion_noise_scaling: bool,
    pub same_field_decode_lift_correction: bool,
    pub homomorphic_same_field_decode_lift_correction: bool,
    pub fusion_to_source_plaintext_bridge_plan: bool,
    pub fusion_to_source_plaintext_noise_correction_skeleton: bool,
    pub fusion_to_source_plaintext_lift_patch: bool,
    pub fusion_to_source_encrypted_lift_patch: bool,
    pub fusion_to_source_encrypted_noise_correction_bridge: bool,
    pub fusion_to_source_recryption_with_encrypted_correction: bool,
    pub fusion_to_source_extracted_crt_plaintext_modulus_switch: bool,
    pub homomorphic_fusion_to_source_plaintext_bridge: bool,
    pub encrypted_coefficient_wrap_repacking: bool,
    pub full_bootstrapping_circuit: bool,
    pub gaps: Vec<BgvBootstrapCircuitGap>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapWrapInterval {
    pub wrap_count: i64,
    pub lower_inclusive: BigInt,
    pub upper_inclusive: BigInt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvBootstrapWrapIntervalPlan {
    pub max_abs_wrap_count: u64,
    pub intervals: Vec<BgvBootstrapWrapInterval>,
}

impl BgvRecryptionKey {
    pub fn encrypted_secret_key(&self) -> &BgvCiphertext {
        &self.encrypted_secret_key
    }

    pub fn source_degree(&self) -> usize {
        self.source_degree
    }

    pub fn source_plain_modulus(&self) -> u64 {
        self.source_plain_modulus
    }
}

pub fn generate_bgv_recryption_key(
    source_params: &BgvParameters,
    source_sk: &SecretKey,
    target_params: BgvParameters,
    target_pk: &PublicKey,
    rng: SecureRng,
) -> Result<BgvRecryptionKey, BgvLevelError> {
    ensure_same_plaintext_space(source_params, &target_params)?;
    let secret_plaintext = secret_key_plaintext(source_params, source_sk)?;
    let mut encryptor = BgvEncryptor::new(target_params.clone(), rng);
    let encrypted_secret_key = encryptor.encrypt_public_bgv(target_pk, &secret_plaintext);
    Ok(BgvRecryptionKey {
        source_degree: source_params.degree(),
        source_plain_modulus: source_params.plain_modulus(),
        encrypted_secret_key,
    })
}

pub fn generate_bgv_phase_residue_recryption_key(
    source_params: &BgvParameters,
    source_sk: &SecretKey,
    target_params: BgvParameters,
    target_pk: &PublicKey,
    rng: SecureRng,
) -> Result<BgvRecryptionKey, BgvLevelError> {
    if source_params.degree() != target_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source_params.degree(),
            actual: target_params.degree(),
        });
    }
    let secret_coefficients = secret_key_centered_bigint_coefficients(source_params, source_sk)?;
    let secret_plaintext =
        plaintext_from_centered_coefficients(&target_params, &secret_coefficients);
    let mut encryptor = BgvEncryptor::new(target_params.clone(), rng);
    let encrypted_secret_key = encryptor.encrypt_public_bgv(target_pk, &secret_plaintext);
    Ok(BgvRecryptionKey {
        source_degree: source_params.degree(),
        source_plain_modulus: source_params.plain_modulus(),
        encrypted_secret_key,
    })
}

pub fn generate_bgv_phase_residue_recryption_key_with_error_scalar(
    source_params: &BgvParameters,
    source_sk: &SecretKey,
    target_params: BgvParameters,
    target_pk: &PublicKey,
    rng: SecureRng,
    error_scalar: u64,
) -> Result<BgvRecryptionKey, BgvLevelError> {
    if source_params.degree() != target_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source_params.degree(),
            actual: target_params.degree(),
        });
    }
    let secret_coefficients = secret_key_centered_bigint_coefficients(source_params, source_sk)?;
    let secret_plaintext =
        plaintext_from_centered_coefficients(&target_params, &secret_coefficients);
    let mut encryptor = BgvEncryptor::new(target_params.clone(), rng);
    let encrypted_secret_key = encryptor.encrypt_public_bgv_with_error_scalar(
        target_pk,
        &secret_plaintext,
        error_scalar,
    )?;
    Ok(BgvRecryptionKey {
        source_degree: source_params.degree(),
        source_plain_modulus: source_params.plain_modulus(),
        encrypted_secret_key,
    })
}

pub fn uncorrected_bgv_recryption(
    ct: &BgvCiphertext,
    key: &BgvRecryptionKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    validate_recryption_input(ct, key)?;

    let factor_inv = ct.int_factor().inverse_mod(ct.params().plain_modulus())?;
    let mut c0 = ciphertext_component_plaintext(ct, 0)?;
    let mut c1 = ciphertext_component_plaintext(ct, 1)?;
    scale_plaintext_inplace(&mut c0, ct.params().plain_modulus(), factor_inv);
    scale_plaintext_inplace(&mut c1, ct.params().plain_modulus(), factor_inv);

    let evaluator = BgvEvaluator::new(key.encrypted_secret_key.params().clone());
    let reduced = evaluator.mul_plain_bgv(&key.encrypted_secret_key, &c1)?;
    evaluator.add_plain_bgv(&reduced, &c0)
}

pub fn bgv_bootstrap_encrypted_phase_mod_plaintext(
    ct: &BgvCiphertext,
    key: &BgvRecryptionKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    validate_recryption_input(ct, key)?;

    let c0 = ciphertext_component_plaintext(ct, 0)?;
    let c1 = ciphertext_component_plaintext(ct, 1)?;
    let evaluator = BgvEvaluator::new(key.encrypted_secret_key.params().clone());
    let secret_product = evaluator.mul_plain_bgv(&key.encrypted_secret_key, &c1)?;
    evaluator.add_plain_bgv(&secret_product, &c0)
}

pub fn bgv_bootstrap_encrypted_phase_residue(
    ct: &BgvCiphertext,
    key: &BgvRecryptionKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    validate_phase_residue_recryption_input(ct, key)?;

    let target_params = key.encrypted_secret_key.params();
    let c0 = ciphertext_component_plaintext_for_params(ct, 0, target_params)?;
    let c1 = ciphertext_component_plaintext_for_params(ct, 1, target_params)?;
    let evaluator = BgvEvaluator::new(target_params.clone());
    let secret_product = evaluator.mul_plain_bgv(&key.encrypted_secret_key, &c1)?;
    evaluator.add_plain_bgv(&secret_product, &c0)
}

pub fn corrected_bgv_recryption_with_encrypted_correction(
    ct: &BgvCiphertext,
    key: &BgvRecryptionKey,
    encrypted_correction: &BgvCiphertext,
) -> Result<BgvCiphertext, BgvLevelError> {
    ensure_same_plaintext_space(
        key.encrypted_secret_key.params(),
        encrypted_correction.params(),
    )?;
    let uncorrected = uncorrected_bgv_recryption(ct, key)?;
    BgvEvaluator::new(key.encrypted_secret_key.params().clone())
        .add_bgv(&uncorrected, encrypted_correction)
}

pub fn bgv_bootstrap_coefficient_carry_correction(
    encrypted_coefficient_wrap_counts: &BgvCiphertext,
    q_mod_plaintext: u64,
    factor_inverse: u64,
) -> Result<BgvCiphertext, BgvLevelError> {
    let params = encrypted_coefficient_wrap_counts.params();
    let t = params.plain_modulus();
    let q_factor = arith::mul_mod_u64(q_mod_plaintext % t, factor_inverse % t, t);
    let correction_factor = arith::sub_mod(0, q_factor, t);
    let correction_scalar = coefficient_scalar_plaintext(params, correction_factor);
    BgvEvaluator::new(params.clone())
        .mul_plain_bgv(encrypted_coefficient_wrap_counts, &correction_scalar)
}

pub fn bgv_recryption_with_encrypted_coefficient_wraps(
    ct: &BgvCiphertext,
    key: &BgvRecryptionKey,
    encrypted_coefficient_wrap_counts: &BgvCiphertext,
) -> Result<BgvCiphertext, BgvLevelError> {
    validate_recryption_input(ct, key)?;
    ensure_same_plaintext_space(
        key.encrypted_secret_key.params(),
        encrypted_coefficient_wrap_counts.params(),
    )?;

    let t = ct.params().plain_modulus();
    let factor_inverse = ct.int_factor().inverse_mod(t)?;
    let q_mod_plaintext = ciphertext_modulus_mod_plaintext(ct.params());
    let encrypted_correction = bgv_bootstrap_coefficient_carry_correction(
        encrypted_coefficient_wrap_counts,
        q_mod_plaintext,
        factor_inverse,
    )?;
    corrected_bgv_recryption_with_encrypted_correction(ct, key, &encrypted_correction)
}

pub fn bgv_bootstrap_coefficient_wrap_plaintext_from_crt_residues(
    target_params: &BgvParameters,
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
    residues_by_coefficient: &[Vec<u64>],
) -> Result<Plaintext, BgvLevelError> {
    if target_params.degree() != crt_plan.degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: crt_plan.degree,
            actual: target_params.degree(),
        });
    }
    let wrap_counts = crt_plan.classify_wrap_counts_from_residues(residues_by_coefficient)?;
    let mut out = Poly::new(target_params.degree(), 1);
    for (dst, wrap) in out.limb_mut(0).iter_mut().zip(wrap_counts.iter()) {
        *dst = signed_i64_to_mod(*wrap, target_params.plain_modulus());
    }
    Ok(Plaintext { value: out })
}

pub fn bgv_recryption_with_extracted_crt_residues(
    ct: &BgvCiphertext,
    key: &BgvRecryptionKey,
    target_pk: &PublicKey,
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
    residues_by_coefficient: &[Vec<u64>],
    rng: SecureRng,
) -> Result<BgvCiphertext, BgvLevelError> {
    validate_recryption_input(ct, key)?;
    let wrap_plaintext = bgv_bootstrap_coefficient_wrap_plaintext_from_crt_residues(
        key.encrypted_secret_key.params(),
        crt_plan,
        residues_by_coefficient,
    )?;
    let mut encryptor = BgvEncryptor::new(key.encrypted_secret_key.params().clone(), rng);
    let encrypted_wraps = encryptor.encrypt_public_bgv(target_pk, &wrap_plaintext);
    bgv_recryption_with_encrypted_coefficient_wraps(ct, key, &encrypted_wraps)
}

pub fn audit_bgv_bootstrap_plaintext_modulus_switch_correction_from_crt_residues(
    ct: &BgvCiphertext,
    target_params: &BgvParameters,
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
    residues_by_coefficient: &[Vec<u64>],
) -> Result<BgvBootstrapPlaintextModulusSwitchCorrectionAudit, BgvLevelError> {
    if ct.params().degree() != target_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: ct.params().degree(),
            actual: target_params.degree(),
        });
    }
    if ct.params().degree() != crt_plan.degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: ct.params().degree(),
            actual: crt_plan.degree,
        });
    }

    let raw_phase = crt_plan.reconstruct_centered_phase(residues_by_coefficient)?;
    let source_plain_modulus = ct.params().plain_modulus();
    let target_plain_modulus = target_params.plain_modulus();
    let source_factor_inverse = ct.int_factor().inverse_mod(source_plain_modulus)?;
    let source_q =
        BigInt::from_biguint(Sign::Plus, rns_big::ciphertext_modulus(ct.params().ring()));
    let source_plaintext_coefficients = raw_phase
        .iter()
        .map(|coefficient| {
            let (centered_phase, _wrap) = centered_residue_and_wrap(coefficient, &source_q);
            arith::mul_mod_u64(
                bigint_mod_u64(&centered_phase, source_plain_modulus),
                source_factor_inverse,
                source_plain_modulus,
            ) % target_plain_modulus
        })
        .collect::<Vec<_>>();
    let direct_target_coefficients = raw_phase
        .iter()
        .map(|coefficient| bigint_mod_u64(coefficient, target_plain_modulus))
        .collect::<Vec<_>>();
    let correction_coefficients = source_plaintext_coefficients
        .iter()
        .zip(direct_target_coefficients.iter())
        .map(|(&source, &direct)| {
            arith::sub_mod(
                source % target_plain_modulus,
                direct % target_plain_modulus,
                target_plain_modulus,
            )
        })
        .collect::<Vec<_>>();
    let nonzero_correction_count = correction_coefficients
        .iter()
        .filter(|&&coefficient| coefficient != 0)
        .count();

    Ok(BgvBootstrapPlaintextModulusSwitchCorrectionAudit {
        source_degree: ct.params().degree(),
        source_plain_modulus,
        target_plain_modulus,
        source_factor_inverse,
        source_plaintext_coefficients,
        direct_target_coefficients,
        correction_coefficients,
        nonzero_correction_count,
    })
}

pub fn bgv_bootstrap_plaintext_modulus_switch_correction_plaintext_from_crt_residues(
    ct: &BgvCiphertext,
    target_params: &BgvParameters,
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
    residues_by_coefficient: &[Vec<u64>],
) -> Result<Plaintext, BgvLevelError> {
    let audit = audit_bgv_bootstrap_plaintext_modulus_switch_correction_from_crt_residues(
        ct,
        target_params,
        crt_plan,
        residues_by_coefficient,
    )?;
    let mut out = Poly::new(target_params.degree(), 1);
    for (dst, coefficient) in out
        .limb_mut(0)
        .iter_mut()
        .zip(audit.correction_coefficients.iter())
    {
        *dst = coefficient % target_params.plain_modulus();
    }
    Ok(Plaintext { value: out })
}

pub fn bgv_bootstrap_plaintext_modulus_switch_with_extracted_crt_residues(
    ct: &BgvCiphertext,
    direct_phase_key: &BgvRecryptionKey,
    target_pk: &PublicKey,
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
    residues_by_coefficient: &[Vec<u64>],
    rng: SecureRng,
) -> Result<BgvCiphertext, BgvLevelError> {
    validate_phase_residue_recryption_input(ct, direct_phase_key)?;
    let target_params = direct_phase_key.encrypted_secret_key.params();
    let correction_plaintext =
        bgv_bootstrap_plaintext_modulus_switch_correction_plaintext_from_crt_residues(
            ct,
            target_params,
            crt_plan,
            residues_by_coefficient,
        )?;
    let direct_phase = bgv_bootstrap_encrypted_phase_residue(ct, direct_phase_key)?;
    let mut encryptor = BgvEncryptor::new(target_params.clone(), rng);
    let encrypted_correction = encryptor.encrypt_public_bgv(target_pk, &correction_plaintext);
    BgvEvaluator::new(target_params.clone()).add_bgv(&direct_phase, &encrypted_correction)
}

pub fn bgv_bootstrap_coefficients_to_slots_matrix(params: &BgvParameters) -> Vec<Vec<u64>> {
    let degree = params.degree();
    let encoder = BgvBatchEncoder::new(params.clone());
    let mut matrix = vec![vec![0u64; degree]; degree];
    for column in 0..degree {
        let mut basis_slots = vec![0u64; degree];
        basis_slots[column] = 1;
        let encoded = encoder.encode(&basis_slots);
        for (row, value) in encoded.value.limb(0).iter().enumerate() {
            matrix[row][column] = *value % params.plain_modulus();
        }
    }
    matrix
}

pub fn bgv_bootstrap_slots_to_coefficients_matrix(params: &BgvParameters) -> Vec<Vec<u64>> {
    let degree = params.degree();
    let encoder = BgvBatchEncoder::new(params.clone());
    let mut matrix = vec![vec![0u64; degree]; degree];
    for column in 0..degree {
        let mut basis = Poly::new(degree, 1);
        basis.limb_mut(0)[column] = 1;
        let decoded = encoder.decode(&Plaintext { value: basis });
        for (row, value) in decoded.iter().enumerate() {
            matrix[row][column] = *value % params.plain_modulus();
        }
    }
    matrix
}

pub fn bgv_bootstrap_coefficients_to_slots_bgv(
    ct: &BgvCiphertext,
    gk: &GaloisKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    BgvEvaluator::new(ct.params().clone()).slot_linear_transform_plain_bgv(
        ct,
        &bgv_bootstrap_coefficients_to_slots_matrix(ct.params()),
        gk,
    )
}

pub fn bgv_bootstrap_slots_to_coefficients_bgv(
    ct: &BgvCiphertext,
    gk: &GaloisKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    BgvEvaluator::new(ct.params().clone()).slot_linear_transform_plain_bgv(
        ct,
        &bgv_bootstrap_slots_to_coefficients_matrix(ct.params()),
        gk,
    )
}

pub fn bgv_bootstrap_single_modulus_wrap_classifier_polynomial(
    source_params: &BgvParameters,
    classifier_params: &BgvParameters,
    phase_bound: &BigInt,
) -> Result<BgvBootstrapSingleModulusWrapClassifierPolynomial, BgvLevelError> {
    const MAX_LOOKUP_POINTS: u64 = 4096;

    if source_params.degree() != classifier_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source_params.degree(),
            actual: classifier_params.degree(),
        });
    }

    let classifier_plain_modulus = classifier_params.plain_modulus();
    if classifier_plain_modulus > MAX_LOOKUP_POINTS {
        return Err(BgvLevelError::BootstrapClassifierLookupTooLarge {
            modulus: classifier_plain_modulus,
            max_points: MAX_LOOKUP_POINTS,
        });
    }

    let required = phase_bound.abs() * BigInt::from(2u8) + BigInt::from(1u8);
    if BigInt::from(classifier_plain_modulus) <= required {
        return Err(
            BgvLevelError::BootstrapSingleModulusClassifierCoverageTooSmall {
                modulus: classifier_plain_modulus,
                required: required.to_u64().unwrap_or(u64::MAX),
            },
        );
    }

    let max_abs_wrap_count = max_abs_wrap_count_for_phase_bound(source_params, phase_bound)?;
    let interval_plan = bgv_bootstrap_wrap_interval_plan(source_params, max_abs_wrap_count)?;
    let phase_bound_abs = phase_bound.abs();
    let mut points = Vec::with_capacity(classifier_plain_modulus as usize);
    for residue in 0..classifier_plain_modulus {
        let centered = centered_residue_mod_plaintext(residue, classifier_plain_modulus);
        let wrap = if centered.abs() <= phase_bound_abs {
            interval_plan.classify(&centered).ok_or(
                BgvLevelError::BootstrapCrtReconstructionOutOfRange {
                    coefficient: residue as usize,
                },
            )?
        } else {
            0
        };
        points.push((residue, signed_i64_to_mod(wrap, classifier_plain_modulus)));
    }

    let scalar_coefficients = interpolate_lagrange_mod(&points, classifier_plain_modulus)?;
    Ok(BgvBootstrapSingleModulusWrapClassifierPolynomial {
        source_degree: source_params.degree(),
        classifier_plain_modulus,
        phase_bound: phase_bound.clone(),
        max_abs_wrap_count,
        scalar_coefficients,
    })
}

pub fn bgv_bootstrap_homomorphic_single_modulus_wrap_classification(
    encrypted_residue_slots: &BgvCiphertext,
    classifier: &BgvBootstrapSingleModulusWrapClassifierPolynomial,
    evk: &EvaluationKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    if encrypted_residue_slots.params().degree() != classifier.source_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: classifier.source_degree,
            actual: encrypted_residue_slots.params().degree(),
        });
    }
    if encrypted_residue_slots.params().plain_modulus() != classifier.classifier_plain_modulus {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: classifier.classifier_plain_modulus,
            actual: encrypted_residue_slots.params().plain_modulus(),
        });
    }

    let coefficients = classifier.plaintext_coefficients(encrypted_residue_slots.params())?;
    BgvEvaluator::new(encrypted_residue_slots.params().clone()).evaluate_polynomial_plain_bgv(
        encrypted_residue_slots,
        &coefficients,
        evk,
    )
}

pub fn bgv_bootstrap_same_field_crt_residue_bridge(
    fusion_params: &BgvParameters,
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
) -> Result<BgvBootstrapSameFieldCrtResidueBridge, BgvLevelError> {
    const MAX_BRIDGE_LOOKUP_POINTS: u64 = 4096;

    if fusion_params.degree() != crt_plan.degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: crt_plan.degree,
            actual: fusion_params.degree(),
        });
    }

    let fusion_plain_modulus = fusion_params.plain_modulus();
    if fusion_plain_modulus > MAX_BRIDGE_LOOKUP_POINTS {
        return Err(BgvLevelError::BootstrapClassifierLookupTooLarge {
            modulus: fusion_plain_modulus,
            max_points: MAX_BRIDGE_LOOKUP_POINTS,
        });
    }
    let required = crt_plan.phase_bound.abs() * BigInt::from(2u8) + BigInt::from(1u8);
    if BigInt::from(fusion_plain_modulus) <= required {
        return Err(
            BgvLevelError::BootstrapSingleModulusClassifierCoverageTooSmall {
                modulus: fusion_plain_modulus,
                required: required.to_u64().unwrap_or(u64::MAX),
            },
        );
    }

    let phase_bound_abs = crt_plan.phase_bound.abs();
    let mut residue_scalar_coefficients = Vec::with_capacity(crt_plan.auxiliary_plain_moduli.len());
    for &residue_modulus in &crt_plan.auxiliary_plain_moduli {
        if fusion_plain_modulus <= residue_modulus {
            return Err(BgvLevelError::BootstrapSameFieldCrtResidueModulusTooLarge {
                classifier_modulus: fusion_plain_modulus,
                residue_modulus,
            });
        }
        let points = (0..fusion_plain_modulus)
            .map(|residue| {
                let centered = centered_residue_mod_plaintext(residue, fusion_plain_modulus);
                let reduced = if centered.abs() <= phase_bound_abs {
                    bigint_mod_u64(&centered, residue_modulus)
                } else {
                    0
                };
                (residue, reduced % fusion_plain_modulus)
            })
            .collect::<Vec<_>>();
        residue_scalar_coefficients.push(interpolate_lagrange_mod(&points, fusion_plain_modulus)?);
    }

    Ok(BgvBootstrapSameFieldCrtResidueBridge {
        source_degree: crt_plan.degree,
        fusion_plain_modulus,
        phase_bound: crt_plan.phase_bound.clone(),
        auxiliary_plain_moduli: crt_plan.auxiliary_plain_moduli.clone(),
        residue_scalar_coefficients,
    })
}

pub fn bgv_bootstrap_homomorphic_same_field_crt_residue_bridge(
    encrypted_raw_phase_slots: &BgvCiphertext,
    bridge: &BgvBootstrapSameFieldCrtResidueBridge,
    evk: &EvaluationKey,
) -> Result<Vec<BgvCiphertext>, BgvLevelError> {
    if encrypted_raw_phase_slots.params().degree() != bridge.source_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: bridge.source_degree,
            actual: encrypted_raw_phase_slots.params().degree(),
        });
    }
    if encrypted_raw_phase_slots.params().plain_modulus() != bridge.fusion_plain_modulus {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: bridge.fusion_plain_modulus,
            actual: encrypted_raw_phase_slots.params().plain_modulus(),
        });
    }

    let polynomials = bridge
        .residue_scalar_coefficients
        .iter()
        .map(|coefficients| {
            coefficients
                .iter()
                .map(|&coefficient| {
                    constant_plaintext(encrypted_raw_phase_slots.params(), coefficient)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    BgvEvaluator::new(encrypted_raw_phase_slots.params().clone()).evaluate_polynomials_plain_bgv(
        encrypted_raw_phase_slots,
        &polynomials,
        evk,
    )
}

pub fn bgv_bootstrap_same_field_decode_polynomial(
    source_ct: &BgvCiphertext,
    fusion_params: &BgvParameters,
    phase_bound: &BigInt,
) -> Result<BgvBootstrapSameFieldDecodePolynomial, BgvLevelError> {
    const MAX_DECODE_LOOKUP_POINTS: u64 = 4096;

    if source_ct.params().degree() != fusion_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source_ct.params().degree(),
            actual: fusion_params.degree(),
        });
    }

    let fusion_plain_modulus = fusion_params.plain_modulus();
    let source_plain_modulus = source_ct.params().plain_modulus();
    if fusion_plain_modulus > MAX_DECODE_LOOKUP_POINTS {
        return Err(BgvLevelError::BootstrapClassifierLookupTooLarge {
            modulus: fusion_plain_modulus,
            max_points: MAX_DECODE_LOOKUP_POINTS,
        });
    }
    if fusion_plain_modulus < source_plain_modulus {
        return Err(
            BgvLevelError::BootstrapSameFieldDecodeSourceModulusTooLarge {
                fusion_modulus: fusion_plain_modulus,
                source_modulus: source_plain_modulus,
            },
        );
    }
    let required = phase_bound.abs() * BigInt::from(2u8) + BigInt::from(1u8);
    if BigInt::from(fusion_plain_modulus) <= required {
        return Err(
            BgvLevelError::BootstrapSingleModulusClassifierCoverageTooSmall {
                modulus: fusion_plain_modulus,
                required: required.to_u64().unwrap_or(u64::MAX),
            },
        );
    }

    let source_q = BigInt::from_biguint(
        Sign::Plus,
        rns_big::ciphertext_modulus(source_ct.params().ring()),
    );
    let factor_inverse = source_ct.int_factor().inverse_mod(source_plain_modulus)?;
    let phase_bound_abs = phase_bound.abs();
    let points = (0..fusion_plain_modulus)
        .map(|residue| {
            let centered = centered_residue_mod_plaintext(residue, fusion_plain_modulus);
            let decoded = if centered.abs() <= phase_bound_abs {
                let (source_centered, _wrap) = centered_residue_and_wrap(&centered, &source_q);
                arith::mul_mod_u64(
                    bigint_mod_u64(&source_centered, source_plain_modulus),
                    factor_inverse,
                    source_plain_modulus,
                )
            } else {
                0
            };
            (residue, decoded % fusion_plain_modulus)
        })
        .collect::<Vec<_>>();
    let scalar_coefficients = interpolate_lagrange_mod(&points, fusion_plain_modulus)?;

    Ok(BgvBootstrapSameFieldDecodePolynomial {
        source_degree: source_ct.params().degree(),
        fusion_plain_modulus,
        source_plain_modulus,
        factor_inverse,
        phase_bound: phase_bound.clone(),
        scalar_coefficients,
    })
}

pub fn bgv_bootstrap_homomorphic_same_field_decode(
    encrypted_raw_phase_slots: &BgvCiphertext,
    decode: &BgvBootstrapSameFieldDecodePolynomial,
    evk: &EvaluationKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    if encrypted_raw_phase_slots.params().degree() != decode.source_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: decode.source_degree,
            actual: encrypted_raw_phase_slots.params().degree(),
        });
    }
    if encrypted_raw_phase_slots.params().plain_modulus() != decode.fusion_plain_modulus {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: decode.fusion_plain_modulus,
            actual: encrypted_raw_phase_slots.params().plain_modulus(),
        });
    }

    let coefficients = decode.plaintext_coefficients(encrypted_raw_phase_slots.params())?;
    BgvEvaluator::new(encrypted_raw_phase_slots.params().clone()).evaluate_polynomial_plain_bgv(
        encrypted_raw_phase_slots,
        &coefficients,
        evk,
    )
}

pub fn audit_bgv_bootstrap_same_field_decode_lift(
    decode: &BgvBootstrapSameFieldDecodePolynomial,
) -> BgvBootstrapSameFieldDecodeLiftAudit {
    let mismatches = (0..decode.fusion_plain_modulus)
        .filter_map(|raw_phase_residue| {
            let target_mod_source =
                decode.evaluate_scalar(raw_phase_residue) % decode.source_plain_modulus;
            let centered_lift_mod_source =
                decode.evaluate_scalar_centered_lift_mod_source(raw_phase_residue);
            (target_mod_source != centered_lift_mod_source).then(|| {
                BgvBootstrapSameFieldDecodeLiftMismatch {
                    raw_phase_residue,
                    target_mod_source,
                    centered_lift_mod_source,
                    correction_mod_source: arith::sub_mod(
                        target_mod_source,
                        centered_lift_mod_source,
                        decode.source_plain_modulus,
                    ),
                }
            })
        })
        .collect::<Vec<_>>();

    BgvBootstrapSameFieldDecodeLiftAudit {
        source_degree: decode.source_degree,
        fusion_plain_modulus: decode.fusion_plain_modulus,
        source_plain_modulus: decode.source_plain_modulus,
        mismatch_count: mismatches.len(),
        mismatches,
    }
}

pub fn bgv_bootstrap_same_field_decode_lift_correction_polynomial(
    decode: &BgvBootstrapSameFieldDecodePolynomial,
) -> Result<BgvBootstrapSameFieldDecodeLiftCorrectionPolynomial, BgvLevelError> {
    let points = (0..decode.fusion_plain_modulus)
        .map(|raw_phase_residue| {
            let target_mod_source =
                decode.evaluate_scalar(raw_phase_residue) % decode.source_plain_modulus;
            let centered_lift_mod_source =
                decode.evaluate_scalar_centered_lift_mod_source(raw_phase_residue);
            (
                raw_phase_residue,
                arith::sub_mod(
                    target_mod_source,
                    centered_lift_mod_source,
                    decode.source_plain_modulus,
                ) % decode.fusion_plain_modulus,
            )
        })
        .collect::<Vec<_>>();
    let scalar_coefficients = interpolate_lagrange_mod(&points, decode.fusion_plain_modulus)?;

    Ok(BgvBootstrapSameFieldDecodeLiftCorrectionPolynomial {
        source_degree: decode.source_degree,
        fusion_plain_modulus: decode.fusion_plain_modulus,
        source_plain_modulus: decode.source_plain_modulus,
        scalar_coefficients,
    })
}

pub fn bgv_bootstrap_homomorphic_same_field_decode_lift_correction(
    encrypted_raw_phase_slots: &BgvCiphertext,
    correction: &BgvBootstrapSameFieldDecodeLiftCorrectionPolynomial,
    evk: &EvaluationKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    if encrypted_raw_phase_slots.params().degree() != correction.source_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: correction.source_degree,
            actual: encrypted_raw_phase_slots.params().degree(),
        });
    }
    if encrypted_raw_phase_slots.params().plain_modulus() != correction.fusion_plain_modulus {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: correction.fusion_plain_modulus,
            actual: encrypted_raw_phase_slots.params().plain_modulus(),
        });
    }

    let coefficients = correction.plaintext_coefficients(encrypted_raw_phase_slots.params())?;
    BgvEvaluator::new(encrypted_raw_phase_slots.params().clone()).evaluate_polynomial_plain_bgv(
        encrypted_raw_phase_slots,
        &coefficients,
        evk,
    )
}

pub fn bgv_bootstrap_same_field_fusion_refresh(
    source_ct: &BgvCiphertext,
    raw_phase_key: &BgvRecryptionKey,
    transform_gk: &GaloisKey,
    evk: &EvaluationKey,
    phase_bound: &BigInt,
) -> Result<BgvCiphertext, BgvLevelError> {
    validate_phase_residue_recryption_input(source_ct, raw_phase_key)?;

    let fusion_params = raw_phase_key.encrypted_secret_key.params();
    let decode = bgv_bootstrap_same_field_decode_polynomial(source_ct, fusion_params, phase_bound)?;
    let encrypted_raw_phase_coefficients =
        bgv_bootstrap_encrypted_phase_residue(source_ct, raw_phase_key)?;
    let encrypted_raw_phase_slots =
        bgv_bootstrap_coefficients_to_slots_bgv(&encrypted_raw_phase_coefficients, transform_gk)?;
    let decoded_slots =
        bgv_bootstrap_homomorphic_same_field_decode(&encrypted_raw_phase_slots, &decode, evk)?;
    bgv_bootstrap_slots_to_coefficients_bgv(&decoded_slots, transform_gk)
}

pub fn bgv_bootstrap_fusion_plaintext_modulus_plan(
    source_params: &BgvParameters,
    phase_bound: &BigInt,
) -> Result<BgvBootstrapFusionPlaintextModulusPlan, BgvLevelError> {
    const MAX_FUSION_LOOKUP_POINTS: u64 = 4096;

    let required = phase_bound.abs() * BigInt::from(2u8) + BigInt::from(1u8);
    let required_lookup_points =
        required
            .to_u64()
            .ok_or(BgvLevelError::BootstrapClassifierLookupTooLarge {
                modulus: u64::MAX,
                max_points: MAX_FUSION_LOOKUP_POINTS,
            })?;
    if required_lookup_points >= MAX_FUSION_LOOKUP_POINTS {
        return Err(BgvLevelError::BootstrapClassifierLookupTooLarge {
            modulus: required_lookup_points,
            max_points: MAX_FUSION_LOOKUP_POINTS,
        });
    }

    let source_plain_modulus = source_params.plain_modulus();
    let step = (source_params.degree() as u64).checked_mul(2).ok_or(
        BgvLevelError::BootstrapCrtPrimeSearchFailed {
            bits: 0,
            step: u64::MAX,
        },
    )?;
    let selected_fusion_plain_modulus = if source_plain_modulus > required_lookup_points {
        source_plain_modulus
    } else {
        let start = source_plain_modulus
            .max(required_lookup_points.saturating_add(1))
            .max(step.saturating_add(1));
        numth::next_ntt_prime(start, step)
            .ok_or(BgvLevelError::BootstrapCrtPrimeSearchFailed { bits: 0, step })?
    };
    if selected_fusion_plain_modulus > MAX_FUSION_LOOKUP_POINTS {
        return Err(BgvLevelError::BootstrapClassifierLookupTooLarge {
            modulus: selected_fusion_plain_modulus,
            max_points: MAX_FUSION_LOOKUP_POINTS,
        });
    }

    let uses_source_plaintext_modulus = selected_fusion_plain_modulus == source_plain_modulus;
    let plaintext_noise_vanishes_mod_source =
        selected_fusion_plain_modulus % source_plain_modulus == 0;
    let requires_small_plaintext_correction = !plaintext_noise_vanishes_mod_source;

    Ok(BgvBootstrapFusionPlaintextModulusPlan {
        source_degree: source_params.degree(),
        source_plain_modulus,
        required_lookup_points,
        selected_fusion_plain_modulus,
        uses_source_plaintext_modulus,
        plaintext_noise_vanishes_mod_source,
        requires_small_plaintext_correction,
    })
}

pub fn bgv_bootstrap_source_compatible_fusion_noise_scale_plan(
    source_params: &BgvParameters,
    fusion_params: &BgvParameters,
) -> Result<BgvBootstrapSourceCompatibleFusionNoiseScalePlan, BgvLevelError> {
    if source_params.degree() != fusion_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source_params.degree(),
            actual: fusion_params.degree(),
        });
    }

    let source_plain_modulus = source_params.plain_modulus();
    let fusion_plain_modulus = fusion_params.plain_modulus();
    let gcd = numth::gcd(source_plain_modulus, fusion_plain_modulus);
    let reduced_fusion = fusion_plain_modulus / gcd;
    let error_scalar = reduced_fusion.checked_mul(source_plain_modulus).ok_or(
        BgvLevelError::BootstrapFusionNoiseScaleOverflow {
            fusion_modulus: fusion_plain_modulus,
            source_modulus: source_plain_modulus,
        },
    )?;
    let bridges_without_noise_correction =
        error_scalar % source_plain_modulus == 0 && error_scalar % fusion_plain_modulus == 0;

    Ok(BgvBootstrapSourceCompatibleFusionNoiseScalePlan {
        source_degree: source_params.degree(),
        source_plain_modulus,
        fusion_plain_modulus,
        error_scalar,
        uses_default_fusion_error_scalar: error_scalar == fusion_plain_modulus,
        bridges_without_noise_correction,
    })
}

pub fn bgv_bootstrap_fusion_to_source_plaintext_bridge_plan(
    fusion_ct: &BgvCiphertext,
    source_params: &BgvParameters,
) -> Result<BgvBootstrapFusionToSourceBridgePlan, BgvLevelError> {
    if fusion_ct.params().degree() != source_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: fusion_ct.params().degree(),
            actual: source_params.degree(),
        });
    }

    let fusion_plain_modulus = fusion_ct.params().plain_modulus();
    let source_plain_modulus = source_params.plain_modulus();
    let same_ciphertext_modulus_chain =
        same_ciphertext_modulus_chain(fusion_ct.params(), source_params);
    let plaintext_noise_vanishes_mod_source = fusion_plain_modulus % source_plain_modulus == 0;
    let plaintext_factor_compatible = numth::mod_inverse(
        fusion_ct.int_factor().value() % source_plain_modulus,
        source_plain_modulus,
    )
    .is_some();
    let can_reinterpret_without_noise_correction = same_ciphertext_modulus_chain
        && plaintext_noise_vanishes_mod_source
        && plaintext_factor_compatible;
    let requires_noise_correction = same_ciphertext_modulus_chain
        && plaintext_factor_compatible
        && !plaintext_noise_vanishes_mod_source;

    Ok(BgvBootstrapFusionToSourceBridgePlan {
        source_degree: source_params.degree(),
        fusion_plain_modulus,
        source_plain_modulus,
        same_ciphertext_modulus_chain,
        plaintext_noise_vanishes_mod_source,
        plaintext_factor_compatible,
        can_reinterpret_without_noise_correction,
        requires_noise_correction,
    })
}

pub fn bgv_bootstrap_compatible_fusion_to_source_plaintext_bridge(
    fusion_ct: &BgvCiphertext,
    source_params: BgvParameters,
) -> Result<BgvCiphertext, BgvLevelError> {
    let plan = bgv_bootstrap_fusion_to_source_plaintext_bridge_plan(fusion_ct, &source_params)?;
    if !plan.same_ciphertext_modulus_chain {
        return Err(BgvLevelError::CiphertextModulusChainMismatch);
    }
    if !plan.plaintext_noise_vanishes_mod_source {
        return Err(
            BgvLevelError::BootstrapFusionToSourceBridgeRequiresNoiseCorrection {
                fusion_modulus: plan.fusion_plain_modulus,
                source_modulus: plan.source_plain_modulus,
            },
        );
    }

    let factor = BgvPlaintextFactor::new(
        fusion_ct.int_factor().value(),
        source_params.plain_modulus(),
    )?;
    BgvCiphertext::with_plaintext_factor(source_params, fusion_ct.raw().clone(), factor)
}

pub fn audit_bgv_bootstrap_fusion_to_source_noise_correction(
    fusion_ct: &BgvCiphertext,
    source_params: &BgvParameters,
    fusion_sk: &SecretKey,
) -> Result<BgvBootstrapFusionToSourceNoiseCorrectionAudit, BgvLevelError> {
    let plan = bgv_bootstrap_fusion_to_source_plaintext_bridge_plan(fusion_ct, source_params)?;
    if !plan.same_ciphertext_modulus_chain {
        return Err(BgvLevelError::CiphertextModulusChainMismatch);
    }
    let factor = BgvPlaintextFactor::new(
        fusion_ct.int_factor().value(),
        source_params.plain_modulus(),
    )?;

    let target_plaintext =
        BgvDecryptor::new(fusion_ct.params().clone(), fusion_sk.clone()).decrypt_bgv(fusion_ct);
    let direct_source_ct = BgvCiphertext::with_plaintext_factor(
        source_params.clone(),
        fusion_ct.raw().clone(),
        factor,
    )?;
    let direct_plaintext =
        BgvDecryptor::new(source_params.clone(), fusion_sk.clone()).decrypt_bgv(&direct_source_ct);

    let source_plain_modulus = source_params.plain_modulus();
    let target_coefficients = target_plaintext
        .value
        .limb(0)
        .iter()
        .map(|value| value % source_plain_modulus)
        .collect::<Vec<_>>();
    let direct_coefficients = direct_plaintext.value.limb(0).to_vec();
    let correction_coefficients = target_coefficients
        .iter()
        .zip(direct_coefficients.iter())
        .map(|(&target, &direct)| {
            arith::sub_mod(
                target % source_plain_modulus,
                direct % source_plain_modulus,
                source_plain_modulus,
            )
        })
        .collect::<Vec<_>>();
    let nonzero_correction_count = correction_coefficients
        .iter()
        .filter(|&&coefficient| coefficient != 0)
        .count();

    Ok(BgvBootstrapFusionToSourceNoiseCorrectionAudit {
        source_degree: source_params.degree(),
        fusion_plain_modulus: fusion_ct.params().plain_modulus(),
        source_plain_modulus,
        target_coefficients,
        direct_coefficients,
        correction_coefficients,
        nonzero_correction_count,
    })
}

pub fn bgv_bootstrap_fusion_to_source_bridge_with_plaintext_correction(
    fusion_ct: &BgvCiphertext,
    source_params: BgvParameters,
    correction: &Plaintext,
) -> Result<BgvCiphertext, BgvLevelError> {
    let plan = bgv_bootstrap_fusion_to_source_plaintext_bridge_plan(fusion_ct, &source_params)?;
    if !plan.same_ciphertext_modulus_chain {
        return Err(BgvLevelError::CiphertextModulusChainMismatch);
    }
    if correction.value.degree() != source_params.degree() {
        return Err(BgvLevelError::PlaintextDegreeMismatch {
            expected: source_params.degree(),
            actual: correction.value.degree(),
        });
    }
    if correction.value.num_moduli() != 1 {
        return Err(BgvLevelError::PlaintextLimbMismatch {
            expected: 1,
            actual: correction.value.num_moduli(),
        });
    }

    let factor = BgvPlaintextFactor::new(
        fusion_ct.int_factor().value(),
        source_params.plain_modulus(),
    )?;
    let direct_source_ct = BgvCiphertext::with_plaintext_factor(
        source_params.clone(),
        fusion_ct.raw().clone(),
        factor,
    )?;
    BgvEvaluator::new(source_params).add_plain_bgv(&direct_source_ct, correction)
}

pub fn bgv_bootstrap_apply_plaintext_lift_correction_to_fusion_ciphertext(
    fusion_ct: &BgvCiphertext,
    source_params: &BgvParameters,
    correction: &Plaintext,
) -> Result<BgvCiphertext, BgvLevelError> {
    let plan = bgv_bootstrap_fusion_to_source_plaintext_bridge_plan(fusion_ct, source_params)?;
    if !plan.same_ciphertext_modulus_chain {
        return Err(BgvLevelError::CiphertextModulusChainMismatch);
    }
    if !plan.plaintext_factor_compatible {
        return Err(BgvLevelError::PlaintextFactorNotInvertible {
            factor: fusion_ct.int_factor().value() % source_params.plain_modulus(),
            modulus: source_params.plain_modulus(),
        });
    }
    if correction.value.degree() != source_params.degree() {
        return Err(BgvLevelError::PlaintextDegreeMismatch {
            expected: source_params.degree(),
            actual: correction.value.degree(),
        });
    }
    if correction.value.num_moduli() != 1 {
        return Err(BgvLevelError::PlaintextLimbMismatch {
            expected: 1,
            actual: correction.value.num_moduli(),
        });
    }

    let lift_scalar_mod_source = fusion_to_source_lift_scalar_mod_source(
        fusion_ct,
        source_params,
        BgvPlaintextFactor::one(),
    )?;
    let adjusted_raw = add_fusion_plaintext_lift_to_raw_c0(
        fusion_ct.raw(),
        fusion_ct.params(),
        correction,
        lift_scalar_mod_source,
    );

    BgvCiphertext::with_metadata(
        fusion_ct.params().clone(),
        adjusted_raw,
        fusion_ct.int_factor(),
        fusion_ct.noise_scale_degree(),
    )
}

pub fn bgv_bootstrap_fusion_to_source_bridge_with_plaintext_lift_correction(
    fusion_ct: &BgvCiphertext,
    source_params: BgvParameters,
    correction: &Plaintext,
) -> Result<BgvCiphertext, BgvLevelError> {
    let adjusted_fusion = bgv_bootstrap_apply_plaintext_lift_correction_to_fusion_ciphertext(
        fusion_ct,
        &source_params,
        correction,
    )?;
    let factor = BgvPlaintextFactor::new(
        fusion_ct.int_factor().value(),
        source_params.plain_modulus(),
    )?;
    BgvCiphertext::with_metadata(
        source_params,
        adjusted_fusion.raw().clone(),
        factor,
        adjusted_fusion.noise_scale_degree(),
    )
}

pub fn bgv_bootstrap_apply_encrypted_lift_correction_to_fusion_ciphertext(
    fusion_ct: &BgvCiphertext,
    source_params: &BgvParameters,
    encrypted_correction: &BgvCiphertext,
) -> Result<BgvCiphertext, BgvLevelError> {
    let plan = bgv_bootstrap_fusion_to_source_plaintext_bridge_plan(fusion_ct, source_params)?;
    if !plan.same_ciphertext_modulus_chain {
        return Err(BgvLevelError::CiphertextModulusChainMismatch);
    }
    if !plan.plaintext_factor_compatible {
        return Err(BgvLevelError::PlaintextFactorNotInvertible {
            factor: fusion_ct.int_factor().value() % source_params.plain_modulus(),
            modulus: source_params.plain_modulus(),
        });
    }
    ensure_same_plaintext_space(fusion_ct.params(), encrypted_correction.params())?;
    if !same_ciphertext_modulus_chain(fusion_ct.params(), encrypted_correction.params()) {
        return Err(BgvLevelError::CiphertextModulusChainMismatch);
    }

    let lift_scalar_mod_source = fusion_to_source_lift_scalar_mod_source(
        fusion_ct,
        source_params,
        encrypted_correction.int_factor(),
    )?;
    let adjusted_raw = add_fusion_encrypted_lift_to_raw(
        fusion_ct.raw(),
        encrypted_correction.raw(),
        fusion_ct.params(),
        lift_scalar_mod_source,
    );

    BgvCiphertext::with_metadata(
        fusion_ct.params().clone(),
        adjusted_raw,
        fusion_ct.int_factor(),
        fusion_ct
            .noise_scale_degree()
            .max(encrypted_correction.noise_scale_degree()),
    )
}

pub fn bgv_bootstrap_fusion_to_source_bridge_with_encrypted_lift_correction(
    fusion_ct: &BgvCiphertext,
    source_params: BgvParameters,
    encrypted_correction: &BgvCiphertext,
) -> Result<BgvCiphertext, BgvLevelError> {
    let adjusted_fusion = bgv_bootstrap_apply_encrypted_lift_correction_to_fusion_ciphertext(
        fusion_ct,
        &source_params,
        encrypted_correction,
    )?;
    let factor = BgvPlaintextFactor::new(
        fusion_ct.int_factor().value(),
        source_params.plain_modulus(),
    )?;
    BgvCiphertext::with_metadata(
        source_params,
        adjusted_fusion.raw().clone(),
        factor,
        adjusted_fusion.noise_scale_degree(),
    )
}

pub fn bgv_bootstrap_fusion_to_source_bridge_with_encrypted_correction(
    fusion_ct: &BgvCiphertext,
    source_params: BgvParameters,
    encrypted_correction: &BgvCiphertext,
) -> Result<BgvCiphertext, BgvLevelError> {
    let plan = bgv_bootstrap_fusion_to_source_plaintext_bridge_plan(fusion_ct, &source_params)?;
    if !plan.same_ciphertext_modulus_chain {
        return Err(BgvLevelError::CiphertextModulusChainMismatch);
    }
    if encrypted_correction.params().degree() != source_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source_params.degree(),
            actual: encrypted_correction.params().degree(),
        });
    }
    if encrypted_correction.params().plain_modulus() != source_params.plain_modulus() {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: source_params.plain_modulus(),
            actual: encrypted_correction.params().plain_modulus(),
        });
    }

    let factor = BgvPlaintextFactor::new(
        fusion_ct.int_factor().value(),
        source_params.plain_modulus(),
    )?;
    let direct_source_ct = BgvCiphertext::with_plaintext_factor(
        source_params.clone(),
        fusion_ct.raw().clone(),
        factor,
    )?;
    BgvEvaluator::new(source_params).add_bgv(&direct_source_ct, encrypted_correction)
}

pub fn bgv_bootstrap_fusion_to_source_recryption_with_encrypted_correction(
    fusion_ct: &BgvCiphertext,
    direct_phase_key: &BgvRecryptionKey,
    encrypted_correction: &BgvCiphertext,
) -> Result<BgvCiphertext, BgvLevelError> {
    validate_phase_residue_recryption_input(fusion_ct, direct_phase_key)?;
    ensure_same_plaintext_space(
        direct_phase_key.encrypted_secret_key.params(),
        encrypted_correction.params(),
    )?;

    let direct_phase = bgv_bootstrap_encrypted_phase_residue(fusion_ct, direct_phase_key)?;
    BgvEvaluator::new(encrypted_correction.params().clone())
        .add_bgv(&direct_phase, encrypted_correction)
}

pub fn bgv_bootstrap_same_field_crt_wrap_classifier(
    classifier_params: &BgvParameters,
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
) -> Result<BgvBootstrapSameFieldCrtWrapClassifier, BgvLevelError> {
    const MAX_SAME_FIELD_CRT_TUPLES: u64 = 4096;

    if classifier_params.degree() != crt_plan.degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: crt_plan.degree,
            actual: classifier_params.degree(),
        });
    }

    let classifier_plain_modulus = classifier_params.plain_modulus();
    let mut tuple_count = 1u64;
    for &residue_modulus in &crt_plan.auxiliary_plain_moduli {
        if classifier_plain_modulus <= residue_modulus {
            return Err(BgvLevelError::BootstrapSameFieldCrtResidueModulusTooLarge {
                classifier_modulus: classifier_plain_modulus,
                residue_modulus,
            });
        }
        tuple_count = tuple_count.checked_mul(residue_modulus).ok_or(
            BgvLevelError::BootstrapSameFieldCrtTupleCountTooLarge {
                tuple_count: u64::MAX,
                max_tuples: MAX_SAME_FIELD_CRT_TUPLES,
            },
        )?;
        if tuple_count > MAX_SAME_FIELD_CRT_TUPLES {
            return Err(BgvLevelError::BootstrapSameFieldCrtTupleCountTooLarge {
                tuple_count,
                max_tuples: MAX_SAME_FIELD_CRT_TUPLES,
            });
        }
    }

    let mut indicator_scalar_coefficients =
        Vec::with_capacity(crt_plan.auxiliary_plain_moduli.len());
    for &residue_modulus in &crt_plan.auxiliary_plain_moduli {
        let mut indicators = Vec::with_capacity(residue_modulus as usize);
        for target_residue in 0..residue_modulus {
            let points = (0..residue_modulus)
                .map(|residue| {
                    (
                        residue,
                        if residue == target_residue {
                            1 % classifier_plain_modulus
                        } else {
                            0
                        },
                    )
                })
                .collect::<Vec<_>>();
            indicators.push(interpolate_lagrange_mod(&points, classifier_plain_modulus)?);
        }
        indicator_scalar_coefficients.push(indicators);
    }

    let mut nonzero_terms = Vec::new();
    enumerate_same_field_crt_terms(
        crt_plan,
        classifier_plain_modulus,
        0,
        &mut Vec::with_capacity(crt_plan.auxiliary_plain_moduli.len()),
        &mut nonzero_terms,
    )?;

    Ok(BgvBootstrapSameFieldCrtWrapClassifier {
        source_degree: crt_plan.degree,
        classifier_plain_modulus,
        auxiliary_plain_moduli: crt_plan.auxiliary_plain_moduli.clone(),
        max_abs_wrap_count: crt_plan.max_abs_wrap_count,
        tuple_count,
        indicator_scalar_coefficients,
        nonzero_terms,
    })
}

pub fn bgv_bootstrap_homomorphic_same_field_crt_wrap_classification(
    encrypted_residue_slots_by_modulus: &[BgvCiphertext],
    classifier: &BgvBootstrapSameFieldCrtWrapClassifier,
    evk: &EvaluationKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    if encrypted_residue_slots_by_modulus.len() != classifier.auxiliary_plain_moduli.len() {
        return Err(BgvLevelError::BootstrapCrtResidueCountMismatch {
            expected: classifier.auxiliary_plain_moduli.len(),
            actual: encrypted_residue_slots_by_modulus.len(),
        });
    }
    let first = encrypted_residue_slots_by_modulus.first().ok_or(
        BgvLevelError::BootstrapCrtResidueCountMismatch {
            expected: classifier.auxiliary_plain_moduli.len(),
            actual: 0,
        },
    )?;
    if first.params().degree() != classifier.source_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: classifier.source_degree,
            actual: first.params().degree(),
        });
    }
    if first.params().plain_modulus() != classifier.classifier_plain_modulus {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: classifier.classifier_plain_modulus,
            actual: first.params().plain_modulus(),
        });
    }

    let evaluator = BgvEvaluator::new(first.params().clone());
    let mut indicator_ciphertexts = Vec::with_capacity(encrypted_residue_slots_by_modulus.len());
    for (modulus_index, residue_ct) in encrypted_residue_slots_by_modulus.iter().enumerate() {
        ensure_same_plaintext_space(first.params(), residue_ct.params())?;
        let polynomials = classifier.indicator_plaintexts(first.params(), modulus_index)?;
        indicator_ciphertexts.push(evaluator.evaluate_polynomials_plain_bgv(
            residue_ct,
            &polynomials,
            evk,
        )?);
    }

    let zero = constant_plaintext(first.params(), 0);
    let mut acc = evaluator.mul_plain_bgv(first, &zero)?;
    for term in &classifier.nonzero_terms {
        let mut term_ct = indicator_ciphertexts[0][term.residues[0] as usize].clone();
        for (modulus_index, &residue) in term.residues.iter().enumerate().skip(1) {
            term_ct = evaluator.mul_relinearized_bgv(
                &term_ct,
                &indicator_ciphertexts[modulus_index][residue as usize],
                evk,
            )?;
        }
        let weighted = evaluator.mul_plain_bgv(
            &term_ct,
            &constant_plaintext(first.params(), term.wrap_scalar),
        )?;
        acc = evaluator.add_bgv(&acc, &weighted)?;
    }

    Ok(acc)
}

pub fn bgv_bootstrap_same_field_plaintext_modulus_switch_correction_classifier(
    source_ct: &BgvCiphertext,
    classifier_params: &BgvParameters,
    target_params: &BgvParameters,
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
) -> Result<BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionClassifier, BgvLevelError> {
    const MAX_SAME_FIELD_CRT_TUPLES: u64 = 4096;

    if source_ct.params().degree() != classifier_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source_ct.params().degree(),
            actual: classifier_params.degree(),
        });
    }
    if source_ct.params().degree() != target_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source_ct.params().degree(),
            actual: target_params.degree(),
        });
    }
    if source_ct.params().degree() != crt_plan.degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source_ct.params().degree(),
            actual: crt_plan.degree,
        });
    }

    let classifier_plain_modulus = classifier_params.plain_modulus();
    let target_plain_modulus = target_params.plain_modulus();
    if classifier_plain_modulus < target_plain_modulus {
        return Err(
            BgvLevelError::BootstrapSameFieldDecodeSourceModulusTooLarge {
                fusion_modulus: classifier_plain_modulus,
                source_modulus: target_plain_modulus,
            },
        );
    }

    let mut tuple_count = 1u64;
    for &residue_modulus in &crt_plan.auxiliary_plain_moduli {
        if classifier_plain_modulus <= residue_modulus {
            return Err(BgvLevelError::BootstrapSameFieldCrtResidueModulusTooLarge {
                classifier_modulus: classifier_plain_modulus,
                residue_modulus,
            });
        }
        tuple_count = tuple_count.checked_mul(residue_modulus).ok_or(
            BgvLevelError::BootstrapSameFieldCrtTupleCountTooLarge {
                tuple_count: u64::MAX,
                max_tuples: MAX_SAME_FIELD_CRT_TUPLES,
            },
        )?;
        if tuple_count > MAX_SAME_FIELD_CRT_TUPLES {
            return Err(BgvLevelError::BootstrapSameFieldCrtTupleCountTooLarge {
                tuple_count,
                max_tuples: MAX_SAME_FIELD_CRT_TUPLES,
            });
        }
    }

    let mut indicator_scalar_coefficients =
        Vec::with_capacity(crt_plan.auxiliary_plain_moduli.len());
    for &residue_modulus in &crt_plan.auxiliary_plain_moduli {
        let mut indicators = Vec::with_capacity(residue_modulus as usize);
        for target_residue in 0..residue_modulus {
            let points = (0..residue_modulus)
                .map(|residue| {
                    (
                        residue,
                        if residue == target_residue {
                            1 % classifier_plain_modulus
                        } else {
                            0
                        },
                    )
                })
                .collect::<Vec<_>>();
            indicators.push(interpolate_lagrange_mod(&points, classifier_plain_modulus)?);
        }
        indicator_scalar_coefficients.push(indicators);
    }

    let source_q = BigInt::from_biguint(
        Sign::Plus,
        rns_big::ciphertext_modulus(source_ct.params().ring()),
    );
    let source_factor_inverse = source_ct
        .int_factor()
        .inverse_mod(source_ct.params().plain_modulus())?;
    let mut nonzero_terms = Vec::new();
    enumerate_plaintext_modulus_switch_correction_terms(
        crt_plan,
        classifier_plain_modulus,
        source_ct.params().plain_modulus(),
        target_plain_modulus,
        source_factor_inverse,
        &source_q,
        0,
        &mut Vec::with_capacity(crt_plan.auxiliary_plain_moduli.len()),
        &mut nonzero_terms,
    )?;

    Ok(
        BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionClassifier {
            source_degree: source_ct.params().degree(),
            classifier_plain_modulus,
            source_plain_modulus: source_ct.params().plain_modulus(),
            target_plain_modulus,
            source_factor_inverse,
            auxiliary_plain_moduli: crt_plan.auxiliary_plain_moduli.clone(),
            tuple_count,
            indicator_scalar_coefficients,
            nonzero_terms,
        },
    )
}

pub fn bgv_bootstrap_homomorphic_same_field_plaintext_modulus_switch_correction(
    encrypted_residue_slots_by_modulus: &[BgvCiphertext],
    classifier: &BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionClassifier,
    evk: &EvaluationKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    if encrypted_residue_slots_by_modulus.len() != classifier.auxiliary_plain_moduli.len() {
        return Err(BgvLevelError::BootstrapCrtResidueCountMismatch {
            expected: classifier.auxiliary_plain_moduli.len(),
            actual: encrypted_residue_slots_by_modulus.len(),
        });
    }
    let first = encrypted_residue_slots_by_modulus.first().ok_or(
        BgvLevelError::BootstrapCrtResidueCountMismatch {
            expected: classifier.auxiliary_plain_moduli.len(),
            actual: 0,
        },
    )?;
    if first.params().degree() != classifier.source_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: classifier.source_degree,
            actual: first.params().degree(),
        });
    }
    if first.params().plain_modulus() != classifier.classifier_plain_modulus {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: classifier.classifier_plain_modulus,
            actual: first.params().plain_modulus(),
        });
    }

    let evaluator = BgvEvaluator::new(first.params().clone());
    let mut indicator_ciphertexts = Vec::with_capacity(encrypted_residue_slots_by_modulus.len());
    for (modulus_index, residue_ct) in encrypted_residue_slots_by_modulus.iter().enumerate() {
        ensure_same_plaintext_space(first.params(), residue_ct.params())?;
        let polynomials = classifier.indicator_plaintexts(first.params(), modulus_index)?;
        indicator_ciphertexts.push(evaluator.evaluate_polynomials_plain_bgv(
            residue_ct,
            &polynomials,
            evk,
        )?);
    }

    let zero = constant_plaintext(first.params(), 0);
    let mut acc = evaluator.mul_plain_bgv(first, &zero)?;
    for term in &classifier.nonzero_terms {
        let mut term_ct = indicator_ciphertexts[0][term.residues[0] as usize].clone();
        for (modulus_index, &residue) in term.residues.iter().enumerate().skip(1) {
            term_ct = evaluator.mul_relinearized_bgv(
                &term_ct,
                &indicator_ciphertexts[modulus_index][residue as usize],
                evk,
            )?;
        }
        let weighted = evaluator.mul_plain_bgv(
            &term_ct,
            &constant_plaintext(first.params(), term.correction_scalar),
        )?;
        acc = evaluator.add_bgv(&acc, &weighted)?;
    }

    Ok(acc)
}

pub fn bgv_bootstrap_homomorphic_same_field_plaintext_modulus_switch(
    source_ct: &BgvCiphertext,
    direct_phase_key: &BgvRecryptionKey,
    encrypted_residue_slots_by_modulus: &[BgvCiphertext],
    classifier: &BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionClassifier,
    transform_gk: &GaloisKey,
    evk: &EvaluationKey,
) -> Result<BgvCiphertext, BgvLevelError> {
    validate_phase_residue_recryption_input(source_ct, direct_phase_key)?;
    if direct_phase_key
        .encrypted_secret_key
        .params()
        .plain_modulus()
        != classifier.classifier_plain_modulus
    {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: direct_phase_key
                .encrypted_secret_key
                .params()
                .plain_modulus(),
            actual: classifier.classifier_plain_modulus,
        });
    }

    let direct_phase = bgv_bootstrap_encrypted_phase_residue(source_ct, direct_phase_key)?;
    let correction_slots =
        bgv_bootstrap_homomorphic_same_field_plaintext_modulus_switch_correction(
            encrypted_residue_slots_by_modulus,
            classifier,
            evk,
        )?;
    let correction_coefficients =
        bgv_bootstrap_slots_to_coefficients_bgv(&correction_slots, transform_gk)?;
    BgvEvaluator::new(direct_phase.params().clone())
        .add_bgv(&direct_phase, &correction_coefficients)
}

pub fn bgv_bootstrap_circuit_readiness(
    ct: &BgvCiphertext,
    key: &BgvRecryptionKey,
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
) -> Result<BgvBootstrapCircuitReadiness, BgvLevelError> {
    validate_recryption_input(ct, key)?;
    if ct.params().degree() != crt_plan.degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: ct.params().degree(),
            actual: crt_plan.degree,
        });
    }

    let gaps = vec![BgvBootstrapCircuitGap::HomomorphicFusionToSourcePlaintextBridge];
    Ok(BgvBootstrapCircuitReadiness {
        source_degree: ct.params().degree(),
        source_q_limbs: ct.q_modulus_count(),
        target_q_limbs: key.encrypted_secret_key.params().q_modulus_count(),
        phase_bound_bits: crt_plan.phase_bound.bits(),
        crt_modulus_product_bits: crt_plan.modulus_product.bits(),
        auxiliary_plain_moduli: crt_plan.auxiliary_plain_moduli.clone(),
        max_abs_wrap_count: crt_plan.max_abs_wrap_count,
        raw_phase_affine_input_layer: true,
        encrypted_phase_mod_plaintext: true,
        encrypted_crt_phase_residues: true,
        crt_wrap_reconstruction_plan: true,
        extracted_crt_residue_repack: true,
        encrypted_coefficient_carry_correction: true,
        corrected_recryption: true,
        homomorphic_single_modulus_residue_classification: true,
        homomorphic_same_field_crt_residue_bridge: true,
        homomorphic_same_field_crt_residue_fusion: true,
        homomorphic_same_field_plaintext_modulus_switch_correction: true,
        homomorphic_same_field_plaintext_modulus_switch: true,
        homomorphic_same_field_phase_decode: true,
        homomorphic_same_field_fusion_refresh: true,
        fusion_plaintext_modulus_planner: true,
        source_compatible_fusion_noise_scaling: true,
        same_field_decode_lift_correction: true,
        homomorphic_same_field_decode_lift_correction: true,
        fusion_to_source_plaintext_bridge_plan: true,
        fusion_to_source_plaintext_noise_correction_skeleton: true,
        fusion_to_source_plaintext_lift_patch: true,
        fusion_to_source_encrypted_lift_patch: true,
        fusion_to_source_encrypted_noise_correction_bridge: true,
        fusion_to_source_recryption_with_encrypted_correction: true,
        fusion_to_source_extracted_crt_plaintext_modulus_switch: true,
        homomorphic_fusion_to_source_plaintext_bridge: false,
        encrypted_coefficient_wrap_repacking: true,
        full_bootstrapping_circuit: false,
        gaps,
    })
}

pub fn bgv_bootstrap_carry_correction_polynomial(
    params: &BgvParameters,
    max_abs_wrap_count: u64,
    q_mod_plaintext: u64,
    factor_inverse: u64,
) -> Result<BgvBootstrapCarryPolynomial, BgvLevelError> {
    let t = params.plain_modulus();
    if max_abs_wrap_count
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .map(|point_count| point_count > t)
        .unwrap_or(true)
    {
        return Err(BgvLevelError::BootstrapCarryBoundTooLarge {
            bound: max_abs_wrap_count,
            modulus: t,
        });
    }

    let q_factor = arith::mul_mod_u64(q_mod_plaintext % t, factor_inverse % t, t);
    let mut points = Vec::with_capacity((2 * max_abs_wrap_count + 1) as usize);
    for wrap in -(max_abs_wrap_count as i64)..=(max_abs_wrap_count as i64) {
        let x = signed_i64_to_mod(wrap, t);
        let signed = (-i128::from(wrap)).rem_euclid(i128::from(t)) as u64;
        let y = arith::mul_mod_u64(signed, q_factor, t);
        points.push((x, y));
    }

    let scalar_coefficients = interpolate_lagrange_mod(&points, t)?;
    Ok(BgvBootstrapCarryPolynomial {
        plain_modulus: t,
        max_abs_wrap_count,
        q_mod_plaintext: q_mod_plaintext % t,
        factor_inverse: factor_inverse % t,
        scalar_coefficients,
    })
}

pub fn bgv_bootstrap_wrap_interval_plan(
    params: &BgvParameters,
    max_abs_wrap_count: u64,
) -> Result<BgvBootstrapWrapIntervalPlan, BgvLevelError> {
    if max_abs_wrap_count > i64::MAX as u64 {
        return Err(BgvLevelError::BootstrapCarryBoundTooLarge {
            bound: max_abs_wrap_count,
            modulus: params.plain_modulus(),
        });
    }

    let q = BigInt::from_biguint(Sign::Plus, rns_big::ciphertext_modulus(params.ring()));
    let half = &q >> 1usize;
    let lower_offset = &q - &half - BigInt::from(1u8);
    let mut intervals = Vec::with_capacity((2 * max_abs_wrap_count + 1) as usize);
    for wrap in -(max_abs_wrap_count as i64)..=(max_abs_wrap_count as i64) {
        let center = &q * wrap;
        intervals.push(BgvBootstrapWrapInterval {
            wrap_count: wrap,
            lower_inclusive: &center - &lower_offset,
            upper_inclusive: center + &half,
        });
    }

    Ok(BgvBootstrapWrapIntervalPlan {
        max_abs_wrap_count,
        intervals,
    })
}

pub fn bgv_bootstrap_phase_linearization_plan(
    ct: &BgvCiphertext,
) -> Result<BgvBootstrapPhaseLinearizationPlan, BgvLevelError> {
    if ct.raw().data.len() != 2 {
        return Err(BgvLevelError::RecryptionCiphertextSize {
            actual: ct.raw().data.len(),
        });
    }

    let c0 = ciphertext_component_centered_coefficients(ct, 0)?;
    let c1 = ciphertext_component_centered_coefficients(ct, 1)?;
    let degree = ct.params().degree();
    let mut max_abs_constant = BigInt::from(0u8);
    let mut max_abs_multiplier = BigInt::from(0u8);
    let mut coefficients = Vec::with_capacity(degree);

    for coefficient_index in 0..degree {
        let constant = c0[coefficient_index].clone();
        max_abs_constant = max_abs_constant.max(constant.abs());
        let mut terms = Vec::with_capacity(degree);
        for secret_coefficient_index in 0..degree {
            let multiplier = if secret_coefficient_index <= coefficient_index {
                c1[coefficient_index - secret_coefficient_index].clone()
            } else {
                -c1[coefficient_index + degree - secret_coefficient_index].clone()
            };
            max_abs_multiplier = max_abs_multiplier.max(multiplier.abs());
            terms.push(BgvBootstrapPhaseTerm {
                secret_coefficient_index,
                multiplier,
            });
        }
        coefficients.push(BgvBootstrapPhaseCoefficientPlan {
            coefficient_index,
            constant,
            terms,
        });
    }

    let ternary_secret_abs_phase_bound =
        &max_abs_constant + &max_abs_multiplier * BigInt::from(degree);
    Ok(BgvBootstrapPhaseLinearizationPlan {
        degree,
        coefficients,
        max_abs_constant,
        max_abs_multiplier,
        ternary_secret_abs_phase_bound,
    })
}

pub fn bgv_bootstrap_crt_wrap_reconstruction_plan(
    params: &BgvParameters,
    phase_plan: &BgvBootstrapPhaseLinearizationPlan,
) -> Result<BgvBootstrapCrtWrapReconstructionPlan, BgvLevelError> {
    let auxiliary_plain_moduli =
        select_auxiliary_plain_moduli(params.degree(), &phase_plan.ternary_secret_abs_phase_bound)?;
    bgv_bootstrap_crt_wrap_reconstruction_plan_from_auxiliary_moduli(
        params,
        phase_plan,
        auxiliary_plain_moduli,
    )
}

pub fn bgv_bootstrap_crt_wrap_reconstruction_plan_from_auxiliary_moduli(
    params: &BgvParameters,
    phase_plan: &BgvBootstrapPhaseLinearizationPlan,
    auxiliary_plain_moduli: Vec<u64>,
) -> Result<BgvBootstrapCrtWrapReconstructionPlan, BgvLevelError> {
    if params.degree() != phase_plan.degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: params.degree(),
            actual: phase_plan.degree,
        });
    }

    let phase_bound = phase_plan.ternary_secret_abs_phase_bound.clone();
    let modulus_product = auxiliary_plain_moduli
        .iter()
        .fold(BigInt::from(1u8), |acc, &modulus| {
            acc * BigInt::from(modulus)
        });
    let required = phase_bound.abs() * BigInt::from(2u8) + BigInt::from(1u8);
    if modulus_product <= required {
        return Err(BgvLevelError::BootstrapCrtModulusProductTooSmall {
            product_bits: modulus_product.bits(),
            required_bits: required.bits(),
        });
    }
    let max_abs_wrap_count = max_abs_wrap_count_for_phase_bound(params, &phase_bound)?;
    let wrap_interval_plan = bgv_bootstrap_wrap_interval_plan(params, max_abs_wrap_count)?;

    Ok(BgvBootstrapCrtWrapReconstructionPlan {
        degree: params.degree(),
        auxiliary_plain_moduli,
        modulus_product,
        phase_bound,
        max_abs_wrap_count,
        wrap_interval_plan,
    })
}

impl BgvBootstrapCarryPolynomial {
    pub fn evaluate_scalar(&self, wrap_count: i64) -> u64 {
        let x = signed_i64_to_mod(wrap_count, self.plain_modulus);
        self.scalar_coefficients
            .iter()
            .rev()
            .fold(0u64, |acc, &coefficient| {
                let product = arith::mul_mod_u64(acc, x, self.plain_modulus);
                arith::add_mod(product, coefficient, self.plain_modulus)
            })
    }

    pub fn plaintext_coefficients(
        &self,
        params: &BgvParameters,
    ) -> Result<Vec<Plaintext>, BgvLevelError> {
        if params.plain_modulus() != self.plain_modulus {
            return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
                expected: self.plain_modulus,
                actual: params.plain_modulus(),
            });
        }
        Ok(self
            .scalar_coefficients
            .iter()
            .map(|&coefficient| constant_plaintext(params, coefficient))
            .collect())
    }
}

impl BgvBootstrapSingleModulusWrapClassifierPolynomial {
    pub fn evaluate_scalar(&self, residue: u64) -> u64 {
        let x = residue % self.classifier_plain_modulus;
        self.scalar_coefficients
            .iter()
            .rev()
            .fold(0u64, |acc, &coefficient| {
                let product = arith::mul_mod_u64(acc, x, self.classifier_plain_modulus);
                arith::add_mod(product, coefficient, self.classifier_plain_modulus)
            })
    }

    pub fn plaintext_coefficients(
        &self,
        params: &BgvParameters,
    ) -> Result<Vec<Plaintext>, BgvLevelError> {
        if params.plain_modulus() != self.classifier_plain_modulus {
            return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
                expected: self.classifier_plain_modulus,
                actual: params.plain_modulus(),
            });
        }
        if params.degree() != self.source_degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.source_degree,
                actual: params.degree(),
            });
        }
        Ok(self
            .scalar_coefficients
            .iter()
            .map(|&coefficient| constant_plaintext(params, coefficient))
            .collect())
    }
}

impl BgvBootstrapSameFieldCrtWrapClassifier {
    pub fn indicator_plaintexts(
        &self,
        params: &BgvParameters,
        modulus_index: usize,
    ) -> Result<Vec<Vec<Plaintext>>, BgvLevelError> {
        if params.degree() != self.source_degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.source_degree,
                actual: params.degree(),
            });
        }
        if params.plain_modulus() != self.classifier_plain_modulus {
            return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
                expected: self.classifier_plain_modulus,
                actual: params.plain_modulus(),
            });
        }
        let indicators = self
            .indicator_scalar_coefficients
            .get(modulus_index)
            .ok_or(BgvLevelError::BootstrapCrtResidueCountMismatch {
                expected: self.auxiliary_plain_moduli.len(),
                actual: modulus_index + 1,
            })?;
        Ok(indicators
            .iter()
            .map(|polynomial| {
                polynomial
                    .iter()
                    .map(|&coefficient| constant_plaintext(params, coefficient))
                    .collect::<Vec<_>>()
            })
            .collect())
    }
}

impl BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionClassifier {
    pub fn indicator_plaintexts(
        &self,
        params: &BgvParameters,
        modulus_index: usize,
    ) -> Result<Vec<Vec<Plaintext>>, BgvLevelError> {
        if params.degree() != self.source_degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.source_degree,
                actual: params.degree(),
            });
        }
        if params.plain_modulus() != self.classifier_plain_modulus {
            return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
                expected: self.classifier_plain_modulus,
                actual: params.plain_modulus(),
            });
        }
        let indicators = self
            .indicator_scalar_coefficients
            .get(modulus_index)
            .ok_or(BgvLevelError::BootstrapCrtResidueCountMismatch {
                expected: self.auxiliary_plain_moduli.len(),
                actual: modulus_index + 1,
            })?;
        Ok(indicators
            .iter()
            .map(|polynomial| {
                polynomial
                    .iter()
                    .map(|&coefficient| constant_plaintext(params, coefficient))
                    .collect::<Vec<_>>()
            })
            .collect())
    }
}

impl BgvBootstrapSameFieldDecodePolynomial {
    pub fn evaluate_scalar(&self, raw_phase_residue: u64) -> u64 {
        let x = raw_phase_residue % self.fusion_plain_modulus;
        self.scalar_coefficients
            .iter()
            .rev()
            .fold(0u64, |acc, &coefficient| {
                let product = arith::mul_mod_u64(acc, x, self.fusion_plain_modulus);
                arith::add_mod(product, coefficient, self.fusion_plain_modulus)
            })
    }

    pub fn evaluate_scalar_centered_lift_mod_source(&self, raw_phase_residue: u64) -> u64 {
        let x = centered_residue_mod_plaintext(raw_phase_residue, self.fusion_plain_modulus);
        let value =
            self.scalar_coefficients
                .iter()
                .rev()
                .fold(BigInt::from(0u8), |acc, &coefficient| {
                    acc * &x
                        + centered_residue_mod_plaintext(coefficient, self.fusion_plain_modulus)
                });
        bigint_mod_u64(&value, self.source_plain_modulus)
    }

    pub fn plaintext_coefficients(
        &self,
        params: &BgvParameters,
    ) -> Result<Vec<Plaintext>, BgvLevelError> {
        if params.degree() != self.source_degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.source_degree,
                actual: params.degree(),
            });
        }
        if params.plain_modulus() != self.fusion_plain_modulus {
            return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
                expected: self.fusion_plain_modulus,
                actual: params.plain_modulus(),
            });
        }
        Ok(self
            .scalar_coefficients
            .iter()
            .map(|&coefficient| constant_plaintext(params, coefficient))
            .collect())
    }
}

impl BgvBootstrapSameFieldDecodeLiftCorrectionPolynomial {
    pub fn evaluate_scalar(&self, raw_phase_residue: u64) -> u64 {
        let x = raw_phase_residue % self.fusion_plain_modulus;
        self.scalar_coefficients
            .iter()
            .rev()
            .fold(0u64, |acc, &coefficient| {
                let product = arith::mul_mod_u64(acc, x, self.fusion_plain_modulus);
                arith::add_mod(product, coefficient, self.fusion_plain_modulus)
            })
    }

    pub fn plaintext_coefficients(
        &self,
        params: &BgvParameters,
    ) -> Result<Vec<Plaintext>, BgvLevelError> {
        if params.degree() != self.source_degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.source_degree,
                actual: params.degree(),
            });
        }
        if params.plain_modulus() != self.fusion_plain_modulus {
            return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
                expected: self.fusion_plain_modulus,
                actual: params.plain_modulus(),
            });
        }
        Ok(self
            .scalar_coefficients
            .iter()
            .map(|&coefficient| constant_plaintext(params, coefficient))
            .collect())
    }
}

impl BgvBootstrapFusionToSourceNoiseCorrectionAudit {
    pub fn correction_plaintext(&self, params: &BgvParameters) -> Result<Plaintext, BgvLevelError> {
        if params.degree() != self.source_degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.source_degree,
                actual: params.degree(),
            });
        }
        if params.plain_modulus() != self.source_plain_modulus {
            return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
                expected: self.source_plain_modulus,
                actual: params.plain_modulus(),
            });
        }

        let mut poly = Poly::new(params.degree(), 1);
        for (dst, &coefficient) in poly
            .limb_mut(0)
            .iter_mut()
            .zip(self.correction_coefficients.iter())
        {
            *dst = coefficient % params.plain_modulus();
        }
        Ok(Plaintext { value: poly })
    }
}

impl BgvBootstrapWrapIntervalPlan {
    pub fn classify(&self, raw_phase_coefficient: &BigInt) -> Option<i64> {
        self.intervals
            .iter()
            .find(|interval| {
                raw_phase_coefficient >= &interval.lower_inclusive
                    && raw_phase_coefficient <= &interval.upper_inclusive
            })
            .map(|interval| interval.wrap_count)
    }
}

impl BgvBootstrapPhaseLinearizationPlan {
    pub fn evaluate_with_secret_key(
        &self,
        params: &BgvParameters,
        sk: &SecretKey,
    ) -> Result<Vec<BigInt>, BgvLevelError> {
        if params.degree() != self.degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.degree,
                actual: params.degree(),
            });
        }
        let secret = secret_key_signed_coefficients(params, sk)?;
        self.evaluate_with_secret_coefficients(&secret)
    }

    pub fn evaluate_mod_plaintext_with_secret_key(
        &self,
        params: &BgvParameters,
        sk: &SecretKey,
    ) -> Result<Vec<u64>, BgvLevelError> {
        let raw = self.evaluate_with_secret_key(params, sk)?;
        Ok(raw
            .iter()
            .map(|coefficient| bigint_mod_u64(coefficient, params.plain_modulus()))
            .collect())
    }

    pub fn evaluate_with_secret_coefficients(
        &self,
        secret_coefficients: &[i64],
    ) -> Result<Vec<BigInt>, BgvLevelError> {
        if secret_coefficients.len() != self.degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: self.degree,
                actual: secret_coefficients.len(),
            });
        }

        Ok(self
            .coefficients
            .iter()
            .map(|coefficient| {
                let mut value = coefficient.constant.clone();
                for term in &coefficient.terms {
                    let secret = secret_coefficients[term.secret_coefficient_index];
                    if secret != 0 && !term.multiplier.is_zero() {
                        value += &term.multiplier * secret;
                    }
                }
                value
            })
            .collect())
    }

    pub fn evaluate_mod_plaintext_with_secret_coefficients(
        &self,
        secret_coefficients: &[i64],
        plain_modulus: u64,
    ) -> Result<Vec<u64>, BgvLevelError> {
        let raw = self.evaluate_with_secret_coefficients(secret_coefficients)?;
        Ok(raw
            .iter()
            .map(|coefficient| bigint_mod_u64(coefficient, plain_modulus))
            .collect())
    }
}

impl BgvBootstrapCrtWrapReconstructionPlan {
    pub fn residues_for_raw_phase(
        &self,
        raw_phase_coefficients: &[BigInt],
    ) -> Result<Vec<Vec<u64>>, BgvLevelError> {
        if raw_phase_coefficients.len() != self.degree {
            return Err(BgvLevelError::BootstrapCrtCoefficientCountMismatch {
                expected: self.degree,
                actual: raw_phase_coefficients.len(),
            });
        }

        Ok(raw_phase_coefficients
            .iter()
            .map(|coefficient| {
                self.auxiliary_plain_moduli
                    .iter()
                    .map(|&modulus| bigint_mod_u64(coefficient, modulus))
                    .collect()
            })
            .collect())
    }

    pub fn reconstruct_centered_coefficient(
        &self,
        residues: &[u64],
    ) -> Result<BigInt, BgvLevelError> {
        if residues.len() != self.auxiliary_plain_moduli.len() {
            return Err(BgvLevelError::BootstrapCrtResidueCountMismatch {
                expected: self.auxiliary_plain_moduli.len(),
                actual: residues.len(),
            });
        }

        let mut value = BigInt::from(0u8);
        for (&residue, &modulus) in residues.iter().zip(self.auxiliary_plain_moduli.iter()) {
            let punctured = &self.modulus_product / BigInt::from(modulus);
            let punctured_mod = bigint_mod_u64(&punctured, modulus);
            let inv = numth::mod_inverse(punctured_mod, modulus).ok_or(
                BgvLevelError::BootstrapCarryInterpolationDenominatorNotInvertible {
                    denominator: punctured_mod,
                    modulus,
                },
            )?;
            value += punctured * BigInt::from(inv) * BigInt::from(residue % modulus);
        }

        value %= &self.modulus_product;
        if value.is_negative() {
            value += &self.modulus_product;
        }
        let half = &self.modulus_product >> 1usize;
        if value > half {
            value -= &self.modulus_product;
        }
        Ok(value)
    }

    pub fn reconstruct_centered_phase(
        &self,
        residues_by_coefficient: &[Vec<u64>],
    ) -> Result<Vec<BigInt>, BgvLevelError> {
        if residues_by_coefficient.len() != self.degree {
            return Err(BgvLevelError::BootstrapCrtCoefficientCountMismatch {
                expected: self.degree,
                actual: residues_by_coefficient.len(),
            });
        }
        residues_by_coefficient
            .iter()
            .map(|residues| self.reconstruct_centered_coefficient(residues))
            .collect()
    }

    pub fn classify_wrap_counts_from_residues(
        &self,
        residues_by_coefficient: &[Vec<u64>],
    ) -> Result<Vec<i64>, BgvLevelError> {
        let raw_phase = self.reconstruct_centered_phase(residues_by_coefficient)?;
        raw_phase
            .iter()
            .enumerate()
            .map(|(coefficient, raw)| {
                self.wrap_interval_plan
                    .classify(raw)
                    .ok_or(BgvLevelError::BootstrapCrtReconstructionOutOfRange { coefficient })
            })
            .collect()
    }
}

pub fn audit_bgv_bootstrap_reduction(
    ct: &BgvCiphertext,
    source_sk: &SecretKey,
) -> Result<BgvBootstrapReductionAudit, BgvLevelError> {
    if ct.raw().data.len() != 2 {
        return Err(BgvLevelError::RecryptionCiphertextSize {
            actual: ct.raw().data.len(),
        });
    }

    let t = ct.params().plain_modulus();
    let factor_inv = ct.int_factor().inverse_mod(t)?;
    let q = rns_big::ciphertext_modulus(ct.params().ring());
    let q_bigint = BigInt::from_biguint(Sign::Plus, q.clone());
    let decryptor = BgvDecryptor::new(ct.params().clone(), source_sk.clone());
    let true_plaintext = decryptor.decrypt_bgv(ct);
    let true_coefficients = true_plaintext.value.limb(0).to_vec();

    let raw_phase_coefficients = raw_binary_phase_coefficients(ct, source_sk)?;
    let secret = secret_key_plaintext(ct.params(), source_sk)?;
    let mut c0 = ciphertext_component_plaintext(ct, 0)?;
    let mut c1 = ciphertext_component_plaintext(ct, 1)?;
    scale_plaintext_inplace(&mut c0, t, factor_inv);
    scale_plaintext_inplace(&mut c1, t, factor_inv);

    let mut component_reduced_coefficients =
        negacyclic_mul_mod(c1.value.limb(0), secret.value.limb(0), t);
    add_coefficients_assign(&mut component_reduced_coefficients, c0.value.limb(0), t);

    let correction_coefficients = true_coefficients
        .iter()
        .zip(component_reduced_coefficients.iter())
        .map(|(actual, reduced)| (actual + t - reduced) % t)
        .collect::<Vec<_>>();
    let nonzero_correction_count = correction_coefficients
        .iter()
        .filter(|&&value| value != 0)
        .count();
    let (wrap_counts, wrap_correction_coefficients, max_abs_wrap_count) =
        wrap_correction_coefficients(&raw_phase_coefficients, &q_bigint, t, factor_inv);

    Ok(BgvBootstrapReductionAudit {
        raw_phase_coefficients,
        true_coefficients,
        component_reduced_coefficients,
        correction_coefficients,
        wrap_counts,
        wrap_correction_coefficients,
        nonzero_correction_count,
        max_abs_wrap_count,
        q_mod_plaintext: (&q % t).to_u64().expect("Q mod t fits in u64"),
        factor_inverse: factor_inv,
    })
}

fn ciphertext_modulus_mod_plaintext(params: &BgvParameters) -> u64 {
    let t = params.plain_modulus();
    (&rns_big::ciphertext_modulus(params.ring()) % t)
        .to_u64()
        .expect("Q mod plaintext modulus fits in u64")
}

fn select_auxiliary_plain_moduli(
    degree: usize,
    phase_bound: &BigInt,
) -> Result<Vec<u64>, BgvLevelError> {
    const AUXILIARY_BITS: u32 = 50;

    let step =
        (degree as u64)
            .checked_mul(2)
            .ok_or(BgvLevelError::BootstrapCrtPrimeSearchFailed {
                bits: AUXILIARY_BITS,
                step: u64::MAX,
            })?;
    let mut search_start = (1u64 << AUXILIARY_BITS) - 1;
    let target = phase_bound.abs() * BigInt::from(2u8) + BigInt::from(1u8);
    let mut product = BigInt::from(1u8);
    let mut moduli = Vec::new();

    while product <= target {
        let modulus = numth::prev_ntt_prime(search_start, step).ok_or(
            BgvLevelError::BootstrapCrtPrimeSearchFailed {
                bits: AUXILIARY_BITS,
                step,
            },
        )?;
        moduli.push(modulus);
        product *= BigInt::from(modulus);
        search_start = modulus.saturating_sub(step);
    }

    Ok(moduli)
}

fn enumerate_same_field_crt_terms(
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
    classifier_plain_modulus: u64,
    modulus_index: usize,
    current: &mut Vec<u64>,
    terms: &mut Vec<BgvBootstrapSameFieldCrtWrapTerm>,
) -> Result<(), BgvLevelError> {
    if modulus_index == crt_plan.auxiliary_plain_moduli.len() {
        let raw = crt_plan.reconstruct_centered_coefficient(current)?;
        let wrap = crt_plan.wrap_interval_plan.classify(&raw).ok_or(
            BgvLevelError::BootstrapCrtReconstructionOutOfRange {
                coefficient: terms.len(),
            },
        )?;
        if wrap != 0 {
            terms.push(BgvBootstrapSameFieldCrtWrapTerm {
                residues: current.clone(),
                wrap_count: wrap,
                wrap_scalar: signed_i64_to_mod(wrap, classifier_plain_modulus),
            });
        }
        return Ok(());
    }

    for residue in 0..crt_plan.auxiliary_plain_moduli[modulus_index] {
        current.push(residue);
        enumerate_same_field_crt_terms(
            crt_plan,
            classifier_plain_modulus,
            modulus_index + 1,
            current,
            terms,
        )?;
        current.pop();
    }

    Ok(())
}

fn enumerate_plaintext_modulus_switch_correction_terms(
    crt_plan: &BgvBootstrapCrtWrapReconstructionPlan,
    classifier_plain_modulus: u64,
    source_plain_modulus: u64,
    target_plain_modulus: u64,
    source_factor_inverse: u64,
    source_q: &BigInt,
    modulus_index: usize,
    current: &mut Vec<u64>,
    terms: &mut Vec<BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionTerm>,
) -> Result<(), BgvLevelError> {
    if modulus_index == crt_plan.auxiliary_plain_moduli.len() {
        let raw = crt_plan.reconstruct_centered_coefficient(current)?;
        let (centered_phase, _wrap) = centered_residue_and_wrap(&raw, source_q);
        let source_plaintext = arith::mul_mod_u64(
            bigint_mod_u64(&centered_phase, source_plain_modulus),
            source_factor_inverse,
            source_plain_modulus,
        ) % target_plain_modulus;
        let direct_target = bigint_mod_u64(&raw, target_plain_modulus);
        let correction = arith::sub_mod(source_plaintext, direct_target, target_plain_modulus);
        if correction != 0 {
            terms.push(BgvBootstrapSameFieldPlaintextModulusSwitchCorrectionTerm {
                residues: current.clone(),
                correction_scalar: correction % classifier_plain_modulus,
            });
        }
        return Ok(());
    }

    for residue in 0..crt_plan.auxiliary_plain_moduli[modulus_index] {
        current.push(residue);
        enumerate_plaintext_modulus_switch_correction_terms(
            crt_plan,
            classifier_plain_modulus,
            source_plain_modulus,
            target_plain_modulus,
            source_factor_inverse,
            source_q,
            modulus_index + 1,
            current,
            terms,
        )?;
        current.pop();
    }

    Ok(())
}

fn fusion_to_source_lift_scalar_mod_source(
    fusion_ct: &BgvCiphertext,
    source_params: &BgvParameters,
    correction_factor: BgvPlaintextFactor,
) -> Result<u64, BgvLevelError> {
    let source_plain_modulus = source_params.plain_modulus();
    let fusion_mod_source = fusion_ct.params().plain_modulus() % source_plain_modulus;
    let fusion_mod_source_inverse = numth::mod_inverse(fusion_mod_source, source_plain_modulus)
        .ok_or(BgvLevelError::PlaintextFactorNotInvertible {
            factor: fusion_mod_source,
            modulus: source_plain_modulus,
        })?;
    let correction_factor_inverse = numth::mod_inverse(
        correction_factor.value() % source_plain_modulus,
        source_plain_modulus,
    )
    .ok_or(BgvLevelError::PlaintextFactorNotInvertible {
        factor: correction_factor.value() % source_plain_modulus,
        modulus: source_plain_modulus,
    })?;
    let source_factor = fusion_ct.int_factor().value() % source_plain_modulus;
    let lift_without_correction_factor = arith::mul_mod_u64(
        source_factor,
        fusion_mod_source_inverse,
        source_plain_modulus,
    );
    Ok(arith::mul_mod_u64(
        lift_without_correction_factor,
        correction_factor_inverse,
        source_plain_modulus,
    ))
}

fn add_fusion_plaintext_lift_to_raw_c0(
    raw: &Ciphertext,
    params: &BgvParameters,
    correction: &Plaintext,
    lift_scalar_mod_source: u64,
) -> Ciphertext {
    let ring = params.ring();
    let mut adjusted = raw.clone();
    if adjusted.is_ntt {
        for poly in adjusted.data.iter_mut() {
            poly.ntt_inverse(ring);
        }
        adjusted.is_ntt = false;
    }

    let fusion_plain_modulus = params.plain_modulus();
    let correction_coefficients = correction.value.limb(0);
    for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        let fusion_factor = fusion_plain_modulus % q;
        let lift_factor = lift_scalar_mod_source % q;
        let c0_limb = adjusted.data[0].limb_mut(limb_idx);
        for (dst, &correction_coefficient) in c0_limb.iter_mut().zip(correction_coefficients.iter())
        {
            let lifted = arith::mul_mod_u64(
                fusion_factor,
                arith::mul_mod_u64(lift_factor, correction_coefficient % q, q),
                q,
            );
            *dst = arith::add_mod(*dst, lifted, q);
        }
    }

    adjusted
}

fn add_fusion_encrypted_lift_to_raw(
    raw: &Ciphertext,
    encrypted_lift: &Ciphertext,
    params: &BgvParameters,
    lift_scalar_mod_source: u64,
) -> Ciphertext {
    let ring = params.ring();
    let mut adjusted = raw.clone();
    if adjusted.is_ntt {
        for poly in adjusted.data.iter_mut() {
            poly.ntt_inverse(ring);
        }
        adjusted.is_ntt = false;
    }

    let mut lift_raw = encrypted_lift.clone();
    if lift_raw.is_ntt {
        for poly in lift_raw.data.iter_mut() {
            poly.ntt_inverse(ring);
        }
        lift_raw.is_ntt = false;
    }

    while adjusted.data.len() < lift_raw.data.len() {
        adjusted
            .data
            .push(Poly::new(params.degree(), params.q_modulus_count()));
    }

    let fusion_plain_modulus = params.plain_modulus();
    for (dst_poly, lift_poly) in adjusted.data.iter_mut().zip(lift_raw.data.iter()) {
        for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
            let q = modulus.value();
            let scalar =
                arith::mul_mod_u64(fusion_plain_modulus % q, lift_scalar_mod_source % q, q);
            let dst_limb = dst_poly.limb_mut(limb_idx);
            let lift_limb = lift_poly.limb(limb_idx);
            for (dst, &lift_coeff) in dst_limb.iter_mut().zip(lift_limb.iter()) {
                let lifted = arith::mul_mod_u64(lift_coeff, scalar, q);
                *dst = arith::add_mod(*dst, lifted, q);
            }
        }
    }

    adjusted
}

fn max_abs_wrap_count_for_phase_bound(
    params: &BgvParameters,
    phase_bound: &BigInt,
) -> Result<u64, BgvLevelError> {
    let q = BigInt::from_biguint(Sign::Plus, rns_big::ciphertext_modulus(params.ring()));
    let (_, positive_wrap) = centered_residue_and_wrap(phase_bound, &q);
    let negative_bound = -phase_bound;
    let (_, negative_wrap) = centered_residue_and_wrap(&negative_bound, &q);
    positive_wrap.abs().max(negative_wrap.abs()).to_u64().ok_or(
        BgvLevelError::BootstrapCarryBoundTooLarge {
            bound: u64::MAX,
            modulus: params.plain_modulus(),
        },
    )
}

fn bigint_mod_u64(value: &BigInt, modulus: u64) -> u64 {
    let modulus_big = BigInt::from(modulus);
    let mut rem = value % &modulus_big;
    if rem.is_negative() {
        rem += modulus_big;
    }
    rem.to_u64().expect("BigInt residue modulo u64 fits")
}

fn centered_residue_mod_plaintext(residue: u64, modulus: u64) -> BigInt {
    let reduced = residue % modulus;
    let half = modulus >> 1usize;
    if reduced > half {
        BigInt::from(reduced) - BigInt::from(modulus)
    } else {
        BigInt::from(reduced)
    }
}

fn validate_recryption_input(
    ct: &BgvCiphertext,
    key: &BgvRecryptionKey,
) -> Result<(), BgvLevelError> {
    if ct.raw().data.len() != 2 {
        return Err(BgvLevelError::RecryptionCiphertextSize {
            actual: ct.raw().data.len(),
        });
    }
    if ct.params().degree() != key.source_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: key.source_degree,
            actual: ct.params().degree(),
        });
    }
    if ct.params().plain_modulus() != key.source_plain_modulus {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: key.source_plain_modulus,
            actual: ct.params().plain_modulus(),
        });
    }
    ensure_same_plaintext_space(ct.params(), key.encrypted_secret_key.params())
}

fn validate_phase_residue_recryption_input(
    ct: &BgvCiphertext,
    key: &BgvRecryptionKey,
) -> Result<(), BgvLevelError> {
    if ct.raw().data.len() != 2 {
        return Err(BgvLevelError::RecryptionCiphertextSize {
            actual: ct.raw().data.len(),
        });
    }
    if ct.params().degree() != key.source_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: key.source_degree,
            actual: ct.params().degree(),
        });
    }
    if ct.params().plain_modulus() != key.source_plain_modulus {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: key.source_plain_modulus,
            actual: ct.params().plain_modulus(),
        });
    }
    if key.encrypted_secret_key.params().degree() != key.source_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: key.source_degree,
            actual: key.encrypted_secret_key.params().degree(),
        });
    }
    Ok(())
}

fn same_ciphertext_modulus_chain(left: &BgvParameters, right: &BgvParameters) -> bool {
    left.ring().rns().moduli() == right.ring().rns().moduli()
}

fn ensure_same_plaintext_space(
    source: &BgvParameters,
    target: &BgvParameters,
) -> Result<(), BgvLevelError> {
    if source.degree() != target.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: source.degree(),
            actual: target.degree(),
        });
    }
    if source.plain_modulus() != target.plain_modulus() {
        return Err(BgvLevelError::CiphertextPlaintextModulusMismatch {
            expected: source.plain_modulus(),
            actual: target.plain_modulus(),
        });
    }
    Ok(())
}

fn secret_key_plaintext(
    params: &BgvParameters,
    sk: &SecretKey,
) -> Result<Plaintext, BgvLevelError> {
    let coefficients = secret_key_centered_bigint_coefficients(params, sk)?;
    Ok(plaintext_from_centered_coefficients(params, &coefficients))
}

fn secret_key_centered_bigint_coefficients(
    params: &BgvParameters,
    sk: &SecretKey,
) -> Result<Vec<BigInt>, BgvLevelError> {
    if sk.value.degree() != params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: params.degree(),
            actual: sk.value.degree(),
        });
    }
    if sk.value.num_moduli() < params.q_modulus_count() {
        return Err(BgvLevelError::CiphertextLimbMismatch {
            expected: params.q_modulus_count(),
            actual: sk.value.num_moduli(),
        });
    }

    let mut coeff = Poly::new(params.degree(), params.q_modulus_count());
    coeff
        .data_mut()
        .copy_from_slice(&sk.value.data()[..params.degree() * params.q_modulus_count()]);
    coeff.ntt_inverse(params.ring());
    Ok(centered_coefficients(params, &coeff))
}

fn ciphertext_component_plaintext(
    ct: &BgvCiphertext,
    component: usize,
) -> Result<Plaintext, BgvLevelError> {
    let mut coeff = ct.raw().data[component].clone();
    if ct.raw().is_ntt {
        coeff.ntt_inverse(ct.params().ring());
    }
    Ok(plaintext_from_centered_poly(ct.params(), &coeff))
}

fn ciphertext_component_plaintext_for_params(
    ct: &BgvCiphertext,
    component: usize,
    target_params: &BgvParameters,
) -> Result<Plaintext, BgvLevelError> {
    if ct.params().degree() != target_params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: ct.params().degree(),
            actual: target_params.degree(),
        });
    }
    let coefficients = ciphertext_component_centered_coefficients(ct, component)?;
    Ok(plaintext_from_centered_coefficients(
        target_params,
        &coefficients,
    ))
}

fn plaintext_from_centered_coefficients(
    params: &BgvParameters,
    coefficients: &[BigInt],
) -> Plaintext {
    let mut out = Poly::new(params.degree(), 1);
    for (dst, coefficient) in out.limb_mut(0).iter_mut().zip(coefficients.iter()) {
        *dst = bigint_mod_u64(coefficient, params.plain_modulus());
    }
    Plaintext { value: out }
}

fn plaintext_from_centered_poly(params: &BgvParameters, poly: &Poly) -> Plaintext {
    let ring = params.ring();
    let degree = ring.degree();
    let size_q = ring.rns().len();
    let t = params.plain_modulus();
    let q = rns_big::ciphertext_modulus(ring);
    let mut out = Poly::new(degree, 1);
    let mut residues = vec![0u64; size_q];
    for coeff_idx in 0..degree {
        for (limb_idx, residue) in residues.iter_mut().enumerate() {
            *residue = poly.limb(limb_idx)[coeff_idx];
        }
        let centered = rns_big::compose_centered(&residues, ring, &q);
        let value = if centered.is_negative() {
            let abs = (-centered)
                .to_biguint()
                .expect("absolute centered coefficient converts to BigUint");
            let rem = (&abs % t).to_u64().expect("remainder fits in u64");
            if rem == 0 { 0 } else { t - rem }
        } else {
            (&centered.to_biguint().expect("non-negative coefficient") % t)
                .to_u64()
                .expect("remainder fits in u64")
        };
        out.limb_mut(0)[coeff_idx] = value;
    }
    Plaintext { value: out }
}

fn scale_plaintext_inplace(plaintext: &mut Plaintext, modulus: u64, scalar: u64) {
    let factor = scalar % modulus;
    for value in plaintext.value.limb_mut(0) {
        *value = arith::mul_mod_u64(*value, factor, modulus);
    }
}

fn constant_plaintext(params: &BgvParameters, value: u64) -> Plaintext {
    BgvBatchEncoder::new(params.clone())
        .encode(&vec![value % params.plain_modulus(); params.degree()])
}

fn coefficient_scalar_plaintext(params: &BgvParameters, value: u64) -> Plaintext {
    let mut poly = Poly::new(params.degree(), 1);
    poly.limb_mut(0)[0] = value % params.plain_modulus();
    Plaintext { value: poly }
}

fn interpolate_lagrange_mod(
    points: &[(u64, u64)],
    modulus: u64,
) -> Result<Vec<u64>, BgvLevelError> {
    let mut result = vec![0u64; points.len()];

    for (i, &(x_i, y_i)) in points.iter().enumerate() {
        let mut basis = vec![1u64];
        let mut denominator = 1u64;

        for (j, &(x_j, _)) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            denominator =
                arith::mul_mod_u64(denominator, arith::sub_mod(x_i, x_j, modulus), modulus);

            let mut next = vec![0u64; basis.len() + 1];
            for (degree, &coefficient) in basis.iter().enumerate() {
                let constant_term =
                    arith::mul_mod_u64(coefficient, arith::sub_mod(0, x_j, modulus), modulus);
                next[degree] = arith::add_mod(next[degree], constant_term, modulus);
                next[degree + 1] = arith::add_mod(next[degree + 1], coefficient, modulus);
            }
            basis = next;
        }

        let inv_den = numth::mod_inverse(denominator, modulus).ok_or(
            BgvLevelError::BootstrapCarryInterpolationDenominatorNotInvertible {
                denominator,
                modulus,
            },
        )?;
        let scale = arith::mul_mod_u64(y_i, inv_den, modulus);
        for (degree, &coefficient) in basis.iter().enumerate() {
            let term = arith::mul_mod_u64(coefficient, scale, modulus);
            result[degree] = arith::add_mod(result[degree], term, modulus);
        }
    }

    while result.len() > 1 && result.last() == Some(&0) {
        result.pop();
    }
    Ok(result)
}

fn raw_binary_phase_coefficients(
    ct: &BgvCiphertext,
    source_sk: &SecretKey,
) -> Result<Vec<BigInt>, BgvLevelError> {
    let c0 = ciphertext_component_centered_coefficients(ct, 0)?;
    let c1 = ciphertext_component_centered_coefficients(ct, 1)?;
    let secret = secret_key_signed_coefficients(ct.params(), source_sk)?;
    let mut phase = negacyclic_mul_centered(&c1, &secret);
    for (dst, addend) in phase.iter_mut().zip(c0.into_iter()) {
        *dst += addend;
    }
    Ok(phase)
}

fn ciphertext_component_centered_coefficients(
    ct: &BgvCiphertext,
    component: usize,
) -> Result<Vec<BigInt>, BgvLevelError> {
    let mut coeff = ct.raw().data[component].clone();
    if ct.raw().is_ntt {
        coeff.ntt_inverse(ct.params().ring());
    }
    Ok(centered_coefficients(ct.params(), &coeff))
}

fn secret_key_signed_coefficients(
    params: &BgvParameters,
    sk: &SecretKey,
) -> Result<Vec<i64>, BgvLevelError> {
    if sk.value.degree() != params.degree() {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: params.degree(),
            actual: sk.value.degree(),
        });
    }
    if sk.value.num_moduli() < params.q_modulus_count() {
        return Err(BgvLevelError::CiphertextLimbMismatch {
            expected: params.q_modulus_count(),
            actual: sk.value.num_moduli(),
        });
    }

    let mut coeff = Poly::new(params.degree(), params.q_modulus_count());
    coeff
        .data_mut()
        .copy_from_slice(&sk.value.data()[..params.degree() * params.q_modulus_count()]);
    coeff.ntt_inverse(params.ring());
    centered_coefficients(params, &coeff)
        .into_iter()
        .map(|value| {
            value.to_i64().ok_or(BgvLevelError::CiphertextLimbMismatch {
                expected: params.q_modulus_count(),
                actual: sk.value.num_moduli(),
            })
        })
        .collect()
}

fn centered_coefficients(params: &BgvParameters, poly: &Poly) -> Vec<BigInt> {
    let ring = params.ring();
    let degree = ring.degree();
    let size_q = ring.rns().len();
    let q = rns_big::ciphertext_modulus(ring);
    let mut residues = vec![0u64; size_q];
    (0..degree)
        .map(|coeff_idx| {
            for (limb_idx, residue) in residues.iter_mut().enumerate() {
                *residue = poly.limb(limb_idx)[coeff_idx];
            }
            rns_big::compose_centered(&residues, ring, &q)
        })
        .collect()
}

fn negacyclic_mul_centered(left: &[BigInt], right: &[i64]) -> Vec<BigInt> {
    let degree = left.len();
    let mut out = vec![BigInt::from(0u8); degree];
    for (i, lhs) in left.iter().enumerate() {
        for (j, &rhs) in right.iter().enumerate() {
            if rhs == 0 {
                continue;
            }
            let mut product = lhs * rhs;
            let index = i + j;
            if index >= degree {
                product = -product;
                out[index - degree] += product;
            } else {
                out[index] += product;
            }
        }
    }
    out
}

fn wrap_correction_coefficients(
    raw_coefficients: &[BigInt],
    q: &BigInt,
    plaintext_modulus: u64,
    factor_inv: u64,
) -> (Vec<i64>, Vec<u64>, u64) {
    let q_mod_t = (q % plaintext_modulus)
        .to_u64()
        .expect("Q mod plaintext modulus fits in u64");
    let q_factor = arith::mul_mod_u64(q_mod_t, factor_inv % plaintext_modulus, plaintext_modulus);
    let mut wrap_counts = Vec::with_capacity(raw_coefficients.len());
    let mut corrections = Vec::with_capacity(raw_coefficients.len());
    let mut max_abs = 0u64;

    for raw in raw_coefficients {
        let (_centered, wrap) = centered_residue_and_wrap(raw, q);
        let wrap_i64 = wrap.to_i64().expect("BGV bootstrap wrap count fits in i64");
        max_abs = max_abs.max(wrap_i64.unsigned_abs());
        let signed = (-i128::from(wrap_i64)).rem_euclid(i128::from(plaintext_modulus)) as u64;
        let correction = arith::mul_mod_u64(signed, q_factor, plaintext_modulus);
        wrap_counts.push(wrap_i64);
        corrections.push(correction);
    }

    (wrap_counts, corrections, max_abs)
}

fn centered_residue_and_wrap(value: &BigInt, modulus: &BigInt) -> (BigInt, BigInt) {
    let mut quotient = value / modulus;
    let mut residue = value % modulus;
    if residue.is_negative() {
        quotient -= 1;
        residue += modulus;
    }

    let half = modulus >> 1usize;
    if residue > half {
        (residue - modulus, quotient + 1)
    } else {
        (residue, quotient)
    }
}

fn signed_i64_to_mod(value: i64, modulus: u64) -> u64 {
    if value >= 0 {
        (value as u64) % modulus
    } else {
        let magnitude = value.unsigned_abs() % modulus;
        if magnitude == 0 {
            0
        } else {
            modulus - magnitude
        }
    }
}

fn negacyclic_mul_mod(left: &[u64], right: &[u64], modulus: u64) -> Vec<u64> {
    let degree = left.len();
    let mut out = vec![0u64; degree];
    for (i, &lhs) in left.iter().enumerate() {
        for (j, &rhs) in right.iter().enumerate() {
            let product = arith::mul_mod_u64(lhs % modulus, rhs % modulus, modulus);
            let index = i + j;
            if index < degree {
                out[index] = arith::add_mod(out[index], product, modulus);
            } else if product != 0 {
                out[index - degree] = arith::sub_mod(out[index - degree], product, modulus);
            }
        }
    }
    out
}

fn add_coefficients_assign(accumulator: &mut [u64], addend: &[u64], modulus: u64) {
    for (dst, &rhs) in accumulator.iter_mut().zip(addend.iter()) {
        *dst = arith::add_mod(*dst, rhs % modulus, modulus);
    }
}
