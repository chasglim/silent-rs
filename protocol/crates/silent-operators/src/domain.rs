use rand::{RngCore, SeedableRng};
use silent_hss::{
    HssBatchEncoder, HssCiphertext, HssContext, HssEncryptor, HssEvaluator, HssShare,
};
use silent_rlwe::PublicKey;
use silent_utils::rng::SecureRng;

use crate::error::OperatorError;

/// Result of converting an HSS/HE ciphertext into additive HSS memory shares.
#[derive(Clone, Debug)]
pub struct HeToShareOutput {
    pub party0: HssShare,
    pub party1: HssShare,
}

/// Explicit domain conversion between SILENT's HSS-backed HE domain and
/// additive-share domain.
///
/// This is the production Rust replacement for the prototype `Share2HE` path.
/// It intentionally exposes the two operations separately:
///
/// - `he_to_hss_shares` runs distributed decryption into fresh additive shares.
/// - `share_coeffs_to_he` encrypts coefficient-domain arithmetic shares.
/// - `share_slots_to_he` encodes SIMD slot shares before encrypting them.
pub struct DomainConverter;

impl DomainConverter {
    pub fn he_to_hss_shares<R: RngCore + ?Sized>(
        context: &HssContext,
        key_share0: &HssShare,
        key_share1: &HssShare,
        ciphertext: &HssCiphertext,
        rng: &mut R,
    ) -> Result<HeToShareOutput, OperatorError> {
        let mut secure_rng = derive_secure_rng(rng);
        let (party0, party1) = HssEvaluator::dec_share_pair(
            context,
            key_share0,
            key_share1,
            ciphertext,
            &mut secure_rng,
        )?;
        Ok(HeToShareOutput { party0, party1 })
    }

    pub fn share_coeffs_to_he<R0: RngCore + ?Sized, R1: RngCore + ?Sized>(
        context: &HssContext,
        public_key: &PublicKey,
        party0_coeffs: &[u64],
        party1_coeffs: &[u64],
        rng0: &mut R0,
        rng1: &mut R1,
    ) -> Result<HssCiphertext, OperatorError> {
        validate_share_shapes(context, party0_coeffs, party1_coeffs)?;

        let pt0 = poly_from_coeffs(context, party0_coeffs);
        let pt1 = poly_from_coeffs(context, party1_coeffs);

        let mut enc0 =
            HssEncryptor::new(context.clone(), public_key.clone(), derive_secure_rng(rng0))?;
        let mut enc1 =
            HssEncryptor::new(context.clone(), public_key.clone(), derive_secure_rng(rng1))?;

        let ct0 = enc0.encrypt_input_poly(&pt0)?;
        let ct1 = enc1.encrypt_input_poly(&pt1)?;
        Ok(HssEvaluator::add(context, &ct0, &ct1))
    }

    pub fn share_slots_to_he<R0: RngCore + ?Sized, R1: RngCore + ?Sized>(
        context: &HssContext,
        public_key: &PublicKey,
        party0_slots: &[u64],
        party1_slots: &[u64],
        rng0: &mut R0,
        rng1: &mut R1,
    ) -> Result<HssCiphertext, OperatorError> {
        validate_share_shapes(context, party0_slots, party1_slots)?;

        let encoder = HssBatchEncoder::from_context(context);
        let pt0 = encoder.encode(&canonicalize(party0_slots, context.plain_modulus()));
        let pt1 = encoder.encode(&canonicalize(party1_slots, context.plain_modulus()));

        let mut enc0 =
            HssEncryptor::new(context.clone(), public_key.clone(), derive_secure_rng(rng0))?;
        let mut enc1 =
            HssEncryptor::new(context.clone(), public_key.clone(), derive_secure_rng(rng1))?;

        let ct0 = enc0.encrypt_input_poly(&pt0)?;
        let ct1 = enc1.encrypt_input_poly(&pt1)?;
        Ok(HssEvaluator::add(context, &ct0, &ct1))
    }

    pub fn memory_shares_to_he<R0: RngCore + ?Sized, R1: RngCore + ?Sized>(
        context: &HssContext,
        public_key: &PublicKey,
        party0: &HssShare,
        party1: &HssShare,
        rng0: &mut R0,
        rng1: &mut R1,
    ) -> Result<HssCiphertext, OperatorError> {
        let p = context.plain_modulus();
        let share0 = HssEvaluator::output_mem(context, party0, p)?;
        let share1 = HssEvaluator::output_mem(context, party1, p)?;
        Self::share_coeffs_to_he(context, public_key, &share0, &share1, rng0, rng1)
    }

    pub fn reconstruct_coeffs(
        context: &HssContext,
        party0: HssShare,
        party1: HssShare,
    ) -> Result<Vec<u64>, OperatorError> {
        Ok(HssEvaluator::reconstruct(context, &[party0, party1])?)
    }
}

fn validate_share_shapes(
    context: &HssContext,
    party0_coeffs: &[u64],
    party1_coeffs: &[u64],
) -> Result<(), OperatorError> {
    if party0_coeffs.len() != party1_coeffs.len() {
        return Err(OperatorError::InvalidParams(
            "Share2HE party coefficient lengths must match",
        ));
    }
    if party0_coeffs.len() > context.degree() {
        return Err(OperatorError::InvalidParams(
            "Share2HE coefficient vector exceeds ring degree",
        ));
    }
    Ok(())
}

fn canonicalize(values: &[u64], modulus: u64) -> Vec<u64> {
    values.iter().map(|value| value % modulus).collect()
}

fn poly_from_coeffs(context: &HssContext, values: &[u64]) -> silent_ring::Poly {
    let mut poly = silent_ring::Poly::new(context.degree(), 1);
    let limb = poly.limb_mut(0);
    for (idx, value) in values.iter().enumerate() {
        limb[idx] = value % context.plain_modulus();
    }
    poly
}

fn derive_secure_rng<R: RngCore + ?Sized>(rng: &mut R) -> SecureRng {
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    SecureRng::from_seed(seed)
}
