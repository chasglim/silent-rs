use silent_hss::HssContext;
use silent_params::{
    CorrectnessMarginBits, DistributionType, HssParams, MaxLinearTerms, MaxRmultDepth, ModulusBits,
    PlaintextModulus, ReconstructionBoundBits, RingDim, RingParams, RlweParams, ScaleBits,
    SecurityLevel, ShareModulusBits,
};
use silent_rlwe::EncryptionParams;

use crate::error::OperatorError;

/// Build canonical HSS metadata for an already materialized RLWE runtime.
///
/// The low-level SILENT core keeps runtime RNS objects and auditable parameter
/// metadata separate. example's prototype code historically passed raw
/// `EncryptionParams`; this helper repairs that boundary so callers can use the
/// context-first HSS APIs without regenerating or changing moduli.
pub fn context_from_runtime(
    name: &'static str,
    runtime: EncryptionParams,
    plaintext_modulus: u64,
) -> Result<HssContext, OperatorError> {
    context_from_runtime_with_security(name, runtime, plaintext_modulus, SecurityLevel::Toy)
}

/// Build HSS metadata for an already materialized RLWE runtime with an explicit
/// security label.
///
/// Use this when the caller can justify the runtime parameters independently
/// from SILENT's preset registry. Test/smoke runtimes should keep using
/// [`context_from_runtime`], which deliberately labels the synthesized tiny
/// runtime as `Toy`.
pub fn context_from_runtime_with_security(
    name: &'static str,
    runtime: EncryptionParams,
    plaintext_modulus: u64,
    security_level: SecurityLevel,
) -> Result<HssContext, OperatorError> {
    let degree = runtime.ring.degree();
    let log_n = RingDim(degree)
        .to_log_n()
        .ok_or(OperatorError::InvalidParams(
            "ring degree must be a power of two",
        ))?;

    let ciphertext_modulus_bits = runtime
        .ring
        .rns()
        .moduli()
        .iter()
        .map(|modulus| ModulusBits(bit_width(modulus.value())))
        .collect::<Vec<_>>();
    let special_modulus_bits = runtime
        .rns_tool
        .base_p()
        .map(|base| {
            base.moduli()
                .iter()
                .map(|modulus| ModulusBits(bit_width(modulus.value())))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let key_switch_modulus_bits = runtime
        .key_switch_modulus
        .map(|modulus| ModulusBits(bit_width(modulus.value())));
    let share_modulus_bits = ciphertext_modulus_bits
        .iter()
        .map(|bits| bits.0)
        .sum::<u16>();

    let params = HssParams {
        name,
        rlwe: RlweParams::new(
            name,
            RingParams::new(log_n, RingDim(degree), DistributionType::Ternary),
            ciphertext_modulus_bits,
            special_modulus_bits,
            key_switch_modulus_bits,
            security_level,
        ),
        plaintext_modulus: PlaintextModulus(plaintext_modulus),
        share_modulus_bits: ShareModulusBits(share_modulus_bits),
        max_linear_terms: MaxLinearTerms(1),
        max_rmult_depth: MaxRmultDepth(0),
        fixed_point_scale_bits: None::<ScaleBits>,
        reconstruction_bound_bits: Some(ReconstructionBoundBits(share_modulus_bits)),
        correctness_margin_bits: Some(CorrectnessMarginBits(0)),
    };

    HssContext::from_runtime_parts(params, runtime).map_err(OperatorError::Param)
}

#[inline]
fn bit_width(value: u64) -> u16 {
    if value == 0 {
        0
    } else {
        (u64::BITS - value.leading_zeros()) as u16
    }
}
