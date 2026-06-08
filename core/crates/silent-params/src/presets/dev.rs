use crate::distribution::DistributionType;
use crate::fhe::{BfvParams, BgvParams, CkksParams};
use crate::hss::HssParams;
use crate::newtypes::{
    CarryModulus, CiphertextModulusLog, CorrectnessMarginBits, DecompositionBaseLog,
    DecompositionLevelCount, EncryptionKeyChoice, GlweDimension, Log2PFail, LogN, LweDimension,
    MaxLinearTerms, MaxRmultDepth, MessageModulus, ModulusBits, MultiplicativeDepth,
    NoiseDistribution, PlaintextModulus, PolynomialSize, ReconstructionBoundBits, RingDim,
    ScaleBits, ShareModulusBits,
};
use crate::ring::RingParams;
use crate::rlwe::RlweParams;
use crate::security::SecurityLevel;
use crate::tfhe::TfheParams;

fn rlwe_4096() -> RlweParams {
    RlweParams::new(
        "dev-rlwe-4096-v1",
        RingParams::new(LogN(12), RingDim(4096), DistributionType::Ternary),
        vec![ModulusBits(54), ModulusBits(55)],
        Vec::new(),
        None,
        SecurityLevel::Classical128,
    )
}

fn rlwe_8192() -> RlweParams {
    RlweParams::new(
        "dev-rlwe-8192-v1",
        RingParams::new(LogN(13), RingDim(8192), DistributionType::Ternary),
        vec![
            ModulusBits(50),
            ModulusBits(50),
            ModulusBits(50),
            ModulusBits(50),
        ],
        Vec::new(),
        Some(ModulusBits(18)),
        SecurityLevel::Classical128,
    )
}

pub fn dev_bfv_4096() -> BfvParams {
    BfvParams {
        name: "dev-bfv-4096-v1",
        rlwe: rlwe_4096(),
        plaintext_modulus: PlaintextModulus(65_537),
        multiplicative_depth: MultiplicativeDepth(1),
    }
}

pub fn dev_bgv_4096() -> BgvParams {
    BgvParams {
        name: "dev-bgv-4096-v1",
        rlwe: rlwe_4096(),
        plaintext_modulus: PlaintextModulus(65_537),
        multiplicative_depth: MultiplicativeDepth(1),
    }
}

pub fn dev_ckks_8192() -> CkksParams {
    CkksParams {
        name: "dev-ckks-8192-v1",
        rlwe: rlwe_8192(),
        default_scale_bits: ScaleBits(40),
        multiplicative_depth: MultiplicativeDepth(3),
    }
}

pub fn dev_hss_4096() -> HssParams {
    HssParams {
        name: "dev-hss-4096-v1",
        rlwe: rlwe_4096(),
        plaintext_modulus: PlaintextModulus(65_537),
        share_modulus_bits: ShareModulusBits(72),
        max_linear_terms: MaxLinearTerms(64),
        max_rmult_depth: MaxRmultDepth(1),
        fixed_point_scale_bits: Some(ScaleBits(16)),
        reconstruction_bound_bits: Some(ReconstructionBoundBits(72)),
        correctness_margin_bits: Some(CorrectnessMarginBits(8)),
    }
}

/// TFHE Classic-PBS shape for development: 2-bit message and 2-bit carry in
/// one short block, 128-bit classical security target, GLWE dimension 1 with
/// `N = 2048`. Field-level match to Zama TFHE-rs
/// **`PARAM_MESSAGE_2_CARRY_2_KS_PBS`** (alias of
/// `V1_4_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128`): LWE 918, GLWE 1,
/// polynomial 2048, PBS (23,1), KS (4,4), TUniform LWE bound 2^45, GLWE 2^17,
/// `log2_p_fail ≈ -129.581`, `EncryptionKeyChoice::Big`.
///
/// Zama-only PBS options (e.g. certain modulus-switch noise-reduction modes) are
/// not modeled in SILENT’s [`TfheParams`](crate::tfhe::TfheParams); this set is
/// still the correct geometry for **apples-to-apples** timing vs TFHE-rs prelude.
pub fn dev_tfhe_n2048() -> TfheParams {
    TfheParams {
        name: "dev-tfhe-n2048-v1",
        lwe_dimension: LweDimension(918),
        glwe_dimension: GlweDimension(1),
        polynomial_size: PolynomialSize(2048),
        ciphertext_modulus_log: CiphertextModulusLog(64),
        message_modulus: MessageModulus(4),
        carry_modulus: CarryModulus(4),
        pbs_base_log: DecompositionBaseLog(23),
        pbs_level: DecompositionLevelCount(1),
        ks_base_log: DecompositionBaseLog(4),
        ks_level: DecompositionLevelCount(4),
        lwe_noise: NoiseDistribution::TUniform { bound_log2: 45 },
        glwe_noise: NoiseDistribution::TUniform { bound_log2: 17 },
        log2_p_fail: Log2PFail(-129.581),
        encryption_key_choice: EncryptionKeyChoice::Big,
        security_level: SecurityLevel::Classical128,
    }
}

/// Same as [`dev_tfhe_n2048`]: explicit name for benchmarks/docs comparing to Zama’s prelude.
#[inline]
pub fn tfhe_rs_prelude_message_2_carry_2_ks_pbs() -> TfheParams {
    dev_tfhe_n2048()
}
