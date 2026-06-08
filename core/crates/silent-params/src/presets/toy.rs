use crate::distribution::DistributionType;
use crate::fhe::{BfvParams, BgvParams};
use crate::hss::HssParams;
use crate::newtypes::{
    CarryModulus, CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount,
    EncryptionKeyChoice, GlweDimension, Log2PFail, LogN, LweDimension, MaxLinearTerms,
    MaxRmultDepth, MessageModulus, ModulusBits, MultiplicativeDepth, NoiseDistribution,
    PlaintextModulus, PolynomialSize, RingDim, ShareModulusBits,
};
use crate::ring::RingParams;
use crate::rlwe::RlweParams;
use crate::security::SecurityLevel;
use crate::tfhe::TfheParams;

pub fn toy_rlwe_1024() -> RlweParams {
    RlweParams::new(
        "toy-rlwe-1024-v1",
        RingParams::new(LogN(10), RingDim(1024), DistributionType::Ternary),
        vec![ModulusBits(27)],
        Vec::new(),
        None,
        SecurityLevel::Toy,
    )
}

pub fn toy_bfv_1024() -> BfvParams {
    BfvParams {
        name: "toy-bfv-1024-v1",
        rlwe: toy_rlwe_1024(),
        plaintext_modulus: PlaintextModulus(257),
        multiplicative_depth: MultiplicativeDepth(1),
    }
}

pub fn toy_bgv_1024() -> BgvParams {
    BgvParams {
        name: "toy-bgv-1024-v1",
        rlwe: toy_rlwe_1024(),
        plaintext_modulus: PlaintextModulus(257),
        multiplicative_depth: MultiplicativeDepth(1),
    }
}

pub fn toy_hss_1024() -> HssParams {
    HssParams {
        name: "toy-hss-1024-v1",
        rlwe: toy_rlwe_1024(),
        plaintext_modulus: PlaintextModulus(257),
        share_modulus_bits: ShareModulusBits(16),
        max_linear_terms: MaxLinearTerms(16),
        max_rmult_depth: MaxRmultDepth(1),
        fixed_point_scale_bits: None,
        reconstruction_bound_bits: None,
        correctness_margin_bits: None,
    }
}

/// Tiny TFHE parameter set, *not for production*. The dimensions and base/level
/// values are intentionally small so that round-trip and bootstrap correctness
/// tests stay within seconds even on a debug build. Security level is `Toy`
/// for this exact reason.
pub fn toy_tfhe_n512() -> TfheParams {
    TfheParams {
        name: "toy-tfhe-n512-v1",
        lwe_dimension: LweDimension(512),
        glwe_dimension: GlweDimension(1),
        polynomial_size: PolynomialSize(512),
        ciphertext_modulus_log: CiphertextModulusLog(64),
        message_modulus: MessageModulus(4),
        carry_modulus: CarryModulus(4),
        pbs_base_log: DecompositionBaseLog(8),
        pbs_level: DecompositionLevelCount(2),
        ks_base_log: DecompositionBaseLog(2),
        ks_level: DecompositionLevelCount(5),
        lwe_noise: NoiseDistribution::Gaussian {
            stddev: 7.069_849_454_709_433e-6,
        },
        glwe_noise: NoiseDistribution::Gaussian {
            stddev: 2.220_446_049_250_313e-16,
        },
        log2_p_fail: Log2PFail(-40.0),
        encryption_key_choice: EncryptionKeyChoice::Big,
        security_level: SecurityLevel::Toy,
    }
}
