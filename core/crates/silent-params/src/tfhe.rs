//! Parameter set for TFHE (CGGI / Classic PBS path).
//!
//! Mirrors the design choices of `tfhe-rs` (`ClassicPBSParameters`) and
//! OpenFHE's `binfhe` Classical context.  TFHE works over a single native
//! power-of-two ciphertext modulus `q = 2^k`, so this struct keeps the chain
//! information out of the picture and instead exposes the dimensions and
//! decomposition parameters that drive every PBS evaluation.

use crate::error::ParamError;
use crate::ids::{
    CanonicalParamEncoding, ParameterSet, ParamsId, params_id_from_canonical, push_str, push_u8,
    push_u16, push_u32, push_u64, push_usize,
};
use crate::newtypes::{
    CarryModulus, CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount,
    EncryptionKeyChoice, GlweDimension, Log2PFail, LweDimension, MessageModulus, NoiseDistribution,
    PolynomialSize,
};
use crate::scheme::SchemeFamily;
use crate::security::SecurityLevel;
use silent_io::HasParamsId as IoHasParamsId;

/// Parameter set for the TFHE Classic PBS path.
///
/// Field semantics mirror `tfhe-rs::ClassicPBSParameters` closely so that
/// existing public preset values can be ported with minimal mental overhead.
#[derive(Clone, Debug, PartialEq)]
pub struct TfheParams {
    /// Friendly name used in registries and `params_id` salts.
    pub name: &'static str,
    /// LWE secret-key dimension (`n` in the literature).
    pub lwe_dimension: LweDimension,
    /// GLWE secret-key dimension (`k` in the literature).
    pub glwe_dimension: GlweDimension,
    /// Polynomial size `N` (must be a power of two).
    pub polynomial_size: PolynomialSize,
    /// `log2(q)` for the native ciphertext modulus.  Only 32 and 64 are
    /// supported by the MVP backend.
    pub ciphertext_modulus_log: CiphertextModulusLog,
    /// Plaintext message modulus (number of distinct messages encoded).
    pub message_modulus: MessageModulus,
    /// Carry buffer (`carry_modulus * message_modulus` is the effective
    /// plaintext modulus before noise).
    pub carry_modulus: CarryModulus,
    /// Decomposition base for the bootstrap key (`B_pbs = 2^pbs_base_log`).
    pub pbs_base_log: DecompositionBaseLog,
    /// Decomposition level count for the bootstrap key (`l_pbs`).
    pub pbs_level: DecompositionLevelCount,
    /// Decomposition base for the LWE key-switch key (`B_ks = 2^ks_base_log`).
    pub ks_base_log: DecompositionBaseLog,
    /// Decomposition level count for the LWE key-switch key (`l_ks`).
    pub ks_level: DecompositionLevelCount,
    /// Noise distribution for the small (LWE) secret key encryption.
    pub lwe_noise: NoiseDistribution,
    /// Noise distribution for the big (GLWE) secret key encryption.
    pub glwe_noise: NoiseDistribution,
    /// Estimated `log2(p_fail)`; informational, used for validation hints.
    pub log2_p_fail: Log2PFail,
    /// Whether ciphertexts are encrypted under the GLWE-derived "big" LWE key
    /// (`KeyswitchBootstrap`) or the LWE "small" key (`BootstrapKeyswitch`).
    pub encryption_key_choice: EncryptionKeyChoice,
    /// Target security level for the parameter set.
    pub security_level: SecurityLevel,
}

impl TfheParams {
    /// Returns the equivalent LWE dimension obtained by sample-extracting a
    /// GLWE encryption (i.e. `k * N`).  This is the dimension of the "big"
    /// LWE key produced after a programmable bootstrap.
    pub fn equivalent_big_lwe_dimension(&self) -> LweDimension {
        self.glwe_dimension
            .equivalent_lwe_dimension(self.polynomial_size)
    }

    /// Total plaintext modulus before noise (`message_modulus * carry_modulus`).
    pub fn total_message_modulus(&self) -> u128 {
        u128::from(self.message_modulus.0) * u128::from(self.carry_modulus.0)
    }

    /// Returns the native ciphertext modulus `q = 2^k` as a `u128`.  Always
    /// fits because `ciphertext_modulus_log <= 64`.
    pub fn ciphertext_modulus(&self) -> u128 {
        1u128 << self.ciphertext_modulus_log.0
    }
}

impl ParameterSet for TfheParams {
    fn name(&self) -> &'static str {
        self.name
    }

    fn scheme_family(&self) -> SchemeFamily {
        SchemeFamily::Tfhe
    }

    fn security_level(&self) -> SecurityLevel {
        self.security_level
    }

    fn params_id(&self) -> ParamsId {
        params_id_from_canonical(b"tfhe", self)
    }

    fn validate(&self) -> Result<(), ParamError> {
        let log_q = u32::from(self.ciphertext_modulus_log.0);
        if !(log_q == 32 || log_q == 64) {
            return Err(ParamError::TfheUnsupportedCiphertextModulusLog(
                self.ciphertext_modulus_log.0,
            ));
        }

        if self.lwe_dimension.0 == 0 {
            return Err(ParamError::TfheZeroDimension { which: "lwe" });
        }
        if self.glwe_dimension.0 == 0 {
            return Err(ParamError::TfheZeroDimension { which: "glwe" });
        }

        if self.polynomial_size.0 == 0 || !self.polynomial_size.is_power_of_two() {
            return Err(ParamError::TfhePolynomialSizeNotPowerOfTwo(
                self.polynomial_size.0,
            ));
        }

        if self.message_modulus.0 < 2 || !self.message_modulus.0.is_power_of_two() {
            return Err(ParamError::TfheModulusNotPowerOfTwo {
                which: "message",
                value: self.message_modulus.0,
            });
        }
        if self.carry_modulus.0 == 0 || !self.carry_modulus.0.is_power_of_two() {
            return Err(ParamError::TfheModulusNotPowerOfTwo {
                which: "carry",
                value: self.carry_modulus.0,
            });
        }

        let total = self.total_message_modulus();
        if total >= self.ciphertext_modulus() {
            return Err(ParamError::TfheMessageSpaceTooLarge {
                total,
                ciphertext_modulus_log: self.ciphertext_modulus_log.0,
            });
        }

        let pbs_budget = u32::from(self.pbs_base_log.0).saturating_mul(u32::from(self.pbs_level.0));
        if pbs_budget == 0 || pbs_budget > log_q {
            return Err(ParamError::TfheDecompositionBudgetExceeded {
                which: "pbs",
                base_log: self.pbs_base_log.0,
                level: self.pbs_level.0,
                ciphertext_modulus_log: self.ciphertext_modulus_log.0,
            });
        }
        let ks_budget = u32::from(self.ks_base_log.0).saturating_mul(u32::from(self.ks_level.0));
        if ks_budget == 0 || ks_budget > log_q {
            return Err(ParamError::TfheDecompositionBudgetExceeded {
                which: "ks",
                base_log: self.ks_base_log.0,
                level: self.ks_level.0,
                ciphertext_modulus_log: self.ciphertext_modulus_log.0,
            });
        }

        // After a PBS, the resulting "big" LWE has dimension k*N. The
        // subsequent key-switch sends it back to the LWE small key, which
        // therefore must have dimension <= k*N.  Conversely, blind rotation
        // requires lwe_dimension > 0 already enforced above.
        let big = self.equivalent_big_lwe_dimension();
        if big.0 < self.lwe_dimension.0 {
            return Err(ParamError::TfheKeyChainInconsistent {
                lwe_dimension: self.lwe_dimension.0,
                big_lwe_dimension: big.0,
            });
        }

        Ok(())
    }
}

impl IoHasParamsId for TfheParams {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(ParameterSet::params_id(self).0)
    }
}

impl CanonicalParamEncoding for TfheParams {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        push_str(out, self.name);
        push_usize(out, self.lwe_dimension.0);
        push_usize(out, self.glwe_dimension.0);
        push_usize(out, self.polynomial_size.0);
        push_u8(out, self.ciphertext_modulus_log.0);
        push_u64(out, self.message_modulus.0);
        push_u64(out, self.carry_modulus.0);
        push_u8(out, self.pbs_base_log.0);
        push_u8(out, self.pbs_level.0);
        push_u8(out, self.ks_base_log.0);
        push_u8(out, self.ks_level.0);
        encode_noise(out, self.lwe_noise);
        encode_noise(out, self.glwe_noise);
        // Encode log2_p_fail as raw bits to keep the canonical encoding stable
        // across platforms regardless of f64 stringification.
        push_u64(out, self.log2_p_fail.0.to_bits());
        push_u8(out, self.encryption_key_choice as u8);
        push_u8(out, self.security_level as u8);
        // Canonical encoding versioning marker so a future tweak can break
        // ParamsIds intentionally without colliding with old ones.
        push_u32(out, 1);
        // Reserved extension tag to keep future ParamsId changes explicit.
        push_u16(out, 0);
    }
}

fn encode_noise(out: &mut Vec<u8>, noise: NoiseDistribution) {
    match noise {
        NoiseDistribution::Gaussian { stddev } => {
            push_u8(out, 0);
            push_u64(out, stddev.to_bits());
        }
        NoiseDistribution::TUniform { bound_log2 } => {
            push_u8(out, 1);
            push_u8(out, bound_log2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;

    fn base() -> TfheParams {
        presets::toy::toy_tfhe_n512()
    }

    #[test]
    fn toy_tfhe_validates() {
        assert!(base().validate().is_ok());
    }

    #[test]
    fn dev_tfhe_validates() {
        assert!(presets::dev::dev_tfhe_n2048().validate().is_ok());
    }

    #[test]
    fn rejects_bad_ciphertext_modulus_log() {
        let mut params = base();
        params.ciphertext_modulus_log = CiphertextModulusLog(40);
        assert!(matches!(
            params.validate(),
            Err(ParamError::TfheUnsupportedCiphertextModulusLog(40))
        ));
    }

    #[test]
    fn rejects_zero_lwe_dimension() {
        let mut params = base();
        params.lwe_dimension = LweDimension(0);
        assert!(matches!(
            params.validate(),
            Err(ParamError::TfheZeroDimension { which: "lwe" })
        ));
    }

    #[test]
    fn rejects_non_power_of_two_polynomial_size() {
        let mut params = base();
        params.polynomial_size = PolynomialSize(513);
        assert!(matches!(
            params.validate(),
            Err(ParamError::TfhePolynomialSizeNotPowerOfTwo(513))
        ));
    }

    #[test]
    fn rejects_non_power_of_two_message_modulus() {
        let mut params = base();
        params.message_modulus = MessageModulus(3);
        assert!(matches!(
            params.validate(),
            Err(ParamError::TfheModulusNotPowerOfTwo {
                which: "message",
                value: 3,
            })
        ));
    }

    #[test]
    fn rejects_oversized_message_space() {
        let mut params = base();
        params.message_modulus = MessageModulus(1u64 << 32);
        params.carry_modulus = CarryModulus(1u64 << 32);
        assert!(matches!(
            params.validate(),
            Err(ParamError::TfheMessageSpaceTooLarge { .. })
        ));
    }

    #[test]
    fn rejects_excessive_decomposition_budget() {
        let mut params = base();
        params.pbs_base_log = DecompositionBaseLog(40);
        params.pbs_level = DecompositionLevelCount(4);
        assert!(matches!(
            params.validate(),
            Err(ParamError::TfheDecompositionBudgetExceeded { which: "pbs", .. })
        ));
    }

    #[test]
    fn rejects_inconsistent_key_chain() {
        let mut params = base();
        params.lwe_dimension = LweDimension(1024 * 1024);
        assert!(matches!(
            params.validate(),
            Err(ParamError::TfheKeyChainInconsistent { .. })
        ));
    }

    #[test]
    fn params_id_is_deterministic_and_distinct() {
        let id1 = ParameterSet::params_id(&base());
        let id2 = ParameterSet::params_id(&base());
        assert_eq!(id1, id2);

        let dev = ParameterSet::params_id(&presets::dev::dev_tfhe_n2048());
        assert_ne!(id1, dev);
    }
}
