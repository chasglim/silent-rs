//! Runtime wrapper around [`silent_params::TfheParams`].
//!
//! Mirrors the role `BfvParameters` plays for BFV: pre-validates the canonical
//! parameter set, exposes ergonomic accessors used by the scheme code, and
//! plugs into [`crate::core::context::HeContext`].

use silent_io::HasParamsId;
use silent_params::{
    CarryModulus, CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount,
    EncryptionKeyChoice, GlweDimension, LweDimension, MessageModulus, NoiseDistribution,
    ParamError, ParameterSet, PolynomialSize, TfheParams,
};
use std::sync::Arc;

use crate::core::context::HeContext;

/// Runtime, pre-validated TFHE parameters used by the scheme code.
#[derive(Clone, Debug)]
pub struct TfheParameters {
    canonical: Arc<TfheParams>,
}

impl TfheParameters {
    /// Construct from a canonical parameter set, validating it eagerly.
    pub fn new(params: TfheParams) -> Result<Self, ParamError> {
        params.validate()?;
        Ok(Self {
            canonical: Arc::new(params),
        })
    }

    pub fn canonical(&self) -> &TfheParams {
        self.canonical.as_ref()
    }

    pub fn lwe_dimension(&self) -> LweDimension {
        self.canonical.lwe_dimension
    }

    pub fn glwe_dimension(&self) -> GlweDimension {
        self.canonical.glwe_dimension
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.canonical.polynomial_size
    }

    pub fn ciphertext_modulus_log(&self) -> CiphertextModulusLog {
        self.canonical.ciphertext_modulus_log
    }

    pub fn message_modulus(&self) -> MessageModulus {
        self.canonical.message_modulus
    }

    pub fn carry_modulus(&self) -> CarryModulus {
        self.canonical.carry_modulus
    }

    pub fn pbs_base_log(&self) -> DecompositionBaseLog {
        self.canonical.pbs_base_log
    }

    pub fn pbs_level(&self) -> DecompositionLevelCount {
        self.canonical.pbs_level
    }

    pub fn ks_base_log(&self) -> DecompositionBaseLog {
        self.canonical.ks_base_log
    }

    pub fn ks_level(&self) -> DecompositionLevelCount {
        self.canonical.ks_level
    }

    pub fn lwe_noise(&self) -> NoiseDistribution {
        self.canonical.lwe_noise
    }

    pub fn glwe_noise(&self) -> NoiseDistribution {
        self.canonical.glwe_noise
    }

    pub fn encryption_key_choice(&self) -> EncryptionKeyChoice {
        self.canonical.encryption_key_choice
    }

    /// `message_modulus * carry_modulus`, the effective plaintext modulus
    /// before noise.
    pub fn total_message_modulus(&self) -> u64 {
        // `validate()` guarantees the product fits in 64 bits because the
        // ciphertext modulus 2^k <= 2^64 dominates it.
        (self.canonical.message_modulus.0).saturating_mul(self.canonical.carry_modulus.0)
    }

    /// Equivalent "big" LWE dimension (`k * N`), used after sample extraction.
    pub fn big_lwe_dimension(&self) -> LweDimension {
        self.canonical.equivalent_big_lwe_dimension()
    }
}

impl HeContext for TfheParameters {
    fn security_level(&self) -> u32 {
        self.canonical.security_level.bits().unwrap_or(0).into()
    }

    fn poly_degree(&self) -> usize {
        self.canonical.polynomial_size.0
    }
}

impl HasParamsId for TfheParameters {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(ParameterSet::params_id(self.canonical.as_ref()).0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use silent_params::presets;

    #[test]
    fn wraps_toy_preset() {
        let p = TfheParameters::new(presets::toy::toy_tfhe_n512()).unwrap();
        assert_eq!(p.lwe_dimension().0, 512);
        assert_eq!(p.glwe_dimension().0, 1);
        assert_eq!(p.polynomial_size().0, 512);
        assert_eq!(p.total_message_modulus(), 16);
        assert_eq!(p.big_lwe_dimension().0, 512);
    }

    #[test]
    fn rejects_invalid_params() {
        let mut params = presets::toy::toy_tfhe_n512();
        params.polynomial_size = PolynomialSize(513);
        assert!(TfheParameters::new(params).is_err());
    }
}
