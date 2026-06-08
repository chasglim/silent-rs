use crate::core::context::HeContext;
use silent_io::HasParamsId;
use silent_math::rns_tool::RnsTool;
use silent_params::{BfvParams, ParamError, ParameterSet};
use silent_ring::RingContext;
use silent_rlwe::EncryptionParams;

use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BfvMulMethod {
    Hps,
    HpsPoverq,
    HpsPoverqLeveled,
    Behz,
}

impl Default for BfvMulMethod {
    fn default() -> Self {
        BfvMulMethod::HpsPoverq
    }
}

#[derive(Clone, Debug)]
pub struct BfvParameters {
    canonical: Arc<BfvParams>,
    mul_method: BfvMulMethod,
    runtime: Arc<EncryptionParams>,
}

impl BfvParameters {
    pub fn new(params: BfvParams) -> Result<Self, ParamError> {
        params.validate()?;
        let rlwe_params = Arc::new(EncryptionParams::from_bfv_parameter_set(&params)?);
        Ok(Self::from_validated_parts(params, rlwe_params))
    }

    pub fn from_runtime_parts(
        params: BfvParams,
        rlwe_params: Arc<EncryptionParams>,
    ) -> Result<Self, ParamError> {
        params.validate()?;
        rlwe_params.validate_against_rlwe_parameter_set(&params.rlwe, params.plaintext_modulus)?;
        Ok(Self::from_validated_parts(params, rlwe_params))
    }

    pub fn from_runtime(
        name: &'static str,
        distribution: silent_params::DistributionType,
        security_level: silent_params::SecurityLevel,
        multiplicative_depth: silent_params::MultiplicativeDepth,
        plain_modulus: silent_params::PlaintextModulus,
        rlwe_params: Arc<EncryptionParams>,
    ) -> Result<Self, ParamError> {
        let degree = rlwe_params.ring.degree();
        let log_n = silent_params::LogN(degree.trailing_zeros() as u8);
        let ciphertext_modulus_bits = rlwe_params
            .ring
            .rns()
            .moduli()
            .iter()
            .map(|modulus| modulus_bits(modulus.value()))
            .collect();
        let special_modulus_bits = rlwe_params
            .rns_tool
            .base_p()
            .map(|base| {
                base.moduli()
                    .iter()
                    .map(|modulus| modulus_bits(modulus.value()))
                    .collect()
            })
            .unwrap_or_default();
        let key_switch_modulus_bits = rlwe_params
            .key_switch_modulus
            .map(|modulus| modulus_bits(modulus.value()));
        let params = BfvParams {
            name,
            rlwe: silent_params::RlweParams::new(
                name,
                silent_params::RingParams::new(
                    log_n,
                    silent_params::RingDim(degree),
                    distribution,
                ),
                ciphertext_modulus_bits,
                special_modulus_bits,
                key_switch_modulus_bits,
                security_level,
            ),
            plaintext_modulus: plain_modulus,
            multiplicative_depth,
        };
        Self::from_runtime_parts(params, rlwe_params)
    }

    fn from_validated_parts(params: BfvParams, rlwe_params: Arc<EncryptionParams>) -> Self {
        Self {
            mul_method: BfvMulMethod::default(),
            canonical: Arc::new(params),
            runtime: rlwe_params,
        }
    }

    pub fn with_mul_method(mut self, mul_method: BfvMulMethod) -> Self {
        self.mul_method = mul_method;
        self
    }

    pub fn canonical(&self) -> &BfvParams {
        self.canonical.as_ref()
    }

    pub fn runtime_params(&self) -> &EncryptionParams {
        self.runtime.as_ref()
    }

    pub fn runtime_params_arc(&self) -> Arc<EncryptionParams> {
        Arc::clone(&self.runtime)
    }

    pub fn ring(&self) -> &RingContext {
        &self.runtime.ring
    }

    pub fn rns_tool(&self) -> &RnsTool {
        &self.runtime.rns_tool
    }

    pub fn degree(&self) -> usize {
        self.canonical.rlwe.ring.ring_dim.0
    }

    pub fn plain_modulus(&self) -> u64 {
        self.canonical.plaintext_modulus.0
    }

    pub fn security_level_bits(&self) -> u32 {
        self.canonical.security_level().bits().unwrap_or(0).into()
    }

    pub fn mul_method(&self) -> BfvMulMethod {
        self.mul_method
    }
}

fn modulus_bits(value: u64) -> silent_params::ModulusBits {
    silent_params::ModulusBits((u64::BITS - value.leading_zeros()) as u16)
}

impl HeContext for BfvParameters {
    fn security_level(&self) -> u32 {
        self.security_level_bits()
    }

    fn poly_degree(&self) -> usize {
        self.degree()
    }
}

impl HasParamsId for BfvParameters {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(ParameterSet::params_id(self.canonical.as_ref()).0)
    }
}
