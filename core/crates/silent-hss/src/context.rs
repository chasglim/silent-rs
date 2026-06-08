use silent_params::{HssParams, ParamError, ParameterSet};
use silent_rlwe::EncryptionParams;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct HssContext {
    canonical: Arc<HssParams>,
    runtime: Arc<EncryptionParams>,
}

impl HssContext {
    pub fn new(params: HssParams) -> Result<Self, ParamError> {
        params.validate()?;
        let runtime = Arc::new(EncryptionParams::from_hss_parameter_set(&params)?);
        Ok(Self::from_validated_parts(params, runtime))
    }

    pub fn from_runtime_parts(
        params: HssParams,
        runtime: EncryptionParams,
    ) -> Result<Self, ParamError> {
        params.validate()?;
        runtime.validate_against_rlwe_parameter_set(&params.rlwe, params.plaintext_modulus)?;
        Ok(Self::from_validated_parts(params, Arc::new(runtime)))
    }

    fn from_validated_parts(params: HssParams, runtime: Arc<EncryptionParams>) -> Self {
        Self {
            canonical: Arc::new(params),
            runtime,
        }
    }

    pub fn params(&self) -> &HssParams {
        self.canonical.as_ref()
    }

    pub fn runtime_params(&self) -> &EncryptionParams {
        self.runtime.as_ref()
    }

    pub fn runtime_params_arc(&self) -> Arc<EncryptionParams> {
        Arc::clone(&self.runtime)
    }

    pub fn plain_modulus(&self) -> u64 {
        self.canonical.plaintext_modulus.0
    }

    pub fn degree(&self) -> usize {
        self.runtime.ring.degree()
    }
}
