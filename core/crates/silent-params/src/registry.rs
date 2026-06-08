use crate::error::ParamError;
use crate::fhe::{BfvParams, BgvParams, CkksParams};
use crate::hss::HssParams;
use crate::ids::{ParameterSet, ParamsId};
use crate::pqc::{PqcKemParams, PqcSigParams};
use crate::presets;
use crate::rlwe::RlweParams;
use crate::scheme::SchemeFamily;
use crate::security::SecurityLevel;
use crate::tfhe::TfheParams;

#[derive(Clone, Debug)]
pub enum RegisteredParameterSet {
    Rlwe(RlweParams),
    Bfv(BfvParams),
    Bgv(BgvParams),
    Ckks(CkksParams),
    Hss(HssParams),
    Tfhe(TfheParams),
    PqcKem(PqcKemParams),
    PqcSig(PqcSigParams),
}

impl RegisteredParameterSet {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rlwe(params) => params.name(),
            Self::Bfv(params) => params.name(),
            Self::Bgv(params) => params.name(),
            Self::Ckks(params) => params.name(),
            Self::Hss(params) => params.name(),
            Self::Tfhe(params) => params.name(),
            Self::PqcKem(params) => params.name(),
            Self::PqcSig(params) => params.name(),
        }
    }

    pub fn scheme_family(&self) -> SchemeFamily {
        match self {
            Self::Rlwe(params) => params.scheme_family(),
            Self::Bfv(params) => params.scheme_family(),
            Self::Bgv(params) => params.scheme_family(),
            Self::Ckks(params) => params.scheme_family(),
            Self::Hss(params) => params.scheme_family(),
            Self::Tfhe(params) => params.scheme_family(),
            Self::PqcKem(params) => params.scheme_family(),
            Self::PqcSig(params) => params.scheme_family(),
        }
    }

    pub fn security_level(&self) -> SecurityLevel {
        match self {
            Self::Rlwe(params) => params.security_level(),
            Self::Bfv(params) => params.security_level(),
            Self::Bgv(params) => params.security_level(),
            Self::Ckks(params) => params.security_level(),
            Self::Hss(params) => params.security_level(),
            Self::Tfhe(params) => params.security_level(),
            Self::PqcKem(params) => params.security_level(),
            Self::PqcSig(params) => params.security_level(),
        }
    }

    pub fn params_id(&self) -> ParamsId {
        match self {
            Self::Rlwe(params) => ParameterSet::params_id(params),
            Self::Bfv(params) => ParameterSet::params_id(params),
            Self::Bgv(params) => ParameterSet::params_id(params),
            Self::Ckks(params) => ParameterSet::params_id(params),
            Self::Hss(params) => ParameterSet::params_id(params),
            Self::Tfhe(params) => ParameterSet::params_id(params),
            Self::PqcKem(params) => params.params_id(),
            Self::PqcSig(params) => params.params_id(),
        }
    }

    pub fn validate(&self) -> Result<(), ParamError> {
        match self {
            Self::Rlwe(params) => params.validate(),
            Self::Bfv(params) => params.validate(),
            Self::Bgv(params) => params.validate(),
            Self::Ckks(params) => params.validate(),
            Self::Hss(params) => params.validate(),
            Self::Tfhe(params) => params.validate(),
            Self::PqcKem(params) => params.validate(),
            Self::PqcSig(params) => params.validate(),
        }
    }
}

pub fn all_presets() -> Vec<RegisteredParameterSet> {
    vec![
        RegisteredParameterSet::Rlwe(presets::toy::toy_rlwe_1024()),
        RegisteredParameterSet::Bfv(presets::toy::toy_bfv_1024()),
        RegisteredParameterSet::Bgv(presets::toy::toy_bgv_1024()),
        RegisteredParameterSet::Hss(presets::toy::toy_hss_1024()),
        RegisteredParameterSet::Tfhe(presets::toy::toy_tfhe_n512()),
        RegisteredParameterSet::Bfv(presets::dev::dev_bfv_4096()),
        RegisteredParameterSet::Bgv(presets::dev::dev_bgv_4096()),
        RegisteredParameterSet::Ckks(presets::dev::dev_ckks_8192()),
        RegisteredParameterSet::Hss(presets::dev::dev_hss_4096()),
        RegisteredParameterSet::Tfhe(presets::dev::dev_tfhe_n2048()),
        RegisteredParameterSet::Bfv(presets::current::current_bfv()),
        RegisteredParameterSet::Bgv(presets::current::current_bgv()),
        RegisteredParameterSet::Ckks(presets::current::current_ckks()),
        RegisteredParameterSet::Hss(presets::current::current_hss()),
        RegisteredParameterSet::PqcKem(presets::current::mlkem512()),
        RegisteredParameterSet::PqcKem(presets::current::mlkem768()),
        RegisteredParameterSet::PqcKem(presets::current::mlkem1024()),
        RegisteredParameterSet::PqcSig(presets::current::mldsa44()),
        RegisteredParameterSet::PqcSig(presets::current::mldsa65()),
        RegisteredParameterSet::PqcSig(presets::current::mldsa87()),
    ]
}

pub fn find_preset(name: &str) -> Option<RegisteredParameterSet> {
    all_presets().into_iter().find(|entry| entry.name() == name)
}
