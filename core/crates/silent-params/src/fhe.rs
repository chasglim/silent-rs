use crate::error::ParamError;
use crate::ids::{
    CanonicalParamEncoding, ParameterSet, ParamsId, params_id_from_canonical, push_str, push_u8,
    push_u16, push_u64,
};
use crate::newtypes::{MultiplicativeDepth, PlaintextModulus, ScaleBits, bits_required_u64};
use crate::rlwe::RlweParams;
use crate::scheme::SchemeFamily;
use crate::security::SecurityLevel;
use silent_io::HasParamsId as IoHasParamsId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BfvParams {
    pub name: &'static str,
    pub rlwe: RlweParams,
    pub plaintext_modulus: PlaintextModulus,
    pub multiplicative_depth: MultiplicativeDepth,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvParams {
    pub name: &'static str,
    pub rlwe: RlweParams,
    pub plaintext_modulus: PlaintextModulus,
    pub multiplicative_depth: MultiplicativeDepth,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CkksParams {
    pub name: &'static str,
    pub rlwe: RlweParams,
    pub default_scale_bits: ScaleBits,
    pub multiplicative_depth: MultiplicativeDepth,
}

impl BfvParams {
    pub fn to_rns_tool_config(&self) -> Result<silent_math::rns_tool::RnsToolConfig, ParamError> {
        self.rlwe.to_rns_tool_config(self.plaintext_modulus)
    }
}

impl BgvParams {
    pub fn to_rns_tool_config(&self) -> Result<silent_math::rns_tool::RnsToolConfig, ParamError> {
        self.rlwe.to_rns_tool_config(self.plaintext_modulus)
    }
}

impl ParameterSet for BfvParams {
    fn name(&self) -> &'static str {
        self.name
    }

    fn scheme_family(&self) -> SchemeFamily {
        SchemeFamily::Bfv
    }

    fn security_level(&self) -> SecurityLevel {
        self.rlwe.security_level
    }

    fn params_id(&self) -> ParamsId {
        params_id_from_canonical(b"bfv", self)
    }

    fn validate(&self) -> Result<(), ParamError> {
        self.rlwe.validate()?;
        if self.plaintext_modulus.0 < 2 {
            return Err(ParamError::PlaintextModulusTooSmall);
        }
        let plain_bits = bits_required_u64(self.plaintext_modulus.0);
        let max_q_bits = self
            .rlwe
            .ciphertext_modulus_bits
            .iter()
            .map(|bits| bits.0)
            .max()
            .unwrap_or(0);
        if plain_bits >= max_q_bits {
            return Err(ParamError::ShareModulusTooSmall {
                share_bits: max_q_bits,
                plain_bits,
            });
        }
        Ok(())
    }
}

impl ParameterSet for BgvParams {
    fn name(&self) -> &'static str {
        self.name
    }

    fn scheme_family(&self) -> SchemeFamily {
        SchemeFamily::Bgv
    }

    fn security_level(&self) -> SecurityLevel {
        self.rlwe.security_level
    }

    fn params_id(&self) -> ParamsId {
        params_id_from_canonical(b"bgv", self)
    }

    fn validate(&self) -> Result<(), ParamError> {
        self.rlwe.validate()?;
        if self.plaintext_modulus.0 < 2 {
            return Err(ParamError::PlaintextModulusTooSmall);
        }
        let plain_bits = bits_required_u64(self.plaintext_modulus.0);
        let min_q_bits = self
            .rlwe
            .ciphertext_modulus_bits
            .iter()
            .map(|bits| bits.0)
            .min()
            .unwrap_or(0);
        if plain_bits >= min_q_bits {
            return Err(ParamError::ShareModulusTooSmall {
                share_bits: min_q_bits,
                plain_bits,
            });
        }
        Ok(())
    }
}

impl ParameterSet for CkksParams {
    fn name(&self) -> &'static str {
        self.name
    }

    fn scheme_family(&self) -> SchemeFamily {
        SchemeFamily::Ckks
    }

    fn security_level(&self) -> SecurityLevel {
        self.rlwe.security_level
    }

    fn params_id(&self) -> ParamsId {
        params_id_from_canonical(b"ckks", self)
    }

    fn validate(&self) -> Result<(), ParamError> {
        self.rlwe.validate()?;
        if self.default_scale_bits.0 == 0 {
            return Err(ParamError::ZeroScaleBits);
        }
        let levels = u16::from(self.default_scale_bits.0)
            .saturating_mul(self.multiplicative_depth.0.saturating_add(1));
        let budget = self.rlwe.ciphertext_modulus_budget_bits().0;
        if levels > budget {
            return Err(ParamError::SecurityBudgetExceeded {
                required: levels,
                max: budget,
                ring_dim: self.rlwe.ring.ring_dim.0,
                security: self.rlwe.security_level,
            });
        }
        Ok(())
    }
}

impl IoHasParamsId for BfvParams {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(ParameterSet::params_id(self).0)
    }
}

impl IoHasParamsId for BgvParams {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(ParameterSet::params_id(self).0)
    }
}

impl CanonicalParamEncoding for BfvParams {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        push_str(out, self.name);
        self.rlwe.encode_canonical(out);
        push_u64(out, self.plaintext_modulus.0);
        push_u16(out, self.multiplicative_depth.0);
    }
}

impl CanonicalParamEncoding for BgvParams {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        push_str(out, self.name);
        self.rlwe.encode_canonical(out);
        push_u64(out, self.plaintext_modulus.0);
        push_u16(out, self.multiplicative_depth.0);
    }
}

impl IoHasParamsId for CkksParams {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(ParameterSet::params_id(self).0)
    }
}

impl CanonicalParamEncoding for CkksParams {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        push_str(out, self.name);
        self.rlwe.encode_canonical(out);
        push_u8(out, self.default_scale_bits.0);
        push_u16(out, self.multiplicative_depth.0);
    }
}
