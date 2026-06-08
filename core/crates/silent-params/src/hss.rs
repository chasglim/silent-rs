use crate::error::ParamError;
use crate::ids::{
    CanonicalParamEncoding, ParameterSet, ParamsId, params_id_from_canonical, push_opt_u8,
    push_opt_u16, push_str, push_u16, push_u32, push_u64,
};
use crate::newtypes::{
    CorrectnessMarginBits, MaxLinearTerms, MaxRmultDepth, PlaintextModulus,
    ReconstructionBoundBits, ScaleBits, ShareModulusBits, bits_required_u64, ceil_log2_u32,
};
use crate::rlwe::RlweParams;
use crate::scheme::SchemeFamily;
use crate::security::SecurityLevel;
use silent_io::HasParamsId as IoHasParamsId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HssParams {
    pub name: &'static str,
    pub rlwe: RlweParams,
    pub plaintext_modulus: PlaintextModulus,
    pub share_modulus_bits: ShareModulusBits,
    pub max_linear_terms: MaxLinearTerms,
    pub max_rmult_depth: MaxRmultDepth,
    pub fixed_point_scale_bits: Option<ScaleBits>,
    pub reconstruction_bound_bits: Option<ReconstructionBoundBits>,
    pub correctness_margin_bits: Option<CorrectnessMarginBits>,
}

impl HssParams {
    pub fn to_rns_tool_config(&self) -> Result<silent_math::rns_tool::RnsToolConfig, ParamError> {
        self.rlwe.to_rns_tool_config(self.plaintext_modulus)
    }

    pub fn estimated_required_budget_bits(&self) -> u16 {
        let base = self
            .reconstruction_bound_bits
            .unwrap_or(ReconstructionBoundBits(self.share_modulus_bits.0))
            .0;
        let linear = ceil_log2_u32(self.max_linear_terms.0);
        let mult_scale = match self.fixed_point_scale_bits {
            Some(scale) => u16::from(scale.0).saturating_mul(self.max_rmult_depth.0),
            None => 0,
        };
        let correctness = self
            .correctness_margin_bits
            .unwrap_or(CorrectnessMarginBits(0))
            .0;
        base.saturating_add(linear)
            .saturating_add(mult_scale)
            .saturating_add(correctness)
    }
}

impl ParameterSet for HssParams {
    fn name(&self) -> &'static str {
        self.name
    }

    fn scheme_family(&self) -> SchemeFamily {
        SchemeFamily::Hss
    }

    fn security_level(&self) -> SecurityLevel {
        self.rlwe.security_level
    }

    fn params_id(&self) -> ParamsId {
        params_id_from_canonical(b"hss", self)
    }

    fn validate(&self) -> Result<(), ParamError> {
        self.rlwe.validate()?;
        if self.plaintext_modulus.0 < 2 {
            return Err(ParamError::PlaintextModulusTooSmall);
        }
        if self.max_linear_terms.0 == 0 {
            return Err(ParamError::ZeroMaxLinearTerms);
        }

        let plain_bits = bits_required_u64(self.plaintext_modulus.0);
        if self.share_modulus_bits.0 < plain_bits {
            return Err(ParamError::ShareModulusTooSmall {
                share_bits: self.share_modulus_bits.0,
                plain_bits,
            });
        }

        if let Some(reconstruction) = self.reconstruction_bound_bits {
            if reconstruction.0 > self.share_modulus_bits.0 {
                return Err(ParamError::ReconstructionBoundExceedsShareModulus {
                    reconstruction_bits: reconstruction.0,
                    share_bits: self.share_modulus_bits.0,
                });
            }
            if let Some(scale) = self.fixed_point_scale_bits {
                if u16::from(scale.0) > reconstruction.0 {
                    return Err(ParamError::ScaleExceedsReconstructionBound {
                        scale_bits: scale.0,
                        reconstruction_bits: reconstruction.0,
                    });
                }
            }
        }

        let required = self.estimated_required_budget_bits();
        let available = self.rlwe.ciphertext_modulus_budget_bits().0;
        if required > available {
            return Err(ParamError::HssBudgetExceeded {
                required,
                available,
            });
        }
        Ok(())
    }
}

impl IoHasParamsId for HssParams {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(ParameterSet::params_id(self).0)
    }
}

impl CanonicalParamEncoding for HssParams {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        push_str(out, self.name);
        self.rlwe.encode_canonical(out);
        push_u64(out, self.plaintext_modulus.0);
        push_u16(out, self.share_modulus_bits.0);
        push_u32(out, self.max_linear_terms.0);
        push_u16(out, self.max_rmult_depth.0);
        push_opt_u8(out, self.fixed_point_scale_bits.map(|bits| bits.0));
        push_opt_u16(out, self.reconstruction_bound_bits.map(|bits| bits.0));
        push_opt_u16(out, self.correctness_margin_bits.map(|bits| bits.0));
    }
}
