use crate::distribution::DistributionType;
use crate::error::ParamError;
use crate::ids::{
    CanonicalParamEncoding, ParameterSet, ParamsId, params_id_from_canonical, push_opt_u16,
    push_slice_u16, push_str, push_u8, push_usize,
};
use crate::newtypes::{ModulusBits, PlaintextModulus};
use crate::ring::RingParams;
use crate::scheme::SchemeFamily;
use crate::security::SecurityLevel;
use crate::standard::find_max_log_q;
use silent_io::HasParamsId as IoHasParamsId;
use silent_math::modulus::{MAX_MODULUS_BITS, Modulus};
use silent_math::numth;
use silent_math::rns::RnsBase;
use silent_math::rns_tool::RnsToolConfig;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RlweParams {
    pub name: &'static str,
    pub ring: RingParams,
    pub ciphertext_modulus_bits: Vec<ModulusBits>,
    pub special_modulus_bits: Vec<ModulusBits>,
    pub key_switch_modulus_bits: Option<ModulusBits>,
    pub security_level: SecurityLevel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedModuli {
    pub ciphertext_moduli: Vec<u64>,
    pub special_moduli: Vec<u64>,
    pub key_switch_modulus: Option<u64>,
}

impl RlweParams {
    pub fn new(
        name: &'static str,
        ring: RingParams,
        ciphertext_modulus_bits: Vec<ModulusBits>,
        special_modulus_bits: Vec<ModulusBits>,
        key_switch_modulus_bits: Option<ModulusBits>,
        security_level: SecurityLevel,
    ) -> Self {
        Self {
            name,
            ring,
            ciphertext_modulus_bits,
            special_modulus_bits,
            key_switch_modulus_bits,
            security_level,
        }
    }

    pub fn ciphertext_modulus_budget_bits(&self) -> ModulusBits {
        ModulusBits(
            self.ciphertext_modulus_bits
                .iter()
                .map(|bits| bits.0)
                .sum::<u16>(),
        )
    }

    pub fn special_modulus_budget_bits(&self) -> ModulusBits {
        ModulusBits(
            self.special_modulus_bits
                .iter()
                .map(|bits| bits.0)
                .sum::<u16>(),
        )
    }

    pub fn gen_moduli(&self) -> Result<GeneratedModuli, ParamError> {
        self.validate()?;

        let step = (self.ring.ring_dim.0 as u64).checked_mul(2).ok_or(
            ParamError::PrimeGenerationFailed {
                bits: 0,
                ring_dim: self.ring.ring_dim.0,
            },
        )?;
        let mut used = BTreeSet::new();
        let mut cursors = BTreeMap::new();

        let ciphertext_moduli = self.generate_prime_chain(
            &self.ciphertext_modulus_bits,
            step,
            &mut used,
            &mut cursors,
        )?;
        let special_moduli =
            self.generate_prime_chain(&self.special_modulus_bits, step, &mut used, &mut cursors)?;
        let key_switch_modulus = match self.key_switch_modulus_bits {
            Some(bits) => Some(self.generate_prime(bits, step, &mut used, &mut cursors)?),
            None => None,
        };

        Ok(GeneratedModuli {
            ciphertext_moduli,
            special_moduli,
            key_switch_modulus,
        })
    }

    pub fn to_rns_tool_config(
        &self,
        plaintext_modulus: PlaintextModulus,
    ) -> Result<RnsToolConfig, ParamError> {
        let generated = self.gen_moduli()?;
        let base_q = RnsBase::from_values(generated.ciphertext_moduli)?;
        let base_t = Modulus::new(plaintext_modulus.0)?;
        let mut config = RnsToolConfig::new(base_q, base_t);
        if !generated.special_moduli.is_empty() {
            config = config.with_base_p(RnsBase::from_values(generated.special_moduli)?);
        }
        if let Some(value) = generated.key_switch_modulus {
            config = config.with_key_switch_modulus(Modulus::new(value)?);
        }
        Ok(config)
    }

    fn validate_modulus_bits(&self, bits: ModulusBits) -> Result<(), ParamError> {
        if bits.0 < 2 || u32::from(bits.0) > MAX_MODULUS_BITS {
            return Err(ParamError::InvalidModulusBits(bits.0));
        }
        Ok(())
    }

    fn generate_prime_chain(
        &self,
        requested_bits: &[ModulusBits],
        step: u64,
        used: &mut BTreeSet<u64>,
        cursors: &mut BTreeMap<u16, u64>,
    ) -> Result<Vec<u64>, ParamError> {
        let mut values = Vec::with_capacity(requested_bits.len());
        for bits in requested_bits {
            values.push(self.generate_prime(*bits, step, used, cursors)?);
        }
        Ok(values)
    }

    fn generate_prime(
        &self,
        bits: ModulusBits,
        step: u64,
        used: &mut BTreeSet<u64>,
        cursors: &mut BTreeMap<u16, u64>,
    ) -> Result<u64, ParamError> {
        let cursor = cursors.entry(bits.0).or_insert(0);
        let mut candidate = if *cursor == 0 {
            numth::first_ntt_prime_with_bits(bits.0 as u32, step)
        } else {
            numth::next_ntt_prime(cursor.saturating_add(step), step)
        }
        .ok_or(ParamError::PrimeGenerationFailed {
            bits: bits.0,
            ring_dim: self.ring.ring_dim.0,
        })?;

        while used.contains(&candidate) {
            candidate = numth::next_ntt_prime(candidate.saturating_add(step), step).ok_or(
                ParamError::PrimeGenerationFailed {
                    bits: bits.0,
                    ring_dim: self.ring.ring_dim.0,
                },
            )?;
        }

        *cursor = candidate;
        used.insert(candidate);
        Ok(candidate)
    }
}

impl ParameterSet for RlweParams {
    fn name(&self) -> &'static str {
        self.name
    }

    fn scheme_family(&self) -> SchemeFamily {
        SchemeFamily::Rlwe
    }

    fn security_level(&self) -> SecurityLevel {
        self.security_level
    }

    fn params_id(&self) -> ParamsId {
        params_id_from_canonical(b"rlwe", self)
    }

    fn validate(&self) -> Result<(), ParamError> {
        self.ring.validate()?;
        if self.ciphertext_modulus_bits.is_empty() {
            return Err(ParamError::EmptyCoefficientModulusChain);
        }

        for bits in &self.ciphertext_modulus_bits {
            self.validate_modulus_bits(*bits)?;
        }
        for bits in &self.special_modulus_bits {
            self.validate_modulus_bits(*bits)?;
        }
        if let Some(bits) = self.key_switch_modulus_bits {
            self.validate_modulus_bits(bits)?;
        }

        let budget = self.ciphertext_modulus_budget_bits();
        match self.security_level {
            SecurityLevel::Toy | SecurityLevel::NotSet => Ok(()),
            _ => {
                let max = find_max_log_q(
                    self.ring.distribution,
                    self.security_level,
                    self.ring.ring_dim,
                )
                .ok_or(ParamError::UnsupportedStandardCombination {
                    distribution: self.ring.distribution,
                    security: self.security_level,
                })?;
                if budget.0 > max.0 {
                    return Err(ParamError::SecurityBudgetExceeded {
                        required: budget.0,
                        max: max.0,
                        ring_dim: self.ring.ring_dim.0,
                        security: self.security_level,
                    });
                }
                Ok(())
            }
        }
    }
}

impl IoHasParamsId for RlweParams {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(ParameterSet::params_id(self).0)
    }
}

impl CanonicalParamEncoding for RlweParams {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        push_str(out, self.name);
        push_u8(out, self.ring.log_n.0);
        push_usize(out, self.ring.ring_dim.0);
        push_u8(out, self.ring.distribution as u8);
        push_slice_u16(
            out,
            &self
                .ciphertext_modulus_bits
                .iter()
                .map(|bits| bits.0)
                .collect::<Vec<_>>(),
        );
        push_slice_u16(
            out,
            &self
                .special_modulus_bits
                .iter()
                .map(|bits| bits.0)
                .collect::<Vec<_>>(),
        );
        push_opt_u16(out, self.key_switch_modulus_bits.map(|bits| bits.0));
        push_u8(out, self.security_level as u8);
    }
}

impl Default for RlweParams {
    fn default() -> Self {
        Self::new(
            "unnamed-rlwe-v1",
            RingParams::new(
                crate::newtypes::LogN(10),
                crate::newtypes::RingDim(1024),
                DistributionType::Ternary,
            ),
            vec![ModulusBits(27)],
            Vec::new(),
            None,
            SecurityLevel::Toy,
        )
    }
}
