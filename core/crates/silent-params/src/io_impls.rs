use crate::distribution::DistributionType;
use crate::fhe::{BfvParams, BgvParams};
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

use silent_io::error::IoError;
use silent_io::object_type::ObjectType;
use silent_io::primitives::*;
use silent_io::traits::{CanonicalDecode, CanonicalEncode, ObjectKind};
use std::io::{Read, Write};

// Helper for encoding/decoding `&'static str` by leaking the decoded string.
// Note: In a production environment, you'd likely want to look up names
// from a registry to avoid memory leaks, but leaking is safe and sound for MVP.
fn write_string(w: &mut impl Write, s: &str) -> Result<usize, IoError> {
    write_len_prefixed_bytes(w, s.as_bytes())
}

fn read_string(r: &mut impl Read) -> Result<&'static str, IoError> {
    let bytes = read_len_prefixed_bytes(r)?;
    let s = String::from_utf8(bytes).map_err(|e| IoError::InvalidEncoding {
        detail: format!("Invalid UTF-8 in string: {}", e),
    })?;
    Ok(Box::leak(s.into_boxed_str()))
}

// ── RingParams ─────────────────────────────────────────────────────────────

impl ObjectKind for RingParams {
    const OBJECT_TYPE: ObjectType = ObjectType::RING_PARAMS;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for RingParams {
    fn encoded_len(&self) -> usize {
        1 + 8 + 1 // log_n(1) + ring_dim(8) + distribution(1)
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut written = 0;
        written += write_u8(w, self.log_n.0)?;
        written += write_u64(w, self.ring_dim.0 as u64)?;
        written += write_u8(w, self.distribution as u8)?;
        Ok(written)
    }
}

impl CanonicalDecode for RingParams {
    fn decode_from<R: Read>(r: &mut R) -> Result<Self, IoError> {
        let log_n = LogN(read_u8(r)?);
        let ring_dim = RingDim(read_u64(r)? as usize);
        let dist_val = read_u8(r)?;
        let distribution = match dist_val {
            0 => DistributionType::Ternary,
            1 => DistributionType::GaussianError,
            2 => DistributionType::Uniform,
            _ => {
                return Err(IoError::InvalidEncoding {
                    detail: format!("Unknown DistributionType: {}", dist_val),
                });
            }
        };
        Ok(RingParams::new(log_n, ring_dim, distribution))
    }
}

// ── RlweParams ─────────────────────────────────────────────────────────────

impl ObjectKind for RlweParams {
    const OBJECT_TYPE: ObjectType = ObjectType::RLWE_PARAMS;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for RlweParams {
    fn encoded_len(&self) -> usize {
        4 + self.name.len()
            + self.ring.encoded_len()
            + 4
            + self.ciphertext_modulus_bits.len() * 2
            + 4
            + self.special_modulus_bits.len() * 2
            + 1
            + self.key_switch_modulus_bits.map_or(0, |_| 2)
            + 1
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut written = 0;
        written += write_string(w, self.name)?;
        written += self.ring.encode_to(w)?;

        let ct_bits: Vec<u16> = self.ciphertext_modulus_bits.iter().map(|b| b.0).collect();
        written += write_slice_with(w, &ct_bits, |&b| {
            let mut buf = Vec::new();
            write_u16(&mut buf, b)?;
            Ok(buf)
        })?;

        let sp_bits: Vec<u16> = self.special_modulus_bits.iter().map(|b| b.0).collect();
        written += write_slice_with(w, &sp_bits, |&b| {
            let mut buf = Vec::new();
            write_u16(&mut buf, b)?;
            Ok(buf)
        })?;

        match self.key_switch_modulus_bits {
            Some(bits) => {
                written += write_bool(w, true)?;
                written += write_u16(w, bits.0)?;
            }
            None => {
                written += write_bool(w, false)?;
            }
        }

        written += write_u8(w, self.security_level as u8)?;
        Ok(written)
    }
}

impl CanonicalDecode for RlweParams {
    fn decode_from<R: Read>(r: &mut R) -> Result<Self, IoError> {
        let name = read_string(r)?;
        let ring = RingParams::decode_from(r)?;

        let ct_bits = read_elements_with(r, 2, |bytes| {
            let mut cur = std::io::Cursor::new(bytes);
            Ok(ModulusBits(read_u16(&mut cur)?))
        })?;

        let sp_bits = read_elements_with(r, 2, |bytes| {
            let mut cur = std::io::Cursor::new(bytes);
            Ok(ModulusBits(read_u16(&mut cur)?))
        })?;

        let has_ks = read_bool(r)?;
        let key_switch_modulus_bits = if has_ks {
            Some(ModulusBits(read_u16(r)?))
        } else {
            None
        };

        let sec_val = read_u8(r)?;
        let security_level = match sec_val {
            0 => SecurityLevel::Toy,
            1 => SecurityLevel::Classical128,
            2 => SecurityLevel::Classical192,
            3 => SecurityLevel::Classical256,
            4 => SecurityLevel::Quantum128,
            5 => SecurityLevel::Quantum192,
            6 => SecurityLevel::Quantum256,
            7 => SecurityLevel::NotSet,
            _ => {
                return Err(IoError::InvalidEncoding {
                    detail: format!("Unknown SecurityLevel: {}", sec_val),
                });
            }
        };

        Ok(RlweParams::new(
            name,
            ring,
            ct_bits,
            sp_bits,
            key_switch_modulus_bits,
            security_level,
        ))
    }
}

// ── BfvParams ──────────────────────────────────────────────────────────────

impl ObjectKind for BfvParams {
    const OBJECT_TYPE: ObjectType = ObjectType::BFV_PARAMS;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for BfvParams {
    fn encoded_len(&self) -> usize {
        4 + self.name.len()
            + self.rlwe.encoded_len()
            + 8 // plaintext_modulus
            + 2 // multiplicative_depth
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut written = 0;
        written += write_string(w, self.name)?;
        written += self.rlwe.encode_to(w)?;
        written += write_u64(w, self.plaintext_modulus.0)?;
        written += write_u16(w, self.multiplicative_depth.0)?;
        Ok(written)
    }
}

impl CanonicalDecode for BfvParams {
    fn decode_from<R: Read>(r: &mut R) -> Result<Self, IoError> {
        let name = read_string(r)?;
        let rlwe = RlweParams::decode_from(r)?;
        let plaintext_modulus = PlaintextModulus(read_u64(r)?);
        let multiplicative_depth = MultiplicativeDepth(read_u16(r)?);

        Ok(BfvParams {
            name,
            rlwe,
            plaintext_modulus,
            multiplicative_depth,
        })
    }
}

// ── BgvParams ──────────────────────────────────────────────────────────────

impl ObjectKind for BgvParams {
    const OBJECT_TYPE: ObjectType = ObjectType::BGV_PARAMS;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for BgvParams {
    fn encoded_len(&self) -> usize {
        4 + self.name.len()
            + self.rlwe.encoded_len()
            + 8 // plaintext_modulus
            + 2 // multiplicative_depth
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut written = 0;
        written += write_string(w, self.name)?;
        written += self.rlwe.encode_to(w)?;
        written += write_u64(w, self.plaintext_modulus.0)?;
        written += write_u16(w, self.multiplicative_depth.0)?;
        Ok(written)
    }
}

impl CanonicalDecode for BgvParams {
    fn decode_from<R: Read>(r: &mut R) -> Result<Self, IoError> {
        let name = read_string(r)?;
        let rlwe = RlweParams::decode_from(r)?;
        let plaintext_modulus = PlaintextModulus(read_u64(r)?);
        let multiplicative_depth = MultiplicativeDepth(read_u16(r)?);

        Ok(BgvParams {
            name,
            rlwe,
            plaintext_modulus,
            multiplicative_depth,
        })
    }
}

// ── HssParams ──────────────────────────────────────────────────────────────

impl ObjectKind for HssParams {
    const OBJECT_TYPE: ObjectType = ObjectType::HSS_PARAMS;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for HssParams {
    fn encoded_len(&self) -> usize {
        4 + self.name.len()
            + self.rlwe.encoded_len()
            + 8 // plaintext_modulus
            + 2 // share_modulus_bits
            + 4 // max_linear_terms
            + 2 // max_rmult_depth
            + 1 + self.fixed_point_scale_bits.map_or(0, |_| 1)
            + 1 + self.reconstruction_bound_bits.map_or(0, |_| 2)
            + 1 + self.correctness_margin_bits.map_or(0, |_| 2)
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut written = 0;
        written += write_string(w, self.name)?;
        written += self.rlwe.encode_to(w)?;
        written += write_u64(w, self.plaintext_modulus.0)?;
        written += write_u16(w, self.share_modulus_bits.0)?;
        written += write_u32(w, self.max_linear_terms.0)?;
        written += write_u16(w, self.max_rmult_depth.0)?;

        match self.fixed_point_scale_bits {
            Some(bits) => {
                written += write_bool(w, true)?;
                written += write_u8(w, bits.0)?;
            }
            None => {
                written += write_bool(w, false)?;
            }
        }

        match self.reconstruction_bound_bits {
            Some(bits) => {
                written += write_bool(w, true)?;
                written += write_u16(w, bits.0)?;
            }
            None => {
                written += write_bool(w, false)?;
            }
        }

        match self.correctness_margin_bits {
            Some(bits) => {
                written += write_bool(w, true)?;
                written += write_u16(w, bits.0)?;
            }
            None => {
                written += write_bool(w, false)?;
            }
        }

        Ok(written)
    }
}

impl CanonicalDecode for HssParams {
    fn decode_from<R: Read>(r: &mut R) -> Result<Self, IoError> {
        let name = read_string(r)?;
        let rlwe = RlweParams::decode_from(r)?;
        let plaintext_modulus = PlaintextModulus(read_u64(r)?);
        let share_modulus_bits = ShareModulusBits(read_u16(r)?);
        let max_linear_terms = MaxLinearTerms(read_u32(r)?);
        let max_rmult_depth = MaxRmultDepth(read_u16(r)?);

        let fixed_point_scale_bits = if read_bool(r)? {
            Some(ScaleBits(read_u8(r)?))
        } else {
            None
        };

        let reconstruction_bound_bits = if read_bool(r)? {
            Some(ReconstructionBoundBits(read_u16(r)?))
        } else {
            None
        };

        let correctness_margin_bits = if read_bool(r)? {
            Some(CorrectnessMarginBits(read_u16(r)?))
        } else {
            None
        };

        Ok(HssParams {
            name,
            rlwe,
            plaintext_modulus,
            share_modulus_bits,
            max_linear_terms,
            max_rmult_depth,
            fixed_point_scale_bits,
            reconstruction_bound_bits,
            correctness_margin_bits,
        })
    }
}

// ── TfheParams ─────────────────────────────────────────────────────────────

impl ObjectKind for TfheParams {
    const OBJECT_TYPE: ObjectType = ObjectType::TFHE_PARAMS;
    const SERIALIZED_VERSION: u16 = 1;
}

fn encode_noise(w: &mut impl Write, noise: NoiseDistribution) -> Result<usize, IoError> {
    let mut written = 0;
    match noise {
        NoiseDistribution::Gaussian { stddev } => {
            written += write_u8(w, 0)?;
            written += write_u64(w, stddev.to_bits())?;
        }
        NoiseDistribution::TUniform { bound_log2 } => {
            written += write_u8(w, 1)?;
            written += write_u8(w, bound_log2)?;
        }
    }
    Ok(written)
}

fn decode_noise(r: &mut impl Read) -> Result<NoiseDistribution, IoError> {
    let tag = read_u8(r)?;
    match tag {
        0 => {
            let stddev = f64::from_bits(read_u64(r)?);
            Ok(NoiseDistribution::Gaussian { stddev })
        }
        1 => {
            let bound_log2 = read_u8(r)?;
            Ok(NoiseDistribution::TUniform { bound_log2 })
        }
        _ => Err(IoError::InvalidEncoding {
            detail: format!("Unknown NoiseDistribution tag: {}", tag),
        }),
    }
}

fn noise_encoded_len(noise: NoiseDistribution) -> usize {
    match noise {
        NoiseDistribution::Gaussian { .. } => 1 + 8,
        NoiseDistribution::TUniform { .. } => 1 + 1,
    }
}

impl CanonicalEncode for TfheParams {
    fn encoded_len(&self) -> usize {
        4 + self.name.len()
            + 8 // lwe_dimension
            + 8 // glwe_dimension
            + 8 // polynomial_size
            + 1 // ciphertext_modulus_log
            + 8 // message_modulus
            + 8 // carry_modulus
            + 1 // pbs_base_log
            + 1 // pbs_level
            + 1 // ks_base_log
            + 1 // ks_level
            + noise_encoded_len(self.lwe_noise)
            + noise_encoded_len(self.glwe_noise)
            + 8 // log2_p_fail
            + 1 // encryption_key_choice
            + 1 // security_level
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut written = 0;
        written += write_string(w, self.name)?;
        written += write_u64(w, self.lwe_dimension.0 as u64)?;
        written += write_u64(w, self.glwe_dimension.0 as u64)?;
        written += write_u64(w, self.polynomial_size.0 as u64)?;
        written += write_u8(w, self.ciphertext_modulus_log.0)?;
        written += write_u64(w, self.message_modulus.0)?;
        written += write_u64(w, self.carry_modulus.0)?;
        written += write_u8(w, self.pbs_base_log.0)?;
        written += write_u8(w, self.pbs_level.0)?;
        written += write_u8(w, self.ks_base_log.0)?;
        written += write_u8(w, self.ks_level.0)?;
        written += encode_noise(w, self.lwe_noise)?;
        written += encode_noise(w, self.glwe_noise)?;
        written += write_u64(w, self.log2_p_fail.0.to_bits())?;
        written += write_u8(w, self.encryption_key_choice as u8)?;
        written += write_u8(w, self.security_level as u8)?;
        Ok(written)
    }
}

impl CanonicalDecode for TfheParams {
    fn decode_from<R: Read>(r: &mut R) -> Result<Self, IoError> {
        let name = read_string(r)?;
        let lwe_dimension = LweDimension(read_u64(r)? as usize);
        let glwe_dimension = GlweDimension(read_u64(r)? as usize);
        let polynomial_size = PolynomialSize(read_u64(r)? as usize);
        let ciphertext_modulus_log = CiphertextModulusLog(read_u8(r)?);
        let message_modulus = MessageModulus(read_u64(r)?);
        let carry_modulus = CarryModulus(read_u64(r)?);
        let pbs_base_log = DecompositionBaseLog(read_u8(r)?);
        let pbs_level = DecompositionLevelCount(read_u8(r)?);
        let ks_base_log = DecompositionBaseLog(read_u8(r)?);
        let ks_level = DecompositionLevelCount(read_u8(r)?);
        let lwe_noise = decode_noise(r)?;
        let glwe_noise = decode_noise(r)?;
        let log2_p_fail = Log2PFail(f64::from_bits(read_u64(r)?));

        let enc_choice_val = read_u8(r)?;
        let encryption_key_choice = match enc_choice_val {
            0 => EncryptionKeyChoice::Big,
            1 => EncryptionKeyChoice::Small,
            _ => {
                return Err(IoError::InvalidEncoding {
                    detail: format!("Unknown EncryptionKeyChoice: {}", enc_choice_val),
                });
            }
        };

        let sec_val = read_u8(r)?;
        let security_level = match sec_val {
            0 => SecurityLevel::Toy,
            1 => SecurityLevel::Classical128,
            2 => SecurityLevel::Classical192,
            3 => SecurityLevel::Classical256,
            4 => SecurityLevel::Quantum128,
            5 => SecurityLevel::Quantum192,
            6 => SecurityLevel::Quantum256,
            7 => SecurityLevel::NotSet,
            _ => {
                return Err(IoError::InvalidEncoding {
                    detail: format!("Unknown SecurityLevel: {}", sec_val),
                });
            }
        };

        Ok(TfheParams {
            name,
            lwe_dimension,
            glwe_dimension,
            polynomial_size,
            ciphertext_modulus_log,
            message_modulus,
            carry_modulus,
            pbs_base_log,
            pbs_level,
            ks_base_log,
            ks_level,
            lwe_noise,
            glwe_noise,
            log2_p_fail,
            encryption_key_choice,
            security_level,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;
    use std::io::Cursor;

    fn roundtrip_test<T: CanonicalEncode + CanonicalDecode + PartialEq + std::fmt::Debug>(
        item: &T,
    ) {
        let mut encoded = Vec::new();
        let len = item.encode_to(&mut encoded).unwrap();
        assert_eq!(len, encoded.len());
        assert_eq!(len, item.encoded_len());

        let mut r = Cursor::new(&encoded);
        let decoded = T::decode_from(&mut r).unwrap();
        assert_eq!(*item, decoded);
    }

    #[test]
    fn test_ring_params_roundtrip() {
        let params = RingParams::new(LogN(10), RingDim(1024), DistributionType::Ternary);
        roundtrip_test(&params);
    }

    #[test]
    fn test_rlwe_params_roundtrip() {
        let params = RlweParams::default();
        roundtrip_test(&params);
    }

    #[test]
    fn test_bfv_params_roundtrip() {
        let params = presets::toy::toy_bfv_1024();
        roundtrip_test(&params);
    }

    #[test]
    fn test_bgv_params_roundtrip() {
        let params = presets::toy::toy_bgv_1024();
        roundtrip_test(&params);
    }

    #[test]
    fn test_tfhe_params_roundtrip() {
        let params = presets::toy::toy_tfhe_n512();
        roundtrip_test(&params);
    }
}
