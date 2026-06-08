//! Canonical serialization for TFHE lattice types and RLWE types.

use crate::bootstrap::LweBootstrapKey;
use crate::ggsw::{GgswCiphertext, GgswCiphertextList};
use crate::glwe::{GlweCiphertext, GlweSecretKey};
use crate::lwe::{LweCiphertext, LweKeyswitchKey, LwePublicKey, LweSecretKey};
use crate::{Ciphertext, EncryptionParams, EvaluationKey, GaloisKey, PublicKey, SecretKey};

use silent_io::error::IoError;
use silent_io::object_type::ObjectType;
use silent_io::primitives::*;
use silent_io::traits::{CanonicalEncode, DecodeWithParams, ObjectKind};
use silent_params::{
    CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount, GlweDimension,
    LweDimension, PolynomialSize,
};
use silent_ring::{NativePoly, Poly};
use std::collections::HashMap;
use std::io::{Read, Write};

const MAX_RLWE_EVALUATION_KEY_ELEMENTS: usize = 4096;
const MAX_RLWE_GALOIS_KEY_ENTRIES: usize = 4096;

// ── LweCiphertext ──────────────────────────────────────────────────────────

impl ObjectKind for LweCiphertext {
    const OBJECT_TYPE: ObjectType = ObjectType::LWE_CIPHERTEXT;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for LweCiphertext {
    fn encoded_len(&self) -> usize {
        4 + 1 + self.as_slice().len() * 8
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += write_u32(w, self.dimension().0 as u32)?;
        n += write_u8(w, self.log_modulus())?;
        for &v in self.as_slice() {
            n += write_u64(w, v)?;
        }
        Ok(n)
    }
}

/// Params: `(LweDimension, CiphertextModulusLog)`
impl DecodeWithParams<(LweDimension, CiphertextModulusLog)> for LweCiphertext {
    fn decode_with_params<R: Read>(
        params: &(LweDimension, CiphertextModulusLog),
        r: &mut R,
    ) -> Result<Self, IoError> {
        let dim = read_u32(r)? as usize;
        let log_mod = read_u8(r)?;
        if dim != params.0.0 {
            return Err(IoError::InvalidEncoding {
                detail: format!("LWE dimension {} != expected {}", dim, params.0.0),
            });
        }
        if log_mod != params.1.0 {
            return Err(IoError::InvalidEncoding {
                detail: format!("log_modulus {} != expected {}", log_mod, params.1.0),
            });
        }
        let size = dim + 1;
        let data = read_u64_slice(r, size)?;
        if log_mod < 64 {
            let modulus = 1u64 << log_mod;
            for (i, &c) in data.iter().enumerate() {
                if c >= modulus {
                    return Err(IoError::CoefficientOutOfRange {
                        index: i,
                        value: c,
                        modulus,
                    });
                }
            }
        }
        Ok(LweCiphertext::from_data(data, params.1))
    }
}

// ── LweSecretKey ───────────────────────────────────────────────────────────

impl ObjectKind for LweSecretKey {
    const OBJECT_TYPE: ObjectType = ObjectType::LWE_SECRET_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for LweSecretKey {
    fn encoded_len(&self) -> usize {
        4 + 1 + self.dimension().0 * 8
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += write_u32(w, self.dimension().0 as u32)?;
        n += write_u8(w, self.log_modulus())?;
        for &v in self.data() {
            n += write_u64(w, v)?;
        }
        Ok(n)
    }
}

impl DecodeWithParams<(LweDimension, CiphertextModulusLog)> for LweSecretKey {
    fn decode_with_params<R: Read>(
        params: &(LweDimension, CiphertextModulusLog),
        r: &mut R,
    ) -> Result<Self, IoError> {
        let dim = read_u32(r)? as usize;
        let log_mod = read_u8(r)?;
        if dim != params.0.0 {
            return Err(IoError::InvalidEncoding {
                detail: format!("LWE SK dimension {} != expected {}", dim, params.0.0),
            });
        }
        if log_mod != params.1.0 {
            return Err(IoError::InvalidEncoding {
                detail: format!("log_modulus {} != expected {}", log_mod, params.1.0),
            });
        }
        let data = read_u64_slice(r, dim)?;
        if log_mod < 64 {
            let modulus = 1u64 << log_mod;
            for (i, &c) in data.iter().enumerate() {
                if c >= modulus {
                    return Err(IoError::CoefficientOutOfRange {
                        index: i,
                        value: c,
                        modulus,
                    });
                }
            }
        }
        Ok(LweSecretKey::from_data(data, params.1))
    }
}

// ── LwePublicKey ───────────────────────────────────────────────────────────

impl ObjectKind for LwePublicKey {
    const OBJECT_TYPE: ObjectType = ObjectType::LWE_PUBLIC_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for LwePublicKey {
    fn encoded_len(&self) -> usize {
        4 + self
            .ciphertexts()
            .iter()
            .map(|c| c.encoded_len())
            .sum::<usize>()
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = write_u32(w, self.count() as u32)?;
        for ct in self.ciphertexts() {
            n += ct.encode_to(w)?;
        }
        Ok(n)
    }
}

impl DecodeWithParams<(LweDimension, CiphertextModulusLog)> for LwePublicKey {
    fn decode_with_params<R: Read>(
        params: &(LweDimension, CiphertextModulusLog),
        r: &mut R,
    ) -> Result<Self, IoError> {
        let count = read_u32(r)? as usize;
        let mut cts = Vec::with_capacity(count);
        for _ in 0..count {
            cts.push(LweCiphertext::decode_with_params(params, r)?);
        }
        Ok(LwePublicKey::from_zero_encryptions(cts, params.0, params.1))
    }
}

// ── LweKeyswitchKey ────────────────────────────────────────────────────────

/// Decode params for KSK.
pub struct LweKskParams {
    pub input_dimension: LweDimension,
    pub output_dimension: LweDimension,
    pub base_log: u8,
    pub level: u8,
    pub log_modulus: CiphertextModulusLog,
}

impl ObjectKind for LweKeyswitchKey {
    const OBJECT_TYPE: ObjectType = ObjectType::LWE_KEYSWITCH_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for LweKeyswitchKey {
    fn encoded_len(&self) -> usize {
        4 + 4
            + 1
            + 1
            + 1
            + self
                .iter_blocks()
                .flat_map(|b| b.iter())
                .map(|c| c.encoded_len())
                .sum::<usize>()
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += write_u32(w, self.input_dimension().0 as u32)?;
        n += write_u32(w, self.output_dimension().0 as u32)?;
        n += write_u8(w, self.base_log())?;
        n += write_u8(w, self.level())?;
        n += write_u8(w, self.log_modulus())?;
        for block in self.iter_blocks() {
            for ct in block {
                n += ct.encode_to(w)?;
            }
        }
        Ok(n)
    }
}

impl DecodeWithParams<LweKskParams> for LweKeyswitchKey {
    fn decode_with_params<R: Read>(params: &LweKskParams, r: &mut R) -> Result<Self, IoError> {
        let in_dim = read_u32(r)? as usize;
        let out_dim = read_u32(r)? as usize;
        let base_log = read_u8(r)?;
        let level = read_u8(r)?;
        let log_mod = read_u8(r)?;
        if in_dim != params.input_dimension.0
            || out_dim != params.output_dimension.0
            || base_log != params.base_log
            || level != params.level
            || log_mod != params.log_modulus.0
        {
            return Err(IoError::InvalidEncoding {
                detail: "LweKeyswitchKey parameter mismatch".into(),
            });
        }
        let mut ksk = LweKeyswitchKey::new(
            params.input_dimension,
            params.output_dimension,
            params.base_log,
            params.level,
            params.log_modulus,
        );
        let ct_params = (params.output_dimension, params.log_modulus);
        for i in 0..in_dim {
            let block = ksk.block_mut(i);
            for ct in block.iter_mut() {
                let decoded = LweCiphertext::decode_with_params(&ct_params, r)?;
                *ct = decoded;
            }
        }
        Ok(ksk)
    }
}

// ── RLWE / BFV-style key and ciphertext types ──────────────────────────────

impl ObjectKind for SecretKey {
    const OBJECT_TYPE: ObjectType = ObjectType::RLWE_SECRET_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for SecretKey {
    fn encoded_len(&self) -> usize {
        self.value.encoded_len()
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        self.value.encode_to(w)
    }
}

impl DecodeWithParams<EncryptionParams> for SecretKey {
    fn decode_with_params<R: Read>(params: &EncryptionParams, r: &mut R) -> Result<Self, IoError> {
        Ok(SecretKey {
            value: Poly::decode_with_params(params.ring.as_ref(), r)?,
        })
    }
}

impl ObjectKind for Ciphertext {
    const OBJECT_TYPE: ObjectType = ObjectType::RLWE_CIPHERTEXT;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for Ciphertext {
    fn encoded_len(&self) -> usize {
        4 + 1
            + 1
            + 1
            + self.seed.map(|_| 64).unwrap_or(0)
            + self.data.iter().map(Poly::encoded_len).sum::<usize>()
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let count: u32 = self
            .data
            .len()
            .try_into()
            .map_err(|_| IoError::InvalidEncoding {
                detail: format!(
                    "RLWE ciphertext component count {} exceeds u32::MAX",
                    self.data.len()
                ),
            })?;
        let mut n = 0;
        n += write_u32(w, count)?;
        n += write_bool(w, self.is_ntt)?;
        n += write_bool(w, self.is_seeded_a)?;
        match self.seed {
            Some(seed) => {
                n += write_bool(w, true)?;
                n += write_bytes(w, &seed)?;
            }
            None => {
                n += write_bool(w, false)?;
            }
        }
        for poly in &self.data {
            n += poly.encode_to(w)?;
        }
        Ok(n)
    }
}

impl DecodeWithParams<EncryptionParams> for Ciphertext {
    fn decode_with_params<R: Read>(params: &EncryptionParams, r: &mut R) -> Result<Self, IoError> {
        let count = read_u32(r)? as usize;
        if count == 0 || count > 16 {
            return Err(IoError::InvalidEncoding {
                detail: format!("RLWE ciphertext component count {count} is unsupported"),
            });
        }
        let is_ntt = read_bool(r)?;
        let is_seeded_a = read_bool(r)?;
        let seed = if read_bool(r)? {
            let mut seed = [0u8; 64];
            read_exact(r, &mut seed)?;
            Some(seed)
        } else {
            None
        };
        if is_seeded_a && seed.is_none() {
            return Err(IoError::InvalidEncoding {
                detail: "RLWE ciphertext marked seeded but seed is absent".into(),
            });
        }

        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            data.push(Poly::decode_with_params(params.ring.as_ref(), r)?);
        }

        Ok(Ciphertext {
            data,
            params: params.clone(),
            is_ntt,
            seed,
            is_seeded_a,
        })
    }
}

impl ObjectKind for PublicKey {
    const OBJECT_TYPE: ObjectType = ObjectType::RLWE_PUBLIC_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for PublicKey {
    fn encoded_len(&self) -> usize {
        self.pk.encoded_len()
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        self.pk.encode_to(w)
    }
}

impl DecodeWithParams<EncryptionParams> for PublicKey {
    fn decode_with_params<R: Read>(params: &EncryptionParams, r: &mut R) -> Result<Self, IoError> {
        Ok(PublicKey {
            pk: Ciphertext::decode_with_params(params, r)?,
        })
    }
}

// ── RLWE key-switching keys ───────────────────────────────────────────────

/// Decode parameters for an RLWE evaluation key.
///
/// The ciphertext parameters describe the modulus basis used by each
/// key-switching ciphertext.  BGV and HPS-style BFV keys use the scheme's
/// active Q basis, while hybrid BFV callers can provide the extended Q∪P/Q∪K
/// basis here.
#[derive(Clone, Debug)]
pub struct EvaluationKeyDecodeParams {
    pub ciphertext_params: EncryptionParams,
    pub expected_element_count: Option<usize>,
}

impl ObjectKind for EvaluationKey {
    const OBJECT_TYPE: ObjectType = ObjectType::RLWE_EVALUATION_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for EvaluationKey {
    fn encoded_len(&self) -> usize {
        4 + self
            .elements
            .iter()
            .map(Ciphertext::encoded_len)
            .sum::<usize>()
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let count: u32 = self
            .elements
            .len()
            .try_into()
            .map_err(|_| IoError::InvalidEncoding {
                detail: format!(
                    "RLWE evaluation key element count {} exceeds u32::MAX",
                    self.elements.len()
                ),
            })?;
        let mut n = write_u32(w, count)?;
        for element in &self.elements {
            n += element.encode_to(w)?;
        }
        Ok(n)
    }
}

impl DecodeWithParams<EvaluationKeyDecodeParams> for EvaluationKey {
    fn decode_with_params<R: Read>(
        params: &EvaluationKeyDecodeParams,
        r: &mut R,
    ) -> Result<Self, IoError> {
        decode_evaluation_key(&params.ciphertext_params, params.expected_element_count, r)
    }
}

impl DecodeWithParams<EncryptionParams> for EvaluationKey {
    fn decode_with_params<R: Read>(params: &EncryptionParams, r: &mut R) -> Result<Self, IoError> {
        decode_evaluation_key(params, None, r)
    }
}

fn decode_evaluation_key<R: Read>(
    ciphertext_params: &EncryptionParams,
    expected_element_count: Option<usize>,
    r: &mut R,
) -> Result<EvaluationKey, IoError> {
    let count = read_u32(r)? as usize;
    if count == 0 || count > MAX_RLWE_EVALUATION_KEY_ELEMENTS {
        return Err(IoError::InvalidEncoding {
            detail: format!("RLWE evaluation key element count {count} is unsupported"),
        });
    }
    if let Some(expected) = expected_element_count
        && count != expected
    {
        return Err(IoError::InvalidEncoding {
            detail: format!("RLWE evaluation key element count {count} != expected {expected}"),
        });
    }

    let mut elements = Vec::with_capacity(count);
    for _ in 0..count {
        elements.push(Ciphertext::decode_with_params(ciphertext_params, r)?);
    }
    Ok(EvaluationKey { elements })
}

/// Decode parameters for an RLWE Galois key set.
#[derive(Clone, Debug)]
pub struct GaloisKeyDecodeParams {
    pub ciphertext_params: EncryptionParams,
    pub expected_galois_elements: Option<Vec<u32>>,
    pub expected_elements_per_key: Option<usize>,
}

impl ObjectKind for GaloisKey {
    const OBJECT_TYPE: ObjectType = ObjectType::RLWE_GALOIS_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for GaloisKey {
    fn encoded_len(&self) -> usize {
        4 + self.keys.iter().map(|_| 4).sum::<usize>()
            + self
                .keys
                .values()
                .map(EvaluationKey::encoded_len)
                .sum::<usize>()
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let count: u32 = self
            .keys
            .len()
            .try_into()
            .map_err(|_| IoError::InvalidEncoding {
                detail: format!(
                    "RLWE Galois key entry count {} exceeds u32::MAX",
                    self.keys.len()
                ),
            })?;
        let mut n = write_u32(w, count)?;
        let mut entries = self.keys.iter().collect::<Vec<_>>();
        entries.sort_by_key(|(galois_element, _)| **galois_element);
        for (&galois_element, key) in entries {
            n += write_u32(w, galois_element)?;
            n += key.encode_to(w)?;
        }
        Ok(n)
    }
}

impl DecodeWithParams<GaloisKeyDecodeParams> for GaloisKey {
    fn decode_with_params<R: Read>(
        params: &GaloisKeyDecodeParams,
        r: &mut R,
    ) -> Result<Self, IoError> {
        decode_galois_key(
            &params.ciphertext_params,
            params.expected_galois_elements.as_deref(),
            params.expected_elements_per_key,
            r,
        )
    }
}

impl DecodeWithParams<EncryptionParams> for GaloisKey {
    fn decode_with_params<R: Read>(params: &EncryptionParams, r: &mut R) -> Result<Self, IoError> {
        decode_galois_key(params, None, None, r)
    }
}

fn decode_galois_key<R: Read>(
    ciphertext_params: &EncryptionParams,
    expected_galois_elements: Option<&[u32]>,
    expected_elements_per_key: Option<usize>,
    r: &mut R,
) -> Result<GaloisKey, IoError> {
    let count = read_u32(r)? as usize;
    if count > MAX_RLWE_GALOIS_KEY_ENTRIES {
        return Err(IoError::InvalidEncoding {
            detail: format!("RLWE Galois key entry count {count} is unsupported"),
        });
    }
    if let Some(expected) = expected_galois_elements
        && count != expected.len()
    {
        return Err(IoError::InvalidEncoding {
            detail: format!(
                "RLWE Galois key entry count {count} != expected {}",
                expected.len()
            ),
        });
    }

    let eval_params = EvaluationKeyDecodeParams {
        ciphertext_params: ciphertext_params.clone(),
        expected_element_count: expected_elements_per_key,
    };
    let mut keys = HashMap::with_capacity(count);
    for _ in 0..count {
        let galois_element = read_u32(r)?;
        if let Some(expected) = expected_galois_elements
            && !expected.contains(&galois_element)
        {
            return Err(IoError::InvalidEncoding {
                detail: format!("unexpected RLWE Galois element {galois_element}"),
            });
        }
        let key = EvaluationKey::decode_with_params(&eval_params, r)?;
        if keys.insert(galois_element, key).is_some() {
            return Err(IoError::InvalidEncoding {
                detail: format!("duplicate RLWE Galois element {galois_element}"),
            });
        }
    }

    if let Some(expected) = expected_galois_elements {
        for &galois_element in expected {
            if !keys.contains_key(&galois_element) {
                return Err(IoError::InvalidEncoding {
                    detail: format!("missing expected RLWE Galois element {galois_element}"),
                });
            }
        }
    }

    Ok(GaloisKey { keys })
}

// ── GlweCiphertext ─────────────────────────────────────────────────────────

type GlweParams = (GlweDimension, PolynomialSize, CiphertextModulusLog);

impl ObjectKind for GlweCiphertext {
    const OBJECT_TYPE: ObjectType = ObjectType::GLWE_CIPHERTEXT;
    const SERIALIZED_VERSION: u16 = 1;
}

fn native_poly_encoded_len(p: &NativePoly) -> usize {
    1 + 4 + p.degree() * 8
}

fn encode_native_poly<W: Write>(w: &mut W, p: &NativePoly) -> Result<usize, IoError> {
    let mut n = 0;
    n += write_u8(w, p.log_modulus())?;
    n += write_u32(w, p.degree() as u32)?;
    for &c in p.coeffs() {
        n += write_u64(w, c)?;
    }
    Ok(n)
}

fn decode_native_poly<R: Read>(
    r: &mut R,
    expected_degree: usize,
    expected_log_mod: u8,
) -> Result<NativePoly, IoError> {
    let log_mod = read_u8(r)?;
    let degree = read_u32(r)? as usize;
    if log_mod != expected_log_mod {
        return Err(IoError::InvalidEncoding {
            detail: format!("NativePoly log_modulus {} != {}", log_mod, expected_log_mod),
        });
    }
    if degree != expected_degree {
        return Err(IoError::InvalidEncoding {
            detail: format!("NativePoly degree {} != {}", degree, expected_degree),
        });
    }
    let coeffs = read_u64_slice(r, degree)?;
    // Reject non-canonical coefficients: must be strictly < 2^log_modulus.
    if expected_log_mod < 64 {
        let modulus = 1u64 << expected_log_mod;
        for (i, &c) in coeffs.iter().enumerate() {
            if c >= modulus {
                return Err(IoError::CoefficientOutOfRange {
                    index: i,
                    value: c,
                    modulus,
                });
            }
        }
    }
    Ok(NativePoly::from_u64(&coeffs, log_mod))
}

impl CanonicalEncode for GlweCiphertext {
    fn encoded_len(&self) -> usize {
        4 + 4
            + 1
            + self
                .polys()
                .iter()
                .map(native_poly_encoded_len)
                .sum::<usize>()
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += write_u32(w, self.glwe_dimension().0 as u32)?;
        n += write_u32(w, self.polynomial_size().0 as u32)?;
        n += write_u8(w, self.log_modulus())?;
        for p in self.polys() {
            n += encode_native_poly(w, p)?;
        }
        Ok(n)
    }
}

impl DecodeWithParams<GlweParams> for GlweCiphertext {
    fn decode_with_params<R: Read>(params: &GlweParams, r: &mut R) -> Result<Self, IoError> {
        let glwe_dim = read_u32(r)? as usize;
        let poly_size = read_u32(r)? as usize;
        let log_mod = read_u8(r)?;
        if glwe_dim != params.0.0 || poly_size != params.1.0 || log_mod != params.2.0 {
            return Err(IoError::InvalidEncoding {
                detail: "GLWE parameter mismatch".into(),
            });
        }
        let count = glwe_dim + 1;
        let mut polys = Vec::with_capacity(count);
        for _ in 0..count {
            polys.push(decode_native_poly(r, poly_size, log_mod)?);
        }
        Ok(GlweCiphertext::from_polys(polys, params.2))
    }
}

// ── GlweSecretKey ──────────────────────────────────────────────────────────

impl ObjectKind for GlweSecretKey {
    const OBJECT_TYPE: ObjectType = ObjectType::GLWE_SECRET_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for GlweSecretKey {
    fn encoded_len(&self) -> usize {
        4 + 4
            + 1
            + self
                .polys()
                .iter()
                .map(native_poly_encoded_len)
                .sum::<usize>()
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += write_u32(w, self.glwe_dimension().0 as u32)?;
        n += write_u32(w, self.polynomial_size().0 as u32)?;
        n += write_u8(w, self.log_modulus())?;
        for p in self.polys() {
            n += encode_native_poly(w, p)?;
        }
        Ok(n)
    }
}

impl DecodeWithParams<GlweParams> for GlweSecretKey {
    fn decode_with_params<R: Read>(params: &GlweParams, r: &mut R) -> Result<Self, IoError> {
        let glwe_dim = read_u32(r)? as usize;
        let poly_size = read_u32(r)? as usize;
        let log_mod = read_u8(r)?;
        if glwe_dim != params.0.0 || poly_size != params.1.0 || log_mod != params.2.0 {
            return Err(IoError::InvalidEncoding {
                detail: "GLWE SK parameter mismatch".into(),
            });
        }
        let mut polys = Vec::with_capacity(glwe_dim);
        for _ in 0..glwe_dim {
            polys.push(decode_native_poly(r, poly_size, log_mod)?);
        }
        Ok(GlweSecretKey::from_polys(polys, params.2))
    }
}

// ── GgswCiphertext ─────────────────────────────────────────────────────────

pub struct GgswDecodeParams {
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub base_log: DecompositionBaseLog,
    pub level: DecompositionLevelCount,
    pub log_modulus: CiphertextModulusLog,
}

impl ObjectKind for GgswCiphertext {
    const OBJECT_TYPE: ObjectType = ObjectType::GGSW_CIPHERTEXT;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for GgswCiphertext {
    fn encoded_len(&self) -> usize {
        1 + 1 + 4 + 4 + 1 + {
            let glwe_size = self.glwe_dimension().glwe_size();
            let per_glwe: usize = 4
                + 4
                + 1
                + glwe_size * native_poly_encoded_len(&self.levels()[0].rows()[0].polys()[0]);
            self.level_count().0 as usize * glwe_size * per_glwe
        }
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += write_u8(w, self.level_count().0)?;
        n += write_u8(w, self.base_log().0)?;
        n += write_u32(w, self.glwe_dimension().0 as u32)?;
        n += write_u32(w, self.polynomial_size().0 as u32)?;
        n += write_u8(w, self.log_modulus())?;
        for level in self.levels() {
            for row in level.rows() {
                n += row.encode_to(w)?;
            }
        }
        Ok(n)
    }
}

impl DecodeWithParams<GgswDecodeParams> for GgswCiphertext {
    fn decode_with_params<R: Read>(params: &GgswDecodeParams, r: &mut R) -> Result<Self, IoError> {
        let level_count = read_u8(r)?;
        let base_log = read_u8(r)?;
        let glwe_dim = read_u32(r)? as usize;
        let poly_size = read_u32(r)? as usize;
        let log_mod = read_u8(r)?;
        if level_count != params.level.0
            || base_log != params.base_log.0
            || glwe_dim != params.glwe_dimension.0
            || poly_size != params.polynomial_size.0
            || log_mod != params.log_modulus.0
        {
            return Err(IoError::InvalidEncoding {
                detail: "GGSW parameter mismatch".into(),
            });
        }
        let glwe_params = (
            params.glwe_dimension,
            params.polynomial_size,
            params.log_modulus,
        );
        // Create a zeroed GGSW then overwrite each level/row in place
        let mut ggsw = GgswCiphertext::zeros(
            params.glwe_dimension,
            params.polynomial_size,
            params.base_log,
            params.level,
            params.log_modulus,
        );
        for lev in ggsw.levels_mut() {
            for row in lev.rows_mut() {
                *row = GlweCiphertext::decode_with_params(&glwe_params, r)?;
            }
        }
        Ok(ggsw)
    }
}

// ── LweBootstrapKey ────────────────────────────────────────────────────────

pub struct BskDecodeParams {
    pub input_lwe_dimension: LweDimension,
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub base_log: DecompositionBaseLog,
    pub level: DecompositionLevelCount,
    pub log_modulus: CiphertextModulusLog,
}

impl ObjectKind for LweBootstrapKey {
    const OBJECT_TYPE: ObjectType = ObjectType::LWE_BOOTSTRAP_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for LweBootstrapKey {
    fn encoded_len(&self) -> usize {
        4 + 4 + 4 + 1 + 1 + 1 + self.list().iter().map(|g| g.encoded_len()).sum::<usize>()
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += write_u32(w, self.input_lwe_dimension().0 as u32)?;
        n += write_u32(w, self.glwe_dimension().0 as u32)?;
        n += write_u32(w, self.polynomial_size().0 as u32)?;
        n += write_u8(w, self.base_log().0)?;
        n += write_u8(w, self.level().0)?;
        n += write_u8(w, self.log_modulus())?;
        for ggsw in self.list().iter() {
            n += ggsw.encode_to(w)?;
        }
        Ok(n)
    }
}

impl DecodeWithParams<BskDecodeParams> for LweBootstrapKey {
    fn decode_with_params<R: Read>(params: &BskDecodeParams, r: &mut R) -> Result<Self, IoError> {
        let in_lwe = read_u32(r)? as usize;
        let glwe_dim = read_u32(r)? as usize;
        let poly_size = read_u32(r)? as usize;
        let base_log = read_u8(r)?;
        let level = read_u8(r)?;
        let log_mod = read_u8(r)?;
        if in_lwe != params.input_lwe_dimension.0
            || glwe_dim != params.glwe_dimension.0
            || poly_size != params.polynomial_size.0
            || base_log != params.base_log.0
            || level != params.level.0
            || log_mod != params.log_modulus.0
        {
            return Err(IoError::InvalidEncoding {
                detail: "BSK parameter mismatch".into(),
            });
        }
        let ggsw_params = GgswDecodeParams {
            glwe_dimension: params.glwe_dimension,
            polynomial_size: params.polynomial_size,
            base_log: params.base_log,
            level: params.level,
            log_modulus: params.log_modulus,
        };
        let mut ggsw_list = Vec::with_capacity(in_lwe);
        for _ in 0..in_lwe {
            ggsw_list.push(GgswCiphertext::decode_with_params(&ggsw_params, r)?);
        }
        let list = GgswCiphertextList::new(ggsw_list);
        Ok(LweBootstrapKey::new(
            list,
            params.input_lwe_dimension,
            params.glwe_dimension,
            params.polynomial_size,
            params.base_log,
            params.level,
            params.log_modulus,
        ))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use silent_math::modulus::Modulus;
    use silent_math::rns::RnsBase;
    use silent_math::rns_tool::RnsToolConfig;
    use std::collections::HashMap;
    use std::io::Cursor;

    fn cm(log: u8) -> CiphertextModulusLog {
        CiphertextModulusLog(log)
    }

    fn test_rlwe_params() -> EncryptionParams {
        let base_q = RnsBase::from_values(vec![17, 97]).expect("base q");
        let base_t = Modulus::new(5).expect("base t");
        let config = RnsToolConfig::new(base_q, base_t);
        EncryptionParams::from_rns_config(8, config).expect("params")
    }

    fn test_rlwe_ciphertext(params: &EncryptionParams, offset: u64) -> Ciphertext {
        let degree = params.ring.degree();
        let num_moduli = params.ring.rns().len();
        let mut polys = Vec::new();
        for component in 0..2 {
            let mut data = Vec::with_capacity(degree * num_moduli);
            for (limb_idx, modulus) in params.ring.rns().moduli().iter().enumerate() {
                for coeff_idx in 0..degree {
                    let value =
                        offset + component as u64 * 31 + limb_idx as u64 * 7 + coeff_idx as u64;
                    data.push(value % modulus.value());
                }
            }
            polys.push(Poly::from_vec(data, degree, num_moduli));
        }
        Ciphertext::new(polys, params.clone(), true)
    }

    fn assert_ciphertext_eq(left: &Ciphertext, right: &Ciphertext) {
        assert_eq!(left.data.len(), right.data.len());
        assert_eq!(left.is_ntt, right.is_ntt);
        assert_eq!(left.seed, right.seed);
        assert_eq!(left.is_seeded_a, right.is_seeded_a);
        assert_eq!(left.params.ring.degree(), right.params.ring.degree());
        assert_eq!(
            left.params.ring.rns().moduli(),
            right.params.ring.rns().moduli()
        );
        for (left_poly, right_poly) in left.data.iter().zip(right.data.iter()) {
            assert_eq!(left_poly, right_poly);
        }
    }

    fn assert_evaluation_key_eq(left: &EvaluationKey, right: &EvaluationKey) {
        assert_eq!(left.elements.len(), right.elements.len());
        for (left_ct, right_ct) in left.elements.iter().zip(right.elements.iter()) {
            assert_ciphertext_eq(left_ct, right_ct);
        }
    }

    #[test]
    fn lwe_ciphertext_roundtrip() {
        let ct = LweCiphertext::from_data(vec![10, 20, 30, 99], cm(64));
        let mut buf = Vec::new();
        let len = ct.encode_to(&mut buf).unwrap();
        assert_eq!(len, ct.encoded_len());
        let params = (LweDimension(3), cm(64));
        let dec = LweCiphertext::decode_with_params(&params, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(ct, dec);
    }

    #[test]
    fn lwe_secret_key_roundtrip() {
        let sk = LweSecretKey::from_data(vec![1, 0, 1, 1], cm(64));
        let mut buf = Vec::new();
        sk.encode_to(&mut buf).unwrap();
        let params = (LweDimension(4), cm(64));
        let dec = LweSecretKey::decode_with_params(&params, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(sk, dec);
    }

    #[test]
    fn glwe_roundtrip() {
        let ct = GlweCiphertext::zeros(GlweDimension(1), PolynomialSize(4), cm(64));
        let mut buf = Vec::new();
        let len = ct.encode_to(&mut buf).unwrap();
        assert_eq!(len, ct.encoded_len());
        let params = (GlweDimension(1), PolynomialSize(4), cm(64));
        let dec = GlweCiphertext::decode_with_params(&params, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(ct, dec);
    }

    #[test]
    fn glwe_secret_key_roundtrip() {
        let polys = vec![NativePoly::from_u64(&[1, 0, 1, 0], 64)];
        let sk = GlweSecretKey::from_polys(polys, cm(64));
        let mut buf = Vec::new();
        sk.encode_to(&mut buf).unwrap();
        let params = (GlweDimension(1), PolynomialSize(4), cm(64));
        let dec = GlweSecretKey::decode_with_params(&params, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(sk, dec);
    }

    #[test]
    fn lwe_pubkey_roundtrip() {
        let cts: Vec<_> = (0..3)
            .map(|_| LweCiphertext::zeros(LweDimension(2), cm(64)))
            .collect();
        let pk = LwePublicKey::from_zero_encryptions(cts, LweDimension(2), cm(64));
        let mut buf = Vec::new();
        pk.encode_to(&mut buf).unwrap();
        let params = (LweDimension(2), cm(64));
        let dec = LwePublicKey::decode_with_params(&params, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(dec.count(), 3);
        assert_eq!(dec.dimension().0, 2);
    }

    #[test]
    fn ksk_roundtrip() {
        let ksk = LweKeyswitchKey::new(LweDimension(3), LweDimension(2), 4, 2, cm(64));
        let mut buf = Vec::new();
        let len = ksk.encode_to(&mut buf).unwrap();
        assert_eq!(len, ksk.encoded_len());
        let params = LweKskParams {
            input_dimension: LweDimension(3),
            output_dimension: LweDimension(2),
            base_log: 4,
            level: 2,
            log_modulus: cm(64),
        };
        let dec = LweKeyswitchKey::decode_with_params(&params, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(dec.input_dimension().0, 3);
        assert_eq!(dec.output_dimension().0, 2);
    }

    #[test]
    fn rlwe_evaluation_key_roundtrip() {
        let params = test_rlwe_params();
        let evk = EvaluationKey {
            elements: vec![
                test_rlwe_ciphertext(&params, 3),
                test_rlwe_ciphertext(&params, 11),
            ],
        };

        let mut buf = Vec::new();
        let len = evk.encode_to(&mut buf).unwrap();
        assert_eq!(len, evk.encoded_len());
        assert_eq!(EvaluationKey::OBJECT_TYPE, ObjectType::RLWE_EVALUATION_KEY);

        let decode_params = EvaluationKeyDecodeParams {
            ciphertext_params: params.clone(),
            expected_element_count: Some(2),
        };
        let decoded =
            EvaluationKey::decode_with_params(&decode_params, &mut Cursor::new(&buf)).unwrap();
        assert_evaluation_key_eq(&decoded, &evk);

        let mismatched = EvaluationKeyDecodeParams {
            ciphertext_params: params.clone(),
            expected_element_count: Some(3),
        };
        assert!(matches!(
            EvaluationKey::decode_with_params(&mismatched, &mut Cursor::new(&buf)),
            Err(IoError::InvalidEncoding { .. })
        ));

        let mut empty = buf.clone();
        empty[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            EvaluationKey::decode_with_params(&params, &mut Cursor::new(&empty)),
            Err(IoError::InvalidEncoding { .. })
        ));
    }

    #[test]
    fn rlwe_galois_key_roundtrip_is_canonical_and_rejects_duplicates() {
        let params = test_rlwe_params();
        let evk_3 = EvaluationKey {
            elements: vec![
                test_rlwe_ciphertext(&params, 19),
                test_rlwe_ciphertext(&params, 23),
            ],
        };
        let evk_5 = EvaluationKey {
            elements: vec![
                test_rlwe_ciphertext(&params, 29),
                test_rlwe_ciphertext(&params, 31),
            ],
        };

        let mut forward = HashMap::new();
        forward.insert(5, evk_5.clone());
        forward.insert(3, evk_3.clone());
        let forward = GaloisKey { keys: forward };

        let mut reverse = HashMap::new();
        reverse.insert(3, evk_3.clone());
        reverse.insert(5, evk_5.clone());
        let reverse = GaloisKey { keys: reverse };

        let mut forward_buf = Vec::new();
        let forward_len = forward.encode_to(&mut forward_buf).unwrap();
        assert_eq!(forward_len, forward.encoded_len());
        assert_eq!(GaloisKey::OBJECT_TYPE, ObjectType::RLWE_GALOIS_KEY);

        let mut reverse_buf = Vec::new();
        reverse.encode_to(&mut reverse_buf).unwrap();
        assert_eq!(forward_buf, reverse_buf);

        let decode_params = GaloisKeyDecodeParams {
            ciphertext_params: params.clone(),
            expected_galois_elements: Some(vec![5, 3]),
            expected_elements_per_key: Some(2),
        };
        let decoded =
            GaloisKey::decode_with_params(&decode_params, &mut Cursor::new(&forward_buf)).unwrap();
        assert_eq!(decoded.keys.len(), 2);
        assert_evaluation_key_eq(decoded.keys.get(&3).unwrap(), &evk_3);
        assert_evaluation_key_eq(decoded.keys.get(&5).unwrap(), &evk_5);

        let unexpected_params = GaloisKeyDecodeParams {
            ciphertext_params: params.clone(),
            expected_galois_elements: Some(vec![3, 7]),
            expected_elements_per_key: Some(2),
        };
        assert!(matches!(
            GaloisKey::decode_with_params(&unexpected_params, &mut Cursor::new(&forward_buf)),
            Err(IoError::InvalidEncoding { .. })
        ));

        let mut duplicate = Vec::new();
        write_u32(&mut duplicate, 2).unwrap();
        write_u32(&mut duplicate, 3).unwrap();
        evk_3.encode_to(&mut duplicate).unwrap();
        write_u32(&mut duplicate, 3).unwrap();
        evk_5.encode_to(&mut duplicate).unwrap();
        assert!(matches!(
            GaloisKey::decode_with_params(&params, &mut Cursor::new(&duplicate)),
            Err(IoError::InvalidEncoding { .. })
        ));
    }
}
