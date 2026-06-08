use crate::context::RingContext;
use crate::native_poly::NativePoly;
use crate::poly::Poly;

use silent_io::error::IoError;
use silent_io::object_type::ObjectType;
use silent_io::primitives::{read_exact, read_u8, read_u32, write_u8, write_u32};
use silent_io::traits::{CanonicalEncode, DecodeWithParams, ObjectKind};
use silent_io::validate::{validate_coefficients, validate_degree, validate_num_moduli};
use std::io::{Read, Write};

// ── Poly ───────────────────────────────────────────────────────────────────

impl ObjectKind for Poly {
    const OBJECT_TYPE: ObjectType = ObjectType::POLY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for Poly {
    fn encoded_len(&self) -> usize {
        1 + 4 + (self.data().len() * 8)
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut written = 0;
        written += write_u8(w, self.num_moduli() as u8)?;
        written += write_u32(w, self.degree() as u32)?;

        let mut buf = [0u8; 8];
        for &coeff in self.data() {
            buf.copy_from_slice(&coeff.to_le_bytes());
            w.write_all(&buf)?;
            written += 8;
        }

        Ok(written)
    }
}

impl DecodeWithParams<RingContext> for Poly {
    fn decode_with_params<R: Read>(context: &RingContext, r: &mut R) -> Result<Self, IoError> {
        let num_moduli_decoded = read_u8(r)?;
        let num_moduli = validate_num_moduli(num_moduli_decoded, context.rns().len())?;

        let degree_decoded = read_u32(r)?;
        let degree = validate_degree(degree_decoded, context.degree())?;

        let total_coeffs = degree * num_moduli;
        let mut data = Vec::with_capacity(total_coeffs);
        let mut buf = [0u8; 8];
        for _ in 0..total_coeffs {
            read_exact(r, &mut buf)?;
            data.push(u64::from_le_bytes(buf));
        }

        let moduli: Vec<u64> = context.rns().moduli().iter().map(|m| m.value()).collect();
        validate_coefficients(&data, &moduli, degree)?;

        Ok(Poly::from_vec(data, degree, num_moduli))
    }
}

// ── NativePoly ─────────────────────────────────────────────────────────────

impl ObjectKind for NativePoly {
    const OBJECT_TYPE: ObjectType = ObjectType::NATIVE_POLY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for NativePoly {
    fn encoded_len(&self) -> usize {
        1 + 4 + (self.degree() * 8)
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut written = 0;
        written += write_u8(w, self.log_modulus())?;
        written += write_u32(w, self.degree() as u32)?;

        let mut buf = [0u8; 8];
        for &coeff in self.coeffs() {
            buf.copy_from_slice(&coeff.to_le_bytes());
            w.write_all(&buf)?;
            written += 8;
        }

        Ok(written)
    }
}

/// Parameters for decoding a `NativePoly` are `(degree, log_modulus)`.
impl DecodeWithParams<(usize, u8)> for NativePoly {
    fn decode_with_params<R: Read>(params: &(usize, u8), r: &mut R) -> Result<Self, IoError> {
        let (expected_degree, expected_log_modulus) = *params;

        let log_modulus = read_u8(r)?;
        if log_modulus != expected_log_modulus {
            return Err(IoError::InvalidEncoding {
                detail: format!(
                    "NativePoly log_modulus {} does not match expected {}",
                    log_modulus, expected_log_modulus
                ),
            });
        }

        let degree_decoded = read_u32(r)?;
        let degree = validate_degree(degree_decoded, expected_degree)?;

        let mut coeffs = Vec::with_capacity(degree);
        let mut buf = [0u8; 8];
        for i in 0..degree {
            read_exact(r, &mut buf)?;
            let c = u64::from_le_bytes(buf);
            // Reject non-canonical coefficients: must be strictly < 2^log_modulus.
            if expected_log_modulus < 64 {
                let modulus = 1u64 << expected_log_modulus;
                if c >= modulus {
                    return Err(IoError::CoefficientOutOfRange {
                        index: i,
                        value: c,
                        modulus,
                    });
                }
            }
            coeffs.push(c);
        }

        Ok(NativePoly::from_u64(&coeffs, expected_log_modulus))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use silent_math::rns::RnsBase;
    use std::io::Cursor;

    fn get_context() -> RingContext {
        let rns = RnsBase::from_values(vec![17, 97]).unwrap();
        RingContext::new(8, rns)
    }

    #[test]
    fn test_poly_roundtrip() {
        let context = get_context();
        let data = vec![
            1, 2, 3, 4, 5, 6, 7, 8, // mod 17
            1, 2, 3, 4, 5, 6, 7, 8, // mod 97
        ];
        let poly = Poly::from_vec(data, 8, 2);

        let mut encoded = Vec::new();
        let len = poly.encode_to(&mut encoded).unwrap();
        assert_eq!(len, poly.encoded_len());
        assert_eq!(len, encoded.len());

        let mut r = Cursor::new(&encoded);
        let decoded = Poly::decode_with_params(&context, &mut r).unwrap();
        assert_eq!(poly, decoded);
    }

    #[test]
    fn test_poly_validation_fails() {
        let context = get_context();
        let mut data = vec![
            1, 2, 3, 4, 5, 6, 7, 8, // mod 17
            1, 2, 3, 4, 5, 6, 7, 8, // mod 97
        ];
        data[0] = 18; // Invalid coefficient for modulus 17
        let poly = Poly::from_vec(data, 8, 2);

        let mut encoded = Vec::new();
        poly.encode_to(&mut encoded).unwrap();

        let mut r = Cursor::new(&encoded);
        let res = Poly::decode_with_params(&context, &mut r);
        assert!(matches!(res, Err(IoError::CoefficientOutOfRange { .. })));
    }

    #[test]
    fn test_native_poly_roundtrip() {
        let coeffs = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let poly = NativePoly::from_u64(&coeffs, 32);

        let mut encoded = Vec::new();
        let len = poly.encode_to(&mut encoded).unwrap();
        assert_eq!(len, poly.encoded_len());
        assert_eq!(len, encoded.len());

        let params = (8, 32);
        let mut r = Cursor::new(&encoded);
        let decoded = NativePoly::decode_with_params(&params, &mut r).unwrap();
        assert_eq!(poly, decoded);
    }

    #[test]
    fn test_native_poly_invalid_log_modulus() {
        let coeffs = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let poly = NativePoly::from_u64(&coeffs, 32);

        let mut encoded = Vec::new();
        poly.encode_to(&mut encoded).unwrap();

        // Expecting log_modulus 64, but encoded has 32
        let params = (8, 64);
        let mut r = Cursor::new(&encoded);
        let res = NativePoly::decode_with_params(&params, &mut r);
        assert!(matches!(res, Err(IoError::InvalidEncoding { .. })));
    }

    #[test]
    fn test_native_poly_rejects_non_canonical_coefficient() {
        // Encode a valid poly with log_modulus=32, then tamper a coefficient
        // to be >= 2^32.
        let coeffs = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let poly = NativePoly::from_u64(&coeffs, 32);

        let mut encoded = Vec::new();
        poly.encode_to(&mut encoded).unwrap();

        // The layout is: 1 byte log_modulus + 4 bytes degree + 8 bytes × N.
        // Tamper the first coefficient (offset 5) to 2^32 = 0x1_0000_0000.
        let coeff_offset = 5;
        let bad_value: u64 = 1u64 << 32;
        encoded[coeff_offset..coeff_offset + 8].copy_from_slice(&bad_value.to_le_bytes());

        let params = (8, 32);
        let mut r = Cursor::new(&encoded);
        let res = NativePoly::decode_with_params(&params, &mut r);
        assert!(
            matches!(res, Err(IoError::CoefficientOutOfRange { index: 0, .. })),
            "Expected CoefficientOutOfRange, got {:?}",
            res
        );
    }
}
