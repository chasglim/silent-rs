//! Validation utilities for canonical encoding.
//!
//! Every function returns [`IoError`] on failure and is designed to be called
//! during [`crate::DecodeWithParams::decode_with_params`] so that malformed
//! data is rejected at the decode boundary.

use crate::error::IoError;

/// Maximum polynomial degree SILENT's ring layer supports.
const MAX_DEGREE: usize = 1 << 17; // 131072

/// Maximum number of moduli in a modulus chain.
const MAX_NUM_MODULI: u8 = 64;

/// Validate that a polynomial degree is positive, a power of two, matches the
/// expected value, and does not exceed the absolute ceiling.
pub fn validate_degree(decoded: u32, expected: usize) -> Result<usize, IoError> {
    let degree = decoded as usize;
    if degree == 0 {
        return Err(IoError::InvalidDegree {
            found: decoded,
            reason: "degree must be positive",
        });
    }
    if !degree.is_power_of_two() {
        return Err(IoError::InvalidDegree {
            found: decoded,
            reason: "degree must be a power of two",
        });
    }
    if degree > MAX_DEGREE {
        return Err(IoError::InvalidDegree {
            found: decoded,
            reason: "degree exceeds absolute maximum",
        });
    }
    if degree != expected {
        return Err(IoError::InvalidDegree {
            found: decoded,
            reason: "degree does not match parameter set",
        });
    }
    Ok(degree)
}

/// Validate that `num_moduli` is positive, matches the expected chain length,
/// and does not exceed the absolute ceiling.
pub fn validate_num_moduli(decoded: u8, expected: usize) -> Result<usize, IoError> {
    if decoded == 0 {
        return Err(IoError::InvalidNumModuli {
            found: decoded,
            reason: "num_moduli must be positive",
        });
    }
    if decoded > MAX_NUM_MODULI {
        return Err(IoError::InvalidNumModuli {
            found: decoded,
            reason: "num_moduli exceeds absolute maximum",
        });
    }
    let n = decoded as usize;
    if n != expected {
        return Err(IoError::InvalidNumModuli {
            found: decoded,
            reason: "num_moduli does not match parameter set",
        });
    }
    Ok(n)
}

/// Validate every coefficient in `data` is strictly less than its corresponding
/// modulus.  Coefficients are indexed sequentially across all limbs.
///
/// `moduli` is the modulus chain (one modulus per limb).  Each limb is
/// `degree` elements wide.
///
/// Returns `Ok(())` or [`IoError::CoefficientOutOfRange`] on the first
/// violation.
pub fn validate_coefficients(data: &[u64], moduli: &[u64], degree: usize) -> Result<(), IoError> {
    let num_moduli = moduli.len();
    if data.len() != degree * num_moduli {
        return Err(IoError::InvalidEncoding {
            detail: format!(
                "coefficient array length {} != degree {} × num_moduli {}",
                data.len(),
                degree,
                num_moduli
            ),
        });
    }
    for (limb_idx, &modulus) in moduli.iter().enumerate().take(num_moduli) {
        let start = limb_idx * degree;
        let limb = &data[start..start + degree];
        for (i, &coeff) in limb.iter().enumerate() {
            if coeff >= modulus {
                return Err(IoError::CoefficientOutOfRange {
                    index: start + i,
                    value: coeff,
                    modulus,
                });
            }
        }
    }
    Ok(())
}

/// Validate that a payload length does not exceed a configurable maximum.
pub fn validate_payload_len(declared: u64, max: u64) -> Result<(), IoError> {
    if declared > max {
        return Err(IoError::PayloadTooLarge {
            declared,
            limit: max,
        });
    }
    Ok(())
}

/// Validate that a frame payload length does not exceed a configurable maximum.
pub fn validate_frame_payload_len(declared: u32, max: u32) -> Result<(), IoError> {
    if declared > max {
        return Err(IoError::PayloadTooLarge {
            declared: declared as u64,
            limit: max as u64,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_degree_ok() {
        assert_eq!(validate_degree(1024, 1024).unwrap(), 1024);
    }

    #[test]
    fn validate_degree_zero() {
        let err = validate_degree(0, 1024).unwrap_err();
        assert!(matches!(err, IoError::InvalidDegree { .. }));
    }

    #[test]
    fn validate_degree_not_power_of_two() {
        let err = validate_degree(1023, 1024).unwrap_err();
        assert!(matches!(err, IoError::InvalidDegree { .. }));
    }

    #[test]
    fn validate_degree_mismatch() {
        let err = validate_degree(2048, 1024).unwrap_err();
        assert!(matches!(err, IoError::InvalidDegree { .. }));
    }

    #[test]
    fn validate_degree_exceeds_max() {
        let err = validate_degree((MAX_DEGREE * 2) as u32, MAX_DEGREE * 2).unwrap_err();
        assert!(matches!(err, IoError::InvalidDegree { .. }));
    }

    #[test]
    fn validate_num_moduli_ok() {
        assert_eq!(validate_num_moduli(5, 5).unwrap(), 5);
    }

    #[test]
    fn validate_num_moduli_zero() {
        let err = validate_num_moduli(0, 5).unwrap_err();
        assert!(matches!(err, IoError::InvalidNumModuli { .. }));
    }

    #[test]
    fn validate_num_moduli_mismatch() {
        let err = validate_num_moduli(3, 5).unwrap_err();
        assert!(matches!(err, IoError::InvalidNumModuli { .. }));
    }

    #[test]
    fn validate_coefficients_ok() {
        let data = vec![0u64, 1, 2, 3, 0, 1, 2, 3];
        let moduli = vec![4u64, 4];
        assert!(validate_coefficients(&data, &moduli, 4).is_ok());
    }

    #[test]
    fn validate_coefficients_out_of_range() {
        let data = vec![0u64, 1, 5, 3]; // coefficient index 2 = 5 >= modulus 4
        let moduli = vec![4u64];
        let err = validate_coefficients(&data, &moduli, 4).unwrap_err();
        assert!(matches!(
            err,
            IoError::CoefficientOutOfRange { index: 2, .. }
        ));
    }

    #[test]
    fn validate_coefficients_wrong_length() {
        let data = vec![0u64, 1, 2]; // too short
        let moduli = vec![4u64, 4];
        let err = validate_coefficients(&data, &moduli, 4).unwrap_err();
        assert!(matches!(err, IoError::InvalidEncoding { .. }));
    }

    #[test]
    fn validate_payload_len_ok() {
        assert!(validate_payload_len(100, 1024).is_ok());
    }

    #[test]
    fn validate_payload_len_too_large() {
        let err = validate_payload_len(2048, 1024).unwrap_err();
        assert!(matches!(err, IoError::PayloadTooLarge { .. }));
    }

    #[test]
    fn validate_frame_payload_len_ok() {
        assert!(validate_frame_payload_len(100, 1024).is_ok());
    }

    #[test]
    fn validate_frame_payload_len_too_large() {
        let err = validate_frame_payload_len(2048, 1024).unwrap_err();
        assert!(matches!(err, IoError::PayloadTooLarge { .. }));
    }
}
