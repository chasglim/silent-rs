use crate::object_type::ObjectType;
use std::fmt;

/// Unified error type for all I/O, encoding, and validation failures in
/// `silent-io`.
///
/// Error messages intentionally avoid embedding sensitive data (secret key
/// bytes, plaintext coefficients, etc.).
#[derive(Debug)]
pub enum IoError {
    /// Wraps a standard library I/O error.
    Io(std::io::Error),

    /// Container magic bytes did not match the expected `b"SFIR"`.
    InvalidMagic {
        /// The four bytes found in the stream.
        actual: [u8; 4],
    },

    /// The container format version is newer than what this library supports.
    UnsupportedVersion {
        /// Version found in the stream.
        found: u16,
        /// Maximum version this library can decode.
        max: u16,
    },

    /// The `object_type` tag in the header does not correspond to any known
    /// [`ObjectType`] variant.
    UnknownObjectType {
        /// The raw u32 tag value from the stream.
        tag: u32,
    },

    /// The header declares a different object type than what the caller
    /// requested.
    ObjectTypeMismatch {
        /// Type that was requested.
        expected: ObjectType,
        /// Type found in the header.
        actual: ObjectType,
    },

    /// The serialized object version is newer than what this library supports.
    ObjectVersionTooNew {
        /// Version found in the stream.
        data_version: u16,
        /// Maximum version this library can decode for this type.
        lib_version: u16,
    },

    /// The `params_id` in the header does not match the caller's expected
    /// parameter set.
    ParamsIdMismatch {
        /// `ParamsId` the caller expected.
        expected: String,
        /// `ParamsId` found in the header.
        actual: String,
    },

    /// The declared payload length exceeds the configured size limit.
    PayloadTooLarge {
        /// Payload length declared in the header.
        declared: u64,
        /// Configured maximum.
        limit: u64,
    },

    /// Coefficient value is out of the valid range \[0, modulus).
    CoefficientOutOfRange {
        /// Index of the offending coefficient.
        index: usize,
        /// The invalid value.
        value: u64,
        /// The modulus that the value must be strictly less than.
        modulus: u64,
    },

    /// The degree field in the encoded object is invalid or inconsistent with
    /// parameters.
    InvalidDegree {
        /// Degree found in the stream.
        found: u32,
        /// Reason the degree is invalid.
        reason: &'static str,
    },

    /// The `num_moduli` field in the encoded object is invalid or inconsistent
    /// with parameters.
    InvalidNumModuli {
        /// Number of moduli found in the stream.
        found: u8,
        /// Reason the count is invalid.
        reason: &'static str,
    },

    /// Container checksum does not match the computed value.
    ChecksumMismatch {
        /// Checksum read from the stream.
        expected: u64,
        /// Checksum computed from the data.
        actual: u64,
    },

    /// The stream ended before the expected number of bytes could be read.
    UnexpectedEof {
        /// Number of bytes expected.
        expected: usize,
        /// Number of bytes remaining in the stream.
        remaining: usize,
    },

    /// The encoded data is malformed in a way that cannot be attributed to a
    /// specific field.
    InvalidEncoding {
        /// Human-readable description.
        detail: String,
    },

    /// A general-purpose validation failure.
    Validation {
        /// Human-readable description.
        detail: String,
    },
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidMagic { actual } => {
                write!(
                    f,
                    "invalid container magic: expected b\"SFIR\", got {actual:02x?}"
                )
            }
            Self::UnsupportedVersion { found, max } => {
                write!(
                    f,
                    "unsupported format version: stream uses v{found}, library supports up to v{max}"
                )
            }
            Self::UnknownObjectType { tag } => {
                write!(f, "unknown object type tag 0x{tag:08x}")
            }
            Self::ObjectTypeMismatch { expected, actual } => {
                write!(
                    f,
                    "object type mismatch: requested {expected}, stream contains {actual}"
                )
            }
            Self::ObjectVersionTooNew {
                data_version,
                lib_version,
            } => {
                write!(
                    f,
                    "object version too new: stream uses v{data_version}, library supports v{lib_version}"
                )
            }
            Self::ParamsIdMismatch { expected, actual } => {
                write!(f, "parameter mismatch: expected {expected}, got {actual}")
            }
            Self::PayloadTooLarge { declared, limit } => {
                write!(
                    f,
                    "payload too large: declared {declared} bytes exceeds limit of {limit}"
                )
            }
            Self::CoefficientOutOfRange {
                index,
                value,
                modulus,
            } => {
                write!(
                    f,
                    "coefficient at index {index} out of range: value {value} >= modulus {modulus}"
                )
            }
            Self::InvalidDegree { found, reason } => {
                write!(f, "invalid degree {found}: {reason}")
            }
            Self::InvalidNumModuli { found, reason } => {
                write!(f, "invalid num_moduli {found}: {reason}")
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "checksum mismatch: expected {expected:016x}, computed {actual:016x}"
                )
            }
            Self::UnexpectedEof {
                expected,
                remaining,
            } => {
                write!(
                    f,
                    "unexpected EOF: expected {expected} bytes, only {remaining} available"
                )
            }
            Self::InvalidEncoding { detail } => {
                write!(f, "invalid encoding: {detail}")
            }
            Self::Validation { detail } => {
                write!(f, "validation error: {detail}")
            }
        }
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ── Convenience constructors ────────────────────────────────────────────────

impl IoError {
    /// Create an [`IoError::PayloadTooLarge`] from a `u64` declared length and
    /// a `usize` limit.
    pub fn payload_too_large(declared: u64, limit: usize) -> Self {
        Self::PayloadTooLarge {
            declared,
            limit: limit as u64,
        }
    }

    /// Create an [`IoError::UnexpectedEof`] from a `usize` expected count.
    pub fn unexpected_eof(expected: usize, remaining: usize) -> Self {
        Self::UnexpectedEof {
            expected,
            remaining,
        }
    }
}
