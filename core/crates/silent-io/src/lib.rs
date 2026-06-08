//! Canonical serialization, persistence, and wire encoding for SILENT core types.
//!
//! This crate defines the **canonical binary codec** — the single source of truth
//! for how cryptographic objects are represented as bytes.  It is the foundation
//! for key storage, ciphertext persistence, test vectors, and network transport.
//!
//! ## Architecture
//!
//! - [`CanonicalEncode`] / [`CanonicalDecode`] / [`DecodeWithParams`]: the three
//!   core encoding traits.  Parameter-dependent crypto objects **must not** implement
//!   `CanonicalDecode`; they decode exclusively via `DecodeWithParams`.
//! - [`ObjectKind`]: static type metadata (object type tag, serialized version).
//! - [`primitives`]: little-endian fixint read/write helpers — the canonical
//!   encoding building blocks.
//! - [`ObjectType`]: u32 type-tag registry for every persistable type.
//!
//! ## Relationship to other crates
//!
//! - `silent-params`: will depend on `silent-io` and re-export `ParamsId`.
//! - `silent-ring` / `silent-rlwe` / `silent-fhe` / `silent-hss`:
//!   each implements `CanonicalEncode` and `DecodeWithParams` for its types
//!   in an `io_impls.rs` module.
//! - `silent-net`: consumes length-delimited frames from `silent-io::framed`.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// I/O errors, encoding failures, and validation rejections.
pub mod error;
/// u32 type-tag registry for every persistable SILENT type.
pub mod object_type;
/// Little-endian fixint read/write primitives.
pub mod primitives;
/// Core encoding traits (`CanonicalEncode`, `DecodeWithParams`, `ObjectKind`, etc.).
pub mod traits;

pub mod container;
pub mod framed;
pub mod header;
pub mod secret;
pub mod validate;
mod xxh3;

pub use error::IoError;
pub use object_type::ObjectType;
pub use traits::{
    CanonicalDecode, CanonicalEncode, DecodeWithParams, EncodeContext, HasParamsId, ObjectKind,
};

/// A 256-bit parameter-set identifier (SHA-256 digest of canonical parameter
/// encoding).  All-zero means "no parameter binding".
///
/// Defined here as a transparent byte wrapper so that `silent-io` has no
/// reverse dependency on `silent-params`.  `silent-params` may re-export
/// this type in the future.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamsId(pub [u8; 32]);

impl ParamsId {
    /// The all-zero `ParamsId`, used when an object has no parameter binding.
    pub const ZERO: Self = Self([0u8; 32]);

    /// View as a byte slice.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Whether this `ParamsId` is the all-zero sentinel.
    pub fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }
}

impl core::fmt::Debug for ParamsId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("ParamsId")
            .field(&hex::encode(&self.0))
            .finish()
    }
}

impl core::fmt::Display for ParamsId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_zero() {
            f.write_str("ParamsId(0)")
        } else {
            let hex_str = hex::encode(&self.0);
            f.write_str(&hex_str[..16])?;
            f.write_str("..")
        }
    }
}

// ── Internal hex formatting (no external dependency) ────────────────────────

mod hex {
    const TABLE: &[u8; 16] = b"0123456789abcdef";

    pub fn encode(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            out.push(TABLE[(b >> 4) as usize] as char);
            out.push(TABLE[(b & 0x0f) as usize] as char);
        }
        out
    }
}

/// Semantic versioning identifier for this crate.
pub const CRATE_ID: &str = "silent-io";

/// Current version of the SILENT container format.
pub const FORMAT_VERSION: u16 = 1;

/// Magic bytes that prefix every SILENT file container.
pub const CONTAINER_MAGIC: &[u8; 4] = b"SFIR";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_id_zero_is_zero() {
        assert!(ParamsId::ZERO.is_zero());
        assert!(!ParamsId([1u8; 32]).is_zero());
    }

    #[test]
    fn params_id_debug_does_not_panic() {
        let id = ParamsId([0xabu8; 32]);
        let _ = format!("{id:?}");
        let _ = format!("{id}");
    }

    #[test]
    fn format_version_is_reasonable() {
        assert!(FORMAT_VERSION >= 1);
    }

    #[test]
    fn container_magic_is_four_bytes() {
        assert_eq!(CONTAINER_MAGIC.len(), 4);
        assert_eq!(CONTAINER_MAGIC, b"SFIR");
    }
}
