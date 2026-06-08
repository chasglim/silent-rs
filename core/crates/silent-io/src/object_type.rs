//! The [`ObjectType`] registry — a `u32`-discriminated enum that identifies
//! every persistable SILENT type in a container or frame header.
//!
//! Tags are grouped by category so that new types can be added without
//! collisions:
//!
//! | Range        | Category                    |
//! |--------------|-----------------------------|
//! | `0x0001-0x0FFF` | ring / math               |
//! | `0x0101-0x01FF` | lattice / TFHE-compatible  |
//! | `0x0201-0x02FF` | RLWE / BFV-style           |
//! | `0x0301-0x03FF` | HSS                        |
//! | `0x0401-0x04FF` | Shortint                   |
//! | `0x1001-0x10FF` | parameter descriptors       |
//! | `0x2001-0x2FFF` | container-level metadata    |
//! | `0x0501-0x05FF` | PQC                        |
//! | `0x3000-0xFFFF` | reserved / application      |

use core::fmt;

/// Numeric type tag embedded in container and frame headers.
///
/// This is deliberately a `u32` rather than a Rust enum so that unknown tags
/// from future library versions can be detected gracefully without crashing.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ObjectType(u32);

// ── Known tags ──────────────────────────────────────────────────────────────

impl ObjectType {
    // ── ring ────────────────────────────────────────────────────────────────

    /// An RNS polynomial over a modulus chain.
    pub const POLY: Self = Self(0x0001);

    /// A native power-of-two modulus polynomial (used by TFHE GLWE).
    pub const NATIVE_POLY: Self = Self(0x0002);

    /// Gadget decomposition parameters (base + level).
    pub const GADGET_DECOMPOSITION: Self = Self(0x0003);

    // ── lattice / TFHE-compatible ───────────────────────────────────────────

    /// An LWE ciphertext under a power-of-two modulus.
    pub const LWE_CIPHERTEXT: Self = Self(0x0101);

    /// An LWE secret key.
    pub const LWE_SECRET_KEY: Self = Self(0x0102);

    /// An LWE public key.
    pub const LWE_PUBLIC_KEY: Self = Self(0x0103);

    /// An LWE key-switching key.
    pub const LWE_KEYSWITCH_KEY: Self = Self(0x0104);

    /// A GLWE ciphertext (generalisation of LWE over a polynomial ring).
    pub const GLWE_CIPHERTEXT: Self = Self(0x0105);

    /// A GLWE secret key.
    pub const GLWE_SECRET_KEY: Self = Self(0x0106);

    /// A GGSW ciphertext (gadget-encrypted GLWE).
    pub const GGSW_CIPHERTEXT: Self = Self(0x0107);

    /// An LWE-to-GLWE bootstrap key.
    pub const LWE_BOOTSTRAP_KEY: Self = Self(0x0108);

    // ── RLWE / BFV-style ────────────────────────────────────────────────────

    /// An RLWE secret key (BFV-style polynomial key).
    pub const RLWE_SECRET_KEY: Self = Self(0x0201);

    /// An RLWE public key.
    pub const RLWE_PUBLIC_KEY: Self = Self(0x0202);

    /// An RLWE ciphertext.
    pub const RLWE_CIPHERTEXT: Self = Self(0x0203);

    /// An RLWE evaluation (relinearisation) key.
    pub const RLWE_EVALUATION_KEY: Self = Self(0x0204);

    /// An RLWE Galois automorphism key.
    pub const RLWE_GALOIS_KEY: Self = Self(0x0205);

    /// A BFV proxy re-encryption key.
    pub const BFV_REKEY: Self = Self(0x0206);

    /// A BGV ciphertext with level and plaintext-factor metadata.
    pub const BGV_CIPHERTEXT: Self = Self(0x0207);

    // ── HSS ─────────────────────────────────────────────────────────────────

    /// An HSS ciphertext (pair of RLWE ciphertexts).
    pub const HSS_CIPHERTEXT: Self = Self(0x0301);

    /// An HSS share.
    pub const HSS_SHARE: Self = Self(0x0302);

    /// An HSS evaluation key.
    pub const HSS_EVAL_KEY: Self = Self(0x0303);

    // ── Shortint ────────────────────────────────────────────────────────────

    /// A Shortint ciphertext.
    pub const SHORTINT_CIPHERTEXT: Self = Self(0x0401);

    /// A Shortint client key.
    pub const SHORTINT_CLIENT_KEY: Self = Self(0x0402);

    /// A Shortint server key.
    pub const SHORTINT_SERVER_KEY: Self = Self(0x0403);

    // ── PQC ──────────────────────────────────────────────────────────────────

    /// An ML-KEM public key (encapsulation key).
    pub const PQC_KEM_PUBLIC_KEY: Self = Self(0x0501);

    /// An ML-KEM secret key blob (encrypted).
    pub const PQC_KEM_SECRET_KEY_BLOB: Self = Self(0x0502);

    /// An ML-KEM ciphertext.
    pub const PQC_KEM_CIPHERTEXT: Self = Self(0x0503);

    /// Canonical KEM parameter set descriptor.
    pub const PQC_KEM_PARAMS: Self = Self(0x0504);

    /// An ML-DSA verification key (public).
    pub const PQC_SIG_VERIFYING_KEY: Self = Self(0x0505);

    /// An ML-DSA signing key blob (encrypted).
    pub const PQC_SIG_SIGNING_KEY_BLOB: Self = Self(0x0506);

    /// An ML-DSA signature.
    pub const PQC_SIG_SIGNATURE: Self = Self(0x0507);

    /// Canonical signature parameter set descriptor.
    pub const PQC_SIG_PARAMS: Self = Self(0x0508);

    // ── parameter descriptors ───────────────────────────────────────────────

    /// Canonical ring parameters.
    pub const RING_PARAMS: Self = Self(0x1001);

    /// Canonical RLWE parameters.
    pub const RLWE_PARAMS: Self = Self(0x1002);

    /// Canonical BFV parameters.
    pub const BFV_PARAMS: Self = Self(0x1003);

    /// Canonical HSS parameters.
    pub const HSS_PARAMS: Self = Self(0x1004);

    /// Canonical TFHE parameters.
    pub const TFHE_PARAMS: Self = Self(0x1005);

    /// Canonical BGV parameters.
    pub const BGV_PARAMS: Self = Self(0x1006);
}

// ── Accessors ───────────────────────────────────────────────────────────────

impl ObjectType {
    /// The raw `u32` tag value.
    #[inline]
    pub const fn to_u32(self) -> u32 {
        self.0
    }

    /// Construct from a raw `u32` tag.  Unknown values are permitted (allows
    /// forward-compatible detection of future tags).
    #[inline]
    pub const fn from_u32(tag: u32) -> Self {
        Self(tag)
    }

    /// A human-readable name for this tag, or `"Unknown"` when the tag is not
    /// known.
    pub fn name(self) -> &'static str {
        match self.0 {
            0x0001 => "Poly",
            0x0002 => "NativePoly",
            0x0003 => "GadgetDecomposition",

            0x0101 => "LweCiphertext",
            0x0102 => "LweSecretKey",
            0x0103 => "LwePublicKey",
            0x0104 => "LweKeyswitchKey",
            0x0105 => "GlweCiphertext",
            0x0106 => "GlweSecretKey",
            0x0107 => "GgswCiphertext",
            0x0108 => "LweBootstrapKey",

            0x0201 => "RlweSecretKey",
            0x0202 => "RlwePublicKey",
            0x0203 => "RlweCiphertext",
            0x0204 => "RlweEvaluationKey",
            0x0205 => "RlweGaloisKey",
            0x0206 => "BfvReKey",
            0x0207 => "BgvCiphertext",

            0x0301 => "HssCiphertext",
            0x0302 => "HssShare",
            0x0303 => "HssEvalKey",

            0x0401 => "ShortintCiphertext",
            0x0402 => "ShortintClientKey",
            0x0403 => "ShortintServerKey",

            0x0501 => "PqcKemPublicKey",
            0x0502 => "PqcKemSecretKeyBlob",
            0x0503 => "PqcKemCiphertext",
            0x0504 => "PqcKemParams",
            0x0505 => "PqcSigVerifyingKey",
            0x0506 => "PqcSigSigningKeyBlob",
            0x0507 => "PqcSigSignature",
            0x0508 => "PqcSigParams",

            0x1001 => "RingParams",
            0x1002 => "RlweParams",
            0x1003 => "BfvParams",
            0x1004 => "HssParams",
            0x1005 => "TfheParams",
            0x1006 => "BgvParams",

            _ => "Unknown",
        }
    }

    /// Returns `true` when this tag corresponds to a known type.
    pub fn is_known(self) -> bool {
        self.name() != "Unknown"
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_known() {
            f.write_str(self.name())
        } else {
            write!(f, "Unknown(0x{:08x})", self.0)
        }
    }
}

impl fmt::Debug for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectType({self})")
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_known_tags_have_names() {
        let tags = [
            ObjectType::POLY,
            ObjectType::NATIVE_POLY,
            ObjectType::GADGET_DECOMPOSITION,
            ObjectType::LWE_CIPHERTEXT,
            ObjectType::LWE_SECRET_KEY,
            ObjectType::LWE_PUBLIC_KEY,
            ObjectType::LWE_KEYSWITCH_KEY,
            ObjectType::GLWE_CIPHERTEXT,
            ObjectType::GLWE_SECRET_KEY,
            ObjectType::GGSW_CIPHERTEXT,
            ObjectType::LWE_BOOTSTRAP_KEY,
            ObjectType::RLWE_SECRET_KEY,
            ObjectType::RLWE_PUBLIC_KEY,
            ObjectType::RLWE_CIPHERTEXT,
            ObjectType::RLWE_EVALUATION_KEY,
            ObjectType::RLWE_GALOIS_KEY,
            ObjectType::BFV_REKEY,
            ObjectType::HSS_CIPHERTEXT,
            ObjectType::HSS_SHARE,
            ObjectType::HSS_EVAL_KEY,
            ObjectType::SHORTINT_CIPHERTEXT,
            ObjectType::SHORTINT_CLIENT_KEY,
            ObjectType::SHORTINT_SERVER_KEY,
            ObjectType::PQC_KEM_PUBLIC_KEY,
            ObjectType::PQC_KEM_SECRET_KEY_BLOB,
            ObjectType::PQC_KEM_CIPHERTEXT,
            ObjectType::PQC_KEM_PARAMS,
            ObjectType::PQC_SIG_VERIFYING_KEY,
            ObjectType::PQC_SIG_SIGNING_KEY_BLOB,
            ObjectType::PQC_SIG_SIGNATURE,
            ObjectType::PQC_SIG_PARAMS,
            ObjectType::RING_PARAMS,
            ObjectType::RLWE_PARAMS,
            ObjectType::BFV_PARAMS,
            ObjectType::HSS_PARAMS,
            ObjectType::TFHE_PARAMS,
            ObjectType::BGV_PARAMS,
        ];
        for tag in tags {
            assert!(tag.is_known(), "tag {tag:?} should be known");
            assert_ne!(tag.name(), "Unknown", "tag {tag:?} has no name");
        }
    }

    #[test]
    fn unknown_tags_are_tolerated() {
        let unknown = ObjectType::from_u32(0xdead);
        assert!(!unknown.is_known());
        assert_eq!(unknown.name(), "Unknown");
        assert_eq!(unknown.to_string(), "Unknown(0x0000dead)");
    }

    #[test]
    fn roundtrip_u32() {
        for tag in [
            ObjectType::POLY,
            ObjectType::RLWE_CIPHERTEXT,
            ObjectType::BFV_PARAMS,
            ObjectType::BGV_PARAMS,
        ] {
            assert_eq!(ObjectType::from_u32(tag.to_u32()), tag);
        }
    }

    #[test]
    fn no_overlapping_tags() {
        let mut seen = std::collections::HashSet::new();
        let tags = [
            ObjectType::POLY.to_u32(),
            ObjectType::NATIVE_POLY.to_u32(),
            ObjectType::GADGET_DECOMPOSITION.to_u32(),
            ObjectType::LWE_CIPHERTEXT.to_u32(),
            ObjectType::LWE_SECRET_KEY.to_u32(),
            ObjectType::LWE_PUBLIC_KEY.to_u32(),
            ObjectType::LWE_KEYSWITCH_KEY.to_u32(),
            ObjectType::GLWE_CIPHERTEXT.to_u32(),
            ObjectType::GLWE_SECRET_KEY.to_u32(),
            ObjectType::GGSW_CIPHERTEXT.to_u32(),
            ObjectType::LWE_BOOTSTRAP_KEY.to_u32(),
            ObjectType::RLWE_SECRET_KEY.to_u32(),
            ObjectType::RLWE_PUBLIC_KEY.to_u32(),
            ObjectType::RLWE_CIPHERTEXT.to_u32(),
            ObjectType::RLWE_EVALUATION_KEY.to_u32(),
            ObjectType::RLWE_GALOIS_KEY.to_u32(),
            ObjectType::BFV_REKEY.to_u32(),
            ObjectType::HSS_CIPHERTEXT.to_u32(),
            ObjectType::HSS_SHARE.to_u32(),
            ObjectType::HSS_EVAL_KEY.to_u32(),
            ObjectType::SHORTINT_CIPHERTEXT.to_u32(),
            ObjectType::SHORTINT_CLIENT_KEY.to_u32(),
            ObjectType::SHORTINT_SERVER_KEY.to_u32(),
            ObjectType::PQC_KEM_PUBLIC_KEY.to_u32(),
            ObjectType::PQC_KEM_SECRET_KEY_BLOB.to_u32(),
            ObjectType::PQC_KEM_CIPHERTEXT.to_u32(),
            ObjectType::PQC_KEM_PARAMS.to_u32(),
            ObjectType::PQC_SIG_VERIFYING_KEY.to_u32(),
            ObjectType::PQC_SIG_SIGNING_KEY_BLOB.to_u32(),
            ObjectType::PQC_SIG_SIGNATURE.to_u32(),
            ObjectType::PQC_SIG_PARAMS.to_u32(),
            ObjectType::RING_PARAMS.to_u32(),
            ObjectType::RLWE_PARAMS.to_u32(),
            ObjectType::BFV_PARAMS.to_u32(),
            ObjectType::HSS_PARAMS.to_u32(),
            ObjectType::TFHE_PARAMS.to_u32(),
            ObjectType::BGV_PARAMS.to_u32(),
        ];
        for tag in tags {
            assert!(seen.insert(tag), "duplicate tag 0x{tag:08x}");
        }
    }
}
