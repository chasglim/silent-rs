use crate::error::IoError;
use crate::object_type::ObjectType;

/// Serialize `self` into a canonical byte representation.
///
/// The encoding is **deterministic**: calling `encode_to` twice on the same
/// logical object must produce identical bytes (assuming no internal RNG or
/// similarly non-deterministic state in the object itself).
///
/// # Implementor's contract
///
/// - `encode_to` must report exactly `encoded_len()` bytes written.
/// - The encoding must be self-contained — a reader with the same parameter
///   set (when applicable) must be able to reconstruct the object via
///   [`DecodeWithParams`].
pub trait CanonicalEncode {
    /// Number of bytes the canonical encoding of `self` occupies.
    fn encoded_len(&self) -> usize;

    /// Write the canonical encoding of `self` to `writer`.
    ///
    /// Returns the number of bytes written (must equal [`encoded_len`]).
    fn encode_to<W: std::io::Write>(&self, writer: &mut W) -> Result<usize, IoError>;

    /// Convenience: encode into a freshly allocated `Vec<u8>`.
    fn encode_to_vec(&self) -> Result<Vec<u8>, IoError> {
        let mut buf = Vec::with_capacity(self.encoded_len());
        self.encode_to(&mut buf)?;
        Ok(buf)
    }
}

/// Deserialize an object from its canonical byte representation **without**
/// any external parameter context.
///
/// # Safety boundary
///
/// This trait is **only** for types that carry their own complete validation
/// context: raw integers, fixed-size arrays, parameter descriptors, metadata.
///
/// **Crypto objects that depend on modulus / degree / modulus chain MUST NOT
/// implement `CanonicalDecode`.**  Use [`DecodeWithParams`] instead.  Violating
/// this rule opens the door to "decode first, validate parameters later" bugs
/// where a ciphertext generated under one parameter set is silently interpreted
/// under another.
pub trait CanonicalDecode: Sized {
    /// Read and decode `Self` from `reader`.
    ///
    /// The reader must be positioned at the start of the encoded object.
    fn decode_from<R: std::io::Read>(reader: &mut R) -> Result<Self, IoError>;
}

/// Deserialize an object from its canonical byte representation **with**
/// external parameter context.
///
/// All modulus-dependent, degree-dependent, and modulus-chain-dependent
/// cryptographic types (e.g. `Poly`, `LweCiphertext`, `Ciphertext`,
/// `EvaluationKey`, `HssShare`) decode exclusively through this trait.
///
/// The `params` reference provides the validation context needed to:
/// - Check that `degree` matches
/// - Check that `num_moduli` matches the modulus chain length
/// - Verify every coefficient is strictly less than its corresponding modulus
/// - Reject any structural mismatch at decode time (fail-fast)
pub trait DecodeWithParams<P>: Sized {
    /// Read and decode `Self` from `reader`, validating against `params`.
    fn decode_with_params<R: std::io::Read>(params: &P, reader: &mut R) -> Result<Self, IoError>;
}

/// Static metadata attached to every persistable SILENT type.
///
/// This trait carries **type-level** metadata only.  Runtime context such as
/// `scheme_id` or `params_id` is provided separately via [`EncodeContext`]
/// when writing a container.
pub trait ObjectKind {
    /// The [`ObjectType`] tag identifying this type in a container header.
    const OBJECT_TYPE: ObjectType;

    /// The serialization format version of this type.  Incremented when the
    /// canonical encoding layout changes in a backward-incompatible way.
    const SERIALIZED_VERSION: u16;
}

/// A type that carries a 256-bit parameter-set identifier.
///
/// Implement this for parameter types passed to [`DecodeWithParams`] so that
/// the container reader can verify the header's `params_id` matches the
/// caller's parameter set.
pub trait HasParamsId {
    /// The 256-bit parameter-set identifier for this parameter set.
    fn params_id(&self) -> crate::ParamsId;
}

/// Runtime context supplied by the caller when writing a container or frame.
///
/// This separates type-level metadata ([`ObjectKind`]) from session-level
/// choices (scheme, parameter set, flags).
#[derive(Clone, Debug)]
pub struct EncodeContext {
    /// Identifier for the cryptographic scheme (e.g. BFV, TFHE, HSS).
    /// Interpreted as a `SchemeFamily` discriminant by `silent-params`.
    pub scheme_id: u16,

    /// The 256-bit parameter-set identifier.  Use [`crate::ParamsId::ZERO`]
    /// when no parameter binding is applicable.
    pub params_id: crate::ParamsId,

    /// Bitfield of encoding flags.
    ///
    /// | Bit | Meaning              |
    /// |-----|----------------------|
    /// | 0   | `is_ntt`             |
    /// | 1   | `is_seeded`          |
    /// | 2   | reserved             |
    /// | 3-7 | reserved             |
    pub flags: u8,
}

impl Default for EncodeContext {
    fn default() -> Self {
        Self {
            scheme_id: 0,
            params_id: crate::ParamsId::ZERO,
            flags: 0,
        }
    }
}
