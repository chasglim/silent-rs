//! Traits for Number Theoretic Transform domains.
//!
//! Provides abstraction over NTT operations so that both ML-KEM
//! (q=3329) and ML-DSA (q=8380417) can reuse the same interface
//! while implementing their own parameter-specific logic.

/// A type that can be forward-NTT transformed.
///
/// Implemented by polynomials and polynomial vectors in ML-KEM / ML-DSA.
pub trait NttForward {
    /// The element type after forward NTT.
    type NttOutput;

    /// Perform the forward NTT transform on `self`.
    fn ntt_forward(&self) -> Self::NttOutput;
}

/// A type that can be inverse-NTT transformed.
pub trait NttInverse {
    /// The element type after inverse NTT.
    type NormalOutput;

    /// Perform the inverse NTT transform on `self`.
    fn ntt_inverse(&self) -> Self::NormalOutput;
}

/// Marker trait for types that support in-place NTT operations.
pub trait NttDomain: NttForward + NttInverse {
    /// The modulus Q of the NTT domain.
    const Q: u32;

    /// The polynomial ring degree N (power of two).
    const N: usize;

    /// Check that `self` is in the Montgomery domain.
    fn is_in_montgomery(&self) -> bool;
}

/// A type that supports Montgomery-domain multiplication.
pub trait MontgomeryMul {
    /// Multiply two NTT-domain elements.
    fn montgomery_mul(&self, rhs: &Self) -> Self;
}

/// A type that can be scaled by a constant (for inverse NTT).
pub trait NttScale {
    /// Scale by the inverse of the ring degree.
    fn ntt_scale(&mut self);
}
