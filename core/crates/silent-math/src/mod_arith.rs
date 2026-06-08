//! Modular arithmetic trait abstractions.
//!
//! Generic traits for modular arithmetic operations that both
//! ML-KEM and ML-DSA can implement against their respective
//! field moduli.

/// Trait for modular reduction in a specific field.
pub trait Reduction<const Q: u32> {
    /// The value type (e.g., `i16` for q=3329, `i32` for q=8380417).
    type Value;

    /// Reduce `value` modulo Q.
    fn reduce(value: Self::Value) -> Self::Value;

    /// Centered reduction: map `value` into [-Q/2, Q/2).
    fn reduce_centered(value: Self::Value) -> Self::Value;
}

/// Trait for Montgomery-domain arithmetic.
pub trait MontgomeryDomain<const Q: u32> {
    /// The base value type.
    type Value;

    /// The Montgomery-domain value type.
    type MontgomeryValue;

    /// Convert a plain value into Montgomery domain.
    fn to_montgomery(value: Self::Value) -> Self::MontgomeryValue;

    /// Convert a Montgomery-domain value back to plain.
    fn from_montgomery(value: &Self::MontgomeryValue) -> Self::Value;

    /// Montgomery multiply: `(a * b) * R^(-1) mod Q`.
    fn montgomery_multiply(
        a: &Self::MontgomeryValue,
        b: &Self::MontgomeryValue,
    ) -> Self::MontgomeryValue;
}
