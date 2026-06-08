use crate::core::context::HeContext;
use silent_rlwe::Plaintext;

/// Abstract interface for encoding/decoding messages.
pub trait HeEncoder {
    type Context: HeContext;

    fn new(context: Self::Context) -> Self;

    /// Encodes a vector of values into a Plaintext.
    /// Interpretation of values depends on scheme (e.g. SIMD slots for BFV).
    fn encode(&self, values: &[u64]) -> Plaintext;

    /// Decodes a Plaintext into a vector of values.
    fn decode(&self, plain: &Plaintext) -> Vec<u64>;
}
