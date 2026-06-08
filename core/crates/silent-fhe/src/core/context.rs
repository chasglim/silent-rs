use std::fmt::Debug;

/// Represents the algorithmic context (parameters, moduli, encodings).
pub trait HeContext: Debug + Clone + Send + Sync {
    /// Returns the underlying RLWE encryption parameters used by logic.
    fn security_level(&self) -> u32;
    fn poly_degree(&self) -> usize;
}
