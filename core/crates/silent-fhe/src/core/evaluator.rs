use crate::core::context::HeContext;
use silent_rlwe::Ciphertext;

// For now, let's assume we operate on silent_rlwe::Ciphertext directly
// or a wrapper. To keep it zero-cost, we might operate on references.

/// Defines standard homomorphic operations.
/// Usage: `evaluator.add(&ct1, &ct2)`
pub trait HeEvaluator {
    type Context: HeContext;

    fn new(context: Self::Context) -> Self;

    fn add(&self, c1: &Ciphertext, c2: &Ciphertext) -> Ciphertext;

    fn sub(&self, c1: &Ciphertext, c2: &Ciphertext) -> Ciphertext;

    fn mul(&self, c1: &Ciphertext, c2: &Ciphertext) -> Ciphertext;

    // fn relinearize(&self, ct: &mut Ciphertext, rk: &RelinKeys);
}
