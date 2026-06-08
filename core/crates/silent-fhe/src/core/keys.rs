//! Generic key-generator interfaces.
//!
//! `KeyGenerator` (the base trait) only requires the ability to sample a
//! *secret* key.  `PublicKeyGen` extends it for schemes that additionally
//! support an asymmetric public-key encryption interface (BFV, CKKS, …); TFHE
//! does **not** implement it for the MVP because the M5 path is symmetric
//! only.
//!
//! The two-trait split keeps the `HeScheme::KeyGen` bound generic over both
//! scheme families: TFHE supplies an `LweSecretKey`, BFV supplies the
//! existing `silent_rlwe::SecretKey`.

/// Sample fresh secret keys for an FHE scheme.
pub trait KeyGenerator {
    /// Concrete secret-key type returned by [`Self::generate_secret_key`].
    type SecretKey;

    /// Sample a fresh secret key.
    fn generate_secret_key(&mut self) -> Self::SecretKey;
}

/// Optional extension trait for schemes that expose asymmetric public-key
/// encryption.
pub trait PublicKeyGen: KeyGenerator {
    /// Concrete public-key type returned by [`Self::generate_public_key`].
    type PublicKey;

    /// Derive a public key from a secret key.
    fn generate_public_key(&mut self, sk: &Self::SecretKey) -> Self::PublicKey;
}

/// Marker trait for evaluation keys if we abstract them later.
pub trait EvalKey: Clone + Send + Sync {}
