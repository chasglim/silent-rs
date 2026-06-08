use silent_rlwe::Ciphertext;

/// Naive TFHE -> BFV repacking key.
///
/// Each entry encrypts one TFHE secret-key coefficient as a BFV constant
/// polynomial. This is large, but it is a genuine homomorphic baseline and
/// does not rely on trusted decryption.
#[derive(Clone, Debug, Default)]
pub struct TfheToBfvRepackKey {
    pub rk: Vec<Ciphertext>,
}
