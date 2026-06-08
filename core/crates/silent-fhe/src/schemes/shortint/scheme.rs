//! [`HeScheme`] marker for the shortint surface (TFHE parameters + [`super::ShortintKeyGenerator`]).

use crate::core::scheme::HeScheme;
use crate::schemes::tfhe::params::TfheParameters;

use super::keygen::ShortintKeyGenerator;

/// Shortint as a first-class scheme next to [`crate::schemes::bfv::scheme::BfvScheme`]:
/// same [`TfheParameters`] / [`HeContext`] as TFHE, with keygen that yields both client and server keys.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShortintScheme;

impl HeScheme for ShortintScheme {
    type Context = TfheParameters;
    type KeyGen = ShortintKeyGenerator;

    fn name(&self) -> &str {
        "TFHE-shortint"
    }
}
