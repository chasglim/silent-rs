//! Marker type and [`HeScheme`] impl for the TFHE family.

use crate::core::scheme::HeScheme;

use super::keys::TfheKeyGenerator;
use super::params::TfheParameters;

#[derive(Clone, Copy, Debug, Default)]
pub struct TfheScheme;

impl TfheScheme {
    pub const NAME: &'static str = "TFHE";
}

impl HeScheme for TfheScheme {
    type Context = TfheParameters;
    type KeyGen = TfheKeyGenerator;

    fn name(&self) -> &str {
        Self::NAME
    }
}
