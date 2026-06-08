use crate::core::scheme::HeScheme;
use crate::schemes::bfv::keys::BfvKeyGenerator;
use crate::schemes::bfv::params::BfvParameters;

pub struct BfvScheme;

impl HeScheme for BfvScheme {
    type Context = BfvParameters;
    type KeyGen = BfvKeyGenerator;

    fn name(&self) -> &str {
        "BFV"
    }
}
