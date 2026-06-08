use crate::core::scheme::HeScheme;
use crate::schemes::bgv::keys::BgvKeyGenerator;
use crate::schemes::bgv::params::BgvParameters;

#[derive(Clone, Debug, Default)]
pub struct BgvScheme;

impl HeScheme for BgvScheme {
    type Context = BgvParameters;
    type KeyGen = BgvKeyGenerator;

    fn name(&self) -> &str {
        "BGV"
    }
}
