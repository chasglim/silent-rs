use crate::core::context::HeContext;
use crate::core::keys::KeyGenerator;

/// A generic trait representing an FHE scheme (e.g. BFV, CKKS).
/// It acts as a factory for Contexts and other high level objects.
pub trait HeScheme {
    type Context: HeContext;
    type KeyGen: KeyGenerator;

    fn name(&self) -> &str;
}
