//! Backend selection for math acceleration.

use crate::modulus::Modulus;
use crate::ntt::NttTables;

pub trait MathBackend {
    fn mod_add(&self, lhs: &mut [u64], rhs: &[u64], modulus: &Modulus);
    fn mod_sub(&self, lhs: &mut [u64], rhs: &[u64], modulus: &Modulus);
    fn mod_mul(&self, lhs: &mut [u64], rhs: &[u64], modulus: &Modulus);
    fn ntt_forward(&self, values: &mut [u64], tables: &NttTables);
    fn ntt_inverse(&self, values: &mut [u64], tables: &NttTables);
}

pub mod simd;
pub use simd::NativeBackend;

#[cfg(feature = "hexl")]
pub mod hexl;
#[cfg(feature = "hexl")]
pub use hexl::HexlBackend;
