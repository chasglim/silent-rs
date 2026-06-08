//! Math backends for SILENT.

pub mod arith;
pub mod backend;
pub mod barrett;
pub mod bigint;
pub mod ct;
pub mod fft64;
pub mod hal;
pub mod mod_arith;
pub mod modulus;
pub mod ntt;
pub mod ntt_traits;
pub mod numth;
pub mod polynomial;
pub mod powerful;
pub mod random;
pub mod rns;
pub mod rns_tool;
pub mod sis;
pub mod torus;
// TODO(ARS24): Matrix-free SIS/Universal hash for succinct VOLE (docs/silent/math.md).
pub mod utils;

#[cfg(test)]
#[path = "tests/rns_tests.rs"]
mod tests;
