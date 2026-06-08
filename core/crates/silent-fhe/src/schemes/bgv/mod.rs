//! BGV over the existing SILENT RLWE/RNS stack.
//!
//! This module uses LSB plaintext embedding: fresh ciphertexts decrypt to
//! `m + t * e`, so decryption reduces the centered phase modulo `t`. Products
//! keep their degree-grown ciphertext shape and are decrypted by the generic
//! secret-key power inner product. The evaluator also provides plaintext
//! addition/subtraction/multiplication and gadget-based relinearization and
//! Galois rotation over the shared SILENT RLWE/RNS layer.

pub mod bootstrap;
pub mod ciphertext;
pub mod crypto;
pub mod encoding;
pub mod keys;
pub mod ops;
pub mod params;
mod rns_big;
pub mod scheme;
