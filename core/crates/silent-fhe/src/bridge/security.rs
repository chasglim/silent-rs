//! Security definitions and assumptions for cross-scheme evaluations.
//!
//! This bridge is deliberately fail-closed. A conversion path may be exposed as
//! public API only after its algebraic scaling, phase convention, key-switching
//! relation, and noise budget are checked in code. Test-only trusted oracles are
//! kept behind `cfg(test)` and must not be used to satisfy public conversions.
//!
//! BFV -> TFHE has two separate plaintext contracts:
//! - `bfv_to_tfhe` returns SILENT TFHE integer ciphertexts and therefore
//!   requires the BFV plaintext modulus to equal the padded TFHE plaintext
//!   modulus `2 * message_modulus * carry_modulus`.
//! - `bfv_to_tfhe_torus_native` follows the SILENT B/FV coefficient contract:
//!   a BFV coefficient `z in Z_p` is interpreted as `z/p in T`.
//! - `bfv_slots_to_tfhe_torus_native` applies the BFV-to-TFHE torus-native switching public
//!   VDM linear map for BFV SIMD slots under the same torus-native `z/p`
//!   plaintext contract.
//!
//! TFHE -> BFV exposes exact zero-noise LWE-dimension-1 backends and a
//! TFHE-to-BFV functional switching toy/audit backend. The SILENT path first uses TFHE PBS
//! to map SILENT TFHE integer messages onto exact `1/p` torus multiples, then
//! applies `KSi = TRGSW_K(S_i)` functional switching with the public
//! `VDM^{-1} * G` transform and modulus-switches the target TRLWE into BFV.
//! This is a correctness implementation for toy/full-decomposition parameters;
//! production noisy parameters still require a reviewed concrete noise proof.
//!
//! The exposed `tfhe_to_bfv_phase_coefficients` helper stops after the real
//! homomorphic phase-repacking stage. It is not a decoded-message conversion:
//! downstream code must still apply a reviewed TorusDecode/Fmod step.
//!
//! The current BFV-Fmod helper is a center-only exact backend. It decodes
//! phases of the form `Δ_T * m` under strict prime-plaintext-modulus
//! parameters; it is not an interval decoder for noisy TFHE phases.
