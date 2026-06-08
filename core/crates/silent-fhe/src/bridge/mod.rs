//! BFV <-> TFHE bridge.
//!
//! The currently supported public BFV -> TFHE paths are coefficient-layout
//! conversion into SILENT TFHE's native integer encoding, and torus-native
//! torus-native coefficient or slot conversion where `z in Z_p` is interpreted
//! as `z/p in T`. TFHE -> BFV exposes exact LWE-dimension-1 baselines and a
//! TFHE-to-BFV functional switching toy/audit path: PBS normalization to exact `1/p`
//! multiples, TRGSW functional switching, and BFV torus-native slot output.
//! Production noisy parameters still require a reviewed concrete noise proof.

pub mod encoding;
pub mod error;
pub mod noise;
pub mod params;
pub mod security;

pub mod bfv_to_tfhe;
pub mod keys;
pub mod tfhe_ops;
pub mod tfhe_to_bfv;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub use bfv_to_tfhe::bfv_to_tfhe_trusted_oracle;
pub use bfv_to_tfhe::{
    BfvToTfheConverter, bfv_slots_to_tfhe_torus_native, bfv_to_tfhe, bfv_to_tfhe_torus_native,
};
pub use error::{BridgeError, BridgeResult};
pub use keys::{BridgeKeyGenerator, BridgeKeys};
pub use params::{BridgeLayout, BridgeParams};
pub use tfhe_ops::{BridgeOpsError, eval_identity, eval_relu};
#[cfg(test)]
pub use tfhe_to_bfv::tfhe_to_bfv_trusted_oracle;
pub use tfhe_to_bfv::{
    TfheToBfvConverter, TfheToBfvFunctionalKey, bfv_phase_to_bfv_messages,
    build_torus_native_normalization_lut, ensure_full_tfhe_to_bfv_preconditions,
    ensure_torus_native_tfhe_to_bfv_preconditions, generate_tfhe_to_bfv_functional_key,
    normalize_lwe_to_bfv_torus_native_with_pbs, tfhe_to_bfv, tfhe_to_bfv_functional_coefficients,
    tfhe_to_bfv_functional_slots, tfhe_to_bfv_phase_coefficients,
    tfhe_to_bfv_torus_native_lwe1_exact, tfhe_to_bfv_with_functional_coefficient_map,
};
