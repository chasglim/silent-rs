//! SILENT inference pipeline composition.
//!
//! The reusable operators come from `silent-operators`; this module exposes
//! the operator set used by the SILENT reproduction binary.

pub use silent_operators::{
    domain::{DomainConverter, HeToShareOutput},
    fixedpoint::FixedPointConfig,
    hss_bridge::context_from_runtime,
    hss_slots::HssSlotEngine,
    linear_map::{LinearMap, LinearMapCrs},
    lookup::{PrivateLookup, PrivateLookupConfig},
    nonlinear::{GeluConfig, LayerNormConfig, NonlinearOps, SoftmaxConfig},
    shares::AdditiveShares,
    truncation::{Truncation, TruncationConfig, TruncationCorrection},
};
