//! Canonical parameter definitions, validation, presets, and identifiers for
//! the SILENT core stack.

pub mod audit;
pub mod distribution;
pub mod error;
pub mod fhe;
pub mod hss;
pub mod ids;
pub mod io_impls;
pub mod newtypes;
pub mod pqc;
pub mod presets;
pub mod recommend;
pub mod registry;
pub mod ring;
pub mod rlwe;
pub mod scheme;
pub mod security;
pub mod standard;
pub mod tfhe;

pub use audit::{
    BgvAuditCode, BgvAuditFinding, BgvAuditReport, BgvAuditSeverity, BgvPresetAuditReport,
    audit_bgv_params, audit_bgv_presets,
};
pub use distribution::DistributionType;
pub use error::ParamError;
pub use fhe::{BfvParams, BgvParams, CkksParams};
pub use hss::HssParams;
pub use ids::{ParameterSet, ParamsId};
pub use newtypes::{
    CarryModulus, CiphertextModulusLog, CorrectnessMarginBits, DecompositionBaseLog,
    DecompositionLevelCount, EncryptionKeyChoice, GlweDimension, Log2PFail, LogN, LweDimension,
    MaxLinearTerms, MaxRmultDepth, MessageModulus, ModulusBits, MultiplicativeDepth,
    NoiseDistribution, PlaintextModulus, PolynomialSize, ReconstructionBoundBits, RingDim,
    ScaleBits, ShareModulusBits,
};
pub use pqc::{PqcKemParams, PqcSigParams};
pub use registry::{RegisteredParameterSet, all_presets, find_preset};
pub use ring::RingParams;
pub use rlwe::{GeneratedModuli, RlweParams};
pub use scheme::SchemeFamily;
pub use security::SecurityLevel;
pub use tfhe::TfheParams;

pub const CRATE_ID: &str = "silent-params";
