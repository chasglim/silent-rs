#![forbid(unsafe_code)]

//! Reusable encrypted operators for SILENT protocol.
//!
//! The modules here are protocol-agnostic building blocks: modular linear
//! algebra, private lookup, fixed-point nonlinear layers, HE/share conversion,
//! and HSS-backed slot operations. Paper-specific pipelines live in examples.

pub mod ahss;
pub mod baselines;
pub mod bks_ahss;
pub mod coefficient_cert;
pub mod comparison;
pub mod conversion;
pub mod direct_reload;
pub mod domain;
pub mod error;
pub mod fixedpoint;
pub mod hat;
pub mod he_bridge;
pub mod hss_bridge;
pub mod hss_slots;
pub mod linear_bridge;
pub mod linear_map;
pub mod lookup;
pub mod modular;
pub mod nonlinear;
pub mod packed_linear_map;
pub mod shares;
pub mod torus_kernels;
pub mod transformer_pipeline;
pub mod truncation;

pub use error::OperatorError;

pub mod prelude {
    pub use crate::ahss;
    pub use crate::baselines;
    pub use crate::bks_ahss;
    pub use crate::coefficient_cert;
    pub use crate::comparison::{Comparison, PublicComparison};
    pub use crate::conversion::{QToPMaskedOpenState, ShareConverter};
    pub use crate::direct_reload;
    pub use crate::domain::{DomainConverter, HeToShareOutput};
    pub use crate::error::OperatorError;
    pub use crate::fixedpoint::FixedPointConfig;
    pub use crate::hat;
    pub use crate::he_bridge;
    pub use crate::hss_slots::HssSlotEngine;
    pub use crate::linear_bridge::{LinearMapQShares, PrivateLinearMap, PrivateLinearMapConfig};
    pub use crate::linear_map::{LinearMap, LinearMapCrs};
    pub use crate::lookup::{PrivateLookup, PrivateLookupConfig, PrivateLookupCrsCache};
    pub use crate::modular;
    pub use crate::nonlinear::{
        BinadeLayerNormConfig, CadscMultiPlan, DyadicOfPmpeLayerNormRsqrtOutput,
        DyadicOfPmpeLayerNormSquareOutput, DyadicOfPmpeOutput, DyadicOfPmpePolynomialConfig,
        DyadicOfPmpeSoftmaxExpOutput, DyadicOfPmpeSoftmaxOutput,
        DyadicOfPmpeSoftmaxReciprocalOutput, DyadicPolynomialGridAudit, DyadicRescaleOutput,
        DyadicScaleProfile, DyadicScaledMulOutput, FastGeluConfig, GeluConfig, LayerNormConfig,
        NoWrapAudit, NonlinearOps, NonlinearProfile, OfPmpeLayerNormSquareOutput, OfPmpeMaskedOpen,
        OfPmpeOfflineMode, OfPmpeOutput, OfPmpePolynomialConfig, OfPmpeProfile,
        OfPmpeSoftmaxOutput, OfPmpeTaylorPack, PackedRlweAheTripleEngine,
        PackedRlweAheTripleProfile, ProfiledShares, RangeSoftmaxConfig, RnsBeaverMulPack,
        RnsCorrelationSource, RnsCrossTermOleTaylorSource, RnsDyadicDomain, RnsDyadicOfPmpeOutput,
        RnsDyadicRescalePack, RnsDyadicTaylorPack, RnsHssCorrelationSource,
        RnsHybridEngineeringCorrelationSource, RnsIdealCorrelationSource,
        RnsMeanShiftSoftmaxExpOutput, RnsPreprocessPlan, RnsRlweAheCorrelationSource,
        RnsRlweAheCrossTermOleTaylorSource, RnsScaledMulOutput, RnsScaledShareTensor,
        RnsShareTensor, RnsTrustedDebugCorrelationSource, ScaleSemantic, ScaledShareTensor,
        SoftmaxConfig,
    };
    pub use crate::packed_linear_map::{
        PackedLinearMap, PackedLinearMapConfig, PackedLinearMapCrs, PackedLinearMapCrsCache,
        PackedLinearMapServerSetup,
    };
    pub use crate::shares::AdditiveShares;
    pub use crate::torus_kernels;
    pub use crate::transformer_pipeline;
    pub use crate::truncation::{Truncation, TruncationConfig, TruncationCorrection};
}
