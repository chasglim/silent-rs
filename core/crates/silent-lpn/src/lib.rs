#![forbid(unsafe_code)]

//! LPN-native building blocks for code-reconciled BKS-style HSS.
//!
//! The crate intentionally keeps networking and operator-specific matrix types
//! out of the core. protocol adapters convert these slice-based kernels to
//! `ModMatrix` and other higher-level operator APIs.

pub mod code;
pub mod error;
pub mod field;
pub mod hss;
pub mod linalg;
pub mod lpn;
pub mod sampler;

pub use code::{LinearCode, PlainCrscOutput, ReedSolomonCode};
pub use error::LpnError;
pub use hss::{
    LpnBksClientDigest, LpnBksHssParams, LpnBksHssPublic, LpnBksHssShareOutput,
    LpnBksPlainCrscServerOutput, LpnBksServerEncoding, LpnBksServerEncodingOutput,
    LpnBksServerPreprocessing, NoisyBilinearOutput, rho_q_error_rate,
};
pub use linalg::FieldMatrix;
pub use lpn::{LpnParams, LpnPublic};
pub use sampler::{SparseVector, sample_bernoulli_error, sample_exact_weight_error};

pub mod prelude {
    pub use crate::{
        FieldMatrix, LinearCode, LpnBksClientDigest, LpnBksHssParams, LpnBksHssPublic,
        LpnBksHssShareOutput, LpnBksPlainCrscServerOutput, LpnBksServerEncoding,
        LpnBksServerEncodingOutput, LpnBksServerPreprocessing, LpnError, LpnParams, LpnPublic,
        PlainCrscOutput, ReedSolomonCode, SparseVector, sample_bernoulli_error,
        sample_exact_weight_error,
    };
}
