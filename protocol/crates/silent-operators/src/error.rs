use core::fmt;

use crate::modular::ModularError;
use silent_math::rns::RnsError;
use silent_params::ParamError;

#[derive(Debug)]
pub enum OperatorError {
    InvalidParams(&'static str),
    Protocol(&'static str),
    Backend(String),
    Modular(ModularError),
    Param(ParamError),
    Rns(RnsError),
}

impl fmt::Display for OperatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParams(msg) => write!(f, "invalid operator parameters: {msg}"),
            Self::Protocol(msg) => write!(f, "operator protocol error: {msg}"),
            Self::Backend(msg) => write!(f, "operator backend error: {msg}"),
            Self::Modular(err) => write!(f, "modular primitive error: {err}"),
            Self::Param(err) => write!(f, "parameter error: {err}"),
            Self::Rns(err) => write!(f, "RNS error: {err:?}"),
        }
    }
}

impl std::error::Error for OperatorError {}

impl From<ModularError> for OperatorError {
    fn from(value: ModularError) -> Self {
        Self::Modular(value)
    }
}

impl From<RnsError> for OperatorError {
    fn from(value: RnsError) -> Self {
        Self::Rns(value)
    }
}

impl From<ParamError> for OperatorError {
    fn from(value: ParamError) -> Self {
        Self::Param(value)
    }
}
