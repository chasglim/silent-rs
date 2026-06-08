use silent_net::frame::FrameError;

use crate::ids::{PartyId, RuntimeRoute};

/// Runtime errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidConfig(&'static str),
    InvalidValue(&'static str),
    DuplicateOperator(&'static str),
    UnknownOperator(String),
    UnknownSymbol(String),
    UnknownPeer(PartyId),
    RouteMismatch {
        expected: RuntimeRoute,
        actual: RuntimeRoute,
    },
    Codec(String),
    Operator(String),
    Network(String),
}

impl core::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid runtime config: {msg}"),
            Self::InvalidValue(msg) => write!(f, "invalid runtime value: {msg}"),
            Self::DuplicateOperator(name) => write!(f, "duplicate operator: {name}"),
            Self::UnknownOperator(name) => write!(f, "unknown operator: {name}"),
            Self::UnknownSymbol(name) => write!(f, "unknown symbol: {name}"),
            Self::UnknownPeer(peer) => write!(f, "unknown peer: {:?}", peer),
            Self::RouteMismatch { expected, actual } => {
                write!(
                    f,
                    "runtime route mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            Self::Codec(msg) => write!(f, "runtime codec error: {msg}"),
            Self::Operator(msg) => write!(f, "operator error: {msg}"),
            Self::Network(msg) => write!(f, "network error: {msg}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<FrameError> for RuntimeError {
    fn from(value: FrameError) -> Self {
        Self::Network(value.to_string())
    }
}

impl From<silent_io::IoError> for RuntimeError {
    fn from(value: silent_io::IoError) -> Self {
        Self::Codec(value.to_string())
    }
}
