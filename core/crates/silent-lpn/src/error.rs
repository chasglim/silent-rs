use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LpnError {
    InvalidModulus(u64),
    InvalidParams(&'static str),
    DecodeFailure(&'static str),
    DimensionMismatch {
        lhs: (usize, usize),
        rhs: (usize, usize),
        context: &'static str,
    },
    VectorLengthMismatch {
        lhs: usize,
        rhs: usize,
        context: &'static str,
    },
    IndexOutOfBounds {
        index: usize,
        upper_bound: usize,
        context: &'static str,
    },
}

impl fmt::Display for LpnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModulus(m) => write!(f, "invalid modulus: {m}"),
            Self::InvalidParams(msg) => write!(f, "invalid parameters: {msg}"),
            Self::DecodeFailure(msg) => write!(f, "decode failure: {msg}"),
            Self::DimensionMismatch { lhs, rhs, context } => write!(
                f,
                "dimension mismatch in {context}: lhs=({},{}) rhs=({},{})",
                lhs.0, lhs.1, rhs.0, rhs.1
            ),
            Self::VectorLengthMismatch { lhs, rhs, context } => {
                write!(
                    f,
                    "vector length mismatch in {context}: lhs={lhs} rhs={rhs}"
                )
            }
            Self::IndexOutOfBounds {
                index,
                upper_bound,
                context,
            } => write!(
                f,
                "index out of bounds in {context}: index={index}, upper_bound={upper_bound}"
            ),
        }
    }
}

impl std::error::Error for LpnError {}
