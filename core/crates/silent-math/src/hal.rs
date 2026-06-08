//! Hardware abstraction layer facade.

pub use crate::backend::{MathBackend, NativeBackend};

#[cfg(feature = "hexl")]
pub use crate::backend::HexlBackend;
