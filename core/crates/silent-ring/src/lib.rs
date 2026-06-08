//! Ring arithmetic for SILENT.

pub mod context;
pub mod gadget;
pub mod io_impls;
pub mod native_poly;
pub mod poly;

pub use context::RingContext;
pub use gadget::GadgetDecomposition;
pub use native_poly::NativePoly;
pub use poly::{Poly, PolyShoup};
