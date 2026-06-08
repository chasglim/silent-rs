//! Network harnesses for local protocol testing and secure channels.

pub mod config;
pub mod frame;
#[cfg(feature = "crypto-tests")]
pub mod integration;
pub mod routing;
pub mod serialize;
pub mod transport;
pub mod typed;

pub const CRATE_ID: &str = "silent-net";
