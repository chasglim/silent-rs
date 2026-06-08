//! Serialization backends gated behind optional features.
//!
//! ## Architecture
//!
//! | Path         | Feature          | Used for                         |
//! |------------- |------------------|----------------------------------|
//! | `proto`      | `proto`          | Control-plane handshake only     |
//! | `flat`       | `flatbuffers`    | Deferred (no auto-switching)     |
//!
//! # Design Decision
//!
//! **Crypto payloads use `silent-io` canonical bytes exclusively.**
//! No other codec path — Protobuf, FlatBuffers, bincode, or serde — is
//! allowed on the cryptographic data plane.  The [`crate::frame::NetworkFrame`]
//! carries all routing metadata; the frame payload is always canonical.
//!
//! Protobuf is acceptable for operator-facing control messages (handshake,
//! discovery, health) and is feature-gated behind `proto`.

pub mod flat;
#[cfg(feature = "proto")]
pub mod proto;

// Every `silent-net` serialization decision is documented in plan.md.
// Do not add second codec paths for crypto payloads without profiling
// evidence and an explicit revision of the plan.
