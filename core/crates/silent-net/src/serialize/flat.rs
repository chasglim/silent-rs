//! FlatBuffers support — deferred.
//!
//! Per plan.md: "FlatBuffers: defer until there is a measured bottleneck and a
//! real zero-copy object layout."
//!
//! There is **no** auto-switching between FlatBuffers and canonical encoding
//! for crypto payloads.  When FlatBuffers is reintroduced, it must come with
//! a benchmark proving it outperforms canonical bytes in a real SILENT
//! workload.
