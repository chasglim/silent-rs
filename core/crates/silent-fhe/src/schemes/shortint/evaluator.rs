//! High-level handle for server-side shortint homomorphic ops (add, mul, LUT PBS).
//!
//! Matches the mental model of [`crate::schemes::bfv::ops::BfvEvaluator`] + secret key:
//! keep a [`ShortintClientKey`] for encrypt/decrypt and a [`ShortintEvaluator`]
//! (wrapping [`super::ShortintServerKey`]) for homomorphic arithmetic.

use std::ops::{Deref, DerefMut};

use super::server_key::ShortintServerKey;

/// Server-side evaluator: **add**, **mul**, comparisons, LUT PBS, carry extract, etc.
///
/// All [`ShortintServerKey`] methods are available via `Deref` (including `mul`,
/// `apply_lookup_table`, `generate_lookup_table`, …).
#[derive(Clone)]
pub struct ShortintEvaluator {
    server: ShortintServerKey,
}

impl ShortintEvaluator {
    pub fn new(server: ShortintServerKey) -> Self {
        Self { server }
    }

    pub fn into_server_key(self) -> ShortintServerKey {
        self.server
    }
}

impl Deref for ShortintEvaluator {
    type Target = ShortintServerKey;

    fn deref(&self) -> &Self::Target {
        &self.server
    }
}

impl DerefMut for ShortintEvaluator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.server
    }
}
