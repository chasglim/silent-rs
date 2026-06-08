//! Shortint-style LWE ciphertext with explicit degree / noise metadata for PBS.

use silent_rlwe::LweCiphertext;

use super::error::ShortintError;
use super::types::{Degree, NoiseLevel};
use crate::schemes::tfhe::params::TfheParameters;

/// One shortint block: an LWE ciphertext plus homomorphic sanity metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShortintCiphertext {
    inner: LweCiphertext,
    plaintext_degree: Degree,
    noise: NoiseLevel,
    mm: u64,
    cm: u64,
}

impl ShortintCiphertext {
    /// Reconstruct from decoded components (used by `io_impls`).
    pub fn from_parts(
        inner: LweCiphertext,
        degree: Degree,
        noise: NoiseLevel,
        mm: u64,
        cm: u64,
    ) -> Self {
        Self {
            inner,
            plaintext_degree: degree,
            noise,
            mm,
            cm,
        }
    }

    #[inline]
    pub fn lwe_ciphertext(&self) -> &LweCiphertext {
        &self.inner
    }

    #[inline]
    pub fn degree(&self) -> Degree {
        self.plaintext_degree
    }

    #[inline]
    pub fn noise_level(&self) -> NoiseLevel {
        self.noise
    }

    #[inline]
    pub fn message_modulus(&self) -> u64 {
        self.mm
    }

    #[inline]
    pub fn carry_modulus(&self) -> u64 {
        self.cm
    }

    /// `true` when the carry buffer is unused (`degree < message_modulus`).
    #[inline]
    pub fn carry_is_empty(&self) -> bool {
        self.plaintext_degree.get() < self.mm
    }

    /// Fresh encryption: degree and noise reflect a single clean message slot.
    pub(crate) fn new_fresh(inner: LweCiphertext, params: &TfheParameters) -> Self {
        let mm = params.message_modulus().0;
        let cm = params.carry_modulus().0;
        Self {
            inner,
            plaintext_degree: Degree::new(mm.saturating_sub(1)),
            noise: NoiseLevel::NOMINAL,
            mm,
            cm,
        }
    }

    pub(crate) fn ensure_compatible(
        &self,
        other: &ShortintCiphertext,
    ) -> Result<(), ShortintError> {
        if self.mm != other.mm
            || self.cm != other.cm
            || self.inner.dimension().0 != other.inner.dimension().0
            || self.inner.log_modulus() != other.inner.log_modulus()
        {
            return Err(ShortintError::ParameterMismatch);
        }
        Ok(())
    }

    pub(crate) fn ensure_params(&self, params: &TfheParameters) -> Result<(), ShortintError> {
        if self.mm != params.message_modulus().0
            || self.cm != params.carry_modulus().0
            || self.inner.dimension().0 != params.lwe_dimension().0
            || self.inner.log_modulus() != params.ciphertext_modulus_log().0
        {
            return Err(ShortintError::ParameterMismatch);
        }
        Ok(())
    }

    pub(crate) fn noise_degree(&self) -> super::types::CiphertextNoiseDegree {
        super::types::CiphertextNoiseDegree {
            degree: self.plaintext_degree,
            noise: self.noise,
        }
    }

    pub(crate) fn inner_mut(&mut self) -> &mut LweCiphertext {
        &mut self.inner
    }

    pub(crate) fn set_degree_noise(&mut self, degree: Degree, noise: NoiseLevel) {
        self.plaintext_degree = degree;
        self.noise = noise;
    }
}
