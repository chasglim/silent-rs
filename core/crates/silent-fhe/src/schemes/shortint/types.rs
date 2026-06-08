//! Plaintext degree and noise budget tracking for short-block TFHE (same
//! accounting model as common shortint implementations).

use std::fmt;

use super::error::ShortintError;

/// Worst-case plaintext magnitude tracked in unary / packed shortint form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Degree(pub u64);

impl fmt::Display for Degree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Degree {
    #[inline]
    pub const fn new(degree: u64) -> Self {
        Self(degree)
    }

    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    #[inline]
    pub fn saturating_mul_scalar(self, scalar: u64) -> Self {
        Self(self.0.saturating_mul(scalar))
    }
}

/// PBS input noise level (additive budget model; cf. typical shortint noise levels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NoiseLevel(pub u64);

impl fmt::Display for NoiseLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl NoiseLevel {
    pub const NOMINAL: Self = Self(1);

    #[inline]
    pub const fn new(level: u64) -> Self {
        Self(level)
    }

    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    #[inline]
    pub fn saturating_mul_scalar(self, scalar: u64) -> Self {
        Self(self.0.saturating_mul(scalar))
    }
}

/// Maximum degree before a carry-cleaning PBS (`message_modulus · carry_modulus − 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaxDegree(pub u64);

impl fmt::Display for MaxDegree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl MaxDegree {
    #[inline]
    pub fn from_msg_carry(message_modulus: u64, carry_modulus: u64) -> Self {
        Self(
            message_modulus
                .saturating_mul(carry_modulus)
                .saturating_sub(1),
        )
    }

    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    pub fn validate(self, degree: Degree) -> Result<(), ShortintError> {
        if degree.get() > self.0 {
            Err(ShortintError::CarryFull {
                degree,
                max_degree: self,
            })
        } else {
            Ok(())
        }
    }
}

/// Noise upper bound derived from `(carry·message − 1) / (message − 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaxNoiseLevel(pub u64);

impl fmt::Display for MaxNoiseLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl MaxNoiseLevel {
    pub fn from_msg_carry(message_modulus: u64, carry_modulus: u64) -> Result<Self, ShortintError> {
        if message_modulus < 2 {
            return Err(ShortintError::MessageModulusTooSmall);
        }
        let level = (carry_modulus
            .saturating_mul(message_modulus)
            .saturating_sub(1))
            / (message_modulus - 1);
        Ok(Self(level))
    }

    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    pub fn validate(self, noise: NoiseLevel) -> Result<(), ShortintError> {
        if noise.get() > self.0 {
            Err(ShortintError::NoiseTooBig {
                noise,
                max_noise: self,
            })
        } else {
            Ok(())
        }
    }
}

/// Joint noise/degree pair for precondition checks.
#[derive(Debug, Clone, Copy)]
pub struct CiphertextNoiseDegree {
    pub degree: Degree,
    pub noise: NoiseLevel,
}
