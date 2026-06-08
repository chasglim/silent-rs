use crate::schemes::bgv::params::{BgvLevelError, BgvParameters};
use silent_math::numth;
use silent_rlwe::Ciphertext;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BgvPlaintextFactor {
    value: u64,
}

impl BgvPlaintextFactor {
    pub fn one() -> Self {
        Self { value: 1 }
    }

    pub fn new(value: u64, plain_modulus: u64) -> Result<Self, BgvLevelError> {
        let value = value % plain_modulus;
        if numth::mod_inverse(value, plain_modulus).is_none() {
            return Err(BgvLevelError::PlaintextFactorNotInvertible {
                factor: value,
                modulus: plain_modulus,
            });
        }
        Ok(Self { value })
    }

    pub fn value(self) -> u64 {
        self.value
    }

    pub fn inverse_mod(self, plain_modulus: u64) -> Result<u64, BgvLevelError> {
        numth::mod_inverse(self.value, plain_modulus).ok_or(
            BgvLevelError::PlaintextFactorNotInvertible {
                factor: self.value,
                modulus: plain_modulus,
            },
        )
    }

    pub fn mul_mod(
        self,
        rhs: Self,
        plain_modulus: u64,
    ) -> Result<BgvPlaintextFactor, BgvLevelError> {
        let value =
            ((u128::from(self.value) * u128::from(rhs.value)) % u128::from(plain_modulus)) as u64;
        BgvPlaintextFactor::new(value, plain_modulus)
    }
}

#[derive(Clone, Debug)]
pub struct BgvCiphertext {
    raw: Ciphertext,
    params: BgvParameters,
    int_factor: BgvPlaintextFactor,
    noise_scale_degree: usize,
}

impl BgvCiphertext {
    pub fn new(params: BgvParameters, raw: Ciphertext) -> Result<Self, BgvLevelError> {
        Self::with_plaintext_factor(params, raw, BgvPlaintextFactor::one())
    }

    pub fn with_plaintext_factor(
        params: BgvParameters,
        raw: Ciphertext,
        int_factor: BgvPlaintextFactor,
    ) -> Result<Self, BgvLevelError> {
        Self::with_metadata(params, raw, int_factor, 1)
    }

    pub fn with_metadata(
        params: BgvParameters,
        raw: Ciphertext,
        int_factor: BgvPlaintextFactor,
        noise_scale_degree: usize,
    ) -> Result<Self, BgvLevelError> {
        validate_raw_ciphertext(&params, &raw)?;
        int_factor.inverse_mod(params.plain_modulus())?;
        Ok(Self {
            raw,
            params,
            int_factor,
            noise_scale_degree,
        })
    }

    pub fn raw(&self) -> &Ciphertext {
        &self.raw
    }

    pub fn into_raw(self) -> Ciphertext {
        self.raw
    }

    pub fn params(&self) -> &BgvParameters {
        &self.params
    }

    pub fn int_factor(&self) -> BgvPlaintextFactor {
        self.int_factor
    }

    pub fn noise_scale_degree(&self) -> usize {
        self.noise_scale_degree
    }

    pub fn q_modulus_count(&self) -> usize {
        self.params.q_modulus_count()
    }
}

fn validate_raw_ciphertext(params: &BgvParameters, raw: &Ciphertext) -> Result<(), BgvLevelError> {
    let expected_degree = params.degree();
    let expected_limbs = params.q_modulus_count();

    if raw.params.ring.degree() != expected_degree {
        return Err(BgvLevelError::CiphertextDegreeMismatch {
            expected: expected_degree,
            actual: raw.params.ring.degree(),
        });
    }
    if raw.params.ring.rns().len() != expected_limbs {
        return Err(BgvLevelError::CiphertextLimbMismatch {
            expected: expected_limbs,
            actual: raw.params.ring.rns().len(),
        });
    }

    for poly in raw.data.iter() {
        if poly.degree() != expected_degree {
            return Err(BgvLevelError::CiphertextDegreeMismatch {
                expected: expected_degree,
                actual: poly.degree(),
            });
        }
        if poly.num_moduli() != expected_limbs {
            return Err(BgvLevelError::CiphertextLimbMismatch {
                expected: expected_limbs,
                actual: poly.num_moduli(),
            });
        }
    }

    Ok(())
}
