//! GLWE containers over a native power-of-two modulus, used by TFHE's
//! programmable bootstrap.
//!
//! `GlweCiphertext` follows the (mask, body) layout familiar from LWE.  The
//! difference is that each entry is now a polynomial in
//! `Z_{2^k}[X] / (X^N + 1)`, stored as a [`NativePoly`].

use silent_params::{CiphertextModulusLog, GlweDimension, PolynomialSize};
use silent_ring::NativePoly;

/// GLWE ciphertext: `k` mask polynomials and one body polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlweCiphertext {
    /// `mask[0..k]` are the `a_i`; `mask[k]` is the body `b`.
    polys: Vec<NativePoly>,
    log_modulus: u8,
}

impl GlweCiphertext {
    /// Create a fresh zero ciphertext.
    pub fn zeros(
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
        log_modulus: CiphertextModulusLog,
    ) -> Self {
        let polys = (0..glwe_dimension.glwe_size())
            .map(|_| NativePoly::zeros(polynomial_size.0, log_modulus.0))
            .collect();
        Self {
            polys,
            log_modulus: log_modulus.0,
        }
    }

    pub fn from_polys(polys: Vec<NativePoly>, log_modulus: CiphertextModulusLog) -> Self {
        assert!(polys.len() >= 1, "at least the body must be present");
        let lm = log_modulus.0;
        for p in &polys {
            assert_eq!(p.log_modulus(), lm, "log_modulus mismatch in GLWE polys");
        }
        Self {
            polys,
            log_modulus: lm,
        }
    }

    pub fn glwe_dimension(&self) -> GlweDimension {
        GlweDimension(self.polys.len() - 1)
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        PolynomialSize(self.polys[0].degree())
    }

    pub fn log_modulus(&self) -> u8 {
        self.log_modulus
    }

    /// `a_0 .. a_{k-1}`.
    pub fn mask(&self) -> &[NativePoly] {
        let k = self.glwe_dimension().0;
        &self.polys[..k]
    }

    /// `a_0 .. a_{k-1}`, mutable.
    pub fn mask_mut(&mut self) -> &mut [NativePoly] {
        let k = self.glwe_dimension().0;
        &mut self.polys[..k]
    }

    pub fn body(&self) -> &NativePoly {
        self.polys.last().expect("body always present")
    }

    pub fn body_mut(&mut self) -> &mut NativePoly {
        self.polys.last_mut().expect("body always present")
    }

    /// All polynomials (mask first, body last).
    pub fn polys(&self) -> &[NativePoly] {
        &self.polys
    }

    pub fn polys_mut(&mut self) -> &mut [NativePoly] {
        &mut self.polys
    }

    /// `self += other`.
    pub fn add_assign(&mut self, other: &GlweCiphertext) {
        debug_assert_eq!(self.log_modulus, other.log_modulus);
        debug_assert_eq!(self.polys.len(), other.polys.len());
        for (a, b) in self.polys.iter_mut().zip(other.polys.iter()) {
            a.add_assign(b);
        }
    }

    /// `self -= other`.
    pub fn sub_assign(&mut self, other: &GlweCiphertext) {
        debug_assert_eq!(self.log_modulus, other.log_modulus);
        debug_assert_eq!(self.polys.len(), other.polys.len());
        for (a, b) in self.polys.iter_mut().zip(other.polys.iter()) {
            a.sub_assign(b);
        }
    }

    /// Multiply every polynomial by the same monomial `X^exponent`
    /// (negacyclic).  Used by blind rotation (the initial accumulator rotation
    /// by `b` and by the CMUX trial polynomial `(X^{a_i} - 1)`).
    pub fn rotate_assign(&mut self, exponent: usize) {
        for p in self.polys.iter_mut() {
            *p = p.mul_monomial(exponent);
        }
    }
}

/// GLWE secret key: `glwe_dimension` polynomials with small coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlweSecretKey {
    polys: Vec<NativePoly>,
    log_modulus: u8,
}

impl GlweSecretKey {
    pub fn from_polys(polys: Vec<NativePoly>, log_modulus: CiphertextModulusLog) -> Self {
        let lm = log_modulus.0;
        for p in &polys {
            assert_eq!(
                p.log_modulus(),
                lm,
                "log_modulus mismatch in GLWE secret key"
            );
        }
        Self {
            polys,
            log_modulus: lm,
        }
    }

    pub fn glwe_dimension(&self) -> GlweDimension {
        GlweDimension(self.polys.len())
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        PolynomialSize(self.polys[0].degree())
    }

    pub fn log_modulus(&self) -> u8 {
        self.log_modulus
    }

    pub fn polys(&self) -> &[NativePoly] {
        &self.polys
    }

    /// Sample-extract the GLWE secret key into the equivalent flat LWE
    /// secret key (length `k * N`), using the canonical TFHE ordering.
    ///
    /// Specifically, `LWE_sk[i*N + j] = GLWE_sk[i].coeffs[j]`.  This is what
    /// the LWE→LWE key-switch consumes after a programmable bootstrap.
    pub fn flatten(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.glwe_dimension().0 * self.polynomial_size().0);
        for poly in &self.polys {
            out.extend_from_slice(poly.coeffs());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cm(log: u8) -> CiphertextModulusLog {
        CiphertextModulusLog(log)
    }

    #[test]
    fn glwe_layout_and_arithmetic() {
        let mut a = GlweCiphertext::zeros(GlweDimension(2), PolynomialSize(4), cm(64));
        let mut b = GlweCiphertext::zeros(GlweDimension(2), PolynomialSize(4), cm(64));

        for poly in a.mask_mut() {
            poly.coeffs_mut().copy_from_slice(&[1, 2, 3, 4]);
        }
        a.body_mut().coeffs_mut().copy_from_slice(&[5, 6, 7, 8]);
        for poly in b.mask_mut() {
            poly.coeffs_mut().copy_from_slice(&[10, 10, 10, 10]);
        }
        b.body_mut().coeffs_mut().copy_from_slice(&[1, 1, 1, 1]);

        a.add_assign(&b);
        assert_eq!(a.mask()[0].coeffs(), &[11, 12, 13, 14]);
        assert_eq!(a.body().coeffs(), &[6, 7, 8, 9]);
    }

    #[test]
    fn glwe_secret_flatten_concat_order() {
        let polys = vec![
            NativePoly::from_u64(&[1, 2, 3, 4], 64),
            NativePoly::from_u64(&[5, 6, 7, 8], 64),
        ];
        let sk = GlweSecretKey::from_polys(polys, cm(64));
        assert_eq!(sk.flatten(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
