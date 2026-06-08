//! Polynomial container for native (power-of-two) modulus rings.
//!
//! Sits next to [`crate::poly::Poly`] (the RNS-prime container BFV uses) and
//! provides the storage shape TFHE / FHEW need: a single `Vec<u64>` holding
//! coefficients of `Z_{2^k}[X] / (X^N + 1)`.  All arithmetic is delegated to
//! [`silent_math::fft64`] so that the polynomial multiplier can be swapped
//! (e.g. schoolbook → 64-bit FFT) without touching call sites.
//!
//! `NativePoly` deliberately stores coefficients in *coefficient* domain.
//! TFHE ciphertexts only ever leave the coefficient domain inside a single
//! polynomial multiplication, where the conversion is owned by the
//! [`silent_math::fft64::NegacyclicMul`] backend.

use silent_math::fft64::{
    NegacyclicMul, SchoolbookMul, add_assign_u64, monomial_mul_u64, negate_assign_u64,
    scalar_mul_assign_u64, sub_assign_u64,
};

/// Polynomial container for `Z_{2^k}[X] / (X^N + 1)` with `k ∈ {32, 64}`.
///
/// Holds coefficients in low-degree-first order (`coeffs[0] = a_0`).  When
/// `log_modulus == 32` only the low 32 bits of each `u64` are meaningful;
/// the rest must be zero, which the type's invariant guarantees by either
/// construction (`zeros`, `from_u64`) or by routing every mutation through
/// `with_modulus_mask`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativePoly {
    coeffs: Vec<u64>,
    log_modulus: u8,
}

impl NativePoly {
    /// Create a polynomial filled with zeros.
    ///
    /// # Panics
    /// Panics if `degree` is zero or not a power of two, or if `log_modulus`
    /// is not 32 or 64.
    pub fn zeros(degree: usize, log_modulus: u8) -> Self {
        Self::check_args(degree, log_modulus);
        Self {
            coeffs: vec![0u64; degree],
            log_modulus,
        }
    }

    /// Build a polynomial from a coefficient slice.
    ///
    /// Each coefficient is reduced modulo `2^log_modulus`.
    pub fn from_u64(coeffs: &[u64], log_modulus: u8) -> Self {
        Self::check_args(coeffs.len(), log_modulus);
        let mut data = coeffs.to_vec();
        if log_modulus < 64 {
            let mask = (1u64 << log_modulus) - 1;
            for c in data.iter_mut() {
                *c &= mask;
            }
        }
        Self {
            coeffs: data,
            log_modulus,
        }
    }

    fn check_args(degree: usize, log_modulus: u8) {
        assert!(
            degree.is_power_of_two() && degree > 0,
            "degree must be a positive power of two"
        );
        assert!(
            log_modulus == 32 || log_modulus == 64,
            "only log_modulus 32 or 64 are supported"
        );
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len()
    }

    pub fn log_modulus(&self) -> u8 {
        self.log_modulus
    }

    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    pub fn coeffs_mut(&mut self) -> &mut [u64] {
        &mut self.coeffs
    }

    /// Mask each coefficient back into `[0, 2^log_modulus)`.  Cheap when
    /// `log_modulus == 64` (no-op) and a single `&` per coefficient otherwise.
    pub fn reduce(&mut self) {
        if self.log_modulus == 64 {
            return;
        }
        let mask = (1u64 << self.log_modulus) - 1;
        for c in self.coeffs.iter_mut() {
            *c &= mask;
        }
    }

    /// In-place addition: `self += other`.
    pub fn add_assign(&mut self, other: &NativePoly) {
        debug_assert_eq!(self.log_modulus, other.log_modulus);
        debug_assert_eq!(self.degree(), other.degree());
        add_assign_u64(&mut self.coeffs, &other.coeffs);
        self.reduce();
    }

    /// In-place subtraction: `self -= other`.
    pub fn sub_assign(&mut self, other: &NativePoly) {
        debug_assert_eq!(self.log_modulus, other.log_modulus);
        debug_assert_eq!(self.degree(), other.degree());
        sub_assign_u64(&mut self.coeffs, &other.coeffs);
        self.reduce();
    }

    /// Negate every coefficient.
    pub fn negate_assign(&mut self) {
        negate_assign_u64(&mut self.coeffs);
        self.reduce();
    }

    /// Multiply every coefficient by a scalar (wrapping).
    pub fn scalar_mul_assign(&mut self, scalar: u64) {
        scalar_mul_assign_u64(&mut self.coeffs, scalar);
        self.reduce();
    }

    /// Negacyclic polynomial multiplication: `out = self * other`.
    ///
    /// Uses the schoolbook backend by default.  Switch to a faster backend
    /// via [`NativePoly::mul_with`].
    pub fn mul(&self, other: &NativePoly) -> NativePoly {
        self.mul_with(other, &SchoolbookMul)
    }

    /// Like [`NativePoly::mul`] but accepts an explicit backend.
    pub fn mul_with<M: NegacyclicMul>(&self, other: &NativePoly, backend: &M) -> NativePoly {
        debug_assert_eq!(self.log_modulus, other.log_modulus);
        debug_assert_eq!(self.degree(), other.degree());
        let mut out = NativePoly::zeros(self.degree(), self.log_modulus);
        backend.mul_assign_u64(&self.coeffs, &other.coeffs, &mut out.coeffs);
        out.reduce();
        out
    }

    /// `out += self * other` with the given backend.  Useful for accumulating
    /// the gadget product in blind rotation.
    pub fn fma_with<M: NegacyclicMul>(
        &self,
        other: &NativePoly,
        out: &mut NativePoly,
        backend: &M,
    ) {
        debug_assert_eq!(self.log_modulus, other.log_modulus);
        debug_assert_eq!(self.log_modulus, out.log_modulus);
        debug_assert_eq!(self.degree(), other.degree());
        debug_assert_eq!(self.degree(), out.degree());
        backend.fma_assign_u64(&self.coeffs, &other.coeffs, &mut out.coeffs);
        out.reduce();
    }

    /// Multiply by a monomial `X^exponent` (negacyclic).
    pub fn mul_monomial(&self, exponent: usize) -> NativePoly {
        let mut out = NativePoly::zeros(self.degree(), self.log_modulus);
        monomial_mul_u64(&self.coeffs, exponent, &mut out.coeffs);
        out.reduce();
        out
    }

    /// Divide by `X^{deg}` — monic monomial division in `Z_q[X]/(X^N+1)`.
    ///
    /// This is the TFHE **initial** blind-rotation step applied to every GLWE
    /// limb of the PBS accumulator, with `deg` the modulus-switched LWE body
    /// in `0 .. 2N` (not a negacyclic `mul_monomial` on the accumulator).
    pub fn wrapping_monic_monomial_div_assign(&mut self, monomial_degree: usize) {
        let n = self.coeffs.len();
        let inp = self.coeffs.clone();
        let remaining_degree = monomial_degree % n;
        let full_cycles_count = monomial_degree / n;
        let out = &mut self.coeffs;

        if remaining_degree == 0 {
            if full_cycles_count.is_multiple_of(2) {
                out.copy_from_slice(&inp);
            } else {
                for (o, i) in out.iter_mut().zip(inp.iter()) {
                    *o = i.wrapping_neg();
                }
            }
        } else if full_cycles_count.is_multiple_of(2) {
            out[..n - remaining_degree].copy_from_slice(&inp[remaining_degree..]);
            for (o, i) in out[n - remaining_degree..]
                .iter_mut()
                .zip(inp[..remaining_degree].iter())
            {
                *o = i.wrapping_neg();
            }
        } else {
            for (o, i) in out[..n - remaining_degree]
                .iter_mut()
                .zip(inp[remaining_degree..].iter())
            {
                *o = i.wrapping_neg();
            }
            out[n - remaining_degree..].copy_from_slice(&inp[..remaining_degree]);
        }
        self.reduce();
    }

    /// TFHE CMUX scratch polynomial: `self ← (input · X^{deg}) - input` in the
    /// **monic monomial** sense (same slice layout as
    /// `polynomial_wrapping_monic_monomial_mul_and_subtract` in TFHE).
    pub fn wrapping_monic_monomial_mul_and_subtract_from(
        &mut self,
        input: &NativePoly,
        monomial_degree: usize,
    ) {
        debug_assert_eq!(self.degree(), input.degree());
        debug_assert_eq!(self.log_modulus(), input.log_modulus());
        let n = input.degree();
        let inp = input.coeffs();
        let out = self.coeffs_mut();
        let remaining_degree = monomial_degree % n;
        let full_cycles_count = monomial_degree / n;

        if remaining_degree == 0 {
            if full_cycles_count.is_multiple_of(2) {
                out.fill(0);
            } else {
                for i in 0..n {
                    out[i] = inp[i].wrapping_neg().wrapping_sub(inp[i]);
                }
            }
        } else if full_cycles_count.is_multiple_of(2) {
            for j in 0..remaining_degree {
                let src = inp[n - remaining_degree + j];
                let src_orig = inp[j];
                out[j] = src.wrapping_neg().wrapping_sub(src_orig);
            }
            for j in remaining_degree..n {
                let src = inp[j - remaining_degree];
                let src_orig = inp[j];
                out[j] = src.wrapping_sub(src_orig);
            }
        } else {
            for j in 0..remaining_degree {
                let src = inp[n - remaining_degree + j];
                let src_orig = inp[j];
                out[j] = src.wrapping_sub(src_orig);
            }
            for j in remaining_degree..n {
                let src = inp[j - remaining_degree];
                let src_orig = inp[j];
                out[j] = src.wrapping_neg().wrapping_sub(src_orig);
            }
        }
        self.reduce();
    }

    /// Returns `(self · X^exponent) - self`.  Frequently appears as the inner
    /// term of a CMUX in TFHE (where the GGSW digits are accumulated against
    /// `(rot - id)`).
    pub fn mul_monomial_minus_one(&self, exponent: usize) -> NativePoly {
        let mut rot = self.mul_monomial(exponent);
        rot.sub_assign(self);
        rot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub_negate_q64() {
        let mut a = NativePoly::from_u64(&[1, 2, 3, 4], 64);
        let b = NativePoly::from_u64(&[10, 20, 30, 40], 64);
        a.add_assign(&b);
        assert_eq!(a.coeffs(), &[11, 22, 33, 44]);
        a.sub_assign(&b);
        assert_eq!(a.coeffs(), &[1, 2, 3, 4]);
        a.negate_assign();
        assert_eq!(
            a.coeffs(),
            &[
                1u64.wrapping_neg(),
                2u64.wrapping_neg(),
                3u64.wrapping_neg(),
                4u64.wrapping_neg(),
            ]
        );
    }

    #[test]
    fn add_sub_q32_keeps_bits_inside_mask() {
        let mut a = NativePoly::from_u64(&[u32::MAX as u64, 1, 0, 0], 32);
        let b = NativePoly::from_u64(&[1, u32::MAX as u64, 7, 0], 32);
        a.add_assign(&b);
        assert_eq!(a.coeffs(), &[0, 0, 7, 0]); // wraps within 32 bits
        a.sub_assign(&b);
        assert_eq!(a.coeffs(), &[u32::MAX as u64, 1, 0, 0]);
    }

    #[test]
    fn schoolbook_mul_negacyclic() {
        // (X + 1) * (X^3 + 1) = X^4 + X^3 + X + 1 -> mod (X^4+1) -> X^3 + X.
        let a = NativePoly::from_u64(&[1, 1, 0, 0], 64);
        let b = NativePoly::from_u64(&[1, 0, 0, 1], 64);
        let c = a.mul(&b);
        assert_eq!(c.coeffs(), &[0, 1, 0, 1]);
    }

    #[test]
    fn wrapping_monic_monomial_div_matches_known_tfhe_pattern() {
        // Coefficient ring matches the TFHE u8 doctest embedded in N=4 (padding 0).
        let mut p = NativePoly::from_u64(&[1, 2, 3, 0], 64);
        p.wrapping_monic_monomial_div_assign(2);
        assert_eq!(p.coeffs(), &[3, 0, u64::MAX, u64::MAX - 1]);
    }

    #[test]
    fn monomial_mul_then_minus_one_is_consistent() {
        let p = NativePoly::from_u64(&[1, 2, 3, 4], 64);
        let r = p.mul_monomial(2);
        let mut expected = p.clone();
        // X^2: shifts up by 2, the wrapped pair is negated.
        expected
            .coeffs_mut()
            .copy_from_slice(&[3u64.wrapping_neg(), 4u64.wrapping_neg(), 1, 2]);
        assert_eq!(r, expected);

        let mut r_minus = p.mul_monomial_minus_one(2);
        r_minus.add_assign(&p);
        assert_eq!(r_minus, r);
    }
}
