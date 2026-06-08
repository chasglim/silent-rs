//! Polynomial container.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    // Stores coefficients for all RNS limbs contiguously.
    // Layout: [limb0_coeff0, limb0_coeff1, ..., limb1_coeff0, ...]
    data: Vec<u64>,
    degree: usize,
    num_moduli: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyShoup {
    // Stores Shoup operands for all RNS limbs contiguously.
    // Layout matches Poly: [limb0_coeff0, limb0_coeff1, ..., limb1_coeff0, ...]
    data: Vec<silent_math::arith::MultiplyUIntModOperand>,
    degree: usize,
    num_moduli: usize,
}

impl Poly {
    pub fn new(degree: usize, num_moduli: usize) -> Self {
        Self {
            data: vec![0; degree * num_moduli],
            degree,
            num_moduli,
        }
    }

    /// Create a polynomial with uninitialized coefficients.
    ///
    /// # Safety
    /// All coefficients must be written before any read.
    pub unsafe fn new_uninit(degree: usize, num_moduli: usize) -> Self {
        let len = degree * num_moduli;
        let mut data = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }
        Self {
            data,
            degree,
            num_moduli,
        }
    }

    pub fn from_vec(data: Vec<u64>, degree: usize, num_moduli: usize) -> Self {
        assert_eq!(data.len(), degree * num_moduli);
        Self {
            data,
            degree,
            num_moduli,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn num_moduli(&self) -> usize {
        self.num_moduli
    }

    pub fn data(&self) -> &[u64] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }

    pub fn limb(&self, limb_idx: usize) -> &[u64] {
        let start = limb_idx * self.degree;
        &self.data[start..start + self.degree]
    }

    pub fn limb_mut(&mut self, limb_idx: usize) -> &mut [u64] {
        let start = limb_idx * self.degree;
        &mut self.data[start..start + self.degree]
    }

    pub fn add_assign(&mut self, other: &Self, context: &crate::RingContext) {
        assert_eq!(self.degree, other.degree);
        assert_eq!(self.num_moduli, other.num_moduli);

        let moduli = context.rns().moduli();
        let degree = self.degree;
        for i in 0..self.num_moduli {
            let q = moduli[i].value();
            let lhs = self.limb_mut(i);
            let rhs = other.limb(i);

            // Optimized: unsafe indexing to eliminate bounds checks
            // Safety: Both lhs and rhs have length == degree, and j < degree
            for j in 0..degree {
                unsafe {
                    let a = *lhs.get_unchecked(j);
                    let b = *rhs.get_unchecked(j);
                    *lhs.get_unchecked_mut(j) = silent_math::arith::add_mod(a, b, q);
                }
            }
        }
    }

    pub fn sub_assign(&mut self, other: &Self, context: &crate::RingContext) {
        assert_eq!(self.degree, other.degree);
        assert_eq!(self.num_moduli, other.num_moduli);

        let moduli = context.rns().moduli();
        let degree = self.degree;
        for i in 0..self.num_moduli {
            let q = moduli[i].value();
            let lhs = self.limb_mut(i);
            let rhs = other.limb(i);

            for j in 0..degree {
                unsafe {
                    let a = *lhs.get_unchecked(j);
                    let b = *rhs.get_unchecked(j);
                    *lhs.get_unchecked_mut(j) = silent_math::arith::sub_mod(a, b, q);
                }
            }
        }
    }

    // Element-wise multiplication (for RNS/NTT domain)
    pub fn mul_assign(&mut self, other: &Self, context: &crate::RingContext) {
        assert_eq!(self.degree, other.degree);
        assert_eq!(self.num_moduli, other.num_moduli);

        let moduli = context.rns().moduli();
        let degree = self.degree;
        for i in 0..self.num_moduli {
            let modulus = &moduli[i];
            let lhs = self.limb_mut(i);
            let rhs = other.limb(i);

            for j in 0..degree {
                unsafe {
                    let a = *lhs.get_unchecked(j);
                    let b = *rhs.get_unchecked(j);
                    *lhs.get_unchecked_mut(j) = silent_math::arith::mul_mod(a, b, modulus);
                }
            }
        }
    }

    pub fn mul_assign_shoup(&mut self, other: &PolyShoup, context: &crate::RingContext) {
        assert_eq!(self.degree, other.degree);
        assert_eq!(self.num_moduli, other.num_moduli);

        let moduli = context.rns().moduli();
        let degree = self.degree;
        for i in 0..self.num_moduli {
            let modulus = moduli[i].value();
            let lhs = self.limb_mut(i);
            let rhs = other.limb(i);

            for j in 0..degree {
                unsafe {
                    let a = *lhs.get_unchecked(j);
                    let shoup = rhs.get_unchecked(j);
                    *lhs.get_unchecked_mut(j) = silent_math::arith::mul_mod_shoup(
                        a,
                        shoup.operand,
                        shoup.quotient,
                        modulus,
                    );
                }
            }
        }
    }

    pub fn mul_assign_shoup_lazy(&mut self, other: &PolyShoup, context: &crate::RingContext) {
        assert_eq!(self.degree, other.degree);
        assert_eq!(self.num_moduli, other.num_moduli);

        let moduli = context.rns().moduli();
        let degree = self.degree;
        for i in 0..self.num_moduli {
            let modulus = moduli[i].value();
            let two_q = modulus << 1;
            let lhs = self.limb_mut(i);
            let rhs = other.limb(i);

            for j in 0..degree {
                unsafe {
                    let a = *lhs.get_unchecked(j);
                    let shoup = rhs.get_unchecked(j);
                    let mut value = silent_math::arith::mul_mod_shoup_lazy(
                        a,
                        shoup.operand,
                        shoup.quotient,
                        modulus,
                    );
                    if value >= two_q {
                        value -= two_q;
                    }
                    *lhs.get_unchecked_mut(j) = value;
                }
            }
        }
    }

    pub fn ntt_forward(&mut self, context: &crate::RingContext) {
        use silent_math::backend::MathBackend;
        assert_eq!(self.num_moduli, context.ntt_tables().len());

        for i in 0..self.num_moduli {
            let tables = &context.ntt_tables()[i];
            let limb = self.limb_mut(i);
            // TODO: Select backend dynamically. Using Native for now.
            silent_math::backend::NativeBackend.ntt_forward(limb, tables);
        }
    }

    pub fn ntt_forward_lazy(&mut self, context: &crate::RingContext) {
        use silent_math::ntt::ntt_forward_lazy;
        assert_eq!(self.num_moduli, context.ntt_tables().len());

        for i in 0..self.num_moduli {
            let tables = &context.ntt_tables()[i];
            let limb = self.limb_mut(i);
            ntt_forward_lazy(limb, tables);
        }
    }

    pub fn ntt_inverse(&mut self, context: &crate::RingContext) {
        use silent_math::backend::MathBackend;
        assert_eq!(self.num_moduli, context.ntt_tables().len());

        for i in 0..self.num_moduli {
            let tables = &context.ntt_tables()[i];
            let limb = self.limb_mut(i);
            silent_math::backend::NativeBackend.ntt_inverse(limb, tables);
        }
    }

    /// Permutes coefficients in NTT domain according to the provided index map.
    /// The map[i] indicates the source index for destination index i.
    /// dest[i] = src[map[i]].
    pub fn apply_automorphism(&mut self, map: &[usize]) {
        let mut temp = vec![0u64; self.degree];
        self.apply_automorphism_with_scratch(map, &mut temp);
    }

    /// Permutes coefficients in NTT domain using a caller-provided scratch buffer.
    pub fn apply_automorphism_with_scratch(&mut self, map: &[usize], scratch: &mut [u64]) {
        assert_eq!(map.len(), self.degree, "Map length must match degree");
        assert_eq!(
            scratch.len(),
            self.degree,
            "Scratch length must match degree"
        );

        for mod_idx in 0..self.num_moduli {
            let limb = self.limb_mut(mod_idx);

            // Permute into scratch
            for (dest_i, &src_i) in map.iter().enumerate() {
                scratch[dest_i] = limb[src_i];
            }

            // Copy back
            limb.copy_from_slice(&scratch[..]);
        }
    }
}

impl PolyShoup {
    pub fn from_poly(poly: &Poly, moduli: &[silent_math::modulus::Modulus]) -> Self {
        assert_eq!(moduli.len(), poly.num_moduli);

        let mut data = Vec::with_capacity(poly.degree * poly.num_moduli);
        for (mod_idx, modulus) in moduli.iter().enumerate() {
            let limb = poly.limb(mod_idx);
            let modulus_value = modulus.value();
            for &value in limb.iter() {
                data.push(silent_math::arith::MultiplyUIntModOperand::new(
                    value,
                    modulus_value,
                ));
            }
        }

        Self {
            data,
            degree: poly.degree,
            num_moduli: poly.num_moduli,
        }
    }

    pub fn limb(&self, limb_idx: usize) -> &[silent_math::arith::MultiplyUIntModOperand] {
        let start = limb_idx * self.degree;
        &self.data[start..start + self.degree]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RingContext;

    use silent_math::rns::RnsBase;

    fn solve_context(degree: usize, qs: Vec<u64>) -> RingContext {
        let rns = RnsBase::from_values(qs).unwrap();
        RingContext::new(degree, rns)
    }

    #[test]
    fn poly_ops_work() {
        let degree = 8;
        let q = 17; // Prime
        let context = solve_context(degree, vec![q]);

        // P1 = [1, 1, ..., 1]
        let mut p1 = Poly::from_vec(vec![1; degree], degree, 1);
        // P2 = [2, 2, ..., 2]
        let p2 = Poly::from_vec(vec![2; degree], degree, 1);

        // P1 += P2 => [3, ...]
        p1.add_assign(&p2, &context);
        assert_eq!(p1.data(), vec![3; degree].as_slice());

        // P1 -= P2 => [1, ...]
        p1.sub_assign(&p2, &context);
        assert_eq!(p1.data(), vec![1; degree].as_slice());

        // P3 = [2, 0, ...] * [3, 0, ...] = [6, 0, ...]
        // Note: mul_assign is element-wise mod q multiplication
        let mut p3 = Poly::from_vec(vec![2; degree], degree, 1);
        let p4 = Poly::from_vec(vec![3; degree], degree, 1);
        p3.mul_assign(&p4, &context);
        assert_eq!(p3.data(), vec![6; degree].as_slice());
    }

    #[test]
    fn poly_ntt_roundtrip() {
        let degree = 8;
        let q = 17; // 17 is prime, 2*8 | (17-1) => 16 | 16. Yes.
        let context = solve_context(degree, vec![q]);

        let original = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut poly = Poly::from_vec(original.clone(), degree, 1);

        poly.ntt_forward(&context);
        assert_ne!(poly.data(), original.as_slice()); // Should assume valid NTT changes data

        poly.ntt_inverse(&context);
        assert_eq!(poly.data(), original.as_slice());
    }
}
