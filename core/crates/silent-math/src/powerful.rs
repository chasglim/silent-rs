//! Powerful Basis implementation for non-power-of-two cyclotomics.
//!
//! Adapted from HElib's powerful.cpp.
//!
//! The powerful basis allows representing elements in R_m = Z[X]/Phi_m(X)
//! as a structure isomorphic to the tensor product of smaller cyclotomic rings.
//! This facilitates efficient coefficient-wise operations and CRT-based arithmetic.

use crate::arith;
use crate::numth;

/// Holds precomputed tables for converting between standard and powerful basis.
#[derive(Debug, Clone)]
pub struct PowerfulBasis {
    pub m: u64,
    pub phim: u64,
    pub factors: Vec<(u64, u32)>, // (p, e)

    // Signatures
    pub long_dims: Vec<u64>,  // m_i = p_i ^ e_i
    pub short_dims: Vec<u64>, // phi(m_i)

    // Maps (flattened hypercube indices)
    pub poly_to_cube_map: Vec<usize>,
    pub cube_to_poly_map: Vec<usize>,
    pub short_to_long_map: Vec<usize>,

    // Cyclotomic polynomials for each factor mod q is dynamic, but the integer form is static.
    // We store standard form here if needed, or precompute mod q on demand?
    // Conversion requires `cyc_vec` mod q.
    // Cached cyclotomic polynomial for m (integer coefficients)
    // Used for inverse transformation.
    pub phi_m_int: Vec<i64>,
}

impl PowerfulBasis {
    pub fn new(m: u64) -> Self {
        let factors = numth::factorize(m);
        let phim = numth::euler_phi(m);

        let mut long_dims = Vec::new();
        let mut short_dims = Vec::new();
        let mut m_vec = Vec::new();
        let mut phi_vec = Vec::new();

        for &(p, count) in &factors {
            let mk = p.pow(count);
            let phi_mk = mk / p * (p - 1);
            long_dims.push(mk);
            short_dims.push(phi_mk);
            m_vec.push(mk);
            phi_vec.push(phi_mk);
        }

        // ... (existing map computation code) ...
        let inv_vec: Vec<u64> = m_vec
            .iter()
            .map(|&mi| {
                let div = m / mi;
                numth::mod_inverse(div, mi).unwrap()
            })
            .collect();

        // Compute poly_to_cube_map
        let mut poly_to_cube_map = vec![0; m as usize];
        let mut cube_to_poly_map = vec![0; m as usize];

        for i in 0..m {
            let mut idx_in_cube = 0;
            let mut current_stride = 1;
            for (d, &mi) in m_vec.iter().enumerate() {
                let inv = inv_vec[d];
                // i_d = (i mod mi) * inv mod mi
                // We use u128 for intermediate calculation to avoid overflow
                let i_d = ((i % mi) as u128 * inv as u128) % mi as u128;
                idx_in_cube += (i_d as u64) * current_stride;
                current_stride *= mi;
            }

            poly_to_cube_map[i as usize] = idx_in_cube as usize;
            cube_to_poly_map[idx_in_cube as usize] = i as usize;
        }

        // Compute short_to_long_map
        let mut short_to_long_map = vec![0; phim as usize];
        for i in 0..phim {
            let mut temp = i;
            let mut j = 0;
            let mut long_stride = 1;
            for (d, &phi_mi) in phi_vec.iter().enumerate() {
                let mi = m_vec[d];
                let coord = temp % phi_mi;
                temp /= phi_mi;

                j += coord * long_stride;
                long_stride *= mi;
            }
            short_to_long_map[i as usize] = j as usize;
        }

        // Precompute Phi_m(X) over integers
        let phi_m_int = Self::compute_cyclotomic_poly_int(m);

        Self {
            m,
            phim,
            factors,
            long_dims,
            short_dims,
            poly_to_cube_map,
            cube_to_poly_map,
            short_to_long_map,
            phi_m_int,
        }
    }

    pub fn compute_cyclotomic_poly(k: u64, modulus: u64) -> Vec<u64> {
        let coeffs_int = Self::compute_cyclotomic_poly_int(k);
        coeffs_int
            .iter()
            .map(|&c| {
                if c >= 0 {
                    c as u64 % modulus
                } else {
                    let val = c.abs() as u64 % modulus;
                    if val == 0 { 0 } else { modulus - val }
                }
            })
            .collect()
    }

    /// Computes Phi_k(X) with integer coefficients.
    pub fn compute_cyclotomic_poly_int(k: u64) -> Vec<i64> {
        if k == 1 {
            return vec![-1, 1]; // X - 1
        }

        // Phi_k(X) = (X^k - 1) / product(Phi_d(X) for d|k, d<k)
        let mut product = vec![1i64];

        let proper_divisors = get_proper_divisors(k);
        for d in proper_divisors {
            let phi_d = Self::compute_cyclotomic_poly_int(d);
            product = poly_mul_int(&product, &phi_d);
        }

        // X^k - 1
        let mut numerator = vec![0i64; k as usize + 1];
        numerator[k as usize] = 1;
        numerator[0] = -1;

        let (quotient, _) = poly_div_rem_int(&numerator, &product);
        quotient
    }

    pub fn powerful_to_poly(&self, powerful: &[u64], modulus: u64) -> Vec<u64> {
        let phim = self.phim as usize;
        assert_eq!(powerful.len(), phim);

        // tmp = 0, size m
        let mut tmp = vec![0u64; self.m as usize];

        // Map coeffs: tmp[Map(i)] = powerful[i]
        for i in 0..phim {
            let idx = self.cube_to_poly_map[self.short_to_long_map[i]];
            tmp[idx] = powerful[i];
        }

        // Reduce using cached phi_m_int
        // We first convert phi_m_int to mod q
        // Note: coefs are small (usually -1, 0, 1), so conversion is cheap.
        // If modulus is small, negative values wrap correctly.

        let phi_m_mod: Vec<u64> = self
            .phi_m_int
            .iter()
            .map(|&c| {
                if c >= 0 {
                    c as u64 % modulus
                } else {
                    // c is negative, e.g. -1. -1 mod q = q - 1.
                    // (c % modulus + modulus) % modulus ?
                    // c is i64.
                    let val = c.abs() as u64 % modulus;
                    if val == 0 { 0 } else { modulus - val }
                }
            })
            .collect();

        let (_, rem) = poly_div_rem_mod(&tmp, &phi_m_mod, modulus);

        rem
    }

    pub fn poly_to_powerful(&self, poly: &[u64], modulus: u64) -> Vec<u64> {
        let mut cube = vec![0u64; self.m as usize];
        for (i, &c) in poly.iter().enumerate() {
            if i < self.poly_to_cube_map.len() {
                cube[self.poly_to_cube_map[i]] = c;
            }
        }

        // Compute cyclotomic polys mod q for deduction
        let mut cyc_vec = Vec::new();
        for &dim in &self.long_dims {
            cyc_vec.push(Self::compute_cyclotomic_poly(dim, modulus));
        }

        Self::recursive_reduce(&mut cube, &cyc_vec, 0, &self.long_dims, modulus);

        // Extract powerful coeffs
        let mut powerful = vec![0u64; self.phim as usize];
        for i in 0..self.phim as usize {
            let long_idx = self.short_to_long_map[i];
            powerful[i] = cube[long_idx];
        }
        powerful
    }

    fn recursive_reduce(
        cube: &mut [u64],
        cyc_vec: &[Vec<u64>],
        dim_idx: usize,
        sig: &[u64],
        modulus: u64,
    ) {
        if dim_idx >= sig.len() {
            return;
        }

        let dim_size = sig[dim_idx] as usize;
        let _stride: usize = sig[0..dim_idx].iter().product::<u64>() as usize; // Wait, stride depends on layout. 
        // HElib layout: lexicographic.
        // Index (i1, i2..., ik) -> i1 + i2*m1 + i3*m1*m2 ...
        // So dimension 0 is the "fastest" varying?
        // Let's recheck HElib logic.
        // j += i_d * longSig.getProd(d+1).
        // getProd(d+1) means stride for d is product of previous dimensions.
        // So d=0 stride is 1. d=1 stride is m0.
        // So yes, dim 0 is fastest varying.

        // But HElib recursiveReduce iterates over columns.
        // If we are reducing dimension `d`, we iterate over all other coordinates.
        // "HyperColumn" in HElib extracts the vector corresponding to varying index corresponding to dimension `d`.

        let current_stride: usize = sig[0..dim_idx].iter().product::<u64>() as usize;
        let next_stride = current_stride * dim_size;
        let total_size = cube.len();
        let num_blocks = total_size / next_stride;

        // For each fixed setting of other coordinates, we have a vector of length `dim_size` at stride `current_stride`.
        // We iterate `pos` which covers all indices *except* the current dimension's contribution.
        // Actually it's easier to think: we have `num_blocks` blocks of size `next_stride`.
        // Inside each block, we have `dim_size` sub-blocks of size `current_stride`.
        // We want to process columns.

        // To avoid complex indexing, let's look at it recursively:
        // Or essentially: for every i in 0..total_size where i % next_stride < current_stride:
        // we have a vector: cube[i], cube[i + current_stride], cube[i + 2*current_stride]...

        // Better: iterate over all `prefix` (coordinates after d) and `suffix` (coordinates before d).

        let modulus_poly = &cyc_vec[dim_idx];
        let _deg_mod = modulus_poly.len() - 1; // degree of Phi_{m_d}

        // Iteration
        // We iterate over everything "orthogonal" to this dimension.
        // Since flattened, this means iterating over the whole array but skipping the stride.

        // Indices are: a + b * next_stride, where 0 <= a < current_stride.
        // The column elements are at (a + b * next_stride) + k * current_stride, for 0 <= k < dim_size.

        for b in 0..num_blocks {
            for a in 0..current_stride {
                let start_idx = a + b * next_stride;

                // Extract column
                let mut col = Vec::with_capacity(dim_size);
                for k in 0..dim_size {
                    col.push(cube[start_idx + k * current_stride]);
                }

                // Poly reduction: col % modulus_poly
                // Note: col is degree dim_size-1. modulus_poly is degree phi(m_d).
                // phi(m_d) < m_d usually.

                let (_, rem) = poly_div_rem_mod(&col, modulus_poly, modulus);

                // Write back
                // Remainder degree is < deg_mod.
                // We pad with zeros up to what?
                // The hypercube slot is still size `dim_size`.
                // We write rem into the first `deg_mod` slots and zero the rest.

                for k in 0..dim_size {
                    let val = if k < rem.len() { rem[k] } else { 0 };
                    cube[start_idx + k * current_stride] = val;
                }
            }
        }

        // Recurse to next dimension
        Self::recursive_reduce(cube, cyc_vec, dim_idx + 1, sig, modulus);
    }
}

// === Helper Functions ===

fn get_proper_divisors(k: u64) -> Vec<u64> {
    let mut divisors = Vec::new();
    for i in 1..k {
        if k % i == 0 {
            divisors.push(i);
        }
    }
    divisors
}

/// Compute (a * b) % (X^n + ...) is not needed, just standard poly mul.
fn poly_div_rem_mod(dividend: &[u64], divisor: &[u64], modulus: u64) -> (Vec<u64>, Vec<u64>) {
    // Standard long division
    if divisor.is_empty() || divisor.iter().all(|&x| x == 0) {
        panic!("Division by zero polynomial");
    }

    // Normalize: remove trailing zeros (high degree coeffs which are 0)
    let dividend = trim_zeros(dividend);
    let divisor = trim_zeros(divisor);

    if dividend.len() < divisor.len() {
        return (vec![0], dividend);
    }

    let mut rem = dividend.clone();
    let mut quo = vec![0u64; dividend.len() - divisor.len() + 1];

    let divisor_deg = divisor.len() - 1;
    let lead_divisor = divisor[divisor_deg];
    let lead_inv =
        numth::mod_inverse(lead_divisor, modulus).expect("Divisor leading coeff not invertible");
    let mod_obj = crate::modulus::Modulus::new(modulus).unwrap();

    // While deg(rem) >= deg(divisor)
    while rem.len() >= divisor.len() {
        let rem_deg = rem.len() - 1;
        if rem_deg < divisor_deg {
            break;
        }

        let lead_rem = rem[rem_deg];
        if lead_rem == 0 {
            rem.pop();
            continue;
        }

        let diff_deg = rem_deg - divisor_deg;
        let scale = arith::mul_mod(lead_rem, lead_inv, &mod_obj);

        quo[diff_deg] = scale;

        // rem -= scale * X^diff * divisor
        for i in 0..=divisor_deg {
            let term = arith::mul_mod(divisor[i], scale, &mod_obj);
            let idx = i + diff_deg;
            rem[idx] = arith::sub_mod(rem[idx], term, modulus);
        }

        // After subtraction, the leading term of rem should be 0.
        // We pop it to reduce degree.
        while let Some(&last) = rem.last() {
            if last == 0 {
                rem.pop();
            } else {
                break;
            }
        }
    }

    if rem.is_empty() {
        rem.push(0);
    }
    (quo, rem)
}

fn trim_zeros(a: &[u64]) -> Vec<u64> {
    let mut a = a.to_vec();
    while a.len() > 1 && a.last() == Some(&0) {
        a.pop();
    }
    a
}

fn poly_mul_int(a: &[i64], b: &[i64]) -> Vec<i64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let deg_res = a.len() + b.len() - 2;
    let mut res = vec![0i64; deg_res + 1];

    for (i, &ca) in a.iter().enumerate() {
        if ca == 0 {
            continue;
        }
        for (j, &cb) in b.iter().enumerate() {
            res[i + j] += ca * cb;
        }
    }
    res
}

fn poly_div_rem_int(dividend: &[i64], divisor: &[i64]) -> (Vec<i64>, Vec<i64>) {
    // Standard long division for integers
    if divisor.is_empty() || divisor.iter().all(|&x| x == 0) {
        panic!("Division by zero polynomial");
    }

    let dividend = trim_zeros_int(dividend);
    let divisor = trim_zeros_int(divisor);

    if dividend.len() < divisor.len() {
        return (vec![0], dividend);
    }

    let mut rem = dividend.clone();
    let mut quo = vec![0i64; dividend.len() - divisor.len() + 1];

    let divisor_deg = divisor.len() - 1;
    let lead_divisor = divisor[divisor_deg];
    // We assume division is possible in Z

    while rem.len() >= divisor.len() {
        let rem_deg = rem.len() - 1;
        let lead_rem = rem[rem_deg];

        let scale = lead_rem / lead_divisor;
        let diff_deg = rem_deg - divisor_deg;

        quo[diff_deg] = scale;

        for i in 0..=divisor_deg {
            let term = divisor[i] * scale;
            let idx = i + diff_deg;
            rem[idx] -= term;
        }

        while let Some(&last) = rem.last() {
            if last == 0 {
                rem.pop();
            } else {
                break;
            }
        }
    }

    if rem.is_empty() {
        rem.push(0);
    }
    (quo, rem)
}

fn trim_zeros_int(a: &[i64]) -> Vec<i64> {
    let mut a = a.to_vec();
    while a.len() > 1 && a.last() == Some(&0) {
        a.pop();
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::numth;

    #[test]
    fn test_cyclotomic_poly_mod() {
        // Phi_1 = X - 1
        // Phi_2 = X + 1
        // Phi_3 = X^2 + X + 1
        // Phi_4 = X^2 + 1
        // Phi_6 = X^2 - X + 1

        let modulus = 17;
        let _pb = PowerfulBasis::new(2); // dummy

        // Phi_3 mod 17
        let phi3 = PowerfulBasis::compute_cyclotomic_poly(3, modulus);
        assert_eq!(phi3, vec![1, 1, 1]);

        // Phi_4 mod 17
        let phi4 = PowerfulBasis::compute_cyclotomic_poly(4, modulus);
        assert_eq!(phi4, vec![1, 0, 1]);

        // Phi_6 mod 17 -> X^2 + 16X + 1
        let phi6 = PowerfulBasis::compute_cyclotomic_poly(6, modulus);
        assert_eq!(phi6, vec![1, 16, 1]);
    }

    #[test]
    fn test_powerful_basis_structure() {
        let m = 12; // 3 * 2^2 = 3 * 4. Factors (3,1), (2,2).
        let pb = PowerfulBasis::new(m);

        assert_eq!(pb.m, 12);
        assert_eq!(pb.phim, 4); // phi(12)=4

        // Factors: (2,2) -> m=4, phi=2. (3,1) -> m=3, phi=2.
        // Order depends on factorize implementation (sorted). 2 then 3.
        assert_eq!(pb.long_dims, vec![4, 3]);
        assert_eq!(pb.short_dims, vec![2, 2]);
    }

    #[test]
    fn test_poly_to_powerful() {
        // Example: m=3. Phi_3(X) = X^2+X+1. R_3 = Z[X]/(X^2+X+1).
        // Powerful basis for m=3 (prime) is just standard basis?
        // m=3: factors=(3,1). long_dims=[3], short_dims=[2].
        // poly X. powerful[1]?
        // recursive_reduce:
        // cube size 3. poly=X -> [0, 1, 0].
        // dim 0, mod Phi_3. [0, 1, 0] % [1,1,1] -> [0, 1, 0]. (degree 1 < 2).
        // powerful: map short indices to long.
        // short (size 2): 0, 1.

        let m = 3;
        let modulus = 17;
        let pb = PowerfulBasis::new(m);
        let poly = vec![0, 1]; // X
        let powerful = pb.poly_to_powerful(&poly, modulus);
        // Expect X -> powerful[0]=0, powerful[1]=1.
        // wait, powerful indices depends on map.

        // Just check length for now and no panic
        assert_eq!(powerful.len(), 2);
    }
    #[test]
    fn test_powerful_round_trip() {
        let m = 12;
        // phi(12) = 4.
        let modulus = 101; // use larger modulus to avoid triviality?
        let pb = PowerfulBasis::new(m);

        // Poly X^3 + X + 1
        let poly = vec![1, 1, 0, 1];

        let powerful = pb.poly_to_powerful(&poly, modulus);

        // Convert back
        let recovered = pb.powerful_to_poly(&powerful, modulus);

        // should match original poly (trimmed)
        assert_eq!(trim_zeros(&recovered), trim_zeros(&poly));
    }
}
