//! Number-theoretic transform helpers.

use crate::arith;
use crate::modulus::Modulus;
use crate::numth;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NttError {
    InvalidDegree,
    DegreeTooLarge,
    ModulusNotPrime,
    NoRoot,
    NoInverse,
}

use crate::arith::MultiplyUIntModOperand;

#[derive(Debug, Clone)]
pub struct NttTables {
    degree: usize,
    modulus: Modulus,
    omega: u64,
    omega_inv: u64,
    psi: u64,
    psi_inv: u64,
    root_pows: Vec<MultiplyUIntModOperand>,
    inv_root_pows: Vec<MultiplyUIntModOperand>,
    inv_root_pows_scaled: Vec<MultiplyUIntModOperand>,
    inv_degree_shoup: MultiplyUIntModOperand,
    negacyclic: bool,
}

impl NttTables {
    pub fn new_cyclic(degree: usize, modulus: Modulus, root: u64) -> Result<Self, NttError> {
        if degree == 0 || !degree.is_power_of_two() {
            return Err(NttError::InvalidDegree);
        }
        if !modulus.is_prime() {
            return Err(NttError::ModulusNotPrime);
        }
        if !numth::is_primitive_root(root, degree as u64, modulus.value()) {
            return Err(NttError::NoRoot);
        }

        let inv_degree =
            numth::mod_inverse(degree as u64, modulus.value()).ok_or(NttError::NoInverse)?;
        let inv_degree_shoup = MultiplyUIntModOperand::new(inv_degree, modulus.value());

        let omega = root;
        let omega_inv = numth::mod_inverse(omega, modulus.value()).ok_or(NttError::NoInverse)?;
        let root_pows = build_root_pows_bitrev(omega, degree, modulus.value());
        let inv_root_pows = build_inv_root_pows_scrambled(omega_inv, degree, modulus.value());
        let inv_root_pows_scaled = inv_root_pows
            .iter()
            .map(|root| {
                let value = arith::mul_mod_u64(root.operand, inv_degree, modulus.value());
                MultiplyUIntModOperand::new(value, modulus.value())
            })
            .collect();

        Ok(Self {
            degree,
            modulus,
            omega,
            omega_inv,
            psi: 0,
            psi_inv: 0,
            root_pows,
            inv_root_pows,
            inv_root_pows_scaled,
            inv_degree_shoup,
            negacyclic: false,
        })
    }

    pub fn new_negacyclic(degree: usize, modulus: Modulus) -> Result<Self, NttError> {
        if degree == 0 || !degree.is_power_of_two() {
            return Err(NttError::InvalidDegree);
        }
        if !modulus.is_prime() {
            return Err(NttError::ModulusNotPrime);
        }

        let degree_u64 = u64::try_from(degree).map_err(|_| NttError::DegreeTooLarge)?;
        let doubled = degree_u64.checked_mul(2).ok_or(NttError::DegreeTooLarge)?;
        let psi = numth::try_primitive_root(doubled, modulus.value()).ok_or(NttError::NoRoot)?;
        let omega = arith::mul_mod(psi, psi, &modulus);

        let inv_degree =
            numth::mod_inverse(degree_u64, modulus.value()).ok_or(NttError::NoInverse)?;
        let inv_degree_shoup = MultiplyUIntModOperand::new(inv_degree, modulus.value());

        let omega_inv = numth::mod_inverse(omega, modulus.value()).ok_or(NttError::NoInverse)?;
        let psi_inv = numth::mod_inverse(psi, modulus.value()).ok_or(NttError::NoInverse)?;
        let root_pows = build_root_pows_bitrev(psi, degree, modulus.value());
        let inv_root_pows = build_inv_root_pows_scrambled(psi_inv, degree, modulus.value());
        let inv_root_pows_scaled = inv_root_pows
            .iter()
            .map(|root| {
                let value = arith::mul_mod_u64(root.operand, inv_degree, modulus.value());
                MultiplyUIntModOperand::new(value, modulus.value())
            })
            .collect();

        Ok(Self {
            degree,
            modulus,
            omega,
            omega_inv,
            psi,
            psi_inv,
            root_pows,
            inv_root_pows,
            inv_root_pows_scaled,
            inv_degree_shoup,
            negacyclic: true,
        })
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn modulus(&self) -> Modulus {
        self.modulus
    }

    pub fn omega(&self) -> u64 {
        self.omega
    }

    pub fn omega_inv(&self) -> u64 {
        self.omega_inv
    }

    pub fn psi(&self) -> u64 {
        self.psi
    }

    pub fn psi_inv(&self) -> u64 {
        self.psi_inv
    }

    pub fn hexl_root(&self) -> Option<u64> {
        if self.negacyclic {
            Some(self.psi)
        } else {
            None
        }
    }

    pub fn is_negacyclic(&self) -> bool {
        self.negacyclic
    }
}

#[cfg(feature = "hexl")]
use crate::backend::{HexlBackend, MathBackend};

pub fn ntt_forward(values: &mut [u64], tables: &NttTables) {
    #[cfg(feature = "hexl")]
    {
        HexlBackend.ntt_forward(values, tables);
    }
    #[cfg(not(feature = "hexl"))]
    {
        ntt_forward_native(values, tables);
    }
}

pub fn ntt_forward_native(values: &mut [u64], tables: &NttTables) {
    debug_assert_eq!(values.len(), tables.degree);
    if values.len() <= 1 {
        return;
    }

    let modulus = tables.modulus.value();
    // Fused strict reduction is enabled
    dwt_forward(values, &tables.root_pows, modulus, true);
}

/// Forward NTT with lazy range (values remain in [0, 4q)).
/// Mirrors SEAL's ntt_negacyclic_harvey_lazy behavior.
pub fn ntt_forward_lazy(values: &mut [u64], tables: &NttTables) {
    // Lazy NTT for now just calls native lazy implementation
    // HEXL doesn't explicitly expose "lazy" forward NTT in the bindings I saw?
    // backend/hexl.rs only has hexl_ntt_forward (which usually is strict or specific).
    // Let's check hexl.rs bindings again?
    // It has `output_mod_factor`.
    // For now, we use native lazy.
    ntt_forward_lazy_native(values, tables);
}

pub fn ntt_forward_lazy_native(values: &mut [u64], tables: &NttTables) {
    debug_assert_eq!(values.len(), tables.degree);
    if values.len() <= 1 {
        return;
    }

    let modulus = tables.modulus.value();
    dwt_forward(values, &tables.root_pows, modulus, false);
}

fn build_root_pows_bitrev(root: u64, degree: usize, modulus: u64) -> Vec<MultiplyUIntModOperand> {
    if degree == 0 {
        return Vec::new();
    }

    let log_n = degree.trailing_zeros();
    let mut pows = vec![MultiplyUIntModOperand::new(0, modulus); degree];
    let mut power = root;
    for i in 1..degree {
        let idx = reverse_bits(i as u64, log_n) as usize;
        pows[idx] = MultiplyUIntModOperand::new(power, modulus);
        power = arith::mul_mod_u64(power, root, modulus);
    }
    pows[0] = MultiplyUIntModOperand::new(1, modulus);
    pows
}

fn build_inv_root_pows_scrambled(
    inv_root: u64,
    degree: usize,
    modulus: u64,
) -> Vec<MultiplyUIntModOperand> {
    if degree == 0 {
        return Vec::new();
    }

    let log_n = degree.trailing_zeros();
    let mut pows = vec![MultiplyUIntModOperand::new(0, modulus); degree];
    let mut power = inv_root;
    for i in 1..degree {
        let idx = (reverse_bits((i - 1) as u64, log_n) + 1) as usize;
        pows[idx] = MultiplyUIntModOperand::new(power, modulus);
        power = arith::mul_mod_u64(power, inv_root, modulus);
    }
    pows[0] = MultiplyUIntModOperand::new(1, modulus);
    pows
}

#[inline(always)]
fn guard(value: u64, two_q: u64) -> u64 {
    let mask = ((value >= two_q) as u64).wrapping_neg();
    value.wrapping_sub(two_q & mask)
}

#[inline(always)]
fn mul_root(value: u64, root: &MultiplyUIntModOperand, modulus: u64) -> u64 {
    arith::mul_mod_shoup_lazy(value, root.operand, root.quotient, modulus)
}

#[inline(always)]
fn mul_scalar(value: u64, scalar: &MultiplyUIntModOperand, modulus: u64) -> u64 {
    arith::mul_mod_shoup_lazy(value, scalar.operand, scalar.quotient, modulus)
}

fn dwt_forward(values: &mut [u64], roots: &[MultiplyUIntModOperand], modulus: u64, strict: bool) {
    let n = values.len();
    if n <= 1 {
        return;
    }

    let two_q = modulus << 1;
    let val_ptr = values.as_mut_ptr();
    let mut root_idx = 0usize;

    let mut gap = n >> 1;
    let mut m = 1usize;

    unsafe {
        while m < (n >> 1) {
            let mut offset = 0usize;
            if gap < 4 {
                for _ in 0..m {
                    root_idx += 1;
                    let r = &*roots.as_ptr().add(root_idx);
                    let mut x_ptr = val_ptr.add(offset);
                    let mut y_ptr = x_ptr.add(gap);
                    for _ in 0..gap {
                        let u = guard(*x_ptr, two_q);
                        let v = mul_root(*y_ptr, r, modulus);
                        *x_ptr = u + v;
                        *y_ptr = u + two_q - v;
                        x_ptr = x_ptr.add(1);
                        y_ptr = y_ptr.add(1);
                    }
                    offset += gap << 1;
                }
            } else {
                for _ in 0..m {
                    root_idx += 1;
                    let r = &*roots.as_ptr().add(root_idx);
                    let mut x_ptr = val_ptr.add(offset);
                    let mut y_ptr = x_ptr.add(gap);
                    let mut j = 0usize;
                    while j < gap {
                        let u0 = guard(*x_ptr, two_q);
                        let v0 = mul_root(*y_ptr, r, modulus);
                        *x_ptr = u0 + v0;
                        *y_ptr = u0 + two_q - v0;

                        let u1 = guard(*x_ptr.add(1), two_q);
                        let v1 = mul_root(*y_ptr.add(1), r, modulus);
                        *x_ptr.add(1) = u1 + v1;
                        *y_ptr.add(1) = u1 + two_q - v1;

                        let u2 = guard(*x_ptr.add(2), two_q);
                        let v2 = mul_root(*y_ptr.add(2), r, modulus);
                        *x_ptr.add(2) = u2 + v2;
                        *y_ptr.add(2) = u2 + two_q - v2;

                        let u3 = guard(*x_ptr.add(3), two_q);
                        let v3 = mul_root(*y_ptr.add(3), r, modulus);
                        *x_ptr.add(3) = u3 + v3;
                        *y_ptr.add(3) = u3 + two_q - v3;

                        x_ptr = x_ptr.add(4);
                        y_ptr = y_ptr.add(4);
                        j += 4;
                    }
                    offset += gap << 1;
                }
            }
            gap >>= 1;
            m <<= 1;
        }

        // Final layer (gap = 1, m = n/2)
        let mut out_ptr = val_ptr;

        if strict {
            // Unrolled loop with reduction
            let mut k = 0;
            while k < m {
                // Process 4 pairs (8 elements) at a time if possible
                // m is usually large (2048).
                if k + 4 <= m {
                    // 1
                    root_idx += 1;
                    let r0 = &*roots.as_ptr().add(root_idx);
                    let u0 = guard(*out_ptr, two_q);
                    let v0 = mul_root(*out_ptr.add(1), r0, modulus);
                    let mut res0_0 = u0 + v0;
                    let mut res0_1 = u0 + two_q - v0;
                    if res0_0 >= two_q {
                        res0_0 -= two_q;
                    }
                    if res0_0 >= modulus {
                        res0_0 -= modulus;
                    }
                    if res0_1 >= two_q {
                        res0_1 -= two_q;
                    }
                    if res0_1 >= modulus {
                        res0_1 -= modulus;
                    }
                    *out_ptr = res0_0;
                    *out_ptr.add(1) = res0_1;
                    out_ptr = out_ptr.add(2);

                    // 2
                    root_idx += 1;
                    let r1 = &*roots.as_ptr().add(root_idx);
                    let u1 = guard(*out_ptr, two_q);
                    let v1 = mul_root(*out_ptr.add(1), r1, modulus);
                    let mut res1_0 = u1 + v1;
                    let mut res1_1 = u1 + two_q - v1;
                    if res1_0 >= two_q {
                        res1_0 -= two_q;
                    }
                    if res1_0 >= modulus {
                        res1_0 -= modulus;
                    }
                    if res1_1 >= two_q {
                        res1_1 -= two_q;
                    }
                    if res1_1 >= modulus {
                        res1_1 -= modulus;
                    }
                    *out_ptr = res1_0;
                    *out_ptr.add(1) = res1_1;
                    out_ptr = out_ptr.add(2);

                    // 3
                    root_idx += 1;
                    let r2 = &*roots.as_ptr().add(root_idx);
                    let u2 = guard(*out_ptr, two_q);
                    let v2 = mul_root(*out_ptr.add(1), r2, modulus);
                    let mut res2_0 = u2 + v2;
                    let mut res2_1 = u2 + two_q - v2;
                    if res2_0 >= two_q {
                        res2_0 -= two_q;
                    }
                    if res2_0 >= modulus {
                        res2_0 -= modulus;
                    }
                    if res2_1 >= two_q {
                        res2_1 -= two_q;
                    }
                    if res2_1 >= modulus {
                        res2_1 -= modulus;
                    }
                    *out_ptr = res2_0;
                    *out_ptr.add(1) = res2_1;
                    out_ptr = out_ptr.add(2);

                    // 4
                    root_idx += 1;
                    let r3 = &*roots.as_ptr().add(root_idx);
                    let u3 = guard(*out_ptr, two_q);
                    let v3 = mul_root(*out_ptr.add(1), r3, modulus);
                    let mut res3_0 = u3 + v3;
                    let mut res3_1 = u3 + two_q - v3;
                    if res3_0 >= two_q {
                        res3_0 -= two_q;
                    }
                    if res3_0 >= modulus {
                        res3_0 -= modulus;
                    }
                    if res3_1 >= two_q {
                        res3_1 -= two_q;
                    }
                    if res3_1 >= modulus {
                        res3_1 -= modulus;
                    }
                    *out_ptr = res3_0;
                    *out_ptr.add(1) = res3_1;
                    out_ptr = out_ptr.add(2);

                    k += 4;
                } else {
                    root_idx += 1;
                    let r = &*roots.as_ptr().add(root_idx);
                    let u = guard(*out_ptr, two_q);
                    let v = mul_root(*out_ptr.add(1), r, modulus);
                    let mut val0 = u + v;
                    let mut val1 = u + two_q - v;
                    if val0 >= two_q {
                        val0 -= two_q;
                    }
                    if val0 >= modulus {
                        val0 -= modulus;
                    }
                    if val1 >= two_q {
                        val1 -= two_q;
                    }
                    if val1 >= modulus {
                        val1 -= modulus;
                    }
                    *out_ptr = val0;
                    *out_ptr.add(1) = val1;
                    out_ptr = out_ptr.add(2);
                    k += 1;
                }
            }
        } else {
            // Lazy reduction (original logic, unrolled)
            let mut k = 0;
            while k < m {
                if k + 4 <= m {
                    // 1
                    root_idx += 1;
                    let r0 = &*roots.as_ptr().add(root_idx);
                    let u0 = guard(*out_ptr, two_q);
                    let v0 = mul_root(*out_ptr.add(1), r0, modulus);
                    *out_ptr = u0 + v0;
                    *out_ptr.add(1) = u0 + two_q - v0;
                    out_ptr = out_ptr.add(2);

                    // 2
                    root_idx += 1;
                    let r1 = &*roots.as_ptr().add(root_idx);
                    let u1 = guard(*out_ptr, two_q);
                    let v1 = mul_root(*out_ptr.add(1), r1, modulus);
                    *out_ptr = u1 + v1;
                    *out_ptr.add(1) = u1 + two_q - v1;
                    out_ptr = out_ptr.add(2);

                    // 3
                    root_idx += 1;
                    let r2 = &*roots.as_ptr().add(root_idx);
                    let u2 = guard(*out_ptr, two_q);
                    let v2 = mul_root(*out_ptr.add(1), r2, modulus);
                    *out_ptr = u2 + v2;
                    *out_ptr.add(1) = u2 + two_q - v2;
                    out_ptr = out_ptr.add(2);

                    // 4
                    root_idx += 1;
                    let r3 = &*roots.as_ptr().add(root_idx);
                    let u3 = guard(*out_ptr, two_q);
                    let v3 = mul_root(*out_ptr.add(1), r3, modulus);
                    *out_ptr = u3 + v3;
                    *out_ptr.add(1) = u3 + two_q - v3;
                    out_ptr = out_ptr.add(2);

                    k += 4;
                } else {
                    root_idx += 1;
                    let r = &*roots.as_ptr().add(root_idx);
                    let u = guard(*out_ptr, two_q);
                    let v = mul_root(*out_ptr.add(1), r, modulus);
                    *out_ptr = u + v;
                    *out_ptr.add(1) = u + two_q - v;
                    out_ptr = out_ptr.add(2);
                    k += 1;
                }
            }
        }
    }
}

fn dwt_inverse(
    values: &mut [u64],
    roots: &[MultiplyUIntModOperand],
    roots_scaled: &[MultiplyUIntModOperand],
    modulus: u64,
    scalar: &MultiplyUIntModOperand,
) {
    let n = values.len();
    if n <= 1 {
        return;
    }

    let two_q = modulus << 1;
    let val_ptr = values.as_mut_ptr();
    let mut root_idx = 0usize;

    let mut gap = 1usize;
    let mut m = n >> 1;

    unsafe {
        while m > 1 {
            let mut offset = 0usize;
            if gap < 4 {
                for _ in 0..m {
                    root_idx += 1;
                    let r = &*roots.as_ptr().add(root_idx);
                    let mut x_ptr = val_ptr.add(offset);
                    let mut y_ptr = x_ptr.add(gap);
                    for _ in 0..gap {
                        let u = *x_ptr;
                        let v = *y_ptr;
                        *x_ptr = guard(u.wrapping_add(v), two_q);
                        *y_ptr = mul_root(u.wrapping_add(two_q).wrapping_sub(v), r, modulus);
                        x_ptr = x_ptr.add(1);
                        y_ptr = y_ptr.add(1);
                    }
                    offset += gap << 1;
                }
            } else {
                for _ in 0..m {
                    root_idx += 1;
                    let r = &*roots.as_ptr().add(root_idx);
                    let mut x_ptr = val_ptr.add(offset);
                    let mut y_ptr = x_ptr.add(gap);
                    let mut j = 0usize;
                    while j < gap {
                        let u0 = *x_ptr;
                        let v0 = *y_ptr;
                        *x_ptr = guard(u0.wrapping_add(v0), two_q);
                        *y_ptr = mul_root(u0.wrapping_add(two_q).wrapping_sub(v0), r, modulus);

                        let u1 = *x_ptr.add(1);
                        let v1 = *y_ptr.add(1);
                        *x_ptr.add(1) = guard(u1.wrapping_add(v1), two_q);
                        *y_ptr.add(1) =
                            mul_root(u1.wrapping_add(two_q).wrapping_sub(v1), r, modulus);

                        let u2 = *x_ptr.add(2);
                        let v2 = *y_ptr.add(2);
                        *x_ptr.add(2) = guard(u2.wrapping_add(v2), two_q);
                        *y_ptr.add(2) =
                            mul_root(u2.wrapping_add(two_q).wrapping_sub(v2), r, modulus);

                        let u3 = *x_ptr.add(3);
                        let v3 = *y_ptr.add(3);
                        *x_ptr.add(3) = guard(u3.wrapping_add(v3), two_q);
                        *y_ptr.add(3) =
                            mul_root(u3.wrapping_add(two_q).wrapping_sub(v3), r, modulus);

                        x_ptr = x_ptr.add(4);
                        y_ptr = y_ptr.add(4);
                        j += 4;
                    }
                    offset += gap << 1;
                }
            }
            gap <<= 1;
            m >>= 1;
        }

        root_idx += 1;
        let scaled_r = &*roots_scaled.as_ptr().add(root_idx);
        let mut x_ptr = val_ptr;
        let mut y_ptr = val_ptr.add(gap);
        if gap < 4 {
            for _ in 0..gap {
                let u = guard(*x_ptr, two_q);
                let v = *y_ptr;
                let sum = guard(u.wrapping_add(v), two_q);
                *x_ptr = mul_scalar(sum, scalar, modulus);
                *y_ptr = mul_root(u.wrapping_add(two_q).wrapping_sub(v), scaled_r, modulus);
                x_ptr = x_ptr.add(1);
                y_ptr = y_ptr.add(1);
            }
        } else {
            let mut j = 0usize;
            while j < gap {
                let u0 = guard(*x_ptr, two_q);
                let v0 = *y_ptr;
                let sum0 = guard(u0.wrapping_add(v0), two_q);
                *x_ptr = mul_scalar(sum0, scalar, modulus);
                *y_ptr = mul_root(u0.wrapping_add(two_q).wrapping_sub(v0), scaled_r, modulus);

                let u1 = guard(*x_ptr.add(1), two_q);
                let v1 = *y_ptr.add(1);
                let sum1 = guard(u1.wrapping_add(v1), two_q);
                *x_ptr.add(1) = mul_scalar(sum1, scalar, modulus);
                *y_ptr.add(1) =
                    mul_root(u1.wrapping_add(two_q).wrapping_sub(v1), scaled_r, modulus);

                let u2 = guard(*x_ptr.add(2), two_q);
                let v2 = *y_ptr.add(2);
                let sum2 = guard(u2.wrapping_add(v2), two_q);
                *x_ptr.add(2) = mul_scalar(sum2, scalar, modulus);
                *y_ptr.add(2) =
                    mul_root(u2.wrapping_add(two_q).wrapping_sub(v2), scaled_r, modulus);

                let u3 = guard(*x_ptr.add(3), two_q);
                let v3 = *y_ptr.add(3);
                let sum3 = guard(u3.wrapping_add(v3), two_q);
                *x_ptr.add(3) = mul_scalar(sum3, scalar, modulus);
                *y_ptr.add(3) =
                    mul_root(u3.wrapping_add(two_q).wrapping_sub(v3), scaled_r, modulus);

                x_ptr = x_ptr.add(4);
                y_ptr = y_ptr.add(4);
                j += 4;
            }
        }
    }
}

pub fn ntt_inverse(values: &mut [u64], tables: &NttTables) {
    #[cfg(feature = "hexl")]
    {
        HexlBackend.ntt_inverse(values, tables);
    }
    #[cfg(not(feature = "hexl"))]
    {
        ntt_inverse_native(values, tables);
    }
}

pub fn ntt_inverse_native(values: &mut [u64], tables: &NttTables) {
    debug_assert_eq!(values.len(), tables.degree);
    if values.len() <= 1 {
        return;
    }

    let modulus = tables.modulus.value();
    dwt_inverse(
        values,
        &tables.inv_root_pows,
        &tables.inv_root_pows_scaled,
        modulus,
        &tables.inv_degree_shoup,
    );

    // Reduce to [0, q) for coefficient-domain operations.
    for value in values.iter_mut() {
        if *value >= modulus {
            *value -= modulus;
        }
    }
}

/// Inverse NTT with lazy range (values remain in [0, 2q)).
/// Mirrors SEAL's inverse_ntt_negacyclic_harvey_lazy behavior.
pub fn ntt_inverse_lazy(values: &mut [u64], tables: &NttTables) {
    ntt_inverse_lazy_native(values, tables);
}

pub fn ntt_inverse_lazy_native(values: &mut [u64], tables: &NttTables) {
    debug_assert_eq!(values.len(), tables.degree);
    if values.len() <= 1 {
        return;
    }

    let modulus = tables.modulus.value();
    dwt_inverse(
        values,
        &tables.inv_root_pows,
        &tables.inv_root_pows_scaled,
        modulus,
        &tables.inv_degree_shoup,
    );
}

pub fn ntt_mul_accumulate(acc: &mut [u64], a: &[u64], b: &[u64], modulus: &Modulus) {
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    for ((accum, lhs), rhs) in acc.iter_mut().zip(a.iter()).zip(b.iter()) {
        let prod = arith::mul_mod(*lhs, *rhs, modulus);
        *accum = arith::add_mod(*accum, prod, modulus.value());
    }
}

pub fn galois_element_from_step(step: i32, degree: usize) -> u32 {
    let m = (degree * 2) as u64;
    let mut step = step;
    let sw = step >> 31;
    step = (step ^ sw) - sw; // abs

    let mut val = 3u64;
    let mut res = 1u64;
    while step > 0 {
        if step & 1 != 0 {
            res = (res * val) % m;
        }
        val = (val * val) % m;
        step >>= 1;
    }
    if sw != 0 {
        res = crate::numth::mod_inverse(res, m).expect("No inverse");
    }
    res as u32
}

pub fn galois_automorphism_map(galois_elt: u32, degree: usize) -> Vec<usize> {
    let mut map = vec![0usize; degree];
    let log_n = degree.trailing_zeros();
    let m = (degree * 2) as u64;
    let galois_elt = galois_elt as u64;

    for i in 0..degree {
        let i_rev = reverse_bits(i as u64, log_n);
        let index_raw = 2 * i_rev + 1;
        let index_mapped = (index_raw * galois_elt) & (m - 1);
        let dest_rev = (index_mapped - 1) >> 1;
        let dest = reverse_bits(dest_rev, log_n) as usize;
        map[i] = dest;
    }
    map
}

pub fn apply_galois_ntt(values: &mut [u64], map: &[usize]) {
    let temp = values.to_vec();
    for (i, &dest) in map.iter().enumerate() {
        values[dest] = temp[i];
    }
}

fn reverse_bits(value: u64, bits: u32) -> u64 {
    if bits == 0 {
        return 0;
    }
    value.reverse_bits() >> (64 - bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulus::Modulus;

    #[test]
    fn negacyclic_roundtrip() {
        let modulus = Modulus::new(17).expect("modulus");
        let tables = NttTables::new_negacyclic(8, modulus).expect("tables");
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let original = data.clone();
        ntt_forward(&mut data, &tables);
        ntt_inverse(&mut data, &tables);
        assert_eq!(data, original);
    }

    #[test]
    fn negacyclic_roundtrip_lazy_matches() {
        let modulus = Modulus::new(17).expect("modulus");
        let tables = NttTables::new_negacyclic(8, modulus).expect("tables");
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let original = data.clone();

        ntt_forward_lazy(&mut data, &tables);
        ntt_inverse_lazy(&mut data, &tables);
        let q = modulus.value();
        for value in data.iter_mut() {
            while *value >= q {
                *value -= q;
            }
        }
        assert_eq!(data, original);
    }

    #[test]
    fn ntt_mul_accumulate_basic() {
        let modulus = Modulus::new(97).expect("modulus");
        let mut acc = vec![1, 2, 3];
        let a = vec![10, 20, 30];
        let b = vec![3, 4, 5];
        ntt_mul_accumulate(&mut acc, &a, &b, &modulus);
        assert_eq!(acc, vec![31, 82, 56]);
    }

    #[test]
    fn tables_reject_invalid_degree() {
        let modulus = Modulus::new(17).expect("modulus");
        assert!(matches!(
            NttTables::new_negacyclic(3, modulus),
            Err(NttError::InvalidDegree)
        ));
    }

    #[test]
    fn negacyclic_convolution_matches_naive() {
        let modulus = Modulus::new(17).expect("modulus");
        let tables = NttTables::new_negacyclic(8, modulus).expect("tables");
        let a = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let b = vec![8u64, 7, 6, 5, 4, 3, 2, 1];

        let mut a_ntt = a.clone();
        let mut b_ntt = b.clone();
        ntt_forward(&mut a_ntt, &tables);
        ntt_forward(&mut b_ntt, &tables);
        for (lhs, rhs) in a_ntt.iter_mut().zip(b_ntt.iter()) {
            *lhs = arith::mul_mod(*lhs, *rhs, &modulus);
        }
        ntt_inverse(&mut a_ntt, &tables);

        let naive = negacyclic_naive(&a, &b, modulus.value());
        assert_eq!(a_ntt, naive);
    }

    fn negacyclic_naive(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        let n = a.len();
        let mut out = vec![0u64; n];
        for i in 0..n {
            for j in 0..n {
                let idx = (i + j) % n;
                let sign = if i + j >= n { modulus - 1 } else { 1 };
                let term =
                    (u128::from(a[i]) * u128::from(b[j]) * u128::from(sign)) % u128::from(modulus);
                let acc = u128::from(out[idx]) + term;
                out[idx] = (acc % u128::from(modulus)) as u64;
            }
        }
        out
    }
}
