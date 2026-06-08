//! Residue number system context.

use crate::arith;
use crate::bigint::BigUint;
use crate::modulus::{Modulus, ModulusError};
use crate::numth;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RnsError {
    EmptyBase,
    NonCoprime,
    InvalidBase,
    InvalidResidues,
    MissingInverse,
    BaseProductTooLarge,
    InvalidMtilde,
    OpenFheDisabled,
    PrimeNotFound,
    Modulus(ModulusError),
}

impl From<ModulusError> for RnsError {
    fn from(err: ModulusError) -> Self {
        Self::Modulus(err)
    }
}

#[derive(Clone, Debug)]
pub struct RnsBase {
    moduli: Vec<Modulus>,
    base_prod: BigUint,
    base_prod_u128: Option<u128>,
    punctured_prod: Vec<BigUint>,
    inv_punctured_prod_mod_base: Vec<u64>,
    inv_punctured_prod_mod_base_shoup: Vec<u64>, // Stores quotient for Shoup mul
}

impl RnsBase {
    pub fn new(moduli: Vec<Modulus>) -> Result<Self, RnsError> {
        if moduli.is_empty() {
            return Err(RnsError::EmptyBase);
        }

        for i in 0..moduli.len() {
            for j in i + 1..moduli.len() {
                if numth::gcd(moduli[i].value(), moduli[j].value()) != 1 {
                    return Err(RnsError::NonCoprime);
                }
            }
        }

        let mut base_prod = BigUint::from_u64(1);
        for modulus in &moduli {
            base_prod = base_prod.mul_u64(modulus.value());
        }
        let base_prod_u128 = base_prod.to_u128();

        let mut punctured_prod = Vec::with_capacity(moduli.len());
        let mut inv_punctured_prod_mod_base = Vec::with_capacity(moduli.len());
        for modulus in &moduli {
            let (quotient, remainder) = base_prod.div_mod_u64(modulus.value());
            if remainder != 0 {
                return Err(RnsError::NonCoprime);
            }
            let q_hat_mod = quotient.mod_u64(modulus.value());
            let inv =
                numth::mod_inverse(q_hat_mod, modulus.value()).ok_or(RnsError::MissingInverse)?;
            punctured_prod.push(quotient);
            inv_punctured_prod_mod_base.push(inv);
        }

        let inv_punctured_prod_mod_base_shoup = inv_punctured_prod_mod_base
            .iter()
            .zip(moduli.iter())
            .map(|(&inv, m)| {
                // Compute floor(inv * 2^64 / m)
                ((u128::from(inv) << 64) / u128::from(m.value())) as u64
            })
            .collect();

        Ok(Self {
            moduli,
            base_prod,
            base_prod_u128,
            punctured_prod,
            inv_punctured_prod_mod_base,
            inv_punctured_prod_mod_base_shoup,
        })
    }

    pub fn from_values(values: Vec<u64>) -> Result<Self, RnsError> {
        let mut moduli = Vec::with_capacity(values.len());
        for value in values {
            moduli.push(Modulus::new(value)?);
        }
        Self::new(moduli)
    }

    pub fn moduli(&self) -> &[Modulus] {
        &self.moduli
    }

    pub fn base_prod(&self) -> &BigUint {
        &self.base_prod
    }

    pub fn base_prod_u128(&self) -> Option<u128> {
        self.base_prod_u128
    }

    pub fn punctured_prod(&self) -> &[BigUint] {
        &self.punctured_prod
    }

    pub fn inv_punctured_prod_mod_base(&self) -> &[u64] {
        &self.inv_punctured_prod_mod_base
    }

    pub fn inv_punctured_prod_mod_base_shoup(&self) -> &[u64] {
        &self.inv_punctured_prod_mod_base_shoup
    }

    pub fn len(&self) -> usize {
        self.moduli.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moduli.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct RnsContext {
    base: RnsBase,
    base_prod_u128: Option<u128>,
}

impl RnsContext {
    pub fn new(base: RnsBase) -> Self {
        let base_prod_u128 = base.base_prod_u128();
        Self {
            base,
            base_prod_u128,
        }
    }

    pub fn base(&self) -> &RnsBase {
        &self.base
    }

    pub fn base_prod_u128(&self) -> Option<u128> {
        self.base_prod_u128
    }

    pub fn decompose_u128(&self, value: u128) -> Vec<u64> {
        self.base
            .moduli()
            .iter()
            .map(|modulus| (value % u128::from(modulus.value())) as u64)
            .collect()
    }

    pub fn compose_u128(&self, residues: &[u64]) -> Result<u128, RnsError> {
        if residues.len() != self.base.len() {
            return Err(RnsError::InvalidResidues);
        }
        let base_prod = self.base_prod_u128.ok_or(RnsError::BaseProductTooLarge)?;

        let mut acc = 0u128;
        for (i, residue) in residues.iter().enumerate() {
            let modulus = self.base.moduli()[i].value() as u128;
            let q_hat = base_prod / modulus;
            let q_hat_mod = (q_hat % modulus) as u64;
            let inv =
                numth::mod_inverse(q_hat_mod, modulus as u64).ok_or(RnsError::MissingInverse)?;
            let term = (u128::from(*residue) * u128::from(inv)) % modulus;
            acc = (acc + term * q_hat) % base_prod;
        }

        Ok(acc)
    }
}

#[derive(Clone, Debug)]
pub struct BaseConverter {
    ibase: RnsBase,
    obase: RnsBase,
    base_change_matrix: Vec<u64>,
}

impl BaseConverter {
    pub const BLOCK_SIZE: usize = 32;

    pub fn new(ibase: RnsBase, obase: RnsBase) -> Result<Self, RnsError> {
        if ibase.is_empty() || obase.is_empty() {
            return Err(RnsError::EmptyBase);
        }

        let ibase_len = ibase.len();
        let obase_len = obase.len();
        let mut base_change_matrix = vec![0u64; obase_len * ibase_len];
        for (j, obase_modulus) in obase.moduli().iter().enumerate() {
            let row_offset = j * ibase_len;
            for (i, punctured) in ibase.punctured_prod().iter().enumerate() {
                base_change_matrix[row_offset + i] = punctured.mod_u64(obase_modulus.value());
            }
        }

        Ok(Self {
            ibase,
            obase,
            base_change_matrix,
        })
    }

    pub fn ibase(&self) -> &RnsBase {
        &self.ibase
    }

    pub fn obase(&self) -> &RnsBase {
        &self.obase
    }

    pub fn fast_convert_into(
        &self,
        input: &[u64],
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        self.fast_convert_array_into(input, 1, output, scratch)
    }

    pub fn fast_convert(&self, input: &[u64]) -> Result<Vec<u64>, RnsError> {
        let mut output = vec![0u64; self.obase.len()];
        // fast_convert_array_into requires scratch of at least ibase_len * BLOCK_SIZE
        // for the general (ibase_len > 1) path.
        let mut scratch = vec![0u64; self.ibase.len() * Self::BLOCK_SIZE];
        self.fast_convert_into(input, &mut output, &mut scratch)?;
        Ok(output)
    }

    pub fn fast_convert_array_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let ibase_len = self.ibase.len();
        let obase_len = self.obase.len();

        if input.len() != count * ibase_len {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != count * obase_len {
            return Err(RnsError::InvalidResidues);
        }
        // Minimal scratch is ibase_len (for 1 coeff).
        if scratch.len() < ibase_len {
            return Err(RnsError::InvalidResidues);
        }

        let ibase_moduli = self.ibase.moduli();
        let ibase_invs = self.ibase.inv_punctured_prod_mod_base();
        let ibase_invs_shoup = self.ibase.inv_punctured_prod_mod_base_shoup();
        let obase_moduli = self.obase.moduli();
        let matrix = &self.base_change_matrix;
        let matrix_ptr = matrix.as_ptr();

        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();

        // Optimized path for single modulus input (common in BFV/CKKS RNS decomposition)
        if ibase_len == 1 {
            // input is [val0, val1, ...]
            // output is [val0%p0, val0%p1, ..., val1%p0, val1%p1, ...] or
            // output is usually column-major or row-major?
            // Input layout: count * ibase_len. (flat: c0_q0, c1_q0...)
            // Output layout: count * obase_len. (flat: c0_p0, c0_p1, ..., c1_p0...)
            // Wait, rns context usually stores [q0_c0, q0_c1..., q1_c0... ] (Limb-major).
            // Let's check `decompose_u128`:
            // `input.len() != count * ibase_len`.
            // In OpenFHE/SILENT, Poly data is Limb-Major.
            // But `fast_convert_array` takes `&[u64]`.
            // Usage in `ops.rs`:
            // `part.approx_switch_into(coeff_buf, degree, comp_buf, temp)`
            // `coeff_buf` comes from `slice::split_at`. `digit_coeff` is `size_q * degree`.
            // `dst.copy_from_slice`.
            // INTT output in `coeff_buf`.
            // The layout of `coeff_buf` in `ops.rs`:
            // It splits `digit_coeff` into chunks of `degree`.
            // And copies `c2_ntt` limbs into it.
            // `c2_ntt` is flattened Poly: [Limb0, Limb1...].
            // So `coeff_buf` contains [Limb_start, Limb_start+1...].
            // This is Limb-Major.
            // BUT `fast_convert_array_into` documentation/logic:
            // `off = i * count + coeff`.
            // `i` is modulus index. `coeff` is coefficient index.
            // Input: [Mod0_Coeff0, Mod0_Coeff1..., Mod1_Coeff0...]
            // So Limb-Major. Correct.
            //
            // Output: [OutMod0_AllCoeffs, OutMod1_AllCoeffs...].

            let modulus_in = ibase_moduli[0].value();

            // Cast pointers to usize to pass to threads (Send+Sync)
            let in_addr = input_ptr as usize;
            let out_addr = output_ptr as usize;

            // Parallelize over output moduli only if worth it
            let use_parallel = obase_moduli.len() >= 4;
            if use_parallel {
                obase_moduli
                    .iter()
                    .enumerate()
                    .for_each(|(j, modulus_out_struct)| {
                        let modulus_out = modulus_out_struct.value();
                        let out_off = j * count;
                        let in_slice =
                            unsafe { std::slice::from_raw_parts(in_addr as *const u64, count) };
                        let out_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out_addr as *mut u64).add(out_off),
                                count,
                            )
                        };

                        if modulus_in <= modulus_out {
                            out_slice.copy_from_slice(in_slice);
                        } else {
                            for (out, &val) in out_slice.iter_mut().zip(in_slice.iter()) {
                                if val >= modulus_out {
                                    if val < 2 * modulus_out {
                                        *out = val - modulus_out;
                                    } else {
                                        *out = val % modulus_out;
                                    }
                                } else {
                                    *out = val;
                                }
                            }
                        }
                    });
            } else {
                // Serial execution for small bases (e.g. P->Q where Q has 2 moduli)
                // This avoids rayon overhead which can be 10-20us.
                let in_slice = unsafe { std::slice::from_raw_parts(in_addr as *const u64, count) };
                for (j, modulus_out_struct) in obase_moduli.iter().enumerate() {
                    let modulus_out = modulus_out_struct.value();
                    let out_off = j * count;
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut((out_addr as *mut u64).add(out_off), count)
                    };

                    if modulus_in <= modulus_out {
                        out_slice.copy_from_slice(in_slice);
                    } else {
                        for (out, &val) in out_slice.iter_mut().zip(in_slice.iter()) {
                            if val >= modulus_out {
                                if val < 2 * modulus_out {
                                    *out = val - modulus_out;
                                } else {
                                    *out = val % modulus_out;
                                }
                            } else {
                                *out = val;
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
        // General Case: Serial execution (optimized for small tasks)
        // Send pointers.
        let in_addr = input_ptr as usize;
        let out_addr = output_ptr as usize;
        let matrix_addr = matrix_ptr as usize;

        // Ensure scratch is large enough for one block
        if scratch.len() < ibase_len * Self::BLOCK_SIZE {
            return Err(RnsError::InvalidResidues);
        }

        // We process in blocks of BLOCK_SIZE directly.
        // No need for "chunks" of 1024 unless for parallelism.
        // Since we are serial, we just loop blocks.

        let mut c_blk = 0;
        while c_blk < count {
            let blk_len = std::cmp::min(Self::BLOCK_SIZE, count - c_blk);
            // scratch usage: just pass scratch pointer
            unsafe {
                self.convert_range_unsafe(
                    in_addr as *const u64,
                    out_addr as *mut u64,
                    scratch.as_mut_ptr(), // Use provided scratch
                    count,
                    c_blk,
                    blk_len, // Pass actual length of this block
                    ibase_len,
                    obase_len,
                    ibase_moduli,
                    ibase_invs,
                    ibase_invs_shoup,
                    obase_moduli,
                    matrix_addr as *const u64,
                );
            }
            c_blk += blk_len;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn convert_range_unsafe(
        &self,
        input_ptr: *const u64,
        output_ptr: *mut u64,
        temp_ptr: *mut u64,
        total_count: usize,
        chunk_start: usize,
        chunk_len: usize,
        ibase_len: usize,
        obase_len: usize,
        ibase_moduli: &[Modulus],
        ibase_invs: &[u64],
        ibase_invs_shoup: &[u64],
        obase_moduli: &[Modulus],
        matrix_ptr: *const u64,
    ) {
        // SAFETY: the caller guarantees that the raw input/output/temp pointers
        // are valid for the full converted range, that the temporary buffer has
        // at least `ibase_len * BLOCK_SIZE` elements, and that matrix/modulus
        // slices are sized consistently with `ibase_len` / `obase_len`.
        unsafe {
            let mut c_blk = 0;
            while c_blk < chunk_len {
                let blk_len = std::cmp::min(Self::BLOCK_SIZE, chunk_len - c_blk);
                let global_offset = chunk_start + c_blk;

                // Phase 1: Input * Inverse -> Temp
                // Use lazy reduction: result in [0, 2q).
                for i in 0..ibase_len {
                    let inv = *ibase_invs.get_unchecked(i);
                    let inv_shoup = *ibase_invs_shoup.get_unchecked(i);
                    let modulus = ibase_moduli.get_unchecked(i).value();
                    let in_off_base = i * total_count + global_offset;
                    let temp_off_base = i * Self::BLOCK_SIZE;

                    for c in 0..blk_len {
                        let val = *input_ptr.add(in_off_base + c);
                        if inv == 1 {
                            *temp_ptr.add(temp_off_base + c) = arith::reduce_once(val, modulus);
                        } else {
                            *temp_ptr.add(temp_off_base + c) =
                                arith::mul_mod_shoup_lazy(val, inv, inv_shoup, modulus);
                        }
                    }
                }

                // Phase 2: Matrix multiplication.
                for j in 0..obase_len {
                    let modulus = obase_moduli.get_unchecked(j);
                    let row_ptr = matrix_ptr.add(j * ibase_len);
                    let out_off_base = j * total_count + global_offset;

                    for c in 0..blk_len {
                        let mut acc_lo: u128 = 0;
                        let mut acc_hi: u128 = 0;

                        let mut i = 0;
                        while i + 8 <= ibase_len {
                            let t0 = *temp_ptr.add(i * Self::BLOCK_SIZE + c) as u128;
                            let m0 = *row_ptr.add(i) as u128;
                            let t1 = *temp_ptr.add((i + 1) * Self::BLOCK_SIZE + c) as u128;
                            let m1 = *row_ptr.add(i + 1) as u128;
                            let t2 = *temp_ptr.add((i + 2) * Self::BLOCK_SIZE + c) as u128;
                            let m2 = *row_ptr.add(i + 2) as u128;
                            let t3 = *temp_ptr.add((i + 3) * Self::BLOCK_SIZE + c) as u128;
                            let m3 = *row_ptr.add(i + 3) as u128;

                            let t4 = *temp_ptr.add((i + 4) * Self::BLOCK_SIZE + c) as u128;
                            let m4 = *row_ptr.add(i + 4) as u128;
                            let t5 = *temp_ptr.add((i + 5) * Self::BLOCK_SIZE + c) as u128;
                            let m5 = *row_ptr.add(i + 5) as u128;
                            let t6 = *temp_ptr.add((i + 6) * Self::BLOCK_SIZE + c) as u128;
                            let m6 = *row_ptr.add(i + 6) as u128;
                            let t7 = *temp_ptr.add((i + 7) * Self::BLOCK_SIZE + c) as u128;
                            let m7 = *row_ptr.add(i + 7) as u128;

                            acc_lo += t0 * m0 + t1 * m1 + t2 * m2 + t3 * m3;
                            acc_hi += t4 * m4 + t5 * m5 + t6 * m6 + t7 * m7;
                            i += 8;
                        }

                        let mut acc = acc_lo + acc_hi;
                        while i < ibase_len {
                            acc += (*temp_ptr.add(i * Self::BLOCK_SIZE + c) as u128)
                                * (*row_ptr.add(i) as u128);
                            i += 1;
                        }
                        *output_ptr.add(out_off_base + c) = modulus.reduce_u128(acc);
                    }
                }
                c_blk += Self::BLOCK_SIZE;
            }
        }
    }

    pub fn fast_convert_array(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        if input.len() != count * self.ibase.len() {
            return Err(RnsError::InvalidResidues);
        }

        let mut output = vec![0u64; count * self.obase.len()];
        let mut scratch = vec![0u64; self.ibase.len() * 32];
        self.fast_convert_array_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    pub fn exact_convert_array(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        if self.obase.len() != 1 {
            return Err(RnsError::InvalidBase);
        }
        if input.len() != count * self.ibase.len() {
            return Err(RnsError::InvalidResidues);
        }
        let plain_modulus = self.obase.moduli()[0].value();
        let rounder = RnsRounder::new(self.ibase.clone(), plain_modulus)?;
        rounder.round_array(input, count)
    }

    pub fn exact_convert_array_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        if self.obase.len() != 1 {
            return Err(RnsError::InvalidBase);
        }
        if input.len() != count * self.ibase.len() {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != count {
            return Err(RnsError::InvalidResidues);
        }
        let plain_modulus = self.obase.moduli()[0].value();
        let rounder = RnsRounder::new(self.ibase.clone(), plain_modulus)?;
        rounder.round_array_into(input, count, output, scratch)
    }
}

#[derive(Clone, Debug)]
pub struct RnsRounder {
    ibase: RnsBase,
    base_prod: BigUint,
    base_prod_half: BigUint,
    plain_modulus: u64,
}

impl RnsRounder {
    pub fn new(base: RnsBase, plain_modulus: u64) -> Result<Self, RnsError> {
        if plain_modulus == 0 {
            return Err(RnsError::InvalidBase);
        }
        let (base_prod_half, _) = base.base_prod().div_mod_u64(2);
        Ok(Self {
            ibase: base.clone(),
            base_prod: base.base_prod().clone(),
            base_prod_half,
            plain_modulus,
        })
    }

    pub fn round(&self, residues: &[u64]) -> Result<u64, RnsError> {
        let value = self.compose_big(residues)?;
        let mut numerator = value.mul_u64(self.plain_modulus);
        numerator.add_assign(&self.base_prod_half);
        let mut quotient = div_big_by_big_small(&numerator, &self.base_prod, self.plain_modulus);
        if quotient >= self.plain_modulus {
            quotient = quotient.wrapping_sub(self.plain_modulus);
        }
        Ok(quotient)
    }

    pub fn round_array(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        if input.len() != count * self.ibase.len() {
            return Err(RnsError::InvalidResidues);
        }
        let mut output = vec![0u64; count];
        let mut scratch = vec![0u64; self.ibase.len()];
        self.round_array_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    pub fn round_array_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        if input.len() != count * self.ibase.len() {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != count {
            return Err(RnsError::InvalidResidues);
        }

        // Optimization: if Q fits in u128, use fast u128 arithmetic
        // Optimization: if Q fits in u128, use fast u128 arithmetic
        if let Some(q) = self.ibase.base_prod_u128() {
            let q_bits = 128 - q.leading_zeros();
            let t_bits = 64 - self.plain_modulus.leading_zeros();

            if q_bits + t_bits + 1 <= 128 {
                const CHUNK_SIZE: usize = 512;
                let mut acc = [0u128; CHUNK_SIZE];
                let moduli = self.ibase.moduli();
                let invs = self.ibase.inv_punctured_prod_mod_base();
                let invs_shoup = self.ibase.inv_punctured_prod_mod_base_shoup();

                let q_half = q / 2;
                let t = self.plain_modulus as u128;
                let size = self.ibase.len();

                for chunk_start in (0..count).step_by(CHUNK_SIZE) {
                    let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, count);
                    let chunk_len = chunk_end - chunk_start;

                    // Clear accumulator
                    acc[..chunk_len].fill(0);

                    // Accumulate CRT components: sum( x_i * inv_i * q_hat_i )
                    for i in 0..size {
                        // Constant for this modulus
                        let m = moduli[i].value(); // u64
                        let inv = invs[i]; // u64
                        let inv_shoup = invs_shoup[i];
                        let q_hat = q / (m as u128); // u128

                        // Process chunk
                        let input_ptr = unsafe { input.as_ptr().add(i * count + chunk_start) };

                        for k in 0..chunk_len {
                            let val = unsafe { *input_ptr.add(k) };
                            // term = (val * inv % m) * q_hat
                            let term_residue = arith::mul_mod_shoup(val, inv, inv_shoup, m);
                            let term = (term_residue as u128) * q_hat;

                            unsafe {
                                *acc.get_unchecked_mut(k) += term;
                            }
                        }
                    }

                    // Final reconstruction and rounding
                    for k in 0..chunk_len {
                        let mut val = unsafe { *acc.get_unchecked(k) };
                        // val is now Sum(x_i * inv_i * q_hat_i)
                        // It CAN be > Q. Ideally val %= Q.
                        // But we can do `val % Q` at the end once.
                        val %= q;

                        // Rounding: (val * t + q/2) / q
                        let numerator = val * t + q_half;
                        let rounded = numerator / q;

                        // Result mod t
                        unsafe {
                            *output.get_unchecked_mut(chunk_start + k) = (rounded % t) as u64;
                        }
                    }
                }
                return Ok(());
            }
        }

        if scratch.len() < self.ibase.len() {
            return Err(RnsError::InvalidResidues);
        }
        let residues = &mut scratch[..self.ibase.len()];
        for coeff in 0..count {
            for i in 0..self.ibase.len() {
                residues[i] = input[i * count + coeff];
            }
            output[coeff] = self.round(residues)?;
        }
        Ok(())
    }

    fn compose_big(&self, residues: &[u64]) -> Result<BigUint, RnsError> {
        if residues.len() != self.ibase.len() {
            return Err(RnsError::InvalidResidues);
        }
        let mut value = BigUint::zero();
        for i in 0..self.ibase.len() {
            let modulus = &self.ibase.moduli()[i];
            let inv = self.ibase.inv_punctured_prod_mod_base()[i];
            let temp = arith::mul_mod(residues[i], inv, modulus);
            let term = self.ibase.punctured_prod()[i].mul_u64(temp);
            value.add_assign(&term);
        }

        Ok(mod_big_small(
            &value,
            &self.base_prod,
            self.ibase.len() as u64,
        ))
    }
}

fn div_big_by_big_small(numerator: &BigUint, denominator: &BigUint, max_quot: u64) -> u64 {
    let mut low = 0u64;
    let mut high = max_quot;
    let mut best = 0u64;

    while low <= high {
        let mid = low + ((high - low) >> 1);
        let prod = denominator.mul_u64(mid);
        match prod.cmp(numerator) {
            std::cmp::Ordering::Greater => {
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            }
            _ => {
                best = mid;
                low = mid + 1;
            }
        }
    }

    best
}

fn mod_big_small(value: &BigUint, modulus: &BigUint, max_quot: u64) -> BigUint {
    let q = div_big_by_big_small(value, modulus, max_quot);
    let mut remainder = value.clone();
    let prod = modulus.mul_u64(q);
    remainder.sub_assign(&prod);
    remainder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rns_context_roundtrip() {
        let base = RnsBase::from_values(vec![3, 5, 7]).expect("base");
        let ctx = RnsContext::new(base);
        let value = 42u128;
        let residues = ctx.decompose_u128(value);
        let recomposed = ctx.compose_u128(&residues).expect("compose");
        assert_eq!(recomposed, value % ctx.base_prod_u128().unwrap());
    }

    #[test]
    fn base_converter_basic() {
        let input = RnsBase::from_values(vec![3, 5, 7]).expect("input");
        let output = RnsBase::from_values(vec![11, 13]).expect("output");
        let converter = BaseConverter::new(input.clone(), output).expect("converter");

        let value = 81u128;
        let residues: Vec<u64> = input
            .moduli()
            .iter()
            .map(|m| (value % u128::from(m.value())) as u64)
            .collect();
        let converted = converter.fast_convert(&residues).expect("convert");

        assert_eq!(converted[0], (value % 11) as u64);
        assert_eq!(converted[1], (value % 13) as u64);
    }

    #[test]
    fn base_converter_array_layout() {
        let input = RnsBase::from_values(vec![3, 5, 7]).expect("input");
        let output = RnsBase::from_values(vec![11, 13]).expect("output");
        let converter = BaseConverter::new(input.clone(), output.clone()).expect("converter");

        let values = [10u128, 22u128, 37u128];
        let count = values.len();
        let mut input_vec = vec![0u64; count * input.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in input.moduli().iter().enumerate() {
                input_vec[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output_vec = converter
            .fast_convert_array(&input_vec, count)
            .expect("convert");
        for (idx, value) in values.iter().enumerate() {
            for (j, _) in output.moduli().iter().enumerate() {
                let residues: Vec<u64> = input
                    .moduli()
                    .iter()
                    .map(|m| (value % u128::from(m.value())) as u64)
                    .collect();
                let expected = converter.fast_convert(&residues).expect("single convert");
                assert_eq!(output_vec[j * count + idx], expected[j]);
            }
        }
    }

    #[test]
    fn base_converter_array_into_matches_alloc() {
        let input = RnsBase::from_values(vec![3, 5, 7]).expect("input");
        let output = RnsBase::from_values(vec![11, 13]).expect("output");
        let converter = BaseConverter::new(input.clone(), output.clone()).expect("converter");

        let values = [9u128, 22u128, 37u128, 41u128];
        let count = values.len();
        let mut input_vec = vec![0u64; count * input.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in input.moduli().iter().enumerate() {
                input_vec[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let expected = converter
            .fast_convert_array(&input_vec, count)
            .expect("convert");
        let mut output_vec = vec![0u64; count * output.len()];
        let mut scratch = vec![0u64; input.len() * BaseConverter::BLOCK_SIZE];
        converter
            .fast_convert_array_into(&input_vec, count, &mut output_vec, &mut scratch)
            .expect("convert");

        assert_eq!(output_vec, expected);
    }

    #[test]
    fn rounder_matches_u128() {
        let base = RnsBase::from_values(vec![3, 5, 7]).expect("base");
        let rounder = RnsRounder::new(base.clone(), 11).expect("rounder");
        let q = base.base_prod_u128().unwrap();
        for x in 0u128..q {
            let residues: Vec<u64> = base
                .moduli()
                .iter()
                .map(|m| (x % u128::from(m.value())) as u64)
                .collect();
            let rounded = rounder.round(&residues).expect("round");
            let expected = ((x * 11 + q / 2) / q) as u64 % 11;
            assert_eq!(rounded, expected);
        }
    }

    #[test]
    fn exact_convert_array_matches_rounder() {
        let ibase = RnsBase::from_values(vec![3, 5, 7]).expect("ibase");
        let obase = RnsBase::from_values(vec![11]).expect("obase");
        let converter = BaseConverter::new(ibase.clone(), obase).expect("converter");
        let rounder = RnsRounder::new(ibase.clone(), 11).expect("rounder");

        let values = [0u128, 1, 42, 77, 104];
        let count = values.len();
        let mut input = vec![0u64; count * ibase.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in ibase.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let exact = converter.exact_convert_array(&input, count).expect("exact");
        let rounded = rounder.round_array(&input, count).expect("round array");
        assert_eq!(exact, rounded);
    }
}
