use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use rand::RngCore;
use rand::SeedableRng;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use silent_math::arith::{
    add_mod, mul_mod_u64 as mul_mod, round_centered_q_to_p as round_q_to_p, sub_mod,
};

use crate::error::OperatorError;
use crate::modular::matrix_mul::{MatrixMulCrs, matrix_mul_setup};
use crate::shares::AdditiveShares;

#[derive(Clone, Debug)]
pub struct PrivateLookupConfig {
    pub output_modulus: u64,
    pub q_modulus_bits: u32,
    pub matrix_rows: usize,
    pub gadget_cols_t: usize,
    pub max_domain: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PrivateLookupCrsKey {
    matrix_rows: usize,
    domain: usize,
    gadget_cols_t: usize,
    q_modulus: u64,
    output_modulus: u64,
}

#[derive(Clone, Debug, Default)]
pub struct PrivateLookupCrsCache {
    inner: Arc<Mutex<HashMap<PrivateLookupCrsKey, Arc<MatrixMulCrs>>>>,
}

impl PrivateLookupCrsCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> Result<usize, OperatorError> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| OperatorError::Backend("lookup CRS cache lock poisoned".to_string()))?
            .len())
    }

    pub fn is_empty(&self) -> Result<bool, OperatorError> {
        Ok(self.len()? == 0)
    }
}

impl Default for PrivateLookupConfig {
    fn default() -> Self {
        Self {
            output_modulus: 65_537,
            q_modulus_bits: 50,
            matrix_rows: 16,
            gadget_cols_t: 8,
            max_domain: 1 << 12,
        }
    }
}

fn radix_digit_support(
    table_domain: usize,
    digit_base: u64,
    threshold_digit: u64,
    include_carry: bool,
) -> Vec<usize> {
    let max_reachable = 2u128 * digit_base as u128 - 1;
    let last_col = usize::min(
        table_domain.saturating_sub(1),
        usize::try_from(max_reachable).unwrap_or(usize::MAX),
    );
    let mut support = Vec::new();
    for value in 0..=last_col {
        let value = value as u64;
        let eq = value % digit_base == threshold_digit;
        let lt = value % digit_base < threshold_digit;
        let carry = include_carry && value >= digit_base;
        if eq || lt || carry {
            support.push(value as usize);
        }
    }
    support
}

fn radix_digit_support_union(
    table_domain: usize,
    digit_base: u64,
    threshold_digits: &[u64],
    include_carry: bool,
) -> Vec<usize> {
    let max_reachable = 2u128 * digit_base as u128 - 1;
    let last_col = usize::min(
        table_domain.saturating_sub(1),
        usize::try_from(max_reachable).unwrap_or(usize::MAX),
    );
    let mut support = Vec::new();
    for value in 0..=last_col {
        let value_u64 = value as u64;
        let digit = value_u64 % digit_base;
        let carry = include_carry && value_u64 >= digit_base;
        if carry
            || threshold_digits
                .iter()
                .any(|&threshold_digit| digit == threshold_digit || digit < threshold_digit)
        {
            support.push(value);
        }
    }
    support
}

impl PrivateLookupConfig {
    fn validate(&self) -> Result<(), OperatorError> {
        if self.output_modulus < 3 {
            return Err(OperatorError::InvalidParams(
                "private point-function lookup output modulus must be >= 3",
            ));
        }
        if self.matrix_rows == 0 || self.gadget_cols_t == 0 {
            return Err(OperatorError::InvalidParams(
                "private point-function lookup matrix-multiplication dimensions must be positive",
            ));
        }
        if self.max_domain == 0 {
            return Err(OperatorError::InvalidParams(
                "private point-function lookup max_domain must be positive",
            ));
        }
        Ok(())
    }
}

/// Exact additive-index lookup using the modular/private point-function backend.
///
/// Given additive shares `i0 + i1 mod table.len()`, this evaluates a distributed
/// point function for the hidden index and locally dots each party's one-hot
/// share with the public table. The selected value is never reconstructed inside
/// the lookup path.
pub struct PrivateLookup {
    cfg: PrivateLookupConfig,
    crs_cache: PrivateLookupCrsCache,
}

impl PrivateLookup {
    pub fn new(cfg: PrivateLookupConfig) -> Result<Self, OperatorError> {
        Self::with_crs_cache(cfg, PrivateLookupCrsCache::new())
    }

    pub fn with_crs_cache(
        cfg: PrivateLookupConfig,
        crs_cache: PrivateLookupCrsCache,
    ) -> Result<Self, OperatorError> {
        cfg.validate()?;
        Ok(Self { cfg, crs_cache })
    }

    pub fn config(&self) -> &PrivateLookupConfig {
        &self.cfg
    }

    pub fn crs_cache(&self) -> PrivateLookupCrsCache {
        self.crs_cache.clone()
    }

    pub fn prewarm_crs<R: RngCore + ?Sized>(
        &self,
        domain: usize,
        output_modulus: u64,
        rng: &mut R,
    ) -> Result<(), OperatorError> {
        let _ = self.crs_for_domain_and_modulus(domain, output_modulus, rng)?;
        Ok(())
    }

    pub fn lookup<R: RngCore + ?Sized>(
        &self,
        index_share0: &[u64],
        index_share1: &[u64],
        table: &[u64],
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let mut results = self.lookup_many(index_share0, index_share1, &[table], rng)?;
        results
            .pop()
            .ok_or_else(|| OperatorError::Backend("lookup_many returned no outputs".to_string()))
    }

    pub fn lookup_many<R: RngCore + ?Sized>(
        &self,
        index_share0: &[u64],
        index_share1: &[u64],
        tables: &[&[u64]],
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError> {
        if index_share0.len() != index_share1.len() {
            return Err(OperatorError::InvalidParams(
                "lookup index share lengths must match",
            ));
        }
        let Some(first_table) = tables.first() else {
            return Err(OperatorError::InvalidParams(
                "lookup_many requires at least one public table",
            ));
        };
        let domain = first_table.len();
        if domain == 0 {
            return Err(OperatorError::InvalidParams(
                "lookup table must be non-empty",
            ));
        }
        if domain > self.cfg.max_domain {
            return Err(OperatorError::InvalidParams(
                "lookup domain exceeds configured max_domain",
            ));
        }
        if tables.iter().any(|table| table.len() != domain) {
            return Err(OperatorError::InvalidParams(
                "lookup_many tables must have the same domain",
            ));
        }

        let support = (0..domain)
            .filter(|&col| {
                tables
                    .iter()
                    .any(|table| table[col] % self.cfg.output_modulus != 0)
            })
            .collect::<Vec<_>>();

        self.lookup_generated_many_with_support(
            index_share0,
            index_share1,
            tables.len(),
            domain,
            &support,
            |table_idx, col| tables[table_idx][col as usize],
            rng,
        )
    }

    pub fn lookup_sparse_many<R, F>(
        &self,
        index_share0: &[u64],
        index_share1: &[u64],
        output_count: usize,
        domain: usize,
        support: &[usize],
        value_at: F,
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(usize, u64) -> u64,
    {
        self.lookup_generated_many_with_support(
            index_share0,
            index_share1,
            output_count,
            domain,
            support,
            value_at,
            rng,
        )
    }

    fn lookup_sparse_many_mod<R, F>(
        &self,
        index_share0: &[u64],
        index_share1: &[u64],
        output_count: usize,
        domain: usize,
        output_modulus: u64,
        support: &[usize],
        value_at: F,
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(usize, u64) -> u64,
    {
        self.lookup_generated_many_with_support_mod(
            index_share0,
            index_share1,
            output_count,
            domain,
            output_modulus,
            support,
            value_at,
            rng,
        )
    }

    pub fn less_than_public<R: RngCore + ?Sized>(
        &self,
        value_share0: &[u64],
        value_share1: &[u64],
        bit_width: usize,
        threshold: u64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if bit_width == 0 || bit_width >= usize::BITS as usize {
            return Err(OperatorError::InvalidParams(
                "bit_width must be in 1..usize::BITS",
            ));
        }
        let domain = 1usize
            .checked_shl(bit_width as u32)
            .ok_or(OperatorError::InvalidParams("comparison domain overflow"))?;
        if domain > self.cfg.max_domain {
            return Err(OperatorError::InvalidParams(
                "comparison domain exceeds configured max_domain",
            ));
        }
        let threshold = usize::min(threshold as usize, domain);
        let table = (0..domain)
            .map(|idx| {
                if idx < threshold {
                    1 % self.cfg.output_modulus
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        self.lookup(value_share0, value_share1, &table, rng)
    }

    /// Carry-aware radix comparison against a public threshold.
    ///
    /// This is the scalable path for large `bit_width`: it evaluates digit
    /// carry, digit equality, digit less-than, and lexicographic aggregation
    /// through repeated small-domain private lookups over `Z_p`, where
    /// `p = output_modulus`. The lookup table size is `p` for every digit, so
    /// the work is linear in the number of radix digits instead of exponential
    /// in `bit_width`.
    ///
    /// Correctness requires `p > 2 * 2^radix_bits - 1`, so the hidden digit sum
    /// `a_k + b_k + carry` never wraps modulo `p` before lookup.
    pub fn less_than_public_radix<R: RngCore + ?Sized>(
        &self,
        value_share0: &[u64],
        value_share1: &[u64],
        bit_width: usize,
        threshold: u64,
        radix_bits: usize,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let mut outputs = self.less_than_public_radix_many(
            value_share0,
            value_share1,
            bit_width,
            &[threshold],
            radix_bits,
            rng,
        )?;
        outputs.pop().ok_or_else(|| {
            OperatorError::Backend("radix comparison returned no output".to_string())
        })
    }

    pub fn less_than_public_radix_many<R: RngCore + ?Sized>(
        &self,
        value_share0: &[u64],
        value_share1: &[u64],
        bit_width: usize,
        thresholds: &[u64],
        radix_bits: usize,
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError> {
        self.less_than_public_radix_many_lookup_internal(
            value_share0,
            value_share1,
            bit_width,
            thresholds,
            radix_bits,
            rng,
        )
    }

    /// Radix comparison with an externally supplied shared-bit multiplication backend.
    ///
    /// The default `less_than_public_radix` keeps the whole comparison in the
    /// private-lookup backend. Callers that already have an arithmetic/HSS
    /// multiplication engine can provide it here for the secret prefix products
    /// while the digit predicates still use the private point-function lookup.
    pub fn less_than_public_radix_with_bit_and<R, F>(
        &self,
        value_share0: &[u64],
        value_share1: &[u64],
        bit_width: usize,
        threshold: u64,
        radix_bits: usize,
        rng: &mut R,
        mut bit_and: F,
    ) -> Result<AdditiveShares, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(&AdditiveShares, &AdditiveShares, &mut R) -> Result<AdditiveShares, OperatorError>,
    {
        self.less_than_public_radix_by(
            value_share0,
            value_share1,
            bit_width,
            threshold,
            radix_bits,
            rng,
            |_lookup, lhs, rhs, rng| bit_and(lhs, rhs, rng),
        )
    }

    pub fn bit_and<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.and_shared_bits(lhs, rhs, rng)
    }

    fn less_than_public_radix_by<R, F>(
        &self,
        value_share0: &[u64],
        value_share1: &[u64],
        bit_width: usize,
        threshold: u64,
        radix_bits: usize,
        rng: &mut R,
        mut bit_and: F,
    ) -> Result<AdditiveShares, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(
            &Self,
            &AdditiveShares,
            &AdditiveShares,
            &mut R,
        ) -> Result<AdditiveShares, OperatorError>,
    {
        if value_share0.len() != value_share1.len() {
            return Err(OperatorError::InvalidParams(
                "radix comparison share lengths must match",
            ));
        }
        if bit_width == 0 || bit_width > 63 {
            return Err(OperatorError::InvalidParams(
                "radix comparison bit_width must be in 1..=63",
            ));
        }
        if radix_bits == 0 || radix_bits > bit_width || radix_bits >= u64::BITS as usize {
            return Err(OperatorError::InvalidParams(
                "radix_bits must be in 1..=bit_width and fit in u64",
            ));
        }
        let radix = 1u64
            .checked_shl(radix_bits as u32)
            .ok_or(OperatorError::InvalidParams("radix overflow"))?;
        let p = self.cfg.output_modulus;
        if (p as u128) <= (2u128 * radix as u128 - 1) {
            return Err(OperatorError::InvalidParams(
                "lookup output modulus must exceed the maximum radix digit sum",
            ));
        }
        let table_domain = usize::try_from(p).map_err(|_| {
            OperatorError::InvalidParams("lookup output modulus exceeds usize domain")
        })?;
        if table_domain > self.cfg.max_domain {
            return Err(OperatorError::InvalidParams(
                "radix comparison requires max_domain >= output_modulus",
            ));
        }

        let input_modulus = 1u128 << bit_width;
        if threshold as u128 >= input_modulus {
            return AdditiveShares::share_public(&vec![1; value_share0.len()], p);
        }
        if threshold == 0 {
            return AdditiveShares::share_public(&vec![0; value_share0.len()], p);
        }

        let digits = bit_width.div_ceil(radix_bits);
        let mut carry = AdditiveShares::share_public(&vec![0; value_share0.len()], p)?;
        let mut eq_digits = Vec::with_capacity(digits);
        let mut lt_digits = Vec::with_capacity(digits);

        for digit_idx in 0..digits {
            let shift = digit_idx * radix_bits;
            let digit_bits = usize::min(radix_bits, bit_width - shift);
            let digit_base = 1u64
                .checked_shl(digit_bits as u32)
                .ok_or(OperatorError::InvalidParams("digit base overflow"))?;
            let digit_mask = digit_base - 1;
            let threshold_digit = ((threshold >> shift) & digit_mask) % digit_base;

            let index0 = value_share0
                .iter()
                .zip(carry.party0().iter())
                .map(|(&share, &carry_share)| {
                    let digit = ((share as u128 % input_modulus) >> shift) as u64 & digit_mask;
                    add_mod(digit % p, carry_share, p)
                })
                .collect::<Vec<_>>();
            let index1 = value_share1
                .iter()
                .zip(carry.party1().iter())
                .map(|(&share, &carry_share)| {
                    let digit = ((share as u128 % input_modulus) >> shift) as u64 & digit_mask;
                    add_mod(digit % p, carry_share, p)
                })
                .collect::<Vec<_>>();

            if digit_idx + 1 < digits {
                let support = radix_digit_support(table_domain, digit_base, threshold_digit, true);
                let mut outputs = self.lookup_generated_many_with_support(
                    &index0,
                    &index1,
                    3,
                    table_domain,
                    &support,
                    |table_idx, digit_sum| match table_idx {
                        0 => u64::from(digit_sum % digit_base == threshold_digit),
                        1 => u64::from(digit_sum % digit_base < threshold_digit),
                        2 => u64::from(digit_sum >= digit_base),
                        _ => 0,
                    },
                    rng,
                )?;
                carry = outputs.pop().ok_or_else(|| {
                    OperatorError::Backend("lookup_many did not return carry output".to_string())
                })?;
                let lt = outputs.pop().ok_or_else(|| {
                    OperatorError::Backend(
                        "lookup_many did not return less-than output".to_string(),
                    )
                })?;
                let eq = outputs.pop().ok_or_else(|| {
                    OperatorError::Backend("lookup_many did not return equality output".to_string())
                })?;
                eq_digits.push(eq);
                lt_digits.push(lt);
            } else {
                let support = radix_digit_support(table_domain, digit_base, threshold_digit, false);
                let mut outputs = self.lookup_generated_many_with_support(
                    &index0,
                    &index1,
                    2,
                    table_domain,
                    &support,
                    |table_idx, digit_sum| match table_idx {
                        0 => u64::from(digit_sum % digit_base == threshold_digit),
                        1 => u64::from(digit_sum % digit_base < threshold_digit),
                        _ => 0,
                    },
                    rng,
                )?;
                let lt = outputs.pop().ok_or_else(|| {
                    OperatorError::Backend(
                        "lookup_many did not return less-than output".to_string(),
                    )
                })?;
                let eq = outputs.pop().ok_or_else(|| {
                    OperatorError::Backend("lookup_many did not return equality output".to_string())
                })?;
                eq_digits.push(eq);
                lt_digits.push(lt);
            }
        }

        let mut result = AdditiveShares::share_public(&vec![0; value_share0.len()], p)?;
        let mut eq_prefix = AdditiveShares::share_public(&vec![1; value_share0.len()], p)?;
        for digit_idx in (0..digits).rev() {
            let less_here = bit_and(self, &eq_prefix, &lt_digits[digit_idx], rng)?;
            self.check_bit_and_output(&less_here, value_share0.len())?;
            result = result.add(&less_here)?;
            eq_prefix = bit_and(self, &eq_prefix, &eq_digits[digit_idx], rng)?;
            self.check_bit_and_output(&eq_prefix, value_share0.len())?;
        }
        Ok(result)
    }

    fn less_than_public_radix_many_lookup_internal<R: RngCore + ?Sized>(
        &self,
        value_share0: &[u64],
        value_share1: &[u64],
        bit_width: usize,
        thresholds: &[u64],
        radix_bits: usize,
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError> {
        if thresholds.is_empty() {
            return Err(OperatorError::InvalidParams(
                "radix comparison requires at least one threshold",
            ));
        }
        if value_share0.len() != value_share1.len() {
            return Err(OperatorError::InvalidParams(
                "radix comparison share lengths must match",
            ));
        }
        if bit_width == 0 || bit_width > 63 {
            return Err(OperatorError::InvalidParams(
                "radix comparison bit_width must be in 1..=63",
            ));
        }
        if radix_bits == 0 || radix_bits > bit_width || radix_bits >= u64::BITS as usize {
            return Err(OperatorError::InvalidParams(
                "radix_bits must be in 1..=bit_width and fit in u64",
            ));
        }
        let radix = 1u64
            .checked_shl(radix_bits as u32)
            .ok_or(OperatorError::InvalidParams("radix overflow"))?;
        let p = self.cfg.output_modulus;
        let narrow_internal_modulus = radix_internal_modulus(radix)?;
        let internal_modulus =
            if can_use_output_modulus_for_radix_internal(p, radix, self.cfg.max_domain) {
                p
            } else {
                narrow_internal_modulus
            };
        let table_domain = usize::try_from(internal_modulus).map_err(|_| {
            OperatorError::InvalidParams("radix comparison internal modulus exceeds usize")
        })?;
        if table_domain > self.cfg.max_domain {
            return Err(OperatorError::InvalidParams(
                "radix comparison internal domain exceeds configured max_domain",
            ));
        }

        let input_modulus = 1u128 << bit_width;
        let mut final_outputs = vec![None; thresholds.len()];
        let mut active = Vec::new();
        for (idx, &threshold) in thresholds.iter().enumerate() {
            if threshold as u128 >= input_modulus {
                final_outputs[idx] = Some(AdditiveShares::share_public(
                    &vec![1; value_share0.len()],
                    p,
                )?);
            } else if threshold == 0 {
                final_outputs[idx] = Some(AdditiveShares::share_public(
                    &vec![0; value_share0.len()],
                    p,
                )?);
            } else {
                active.push((idx, threshold));
            }
        }
        if active.is_empty() {
            return final_outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        OperatorError::Backend(
                            "radix comparison output was not initialized".to_string(),
                        )
                    })
                })
                .collect();
        }

        let digits = bit_width.div_ceil(radix_bits);
        let slot_count = value_share0.len();
        let mut carry = AdditiveShares::share_public(&vec![0; slot_count], internal_modulus)?;
        let mut eq_digits = vec![Vec::with_capacity(digits); active.len()];
        let mut lt_digits = vec![Vec::with_capacity(digits); active.len()];

        for digit_idx in 0..digits {
            let shift = digit_idx * radix_bits;
            let digit_bits = usize::min(radix_bits, bit_width - shift);
            let digit_base = 1u64
                .checked_shl(digit_bits as u32)
                .ok_or(OperatorError::InvalidParams("digit base overflow"))?;
            let digit_mask = digit_base - 1;
            let include_carry = digit_idx + 1 < digits;
            let threshold_digits = active
                .iter()
                .map(|(_, threshold)| ((*threshold >> shift) & digit_mask) % digit_base)
                .collect::<Vec<_>>();

            let index0 = value_share0
                .iter()
                .zip(carry.party0().iter())
                .map(|(&share, &carry_share)| {
                    let digit = ((share as u128 % input_modulus) >> shift) as u64 & digit_mask;
                    add_mod(digit % internal_modulus, carry_share, internal_modulus)
                })
                .collect::<Vec<_>>();
            let index1 = value_share1
                .iter()
                .zip(carry.party1().iter())
                .map(|(&share, &carry_share)| {
                    let digit = ((share as u128 % input_modulus) >> shift) as u64 & digit_mask;
                    add_mod(digit % internal_modulus, carry_share, internal_modulus)
                })
                .collect::<Vec<_>>();

            let support = radix_digit_support_union(
                table_domain,
                digit_base,
                &threshold_digits,
                include_carry,
            );
            let output_count = active.len() * 2 + usize::from(include_carry);
            let outputs = self.lookup_sparse_many_mod(
                &index0,
                &index1,
                output_count,
                table_domain,
                internal_modulus,
                &support,
                |table_idx, digit_sum| {
                    let threshold_idx = table_idx / 2;
                    if threshold_idx < threshold_digits.len() {
                        let threshold_digit = threshold_digits[threshold_idx];
                        if table_idx % 2 == 0 {
                            u64::from(digit_sum % digit_base == threshold_digit)
                        } else {
                            u64::from(digit_sum % digit_base < threshold_digit)
                        }
                    } else {
                        u64::from(include_carry && digit_sum >= digit_base)
                    }
                },
                rng,
            )?;
            for threshold_idx in 0..active.len() {
                eq_digits[threshold_idx].push(outputs[threshold_idx * 2].clone());
                lt_digits[threshold_idx].push(outputs[threshold_idx * 2 + 1].clone());
            }
            if include_carry {
                carry = outputs[active.len() * 2].clone();
            }
        }

        let mut results = (0..active.len())
            .map(|_| AdditiveShares::share_public(&vec![0; slot_count], internal_modulus))
            .collect::<Result<Vec<_>, _>>()?;
        let mut eq_prefixes = (0..active.len())
            .map(|_| AdditiveShares::share_public(&vec![1; slot_count], internal_modulus))
            .collect::<Result<Vec<_>, _>>()?;

        for digit_idx in (0..digits).rev() {
            let batch_pairs = active.len() * 2;
            let mut lhs0 = Vec::with_capacity(batch_pairs * slot_count);
            let mut lhs1 = Vec::with_capacity(batch_pairs * slot_count);
            let mut rhs0 = Vec::with_capacity(batch_pairs * slot_count);
            let mut rhs1 = Vec::with_capacity(batch_pairs * slot_count);

            for threshold_idx in 0..active.len() {
                append_share(&mut lhs0, &mut lhs1, &eq_prefixes[threshold_idx]);
                append_share(&mut rhs0, &mut rhs1, &lt_digits[threshold_idx][digit_idx]);
                append_share(&mut lhs0, &mut lhs1, &eq_prefixes[threshold_idx]);
                append_share(&mut rhs0, &mut rhs1, &eq_digits[threshold_idx][digit_idx]);
            }

            let lhs_batch = AdditiveShares::new(internal_modulus, lhs0, lhs1)?;
            let rhs_batch = AdditiveShares::new(internal_modulus, rhs0, rhs1)?;
            let and_batch =
                self.and_shared_bits_mod(&lhs_batch, &rhs_batch, internal_modulus, rng)?;
            self.check_bit_output_mod(&and_batch, batch_pairs * slot_count, internal_modulus)?;

            for threshold_idx in 0..active.len() {
                let less_start = threshold_idx * 2 * slot_count;
                let eq_start = less_start + slot_count;
                let less_here = share_slice(&and_batch, less_start, slot_count)?;
                let next_eq = share_slice(&and_batch, eq_start, slot_count)?;
                results[threshold_idx] = results[threshold_idx].add(&less_here)?;
                eq_prefixes[threshold_idx] = next_eq;
            }
        }

        let converted_results = if internal_modulus == p {
            results
        } else {
            let mut idx0 = Vec::with_capacity(active.len() * slot_count);
            let mut idx1 = Vec::with_capacity(active.len() * slot_count);
            for result in &results {
                append_share(&mut idx0, &mut idx1, result);
            }
            let mut converted = self.lookup_sparse_many_mod(
                &idx0,
                &idx1,
                1,
                table_domain,
                p,
                &[1],
                |_table_idx, col| u64::from(col == 1),
                rng,
            )?;
            let converted_batch = converted.pop().ok_or_else(|| {
                OperatorError::Backend("boolean modulus conversion returned no output".to_string())
            })?;
            (0..active.len())
                .map(|idx| share_slice(&converted_batch, idx * slot_count, slot_count))
                .collect::<Result<Vec<_>, _>>()?
        };

        for (threshold_idx, (output_idx, _)) in active.iter().enumerate() {
            final_outputs[*output_idx] = Some(converted_results[threshold_idx].clone());
        }

        final_outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    OperatorError::Backend(
                        "radix comparison output was not initialized".to_string(),
                    )
                })
            })
            .collect()
    }

    fn and_shared_bits<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.and_shared_bits_mod(lhs, rhs, self.cfg.output_modulus, rng)
    }

    fn and_shared_bits_mod<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        output_modulus: u64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if lhs.modulus() != output_modulus || rhs.modulus() != output_modulus {
            return Err(OperatorError::InvalidParams(
                "shared bit AND requires shares over the requested output modulus",
            ));
        }
        if lhs.len() != rhs.len() {
            return Err(OperatorError::InvalidParams(
                "shared bit AND input lengths must match",
            ));
        }
        let p = output_modulus;
        if p <= 3 {
            return Err(OperatorError::InvalidParams(
                "shared bit AND requires output_modulus > 3",
            ));
        }
        let table_domain = usize::try_from(p).map_err(|_| {
            OperatorError::InvalidParams("lookup output modulus exceeds usize domain")
        })?;
        if table_domain > self.cfg.max_domain {
            return Err(OperatorError::InvalidParams(
                "shared bit AND requires max_domain >= output_modulus",
            ));
        }
        let index0 = lhs
            .party0()
            .iter()
            .zip(rhs.party0().iter())
            .map(|(&a, &b)| add_mod(a, (2 * (b as u128) % p as u128) as u64, p))
            .collect::<Vec<_>>();
        let index1 = lhs
            .party1()
            .iter()
            .zip(rhs.party1().iter())
            .map(|(&a, &b)| add_mod(a, (2 * (b as u128) % p as u128) as u64, p))
            .collect::<Vec<_>>();
        let support = [3usize];
        let mut outputs = self.lookup_generated_many_with_support_mod(
            &index0,
            &index1,
            1,
            table_domain,
            output_modulus,
            &support,
            |_table_idx, value| u64::from(value == 3),
            rng,
        )?;
        outputs.pop().ok_or_else(|| {
            OperatorError::Backend("lookup_generated_many returned no AND output".to_string())
        })
    }

    fn check_bit_and_output(
        &self,
        shares: &AdditiveShares,
        len: usize,
    ) -> Result<(), OperatorError> {
        self.check_bit_output_mod(shares, len, self.cfg.output_modulus)
    }

    fn check_bit_output_mod(
        &self,
        shares: &AdditiveShares,
        len: usize,
        output_modulus: u64,
    ) -> Result<(), OperatorError> {
        if shares.modulus() != output_modulus {
            return Err(OperatorError::InvalidParams(
                "shared bit AND backend returned shares over a different modulus",
            ));
        }
        if shares.len() != len {
            return Err(OperatorError::InvalidParams(
                "shared bit AND backend returned the wrong number of slots",
            ));
        }
        Ok(())
    }

    fn scalar_q_modulus(&self, output_modulus: u64) -> Result<u64, OperatorError> {
        if output_modulus < 2 {
            return Err(OperatorError::InvalidParams(
                "private point-function output modulus must be >= 2",
            ));
        }
        let p_bits = u64::BITS - output_modulus.leading_zeros();
        let scale_bits = self.cfg.q_modulus_bits.saturating_sub(p_bits).clamp(1, 62);
        let scale = 1u64
            .checked_shl(scale_bits)
            .ok_or(OperatorError::InvalidParams(
                "private point-function q scale overflow",
            ))?;
        output_modulus
            .checked_mul(scale)
            .ok_or(OperatorError::InvalidParams(
                "private point-function q modulus overflow",
            ))
    }

    fn crs_for_domain_and_modulus<R: RngCore + ?Sized>(
        &self,
        domain: usize,
        output_modulus: u64,
        rng: &mut R,
    ) -> Result<Arc<MatrixMulCrs>, OperatorError> {
        let q = self.scalar_q_modulus(output_modulus)?;
        let cache_key = PrivateLookupCrsKey {
            matrix_rows: self.cfg.matrix_rows,
            domain,
            gadget_cols_t: self.cfg.gadget_cols_t,
            q_modulus: q,
            output_modulus,
        };
        let mut cache =
            self.crs_cache.inner.lock().map_err(|_| {
                OperatorError::Backend("lookup CRS cache lock poisoned".to_string())
            })?;
        if let Some(crs) = cache.get(&cache_key) {
            return Ok(Arc::clone(crs));
        }

        let crs = Arc::new(
            matrix_mul_setup(self.cfg.matrix_rows, domain, self.cfg.gadget_cols_t, q, rng)
                .map_err(OperatorError::Modular)?,
        );
        cache.insert(cache_key, Arc::clone(&crs));
        Ok(crs)
    }

    fn lookup_generated_many_with_support<R, F>(
        &self,
        index_share0: &[u64],
        index_share1: &[u64],
        output_count: usize,
        domain: usize,
        support: &[usize],
        value_at: F,
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(usize, u64) -> u64,
    {
        self.lookup_generated_many_with_support_mod(
            index_share0,
            index_share1,
            output_count,
            domain,
            self.cfg.output_modulus,
            support,
            value_at,
            rng,
        )
    }

    fn lookup_generated_many_with_support_mod<R, F>(
        &self,
        index_share0: &[u64],
        index_share1: &[u64],
        output_count: usize,
        domain: usize,
        output_modulus: u64,
        support: &[usize],
        mut value_at: F,
        rng: &mut R,
    ) -> Result<Vec<AdditiveShares>, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(usize, u64) -> u64,
    {
        if index_share0.len() != index_share1.len() {
            return Err(OperatorError::InvalidParams(
                "lookup index share lengths must match",
            ));
        }
        if output_count == 0 {
            return Err(OperatorError::InvalidParams(
                "lookup_generated_many requires at least one output",
            ));
        }
        if output_modulus < 2 {
            return Err(OperatorError::InvalidParams(
                "lookup output modulus must be >= 2",
            ));
        }
        if domain == 0 {
            return Err(OperatorError::InvalidParams(
                "lookup domain must be non-empty",
            ));
        }
        if domain > self.cfg.max_domain {
            return Err(OperatorError::InvalidParams(
                "lookup domain exceeds configured max_domain",
            ));
        }
        if support.iter().any(|&col| col >= domain) {
            return Err(OperatorError::InvalidParams(
                "lookup support contains a column outside the domain",
            ));
        }

        let crs = self.crs_for_domain_and_modulus(domain, output_modulus, rng)?;
        let support_values = support
            .iter()
            .map(|&col| {
                (0..output_count)
                    .map(|output_idx| value_at(output_idx, col as u64) % output_modulus)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut out0 = (0..output_count)
            .map(|_| Vec::with_capacity(index_share0.len()))
            .collect::<Vec<_>>();
        let mut out1 = (0..output_count)
            .map(|_| Vec::with_capacity(index_share1.len()))
            .collect::<Vec<_>>();

        if should_parallelize_lookup(index_share0.len(), support.len()) {
            let workers = usize::min(
                index_share0.len(),
                thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            );
            let chunk = index_share0.len().div_ceil(workers);
            let mut seeds = Vec::with_capacity(workers);
            for _ in 0..workers {
                let mut seed = [0u8; 32];
                rng.fill_bytes(&mut seed);
                seeds.push(seed);
            }
            let mut parts = Vec::with_capacity(workers);
            thread::scope(|scope| {
                let mut handles = Vec::with_capacity(workers);
                for (worker_idx, start) in (0..index_share0.len()).step_by(chunk).enumerate() {
                    let end = usize::min(start + chunk, index_share0.len());
                    let seed = seeds[worker_idx];
                    handles.push(scope.spawn({
                        let crs = &crs;
                        let support_values = &support_values;
                        move || -> Result<_, OperatorError> {
                            let mut local_rng = StdRng::from_seed(seed);
                            let mut part0 = (0..output_count)
                                .map(|_| Vec::with_capacity(end - start))
                                .collect::<Vec<_>>();
                            let mut part1 = (0..output_count)
                                .map(|_| Vec::with_capacity(end - start))
                                .collect::<Vec<_>>();
                            let mut acc0 = vec![0u64; output_count];
                            let mut acc1 = vec![0u64; output_count];
                            let mut v_col = Vec::with_capacity(crs.v.rows());
                            let q_dist = Uniform::new(0, crs.v.modulus());
                            for offset in start..end {
                                let idx0 = (index_share0[offset] as usize) % domain;
                                let idx1 = (index_share1[offset] as usize) % domain;
                                point_function_accumulate_values_into(
                                    crs,
                                    idx0,
                                    idx1,
                                    output_count,
                                    support,
                                    support_values,
                                    output_modulus,
                                    &mut v_col,
                                    &mut acc0,
                                    &mut acc1,
                                    &q_dist,
                                    &mut local_rng,
                                )?;
                                for output_idx in 0..output_count {
                                    part0[output_idx].push(acc0[output_idx]);
                                    part1[output_idx].push(acc1[output_idx]);
                                }
                            }
                            Ok((start, part0, part1))
                        }
                    }));
                }
                for handle in handles {
                    parts.push(handle.join().map_err(|_| {
                        OperatorError::Backend("lookup worker panicked".to_string())
                    })??);
                }
                Ok::<_, OperatorError>(())
            })?;
            parts.sort_by_key(|(start, _, _)| *start);
            for (_, part0, part1) in parts {
                for output_idx in 0..output_count {
                    out0[output_idx].extend(part0[output_idx].iter().copied());
                    out1[output_idx].extend(part1[output_idx].iter().copied());
                }
            }
        } else {
            let mut acc0 = vec![0u64; output_count];
            let mut acc1 = vec![0u64; output_count];
            let mut v_col = Vec::with_capacity(crs.v.rows());
            let q_dist = Uniform::new(0, crs.v.modulus());
            for (&i0, &i1) in index_share0.iter().zip(index_share1.iter()) {
                let idx0 = (i0 as usize) % domain;
                let idx1 = (i1 as usize) % domain;
                point_function_accumulate_values_into(
                    &crs,
                    idx0,
                    idx1,
                    output_count,
                    support,
                    &support_values,
                    output_modulus,
                    &mut v_col,
                    &mut acc0,
                    &mut acc1,
                    &q_dist,
                    rng,
                )?;
                for output_idx in 0..output_count {
                    out0[output_idx].push(acc0[output_idx]);
                    out1[output_idx].push(acc1[output_idx]);
                }
            }
        }

        out0.into_iter()
            .zip(out1)
            .map(|(party0, party1)| AdditiveShares::new(output_modulus, party0, party1))
            .collect()
    }
}

fn should_parallelize_lookup(slots: usize, support_len: usize) -> bool {
    let work_items = slots.saturating_mul(support_len.max(1));
    ((slots >= 64 && work_items >= 512) || (slots > 1 && support_len >= 16 && work_items >= 4096))
        && thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            > 1
}

fn radix_internal_modulus(radix: u64) -> Result<u64, OperatorError> {
    radix
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .filter(|&value| value > 3)
        .ok_or(OperatorError::InvalidParams(
            "radix comparison internal modulus overflow",
        ))
}

fn can_use_output_modulus_for_radix_internal(p: u64, radix: u64, max_domain: usize) -> bool {
    const OUTPUT_MODULUS_DIRECT_DOMAIN_LIMIT: u64 = 257;
    (p as u128) > (2u128 * radix as u128 - 1)
        && p <= OUTPUT_MODULUS_DIRECT_DOMAIN_LIMIT
        && usize::try_from(p)
            .map(|domain| domain <= max_domain)
            .unwrap_or(false)
}

fn append_share(party0: &mut Vec<u64>, party1: &mut Vec<u64>, shares: &AdditiveShares) {
    party0.extend_from_slice(shares.party0());
    party1.extend_from_slice(shares.party1());
}

fn share_slice(
    shares: &AdditiveShares,
    start: usize,
    len: usize,
) -> Result<AdditiveShares, OperatorError> {
    let end = start
        .checked_add(len)
        .ok_or(OperatorError::InvalidParams("share slice range overflow"))?;
    if end > shares.len() {
        return Err(OperatorError::InvalidParams(
            "share slice range exceeds share length",
        ));
    }
    AdditiveShares::new(
        shares.modulus(),
        shares.party0()[start..end].to_vec(),
        shares.party1()[start..end].to_vec(),
    )
}

fn point_function_accumulate_values_into<R: RngCore + ?Sized>(
    crs: &MatrixMulCrs,
    idx0: usize,
    idx1: usize,
    output_count: usize,
    support: &[usize],
    support_values: &[Vec<u64>],
    p: u64,
    v_col: &mut Vec<u64>,
    acc0: &mut [u64],
    acc1: &mut [u64],
    q_dist: &Uniform<u64>,
    rng: &mut R,
) -> Result<(), OperatorError> {
    if acc0.len() != output_count || acc1.len() != output_count {
        return Err(OperatorError::InvalidParams(
            "point-function accumulator length must match output_count",
        ));
    }
    let q = crs.v.modulus();
    if q % p != 0 {
        return Err(OperatorError::InvalidParams(
            "private point-function q modulus must be divisible by output_modulus",
        ));
    }
    let rows = crs.v.rows();
    let domain = crs.v.cols();
    if idx0 >= domain || idx1 >= domain {
        return Err(OperatorError::InvalidParams(
            "private point-function index exceeds CRS domain",
        ));
    }

    let delta = q / p;
    let selected_col = (idx0 + idx1) % domain;
    v_col.clear();
    for row in 0..rows {
        v_col.push(crs.v.get(row, idx0).map_err(OperatorError::Modular)?);
    }
    acc0.fill(0);
    acc1.fill(0);

    for (support_idx, &col) in support.iter().enumerate() {
        let mut vt_s_at_idx0 = 0u64;
        for &v in v_col.iter() {
            let sample = q_dist.sample(rng);
            vt_s_at_idx0 = add_mod(vt_s_at_idx0, mul_mod(v, sample, q), q);
        }

        let party0_q = if col == selected_col {
            add_mod(vt_s_at_idx0, delta, q)
        } else {
            vt_s_at_idx0
        };
        let party1_q = vt_s_at_idx0;
        let party0_p = round_q_to_p(party0_q, q, p);
        let party1_p = round_q_to_p(party1_q, q, p);
        let share0 = party0_p % p;
        let share1 = sub_mod(0, party1_p % p, p);
        for output_idx in 0..output_count {
            let value = support_values[support_idx][output_idx];
            if value != 0 {
                acc0[output_idx] = add_mod(acc0[output_idx], mul_mod(share0, value, p), p);
                acc1[output_idx] = add_mod(acc1[output_idx], mul_mod(share1, value, p), p);
            }
        }
    }

    Ok(())
}
