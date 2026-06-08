use rand::RngCore;

use crate::error::OperatorError;
use crate::lookup::PrivateLookup;
use crate::shares::AdditiveShares;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicComparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Predicate generation for additively shared integers.
///
/// Inputs are interpreted in `Z_{2^bit_width}`. The output is an additive
/// sharing of a Boolean selector in the lookup operator's output modulus.
pub struct Comparison<'a> {
    lookup: &'a PrivateLookup,
    bit_width: usize,
    domain: usize,
}

impl<'a> Comparison<'a> {
    pub fn new(lookup: &'a PrivateLookup, bit_width: usize) -> Result<Self, OperatorError> {
        if bit_width == 0 || bit_width >= usize::BITS as usize {
            return Err(OperatorError::InvalidParams(
                "comparison bit_width must be in 1..usize::BITS",
            ));
        }
        let domain = 1usize
            .checked_shl(bit_width as u32)
            .ok_or(OperatorError::InvalidParams("comparison domain overflow"))?;
        if domain > lookup.config().max_domain {
            return Err(OperatorError::InvalidParams(
                "comparison domain exceeds lookup max_domain",
            ));
        }
        Ok(Self {
            lookup,
            bit_width,
            domain,
        })
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    pub fn domain(&self) -> usize {
        self.domain
    }

    pub fn compare_public<R: RngCore + ?Sized>(
        &self,
        value_share0: &[u64],
        value_share1: &[u64],
        rhs: u64,
        op: PublicComparison,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let modulus = self.lookup.config().output_modulus;
        let table = (0..self.domain)
            .map(|idx| {
                if eval_public(idx as u64, rhs, self.domain as u64, op) {
                    1 % modulus
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        self.lookup.lookup(value_share0, value_share1, &table, rng)
    }

    pub fn less_than_public<R: RngCore + ?Sized>(
        &self,
        value_share0: &[u64],
        value_share1: &[u64],
        rhs: u64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        self.compare_public(value_share0, value_share1, rhs, PublicComparison::Lt, rng)
    }

    pub fn range_public<R: RngCore + ?Sized>(
        &self,
        value_share0: &[u64],
        value_share1: &[u64],
        lower_inclusive: u64,
        upper_exclusive: u64,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let modulus = self.lookup.config().output_modulus;
        let table = (0..self.domain)
            .map(|idx| {
                let idx = idx as u64;
                if lower_inclusive <= idx && idx < upper_exclusive {
                    1 % modulus
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        self.lookup.lookup(value_share0, value_share1, &table, rng)
    }
}

fn eval_public(lhs: u64, rhs: u64, domain: u64, op: PublicComparison) -> bool {
    match op {
        PublicComparison::Eq => lhs == rhs,
        PublicComparison::Ne => lhs != rhs,
        PublicComparison::Lt => lhs < rhs,
        PublicComparison::Le => lhs <= rhs,
        PublicComparison::Gt => lhs > rhs && rhs < domain,
        PublicComparison::Ge => lhs >= rhs && rhs < domain,
    }
}
