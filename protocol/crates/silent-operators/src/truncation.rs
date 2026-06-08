use rand::RngCore;
use silent_math::arith::{add_mod, mul_mod_u64 as mul_mod, sub_mod};

use crate::error::OperatorError;
use crate::lookup::PrivateLookup;
use crate::shares::AdditiveShares;

#[derive(Clone, Debug)]
pub struct TruncationConfig {
    /// Input ring is Z_{2^bit_width}; local shares must be canonical in this range.
    pub bit_width: usize,
    /// Right-shift amount d.
    pub shift: usize,
    /// Output arithmetic share modulus. This should match the nonlinear stack's
    /// share modulus and must match the lookup predicate modulus.
    pub output_modulus: u64,
}

impl TruncationConfig {
    fn validate(&self) -> Result<(), OperatorError> {
        if self.bit_width == 0 || self.bit_width >= 63 {
            return Err(OperatorError::InvalidParams(
                "truncation bit_width must be in 1..63",
            ));
        }
        if self.shift == 0 || self.shift > self.bit_width {
            return Err(OperatorError::InvalidParams(
                "truncation shift must be in 1..=bit_width",
            ));
        }
        if self.output_modulus < 3 {
            return Err(OperatorError::InvalidParams(
                "truncation output modulus must be >= 3",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TruncationCorrection {
    pub low_carry: AdditiveShares,
    pub modular_wrap: AdditiveShares,
}

/// One-way truncation bridge from the example paper.
///
/// The correction bits are obtained through the configured private point-function lookup backend
/// without reconstructing the input value. The final local arithmetic formula is
/// the paper's Appendix correctness equation:
///
/// floor(x / 2^d) = floor(x_C / 2^d) + floor(x_S / 2^d)
///                  + omega_d - omega_n * 2^{n-d}.
pub struct Truncation<'a> {
    cfg: TruncationConfig,
    lookup: &'a PrivateLookup,
}

impl<'a> Truncation<'a> {
    pub fn new(cfg: TruncationConfig, lookup: &'a PrivateLookup) -> Result<Self, OperatorError> {
        cfg.validate()?;
        if lookup.config().output_modulus != cfg.output_modulus {
            return Err(OperatorError::InvalidParams(
                "truncation output modulus must match lookup output modulus",
            ));
        }
        Ok(Self { cfg, lookup })
    }

    pub fn correction_bits<R: RngCore + ?Sized>(
        &self,
        party0: &[u64],
        party1: &[u64],
        rng: &mut R,
    ) -> Result<TruncationCorrection, OperatorError> {
        if party0.len() != party1.len() {
            return Err(OperatorError::InvalidParams(
                "truncation share lengths must match",
            ));
        }
        let n_modulus = 1u64 << self.cfg.bit_width;
        let d_modulus = 1u64 << self.cfg.shift;

        let low0 = party0.iter().map(|v| v % d_modulus).collect::<Vec<_>>();
        let low1 = party1.iter().map(|v| v % d_modulus).collect::<Vec<_>>();
        let low_lt =
            self.lookup
                .less_than_public(&low0, &low1, self.cfg.shift + 1, d_modulus, rng)?;
        let low_carry = one_minus(&low_lt)?;

        let full0 = party0.iter().map(|v| v % n_modulus).collect::<Vec<_>>();
        let full1 = party1.iter().map(|v| v % n_modulus).collect::<Vec<_>>();
        let wrap_lt =
            self.lookup
                .less_than_public(&full0, &full1, self.cfg.bit_width + 1, n_modulus, rng)?;
        let modular_wrap = one_minus(&wrap_lt)?;

        Ok(TruncationCorrection {
            low_carry,
            modular_wrap,
        })
    }

    pub fn correction_bits_radix<R: RngCore + ?Sized>(
        &self,
        party0: &[u64],
        party1: &[u64],
        radix_bits: usize,
        rng: &mut R,
    ) -> Result<TruncationCorrection, OperatorError> {
        if party0.len() != party1.len() {
            return Err(OperatorError::InvalidParams(
                "truncation share lengths must match",
            ));
        }
        let n_modulus = 1u64 << self.cfg.bit_width;
        let d_modulus = 1u64 << self.cfg.shift;

        let low0 = party0.iter().map(|v| v % d_modulus).collect::<Vec<_>>();
        let low1 = party1.iter().map(|v| v % d_modulus).collect::<Vec<_>>();
        let low_lt = self.lookup.less_than_public_radix(
            &low0,
            &low1,
            self.cfg.shift + 1,
            d_modulus,
            radix_bits.min(self.cfg.shift + 1),
            rng,
        )?;
        let low_carry = one_minus(&low_lt)?;

        let full0 = party0.iter().map(|v| v % n_modulus).collect::<Vec<_>>();
        let full1 = party1.iter().map(|v| v % n_modulus).collect::<Vec<_>>();
        let wrap_lt = self.lookup.less_than_public_radix(
            &full0,
            &full1,
            self.cfg.bit_width + 1,
            n_modulus,
            radix_bits.min(self.cfg.bit_width + 1),
            rng,
        )?;
        let modular_wrap = one_minus(&wrap_lt)?;

        Ok(TruncationCorrection {
            low_carry,
            modular_wrap,
        })
    }

    pub fn correction_bits_radix_with_bit_and<R, F>(
        &self,
        party0: &[u64],
        party1: &[u64],
        radix_bits: usize,
        rng: &mut R,
        mut bit_and: F,
    ) -> Result<TruncationCorrection, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(&AdditiveShares, &AdditiveShares, &mut R) -> Result<AdditiveShares, OperatorError>,
    {
        if party0.len() != party1.len() {
            return Err(OperatorError::InvalidParams(
                "truncation share lengths must match",
            ));
        }
        let n_modulus = 1u64 << self.cfg.bit_width;
        let d_modulus = 1u64 << self.cfg.shift;

        let low0 = party0.iter().map(|v| v % d_modulus).collect::<Vec<_>>();
        let low1 = party1.iter().map(|v| v % d_modulus).collect::<Vec<_>>();
        let low_lt = self.lookup.less_than_public_radix_with_bit_and(
            &low0,
            &low1,
            self.cfg.shift + 1,
            d_modulus,
            radix_bits.min(self.cfg.shift + 1),
            rng,
            &mut bit_and,
        )?;
        let low_carry = one_minus(&low_lt)?;

        let full0 = party0.iter().map(|v| v % n_modulus).collect::<Vec<_>>();
        let full1 = party1.iter().map(|v| v % n_modulus).collect::<Vec<_>>();
        let wrap_lt = self.lookup.less_than_public_radix_with_bit_and(
            &full0,
            &full1,
            self.cfg.bit_width + 1,
            n_modulus,
            radix_bits.min(self.cfg.bit_width + 1),
            rng,
            &mut bit_and,
        )?;
        let modular_wrap = one_minus(&wrap_lt)?;

        Ok(TruncationCorrection {
            low_carry,
            modular_wrap,
        })
    }

    pub fn truncate<R: RngCore + ?Sized>(
        &self,
        party0: &[u64],
        party1: &[u64],
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let corr = self.correction_bits(party0, party1, rng)?;
        self.truncate_with_correction(party0, party1, &corr)
    }

    pub fn truncate_radix<R: RngCore + ?Sized>(
        &self,
        party0: &[u64],
        party1: &[u64],
        radix_bits: usize,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        let corr = self.correction_bits_radix(party0, party1, radix_bits, rng)?;
        self.truncate_with_correction(party0, party1, &corr)
    }

    pub fn truncate_radix_with_bit_and<R, F>(
        &self,
        party0: &[u64],
        party1: &[u64],
        radix_bits: usize,
        rng: &mut R,
        bit_and: F,
    ) -> Result<AdditiveShares, OperatorError>
    where
        R: RngCore + ?Sized,
        F: FnMut(&AdditiveShares, &AdditiveShares, &mut R) -> Result<AdditiveShares, OperatorError>,
    {
        let corr =
            self.correction_bits_radix_with_bit_and(party0, party1, radix_bits, rng, bit_and)?;
        self.truncate_with_correction(party0, party1, &corr)
    }

    pub fn truncate_with_correction(
        &self,
        party0: &[u64],
        party1: &[u64],
        corr: &TruncationCorrection,
    ) -> Result<AdditiveShares, OperatorError> {
        if party0.len() != party1.len()
            || party0.len() != corr.low_carry.len()
            || party0.len() != corr.modular_wrap.len()
        {
            return Err(OperatorError::InvalidParams(
                "truncation correction lengths must match input shares",
            ));
        }
        let p = self.cfg.output_modulus;
        let n_modulus = 1u64 << self.cfg.bit_width;
        let wrap_weight = (1u64 << (self.cfg.bit_width - self.cfg.shift)) % p;

        let mut out0 = Vec::with_capacity(party0.len());
        let mut out1 = Vec::with_capacity(party0.len());
        for i in 0..party0.len() {
            let base0 = (party0[i] % n_modulus) >> self.cfg.shift;
            let base1 = (party1[i] % n_modulus) >> self.cfg.shift;

            let low0 = corr.low_carry.party0()[i];
            let low1 = corr.low_carry.party1()[i];
            let wrap0 = mul_mod(corr.modular_wrap.party0()[i], wrap_weight, p);
            let wrap1 = mul_mod(corr.modular_wrap.party1()[i], wrap_weight, p);

            out0.push(sub_mod(add_mod(base0 % p, low0, p), wrap0, p));
            out1.push(sub_mod(add_mod(base1 % p, low1, p), wrap1, p));
        }

        AdditiveShares::new(p, out0, out1)
    }
}

fn one_minus(bits: &AdditiveShares) -> Result<AdditiveShares, OperatorError> {
    let p = bits.modulus();
    let p0 = bits.party0().iter().map(|&v| sub_mod(1, v, p)).collect();
    let p1 = bits.party1().iter().map(|&v| sub_mod(0, v, p)).collect();
    AdditiveShares::new(p, p0, p1)
}
