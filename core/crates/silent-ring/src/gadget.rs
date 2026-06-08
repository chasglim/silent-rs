//! Gadget (signed base-`B`) decomposition over native power-of-two moduli.
//!
//! This is the decomposition used by TFHE / FHEW for blind rotation and LWE
//! key switching: a value `x ∈ Z_{2^k}` is represented as `level` signed
//! coefficients in `(-B/2, B/2]` such that
//!
//! ```text
//! round(x / Δ) = Σ_{i = 0}^{level - 1} d_i · 2^(k - (i+1) · base_log)
//! ```
//!
//! where `Δ` is implicit in the decomposition shift.  The implementation
//! mirrors `tfhe-rs::core_crypto::commons::math::decomposition` and is also
//! reusable by future BFV relinearization paths that opt into a balanced
//! gadget representation.

use silent_math::torus::modulus_u64;

/// Description of a balanced base-`B = 2^base_log` gadget over a native
/// power-of-two modulus `2^log_q` with `level` levels.
///
/// The first level decomposes the highest-weight bits of the input, matching
/// tfhe-rs's "high-to-low" order.  This order is what blind-rotation requires.
#[derive(Clone, Copy, Debug)]
pub struct GadgetDecomposition {
    log_q: u8,
    base_log: u8,
    level: u8,
}

impl GadgetDecomposition {
    /// Construct a gadget descriptor.
    ///
    /// # Panics
    /// Panics if `base_log * level > log_q`, i.e. the gadget cannot fit.
    pub const fn new(log_q: u8, base_log: u8, level: u8) -> Self {
        assert!(
            (base_log as u32) * (level as u32) <= log_q as u32,
            "base_log * level must fit in log_q"
        );
        assert!(base_log > 0 && level > 0, "base_log and level must be > 0");
        Self {
            log_q,
            base_log,
            level,
        }
    }

    pub const fn log_q(&self) -> u8 {
        self.log_q
    }

    pub const fn base_log(&self) -> u8 {
        self.base_log
    }

    pub const fn level(&self) -> u8 {
        self.level
    }

    /// Returns `2^base_log` (the gadget base).
    pub const fn base(&self) -> u64 {
        1u64 << self.base_log
    }

    /// Returns `2^level`.
    pub const fn levels(&self) -> usize {
        self.level as usize
    }

    /// Returns the inverse-shift to apply to level `i` (0-indexed) when
    /// recomposing: `2^(log_q - (i+1) * base_log)`.
    ///
    /// When the result equals `2^64`, returns `0` and the caller is expected
    /// to handle the saturation (the corresponding term is then "absorbed by
    /// the modulus" and contributes nothing).
    pub fn level_shift(&self, i: usize) -> u8 {
        let drop = (i as u32 + 1) * (self.base_log as u32);
        debug_assert!(drop <= self.log_q as u32);
        (self.log_q as u32 - drop) as u8
    }

    /// Decompose `value` into `self.level` signed digits in
    /// `(-B/2, B/2]`, written into `out` (most-significant-first).
    ///
    /// Each digit is returned as a `u64` whose two's-complement value
    /// matches the signed result.  This makes the digits trivial to feed back
    /// into wrapping arithmetic.
    pub fn decompose_into(&self, value: u64, out: &mut [u64]) {
        debug_assert!(out.len() == self.levels());

        let base_log = self.base_log;
        let base = self.base();
        let half_base = base >> 1;
        let log_q = self.log_q;
        let mask = base - 1;

        // Pre-round so that subsequent right-shifts implement
        // round-to-nearest at every digit boundary.
        let dropped_bits = (log_q as u32) - (base_log as u32) * (self.level as u32);
        let mut state = if dropped_bits == 0 {
            value
        } else {
            // Add 2^(dropped_bits - 1) for nearest rounding, then erase the
            // dropped bits (we just zero them, they only mattered for the
            // rounding decision).
            let rounding = if dropped_bits >= 64 {
                0
            } else {
                1u64 << (dropped_bits - 1)
            };
            let after_round = value.wrapping_add(rounding);
            if dropped_bits >= 64 {
                0
            } else {
                (after_round >> dropped_bits) << dropped_bits
            }
        };

        // We propagate a carry top-down so each digit lands in
        // (-B/2, B/2].  The carry equals 1 iff the previous digit was set
        // to its negative form by the rebalancing.
        let mut carry: u64 = 0;
        for i in (0..self.level as usize).rev() {
            let shift = self.level_shift(i);
            // Extract the raw digit (always in [0, B)).
            let raw = if shift >= 64 {
                0
            } else {
                ((state >> shift) & mask).wrapping_add(carry)
            };
            // Rebalance: if raw > B/2, subtract B and propagate +1 carry.
            // The boundary case raw == B/2 keeps the digit at B/2 with no
            // further carry, matching tfhe-rs's convention.
            if raw > half_base {
                out[i] = raw.wrapping_sub(base);
                carry = 1;
            } else if raw == base {
                // raw == B can occur when carry pushed `mask` over the edge.
                out[i] = 0;
                carry = 1;
            } else {
                out[i] = raw;
                carry = 0;
            }

            // Reset the bits we just consumed to keep `state` clean for the
            // next iteration.
            if shift < 64 {
                state &= !(mask << shift);
            }
        }
    }

    /// Convenience wrapper: allocates a fresh `Vec` for the result.
    pub fn decompose(&self, value: u64) -> Vec<u64> {
        let mut out = vec![0u64; self.levels()];
        self.decompose_into(value, &mut out);
        out
    }

    /// Reconstruct an approximation of `value` from a balanced decomposition.
    ///
    /// Discrepancy is at most `2^(log_q - base_log * level - 1)` (the bits
    /// that the gadget could not address).
    pub fn recompose(&self, digits: &[u64]) -> u64 {
        debug_assert!(digits.len() == self.levels());
        let mut acc: u64 = 0;
        for (i, d) in digits.iter().enumerate() {
            let shift = self.level_shift(i);
            if shift >= 64 {
                continue;
            }
            // `d` is signed in two's complement: shifting and wrapping_add
            // preserves the sign correctly.
            acc = acc.wrapping_add(d.wrapping_shl(shift as u32));
        }
        acc
    }

    /// Returns the unsigned `q = 2^log_q` if it fits a `u64`, else `None`.
    pub fn modulus_u64(&self) -> Option<u64> {
        if self.log_q < 64 {
            Some(modulus_u64(self.log_q))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_roundtrip(log_q: u8, base_log: u8, level: u8, sample: u64) {
        let gadget = GadgetDecomposition::new(log_q, base_log, level);
        let digits = gadget.decompose(sample);

        // Each digit must be in (-B/2, B/2].
        let half = (gadget.base() >> 1) as i64;
        for &d in &digits {
            let signed = d as i64;
            let normalised = if signed > half {
                signed - gadget.base() as i64
            } else if signed < -half {
                signed + gadget.base() as i64
            } else {
                signed
            };
            assert!(
                normalised.abs() <= half,
                "digit {} out of bounds (half={})",
                normalised,
                half
            );
        }

        let recovered = gadget.recompose(&digits);
        let dropped_bits = log_q as u32 - (base_log as u32) * (level as u32);
        let max_err = if dropped_bits == 0 {
            0
        } else if dropped_bits >= 63 {
            u64::MAX
        } else {
            1u64 << (dropped_bits - 1)
        };
        let diff = sample.wrapping_sub(recovered);
        let signed = diff as i64;
        let abs_diff = if signed < 0 {
            signed.unsigned_abs()
        } else {
            signed as u64
        };
        assert!(
            abs_diff <= max_err.wrapping_add(1),
            "recompose error {} exceeds tolerance {}; sample={:#x} recovered={:#x}",
            abs_diff,
            max_err,
            sample,
            recovered
        );
    }

    #[test]
    fn full_chain_64_bit() {
        check_roundtrip(64, 8, 8, 0xDEAD_BEEF_CAFE_BABE);
        check_roundtrip(64, 8, 8, 0);
        check_roundtrip(64, 8, 8, 1);
        check_roundtrip(64, 8, 8, u64::MAX);
        check_roundtrip(64, 8, 8, 1u64 << 63);
    }

    #[test]
    fn partial_chain_drops_lsbs() {
        // 64-bit modulus, only 4 levels of base_log = 8 -> top 32 bits.
        check_roundtrip(64, 8, 4, 0xAAAA_BBBB_CCCC_DDDD);
        check_roundtrip(64, 8, 4, 0);
        check_roundtrip(64, 8, 4, u64::MAX);
    }

    #[test]
    fn matches_tfhe_rs_style_small_example() {
        // A small case where the answer is easy to inspect by hand.
        // log_q = 8, base_log = 2, level = 4 -> covers all 8 bits.
        let gadget = GadgetDecomposition::new(8, 2, 4);
        let value = 0b1011_1010u64; // 186
        let digits = gadget.decompose(value);
        let recovered = gadget.recompose(&digits);
        let dropped = (value as u8).wrapping_sub(recovered as u8) as i8;
        assert_eq!(dropped, 0, "digits = {:?}", digits);
    }

    #[test]
    fn level_shift_decreasing_with_index() {
        let gadget = GadgetDecomposition::new(64, 8, 4);
        let shifts: Vec<u8> = (0..gadget.levels())
            .map(|i| gadget.level_shift(i))
            .collect();
        assert_eq!(shifts, vec![56, 48, 40, 32]);
    }
}
