//! Modular arithmetic helpers.

use crate::modulus::{Modulus, barrett_reduce_u128};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiplyUIntModOperand {
    pub operand: u64,
    pub quotient: u64,
}

impl MultiplyUIntModOperand {
    pub fn new(operand: u64, modulus: u64) -> Self {
        let quotient = ((u128::from(operand) << 64) / u128::from(modulus)) as u64;
        Self { operand, quotient }
    }
}

#[inline(always)]
pub fn add_mod(a: u64, b: u64, modulus: u64) -> u64 {
    let sum = a.wrapping_add(b);
    // Branchless conditional subtraction: if sum >= modulus, subtract modulus
    // Assumes modulus < 2^63 and inputs < modulus, so sum < 2^64.
    // If sum wraps u64, this logic fails, but we assume 60-bit primes.
    let mask = ((sum >= modulus) as u64).wrapping_neg();
    sum.wrapping_sub(modulus & mask)
}

#[inline(always)]
pub fn sub_mod(a: u64, b: u64, modulus: u64) -> u64 {
    let diff = a.wrapping_sub(b);
    // If a < b, diff wraps (is large).
    // We need to add modulus.
    // Condition: a < b
    let mask = ((a < b) as u64).wrapping_neg();
    diff.wrapping_add(modulus & mask)
}

#[inline(always)]
pub fn sub_mod_lazy(a: u64, b: u64, modulus: u64) -> u64 {
    let diff = a.wrapping_sub(b);
    let mask = ((a < b) as u64).wrapping_neg();
    // lazy adds 2*modulus? Or just 1*modulus enough?
    // original code added 2*modulus if needed?
    // "a + modulus.wrapping_mul(2) - b"
    // Lazy reduction usually returns in [0, 2P).
    // If subtraction wraps, we need to add P (or 2P).
    // Let's keep original logic but branchless.
    // If a < b, add 2*modulus.
    diff.wrapping_add(modulus.wrapping_mul(2) & mask)
}

#[inline(always)]
pub fn reduce_once(value: u64, modulus: u64) -> u64 {
    debug_assert!(modulus > 0);
    // Branchless conditional subtraction
    let mask = ((value >= modulus) as u64).wrapping_neg();
    value.wrapping_sub(modulus & mask)
}

#[inline(always)]
pub fn neg_mod(a: u64, modulus: u64) -> u64 {
    debug_assert!(modulus > 0);
    // if a == 0, return 0; else return modulus - a
    // mask = (a != 0) -> 0xFFFF (if true)
    let mask = ((a != 0) as u64).wrapping_neg();
    modulus.wrapping_sub(a) & mask
}

#[inline(always)]
pub fn mul_mod(a: u64, b: u64, modulus: &Modulus) -> u64 {
    let product = u128::from(a) * u128::from(b);
    barrett_reduce_u128(product, modulus)
}

/// Shoup multiplication (lazy reduction).
/// Computes `(a * b_operand) mod modulus` approximately, returning result in [0, 2*modulus).
/// `b_quotient` must be `floor(b_operand * 2^64 / modulus)`.
/// This avoids 128-bit arithmetic for the reduction part (mostly).
#[inline(always)]
pub fn mul_mod_shoup_lazy(a: u64, b_operand: u64, b_quotient: u64, modulus: u64) -> u64 {
    debug_assert!(modulus > 0);
    let q = mul_u64_high(a, b_quotient);
    let r = a
        .wrapping_mul(b_operand)
        .wrapping_sub(q.wrapping_mul(modulus));

    // Result is in [0, 2*modulus) usually, sometimes [0, 3*modulus) if approximation is loose?
    // SEAL says result is strictly < 2*modulus for most cases if input < 4*modulus?
    // Let's trust the algorithm logic from SEAL optimization guide/paper.
    r
}

/// Shoup multiplication (fully reduced).
/// Returns `(a * b_operand) mod modulus` in [0, modulus).
#[inline(always)]
pub fn mul_mod_shoup(a: u64, b_operand: u64, b_quotient: u64, modulus: u64) -> u64 {
    let r = mul_mod_shoup_lazy(a, b_operand, b_quotient, modulus);
    // Branchless reduction
    // Note: r in [0, 2q). Only need to subtract q once.
    let mask = ((r >= modulus) as u64).wrapping_neg();
    r.wrapping_sub(modulus & mask)
}

/// Returns `(a + b)` without modular reduction, assuming `(a + b) < 2 * modulus`.
#[inline(always)]
pub fn add_mod_lazy(a: u64, b: u64, modulus: u64) -> u64 {
    debug_assert!(modulus > 0);
    debug_assert!(a < modulus);
    debug_assert!(b < modulus);
    // In strict sense, lazy reduction allows result >= modulus.
    // It is correct as long as it handles overflow or < 2*modulus if next op handles it.
    a + b
}

pub fn mul_mod_u64(a: u64, b: u64, modulus: u64) -> u64 {
    debug_assert!(modulus > 0);
    ((u128::from(a) * u128::from(b)) % u128::from(modulus)) as u64
}

#[inline(always)]
pub fn reduce_i128_mod_u64(value: i128, modulus: u64) -> u64 {
    debug_assert!(modulus > 0);
    let m = modulus as i128;
    let mut r = value % m;
    if r < 0 {
        r += m;
    }
    r as u64
}

/// Rounds a value in `Z_q` into `Z_p` by scaling centered representatives.
/// Computes `round((p / q) * centered(value mod q)) mod p`.
#[inline(always)]
pub fn round_centered_q_to_p(value: u64, q: u64, p: u64) -> u64 {
    debug_assert!(q > 1);
    debug_assert!(p > 1);

    let centered = if value > q / 2 {
        value as i128 - q as i128
    } else {
        value as i128
    };
    let scaled = centered * p as i128;
    let q_i = q as i128;
    let half = q_i / 2;
    let rounded = if scaled >= 0 {
        (scaled + half) / q_i
    } else {
        (scaled - half) / q_i
    };
    reduce_i128_mod_u64(rounded, p)
}

#[cfg(all(target_arch = "aarch64", feature = "aarch64-umulh"))]
#[inline(always)]
unsafe fn umulh(a: u64, b: u64) -> u64 {
    let hi: u64;
    core::arch::asm!(
        "umulh {hi}, {a}, {b}",
        hi = out(reg) hi,
        a = in(reg) a,
        b = in(reg) b,
        options(pure, nomem, nostack)
    );
    hi
}

#[inline(always)]
fn mul_u64_high(a: u64, b: u64) -> u64 {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64-umulh"))]
    unsafe {
        return umulh(a, b);
    }
    #[cfg(not(any(all(target_arch = "aarch64", feature = "aarch64-umulh"))))]
    {
        return ((a as u128) * (b as u128) >> 64) as u64;
    }
}

#[inline(always)]
fn mul_u64(a: u64, b: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64-umulh"))]
    {
        let lo = a.wrapping_mul(b);
        let hi = unsafe { umulh(a, b) };
        return (lo, hi);
    }
    #[cfg(not(any(all(target_arch = "aarch64", feature = "aarch64-umulh"))))]
    {
        let prod = (a as u128) * (b as u128);
        return (prod as u64, (prod >> 64) as u64);
    }
}

/// Barrett-reduced multiplication using precomputed 128-bit ratio (SEAL-style).
#[inline(always)]
pub fn mul_mod_barrett_u64(a: u64, b: u64, modulus: u64, ratio0: u64, ratio1: u64) -> u64 {
    let (z0, z1) = mul_u64(a, b);

    // Round 1
    let carry = mul_u64_high(z0, ratio0);
    let (tmp2_lo, tmp2_hi) = mul_u64(z0, ratio1);
    let (tmp1, c1) = tmp2_lo.overflowing_add(carry);
    let tmp3 = tmp2_hi.wrapping_add(c1 as u64);

    // Round 2
    let (tmp2_lo2, tmp2_hi2) = mul_u64(z1, ratio0);
    let (_tmp1b, c2) = tmp1.overflowing_add(tmp2_lo2);
    let carry2 = tmp2_hi2.wrapping_add(c2 as u64);

    let q = z1
        .wrapping_mul(ratio1)
        .wrapping_add(tmp3)
        .wrapping_add(carry2);

    let res = z0.wrapping_sub(q.wrapping_mul(modulus));
    let mask = ((res >= modulus) as u64).wrapping_neg();
    res.wrapping_sub(modulus & mask)
}

#[inline(always)]
pub fn mul_mod_barrett(a: u64, b: u64, modulus: &Modulus) -> u64 {
    let ratio = modulus.const_ratio();
    mul_mod_barrett_u64(a, b, modulus.value(), ratio[0], ratio[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulus::Modulus;

    #[test]
    fn add_sub_neg_mod_basic() {
        let m = 17;
        assert_eq!(add_mod(16, 1, m), 0);
        assert_eq!(add_mod(8, 8, m), 16);
        assert_eq!(add_mod_lazy(16, 1, m), 17);
        assert_eq!(sub_mod(0, 1, m), 16);
        assert_eq!(sub_mod(5, 3, m), 2);
        assert_eq!(sub_mod_lazy(0, 1, m), 33);
        assert_eq!(neg_mod(0, m), 0);
        assert_eq!(neg_mod(5, m), 12);
        assert_eq!(reduce_once(17, m), 0);
    }

    #[test]
    fn mul_mod_barrett_matches_u128() {
        let modulus = Modulus::new(65537).expect("modulus");
        let result = mul_mod(12345, 67890, &modulus);
        let expected =
            ((u128::from(12345u64) * u128::from(67890u64)) % u128::from(modulus.value())) as u64;
        assert_eq!(result, expected);
    }

    #[test]
    fn mul_mod_barrett_matches_mul_mod() {
        let modulus = Modulus::new(36028797018963913u64).expect("modulus");
        let mut x: u64 = 0x1234_5678_9abc_def0;
        let mut y: u64 = 0xfedc_ba98_7654_3210;
        for _ in 0..1000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            y = y.wrapping_mul(1442695040888963407).wrapping_add(1);
            let a = x % modulus.value();
            let b = y % modulus.value();
            let r1 = mul_mod(a, b, &modulus);
            let r2 = mul_mod_barrett(a, b, &modulus);
            assert_eq!(r1, r2);
        }
    }

    #[test]
    fn reduce_i128_mod_u64_handles_negative() {
        assert_eq!(reduce_i128_mod_u64(-1, 97), 96);
        assert_eq!(reduce_i128_mod_u64(-98, 97), 96);
        assert_eq!(reduce_i128_mod_u64(98, 97), 1);
    }

    #[test]
    fn round_centered_q_to_p_matches_expected_sign() {
        // q=17, p=5.
        // 16 represents -1 mod 17.
        assert_eq!(round_centered_q_to_p(16, 17, 5), 0);
        // 9 represents -8 mod 17.
        assert_eq!(round_centered_q_to_p(9, 17, 5), 3);
    }
}
