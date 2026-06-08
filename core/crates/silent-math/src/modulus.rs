//! Modulus representation with Barrett ratio precomputation.

use crate::numth;

pub const MAX_MODULUS_BITS: u32 = 61;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Modulus {
    value: u64,
    bit_count: u32,
    const_ratio: [u64; 3],
    is_prime: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModulusError {
    Zero,
    One,
    TooLarge,
}

impl Modulus {
    pub fn new(value: u64) -> Result<Self, ModulusError> {
        if value == 0 {
            return Err(ModulusError::Zero);
        }
        if value == 1 {
            return Err(ModulusError::One);
        }
        if value >> MAX_MODULUS_BITS != 0 {
            return Err(ModulusError::TooLarge);
        }

        let bit_count = bit_count_u64(value);
        let const_ratio = barrett_ratio(value);
        let is_prime = numth::is_prime(value);

        Ok(Self {
            value,
            bit_count,
            const_ratio,
            is_prime,
        })
    }

    pub fn value(&self) -> u64 {
        self.value
    }

    pub fn bit_count(&self) -> u32 {
        self.bit_count
    }

    pub fn const_ratio(&self) -> [u64; 3] {
        self.const_ratio
    }

    pub fn is_prime(&self) -> bool {
        self.is_prime
    }

    pub fn reduce_u64(&self, value: u64) -> u64 {
        barrett_reduce_u128(u128::from(value), self)
    }

    pub fn reduce_u64_fast(&self, value: u64) -> u64 {
        // SEAL barrett_reduce_64: requires modulus <= 63 bits.
        let m = self.value();
        let tmp1 = ((value as u128 * self.const_ratio[1] as u128) >> 64) as u64;
        let mut tmp0 = value.wrapping_sub(tmp1.wrapping_mul(m));
        if tmp0 >= m {
            tmp0 = tmp0.wrapping_sub(m);
        }
        tmp0
    }

    pub fn reduce_u128(&self, value: u128) -> u64 {
        barrett_reduce_u128(value, self)
    }
}

#[inline(always)]
pub fn barrett_reduce_u128(value: u128, modulus: &Modulus) -> u64 {
    let m = modulus.value();
    debug_assert!(m > 1);

    // SEAL's barrett_reduce_128 logic adapted for Rust
    // Input: value (128-bit)
    // Const Ratio: floor(2^128 / m) -> stored as [u64; 3] in modulus (only low 2 needed for < 64-bit m)
    // We assume m < 2^62 to refer to SEAL's constraints, but here let's stick to the math.

    // We want q = floor(value / m)
    // approx q = floor(value * mu / 2^128)

    // mu is 128-bit: (const_ratio[1] << 64) | const_ratio[0]
    let mu_lo = modulus.const_ratio[0];
    let mu_hi = modulus.const_ratio[1];

    let a_lo = value as u64;
    let a_hi = (value >> 64) as u64;

    // Multiply value * mu. We only need the high 128 bits of the 256-bit product.
    // product = (a_hi*2^64 + a_lo) * (mu_hi*2^64 + mu_lo)
    //         = a_hi*mu_hi*2^128 + (a_hi*mu_lo + a_lo*mu_hi)*2^64 + a_lo*mu_lo

    // 1. Term a_lo * mu_lo -> 128-bit. We only care about carry into bit 128?
    // Wait, we need floor(product / 2^128). So we need the bits [128..256].

    // Low 128 bits of a_lo * mu_lo are relevant only for carry.
    let p0 = u128::from(a_lo) * u128::from(mu_lo);
    let p0_hi = (p0 >> 64) as u64; // Carry to bit 64

    // Middle terms: a_hi * mu_lo and a_lo * mu_hi
    // (a_hi * mu_lo) * 2^64
    let p1 = u128::from(a_hi) * u128::from(mu_lo);
    // (a_lo * mu_hi) * 2^64
    let p2 = u128::from(a_lo) * u128::from(mu_hi);

    // Add middle terms at bit position 64.
    // We are interested in carries into bit 128.

    // lo parts of p1, p2 affect bit 64..127
    // hi parts of p1, p2 affect bit 128..191

    let p1_lo = p1 as u64;
    let p1_hi = (p1 >> 64) as u64;

    let p2_lo = p2 as u64;
    let p2_hi = (p2 >> 64) as u64;

    // Sum at pos 64: p0_hi + p1_lo + p2_lo
    let (s1, c1) = p0_hi.overflowing_add(p1_lo);
    let (_, c2) = s1.overflowing_add(p2_lo);

    // Carry to bit 128 is c1 + c2.
    let carry_to_128 = (c1 as u64) + (c2 as u64);

    // High term: a_hi * mu_hi * 2^128
    let p3 = u128::from(a_hi) * u128::from(mu_hi);
    let p3_lo = p3 as u64;

    // Total q (bits 128..191 roughly, fitting in u64)
    // q = p3_lo + p1_hi + p2_hi + carry_to_128

    // NOTE: In Rust optimized build, this is reasonably efficient but verify against SEAL's logic.
    // SEAL:
    // multiply_uint64_hw64(input[0], const_ratio[0], &carry); // a_lo * mu_lo >> 64
    // multiply_uint64(input[0], const_ratio[1], tmp2); // a_lo * mu_hi
    // tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1); // High part calc

    // Let's stick to the Rust expression which compiles down well.
    let q = p3_lo
        .wrapping_add(p1_hi)
        .wrapping_add(p2_hi)
        .wrapping_add(carry_to_128);

    // Subtraction: result = value - q * m
    // Since we computed q ~ floor(value / m), result should be in [0, 2m) roughly.
    let q_m = (q as u128) * (m as u128);
    let q_m_lo = q_m as u64;

    // value (low 64) - q*m (low 64)
    let mut result = a_lo.wrapping_sub(q_m_lo);

    // Branchless conditional subtraction
    let mask = ((result >= m) as u64).wrapping_neg();
    result = result.wrapping_sub(m & mask);
    // Occasionally it might need a second one if precision was slightly off,
    // but with 128-bit ratio for 60-bit modulus, it should be stable.
    // SEAL does: return SEAL_COND_SELECT(tmp3 >= modulus.value(), tmp3 - modulus.value(), tmp3);
    // So one check is enough.

    result
}

fn bit_count_u64(value: u64) -> u32 {
    64 - value.leading_zeros()
}

fn barrett_ratio(modulus: u64) -> [u64; 3] {
    let numerator = [0u64, 0u64, 1u64];
    let (quotient, remainder) = divide_u192_by_u64(numerator, modulus);
    [quotient[0], quotient[1], remainder]
}

fn divide_u192_by_u64(numerator: [u64; 3], divisor: u64) -> ([u64; 3], u64) {
    let mut quotient = [0u64; 3];
    let mut rem: u128 = 0;

    for i in (0..3).rev() {
        let num = (rem << 64) | u128::from(numerator[i]);
        let q = num / u128::from(divisor);
        rem = num % u128::from(divisor);
        quotient[i] = q as u64;
    }

    (quotient, rem as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulus_rejects_invalid_values() {
        assert_eq!(Modulus::new(0), Err(ModulusError::Zero));
        assert_eq!(Modulus::new(1), Err(ModulusError::One));
        assert_eq!(Modulus::new(1u64 << 61), Err(ModulusError::TooLarge));
    }

    #[test]
    fn modulus_prime_flag() {
        let prime = Modulus::new(17).expect("prime modulus");
        let composite = Modulus::new(21).expect("composite modulus");
        assert!(prime.is_prime());
        assert!(!composite.is_prime());
    }

    #[test]
    fn barrett_reduce_matches_mod() {
        let modulus = Modulus::new(65537).expect("modulus");
        let test_values = [
            0u128,
            1,
            2,
            3,
            123456789,
            9876543210123,
            (modulus.value() as u128) * (modulus.value() as u128) - 1,
        ];

        for value in test_values {
            let reduced = barrett_reduce_u128(value, &modulus);
            let expected = (value % modulus.value() as u128) as u64;
            assert_eq!(reduced, expected);
        }

        for i in 0u128..1024 {
            let value = i * 1337 + 42;
            let reduced = barrett_reduce_u128(value, &modulus);
            let expected = (value % modulus.value() as u128) as u64;
            assert_eq!(reduced, expected);
        }

        for i in 0u64..1024 {
            let reduced = modulus.reduce_u64(i);
            let expected = i % modulus.value();
            assert_eq!(reduced, expected);
        }
    }
}
