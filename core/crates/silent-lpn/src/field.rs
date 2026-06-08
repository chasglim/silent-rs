use crate::error::LpnError;
use silent_math::arith::{add_mod, mul_mod_u64 as mul_mod, reduce_i128_mod_u64, sub_mod};

#[inline]
pub fn ensure_modulus(modulus: u64) -> Result<(), LpnError> {
    if modulus < 2 {
        return Err(LpnError::InvalidModulus(modulus));
    }
    Ok(())
}

#[inline]
pub fn add(a: u64, b: u64, modulus: u64) -> u64 {
    add_mod(a, b, modulus)
}

#[inline]
pub fn sub(a: u64, b: u64, modulus: u64) -> u64 {
    sub_mod(a, b, modulus)
}

#[inline]
pub fn neg(a: u64, modulus: u64) -> u64 {
    if a == 0 { 0 } else { modulus - a }
}

#[inline]
pub fn mul(a: u64, b: u64, modulus: u64) -> u64 {
    mul_mod(a, b, modulus)
}

#[inline]
pub fn reduce(value: i128, modulus: u64) -> u64 {
    reduce_i128_mod_u64(value, modulus)
}

pub fn pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut acc = 1 % modulus;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul(acc, base, modulus);
        }
        base = mul(base, base, modulus);
        exp >>= 1;
    }
    acc
}

pub fn inv(value: u64, modulus: u64) -> Option<u64> {
    if value == 0 || modulus < 2 {
        return None;
    }
    let mut t = 0i128;
    let mut new_t = 1i128;
    let mut r = modulus as i128;
    let mut new_r = value as i128;

    while new_r != 0 {
        let q = r / new_r;
        let next_t = t - q * new_t;
        t = new_t;
        new_t = next_t;

        let next_r = r - q * new_r;
        r = new_r;
        new_r = next_r;
    }

    if r != 1 {
        return None;
    }
    Some(reduce(t, modulus))
}

#[inline]
pub fn div(a: u64, b: u64, modulus: u64) -> Option<u64> {
    inv(b, modulus).map(|b_inv| mul(a, b_inv, modulus))
}

pub fn hamming_weight(values: &[u64]) -> usize {
    values.iter().filter(|&&v| v != 0).count()
}
