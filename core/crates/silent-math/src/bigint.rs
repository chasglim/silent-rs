//! Fixed-width bigint helpers for RNS operations.

use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BigInt<const LIMBS: usize>(pub [u64; LIMBS]);

impl<const LIMBS: usize> BigInt<LIMBS> {
    pub const fn zero() -> Self {
        Self([0u64; LIMBS])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigUint {
    limbs: Vec<u64>,
}

impl BigUint {
    pub fn zero() -> Self {
        Self { limbs: vec![0] }
    }

    pub fn from_u64(value: u64) -> Self {
        if value == 0 {
            Self::zero()
        } else {
            Self { limbs: vec![value] }
        }
    }

    pub fn from_u128(value: u128) -> Self {
        let low = value as u64;
        let high = (value >> 64) as u64;
        if high == 0 {
            Self::from_u64(low)
        } else {
            Self {
                limbs: vec![low, high],
            }
        }
    }

    pub fn to_u128(&self) -> Option<u128> {
        match self.limbs.len() {
            0 => Some(0),
            1 => Some(u128::from(self.limbs[0])),
            2 => Some(u128::from(self.limbs[0]) | (u128::from(self.limbs[1]) << 64)),
            _ => None,
        }
    }

    pub fn limbs(&self) -> &[u64] {
        &self.limbs
    }

    pub fn mul_u64(&self, rhs: u64) -> Self {
        if rhs == 0 {
            return Self::zero();
        }
        let mut out = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry: u128 = 0;
        for &limb in &self.limbs {
            let prod = u128::from(limb) * u128::from(rhs) + carry;
            out.push(prod as u64);
            carry = prod >> 64;
        }
        if carry != 0 {
            out.push(carry as u64);
        }
        let mut result = Self { limbs: out };
        result.normalize();
        result
    }

    pub fn mul_big(&self, rhs: &Self) -> Self {
        if self.limbs.len() == 1 && self.limbs[0] == 0 {
            return Self::zero();
        }
        if rhs.limbs.len() == 1 && rhs.limbs[0] == 0 {
            return Self::zero();
        }
        let mut out = vec![0u64; self.limbs.len() + rhs.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry: u128 = 0;
            for (j, &b) in rhs.limbs.iter().enumerate() {
                let idx = i + j;
                let prod = u128::from(a) * u128::from(b) + u128::from(out[idx]) + carry;
                out[idx] = prod as u64;
                carry = prod >> 64;
            }
            let mut k = i + rhs.limbs.len();
            while carry != 0 {
                let sum = u128::from(out[k]) + carry;
                out[k] = sum as u64;
                carry = sum >> 64;
                k += 1;
            }
        }
        let mut result = Self { limbs: out };
        result.normalize();
        result
    }

    pub fn add_assign(&mut self, other: &Self) {
        let max_len = self.limbs.len().max(other.limbs.len());
        self.limbs.resize(max_len, 0);
        let mut carry: u128 = 0;
        for i in 0..max_len {
            let a = self.limbs[i];
            let b = other.limbs.get(i).copied().unwrap_or(0);
            let sum = u128::from(a) + u128::from(b) + carry;
            self.limbs[i] = sum as u64;
            carry = sum >> 64;
        }
        if carry != 0 {
            self.limbs.push(carry as u64);
        }
        self.normalize();
    }

    pub fn sub_assign(&mut self, other: &Self) {
        debug_assert!(&*self >= other);
        let mut borrow: u128 = 0;
        for i in 0..self.limbs.len() {
            let a = u128::from(self.limbs[i]);
            let b = u128::from(other.limbs.get(i).copied().unwrap_or(0));
            let sub = a.wrapping_sub(b + borrow);
            self.limbs[i] = sub as u64;
            borrow = if a < b + borrow { 1 } else { 0 };
        }
        self.normalize();
    }

    pub fn div_mod_u64(&self, divisor: u64) -> (Self, u64) {
        assert!(divisor != 0);
        let mut quotient = vec![0u64; self.limbs.len()];
        let mut rem: u128 = 0;
        for i in (0..self.limbs.len()).rev() {
            let num = (rem << 64) | u128::from(self.limbs[i]);
            let q = num / u128::from(divisor);
            rem = num % u128::from(divisor);
            quotient[i] = q as u64;
        }
        let mut result = Self { limbs: quotient };
        result.normalize();
        (result, rem as u64)
    }

    pub fn mod_u64(&self, modulus: u64) -> u64 {
        assert!(modulus != 0);
        let mut rem: u128 = 0;
        for limb in self.limbs.iter().rev() {
            rem = (rem << 64) | u128::from(*limb);
            rem %= u128::from(modulus);
        }
        rem as u64
    }

    fn normalize(&mut self) {
        while self.limbs.len() > 1 && *self.limbs.last().unwrap() == 0 {
            self.limbs.pop();
        }
    }
}

impl Ord for BigUint {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for (a, b) in self.limbs.iter().rev().zip(other.limbs.iter().rev()) {
            if a != b {
                return a.cmp(b);
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biguint_mul_u64_matches_u128() {
        let value = BigUint::from_u128((1u128 << 96) + 1234);
        let scaled = value.mul_u64(42);
        let expected = value.to_u128().unwrap() * 42u128;
        assert_eq!(scaled.to_u128(), Some(expected));
    }

    #[test]
    fn biguint_div_mod_u64_matches_u128() {
        let value = BigUint::from_u128((1u128 << 120) + 987654321);
        let (quotient, remainder) = value.div_mod_u64(97);
        let expected = value.to_u128().unwrap();
        assert_eq!(quotient.to_u128(), Some(expected / 97));
        assert_eq!(remainder, (expected % 97) as u64);
    }

    #[test]
    fn biguint_mod_u64_matches_u128() {
        let value = BigUint::from_u128((1u128 << 100) + 55);
        assert_eq!(value.mod_u64(101), (value.to_u128().unwrap() % 101) as u64);
    }

    #[test]
    fn biguint_sub_assign_basic() {
        let mut value = BigUint::from_u128((1u128 << 96) + 500);
        let sub = BigUint::from_u64(500);
        value.sub_assign(&sub);
        assert_eq!(value.to_u128(), Some(1u128 << 96));
    }
}
