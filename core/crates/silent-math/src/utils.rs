//! Constant-time helpers and rounding primitives.

pub fn ct_lt(a: u64, b: u64) -> u64 {
    u64::from(a < b)
}

pub fn ct_select(mask: u64, on_true: u64, on_false: u64) -> u64 {
    let mask = 0u64.wrapping_sub(mask & 1);
    (on_true & mask) | (on_false & !mask)
}

pub fn reduce_once_ct(value: u64, modulus: u64) -> u64 {
    let tmp = value.wrapping_sub(modulus);
    let lt = ct_lt(value, modulus);
    ct_select(lt, value, tmp)
}

pub fn divide_and_round(x: u64, p: u64, q: u64) -> u64 {
    let numerator = u128::from(x) * u128::from(p);
    let quotient = numerator / u128::from(q);
    let remainder = numerator % u128::from(q);
    if remainder * 2 >= u128::from(q) {
        (quotient + 1) as u64
    } else {
        quotient as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_select_basic() {
        assert_eq!(ct_select(1, 10, 20), 10);
        assert_eq!(ct_select(0, 10, 20), 20);
    }

    #[test]
    fn reduce_once_ct_basic() {
        assert_eq!(reduce_once_ct(16, 17), 16);
        assert_eq!(reduce_once_ct(17, 17), 0);
        assert_eq!(reduce_once_ct(25, 17), 8);
    }

    #[test]
    fn divide_and_round_basic() {
        assert_eq!(divide_and_round(5, 3, 7), 2);
        assert_eq!(divide_and_round(6, 3, 7), 3);
    }
}
