//! Number-theoretic utilities.

pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    debug_assert!(modulus > 0);
    if modulus == 1 {
        return 0;
    }
    base %= modulus;
    let mut result = 1 % modulus;

    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exp >>= 1;
    }
    result
}

pub fn mod_inverse(value: u64, modulus: u64) -> Option<u64> {
    if modulus == 0 {
        return None;
    }

    let (gcd, x, _) = extended_gcd(value as i128, modulus as i128);
    if gcd != 1 {
        return None;
    }

    let mut x = x % modulus as i128;
    if x < 0 {
        x += modulus as i128;
    }
    Some(x as u64)
}

pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }

    const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &p in SMALL_PRIMES.iter() {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }

    let mut d = n - 1;
    let s = d.trailing_zeros();
    d >>= s;

    for &a in SMALL_PRIMES.iter() {
        let mut x = mod_pow(a % n, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut witness = true;
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                witness = false;
                break;
            }
        }
        if witness {
            return false;
        }
    }

    true
}

pub fn is_primitive_root(root: u64, degree: u64, modulus: u64) -> bool {
    if degree == 0 || !degree.is_power_of_two() {
        return false;
    }
    if modulus == 0 {
        return false;
    }

    if mod_pow(root, degree, modulus) != 1 {
        return false;
    }
    if mod_pow(root, degree >> 1, modulus) == 1 {
        return false;
    }
    true
}

pub fn factorize(mut n: u64) -> Vec<(u64, u32)> {
    let mut factors = Vec::new();
    if n == 0 {
        return factors;
    }
    if n == 1 {
        return vec![(1, 1)];
    }

    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            let mut count = 0;
            while n % d == 0 {
                count += 1;
                n /= d;
            }
            factors.push((d, count));
        }
        d += 1;
    }
    if n > 1 {
        factors.push((n, 1));
    }
    factors
}

pub fn euler_phi(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let factors = factorize(n);
    let mut result = n;
    for (p, _) in factors {
        if p > 1 {
            // Handle 1 case if factorize returns (1,1) for n=1 strictly, but our factorize logic skips 1 for prime factors usually.
            result = result / p * (p - 1);
        }
    }
    result
}

pub fn try_primitive_root(degree: u64, modulus: u64) -> Option<u64> {
    if degree == 0 || !degree.is_power_of_two() {
        return None;
    }
    if modulus < 3 {
        return None;
    }
    if (modulus - 1) % degree != 0 {
        return None;
    }

    let exponent = (modulus - 1) / degree;
    for candidate in 2..modulus {
        let root = mod_pow(candidate % modulus, exponent, modulus);
        if root == 1 {
            continue;
        }
        if is_primitive_root(root, degree, modulus) {
            return Some(root);
        }
    }

    None
}

/// Returns the largest prime < start such that prime ≡ 1 (mod step).
pub fn prev_ntt_prime(start: u64, step: u64) -> Option<u64> {
    if step == 0 || start <= step + 1 {
        return None;
    }
    let mut candidate = ((start - 1) / step) * step + 1;
    if candidate >= start {
        candidate = candidate.saturating_sub(step);
    }
    while candidate > step {
        if is_prime(candidate) {
            return Some(candidate);
        }
        if candidate <= step {
            break;
        }
        candidate -= step;
    }
    None
}

/// Returns the smallest prime >= start such that prime ≡ 1 (mod step).
pub fn next_ntt_prime(start: u64, step: u64) -> Option<u64> {
    if step == 0 {
        return None;
    }
    let start = if start <= 1 { 1 } else { start };
    let k = (start.saturating_sub(1) + step - 1) / step;
    let mut candidate = k.saturating_mul(step).saturating_add(1);
    loop {
        if candidate <= step {
            return None;
        }
        if is_prime(candidate) {
            return Some(candidate);
        }
        candidate = match candidate.checked_add(step) {
            Some(next) => next,
            None => return None,
        };
    }
}

/// Returns the smallest prime with the given bit length such that prime ≡ 1 (mod step).
pub fn first_ntt_prime_with_bits(bits: u32, step: u64) -> Option<u64> {
    if bits == 0 || bits >= 64 || step == 0 {
        return None;
    }
    let lower = 1u64 << (bits - 1);
    let upper = (1u64 << bits).saturating_sub(1);
    let k = (lower.saturating_sub(1) + step - 1) / step;
    let mut candidate = k.saturating_mul(step).saturating_add(1);
    while candidate <= upper {
        if is_prime(candidate) {
            return Some(candidate);
        }
        candidate = match candidate.checked_add(step) {
            Some(next) => next,
            None => return None,
        };
    }
    None
}

pub fn mul_mod(a: u64, b: u64, modulus: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(modulus)) as u64
}

fn extended_gcd(mut a: i128, mut b: i128) -> (i128, i128, i128) {
    let mut x0: i128 = 1;
    let mut x1: i128 = 0;
    let mut y0: i128 = 0;
    let mut y1: i128 = 1;

    while b != 0 {
        let q = a / b;
        let r = a % b;
        a = b;
        b = r;

        let x2 = x0 - q * x1;
        x0 = x1;
        x1 = x2;

        let y2 = y0 - q * y1;
        y0 = y1;
        y1 = y2;
    }

    (a, x0, y0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_basic() {
        assert_eq!(gcd(54, 24), 6);
        assert_eq!(gcd(17, 13), 1);
    }

    #[test]
    fn mod_pow_basic() {
        assert_eq!(mod_pow(3, 5, 17), 5);
        assert_eq!(mod_pow(2, 8, 17), 1);
    }

    #[test]
    fn mod_inverse_basic() {
        assert_eq!(mod_inverse(5, 17), Some(7));
        assert_eq!(mod_inverse(0, 17), None);
        assert_eq!(mod_inverse(6, 21), None);
    }

    #[test]
    fn is_prime_basic() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(17));
        assert!(is_prime(65537));
        assert!(!is_prime(1));
        assert!(!is_prime(9));
        assert!(!is_prime(21));
    }

    #[test]
    fn primitive_root_basic() {
        assert!(is_primitive_root(9, 8, 17));
        assert!(!is_primitive_root(4, 8, 17));
        assert_eq!(try_primitive_root(8, 17), Some(9));
    }

    #[test]
    fn factorize_basic() {
        assert_eq!(factorize(12), vec![(2, 2), (3, 1)]);
        assert_eq!(factorize(17), vec![(17, 1)]);
        assert_eq!(factorize(100), vec![(2, 2), (5, 2)]);
    }

    #[test]
    fn euler_phi_basic() {
        assert_eq!(euler_phi(1), 1);
        assert_eq!(euler_phi(2), 1);
        assert_eq!(euler_phi(3), 2);
        assert_eq!(euler_phi(4), 2);
        assert_eq!(euler_phi(10), 4); // 1, 3, 7, 9
        assert_eq!(euler_phi(12), 4); // 1, 5, 7, 11
    }
}
