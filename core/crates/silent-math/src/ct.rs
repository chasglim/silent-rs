//! Constant-time helpers for cryptographic operations.
//!
//! Thin wrappers around the `subtle` crate, providing:
//! - Constant-time equality comparison
//! - Constant-time conditional select
//! - Constant-time conditional assignment
//!
//! These are used throughout PQC implementations to prevent
//! timing side-channels on secret-dependent code paths.

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

/// Return `true` if `a` and `b` are equal in constant time.
#[inline]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}

/// Constant-time select between two byte arrays.
///
/// Returns `a` if `condition` is true, `b` otherwise.
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn ct_select(a: &[u8], b: &[u8], condition: bool) -> Vec<u8> {
    assert_eq!(a.len(), b.len(), "ct_select requires equal-length inputs");
    let choice = Choice::from(condition as u8);
    let mut result = vec![0u8; a.len()];
    for i in 0..a.len() {
        // conditional_select(a, b, choice) returns b when choice is true.
        // We want to return a when condition is true, so swap args.
        result[i] = u8::conditional_select(&b[i], &a[i], choice);
    }
    result
}

/// Constant-time conditional assign.
///
/// Writes `src` into `dst` if `condition` is true, leaves `dst` unchanged
/// otherwise. Panics if lengths differ.
#[inline]
pub fn ct_assign(dst: &mut [u8], src: &[u8], condition: bool) {
    assert_eq!(
        dst.len(),
        src.len(),
        "ct_assign requires equal-length inputs"
    );
    let choice = Choice::from(condition as u8);
    for i in 0..dst.len() {
        dst[i] = u8::conditional_select(&dst[i], &src[i], choice);
    }
}

/// Constant-time conditional swap of two byte arrays.
#[inline]
pub fn ct_swap(a: &mut [u8], b: &mut [u8], condition: bool) {
    assert_eq!(a.len(), b.len(), "ct_swap requires equal-length inputs");
    let choice = Choice::from(condition as u8);
    for i in 0..a.len() {
        let tmp = u8::conditional_select(&a[i], &b[i], choice);
        let opposite = u8::conditional_select(&b[i], &a[i], choice);
        a[i] = tmp;
        b[i] = opposite;
    }
}

/// Constant-time comparison of two `u8` slices, returning a `Choice`.
#[inline]
pub fn ct_eq_u8_slice(a: &[u8], b: &[u8]) -> Choice {
    if a.len() != b.len() {
        return Choice::from(0);
    }
    a.ct_eq(b)
}

/// Return `true` if all bytes are zero, in constant time.
#[inline]
pub fn ct_is_zero(buf: &[u8]) -> bool {
    let mut acc = 0u8;
    for &b in buf {
        acc |= b;
    }
    acc.ct_eq(&0u8).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_works() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab"));
    }

    #[test]
    fn ct_select_true() {
        let result = ct_select(b"hello", b"world", true);
        assert_eq!(&result[..], b"hello");
    }

    #[test]
    fn ct_select_false() {
        let result = ct_select(b"hello", b"world", false);
        assert_eq!(&result[..], b"world");
    }

    #[test]
    fn ct_assign_true() {
        let mut dst = vec![0u8; 5];
        ct_assign(&mut dst, b"hello", true);
        assert_eq!(&dst[..], b"hello");
    }

    #[test]
    fn ct_assign_false() {
        let mut dst = b"hello".to_vec();
        ct_assign(&mut dst, b"world", false);
        assert_eq!(&dst[..], b"hello");
    }

    #[test]
    fn ct_is_zero_works() {
        assert!(ct_is_zero(&[0u8; 32]));
        assert!(!ct_is_zero(&[1u8; 32]));
    }
}
