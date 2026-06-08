//! Portable SIMD backend (fallback).

use crate::arith;
use crate::backend::MathBackend;
use crate::modulus::Modulus;
use crate::ntt::{NttTables, ntt_forward_native, ntt_inverse_native};

#[derive(Debug, Default, Clone, Copy)]
pub struct NativeBackend;

impl MathBackend for NativeBackend {
    fn mod_add(&self, lhs: &mut [u64], rhs: &[u64], modulus: &Modulus) {
        assert_eq!(lhs.len(), rhs.len());
        let m = modulus.value();
        for (l, r) in lhs.iter_mut().zip(rhs.iter()) {
            *l = arith::add_mod(*l, *r, m);
        }
    }

    fn mod_sub(&self, lhs: &mut [u64], rhs: &[u64], modulus: &Modulus) {
        assert_eq!(lhs.len(), rhs.len());
        let m = modulus.value();
        for (l, r) in lhs.iter_mut().zip(rhs.iter()) {
            *l = arith::sub_mod(*l, *r, m);
        }
    }

    fn mod_mul(&self, lhs: &mut [u64], rhs: &[u64], modulus: &Modulus) {
        assert_eq!(lhs.len(), rhs.len());
        for (l, r) in lhs.iter_mut().zip(rhs.iter()) {
            *l = arith::mul_mod(*l, *r, modulus);
        }
    }

    fn ntt_forward(&self, values: &mut [u64], tables: &NttTables) {
        ntt_forward_native(values, tables);
    }

    fn ntt_inverse(&self, values: &mut [u64], tables: &NttTables) {
        ntt_inverse_native(values, tables);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulus::Modulus;
    use crate::ntt::NttTables;

    #[test]
    fn native_backend_mod_ops() {
        let backend = NativeBackend;
        let modulus = Modulus::new(97).expect("modulus");
        let mut lhs = vec![5, 20, 90];
        let rhs = vec![12, 90, 10];

        backend.mod_add(&mut lhs, &rhs, &modulus);
        assert_eq!(lhs, vec![17, 13, 3]);

        backend.mod_sub(&mut lhs, &rhs, &modulus);
        assert_eq!(lhs, vec![5, 20, 90]);

        backend.mod_mul(&mut lhs, &rhs, &modulus);
        assert_eq!(lhs, vec![60, 54, 27]);
    }

    #[test]
    fn native_backend_ntt_roundtrip() {
        let backend = NativeBackend;
        let modulus = Modulus::new(17).expect("modulus");
        let tables = NttTables::new_negacyclic(8, modulus).expect("tables");
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let original = data.clone();

        backend.ntt_forward(&mut data, &tables);
        backend.ntt_inverse(&mut data, &tables);

        assert_eq!(data, original);
    }
}
