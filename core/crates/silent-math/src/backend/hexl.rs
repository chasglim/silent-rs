//! Intel HEXL backend (FFI).

use crate::backend::{MathBackend, NativeBackend};
use crate::modulus::Modulus;
use crate::ntt::NttTables;

#[derive(Debug, Default, Clone, Copy)]
pub struct HexlBackend;

mod ffi {
    extern "C" {
        pub fn hexl_eltwise_add_mod(
            result: *mut u64,
            op1: *const u64,
            op2: *const u64,
            n: usize,
            modulus: u64,
        );
        pub fn hexl_eltwise_sub_mod(
            result: *mut u64,
            op1: *const u64,
            op2: *const u64,
            n: usize,
            modulus: u64,
        );
        pub fn hexl_eltwise_mul_mod(
            result: *mut u64,
            op1: *const u64,
            op2: *const u64,
            n: usize,
            modulus: u64,
            input_mod_factor: u64,
        );
        pub fn hexl_ntt_forward(
            data: *mut u64,
            n: usize,
            modulus: u64,
            root: u64,
            input_mod_factor: u64,
            output_mod_factor: u64,
        );
        pub fn hexl_ntt_inverse(
            data: *mut u64,
            n: usize,
            modulus: u64,
            root: u64,
            input_mod_factor: u64,
            output_mod_factor: u64,
        );
    }
}

impl MathBackend for HexlBackend {
    fn mod_add(&self, lhs: &mut [u64], rhs: &[u64], modulus: &Modulus) {
        assert_eq!(lhs.len(), rhs.len());
        unsafe {
            ffi::hexl_eltwise_add_mod(
                lhs.as_mut_ptr(),
                lhs.as_ptr(),
                rhs.as_ptr(),
                lhs.len(),
                modulus.value(),
            );
        }
    }

    fn mod_sub(&self, lhs: &mut [u64], rhs: &[u64], modulus: &Modulus) {
        assert_eq!(lhs.len(), rhs.len());
        unsafe {
            ffi::hexl_eltwise_sub_mod(
                lhs.as_mut_ptr(),
                lhs.as_ptr(),
                rhs.as_ptr(),
                lhs.len(),
                modulus.value(),
            );
        }
    }

    fn mod_mul(&self, lhs: &mut [u64], rhs: &[u64], modulus: &Modulus) {
        assert_eq!(lhs.len(), rhs.len());
        unsafe {
            ffi::hexl_eltwise_mul_mod(
                lhs.as_mut_ptr(),
                lhs.as_ptr(),
                rhs.as_ptr(),
                lhs.len(),
                modulus.value(),
                1,
            );
        }
    }

    fn ntt_forward(&self, values: &mut [u64], tables: &NttTables) {
        assert_eq!(values.len(), tables.degree());
        if let Some(root) = tables.hexl_root() {
            unsafe {
                ffi::hexl_ntt_forward(
                    values.as_mut_ptr(),
                    values.len(),
                    tables.modulus().value(),
                    root,
                    1,
                    1,
                );
            }
        } else {
            NativeBackend.ntt_forward(values, tables);
        }
    }

    fn ntt_inverse(&self, values: &mut [u64], tables: &NttTables) {
        assert_eq!(values.len(), tables.degree());
        if let Some(root) = tables.hexl_root() {
            unsafe {
                ffi::hexl_ntt_inverse(
                    values.as_mut_ptr(),
                    values.len(),
                    tables.modulus().value(),
                    root,
                    1,
                    1,
                );
            }
        } else {
            NativeBackend.ntt_inverse(values, tables);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::NativeBackend;

    #[test]
    fn hexl_matches_native_mod_ops() {
        let modulus = Modulus::new(97).expect("modulus");
        let rhs = vec![12u64, 90, 10, 77];
        let mut native = vec![5u64, 20, 90, 1];
        let mut hexl = native.clone();

        NativeBackend.mod_add(&mut native, &rhs, &modulus);
        HexlBackend.mod_add(&mut hexl, &rhs, &modulus);
        assert_eq!(native, hexl);

        NativeBackend.mod_sub(&mut native, &rhs, &modulus);
        HexlBackend.mod_sub(&mut hexl, &rhs, &modulus);
        assert_eq!(native, hexl);

        NativeBackend.mod_mul(&mut native, &rhs, &modulus);
        HexlBackend.mod_mul(&mut hexl, &rhs, &modulus);
        assert_eq!(native, hexl);
    }

    #[test]
    fn hexl_matches_native_ntt() {
        let modulus = Modulus::new(17).expect("modulus");
        let tables = NttTables::new_negacyclic(8, modulus).expect("tables");
        let mut native = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let mut hexl = native.clone();

        NativeBackend.ntt_forward(&mut native, &tables);
        HexlBackend.ntt_forward(&mut hexl, &tables);
        assert_eq!(native, hexl);

        NativeBackend.ntt_inverse(&mut native, &tables);
        HexlBackend.ntt_inverse(&mut hexl, &tables);
        assert_eq!(native, hexl);
    }
}
