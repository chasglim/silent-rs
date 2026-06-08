use silent_math::rns::RnsBase;
use silent_ring::{Poly, RingContext};

/// Generates the coefficient-to-slot matrix used by SILENT's BFV
/// [`BatchEncoder`](crate::schemes::bfv::encoding::BatchEncoder).
///
/// Keep this tied to [`RingContext`]'s NTT tables instead of duplicating a
/// Vandermonde slot order: the bridge must recover exactly the slots produced
/// by `BatchEncoder::decode`, whose implementation is `Poly::ntt_forward` over
/// the plaintext ring.
pub fn generate_coeff_to_slot_matrix(n: usize, t: u64) -> Option<Vec<Vec<u64>>> {
    if !n.is_power_of_two() || (t - 1) % (2 * n as u64) != 0 {
        return None;
    }
    let ring = plaintext_ring(n, t)?;
    let mut matrix = vec![vec![0u64; n]; n];
    for coeff in 0..n {
        let mut poly = Poly::new(n, 1);
        poly.limb_mut(0)[coeff] = 1;
        poly.ntt_forward(&ring);
        for slot in 0..n {
            matrix[slot][coeff] = poly.limb(0)[slot];
        }
    }
    Some(matrix)
}

/// Generates the slot-to-coefficient matrix used by SILENT's BFV
/// [`BatchEncoder`](crate::schemes::bfv::encoding::BatchEncoder).
pub fn generate_slot_to_coeff_matrix(n: usize, t: u64) -> Option<Vec<Vec<u64>>> {
    if !n.is_power_of_two() || (t - 1) % (2 * n as u64) != 0 {
        return None;
    }
    let ring = plaintext_ring(n, t)?;
    let mut inv_matrix = vec![vec![0u64; n]; n];
    for slot in 0..n {
        let mut poly = Poly::new(n, 1);
        poly.limb_mut(0)[slot] = 1;
        poly.ntt_inverse(&ring);
        for coeff in 0..n {
            inv_matrix[coeff][slot] = poly.limb(0)[coeff];
        }
    }
    Some(inv_matrix)
}

fn plaintext_ring(n: usize, t: u64) -> Option<RingContext> {
    let base = RnsBase::from_values(vec![t]).ok()?;
    Some(RingContext::new(n, base))
}
