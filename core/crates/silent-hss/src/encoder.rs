use crate::HssContext;
use silent_math::rns::RnsBase;
use silent_ring::{Poly, RingContext};

pub struct HssBatchEncoder {
    degree: usize,
    _plain_modulus: u64,
    plain_ring: RingContext,
    matrix_reps_index_map: Vec<usize>,
}

impl HssBatchEncoder {
    pub fn new(degree: usize, plain_modulus: u64) -> Self {
        let rns_t = RnsBase::from_values(vec![plain_modulus])
            .expect("Failed to create RNS for plain modulus");
        // RingContext will check if plain_modulus supports NTT for the degree
        let plain_ring = RingContext::new(degree, rns_t);
        let matrix_reps_index_map = Self::build_matrix_reps_index_map(degree);

        Self {
            degree,
            _plain_modulus: plain_modulus,
            plain_ring,
            matrix_reps_index_map,
        }
    }

    pub fn from_context(context: &HssContext) -> Self {
        Self::new(context.degree(), context.plain_modulus())
    }

    /// Encodes a vector of values into a polynomial (Inverse NTT).
    /// The input values are treated as being in the "slots" (evaluations at roots).
    pub fn encode(&self, values: &[u64]) -> Poly {
        let mut poly = unsafe { Poly::new_uninit(self.degree, 1) };
        let limb = poly.limb_mut(0);

        limb.fill(0);
        let len = values.len().min(self.degree);
        for (i, &value) in values.iter().take(len).enumerate() {
            let coeff_index = self.matrix_reps_index_map[i];
            limb[coeff_index] = value;
        }

        // Inverse NTT to map slots to coefficients
        poly.ntt_inverse(&self.plain_ring);

        poly
    }

    /// Decodes a polynomial into a vector of values (Forward NTT).
    pub fn decode(&self, poly: &Poly) -> Vec<u64> {
        let mut temp = poly.clone();
        // Forward NTT to map coeffs to slots
        temp.ntt_forward(&self.plain_ring);
        let coeffs = temp.limb(0);
        let mut out = vec![0u64; self.degree];
        for i in 0..self.degree {
            let coeff_index = self.matrix_reps_index_map[i];
            out[i] = coeffs[coeff_index];
        }
        out
    }

    fn build_matrix_reps_index_map(slot_count: usize) -> Vec<usize> {
        assert!(
            slot_count.is_power_of_two(),
            "slot_count must be a power of two"
        );
        let logn = slot_count.trailing_zeros() as usize;
        let row_size = slot_count >> 1;
        let m = slot_count << 1;
        let generator = 3usize;

        let mut map = vec![0usize; slot_count];
        let mut pos = 1usize;
        for i in 0..row_size {
            let index1 = (pos - 1) >> 1;
            let index2 = (m - pos - 1) >> 1;

            map[i] = Self::reverse_bits(index1, logn);
            map[row_size | i] = Self::reverse_bits(index2, logn);

            pos = (pos * generator) & (m - 1);
        }
        map
    }

    fn reverse_bits(mut value: usize, bit_count: usize) -> usize {
        let mut out = 0usize;
        for _ in 0..bit_count {
            out = (out << 1) | (value & 1);
            value >>= 1;
        }
        out
    }
}
