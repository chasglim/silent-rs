//! Extract BFV coefficients into LWE ciphertexts over the full BFV ciphertext modulus.

use silent_math::arith;
use silent_rlwe::Ciphertext;

use super::{BridgeError, BridgeParams};

#[derive(Clone, Debug)]
pub struct ExtractedBfvLwe {
    residues_q: Vec<u64>,
    dimension: usize,
}

impl ExtractedBfvLwe {
    pub fn new(dimension: usize, residues_q: Vec<u64>) -> Self {
        Self {
            residues_q,
            dimension,
        }
    }

    pub fn residues_q(&self) -> &[u64] {
        &self.residues_q
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

pub fn extract_coefficient_lwe(
    ct: &Ciphertext,
    index: usize,
    params: &BridgeParams,
) -> Result<ExtractedBfvLwe, BridgeError> {
    let ring = params.bfv_params.ring();
    let degree = ring.degree();
    let size_q = ring.rns().len();
    if index >= degree {
        return Err(BridgeError::SlotIndexOutOfRange(index, degree));
    }
    if ct.data.len() != 2 {
        return Err(BridgeError::InvalidBfvCiphertext(ct.data.len()));
    }

    let mut c0 = ct.data[0].clone();
    let mut c1 = ct.data[1].clone();

    if ct.is_ntt {
        c0.ntt_inverse(ring);
        c1.ntt_inverse(ring);
    }

    let lwe_dimension = degree;
    let lwe_size = lwe_dimension + 1;

    let mut residues_q = vec![0u64; size_q * lwe_size];

    for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
        let qi = modulus.value();
        let limb_offset = limb_idx * lwe_size;

        // Body coefficient (last element of LWE ciphertext)
        // This is c0[index]
        residues_q[limb_offset + lwe_dimension] = c0.limb(limb_idx)[index];

        // Mask coefficients (first 'degree' elements of LWE ciphertext)
        // Follow the same negacyclic sample-extraction layout as TFHE GLWE->LWE:
        // [c1[index], c1[index-1], ..., c1[0], -c1[n-1], ..., -c1[index+1]]
        for j in 0..degree {
            let c1_val = c1.limb(limb_idx)[if j <= index {
                index - j
            } else {
                degree + index - j
            }];
            residues_q[limb_offset + j] = if j <= index {
                c1_val
            } else {
                arith::neg_mod(c1_val, qi)
            };
        }
    }

    Ok(ExtractedBfvLwe {
        residues_q,
        dimension: lwe_dimension,
    })
}

pub fn extract_lwe_from_bfv(
    ct: &Ciphertext,
    params: &BridgeParams,
) -> Result<Vec<ExtractedBfvLwe>, BridgeError> {
    let count = params.num_slots.min(params.bfv_params.degree());
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        out.push(extract_coefficient_lwe(ct, index, params)?);
    }
    Ok(out)
}
