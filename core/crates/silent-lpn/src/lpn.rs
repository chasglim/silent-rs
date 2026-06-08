use crate::error::LpnError;
use crate::field::{add, ensure_modulus, mul};
use crate::linalg::FieldMatrix;
use crate::sampler::SparseVector;
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnParams {
    pub modulus: u64,
    pub input_len: usize,
    pub kappa: usize,
    pub mask_len: usize,
    pub client_weight: usize,
}

impl LpnParams {
    pub fn validate(&self) -> Result<(), LpnError> {
        ensure_modulus(self.modulus)?;
        if self.input_len == 0 || self.kappa == 0 || self.mask_len == 0 {
            return Err(LpnError::InvalidParams(
                "LPN params require positive input_len, kappa, and mask_len",
            ));
        }
        if self.client_weight > self.mask_len {
            return Err(LpnError::InvalidParams(
                "LPN client weight must be <= mask_len",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnPublic {
    pub params: LpnParams,
    pub a: FieldMatrix,
    pub b: FieldMatrix,
}

impl LpnPublic {
    pub fn setup<R: Rng + ?Sized>(params: LpnParams, rng: &mut R) -> Result<Self, LpnError> {
        params.validate()?;
        let a = FieldMatrix::random(params.kappa, params.input_len, params.modulus, rng)?;
        let b = FieldMatrix::random(params.kappa, params.mask_len, params.modulus, rng)?;
        Ok(Self { params, a, b })
    }

    pub fn sample_client_mask<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<SparseVector, LpnError> {
        SparseVector::exact_weight(
            self.params.mask_len,
            self.params.client_weight,
            self.params.modulus,
            rng,
        )
    }

    pub fn digest(&self, x: &[u64], u: &SparseVector) -> Result<Vec<u64>, LpnError> {
        if x.len() != self.params.input_len {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.input_len,
                rhs: x.len(),
                context: "LpnPublic::digest x",
            });
        }
        if u.len() != self.params.mask_len || u.modulus() != self.params.modulus {
            return Err(LpnError::InvalidParams(
                "LpnPublic::digest mask length or modulus mismatch",
            ));
        }
        let q = self.params.modulus;
        let mut out = self.a.mul_vec(x)?;
        for row in 0..self.params.kappa {
            let mut bu = 0;
            for (&pos, &value) in u.positions().iter().zip(u.values().iter()) {
                bu = add(bu, mul(self.b.get(row, pos)?, value, q), q);
            }
            out[row] = add(out[row], bu, q);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::dot;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn digest_matches_dense_formula() {
        let mut rng = StdRng::seed_from_u64(10);
        let params = LpnParams {
            modulus: 97,
            input_len: 4,
            kappa: 3,
            mask_len: 5,
            client_weight: 2,
        };
        let lpn = LpnPublic::setup(params, &mut rng).unwrap();
        let x = vec![1, 2, 3, 4];
        let u = SparseVector::from_positions_values(5, 97, vec![1, 3], vec![7, 9]).unwrap();
        let digest = lpn.digest(&x, &u).unwrap();
        let dense_u = u.to_dense();
        for row in 0..3 {
            let a_row = lpn.a.row(row).unwrap();
            let b_row = lpn.b.row(row).unwrap();
            let expected = (dot(&a_row, &x, 97).unwrap() + dot(&b_row, &dense_u, 97).unwrap()) % 97;
            assert_eq!(digest[row], expected);
        }
    }
}
