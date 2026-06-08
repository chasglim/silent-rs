use crate::error::LpnError;
use crate::field::{add, ensure_modulus, mul};
use rand::Rng;
use rand::seq::SliceRandom;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseVector {
    len: usize,
    modulus: u64,
    positions: Vec<usize>,
    values: Vec<u64>,
}

impl SparseVector {
    pub fn exact_weight<R: Rng + ?Sized>(
        len: usize,
        weight: usize,
        modulus: u64,
        rng: &mut R,
    ) -> Result<Self, LpnError> {
        ensure_modulus(modulus)?;
        if weight > len {
            return Err(LpnError::InvalidParams(
                "SparseVector::exact_weight requires weight <= len",
            ));
        }
        let mut positions = (0..len).collect::<Vec<_>>();
        positions.shuffle(rng);
        positions.truncate(weight);
        positions.sort_unstable();

        let values = positions
            .iter()
            .map(|_| rng.gen_range(1..modulus))
            .collect::<Vec<_>>();
        Ok(Self {
            len,
            modulus,
            positions,
            values,
        })
    }

    pub fn from_positions_values(
        len: usize,
        modulus: u64,
        positions: Vec<usize>,
        values: Vec<u64>,
    ) -> Result<Self, LpnError> {
        ensure_modulus(modulus)?;
        if positions.len() != values.len() {
            return Err(LpnError::VectorLengthMismatch {
                lhs: positions.len(),
                rhs: values.len(),
                context: "SparseVector::from_positions_values",
            });
        }
        let mut pairs = positions
            .into_iter()
            .zip(values.into_iter())
            .map(|(pos, value)| {
                if pos >= len {
                    Err(LpnError::IndexOutOfBounds {
                        index: pos,
                        upper_bound: len,
                        context: "SparseVector::from_positions_values",
                    })
                } else if value % modulus == 0 {
                    Err(LpnError::InvalidParams(
                        "SparseVector values must be nonzero modulo modulus",
                    ))
                } else {
                    Ok((pos, value % modulus))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        pairs.sort_unstable_by_key(|(pos, _)| *pos);
        for window in pairs.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(LpnError::InvalidParams(
                    "SparseVector positions must be unique",
                ));
            }
        }
        let (positions, values) = pairs.into_iter().unzip();
        Ok(Self {
            len,
            modulus,
            positions,
            values,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    pub fn weight(&self) -> usize {
        self.positions.len()
    }

    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    pub fn values(&self) -> &[u64] {
        &self.values
    }

    pub fn to_dense(&self) -> Vec<u64> {
        let mut dense = vec![0; self.len];
        for (&pos, &value) in self.positions.iter().zip(self.values.iter()) {
            dense[pos] = value;
        }
        dense
    }

    pub fn dot_dense(&self, dense: &[u64]) -> Result<u64, LpnError> {
        if dense.len() != self.len {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.len,
                rhs: dense.len(),
                context: "SparseVector::dot_dense",
            });
        }
        let mut acc = 0;
        for (&pos, &value) in self.positions.iter().zip(self.values.iter()) {
            acc = add(
                acc,
                mul(value, dense[pos] % self.modulus, self.modulus),
                self.modulus,
            );
        }
        Ok(acc)
    }
}

pub fn sample_bernoulli_error<R: Rng + ?Sized>(
    len: usize,
    tau: f64,
    modulus: u64,
    rng: &mut R,
) -> Result<Vec<u64>, LpnError> {
    ensure_modulus(modulus)?;
    if !(0.0..=1.0).contains(&tau) {
        return Err(LpnError::InvalidParams(
            "sample_bernoulli_error requires tau in [0,1]",
        ));
    }
    let mut out = vec![0; len];
    for value in &mut out {
        if rng.gen_bool(tau) {
            *value = rng.gen_range(1..modulus);
        }
    }
    Ok(out)
}

pub fn sample_exact_weight_error<R: Rng + ?Sized>(
    len: usize,
    weight: usize,
    modulus: u64,
    rng: &mut R,
) -> Result<Vec<u64>, LpnError> {
    Ok(SparseVector::exact_weight(len, weight, modulus, rng)?.to_dense())
}

pub fn sample_uniform_vector<R: Rng + ?Sized>(
    len: usize,
    modulus: u64,
    rng: &mut R,
) -> Result<Vec<u64>, LpnError> {
    ensure_modulus(modulus)?;
    Ok((0..len).map(|_| rng.gen_range(0..modulus)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn exact_weight_sampler_has_requested_weight() {
        let mut rng = StdRng::seed_from_u64(9);
        let sample = SparseVector::exact_weight(32, 7, 97, &mut rng).unwrap();
        assert_eq!(sample.weight(), 7);
        assert_eq!(sample.to_dense().iter().filter(|&&v| v != 0).count(), 7);
    }
}
