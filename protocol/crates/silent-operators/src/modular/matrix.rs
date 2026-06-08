use crate::modular::error::{ModularError, ensure_modulus};
use silent_math::arith::{
    add_mod, mul_mod_u64 as mul_mod, reduce_i128_mod_u64 as reduce_i128, sub_mod,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModVector {
    modulus: u64,
    data: Vec<u64>,
}

impl ModVector {
    pub fn new(len: usize, modulus: u64) -> Result<Self, ModularError> {
        ensure_modulus(modulus)?;
        Ok(Self {
            modulus,
            data: vec![0; len],
        })
    }

    pub fn from_values(values: Vec<u64>, modulus: u64) -> Result<Self, ModularError> {
        ensure_modulus(modulus)?;
        let normalized = values
            .into_iter()
            .map(|v| v % modulus)
            .collect::<Vec<u64>>();
        Ok(Self {
            modulus,
            data: normalized,
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.data
    }

    pub fn get(&self, index: usize) -> Result<u64, ModularError> {
        self.data
            .get(index)
            .copied()
            .ok_or(ModularError::IndexOutOfBounds {
                index,
                upper_bound: self.len(),
                context: "ModVector::get",
            })
    }

    pub fn set(&mut self, index: usize, value: i128) -> Result<(), ModularError> {
        let upper_bound = self.len();
        let slot = self
            .data
            .get_mut(index)
            .ok_or(ModularError::IndexOutOfBounds {
                index,
                upper_bound,
                context: "ModVector::set",
            })?;
        *slot = reduce_i128(value, self.modulus);
        Ok(())
    }

    pub fn dot(&self, other: &Self) -> Result<u64, ModularError> {
        if self.modulus != other.modulus {
            return Err(ModularError::InvalidParams(
                "dot product requires equal moduli",
            ));
        }
        if self.len() != other.len() {
            return Err(ModularError::VectorLengthMismatch {
                lhs: self.len(),
                rhs: other.len(),
                context: "ModVector::dot",
            });
        }

        let mut acc = 0u64;
        for (lhs, rhs) in self.data.iter().zip(other.data.iter()) {
            acc = add_mod(acc, mul_mod(*lhs, *rhs, self.modulus), self.modulus);
        }
        Ok(acc)
    }

    pub fn add_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        if self.modulus != other.modulus {
            return Err(ModularError::InvalidParams(
                "vector add_assign requires equal moduli",
            ));
        }
        if self.len() != other.len() {
            return Err(ModularError::VectorLengthMismatch {
                lhs: self.len(),
                rhs: other.len(),
                context: "ModVector::add_assign",
            });
        }
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            *lhs = add_mod(*lhs, *rhs, self.modulus);
        }
        Ok(())
    }

    pub fn sub_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        if self.modulus != other.modulus {
            return Err(ModularError::InvalidParams(
                "vector sub_assign requires equal moduli",
            ));
        }
        if self.len() != other.len() {
            return Err(ModularError::VectorLengthMismatch {
                lhs: self.len(),
                rhs: other.len(),
                context: "ModVector::sub_assign",
            });
        }
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            *lhs = sub_mod(*lhs, *rhs, self.modulus);
        }
        Ok(())
    }

    pub fn scale_assign(&mut self, scalar: u64) {
        let scalar = scalar % self.modulus;
        for value in &mut self.data {
            *value = mul_mod(*value, scalar, self.modulus);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModMatrix {
    rows: usize,
    cols: usize,
    modulus: u64,
    data: Vec<u64>,
}

impl ModMatrix {
    pub fn zeros(rows: usize, cols: usize, modulus: u64) -> Result<Self, ModularError> {
        ensure_modulus(modulus)?;
        Ok(Self {
            rows,
            cols,
            modulus,
            data: vec![0; rows * cols],
        })
    }

    pub fn from_values(
        rows: usize,
        cols: usize,
        values: Vec<u64>,
        modulus: u64,
    ) -> Result<Self, ModularError> {
        ensure_modulus(modulus)?;
        if values.len() != rows * cols {
            return Err(ModularError::DimensionMismatch {
                lhs: (rows, cols),
                rhs: (1, values.len()),
                context: "ModMatrix::from_values (rows*cols != values.len())",
            });
        }
        let normalized = values.into_iter().map(|v| v % modulus).collect();
        Ok(Self {
            rows,
            cols,
            modulus,
            data: normalized,
        })
    }

    #[inline]
    fn index(&self, row: usize, col: usize, context: &'static str) -> Result<usize, ModularError> {
        if row >= self.rows {
            return Err(ModularError::IndexOutOfBounds {
                index: row,
                upper_bound: self.rows,
                context,
            });
        }
        if col >= self.cols {
            return Err(ModularError::IndexOutOfBounds {
                index: col,
                upper_bound: self.cols,
                context,
            });
        }
        Ok(row * self.cols + col)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    pub fn get(&self, row: usize, col: usize) -> Result<u64, ModularError> {
        let idx = self.index(row, col, "ModMatrix::get")?;
        Ok(self.data[idx])
    }

    pub fn set(&mut self, row: usize, col: usize, value: i128) -> Result<(), ModularError> {
        let idx = self.index(row, col, "ModMatrix::set")?;
        self.data[idx] = reduce_i128(value, self.modulus);
        Ok(())
    }

    pub fn row_vec(&self, row: usize) -> Result<ModVector, ModularError> {
        if row >= self.rows {
            return Err(ModularError::IndexOutOfBounds {
                index: row,
                upper_bound: self.rows,
                context: "ModMatrix::row_vec",
            });
        }
        let start = row * self.cols;
        let end = start + self.cols;
        ModVector::from_values(self.data[start..end].to_vec(), self.modulus)
    }

    pub fn col_vec(&self, col: usize) -> Result<ModVector, ModularError> {
        if col >= self.cols {
            return Err(ModularError::IndexOutOfBounds {
                index: col,
                upper_bound: self.cols,
                context: "ModMatrix::col_vec",
            });
        }
        let mut out = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            out.push(self.data[row * self.cols + col]);
        }
        ModVector::from_values(out, self.modulus)
    }

    pub fn add_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        if self.modulus != other.modulus {
            return Err(ModularError::InvalidParams(
                "matrix add_assign requires equal moduli",
            ));
        }
        if self.rows != other.rows || self.cols != other.cols {
            return Err(ModularError::DimensionMismatch {
                lhs: (self.rows, self.cols),
                rhs: (other.rows, other.cols),
                context: "ModMatrix::add_assign",
            });
        }
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            *lhs = add_mod(*lhs, *rhs, self.modulus);
        }
        Ok(())
    }

    pub fn sub_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        if self.modulus != other.modulus {
            return Err(ModularError::InvalidParams(
                "matrix sub_assign requires equal moduli",
            ));
        }
        if self.rows != other.rows || self.cols != other.cols {
            return Err(ModularError::DimensionMismatch {
                lhs: (self.rows, self.cols),
                rhs: (other.rows, other.cols),
                context: "ModMatrix::sub_assign",
            });
        }
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            *lhs = sub_mod(*lhs, *rhs, self.modulus);
        }
        Ok(())
    }

    pub fn scale_assign(&mut self, scalar: u64) {
        let scalar = scalar % self.modulus;
        for value in &mut self.data {
            *value = mul_mod(*value, scalar, self.modulus);
        }
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, ModularError> {
        if self.modulus != rhs.modulus {
            return Err(ModularError::InvalidParams(
                "matrix multiplication requires equal moduli",
            ));
        }
        if self.cols != rhs.rows {
            return Err(ModularError::DimensionMismatch {
                lhs: (self.rows, self.cols),
                rhs: (rhs.rows, rhs.cols),
                context: "ModMatrix::mul",
            });
        }

        let mut out = ModMatrix::zeros(self.rows, rhs.cols, self.modulus)?;
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut acc = 0u64;
                for k in 0..self.cols {
                    let lhs = self.get(i, k)?;
                    let rv = rhs.get(k, j)?;
                    acc = add_mod(acc, mul_mod(lhs, rv, self.modulus), self.modulus);
                }
                out.set(i, j, acc as i128)?;
            }
        }
        Ok(out)
    }

    pub fn slice_cols(&self, start: usize, end: usize) -> Result<Self, ModularError> {
        if start > end || end > self.cols {
            return Err(ModularError::InvalidParams(
                "invalid column range in ModMatrix::slice_cols",
            ));
        }
        let width = end - start;
        let mut out = ModMatrix::zeros(self.rows, width, self.modulus)?;
        for row in 0..self.rows {
            for col in 0..width {
                out.set(row, col, self.get(row, start + col)? as i128)?;
            }
        }
        Ok(out)
    }
}
