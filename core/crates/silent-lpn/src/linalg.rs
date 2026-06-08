use crate::error::LpnError;
use crate::field::{add, ensure_modulus, inv, mul, neg, reduce, sub};
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldMatrix {
    rows: usize,
    cols: usize,
    modulus: u64,
    data: Vec<u64>,
}

impl FieldMatrix {
    pub fn zeros(rows: usize, cols: usize, modulus: u64) -> Result<Self, LpnError> {
        ensure_modulus(modulus)?;
        Ok(Self {
            rows,
            cols,
            modulus,
            data: vec![0; rows * cols],
        })
    }

    pub fn identity(size: usize, modulus: u64) -> Result<Self, LpnError> {
        let mut out = Self::zeros(size, size, modulus)?;
        for i in 0..size {
            out.set(i, i, 1)?;
        }
        Ok(out)
    }

    pub fn random<R: Rng + ?Sized>(
        rows: usize,
        cols: usize,
        modulus: u64,
        rng: &mut R,
    ) -> Result<Self, LpnError> {
        ensure_modulus(modulus)?;
        let mut out = Self::zeros(rows, cols, modulus)?;
        for value in out.data.iter_mut() {
            *value = rng.gen_range(0..modulus);
        }
        Ok(out)
    }

    pub fn from_values(
        rows: usize,
        cols: usize,
        values: Vec<u64>,
        modulus: u64,
    ) -> Result<Self, LpnError> {
        ensure_modulus(modulus)?;
        if values.len() != rows * cols {
            return Err(LpnError::DimensionMismatch {
                lhs: (rows, cols),
                rhs: (1, values.len()),
                context: "FieldMatrix::from_values",
            });
        }
        Ok(Self {
            rows,
            cols,
            modulus,
            data: values.into_iter().map(|v| v % modulus).collect(),
        })
    }

    pub fn from_rows(rows: &[Vec<u64>], modulus: u64) -> Result<Self, LpnError> {
        ensure_modulus(modulus)?;
        let row_count = rows.len();
        let col_count = rows.first().map_or(0, Vec::len);
        let mut data = Vec::with_capacity(row_count * col_count);
        for row in rows {
            if row.len() != col_count {
                return Err(LpnError::InvalidParams(
                    "FieldMatrix::from_rows requires rectangular input",
                ));
            }
            data.extend(row.iter().map(|v| v % modulus));
        }
        Self::from_values(row_count, col_count, data, modulus)
    }

    #[inline]
    fn index(&self, row: usize, col: usize, context: &'static str) -> Result<usize, LpnError> {
        if row >= self.rows {
            return Err(LpnError::IndexOutOfBounds {
                index: row,
                upper_bound: self.rows,
                context,
            });
        }
        if col >= self.cols {
            return Err(LpnError::IndexOutOfBounds {
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

    pub fn as_slice(&self) -> &[u64] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> Result<u64, LpnError> {
        let idx = self.index(row, col, "FieldMatrix::get")?;
        Ok(self.data[idx])
    }

    pub fn set(&mut self, row: usize, col: usize, value: u64) -> Result<(), LpnError> {
        let idx = self.index(row, col, "FieldMatrix::set")?;
        self.data[idx] = value % self.modulus;
        Ok(())
    }

    pub fn set_signed(&mut self, row: usize, col: usize, value: i128) -> Result<(), LpnError> {
        let idx = self.index(row, col, "FieldMatrix::set_signed")?;
        self.data[idx] = reduce(value, self.modulus);
        Ok(())
    }

    pub fn row(&self, row: usize) -> Result<Vec<u64>, LpnError> {
        if row >= self.rows {
            return Err(LpnError::IndexOutOfBounds {
                index: row,
                upper_bound: self.rows,
                context: "FieldMatrix::row",
            });
        }
        let start = row * self.cols;
        Ok(self.data[start..start + self.cols].to_vec())
    }

    pub fn col(&self, col: usize) -> Result<Vec<u64>, LpnError> {
        if col >= self.cols {
            return Err(LpnError::IndexOutOfBounds {
                index: col,
                upper_bound: self.cols,
                context: "FieldMatrix::col",
            });
        }
        let mut out = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            out.push(self.data[row * self.cols + col]);
        }
        Ok(out)
    }

    pub fn mul_vec(&self, vector: &[u64]) -> Result<Vec<u64>, LpnError> {
        if self.cols != vector.len() {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.cols,
                rhs: vector.len(),
                context: "FieldMatrix::mul_vec",
            });
        }
        let mut out = vec![0; self.rows];
        for row in 0..self.rows {
            let mut acc = 0;
            for col in 0..self.cols {
                acc = add(
                    acc,
                    mul(
                        self.data[row * self.cols + col],
                        vector[col] % self.modulus,
                        self.modulus,
                    ),
                    self.modulus,
                );
            }
            out[row] = acc;
        }
        Ok(out)
    }

    pub fn transpose_mul_vec(&self, vector: &[u64]) -> Result<Vec<u64>, LpnError> {
        if self.rows != vector.len() {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.rows,
                rhs: vector.len(),
                context: "FieldMatrix::transpose_mul_vec",
            });
        }
        let mut out = vec![0; self.cols];
        for col in 0..self.cols {
            let mut acc = 0;
            for row in 0..self.rows {
                acc = add(
                    acc,
                    mul(
                        self.data[row * self.cols + col],
                        vector[row] % self.modulus,
                        self.modulus,
                    ),
                    self.modulus,
                );
            }
            out[col] = acc;
        }
        Ok(out)
    }

    pub fn mul_matrix(&self, rhs: &Self) -> Result<Self, LpnError> {
        if self.modulus != rhs.modulus {
            return Err(LpnError::InvalidParams(
                "FieldMatrix::mul_matrix requires equal moduli",
            ));
        }
        if self.cols != rhs.rows {
            return Err(LpnError::DimensionMismatch {
                lhs: (self.rows, self.cols),
                rhs: (rhs.rows, rhs.cols),
                context: "FieldMatrix::mul_matrix",
            });
        }
        let mut out = Self::zeros(self.rows, rhs.cols, self.modulus)?;
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut acc = 0;
                for k in 0..self.cols {
                    acc = add(
                        acc,
                        mul(
                            self.data[i * self.cols + k],
                            rhs.data[k * rhs.cols + j],
                            self.modulus,
                        ),
                        self.modulus,
                    );
                }
                out.set(i, j, acc)?;
            }
        }
        Ok(out)
    }

    pub fn transpose(&self) -> Result<Self, LpnError> {
        let mut out = Self::zeros(self.cols, self.rows, self.modulus)?;
        for row in 0..self.rows {
            for col in 0..self.cols {
                out.set(col, row, self.data[row * self.cols + col])?;
            }
        }
        Ok(out)
    }
}

pub fn dot(lhs: &[u64], rhs: &[u64], modulus: u64) -> Result<u64, LpnError> {
    if lhs.len() != rhs.len() {
        return Err(LpnError::VectorLengthMismatch {
            lhs: lhs.len(),
            rhs: rhs.len(),
            context: "dot",
        });
    }
    let mut acc = 0;
    for (&l, &r) in lhs.iter().zip(rhs.iter()) {
        acc = add(acc, mul(l % modulus, r % modulus, modulus), modulus);
    }
    Ok(acc)
}

pub fn add_vectors(lhs: &[u64], rhs: &[u64], modulus: u64) -> Result<Vec<u64>, LpnError> {
    if lhs.len() != rhs.len() {
        return Err(LpnError::VectorLengthMismatch {
            lhs: lhs.len(),
            rhs: rhs.len(),
            context: "add_vectors",
        });
    }
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| add(l % modulus, r % modulus, modulus))
        .collect())
}

pub fn sub_vectors(lhs: &[u64], rhs: &[u64], modulus: u64) -> Result<Vec<u64>, LpnError> {
    if lhs.len() != rhs.len() {
        return Err(LpnError::VectorLengthMismatch {
            lhs: lhs.len(),
            rhs: rhs.len(),
            context: "sub_vectors",
        });
    }
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| sub(l % modulus, r % modulus, modulus))
        .collect())
}

pub fn scale_add_assign(
    dst: &mut [u64],
    scalar: u64,
    src: &[u64],
    modulus: u64,
) -> Result<(), LpnError> {
    if dst.len() != src.len() {
        return Err(LpnError::VectorLengthMismatch {
            lhs: dst.len(),
            rhs: src.len(),
            context: "scale_add_assign",
        });
    }
    let scalar = scalar % modulus;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = add(*d, mul(s % modulus, scalar, modulus), modulus);
    }
    Ok(())
}

pub fn solve_linear_system(
    matrix: &[Vec<u64>],
    rhs: &[u64],
    modulus: u64,
) -> Result<Option<Vec<u64>>, LpnError> {
    ensure_modulus(modulus)?;
    if matrix.len() != rhs.len() {
        return Err(LpnError::VectorLengthMismatch {
            lhs: matrix.len(),
            rhs: rhs.len(),
            context: "solve_linear_system rows",
        });
    }
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    for row in matrix {
        if row.len() != cols {
            return Err(LpnError::InvalidParams(
                "solve_linear_system requires rectangular matrix",
            ));
        }
    }
    let mut aug = vec![vec![0; cols + 1]; rows];
    for row in 0..rows {
        for col in 0..cols {
            aug[row][col] = matrix[row][col] % modulus;
        }
        aug[row][cols] = rhs[row] % modulus;
    }

    let mut pivot_for_col = vec![None; cols];
    let mut pivot_row = 0usize;
    for col in 0..cols {
        let pivot = (pivot_row..rows).find(|&row| aug[row][col] != 0);
        let Some(pivot) = pivot else {
            continue;
        };
        aug.swap(pivot_row, pivot);
        let inv_pivot = inv(aug[pivot_row][col], modulus).ok_or(LpnError::InvalidParams(
            "solve_linear_system encountered non-invertible pivot",
        ))?;
        for c in col..=cols {
            aug[pivot_row][c] = mul(aug[pivot_row][c], inv_pivot, modulus);
        }
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = aug[row][col];
            if factor == 0 {
                continue;
            }
            for c in col..=cols {
                let term = mul(factor, aug[pivot_row][c], modulus);
                aug[row][c] = sub(aug[row][c], term, modulus);
            }
        }
        pivot_for_col[col] = Some(pivot_row);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }

    for row in &aug {
        if row[..cols].iter().all(|&v| v == 0) && row[cols] != 0 {
            return Ok(None);
        }
    }

    let mut solution = vec![0; cols];
    for col in 0..cols {
        if let Some(row) = pivot_for_col[col] {
            solution[col] = aug[row][cols];
        }
    }
    Ok(Some(solution))
}

pub fn invert_square(matrix: &FieldMatrix) -> Result<FieldMatrix, LpnError> {
    if matrix.rows != matrix.cols {
        return Err(LpnError::DimensionMismatch {
            lhs: (matrix.rows, matrix.cols),
            rhs: (matrix.cols, matrix.cols),
            context: "invert_square",
        });
    }
    let n = matrix.rows;
    let q = matrix.modulus;
    let mut aug = vec![vec![0; n * 2]; n];
    for row in 0..n {
        for col in 0..n {
            aug[row][col] = matrix.get(row, col)?;
        }
        aug[row][n + row] = 1;
    }

    for col in 0..n {
        let pivot = (col..n)
            .find(|&row| aug[row][col] != 0)
            .ok_or(LpnError::InvalidParams(
                "invert_square received a singular matrix",
            ))?;
        aug.swap(col, pivot);
        let inv_pivot = inv(aug[col][col], q).ok_or(LpnError::InvalidParams(
            "invert_square encountered non-invertible pivot",
        ))?;
        for c in col..(2 * n) {
            aug[col][c] = mul(aug[col][c], inv_pivot, q);
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            if factor == 0 {
                continue;
            }
            for c in col..(2 * n) {
                let term = mul(factor, aug[col][c], q);
                aug[row][c] = sub(aug[row][c], term, q);
            }
        }
    }

    let mut out = FieldMatrix::zeros(n, n, q)?;
    for row in 0..n {
        for col in 0..n {
            out.set(row, col, aug[row][n + col])?;
        }
    }
    Ok(out)
}

pub fn nullspace_basis_rows(matrix: &FieldMatrix) -> Result<FieldMatrix, LpnError> {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let q = matrix.modulus;
    let mut rref = vec![vec![0; cols]; rows];
    for row in 0..rows {
        for col in 0..cols {
            rref[row][col] = matrix.get(row, col)?;
        }
    }

    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0usize;
    for col in 0..cols {
        let pivot = (pivot_row..rows).find(|&row| rref[row][col] != 0);
        let Some(pivot) = pivot else {
            continue;
        };
        rref.swap(pivot_row, pivot);
        let inv_pivot = inv(rref[pivot_row][col], q).ok_or(LpnError::InvalidParams(
            "nullspace_basis_rows encountered non-invertible pivot",
        ))?;
        for c in col..cols {
            rref[pivot_row][c] = mul(rref[pivot_row][c], inv_pivot, q);
        }
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = rref[row][col];
            if factor == 0 {
                continue;
            }
            for c in col..cols {
                let term = mul(factor, rref[pivot_row][c], q);
                rref[row][c] = sub(rref[row][c], term, q);
            }
        }
        pivot_cols.push((pivot_row, col));
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }

    let mut is_pivot = vec![false; cols];
    for &(_, col) in &pivot_cols {
        is_pivot[col] = true;
    }
    let free_cols = (0..cols).filter(|&col| !is_pivot[col]).collect::<Vec<_>>();
    let mut basis_rows = Vec::with_capacity(free_cols.len());
    for free_col in free_cols {
        let mut vector = vec![0; cols];
        vector[free_col] = 1;
        for &(row, pivot_col) in &pivot_cols {
            vector[pivot_col] = neg(rref[row][free_col], q);
        }
        basis_rows.push(vector);
    }
    FieldMatrix::from_rows(&basis_rows, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_solve_handles_overdetermined_system() {
        let matrix = vec![vec![1, 1], vec![1, 2], vec![2, 3]];
        let rhs = vec![3, 5, 8];
        let sol = solve_linear_system(&matrix, &rhs, 97)
            .expect("solve")
            .expect("consistent");
        assert_eq!(sol, vec![1, 2]);
    }

    #[test]
    fn nullspace_basis_annihilates_matrix() {
        let matrix = FieldMatrix::from_values(2, 3, vec![1, 0, 1, 0, 1, 1], 97).unwrap();
        let basis = nullspace_basis_rows(&matrix).unwrap();
        assert_eq!(basis.rows(), 1);
        let product = matrix.mul_vec(&basis.row(0).unwrap()).unwrap();
        assert_eq!(product, vec![0, 0]);
    }
}
