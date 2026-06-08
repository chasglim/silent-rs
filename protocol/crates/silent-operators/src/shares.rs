use rand::RngCore;
use silent_math::arith::{add_mod, mul_mod_u64 as mul_mod, sub_mod};
use std::thread;

use crate::error::OperatorError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdditiveShares {
    modulus: u64,
    party0: Vec<u64>,
    party1: Vec<u64>,
}

impl AdditiveShares {
    pub fn new(modulus: u64, party0: Vec<u64>, party1: Vec<u64>) -> Result<Self, OperatorError> {
        if modulus < 2 {
            return Err(OperatorError::InvalidParams("share modulus must be >= 2"));
        }
        if party0.len() != party1.len() {
            return Err(OperatorError::InvalidParams(
                "additive share lengths must match",
            ));
        }
        Ok(Self {
            modulus,
            party0: party0.into_iter().map(|v| v % modulus).collect(),
            party1: party1.into_iter().map(|v| v % modulus).collect(),
        })
    }

    pub fn share_public(values: &[u64], modulus: u64) -> Result<Self, OperatorError> {
        Self::new(
            modulus,
            values.iter().map(|v| v % modulus).collect(),
            vec![0; values.len()],
        )
    }

    pub fn share_with_rng<R: RngCore + ?Sized>(
        values: &[u64],
        modulus: u64,
        rng: &mut R,
    ) -> Result<Self, OperatorError> {
        if modulus < 2 {
            return Err(OperatorError::InvalidParams("share modulus must be >= 2"));
        }
        let mut party0 = Vec::with_capacity(values.len());
        let mut party1 = Vec::with_capacity(values.len());
        for value in values {
            let mask = rng.next_u64() % modulus;
            party0.push(mask);
            party1.push(sub_mod(value % modulus, mask, modulus));
        }
        Self::new(modulus, party0, party1)
    }

    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    pub fn len(&self) -> usize {
        self.party0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.party0.is_empty()
    }

    pub fn party0(&self) -> &[u64] {
        &self.party0
    }

    pub fn party1(&self) -> &[u64] {
        &self.party1
    }

    pub fn reconstruct(&self) -> Vec<u64> {
        self.party0
            .iter()
            .zip(self.party1.iter())
            .map(|(&a, &b)| add_mod(a, b, self.modulus))
            .collect()
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, OperatorError> {
        self.check_compatible(rhs, "add")?;
        let p0 = self
            .party0
            .iter()
            .zip(rhs.party0.iter())
            .map(|(&a, &b)| add_mod(a, b, self.modulus))
            .collect();
        let p1 = self
            .party1
            .iter()
            .zip(rhs.party1.iter())
            .map(|(&a, &b)| add_mod(a, b, self.modulus))
            .collect();
        Self::new(self.modulus, p0, p1)
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, OperatorError> {
        self.check_compatible(rhs, "sub")?;
        let p0 = self
            .party0
            .iter()
            .zip(rhs.party0.iter())
            .map(|(&a, &b)| sub_mod(a, b, self.modulus))
            .collect();
        let p1 = self
            .party1
            .iter()
            .zip(rhs.party1.iter())
            .map(|(&a, &b)| sub_mod(a, b, self.modulus))
            .collect();
        Self::new(self.modulus, p0, p1)
    }

    pub fn add_public_to_party0(&self, values: &[u64]) -> Result<Self, OperatorError> {
        if values.len() != self.len() {
            return Err(OperatorError::InvalidParams(
                "public addend length must match shares",
            ));
        }
        let p0 = self
            .party0
            .iter()
            .zip(values.iter())
            .map(|(&a, &b)| add_mod(a, b % self.modulus, self.modulus))
            .collect();
        Self::new(self.modulus, p0, self.party1.clone())
    }

    pub fn mul_public_scalar(&self, scalar: u64) -> Result<Self, OperatorError> {
        let scalar = scalar % self.modulus;
        let p0 = self
            .party0
            .iter()
            .map(|&v| mul_mod(v, scalar, self.modulus))
            .collect();
        let p1 = self
            .party1
            .iter()
            .map(|&v| mul_mod(v, scalar, self.modulus))
            .collect();
        Self::new(self.modulus, p0, p1)
    }

    pub fn matmul_public(&self, matrix: &[Vec<u64>]) -> Result<Self, OperatorError> {
        if matrix.is_empty() {
            return Err(OperatorError::InvalidParams(
                "public matrix must be non-empty",
            ));
        }
        if matrix[0].len() != self.len() {
            return Err(OperatorError::InvalidParams(
                "public matrix column count must match share length",
            ));
        }
        if matrix.iter().any(|row| row.len() != matrix[0].len()) {
            return Err(OperatorError::InvalidParams(
                "public matrix must be rectangular",
            ));
        }

        let mut p0 = Vec::with_capacity(matrix.len());
        let mut p1 = Vec::with_capacity(matrix.len());
        for row in matrix {
            p0.push(dot_public_table(&self.party0, row, self.modulus)?);
            p1.push(dot_public_table(&self.party1, row, self.modulus)?);
        }
        Self::new(self.modulus, p0, p1)
    }

    /// Matrix-level public linear map on additive shares.
    ///
    /// `self` is interpreted as a row-major `rows x inner_dim` matrix `X`.
    /// `weights` is a row-major `inner_dim x output_dim` matrix `W`.
    /// The output is a row-major `rows x output_dim` sharing of `X * W`.
    ///
    /// This is the CPU tensor-level path used by Transformer examples.  It
    /// avoids per-token allocations and keeps each row's output accumulation in
    /// cache while scanning contiguous rows of `W`.
    pub fn matmul_public_row_major(
        &self,
        rows: usize,
        inner_dim: usize,
        weights: &[u64],
        output_dim: usize,
    ) -> Result<Self, OperatorError> {
        validate_row_major_matmul(self.len(), rows, inner_dim, weights.len(), output_dim)?;
        let p0 = matmul_public_row_major_party(
            &self.party0,
            rows,
            inner_dim,
            weights,
            output_dim,
            self.modulus,
        )?;
        let p1 = matmul_public_row_major_party(
            &self.party1,
            rows,
            inner_dim,
            weights,
            output_dim,
            self.modulus,
        )?;
        Self::new(self.modulus, p0, p1)
    }

    fn check_compatible(&self, rhs: &Self, op: &'static str) -> Result<(), OperatorError> {
        if self.modulus != rhs.modulus {
            return Err(OperatorError::InvalidParams(match op {
                "add" => "cannot add shares with different moduli",
                "sub" => "cannot subtract shares with different moduli",
                _ => "incompatible additive shares",
            }));
        }
        if self.len() != rhs.len() {
            return Err(OperatorError::InvalidParams(
                "additive share lengths must match",
            ));
        }
        Ok(())
    }
}

fn validate_row_major_matmul(
    share_len: usize,
    rows: usize,
    inner_dim: usize,
    weight_len: usize,
    output_dim: usize,
) -> Result<(), OperatorError> {
    if rows == 0 || inner_dim == 0 || output_dim == 0 {
        return Err(OperatorError::InvalidParams(
            "row-major matmul dimensions must be positive",
        ));
    }
    if share_len != rows.saturating_mul(inner_dim) {
        return Err(OperatorError::InvalidParams(
            "share length must equal rows * inner_dim",
        ));
    }
    if weight_len != inner_dim.saturating_mul(output_dim) {
        return Err(OperatorError::InvalidParams(
            "public weight length must equal inner_dim * output_dim",
        ));
    }
    Ok(())
}

fn matmul_public_row_major_party(
    input: &[u64],
    rows: usize,
    inner_dim: usize,
    weights: &[u64],
    output_dim: usize,
    modulus: u64,
) -> Result<Vec<u64>, OperatorError> {
    let workers = usize::min(
        rows,
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    );
    if workers <= 1 || rows < 4 {
        return Ok(matmul_public_row_range(
            input, 0, rows, inner_dim, weights, output_dim, modulus,
        ));
    }

    let chunk = rows.div_ceil(workers);
    let mut parts = Vec::with_capacity(workers);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..rows).step_by(chunk) {
            let end = usize::min(start + chunk, rows);
            handles.push(scope.spawn(move || {
                (
                    start,
                    matmul_public_row_range(
                        input, start, end, inner_dim, weights, output_dim, modulus,
                    ),
                )
            }));
        }
        for handle in handles {
            parts.push(handle.join().map_err(|_| {
                OperatorError::Backend("row-major matmul worker panicked".to_string())
            })?);
        }
        Ok::<_, OperatorError>(())
    })?;

    let mut out = vec![0u64; rows * output_dim];
    for (start, part) in parts {
        let offset = start * output_dim;
        out[offset..offset + part.len()].copy_from_slice(&part);
    }
    Ok(out)
}

fn matmul_public_row_range(
    input: &[u64],
    start_row: usize,
    end_row: usize,
    inner_dim: usize,
    weights: &[u64],
    output_dim: usize,
    modulus: u64,
) -> Vec<u64> {
    let modulus_u128 = modulus as u128;
    let mut out = vec![0u64; (end_row - start_row) * output_dim];
    let mut acc = vec![0u128; output_dim];
    for row in start_row..end_row {
        acc.fill(0);
        let input_row = &input[row * inner_dim..(row + 1) * inner_dim];
        for (k, &share) in input_row.iter().enumerate() {
            let a = (share % modulus) as u128;
            if a == 0 {
                continue;
            }
            let weight_row = &weights[k * output_dim..(k + 1) * output_dim];
            for (slot, &weight) in acc.iter_mut().zip(weight_row.iter()) {
                add_product_acc(slot, a, (weight % modulus) as u128, modulus_u128);
            }
            if k % 256 == 255 {
                for slot in acc.iter_mut() {
                    *slot %= modulus_u128;
                }
            }
        }
        let out_row = &mut out[(row - start_row) * output_dim..(row - start_row + 1) * output_dim];
        for (dst, value) in out_row.iter_mut().zip(acc.iter()) {
            *dst = (*value % modulus_u128) as u64;
        }
    }
    out
}

#[inline]
fn add_product_acc(acc: &mut u128, lhs: u128, rhs: u128, modulus: u128) {
    let product = lhs * rhs;
    if let Some(next) = acc.checked_add(product) {
        *acc = next;
    } else {
        *acc = (*acc % modulus + product % modulus) % modulus;
    }
}

pub fn dot_public_table(
    share_values: &[u64],
    table: &[u64],
    modulus: u64,
) -> Result<u64, OperatorError> {
    if share_values.len() != table.len() {
        return Err(OperatorError::InvalidParams(
            "share vector and public table lengths must match",
        ));
    }
    let mut acc = 0u64;
    for (&share, &value) in share_values.iter().zip(table.iter()) {
        let term = mul_mod(share % modulus, value % modulus, modulus);
        acc = add_mod(acc, term, modulus);
    }
    Ok(acc)
}
