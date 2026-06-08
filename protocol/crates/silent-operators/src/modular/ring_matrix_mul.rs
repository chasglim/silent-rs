use rand::Rng;

use crate::modular::error::{ModularError, ensure_modulus};
use silent_math::arith::{
    add_mod, mul_mod_u64 as mul_mod, reduce_i128_mod_u64 as reduce_i128,
    round_centered_q_to_p as round_q_to_p, sub_mod,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyElem {
    modulus: u64,
    coeffs: Vec<u64>,
}

impl PolyElem {
    pub fn zeros(degree: usize, modulus: u64) -> Result<Self, ModularError> {
        ensure_modulus(modulus)?;
        if degree == 0 {
            return Err(ModularError::InvalidParams("poly degree must be positive"));
        }
        Ok(Self {
            modulus,
            coeffs: vec![0; degree],
        })
    }

    pub fn constant(degree: usize, modulus: u64, value: u64) -> Result<Self, ModularError> {
        let mut out = Self::zeros(degree, modulus)?;
        out.coeffs[0] = value % modulus;
        Ok(out)
    }

    pub fn from_coeffs(coeffs: Vec<u64>, modulus: u64) -> Result<Self, ModularError> {
        ensure_modulus(modulus)?;
        if coeffs.is_empty() {
            return Err(ModularError::InvalidParams(
                "poly coefficient vector must be non-empty",
            ));
        }
        Ok(Self {
            modulus,
            coeffs: coeffs.into_iter().map(|v| v % modulus).collect(),
        })
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len()
    }

    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    pub fn coeff(&self, index: usize) -> Result<u64, ModularError> {
        self.coeffs
            .get(index)
            .copied()
            .ok_or(ModularError::IndexOutOfBounds {
                index,
                upper_bound: self.degree(),
                context: "PolyElem::coeff",
            })
    }

    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    pub fn add_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        self.check_compatible(other, "PolyElem::add_assign")?;
        for (lhs, rhs) in self.coeffs.iter_mut().zip(other.coeffs.iter()) {
            *lhs = add_mod(*lhs, *rhs, self.modulus);
        }
        Ok(())
    }

    pub fn sub_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        self.check_compatible(other, "PolyElem::sub_assign")?;
        for (lhs, rhs) in self.coeffs.iter_mut().zip(other.coeffs.iter()) {
            *lhs = sub_mod(*lhs, *rhs, self.modulus);
        }
        Ok(())
    }

    pub fn scale_assign(&mut self, scalar: u64) {
        let scalar = scalar % self.modulus;
        for coeff in &mut self.coeffs {
            *coeff = mul_mod(*coeff, scalar, self.modulus);
        }
    }

    pub fn negate_assign(&mut self) {
        for coeff in &mut self.coeffs {
            if *coeff != 0 {
                *coeff = self.modulus - *coeff;
            }
        }
    }

    pub fn mul_negacyclic(&self, rhs: &Self) -> Result<Self, ModularError> {
        self.check_compatible(rhs, "PolyElem::mul_negacyclic")?;
        let n = self.degree();
        let q = self.modulus;
        let mut out = vec![0u64; n];
        for i in 0..n {
            for j in 0..n {
                let prod = mul_mod(self.coeffs[i], rhs.coeffs[j], q);
                let idx = i + j;
                if idx < n {
                    out[idx] = add_mod(out[idx], prod, q);
                } else {
                    out[idx - n] = sub_mod(out[idx - n], prod, q);
                }
            }
        }
        Self::from_coeffs(out, q)
    }

    fn check_compatible(&self, other: &Self, context: &'static str) -> Result<(), ModularError> {
        if self.modulus != other.modulus || self.degree() != other.degree() {
            return Err(ModularError::InvalidParams(context));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyVector {
    modulus: u64,
    degree: usize,
    data: Vec<PolyElem>,
}

impl PolyVector {
    pub fn new(len: usize, degree: usize, modulus: u64) -> Result<Self, ModularError> {
        ensure_modulus(modulus)?;
        if degree == 0 {
            return Err(ModularError::InvalidParams("poly degree must be positive"));
        }
        let zero = PolyElem::zeros(degree, modulus)?;
        Ok(Self {
            modulus,
            degree,
            data: vec![zero; len],
        })
    }

    pub fn from_elems(
        elems: Vec<PolyElem>,
        degree: usize,
        modulus: u64,
    ) -> Result<Self, ModularError> {
        ensure_modulus(modulus)?;
        if degree == 0 {
            return Err(ModularError::InvalidParams("poly degree must be positive"));
        }
        for elem in &elems {
            if elem.modulus() != modulus || elem.degree() != degree {
                return Err(ModularError::InvalidParams(
                    "PolyVector::from_elems incompatible element",
                ));
            }
        }
        Ok(Self {
            modulus,
            degree,
            data: elems,
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

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn get(&self, index: usize) -> Result<&PolyElem, ModularError> {
        self.data.get(index).ok_or(ModularError::IndexOutOfBounds {
            index,
            upper_bound: self.len(),
            context: "PolyVector::get",
        })
    }

    pub fn set(&mut self, index: usize, value: PolyElem) -> Result<(), ModularError> {
        if value.modulus() != self.modulus || value.degree() != self.degree {
            return Err(ModularError::InvalidParams(
                "PolyVector::set incompatible value",
            ));
        }
        let upper_bound = self.len();
        let slot = self
            .data
            .get_mut(index)
            .ok_or(ModularError::IndexOutOfBounds {
                index,
                upper_bound,
                context: "PolyVector::set",
            })?;
        *slot = value;
        Ok(())
    }

    pub fn dot(&self, other: &Self) -> Result<PolyElem, ModularError> {
        if self.modulus != other.modulus || self.degree != other.degree {
            return Err(ModularError::InvalidParams(
                "PolyVector::dot incompatible vectors",
            ));
        }
        if self.len() != other.len() {
            return Err(ModularError::VectorLengthMismatch {
                lhs: self.len(),
                rhs: other.len(),
                context: "PolyVector::dot",
            });
        }
        let mut acc = PolyElem::zeros(self.degree, self.modulus)?;
        for (lhs, rhs) in self.data.iter().zip(other.data.iter()) {
            let term = lhs.mul_negacyclic(rhs)?;
            acc.add_assign(&term)?;
        }
        Ok(acc)
    }

    pub fn add_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        if self.modulus != other.modulus || self.degree != other.degree {
            return Err(ModularError::InvalidParams(
                "PolyVector::add_assign incompatible vectors",
            ));
        }
        if self.len() != other.len() {
            return Err(ModularError::VectorLengthMismatch {
                lhs: self.len(),
                rhs: other.len(),
                context: "PolyVector::add_assign",
            });
        }
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            lhs.add_assign(rhs)?;
        }
        Ok(())
    }

    pub fn sub_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        if self.modulus != other.modulus || self.degree != other.degree {
            return Err(ModularError::InvalidParams(
                "PolyVector::sub_assign incompatible vectors",
            ));
        }
        if self.len() != other.len() {
            return Err(ModularError::VectorLengthMismatch {
                lhs: self.len(),
                rhs: other.len(),
                context: "PolyVector::sub_assign",
            });
        }
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            lhs.sub_assign(rhs)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyMatrix {
    rows: usize,
    cols: usize,
    modulus: u64,
    degree: usize,
    data: Vec<PolyElem>,
}

impl PolyMatrix {
    pub fn zeros(
        rows: usize,
        cols: usize,
        degree: usize,
        modulus: u64,
    ) -> Result<Self, ModularError> {
        ensure_modulus(modulus)?;
        if degree == 0 {
            return Err(ModularError::InvalidParams("poly degree must be positive"));
        }
        let zero = PolyElem::zeros(degree, modulus)?;
        Ok(Self {
            rows,
            cols,
            modulus,
            degree,
            data: vec![zero; rows * cols],
        })
    }

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

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn get(&self, row: usize, col: usize) -> Result<&PolyElem, ModularError> {
        let idx = self.index(row, col, "PolyMatrix::get")?;
        Ok(&self.data[idx])
    }

    pub fn set(&mut self, row: usize, col: usize, value: PolyElem) -> Result<(), ModularError> {
        if value.modulus() != self.modulus || value.degree() != self.degree {
            return Err(ModularError::InvalidParams(
                "PolyMatrix::set incompatible value",
            ));
        }
        let idx = self.index(row, col, "PolyMatrix::set")?;
        self.data[idx] = value;
        Ok(())
    }

    pub fn set_constant(&mut self, row: usize, col: usize, value: u64) -> Result<(), ModularError> {
        self.set(
            row,
            col,
            PolyElem::constant(self.degree, self.modulus, value)?,
        )
    }

    pub fn row_vec(&self, row: usize) -> Result<PolyVector, ModularError> {
        if row >= self.rows {
            return Err(ModularError::IndexOutOfBounds {
                index: row,
                upper_bound: self.rows,
                context: "PolyMatrix::row_vec",
            });
        }
        let start = row * self.cols;
        let end = start + self.cols;
        PolyVector::from_elems(self.data[start..end].to_vec(), self.degree, self.modulus)
    }

    pub fn col_vec(&self, col: usize) -> Result<PolyVector, ModularError> {
        if col >= self.cols {
            return Err(ModularError::IndexOutOfBounds {
                index: col,
                upper_bound: self.cols,
                context: "PolyMatrix::col_vec",
            });
        }
        let mut out = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            out.push(self.data[row * self.cols + col].clone());
        }
        PolyVector::from_elems(out, self.degree, self.modulus)
    }

    pub fn slice_cols(&self, start: usize, end: usize) -> Result<Self, ModularError> {
        if start > end || end > self.cols {
            return Err(ModularError::IndexOutOfBounds {
                index: end,
                upper_bound: self.cols,
                context: "PolyMatrix::slice_cols",
            });
        }
        let new_cols = end - start;
        let mut out = Self::zeros(self.rows, new_cols, self.degree, self.modulus)?;
        for row in 0..self.rows {
            for col in 0..new_cols {
                out.set(row, col, self.get(row, start + col)?.clone())?;
            }
        }
        Ok(out)
    }

    pub fn add_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        self.check_compatible(other, "PolyMatrix::add_assign")?;
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            lhs.add_assign(rhs)?;
        }
        Ok(())
    }

    pub fn sub_assign(&mut self, other: &Self) -> Result<(), ModularError> {
        self.check_compatible(other, "PolyMatrix::sub_assign")?;
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            lhs.sub_assign(rhs)?;
        }
        Ok(())
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, ModularError> {
        if self.modulus != rhs.modulus || self.degree != rhs.degree {
            return Err(ModularError::InvalidParams(
                "PolyMatrix::mul incompatible matrices",
            ));
        }
        if self.cols != rhs.rows {
            return Err(ModularError::DimensionMismatch {
                lhs: (self.rows, self.cols),
                rhs: (rhs.rows, rhs.cols),
                context: "PolyMatrix::mul",
            });
        }
        let mut out = Self::zeros(self.rows, rhs.cols, self.degree, self.modulus)?;
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut acc = PolyElem::zeros(self.degree, self.modulus)?;
                for k in 0..self.cols {
                    let term = self.get(i, k)?.mul_negacyclic(rhs.get(k, j)?)?;
                    acc.add_assign(&term)?;
                }
                out.set(i, j, acc)?;
            }
        }
        Ok(out)
    }

    fn check_compatible(&self, other: &Self, context: &'static str) -> Result<(), ModularError> {
        if self.modulus != other.modulus || self.degree != other.degree {
            return Err(ModularError::InvalidParams(context));
        }
        if self.rows != other.rows || self.cols != other.cols {
            return Err(ModularError::DimensionMismatch {
                lhs: (self.rows, self.cols),
                rhs: (other.rows, other.cols),
                context,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct RingMatrixMulCrs {
    pub v: PolyMatrix,
    pub u: PolyMatrix,
}

#[derive(Clone, Debug)]
pub struct RingMatrixMulPublicA {
    pub d: Vec<PolyVector>,
}

#[derive(Clone, Debug)]
pub struct RingMatrixMulStateA {
    pub a: PolyMatrix,
    pub r: Vec<PolyVector>,
}

#[derive(Clone, Debug)]
pub struct RingMatrixMulPublicB {
    pub e: Vec<(PolyVector, PolyVector)>,
}

#[derive(Clone, Debug)]
pub struct RingMatrixMulStateB {
    pub s: Vec<PolyVector>,
}

fn sample_bounded<R: Rng + ?Sized>(bound: i64, rng: &mut R) -> Result<i64, ModularError> {
    if bound < 0 {
        return Err(ModularError::InvalidParams(
            "noise bound must be non-negative",
        ));
    }
    if bound == 0 {
        return Ok(0);
    }
    Ok(rng.gen_range(-bound..=bound))
}

fn sample_noise_poly<R: Rng + ?Sized>(
    degree: usize,
    modulus: u64,
    bound: i64,
    rng: &mut R,
) -> Result<PolyElem, ModularError> {
    let mut coeffs = vec![0u64; degree];
    for coeff in coeffs.iter_mut().take(degree) {
        let sample = sample_bounded(bound, rng)?;
        *coeff = reduce_i128(sample as i128, modulus);
    }
    PolyElem::from_coeffs(coeffs, modulus)
}

fn sample_uniform_poly<R: Rng + ?Sized>(
    degree: usize,
    modulus: u64,
    rng: &mut R,
) -> Result<PolyElem, ModularError> {
    let mut coeffs = vec![0u64; degree];
    for coeff in coeffs.iter_mut().take(degree) {
        *coeff = rng.gen_range(0..modulus);
    }
    PolyElem::from_coeffs(coeffs, modulus)
}

fn matrix_times_vector(
    matrix: &PolyMatrix,
    vector: &PolyVector,
) -> Result<PolyVector, ModularError> {
    if matrix.cols() != vector.len() {
        return Err(ModularError::DimensionMismatch {
            lhs: (matrix.rows(), matrix.cols()),
            rhs: (vector.len(), 1),
            context: "ring_matrix_mul::matrix_times_vector",
        });
    }
    if matrix.modulus() != vector.modulus() || matrix.degree() != vector.degree() {
        return Err(ModularError::InvalidParams(
            "ring_matrix_mul::matrix_times_vector incompatible matrix/vector",
        ));
    }
    let mut out = PolyVector::new(matrix.rows(), matrix.degree(), matrix.modulus())?;
    for row in 0..matrix.rows() {
        let mut acc = PolyElem::zeros(matrix.degree(), matrix.modulus())?;
        for col in 0..matrix.cols() {
            let term = matrix.get(row, col)?.mul_negacyclic(vector.get(col)?)?;
            acc.add_assign(&term)?;
        }
        out.set(row, acc)?;
    }
    Ok(out)
}

fn matrix_transpose_times_vector(
    matrix: &PolyMatrix,
    vector: &PolyVector,
) -> Result<PolyVector, ModularError> {
    if matrix.rows() != vector.len() {
        return Err(ModularError::DimensionMismatch {
            lhs: (matrix.rows(), matrix.cols()),
            rhs: (vector.len(), 1),
            context: "ring_matrix_mul::matrix_transpose_times_vector",
        });
    }
    if matrix.modulus() != vector.modulus() || matrix.degree() != vector.degree() {
        return Err(ModularError::InvalidParams(
            "ring_matrix_mul::matrix_transpose_times_vector incompatible matrix/vector",
        ));
    }
    let mut out = PolyVector::new(matrix.cols(), matrix.degree(), matrix.modulus())?;
    for col in 0..matrix.cols() {
        let mut acc = PolyElem::zeros(matrix.degree(), matrix.modulus())?;
        for row in 0..matrix.rows() {
            let term = matrix.get(row, col)?.mul_negacyclic(vector.get(row)?)?;
            acc.add_assign(&term)?;
        }
        out.set(col, acc)?;
    }
    Ok(out)
}

pub fn ring_matrix_mul_setup<R: Rng + ?Sized>(
    n: usize,
    m: usize,
    t: usize,
    degree: usize,
    q: u64,
    rng: &mut R,
) -> Result<RingMatrixMulCrs, ModularError> {
    ensure_modulus(q)?;
    if n == 0 || m == 0 || t == 0 || degree == 0 {
        return Err(ModularError::InvalidParams(
            "n, m, t and degree must all be strictly positive",
        ));
    }

    let mut v = PolyMatrix::zeros(n, m, degree, q)?;
    let mut u = PolyMatrix::zeros(n, t, degree, q)?;
    for row in 0..n {
        for col in 0..m {
            v.set(row, col, sample_uniform_poly(degree, q, rng)?)?;
        }
    }
    for row in 0..n {
        for col in 0..t {
            u.set(row, col, sample_uniform_poly(degree, q, rng)?)?;
        }
    }
    Ok(RingMatrixMulCrs { v, u })
}

pub fn encode_a_ring<R: Rng + ?Sized>(
    crs: &RingMatrixMulCrs,
    a: &PolyMatrix,
    noise_bound: i64,
    rng: &mut R,
) -> Result<(RingMatrixMulPublicA, RingMatrixMulStateA), ModularError> {
    if a.cols() != crs.v.cols() {
        return Err(ModularError::DimensionMismatch {
            lhs: (a.rows(), a.cols()),
            rhs: (a.rows(), crs.v.cols()),
            context: "encode_a_ring expects A.cols == V.cols",
        });
    }
    if a.modulus() != crs.v.modulus()
        || a.modulus() != crs.u.modulus()
        || a.degree() != crs.v.degree()
        || a.degree() != crs.u.degree()
    {
        return Err(ModularError::InvalidParams(
            "encode_a_ring requires A, V, U over the same ring",
        ));
    }

    let ell = a.rows();
    let t = crs.u.cols();
    let degree = a.degree();
    let q = a.modulus();
    let mut d_list = Vec::with_capacity(ell);
    let mut r_list = Vec::with_capacity(ell);

    for i in 0..ell {
        let mut r_i = PolyVector::new(t, degree, q)?;
        for j in 0..t {
            r_i.set(j, sample_noise_poly(degree, q, noise_bound, rng)?)?;
        }

        let a_row = a.row_vec(i)?;
        let va = matrix_times_vector(&crs.v, &a_row)?;
        let ur = matrix_times_vector(&crs.u, &r_i)?;
        let mut d_i = ur.clone();
        d_i.add_assign(&va)?;

        d_list.push(d_i);
        r_list.push(r_i);
    }

    Ok((
        RingMatrixMulPublicA { d: d_list },
        RingMatrixMulStateA {
            a: a.clone(),
            r: r_list,
        },
    ))
}

pub fn encode_b_ring<R: Rng + ?Sized>(
    crs: &RingMatrixMulCrs,
    b: &PolyMatrix,
    p: u64,
    noise_bound: i64,
    rng: &mut R,
) -> Result<(RingMatrixMulPublicB, RingMatrixMulStateB), ModularError> {
    ensure_modulus(p)?;
    encode_b_ring_internal(crs, b, noise_bound, rng, Some(p))
}

pub fn encode_b_raw_ring<R: Rng + ?Sized>(
    crs: &RingMatrixMulCrs,
    b: &PolyMatrix,
    noise_bound: i64,
    rng: &mut R,
) -> Result<(RingMatrixMulPublicB, RingMatrixMulStateB), ModularError> {
    encode_b_ring_internal(crs, b, noise_bound, rng, None)
}

fn encode_b_ring_internal<R: Rng + ?Sized>(
    crs: &RingMatrixMulCrs,
    b: &PolyMatrix,
    noise_bound: i64,
    rng: &mut R,
    scaled_plain_modulus: Option<u64>,
) -> Result<(RingMatrixMulPublicB, RingMatrixMulStateB), ModularError> {
    if b.rows() != crs.v.cols() {
        return Err(ModularError::DimensionMismatch {
            lhs: (b.rows(), b.cols()),
            rhs: (crs.v.cols(), b.cols()),
            context: "encode_b_ring expects B.rows == V.cols",
        });
    }
    if b.modulus() != crs.v.modulus()
        || b.modulus() != crs.u.modulus()
        || b.degree() != crs.v.degree()
        || b.degree() != crs.u.degree()
    {
        return Err(ModularError::InvalidParams(
            "encode_b_ring requires B, V, U over the same ring",
        ));
    }

    let q = b.modulus();
    let degree = b.degree();
    let n = crs.v.rows();
    let m = crs.v.cols();
    let t = crs.u.cols();
    let k = b.cols();
    let delta = if let Some(p) = scaled_plain_modulus {
        q / p
    } else {
        1
    };

    let mut e_list = Vec::with_capacity(k);
    let mut s_list = Vec::with_capacity(k);

    for i in 0..k {
        let mut s_i = PolyVector::new(n, degree, q)?;
        for idx in 0..n {
            s_i.set(idx, sample_uniform_poly(degree, q, rng)?)?;
        }

        let mut w_i = PolyVector::new(m, degree, q)?;
        for idx in 0..m {
            w_i.set(idx, sample_noise_poly(degree, q, noise_bound, rng)?)?;
        }
        let mut w_i_prime = PolyVector::new(t, degree, q)?;
        for idx in 0..t {
            w_i_prime.set(idx, sample_noise_poly(degree, q, noise_bound, rng)?)?;
        }

        let vt_s = matrix_transpose_times_vector(&crs.v, &s_i)?;
        let ut_s = matrix_transpose_times_vector(&crs.u, &s_i)?;
        let b_col = b.col_vec(i)?;

        let mut e0 = PolyVector::new(m, degree, q)?;
        for j in 0..m {
            let mut scaled_b = b_col.get(j)?.clone();
            scaled_b.scale_assign(delta);
            let mut val = vt_s.get(j)?.clone();
            val.add_assign(&scaled_b)?;
            val.add_assign(w_i.get(j)?)?;
            e0.set(j, val)?;
        }

        let mut e1 = PolyVector::new(t, degree, q)?;
        for j in 0..t {
            let mut val = ut_s.get(j)?.clone();
            val.add_assign(w_i_prime.get(j)?)?;
            e1.set(j, val)?;
        }

        e_list.push((e0, e1));
        s_list.push(s_i);
    }

    Ok((
        RingMatrixMulPublicB { e: e_list },
        RingMatrixMulStateB { s: s_list },
    ))
}

pub fn decode_a_ring(
    _crs: &RingMatrixMulCrs,
    pe_b: &RingMatrixMulPublicB,
    st_a: &RingMatrixMulStateA,
) -> Result<PolyMatrix, ModularError> {
    let ell = st_a.a.rows();
    let k = pe_b.e.len();
    if st_a.r.len() != ell {
        return Err(ModularError::InvalidParams(
            "decode_a_ring: malformed state A (r length mismatch)",
        ));
    }
    let mut z_a = PolyMatrix::zeros(ell, k, st_a.a.degree(), st_a.a.modulus())?;
    for i in 0..ell {
        let row_a = st_a.a.row_vec(i)?;
        let r_i = &st_a.r[i];
        for j in 0..k {
            let (e0, e1) = &pe_b.e[j];
            let mut val = e0.dot(&row_a)?;
            let term1 = e1.dot(r_i)?;
            val.add_assign(&term1)?;
            z_a.set(i, j, val)?;
        }
    }
    Ok(z_a)
}

pub fn decode_b_ring(
    _crs: &RingMatrixMulCrs,
    pe_a: &RingMatrixMulPublicA,
    st_b: &RingMatrixMulStateB,
) -> Result<PolyMatrix, ModularError> {
    let k = st_b.s.len();
    let l = pe_a.d.len();
    if k == 0 || l == 0 {
        return Err(ModularError::InvalidParams(
            "decode_b_ring requires non-empty public/state vectors",
        ));
    }
    let degree = st_b.s[0].degree();
    let q = st_b.s[0].modulus();
    let mut z_b = PolyMatrix::zeros(l, k, degree, q)?;
    for i in 0..l {
        let d_i = &pe_a.d[i];
        if d_i.degree() != degree || d_i.modulus() != q {
            return Err(ModularError::InvalidParams(
                "decode_b_ring modulus/degree mismatch between s_j and d_i",
            ));
        }
        for j in 0..k {
            z_b.set(i, j, st_b.s[j].dot(d_i)?)?;
        }
    }
    Ok(z_b)
}

pub fn reconstruct_sub_q_ring(
    z_a: &PolyMatrix,
    z_b: &PolyMatrix,
) -> Result<PolyMatrix, ModularError> {
    if z_a.rows() != z_b.rows() || z_a.cols() != z_b.cols() {
        return Err(ModularError::DimensionMismatch {
            lhs: (z_a.rows(), z_a.cols()),
            rhs: (z_b.rows(), z_b.cols()),
            context: "reconstruct_sub_q_ring",
        });
    }
    if z_a.modulus() != z_b.modulus() || z_a.degree() != z_b.degree() {
        return Err(ModularError::InvalidParams(
            "reconstruct_sub_q_ring requires equal ring parameters",
        ));
    }
    let mut out = z_a.clone();
    out.sub_assign(z_b)?;
    Ok(out)
}

pub fn round_matrix_q_to_p_ring(matrix_q: &PolyMatrix, p: u64) -> Result<PolyMatrix, ModularError> {
    ensure_modulus(p)?;
    let q = matrix_q.modulus();
    let degree = matrix_q.degree();
    let mut out = PolyMatrix::zeros(matrix_q.rows(), matrix_q.cols(), degree, p)?;
    for row in 0..matrix_q.rows() {
        for col in 0..matrix_q.cols() {
            let in_elem = matrix_q.get(row, col)?;
            let mut coeffs = vec![0u64; degree];
            for (idx, coeff) in coeffs.iter_mut().enumerate().take(degree) {
                *coeff = round_q_to_p(in_elem.coeff(idx)?, q, p);
            }
            out.set(row, col, PolyElem::from_coeffs(coeffs, p)?)?;
        }
    }
    Ok(out)
}
