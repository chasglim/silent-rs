use rand::Rng;

use crate::modular::error::{ModularError, ensure_modulus};
use crate::modular::matrix::{ModMatrix, ModVector};
use silent_math::arith::{
    add_mod, mul_mod_u64 as mul_mod, reduce_i128_mod_u64 as reduce_i128,
    round_centered_q_to_p as round_q_to_p,
};

#[derive(Clone, Debug)]
pub struct MatrixMulCrs {
    /// V in Z_q^{n x m}
    pub v: ModMatrix,
    /// U in Z_q^{n x t}
    pub u: ModMatrix,
}

#[derive(Clone, Debug)]
pub struct MatrixMulPublicA {
    /// d_i vectors, i in [ell]
    pub d: Vec<ModVector>,
}

#[derive(Clone, Debug)]
pub struct MatrixMulStateA {
    pub a: ModMatrix,
    pub r: Vec<ModVector>,
}

#[derive(Clone, Debug)]
pub struct MatrixMulPublicB {
    /// (e_{j,0}, e_{j,1}) pairs, j in [k]
    pub e: Vec<(ModVector, ModVector)>,
}

#[derive(Clone, Debug)]
pub struct MatrixMulStateB {
    pub s: Vec<ModVector>,
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

fn matrix_times_vector(matrix: &ModMatrix, vector: &ModVector) -> Result<ModVector, ModularError> {
    if matrix.cols() != vector.len() {
        return Err(ModularError::DimensionMismatch {
            lhs: (matrix.rows(), matrix.cols()),
            rhs: (vector.len(), 1),
            context: "matrix_times_vector",
        });
    }
    if matrix.modulus() != vector.modulus() {
        return Err(ModularError::InvalidParams(
            "matrix_times_vector requires equal moduli",
        ));
    }

    let q = matrix.modulus();
    let mut out = ModVector::new(matrix.rows(), q)?;
    for row in 0..matrix.rows() {
        let mut acc = 0u64;
        for col in 0..matrix.cols() {
            let lhs = matrix.get(row, col)?;
            let rhs = vector.get(col)?;
            acc = add_mod(acc, mul_mod(lhs, rhs, q), q);
        }
        out.set(row, acc as i128)?;
    }
    Ok(out)
}

fn matrix_transpose_times_vector(
    matrix: &ModMatrix,
    vector: &ModVector,
) -> Result<ModVector, ModularError> {
    if matrix.rows() != vector.len() {
        return Err(ModularError::DimensionMismatch {
            lhs: (matrix.rows(), matrix.cols()),
            rhs: (vector.len(), 1),
            context: "matrix_transpose_times_vector",
        });
    }
    if matrix.modulus() != vector.modulus() {
        return Err(ModularError::InvalidParams(
            "matrix_transpose_times_vector requires equal moduli",
        ));
    }

    let q = matrix.modulus();
    let mut out = ModVector::new(matrix.cols(), q)?;
    for col in 0..matrix.cols() {
        let mut acc = 0u64;
        for row in 0..matrix.rows() {
            let lhs = matrix.get(row, col)?;
            let rhs = vector.get(row)?;
            acc = add_mod(acc, mul_mod(lhs, rhs, q), q);
        }
        out.set(col, acc as i128)?;
    }
    Ok(out)
}

/// matrix-multiplication.Setup(1^lambda): sample random CRS matrices V and U in Z_q.
pub fn matrix_mul_setup<R: Rng + ?Sized>(
    n: usize,
    m: usize,
    t: usize,
    q: u64,
    rng: &mut R,
) -> Result<MatrixMulCrs, ModularError> {
    ensure_modulus(q)?;
    if n == 0 || m == 0 || t == 0 {
        return Err(ModularError::InvalidParams(
            "n, m, t must all be strictly positive",
        ));
    }

    let mut v = ModMatrix::zeros(n, m, q)?;
    let mut u = ModMatrix::zeros(n, t, q)?;
    for row in 0..n {
        for col in 0..m {
            let sample = rng.gen_range(0..q);
            v.set(row, col, sample as i128)?;
        }
    }
    for row in 0..n {
        for col in 0..t {
            let sample = rng.gen_range(0..q);
            u.set(row, col, sample as i128)?;
        }
    }
    Ok(MatrixMulCrs { v, u })
}

/// matrix-multiplication.EncodeA(crs, A) .
pub fn encode_a<R: Rng + ?Sized>(
    crs: &MatrixMulCrs,
    a: &ModMatrix,
    noise_bound: i64,
    rng: &mut R,
) -> Result<(MatrixMulPublicA, MatrixMulStateA), ModularError> {
    if a.cols() != crs.v.cols() {
        return Err(ModularError::DimensionMismatch {
            lhs: (a.rows(), a.cols()),
            rhs: (a.rows(), crs.v.cols()),
            context: "encode_a expects A.cols == V.cols",
        });
    }
    if a.modulus() != crs.v.modulus() || a.modulus() != crs.u.modulus() {
        return Err(ModularError::InvalidParams(
            "encode_a requires A, V, U over the same modulus",
        ));
    }

    let q = a.modulus();
    let ell = a.rows();
    let t = crs.u.cols();
    let mut d_list = Vec::with_capacity(ell);
    let mut r_list = Vec::with_capacity(ell);

    for i in 0..ell {
        let mut r_i = ModVector::new(t, q)?;
        for j in 0..t {
            let noise = sample_bounded(noise_bound, rng)?;
            r_i.set(j, reduce_i128(noise as i128, q) as i128)?;
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
        MatrixMulPublicA { d: d_list },
        MatrixMulStateA {
            a: a.clone(),
            r: r_list,
        },
    ))
}

/// matrix-multiplication.EncodeB(crs, B) .
pub fn encode_b<R: Rng + ?Sized>(
    crs: &MatrixMulCrs,
    b: &ModMatrix,
    p: u64,
    noise_bound: i64,
    rng: &mut R,
) -> Result<(MatrixMulPublicB, MatrixMulStateB), ModularError> {
    ensure_modulus(p)?;
    encode_b_internal(crs, b, noise_bound, rng, Some(p))
}

/// Raw embedding variant: computes shares of A*B directly in Z_q (no q->p scaling).
pub fn encode_b_raw<R: Rng + ?Sized>(
    crs: &MatrixMulCrs,
    b: &ModMatrix,
    noise_bound: i64,
    rng: &mut R,
) -> Result<(MatrixMulPublicB, MatrixMulStateB), ModularError> {
    encode_b_internal(crs, b, noise_bound, rng, None)
}

fn encode_b_internal<R: Rng + ?Sized>(
    crs: &MatrixMulCrs,
    b: &ModMatrix,
    noise_bound: i64,
    rng: &mut R,
    scaled_plain_modulus: Option<u64>,
) -> Result<(MatrixMulPublicB, MatrixMulStateB), ModularError> {
    if b.rows() != crs.v.cols() {
        return Err(ModularError::DimensionMismatch {
            lhs: (b.rows(), b.cols()),
            rhs: (crs.v.cols(), b.cols()),
            context: "encode_b expects B.rows == V.cols",
        });
    }
    if b.modulus() != crs.v.modulus() || b.modulus() != crs.u.modulus() {
        return Err(ModularError::InvalidParams(
            "encode_b requires B, V, U over the same modulus",
        ));
    }

    let q = b.modulus();
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
        let mut s_i = ModVector::new(n, q)?;
        for idx in 0..n {
            s_i.set(idx, rng.gen_range(0..q) as i128)?;
        }

        let mut w_i = ModVector::new(m, q)?;
        for idx in 0..m {
            w_i.set(idx, sample_bounded(noise_bound, rng)? as i128)?;
        }
        let mut w_i_prime = ModVector::new(t, q)?;
        for idx in 0..t {
            w_i_prime.set(idx, sample_bounded(noise_bound, rng)? as i128)?;
        }

        let vt_s = matrix_transpose_times_vector(&crs.v, &s_i)?;
        let ut_s = matrix_transpose_times_vector(&crs.u, &s_i)?;
        let b_col = b.col_vec(i)?;

        let mut e0 = ModVector::new(m, q)?;
        for j in 0..m {
            let scaled_b = mul_mod(delta, b_col.get(j)?, q);
            let val = add_mod(add_mod(vt_s.get(j)?, scaled_b, q), w_i.get(j)?, q);
            e0.set(j, val as i128)?;
        }

        let mut e1 = ModVector::new(t, q)?;
        for j in 0..t {
            let val = add_mod(ut_s.get(j)?, w_i_prime.get(j)?, q);
            e1.set(j, val as i128)?;
        }

        e_list.push((e0, e1));
        s_list.push(s_i);
    }

    Ok((
        MatrixMulPublicB { e: e_list },
        MatrixMulStateB { s: s_list },
    ))
}

/// matrix-multiplication.DecodeA(crs, peB, stA) .
pub fn decode_a(
    _crs: &MatrixMulCrs,
    pe_b: &MatrixMulPublicB,
    st_a: &MatrixMulStateA,
) -> Result<ModMatrix, ModularError> {
    let q = st_a.a.modulus();
    let ell = st_a.a.rows();
    let k = pe_b.e.len();
    if st_a.r.len() != ell {
        return Err(ModularError::InvalidParams(
            "state A malformed: r length != number of A rows",
        ));
    }

    let mut z_a = ModMatrix::zeros(ell, k, q)?;
    for i in 0..ell {
        let row_a = st_a.a.row_vec(i)?;
        let r_i = &st_a.r[i];
        for j in 0..k {
            let (e0, e1) = &pe_b.e[j];
            let term0 = e0.dot(&row_a)?;
            let term1 = e1.dot(r_i)?;
            z_a.set(i, j, add_mod(term0, term1, q) as i128)?;
        }
    }
    Ok(z_a)
}

/// matrix-multiplication.DecodeB(crs, peA, stB) .
pub fn decode_b(
    _crs: &MatrixMulCrs,
    pe_a: &MatrixMulPublicA,
    st_b: &MatrixMulStateB,
) -> Result<ModMatrix, ModularError> {
    let k = st_b.s.len();
    let l = pe_a.d.len();
    if k == 0 || l == 0 {
        return Err(ModularError::InvalidParams(
            "decode_b requires non-empty public/state vectors",
        ));
    }

    let q = st_b.s[0].modulus();
    let mut z_b = ModMatrix::zeros(l, k, q)?;
    for i in 0..l {
        let d_i = &pe_a.d[i];
        if d_i.modulus() != q {
            return Err(ModularError::InvalidParams(
                "decode_b modulus mismatch between s_j and d_i",
            ));
        }
        for j in 0..k {
            let val = st_b.s[j].dot(d_i)?;
            z_b.set(i, j, val as i128)?;
        }
    }
    Ok(z_b)
}

/// Subtractive reconstruction used by modular matrix-multiplication:
/// reconstructed = Z_A - Z_B (mod q).
pub fn reconstruct_sub_q(z_a: &ModMatrix, z_b: &ModMatrix) -> Result<ModMatrix, ModularError> {
    if z_a.rows() != z_b.rows() || z_a.cols() != z_b.cols() {
        return Err(ModularError::DimensionMismatch {
            lhs: (z_a.rows(), z_a.cols()),
            rhs: (z_b.rows(), z_b.cols()),
            context: "reconstruct_sub_q",
        });
    }
    if z_a.modulus() != z_b.modulus() {
        return Err(ModularError::InvalidParams(
            "reconstruct_sub_q requires equal moduli",
        ));
    }
    let mut out = z_a.clone();
    out.sub_assign(z_b)?;
    Ok(out)
}

/// Locally rounds a share matrix from Z_q to Z_p.
pub fn round_matrix_q_to_p(matrix_q: &ModMatrix, p: u64) -> Result<ModMatrix, ModularError> {
    ensure_modulus(p)?;
    let q = matrix_q.modulus();
    let mut out = ModMatrix::zeros(matrix_q.rows(), matrix_q.cols(), p)?;
    for row in 0..matrix_q.rows() {
        for col in 0..matrix_q.cols() {
            let rounded = round_q_to_p(matrix_q.get(row, col)?, q, p);
            out.set(row, col, rounded as i128)?;
        }
    }
    Ok(out)
}
