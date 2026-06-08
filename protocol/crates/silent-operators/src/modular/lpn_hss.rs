use rand::Rng;
use silent_lpn::{LpnBksHssParams, LpnBksHssPublic, LpnError};
use silent_math::arith::add_mod;

use crate::modular::error::{ModularError, ensure_modulus};
use crate::modular::matrix::{ModMatrix, ModVector};

impl From<LpnError> for ModularError {
    fn from(value: LpnError) -> Self {
        ModularError::Backend(value.to_string())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LpnHssMatrixMulConfig {
    pub kappa: usize,
    pub mask_len: usize,
    pub client_weight: usize,
    pub code_len: usize,
    pub decode_radius: usize,
    pub noise_rate_x: f64,
    pub noise_rate_u: f64,
    pub block_width: Option<usize>,
}

impl LpnHssMatrixMulConfig {
    pub fn toy(output_len: usize) -> Self {
        Self {
            kappa: 8,
            mask_len: 8,
            client_weight: 2,
            code_len: output_len + 4,
            decode_radius: 2,
            noise_rate_x: 0.0,
            noise_rate_u: 0.0,
            block_width: None,
        }
    }

    pub fn to_hss_params(
        &self,
        modulus: u64,
        input_len: usize,
        output_len: usize,
    ) -> LpnBksHssParams {
        LpnBksHssParams {
            modulus,
            input_len,
            output_len,
            code_len: self.code_len,
            decode_radius: self.decode_radius,
            kappa: self.kappa,
            mask_len: self.mask_len,
            client_weight: self.client_weight,
            noise_rate_x: self.noise_rate_x,
            noise_rate_u: self.noise_rate_u,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnHssMatrixMulOutput {
    pub client_share: ModMatrix,
    pub server_share: ModMatrix,
    pub block_count: usize,
    pub decoded_error_weight: usize,
    pub comm_field_elements: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnHssVectorOutput {
    pub client_share: ModVector,
    pub server_share: ModVector,
    pub decoded_error_weight: usize,
    pub comm_field_elements: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnHssMatrixTripleOutput {
    pub a_client_share: ModMatrix,
    pub a_server_share: ModMatrix,
    pub b_client_share: ModMatrix,
    pub b_server_share: ModMatrix,
    pub c_client_share: ModMatrix,
    pub c_server_share: ModMatrix,
    pub block_count: usize,
    pub decoded_error_weight: usize,
    pub comm_field_elements: usize,
}

pub fn matrix_vector_lpn_hss<R: Rng + ?Sized>(
    server_matrix: &ModMatrix,
    client_vector: &ModVector,
    cfg: &LpnHssMatrixMulConfig,
    rng: &mut R,
) -> Result<LpnHssVectorOutput, ModularError> {
    if server_matrix.modulus() != client_vector.modulus() {
        return Err(ModularError::InvalidParams(
            "matrix_vector_lpn_hss requires equal moduli",
        ));
    }
    if server_matrix.cols() != client_vector.len() {
        return Err(ModularError::DimensionMismatch {
            lhs: (server_matrix.rows(), server_matrix.cols()),
            rhs: (client_vector.len(), 1),
            context: "matrix_vector_lpn_hss",
        });
    }

    let q = server_matrix.modulus();
    ensure_modulus(q)?;
    let params = cfg.to_hss_params(q, server_matrix.cols(), server_matrix.rows());
    params.validate().map_err(ModularError::from)?;
    let hss = LpnBksHssPublic::setup(params.clone(), rng)?;
    let mut server_vectors = Vec::with_capacity(server_matrix.rows());
    for row in 0..server_matrix.rows() {
        server_vectors.push(server_matrix.row_vec(row)?.as_slice().to_vec());
    }

    let preprocessing = hss.server_preprocess_encodings(&server_vectors, rng)?;
    let digest = hss.client_digest(client_vector.as_slice(), rng)?;
    let client_codeword_share = hss.client_codeword_share_from_encodings(
        client_vector.as_slice(),
        &digest.client_mask,
        &preprocessing.encodings,
    )?;
    let server_codeword_share =
        hss.server_codeword_share_from_preprocessing(&digest.digest, &preprocessing)?;
    let crsc = hss.syndrome_key_crsc_convert(
        &client_codeword_share,
        &server_codeword_share,
        &digest.digest,
        &preprocessing.syndrome_key,
    )?;
    let decoded_error_weight = crsc.error.iter().filter(|&&value| value != 0).count();
    let client_share = ModVector::from_values(crsc.client_share, q)?;
    let server_share = ModVector::from_values(crsc.server_share, q)?;
    Ok(LpnHssVectorOutput {
        client_share,
        server_share,
        decoded_error_weight,
        comm_field_elements: syndrome_key_invocation_field_elements(&params),
    })
}

pub fn matrix_mul_lpn_hss<R: Rng + ?Sized>(
    lhs: &ModMatrix,
    rhs: &ModMatrix,
    cfg: &LpnHssMatrixMulConfig,
    rng: &mut R,
) -> Result<LpnHssMatrixMulOutput, ModularError> {
    if lhs.modulus() != rhs.modulus() {
        return Err(ModularError::InvalidParams(
            "matrix_mul_lpn_hss requires equal moduli",
        ));
    }
    if lhs.cols() != rhs.rows() {
        return Err(ModularError::DimensionMismatch {
            lhs: (lhs.rows(), lhs.cols()),
            rhs: (rhs.rows(), rhs.cols()),
            context: "matrix_mul_lpn_hss",
        });
    }
    if lhs.rows() == 0 || lhs.cols() == 0 || rhs.cols() == 0 {
        return Err(ModularError::InvalidParams(
            "matrix_mul_lpn_hss requires non-empty matrices",
        ));
    }

    let q = lhs.modulus();
    ensure_modulus(q)?;
    let block_width = cfg.block_width.unwrap_or(lhs.cols()).min(lhs.cols());
    if block_width == 0 {
        return Err(ModularError::InvalidParams(
            "matrix_mul_lpn_hss block width must be positive",
        ));
    }

    let mut client_share = ModMatrix::zeros(lhs.rows(), rhs.cols(), q)?;
    let mut server_share = ModMatrix::zeros(lhs.rows(), rhs.cols(), q)?;
    let mut block_count = 0usize;
    let mut decoded_error_weight = 0usize;
    let mut comm_field_elements = 0usize;

    for start in (0..lhs.cols()).step_by(block_width) {
        let end = (start + block_width).min(lhs.cols());
        let width = end - start;
        let params = cfg.to_hss_params(q, width, rhs.cols());
        params.validate().map_err(ModularError::from)?;
        let hss = LpnBksHssPublic::setup(params.clone(), rng)?;
        let server_vectors = rhs_column_blocks(rhs, start, end)?;
        for row in 0..lhs.rows() {
            let preprocessing = hss.server_preprocess_encodings(&server_vectors, rng)?;
            let x = lhs_row_block(lhs, row, start, end)?;
            let digest = hss.client_digest(&x, rng)?;
            let client_codeword_share = hss.client_codeword_share_from_encodings(
                &x,
                &digest.client_mask,
                &preprocessing.encodings,
            )?;
            let server_codeword_share =
                hss.server_codeword_share_from_preprocessing(&digest.digest, &preprocessing)?;
            let crsc = hss.syndrome_key_crsc_convert(
                &client_codeword_share,
                &server_codeword_share,
                &digest.digest,
                &preprocessing.syndrome_key,
            )?;
            for col in 0..rhs.cols() {
                add_matrix_entry(&mut client_share, row, col, crsc.client_share[col])?;
                add_matrix_entry(&mut server_share, row, col, crsc.server_share[col])?;
            }
            decoded_error_weight += crsc.error.iter().filter(|&&value| value != 0).count();
            comm_field_elements += syndrome_key_invocation_field_elements(&params);
        }
        block_count += 1;
    }

    Ok(LpnHssMatrixMulOutput {
        client_share,
        server_share,
        block_count,
        decoded_error_weight,
        comm_field_elements,
    })
}

pub fn matrix_triple_lpn_hss<R: Rng + ?Sized>(
    rows: usize,
    inner: usize,
    cols: usize,
    modulus: u64,
    cfg: &LpnHssMatrixMulConfig,
    rng: &mut R,
) -> Result<LpnHssMatrixTripleOutput, ModularError> {
    ensure_modulus(modulus)?;
    if rows == 0 || inner == 0 || cols == 0 {
        return Err(ModularError::InvalidParams(
            "matrix_triple_lpn_hss dimensions must be positive",
        ));
    }

    let a_client_share = random_matrix(rows, inner, modulus, rng)?;
    let b_client_share = random_matrix(inner, cols, modulus, rng)?;
    let a_server_share = random_matrix(rows, inner, modulus, rng)?;
    let b_server_share = random_matrix(inner, cols, modulus, rng)?;

    let client_local = a_client_share.mul(&b_client_share)?;
    let server_local = a_server_share.mul(&b_server_share)?;
    let cross_u = matrix_mul_lpn_hss(&a_client_share, &b_server_share, cfg, rng)?;
    let b_client_t = matrix_transpose(&b_client_share)?;
    let a_server_t = matrix_transpose(&a_server_share)?;
    let mut transpose_cfg = cfg.clone();
    transpose_cfg.block_width = cfg.block_width.map(|width| width.min(inner));
    let cross_v_t = matrix_mul_lpn_hss(&b_client_t, &a_server_t, &transpose_cfg, rng)?;
    let cross_v_client = matrix_transpose(&cross_v_t.client_share)?;
    let cross_v_server = matrix_transpose(&cross_v_t.server_share)?;
    let mut c_client_share = client_local;
    c_client_share.add_assign(&cross_u.client_share)?;
    c_client_share.add_assign(&cross_v_client)?;
    let mut c_server_share = server_local;
    c_server_share.add_assign(&cross_u.server_share)?;
    c_server_share.add_assign(&cross_v_server)?;

    Ok(LpnHssMatrixTripleOutput {
        a_client_share,
        a_server_share,
        b_client_share,
        b_server_share,
        c_client_share,
        c_server_share,
        block_count: cross_u.block_count + cross_v_t.block_count,
        decoded_error_weight: cross_u.decoded_error_weight + cross_v_t.decoded_error_weight,
        comm_field_elements: cross_u.comm_field_elements + cross_v_t.comm_field_elements,
    })
}

pub fn reconstruct_add_q(lhs: &ModMatrix, rhs: &ModMatrix) -> Result<ModMatrix, ModularError> {
    if lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols() {
        return Err(ModularError::DimensionMismatch {
            lhs: (lhs.rows(), lhs.cols()),
            rhs: (rhs.rows(), rhs.cols()),
            context: "reconstruct_add_q",
        });
    }
    if lhs.modulus() != rhs.modulus() {
        return Err(ModularError::InvalidParams(
            "reconstruct_add_q requires equal moduli",
        ));
    }
    let q = lhs.modulus();
    let mut out = ModMatrix::zeros(lhs.rows(), lhs.cols(), q)?;
    for row in 0..lhs.rows() {
        for col in 0..lhs.cols() {
            out.set(
                row,
                col,
                add_mod(lhs.get(row, col)?, rhs.get(row, col)?, q) as i128,
            )?;
        }
    }
    Ok(out)
}

fn rhs_column_blocks(
    rhs: &ModMatrix,
    start: usize,
    end: usize,
) -> Result<Vec<Vec<u64>>, ModularError> {
    let mut out = Vec::with_capacity(rhs.cols());
    for col in 0..rhs.cols() {
        let mut vector = Vec::with_capacity(end - start);
        for row in start..end {
            vector.push(rhs.get(row, col)?);
        }
        out.push(vector);
    }
    Ok(out)
}

fn lhs_row_block(
    lhs: &ModMatrix,
    row: usize,
    start: usize,
    end: usize,
) -> Result<Vec<u64>, ModularError> {
    let mut out = Vec::with_capacity(end - start);
    for col in start..end {
        out.push(lhs.get(row, col)?);
    }
    Ok(out)
}

fn add_matrix_entry(
    matrix: &mut ModMatrix,
    row: usize,
    col: usize,
    value: u64,
) -> Result<(), ModularError> {
    let q = matrix.modulus();
    let next = add_mod(matrix.get(row, col)?, value % q, q);
    matrix.set(row, col, next as i128)
}

fn random_matrix<R: Rng + ?Sized>(
    rows: usize,
    cols: usize,
    modulus: u64,
    rng: &mut R,
) -> Result<ModMatrix, ModularError> {
    let mut out = ModMatrix::zeros(rows, cols, modulus)?;
    for row in 0..rows {
        for col in 0..cols {
            out.set(row, col, rng.gen_range(0..modulus) as i128)?;
        }
    }
    Ok(out)
}

fn matrix_transpose(matrix: &ModMatrix) -> Result<ModMatrix, ModularError> {
    let mut out = ModMatrix::zeros(matrix.cols(), matrix.rows(), matrix.modulus())?;
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            out.set(col, row, matrix.get(row, col)? as i128)?;
        }
    }
    Ok(out)
}

fn syndrome_key_invocation_field_elements(params: &LpnBksHssParams) -> usize {
    params.code_len * (params.input_len + params.mask_len)
        + (params.code_len - params.output_len) * params.kappa
        + params.kappa
}
