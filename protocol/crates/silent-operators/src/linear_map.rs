use rand::Rng;

use crate::modular::{
    PolyElem, PolyMatrix, RingMatrixMulCrs, RingMatrixMulPublicA, RingMatrixMulPublicB,
    RingMatrixMulStateA, RingMatrixMulStateB, decode_a_ring, decode_b_ring, encode_a_ring,
    encode_b_raw_ring, encode_b_ring, ring_matrix_mul_setup,
};

use crate::error::OperatorError;

#[derive(Clone, Debug)]
pub struct LinearMapCrs {
    pub ring_crs: RingMatrixMulCrs,
    pub inner_dim: usize,
    pub lwe_rows_n: usize,
    pub gadget_cols_t: usize,
    pub poly_degree: usize,
    pub q_modulus: u64,
    pub p_modulus: u64,
}

#[derive(Clone, Debug)]
pub struct LinearMapServerState {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct LinearMapClientState {
    pub state_a: RingMatrixMulStateA,
    pub inner_dim: usize,
    pub simd_lanes: usize,
}

#[derive(Clone, Debug)]
pub struct LinearMapServerSetup {
    pub public_b: RingMatrixMulPublicB,
    pub state_b: RingMatrixMulStateB,
    pub state: LinearMapServerState,
    pub crs: LinearMapCrs,
}

/// Row-blocked setup for factored-key style linear evaluation.
///
/// The same CRS is reused across output row blocks and the client sends a
/// single query; server/client extraction is then performed block-by-block and
/// concatenated.
#[derive(Clone, Debug)]
pub struct LinearMapBlockedServerSetup {
    pub blocks: Vec<LinearMapServerSetup>,
    pub total_rows: usize,
    pub cols: usize,
    pub row_block_size: usize,
}

#[derive(Clone, Debug)]
pub struct LinearMapClientQuery {
    pub public_a: RingMatrixMulPublicA,
    pub state: LinearMapClientState,
}

pub struct LinearMap;

impl LinearMap {
    pub fn setup<R: Rng + ?Sized>(
        inner_dim: usize,
        poly_degree: usize,
        q_modulus: u64,
        p_modulus: u64,
        lwe_rows_n: usize,
        gadget_cols_t: usize,
        rng: &mut R,
    ) -> Result<LinearMapCrs, OperatorError> {
        if inner_dim == 0 {
            return Err(OperatorError::InvalidParams("inner_dim must be > 0"));
        }
        if poly_degree == 0 {
            return Err(OperatorError::InvalidParams("poly_degree must be > 0"));
        }
        if p_modulus < 2 || q_modulus < 2 {
            return Err(OperatorError::InvalidParams("invalid moduli"));
        }
        if q_modulus < p_modulus {
            return Err(OperatorError::InvalidParams(
                "q_modulus must be >= p_modulus",
            ));
        }
        if q_modulus % p_modulus != 0 {
            return Err(OperatorError::InvalidParams(
                "q_modulus must be divisible by p_modulus",
            ));
        }
        if lwe_rows_n == 0 || gadget_cols_t == 0 {
            return Err(OperatorError::InvalidParams(
                "lwe_rows_n and gadget_cols_t must be > 0",
            ));
        }
        if lwe_rows_n < inner_dim {
            return Err(OperatorError::InvalidParams(
                "lwe_rows_n must be >= inner_dim",
            ));
        }
        let ring_crs = ring_matrix_mul_setup(
            lwe_rows_n,
            inner_dim,
            gadget_cols_t,
            poly_degree,
            q_modulus,
            rng,
        )?;

        Ok(LinearMapCrs {
            ring_crs,
            inner_dim,
            lwe_rows_n,
            gadget_cols_t,
            poly_degree,
            q_modulus,
            p_modulus,
        })
    }

    pub fn server_setup<R: Rng + ?Sized>(
        matrix_w: &[Vec<u64>],
        crs: &LinearMapCrs,
        noise_bound: i64,
        use_scaled_encoding: bool,
        rng: &mut R,
    ) -> Result<LinearMapServerSetup, OperatorError> {
        let rows = matrix_w.len();
        if rows == 0 {
            return Err(OperatorError::InvalidParams("matrix_w must be non-empty"));
        }
        let cols = matrix_w[0].len();
        if cols == 0 {
            return Err(OperatorError::InvalidParams(
                "matrix_w must have non-empty rows",
            ));
        }
        if cols != crs.inner_dim {
            return Err(OperatorError::InvalidParams(
                "matrix_w column count must equal CRS inner_dim",
            ));
        }
        for row in matrix_w {
            if row.len() != cols {
                return Err(OperatorError::InvalidParams("matrix_w must be rectangular"));
            }
        }

        // matrix-multiplication/linear-map computes A(l x m) * B(m x k). For y = W(m_out x n_in) * x(n_in),
        // we set A := x (1 x n_in), B := W^T (n_in x m_out), so output is 1 x m_out.
        let mut b_matrix = PolyMatrix::zeros(cols, rows, crs.poly_degree, crs.q_modulus)?;
        for (i, row) in matrix_w.iter().enumerate() {
            for (j, &weight) in row.iter().enumerate() {
                b_matrix.set(
                    j,
                    i,
                    scalar_to_poly(weight % crs.p_modulus, crs.poly_degree, crs.q_modulus)?,
                )?;
            }
        }

        let (public_b, state_b) = if use_scaled_encoding {
            encode_b_ring(&crs.ring_crs, &b_matrix, crs.p_modulus, noise_bound, rng)?
        } else {
            encode_b_raw_ring(&crs.ring_crs, &b_matrix, noise_bound, rng)?
        };

        Ok(LinearMapServerSetup {
            public_b,
            state_b,
            state: LinearMapServerState { rows, cols },
            crs: crs.clone(),
        })
    }

    pub fn server_setup_row_blocked<R: Rng + ?Sized>(
        matrix_w: &[Vec<u64>],
        crs: &LinearMapCrs,
        noise_bound: i64,
        use_scaled_encoding: bool,
        row_block_size: usize,
        rng: &mut R,
    ) -> Result<LinearMapBlockedServerSetup, OperatorError> {
        if matrix_w.is_empty() {
            return Err(OperatorError::InvalidParams("matrix_w must be non-empty"));
        }
        if row_block_size == 0 {
            return Err(OperatorError::InvalidParams("row_block_size must be > 0"));
        }
        let rows = matrix_w.len();
        let cols = matrix_w[0].len();
        for row in matrix_w {
            if row.len() != cols {
                return Err(OperatorError::InvalidParams("matrix_w must be rectangular"));
            }
        }
        if cols != crs.inner_dim {
            return Err(OperatorError::InvalidParams(
                "matrix_w column count must equal CRS inner_dim",
            ));
        }

        let mut blocks = Vec::new();
        let mut start = 0usize;
        while start < rows {
            let end = usize::min(start + row_block_size, rows);
            let block_setup = Self::server_setup(
                &matrix_w[start..end],
                crs,
                noise_bound,
                use_scaled_encoding,
                rng,
            )?;
            blocks.push(block_setup);
            start = end;
        }

        Ok(LinearMapBlockedServerSetup {
            blocks,
            total_rows: rows,
            cols,
            row_block_size,
        })
    }

    pub fn client_query<R: Rng + ?Sized>(
        vector_x: &[u64],
        crs: &LinearMapCrs,
        noise_bound: i64,
        rng: &mut R,
    ) -> Result<LinearMapClientQuery, OperatorError> {
        if vector_x.is_empty() || vector_x.len() % crs.inner_dim != 0 {
            return Err(OperatorError::InvalidParams(
                "vector_x length must be a positive multiple of CRS inner_dim",
            ));
        }
        let simd_lanes = vector_x.len() / crs.inner_dim;
        if simd_lanes > crs.poly_degree {
            return Err(OperatorError::InvalidParams(
                "SIMD lane count exceeds polynomial degree",
            ));
        }

        let mut a_matrix = PolyMatrix::zeros(1, crs.inner_dim, crs.poly_degree, crs.q_modulus)?;
        for j in 0..crs.inner_dim {
            let mut coeffs = vec![0u64; crs.poly_degree];
            for lane in 0..simd_lanes {
                let idx = lane * crs.inner_dim + j;
                coeffs[lane] = vector_x[idx] % crs.p_modulus;
            }
            a_matrix.set(0, j, PolyElem::from_coeffs(coeffs, crs.q_modulus)?)?;
        }

        let (public_a, state_a) = encode_a_ring(&crs.ring_crs, &a_matrix, noise_bound, rng)?;

        Ok(LinearMapClientQuery {
            public_a,
            state: LinearMapClientState {
                state_a,
                inner_dim: crs.inner_dim,
                simd_lanes,
            },
        })
    }

    pub fn server_extract(
        query: &LinearMapClientQuery,
        server_setup: &LinearMapServerSetup,
    ) -> Result<Vec<u64>, OperatorError> {
        if query.state.inner_dim != server_setup.state.cols {
            return Err(OperatorError::InvalidParams(
                "query inner_dim does not match server setup column dimension",
            ));
        }
        let simd_lanes = query.state.simd_lanes;
        let z_b = decode_b_ring(
            &server_setup.crs.ring_crs,
            &query.public_a,
            &server_setup.state_b,
        )?;

        let mut out = Vec::with_capacity(server_setup.state.rows * simd_lanes);
        for i in 0..server_setup.state.rows {
            let poly = z_b.get(0, i)?;
            for lane in 0..simd_lanes {
                let v = poly.coeff(lane)?;
                out.push(neg_mod(v, server_setup.crs.q_modulus));
            }
        }
        Ok(out)
    }

    pub fn server_extract_row_blocked(
        query: &LinearMapClientQuery,
        server_setup: &LinearMapBlockedServerSetup,
    ) -> Result<Vec<u64>, OperatorError> {
        let mut out = Vec::with_capacity(server_setup.total_rows * query.state.simd_lanes);
        for block in &server_setup.blocks {
            let part = Self::server_extract(query, block)?;
            out.extend_from_slice(&part);
        }
        Ok(out)
    }

    pub fn client_extract(
        server_setup: &LinearMapServerSetup,
        client_state: &LinearMapClientState,
        result_size: usize,
    ) -> Result<Vec<u64>, OperatorError> {
        if client_state.inner_dim != server_setup.state.cols {
            return Err(OperatorError::InvalidParams(
                "client inner_dim does not match server setup column dimension",
            ));
        }
        if result_size == 0 || result_size > server_setup.state.rows {
            return Err(OperatorError::InvalidParams(
                "result_size must be in [1, rows]",
            ));
        }

        let z_a = decode_a_ring(
            &server_setup.crs.ring_crs,
            &server_setup.public_b,
            &client_state.state_a,
        )?;

        let mut out = Vec::with_capacity(result_size * client_state.simd_lanes);
        for i in 0..result_size {
            let poly = z_a.get(0, i)?;
            for lane in 0..client_state.simd_lanes {
                out.push(poly.coeff(lane)?);
            }
        }
        Ok(out)
    }

    pub fn client_extract_row_blocked(
        server_setup: &LinearMapBlockedServerSetup,
        client_state: &LinearMapClientState,
        result_size: usize,
    ) -> Result<Vec<u64>, OperatorError> {
        if result_size == 0 || result_size > server_setup.total_rows {
            return Err(OperatorError::InvalidParams(
                "result_size must be in [1, total_rows]",
            ));
        }

        let mut out = Vec::with_capacity(result_size * client_state.simd_lanes);
        let mut remaining = result_size;
        for block in &server_setup.blocks {
            if remaining == 0 {
                break;
            }
            let take = usize::min(remaining, block.state.rows);
            let part = Self::client_extract(block, client_state, take)?;
            out.extend_from_slice(&part);
            remaining -= take;
        }
        Ok(out)
    }

    pub fn reconstruct_q(
        server_share: &[u64],
        client_share: &[u64],
        q_modulus: u64,
    ) -> Result<Vec<u64>, OperatorError> {
        if q_modulus < 2 {
            return Err(OperatorError::InvalidParams("q_modulus must be >= 2"));
        }
        if server_share.len() != client_share.len() {
            return Err(OperatorError::InvalidParams(
                "server/client share lengths must match",
            ));
        }
        if server_share
            .iter()
            .zip(client_share.iter())
            .any(|(&s, &c)| s >= q_modulus || c >= q_modulus)
        {
            return Err(OperatorError::InvalidParams(
                "shares must be canonical elements in Z_q",
            ));
        }
        let mut out = Vec::with_capacity(server_share.len());
        for (&s, &c) in server_share.iter().zip(client_share.iter()) {
            out.push(add_mod_u64(s, c, q_modulus));
        }
        Ok(out)
    }

    pub fn decode_q_to_p(values_q: &[u64], q_modulus: u64, p_modulus: u64) -> Vec<u64> {
        values_q
            .iter()
            .map(|&v| {
                let num = (v as u128) * (p_modulus as u128) + (q_modulus as u128 / 2);
                ((num / q_modulus as u128) as u64) % p_modulus
            })
            .collect()
    }
}

fn scalar_to_poly(value: u64, degree: usize, modulus: u64) -> Result<PolyElem, OperatorError> {
    let mut coeffs = vec![0u64; degree];
    coeffs[0] = value % modulus;
    Ok(PolyElem::from_coeffs(coeffs, modulus)?)
}

fn neg_mod(value: u64, modulus: u64) -> u64 {
    if value == 0 {
        0
    } else {
        modulus - (value % modulus)
    }
}

#[inline]
fn add_mod_u64(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 + rhs as u128) % modulus as u128) as u64
}
