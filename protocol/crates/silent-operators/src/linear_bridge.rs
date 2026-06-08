use rand::Rng;

use crate::error::OperatorError;
use crate::linear_map::{LinearMap, LinearMapCrs, LinearMapServerSetup};
use crate::shares::AdditiveShares;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrivateLinearMapConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub poly_degree: usize,
    pub q_modulus: u64,
    pub p_modulus: u64,
    pub lwe_rows_n: usize,
    pub gadget_cols_t: usize,
    pub setup_noise_bound: i64,
    pub query_noise_bound: i64,
}

impl PrivateLinearMapConfig {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.input_dim == 0 || self.output_dim == 0 {
            return Err(OperatorError::InvalidParams(
                "private linear map dimensions must be positive",
            ));
        }
        if self.q_modulus < self.p_modulus || self.p_modulus < 2 {
            return Err(OperatorError::InvalidParams(
                "private linear map requires q_modulus >= p_modulus >= 2",
            ));
        }
        if self.q_modulus % self.p_modulus != 0 {
            return Err(OperatorError::InvalidParams(
                "private linear map q_modulus must be divisible by p_modulus",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearMapQShares {
    party0_q: Vec<u64>,
    party1_q: Vec<u64>,
    q_modulus: u64,
    p_modulus: u64,
    output_dim: usize,
    simd_lanes: usize,
}

impl LinearMapQShares {
    pub fn new(
        party0_q: Vec<u64>,
        party1_q: Vec<u64>,
        q_modulus: u64,
        p_modulus: u64,
        output_dim: usize,
        simd_lanes: usize,
    ) -> Result<Self, OperatorError> {
        if party0_q.len() != party1_q.len() {
            return Err(OperatorError::InvalidParams(
                "linear map q-share lengths must match",
            ));
        }
        if output_dim == 0 || simd_lanes == 0 || party0_q.len() != output_dim * simd_lanes {
            return Err(OperatorError::InvalidParams(
                "linear map q-share shape does not match output dimensions",
            ));
        }
        if q_modulus < p_modulus || p_modulus < 2 {
            return Err(OperatorError::InvalidParams(
                "invalid moduli for linear map q-shares",
            ));
        }
        if party0_q
            .iter()
            .chain(party1_q.iter())
            .any(|&v| v >= q_modulus)
        {
            return Err(OperatorError::InvalidParams(
                "linear map q-shares must be canonical elements in Z_q",
            ));
        }
        Ok(Self {
            party0_q,
            party1_q,
            q_modulus,
            p_modulus,
            output_dim,
            simd_lanes,
        })
    }

    pub fn party0_q(&self) -> &[u64] {
        &self.party0_q
    }

    pub fn party1_q(&self) -> &[u64] {
        &self.party1_q
    }

    pub fn q_modulus(&self) -> u64 {
        self.q_modulus
    }

    pub fn p_modulus(&self) -> u64 {
        self.p_modulus
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn simd_lanes(&self) -> usize {
        self.simd_lanes
    }

    pub fn len(&self) -> usize {
        self.party0_q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.party0_q.is_empty()
    }

    pub fn reconstruct_p(&self) -> Result<Vec<u64>, OperatorError> {
        let q = LinearMap::reconstruct_q(&self.party0_q, &self.party1_q, self.q_modulus)?;
        Ok(LinearMap::decode_q_to_p(&q, self.q_modulus, self.p_modulus))
    }
}

#[derive(Clone, Debug)]
pub struct PrivateLinearMap {
    cfg: PrivateLinearMapConfig,
    crs: LinearMapCrs,
    setup: LinearMapServerSetup,
}

impl PrivateLinearMap {
    pub fn setup<R: Rng + ?Sized>(
        matrix_w: &[Vec<u64>],
        cfg: PrivateLinearMapConfig,
        rng: &mut R,
    ) -> Result<Self, OperatorError> {
        cfg.validate()?;
        validate_matrix_shape(matrix_w, cfg.output_dim, cfg.input_dim)?;
        let crs = LinearMap::setup(
            cfg.input_dim,
            cfg.poly_degree,
            cfg.q_modulus,
            cfg.p_modulus,
            cfg.lwe_rows_n,
            cfg.gadget_cols_t,
            rng,
        )?;
        let setup = LinearMap::server_setup(matrix_w, &crs, cfg.setup_noise_bound, true, rng)?;
        Ok(Self { cfg, crs, setup })
    }

    pub fn config(&self) -> &PrivateLinearMapConfig {
        &self.cfg
    }

    pub fn crs(&self) -> &LinearMapCrs {
        &self.crs
    }

    pub fn apply_q<R: Rng + ?Sized>(
        &self,
        input: &[u64],
        rng: &mut R,
    ) -> Result<LinearMapQShares, OperatorError> {
        if input.is_empty() || input.len() % self.cfg.input_dim != 0 {
            return Err(OperatorError::InvalidParams(
                "private linear input length must be a positive multiple of input_dim",
            ));
        }
        let simd_lanes = input.len() / self.cfg.input_dim;
        let query = LinearMap::client_query(input, &self.crs, self.cfg.query_noise_bound, rng)?;
        let party1_q = LinearMap::server_extract(&query, &self.setup)?;
        let party0_q = LinearMap::client_extract(&self.setup, &query.state, self.cfg.output_dim)?;
        LinearMapQShares::new(
            party0_q,
            party1_q,
            self.cfg.q_modulus,
            self.cfg.p_modulus,
            self.cfg.output_dim,
            simd_lanes,
        )
    }

    pub fn apply_with_converter<R, F>(
        &self,
        input: &[u64],
        rng: &mut R,
        mut converter: F,
    ) -> Result<AdditiveShares, OperatorError>
    where
        R: Rng + ?Sized,
        F: FnMut(&LinearMapQShares, &mut R) -> Result<AdditiveShares, OperatorError>,
    {
        let q_shares = self.apply_q(input, rng)?;
        converter(&q_shares, rng)
    }
}

fn validate_matrix_shape(
    matrix: &[Vec<u64>],
    output_dim: usize,
    input_dim: usize,
) -> Result<(), OperatorError> {
    if matrix.len() != output_dim {
        return Err(OperatorError::InvalidParams(
            "private linear matrix row count must match output_dim",
        ));
    }
    if matrix.iter().any(|row| row.len() != input_dim) {
        return Err(OperatorError::InvalidParams(
            "private linear matrix column count must match input_dim",
        ));
    }
    Ok(())
}
