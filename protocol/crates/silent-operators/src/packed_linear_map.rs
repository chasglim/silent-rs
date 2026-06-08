use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use silent_math::arith::{add_mod, reduce_i128_mod_u64, sub_mod};
use silent_ring::{Poly, RingContext};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::OperatorError;
use crate::linear_bridge::LinearMapQShares;
use crate::linear_map::LinearMap;

#[derive(Clone, Debug)]
pub struct PackedLinearMapConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub ring: RingContext,
    pub plaintext_modulus: u64,
    pub noise_bound: i64,
}

impl PackedLinearMapConfig {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.input_dim == 0 || self.output_dim == 0 {
            return Err(OperatorError::InvalidParams(
                "packed linear dimensions must be positive",
            ));
        }
        if self.ring.degree() == 0 || self.ring.degree() < self.input_dim {
            return Err(OperatorError::InvalidParams(
                "packed linear ring degree must be at least input_dim",
            ));
        }
        if self.ring.rns().is_empty() {
            return Err(OperatorError::InvalidParams(
                "packed linear ring must have at least one modulus",
            ));
        }
        if self.plaintext_modulus < 2 {
            return Err(OperatorError::InvalidParams(
                "packed linear plaintext modulus must be >= 2",
            ));
        }
        if self.q_modulus() <= self.plaintext_modulus {
            return Err(OperatorError::InvalidParams(
                "packed linear q modulus must be larger than plaintext modulus",
            ));
        }
        if self.noise_bound < 0 {
            return Err(OperatorError::InvalidParams(
                "packed linear noise bound must be non-negative",
            ));
        }
        Ok(())
    }

    pub fn q_modulus(&self) -> u64 {
        self.ring.rns().moduli()[0].value()
    }

    pub fn max_block_outputs(&self) -> usize {
        usize::max(1, self.ring.degree() / self.input_dim)
    }
}

#[derive(Clone, Debug)]
pub struct PackedLinearMapCrs {
    pub cfg: PackedLinearMapConfig,
    u_ntt: Poly,
    v_ntt: Poly,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PackedLinearMapCrsKey {
    input_dim: usize,
    output_dim: usize,
    degree: usize,
    q_moduli: Vec<u64>,
    plaintext_modulus: u64,
    noise_bound: i64,
}

#[derive(Clone, Debug, Default)]
pub struct PackedLinearMapCrsCache {
    inner: Arc<Mutex<HashMap<PackedLinearMapCrsKey, Arc<PackedLinearMapCrs>>>>,
}

impl PackedLinearMapCrsCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> Result<usize, OperatorError> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| {
                OperatorError::Backend("packed linear CRS cache lock poisoned".to_string())
            })?
            .len())
    }

    pub fn is_empty(&self) -> Result<bool, OperatorError> {
        Ok(self.len()? == 0)
    }
}

#[derive(Clone, Debug)]
pub struct PackedLinearMapServerSetup {
    blocks: Vec<PackedLinearMapBlock>,
    output_dim: usize,
}

#[derive(Clone, Debug)]
struct PackedLinearMapBlock {
    start_col: usize,
    end_col: usize,
    e0_ntt: Poly,
    e1_ntt: Poly,
    s_ntt: Poly,
}

#[derive(Clone, Debug)]
pub struct PackedLinearMapClientQuery {
    rows: usize,
    a_ntt: Vec<Poly>,
    r_ntt: Vec<Poly>,
    d_ntt: Vec<Poly>,
}

impl PackedLinearMapClientQuery {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn online_payload_len_u64(&self) -> usize {
        self.d_ntt.iter().map(|poly| poly.data().len()).sum()
    }

    pub fn online_payload_u64s(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.online_payload_len_u64());
        for poly in &self.d_ntt {
            out.extend_from_slice(poly.data());
        }
        out
    }
}

pub struct PackedLinearMap;

impl PackedLinearMap {
    pub fn setup<R: Rng + ?Sized>(
        cfg: PackedLinearMapConfig,
        rng: &mut R,
    ) -> Result<PackedLinearMapCrs, OperatorError> {
        cfg.validate()?;
        let u_ntt = sample_uniform_ntt(&cfg.ring, rng);
        let v_ntt = sample_uniform_ntt(&cfg.ring, rng);
        Ok(PackedLinearMapCrs { cfg, u_ntt, v_ntt })
    }

    pub fn setup_cached<R: Rng + ?Sized>(
        cfg: PackedLinearMapConfig,
        cache: &PackedLinearMapCrsCache,
        rng: &mut R,
    ) -> Result<Arc<PackedLinearMapCrs>, OperatorError> {
        cfg.validate()?;
        let key = packed_linear_crs_key(&cfg);
        let mut entries = cache.inner.lock().map_err(|_| {
            OperatorError::Backend("packed linear CRS cache lock poisoned".to_string())
        })?;
        if let Some(crs) = entries.get(&key) {
            return Ok(Arc::clone(crs));
        }
        let crs = Arc::new(Self::setup(cfg, rng)?);
        entries.insert(key, Arc::clone(&crs));
        Ok(crs)
    }

    pub fn server_setup<R: Rng + ?Sized>(
        matrix_b_row_major: &[u64],
        crs: &PackedLinearMapCrs,
        rng: &mut R,
    ) -> Result<PackedLinearMapServerSetup, OperatorError> {
        let cfg = &crs.cfg;
        validate_b_matrix(matrix_b_row_major, cfg.input_dim, cfg.output_dim)?;
        let block_cols = cfg.max_block_outputs();
        let mut blocks = Vec::with_capacity(cfg.output_dim.div_ceil(block_cols));
        let mut start_col = 0usize;
        while start_col < cfg.output_dim {
            let end_col = usize::min(start_col + block_cols, cfg.output_dim);
            blocks.push(server_setup_block(
                matrix_b_row_major,
                crs,
                start_col,
                end_col,
                rng,
            )?);
            start_col = end_col;
        }
        Ok(PackedLinearMapServerSetup {
            blocks,
            output_dim: cfg.output_dim,
        })
    }

    pub fn client_query<R: RngCore + ?Sized>(
        matrix_a_row_major: &[u64],
        crs: &PackedLinearMapCrs,
        rng: &mut R,
    ) -> Result<PackedLinearMapClientQuery, OperatorError> {
        let cfg = &crs.cfg;
        if matrix_a_row_major.is_empty() || matrix_a_row_major.len() % cfg.input_dim != 0 {
            return Err(OperatorError::InvalidParams(
                "packed linear input must be rows*input_dim",
            ));
        }
        if matrix_a_row_major
            .iter()
            .any(|&value| value >= cfg.plaintext_modulus)
        {
            return Err(OperatorError::InvalidParams(
                "packed linear input values must be in Z_p",
            ));
        }
        let rows = matrix_a_row_major.len() / cfg.input_dim;
        if rows > cfg.ring.degree() {
            return Err(OperatorError::InvalidParams(
                "packed linear row count exceeds ring degree",
            ));
        }

        let seeds = (0..rows).map(|_| rng.next_u64()).collect::<Vec<_>>();
        let triples = matrix_a_row_major
            .par_chunks(cfg.input_dim)
            .zip(seeds.into_par_iter())
            .map(|(row, seed)| {
                let mut row_rng = StdRng::seed_from_u64(seed);
                let a_ntt = encode_vector_ntt(row, cfg)?;
                let r_ntt = sample_small_ntt(&cfg.ring, cfg.noise_bound, &mut row_rng)?;
                let mut d_ntt = crs.u_ntt.clone();
                d_ntt.mul_assign(&r_ntt, &cfg.ring);
                let mut va = crs.v_ntt.clone();
                va.mul_assign(&a_ntt, &cfg.ring);
                d_ntt.add_assign(&va, &cfg.ring);
                Ok::<_, OperatorError>((a_ntt, r_ntt, d_ntt))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut a_ntt = Vec::with_capacity(rows);
        let mut r_ntt = Vec::with_capacity(rows);
        let mut d_ntt = Vec::with_capacity(rows);
        for (a, r, d) in triples {
            a_ntt.push(a);
            r_ntt.push(r);
            d_ntt.push(d);
        }
        Ok(PackedLinearMapClientQuery {
            rows,
            a_ntt,
            r_ntt,
            d_ntt,
        })
    }

    pub fn extract_q_shares(
        setup: &PackedLinearMapServerSetup,
        query: &PackedLinearMapClientQuery,
        crs: &PackedLinearMapCrs,
    ) -> Result<(Vec<u64>, Vec<u64>), OperatorError> {
        if setup.output_dim != crs.cfg.output_dim {
            return Err(OperatorError::InvalidParams(
                "packed linear setup output_dim mismatch",
            ));
        }
        if query.a_ntt.len() != query.rows
            || query.r_ntt.len() != query.rows
            || query.d_ntt.len() != query.rows
        {
            return Err(OperatorError::InvalidParams(
                "packed linear malformed client query",
            ));
        }

        let len = query.rows * crs.cfg.output_dim;
        let q = crs.cfg.q_modulus();
        let block_shares = setup
            .blocks
            .par_iter()
            .map(|block| extract_block_shares(block, query, crs))
            .collect::<Result<Vec<_>, _>>()?;
        let mut party0_q = vec![0u64; len];
        let mut party1_q = vec![0u64; len];
        for (block_party0, block_party1, start_col, end_col) in block_shares {
            for local_col in 0..(end_col - start_col) {
                let col = start_col + local_col;
                for row in 0..query.rows {
                    let src = local_col * query.rows + row;
                    let dst = col * query.rows + row;
                    party0_q[dst] = block_party0[src];
                    party1_q[dst] = block_party1[src];
                }
            }
        }
        debug_assert!(party0_q.iter().chain(party1_q.iter()).all(|&v| v < q));
        Ok((party0_q, party1_q))
    }

    pub fn apply_q<R: RngCore + ?Sized>(
        matrix_a_row_major: &[u64],
        setup: &PackedLinearMapServerSetup,
        crs: &PackedLinearMapCrs,
        rng: &mut R,
    ) -> Result<LinearMapQShares, OperatorError> {
        let query = Self::client_query(matrix_a_row_major, crs, rng)?;
        let (party0_q, party1_q) = Self::extract_q_shares(setup, &query, crs)?;
        LinearMapQShares::new(
            party0_q,
            party1_q,
            crs.cfg.q_modulus(),
            crs.cfg.plaintext_modulus,
            crs.cfg.output_dim,
            query.rows,
        )
    }

    pub fn reconstruct_p(q_shares: &LinearMapQShares) -> Result<Vec<u64>, OperatorError> {
        let q = LinearMap::reconstruct_q(
            q_shares.party0_q(),
            q_shares.party1_q(),
            q_shares.q_modulus(),
        )?;
        Ok(LinearMap::decode_q_to_p(
            &q,
            q_shares.q_modulus(),
            q_shares.p_modulus(),
        ))
    }
}

fn server_setup_block<R: Rng + ?Sized>(
    matrix_b_row_major: &[u64],
    crs: &PackedLinearMapCrs,
    start_col: usize,
    end_col: usize,
    rng: &mut R,
) -> Result<PackedLinearMapBlock, OperatorError> {
    let cfg = &crs.cfg;
    let b_ntt = encode_matrix_block_ntt(matrix_b_row_major, cfg, start_col, end_col)?;
    let s_ntt = sample_small_ntt(&cfg.ring, cfg.noise_bound, rng)?;
    let eta0_ntt = sample_small_ntt(&cfg.ring, cfg.noise_bound, rng)?;
    let eta1_ntt = sample_small_ntt(&cfg.ring, cfg.noise_bound, rng)?;

    let mut e0_ntt = crs.v_ntt.clone();
    e0_ntt.mul_assign(&s_ntt, &cfg.ring);
    e0_ntt.add_assign(&b_ntt, &cfg.ring);
    e0_ntt.add_assign(&eta0_ntt, &cfg.ring);

    let mut e1_ntt = crs.u_ntt.clone();
    e1_ntt.mul_assign(&s_ntt, &cfg.ring);
    e1_ntt.add_assign(&eta1_ntt, &cfg.ring);

    Ok(PackedLinearMapBlock {
        start_col,
        end_col,
        e0_ntt,
        e1_ntt,
        s_ntt,
    })
}

fn extract_block_shares(
    block: &PackedLinearMapBlock,
    query: &PackedLinearMapClientQuery,
    crs: &PackedLinearMapCrs,
) -> Result<(Vec<u64>, Vec<u64>, usize, usize), OperatorError> {
    let cfg = &crs.cfg;
    let block_cols = block.end_col - block.start_col;
    let rows = query.rows;
    let q = cfg.q_modulus();
    let mut party0_q = vec![0u64; block_cols * rows];
    let mut party1_q = vec![0u64; block_cols * rows];

    for row in 0..rows {
        let mut server_poly = block.s_ntt.clone();
        server_poly.mul_assign(&query.d_ntt[row], &cfg.ring);
        server_poly.ntt_inverse(&cfg.ring);

        let mut client_poly = block.e0_ntt.clone();
        client_poly.mul_assign(&query.a_ntt[row], &cfg.ring);
        let mut term = block.e1_ntt.clone();
        term.mul_assign(&query.r_ntt[row], &cfg.ring);
        client_poly.add_assign(&term, &cfg.ring);
        client_poly.ntt_inverse(&cfg.ring);

        let server_limb = server_poly.limb(0);
        let client_limb = client_poly.limb(0);
        for local_col in 0..block_cols {
            let coeff_idx = local_col * cfg.input_dim + (cfg.input_dim - 1);
            let dst = local_col * rows + row;
            party0_q[dst] = client_limb[coeff_idx] % q;
            party1_q[dst] = neg_mod(server_limb[coeff_idx] % q, q);
        }
    }

    Ok((party0_q, party1_q, block.start_col, block.end_col))
}

fn encode_vector_ntt(values: &[u64], cfg: &PackedLinearMapConfig) -> Result<Poly, OperatorError> {
    if values.len() != cfg.input_dim {
        return Err(OperatorError::InvalidParams(
            "packed linear vector length mismatch",
        ));
    }
    let mut poly = Poly::new(cfg.ring.degree(), cfg.ring.rns().len());
    for (limb_idx, modulus) in cfg.ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        let limb = poly.limb_mut(limb_idx);
        for (idx, &value) in values.iter().enumerate() {
            limb[idx] = value % q;
        }
    }
    poly.ntt_forward(&cfg.ring);
    Ok(poly)
}

fn encode_matrix_block_ntt(
    matrix_b_row_major: &[u64],
    cfg: &PackedLinearMapConfig,
    start_col: usize,
    end_col: usize,
) -> Result<Poly, OperatorError> {
    if start_col >= end_col || end_col > cfg.output_dim {
        return Err(OperatorError::InvalidParams(
            "packed linear invalid matrix block",
        ));
    }
    if (end_col - start_col) * cfg.input_dim > cfg.ring.degree() {
        return Err(OperatorError::InvalidParams(
            "packed linear matrix block exceeds ring degree",
        ));
    }
    let mut poly = Poly::new(cfg.ring.degree(), cfg.ring.rns().len());
    for (limb_idx, modulus) in cfg.ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        let delta = q / cfg.plaintext_modulus;
        let limb = poly.limb_mut(limb_idx);
        for local_col in 0..(end_col - start_col) {
            let col = start_col + local_col;
            for inner in 0..cfg.input_dim {
                let raw = matrix_b_row_major[inner * cfg.output_dim + col] % cfg.plaintext_modulus;
                let coeff_idx = local_col * cfg.input_dim + (cfg.input_dim - 1 - inner);
                limb[coeff_idx] = ((raw as u128 * delta as u128) % q as u128) as u64;
            }
        }
    }
    poly.ntt_forward(&cfg.ring);
    Ok(poly)
}

fn sample_uniform_ntt<R: Rng + ?Sized>(ring: &RingContext, rng: &mut R) -> Poly {
    let mut poly = Poly::new(ring.degree(), ring.rns().len());
    for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        for coeff in poly.limb_mut(limb_idx) {
            *coeff = rng.gen_range(0..q);
        }
    }
    poly
}

fn sample_small_ntt<R: Rng + ?Sized>(
    ring: &RingContext,
    bound: i64,
    rng: &mut R,
) -> Result<Poly, OperatorError> {
    if bound < 0 {
        return Err(OperatorError::InvalidParams(
            "packed linear noise bound must be non-negative",
        ));
    }
    let mut poly = Poly::new(ring.degree(), ring.rns().len());
    for (limb_idx, modulus) in ring.rns().moduli().iter().enumerate() {
        let q = modulus.value();
        for coeff in poly.limb_mut(limb_idx) {
            let sample = if bound == 0 {
                0
            } else {
                rng.gen_range(-bound..=bound)
            };
            *coeff = reduce_i128_mod_u64(sample as i128, q);
        }
    }
    poly.ntt_forward(ring);
    Ok(poly)
}

fn validate_b_matrix(
    matrix_b_row_major: &[u64],
    input_dim: usize,
    output_dim: usize,
) -> Result<(), OperatorError> {
    if matrix_b_row_major.len() != input_dim * output_dim {
        return Err(OperatorError::InvalidParams(
            "packed linear matrix length must be input_dim*output_dim",
        ));
    }
    Ok(())
}

fn packed_linear_crs_key(cfg: &PackedLinearMapConfig) -> PackedLinearMapCrsKey {
    PackedLinearMapCrsKey {
        input_dim: cfg.input_dim,
        output_dim: cfg.output_dim,
        degree: cfg.ring.degree(),
        q_moduli: cfg
            .ring
            .rns()
            .moduli()
            .iter()
            .map(|modulus| modulus.value())
            .collect(),
        plaintext_modulus: cfg.plaintext_modulus,
        noise_bound: cfg.noise_bound,
    }
}

#[inline]
fn neg_mod(value: u64, modulus: u64) -> u64 {
    sub_mod(0, value, modulus)
}

#[allow(dead_code)]
fn reconstruct_for_test(lhs: &[u64], rhs: &[u64], q: u64) -> Vec<u64> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| add_mod(a, b, q))
        .collect()
}
