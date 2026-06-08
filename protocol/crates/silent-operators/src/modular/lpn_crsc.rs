use rand::RngCore;
use silent_lpn::{LinearCode, ReedSolomonCode, field::inv};
use silent_math::arith::{add_mod, mul_mod_u64 as mul_mod, sub_mod};
use std::time::Instant;

use crate::error::OperatorError;
use crate::lookup::PrivateLookup;
use crate::shares::AdditiveShares;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedCrscConfig {
    pub max_error_candidates: usize,
}

impl Default for DistributedCrscConfig {
    fn default() -> Self {
        Self {
            max_error_candidates: 1 << 16,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedCrscOutput {
    pub client_share: Vec<u64>,
    pub server_share: Vec<u64>,
    pub error_candidate_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DistributedCrscProfile {
    pub syndrome_us: u128,
    pub distributed_ddec_us: u128,
    pub recover_us: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedCrscBatchOutput {
    pub outputs: Vec<DistributedCrscOutput>,
    pub profile: DistributedCrscProfile,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SyndromeKeyCrscProfile {
    pub syndrome_reconstruction_us: u128,
    pub decode_us: u128,
    pub recover_us: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyndromeKeyCrscBatchOutput {
    pub outputs: Vec<DistributedCrscOutput>,
    pub profile: SyndromeKeyCrscProfile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreprocessedCrscKeys {
    pub batch_size: usize,
    pub error_candidate_count: usize,
}

pub fn syndrome_key_crsc_convert_many_profiled(
    code: &ReedSolomonCode,
    client_codeword_shares: &[Vec<u64>],
    server_codeword_shares: &[Vec<u64>],
    digests: &[Vec<u64>],
    syndrome_key: &[Vec<u64>],
) -> Result<SyndromeKeyCrscBatchOutput, OperatorError> {
    if client_codeword_shares.is_empty() {
        return Err(OperatorError::InvalidParams(
            "syndrome-key CRSC batch must be non-empty",
        ));
    }
    if client_codeword_shares.len() != server_codeword_shares.len()
        || client_codeword_shares.len() != digests.len()
    {
        return Err(OperatorError::InvalidParams(
            "syndrome-key CRSC batch lengths must match",
        ));
    }
    if client_codeword_shares
        .iter()
        .chain(server_codeword_shares.iter())
        .any(|share| share.len() != code.n())
    {
        return Err(OperatorError::InvalidParams(
            "syndrome-key CRSC codeword share length mismatch",
        ));
    }
    let syndrome_rows = code.n() - code.r();
    if syndrome_key.len() != syndrome_rows {
        return Err(OperatorError::InvalidParams(
            "syndrome-key CRSC key row count mismatch",
        ));
    }
    let kappa = syndrome_key.first().map_or(0, Vec::len);
    if kappa == 0 || syndrome_key.iter().any(|row| row.len() != kappa) {
        return Err(OperatorError::InvalidParams(
            "syndrome-key CRSC key must be rectangular and non-empty",
        ));
    }
    if digests.iter().any(|digest| digest.len() != kappa) {
        return Err(OperatorError::InvalidParams(
            "syndrome-key CRSC digest length mismatch",
        ));
    }

    let candidate_count = support_candidate_count(code)?;
    let modulus = code.modulus();

    let syndrome_start = Instant::now();
    let mut syndromes = Vec::with_capacity(client_codeword_shares.len());
    for (client_share, digest) in client_codeword_shares.iter().zip(digests.iter()) {
        let mut syndrome = code
            .syndrome(client_share)
            .map_err(|err| OperatorError::Backend(err.to_string()))?;
        for row in 0..syndrome_rows {
            let mut key_dot_digest = 0u64;
            for (key, &d) in syndrome_key[row].iter().zip(digest.iter()) {
                key_dot_digest = add_mod(key_dot_digest, mul_mod(*key, d, modulus), modulus);
            }
            syndrome[row] = sub_mod(syndrome[row], key_dot_digest, modulus);
        }
        syndromes.push(syndrome);
    }
    let syndrome_reconstruction_us = syndrome_start.elapsed().as_micros();

    let decode_start = Instant::now();
    let errors = syndromes
        .iter()
        .map(|syndrome| {
            code.decode_error_from_syndrome(syndrome)
                .map_err(|err| OperatorError::Backend(err.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let decode_us = decode_start.elapsed().as_micros();

    let recover_start = Instant::now();
    let mut outputs = Vec::with_capacity(client_codeword_shares.len());
    for ((client_codeword_share, server_codeword_share), error) in client_codeword_shares
        .iter()
        .zip(server_codeword_shares.iter())
        .zip(errors.iter())
    {
        let mut client_clean = Vec::with_capacity(code.n());
        for idx in 0..code.n() {
            client_clean.push(sub_mod(
                client_codeword_share[idx] % modulus,
                error[idx] % modulus,
                modulus,
            ));
        }
        let client_share = code
            .recover(&client_clean)
            .map_err(|err| OperatorError::Backend(err.to_string()))?;
        let server_share = code
            .recover(server_codeword_share)
            .map_err(|err| OperatorError::Backend(err.to_string()))?;
        outputs.push(DistributedCrscOutput {
            client_share,
            server_share,
            error_candidate_count: candidate_count,
        });
    }
    let recover_us = recover_start.elapsed().as_micros();

    Ok(SyndromeKeyCrscBatchOutput {
        outputs,
        profile: SyndromeKeyCrscProfile {
            syndrome_reconstruction_us,
            decode_us,
            recover_us,
        },
    })
}

pub fn syndrome_key_crsc_convert_many(
    code: &ReedSolomonCode,
    client_codeword_shares: &[Vec<u64>],
    server_codeword_shares: &[Vec<u64>],
    digests: &[Vec<u64>],
    syndrome_key: &[Vec<u64>],
) -> Result<Vec<DistributedCrscOutput>, OperatorError> {
    Ok(syndrome_key_crsc_convert_many_profiled(
        code,
        client_codeword_shares,
        server_codeword_shares,
        digests,
        syndrome_key,
    )?
    .outputs)
}

pub fn preprocess_crsc_keys_many<R: RngCore + ?Sized>(
    code: &ReedSolomonCode,
    lookup: &PrivateLookup,
    cfg: &DistributedCrscConfig,
    batch_size: usize,
    rng: &mut R,
) -> Result<PreprocessedCrscKeys, OperatorError> {
    if batch_size == 0 {
        return Err(OperatorError::InvalidParams(
            "CRSC preprocessing batch must be non-empty",
        ));
    }
    if lookup.config().output_modulus != code.modulus() {
        return Err(OperatorError::InvalidParams(
            "CRSC preprocessing lookup output modulus must match code modulus",
        ));
    }
    if code.modulus() as usize > lookup.config().max_domain {
        return Err(OperatorError::InvalidParams(
            "CRSC preprocessing requires lookup max_domain >= code modulus",
        ));
    }
    let error_candidate_count = support_candidate_count(code)?;
    if error_candidate_count > cfg.max_error_candidates {
        return Err(OperatorError::InvalidParams(
            "CRSC preprocessing support candidate set exceeds max_error_candidates",
        ));
    }
    lookup.prewarm_crs(code.modulus() as usize, code.modulus(), rng)?;
    Ok(PreprocessedCrscKeys {
        batch_size,
        error_candidate_count,
    })
}

pub fn convert_many_local<R: RngCore + ?Sized>(
    code: &ReedSolomonCode,
    client_codeword_shares: &[Vec<u64>],
    server_codeword_shares: &[Vec<u64>],
    lookup: &PrivateLookup,
    keys: &PreprocessedCrscKeys,
    rng: &mut R,
) -> Result<Vec<DistributedCrscOutput>, OperatorError> {
    if client_codeword_shares.len() != keys.batch_size {
        return Err(OperatorError::InvalidParams(
            "local CRSC convert batch size does not match preprocessing",
        ));
    }
    Ok(distributed_lookup_crsc_convert_many_profiled(
        code,
        client_codeword_shares,
        server_codeword_shares,
        lookup,
        &DistributedCrscConfig {
            max_error_candidates: keys.error_candidate_count,
        },
        rng,
    )?
    .outputs)
}

pub fn distributed_lookup_crsc_convert<R: RngCore + ?Sized>(
    code: &ReedSolomonCode,
    client_codeword_share: &[u64],
    server_codeword_share: &[u64],
    lookup: &PrivateLookup,
    cfg: &DistributedCrscConfig,
    rng: &mut R,
) -> Result<DistributedCrscOutput, OperatorError> {
    let mut outputs = distributed_lookup_crsc_convert_many(
        code,
        &[client_codeword_share.to_vec()],
        &[server_codeword_share.to_vec()],
        lookup,
        cfg,
        rng,
    )?;
    outputs
        .pop()
        .ok_or_else(|| OperatorError::Backend("distributed CRSC returned no output".to_string()))
}

pub fn distributed_lookup_crsc_convert_many<R: RngCore + ?Sized>(
    code: &ReedSolomonCode,
    client_codeword_shares: &[Vec<u64>],
    server_codeword_shares: &[Vec<u64>],
    lookup: &PrivateLookup,
    cfg: &DistributedCrscConfig,
    rng: &mut R,
) -> Result<Vec<DistributedCrscOutput>, OperatorError> {
    Ok(distributed_lookup_crsc_convert_many_profiled(
        code,
        client_codeword_shares,
        server_codeword_shares,
        lookup,
        cfg,
        rng,
    )?
    .outputs)
}

pub fn distributed_lookup_crsc_convert_many_profiled<R: RngCore + ?Sized>(
    code: &ReedSolomonCode,
    client_codeword_shares: &[Vec<u64>],
    server_codeword_shares: &[Vec<u64>],
    lookup: &PrivateLookup,
    cfg: &DistributedCrscConfig,
    rng: &mut R,
) -> Result<DistributedCrscBatchOutput, OperatorError> {
    if lookup.config().output_modulus != code.modulus() {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC lookup output modulus must match the code modulus",
        ));
    }
    if client_codeword_shares.is_empty() {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC batch must be non-empty",
        ));
    }
    if client_codeword_shares.len() != server_codeword_shares.len() {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC client/server batch lengths must match",
        ));
    }
    if client_codeword_shares
        .iter()
        .chain(server_codeword_shares.iter())
        .any(|share| share.len() != code.n())
    {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC codeword share length mismatch",
        ));
    }
    if code.modulus() as usize > lookup.config().max_domain {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC requires lookup max_domain >= code modulus",
        ));
    }

    let candidate_count = support_candidate_count(code)?;
    if candidate_count > cfg.max_error_candidates {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC support candidate set exceeds max_error_candidates",
        ));
    }

    let syndrome_start = Instant::now();
    let syndrome_client = client_codeword_shares
        .iter()
        .map(|share| code.syndrome(share))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| OperatorError::Backend(err.to_string()))?;
    let syndrome_server = server_codeword_shares
        .iter()
        .map(|share| code.syndrome(share))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| OperatorError::Backend(err.to_string()))?;
    let syndrome_us = syndrome_start.elapsed().as_micros();

    let ddec_start = Instant::now();
    let error_shares = distributed_decode_error_from_syndrome_shares_batched(
        code,
        &syndrome_client,
        &syndrome_server,
        lookup,
        rng,
    )?;
    let distributed_ddec_us = ddec_start.elapsed().as_micros();

    let recover_start = Instant::now();
    let mut outputs = Vec::with_capacity(client_codeword_shares.len());
    for ((client_codeword_share, server_codeword_share), error_share) in client_codeword_shares
        .iter()
        .zip(server_codeword_shares.iter())
        .zip(error_shares.iter())
    {
        let mut client_clean = Vec::with_capacity(code.n());
        let mut server_clean = Vec::with_capacity(code.n());
        for idx in 0..code.n() {
            client_clean.push(sub_mod(
                client_codeword_share[idx] % code.modulus(),
                error_share.party0()[idx],
                code.modulus(),
            ));
            server_clean.push(sub_mod(
                server_codeword_share[idx] % code.modulus(),
                error_share.party1()[idx],
                code.modulus(),
            ));
        }

        let client_share = code
            .recover(&client_clean)
            .map_err(|err| OperatorError::Backend(err.to_string()))?;
        let server_share = code
            .recover(&server_clean)
            .map_err(|err| OperatorError::Backend(err.to_string()))?;
        outputs.push(DistributedCrscOutput {
            client_share,
            server_share,
            error_candidate_count: candidate_count,
        });
    }
    let recover_us = recover_start.elapsed().as_micros();
    Ok(DistributedCrscBatchOutput {
        outputs,
        profile: DistributedCrscProfile {
            syndrome_us,
            distributed_ddec_us,
            recover_us,
        },
    })
}

fn distributed_decode_error_from_syndrome_shares_batched<R: RngCore + ?Sized>(
    code: &ReedSolomonCode,
    syndrome_client: &[Vec<u64>],
    syndrome_server: &[Vec<u64>],
    lookup: &PrivateLookup,
    rng: &mut R,
) -> Result<Vec<AdditiveShares>, OperatorError> {
    if code.t() > 2 {
        return Err(OperatorError::InvalidParams(
            "support-only distributed CRSC currently supports decode_radius <= 2",
        ));
    }
    if syndrome_client.len() != syndrome_server.len() {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC syndrome batch lengths must match",
        ));
    }
    let batch = syndrome_client.len();
    let syndrome_len = code.n() - code.r();
    if syndrome_client
        .iter()
        .chain(syndrome_server.iter())
        .any(|syndrome| syndrome.len() != syndrome_len)
    {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC syndrome share length mismatch",
        ));
    }

    let modulus = code.modulus();
    let mut error_client = vec![vec![0; code.n()]; batch];
    let mut error_server = vec![vec![0; code.n()]; batch];

    if code.t() >= 1 {
        for pos in 0..code.n() {
            let support = [pos];
            let values =
                solve_support_values_batch(code, &support, syndrome_client, syndrome_server)?;
            let mut indicator = residual_zero_indicator(
                code,
                &support,
                &values,
                syndrome_client,
                syndrome_server,
                lookup,
                rng,
            )?;
            let nonzero = nonzero_indicator(&values[0], lookup, rng)?;
            indicator = lookup.bit_and(&indicator, &nonzero, rng)?;
            add_selected_value_to_errors(
                support[0],
                &indicator,
                &values[0],
                &mut error_client,
                &mut error_server,
                rng,
            )?;
        }
    }

    if code.t() >= 2 {
        for first in 0..code.n() {
            for second in (first + 1)..code.n() {
                let support = [first, second];
                let values =
                    solve_support_values_batch(code, &support, syndrome_client, syndrome_server)?;
                let mut indicator = residual_zero_indicator(
                    code,
                    &support,
                    &values,
                    syndrome_client,
                    syndrome_server,
                    lookup,
                    rng,
                )?;
                for value in &values {
                    let nonzero = nonzero_indicator(value, lookup, rng)?;
                    indicator = lookup.bit_and(&indicator, &nonzero, rng)?;
                }
                for (&position, value) in support.iter().zip(values.iter()) {
                    add_selected_value_to_errors(
                        position,
                        &indicator,
                        value,
                        &mut error_client,
                        &mut error_server,
                        rng,
                    )?;
                }
            }
        }
    }

    error_client
        .into_iter()
        .zip(error_server)
        .map(|(client, server)| AdditiveShares::new(modulus, client, server))
        .collect()
}

fn zero_indicator_for_shared_values<R: RngCore + ?Sized>(
    value_client: &[u64],
    value_server: &[u64],
    lookup: &PrivateLookup,
    rng: &mut R,
) -> Result<AdditiveShares, OperatorError> {
    let modulus = lookup.config().output_modulus;
    let mut outputs = lookup.lookup_sparse_many(
        value_client,
        value_server,
        1,
        modulus as usize,
        &[0],
        |_, value| u64::from(value == 0),
        rng,
    )?;
    outputs
        .pop()
        .ok_or_else(|| OperatorError::Backend("CRSC zero lookup returned no output".to_string()))
}

fn nonzero_indicator<R: RngCore + ?Sized>(
    value: &AdditiveShares,
    lookup: &PrivateLookup,
    rng: &mut R,
) -> Result<AdditiveShares, OperatorError> {
    let zero = zero_indicator_for_shared_values(value.party0(), value.party1(), lookup, rng)?;
    AdditiveShares::share_public(&vec![1; value.len()], value.modulus())?.sub(&zero)
}

fn residual_zero_indicator<R: RngCore + ?Sized>(
    code: &ReedSolomonCode,
    support: &[usize],
    values: &[AdditiveShares],
    syndrome_client: &[Vec<u64>],
    syndrome_server: &[Vec<u64>],
    lookup: &PrivateLookup,
    rng: &mut R,
) -> Result<AdditiveShares, OperatorError> {
    if support.len() != values.len() {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC support/value length mismatch",
        ));
    }
    let modulus = code.modulus();
    let batch = syndrome_client.len();
    let mut indicator = AdditiveShares::share_public(&vec![1; batch], modulus)?;
    for row in 0..code.parity_check().rows() {
        let mut residual_client = Vec::with_capacity(batch);
        let mut residual_server = Vec::with_capacity(batch);
        for slot in 0..batch {
            let mut client = syndrome_client[slot][row] % modulus;
            let mut server = syndrome_server[slot][row] % modulus;
            for (&position, value) in support.iter().zip(values.iter()) {
                let h = code
                    .parity_check()
                    .get(row, position)
                    .map_err(|err| OperatorError::Backend(err.to_string()))?;
                client = sub_mod(client, mul_mod(h, value.party0()[slot], modulus), modulus);
                server = sub_mod(server, mul_mod(h, value.party1()[slot], modulus), modulus);
            }
            residual_client.push(client);
            residual_server.push(server);
        }
        let eq = zero_indicator_for_shared_values(&residual_client, &residual_server, lookup, rng)?;
        indicator = lookup.bit_and(&indicator, &eq, rng)?;
    }
    Ok(indicator)
}

fn solve_support_values_batch(
    code: &ReedSolomonCode,
    support: &[usize],
    syndrome_client: &[Vec<u64>],
    syndrome_server: &[Vec<u64>],
) -> Result<Vec<AdditiveShares>, OperatorError> {
    match support.len() {
        1 => solve_weight_one_values_batch(code, support[0], syndrome_client, syndrome_server),
        2 => solve_weight_two_values_batch(
            code,
            support[0],
            support[1],
            syndrome_client,
            syndrome_server,
        ),
        _ => Err(OperatorError::InvalidParams(
            "support-only distributed CRSC supports support weight 1 or 2",
        )),
    }
}

fn solve_weight_one_values_batch(
    code: &ReedSolomonCode,
    position: usize,
    syndrome_client: &[Vec<u64>],
    syndrome_server: &[Vec<u64>],
) -> Result<Vec<AdditiveShares>, OperatorError> {
    let modulus = code.modulus();
    let (pivot_row, h_inv) = (0..code.parity_check().rows())
        .find_map(|row| {
            let h = code.parity_check().get(row, position).ok()?;
            inv(h, modulus).map(|h_inv| (row, h_inv))
        })
        .ok_or(OperatorError::InvalidParams(
            "distributed CRSC could not find an invertible single-column pivot",
        ))?;
    let client = syndrome_client
        .iter()
        .map(|syndrome| mul_mod(syndrome[pivot_row], h_inv, modulus))
        .collect::<Vec<_>>();
    let server = syndrome_server
        .iter()
        .map(|syndrome| mul_mod(syndrome[pivot_row], h_inv, modulus))
        .collect::<Vec<_>>();
    Ok(vec![AdditiveShares::new(modulus, client, server)?])
}

fn solve_weight_two_values_batch(
    code: &ReedSolomonCode,
    first: usize,
    second: usize,
    syndrome_client: &[Vec<u64>],
    syndrome_server: &[Vec<u64>],
) -> Result<Vec<AdditiveShares>, OperatorError> {
    let modulus = code.modulus();
    let mut pivot = None;
    for row_a in 0..code.parity_check().rows() {
        let h_ai = h_at(code, row_a, first)?;
        let h_aj = h_at(code, row_a, second)?;
        for row_b in (row_a + 1)..code.parity_check().rows() {
            let h_bi = h_at(code, row_b, first)?;
            let h_bj = h_at(code, row_b, second)?;
            let det = sub_mod(
                mul_mod(h_ai, h_bj, modulus),
                mul_mod(h_aj, h_bi, modulus),
                modulus,
            );
            if let Some(det_inv) = inv(det, modulus) {
                pivot = Some((row_a, row_b, h_ai, h_aj, h_bi, h_bj, det_inv));
                break;
            }
        }
        if pivot.is_some() {
            break;
        }
    }
    let Some((row_a, row_b, h_ai, h_aj, h_bi, h_bj, det_inv)) = pivot else {
        return Err(OperatorError::InvalidParams(
            "distributed CRSC could not find an invertible two-column pivot",
        ));
    };

    let solve_party = |syndromes: &[Vec<u64>]| {
        let mut first_values = Vec::with_capacity(syndromes.len());
        let mut second_values = Vec::with_capacity(syndromes.len());
        for syndrome in syndromes {
            let sigma_a = syndrome[row_a];
            let sigma_b = syndrome[row_b];
            let first_num = sub_mod(
                mul_mod(sigma_a, h_bj, modulus),
                mul_mod(sigma_b, h_aj, modulus),
                modulus,
            );
            let second_num = sub_mod(
                mul_mod(h_ai, sigma_b, modulus),
                mul_mod(h_bi, sigma_a, modulus),
                modulus,
            );
            first_values.push(mul_mod(first_num, det_inv, modulus));
            second_values.push(mul_mod(second_num, det_inv, modulus));
        }
        (first_values, second_values)
    };
    let (first_client, second_client) = solve_party(syndrome_client);
    let (first_server, second_server) = solve_party(syndrome_server);
    Ok(vec![
        AdditiveShares::new(modulus, first_client, first_server)?,
        AdditiveShares::new(modulus, second_client, second_server)?,
    ])
}

fn add_selected_value_to_errors<R: RngCore + ?Sized>(
    position: usize,
    indicator: &AdditiveShares,
    value: &AdditiveShares,
    error_client: &mut [Vec<u64>],
    error_server: &mut [Vec<u64>],
    rng: &mut R,
) -> Result<(), OperatorError> {
    let selected = beaver_mul_shared(indicator, value, rng)?;
    let modulus = value.modulus();
    for slot in 0..selected.len() {
        error_client[slot][position] = add_mod(
            error_client[slot][position],
            selected.party0()[slot],
            modulus,
        );
        error_server[slot][position] = add_mod(
            error_server[slot][position],
            selected.party1()[slot],
            modulus,
        );
    }
    Ok(())
}

fn beaver_mul_shared<R: RngCore + ?Sized>(
    lhs: &AdditiveShares,
    rhs: &AdditiveShares,
    rng: &mut R,
) -> Result<AdditiveShares, OperatorError> {
    if lhs.modulus() != rhs.modulus() || lhs.len() != rhs.len() {
        return Err(OperatorError::InvalidParams(
            "shared multiplication requires compatible additive shares",
        ));
    }
    let modulus = lhs.modulus();
    let len = lhs.len();
    let mut a = Vec::with_capacity(len);
    let mut b = Vec::with_capacity(len);
    let mut c = Vec::with_capacity(len);
    for _ in 0..len {
        let av = rng.next_u64() % modulus;
        let bv = rng.next_u64() % modulus;
        a.push(av);
        b.push(bv);
        c.push(mul_mod(av, bv, modulus));
    }
    let a_share = AdditiveShares::share_with_rng(&a, modulus, rng)?;
    let b_share = AdditiveShares::share_with_rng(&b, modulus, rng)?;
    let c_share = AdditiveShares::share_with_rng(&c, modulus, rng)?;

    // Beaver multiplication opens only the masked differences d=x-a and e=y-b.
    let mut d_open = Vec::with_capacity(len);
    let mut e_open = Vec::with_capacity(len);
    for idx in 0..len {
        let d0 = sub_mod(lhs.party0()[idx], a_share.party0()[idx], modulus);
        let d1 = sub_mod(lhs.party1()[idx], a_share.party1()[idx], modulus);
        let e0 = sub_mod(rhs.party0()[idx], b_share.party0()[idx], modulus);
        let e1 = sub_mod(rhs.party1()[idx], b_share.party1()[idx], modulus);
        d_open.push(add_mod(d0, d1, modulus));
        e_open.push(add_mod(e0, e1, modulus));
    }

    let mut out0 = Vec::with_capacity(len);
    let mut out1 = Vec::with_capacity(len);
    for idx in 0..len {
        let d = d_open[idx];
        let e = e_open[idx];
        let p0 = add_mod(
            add_mod(
                add_mod(
                    c_share.party0()[idx],
                    mul_mod(d, b_share.party0()[idx], modulus),
                    modulus,
                ),
                mul_mod(e, a_share.party0()[idx], modulus),
                modulus,
            ),
            mul_mod(d, e, modulus),
            modulus,
        );
        let p1 = add_mod(
            add_mod(
                c_share.party1()[idx],
                mul_mod(d, b_share.party1()[idx], modulus),
                modulus,
            ),
            mul_mod(e, a_share.party1()[idx], modulus),
            modulus,
        );
        out0.push(p0);
        out1.push(p1);
    }
    AdditiveShares::new(modulus, out0, out1)
}

fn support_candidate_count(code: &ReedSolomonCode) -> Result<usize, OperatorError> {
    if code.t() > 2 {
        return Err(OperatorError::InvalidParams(
            "support-only distributed CRSC currently supports decode_radius <= 2",
        ));
    }
    let mut total = 1usize;
    if code.t() >= 1 {
        total = total
            .checked_add(code.n())
            .ok_or(OperatorError::InvalidParams(
                "support candidate count overflow",
            ))?;
    }
    if code.t() >= 2 {
        total = total
            .checked_add(
                code.n().checked_mul(code.n().saturating_sub(1)).ok_or(
                    OperatorError::InvalidParams("support candidate count overflow"),
                )? / 2,
            )
            .ok_or(OperatorError::InvalidParams(
                "support candidate count overflow",
            ))?;
    }
    Ok(total)
}

fn h_at(code: &ReedSolomonCode, row: usize, col: usize) -> Result<u64, OperatorError> {
    code.parity_check()
        .get(row, col)
        .map_err(|err| OperatorError::Backend(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use silent_lpn::LinearCode;

    #[test]
    fn distributed_lookup_crsc_recovers_without_opening_syndrome() {
        let code = ReedSolomonCode::new(17, 5, 3, 1).unwrap();
        let msg = vec![3, 4, 5];
        let clean = code.encode(&msg).unwrap();
        let mut noisy = clean.clone();
        noisy[2] = add_mod(noisy[2], 7, 17);
        let client_codeword_share = vec![2, 7, 9, 1, 12];
        let server_codeword_share = noisy
            .iter()
            .zip(client_codeword_share.iter())
            .map(|(&value, &share)| sub_mod(value, share, 17))
            .collect::<Vec<_>>();
        let lookup = PrivateLookup::new(crate::lookup::PrivateLookupConfig {
            output_modulus: 17,
            q_modulus_bits: 20,
            matrix_rows: 6,
            gadget_cols_t: 4,
            max_domain: 17,
        })
        .unwrap();
        let mut rng = StdRng::seed_from_u64(991);
        let out = distributed_lookup_crsc_convert(
            &code,
            &client_codeword_share,
            &server_codeword_share,
            &lookup,
            &DistributedCrscConfig::default(),
            &mut rng,
        )
        .unwrap();
        let opened = out
            .client_share
            .iter()
            .zip(out.server_share.iter())
            .map(|(&lhs, &rhs)| add_mod(lhs, rhs, 17))
            .collect::<Vec<_>>();
        assert_eq!(opened, msg);
        assert_eq!(out.error_candidate_count, 1 + 5);
    }

    #[test]
    fn support_only_crsc_uses_support_count_for_large_field() {
        let code = ReedSolomonCode::new(257, 18, 16, 1).unwrap();
        let msg = (0..code.r())
            .map(|idx| (idx as u64 + 11) % 257)
            .collect::<Vec<_>>();
        let clean = code.encode(&msg).unwrap();
        let mut noisy = clean.clone();
        noisy[13] = add_mod(noisy[13], 91, 257);
        let client_codeword_share = (0..code.n())
            .map(|idx| ((idx * 19 + 3) as u64) % 257)
            .collect::<Vec<_>>();
        let server_codeword_share = noisy
            .iter()
            .zip(client_codeword_share.iter())
            .map(|(&value, &share)| sub_mod(value, share, 257))
            .collect::<Vec<_>>();
        let lookup = PrivateLookup::new(crate::lookup::PrivateLookupConfig {
            output_modulus: 257,
            q_modulus_bits: 26,
            matrix_rows: 6,
            gadget_cols_t: 4,
            max_domain: 257,
        })
        .unwrap();
        let mut rng = StdRng::seed_from_u64(4421);
        let out = distributed_lookup_crsc_convert(
            &code,
            &client_codeword_share,
            &server_codeword_share,
            &lookup,
            &DistributedCrscConfig::default(),
            &mut rng,
        )
        .unwrap();
        let opened = out
            .client_share
            .iter()
            .zip(out.server_share.iter())
            .map(|(&lhs, &rhs)| add_mod(lhs, rhs, 257))
            .collect::<Vec<_>>();
        assert_eq!(opened, msg);
        assert_eq!(out.error_candidate_count, 1 + 18);
    }

    #[test]
    fn support_only_crsc_handles_weight_two_batch() {
        let code = ReedSolomonCode::new(17, 9, 1, 2).unwrap();
        let lookup = PrivateLookup::new(crate::lookup::PrivateLookupConfig {
            output_modulus: 17,
            q_modulus_bits: 24,
            matrix_rows: 6,
            gadget_cols_t: 4,
            max_domain: 17,
        })
        .unwrap();
        let msg0 = vec![6];
        let msg1 = vec![12];
        let mut noisy0 = code.encode(&msg0).unwrap();
        let mut noisy1 = code.encode(&msg1).unwrap();
        noisy0[1] = add_mod(noisy0[1], 3, 17);
        noisy0[7] = add_mod(noisy0[7], 5, 17);
        noisy1[2] = add_mod(noisy1[2], 9, 17);
        noisy1[6] = add_mod(noisy1[6], 4, 17);
        let client0 = (0..code.n())
            .map(|idx| ((idx * 5 + 4) as u64) % 17)
            .collect::<Vec<_>>();
        let client1 = (0..code.n())
            .map(|idx| ((idx * 7 + 2) as u64) % 17)
            .collect::<Vec<_>>();
        let server0 = noisy0
            .iter()
            .zip(client0.iter())
            .map(|(&value, &share)| sub_mod(value, share, 17))
            .collect::<Vec<_>>();
        let server1 = noisy1
            .iter()
            .zip(client1.iter())
            .map(|(&value, &share)| sub_mod(value, share, 17))
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(91);
        let out = distributed_lookup_crsc_convert_many(
            &code,
            &[client0, client1],
            &[server0, server1],
            &lookup,
            &DistributedCrscConfig::default(),
            &mut rng,
        )
        .unwrap();
        let opened = out
            .iter()
            .map(|row| {
                row.client_share
                    .iter()
                    .zip(row.server_share.iter())
                    .map(|(&lhs, &rhs)| add_mod(lhs, rhs, 17))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(opened, vec![msg0, msg1]);
        assert_eq!(out[0].error_candidate_count, 1 + 9 + 36);
        assert_eq!(out[1].error_candidate_count, 1 + 9 + 36);
    }
}
