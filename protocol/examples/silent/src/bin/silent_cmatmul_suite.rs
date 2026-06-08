use rand::SeedableRng;
use rand::rngs::StdRng;
use silent_math::arith::add_mod;
use silent_operators::linear_bridge::LinearMapQShares;
use silent_operators::packed_linear_map::{
    PackedLinearMap, PackedLinearMapConfig, PackedLinearMapCrsCache,
};
use silent_params::{
    DistributionType, ModulusBits, PlaintextModulus, RingDim, RingParams, RlweParams, SecurityLevel,
};
use silent_rlwe::EncryptionParams;
use std::time::{Duration, Instant};

const P: u64 = 65_537;

#[derive(Clone, Copy)]
struct Case {
    group: &'static str,
    operator: &'static str,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    degree: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut iterations = 1usize;
    let mut degree_override = std::env::var("SILENT_CMATMUL_DEGREE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--degree" => {
                degree_override = args
                    .next()
                    .ok_or("--degree requires a power-of-two degree")?
                    .parse::<usize>()
                    .map(Some)?;
            }
            "--iterations" | "-n" => {
                iterations = args
                    .next()
                    .ok_or("--iterations requires a positive integer")?
                    .parse::<usize>()?
                    .max(1);
            }
            value => iterations = value.parse::<usize>()?.max(1),
        }
    }

    let cases = [
        Case {
            group: "Batch = 32",
            operator: "Q/K/V projection",
            rows: 32,
            input_dim: 768,
            output_dim: 64,
            degree: 2_048,
        },
        Case {
            group: "Batch = 32",
            operator: "Output projection",
            rows: 32,
            input_dim: 768,
            output_dim: 768,
            degree: 2_048,
        },
        Case {
            group: "Batch = 32",
            operator: "FFN expansion",
            rows: 32,
            input_dim: 768,
            output_dim: 3072,
            degree: 2_048,
        },
        Case {
            group: "Batch = 32",
            operator: "FFN contraction",
            rows: 32,
            input_dim: 3072,
            output_dim: 768,
            degree: 4_096,
        },
        Case {
            group: "Batch = 64",
            operator: "Q/K/V projection",
            rows: 64,
            input_dim: 768,
            output_dim: 64,
            degree: 2_048,
        },
        Case {
            group: "Batch = 64",
            operator: "Output projection",
            rows: 64,
            input_dim: 768,
            output_dim: 768,
            degree: 2_048,
        },
        Case {
            group: "Batch = 64",
            operator: "FFN expansion",
            rows: 64,
            input_dim: 768,
            output_dim: 3072,
            degree: 2_048,
        },
        Case {
            group: "Batch = 64",
            operator: "FFN contraction",
            rows: 64,
            input_dim: 3072,
            output_dim: 768,
            degree: 4_096,
        },
        Case {
            group: "Batch = 128",
            operator: "Q/K/V projection",
            rows: 128,
            input_dim: 768,
            output_dim: 64,
            degree: 2_048,
        },
        Case {
            group: "Batch = 128",
            operator: "Output projection",
            rows: 128,
            input_dim: 768,
            output_dim: 768,
            degree: 2_048,
        },
        Case {
            group: "Batch = 128",
            operator: "FFN expansion",
            rows: 128,
            input_dim: 768,
            output_dim: 3072,
            degree: 2_048,
        },
        Case {
            group: "Batch = 128",
            operator: "FFN contraction",
            rows: 128,
            input_dim: 3072,
            output_dim: 768,
            degree: 4_096,
        },
        Case {
            group: "Batch = 128",
            operator: "Classification",
            rows: 1,
            input_dim: 768,
            output_dim: 768,
            degree: 2_048,
        },
        Case {
            group: "Batch = 256",
            operator: "Q/K/V projection",
            rows: 256,
            input_dim: 768,
            output_dim: 64,
            degree: 2_048,
        },
        Case {
            group: "Batch = 256",
            operator: "Output projection",
            rows: 256,
            input_dim: 768,
            output_dim: 768,
            degree: 2_048,
        },
        Case {
            group: "Batch = 256",
            operator: "FFN expansion",
            rows: 256,
            input_dim: 768,
            output_dim: 3072,
            degree: 2_048,
        },
        Case {
            group: "Batch = 256",
            operator: "FFN contraction",
            rows: 256,
            input_dim: 3072,
            output_dim: 768,
            degree: 4_096,
        },
    ];

    println!(
        "group,operator,rows,input_dim,output_dim,system,iterations,offline_ms,online_ms,total_ms,client_query_bytes,offline_preprocess_bytes,poly_degree,limbs,blocks,block_cols,checksum,correct"
    );
    for (idx, base_case) in cases.iter().enumerate() {
        let mut case = *base_case;
        if let Some(degree) = degree_override {
            case.degree = degree;
        }
        let mut rng = StdRng::seed_from_u64(0xC0DE_0000 + idx as u64);
        let row = bench_rpm(case, iterations, &mut rng)?;
        println!("{row}");
    }
    Ok(())
}

fn bench_rpm(
    case: Case,
    iterations: usize,
    rng: &mut StdRng,
) -> Result<String, Box<dyn std::error::Error>> {
    let params = build_packed_linear_encryption_params(case.degree, P)?;
    let cfg = PackedLinearMapConfig {
        input_dim: case.input_dim,
        output_dim: case.output_dim,
        ring: params.ring.as_ref().clone(),
        plaintext_modulus: P,
        noise_bound: 0,
    };
    let cache = PackedLinearMapCrsCache::new();
    let crs = PackedLinearMap::setup_cached(cfg, &cache, rng)?;
    let input = deterministic_plain_matrix_flat(case.rows, case.input_dim, P);
    let matrix = deterministic_dense_matrix_flat(case.input_dim, case.output_dim, P);
    let expected = plain_matmul_output_major(
        &input,
        &matrix,
        case.rows,
        case.input_dim,
        case.output_dim,
        P,
    );

    let setup_start = Instant::now();
    let setup = PackedLinearMap::server_setup(&matrix, crs.as_ref(), rng)?;
    let offline = setup_start.elapsed();

    let mut online = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let query = PackedLinearMap::client_query(&input, crs.as_ref(), rng)?;
        let (party0_q, party1_q) = PackedLinearMap::extract_q_shares(&setup, &query, crs.as_ref())?;
        online += start.elapsed();
        let q_shares = LinearMapQShares::new(
            party0_q.clone(),
            party1_q.clone(),
            crs.cfg.q_modulus(),
            crs.cfg.plaintext_modulus,
            crs.cfg.output_dim,
            query.rows(),
        )?;
        let reconstructed = PackedLinearMap::reconstruct_p(&q_shares)?;
        if reconstructed != expected {
            return Err(format!(
                "RPM-CNIM correctness mismatch for {} {} degree {}",
                case.group, case.operator, case.degree
            )
            .into());
        }
        checksum = checksum.wrapping_add(
            party0_q
                .iter()
                .chain(party1_q.iter())
                .fold(0u64, |acc, &v| add_mod(acc, v % P, P)),
        );
    }

    let block_cols = crs.cfg.max_block_outputs();
    let blocks = case.output_dim.div_ceil(block_cols);
    let limbs = crs.cfg.ring.rns().len();
    let poly_bytes = case.degree as u64 * limbs as u64 * 8;
    let query_bytes = case.rows as u64 * poly_bytes;
    let offline_bytes = blocks as u64 * 2 * poly_bytes;
    let online_ms = online.as_secs_f64() * 1000.0 / iterations as f64;
    let offline_ms = offline.as_secs_f64() * 1000.0;
    let total_ms = online_ms + offline_ms;

    Ok(format!(
        "{},{},{},{},{},RPM-CNIM,{},{offline_ms:.3},{online_ms:.3},{total_ms:.3},{query_bytes},{offline_bytes},{},{limbs},{blocks},{block_cols},{checksum},true",
        case.group,
        case.operator,
        case.rows,
        case.input_dim,
        case.output_dim,
        iterations,
        case.degree
    ))
}

fn build_packed_linear_encryption_params(
    degree: usize,
    plaintext_modulus: u64,
) -> Result<EncryptionParams, Box<dyn std::error::Error>> {
    if !degree.is_power_of_two() {
        return Err("packed linear degree must be a power of two".into());
    }
    let log_n = RingDim(degree)
        .to_log_n()
        .ok_or("packed linear degree must fit SILENT ring metadata")?;
    let q_bits = if degree <= 2048 {
        vec![ModulusBits(54)]
    } else if degree <= 4096 {
        vec![ModulusBits(54), ModulusBits(55)]
    } else {
        vec![ModulusBits(55), ModulusBits(55)]
    };
    let rlwe = RlweParams::new(
        "silent-cmatmul-suite-rpm-v1",
        RingParams::new(log_n, RingDim(degree), DistributionType::Ternary),
        q_bits,
        Vec::new(),
        None,
        SecurityLevel::Classical128,
    );
    Ok(EncryptionParams::from_rlwe_parameter_set(
        &rlwe,
        PlaintextModulus(plaintext_modulus),
    )?)
}

fn deterministic_dense_matrix_flat(rows: usize, cols: usize, p: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let raw = ((row * 131 + col * 17 + 7) % 31) as i64 - 15;
            out.push(signed_mod(raw, p));
        }
    }
    out
}

fn deterministic_plain_matrix_flat(rows: usize, cols: usize, p: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            out.push(((row * 19 + col * 7 + 3) % 23) as u64 % p);
        }
    }
    out
}

fn plain_matmul_output_major(
    lhs_row_major: &[u64],
    rhs_row_major: &[u64],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    p: u64,
) -> Vec<u64> {
    let mut out = vec![0u64; rows * output_dim];
    for col in 0..output_dim {
        for row in 0..rows {
            let mut acc = 0u64;
            for inner in 0..input_dim {
                let lhs = lhs_row_major[row * input_dim + inner];
                let rhs = rhs_row_major[inner * output_dim + col];
                acc = add_mod(acc, ((lhs as u128 * rhs as u128) % p as u128) as u64, p);
            }
            out[col * rows + row] = acc;
        }
    }
    out
}

fn signed_mod(value: i64, p: u64) -> u64 {
    let p_i = p as i64;
    let mut value = value % p_i;
    if value < 0 {
        value += p_i;
    }
    value as u64
}
