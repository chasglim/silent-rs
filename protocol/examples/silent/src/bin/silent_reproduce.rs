use rand::SeedableRng;
use rand::rngs::StdRng;
use silent_hss::{HssContext, HssKeyGenerator};
use silent_math::modulus::Modulus;
use silent_math::numth::{first_ntt_prime_with_bits, next_ntt_prime};
use silent_math::rns::RnsBase;
use silent_math::rns_tool::RnsToolConfig;
use silent_net::frame::MessageKind;
use silent_net::transport::memory::MemoryConnection;
use silent_net::transport::tcp::TcpTransport;
use silent_net::transport::{Connection, Stream};
use silent_operators::fixedpoint::FixedPointConfig;
use silent_operators::hss_bridge::context_from_runtime;
use silent_operators::hss_slots::HssSlotEngine;
use silent_operators::linear_map::LinearMap;
use silent_operators::lookup::{PrivateLookup, PrivateLookupConfig};
use silent_operators::nonlinear::{GeluConfig, LayerNormConfig, NonlinearOps, SoftmaxConfig};
use silent_operators::shares::AdditiveShares;
use silent_operators::truncation::{Truncation, TruncationConfig};
use silent_params::{RegisteredParameterSet, find_preset};
use silent_protocol_runtime::{
    FieldType, PartyId, RuntimeChannel, RuntimeConfig, RuntimeContext, RuntimeError, RuntimeMesh,
    RuntimeSession, SessionId, TaskId, Value, WireContext,
};
use silent_rlwe::EncryptionParams;
use silent_utils::rng::SecureRng;
use std::time::{Duration, Instant};
use tokio::runtime::{Builder, Runtime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (iterations, mode, include_legacy_diagnostics) = parse_cli()?;

    println!("section,case,iterations,total_ms,avg_ms,extra");
    bench_linear_map(iterations, mode)?;
    bench_hss_mul(iterations, mode)?;
    if include_legacy_diagnostics {
        bench_lookup_and_trunc(iterations, mode)?;
        bench_transformer_nonlinears(iterations, mode)?;
    }
    bench_runtime_transport(iterations, mode)?;
    print_barrier_model();
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExperimentMode {
    Smoke,
    Full,
}

impl ExperimentMode {
    fn parse(value: &str) -> Result<Self, RuntimeError> {
        match value {
            "smoke" => Ok(Self::Smoke),
            "full" => Ok(Self::Full),
            _ => Err(RuntimeError::InvalidConfig(
                "SILENT experiment mode must be smoke or full",
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Full => "full",
        }
    }

    fn default_iterations(self) -> usize {
        match self {
            Self::Smoke => 3,
            Self::Full => 1,
        }
    }

    fn lookup_modulus(self) -> u64 {
        match self {
            Self::Smoke => 97,
            Self::Full => 65_537,
        }
    }

    fn lookup_radix_bits(self) -> usize {
        match self {
            Self::Smoke => 5,
            Self::Full => 15,
        }
    }

    fn lookup_config(self) -> PrivateLookupConfig {
        match self {
            Self::Smoke => PrivateLookupConfig {
                output_modulus: 97,
                q_modulus_bits: 50,
                matrix_rows: 16,
                gadget_cols_t: 6,
                max_domain: 128,
            },
            Self::Full => PrivateLookupConfig {
                output_modulus: 65_537,
                q_modulus_bits: 62,
                matrix_rows: 16,
                gadget_cols_t: 8,
                max_domain: 65_537,
            },
        }
    }
}

fn parse_cli() -> Result<(usize, ExperimentMode, bool), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).peekable();
    let mut iterations = std::env::var("ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let mut mode = std::env::var("SILENT_PROFILE")
        .or_else(|_| std::env::var("MODE"))
        .ok()
        .map(|value| ExperimentMode::parse(&value))
        .transpose()?;
    let mut include_legacy_diagnostics = std::env::var("SILENT_LEGACY_DIAGNOSTICS")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--legacy-diagnostics" => include_legacy_diagnostics = true,
            "--mode" | "--profile" => {
                let value = args
                    .next()
                    .ok_or(RuntimeError::InvalidConfig("--mode requires smoke or full"))?;
                mode = Some(ExperimentMode::parse(&value)?);
            }
            "--iterations" | "-n" => {
                let value = args.next().ok_or(RuntimeError::InvalidConfig(
                    "--iterations requires a positive integer",
                ))?;
                iterations = Some(value.parse::<usize>()?);
            }
            "smoke" | "full" => mode = Some(ExperimentMode::parse(&arg)?),
            value => {
                iterations = Some(value.parse::<usize>()?);
            }
        }
    }

    let mode = mode.unwrap_or(ExperimentMode::Smoke);
    Ok((
        iterations
            .unwrap_or_else(|| mode.default_iterations())
            .max(1),
        mode,
        include_legacy_diagnostics,
    ))
}

fn bench_linear_map(
    iterations: usize,
    mode: ExperimentMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(0xC111);
    let p = mode.lookup_modulus();
    let dim = match mode {
        ExperimentMode::Smoke => 8,
        ExperimentMode::Full => 32,
    };
    let q = match mode {
        ExperimentMode::Smoke => p * 65_537u64,
        ExperimentMode::Full => p * (1u64 << 32),
    };
    let poly_degree = match mode {
        ExperimentMode::Smoke => 16,
        ExperimentMode::Full => 64,
    };
    let lwe_rows_n = match mode {
        ExperimentMode::Smoke => 16,
        ExperimentMode::Full => dim * 2,
    };
    let crs = LinearMap::setup(dim, poly_degree, q, p, lwe_rows_n, 8, &mut rng)?;
    let matrix = (0..dim)
        .map(|i| (0..dim).map(|j| ((i + 2 * j + 1) as u64) % p).collect())
        .collect::<Vec<Vec<u64>>>();
    let vector = (0..dim).map(|i| (i as u64 + 3) % p).collect::<Vec<_>>();

    let mut elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let setup = LinearMap::server_setup(&matrix, &crs, 0, true, &mut rng)?;
        let query = LinearMap::client_query(&vector, &crs, 0, &mut rng)?;
        let server = LinearMap::server_extract(&query, &setup)?;
        let client = LinearMap::client_extract(&setup, &query.state, matrix.len())?;
        let rec_q = LinearMap::reconstruct_q(&server, &client, crs.q_modulus)?;
        let _decoded = LinearMap::decode_q_to_p(&rec_q, crs.q_modulus, crs.p_modulus);
        elapsed += start.elapsed();
    }
    let case = format!("rpm_cnim_nimvm_{dim}x{dim}");
    let extra = format!(
        "profile={},p={p},includes_offline_setup=true",
        mode.as_str()
    );
    print_row("linear", &case, iterations, elapsed, &extra);

    let setup = LinearMap::server_setup(&matrix, &crs, 0, true, &mut rng)?;
    let mut online_elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let query = LinearMap::client_query(&vector, &crs, 0, &mut rng)?;
        let server = LinearMap::server_extract(&query, &setup)?;
        let client = LinearMap::client_extract(&setup, &query.state, matrix.len())?;
        let rec_q = LinearMap::reconstruct_q(&server, &client, crs.q_modulus)?;
        let _decoded = LinearMap::decode_q_to_p(&rec_q, crs.q_modulus, crs.p_modulus);
        online_elapsed += start.elapsed();
    }
    let case = format!("rpm_cnim_nimvm_{dim}x{dim}_online");
    let extra = format!(
        "profile={},p={p},server_setup=offline,row_block=monolithic",
        mode.as_str()
    );
    print_row("linear", &case, iterations, online_elapsed, &extra);

    let mut blocked_elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let setup = LinearMap::server_setup_row_blocked(&matrix, &crs, 0, true, 4, &mut rng)?;
        let query = LinearMap::client_query(&vector, &crs, 0, &mut rng)?;
        let server = LinearMap::server_extract_row_blocked(&query, &setup)?;
        let client = LinearMap::client_extract_row_blocked(&setup, &query.state, matrix.len())?;
        let rec_q = LinearMap::reconstruct_q(&server, &client, crs.q_modulus)?;
        let _decoded = LinearMap::decode_q_to_p(&rec_q, crs.q_modulus, crs.p_modulus);
        blocked_elapsed += start.elapsed();
    }
    let case = format!("rpm_cnim_nimvm_{dim}x{dim}_row_block4");
    let extra = format!(
        "profile={},p={p},row_block=4,includes_offline_setup=true",
        mode.as_str()
    );
    print_row("linear", &case, iterations, blocked_elapsed, &extra);

    let setup = LinearMap::server_setup_row_blocked(&matrix, &crs, 0, true, 4, &mut rng)?;
    let mut blocked_online_elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let query = LinearMap::client_query(&vector, &crs, 0, &mut rng)?;
        let server = LinearMap::server_extract_row_blocked(&query, &setup)?;
        let client = LinearMap::client_extract_row_blocked(&setup, &query.state, matrix.len())?;
        let rec_q = LinearMap::reconstruct_q(&server, &client, crs.q_modulus)?;
        let _decoded = LinearMap::decode_q_to_p(&rec_q, crs.q_modulus, crs.p_modulus);
        blocked_online_elapsed += start.elapsed();
    }
    let case = format!("rpm_cnim_nimvm_{dim}x{dim}_row_block4_online");
    let extra = format!(
        "profile={},p={p},server_setup=offline,row_block=4",
        mode.as_str()
    );
    print_row("linear", &case, iterations, blocked_online_elapsed, &extra);
    Ok(())
}

fn bench_hss_mul(
    iterations: usize,
    mode: ExperimentMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = match mode {
        ExperimentMode::Smoke => build_toy_engine(16, 97, "inference-silent-reproduce-hss")?,
        ExperimentMode::Full => build_hss_preset_engine("current-hss-v1")?,
    };
    let p = engine.context().plain_modulus();
    let lanes = match mode {
        ExperimentMode::Smoke => 8,
        ExperimentMode::Full => 64,
    };
    let lhs_values = (0..lanes)
        .map(|idx| (2 * idx as u64 + 3) % p)
        .collect::<Vec<_>>();
    let rhs_values = (0..lanes)
        .map(|idx| (5 * idx as u64 + 7) % p)
        .collect::<Vec<_>>();
    let mut share_rng = StdRng::seed_from_u64(0x5155);
    let lhs = AdditiveShares::share_with_rng(&lhs_values, p, &mut share_rng)?;
    let rhs = AdditiveShares::share_with_rng(&rhs_values, p, &mut share_rng)?;
    let mut rng = StdRng::seed_from_u64(0x5156);

    let mut elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = engine.mul_slots(&lhs, &rhs, &mut rng)?;
        elapsed += start.elapsed();
    }
    print_row(
        "hss",
        &format!("share2he_d2s_hss_mul_slots_{lanes}"),
        iterations,
        elapsed,
        &format!(
            "profile={},degree={},p={p}",
            mode.as_str(),
            engine.context().degree()
        ),
    );
    Ok(())
}

fn bench_lookup_and_trunc(
    iterations: usize,
    mode: ExperimentMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let lookup = PrivateLookup::new(mode.lookup_config())?;
    let p = lookup.config().output_modulus;
    let radix_bits = mode.lookup_radix_bits();
    let bit_and_engine = match mode {
        ExperimentMode::Smoke => build_toy_engine(16, p, "inference-silent-reproduce-lookup-and")?,
        ExperimentMode::Full => build_hss_preset_engine("current-hss-v1")?,
    };
    if bit_and_engine.context().plain_modulus() != p {
        return Err(Box::new(RuntimeError::InvalidValue(
            "lookup bit-AND HSS plaintext modulus mismatch",
        )));
    }
    let mut rng = StdRng::seed_from_u64(0x5157);

    for bit_width in [3usize, 4, 5, 6] {
        let domain = 1u64 << bit_width;
        let threshold = domain / 2;
        let share0 = vec![1u64, threshold.saturating_sub(1)];
        let share1 = vec![2u64, 1];
        let mut cmp_elapsed = Duration::ZERO;
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = lookup.less_than_public(&share0, &share1, bit_width, threshold, &mut rng)?;
            cmp_elapsed += start.elapsed();
        }
        let case = format!("nidpf_less_than_domain{domain}_batch2");
        let extra = format!(
            "profile={},p={p},bit_width={bit_width},implemented_small_domain_sweep",
            mode.as_str()
        );
        print_row("nonlinear", &case, iterations, cmp_elapsed, &extra);
    }

    for bit_width in [8usize, 16, 32] {
        let case_radix_bits = radix_bits.min(bit_width);
        let threshold = 1u64 << (bit_width - 1);
        let values = vec![
            0,
            threshold - 1,
            threshold,
            u32::MAX as u64 & ((1u64 << bit_width) - 1),
        ];
        let (share0, share1) = share_mod_power_of_two(&values, bit_width);
        let mut elapsed = Duration::ZERO;
        for _ in 0..iterations {
            let start = Instant::now();
            let got = lookup.less_than_public_radix_with_bit_and(
                &share0,
                &share1,
                bit_width,
                threshold,
                case_radix_bits,
                &mut rng,
                |lhs, rhs, rng| bit_and_engine.mul_slots(lhs, rhs, rng),
            )?;
            elapsed += start.elapsed();
            let reconstructed = got.reconstruct();
            let expected = values
                .iter()
                .map(|&value| (value < threshold) as u64)
                .collect::<Vec<_>>();
            if reconstructed != expected {
                return Err(Box::new(RuntimeError::InvalidValue(
                    "radix comparison mismatch",
                )));
            }
        }
        let case = format!("nidcf_radix_less_than_{bit_width}bit_batch4");
        print_row(
            "nonlinear",
            &case,
            iterations,
            elapsed,
            &format!(
                "profile={},p={p},radix_bits={case_radix_bits},lookup_domain={p},bit_and=hss_slots",
                mode.as_str()
            ),
        );
    }

    let share0 = vec![7u64, 31];
    let share1 = vec![6u64, 2];
    let trunc = Truncation::new(
        TruncationConfig {
            bit_width: 5,
            shift: 2,
            output_modulus: p,
        },
        &lookup,
    )?;
    let mut trunc_elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = trunc.truncate(&share0, &share1, &mut rng)?;
        trunc_elapsed += start.elapsed();
    }
    print_row(
        "nonlinear",
        "nitrunc_n5_d2_batch2",
        iterations,
        trunc_elapsed,
        &format!("profile={},uses low+wrap correction", mode.as_str()),
    );

    let trunc32 = Truncation::new(
        TruncationConfig {
            bit_width: 32,
            shift: 13,
            output_modulus: p,
        },
        &lookup,
    )?;
    let values32 = vec![0u64, 8191, 8192, 1_000_000];
    let (share32_0, share32_1) = share_mod_power_of_two(&values32, 32);
    let mut trunc32_elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let got = trunc32.truncate_radix_with_bit_and(
            &share32_0,
            &share32_1,
            radix_bits,
            &mut rng,
            |lhs, rhs, rng| bit_and_engine.mul_slots(lhs, rhs, rng),
        )?;
        trunc32_elapsed += start.elapsed();
        let reconstructed = got.reconstruct();
        let expected = values32
            .iter()
            .map(|&value| (value >> 13) % got.modulus())
            .collect::<Vec<_>>();
        if reconstructed != expected {
            return Err(Box::new(RuntimeError::InvalidValue(
                "radix truncation mismatch",
            )));
        }
    }
    print_row(
        "nonlinear",
        "nitrunc_radix_n32_d13_batch4",
        iterations,
        trunc32_elapsed,
        &format!(
            "profile={},p={p},radix_bits={radix_bits},lookup_domain={p},bit_and=hss_slots",
            mode.as_str()
        ),
    );
    Ok(())
}

fn bench_transformer_nonlinears(
    iterations: usize,
    mode: ExperimentMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let (engine, lookup, fixed) = match mode {
        ExperimentMode::Smoke => {
            let engine = build_toy_engine(8, 17, "inference-silent-reproduce-nonlinear")?;
            let lookup = PrivateLookup::new(PrivateLookupConfig {
                output_modulus: 17,
                q_modulus_bits: 50,
                matrix_rows: 17,
                gadget_cols_t: 6,
                max_domain: 32,
            })?;
            (engine, lookup, FixedPointConfig::new(17, 2)?)
        }
        ExperimentMode::Full => {
            let engine = build_hss_preset_engine("current-hss-v1")?;
            let p = engine.context().plain_modulus();
            let lookup = PrivateLookup::new(PrivateLookupConfig {
                output_modulus: p,
                q_modulus_bits: 62,
                matrix_rows: 16,
                gadget_cols_t: 8,
                max_domain: p as usize,
            })?;
            (engine, lookup, FixedPointConfig::new(p, 16)?)
        }
    };
    let stack = NonlinearOps::new(fixed, &lookup, &engine)?;

    let mut share_rng = StdRng::seed_from_u64(0x5158);
    let gelu_input = AdditiveShares::share_with_rng(
        &[
            fixed.encode_f64(-1.0),
            fixed.encode_f64(0.0),
            fixed.encode_f64(0.5),
            fixed.encode_f64(1.0),
        ],
        fixed.modulus,
        &mut share_rng,
    )?;
    let softmax_input = AdditiveShares::share_with_rng(
        &[fixed.encode_f64(-1.0), fixed.encode_f64(0.0)],
        fixed.modulus,
        &mut share_rng,
    )?;
    let layernorm_input = AdditiveShares::share_with_rng(
        &[fixed.encode_f64(-1.0), fixed.encode_f64(1.0)],
        fixed.modulus,
        &mut share_rng,
    )?;

    let mut rng = StdRng::seed_from_u64(0x5159);
    let mut gelu_elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = stack.gelu(
            &gelu_input,
            &GeluConfig {
                t1: -0.5,
                t2: 0.5,
                coeffs: [0.0, 0.5, 0.0, 0.0],
            },
            &mut rng,
        )?;
        gelu_elapsed += start.elapsed();
    }
    print_row(
        "nonlinear",
        "gelu_piecewise_selector_hss_poly",
        iterations,
        gelu_elapsed,
        &format!(
            "profile={},p={},scale={}",
            mode.as_str(),
            fixed.modulus,
            fixed.scale
        ),
    );

    let mut softmax_elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = stack.softmax(&softmax_input, &SoftmaxConfig::default(), &mut rng)?;
        softmax_elapsed += start.elapsed();
    }
    print_row(
        "nonlinear",
        "softmax_max_exp_recip_hss",
        iterations,
        softmax_elapsed,
        &format!(
            "profile={},row=2,p={},scale={}",
            mode.as_str(),
            fixed.modulus,
            fixed.scale
        ),
    );

    let mut layernorm_elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = stack.layer_norm(
            &layernorm_input,
            &LayerNormConfig {
                epsilon: 0.0,
                variance_clip_max: 4.0,
                gamma: Vec::new(),
                beta: Vec::new(),
            },
            &mut rng,
        )?;
        layernorm_elapsed += start.elapsed();
    }
    print_row(
        "nonlinear",
        "layernorm_mean_var_rsqrt_hss",
        iterations,
        layernorm_elapsed,
        &format!(
            "profile={},row=2,p={},scale={}",
            mode.as_str(),
            fixed.modulus,
            fixed.scale
        ),
    );
    Ok(())
}

fn print_barrier_model() {
    let layers = 12u64;
    let barriers = 2 * layers;
    let lan_rtt_ms = 0.5f64;
    let wan_rtt_ms = 4.0f64;
    println!(
        "e2e_model,bert_base_refresh_barriers,1,{:.3},{:.3},R={} LAN_RTT_ms={} WAN_RTT_ms={} WAN_RTT_component_ms={:.3}",
        barriers as f64 * wan_rtt_ms,
        barriers as f64 * wan_rtt_ms,
        barriers,
        lan_rtt_ms,
        wan_rtt_ms,
        barriers as f64 * wan_rtt_ms
    );
}

fn bench_runtime_transport(
    iterations: usize,
    mode: ExperimentMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Builder::new_current_thread().enable_all().build()?;
    let value = runtime_roundtrip_value()?;

    let (left, right) = MemoryConnection::pair();
    let (stream_left, stream_right) = runtime.block_on(async {
        let stream_left = left.open_stream().await?;
        let stream_right = right.accept_stream().await?;
        Ok::<_, std::io::Error>((stream_left, stream_right))
    })?;
    let (mut session_left, mut session_right) =
        runtime_sessions_from_streams(stream_left, stream_right, SessionId(2026))?;
    let elapsed = measure_runtime_value_roundtrip(
        &runtime,
        &mut session_left,
        &mut session_right,
        &value,
        iterations,
    )?;
    print_runtime_transport_row(
        "silent_net_value_roundtrip_u64x8",
        iterations,
        elapsed,
        mode,
        "memory",
        None,
        session_left.stats(),
        session_right.stats(),
    );

    let (stream_left, stream_right, endpoint) = runtime.block_on(tcp_loopback_stream_pair())?;
    let (mut session_left, mut session_right) =
        runtime_sessions_from_streams(stream_left, stream_right, SessionId(2027))?;
    let tcp_elapsed = measure_runtime_value_roundtrip(
        &runtime,
        &mut session_left,
        &mut session_right,
        &value,
        iterations,
    )?;
    print_runtime_transport_row(
        "silent_net_tcp_loopback_value_roundtrip_u64x8",
        iterations,
        tcp_elapsed,
        mode,
        "tcp_loopback",
        Some(endpoint.as_str()),
        session_left.stats(),
        session_right.stats(),
    );
    print_runtime_latency_model(mode, iterations, tcp_elapsed);
    Ok(())
}

fn runtime_roundtrip_value() -> Result<Value, RuntimeError> {
    Value::public_u64(
        vec![3, 5, 8, 13, 21, 34, 55, 89],
        FieldType::Ring64,
        vec![8],
    )
}

fn runtime_sessions_from_streams(
    stream_left: Box<dyn Stream>,
    stream_right: Box<dyn Stream>,
    session_id: SessionId,
) -> Result<(RuntimeSession, RuntimeSession), RuntimeError> {
    let wire = WireContext::default();
    let cfg = RuntimeConfig::two_party_inference();
    let mut mesh_left = RuntimeMesh::new(PartyId(0));
    let mut mesh_right = RuntimeMesh::new(PartyId(1));
    mesh_left.insert_peer(PartyId(1), RuntimeChannel::new(stream_left, wire.clone()))?;
    mesh_right.insert_peer(PartyId(0), RuntimeChannel::new(stream_right, wire.clone()))?;
    let ctx_left = RuntimeContext::new(cfg.clone(), PartyId(0), session_id, wire.clone())?;
    let ctx_right = RuntimeContext::new(cfg, PartyId(1), session_id, wire)?;
    Ok((
        RuntimeSession::with_mesh(ctx_left, mesh_left)?,
        RuntimeSession::with_mesh(ctx_right, mesh_right)?,
    ))
}

fn measure_runtime_value_roundtrip(
    runtime: &Runtime,
    session_left: &mut RuntimeSession,
    session_right: &mut RuntimeSession,
    value: &Value,
    iterations: usize,
) -> Result<Duration, RuntimeError> {
    let mut elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        let got = runtime.block_on(async {
            let route = session_left
                .send_value(PartyId(1), TaskId(17), MessageKind::Data, value)
                .await?;
            session_right.recv_expected_value(PartyId(0), route).await
        })?;
        if got != *value {
            return Err(RuntimeError::InvalidValue(
                "runtime value roundtrip mismatch",
            ));
        }
        elapsed += start.elapsed();
    }
    Ok(elapsed)
}

async fn tcp_loopback_stream_pair()
-> Result<(Box<dyn Stream>, Box<dyn Stream>, String), RuntimeError> {
    let listener = TcpTransport::new()
        .bind("127.0.0.1:0")
        .await
        .map_err(|err| RuntimeError::Network(err.to_string()))?;
    let endpoint = listener
        .local_addr()
        .map_err(|err| RuntimeError::Network(err.to_string()))?
        .to_string();
    let accept_task = tokio::spawn(async move {
        let conn = listener
            .accept()
            .await
            .map_err(|err| RuntimeError::Network(err.to_string()))?;
        conn.open_stream()
            .await
            .map_err(|err| RuntimeError::Network(err.to_string()))
    });
    let client_conn = TcpTransport::new()
        .connect(&endpoint)
        .await
        .map_err(|err| RuntimeError::Network(err.to_string()))?;
    let stream_left = client_conn
        .open_stream()
        .await
        .map_err(|err| RuntimeError::Network(err.to_string()))?;
    let stream_right = accept_task
        .await
        .map_err(|err| RuntimeError::Network(err.to_string()))??;
    Ok((stream_left, stream_right, endpoint))
}

fn print_runtime_transport_row(
    case: &str,
    iterations: usize,
    elapsed: Duration,
    mode: ExperimentMode,
    transport: &str,
    endpoint: Option<&str>,
    sent: silent_protocol_runtime::RuntimeStats,
    received: silent_protocol_runtime::RuntimeStats,
) {
    let mut extra = format!(
        "profile={},transport={transport},frames_sent={},frames_received={},bytes_sent={},bytes_received={}",
        mode.as_str(),
        sent.frames_sent,
        received.frames_received,
        sent.bytes_sent,
        received.bytes_received
    );
    if let Some(endpoint) = endpoint {
        extra.push_str(&format!(",endpoint={endpoint}"));
    }
    print_row("runtime", case, iterations, elapsed, &extra);
}

fn print_runtime_latency_model(mode: ExperimentMode, iterations: usize, tcp_elapsed: Duration) {
    if iterations == 0 {
        return;
    }
    let barriers = 24f64;
    let tcp_rtt_ms = tcp_elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let tcp_component = Duration::from_secs_f64((tcp_rtt_ms * barriers) / 1000.0);
    let lan_rtt_ms = 0.5f64;
    let wan_rtt_ms = 4.0f64;
    print_row(
        "runtime_model",
        "bert_base_refresh_barriers_tcp_loopback",
        1,
        tcp_component,
        &format!(
            "profile={},R=24,tcp_loopback_rtt_ms={tcp_rtt_ms:.6},LAN_RTT_ms={lan_rtt_ms},WAN_RTT_ms={wan_rtt_ms},LAN_component_ms={:.3},WAN_component_ms={:.3}",
            mode.as_str(),
            barriers * lan_rtt_ms,
            barriers * wan_rtt_ms,
        ),
    );
}

fn build_toy_engine(
    degree: usize,
    plain_modulus: u64,
    name: &'static str,
) -> Result<HssSlotEngine, Box<dyn std::error::Error>> {
    let step = 2 * degree as u64;
    let q1 = first_ntt_prime_with_bits(50, step).ok_or("q1")?;
    let q2 = next_ntt_prime(q1 + 1, step).ok_or("q2")?;
    let q3 = next_ntt_prime(q2 + 1, step).ok_or("q3")?;
    let q4 = next_ntt_prime(q3 + 1, step).ok_or("q4")?;
    let base_q = RnsBase::from_values(vec![q1, q2, q3, q4]).map_err(|err| format!("{err:?}"))?;
    let runtime = EncryptionParams::from_rns_config(
        degree,
        RnsToolConfig::new(
            base_q,
            Modulus::new(plain_modulus).map_err(|err| format!("{err:?}"))?,
        ),
    )
    .map_err(|err| format!("{err:?}"))?;
    let context = context_from_runtime(name, runtime, plain_modulus)?;

    let mut keygen = HssKeyGenerator::new(context.clone(), SecureRng::from_seed([29u8; 32]));
    let sk = keygen.generate_secret_key();
    let pk = keygen.generate_public_key(&sk);
    let (share0, share1) = keygen.split_secret_key(&sk);
    Ok(HssSlotEngine::new(context, pk, share0, share1))
}

fn build_hss_preset_engine(name: &str) -> Result<HssSlotEngine, Box<dyn std::error::Error>> {
    let preset = find_preset(name).ok_or_else(|| format!("missing HSS preset {name}"))?;
    let RegisteredParameterSet::Hss(params) = preset else {
        return Err(format!("preset {name} is not an HSS parameter set").into());
    };
    let context = HssContext::new(params)?;
    let mut keygen = HssKeyGenerator::new(context.clone(), SecureRng::from_seed([41u8; 32]));
    let sk = keygen.generate_secret_key();
    let pk = keygen.generate_public_key(&sk);
    let (share0, share1) = keygen.split_secret_key(&sk);
    Ok(HssSlotEngine::new(context, pk, share0, share1))
}

fn print_row(section: &str, case: &str, iterations: usize, elapsed: Duration, extra: &str) {
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / iterations as f64;
    println!("{section},{case},{iterations},{total_ms:.3},{avg_ms:.3},{extra}");
}

fn share_mod_power_of_two(values: &[u64], bit_width: usize) -> (Vec<u64>, Vec<u64>) {
    let modulus = 1u64 << bit_width;
    let mask = modulus - 1;
    let share0 = values
        .iter()
        .enumerate()
        .map(|(idx, _)| ((idx as u64 * 0x9e37_79b1) + 17) & mask)
        .collect::<Vec<_>>();
    let share1 = values
        .iter()
        .zip(share0.iter())
        .map(|(&value, &mask_share)| value.wrapping_sub(mask_share) & mask)
        .collect::<Vec<_>>();
    (share0, share1)
}
