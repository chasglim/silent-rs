use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use silent_hss::{HssContext, HssKeyGenerator};
use silent_math::modulus::Modulus;
use silent_math::numth::{first_ntt_prime_with_bits, next_ntt_prime};
use silent_math::rns::RnsBase;
use silent_math::rns_tool::RnsToolConfig;
use silent_net::frame::MessageKind;
use silent_net::transport::tcp::TcpTransport;
use silent_net::transport::{Connection, Stream};
use silent_operators::conversion::ShareConverter;
use silent_operators::fixedpoint::FixedPointConfig;
use silent_operators::hss_bridge::context_from_runtime;
use silent_operators::hss_slots::HssSlotEngine;
use silent_operators::linear_bridge::{
    LinearMapQShares, PrivateLinearMap, PrivateLinearMapConfig,
};
use silent_operators::lookup::{PrivateLookup, PrivateLookupConfig};
use silent_operators::nonlinear::{GeluConfig, LayerNormConfig, NonlinearOps, SoftmaxConfig};
use silent_operators::shares::AdditiveShares;
use silent_params::{RegisteredParameterSet, find_preset};
use silent_protocol_runtime::{
    PartyId, RuntimeChannel, RuntimeConfig, RuntimeContext, RuntimeError, RuntimeMesh,
    RuntimeSession, SessionId, TaskId, WireContext,
};
use silent_rlwe::EncryptionParams;
use silent_utils::rng::SecureRng;
use std::time::{Duration, Instant};
use tokio::runtime::{Builder, Runtime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (iterations, mode, no_header) = parse_cli()?;
    if !no_header {
        println!("section,case,iterations,total_ms,avg_ms,extra");
    }

    let setup_start = Instant::now();
    let stack = SilentStack::new(mode)?;
    let nonlinear = NonlinearOps::new(stack.fixed, &stack.lookup, &stack.hss)?;
    let setup_elapsed = setup_start.elapsed();
    print_row(
        "e2e",
        "silent_thor_style_transformer_setup",
        1,
        setup_elapsed,
        &format!(
            "profile={}|hidden={}|p={}|scale={}|hss_degree={}",
            mode.as_str(),
            stack.model.hidden,
            stack.fixed.modulus,
            stack.fixed.scale,
            stack.hss.context().degree()
        ),
    );

    let mut q_to_p_bridge = RuntimeQToPBridge::tcp_loopback()?;
    let mut online_elapsed = Duration::ZERO;
    let mut last_probs = Vec::new();
    for iter in 0..iterations {
        let mut protocol_rng = StdRng::seed_from_u64(0x51E2_1000 + iter as u64);
        let input = stack.input_plain();

        let start = Instant::now();
        let hidden = stack.input_projection.apply_with_converter(
            &input,
            &mut protocol_rng,
            |q_shares, rng| {
                q_to_p_bridge
                    .convert(q_shares, rng)
                    .map_err(|err| silent_operators::OperatorError::Backend(err.to_string()))
            },
        )?;
        let probs = stack
            .model
            .forward(&hidden, &nonlinear, &mut protocol_rng)?;
        online_elapsed += start.elapsed();

        stack.validate_output(&probs)?;
        last_probs = probs.reconstruct();
    }

    let decoded = last_probs
        .iter()
        .map(|&value| format!("{:.4}", stack.fixed.decode_f64(value)))
        .collect::<Vec<_>>()
        .join("|");
    let sum = last_probs
        .iter()
        .map(|&value| stack.fixed.decode_f64(value))
        .sum::<f64>();
    print_row(
        "e2e",
        "silent_thor_style_transformer_online",
        iterations,
        online_elapsed,
        &format!(
            "profile={}|hidden={}|blocks=1|linear=rpm_cnim_q_to_p_runtime_bridge+additive_share_public_matmul|attention=qk_softmax_v|nonlinear=hss_lookup|runtime_transport=tcp_loopback|decoded_probs={decoded}|prob_sum={sum:.4}",
            mode.as_str(),
            stack.model.hidden,
        ),
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExperimentMode {
    Smoke,
    Full,
}

impl ExperimentMode {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "smoke" => Ok(Self::Smoke),
            "full" => Ok(Self::Full),
            _ => Err(format!("mode must be smoke or full, got {value}").into()),
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
    let mut no_header = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" | "--profile" => {
                let value = args.next().ok_or("--mode requires smoke or full")?;
                mode = Some(ExperimentMode::parse(&value)?);
            }
            "--iterations" | "-n" => {
                let value = args.next().ok_or("--iterations requires a value")?;
                iterations = Some(value.parse::<usize>()?);
            }
            "--no-header" => no_header = true,
            "smoke" | "full" => mode = Some(ExperimentMode::parse(&arg)?),
            value => iterations = Some(value.parse::<usize>()?),
        }
    }

    let mode = mode.unwrap_or(ExperimentMode::Smoke);
    Ok((
        iterations
            .unwrap_or_else(|| mode.default_iterations())
            .max(1),
        mode,
        no_header,
    ))
}

struct SilentStack {
    fixed: FixedPointConfig,
    hss: HssSlotEngine,
    lookup: PrivateLookup,
    input_projection: PrivateLinearMap,
    model: TransformerBlock,
}

impl SilentStack {
    fn new(mode: ExperimentMode) -> Result<Self, Box<dyn std::error::Error>> {
        let (hss, lookup, fixed) = match mode {
            ExperimentMode::Smoke => {
                let hss = build_toy_engine(8, 17, "silent-e2e-transformer-smoke")?;
                let lookup = PrivateLookup::new(PrivateLookupConfig {
                    output_modulus: 17,
                    q_modulus_bits: 50,
                    matrix_rows: 17,
                    gadget_cols_t: 6,
                    max_domain: 32,
                })?;
                (hss, lookup, FixedPointConfig::new(17, 2)?)
            }
            ExperimentMode::Full => {
                let hss = build_hss_preset_engine("current-hss-v1")?;
                let p = hss.context().plain_modulus();
                let lookup = PrivateLookup::new(PrivateLookupConfig {
                    output_modulus: p,
                    q_modulus_bits: 62,
                    matrix_rows: 16,
                    gadget_cols_t: 8,
                    max_domain: p as usize,
                })?;
                (hss, lookup, FixedPointConfig::new(p, 16)?)
            }
        };

        let mut setup_rng = StdRng::seed_from_u64(0x51E2_5155);
        let input_projection = PrivateLinearMap::setup(
            &bootstrap_matrix(fixed.modulus),
            private_linear_config(mode, fixed, 4, 4),
            &mut setup_rng,
        )?;

        Ok(Self {
            fixed,
            hss,
            lookup,
            input_projection,
            model: TransformerBlock::new(fixed, 4),
        })
    }

    fn input_plain(&self) -> Vec<u64> {
        [-0.5, 0.25, 0.75, -0.25]
            .iter()
            .map(|&value| self.fixed.encode_f64(value))
            .collect::<Vec<_>>()
    }

    fn validate_output(&self, probs: &AdditiveShares) -> Result<(), Box<dyn std::error::Error>> {
        if probs.len() != 2 || probs.modulus() != self.fixed.modulus {
            return Err("classifier output shape/modulus mismatch".into());
        }
        let decoded = probs
            .reconstruct()
            .iter()
            .map(|&value| self.fixed.decode_f64(value))
            .collect::<Vec<_>>();
        if decoded.iter().any(|value| !value.is_finite()) {
            return Err("classifier output contains a non-finite value".into());
        }
        let sum = decoded.iter().sum::<f64>();
        let tolerance = 2.0 / self.fixed.scale as f64;
        if (sum - 1.0).abs() > tolerance {
            return Err(format!("softmax probability sum check failed: {sum:.6}").into());
        }
        Ok(())
    }
}

struct RuntimeQToPBridge {
    runtime: Runtime,
    client: RuntimeSession,
    server: RuntimeSession,
}

impl RuntimeQToPBridge {
    fn tcp_loopback() -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = Builder::new_current_thread().enable_all().build()?;
        let (client_stream, server_stream) = runtime.block_on(tcp_loopback_stream_pair())?;
        let (client, server) =
            runtime_sessions_from_streams(client_stream, server_stream, SessionId(4520))?;
        Ok(Self {
            runtime,
            client,
            server,
        })
    }

    fn convert(
        &mut self,
        q_shares: &LinearMapQShares,
        rng: &mut StdRng,
    ) -> Result<AdditiveShares, Box<dyn std::error::Error>> {
        let mut client_rng = StdRng::seed_from_u64(rng.next_u64());
        let mut server_rng = StdRng::seed_from_u64(rng.next_u64());
        let client_state = ShareConverter::mask_q_to_p(
            q_shares.party0_q(),
            q_shares.q_modulus(),
            q_shares.p_modulus(),
            &mut client_rng,
        )?;
        let server_state = ShareConverter::mask_q_to_p(
            q_shares.party1_q(),
            q_shares.q_modulus(),
            q_shares.p_modulus(),
            &mut server_rng,
        )?;

        let (server_payload_for_client, client_payload_for_server) =
            self.runtime.block_on(async {
                let client_route = self
                    .client
                    .send_u64_vec(
                        PartyId(1),
                        TaskId(4521),
                        MessageKind::Data,
                        client_state.masked_q(),
                    )
                    .await?;
                let server_route = self
                    .server
                    .send_u64_vec(
                        PartyId(0),
                        TaskId(4521),
                        MessageKind::Data,
                        server_state.masked_q(),
                    )
                    .await?;
                let server_payload_for_client = self
                    .client
                    .recv_expected_u64_vec(PartyId(1), server_route)
                    .await?;
                let client_payload_for_server = self
                    .server
                    .recv_expected_u64_vec(PartyId(0), client_route)
                    .await?;
                Ok::<_, RuntimeError>((server_payload_for_client, client_payload_for_server))
            })?;

        let client_p = ShareConverter::finish_q_to_p(0, &client_state, &server_payload_for_client)?;
        let server_p = ShareConverter::finish_q_to_p(1, &server_state, &client_payload_for_server)?;
        Ok(AdditiveShares::new(
            q_shares.p_modulus(),
            client_p,
            server_p,
        )?)
    }
}

struct TransformerBlock {
    hidden: usize,
    q_proj: Vec<Vec<u64>>,
    k_proj: Vec<Vec<u64>>,
    v_proj: Vec<Vec<u64>>,
    out_proj: Vec<Vec<u64>>,
    ff1: Vec<Vec<u64>>,
    ff2: Vec<Vec<u64>>,
    classifier: Vec<Vec<u64>>,
    classifier_bias: Vec<u64>,
    gelu: GeluConfig,
    softmax: SoftmaxConfig,
    ln: LayerNormConfig,
}

impl TransformerBlock {
    fn new(fixed: FixedPointConfig, hidden: usize) -> Self {
        Self {
            hidden,
            q_proj: diag_shift_matrix(hidden, 1, 1, fixed.modulus),
            k_proj: diag_shift_matrix(hidden, 2, -1, fixed.modulus),
            v_proj: identity_matrix(hidden, fixed.modulus),
            out_proj: diag_shift_matrix(hidden, 3, 1, fixed.modulus),
            ff1: diag_shift_matrix(hidden, 1, -1, fixed.modulus),
            ff2: diag_shift_matrix(hidden, 2, 1, fixed.modulus),
            classifier: classifier_matrix(hidden, fixed.modulus),
            classifier_bias: vec![fixed.encode_f64(0.25), fixed.encode_f64(-0.25)],
            gelu: GeluConfig::default(),
            softmax: SoftmaxConfig::default(),
            ln: LayerNormConfig {
                epsilon: 0.25,
                variance_clip_max: 4.0,
                gamma: vec![1.0; hidden],
                beta: vec![0.0; hidden],
            },
        }
    }

    fn forward(
        &self,
        input: &AdditiveShares,
        nonlinear: &NonlinearOps<'_>,
        rng: &mut StdRng,
    ) -> Result<AdditiveShares, Box<dyn std::error::Error>> {
        let q = input.matmul_public(&self.q_proj)?;
        let k = input.matmul_public(&self.k_proj)?;
        let v = input.matmul_public(&self.v_proj)?;

        let scores = nonlinear.mul_fixed(&q, &k, rng)?;
        let attention = nonlinear.softmax(&scores, &self.softmax, rng)?;
        let context = nonlinear.mul_fixed(&attention, &v, rng)?;
        let attended = context.matmul_public(&self.out_proj)?;
        let norm1 = nonlinear.layer_norm(&input.add(&attended)?, &self.ln, rng)?;

        let ff_hidden = norm1.matmul_public(&self.ff1)?;
        let ff_activated = nonlinear.gelu(&ff_hidden, &self.gelu, rng)?;
        let ff_out = ff_activated.matmul_public(&self.ff2)?;
        let norm2 = nonlinear.layer_norm(&norm1.add(&ff_out)?, &self.ln, rng)?;

        let logits = norm2
            .matmul_public(&self.classifier)?
            .add_public_to_party0(&self.classifier_bias)?;
        Ok(nonlinear.softmax(&logits, &self.softmax, rng)?)
    }
}

fn identity_matrix(size: usize, p: u64) -> Vec<Vec<u64>> {
    let mut matrix = vec![vec![0; size]; size];
    for (idx, row) in matrix.iter_mut().enumerate() {
        row[idx] = 1 % p;
    }
    matrix
}

fn diag_shift_matrix(size: usize, shift: usize, shifted_sign: i64, p: u64) -> Vec<Vec<u64>> {
    let mut matrix = identity_matrix(size, p);
    for (idx, row) in matrix.iter_mut().enumerate() {
        let shifted = (idx + shift) % size;
        row[shifted] = add_signed_mod(row[shifted], shifted_sign, p);
    }
    matrix
}

fn classifier_matrix(hidden: usize, p: u64) -> Vec<Vec<u64>> {
    let mut matrix = vec![vec![0; hidden]; 2];
    for idx in 0..hidden {
        let sign = if idx % 2 == 0 { 1 } else { -1 };
        matrix[idx % 2][idx] = signed_mod(sign, p);
    }
    matrix
}

fn bootstrap_matrix(p: u64) -> Vec<Vec<u64>> {
    vec![
        vec![1, 1, 0, 0],
        vec![0, 1, 1, 0],
        vec![0, 0, 1, 1],
        vec![1, 0, 0, p - 1],
    ]
}

fn private_linear_config(
    mode: ExperimentMode,
    fixed: FixedPointConfig,
    input_dim: usize,
    output_dim: usize,
) -> PrivateLinearMapConfig {
    PrivateLinearMapConfig {
        input_dim,
        output_dim,
        poly_degree: match mode {
            ExperimentMode::Smoke => 8,
            ExperimentMode::Full => 16,
        },
        q_modulus: match mode {
            ExperimentMode::Smoke => fixed.modulus * 65_537,
            ExperimentMode::Full => fixed.modulus * (1u64 << 32),
        },
        p_modulus: fixed.modulus,
        lwe_rows_n: match mode {
            ExperimentMode::Smoke => input_dim,
            ExperimentMode::Full => input_dim * 2,
        },
        gadget_cols_t: 6,
        setup_noise_bound: 0,
        query_noise_bound: 0,
    }
}

fn signed_mod(value: i64, p: u64) -> u64 {
    let p_i = p as i64;
    let mut value = value % p_i;
    if value < 0 {
        value += p_i;
    }
    value as u64
}

fn add_signed_mod(base: u64, value: i64, p: u64) -> u64 {
    signed_mod(base as i64 + value, p)
}

async fn tcp_loopback_stream_pair() -> Result<(Box<dyn Stream>, Box<dyn Stream>), RuntimeError> {
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
    let client_stream = client_conn
        .open_stream()
        .await
        .map_err(|err| RuntimeError::Network(err.to_string()))?;
    let server_stream = accept_task
        .await
        .map_err(|err| RuntimeError::Network(err.to_string()))??;
    Ok((client_stream, server_stream))
}

fn runtime_sessions_from_streams(
    client_stream: Box<dyn Stream>,
    server_stream: Box<dyn Stream>,
    session_id: SessionId,
) -> Result<(RuntimeSession, RuntimeSession), RuntimeError> {
    let wire = WireContext::default();
    let cfg = RuntimeConfig::two_party_inference();
    let mut client_mesh = RuntimeMesh::new(PartyId(0));
    let mut server_mesh = RuntimeMesh::new(PartyId(1));
    client_mesh.insert_peer(PartyId(1), RuntimeChannel::new(client_stream, wire.clone()))?;
    server_mesh.insert_peer(PartyId(0), RuntimeChannel::new(server_stream, wire.clone()))?;
    let client_ctx = RuntimeContext::new(cfg.clone(), PartyId(0), session_id, wire.clone())?;
    let server_ctx = RuntimeContext::new(cfg, PartyId(1), session_id, wire)?;
    Ok((
        RuntimeSession::with_mesh(client_ctx, client_mesh)?,
        RuntimeSession::with_mesh(server_ctx, server_mesh)?,
    ))
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

    let mut keygen = HssKeyGenerator::new(context.clone(), SecureRng::from_seed([51u8; 32]));
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
    let mut keygen = HssKeyGenerator::new(context.clone(), SecureRng::from_seed([52u8; 32]));
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
