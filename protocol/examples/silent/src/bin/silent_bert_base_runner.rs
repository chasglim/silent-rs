use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use silent_hss::{HssContext, HssKeyGenerator};
use silent_math::arith::add_mod;
use silent_math::modulus::Modulus;
use silent_math::numth::{first_ntt_prime_with_bits, next_ntt_prime};
use silent_math::rns::RnsBase;
use silent_math::rns_tool::RnsToolConfig;
use silent_net::frame::MessageKind;
use silent_net::transport::tcp::TcpTransport;
use silent_net::transport::{Connection, Stream};
use silent_operators::conversion::ShareConverter;
use silent_operators::error::OperatorError;
use silent_operators::fixedpoint::FixedPointConfig;
use silent_operators::hss_bridge::context_from_runtime;
use silent_operators::hss_slots::HssSlotEngine;
use silent_operators::linear_bridge::LinearMapQShares;
use silent_operators::lookup::{PrivateLookup, PrivateLookupConfig};
use silent_operators::nonlinear::{
    BinadeLayerNormConfig, DyadicOfPmpePolynomialConfig, FastGeluConfig, GeluConfig,
    LayerNormConfig, NonlinearOps, NonlinearProfile, OfPmpeOfflineMode, OfPmpePolynomialConfig,
    OfPmpeProfile, PackedRlweAheTripleEngine, RangeSoftmaxConfig, RnsCorrelationSource,
    RnsCrossTermOleTaylorSource, RnsDyadicDomain, RnsHssCorrelationSource,
    RnsHybridEngineeringCorrelationSource, RnsRlweAheCorrelationSource,
    RnsRlweAheCrossTermOleTaylorSource, RnsScaledShareTensor, ScaleSemantic, SoftmaxConfig,
};
use silent_operators::packed_linear_map::{
    PackedLinearMap, PackedLinearMapConfig, PackedLinearMapCrsCache,
};
use silent_operators::shares::AdditiveShares;
use silent_params::{
    CorrectnessMarginBits, DistributionType, HssParams, MaxLinearTerms, MaxRmultDepth, ModulusBits,
    PlaintextModulus, ReconstructionBoundBits, RegisteredParameterSet, RingDim, RingParams,
    RlweParams, ScaleBits, SecurityLevel, ShareModulusBits, find_preset,
};
use silent_protocol_runtime::{
    PartyId, RuntimeChannel, RuntimeConfig, RuntimeContext, RuntimeError, RuntimeMesh,
    RuntimeSession, RuntimeStats, SessionId, TaskId, WireContext,
};
use silent_rlwe::EncryptionParams;
use silent_utils::rng::SecureRng;
use std::time::{Duration, Instant};
use tokio::runtime::{Builder, Runtime};

const BERT_LAYERS: usize = 12;
const BERT_SEQ: usize = 128;
const BERT_HIDDEN: usize = 768;
const BERT_HEADS: usize = 12;
const BERT_HEAD_DIM: usize = 64;
const BERT_FFN: usize = 3072;
const HSS_PLAINTEXT_MODULUS: u64 = 65_537;
const DEFAULT_FIXED_SCALE: u64 = 256;
const PAPER_BERT_BASE_COMM_GB: f64 = 4.88;
const LAN_RTT_MS: f64 = 0.5;
const WAN_RTT_MS: f64 = 4.0;
const LAN_BW_MBPS: f64 = 1000.0;
const WAN_BW_MBPS: f64 = 400.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = CliOptions::parse()?;
    if !opts.no_header {
        println!("section,case,iterations,total_ms,avg_ms,extra");
    }

    if opts.meanbeta_sweep {
        let mut rng = StdRng::seed_from_u64(0x5F07_5AEE);
        return run_meanbeta_softmax_plaintext_sweep(&opts, &mut rng);
    }
    if opts.direction_validation {
        let fixed = FixedPointConfig::new(HSS_PLAINTEXT_MODULUS, opts.scale)?;
        let mut rng = StdRng::seed_from_u64(0xD1EC_7100);
        return run_silent_direction_validation(&opts, fixed, &mut rng);
    }

    if opts.nonlinear_only && opts.pmpe_only && opts.pmpe_rns {
        let fixed = FixedPointConfig::new(HSS_PLAINTEXT_MODULUS, opts.scale)?;
        let mut rng = StdRng::seed_from_u64(0xBEE7_BA5E);
        if opts.pmpe_bert_shapes {
            let gelu_points = BERT_LAYERS * BERT_SEQ * BERT_FFN;
            let softmax_exp_points = BERT_LAYERS * BERT_HEADS * BERT_SEQ * BERT_SEQ;
            let layernorm_square_points = 2 * BERT_LAYERS * BERT_SEQ * BERT_HIDDEN;
            return run_rns_dyadic_of_pmpe_bert_shape_benches(
                &opts,
                fixed,
                &mut rng,
                gelu_points,
                softmax_exp_points,
                layernorm_square_points,
            );
        }
        return run_rns_dyadic_of_pmpe_benches(&opts, fixed, &mut rng);
    }

    let setup_start = Instant::now();
    let hss = match opts.hss_degree {
        Some(degree) => build_hss_degree_engine(degree)?,
        None => build_hss_preset_engine("current-hss-v1")?,
    };
    let p = hss.context().plain_modulus();
    let simd_chunk = hss.context().degree();
    let gelu_lanes = opts.gelu_lanes.unwrap_or(simd_chunk);
    if gelu_lanes == 0 || gelu_lanes > simd_chunk {
        return Err(format!("--gelu-lanes must be in 1..={simd_chunk}").into());
    }
    let softmax_batch_rows = usize::max(1, simd_chunk / BERT_SEQ);
    let layernorm_batch_rows = usize::max(1, simd_chunk / BERT_HIDDEN);
    let fixed = FixedPointConfig::new(p, opts.scale)?;
    let lookup = PrivateLookup::new(PrivateLookupConfig {
        output_modulus: p,
        q_modulus_bits: 62,
        matrix_rows: 16,
        gadget_cols_t: 8,
        max_domain: p as usize,
    })?;
    let nonlinear = NonlinearOps::new(fixed, &lookup, &hss)?;
    let mut public_param_rng = StdRng::seed_from_u64(0xC5A5_0001);
    lookup.prewarm_crs(p as usize, p, &mut public_param_rng)?;
    lookup.prewarm_crs(129, 129, &mut public_param_rng)?;
    lookup.prewarm_crs(129, p, &mut public_param_rng)?;
    let setup_elapsed = setup_start.elapsed();
    print_row(
        "bert_probe",
        "setup_current_hss_v1_nonlinear",
        1,
        setup_elapsed,
        &format!(
            "p={p}|scale={}|hss_degree={simd_chunk}|lookup_max_domain={}|lookup_crs_entries={}",
            fixed.scale,
            lookup.config().max_domain,
            lookup.crs_cache().len()?
        ),
    );

    let mut rng = StdRng::seed_from_u64(0xBEE7_BA5E);
    let packed_crs_cache = PackedLinearMapCrsCache::new();
    if opts.nonlinear_only {
        if opts.pmpe_only {
            run_of_pmpe_benches(&opts, fixed, &nonlinear, &mut rng)?;
            return Ok(());
        }
        let _ = run_nonlinear_benches(
            &opts,
            fixed,
            simd_chunk,
            softmax_batch_rows,
            gelu_lanes,
            layernorm_batch_rows,
            &hss,
            &nonlinear,
            &mut rng,
        )?;
        return Ok(());
    }

    let linear_qkv = bench_public_matmul_matrix(
        opts.iterations,
        BERT_SEQ,
        BERT_HIDDEN,
        BERT_HIDDEN * 3,
        fixed,
        &mut rng,
        "share_public_matmul_matrix_128x768_by_768x2304",
    )?;
    let linear_768 = bench_public_matmul_matrix(
        opts.iterations,
        BERT_SEQ,
        BERT_HIDDEN,
        BERT_HIDDEN,
        fixed,
        &mut rng,
        "share_public_matmul_matrix_128x768_by_768x768",
    )?;
    let linear_3072 = bench_public_matmul_matrix(
        opts.iterations,
        BERT_SEQ,
        BERT_HIDDEN,
        BERT_FFN,
        fixed,
        &mut rng,
        "share_public_matmul_matrix_128x768_by_768x3072",
    )?;
    let linear_3072_to_768 = bench_public_matmul_matrix(
        opts.iterations,
        BERT_SEQ,
        BERT_FFN,
        BERT_HIDDEN,
        fixed,
        &mut rng,
        "share_public_matmul_matrix_128x3072_by_3072x768",
    )?;
    let packed_linear_768 = if opts.packed_linear {
        Some(bench_packed_private_linear_map_matrix(
            opts.iterations,
            BERT_SEQ,
            BERT_HIDDEN,
            BERT_HIDDEN,
            fixed,
            opts.packed_linear_degree.unwrap_or(32_768),
            &packed_crs_cache,
            &mut rng,
            "packed_private_linear_map_128x768_by_768x768_online",
        )?)
    } else {
        None
    };
    let packed_attention = if opts.packed_linear {
        Some(bench_packed_shared_matmul_matrix(
            opts.iterations,
            BERT_SEQ,
            BERT_HIDDEN,
            BERT_HEADS * BERT_SEQ,
            fixed,
            opts.packed_linear_degree.unwrap_or(2048),
            &packed_crs_cache,
            &mut rng,
            "packed_shared_matmul_attention_128x768_by_768x1536",
        )?)
    } else {
        None
    };
    let packed_context = if opts.packed_linear {
        Some(bench_packed_shared_matmul_matrix(
            opts.iterations,
            BERT_SEQ,
            BERT_HEADS * BERT_SEQ,
            BERT_HIDDEN,
            fixed,
            opts.packed_linear_degree.unwrap_or(2048),
            &packed_crs_cache,
            &mut rng,
            "packed_shared_matmul_context_128x1536_by_1536x768",
        )?)
    } else {
        None
    };
    let packed_attention_head = if opts.packed_linear {
        Some(bench_packed_shared_matmul_matrix(
            opts.iterations,
            BERT_SEQ,
            BERT_HEAD_DIM,
            BERT_SEQ,
            fixed,
            opts.packed_linear_degree.unwrap_or(2048),
            &packed_crs_cache,
            &mut rng,
            "packed_shared_matmul_attention_head_128x64_by_64x128",
        )?)
    } else {
        None
    };
    let packed_context_head = if opts.packed_linear {
        Some(bench_packed_shared_matmul_matrix(
            opts.iterations,
            BERT_SEQ,
            BERT_SEQ,
            BERT_HEAD_DIM,
            fixed,
            opts.packed_linear_degree.unwrap_or(2048),
            &packed_crs_cache,
            &mut rng,
            "packed_shared_matmul_context_head_128x128_by_128x64",
        )?)
    } else {
        None
    };
    if opts.linear_only {
        return Ok(());
    }
    let nonlinear_benches = run_nonlinear_benches(
        &opts,
        fixed,
        simd_chunk,
        softmax_batch_rows,
        gelu_lanes,
        layernorm_batch_rows,
        &hss,
        &nonlinear,
        &mut rng,
    )?;
    let bridge = if opts.skip_bridge {
        BenchResult {
            avg_ms: 0.0,
            bytes_sent: 0,
            bytes_received: 0,
        }
    } else {
        bench_q_to_p_bridge_bert_input(opts.iterations, fixed, &mut rng)?
    };

    print_bert_base_scaled_model(
        linear_qkv,
        linear_768,
        linear_3072,
        linear_3072_to_768,
        nonlinear_benches.hss_mul_raw_4096,
        nonlinear_benches.hss_mul_4096,
        nonlinear_benches.attention_dot_64,
        nonlinear_benches.context_dot_128,
        nonlinear_benches.softmax_128,
        nonlinear_benches.softmax_32x128,
        nonlinear_benches.softmax_poly_32x128,
        nonlinear_benches.softmax_approx_32x128,
        nonlinear_benches.softmax_specialized_32x128,
        nonlinear_benches.layernorm_768,
        nonlinear_benches.layernorm_5x768,
        nonlinear_benches.layernorm_approx_5x768,
        nonlinear_benches.layernorm_specialized_5x768,
        nonlinear_benches.gelu,
        nonlinear_benches.gelu_specialized,
        gelu_lanes,
        simd_chunk,
        softmax_batch_rows,
        layernorm_batch_rows,
        bridge.avg_ms,
        bridge.bytes_sent + bridge.bytes_received,
        packed_linear_768,
        packed_attention,
        packed_context,
        packed_attention_head,
        packed_context_head,
    );

    Ok(())
}

#[derive(Clone, Copy)]
struct CliOptions {
    iterations: usize,
    gelu_lanes: Option<usize>,
    hss_degree: Option<usize>,
    scale: u64,
    linear_only: bool,
    nonlinear_only: bool,
    skip_bridge: bool,
    packed_linear: bool,
    packed_linear_degree: Option<usize>,
    poly_softmax: bool,
    poly_softmax_squarings: usize,
    specialized_nonlinear: bool,
    approx_trunc: bool,
    pmpe_nonlinear: bool,
    pmpe_only: bool,
    pmpe_lanes: usize,
    pmpe_runtime_net: bool,
    pmpe_streaming: bool,
    pmpe_chunk_lanes: usize,
    pmpe_bert_shapes: bool,
    pmpe_dyadic: bool,
    pmpe_rns: bool,
    meanbeta_sweep: bool,
    direction_validation: bool,
    trusted_debug_offline: bool,
    highscale_hss_degree: usize,
    highscale_hss_slots: usize,
    packed_ahe_probe_degree: usize,
    packed_ahe_probe_lanes: usize,
    packed_ahe_probe_q_count: usize,
    packed_ahe_probe_q_bits: u32,
    packed_ahe_probe_plain_bits: u32,
    run_hss_wrapper_probe: bool,
    no_header: bool,
}

impl CliOptions {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut iterations = std::env::var("ITERATIONS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        let mut gelu_lanes = std::env::var("SILENT_BERT_GELU_LANES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let mut hss_degree = std::env::var("SILENT_BERT_HSS_DEGREE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let mut scale = std::env::var("SILENT_BERT_SCALE")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_FIXED_SCALE);
        let mut linear_only = std::env::var("SILENT_BERT_LINEAR_ONLY")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut nonlinear_only = std::env::var("SILENT_BERT_NONLINEAR_ONLY")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut skip_bridge = std::env::var("SILENT_BERT_SKIP_BRIDGE")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut packed_linear = std::env::var("SILENT_BERT_PACKED_LINEAR")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut packed_linear_degree = std::env::var("SILENT_BERT_PACKED_LINEAR_DEGREE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let mut poly_softmax = std::env::var("SILENT_BERT_POLY_SOFTMAX")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut poly_softmax_squarings = std::env::var("SILENT_BERT_POLY_SOFTMAX_SQUARINGS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(7);
        let mut specialized_nonlinear = std::env::var("SILENT_BERT_SPECIALIZED_NONLINEAR")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut approx_trunc = std::env::var("SILENT_BERT_APPROX_TRUNC")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut pmpe_nonlinear = std::env::var("SILENT_BERT_PMPE_NONLINEAR")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut pmpe_only = std::env::var("SILENT_BERT_PMPE_ONLY")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut pmpe_lanes = std::env::var("SILENT_BERT_PMPE_LANES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1 << 20);
        let mut pmpe_runtime_net = std::env::var("SILENT_BERT_PMPE_RUNTIME_NET")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut pmpe_streaming = std::env::var("SILENT_BERT_PMPE_STREAMING")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut pmpe_chunk_lanes = std::env::var("SILENT_BERT_PMPE_CHUNK_LANES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1 << 20);
        let mut pmpe_bert_shapes = std::env::var("SILENT_BERT_PMPE_BERT_SHAPES")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut pmpe_dyadic = std::env::var("SILENT_BERT_PMPE_DYADIC")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut pmpe_rns = std::env::var("SILENT_BERT_PMPE_RNS")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut meanbeta_sweep = std::env::var("SILENT_BERT_MEANBETA_SWEEP")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut direction_validation = std::env::var("SILENT_DIRECTION_VALIDATION")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut trusted_debug_offline = std::env::var("SILENT_TRUSTED_DEBUG_OFFLINE")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut highscale_hss_degree = std::env::var("SILENT_HIGHSCALE_HSS_DEGREE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8);
        let mut highscale_hss_slots = std::env::var("SILENT_HIGHSCALE_HSS_SLOTS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(6);
        let mut packed_ahe_probe_degree = std::env::var("SILENT_PACKED_AHE_PROBE_DEGREE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(128);
        let mut packed_ahe_probe_lanes = std::env::var("SILENT_PACKED_AHE_PROBE_LANES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(128);
        let mut packed_ahe_probe_q_count = std::env::var("SILENT_PACKED_AHE_PROBE_Q_COUNT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8);
        let mut packed_ahe_probe_q_bits = std::env::var("SILENT_PACKED_AHE_PROBE_Q_BITS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(50);
        let mut packed_ahe_probe_plain_bits = std::env::var("SILENT_PACKED_AHE_PROBE_PLAIN_BITS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(61);
        let mut run_hss_wrapper_probe = std::env::var("SILENT_RUN_HSS_WRAPPER_PROBE")
            .ok()
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut no_header = false;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--iterations" | "-n" => {
                    iterations = args
                        .next()
                        .ok_or("--iterations requires a positive integer")?
                        .parse::<usize>()?;
                }
                "--gelu-lanes" => {
                    gelu_lanes = args
                        .next()
                        .ok_or("--gelu-lanes requires a positive integer")?
                        .parse::<usize>()
                        .map(Some)?;
                }
                "--hss-degree" => {
                    hss_degree = args
                        .next()
                        .ok_or("--hss-degree requires a positive integer")?
                        .parse::<usize>()
                        .map(Some)?;
                }
                "--scale" => {
                    scale = args
                        .next()
                        .ok_or("--scale requires a positive integer")?
                        .parse::<u64>()?;
                }
                "--no-header" => no_header = true,
                "--linear-only" => linear_only = true,
                "--nonlinear-only" => nonlinear_only = true,
                "--skip-bridge" => skip_bridge = true,
                "--packed-linear" => packed_linear = true,
                "--packed-linear-degree" => {
                    packed_linear_degree = args
                        .next()
                        .ok_or("--packed-linear-degree requires a positive integer")?
                        .parse::<usize>()
                        .map(Some)?;
                }
                "--poly-softmax" => poly_softmax = true,
                "--poly-softmax-squarings" => {
                    poly_softmax = true;
                    poly_softmax_squarings = args
                        .next()
                        .ok_or("--poly-softmax-squarings requires a positive integer")?
                        .parse::<usize>()?;
                }
                "--specialized-nonlinear" => specialized_nonlinear = true,
                "--approx-trunc" => approx_trunc = true,
                "--pmpe-nonlinear" => pmpe_nonlinear = true,
                "--pmpe-only" => {
                    pmpe_only = true;
                    pmpe_nonlinear = true;
                    nonlinear_only = true;
                }
                "--pmpe-lanes" => {
                    pmpe_lanes = args
                        .next()
                        .ok_or("--pmpe-lanes requires a positive integer")?
                        .parse::<usize>()?;
                }
                "--pmpe-runtime-net" => pmpe_runtime_net = true,
                "--pmpe-streaming" => pmpe_streaming = true,
                "--pmpe-chunk-lanes" => {
                    pmpe_streaming = true;
                    pmpe_chunk_lanes = args
                        .next()
                        .ok_or("--pmpe-chunk-lanes requires a positive integer")?
                        .parse::<usize>()?;
                }
                "--pmpe-bert-shapes" => pmpe_bert_shapes = true,
                "--pmpe-dyadic" => pmpe_dyadic = true,
                "--pmpe-rns" => {
                    pmpe_rns = true;
                    pmpe_dyadic = true;
                }
                "--meanbeta-softmax-sweep" => meanbeta_sweep = true,
                "--silent-direction-validation" => direction_validation = true,
                "--trusted-debug-offline" => trusted_debug_offline = true,
                "--highscale-hss-degree" => {
                    highscale_hss_degree = args
                        .next()
                        .ok_or("--highscale-hss-degree requires a positive integer")?
                        .parse::<usize>()?;
                }
                "--highscale-hss-slots" => {
                    highscale_hss_slots = args
                        .next()
                        .ok_or("--highscale-hss-slots requires a positive integer")?
                        .parse::<usize>()?;
                }
                "--packed-ahe-probe-degree" => {
                    packed_ahe_probe_degree = args
                        .next()
                        .ok_or("--packed-ahe-probe-degree requires a positive integer")?
                        .parse::<usize>()?;
                }
                "--packed-ahe-probe-lanes" => {
                    packed_ahe_probe_lanes = args
                        .next()
                        .ok_or("--packed-ahe-probe-lanes requires a positive integer")?
                        .parse::<usize>()?;
                }
                "--packed-ahe-probe-q-count" => {
                    packed_ahe_probe_q_count = args
                        .next()
                        .ok_or("--packed-ahe-probe-q-count requires a positive integer")?
                        .parse::<usize>()?;
                }
                "--packed-ahe-probe-q-bits" => {
                    packed_ahe_probe_q_bits = args
                        .next()
                        .ok_or("--packed-ahe-probe-q-bits requires a positive integer")?
                        .parse::<u32>()?;
                }
                "--packed-ahe-probe-plain-bits" => {
                    packed_ahe_probe_plain_bits = args
                        .next()
                        .ok_or("--packed-ahe-probe-plain-bits requires a positive integer")?
                        .parse::<u32>()?;
                }
                "--run-hss-wrapper-probe" => run_hss_wrapper_probe = true,
                value => iterations = value.parse::<usize>()?,
            }
        }
        Ok(Self {
            iterations: iterations.max(1),
            gelu_lanes,
            hss_degree,
            scale,
            linear_only,
            nonlinear_only,
            skip_bridge,
            packed_linear,
            packed_linear_degree,
            poly_softmax,
            poly_softmax_squarings,
            specialized_nonlinear,
            approx_trunc,
            pmpe_nonlinear,
            pmpe_only,
            pmpe_lanes,
            pmpe_runtime_net,
            pmpe_streaming,
            pmpe_chunk_lanes,
            pmpe_bert_shapes,
            pmpe_dyadic,
            pmpe_rns,
            meanbeta_sweep,
            direction_validation,
            trusted_debug_offline,
            highscale_hss_degree: highscale_hss_degree.max(1),
            highscale_hss_slots: highscale_hss_slots.max(1),
            packed_ahe_probe_degree: packed_ahe_probe_degree.max(1),
            packed_ahe_probe_lanes: packed_ahe_probe_lanes.max(1),
            packed_ahe_probe_q_count: packed_ahe_probe_q_count.max(1),
            packed_ahe_probe_q_bits: packed_ahe_probe_q_bits.max(1),
            packed_ahe_probe_plain_bits: packed_ahe_probe_plain_bits.max(1),
            run_hss_wrapper_probe,
            no_header,
        })
    }
}

struct NonlinearBenchSet {
    hss_mul_raw_4096: f64,
    hss_mul_4096: f64,
    attention_dot_64: f64,
    context_dot_128: f64,
    softmax_128: f64,
    softmax_32x128: f64,
    softmax_poly_32x128: Option<f64>,
    softmax_approx_32x128: Option<f64>,
    softmax_specialized_32x128: Option<f64>,
    layernorm_768: f64,
    layernorm_5x768: f64,
    layernorm_approx_5x768: Option<f64>,
    layernorm_specialized_5x768: Option<f64>,
    gelu: f64,
    gelu_specialized: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct MeanBetaSweepMetrics {
    rows: usize,
    cols: usize,
    beta: f64,
    domain_b: f64,
    degree: usize,
    avg_l1: f64,
    avg_kl: f64,
    max_l1: f64,
    top1_agreement: f64,
    out_hi_rate: f64,
    out_lo_rate: f64,
    negative_rate: f64,
    bad_rows: usize,
    poly_min_observed: f64,
    poly_max_observed: f64,
}

#[derive(Clone, Debug)]
struct MeanBetaSoftmaxPlan {
    beta_source: &'static str,
    metrics: MeanBetaSweepMetrics,
    coeffs: Vec<f64>,
    coeff_checksum: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct DirectionOperatorStats {
    offline_us: u64,
    online_us: u64,
    offline_bytes: u64,
    online_bytes: u64,
    pack_size_bytes: u64,
    peak_pack_resident_bytes: u64,
    oneflow_phases: usize,
    request_response_rounds: usize,
    online_secret_secret_mul: usize,
    online_trunc: usize,
    trusted_debug_offline_us: u64,
    secure_offline_us: u64,
}

impl DirectionOperatorStats {
    fn from_profile(profile: OfPmpeProfile) -> Self {
        Self {
            offline_us: profile.offline_us,
            online_us: profile.online_us,
            offline_bytes: profile.offline_bytes,
            online_bytes: profile.online_bytes,
            pack_size_bytes: profile.pack_size_bytes,
            peak_pack_resident_bytes: profile.peak_pack_resident_bytes,
            oneflow_phases: profile.num_oneflow_phases,
            request_response_rounds: profile.num_request_response_rounds,
            online_secret_secret_mul: profile.online_secret_secret_mul,
            online_trunc: profile.online_trunc,
            trusted_debug_offline_us: profile.trusted_debug_offline_us,
            secure_offline_us: profile.secure_offline_us,
        }
    }

    fn add_assign(&mut self, rhs: Self) {
        self.offline_us = self.offline_us.saturating_add(rhs.offline_us);
        self.online_us = self.online_us.saturating_add(rhs.online_us);
        self.offline_bytes = self.offline_bytes.saturating_add(rhs.offline_bytes);
        self.online_bytes = self.online_bytes.saturating_add(rhs.online_bytes);
        self.pack_size_bytes = self.pack_size_bytes.saturating_add(rhs.pack_size_bytes);
        self.peak_pack_resident_bytes = self
            .peak_pack_resident_bytes
            .max(rhs.peak_pack_resident_bytes);
        self.oneflow_phases = self.oneflow_phases.saturating_add(rhs.oneflow_phases);
        self.request_response_rounds = self
            .request_response_rounds
            .saturating_add(rhs.request_response_rounds);
        self.online_secret_secret_mul = self
            .online_secret_secret_mul
            .saturating_add(rhs.online_secret_secret_mul);
        self.online_trunc = self.online_trunc.saturating_add(rhs.online_trunc);
        self.trusted_debug_offline_us = self
            .trusted_debug_offline_us
            .saturating_add(rhs.trusted_debug_offline_us);
        self.secure_offline_us = self.secure_offline_us.saturating_add(rhs.secure_offline_us);
    }

    fn scale_to(self, source_slots: usize, target_slots: usize) -> Self {
        if source_slots == 0 || source_slots == target_slots {
            return self;
        }
        let scale_u64 = |value: u64| -> u64 {
            ((value as u128).saturating_mul(target_slots as u128) / source_slots as u128)
                .min(u64::MAX as u128) as u64
        };
        let scale_usize = |value: usize| -> usize {
            ((value as u128).saturating_mul(target_slots as u128) / source_slots as u128)
                .min(usize::MAX as u128) as usize
        };
        Self {
            offline_us: scale_u64(self.offline_us),
            online_us: scale_u64(self.online_us),
            offline_bytes: scale_u64(self.offline_bytes),
            online_bytes: scale_u64(self.online_bytes),
            pack_size_bytes: scale_u64(self.pack_size_bytes),
            peak_pack_resident_bytes: self.peak_pack_resident_bytes,
            oneflow_phases: scale_usize(self.oneflow_phases),
            request_response_rounds: scale_usize(self.request_response_rounds),
            online_secret_secret_mul: scale_usize(self.online_secret_secret_mul),
            online_trunc: scale_usize(self.online_trunc),
            trusted_debug_offline_us: scale_u64(self.trusted_debug_offline_us),
            secure_offline_us: scale_u64(self.secure_offline_us),
        }
    }
}

fn run_meanbeta_softmax_plaintext_sweep(
    opts: &CliOptions,
    rng: &mut StdRng,
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = usize::max(32, opts.pmpe_lanes / BERT_SEQ);
    let cols = BERT_SEQ;
    let start = Instant::now();
    let scores = generate_attention_score_rows(rows, cols, rng);
    let beta_stats = max_minus_mean_percentiles(&scores, rows, cols);
    let beta_candidates = [
        ("zero", 0.0),
        ("p90", beta_stats[0]),
        ("p95", beta_stats[1]),
        ("p99", beta_stats[2]),
        ("p999", beta_stats[3]),
    ];
    print_row(
        "plaintext_sweep",
        "meanbeta_softmax_calibration",
        1,
        start.elapsed(),
        &format!(
            "rows={rows}|cols={cols}|score_model=deterministic_attention_mixture|beta_p90={:.6}|beta_p95={:.6}|beta_p99={:.6}|beta_p999={:.6}",
            beta_stats[0], beta_stats[1], beta_stats[2], beta_stats[3]
        ),
    );

    let mut best_valid: Option<(MeanBetaSweepMetrics, &'static str, u64)> = None;
    for &(beta_name, beta) in &beta_candidates {
        for domain_b in [6.0, 8.0, 10.0, 12.0] {
            for degree in 3..=7 {
                let iter_start = Instant::now();
                let coeffs = fit_exp_polynomial_on_negative_interval(degree, domain_b, 512)?;
                let metrics = evaluate_meanbeta_softmax_poly(
                    &scores, rows, cols, beta, domain_b, degree, &coeffs,
                )?;
                let coeff_checksum = coeffs
                    .iter()
                    .fold(0u64, |acc, coeff| acc.wrapping_add(coeff.to_bits()));
                print_row(
                    "plaintext_sweep",
                    &format!("meanbeta_{beta_name}_B{domain_b:.0}_deg{degree}"),
                    1,
                    iter_start.elapsed(),
                    &format!(
                        "rows={}|cols={}|beta={:.6}|domain=[-{:.1};0]|degree={}|avg_l1={:.6e}|avg_kl={:.6e}|max_l1={:.6e}|top1_agreement={:.6}|out_hi_rate={:.6e}|out_lo_rate={:.6e}|negative_rate={:.6e}|bad_rows={}|poly_min_observed={:.6e}|poly_max_observed={:.6e}|coeff_checksum={coeff_checksum}",
                        metrics.rows,
                        metrics.cols,
                        metrics.beta,
                        metrics.domain_b,
                        metrics.degree,
                        metrics.avg_l1,
                        metrics.avg_kl,
                        metrics.max_l1,
                        metrics.top1_agreement,
                        metrics.out_hi_rate,
                        metrics.out_lo_rate,
                        metrics.negative_rate,
                        metrics.bad_rows,
                        metrics.poly_min_observed,
                        metrics.poly_max_observed,
                    ),
                );
                let valid = metrics.negative_rate == 0.0
                    && metrics.out_hi_rate <= 1e-3
                    && metrics.out_lo_rate <= 1e-2
                    && metrics.bad_rows == 0;
                if valid
                    && best_valid
                        .map(|(best, _, _)| metrics.avg_kl < best.avg_kl)
                        .unwrap_or(true)
                {
                    best_valid = Some((metrics, beta_name, coeff_checksum));
                }
            }
        }
    }
    if let Some((metrics, beta_name, coeff_checksum)) = best_valid {
        print_row(
            "plaintext_sweep",
            "meanbeta_softmax_best_valid",
            1,
            start.elapsed(),
            &format!(
                "beta_source={beta_name}|rows={}|cols={}|beta={:.6}|domain=[-{:.1};0]|degree={}|avg_l1={:.6e}|avg_kl={:.6e}|max_l1={:.6e}|top1_agreement={:.6}|out_hi_rate={:.6e}|out_lo_rate={:.6e}|negative_rate={:.6e}|bad_rows={}|poly_min_observed={:.6e}|poly_max_observed={:.6e}|coeff_checksum={coeff_checksum}",
                metrics.rows,
                metrics.cols,
                metrics.beta,
                metrics.domain_b,
                metrics.degree,
                metrics.avg_l1,
                metrics.avg_kl,
                metrics.max_l1,
                metrics.top1_agreement,
                metrics.out_hi_rate,
                metrics.out_lo_rate,
                metrics.negative_rate,
                metrics.bad_rows,
                metrics.poly_min_observed,
                metrics.poly_max_observed,
            ),
        );
    }
    Ok(())
}

fn run_silent_direction_validation(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    rng: &mut StdRng,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let validation_rows = if opts.pmpe_bert_shapes {
        BERT_LAYERS * BERT_HEADS * BERT_SEQ
    } else {
        usize::max(1024, opts.pmpe_lanes / BERT_SEQ)
    };
    let f = frac_bits_from_scale(fixed.scale)?;
    let shifted_frac_bits = f + BERT_SEQ.trailing_zeros();
    let domain4 = RnsDyadicDomain::limbs_61bit(4)?;
    let meanbeta_plan = select_meanbeta_softmax_plan(
        validation_rows,
        BERT_SEQ,
        Some((shifted_frac_bits, f, domain4.modulus_bits())),
        rng,
    )?;
    let plaintext_pass = meanbeta_plan.metrics.avg_kl <= 1e-4
        && meanbeta_plan.metrics.avg_l1 <= 1.5e-2
        && meanbeta_plan.metrics.negative_rate == 0.0
        && meanbeta_plan.metrics.out_hi_rate <= 1e-3
        && meanbeta_plan.metrics.out_lo_rate <= 1e-2
        && meanbeta_plan.metrics.top1_agreement >= 0.999
        && meanbeta_plan.metrics.bad_rows == 0;
    print_row(
        "direction_validation",
        "gate1_plaintext_meanbeta_softmax",
        1,
        start.elapsed(),
        &format!(
            "status={}|beta_source={}|rows={}|cols={}|beta={:.6}|domain=[-{:.1};0]|degree={}|avg_l1={:.6e}|avg_kl={:.6e}|max_l1={:.6e}|top1_agreement={:.6}|out_hi_rate={:.6e}|out_lo_rate={:.6e}|negative_rate={:.6e}|bad_rows={}|coeff_checksum={}",
            pass_fail(plaintext_pass),
            meanbeta_plan.beta_source,
            meanbeta_plan.metrics.rows,
            meanbeta_plan.metrics.cols,
            meanbeta_plan.metrics.beta,
            meanbeta_plan.metrics.domain_b,
            meanbeta_plan.metrics.degree,
            meanbeta_plan.metrics.avg_l1,
            meanbeta_plan.metrics.avg_kl,
            meanbeta_plan.metrics.max_l1,
            meanbeta_plan.metrics.top1_agreement,
            meanbeta_plan.metrics.out_hi_rate,
            meanbeta_plan.metrics.out_lo_rate,
            meanbeta_plan.metrics.negative_rate,
            meanbeta_plan.metrics.bad_rows,
            meanbeta_plan.coeff_checksum,
        ),
    );

    let operator_start = Instant::now();
    let offline_backend = if opts.trusted_debug_offline {
        "trusted_debug"
    } else {
        "rns_hybrid_ideal_taylor_beaver_coin_tossed_rescale"
    };
    let chunk = opts.pmpe_chunk_lanes.max(1);
    let operator_slots = if opts.pmpe_bert_shapes {
        BERT_LAYERS * BERT_SEQ * BERT_FFN
    } else {
        opts.pmpe_lanes.max(1)
    };
    let gelu_target_slots = BERT_LAYERS * BERT_SEQ * BERT_FFN;
    let softmax_target_slots = BERT_LAYERS * BERT_HEADS * BERT_SEQ * BERT_SEQ;
    let layernorm_target_slots = 2 * BERT_LAYERS * BERT_SEQ * BERT_HIDDEN;
    let softmax_rows_source = if opts.pmpe_bert_shapes {
        BERT_LAYERS * BERT_HEADS * BERT_SEQ
    } else {
        usize::max(1, operator_slots / BERT_SEQ)
    };
    let softmax_source_slots = softmax_rows_source * BERT_SEQ;
    let activation_bound_bits = f.saturating_add(3);
    let domain2 = RnsDyadicDomain::two_limb_61bit()?;

    let gelu_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::gelu_global_degree6_probe(chunk).coeffs,
        f,
        f,
        56,
        domain2.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modulus_margin_bits(10);
    let gelu_input = rns_shared_vector(operator_slots, fixed, &domain2, rng)?;
    let gelu = if opts.trusted_debug_offline {
        domain2.dyadic_of_pmpe_profiled(&gelu_input, &gelu_cfg, rng)?
    } else {
        let mut source = RnsHybridEngineeringCorrelationSource::new(rng);
        domain2.dyadic_of_pmpe_profiled_with_source(&gelu_input, &gelu_cfg, &mut source)?
    };
    let gelu_stats = DirectionOperatorStats::from_profile(gelu.profile)
        .scale_to(operator_slots, gelu_target_slots);

    let square_cfg = DyadicOfPmpePolynomialConfig::square(f, chunk)?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modulus_margin_bits(10);
    let square_input = rns_shared_vector(operator_slots, fixed, &domain2, rng)?;
    let square = if opts.trusted_debug_offline {
        domain2.dyadic_of_pmpe_profiled(&square_input, &square_cfg, rng)?
    } else {
        let mut source = RnsHybridEngineeringCorrelationSource::new(rng);
        domain2.dyadic_of_pmpe_profiled_with_source(&square_input, &square_cfg, &mut source)?
    };
    let layernorm_stats = DirectionOperatorStats::from_profile(square.profile)
        .scale_to(operator_slots, layernorm_target_slots);

    let shifted_bound_bits = shifted_frac_bits.saturating_add(4);
    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &meanbeta_plan.coeffs,
        shifted_frac_bits,
        f,
        direction_exp_guard_bits(meanbeta_plan.metrics.degree, shifted_frac_bits, f),
        domain4.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(shifted_bound_bits)?
    .with_modulus_margin_bits(10);
    let exp_frac_bits = exp_cfg.output_frac_bits_with_guard()?;
    let row_sum_bound_bits = exp_frac_bits
        .saturating_add(BERT_SEQ.trailing_zeros())
        .saturating_add(1);
    let rescale_mask_bits = row_sum_bound_bits.saturating_add(4).min(126);
    let reciprocal_input_frac_bits = f.saturating_mul(2);
    let reciprocal_input_bound_bits = reciprocal_input_frac_bits
        .saturating_add(BERT_SEQ.next_power_of_two().trailing_zeros())
        .saturating_add(2);
    let reciprocal_center = (BERT_SEQ as f64) * (-meanbeta_plan.metrics.beta).exp();
    let reciprocal_coeffs = reciprocal_taylor_coeffs(reciprocal_center, 3);
    let reciprocal_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &reciprocal_coeffs,
        reciprocal_input_frac_bits,
        reciprocal_input_frac_bits,
        72,
        domain4.modulus_bits(),
        true,
        softmax_rows_source.max(1),
    )?
    .with_input_abs_bound_bits(reciprocal_input_bound_bits)?
    .with_modulus_margin_bits(10);
    let exp_audit = domain4.no_wrap_audit(&exp_cfg)?;
    let reciprocal_audit = domain4.no_wrap_audit(&reciprocal_cfg)?;
    let softmax_input = rns_shared_vector(softmax_source_slots, fixed, &domain4, rng)?;
    let softmax = if opts.trusted_debug_offline {
        domain4.softmax_mean_shift_of_pmpe_rescaled_profiled(
            &softmax_input,
            softmax_rows_source,
            BERT_SEQ,
            f,
            meanbeta_plan.metrics.beta,
            &exp_cfg,
            reciprocal_input_frac_bits,
            row_sum_bound_bits,
            rescale_mask_bits,
            &reciprocal_cfg,
            rng,
        )?
    } else {
        let mut source = RnsHybridEngineeringCorrelationSource::new(rng)
            .with_rescale_statistical_security_bits(4);
        domain4.softmax_mean_shift_of_pmpe_rescaled_profiled_with_source(
            &softmax_input,
            softmax_rows_source,
            BERT_SEQ,
            f,
            meanbeta_plan.metrics.beta,
            &exp_cfg,
            reciprocal_input_frac_bits,
            row_sum_bound_bits,
            rescale_mask_bits,
            &reciprocal_cfg,
            &mut source,
        )?
    };
    let mut softmax_source_stats =
        DirectionOperatorStats::from_profile(softmax.exp_phase.exp_profile);
    softmax_source_stats.add_assign(DirectionOperatorStats::from_profile(
        softmax
            .rescale_profile
            .ok_or("direction validation missing rescale profile")?,
    ));
    softmax_source_stats.add_assign(DirectionOperatorStats::from_profile(
        softmax.reciprocal_profile,
    ));
    softmax_source_stats.add_assign(DirectionOperatorStats::from_profile(
        softmax.broadcast_mul_profile,
    ));
    let softmax_stats = softmax_source_stats.scale_to(softmax_source_slots, softmax_target_slots);

    let mut total = DirectionOperatorStats::default();
    total.add_assign(gelu_stats);
    total.add_assign(softmax_stats);
    total.add_assign(layernorm_stats);
    let online_ms = total.online_us as f64 / 1000.0;
    let offline_ms = total.offline_us as f64 / 1000.0;
    let online_gb = total.online_bytes as f64 / 1_000_000_000.0;
    let lan_ms = online_ms + bandwidth_ms(online_gb, LAN_BW_MBPS);
    let wan_ms = online_ms + bandwidth_ms(online_gb, WAN_BW_MBPS);
    let operator_pass = online_ms <= 60_000.0
        && lan_ms <= 80_000.0
        && total.request_response_rounds == 0
        && total.online_secret_secret_mul == 0
        && online_gb <= 1.5
        && exp_audit.ok
        && reciprocal_audit.ok;
    print_row(
        "direction_validation",
        "gate2_bert_shape_nonlinear_operator_budget",
        1,
        operator_start.elapsed(),
        &format!(
            "status={}|offline_backend={offline_backend}|source_slots={operator_slots}|softmax_source_rows={softmax_rows_source}|scaled_to=bert_base_12layer_128token|online_ms={online_ms:.3}|offline_preprocess_ms={offline_ms:.3}|lan_ms={lan_ms:.3}|wan_ms={wan_ms:.3}|online_gb={online_gb:.6}|offline_gb={:.6}|pack_size_gb={:.6}|peak_pack_gb={:.6}|oneflow_phases={}|request_response_rounds={}|online_secret_secret_mul={}|online_trunc={}|rescale_mask_bits={rescale_mask_bits}|rescale_statistical_security_bits=4|exp_no_wrap_ok={}|exp_signed_margin_bits={}|reciprocal_no_wrap_ok={}|reciprocal_signed_margin_bits={}",
            pass_fail(operator_pass),
            total.offline_bytes as f64 / 1_000_000_000.0,
            total.pack_size_bytes as f64 / 1_000_000_000.0,
            total.peak_pack_resident_bytes as f64 / 1_000_000_000.0,
            total.oneflow_phases,
            total.request_response_rounds,
            total.online_secret_secret_mul,
            total.online_trunc,
            exp_audit.ok,
            exp_audit.signed_margin_bits,
            reciprocal_audit.ok,
            reciprocal_audit.signed_margin_bits,
        ),
    );

    let secure_offline_pass = total.secure_offline_us > 0 && total.trusted_debug_offline_us == 0;
    print_row(
        "direction_validation",
        "gate3_secure_offline_status",
        1,
        start.elapsed(),
        &format!(
            "status={}|offline_backend={offline_backend}|secure_offline_us={}|trusted_debug_offline_us={}|paper_security_ready={}|reason={}",
            pass_fail(secure_offline_pass),
            total.secure_offline_us,
            total.trusted_debug_offline_us,
            if secure_offline_pass && !opts.trusted_debug_offline {
                "false_hybrid_source_still_uses_ideal_taylor_beaver_until_high_scale_hss_or_vole_backend_is_wired"
            } else {
                "false"
            },
            if secure_offline_pass {
                "rns_hybrid_engineering_source_active_coin_tossed_rescale"
            } else if opts.trusted_debug_offline {
                "trusted_debug_offline_forced"
            } else {
                "correlation_source_interface_exists_but_runner_still_uses_trusted_debug"
            }
        ),
    );

    let hss_gate_start = Instant::now();
    let hss_correlation_pass = run_hss_correlation_source_smoke(rng).unwrap_or(false);
    print_row(
        "direction_validation",
        "gate3b_hss_correlation_backend_smoke",
        1,
        hss_gate_start.elapsed(),
        &format!(
            "status={}|scope=taylor_powers_beaver_mul_and_coin_tossed_rescale_packs|note=full_bert_shape_runner_still_uses_rns_ideal_for_large_scale_correlation_until_high_scale_hss_is_wired",
            pass_fail(hss_correlation_pass)
        ),
    );

    let highscale_hss_gate_start = Instant::now();
    let highscale_hss_correlation_pass = run_highscale_hss_correlation_source_probe(
        rng,
        opts.highscale_hss_degree,
        opts.highscale_hss_slots,
    )
    .unwrap_or(false);
    print_row(
        "direction_validation",
        "gate3c_highscale_hss_correlation_probe",
        1,
        highscale_hss_gate_start.elapsed(),
        &format!(
            "status={}|scope=two_61bit_ntt_limbs_taylor_beaver_and_coin_tossed_rescale|degree={}|slots={}|note=probe_not_full_shape_unless_slots_matches_target",
            pass_fail(highscale_hss_correlation_pass),
            opts.highscale_hss_degree,
            opts.highscale_hss_slots
        ),
    );

    let ahe_gate_start = Instant::now();
    let rlwe_ahe_correlation_pass = run_rlwe_ahe_correlation_source_smoke(rng).unwrap_or(false);
    print_row(
        "direction_validation",
        "gate3d_packed_rlwe_ahe_beaver_probe",
        1,
        ahe_gate_start.elapsed(),
        &format!(
            "status={}|scope=packed_rlwe_ahe_beaver_triples|note=whisper_hss_cross_term_path_is_only_a_reference_not_the_target_backend",
            pass_fail(rlwe_ahe_correlation_pass)
        ),
    );

    let ahe_operator_start = Instant::now();
    let (ahe_operator_pass, ahe_sample_stats, ahe_scaled_stats, ahe_error) =
        match run_packed_ahe_operator_sample_probe(opts, fixed, f, &meanbeta_plan, rng) {
            Ok((pass, sample, scaled)) => (pass, sample, scaled, "none".to_string()),
            Err(err) => (
                false,
                DirectionOperatorStats::default(),
                DirectionOperatorStats::default(),
                err.to_string().replace(['|', ',', '\n'], "_"),
            ),
        };
    let ahe_sample_online_ms = ahe_sample_stats.online_us as f64 / 1000.0;
    let ahe_sample_offline_ms = ahe_sample_stats.offline_us as f64 / 1000.0;
    let ahe_scaled_online_ms = ahe_scaled_stats.online_us as f64 / 1000.0;
    let ahe_scaled_offline_ms = ahe_scaled_stats.offline_us as f64 / 1000.0;
    print_row(
        "direction_validation",
        "gate3e_packed_rlwe_ahe_operator_sample",
        1,
        ahe_operator_start.elapsed(),
        &format!(
            "status={}|paper_protocol=rlwe_ahe_case_ii|target_ring=rns_prime_limbs|noise_flooding=pending|degree={}|plain_bits={}|q_bits={}|q_count={}|source_lanes={}|sample_online_ms={ahe_sample_online_ms:.3}|sample_offline_ms={ahe_sample_offline_ms:.3}|sample_offline_gb={:.6}|scaled_online_ms={ahe_scaled_online_ms:.3}|scaled_offline_ms={ahe_scaled_offline_ms:.3}|scaled_offline_gb={:.6}|oneflow_phases={}|request_response_rounds={}|online_secret_secret_mul={}|trusted_debug_offline_us={}|error={}|note=sampled_real_packed_rlwe_ahe_correlation_source_scaled_to_bert_shapes",
            pass_fail(ahe_operator_pass),
            opts.packed_ahe_probe_degree,
            opts.packed_ahe_probe_plain_bits,
            opts.packed_ahe_probe_q_bits,
            opts.packed_ahe_probe_q_count,
            opts.packed_ahe_probe_lanes,
            ahe_sample_stats.offline_bytes as f64 / 1_000_000_000.0,
            ahe_scaled_stats.offline_bytes as f64 / 1_000_000_000.0,
            ahe_scaled_stats.oneflow_phases,
            ahe_scaled_stats.request_response_rounds,
            ahe_scaled_stats.online_secret_secret_mul,
            ahe_scaled_stats.trusted_debug_offline_us,
            ahe_error,
        ),
    );
    let ahe_generic_backend_paper_ready = ahe_operator_pass
        && ahe_scaled_stats.offline_us <= 120_000_000
        && ahe_scaled_stats.offline_bytes <= 8_000_000_000;
    print_row(
        "direction_validation",
        "gate3f_real_correlation_backend_decision",
        1,
        ahe_operator_start.elapsed(),
        &format!(
            "status={}|backend=packed_rlwe_ahe_generic_beaver|paper_protocol=rlwe_ahe_case_ii|target_ring=rns_prime_limbs|noise_flooding=pending|degree={}|plain_bits={}|q_bits={}|q_count={}|decision={}|offline_ms={ahe_scaled_offline_ms:.3}|offline_gb={:.6}|threshold_offline_ms=120000|threshold_offline_gb=8.000000|next_backend=specialized_pmpe_vole_or_pcg_taylor_correlation|note=generic_beaver_power_generation_is_a_security_bridge_not_the_final_full_bert_pmpe_preprocess_backend",
            pass_fail(ahe_generic_backend_paper_ready),
            opts.packed_ahe_probe_degree,
            opts.packed_ahe_probe_plain_bits,
            opts.packed_ahe_probe_q_bits,
            opts.packed_ahe_probe_q_count,
            if ahe_generic_backend_paper_ready {
                "KEEP_AS_PAPER_BACKEND"
            } else {
                "REJECT_GENERIC_AHE_BEAVER_FOR_FULL_PMPE_TAYLOR"
            },
            ahe_scaled_stats.offline_bytes as f64 / 1_000_000_000.0,
        ),
    );

    let hss_taylor_start = Instant::now();
    let (hss_taylor_pass, hss_taylor_sample_stats, hss_taylor_scaled_stats, hss_taylor_error) =
        if opts.run_hss_wrapper_probe {
            match run_rms_hss_taylor_operator_sample_probe(opts, fixed, f, &meanbeta_plan, rng) {
                Ok((pass, sample, scaled)) => (pass, sample, scaled, "none".to_string()),
                Err(err) => (
                    false,
                    DirectionOperatorStats::default(),
                    DirectionOperatorStats::default(),
                    err.to_string().replace(['|', ',', '\n'], "_"),
                ),
            }
        } else {
            (
                true,
                DirectionOperatorStats::default(),
                DirectionOperatorStats::default(),
                "skipped_by_default_after_cpu_bound_negative_baseline".to_string(),
            )
        };
    let hss_taylor_sample_online_ms = hss_taylor_sample_stats.online_us as f64 / 1000.0;
    let hss_taylor_sample_offline_ms = hss_taylor_sample_stats.offline_us as f64 / 1000.0;
    let hss_taylor_scaled_online_ms = hss_taylor_scaled_stats.online_us as f64 / 1000.0;
    let hss_taylor_scaled_offline_ms = hss_taylor_scaled_stats.offline_us as f64 / 1000.0;
    let hss_taylor_status = if opts.run_hss_wrapper_probe {
        pass_fail(hss_taylor_pass)
    } else {
        "SKIP"
    };
    print_row(
        "direction_validation",
        "gate3g_rms_hss_wrapper_taylor_backend_sample",
        1,
        hss_taylor_start.elapsed(),
        &format!(
            "status={}|backend=rms_hss_wrapper_taylor|target=negative_baseline_only|enabled={}|degree={}|plain_bits={}|q_bits={}|q_count={}|source_lanes={}|sample_online_ms={hss_taylor_sample_online_ms:.3}|sample_offline_ms={hss_taylor_sample_offline_ms:.3}|sample_offline_gb={:.6}|scaled_online_ms={hss_taylor_scaled_online_ms:.3}|scaled_offline_ms={hss_taylor_scaled_offline_ms:.3}|scaled_offline_gb={:.6}|oneflow_phases={}|request_response_rounds={}|online_secret_secret_mul={}|trusted_debug_offline_us={}|error={}|note=current_hss_wrapper_is_not_mainline_run_with_run_hss_wrapper_probe_to_reproduce_negative_baseline",
            hss_taylor_status,
            opts.run_hss_wrapper_probe,
            opts.packed_ahe_probe_degree,
            opts.packed_ahe_probe_plain_bits,
            opts.packed_ahe_probe_q_bits,
            opts.packed_ahe_probe_q_count,
            opts.packed_ahe_probe_lanes,
            hss_taylor_sample_stats.offline_bytes as f64 / 1_000_000_000.0,
            hss_taylor_scaled_stats.offline_bytes as f64 / 1_000_000_000.0,
            hss_taylor_scaled_stats.oneflow_phases,
            hss_taylor_scaled_stats.request_response_rounds,
            hss_taylor_scaled_stats.online_secret_secret_mul,
            hss_taylor_scaled_stats.trusted_debug_offline_us,
            hss_taylor_error,
        ),
    );

    let cross_term_start = Instant::now();
    let (cross_term_pass, cross_term_sample_stats, cross_term_scaled_stats, cross_term_error) =
        match run_cross_term_ole_taylor_operator_sample_probe(opts, fixed, f, &meanbeta_plan, rng) {
            Ok((pass, sample, scaled)) => (pass, sample, scaled, "none".to_string()),
            Err(err) => (
                false,
                DirectionOperatorStats::default(),
                DirectionOperatorStats::default(),
                err.to_string().replace(['|', ',', '\n'], "_"),
            ),
        };
    let cross_term_sample_online_ms = cross_term_sample_stats.online_us as f64 / 1000.0;
    let cross_term_sample_offline_ms = cross_term_sample_stats.offline_us as f64 / 1000.0;
    let cross_term_scaled_online_ms = cross_term_scaled_stats.online_us as f64 / 1000.0;
    let cross_term_scaled_offline_ms = cross_term_scaled_stats.offline_us as f64 / 1000.0;
    print_row(
        "direction_validation",
        "gate3h_cross_term_ole_taylor_backend_sample",
        1,
        cross_term_start.elapsed(),
        &format!(
            "status={}|backend=binomial_cross_term_ole_taylor|target=pmpe_taylor_correlations_only|degree={}|plain_bits={}|source_lanes={}|sample_online_ms={cross_term_sample_online_ms:.3}|sample_offline_ms={cross_term_sample_offline_ms:.3}|sample_offline_gb={:.6}|scaled_online_ms={cross_term_scaled_online_ms:.3}|scaled_offline_ms={cross_term_scaled_offline_ms:.3}|scaled_offline_gb={:.6}|oneflow_phases={}|request_response_rounds={}|online_secret_secret_mul={}|trusted_debug_offline_us={}|error={}|note=native_binomial_cross_term_basis_no_generic_beaver_no_hss_wrapper",
            pass_fail(cross_term_pass),
            opts.packed_ahe_probe_degree,
            opts.packed_ahe_probe_plain_bits,
            opts.packed_ahe_probe_lanes,
            cross_term_sample_stats.offline_bytes as f64 / 1_000_000_000.0,
            cross_term_scaled_stats.offline_bytes as f64 / 1_000_000_000.0,
            cross_term_scaled_stats.oneflow_phases,
            cross_term_scaled_stats.request_response_rounds,
            cross_term_scaled_stats.online_secret_secret_mul,
            cross_term_scaled_stats.trusted_debug_offline_us,
            cross_term_error,
        ),
    );

    let ahe_cross_term_start = Instant::now();
    let (
        ahe_cross_term_pass,
        ahe_cross_term_sample_stats,
        ahe_cross_term_scaled_stats,
        ahe_cross_term_error,
    ) = match run_rlwe_ahe_cross_term_ole_taylor_operator_sample_probe(
        opts,
        fixed,
        f,
        &meanbeta_plan,
        rng,
    ) {
        Ok((pass, sample, scaled)) => (pass, sample, scaled, "none".to_string()),
        Err(err) => (
            false,
            DirectionOperatorStats::default(),
            DirectionOperatorStats::default(),
            err.to_string().replace(['|', ',', '\n'], "_"),
        ),
    };
    let ahe_cross_term_sample_online_ms = ahe_cross_term_sample_stats.online_us as f64 / 1000.0;
    let ahe_cross_term_sample_offline_ms = ahe_cross_term_sample_stats.offline_us as f64 / 1000.0;
    let ahe_cross_term_scaled_online_ms = ahe_cross_term_scaled_stats.online_us as f64 / 1000.0;
    let ahe_cross_term_scaled_offline_ms = ahe_cross_term_scaled_stats.offline_us as f64 / 1000.0;
    print_row(
        "direction_validation",
        "gate3i_rlwe_ahe_cross_term_ole_taylor_backend_sample",
        1,
        ahe_cross_term_start.elapsed(),
        &format!(
            "status={}|backend=rlwe_ahe_cross_term_ole_taylor|target=pmpe_taylor_correlations_only|degree={}|plain_bits={}|q_bits={}|q_count={}|source_lanes={}|sample_online_ms={ahe_cross_term_sample_online_ms:.3}|sample_offline_ms={ahe_cross_term_sample_offline_ms:.3}|sample_offline_gb={:.6}|scaled_online_ms={ahe_cross_term_scaled_online_ms:.3}|scaled_offline_ms={ahe_cross_term_scaled_offline_ms:.3}|scaled_offline_gb={:.6}|oneflow_phases={}|request_response_rounds={}|online_secret_secret_mul={}|trusted_debug_offline_us={}|error={}|note=real_rlwe_ahe_cross_term_ole_generates_taylor_pack_without_generic_beaver",
            pass_fail(ahe_cross_term_pass),
            opts.packed_ahe_probe_degree,
            opts.packed_ahe_probe_plain_bits,
            opts.packed_ahe_probe_q_bits,
            opts.packed_ahe_probe_q_count,
            opts.packed_ahe_probe_lanes,
            ahe_cross_term_sample_stats.offline_bytes as f64 / 1_000_000_000.0,
            ahe_cross_term_scaled_stats.offline_bytes as f64 / 1_000_000_000.0,
            ahe_cross_term_scaled_stats.oneflow_phases,
            ahe_cross_term_scaled_stats.request_response_rounds,
            ahe_cross_term_scaled_stats.online_secret_secret_mul,
            ahe_cross_term_scaled_stats.trusted_debug_offline_us,
            ahe_cross_term_error,
        ),
    );

    let bumblebee_pass =
        operator_pass && lan_ms <= 170_000.0 && online_gb <= PAPER_BERT_BASE_COMM_GB;
    print_row(
        "direction_validation",
        "gate4_bumblebee_aligned_budget",
        1,
        start.elapsed(),
        &format!(
            "status={}|silent_nonlinear_lan_ms={lan_ms:.3}|silent_nonlinear_wan_ms={wan_ms:.3}|silent_online_gb={online_gb:.6}|bumblebee_reference_comm_gb={PAPER_BERT_BASE_COMM_GB}|budget_note=nonlinear_only_not_full_model",
            pass_fail(bumblebee_pass),
        ),
    );

    let engineering_decision =
        if plaintext_pass && operator_pass && secure_offline_pass && bumblebee_pass {
            "GO_ENGINEERING"
        } else if plaintext_pass && operator_pass && bumblebee_pass {
            "CONDITIONAL_ENGINEERING"
        } else {
            "NO_GO"
        };
    let paper_readiness = if engineering_decision == "GO_ENGINEERING" && opts.trusted_debug_offline
    {
        "NO_TRUSTED_DEBUG"
    } else if engineering_decision == "GO_ENGINEERING" && !hss_correlation_pass {
        "BLOCKED_ON_HSS_CORRELATION_BACKEND"
    } else if engineering_decision == "GO_ENGINEERING" && !highscale_hss_correlation_pass {
        "BLOCKED_ON_HIGHSCALE_HSS_CORRELATION_BACKEND"
    } else if engineering_decision == "GO_ENGINEERING" && !rlwe_ahe_correlation_pass {
        "BLOCKED_ON_PACKED_RLWE_AHE_BEAVER_BACKEND"
    } else if engineering_decision == "GO_ENGINEERING" {
        "BLOCKED_ON_REAL_CORRELATION_BACKEND_AND_MODEL_ACCURACY"
    } else if plaintext_pass && operator_pass && bumblebee_pass {
        "BLOCKED_ON_SECURE_OFFLINE"
    } else {
        "NO_DIRECTION_NOT_VALIDATED"
    };
    let blocker = if !plaintext_pass {
        "plaintext_accuracy"
    } else if !operator_pass {
        "operator_budget"
    } else if !bumblebee_pass {
        "bumblebee_budget"
    } else if !secure_offline_pass {
        "secure_offline"
    } else if !hss_correlation_pass {
        "hss_correlation_backend"
    } else if !highscale_hss_correlation_pass {
        "highscale_hss_correlation_backend"
    } else if !rlwe_ahe_correlation_pass {
        "packed_rlwe_ahe_beaver_backend"
    } else {
        "none"
    };
    print_row(
        "direction_validation",
        "silent_ofpmpe_direction_decision",
        1,
        start.elapsed(),
        &format!(
            "engineering_decision={engineering_decision}|paper_readiness={paper_readiness}|offline_backend={offline_backend}|blocker={blocker}|meaning=GO_ENGINEERING_means_protocol_shape_budget_and_non_trusted_preprocess_accounting_pass;paper_readiness_requires_real_correlation_backend_and_real_model_accuracy"
        ),
    );
    Ok(())
}

fn run_highscale_hss_correlation_source_probe(
    rng: &mut StdRng,
    degree: usize,
    slots: usize,
) -> Result<bool, Box<dyn std::error::Error>> {
    if !degree.is_power_of_two() {
        return Err("high-scale HSS degree must be a power of two".into());
    }
    let domain = RnsDyadicDomain::two_limb_61bit_ntt(degree)?;
    let p0 = domain.moduli()[0];
    let p1 = domain.moduli()[1];
    let engines = vec![
        build_hss_plain_modulus_engine(degree, p0)?,
        build_hss_plain_modulus_engine(degree, p1)?,
    ];
    let cfg =
        DyadicOfPmpePolynomialConfig::from_integer_coeffs(vec![3, 2, 1], 0, 0, 0, 122, true, 8)?
            .with_input_abs_bound_bits(4)?
            .with_modulus_margin_bits(10);
    let values = (0..slots)
        .map(|idx| (idx as i128 % 7) - 3)
        .collect::<Vec<_>>();
    let input = domain.share_with_rng(&values, rng)?;
    let lhs_values = (0..slots)
        .map(|idx| (idx as i128 % 11) - 5)
        .collect::<Vec<_>>();
    let rhs_values = (0..slots)
        .map(|idx| 6 - (idx as i128 % 13))
        .collect::<Vec<_>>();
    let lhs = RnsScaledShareTensor {
        shares: domain.share_with_rng(&lhs_values, rng)?,
        frac_bits: 0,
        guard_bits: 0,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let rhs = RnsScaledShareTensor {
        shares: domain.share_with_rng(&rhs_values, rng)?,
        frac_bits: 0,
        guard_bits: 0,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let rescale_values = (0..slots)
        .map(|idx| 16 + 8 * (idx as i128 % 8))
        .collect::<Vec<_>>();
    let rescale_input = RnsScaledShareTensor {
        shares: domain.share_with_rng(&rescale_values, rng)?,
        frac_bits: 4,
        guard_bits: 4,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let mut source =
        RnsHssCorrelationSource::new(&domain, engines, rng)?.with_coin_tossed_rescale(2);
    let out = domain.dyadic_of_pmpe_profiled_with_source(&input, &cfg, &mut source)?;
    let expected = values
        .iter()
        .map(|&value| dyadic_integer_polynomial_value_i128_for_runner(value, &cfg.coeffs_num))
        .collect::<Vec<_>>();
    if out.tensor.shares.reconstruct_centered_i128()? != expected {
        return Ok(false);
    }

    let mut pack = source.beaver_mul_pack(&domain, lhs_values.len())?;
    let product = domain.mul_scaled_with_beaver_profiled(&lhs, &rhs, &mut pack)?;
    let expected_product = lhs_values
        .iter()
        .zip(rhs_values.iter())
        .map(|(&lhs, &rhs)| lhs * rhs)
        .collect::<Vec<_>>();
    let rescaled = domain.approximate_rescale_nonnegative_profiled_with_source(
        &rescale_input,
        2,
        4,
        8,
        &mut source,
    )?;
    Ok(
        product.tensor.shares.reconstruct_centered_i128()? == expected_product
            && rescaled
                .tensor
                .shares
                .reconstruct_nonnegative_i128_bounded(8)?
                == rescale_values
                    .iter()
                    .map(|&value| value >> 2)
                    .collect::<Vec<_>>()
            && out.profile.trusted_debug_offline_us == 0
            && product.profile.trusted_debug_offline_us == 0
            && rescaled.profile.trusted_debug_offline_us == 0,
    )
}

fn run_rlwe_ahe_correlation_source_smoke(
    rng: &mut StdRng,
) -> Result<bool, Box<dyn std::error::Error>> {
    let domain = RnsDyadicDomain::from_moduli(vec![97, 193])?;
    let engines = vec![
        build_packed_ahe_plain_modulus_engine(8, 97, 4, 50)?,
        build_packed_ahe_plain_modulus_engine(8, 193, 4, 50)?,
    ];
    let cfg =
        DyadicOfPmpePolynomialConfig::from_integer_coeffs(vec![3, 2, 1], 0, 0, 0, 15, true, 8)?
            .with_input_abs_bound_bits(3)?
            .with_modulus_margin_bits(2);
    let values = vec![-2, -1, 0, 1, 2, 3];
    let input = domain.share_with_rng(&values, rng)?;
    let lhs_values = vec![2, -3, 4, -5, 6, -7];
    let rhs_values = vec![6, 7, -2, -3, 5, 4];
    let lhs = RnsScaledShareTensor {
        shares: domain.share_with_rng(&lhs_values, rng)?,
        frac_bits: 0,
        guard_bits: 0,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let rhs = RnsScaledShareTensor {
        shares: domain.share_with_rng(&rhs_values, rng)?,
        frac_bits: 0,
        guard_bits: 0,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let mut source = RnsRlweAheCorrelationSource::new(&domain, engines, rng)?
        .with_rescale_statistical_security_bits(2);
    let out = domain.dyadic_of_pmpe_profiled_with_source(&input, &cfg, &mut source)?;
    let expected = values
        .iter()
        .map(|&value| dyadic_integer_polynomial_value_i128_for_runner(value, &cfg.coeffs_num))
        .collect::<Vec<_>>();
    if out.tensor.shares.reconstruct_centered_i128()? != expected {
        return Ok(false);
    }

    let mut pack = source.beaver_mul_pack(&domain, lhs_values.len())?;
    if pack.offline_bytes <= pack.pack_size_bytes {
        return Ok(false);
    }
    let product = domain.mul_scaled_with_beaver_profiled(&lhs, &rhs, &mut pack)?;
    let expected_product = lhs_values
        .iter()
        .zip(rhs_values.iter())
        .map(|(&lhs, &rhs)| lhs * rhs)
        .collect::<Vec<_>>();
    Ok(
        product.tensor.shares.reconstruct_centered_i128()? == expected_product
            && out.profile.trusted_debug_offline_us == 0
            && product.profile.trusted_debug_offline_us == 0,
    )
}

fn run_packed_ahe_operator_sample_probe(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    f: u32,
    meanbeta_plan: &MeanBetaSoftmaxPlan,
    rng: &mut StdRng,
) -> Result<(bool, DirectionOperatorStats, DirectionOperatorStats), Box<dyn std::error::Error>> {
    let degree = opts.packed_ahe_probe_degree;
    if !degree.is_power_of_two() {
        return Err("packed AHE operator probe degree must be a power of two".into());
    }
    let lanes = opts.packed_ahe_probe_lanes.max(BERT_SEQ);
    let chunk = degree.max(1);
    let softmax_rows_source = usize::max(1, lanes / BERT_SEQ);
    let softmax_source_slots = softmax_rows_source * BERT_SEQ;
    let activation_bound_bits = f.saturating_add(3);
    let shifted_frac_bits = f + BERT_SEQ.trailing_zeros();
    let shifted_bound_bits = shifted_frac_bits.saturating_add(4);

    let plain_bits = opts.packed_ahe_probe_plain_bits;
    let domain2 = RnsDyadicDomain::ntt_prime_limbs(plain_bits, 2, degree)?;
    let domain4 = RnsDyadicDomain::ntt_prime_limbs(plain_bits, 4, degree)?;
    let domain2_engines = domain2
        .moduli()
        .iter()
        .map(|&modulus| {
            build_packed_ahe_plain_modulus_engine(
                degree,
                modulus,
                opts.packed_ahe_probe_q_count,
                opts.packed_ahe_probe_q_bits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let domain4_engines = domain4
        .moduli()
        .iter()
        .map(|&modulus| {
            build_packed_ahe_plain_modulus_engine(
                degree,
                modulus,
                opts.packed_ahe_probe_q_count,
                opts.packed_ahe_probe_q_bits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let gelu_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::gelu_global_degree6_probe(chunk).coeffs,
        f,
        f,
        56,
        domain2.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modulus_margin_bits(10);
    let gelu_input = rns_shared_vector(lanes, fixed, &domain2, rng)?;
    let mut gelu_source = RnsRlweAheCorrelationSource::new(&domain2, domain2_engines.clone(), rng)?
        .with_rescale_statistical_security_bits(4);
    let gelu =
        domain2.dyadic_of_pmpe_profiled_with_source(&gelu_input, &gelu_cfg, &mut gelu_source)?;
    let gelu_sample_stats = DirectionOperatorStats::from_profile(gelu.profile);
    let gelu_scaled_stats = gelu_sample_stats.scale_to(lanes, BERT_LAYERS * BERT_SEQ * BERT_FFN);

    let square_cfg = DyadicOfPmpePolynomialConfig::square(f, chunk)?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modulus_margin_bits(10);
    let square_input = rns_shared_vector(lanes, fixed, &domain2, rng)?;
    let mut square_source = RnsRlweAheCorrelationSource::new(&domain2, domain2_engines, rng)?
        .with_rescale_statistical_security_bits(4);
    let square = domain2.dyadic_of_pmpe_profiled_with_source(
        &square_input,
        &square_cfg,
        &mut square_source,
    )?;
    let square_sample_stats = DirectionOperatorStats::from_profile(square.profile);
    let square_scaled_stats =
        square_sample_stats.scale_to(lanes, 2 * BERT_LAYERS * BERT_SEQ * BERT_HIDDEN);

    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &meanbeta_plan.coeffs,
        shifted_frac_bits,
        f,
        direction_exp_guard_bits(meanbeta_plan.metrics.degree, shifted_frac_bits, f),
        domain4.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(shifted_bound_bits)?
    .with_modulus_margin_bits(10);
    let exp_frac_bits = exp_cfg.output_frac_bits_with_guard()?;
    let row_sum_bound_bits = exp_frac_bits
        .saturating_add(BERT_SEQ.trailing_zeros())
        .saturating_add(1);
    let rescale_mask_bits = row_sum_bound_bits.saturating_add(4).min(126);
    let reciprocal_input_frac_bits = f.saturating_mul(2);
    let reciprocal_input_bound_bits = reciprocal_input_frac_bits
        .saturating_add(BERT_SEQ.next_power_of_two().trailing_zeros())
        .saturating_add(2);
    let reciprocal_center = (BERT_SEQ as f64) * (-meanbeta_plan.metrics.beta).exp();
    let reciprocal_coeffs = reciprocal_taylor_coeffs(reciprocal_center, 3);
    let reciprocal_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &reciprocal_coeffs,
        reciprocal_input_frac_bits,
        reciprocal_input_frac_bits,
        72,
        domain4.modulus_bits(),
        true,
        softmax_rows_source.max(1),
    )?
    .with_input_abs_bound_bits(reciprocal_input_bound_bits)?
    .with_modulus_margin_bits(10);
    let softmax_input = rns_shared_vector(softmax_source_slots, fixed, &domain4, rng)?;
    let mut softmax_source = RnsRlweAheCorrelationSource::new(&domain4, domain4_engines, rng)?
        .with_rescale_statistical_security_bits(4);
    let softmax = domain4.softmax_mean_shift_of_pmpe_rescaled_profiled_with_source(
        &softmax_input,
        softmax_rows_source,
        BERT_SEQ,
        f,
        meanbeta_plan.metrics.beta,
        &exp_cfg,
        reciprocal_input_frac_bits,
        row_sum_bound_bits,
        rescale_mask_bits,
        &reciprocal_cfg,
        &mut softmax_source,
    )?;
    let mut softmax_sample_stats =
        DirectionOperatorStats::from_profile(softmax.exp_phase.exp_profile);
    softmax_sample_stats.add_assign(DirectionOperatorStats::from_profile(
        softmax
            .rescale_profile
            .ok_or("packed AHE probe missing rescale profile")?,
    ));
    softmax_sample_stats.add_assign(DirectionOperatorStats::from_profile(
        softmax.reciprocal_profile,
    ));
    softmax_sample_stats.add_assign(DirectionOperatorStats::from_profile(
        softmax.broadcast_mul_profile,
    ));
    let softmax_scaled_stats = softmax_sample_stats.scale_to(
        softmax_source_slots,
        BERT_LAYERS * BERT_HEADS * BERT_SEQ * BERT_SEQ,
    );

    let mut sample_total = DirectionOperatorStats::default();
    sample_total.add_assign(gelu_sample_stats);
    sample_total.add_assign(square_sample_stats);
    sample_total.add_assign(softmax_sample_stats);

    let mut scaled_total = DirectionOperatorStats::default();
    scaled_total.add_assign(gelu_scaled_stats);
    scaled_total.add_assign(square_scaled_stats);
    scaled_total.add_assign(softmax_scaled_stats);

    let pass = sample_total.trusted_debug_offline_us == 0
        && sample_total.request_response_rounds == 0
        && sample_total.online_secret_secret_mul == 0
        && sample_total.secure_offline_us > 0;
    Ok((pass, sample_total, scaled_total))
}

fn run_rms_hss_taylor_operator_sample_probe(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    f: u32,
    meanbeta_plan: &MeanBetaSoftmaxPlan,
    rng: &mut StdRng,
) -> Result<(bool, DirectionOperatorStats, DirectionOperatorStats), Box<dyn std::error::Error>> {
    let degree = opts.packed_ahe_probe_degree;
    if !degree.is_power_of_two() {
        return Err("RMS-HSS Taylor probe degree must be a power of two".into());
    }
    let lanes = opts.packed_ahe_probe_lanes.max(BERT_SEQ);
    let chunk = degree.max(1);
    let softmax_rows_source = usize::max(1, lanes / BERT_SEQ);
    let softmax_source_slots = softmax_rows_source * BERT_SEQ;
    let activation_bound_bits = f.saturating_add(3);
    let shifted_frac_bits = f + BERT_SEQ.trailing_zeros();
    let shifted_bound_bits = shifted_frac_bits.saturating_add(4);
    let plain_bits = opts.packed_ahe_probe_plain_bits;

    let domain2 = RnsDyadicDomain::ntt_prime_limbs(plain_bits, 2, degree)?;
    let domain4 = RnsDyadicDomain::ntt_prime_limbs(plain_bits, 4, degree)?;
    let domain2_engines = domain2
        .moduli()
        .iter()
        .map(|&modulus| {
            build_hss_plain_modulus_engine_with_q_count(
                degree,
                modulus,
                opts.packed_ahe_probe_q_count,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let domain4_engines = domain4
        .moduli()
        .iter()
        .map(|&modulus| {
            build_hss_plain_modulus_engine_with_q_count(
                degree,
                modulus,
                opts.packed_ahe_probe_q_count,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let gelu_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::gelu_global_degree6_probe(chunk).coeffs,
        f,
        f,
        56,
        domain2.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modulus_margin_bits(10);
    let gelu_input = rns_shared_vector(lanes, fixed, &domain2, rng)?;
    let mut gelu_source = RnsHssCorrelationSource::new(&domain2, domain2_engines.clone(), rng)?;
    let gelu =
        domain2.dyadic_of_pmpe_profiled_with_source(&gelu_input, &gelu_cfg, &mut gelu_source)?;
    let gelu_sample_stats = DirectionOperatorStats::from_profile(gelu.profile);
    let gelu_scaled_stats = gelu_sample_stats.scale_to(lanes, BERT_LAYERS * BERT_SEQ * BERT_FFN);

    let square_cfg = DyadicOfPmpePolynomialConfig::square(f, chunk)?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modulus_margin_bits(10);
    let square_input = rns_shared_vector(lanes, fixed, &domain2, rng)?;
    let mut square_source = RnsHssCorrelationSource::new(&domain2, domain2_engines, rng)?;
    let square = domain2.dyadic_of_pmpe_profiled_with_source(
        &square_input,
        &square_cfg,
        &mut square_source,
    )?;
    let square_sample_stats = DirectionOperatorStats::from_profile(square.profile);
    let square_scaled_stats =
        square_sample_stats.scale_to(lanes, 2 * BERT_LAYERS * BERT_SEQ * BERT_HIDDEN);

    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &meanbeta_plan.coeffs,
        shifted_frac_bits,
        f,
        direction_exp_guard_bits(meanbeta_plan.metrics.degree, shifted_frac_bits, f),
        domain4.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(shifted_bound_bits)?
    .with_modulus_margin_bits(10);
    let softmax_input = rns_shared_vector(softmax_source_slots, fixed, &domain4, rng)?;
    let mut softmax_source = RnsHssCorrelationSource::new(&domain4, domain4_engines, rng)?;
    let softmax_exp = domain4.softmax_mean_shift_of_pmpe_exp_profiled_with_source(
        &softmax_input,
        softmax_rows_source,
        BERT_SEQ,
        f,
        meanbeta_plan.metrics.beta,
        &exp_cfg,
        &mut softmax_source,
    )?;
    let softmax_sample_stats = DirectionOperatorStats::from_profile(softmax_exp.exp_profile);
    let softmax_scaled_stats = softmax_sample_stats.scale_to(
        softmax_source_slots,
        BERT_LAYERS * BERT_HEADS * BERT_SEQ * BERT_SEQ,
    );

    let mut sample_total = DirectionOperatorStats::default();
    sample_total.add_assign(gelu_sample_stats);
    sample_total.add_assign(square_sample_stats);
    sample_total.add_assign(softmax_sample_stats);

    let mut scaled_total = DirectionOperatorStats::default();
    scaled_total.add_assign(gelu_scaled_stats);
    scaled_total.add_assign(square_scaled_stats);
    scaled_total.add_assign(softmax_scaled_stats);

    let pass = sample_total.trusted_debug_offline_us == 0
        && sample_total.request_response_rounds == 0
        && sample_total.online_secret_secret_mul == 0
        && sample_total.secure_offline_us > 0;
    Ok((pass, sample_total, scaled_total))
}

fn run_cross_term_ole_taylor_operator_sample_probe(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    f: u32,
    meanbeta_plan: &MeanBetaSoftmaxPlan,
    rng: &mut StdRng,
) -> Result<(bool, DirectionOperatorStats, DirectionOperatorStats), Box<dyn std::error::Error>> {
    let degree = opts.packed_ahe_probe_degree;
    if !degree.is_power_of_two() {
        return Err("Cross-term OLE Taylor probe degree must be a power of two".into());
    }
    let lanes = opts.packed_ahe_probe_lanes.max(BERT_SEQ);
    let chunk = degree.max(1);
    let softmax_rows_source = usize::max(1, lanes / BERT_SEQ);
    let softmax_source_slots = softmax_rows_source * BERT_SEQ;
    let activation_bound_bits = f.saturating_add(3);
    let shifted_frac_bits = f + BERT_SEQ.trailing_zeros();
    let shifted_bound_bits = shifted_frac_bits.saturating_add(4);
    let plain_bits = opts.packed_ahe_probe_plain_bits;

    let domain2 = RnsDyadicDomain::ntt_prime_limbs(plain_bits, 2, degree)?;
    let domain4 = RnsDyadicDomain::ntt_prime_limbs(plain_bits, 4, degree)?;

    let gelu_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::gelu_global_degree6_probe(chunk).coeffs,
        f,
        f,
        56,
        domain2.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modulus_margin_bits(10);
    let gelu_input = rns_shared_vector(lanes, fixed, &domain2, rng)?;
    let mut gelu_source = RnsCrossTermOleTaylorSource::new(rng);
    let gelu =
        domain2.dyadic_of_pmpe_profiled_with_source(&gelu_input, &gelu_cfg, &mut gelu_source)?;
    let gelu_sample_stats = DirectionOperatorStats::from_profile(gelu.profile);
    let gelu_scaled_stats = gelu_sample_stats.scale_to(lanes, BERT_LAYERS * BERT_SEQ * BERT_FFN);

    let square_cfg = DyadicOfPmpePolynomialConfig::square(f, chunk)?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modulus_margin_bits(10);
    let square_input = rns_shared_vector(lanes, fixed, &domain2, rng)?;
    let mut square_source = RnsCrossTermOleTaylorSource::new(rng);
    let square = domain2.dyadic_of_pmpe_profiled_with_source(
        &square_input,
        &square_cfg,
        &mut square_source,
    )?;
    let square_sample_stats = DirectionOperatorStats::from_profile(square.profile);
    let square_scaled_stats =
        square_sample_stats.scale_to(lanes, 2 * BERT_LAYERS * BERT_SEQ * BERT_HIDDEN);

    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &meanbeta_plan.coeffs,
        shifted_frac_bits,
        f,
        direction_exp_guard_bits(meanbeta_plan.metrics.degree, shifted_frac_bits, f),
        domain4.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(shifted_bound_bits)?
    .with_modulus_margin_bits(10);
    let softmax_input = rns_shared_vector(softmax_source_slots, fixed, &domain4, rng)?;
    let mut softmax_source = RnsCrossTermOleTaylorSource::new(rng);
    let softmax_exp = domain4.softmax_mean_shift_of_pmpe_exp_profiled_with_source(
        &softmax_input,
        softmax_rows_source,
        BERT_SEQ,
        f,
        meanbeta_plan.metrics.beta,
        &exp_cfg,
        &mut softmax_source,
    )?;
    let softmax_sample_stats = DirectionOperatorStats::from_profile(softmax_exp.exp_profile);
    let softmax_scaled_stats = softmax_sample_stats.scale_to(
        softmax_source_slots,
        BERT_LAYERS * BERT_HEADS * BERT_SEQ * BERT_SEQ,
    );

    let mut sample_total = DirectionOperatorStats::default();
    sample_total.add_assign(gelu_sample_stats);
    sample_total.add_assign(square_sample_stats);
    sample_total.add_assign(softmax_sample_stats);

    let mut scaled_total = DirectionOperatorStats::default();
    scaled_total.add_assign(gelu_scaled_stats);
    scaled_total.add_assign(square_scaled_stats);
    scaled_total.add_assign(softmax_scaled_stats);

    let pass = sample_total.trusted_debug_offline_us == 0
        && sample_total.request_response_rounds == 0
        && sample_total.online_secret_secret_mul == 0
        && sample_total.secure_offline_us > 0;
    Ok((pass, sample_total, scaled_total))
}

fn run_rlwe_ahe_cross_term_ole_taylor_operator_sample_probe(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    f: u32,
    meanbeta_plan: &MeanBetaSoftmaxPlan,
    rng: &mut StdRng,
) -> Result<(bool, DirectionOperatorStats, DirectionOperatorStats), Box<dyn std::error::Error>> {
    let degree = opts.packed_ahe_probe_degree;
    if !degree.is_power_of_two() {
        return Err("RLWE-AHE Cross-term OLE Taylor probe degree must be a power of two".into());
    }
    let lanes = opts.packed_ahe_probe_lanes.max(BERT_SEQ);
    let chunk = degree.max(1);
    let softmax_rows_source = usize::max(1, lanes / BERT_SEQ);
    let softmax_source_slots = softmax_rows_source * BERT_SEQ;
    let activation_bound_bits = f.saturating_add(3);
    let shifted_frac_bits = f + BERT_SEQ.trailing_zeros();
    let shifted_bound_bits = shifted_frac_bits.saturating_add(4);
    let plain_bits = opts.packed_ahe_probe_plain_bits;

    let domain2 = RnsDyadicDomain::ntt_prime_limbs(plain_bits, 2, degree)?;
    let domain4 = RnsDyadicDomain::ntt_prime_limbs(plain_bits, 4, degree)?;
    let domain2_engines = domain2
        .moduli()
        .iter()
        .map(|&modulus| {
            build_packed_ahe_plain_modulus_engine(
                degree,
                modulus,
                opts.packed_ahe_probe_q_count,
                opts.packed_ahe_probe_q_bits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let domain4_engines = domain4
        .moduli()
        .iter()
        .map(|&modulus| {
            build_packed_ahe_plain_modulus_engine(
                degree,
                modulus,
                opts.packed_ahe_probe_q_count,
                opts.packed_ahe_probe_q_bits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let gelu_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::gelu_global_degree6_probe(chunk).coeffs,
        f,
        f,
        56,
        domain2.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modulus_margin_bits(10);
    let gelu_input = rns_shared_vector(lanes, fixed, &domain2, rng)?;
    let mut gelu_source =
        RnsRlweAheCrossTermOleTaylorSource::new(&domain2, domain2_engines.clone(), rng)?;
    let gelu =
        domain2.dyadic_of_pmpe_profiled_with_source(&gelu_input, &gelu_cfg, &mut gelu_source)?;
    let gelu_sample_stats = DirectionOperatorStats::from_profile(gelu.profile);
    let gelu_scaled_stats = gelu_sample_stats.scale_to(lanes, BERT_LAYERS * BERT_SEQ * BERT_FFN);

    let square_cfg = DyadicOfPmpePolynomialConfig::square(f, chunk)?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modulus_margin_bits(10);
    let square_input = rns_shared_vector(lanes, fixed, &domain2, rng)?;
    let mut square_source =
        RnsRlweAheCrossTermOleTaylorSource::new(&domain2, domain2_engines, rng)?;
    let square = domain2.dyadic_of_pmpe_profiled_with_source(
        &square_input,
        &square_cfg,
        &mut square_source,
    )?;
    let square_sample_stats = DirectionOperatorStats::from_profile(square.profile);
    let square_scaled_stats =
        square_sample_stats.scale_to(lanes, 2 * BERT_LAYERS * BERT_SEQ * BERT_HIDDEN);

    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &meanbeta_plan.coeffs,
        shifted_frac_bits,
        f,
        direction_exp_guard_bits(meanbeta_plan.metrics.degree, shifted_frac_bits, f),
        domain4.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(shifted_bound_bits)?
    .with_modulus_margin_bits(10);
    let softmax_input = rns_shared_vector(softmax_source_slots, fixed, &domain4, rng)?;
    let mut softmax_source =
        RnsRlweAheCrossTermOleTaylorSource::new(&domain4, domain4_engines, rng)?;
    let softmax_exp = domain4.softmax_mean_shift_of_pmpe_exp_profiled_with_source(
        &softmax_input,
        softmax_rows_source,
        BERT_SEQ,
        f,
        meanbeta_plan.metrics.beta,
        &exp_cfg,
        &mut softmax_source,
    )?;
    let softmax_sample_stats = DirectionOperatorStats::from_profile(softmax_exp.exp_profile);
    let softmax_scaled_stats = softmax_sample_stats.scale_to(
        softmax_source_slots,
        BERT_LAYERS * BERT_HEADS * BERT_SEQ * BERT_SEQ,
    );

    let mut sample_total = DirectionOperatorStats::default();
    sample_total.add_assign(gelu_sample_stats);
    sample_total.add_assign(square_sample_stats);
    sample_total.add_assign(softmax_sample_stats);

    let mut scaled_total = DirectionOperatorStats::default();
    scaled_total.add_assign(gelu_scaled_stats);
    scaled_total.add_assign(square_scaled_stats);
    scaled_total.add_assign(softmax_scaled_stats);

    let pass = sample_total.trusted_debug_offline_us == 0
        && sample_total.request_response_rounds == 0
        && sample_total.online_secret_secret_mul == 0
        && sample_total.secure_offline_us > 0;
    Ok((pass, sample_total, scaled_total))
}

fn run_hss_correlation_source_smoke(rng: &mut StdRng) -> Result<bool, Box<dyn std::error::Error>> {
    let domain = RnsDyadicDomain::from_moduli(vec![97, 193])?;
    let engines = vec![
        build_hss_plain_modulus_engine(8, 97)?,
        build_hss_plain_modulus_engine(8, 193)?,
    ];
    let cfg =
        DyadicOfPmpePolynomialConfig::from_integer_coeffs(vec![3, 2, 1], 0, 0, 0, 15, true, 8)?
            .with_input_abs_bound_bits(3)?
            .with_modulus_margin_bits(2);
    let values = vec![-2, -1, 0, 1, 2, 3];
    let input = domain.share_with_rng(&values, rng)?;
    let lhs_values = vec![2, -3, 4, -5];
    let rhs_values = vec![6, 7, -2, -3];
    let lhs = RnsScaledShareTensor {
        shares: domain.share_with_rng(&lhs_values, rng)?,
        frac_bits: 0,
        guard_bits: 0,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let rhs = RnsScaledShareTensor {
        shares: domain.share_with_rng(&rhs_values, rng)?,
        frac_bits: 0,
        guard_bits: 0,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let rescale_input = RnsScaledShareTensor {
        shares: domain.share_with_rng(&[16, 24, 40, 56], rng)?,
        frac_bits: 4,
        guard_bits: 4,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let mut source =
        RnsHssCorrelationSource::new(&domain, engines, rng)?.with_coin_tossed_rescale(2);
    let out = domain.dyadic_of_pmpe_profiled_with_source(&input, &cfg, &mut source)?;
    let expected = values
        .iter()
        .map(|&value| dyadic_integer_polynomial_value_i128_for_runner(value, &cfg.coeffs_num))
        .collect::<Vec<_>>();
    if out.tensor.shares.reconstruct_centered_i128()? != expected {
        return Ok(false);
    }

    let mut pack = source.beaver_mul_pack(&domain, lhs_values.len())?;
    let product = domain.mul_scaled_with_beaver_profiled(&lhs, &rhs, &mut pack)?;
    let expected_product = lhs_values
        .iter()
        .zip(rhs_values.iter())
        .map(|(&lhs, &rhs)| lhs * rhs)
        .collect::<Vec<_>>();
    let rescaled = domain.approximate_rescale_nonnegative_profiled_with_source(
        &rescale_input,
        2,
        4,
        8,
        &mut source,
    )?;
    Ok(
        product.tensor.shares.reconstruct_centered_i128()? == expected_product
            && rescaled
                .tensor
                .shares
                .reconstruct_nonnegative_i128_bounded(8)?
                == vec![4, 6, 10, 14]
            && out.profile.trusted_debug_offline_us == 0
            && product.profile.trusted_debug_offline_us == 0
            && rescaled.profile.trusted_debug_offline_us == 0,
    )
}

fn select_meanbeta_softmax_plan(
    rows: usize,
    cols: usize,
    dyadic_constraint: Option<(u32, u32, u32)>,
    rng: &mut StdRng,
) -> Result<MeanBetaSoftmaxPlan, Box<dyn std::error::Error>> {
    let scores = generate_attention_score_rows(rows, cols, rng);
    let beta_stats = max_minus_mean_percentiles(&scores, rows, cols);
    let beta_candidates = [
        ("p90", beta_stats[0]),
        ("p95", beta_stats[1]),
        ("p99", beta_stats[2]),
        ("p999", beta_stats[3]),
    ];
    let mut best: Option<MeanBetaSoftmaxPlan> = None;
    for &(beta_source, beta) in &beta_candidates {
        for domain_b in [6.0, 8.0, 10.0, 12.0] {
            for degree in 3..=7 {
                let coeffs = fit_exp_polynomial_on_negative_interval(degree, domain_b, 512)?;
                let metrics = evaluate_meanbeta_softmax_poly(
                    &scores, rows, cols, beta, domain_b, degree, &coeffs,
                )?;
                let valid = metrics.negative_rate == 0.0
                    && metrics.out_hi_rate <= 1e-3
                    && metrics.out_lo_rate <= 1e-2
                    && metrics.bad_rows == 0;
                if !valid {
                    continue;
                }
                if let Some((input_frac_bits, output_frac_bits, modulus_bits)) = dyadic_constraint {
                    let guard_bits =
                        direction_exp_guard_bits(degree, input_frac_bits, output_frac_bits);
                    if DyadicOfPmpePolynomialConfig::from_real_coeffs(
                        &coeffs,
                        input_frac_bits,
                        output_frac_bits,
                        guard_bits,
                        modulus_bits,
                        true,
                        cols,
                    )
                    .and_then(|cfg| {
                        cfg.with_input_abs_bound_bits(input_frac_bits.saturating_add(4))
                    })
                    .is_err()
                    {
                        continue;
                    }
                }
                let coeff_checksum = coeffs
                    .iter()
                    .fold(0u64, |acc, coeff| acc.wrapping_add(coeff.to_bits()));
                let candidate = MeanBetaSoftmaxPlan {
                    beta_source,
                    metrics,
                    coeffs,
                    coeff_checksum,
                };
                if best
                    .as_ref()
                    .map(|plan| candidate.metrics.avg_kl < plan.metrics.avg_kl)
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
            }
        }
    }
    best.ok_or_else(|| "no valid MeanBetaShift softmax plan found".into())
}

fn pass_fail(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

fn direction_exp_guard_bits(degree: usize, input_frac_bits: u32, output_frac_bits: u32) -> u32 {
    let required = (degree as u32)
        .saturating_mul(input_frac_bits)
        .saturating_sub(output_frac_bits)
        .saturating_add(16);
    required
        .max(104)
        .min(120u32.saturating_sub(output_frac_bits))
}

#[derive(Clone, Copy)]
struct BenchResult {
    avg_ms: f64,
    bytes_sent: u64,
    bytes_received: u64,
}

#[derive(Clone, Copy)]
struct PhaseTiming {
    online_ms: f64,
    preprocess_ms: f64,
}

fn run_nonlinear_benches(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    simd_chunk: usize,
    softmax_batch_rows: usize,
    gelu_lanes: usize,
    layernorm_batch_rows: usize,
    hss: &HssSlotEngine,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<NonlinearBenchSet, Box<dyn std::error::Error>> {
    let hss_mul_raw_4096 = bench_hss_mul_raw_slots(opts.iterations, simd_chunk, fixed, hss, rng)?;
    let hss_mul_4096 =
        bench_hss_mul_fixed_slots(opts.iterations, simd_chunk, fixed, nonlinear, rng)?;
    if opts.approx_trunc {
        let _ = bench_hss_mul_fixed_nonnegative_slots(
            opts.iterations,
            simd_chunk,
            fixed,
            nonlinear,
            rng,
        )?;
        let _ = bench_hss_mul_fixed_nonnegative_approx_slots(
            opts.iterations,
            simd_chunk,
            fixed,
            nonlinear,
            rng,
        )?;
    }
    let attention_case = format!("hss_dot_fixed_groups64_slots{simd_chunk}");
    let attention_dot_64 = bench_hss_dot_fixed_slots(
        opts.iterations,
        simd_chunk,
        BERT_HEAD_DIM,
        fixed,
        nonlinear,
        rng,
        &attention_case,
    )?;
    let context_case = format!("hss_dot_fixed_groups128_slots{simd_chunk}");
    let context_dot_128 = bench_hss_dot_fixed_slots(
        opts.iterations,
        simd_chunk,
        BERT_SEQ,
        fixed,
        nonlinear,
        rng,
        &context_case,
    )?;
    let softmax_128 = bench_softmax_row_128(opts.iterations, fixed, nonlinear, rng)?;
    let softmax_32x128 =
        bench_softmax_rows(opts.iterations, softmax_batch_rows, fixed, nonlinear, rng)?;
    let softmax_poly_32x128 = if opts.poly_softmax {
        match bench_softmax_rows_polynomial_exp(
            opts.iterations,
            softmax_batch_rows,
            opts.poly_softmax_squarings,
            fixed,
            nonlinear,
            rng,
        ) {
            Ok(ms) => Some(ms),
            Err(err)
                if matches!(
                    err.downcast_ref::<OperatorError>(),
                    Some(OperatorError::InvalidParams(_))
                ) =>
            {
                let case = format!("softmax_rows_polynomial_exp_{softmax_batch_rows}x{BERT_SEQ}");
                print_row(
                    "bert_probe",
                    &case,
                    1,
                    Duration::ZERO,
                    &format!("skipped=true|reason={}", extra_safe(&err.to_string())),
                );
                None
            }
            Err(err) => return Err(err),
        }
    } else {
        None
    };
    let softmax_approx_32x128 = if opts.approx_trunc {
        Some(bench_softmax_rows_approx_trunc(
            opts.iterations,
            softmax_batch_rows,
            fixed,
            nonlinear,
            rng,
        )?)
    } else {
        None
    };
    let softmax_specialized_32x128 = if opts.specialized_nonlinear {
        match bench_range_softmax_rows(
            opts.iterations,
            softmax_batch_rows,
            fixed,
            nonlinear,
            opts.approx_trunc,
            rng,
        ) {
            Ok(ms) => Some(ms),
            Err(err)
                if matches!(
                    err.downcast_ref::<OperatorError>(),
                    Some(OperatorError::InvalidParams(_))
                ) =>
            {
                let case = format!("range_softmax_rows_{softmax_batch_rows}x{BERT_SEQ}");
                print_row(
                    "bert_probe",
                    &case,
                    1,
                    Duration::ZERO,
                    &format!("skipped=true|reason={}", extra_safe(&err.to_string())),
                );
                None
            }
            Err(err) => return Err(err),
        }
    } else {
        None
    };
    if opts.pmpe_nonlinear {
        if opts.pmpe_dyadic {
            let _ = bench_softmax_rows_dyadic_ofpmpe_exp_phase(
                opts.iterations,
                softmax_batch_rows,
                fixed,
                nonlinear,
                opts.pmpe_chunk_lanes,
                rng,
            )?;
            let _ = bench_softmax_rows_dyadic_ofpmpe_full(
                opts.iterations,
                softmax_batch_rows,
                fixed,
                nonlinear,
                opts.pmpe_chunk_lanes,
                rng,
            )?;
        } else {
            let _ = bench_softmax_rows_ofpmpe_exp(
                opts.iterations,
                softmax_batch_rows,
                fixed,
                nonlinear,
                opts.approx_trunc,
                opts.pmpe_chunk_lanes,
                rng,
            )?;
        }
    }
    let layernorm_768 = bench_layernorm_row_768(opts.iterations, fixed, nonlinear, rng)?;
    let layernorm_5x768 =
        bench_layernorm_rows(opts.iterations, layernorm_batch_rows, fixed, nonlinear, rng)?;
    let layernorm_approx_5x768 = if opts.approx_trunc {
        Some(bench_layernorm_rows_approx_trunc(
            opts.iterations,
            layernorm_batch_rows,
            fixed,
            nonlinear,
            rng,
        )?)
    } else {
        None
    };
    let layernorm_specialized_5x768 = if opts.specialized_nonlinear {
        match bench_binade_layernorm_rows(
            opts.iterations,
            layernorm_batch_rows,
            fixed,
            nonlinear,
            opts.approx_trunc,
            rng,
        ) {
            Ok(ms) => Some(ms),
            Err(err)
                if matches!(
                    err.downcast_ref::<OperatorError>(),
                    Some(OperatorError::InvalidParams(_))
                ) =>
            {
                let case = format!("binade_layernorm_rows_{layernorm_batch_rows}x{BERT_HIDDEN}");
                print_row(
                    "bert_probe",
                    &case,
                    1,
                    Duration::ZERO,
                    &format!("skipped=true|reason={}", extra_safe(&err.to_string())),
                );
                None
            }
            Err(err) => return Err(err),
        }
    } else {
        None
    };
    let gelu = bench_gelu_slots(opts.iterations, gelu_lanes, fixed, nonlinear, rng)?;
    let gelu_specialized = if opts.specialized_nonlinear {
        Some(bench_fast_gelu_slots(
            opts.iterations,
            gelu_lanes,
            fixed,
            nonlinear,
            opts.approx_trunc,
            rng,
        )?)
    } else {
        None
    };
    if opts.pmpe_nonlinear {
        run_of_pmpe_benches(opts, fixed, nonlinear, rng)?;
    }

    Ok(NonlinearBenchSet {
        hss_mul_raw_4096,
        hss_mul_4096,
        attention_dot_64,
        context_dot_128,
        softmax_128,
        softmax_32x128,
        softmax_poly_32x128,
        softmax_approx_32x128,
        softmax_specialized_32x128,
        layernorm_768,
        layernorm_5x768,
        layernorm_approx_5x768,
        layernorm_specialized_5x768,
        gelu,
        gelu_specialized,
    })
}

fn bench_public_matmul_matrix(
    iterations: usize,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    fixed: FixedPointConfig,
    rng: &mut StdRng,
    case: &'static str,
) -> Result<f64, Box<dyn std::error::Error>> {
    let input = shared_vector(rows * input_dim, fixed, rng)?;
    let matrix = deterministic_dense_matrix_flat(input_dim, output_dim, fixed.modulus);

    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = input.matmul_public_row_major(rows, input_dim, &matrix, output_dim)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }

    print_row(
        "bert_probe",
        case,
        iterations,
        elapsed,
        &format!(
            "rows={rows}|input_dim={input_dim}|output_dim={output_dim}|weights=row_major_dense_deterministic|checksum={checksum}"
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_packed_private_linear_map_matrix(
    iterations: usize,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    fixed: FixedPointConfig,
    poly_degree: usize,
    crs_cache: &PackedLinearMapCrsCache,
    rng: &mut StdRng,
    case: &'static str,
) -> Result<PhaseTiming, Box<dyn std::error::Error>> {
    let linear_params = build_packed_linear_encryption_params(poly_degree, fixed.modulus)?;
    let ring = linear_params.ring.as_ref().clone();
    let cfg = PackedLinearMapConfig {
        input_dim,
        output_dim,
        ring,
        plaintext_modulus: fixed.modulus,
        noise_bound: 0,
    };
    let crs = PackedLinearMap::setup_cached(cfg, crs_cache, rng)?;
    let input = deterministic_plain_matrix_flat(rows, input_dim, fixed.modulus);
    let matrix = deterministic_dense_matrix_flat(input_dim, output_dim, fixed.modulus);

    let setup_start = Instant::now();
    let setup = PackedLinearMap::server_setup(&matrix, crs.as_ref(), rng)?;
    let setup_elapsed = setup_start.elapsed();
    let block_cols = crs.cfg.max_block_outputs();
    let blocks = output_dim.div_ceil(block_cols);
    let limb_count = crs.cfg.ring.rns().len();
    let poly_bytes = poly_degree as u64 * limb_count as u64 * 8;
    print_row(
        "bert_probe",
        "packed_private_linear_map_128x768_by_768x768_offline_setup",
        1,
        setup_elapsed,
        &format!(
            "phase=offline_preprocess|rows={rows}|input_dim={input_dim}|output_dim={output_dim}|poly_degree={poly_degree}|q0={}|limbs={limb_count}|blocks={blocks}|block_cols={block_cols}|server_preprocess_bytes={}|packed_crs_entries={}|weights=deterministic_dense",
            crs.cfg.q_modulus(),
            blocks as u64 * 2 * poly_bytes,
            crs_cache.len()?
        ),
    );

    let mut compute_elapsed = Duration::ZERO;
    let mut net_elapsed = Duration::ZERO;
    let mut link = RuntimeOneWayU64::tcp_loopback(SessionId(768130))?;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let query_start = Instant::now();
        let query = PackedLinearMap::client_query(&input, crs.as_ref(), rng)?;
        compute_elapsed += query_start.elapsed();

        let payload = query.online_payload_u64s();
        let net_start = Instant::now();
        link.send_client_to_server(TaskId(768131), &payload)?;
        net_elapsed += net_start.elapsed();

        let extract_start = Instant::now();
        let (party0_q, party1_q) = PackedLinearMap::extract_q_shares(&setup, &query, crs.as_ref())?;
        compute_elapsed += extract_start.elapsed();
        let q_shares = LinearMapQShares::new(
            party0_q,
            party1_q,
            crs.cfg.q_modulus(),
            crs.cfg.plaintext_modulus,
            crs.cfg.output_dim,
            query.rows(),
        )?;
        checksum = checksum.wrapping_add(
            q_shares
                .party0_q()
                .iter()
                .chain(q_shares.party1_q().iter())
                .fold(0u64, |acc, &v| {
                    add_mod(acc, v % fixed.modulus, fixed.modulus)
                }),
        );
    }

    print_row(
        "bert_probe",
        case,
        iterations,
        compute_elapsed,
        &format!(
            "phase=online|rows={rows}|input_dim={input_dim}|output_dim={output_dim}|poly_degree={poly_degree}|q0={}|limbs={limb_count}|blocks={blocks}|block_cols={block_cols}|client_query_bytes={}|offline_server_preprocess_bytes={}|offline_preprocess_ms={:.3}|packed_crs_entries={}|checksum={checksum}|model=real_packed_rlwe_linear_map",
            crs.cfg.q_modulus(),
            rows as u64 * poly_bytes,
            blocks as u64 * 2 * poly_bytes,
            setup_elapsed.as_secs_f64() * 1000.0,
            crs_cache.len()?
        ),
    );
    let stats = link.stats();
    let total_with_net = compute_elapsed + net_elapsed;
    let net_case = format!("{case}_tcp_loopback");
    print_row(
        "bert_probe",
        &net_case,
        iterations,
        total_with_net,
        &format!(
            "phase=online_with_transport|online_compute_ms={:.3}|net_ms={:.3}|offline_preprocess_ms={:.3}|frames_sent={}|frames_received={}|bytes_sent={}|bytes_received={}|model=real_packed_rlwe_linear_map_core_net",
            compute_elapsed.as_secs_f64() * 1000.0,
            net_elapsed.as_secs_f64() * 1000.0,
            setup_elapsed.as_secs_f64() * 1000.0,
            stats.frames_sent,
            stats.frames_received,
            stats.bytes_sent,
            stats.bytes_received
        ),
    );
    Ok(PhaseTiming {
        online_ms: avg_ms(compute_elapsed, iterations),
        preprocess_ms: setup_elapsed.as_secs_f64() * 1000.0,
    })
}

fn bench_packed_shared_matmul_matrix(
    iterations: usize,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    fixed: FixedPointConfig,
    poly_degree: usize,
    crs_cache: &PackedLinearMapCrsCache,
    rng: &mut StdRng,
    case: &'static str,
) -> Result<PhaseTiming, Box<dyn std::error::Error>> {
    let linear_params = build_packed_linear_encryption_params(poly_degree, fixed.modulus)?;
    let cfg = PackedLinearMapConfig {
        input_dim,
        output_dim,
        ring: linear_params.ring.as_ref().clone(),
        plaintext_modulus: fixed.modulus,
        noise_bound: 0,
    };
    let crs = PackedLinearMap::setup_cached(cfg, crs_cache, rng)?;
    let lhs_plain = deterministic_plain_matrix_flat(rows, input_dim, fixed.modulus);
    let rhs_plain = deterministic_dense_matrix_flat(input_dim, output_dim, fixed.modulus);
    let lhs = AdditiveShares::share_with_rng(&lhs_plain, fixed.modulus, rng)?;
    let rhs = AdditiveShares::share_with_rng(&rhs_plain, fixed.modulus, rng)?;

    let block_cols = crs.cfg.max_block_outputs();
    let blocks = output_dim.div_ceil(block_cols);
    let limb_count = crs.cfg.ring.rns().len();
    let poly_bytes = poly_degree as u64 * limb_count as u64 * 8;

    let mut preprocess_elapsed = Duration::ZERO;
    let mut online_elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let seed01 = rng.next_u64();
        let seed10 = rng.next_u64();
        let seed_q01_0 = rng.next_u64();
        let seed_q01_1 = rng.next_u64();
        let seed_q10_0 = rng.next_u64();
        let seed_q10_1 = rng.next_u64();
        let prep_start = Instant::now();
        let (setup_rhs1, setup_rhs0) = std::thread::scope(|scope| -> Result<_, String> {
            let setup01 = scope.spawn(|| -> Result<_, String> {
                let mut local_rng = StdRng::seed_from_u64(seed01);
                PackedLinearMap::server_setup(rhs.party1(), crs.as_ref(), &mut local_rng)
                    .map_err(|err| err.to_string())
            });
            let setup10 = scope.spawn(|| -> Result<_, String> {
                let mut local_rng = StdRng::seed_from_u64(seed10);
                PackedLinearMap::server_setup(rhs.party0(), crs.as_ref(), &mut local_rng)
                    .map_err(|err| err.to_string())
            });
            let setup_rhs1 = setup01
                .join()
                .map_err(|_| "packed shared matmul setup01 worker panicked".to_string())??;
            let setup_rhs0 = setup10
                .join()
                .map_err(|_| "packed shared matmul setup10 worker panicked".to_string())??;
            Ok((setup_rhs1, setup_rhs0))
        })
        .map_err(std::io::Error::other)?;
        preprocess_elapsed += prep_start.elapsed();

        let cross_start = Instant::now();
        let ((cross01_0, cross01_1), (cross10_0, cross10_1)) =
            std::thread::scope(|scope| -> Result<_, String> {
                let cross01 = scope.spawn(|| -> Result<_, String> {
                    let mut local_rng = StdRng::seed_from_u64(seed01);
                    let q_cross01 = PackedLinearMap::apply_q(
                        lhs.party0(),
                        &setup_rhs1,
                        crs.as_ref(),
                        &mut local_rng,
                    )
                    .map_err(|err| err.to_string())?;
                    let mut q_rng0 = StdRng::seed_from_u64(seed_q01_0);
                    let mut q_rng1 = StdRng::seed_from_u64(seed_q01_1);
                    ShareConverter::convert_pair_local(
                        q_cross01.party0_q(),
                        q_cross01.party1_q(),
                        q_cross01.q_modulus(),
                        q_cross01.p_modulus(),
                        &mut q_rng0,
                        &mut q_rng1,
                    )
                    .map_err(|err| err.to_string())
                });
                let cross10 = scope.spawn(|| -> Result<_, String> {
                    let mut local_rng = StdRng::seed_from_u64(seed10);
                    let q_cross10 = PackedLinearMap::apply_q(
                        lhs.party1(),
                        &setup_rhs0,
                        crs.as_ref(),
                        &mut local_rng,
                    )
                    .map_err(|err| err.to_string())?;
                    let mut q_rng0 = StdRng::seed_from_u64(seed_q10_0);
                    let mut q_rng1 = StdRng::seed_from_u64(seed_q10_1);
                    ShareConverter::convert_pair_local(
                        q_cross10.party0_q(),
                        q_cross10.party1_q(),
                        q_cross10.q_modulus(),
                        q_cross10.p_modulus(),
                        &mut q_rng0,
                        &mut q_rng1,
                    )
                    .map_err(|err| err.to_string())
                });
                let cross01 = cross01
                    .join()
                    .map_err(|_| "packed shared matmul cross01 worker panicked".to_string())??;
                let cross10 = cross10
                    .join()
                    .map_err(|_| "packed shared matmul cross10 worker panicked".to_string())??;
                Ok((cross01, cross10))
            })
            .map_err(std::io::Error::other)?;
        online_elapsed += cross_start.elapsed();

        let local_start = Instant::now();
        let local0 = plain_matmul_output_major(
            lhs.party0(),
            rhs.party0(),
            rows,
            input_dim,
            output_dim,
            fixed.modulus,
        );
        let local1 = plain_matmul_output_major(
            lhs.party1(),
            rhs.party1(),
            rows,
            input_dim,
            output_dim,
            fixed.modulus,
        );
        online_elapsed += local_start.elapsed();

        checksum = checksum.wrapping_add(
            local0
                .iter()
                .zip(local1.iter())
                .zip(cross01_0.iter().zip(cross01_1.iter()))
                .zip(cross10_0.iter().zip(cross10_1.iter()))
                .fold(
                    0u64,
                    |acc, (((&l0, &l1), (&c010, &c011)), (&c100, &c101))| {
                        let value = add_mod(
                            add_mod(
                                add_mod(l0, l1, fixed.modulus),
                                add_mod(c010, c011, fixed.modulus),
                                fixed.modulus,
                            ),
                            add_mod(c100, c101, fixed.modulus),
                            fixed.modulus,
                        );
                        add_mod(acc, value, fixed.modulus)
                    },
                ),
        );
    }

    print_row(
        "bert_probe",
        &format!("{case}_offline_preprocess"),
        iterations,
        preprocess_elapsed,
        &format!(
            "phase=offline_preprocess|rows={rows}|input_dim={input_dim}|output_dim={output_dim}|poly_degree={poly_degree}|q0={}|limbs={limb_count}|blocks={blocks}|block_cols={block_cols}|cross_protocols=2|dynamic_server_preprocess_bytes={}|packed_crs_entries={}|model=real_packed_rlwe_shared_matmul",
            crs.cfg.q_modulus(),
            2 * blocks as u64 * 2 * poly_bytes,
            crs_cache.len()?
        ),
    );
    print_row(
        "bert_probe",
        case,
        iterations,
        online_elapsed,
        &format!(
            "phase=online|rows={rows}|input_dim={input_dim}|output_dim={output_dim}|poly_degree={poly_degree}|q0={}|limbs={limb_count}|blocks={blocks}|block_cols={block_cols}|cross_protocols=2|client_query_bytes={}|offline_dynamic_server_preprocess_bytes={}|offline_preprocess_ms={:.3}|packed_crs_entries={}|q_to_p=masked_open_local|checksum={checksum}|model=real_packed_rlwe_shared_matmul",
            crs.cfg.q_modulus(),
            2 * rows as u64 * poly_bytes,
            2 * blocks as u64 * 2 * poly_bytes,
            preprocess_elapsed.as_secs_f64() * 1000.0,
            crs_cache.len()?
        ),
    );
    Ok(PhaseTiming {
        online_ms: avg_ms(online_elapsed, iterations),
        preprocess_ms: avg_ms(preprocess_elapsed, iterations),
    })
}

fn bench_hss_mul_raw_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    hss: &HssSlotEngine,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let lhs = shared_vector(lanes, fixed, rng)?;
    let rhs = shared_vector(lanes, fixed, rng)?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = hss.mul_slots(&lhs, &rhs, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("hss_mul_raw_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!("lanes={lanes}|rescale=none|checksum={checksum}"),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_hss_mul_fixed_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let lhs = shared_vector(lanes, fixed, rng)?;
    let rhs = shared_vector(lanes, fixed, rng)?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.mul_fixed(&lhs, &rhs, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("hss_mul_fixed_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!("lanes={lanes}|rescale=lookup|checksum={checksum}"),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_hss_mul_fixed_nonnegative_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let input = shared_vector(lanes, fixed, rng)?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.mul_fixed_nonnegative(&input, &input, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("hss_mul_fixed_nonnegative_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!("lanes={lanes}|rescale=known_msb_exact|checksum={checksum}"),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_hss_mul_fixed_nonnegative_approx_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let input = shared_vector(lanes, fixed, rng)?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.mul_fixed_nonnegative_approx(&input, &input, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("hss_mul_fixed_nonnegative_approx_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!("lanes={lanes}|rescale=known_msb_approx|checksum={checksum}"),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_hss_dot_fixed_slots(
    iterations: usize,
    lanes: usize,
    group_size: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
    case: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    let lhs = shared_vector(lanes, fixed, rng)?;
    let rhs = shared_vector(lanes, fixed, rng)?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.dot_fixed(&lhs, &rhs, group_size, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    print_row(
        "bert_probe",
        case,
        iterations,
        elapsed,
        &format!(
            "lanes={lanes}|group_size={group_size}|groups={}|rescale=after_group_sum|checksum={checksum}",
            lanes / group_size
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_softmax_row_128(
    iterations: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let row = shared_vector(BERT_SEQ, fixed, rng)?;
    let cfg = SoftmaxConfig::default();
    let mut elapsed = Duration::ZERO;
    let mut last_sum = 0.0;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.softmax(&row, &cfg, rng)?;
        elapsed += start.elapsed();
        last_sum = out
            .reconstruct()
            .iter()
            .map(|&v| fixed.decode_f64(v))
            .sum::<f64>();
    }
    print_row(
        "bert_probe",
        "softmax_row_128",
        iterations,
        elapsed,
        &format!("row_len={BERT_SEQ}|sum={last_sum:.4}"),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_softmax_rows(
    iterations: usize,
    rows: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cols = BERT_SEQ;
    let batch = shared_vector(rows * cols, fixed, rng)?;
    let cfg = SoftmaxConfig::default();
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.softmax_rows(&batch, rows, cols, &cfg, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("softmax_rows_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "rows={rows}|cols={cols}|slots={}|checksum={checksum}",
            rows * cols
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_softmax_rows_approx_trunc(
    iterations: usize,
    rows: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cols = BERT_SEQ;
    let batch = shared_vector(rows * cols, fixed, rng)?;
    let cfg = SoftmaxConfig::default();
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.softmax_rows_approx_trunc(&batch, rows, cols, &cfg, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("softmax_rows_approx_trunc_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "rows={rows}|cols={cols}|slots={}|trunc=known_msb_approx|checksum={checksum}",
            rows * cols
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_softmax_rows_polynomial_exp(
    iterations: usize,
    rows: usize,
    squarings: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cols = BERT_SEQ;
    let batch = shared_vector(rows * cols, fixed, rng)?;
    let cfg = SoftmaxConfig::default();
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out =
            nonlinear.softmax_rows_polynomial_exp(&batch, rows, cols, &cfg, squarings, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("softmax_rows_polynomial_exp_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "rows={rows}|cols={cols}|slots={}|exp_squarings={squarings}|checksum={checksum}",
            rows * cols
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_range_softmax_rows(
    iterations: usize,
    rows: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    approximate_truncation: bool,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cols = BERT_SEQ;
    let batch = shared_vector(rows * cols, fixed, rng)?;
    let cfg = RangeSoftmaxConfig {
        approximate_truncation,
        ..RangeSoftmaxConfig::calibrated_unit_interval()
    };
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut profile = NonlinearProfile::default();
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.range_softmax_rows_profiled(&batch, rows, cols, &cfg, rng)?;
        elapsed += start.elapsed();
        profile.selectors += out.profile.selectors;
        profile.poly_mul += out.profile.poly_mul;
        profile.trunc += out.profile.trunc;
        profile.refresh += out.profile.refresh;
        profile.online_ms += out.profile.online_ms;
        profile.transport_bytes += out.profile.transport_bytes;
        checksum = checksum.wrapping_add(
            out.shares
                .reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("range_softmax_rows_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=sac_range_softmax|rows={rows}|cols={cols}|slots={}|{}|exp_clip_min={:.3}|exp_clip_max={:.3}|use_row_max={}|public_input_shift={:.3}|approx_trunc={}|checksum={checksum}",
            rows * cols,
            profile_extra(profile),
            cfg.exp_clip_min,
            cfg.exp_clip_max,
            cfg.use_row_max,
            cfg.public_input_shift,
            cfg.approximate_truncation
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_softmax_rows_ofpmpe_exp(
    iterations: usize,
    rows: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    approximate_truncation: bool,
    chunk_lanes: usize,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cols = BERT_SEQ;
    let input = shared_vector(rows * cols, fixed, rng)?;
    let exp_cfg = OfPmpePolynomialConfig::exp_negative_degree6_probe(chunk_lanes.max(1));
    let cfg = SoftmaxConfig::default();
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut exp_profile = OfPmpeProfile::default();
    let mut operator_online_ms = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.softmax_rows_of_pmpe_exp_profiled(
            &input,
            rows,
            cols,
            &cfg,
            &exp_cfg,
            approximate_truncation,
            rng,
        )?;
        elapsed += start.elapsed();
        operator_online_ms += out.online_ms;
        exp_profile = out.exp_profile;
        checksum = checksum.wrapping_add(
            out.shares
                .reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("softmax_rows_ofpmpe_exp_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=softmax_existing_max_ofpmpe_exp_existing_recip|rows={rows}|cols={cols}|approx_trunc={approximate_truncation}|operator_online_ms_avg={:.3}|exp_{}|checksum={checksum}",
            operator_online_ms as f64 / iterations as f64,
            of_pmpe_profile_extra(exp_profile)
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_softmax_rows_dyadic_ofpmpe_exp_phase(
    iterations: usize,
    rows: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    chunk_lanes: usize,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cols = BERT_SEQ;
    let input = shared_vector(rows * cols, fixed, rng)?;
    let f = frac_bits_from_scale(fixed.scale)?;
    let activation_bound_bits = f.saturating_add(3);
    let modulus_bits = modulus_bit_width_for_runner(fixed.modulus);
    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::exp_negative_degree6_probe(chunk_lanes.max(1)).coeffs,
        f,
        f,
        56,
        modulus_bits,
        true,
        chunk_lanes.max(1),
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modular_wrap_allowed(true);
    let cfg = SoftmaxConfig::default();
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut exp_profile = OfPmpeProfile::default();
    let mut operator_online_ms = 0u64;
    let mut scale_profile = None;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.softmax_rows_dyadic_of_pmpe_exp_phase_profiled(
            &input, rows, cols, &cfg, &exp_cfg, rng,
        )?;
        elapsed += start.elapsed();
        operator_online_ms += out.online_ms;
        exp_profile = out.exp_profile;
        scale_profile = Some(out.exp_scale_profile);
        checksum = checksum.wrapping_add(
            out.row_sums
                .shares
                .reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let scale_profile = scale_profile.ok_or("missing dyadic softmax exp scale profile")?;
    let case = format!("softmax_rows_dyadic_ofpmpe_exp_phase_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=softmax_existing_max_dyadic_ofpmpe_exp_highscale_sum|rows={rows}|cols={cols}|operator_online_ms_avg={:.3}|input_frac_bits={}|output_frac_bits={}|guard_bits={}|effective_output_frac_bits={}|input_abs_bound_bits={}|estimated_output_bound_bits={}|modulus_bits={}|modulus_margin_bits={}|modulus_wrap_safe={}|allow_modular_wrap={}|exp_{}|row_sum_checksum={checksum}",
            operator_online_ms as f64 / iterations as f64,
            scale_profile.input_frac_bits,
            scale_profile.output_frac_bits,
            scale_profile.guard_bits,
            scale_profile.effective_output_frac_bits,
            format_opt_u32(scale_profile.input_abs_bound_bits),
            format_opt_u32(scale_profile.output_bound_bits),
            scale_profile.modulus_bits,
            scale_profile.modulus_margin_bits,
            format_opt_bool(scale_profile.modulus_wrap_safe),
            exp_cfg.allow_modular_wrap,
            of_pmpe_profile_extra(exp_profile)
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_softmax_rows_dyadic_ofpmpe_full(
    iterations: usize,
    rows: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    chunk_lanes: usize,
    rng: &mut StdRng,
) -> Result<Option<f64>, Box<dyn std::error::Error>> {
    let cols = BERT_SEQ;
    let f = frac_bits_from_scale(fixed.scale)?;
    let activation_bound_bits = f.saturating_add(3);
    let modulus_bits = modulus_bit_width_for_runner(fixed.modulus);
    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::exp_negative_degree6_probe(chunk_lanes.max(1)).coeffs,
        f,
        f,
        56,
        modulus_bits,
        true,
        chunk_lanes.max(1),
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modular_wrap_allowed(true);
    let modulus_wrap_safe = exp_cfg.modulus_wrap_safe(modulus_bits);
    let case = format!("softmax_rows_dyadic_ofpmpe_full_{rows}x{cols}");
    if modulus_wrap_safe == Some(false) {
        print_row(
            "bert_probe",
            &case,
            iterations,
            Duration::ZERO,
            &format!(
                "path=softmax_existing_max_dyadic_ofpmpe_full|skipped=true|reason=dyadic_exp_denominator_not_nowrap|rows={rows}|cols={cols}|input_frac_bits={f}|denominator_frac_bits={f}|prob_frac_bits=deferred|guard_bits={}|estimated_output_bound_bits={}|modulus_bits={modulus_bits}|modulus_margin_bits={}|modulus_wrap_safe={}|allow_modular_wrap={}",
                exp_cfg.guard_bits,
                format_opt_u32(exp_cfg.output_bound_bits),
                exp_cfg.modulus_margin_bits,
                format_opt_bool(modulus_wrap_safe),
                exp_cfg.allow_modular_wrap,
            ),
        );
        return Ok(None);
    }

    let cfg = SoftmaxConfig::default();
    let input = shared_vector(rows * cols, fixed, rng)?;
    let prob_frac_bits = f
        .checked_mul(2)
        .ok_or("softmax probability frac_bits overflow")?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut operator_online_us = 0u64;
    let mut reciprocal_us = 0u64;
    let mut rescale_us = 0u64;
    let mut broadcast_mul_us = 0u64;
    let mut exp_profile = OfPmpeProfile::default();
    let mut broadcast_shift_bits = 0u32;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.softmax_rows_dyadic_of_pmpe_profiled(
            &input,
            rows,
            cols,
            &cfg,
            &exp_cfg,
            f,
            prob_frac_bits,
            rng,
        )?;
        elapsed += start.elapsed();
        operator_online_us = operator_online_us.saturating_add(out.online_us);
        reciprocal_us = reciprocal_us.saturating_add(out.reciprocal_phase.reciprocal_us);
        rescale_us = rescale_us.saturating_add(out.reciprocal_phase.rescale.online_us);
        broadcast_mul_us = broadcast_mul_us.saturating_add(out.broadcast_mul.online_us);
        broadcast_shift_bits = out.broadcast_mul.shift_bits;
        exp_profile = out.reciprocal_phase.exp_phase.exp_profile;
        checksum = checksum.wrapping_add(
            out.probs
                .shares
                .reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=softmax_existing_max_dyadic_ofpmpe_full|skipped=false|rows={rows}|cols={cols}|operator_online_us_avg={:.3}|exp_frac_bits={}|denominator_frac_bits={f}|prob_frac_bits={prob_frac_bits}|broadcast_shift_bits={broadcast_shift_bits}|reciprocal_us_avg={:.3}|denom_rescale_us_avg={:.3}|broadcast_mul_us_avg={:.3}|modulus_bits={modulus_bits}|modulus_wrap_safe={}|allow_modular_wrap={}|exp_{}|checksum={checksum}",
            operator_online_us as f64 / iterations as f64,
            exp_cfg.output_frac_bits_with_guard()?,
            reciprocal_us as f64 / iterations as f64,
            rescale_us as f64 / iterations as f64,
            broadcast_mul_us as f64 / iterations as f64,
            format_opt_bool(modulus_wrap_safe),
            exp_cfg.allow_modular_wrap,
            of_pmpe_profile_extra(exp_profile),
        ),
    );
    Ok(Some(avg_ms(elapsed, iterations)))
}

fn bench_layernorm_row_768(
    iterations: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let row = shared_vector(BERT_HIDDEN, fixed, rng)?;
    let cfg = LayerNormConfig {
        epsilon: 0.25,
        variance_clip_max: 4.0,
        gamma: vec![1.0; BERT_HIDDEN],
        beta: vec![0.0; BERT_HIDDEN],
    };
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.layer_norm(&row, &cfg, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    print_row(
        "bert_probe",
        "layernorm_row_768",
        iterations,
        elapsed,
        &format!("row_len={BERT_HIDDEN}|epsilon=0.25|checksum={checksum}"),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_layernorm_rows(
    iterations: usize,
    rows: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cols = BERT_HIDDEN;
    let batch = shared_vector(rows * cols, fixed, rng)?;
    let cfg = LayerNormConfig {
        epsilon: 0.25,
        variance_clip_max: 4.0,
        gamma: vec![1.0; cols],
        beta: vec![0.0; cols],
    };
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.layer_norm_rows(&batch, rows, cols, &cfg, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("layernorm_rows_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "rows={rows}|cols={cols}|slots={}|epsilon=0.25|checksum={checksum}",
            rows * cols
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_layernorm_rows_approx_trunc(
    iterations: usize,
    rows: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cols = BERT_HIDDEN;
    let batch = shared_vector(rows * cols, fixed, rng)?;
    let cfg = LayerNormConfig {
        epsilon: 0.25,
        variance_clip_max: 4.0,
        gamma: vec![1.0; cols],
        beta: vec![0.0; cols],
    };
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.layer_norm_rows_approx_trunc(&batch, rows, cols, &cfg, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("layernorm_rows_approx_trunc_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "rows={rows}|cols={cols}|slots={}|epsilon=0.25|trunc=known_msb_approx|checksum={checksum}",
            rows * cols
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_binade_layernorm_rows(
    iterations: usize,
    rows: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    approximate_truncation: bool,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cols = BERT_HIDDEN;
    let batch = shared_vector(rows * cols, fixed, rng)?;
    let cfg = BinadeLayerNormConfig {
        epsilon: 0.25,
        variance_clip_max: 4.0,
        gamma: vec![1.0; cols],
        beta: vec![0.0; cols],
        approximate_truncation,
        ..BinadeLayerNormConfig::default()
    };
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut profile = NonlinearProfile::default();
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.binade_layer_norm_rows_profiled(&batch, rows, cols, &cfg, rng)?;
        elapsed += start.elapsed();
        profile.selectors += out.profile.selectors;
        profile.poly_mul += out.profile.poly_mul;
        profile.trunc += out.profile.trunc;
        profile.refresh += out.profile.refresh;
        profile.online_ms += out.profile.online_ms;
        profile.transport_bytes += out.profile.transport_bytes;
        checksum = checksum.wrapping_add(
            out.shares
                .reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("binade_layernorm_rows_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=sac_binade_layernorm|rows={rows}|cols={cols}|slots={}|{}|epsilon=0.25|binades={}|approx_trunc={}|checksum={checksum}",
            rows * cols,
            profile_extra(profile),
            cfg.binade_thresholds.len() + 1,
            cfg.approximate_truncation
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_gelu_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let input = shared_vector(lanes, fixed, rng)?;
    let cfg = GeluConfig::default();
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.gelu(&input, &cfg, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("gelu_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!("lanes={lanes}|piecewise_mid_poly_lut=true|checksum={checksum}"),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_fast_gelu_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    approximate_truncation: bool,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let input = shared_vector(lanes, fixed, rng)?;
    let cfg = FastGeluConfig {
        approximate_truncation,
        ..FastGeluConfig::calibrated_unit_quadratic()
    };
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut profile = NonlinearProfile::default();
    for _ in 0..iterations {
        let start = Instant::now();
        let out = nonlinear.fast_gelu_profiled(&input, &cfg, rng)?;
        elapsed += start.elapsed();
        profile.selectors += out.profile.selectors;
        profile.poly_mul += out.profile.poly_mul;
        profile.trunc += out.profile.trunc;
        profile.refresh += out.profile.refresh;
        profile.online_ms += out.profile.online_ms;
        profile.transport_bytes += out.profile.transport_bytes;
        checksum = checksum.wrapping_add(
            out.shares
                .reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let case = format!("fast_gelu_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=sac_fast_gelu|lanes={lanes}|{}|clip_bound={:.3}|middle_intervals={}|approx_trunc={}|checksum={checksum}",
            profile_extra(profile),
            cfg.clip_bound,
            cfg.middle_thresholds.len() + 1,
            cfg.approximate_truncation
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn run_of_pmpe_benches(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<(), Box<dyn std::error::Error>> {
    if opts.pmpe_bert_shapes {
        return run_of_pmpe_bert_shape_benches(opts, fixed, nonlinear, rng);
    }
    if opts.pmpe_rns {
        return run_rns_dyadic_of_pmpe_benches(opts, fixed, rng);
    }
    let lanes = opts.pmpe_lanes.max(1);
    if opts.pmpe_dyadic {
        return run_dyadic_of_pmpe_benches(opts, fixed, nonlinear, rng);
    }
    bench_of_pmpe_poly_slots(
        opts.iterations,
        lanes,
        fixed,
        nonlinear,
        "of_pmpe_poly_deg3",
        &[0.25, -0.5, 0.125, 0.03125],
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        opts.pmpe_chunk_lanes,
        rng,
    )?;
    bench_of_pmpe_poly_slots(
        opts.iterations,
        lanes,
        fixed,
        nonlinear,
        "of_pmpe_poly_deg6",
        &[0.05, 0.5, 0.25, -0.125, 0.03125, -0.0078125, 0.001953125],
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        opts.pmpe_chunk_lanes,
        rng,
    )?;
    bench_of_pmpe_poly_slots(
        opts.iterations,
        lanes,
        fixed,
        nonlinear,
        "of_pmpe_gelu_calibrated_deg2",
        &[0.00489279750771025, 0.5, 0.348188960031138],
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        opts.pmpe_chunk_lanes,
        rng,
    )?;
    bench_of_pmpe_poly_slots(
        opts.iterations,
        lanes,
        fixed,
        nonlinear,
        "of_pmpe_exp_taylor_deg5",
        &[1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0],
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        opts.pmpe_chunk_lanes,
        rng,
    )?;
    bench_of_pmpe_poly_slots(
        opts.iterations,
        lanes,
        fixed,
        nonlinear,
        "of_pmpe_square",
        &[0.0, 0.0, 1.0],
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        opts.pmpe_chunk_lanes,
        rng,
    )?;
    Ok(())
}

fn run_of_pmpe_bert_shape_benches(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<(), Box<dyn std::error::Error>> {
    let gelu_points = BERT_LAYERS * BERT_SEQ * BERT_FFN;
    let softmax_exp_points = BERT_LAYERS * BERT_HEADS * BERT_SEQ * BERT_SEQ;
    let layernorm_square_points = 2 * BERT_LAYERS * BERT_SEQ * BERT_HIDDEN;
    if opts.pmpe_rns {
        return run_rns_dyadic_of_pmpe_bert_shape_benches(
            opts,
            fixed,
            rng,
            gelu_points,
            softmax_exp_points,
            layernorm_square_points,
        );
    }
    if opts.pmpe_dyadic {
        return run_dyadic_of_pmpe_bert_shape_benches(
            opts,
            fixed,
            nonlinear,
            rng,
            gelu_points,
            softmax_exp_points,
            layernorm_square_points,
        );
    }
    bench_of_pmpe_poly_slots(
        opts.iterations,
        gelu_points,
        fixed,
        nonlinear,
        "bert_shape_gelu_of_pmpe_poly_deg6",
        &[0.05, 0.5, 0.25, -0.125, 0.03125, -0.0078125, 0.001953125],
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        opts.pmpe_chunk_lanes,
        rng,
    )?;
    bench_of_pmpe_poly_slots(
        opts.iterations,
        softmax_exp_points,
        fixed,
        nonlinear,
        "bert_shape_softmax_exp_of_pmpe_deg5",
        &[1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0],
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        opts.pmpe_chunk_lanes,
        rng,
    )?;
    bench_of_pmpe_poly_slots(
        opts.iterations,
        layernorm_square_points,
        fixed,
        nonlinear,
        "bert_shape_layernorm_square_of_pmpe",
        &[0.0, 0.0, 1.0],
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        opts.pmpe_chunk_lanes,
        rng,
    )?;
    Ok(())
}

fn run_dyadic_of_pmpe_benches(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
) -> Result<(), Box<dyn std::error::Error>> {
    let f = frac_bits_from_scale(fixed.scale)?;
    let lanes = opts.pmpe_lanes.max(1);
    let modulus_bits = modulus_bit_width_for_runner(fixed.modulus);
    let activation_bound_bits = f.saturating_add(3);
    let chunk = opts.pmpe_chunk_lanes.max(1);
    let cases = [
        (
            "dyadic_of_pmpe_poly_deg3",
            vec![0.25, -0.5, 0.125, 0.03125],
            f,
            32,
        ),
        (
            "dyadic_of_pmpe_poly_deg6",
            vec![0.05, 0.5, 0.25, -0.125, 0.03125, -0.0078125, 0.001953125],
            f,
            56,
        ),
        (
            "dyadic_of_pmpe_gelu_deg6",
            OfPmpePolynomialConfig::gelu_global_degree6_probe(chunk).coeffs,
            f,
            56,
        ),
        (
            "dyadic_of_pmpe_exp_deg6",
            OfPmpePolynomialConfig::exp_negative_degree6_probe(chunk).coeffs,
            f,
            56,
        ),
    ];
    for (case, coeffs, output_frac_bits, guard_bits) in cases {
        let cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
            &coeffs,
            f,
            output_frac_bits,
            guard_bits,
            modulus_bits,
            true,
            chunk,
        )?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modular_wrap_allowed(true);
        bench_dyadic_of_pmpe_slots(
            opts.iterations,
            lanes,
            fixed,
            nonlinear,
            case,
            cfg,
            opts.pmpe_runtime_net,
            opts.pmpe_streaming,
            rng,
        )?;
    }
    let square_cfg = DyadicOfPmpePolynomialConfig::square(f, chunk)?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modular_wrap_allowed(true);
    bench_dyadic_of_pmpe_slots(
        opts.iterations,
        lanes,
        fixed,
        nonlinear,
        "dyadic_of_pmpe_square",
        square_cfg,
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        rng,
    )?;
    Ok(())
}

fn run_dyadic_of_pmpe_bert_shape_benches(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    rng: &mut StdRng,
    gelu_points: usize,
    softmax_exp_points: usize,
    layernorm_square_points: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let f = frac_bits_from_scale(fixed.scale)?;
    let modulus_bits = modulus_bit_width_for_runner(fixed.modulus);
    let activation_bound_bits = f.saturating_add(3);
    let chunk = opts.pmpe_chunk_lanes.max(1);
    let gelu_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::gelu_global_degree6_probe(chunk).coeffs,
        f,
        f,
        56,
        modulus_bits,
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modular_wrap_allowed(true);
    bench_dyadic_of_pmpe_slots(
        opts.iterations,
        gelu_points,
        fixed,
        nonlinear,
        "bert_shape_dyadic_gelu_of_pmpe_deg6",
        gelu_cfg,
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        rng,
    )?;
    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::exp_negative_degree6_probe(chunk).coeffs,
        f,
        f,
        56,
        modulus_bits,
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modular_wrap_allowed(true);
    bench_dyadic_of_pmpe_slots(
        opts.iterations,
        softmax_exp_points,
        fixed,
        nonlinear,
        "bert_shape_dyadic_softmax_exp_of_pmpe_deg6",
        exp_cfg,
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        rng,
    )?;
    let square_cfg = DyadicOfPmpePolynomialConfig::square(f, chunk)?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modular_wrap_allowed(true);
    bench_dyadic_of_pmpe_slots(
        opts.iterations,
        layernorm_square_points,
        fixed,
        nonlinear,
        "bert_shape_dyadic_layernorm_square_of_pmpe",
        square_cfg,
        opts.pmpe_runtime_net,
        opts.pmpe_streaming,
        rng,
    )?;
    Ok(())
}

fn run_rns_dyadic_of_pmpe_benches(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    rng: &mut StdRng,
) -> Result<(), Box<dyn std::error::Error>> {
    let f = frac_bits_from_scale(fixed.scale)?;
    let lanes = opts.pmpe_lanes.max(1);
    let activation_bound_bits = f.saturating_add(3);
    let chunk = opts.pmpe_chunk_lanes.max(1);
    let domain = RnsDyadicDomain::two_limb_61bit()?;
    let cases = [
        (
            "rns_dyadic_of_pmpe_poly_deg3",
            vec![0.25, -0.5, 0.125, 0.03125],
            f,
            32,
        ),
        (
            "rns_dyadic_of_pmpe_poly_deg6",
            vec![0.05, 0.5, 0.25, -0.125, 0.03125, -0.0078125, 0.001953125],
            f,
            56,
        ),
        (
            "rns_dyadic_of_pmpe_gelu_deg6",
            OfPmpePolynomialConfig::gelu_global_degree6_probe(chunk).coeffs,
            f,
            56,
        ),
        (
            "rns_dyadic_of_pmpe_exp_deg6",
            OfPmpePolynomialConfig::exp_negative_degree6_probe(chunk).coeffs,
            f,
            56,
        ),
    ];
    for (case, coeffs, output_frac_bits, guard_bits) in cases {
        let cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
            &coeffs,
            f,
            output_frac_bits,
            guard_bits,
            domain.modulus_bits(),
            true,
            chunk,
        )?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modulus_margin_bits(10);
        bench_rns_dyadic_of_pmpe_slots(opts.iterations, lanes, fixed, &domain, case, cfg, rng)?;
    }
    let square_cfg = DyadicOfPmpePolynomialConfig::square(f, chunk)?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modulus_margin_bits(10);
    bench_rns_dyadic_of_pmpe_slots(
        opts.iterations,
        lanes,
        fixed,
        &domain,
        "rns_dyadic_of_pmpe_square",
        square_cfg,
        rng,
    )?;
    let mean_shift_rows = usize::max(1, lanes / BERT_SEQ);
    bench_rns_mean_shift_softmax_exp_phase(
        opts.iterations,
        mean_shift_rows,
        BERT_SEQ,
        fixed,
        opts.pmpe_chunk_lanes.max(1),
        0.0,
        rng,
    )?;
    bench_rns_mean_shift_softmax_full_probe(
        opts.iterations,
        mean_shift_rows,
        BERT_SEQ,
        fixed,
        opts.pmpe_chunk_lanes.max(1),
        0.0,
        rng,
    )?;
    bench_rns_beaver_mul_slots(opts.iterations, lanes, fixed, 4, rng)?;
    Ok(())
}

fn run_rns_dyadic_of_pmpe_bert_shape_benches(
    opts: &CliOptions,
    fixed: FixedPointConfig,
    rng: &mut StdRng,
    gelu_points: usize,
    softmax_exp_points: usize,
    layernorm_square_points: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let f = frac_bits_from_scale(fixed.scale)?;
    let activation_bound_bits = f.saturating_add(3);
    let chunk = opts.pmpe_chunk_lanes.max(1);
    let domain = RnsDyadicDomain::two_limb_61bit()?;
    let gelu_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::gelu_global_degree6_probe(chunk).coeffs,
        f,
        f,
        56,
        domain.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modulus_margin_bits(10);
    bench_rns_dyadic_of_pmpe_slots(
        opts.iterations,
        gelu_points,
        fixed,
        &domain,
        "bert_shape_rns_dyadic_gelu_of_pmpe_deg6",
        gelu_cfg,
        rng,
    )?;
    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::exp_negative_degree6_probe(chunk).coeffs,
        f,
        f,
        56,
        domain.modulus_bits(),
        true,
        chunk,
    )?
    .with_input_abs_bound_bits(activation_bound_bits)?
    .with_modulus_margin_bits(10);
    bench_rns_dyadic_of_pmpe_slots(
        opts.iterations,
        softmax_exp_points,
        fixed,
        &domain,
        "bert_shape_rns_dyadic_softmax_exp_of_pmpe_deg6",
        exp_cfg,
        rng,
    )?;
    let square_cfg = DyadicOfPmpePolynomialConfig::square(f, chunk)?
        .with_input_abs_bound_bits(activation_bound_bits)?
        .with_modulus_margin_bits(10);
    bench_rns_dyadic_of_pmpe_slots(
        opts.iterations,
        layernorm_square_points,
        fixed,
        &domain,
        "bert_shape_rns_dyadic_layernorm_square_of_pmpe",
        square_cfg,
        rng,
    )?;
    bench_rns_mean_shift_softmax_exp_phase(
        opts.iterations,
        softmax_exp_points / BERT_SEQ,
        BERT_SEQ,
        fixed,
        opts.pmpe_chunk_lanes.max(1),
        0.0,
        rng,
    )?;
    bench_rns_mean_shift_softmax_full_probe(
        opts.iterations,
        softmax_exp_points / BERT_SEQ,
        BERT_SEQ,
        fixed,
        opts.pmpe_chunk_lanes.max(1),
        0.0,
        rng,
    )?;
    Ok(())
}

fn bench_rns_mean_shift_softmax_exp_phase(
    iterations: usize,
    rows: usize,
    cols: usize,
    fixed: FixedPointConfig,
    chunk_lanes: usize,
    beta_shift: f64,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    if !cols.is_power_of_two() {
        return Err("mean-shift softmax benchmark requires power-of-two cols".into());
    }
    let f = frac_bits_from_scale(fixed.scale)?;
    let shifted_frac_bits = f + cols.trailing_zeros();
    let domain = RnsDyadicDomain::limbs_61bit(4)?;
    let shifted_bound_bits = shifted_frac_bits.saturating_add(4);
    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::exp_negative_degree6_probe(chunk_lanes.max(1)).coeffs,
        shifted_frac_bits,
        f,
        104,
        domain.modulus_bits(),
        true,
        chunk_lanes.max(1),
    )?
    .with_input_abs_bound_bits(shifted_bound_bits)?
    .with_modulus_margin_bits(10);
    let audit = domain.no_wrap_audit(&exp_cfg)?;
    if !audit.ok {
        return Err(format!("RNS mean-shift softmax audit failed: {audit:?}").into());
    }
    let input = rns_shared_vector(rows * cols, fixed, &domain, rng)?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut mean_shift_us = 0u64;
    let mut sum_us = 0u64;
    let mut operator_online_us = 0u64;
    let mut exp_profile = OfPmpeProfile::default();
    for _ in 0..iterations {
        let start = Instant::now();
        let out = domain.softmax_mean_shift_of_pmpe_exp_profiled(
            &input, rows, cols, f, beta_shift, &exp_cfg, rng,
        )?;
        elapsed += start.elapsed();
        mean_shift_us = mean_shift_us.saturating_add(out.mean_shift_us);
        sum_us = sum_us.saturating_add(out.sum_us);
        operator_online_us = operator_online_us.saturating_add(out.online_us);
        exp_profile = out.exp_profile;
        checksum = out
            .row_sums
            .shares
            .reconstruct_residues()
            .iter()
            .flat_map(|limb| limb.iter())
            .fold(checksum, |acc, &value| acc.wrapping_add(value));
    }
    let case = format!("rns_mean_shift_softmax_exp_phase_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=rns_mean_shift_ofpmpe_exp|rows={rows}|cols={cols}|beta_shift={beta_shift:.6}|score_frac_bits={f}|shifted_frac_bits={shifted_frac_bits}|output_frac_bits={}|guard_bits={}|effective_output_frac_bits={}|input_abs_bound_bits={}|estimated_output_bound_bits={}|rns_limbs={}|crt_modulus_bits={}|modulus_margin_bits={}|signed_margin_bits={}|no_wrap_ok={}|operator_online_us_avg={:.3}|mean_shift_us_avg={:.3}|sum_us_avg={:.3}|exp_{}|checksum={checksum}",
            exp_cfg.output_frac_bits,
            exp_cfg.guard_bits,
            exp_cfg.output_frac_bits_with_guard()?,
            format_opt_u32(exp_cfg.input_abs_bound_bits),
            format_opt_u32(exp_cfg.output_bound_bits),
            domain.moduli().len(),
            audit.modulus_bits,
            exp_cfg.modulus_margin_bits,
            audit.signed_margin_bits,
            audit.ok,
            operator_online_us as f64 / iterations as f64,
            mean_shift_us as f64 / iterations as f64,
            sum_us as f64 / iterations as f64,
            of_pmpe_profile_extra(exp_profile),
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_rns_mean_shift_softmax_full_probe(
    iterations: usize,
    rows: usize,
    cols: usize,
    fixed: FixedPointConfig,
    chunk_lanes: usize,
    beta_shift: f64,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    if !cols.is_power_of_two() {
        return Err("mean-shift softmax benchmark requires power-of-two cols".into());
    }
    let f = frac_bits_from_scale(fixed.scale)?;
    let shifted_frac_bits = f + cols.trailing_zeros();
    let domain = RnsDyadicDomain::limbs_61bit(4)?;
    let shifted_bound_bits = shifted_frac_bits.saturating_add(4);
    let exp_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &OfPmpePolynomialConfig::exp_negative_degree6_probe(chunk_lanes.max(1)).coeffs,
        shifted_frac_bits,
        f,
        104,
        domain.modulus_bits(),
        true,
        chunk_lanes.max(1),
    )?
    .with_input_abs_bound_bits(shifted_bound_bits)?
    .with_modulus_margin_bits(10);
    let exp_audit = domain.no_wrap_audit(&exp_cfg)?;
    if !exp_audit.ok {
        return Err(format!("RNS mean-shift exp audit failed: {exp_audit:?}").into());
    }
    let exp_frac_bits = exp_cfg.output_frac_bits_with_guard()?;
    let row_sum_bound_bits = exp_frac_bits
        .saturating_add(cols.trailing_zeros())
        .saturating_add(1);
    let reciprocal_input_frac_bits = f.saturating_mul(2);
    let reciprocal_input_bound_bits = reciprocal_input_frac_bits
        .saturating_add(cols.next_power_of_two().trailing_zeros())
        .saturating_add(2);
    let reciprocal_guard_bits = 72;
    let reciprocal_degree = 3;
    let reciprocal_center = (cols as f64) * (-beta_shift).exp();
    let reciprocal_coeffs = reciprocal_taylor_coeffs(reciprocal_center, reciprocal_degree);
    let reciprocal_cfg = DyadicOfPmpePolynomialConfig::from_real_coeffs(
        &reciprocal_coeffs,
        reciprocal_input_frac_bits,
        reciprocal_input_frac_bits,
        reciprocal_guard_bits,
        domain.modulus_bits(),
        true,
        rows.max(1),
    )?
    .with_input_abs_bound_bits(reciprocal_input_bound_bits)?
    .with_modulus_margin_bits(10);
    let reciprocal_audit = domain.no_wrap_audit(&reciprocal_cfg)?;
    if !reciprocal_audit.ok {
        return Err(format!("RNS mean-shift reciprocal audit failed: {reciprocal_audit:?}").into());
    }
    let product_bound_bits = exp_cfg
        .output_bound_bits
        .and_then(|exp_bits| {
            reciprocal_cfg
                .output_bound_bits
                .and_then(|recip_bits| exp_bits.checked_add(recip_bits))
        })
        .ok_or("missing RNS mean-shift product bound")?;
    let product_required_bits = product_bound_bits.saturating_add(10).saturating_add(1);
    let product_signed_margin_bits = domain.modulus_bits() as i32 - product_required_bits as i32;
    if product_signed_margin_bits < 0 {
        return Err(format!(
            "RNS mean-shift broadcast product audit failed: bound_bits={product_bound_bits}, modulus_bits={}, signed_margin_bits={product_signed_margin_bits}",
            domain.modulus_bits()
        )
        .into());
    }
    let preprocess_plan = domain.mean_shift_softmax_preprocess_plan(
        rows,
        cols,
        &exp_cfg,
        exp_frac_bits.saturating_sub(reciprocal_input_frac_bits),
        row_sum_bound_bits,
        120,
        &reciprocal_cfg,
    )?;
    let input = rns_shared_vector(rows * cols, fixed, &domain, rng)?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut operator_online_us = 0u64;
    let mut repeat_us = 0u64;
    let mut exp_profile = OfPmpeProfile::default();
    let mut rescale_profile = OfPmpeProfile::default();
    let mut reciprocal_profile = OfPmpeProfile::default();
    let mut broadcast_profile = OfPmpeProfile::default();
    let mut product_frac_bits = 0u32;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = domain.softmax_mean_shift_of_pmpe_rescaled_profiled(
            &input,
            rows,
            cols,
            f,
            beta_shift,
            &exp_cfg,
            reciprocal_input_frac_bits,
            row_sum_bound_bits,
            120,
            &reciprocal_cfg,
            rng,
        )?;
        elapsed += start.elapsed();
        operator_online_us = operator_online_us.saturating_add(out.online_us);
        repeat_us = repeat_us.saturating_add(out.repeat_us);
        exp_profile = out.exp_phase.exp_profile;
        rescale_profile = out
            .rescale_profile
            .ok_or("missing RNS mean-shift rescale profile")?;
        reciprocal_profile = out.reciprocal_profile;
        broadcast_profile = out.broadcast_mul_profile;
        product_frac_bits = out.probs.frac_bits;
        checksum = out
            .probs
            .shares
            .reconstruct_residues()
            .iter()
            .flat_map(|limb| limb.iter())
            .fold(checksum, |acc, &value| acc.wrapping_add(value));
    }
    let total_oneflow_phases = exp_profile
        .num_oneflow_phases
        .saturating_add(rescale_profile.num_oneflow_phases)
        .saturating_add(reciprocal_profile.num_oneflow_phases)
        .saturating_add(broadcast_profile.num_oneflow_phases);
    let total_request_response_rounds = exp_profile
        .num_request_response_rounds
        .saturating_add(rescale_profile.num_request_response_rounds)
        .saturating_add(reciprocal_profile.num_request_response_rounds)
        .saturating_add(broadcast_profile.num_request_response_rounds);
    let case = format!("rns_mean_shift_softmax_full_probe_{rows}x{cols}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=rns_mean_shift_ofpmpe_softmax_full_probe|rows={rows}|cols={cols}|beta_shift={beta_shift:.6}|score_frac_bits={f}|shifted_frac_bits={shifted_frac_bits}|exp_frac_bits={exp_frac_bits}|row_sum_rescale_target_frac_bits={reciprocal_input_frac_bits}|row_sum_rescale_mask_bits=120|reciprocal_input_frac_bits={}|reciprocal_output_frac_bits={}|reciprocal_guard_bits={}|product_frac_bits={product_frac_bits}|product_estimated_output_bound_bits={product_bound_bits}|product_signed_margin_bits={product_signed_margin_bits}|product_no_wrap_ok={}|reciprocal_kind=taylor_centered_degree{reciprocal_degree}|reciprocal_center={reciprocal_center:.6}|rns_limbs={}|crt_modulus_bits={}|preprocess_random_masks={}|preprocess_beaver_triples={}|preprocess_truncation_masks={}|preprocess_requires_beaver_source={}|preprocess_requires_truncation_source={}|preprocess_requires_external_correlation_source={}|preprocess_pack_size_bytes={}|preprocess_peak_pack_resident_bytes={}|preprocess_streaming_pack={}|preprocess_one_time={}|exp_signed_margin_bits={}|reciprocal_signed_margin_bits={}|exp_no_wrap_ok={}|reciprocal_no_wrap_ok={}|operator_online_us_avg={:.3}|repeat_us_avg={:.3}|total_oneflow_phases={total_oneflow_phases}|total_request_response_rounds={total_request_response_rounds}|exp_{}|rescale_{}|reciprocal_{}|broadcast_{}|checksum={checksum}",
            reciprocal_cfg.input_frac_bits,
            reciprocal_cfg.output_frac_bits,
            reciprocal_cfg.guard_bits,
            product_signed_margin_bits >= 0,
            domain.moduli().len(),
            domain.modulus_bits(),
            preprocess_plan.random_masks,
            preprocess_plan.beaver_triples,
            preprocess_plan.truncation_masks,
            preprocess_plan.requires_beaver_source(),
            preprocess_plan.requires_truncation_source(),
            preprocess_plan.requires_external_correlation_source(),
            preprocess_plan.pack_size_bytes,
            preprocess_plan.peak_pack_resident_bytes,
            preprocess_plan.streaming_pack,
            preprocess_plan.one_time,
            exp_audit.signed_margin_bits,
            reciprocal_audit.signed_margin_bits,
            exp_audit.ok,
            reciprocal_audit.ok,
            operator_online_us as f64 / iterations as f64,
            repeat_us as f64 / iterations as f64,
            of_pmpe_profile_extra(exp_profile),
            of_pmpe_profile_extra(rescale_profile),
            of_pmpe_profile_extra(reciprocal_profile),
            of_pmpe_profile_extra(broadcast_profile),
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_rns_beaver_mul_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    limbs: usize,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let f = frac_bits_from_scale(fixed.scale)?;
    let domain = RnsDyadicDomain::limbs_61bit(limbs)?;
    let lhs = RnsScaledShareTensor {
        shares: rns_shared_vector(lanes, fixed, &domain, rng)?,
        frac_bits: f,
        guard_bits: 0,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let rhs = RnsScaledShareTensor {
        shares: rns_shared_vector(lanes, fixed, &domain, rng)?,
        frac_bits: f,
        guard_bits: 0,
        semantic: ScaleSemantic::DyadicNumerator,
    };
    let mut elapsed = Duration::ZERO;
    let mut profile = OfPmpeProfile::default();
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        let mut pack = domain.beaver_mul_pack(lanes, rng)?;
        let out = domain.mul_scaled_with_beaver_profiled(&lhs, &rhs, &mut pack)?;
        elapsed += start.elapsed();
        profile = out.profile;
        checksum = out
            .tensor
            .shares
            .reconstruct_residues()
            .iter()
            .flat_map(|limb| limb.iter())
            .fold(checksum, |acc, &value| acc.wrapping_add(value));
    }
    let case = format!("rns_beaver_mul_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=rns_beaver_mul|lanes={lanes}|lhs_frac_bits={f}|rhs_frac_bits={f}|product_frac_bits={}|rns_limbs={}|crt_modulus_bits={}|offline_ms_avg={:.3}|online_ms_avg={:.3}|offline_bytes_avg={}|online_bytes_avg={}|{}|checksum={checksum}",
            f.saturating_mul(2),
            domain.moduli().len(),
            domain.modulus_bits(),
            profile.offline_ms as f64 / iterations as f64,
            profile.online_ms as f64 / iterations as f64,
            profile.offline_bytes / iterations as u64,
            profile.online_bytes / iterations as u64,
            of_pmpe_profile_extra(profile),
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_rns_dyadic_of_pmpe_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    domain: &RnsDyadicDomain,
    case_prefix: &str,
    cfg: DyadicOfPmpePolynomialConfig,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let input = rns_shared_vector(lanes, fixed, domain, rng)?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0i128;
    let mut profile = OfPmpeProfile::default();
    let audit = domain.no_wrap_audit(&cfg)?;
    if !audit.ok {
        return Err(format!("RNS no-wrap audit failed for {case_prefix}: {audit:?}").into());
    }
    for _ in 0..iterations {
        let start = Instant::now();
        let out = domain.dyadic_of_pmpe_profiled(&input, &cfg, rng)?;
        elapsed += start.elapsed();
        profile = out.profile;
        checksum = out
            .tensor
            .shares
            .reconstruct_centered_i128()?
            .iter()
            .fold(checksum, |acc, &value| acc.wrapping_add(value));
    }
    let case = format!("{case_prefix}_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=rns_dyadic_of_pmpe|lanes={lanes}|degree={}|coeffs={}|input_frac_bits={}|output_frac_bits={}|guard_bits={}|effective_output_frac_bits={}|input_abs_bound_bits={}|estimated_output_bound_bits={}|rns_limbs={}|crt_modulus_bits={}|modulus_margin_bits={}|signed_margin_bits={}|no_wrap_ok={}|offline_ms_avg={:.3}|online_ms_avg={:.3}|offline_bytes_avg={}|online_bytes_avg={}|{}|checksum={checksum}",
            cfg.degree,
            cfg.coeffs_num.len(),
            cfg.input_frac_bits,
            cfg.output_frac_bits,
            cfg.guard_bits,
            cfg.output_frac_bits_with_guard()?,
            format_opt_u32(cfg.input_abs_bound_bits),
            format_opt_u32(cfg.output_bound_bits),
            domain.moduli().len(),
            audit.modulus_bits,
            cfg.modulus_margin_bits,
            audit.signed_margin_bits,
            audit.ok,
            profile.offline_ms as f64 / iterations as f64,
            profile.online_ms as f64 / iterations as f64,
            profile.offline_bytes / iterations as u64,
            profile.online_bytes / iterations as u64,
            of_pmpe_profile_extra(profile),
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_of_pmpe_poly_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    case_prefix: &str,
    coeffs: &[f64],
    runtime_net: bool,
    streaming: bool,
    chunk_lanes: usize,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    if streaming {
        return bench_of_pmpe_poly_slots_streaming(
            iterations,
            lanes,
            fixed,
            nonlinear,
            case_prefix,
            coeffs,
            runtime_net,
            chunk_lanes,
            rng,
        );
    }
    let input = shared_vector(lanes, fixed, rng)?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut offline_ms = 0u64;
    let mut online_ms = 0u64;
    let mut online_bytes = 0u64;
    let mut offline_bytes = 0u64;
    let mut runtime_net_ms = 0.0f64;
    let mut runtime_stats = RuntimeStats::default();
    let mut profile = OfPmpeProfile::default();
    for _ in 0..iterations {
        let start = Instant::now();
        let mut pack = nonlinear.of_pmpe_taylor_pack(lanes, coeffs, rng)?;
        let runtime_delta = if runtime_net {
            Some(input.sub(&pack.masks)?)
        } else {
            None
        };
        let out = nonlinear.of_pmpe_eval(&input, &mut pack)?;
        if runtime_net {
            let delta = runtime_delta
                .as_ref()
                .ok_or("missing OF-PMPE runtime delta")?;
            let mut link = RuntimeOneWayU64::tcp_loopback(SessionId(768230))?;
            let net_start = Instant::now();
            link.exchange_oneflow(TaskId(768231), delta.party0(), delta.party1())?;
            runtime_net_ms += net_start.elapsed().as_secs_f64() * 1000.0;
            runtime_stats.merge(link.stats());
        }
        elapsed += start.elapsed();
        offline_ms = offline_ms.saturating_add(out.profile.offline_ms);
        online_ms = online_ms.saturating_add(out.profile.online_ms);
        online_bytes = online_bytes.saturating_add(out.profile.online_bytes);
        offline_bytes = offline_bytes.saturating_add(out.profile.offline_bytes);
        profile = out.profile;
        checksum = checksum.wrapping_add(
            out.shares
                .reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }
    let avg_offline_ms = offline_ms as f64 / iterations as f64;
    let avg_online_ms = online_ms as f64 / iterations as f64;
    let avg_offline_bytes = offline_bytes / iterations as u64;
    let avg_online_bytes = online_bytes / iterations as u64;
    let avg_runtime_net_ms = runtime_net_ms / iterations as f64;
    let case = format!("{case_prefix}_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=of_pmpe|lanes={lanes}|degree={}|coeffs={}|offline_ms_avg={avg_offline_ms:.3}|online_ms_avg={avg_online_ms:.3}|offline_bytes_avg={avg_offline_bytes}|online_bytes_avg={avg_online_bytes}|runtime_net_enabled={runtime_net}|runtime_net_ms_avg={avg_runtime_net_ms:.3}|runtime_frames_sent={}|runtime_frames_received={}|runtime_bytes_sent={}|runtime_bytes_received={}|runtime_request_response_rounds=0|{}|checksum={checksum}",
            coeffs.len() - 1,
            coeffs.len(),
            runtime_stats.frames_sent / iterations as u64,
            runtime_stats.frames_received / iterations as u64,
            runtime_stats.bytes_sent / iterations as u64,
            runtime_stats.bytes_received / iterations as u64,
            of_pmpe_profile_extra(profile)
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_of_pmpe_poly_slots_streaming(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    case_prefix: &str,
    coeffs: &[f64],
    runtime_net: bool,
    chunk_lanes: usize,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let input = shared_vector(lanes, fixed, rng)?;
    let chunk_lanes = chunk_lanes.max(1);
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut offline_ms = 0u64;
    let mut online_ms = 0u64;
    let mut online_bytes = 0u64;
    let mut offline_bytes = 0u64;
    let mut runtime_net_ms = 0.0f64;
    let mut runtime_stats = RuntimeStats::default();
    let mut profile = OfPmpeProfile::default();
    for _ in 0..iterations {
        let start = Instant::now();
        let mut iter_profile = OfPmpeProfile {
            offline_mode: OfPmpeOfflineMode::TrustedDebug,
            degree: coeffs.len() - 1,
            slots: lanes,
            num_oneflow_phases: 1,
            public_linear_terms: coeffs.len(),
            pack_size_bytes: of_pmpe_pack_bytes_for_runner(lanes, coeffs.len() - 1),
            streaming_pack: true,
            ..OfPmpeProfile::default()
        };
        let mut link = if runtime_net {
            Some(RuntimeOneWayU64::tcp_loopback(SessionId(768230))?)
        } else {
            None
        };
        for start_idx in (0..lanes).step_by(chunk_lanes) {
            let end_idx = usize::min(start_idx + chunk_lanes, lanes);
            let chunk_input = AdditiveShares::new(
                fixed.modulus,
                input.party0()[start_idx..end_idx].to_vec(),
                input.party1()[start_idx..end_idx].to_vec(),
            )?;
            let mut pack = nonlinear.of_pmpe_taylor_pack(chunk_input.len(), coeffs, rng)?;
            let runtime_delta = if runtime_net {
                Some(chunk_input.sub(&pack.masks)?)
            } else {
                None
            };
            let out = nonlinear.of_pmpe_eval(&chunk_input, &mut pack)?;
            if let Some(link) = link.as_mut() {
                let delta = runtime_delta
                    .as_ref()
                    .ok_or("missing streaming OF-PMPE delta")?;
                let net_start = Instant::now();
                link.exchange_oneflow(TaskId(768231), delta.party0(), delta.party1())?;
                runtime_net_ms += net_start.elapsed().as_secs_f64() * 1000.0;
            }
            merge_of_pmpe_profile(&mut iter_profile, out.profile);
            checksum = checksum.wrapping_add(
                out.shares
                    .reconstruct()
                    .iter()
                    .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
            );
        }
        if let Some(link) = link {
            runtime_stats.merge(link.stats());
        }
        elapsed += start.elapsed();
        offline_ms = offline_ms.saturating_add(iter_profile.offline_ms);
        online_ms = online_ms.saturating_add(iter_profile.online_ms);
        online_bytes = online_bytes.saturating_add(iter_profile.online_bytes);
        offline_bytes = offline_bytes.saturating_add(iter_profile.offline_bytes);
        profile = iter_profile;
    }
    let avg_offline_ms = offline_ms as f64 / iterations as f64;
    let avg_online_ms = online_ms as f64 / iterations as f64;
    let avg_offline_bytes = offline_bytes / iterations as u64;
    let avg_online_bytes = online_bytes / iterations as u64;
    let avg_runtime_net_ms = runtime_net_ms / iterations as f64;
    let case = format!("{case_prefix}_streaming_slots_{lanes}");
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=of_pmpe|lanes={lanes}|degree={}|coeffs={}|chunk_lanes={chunk_lanes}|offline_ms_avg={avg_offline_ms:.3}|online_ms_avg={avg_online_ms:.3}|offline_bytes_avg={avg_offline_bytes}|online_bytes_avg={avg_online_bytes}|runtime_net_enabled={runtime_net}|runtime_net_ms_avg={avg_runtime_net_ms:.3}|runtime_frames_sent={}|runtime_frames_received={}|runtime_bytes_sent={}|runtime_bytes_received={}|runtime_request_response_rounds=0|{}|checksum={checksum}",
            coeffs.len() - 1,
            coeffs.len(),
            runtime_stats.frames_sent / iterations as u64,
            runtime_stats.frames_received / iterations as u64,
            runtime_stats.bytes_sent / iterations as u64,
            runtime_stats.bytes_received / iterations as u64,
            of_pmpe_profile_extra(profile)
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_dyadic_of_pmpe_slots(
    iterations: usize,
    lanes: usize,
    fixed: FixedPointConfig,
    nonlinear: &NonlinearOps<'_>,
    case_prefix: &str,
    mut cfg: DyadicOfPmpePolynomialConfig,
    runtime_net: bool,
    streaming: bool,
    rng: &mut StdRng,
) -> Result<f64, Box<dyn std::error::Error>> {
    let input = shared_vector(lanes, fixed, rng)?;
    cfg.streaming = streaming;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    let mut offline_ms = 0u64;
    let mut online_ms = 0u64;
    let mut online_bytes = 0u64;
    let mut offline_bytes = 0u64;
    let mut runtime_net_ms = 0.0f64;
    let mut runtime_stats = RuntimeStats::default();
    let mut profile = OfPmpeProfile::default();

    for _ in 0..iterations {
        let start = Instant::now();
        if streaming {
            let mut iter_profile = OfPmpeProfile {
                offline_mode: OfPmpeOfflineMode::TrustedDebug,
                degree: cfg.degree,
                slots: lanes,
                num_oneflow_phases: 1,
                public_linear_terms: cfg.coeffs_num.len(),
                pack_size_bytes: of_pmpe_pack_bytes_for_runner(lanes, cfg.degree),
                streaming_pack: true,
                ..OfPmpeProfile::default()
            };
            let mut link = if runtime_net {
                Some(RuntimeOneWayU64::tcp_loopback(SessionId(768250))?)
            } else {
                None
            };
            for start_idx in (0..lanes).step_by(cfg.chunk_slots) {
                let end_idx = usize::min(start_idx + cfg.chunk_slots, lanes);
                let chunk_input = AdditiveShares::new(
                    fixed.modulus,
                    input.party0()[start_idx..end_idx].to_vec(),
                    input.party1()[start_idx..end_idx].to_vec(),
                )?;
                let mut pack =
                    nonlinear.dyadic_of_pmpe_taylor_pack(chunk_input.len(), &cfg, rng)?;
                let runtime_delta = if runtime_net {
                    Some(chunk_input.sub(&pack.masks)?)
                } else {
                    None
                };
                let out = nonlinear.of_pmpe_eval(&chunk_input, &mut pack)?;
                if let Some(link) = link.as_mut() {
                    let delta = runtime_delta
                        .as_ref()
                        .ok_or("missing dyadic OF-PMPE delta")?;
                    let net_start = Instant::now();
                    link.exchange_oneflow(TaskId(768251), delta.party0(), delta.party1())?;
                    runtime_net_ms += net_start.elapsed().as_secs_f64() * 1000.0;
                }
                merge_of_pmpe_profile(&mut iter_profile, out.profile);
                checksum = checksum.wrapping_add(
                    out.shares
                        .reconstruct()
                        .iter()
                        .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
                );
            }
            if let Some(link) = link {
                runtime_stats.merge(link.stats());
            }
            profile = iter_profile;
        } else {
            let mut pack = nonlinear.dyadic_of_pmpe_taylor_pack(lanes, &cfg, rng)?;
            let runtime_delta = if runtime_net {
                Some(input.sub(&pack.masks)?)
            } else {
                None
            };
            let out = nonlinear.of_pmpe_eval(&input, &mut pack)?;
            if runtime_net {
                let delta = runtime_delta
                    .as_ref()
                    .ok_or("missing dyadic OF-PMPE delta")?;
                let mut link = RuntimeOneWayU64::tcp_loopback(SessionId(768250))?;
                let net_start = Instant::now();
                link.exchange_oneflow(TaskId(768251), delta.party0(), delta.party1())?;
                runtime_net_ms += net_start.elapsed().as_secs_f64() * 1000.0;
                runtime_stats.merge(link.stats());
            }
            profile = out.profile;
            checksum = checksum.wrapping_add(
                out.shares
                    .reconstruct()
                    .iter()
                    .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
            );
        }
        elapsed += start.elapsed();
        offline_ms = offline_ms.saturating_add(profile.offline_ms);
        online_ms = online_ms.saturating_add(profile.online_ms);
        online_bytes = online_bytes.saturating_add(profile.online_bytes);
        offline_bytes = offline_bytes.saturating_add(profile.offline_bytes);
    }

    let avg_offline_ms = offline_ms as f64 / iterations as f64;
    let avg_online_ms = online_ms as f64 / iterations as f64;
    let avg_offline_bytes = offline_bytes / iterations as u64;
    let avg_online_bytes = online_bytes / iterations as u64;
    let avg_runtime_net_ms = runtime_net_ms / iterations as f64;
    let actual_modulus_bits = modulus_bit_width_for_runner(fixed.modulus);
    let modulus_wrap_safe = cfg.modulus_wrap_safe(actual_modulus_bits);
    let case = if streaming {
        format!("{case_prefix}_streaming_slots_{lanes}")
    } else {
        format!("{case_prefix}_slots_{lanes}")
    };
    print_row(
        "bert_probe",
        &case,
        iterations,
        elapsed,
        &format!(
            "path=dyadic_of_pmpe|lanes={lanes}|degree={}|coeffs={}|input_frac_bits={}|output_frac_bits={}|guard_bits={}|effective_output_frac_bits={}|input_abs_bound_bits={}|estimated_output_bound_bits={}|modulus_bits={actual_modulus_bits}|modulus_margin_bits={}|modulus_wrap_safe={}|allow_modular_wrap={}|chunk_lanes={}|offline_ms_avg={avg_offline_ms:.3}|online_ms_avg={avg_online_ms:.3}|offline_bytes_avg={avg_offline_bytes}|online_bytes_avg={avg_online_bytes}|runtime_net_enabled={runtime_net}|runtime_net_ms_avg={avg_runtime_net_ms:.3}|runtime_frames_sent={}|runtime_frames_received={}|runtime_bytes_sent={}|runtime_bytes_received={}|runtime_request_response_rounds=0|{}|checksum={checksum}",
            cfg.degree,
            cfg.coeffs_num.len(),
            cfg.input_frac_bits,
            cfg.output_frac_bits,
            cfg.guard_bits,
            cfg.output_frac_bits_with_guard()?,
            format_opt_u32(cfg.input_abs_bound_bits),
            format_opt_u32(cfg.output_bound_bits),
            cfg.modulus_margin_bits,
            format_opt_bool(modulus_wrap_safe),
            cfg.allow_modular_wrap,
            cfg.chunk_slots,
            runtime_stats.frames_sent / iterations as u64,
            runtime_stats.frames_received / iterations as u64,
            runtime_stats.bytes_sent / iterations as u64,
            runtime_stats.bytes_received / iterations as u64,
            of_pmpe_profile_extra(profile)
        ),
    );
    Ok(avg_ms(elapsed, iterations))
}

fn bench_q_to_p_bridge_bert_input(
    iterations: usize,
    fixed: FixedPointConfig,
    rng: &mut StdRng,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    let q_modulus = fixed.modulus * (1u64 << 32);
    let len = BERT_SEQ * BERT_HIDDEN;
    let mut bridge = RuntimeQToPBridge::tcp_loopback()?;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;

    for _ in 0..iterations {
        let mut party0 = Vec::with_capacity(len);
        let mut party1 = Vec::with_capacity(len);
        for _ in 0..len {
            party0.push(rng.next_u64() % q_modulus);
            party1.push(rng.next_u64() % q_modulus);
        }
        let q_shares = LinearMapQShares::new(
            party0,
            party1,
            q_modulus,
            fixed.modulus,
            BERT_HIDDEN,
            BERT_SEQ,
        )?;
        let start = Instant::now();
        let out = bridge.convert(&q_shares, rng)?;
        elapsed += start.elapsed();
        checksum = checksum.wrapping_add(
            out.reconstruct()
                .iter()
                .fold(0u64, |acc, &v| add_mod(acc, v, fixed.modulus)),
        );
    }

    let stats = bridge.stats();
    print_row(
        "bert_probe",
        "q_to_p_bridge_values_98304_tcp_loopback",
        iterations,
        elapsed,
        &format!(
            "values={len}|q_modulus={q_modulus}|p={}|frames_sent={}|frames_received={}|bytes_sent={}|bytes_received={}|checksum={checksum}",
            fixed.modulus,
            stats.frames_sent,
            stats.frames_received,
            stats.bytes_sent,
            stats.bytes_received,
        ),
    );
    Ok(BenchResult {
        avg_ms: avg_ms(elapsed, iterations),
        bytes_sent: stats.bytes_sent,
        bytes_received: stats.bytes_received,
    })
}

fn print_bert_base_scaled_model(
    linear_qkv_ms: f64,
    linear_768_ms: f64,
    linear_3072_ms: f64,
    linear_3072_to_768_ms: f64,
    hss_mul_raw_4096_ms: f64,
    hss_mul_4096_ms: f64,
    attention_dot_64_ms: f64,
    context_dot_128_ms: f64,
    softmax_128_ms: f64,
    softmax_32x128_ms: f64,
    softmax_poly_32x128_ms: Option<f64>,
    softmax_approx_32x128_ms: Option<f64>,
    softmax_specialized_32x128_ms: Option<f64>,
    layernorm_768_ms: f64,
    layernorm_5x768_ms: f64,
    layernorm_approx_5x768_ms: Option<f64>,
    layernorm_specialized_5x768_ms: Option<f64>,
    gelu_4096_ms: f64,
    gelu_specialized_4096_ms: Option<f64>,
    gelu_chunk_lanes: usize,
    simd_chunk: usize,
    softmax_batch_rows: usize,
    layernorm_batch_rows: usize,
    q_to_p_bridge_ms: f64,
    q_to_p_bridge_bytes: u64,
    packed_linear_768: Option<PhaseTiming>,
    packed_attention: Option<PhaseTiming>,
    packed_context: Option<PhaseTiming>,
    packed_attention_head: Option<PhaseTiming>,
    packed_context_head: Option<PhaseTiming>,
) {
    let qkv_linear = BERT_LAYERS;
    let out_linear = BERT_LAYERS;
    let ff1_linear = BERT_LAYERS;
    let ff2_linear = BERT_LAYERS;
    let layernorm_rows = BERT_LAYERS * 2 * BERT_SEQ;
    let layernorm_batches = layernorm_rows.div_ceil(layernorm_batch_rows);
    let softmax_rows = BERT_LAYERS * BERT_HEADS * BERT_SEQ;
    let softmax_batches = softmax_rows.div_ceil(softmax_batch_rows);
    let attention_products = BERT_LAYERS * BERT_HEADS * BERT_SEQ * BERT_SEQ * BERT_HEAD_DIM;
    let context_products = attention_products;
    let attention_chunks = attention_products.div_ceil(simd_chunk);
    let context_chunks = context_products.div_ceil(simd_chunk);
    let gelu_elements = BERT_LAYERS * BERT_SEQ * BERT_FFN;
    let gelu_chunks = gelu_elements.div_ceil(gelu_chunk_lanes);

    let linear_ms = qkv_linear as f64 * linear_qkv_ms
        + out_linear as f64 * linear_768_ms
        + ff1_linear as f64 * linear_3072_ms
        + ff2_linear as f64 * linear_3072_to_768_ms;
    let packed_linear_projection_ms =
        packed_linear_768.map(|timing| (qkv_linear * 3 + out_linear) as f64 * timing.online_ms);
    let packed_linear_projection_preprocess_ms =
        packed_linear_768.map(|timing| (qkv_linear * 3 + out_linear) as f64 * timing.preprocess_ms);
    let hss_smatmul_ms = (attention_chunks + context_chunks) as f64 * hss_mul_4096_ms;
    let dot_smatmul_ms =
        attention_chunks as f64 * attention_dot_64_ms + context_chunks as f64 * context_dot_128_ms;
    let packed_smatmul_ms = packed_attention
        .zip(packed_context)
        .map(|(attention, context)| BERT_LAYERS as f64 * (attention.online_ms + context.online_ms));
    let packed_smatmul_preprocess_ms =
        packed_attention
            .zip(packed_context)
            .map(|(attention, context)| {
                BERT_LAYERS as f64 * (attention.preprocess_ms + context.preprocess_ms)
            });
    let packed_head_smatmul_ms =
        packed_attention_head
            .zip(packed_context_head)
            .map(|(attention, context)| {
                (BERT_LAYERS * BERT_HEADS) as f64 * (attention.online_ms + context.online_ms)
            });
    let packed_head_smatmul_preprocess_ms =
        packed_attention_head
            .zip(packed_context_head)
            .map(|(attention, context)| {
                (BERT_LAYERS * BERT_HEADS) as f64
                    * (attention.preprocess_ms + context.preprocess_ms)
            });
    let softmax_ms = softmax_rows as f64 * softmax_128_ms;
    let softmax_batched_ms = softmax_batches as f64 * softmax_32x128_ms;
    let softmax_poly_batched_ms = softmax_poly_32x128_ms.map(|ms| softmax_batches as f64 * ms);
    let softmax_approx_batched_ms = softmax_approx_32x128_ms.map(|ms| softmax_batches as f64 * ms);
    let softmax_specialized_batched_ms =
        softmax_specialized_32x128_ms.map(|ms| softmax_batches as f64 * ms);
    let mut selected_softmax_ms = softmax_batched_ms;
    if let Some(poly_ms) = softmax_poly_batched_ms.filter(|&poly_ms| poly_ms < selected_softmax_ms)
    {
        selected_softmax_ms = poly_ms;
    }
    if let Some(approx_ms) =
        softmax_approx_batched_ms.filter(|&approx_ms| approx_ms < selected_softmax_ms)
    {
        selected_softmax_ms = approx_ms;
    }
    if let Some(specialized_ms) = softmax_specialized_batched_ms
        .filter(|&specialized_ms| specialized_ms < selected_softmax_ms)
    {
        selected_softmax_ms = specialized_ms;
    }
    let layernorm_ms = layernorm_rows as f64 * layernorm_768_ms;
    let layernorm_batched_ms = layernorm_batches as f64 * layernorm_5x768_ms;
    let layernorm_approx_batched_ms =
        layernorm_approx_5x768_ms.map(|ms| layernorm_batches as f64 * ms);
    let layernorm_specialized_batched_ms =
        layernorm_specialized_5x768_ms.map(|ms| layernorm_batches as f64 * ms);
    let mut selected_layernorm_ms = layernorm_batched_ms;
    if let Some(approx_ms) =
        layernorm_approx_batched_ms.filter(|&approx_ms| approx_ms < selected_layernorm_ms)
    {
        selected_layernorm_ms = approx_ms;
    }
    if let Some(specialized_ms) = layernorm_specialized_batched_ms
        .filter(|&specialized_ms| specialized_ms < selected_layernorm_ms)
    {
        selected_layernorm_ms = specialized_ms;
    }
    let gelu_ms = gelu_chunks as f64 * gelu_4096_ms;
    let gelu_specialized_ms = gelu_specialized_4096_ms.map(|ms| gelu_chunks as f64 * ms);
    let selected_gelu_ms = gelu_specialized_ms
        .filter(|&specialized_ms| specialized_ms < gelu_ms)
        .unwrap_or(gelu_ms);
    let selected_smatmul_ms = packed_head_smatmul_ms
        .or(packed_smatmul_ms)
        .unwrap_or(dot_smatmul_ms);
    let selected_smatmul_preprocess_ms = if packed_head_smatmul_ms.is_some() {
        packed_head_smatmul_preprocess_ms.unwrap_or(0.0)
    } else {
        packed_smatmul_preprocess_ms.unwrap_or(0.0)
    };
    let preprocess_ms =
        packed_linear_projection_preprocess_ms.unwrap_or(0.0) + selected_smatmul_preprocess_ms;
    let compute_ms = linear_ms
        + selected_smatmul_ms
        + selected_softmax_ms
        + selected_layernorm_ms
        + selected_gelu_ms;
    let refresh_barriers = 2 * BERT_LAYERS;
    let lan_ms = compute_ms
        + refresh_barriers as f64 * LAN_RTT_MS
        + bandwidth_ms(PAPER_BERT_BASE_COMM_GB, LAN_BW_MBPS);
    let wan_ms = compute_ms
        + refresh_barriers as f64 * WAN_RTT_MS
        + bandwidth_ms(PAPER_BERT_BASE_COMM_GB, WAN_BW_MBPS);

    let total = Duration::from_secs_f64(compute_ms / 1000.0);
    print_row(
        "bert_base_diagnostic",
        "bert_base_12layer_128token_packed_cpu_operator_mix_compute",
        1,
        total,
        &format!(
            "phase=online|layers={BERT_LAYERS}|seq={BERT_SEQ}|hidden={BERT_HIDDEN}|heads={BERT_HEADS}|head_dim={BERT_HEAD_DIM}|ffn={BERT_FFN}|hss_simd_chunk={simd_chunk}|qkv_linear_calls={qkv_linear}|out_linear_calls={out_linear}|ff1_calls={ff1_linear}|ff2_calls={ff2_linear}|attention_hss_chunks={attention_chunks}|context_chunks={context_chunks}|softmax_rows={softmax_rows}|softmax_batch_rows={softmax_batch_rows}|softmax_batches={softmax_batches}|layernorm_rows={layernorm_rows}|layernorm_batch_rows={layernorm_batch_rows}|layernorm_batches={layernorm_batches}|gelu_chunk_lanes={gelu_chunk_lanes}|gelu_chunks={gelu_chunks}|linear_qkv_ms={:.3}|linear_ms={linear_ms:.3}|packed_linear_768_online_ms={:.3}|packed_projection_online_equiv_ms={:.3}|packed_projection_offline_preprocess_equiv_ms={:.3}|hss_mul_raw_ms={hss_mul_raw_4096_ms:.3}|hss_mul_fixed_ms={hss_mul_4096_ms:.3}|smatmul_elementwise_rescale_ms={hss_smatmul_ms:.3}|attention_dot64_ms={attention_dot_64_ms:.3}|context_dot128_ms={context_dot_128_ms:.3}|smatmul_dot_rescale_ms={dot_smatmul_ms:.3}|packed_attention_online_ms={:.3}|packed_context_online_ms={:.3}|packed_smatmul_online_ms={:.3}|packed_smatmul_offline_preprocess_ms={:.3}|packed_attention_head_online_ms={:.3}|packed_context_head_online_ms={:.3}|packed_head_smatmul_online_ms={:.3}|packed_head_smatmul_offline_preprocess_ms={:.3}|selected_smatmul_ms={selected_smatmul_ms:.3}|softmax_rowwise_ms={softmax_ms:.3}|softmax_batched_ms={softmax_batched_ms:.3}|softmax_poly_batched_ms={:.3}|softmax_approx_batched_ms={:.3}|softmax_specialized_batched_ms={:.3}|selected_softmax_ms={selected_softmax_ms:.3}|layernorm_rowwise_ms={layernorm_ms:.3}|layernorm_batched_ms={layernorm_batched_ms:.3}|layernorm_approx_batched_ms={:.3}|layernorm_specialized_batched_ms={:.3}|selected_layernorm_ms={selected_layernorm_ms:.3}|gelu_ms={gelu_ms:.3}|gelu_specialized_ms={:.3}|selected_gelu_ms={selected_gelu_ms:.3}|q_to_p_bridge_98304_ms={q_to_p_bridge_ms:.3}|q_to_p_bridge_bytes_total={q_to_p_bridge_bytes}|model=diagnostic_packed_cpu_operator_mix_not_paper_e2e",
            qkv_linear as f64 * linear_qkv_ms,
            packed_linear_768
                .map(|timing| timing.online_ms)
                .unwrap_or(0.0),
            packed_linear_projection_ms.unwrap_or(0.0),
            packed_linear_projection_preprocess_ms.unwrap_or(0.0),
            packed_attention
                .map(|timing| timing.online_ms)
                .unwrap_or(0.0),
            packed_context.map(|timing| timing.online_ms).unwrap_or(0.0),
            packed_smatmul_ms.unwrap_or(0.0),
            packed_smatmul_preprocess_ms.unwrap_or(0.0),
            packed_attention_head
                .map(|timing| timing.online_ms)
                .unwrap_or(0.0),
            packed_context_head
                .map(|timing| timing.online_ms)
                .unwrap_or(0.0),
            packed_head_smatmul_ms.unwrap_or(0.0),
            packed_head_smatmul_preprocess_ms.unwrap_or(0.0),
            softmax_poly_batched_ms.unwrap_or(0.0),
            softmax_approx_batched_ms.unwrap_or(0.0),
            softmax_specialized_batched_ms.unwrap_or(0.0),
            layernorm_approx_batched_ms.unwrap_or(0.0),
            layernorm_specialized_batched_ms.unwrap_or(0.0),
            gelu_specialized_ms.unwrap_or(0.0)
        ),
    );

    print_row(
        "bert_base_diagnostic",
        "bert_base_12layer_128token_packed_cpu_operator_mix_offline_preprocess",
        1,
        Duration::from_secs_f64(preprocess_ms / 1000.0),
        &format!(
            "phase=offline_preprocess|packed_projection_offline_preprocess_equiv_ms={:.3}|selected_smatmul_offline_preprocess_ms={selected_smatmul_preprocess_ms:.3}|model=diagnostic_packed_cpu_operator_mix_not_paper_e2e",
            packed_linear_projection_preprocess_ms.unwrap_or(0.0)
        ),
    );

    println!(
        "bert_base_diagnostic,bert_base_12layer_128token_packed_cpu_operator_mix_lan,1,{lan_ms:.3},{lan_ms:.3},phase=online_with_transport|online_compute_ms={compute_ms:.3}|offline_preprocess_ms={preprocess_ms:.3}|refresh_barriers={refresh_barriers}|rtt_ms={LAN_RTT_MS}|comm_gb={PAPER_BERT_BASE_COMM_GB}|bandwidth_mbps={LAN_BW_MBPS}|comm_ms={:.3}|model=diagnostic_packed_cpu_operator_mix_not_paper_e2e",
        bandwidth_ms(PAPER_BERT_BASE_COMM_GB, LAN_BW_MBPS)
    );
    println!(
        "bert_base_diagnostic,bert_base_12layer_128token_packed_cpu_operator_mix_wan,1,{wan_ms:.3},{wan_ms:.3},phase=online_with_transport|online_compute_ms={compute_ms:.3}|offline_preprocess_ms={preprocess_ms:.3}|refresh_barriers={refresh_barriers}|rtt_ms={WAN_RTT_MS}|comm_gb={PAPER_BERT_BASE_COMM_GB}|bandwidth_mbps={WAN_BW_MBPS}|comm_ms={:.3}|model=diagnostic_packed_cpu_operator_mix_not_paper_e2e",
        bandwidth_ms(PAPER_BERT_BASE_COMM_GB, WAN_BW_MBPS)
    );
}

fn bandwidth_ms(gb: f64, mbps: f64) -> f64 {
    gb * 8.0 * 1000.0 / mbps * 1000.0
}

fn shared_vector(
    len: usize,
    fixed: FixedPointConfig,
    rng: &mut StdRng,
) -> Result<AdditiveShares, Box<dyn std::error::Error>> {
    let values = (0..len)
        .map(|idx| {
            let centered = (idx as i64 % 17) - 8;
            fixed.encode_f64(centered as f64 / 16.0)
        })
        .collect::<Vec<_>>();
    Ok(AdditiveShares::share_with_rng(&values, fixed.modulus, rng)?)
}

fn rns_shared_vector(
    len: usize,
    fixed: FixedPointConfig,
    domain: &RnsDyadicDomain,
    rng: &mut StdRng,
) -> Result<silent_operators::nonlinear::RnsShareTensor, Box<dyn std::error::Error>> {
    let values = (0..len)
        .map(|idx| {
            let centered = (idx as i64 % 17) - 8;
            (centered as i128 * fixed.scale as i128) / 16
        })
        .collect::<Vec<_>>();
    Ok(domain.share_with_rng(&values, rng)?)
}

fn reciprocal_taylor_coeffs(center: f64, degree: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0f64; degree + 1];
    for k in 0..=degree {
        let taylor_coeff = if k % 2 == 0 {
            center.powi(-((k + 1) as i32))
        } else {
            -center.powi(-((k + 1) as i32))
        };
        for (j, coeff) in coeffs.iter_mut().enumerate().take(k + 1) {
            *coeff += taylor_coeff * binomial_f64(k, j) * (-center).powi((k - j) as i32);
        }
    }
    coeffs
}

fn binomial_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut acc = 1.0;
    for idx in 0..k {
        acc *= (n - idx) as f64;
        acc /= (idx + 1) as f64;
    }
    acc
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

fn avg_ms(elapsed: Duration, iterations: usize) -> f64 {
    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn profile_extra(profile: NonlinearProfile) -> String {
    format!(
        "selectors={}|poly_mul={}|trunc={}|refresh={}|profile_online_ms={}|transport_bytes={}",
        profile.selectors,
        profile.poly_mul,
        profile.trunc,
        profile.refresh,
        profile.online_ms,
        profile.transport_bytes
    )
}

fn merge_of_pmpe_profile(acc: &mut OfPmpeProfile, chunk: OfPmpeProfile) {
    acc.num_request_response_rounds += chunk.num_request_response_rounds;
    acc.num_masked_opens += chunk.num_masked_opens;
    acc.num_opened_elements += chunk.num_opened_elements;
    acc.num_flushes += chunk.num_flushes;
    acc.num_network_flushes += chunk.num_network_flushes;
    acc.online_secret_secret_mul += chunk.online_secret_secret_mul;
    acc.online_trunc += chunk.online_trunc;
    acc.num_fresh_masks += chunk.num_fresh_masks;
    acc.num_reused_masks += chunk.num_reused_masks;
    acc.offline_ms += chunk.offline_ms;
    acc.online_ms += chunk.online_ms;
    acc.offline_us += chunk.offline_us;
    acc.online_us += chunk.online_us;
    acc.secure_offline_ms += chunk.secure_offline_ms;
    acc.trusted_debug_offline_ms += chunk.trusted_debug_offline_ms;
    acc.secure_offline_us += chunk.secure_offline_us;
    acc.trusted_debug_offline_us += chunk.trusted_debug_offline_us;
    acc.offline_bytes += chunk.offline_bytes;
    acc.online_bytes += chunk.online_bytes;
    acc.oneflow_payload_bytes += chunk.oneflow_payload_bytes;
    acc.peak_pack_resident_bytes =
        u64::max(acc.peak_pack_resident_bytes, chunk.peak_pack_resident_bytes);
}

fn of_pmpe_pack_bytes_for_runner(slots: usize, degree: usize) -> u64 {
    (slots as u64)
        .saturating_mul((degree as u64).saturating_add(2))
        .saturating_mul(2)
        .saturating_mul(8)
}

fn frac_bits_from_scale(scale: u64) -> Result<u32, Box<dyn std::error::Error>> {
    if scale == 0 || !scale.is_power_of_two() {
        return Err("dyadic OF-PMPE requires power-of-two fixed scale".into());
    }
    Ok(scale.trailing_zeros())
}

fn modulus_bit_width_for_runner(modulus: u64) -> u32 {
    u64::BITS - modulus.saturating_sub(1).leading_zeros()
}

fn format_opt_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn format_opt_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "unknown",
    }
}

fn of_pmpe_profile_extra(profile: OfPmpeProfile) -> String {
    format!(
        "offline_mode={}|num_oneflow_phases={}|num_request_response_rounds={}|num_masked_opens={}|num_opened_elements={}|num_flushes={}|num_network_flushes={}|public_linear_terms={}|online_secret_secret_mul={}|online_trunc={}|num_fresh_masks={}|num_reused_masks={}|profile_offline_ms={}|profile_online_ms={}|profile_offline_us={}|profile_online_us={}|secure_offline_ms={}|trusted_debug_offline_ms={}|secure_offline_us={}|trusted_debug_offline_us={}|offline_bytes={}|online_bytes={}|oneflow_payload_bytes={}|pack_size_bytes={}|peak_pack_resident_bytes={}|streaming_pack={}",
        of_pmpe_offline_mode(profile.offline_mode),
        profile.num_oneflow_phases,
        profile.num_request_response_rounds,
        profile.num_masked_opens,
        profile.num_opened_elements,
        profile.num_flushes,
        profile.num_network_flushes,
        profile.public_linear_terms,
        profile.online_secret_secret_mul,
        profile.online_trunc,
        profile.num_fresh_masks,
        profile.num_reused_masks,
        profile.offline_ms,
        profile.online_ms,
        profile.offline_us,
        profile.online_us,
        profile.secure_offline_ms,
        profile.trusted_debug_offline_ms,
        profile.secure_offline_us,
        profile.trusted_debug_offline_us,
        profile.offline_bytes,
        profile.online_bytes,
        profile.oneflow_payload_bytes,
        profile.pack_size_bytes,
        profile.peak_pack_resident_bytes,
        profile.streaming_pack
    )
}

fn of_pmpe_offline_mode(mode: silent_operators::nonlinear::OfPmpeOfflineMode) -> &'static str {
    match mode {
        silent_operators::nonlinear::OfPmpeOfflineMode::TrustedDebug => "trusted_debug",
        silent_operators::nonlinear::OfPmpeOfflineMode::SecurePreprocess => "secure_preprocess",
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
            runtime_sessions_from_streams(client_stream, server_stream, SessionId(768128))?;
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
                        TaskId(768129),
                        MessageKind::Data,
                        client_state.masked_q(),
                    )
                    .await?;
                let server_route = self
                    .server
                    .send_u64_vec(
                        PartyId(0),
                        TaskId(768129),
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

    fn stats(&self) -> RuntimeStats {
        let mut stats = self.client.stats();
        stats.merge(self.server.stats());
        stats
    }
}

struct RuntimeOneWayU64 {
    runtime: Runtime,
    client: RuntimeSession,
    server: RuntimeSession,
}

impl RuntimeOneWayU64 {
    fn tcp_loopback(session_id: SessionId) -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = Builder::new_current_thread().enable_all().build()?;
        let (client_stream, server_stream) = runtime.block_on(tcp_loopback_stream_pair())?;
        let (client, server) =
            runtime_sessions_from_streams(client_stream, server_stream, session_id)?;
        Ok(Self {
            runtime,
            client,
            server,
        })
    }

    fn send_client_to_server(
        &mut self,
        task: TaskId,
        payload: &[u64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        const MAX_U64S_PER_FRAME: usize = 120_000;
        self.runtime.block_on(async {
            for chunk in payload.chunks(MAX_U64S_PER_FRAME) {
                let route = self
                    .client
                    .send_u64_vec(PartyId(1), task, MessageKind::Data, chunk)
                    .await?;
                let received = self.server.recv_expected_u64_vec(PartyId(0), route).await?;
                if received.len() != chunk.len() {
                    return Err(RuntimeError::Codec(
                        "packed linear query payload length mismatch".to_string(),
                    ));
                }
            }
            Ok::<_, RuntimeError>(())
        })?;
        Ok(())
    }

    fn exchange_oneflow(
        &mut self,
        task: TaskId,
        client_payload: &[u64],
        server_payload: &[u64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if client_payload.len() != server_payload.len() {
            return Err("one-flow exchange payload lengths must match".into());
        }
        const MAX_U64S_PER_FRAME: usize = 16_000;
        self.runtime.block_on(async {
            for (client_chunk, server_chunk) in client_payload
                .chunks(MAX_U64S_PER_FRAME)
                .zip(server_payload.chunks(MAX_U64S_PER_FRAME))
            {
                let (client_route, server_route) = tokio::try_join!(
                    self.client
                        .send_u64_vec(PartyId(1), task, MessageKind::Data, client_chunk),
                    self.server
                        .send_u64_vec(PartyId(0), task, MessageKind::Data, server_chunk),
                )?;
                let (server_seen, client_seen) = tokio::try_join!(
                    self.client.recv_expected_u64_vec(PartyId(1), server_route),
                    self.server.recv_expected_u64_vec(PartyId(0), client_route),
                )?;
                if server_seen.len() != server_chunk.len()
                    || client_seen.len() != client_chunk.len()
                {
                    return Err(RuntimeError::Codec(
                        "OF-PMPE one-flow payload length mismatch".to_string(),
                    ));
                }
            }
            Ok::<_, RuntimeError>(())
        })?;
        Ok(())
    }

    fn stats(&self) -> RuntimeStats {
        let mut stats = self.client.stats();
        stats.merge(self.server.stats());
        stats
    }
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
        "bert-packed-linear-rlwe-v1",
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

fn build_hss_preset_engine(name: &str) -> Result<HssSlotEngine, Box<dyn std::error::Error>> {
    let preset = find_preset(name).ok_or_else(|| format!("missing HSS preset {name}"))?;
    let RegisteredParameterSet::Hss(params) = preset else {
        return Err(format!("preset {name} is not an HSS parameter set").into());
    };
    let context = HssContext::new(params)?;
    let mut keygen = HssKeyGenerator::new(context.clone(), SecureRng::from_seed([53u8; 32]));
    let sk = keygen.generate_secret_key();
    let pk = keygen.generate_public_key(&sk);
    let (share0, share1) = keygen.split_secret_key(&sk);
    Ok(HssSlotEngine::new(context, pk, share0, share1))
}

fn build_hss_plain_modulus_engine(
    degree: usize,
    plain_modulus: u64,
) -> Result<HssSlotEngine, Box<dyn std::error::Error>> {
    let q_count = if plain_modulus >= (1u64 << 50) || degree >= 1024 {
        8
    } else {
        4
    };
    build_hss_plain_modulus_engine_with_q_count(degree, plain_modulus, q_count)
}

fn build_hss_plain_modulus_engine_with_q_count(
    degree: usize,
    plain_modulus: u64,
    q_count: usize,
) -> Result<HssSlotEngine, Box<dyn std::error::Error>> {
    if !degree.is_power_of_two() {
        return Err("HSS correlation smoke degree must be a power of two".into());
    }
    let step = 2 * degree as u64;
    let q_count = q_count.max(1);
    let mut q_values = Vec::with_capacity(q_count);
    let mut q = first_ntt_prime_with_bits(50, step).ok_or("failed to find q1")?;
    q_values.push(q);
    for idx in 1..q_count {
        q = next_ntt_prime(q + 1, step).ok_or_else(|| format!("failed to find q{}", idx + 1))?;
        q_values.push(q);
    }
    let base_q = RnsBase::from_values(q_values).map_err(|err| format!("{err:?}"))?;
    let runtime = EncryptionParams::from_rns_config(
        degree,
        RnsToolConfig::new(
            base_q,
            Modulus::new(plain_modulus).map_err(|err| format!("{err:?}"))?,
        ),
    )
    .map_err(|err| format!("{err:?}"))?;
    let context = context_from_runtime("silent-hss-correlation-smoke-v1", runtime, plain_modulus)?;
    let seed_byte = (plain_modulus as u8).wrapping_mul(31).wrapping_add(7);
    let mut keygen = HssKeyGenerator::new(context.clone(), SecureRng::from_seed([seed_byte; 32]));
    let sk = keygen.generate_secret_key();
    let pk = keygen.generate_public_key(&sk);
    let (share0, share1) = keygen.split_secret_key(&sk);
    Ok(HssSlotEngine::new(context, pk, share0, share1))
}

fn build_packed_ahe_plain_modulus_engine(
    degree: usize,
    plain_modulus: u64,
    q_count: usize,
    q_bits: u32,
) -> Result<PackedRlweAheTripleEngine, Box<dyn std::error::Error>> {
    if !degree.is_power_of_two() {
        return Err("packed RLWE-AHE smoke degree must be a power of two".into());
    }
    let step = 2 * degree as u64;
    let q_count = q_count.max(1);
    let mut q_values = Vec::with_capacity(q_count);
    let mut q = first_ntt_prime_with_bits(q_bits, step).ok_or("failed to find q1")?;
    q_values.push(q);
    for idx in 1..q_count {
        q = next_ntt_prime(q + 1, step).ok_or_else(|| format!("failed to find q{}", idx + 1))?;
        q_values.push(q);
    }
    let base_q = RnsBase::from_values(q_values).map_err(|err| format!("{err:?}"))?;
    let runtime = EncryptionParams::from_rns_config(
        degree,
        RnsToolConfig::new(
            base_q,
            Modulus::new(plain_modulus).map_err(|err| format!("{err:?}"))?,
        ),
    )
    .map_err(|err| format!("{err:?}"))?;
    let context =
        context_from_runtime("silent-packed-ahe-triples-smoke-v1", runtime, plain_modulus)?;
    let seed_byte = (plain_modulus as u8).wrapping_mul(17).wrapping_add(11);
    let mut keygen = HssKeyGenerator::new(context.clone(), SecureRng::from_seed([seed_byte; 32]));
    let sk = keygen.generate_secret_key();
    let pk = keygen.generate_public_key(&sk);
    Ok(PackedRlweAheTripleEngine::new(context, pk, sk)?)
}

fn build_hss_degree_engine(degree: usize) -> Result<HssSlotEngine, Box<dyn std::error::Error>> {
    if !degree.is_power_of_two() {
        return Err("--hss-degree must be a power of two".into());
    }
    if degree > 32_768 {
        return Err("--hss-degree must be <= 32768 for plaintext modulus 65537 batching".into());
    }
    let log_n = RingDim(degree)
        .to_log_n()
        .ok_or("--hss-degree must fit SILENT ring metadata")?;
    let params = HssParams {
        name: "bert-hss-custom-v1",
        rlwe: RlweParams::new(
            "bert-rlwe-hss-custom-v1",
            RingParams::new(log_n, RingDim(degree), DistributionType::Ternary),
            vec![ModulusBits(55), ModulusBits(55)],
            Vec::new(),
            None,
            SecurityLevel::Classical128,
        ),
        plaintext_modulus: PlaintextModulus(HSS_PLAINTEXT_MODULUS),
        share_modulus_bits: ShareModulusBits(110),
        max_linear_terms: MaxLinearTerms(128),
        max_rmult_depth: MaxRmultDepth(1),
        fixed_point_scale_bits: Some(ScaleBits(16)),
        reconstruction_bound_bits: Some(ReconstructionBoundBits(72)),
        correctness_margin_bits: Some(CorrectnessMarginBits(8)),
    };
    let context = HssContext::new(params)?;
    let mut keygen = HssKeyGenerator::new(context.clone(), SecureRng::from_seed([67u8; 32]));
    let sk = keygen.generate_secret_key();
    let pk = keygen.generate_public_key(&sk);
    let (share0, share1) = keygen.split_secret_key(&sk);
    Ok(HssSlotEngine::new(context, pk, share0, share1))
}

fn dyadic_integer_polynomial_value_i128_for_runner(x: i128, coeffs: &[i128]) -> i128 {
    let mut acc = 0i128;
    let mut power = 1i128;
    for &coeff in coeffs {
        acc += coeff * power;
        power *= x;
    }
    acc
}

fn print_row(section: &str, case: &str, iterations: usize, elapsed: Duration, extra: &str) {
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / iterations as f64;
    println!("{section},{case},{iterations},{total_ms:.3},{avg_ms:.3},{extra}");
}

fn extra_safe(value: &str) -> String {
    value.replace(',', ";").replace('|', ";").replace('\n', " ")
}

fn generate_attention_score_rows(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut scores = Vec::with_capacity(rows.saturating_mul(cols));
    for row in 0..rows {
        let layer = row % BERT_LAYERS;
        let head = (row / BERT_LAYERS) % BERT_HEADS;
        let sigma = 0.55 + 0.035 * layer as f64 + 0.018 * head as f64;
        let focus = (row.wrapping_mul(37).wrapping_add(head * 13)) % cols;
        let rare_focus = row % 97 == 0;
        for col in 0..cols {
            let wave = 0.20
                * ((col as f64 + 1.0) * (head as f64 + 1.0) / cols as f64 + layer as f64 * 0.07)
                    .sin();
            let distance = usize::abs_diff(col, focus).min(cols - usize::abs_diff(col, focus));
            let locality = 0.45 * (-(distance as f64) / 18.0).exp();
            let rare_spike = if rare_focus && col == focus {
                2.75
            } else {
                0.0
            };
            scores.push(sample_standard_normal(rng) * sigma + wave + locality + rare_spike);
        }
    }
    scores
}

fn max_minus_mean_percentiles(scores: &[f64], rows: usize, cols: usize) -> [f64; 4] {
    let mut spreads = Vec::with_capacity(rows);
    for row in scores.chunks_exact(cols) {
        let mean = row.iter().sum::<f64>() / cols as f64;
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        spreads.push(max - mean);
    }
    spreads.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    [
        percentile_sorted(&spreads, 0.90),
        percentile_sorted(&spreads, 0.95),
        percentile_sorted(&spreads, 0.99),
        percentile_sorted(&spreads, 0.999),
    ]
}

fn percentile_sorted(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * p).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn fit_exp_polynomial_on_negative_interval(
    degree: usize,
    domain_b: f64,
    samples: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = degree + 1;
    let mut normal = vec![vec![0.0f64; n + 1]; n];
    for sample in 0..samples.max(2) {
        let t = sample as f64 / (samples.max(2) - 1) as f64;
        let x = -domain_b + domain_b * t;
        let y = x.exp();
        let mut powers = vec![1.0f64; 2 * degree + 1];
        for idx in 1..powers.len() {
            powers[idx] = powers[idx - 1] * x;
        }
        for row in 0..n {
            for col in 0..n {
                normal[row][col] += powers[row + col];
            }
            normal[row][n] += y * powers[row];
        }
    }
    solve_linear_system(normal).ok_or_else(|| "singular exp polynomial fit".into())
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let n = matrix.len();
    for pivot in 0..n {
        let mut best = pivot;
        let mut best_abs = matrix[pivot][pivot].abs();
        for row in pivot + 1..n {
            let candidate = matrix[row][pivot].abs();
            if candidate > best_abs {
                best = row;
                best_abs = candidate;
            }
        }
        if best_abs < 1e-18 || !best_abs.is_finite() {
            return None;
        }
        matrix.swap(pivot, best);
        let pivot_value = matrix[pivot][pivot];
        for col in pivot..=n {
            matrix[pivot][col] /= pivot_value;
        }
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in pivot..=n {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
        }
    }
    Some(matrix.into_iter().map(|row| row[n]).collect())
}

fn evaluate_meanbeta_softmax_poly(
    scores: &[f64],
    rows: usize,
    cols: usize,
    beta: f64,
    domain_b: f64,
    degree: usize,
    coeffs: &[f64],
) -> Result<MeanBetaSweepMetrics, Box<dyn std::error::Error>> {
    let mut total_l1 = 0.0;
    let mut total_kl = 0.0;
    let mut max_l1 = 0.0f64;
    let mut top1_hits = 0usize;
    let mut out_hi = 0usize;
    let mut out_lo = 0usize;
    let mut negative = 0usize;
    let mut bad_rows = 0usize;
    let mut poly_min_observed = f64::INFINITY;
    let mut poly_max_observed = f64::NEG_INFINITY;
    let mut exact = vec![0.0f64; cols];
    let mut approx_exp = vec![0.0f64; cols];
    let eps = 1e-15;

    for row in scores.chunks_exact(cols) {
        let mean = row.iter().sum::<f64>() / cols as f64;
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exact_denom = row.iter().map(|&value| (value - max).exp()).sum::<f64>();
        let mut exact_top = 0usize;
        for (idx, &value) in row.iter().enumerate() {
            exact[idx] = (value - max).exp() / exact_denom;
            if exact[idx] > exact[exact_top] {
                exact_top = idx;
            }
        }

        let mut approx_sum = 0.0;
        for (idx, &value) in row.iter().enumerate() {
            let u = value - mean - beta;
            if u > 0.0 {
                out_hi += 1;
            }
            if u < -domain_b {
                out_lo += 1;
            }
            let poly = eval_poly(coeffs, u);
            poly_min_observed = poly_min_observed.min(poly);
            poly_max_observed = poly_max_observed.max(poly);
            if poly < 0.0 {
                negative += 1;
            }
            approx_exp[idx] = poly;
            approx_sum += poly;
        }
        if approx_sum <= 0.0 || !approx_sum.is_finite() {
            bad_rows += 1;
            continue;
        }

        let mut row_l1 = 0.0;
        let mut row_kl = 0.0;
        let mut approx_top = 0usize;
        let mut approx_top_value = f64::NEG_INFINITY;
        for idx in 0..cols {
            let q = approx_exp[idx] / approx_sum;
            if q > approx_top_value {
                approx_top_value = q;
                approx_top = idx;
            }
            row_l1 += (exact[idx] - q).abs();
            row_kl += exact[idx] * (exact[idx] / q.max(eps)).ln();
        }
        total_l1 += row_l1;
        total_kl += row_kl;
        max_l1 = max_l1.max(row_l1);
        if approx_top == exact_top {
            top1_hits += 1;
        }
    }

    let denom_rows = rows.saturating_sub(bad_rows).max(1) as f64;
    let denom_elements = rows.saturating_mul(cols).max(1) as f64;
    Ok(MeanBetaSweepMetrics {
        rows,
        cols,
        beta,
        domain_b,
        degree,
        avg_l1: total_l1 / denom_rows,
        avg_kl: total_kl / denom_rows,
        max_l1,
        top1_agreement: top1_hits as f64 / denom_rows,
        out_hi_rate: out_hi as f64 / denom_elements,
        out_lo_rate: out_lo as f64 / denom_elements,
        negative_rate: negative as f64 / denom_elements,
        bad_rows,
        poly_min_observed,
        poly_max_observed,
    })
}

fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &coeff| acc * x + coeff)
}

fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = uniform_open01(rng);
    let u2 = uniform_open01(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn uniform_open01(rng: &mut StdRng) -> f64 {
    let value = rng.next_u64() >> 11;
    (value as f64 + 0.5) / ((1u64 << 53) as f64)
}
