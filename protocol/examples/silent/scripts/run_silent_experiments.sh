#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/target/silent-experiments}"

MODE="${MODE:-${SILENT_PROFILE:-smoke}}"
ITERATIONS="${ITERATIONS:-}"
if [[ $# -ge 1 ]]; then
  case "$1" in
    smoke|full)
      MODE="$1"
      ;;
    *)
      ITERATIONS="$1"
      ;;
  esac
fi
if [[ $# -ge 2 ]]; then
  MODE="$2"
fi
case "${MODE}" in
  smoke)
    ITERATIONS="${ITERATIONS:-5}"
    ;;
  full)
    ITERATIONS="${ITERATIONS:-1}"
    ;;
  *)
    echo "[silent] invalid mode '${MODE}', expected smoke or full" >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_DIR}"

ts="$(date +%Y%m%d_%H%M%S)"
MEASURED_CSV="${OUT_DIR}/silent_${MODE}_measured_${ts}.csv"
PARAMS_CSV="${OUT_DIR}/silent_${MODE}_parameters_${ts}.csv"
CMATMUL_CSV="${OUT_DIR}/silent_${MODE}_cmatmul_suite_${ts}.csv"
PMPE_AUDIT_CSV="${OUT_DIR}/silent_${MODE}_pmpe_rns_audit_${ts}.csv"
WAN_SENSITIVITY_CSV="${OUT_DIR}/silent_${MODE}_wan_sensitivity_${ts}.csv"
RPM_QUERY_AUDIT_CSV="${OUT_DIR}/silent_${MODE}_rpm_query_hiding_entropy_${ts}.csv"
SUMMARY_MD="${OUT_DIR}/silent_${MODE}_reproduction_${ts}.md"

echo "[silent] running ${MODE} Rust microbenchmarks (${ITERATIONS} iterations)"
cargo run \
  --release \
  --manifest-path "${ROOT_DIR}/Cargo.toml" \
  -p silent \
  --bin silent_reproduce \
  -- --iterations "${ITERATIONS}" --mode "${MODE}" | tee "${MEASURED_CSV}"

echo "[silent] running ${MODE} TCP multi-process runtime harness (${ITERATIONS} iterations)"
cargo run \
  --release \
  --manifest-path "${ROOT_DIR}/Cargo.toml" \
  -p silent \
  --bin silent_runtime_net_harness \
  -- --iterations "${ITERATIONS}" --mode "${MODE}" --no-header | tee -a "${MEASURED_CSV}"

echo "[silent] running ${MODE} THOR-style end-to-end transformer harness (${ITERATIONS} iterations)"
cargo run \
  --release \
  --manifest-path "${ROOT_DIR}/Cargo.toml" \
  -p silent \
  --bin silent_e2e_transformer \
  -- --iterations "${ITERATIONS}" --mode "${MODE}" --no-header | tee -a "${MEASURED_CSV}"

SILENT_DIRECTION_LANES="${SILENT_DIRECTION_LANES:-262144}"
SILENT_DIRECTION_CHUNK="${SILENT_DIRECTION_CHUNK:-8192}"
echo "[silent] running OF-PMPE nonlinear direction validator (${SILENT_DIRECTION_LANES} sampled lanes)"
cargo run \
  --release \
  --manifest-path "${ROOT_DIR}/Cargo.toml" \
  -p silent \
  --bin silent_bert_base_runner \
  -- --silent-direction-validation \
     --pmpe-lanes "${SILENT_DIRECTION_LANES}" \
     --pmpe-chunk-lanes "${SILENT_DIRECTION_CHUNK}" \
     --no-header | tee -a "${MEASURED_CSV}"

echo "[silent] exporting audited parameter manifest"
cargo run \
  --release \
  --manifest-path "${ROOT_DIR}/Cargo.toml" \
  -p silent \
  --bin silent_params_report | tee "${PARAMS_CSV}"

echo "[silent] running RPM-CNIM BERT-shaped CMatMul audit"
cargo run \
  --release \
  --manifest-path "${ROOT_DIR}/Cargo.toml" \
  -p silent \
  --bin silent_cmatmul_suite \
  -- --iterations 1 | tee "${CMATMUL_CSV}"

echo "[silent] running RNS OF-PMPE BERT-shape no-wrap audit"
cargo run \
  --release \
  --manifest-path "${ROOT_DIR}/Cargo.toml" \
  -p silent \
  --bin silent_bert_base_runner \
  -- --pmpe-only --pmpe-rns --pmpe-bert-shapes --pmpe-lanes 4096 --pmpe-chunk-lanes 1024 --no-header | tee "${PMPE_AUDIT_CSV}"

echo "[silent] generating WAN sensitivity model"
python3 "${ROOT_DIR}/protocol/examples/silent/scripts/make_wan_sensitivity.py" | tee "${WAN_SENSITIVITY_CSV}"

RPM_QUERY_AUDIT_ROWS=0
RPM_QUERY_AUDIT_STATUS="skipped; set SILENT_RUN_RPM_QUERY_AUDIT=1 and provide SILENT_RPM_KAPPA_{2048,4096} plus SILENT_RPM_MASK_HINF_{2048,4096}"
if [[ "${SILENT_RUN_RPM_QUERY_AUDIT:-0}" == "1" ]]; then
  echo "[silent] generating optional RPM-CNIM query-hiding entropy audit"
  python3 "${ROOT_DIR}/protocol/examples/silent/scripts/make_rpm_query_hiding_audit.py" \
    --kappa-2048 "${SILENT_RPM_KAPPA_2048:?missing SILENT_RPM_KAPPA_2048}" \
    --kappa-4096 "${SILENT_RPM_KAPPA_4096:?missing SILENT_RPM_KAPPA_4096}" \
    --mask-min-entropy-per-coeff-2048 "${SILENT_RPM_MASK_HINF_2048:?missing SILENT_RPM_MASK_HINF_2048}" \
    --mask-min-entropy-per-coeff-4096 "${SILENT_RPM_MASK_HINF_4096:?missing SILENT_RPM_MASK_HINF_4096}" \
    | tee "${RPM_QUERY_AUDIT_CSV}"
  RPM_QUERY_AUDIT_ROWS="$(( $(wc -l < "${RPM_QUERY_AUDIT_CSV}") - 1 ))"
  RPM_QUERY_AUDIT_STATUS="generated; entropy-only candidate audit, not a projected-error or RLWE-estimator certificate"
else
  printf '%s\n' 'operator,N,log2_q,kappa,mask_min_entropy_per_coeff,q_rpm,h_inf_bits,lhl_required_bits,entropy_slack_bits,statistical_lhl_pass,paper_usable,paper_usable_reason,scope' > "${RPM_QUERY_AUDIT_CSV}"
fi

MEASURED_ROWS="$(( $(wc -l < "${MEASURED_CSV}") - 1 ))"
PARAM_ROWS="$(( $(wc -l < "${PARAMS_CSV}") - 1 ))"
CMATMUL_ROWS="$(( $(wc -l < "${CMATMUL_CSV}") - 1 ))"
PMPE_AUDIT_ROWS="$(( $(wc -l < "${PMPE_AUDIT_CSV}") ))"
WAN_SENSITIVITY_ROWS="$(( $(wc -l < "${WAN_SENSITIVITY_CSV}") - 1 ))"

cat > "${SUMMARY_MD}" <<EOF
# SILENT Experiment Reproduction

Generated: ${ts}
Profile: ${MODE}

## Measured Rust Artifact

- File: \`${MEASURED_CSV}\`
- Command: \`cargo run --release --manifest-path ${ROOT_DIR}/Cargo.toml -p silent --bin silent_reproduce -- --iterations ${ITERATIONS} --mode ${MODE}\`
- Rows: ${MEASURED_ROWS}
- Scope: Rust implementation microbenchmarks for RPM-CNIM/NIMVM with and without offline server setup, row-blocked linear ablation, typed Share2HE/D2S-Refresh/HSS slot multiplication, runtime session/value transport over memory, TCP loopback, a spawned two-process TCP harness, and the BERT refresh-barrier RTT model. Legacy lookup/nonlinear diagnostics are opt-in through \`--legacy-diagnostics\` and are not part of this default artifact run.
- Baseline/comparison-system experiments are intentionally not run by this artifact script.

## Parameter Manifest

- File: \`${PARAMS_CSV}\`
- Command: \`cargo run --release --manifest-path ${ROOT_DIR}/Cargo.toml -p silent --bin silent_params_report\`
- Rows: ${PARAM_ROWS}
- Scope: validated SILENT parameter presets used for audited BFV/HSS/PQ metadata, plus explicitly labeled toy-size smoke parameters used only for fast correctness and microbenchmark reproduction.
- Profile policy: \`smoke\` is the fast CI/correctness profile; \`full\` switches measured operators to audited-size HSS, larger lookup parameters, and HSS-backed shared-bit multiplication where the current Rust implementation supports them.

## Added Review-Response Audits

- File: \`${CMATMUL_CSV}\`
- Rows: ${CMATMUL_ROWS}
- Scope: BERT-shaped \`RPM-CNIM\` CMatMul block geometry, online/offline bytes, timings, bounded-noise setting, and correctness checks.
- File: \`${PMPE_AUDIT_CSV}\`
- Rows: ${PMPE_AUDIT_ROWS}
- Scope: RNS dyadic \`OF-PMPE\` BERT-shape no-wrap audits for GeLU, Softmax, and LayerNorm templates.
- File: \`${WAN_SENSITIVITY_CSV}\`
- Rows: ${WAN_SENSITIVITY_ROWS}
- Scope: analytic WAN sensitivity model from measured online bytes and public availability-flight counts. This is not a substitute for Linux \`tc\` or real-cloud measurements.
- File: \`${RPM_QUERY_AUDIT_CSV}\`
- Rows: ${RPM_QUERY_AUDIT_ROWS}
- Scope: optional RPM-CNIM randomized-digest entropy audit. Status: ${RPM_QUERY_AUDIT_STATUS}. Even when generated, rows carry \`paper_usable=false\` until the matching projected-error and RLWE-estimator audit also pass.

## Paper-to-Artifact Map

- Typed transitions / \`Share2HE\` / \`D2S-Refresh\`: \`protocol/examples/silent/src/bin/silent_reproduce.rs\` emits \`hss,share2he_d2s_hss_mul_slots_*\`; the reusable implementation path is \`protocol/crates/silent-operators/src/hss_slots.rs\`, backed by \`core/crates/silent-hss/src/ops.rs\` and \`core/crates/silent-hss/src/keygen.rs\`.
- \`D2S-Refresh\` availability barriers: \`protocol/examples/silent/src/bin/silent_params_report.rs\` emits \`protocol_model,bert_base_refresh_barriers\`.
- Appendix C parameter audit: \`protocol/examples/silent/appendix_c/README.md\` maps the paper audit to \`${PARAMS_CSV}\`, \`${CMATMUL_CSV}\`, \`${PMPE_AUDIT_CSV}\`, \`${RPM_QUERY_AUDIT_CSV}\`, and the stricter complete-audit script \`protocol/examples/silent/scripts/make_rpm_parameter_audit.py\`; reusable preset labels such as \`current-bfv-v1\` and \`current-hss-v1\` are defined in \`core/crates/silent-params/src/presets/current.rs\`.

## Mapping

- \`linear,rpm_cnim_nimvm_{dim}x{dim}\`: RPM-CNIM/NIMVM operator benchmark. \`smoke\` uses 8x8 toy-size parameters; \`full\` uses 32x32 with p=65537.
- \`linear,rpm_cnim_nimvm_{dim}x{dim}_online\`: same linear map with server-side static setup excluded from timed online path.
- \`linear,rpm_cnim_nimvm_{dim}x{dim}_row_block4\`: row-blocked RPM-CNIM ablation.
- \`linear,rpm_cnim_nimvm_{dim}x{dim}_row_block4_online\`: row-blocked online path with server setup excluded.
- \`hss,share2he_d2s_hss_mul_slots_{lanes}\`: typed Share2HE/D2S-Refresh conversion reproducer plus HSS multiplication path. \`full\` uses \`current-hss-v1\`.
- \`runtime,silent_net_value_roundtrip_u64x8\`: \`RuntimeSession\` value codec plus in-memory \`silent-net\` framed transport, including frame and byte counters in \`extra\`.
- \`runtime,silent_net_tcp_loopback_value_roundtrip_u64x8\`: same runtime value path over real TCP loopback sockets through \`silent-net::transport::tcp\`.
- \`runtime,silent_net_tcp_multiprocess_pingpong_u64x8\`: two spawned OS processes, one listening and one connecting over TCP, running a RuntimeSession ping-pong through \`silent-net\`.
- \`direction_validation,gate{1,2,3,4}_*\` and \`direction_validation,silent_ofpmpe_direction_decision\`: RNS dyadic OF-PMPE nonlinear direction gates for MeanBetaShift softmax, BERT-base nonlinear online/transport accounting, non-trusted preprocess accounting, BumbleBee-aligned nonlinear communication budget, and a concrete RLWE-AHE cross-term TaylorCorr sample gate.
- \`e2e,silent_thor_style_transformer_setup\` / \`e2e,silent_thor_style_transformer_online\`: compact one-block integration sanity harness that exercises reusable runtime, linear, conversion, and nonlinear operators; the BERT-base paper numbers come from the component-composed rows and the BERT-shaped audits above.
- \`runtime_model,bert_base_refresh_barriers_tcp_loopback\`: measured TCP-loopback RTT scaled to the paper's \(R=24\) refresh barriers, with LAN/WAN RTT components reported in \`extra\`.
- \`e2e_model,bert_base_refresh_barriers\`: \(R=24\) explicit D2S-Refresh availability barriers.
EOF

echo "[silent] measured CSV: ${MEASURED_CSV}"
echo "[silent] parameter CSV: ${PARAMS_CSV}"
echo "[silent] cmatmul audit CSV: ${CMATMUL_CSV}"
echo "[silent] pmpe audit CSV: ${PMPE_AUDIT_CSV}"
echo "[silent] WAN sensitivity CSV: ${WAN_SENSITIVITY_CSV}"
echo "[silent] optional RPM query-hiding entropy CSV: ${RPM_QUERY_AUDIT_CSV} (${RPM_QUERY_AUDIT_STATUS})"
echo "[silent] summary: ${SUMMARY_MD}"
