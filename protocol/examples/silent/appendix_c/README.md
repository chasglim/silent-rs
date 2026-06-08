# Appendix C Parameter Audit

This directory is a paper-to-artifact index for Appendix C.  It intentionally
does not introduce new numbers; it points reviewers to the scripts and outputs
that correspond to the paper's parameter-audit summary.

## Reproduction Outputs

Run the standard reproduction script from the artifact root:

```bash
protocol/examples/silent/scripts/run_silent_experiments.sh smoke
```

It writes the following audit-adjacent outputs under
`target/silent-experiments/`:

- `silent_<mode>_parameters_<timestamp>.csv`: emitted by
  `protocol/examples/silent/src/bin/silent_params_report.rs`; labels the
  validated BFV, HSS, and protocol-model parameter presets used by the artifact.
  The reusable preset definitions live in
  `core/crates/silent-params/src/presets/current.rs`.
- `silent_<mode>_cmatmul_suite_<timestamp>.csv`: emitted by
  `protocol/examples/silent/src/bin/silent_cmatmul_suite.rs`; records
  BERT-shaped RPM-CNIM block geometry, byte counts, timings, and correctness
  checks.
- `silent_<mode>_pmpe_rns_audit_<timestamp>.csv`: emitted by
  `protocol/examples/silent/src/bin/silent_bert_base_runner.rs
  --pmpe-only --pmpe-rns --pmpe-bert-shapes`; records OF-PMPE fixed-point
  no-wrap checks for the BERT-base nonlinear templates.
- `silent_<mode>_rpm_query_hiding_entropy_<timestamp>.csv`: emitted only when
  `SILENT_RUN_RPM_QUERY_AUDIT=1`; records randomized-digest entropy margins
  and is not a complete parameter-security certificate by itself.

## Complete RPM-CNIM Audit Script

Use `protocol/examples/silent/scripts/make_rpm_parameter_audit.py` for the complete
Appendix C-style RPM-CNIM parameter audit.  It combines randomized-digest
entropy, projected-error slack, and externally supplied RLWE-estimator bits.
Rows remain `paper_usable=false` unless all conditions pass and the caller
explicitly supplies `--allow-paper-usable`.

The README one level up contains a CSV-shape example.  Do not cite example
output unless every bound comes from the audited deployment parameter file and
the matching estimator rows are attached.
