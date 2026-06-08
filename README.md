# SILENT Artifact Code

This directory is a standalone copy of the SILENT artifact code extracted from a
larger development workspace. It keeps the SILENT experiment runners plus the
local support crates needed to build and run them.

## Layout

- `protocol/examples/silent`: SILENT experiment crate and scripts.
- `protocol/crates`: small runtime/operator crates used by SILENT.
- `core/crates`: minimal cryptographic support crates required by the
  SILENT artifact.

## Quick Check

```bash
cargo check --manifest-path Cargo.toml -p silent
```

## One-Command Experiments

Run the SILENT artifact experiments from the repository root:

```bash
protocol/examples/silent/scripts/run_silent_experiments.sh smoke
```

This runs SILENT's own Rust experiment runners and audits: microbenchmarks,
runtime transport, end-to-end transformer harness, parameter manifest,
BERT-shaped RPM-CNIM CMatMul audit, RNS OF-PMPE no-wrap audit, and WAN
sensitivity model. It does not run baseline or comparison-system experiments.

For a single-iteration smoke run:

```bash
protocol/examples/silent/scripts/run_silent_experiments.sh 1 smoke
```

For the heavier profile used by the artifact runners where supported:

```bash
protocol/examples/silent/scripts/run_silent_experiments.sh full
```

The script writes CSV and Markdown outputs to `target/silent-experiments/` by
default. Set `OUT_DIR=...` to use a different output directory. The `target/`
directory is generated output and is intentionally ignored by git.
