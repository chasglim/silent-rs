#!/usr/bin/env python3
"""Build a conservative RPM-CNIM parameter-audit CSV.

The audit combines three conditions used by the paper proof:

1. randomized-digest LHL entropy margin,
2. projected-error correctness margin, and
3. externally supplied RLWE estimator bits.

The script intentionally does not run a lattice estimator and does not infer
mask/error bounds. Rows stay ``paper_usable=false`` unless the caller provides
all required inputs and explicitly passes ``--allow-paper-usable``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys


BERT_LAYERS = 12
BERT_SEQ = 128
DEFAULT_P = 65537


SCHEDULE = [
    ("Q/K/V projection", 768, 768, 2048, 384, 3),
    ("Output projection", 768, 768, 2048, 384, 1),
    ("FFN-1 expansion", 768, 3072, 2048, 1536, 1),
    ("FFN-2 contraction", 3072, 768, 4096, 768, 1),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-bits", type=int, default=128)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--p", type=int, default=DEFAULT_P)
    parser.add_argument("--log2-q-2048", type=float, required=True)
    parser.add_argument("--log2-q-4096", type=float, required=True)
    parser.add_argument("--kappa-2048", type=int, required=True)
    parser.add_argument("--kappa-4096", type=int, required=True)
    parser.add_argument("--mask-min-entropy-per-coeff-2048", type=float, required=True)
    parser.add_argument("--mask-min-entropy-per-coeff-4096", type=float, required=True)
    parser.add_argument("--eta0-linf-2048", type=float, required=True)
    parser.add_argument("--eta1-linf-2048", type=float, required=True)
    parser.add_argument("--eta0-linf-4096", type=float, required=True)
    parser.add_argument("--eta1-linf-4096", type=float, required=True)
    parser.add_argument(
        "--activation-coeff-linf-bound",
        type=float,
        default=None,
        help="defaults to centered plaintext worst case floor((p-1)/2)",
    )
    parser.add_argument("--mask-coeff-linf-bound-2048", type=float, required=True)
    parser.add_argument("--mask-coeff-linf-bound-4096", type=float, required=True)
    parser.add_argument("--rlwe-classical-bits-2048", type=float, required=True)
    parser.add_argument("--rlwe-classical-bits-4096", type=float, required=True)
    parser.add_argument("--rlwe-quantum-bits-2048", type=float, default=None)
    parser.add_argument("--rlwe-quantum-bits-4096", type=float, default=None)
    parser.add_argument("--estimator-status-2048", required=True)
    parser.add_argument("--estimator-status-4096", required=True)
    parser.add_argument("--min-estimator-bits", type=float, default=128.0)
    parser.add_argument("--allow-paper-usable", action="store_true")
    return parser.parse_args()


def degree_args(args: argparse.Namespace, degree: int) -> dict[str, float | int | str | None]:
    if degree == 2048:
        return {
            "log2_q": args.log2_q_2048,
            "kappa": args.kappa_2048,
            "mask_h": args.mask_min_entropy_per_coeff_2048,
            "eta0": args.eta0_linf_2048,
            "eta1": args.eta1_linf_2048,
            "mask_coeff": args.mask_coeff_linf_bound_2048,
            "classical": args.rlwe_classical_bits_2048,
            "quantum": args.rlwe_quantum_bits_2048,
            "status": args.estimator_status_2048,
        }
    return {
        "log2_q": args.log2_q_4096,
        "kappa": args.kappa_4096,
        "mask_h": args.mask_min_entropy_per_coeff_4096,
        "eta0": args.eta0_linf_4096,
        "eta1": args.eta1_linf_4096,
        "mask_coeff": args.mask_coeff_linf_bound_4096,
        "classical": args.rlwe_classical_bits_4096,
        "quantum": args.rlwe_quantum_bits_4096,
        "status": args.estimator_status_4096,
    }


def log2_sum(a: float, b: float) -> float:
    if a <= 0.0 and b <= 0.0:
        return float("-inf")
    if a <= 0.0:
        return math.log2(b)
    if b <= 0.0:
        return math.log2(a)
    hi = max(a, b)
    lo = min(a, b)
    return math.log2(hi) + math.log2(1.0 + lo / hi)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> int:
    args = parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "operator",
            "N",
            "input_dim",
            "output_dim",
            "blocks",
            "kappa",
            "log2_q",
            "p",
            "q_rpm",
            "h_inf_bits",
            "lhl_required_bits",
            "entropy_slack_bits",
            "statistical_lhl_pass",
            "b_eta0_linf",
            "b_eta1_linf",
            "b_a_l1",
            "b_r_l1",
            "projected_error_log2",
            "delta_half_log2",
            "projected_error_slack_bits",
            "projected_error_pass",
            "estimator_status",
            "rlwe_classical_bits",
            "rlwe_quantum_bits",
            "estimator_pass",
            "paper_usable",
            "paper_usable_reason",
            "scope",
        ]
    )

    q_rpm = (
        sum(
            BERT_LAYERS * multiplicity * BERT_SEQ * blocks
            for _, _, _, _, blocks, multiplicity in SCHEDULE
        )
        * args.slots
    )
    log_q_rpm = math.ceil(math.log2(q_rpm))
    activation_coeff = args.activation_coeff_linf_bound
    if activation_coeff is None:
        activation_coeff = (args.p - 1) / 2.0
    log2_p = math.log2(args.p)

    for operator, input_dim, output_dim, degree, blocks, _multiplicity in SCHEDULE:
        params = degree_args(args, degree)
        log2_q = float(params["log2_q"])
        kappa = int(params["kappa"])
        mask_h = float(params["mask_h"])
        eta0 = float(params["eta0"])
        eta1 = float(params["eta1"])
        mask_coeff = float(params["mask_coeff"])
        classical = float(params["classical"])
        quantum = params["quantum"]
        quantum_value = "" if quantum is None else f"{float(quantum):.3f}"
        status = str(params["status"])

        h_inf = kappa * degree * mask_h
        lhl_required = degree * log2_q + 2 * (args.lambda_bits + log_q_rpm)
        entropy_slack = h_inf - lhl_required
        entropy_pass = entropy_slack >= 0.0

        b_a_l1 = input_dim * activation_coeff
        b_r_l1 = degree * mask_coeff
        term0 = eta0 * b_a_l1
        term1 = kappa * eta1 * b_r_l1
        projected_log2 = log2_sum(term0, term1)
        delta_half_log2 = log2_q - log2_p - 1.0
        projected_slack = delta_half_log2 - projected_log2
        projected_pass = projected_slack > 0.0

        estimator_bits = classical
        if quantum is not None:
            estimator_bits = min(estimator_bits, float(quantum))
        estimator_pass = estimator_bits >= args.min_estimator_bits and status.lower() in {
            "pass",
            "screen-pass",
            "estimator-pass",
        }

        paper_usable = (
            args.allow_paper_usable and entropy_pass and projected_pass and estimator_pass
        )
        if paper_usable:
            reason = "complete_audit_passed_with_explicit_allow_paper_usable"
        else:
            reason = (
                "candidate_only_require_complete_parameter_file_and_manual_review"
                if entropy_pass and projected_pass and estimator_pass
                else "audit_condition_failed_or_incomplete"
            )

        writer.writerow(
            [
                operator,
                degree,
                input_dim,
                output_dim,
                blocks,
                kappa,
                f"{log2_q:.3f}",
                args.p,
                q_rpm,
                f"{h_inf:.3f}",
                f"{lhl_required:.3f}",
                f"{entropy_slack:.3f}",
                bool_text(entropy_pass),
                f"{eta0:.6g}",
                f"{eta1:.6g}",
                f"{b_a_l1:.6g}",
                f"{b_r_l1:.6g}",
                f"{projected_log2:.3f}",
                f"{delta_half_log2:.3f}",
                f"{projected_slack:.3f}",
                bool_text(projected_pass),
                status,
                f"{classical:.3f}",
                quantum_value,
                bool_text(estimator_pass),
                bool_text(paper_usable),
                reason,
                "rpm_parameter_audit_requires_external_estimator",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
