#!/usr/bin/env python3
"""Compute RPM-CNIM multi-query LHL entropy margins for candidate parameters.

This script checks only the randomized-digest entropy side of the audit. It
does not certify projected-error correctness; that must be checked against the
same mask/error distribution before any row is used in the paper.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys


BERT_LAYERS = 12
BERT_SEQ = 128


SCHEDULE = [
    ("Q/K/V projection", 768, 768, 2048, 384, 3),
    ("Output projection", 768, 768, 2048, 384, 1),
    ("FFN-1 expansion", 768, 3072, 2048, 1536, 1),
    ("FFN-2 contraction", 3072, 768, 4096, 768, 1),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-bits", type=int, default=128)
    parser.add_argument("--slots", type=int, default=1, help="online inference slots under one seed")
    parser.add_argument("--log2-q-2048", type=float, default=54.0)
    parser.add_argument("--log2-q-4096", type=float, default=109.0)
    parser.add_argument("--kappa", type=int, default=None)
    parser.add_argument("--kappa-2048", type=int, default=None)
    parser.add_argument("--kappa-4096", type=int, default=None)
    parser.add_argument(
        "--mask-min-entropy-per-coeff",
        type=float,
        default=None,
        help="fallback min-entropy in bits for one coefficient of one mask polynomial",
    )
    parser.add_argument("--mask-min-entropy-per-coeff-2048", type=float, default=None)
    parser.add_argument("--mask-min-entropy-per-coeff-4096", type=float, default=None)
    return parser.parse_args()


def choose(name: str, degree: int, degree_value: float | int | None, fallback: float | int | None) -> float | int:
    if degree_value is not None:
        return degree_value
    if fallback is not None:
        return fallback
    raise SystemExit(
        f"missing --{name}-{degree} or fallback --{name}; "
        "do not infer mask entropy from log2(q) unless the parameter file says so"
    )


def main() -> int:
    args = parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "operator",
            "N",
            "log2_q",
            "kappa",
            "mask_min_entropy_per_coeff",
            "q_rpm",
            "h_inf_bits",
            "lhl_required_bits",
            "entropy_slack_bits",
            "statistical_lhl_pass",
            "paper_usable",
            "paper_usable_reason",
            "scope",
        ]
    )

    total_q = sum(
        BERT_LAYERS * multiplicity * BERT_SEQ * blocks
        for _, _, _, _, blocks, multiplicity in SCHEDULE
    )
    q_rpm = total_q * args.slots
    log_q_rpm = math.ceil(math.log2(q_rpm))

    for operator, _m, _k, degree, _blocks, _multiplicity in SCHEDULE:
        if degree == 2048:
            log2_q = args.log2_q_2048
            kappa = int(choose("kappa", degree, args.kappa_2048, args.kappa))
            mask_h = float(
                choose(
                    "mask-min-entropy-per-coeff",
                    degree,
                    args.mask_min_entropy_per_coeff_2048,
                    args.mask_min_entropy_per_coeff,
                )
            )
        else:
            log2_q = args.log2_q_4096
            kappa = int(choose("kappa", degree, args.kappa_4096, args.kappa))
            mask_h = float(
                choose(
                    "mask-min-entropy-per-coeff",
                    degree,
                    args.mask_min_entropy_per_coeff_4096,
                    args.mask_min_entropy_per_coeff,
                )
            )
        h_inf = kappa * degree * mask_h
        required = degree * log2_q + 2 * (args.lambda_bits + log_q_rpm)
        slack = h_inf - required
        writer.writerow(
            [
                operator,
                degree,
                f"{log2_q:.3f}",
                kappa,
                f"{mask_h:.3f}",
                q_rpm,
                f"{h_inf:.3f}",
                f"{required:.3f}",
                f"{slack:.3f}",
                slack >= 0.0,
                False,
                "entropy_only_requires_projected_error_and_rlwe_estimator_audit_before_paper_use",
                "entropy_only_not_projected_error",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
