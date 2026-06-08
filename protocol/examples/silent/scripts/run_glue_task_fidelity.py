#!/usr/bin/env python3
"""Task-level fidelity probe for SILENT-style BERT nonlinear approximations.

This script is deliberately conservative:

- It writes CSV evidence, but it does not edit the paper.
- It marks approximation knobs in every row so partial probes are not confused
  with the final protocol semantics.
- It exits with a dependency error instead of silently fabricating GLUE data.

For paper use, run full dev splits and compare the generated CSV against the
exact approximation semantics used by the protocol implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TASK_COLUMNS = {
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "mrpc": ("sentence1", "sentence2"),
}

LABEL_MAP = {
    "sst2": {"0": 0, "1": 1},
    "qnli": {"entailment": 0, "not_entailment": 1, "0": 0, "1": 1},
    "mrpc": {"0": 0, "1": 1},
}

DEFAULT_MODELS = {
    "sst2": "textattack/bert-base-uncased-SST-2",
    "qnli": "textattack/bert-base-uncased-QNLI",
    "mrpc": "textattack/bert-base-uncased-MRPC",
}

GELU_COEFFS_DEG6 = [
    0.11789354887618581,
    0.5000000000000826,
    0.21580646543889759,
    -1.047558059695886e-14,
    -0.0077040940591314375,
    2.60145656290972e-16,
    0.00011217412609841613,
]

GELU_COEFFS_CALIBRATED_DEG2 = [
    0.00489279750771025,
    0.5,
    0.348188960031138,
]

GELU_COEFFS_FIT_DEG6_DOMAIN3 = [
    0.0084554200525157781,
    0.50000000000000056,
    0.36074596760668393,
    1.4878358876936727e-16,
    -0.037845632863519531,
    -1.3862100840421431e-17,
    0.0018194973154685846,
]

GELU_COEFFS_FIT_DEG6_DOMAIN4 = [
    0.031945655783768248,
    0.49999999999999983,
    0.30978590593117356,
    1.0758715969157264e-17,
    -0.022424334244876963,
    8.4796843682052862e-18,
    0.00068614588871220115,
]


@dataclasses.dataclass
class Example:
    text_a: str
    text_b: str | None
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_COLUMNS), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--local-glue-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means full split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--approx-gelu", action="store_true")
    parser.add_argument(
        "--gelu-template",
        choices=["global_deg6", "calibrated_deg2", "fit_deg6_domain3", "fit_deg6_domain4"],
        default="global_deg6",
    )
    parser.add_argument(
        "--gelu-clip-bound",
        type=float,
        default=0.0,
        help="0 disables clipping; positive b clamps GeLU inputs to [-b,b]",
    )
    parser.add_argument("--gelu-clip-min", type=float, default=None)
    parser.add_argument("--gelu-clip-max", type=float, default=None)
    parser.add_argument(
        "--gelu-coeffs",
        default=None,
        help="comma-separated low-to-high coefficients; overrides --gelu-template",
    )
    parser.add_argument("--approx-softmax", action="store_true")
    parser.add_argument("--softmax-domain", type=float, default=6.0)
    parser.add_argument("--softmax-degree", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("glue_task_fidelity.csv"))
    return parser.parse_args()


def require_datasets():
    try:
        import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency reporting path
        raise SystemExit(
            "Missing optional dependency 'datasets'. Install it or pass "
            "--local-glue-dir pointing at GLUE TSV files. Original error: "
            f"{type(exc).__name__}: {exc}"
        )
    return datasets


def load_examples(task: str, split: str, local_glue_dir: Path | None, limit: int) -> list[Example]:
    if local_glue_dir is None:
        datasets = require_datasets()
        ds = datasets.load_dataset("glue", task, split=split)
        col_a, col_b = TASK_COLUMNS[task]
        examples = [
            Example(str(row[col_a]), str(row[col_b]) if col_b else None, int(row["label"]))
            for row in ds
        ]
    else:
        examples = load_local_glue_tsv(task, split, local_glue_dir)

    if limit and len(examples) > limit:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(examples), size=limit, replace=False))
        examples = [examples[int(i)] for i in idx]
    return examples


def load_local_glue_tsv(task: str, split: str, root: Path) -> list[Example]:
    candidates = [
        root / task.upper() / f"{split}.tsv",
        root / task / f"{split}.tsv",
        root / f"{task}_{split}.tsv",
        root / f"{split}.tsv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise SystemExit(f"Could not find local GLUE TSV for {task}/{split} under {root}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    examples: list[Example] = []
    if task == "sst2":
        for row in rows:
            examples.append(Example(row["sentence"], None, parse_label(task, row["label"])))
    elif task == "qnli":
        for row in rows:
            examples.append(Example(row["question"], row["sentence"], parse_label(task, row["label"])))
    elif task == "mrpc":
        for row in rows:
            text_a = row.get("sentence1") or row.get("#1 String")
            text_b = row.get("sentence2") or row.get("#2 String")
            label = row.get("label") or row.get("Quality")
            if text_a is None or text_b is None or label is None:
                raise SystemExit(f"Unsupported MRPC TSV columns in {path}")
            examples.append(Example(text_a, text_b, parse_label(task, label)))
    else:  # pragma: no cover
        raise SystemExit(f"Unsupported task {task}")
    return examples


def parse_label(task: str, value: Any) -> int:
    text = str(value)
    if text in LABEL_MAP[task]:
        return LABEL_MAP[task][text]
    return int(value)


def gelu_coeffs(name: str, override: str | None = None) -> list[float]:
    if override:
        return [float(item.strip()) for item in override.split(",") if item.strip()]
    if name == "global_deg6":
        return GELU_COEFFS_DEG6
    if name == "calibrated_deg2":
        return GELU_COEFFS_CALIBRATED_DEG2
    if name == "fit_deg6_domain3":
        return GELU_COEFFS_FIT_DEG6_DOMAIN3
    if name == "fit_deg6_domain4":
        return GELU_COEFFS_FIT_DEG6_DOMAIN4
    raise ValueError(f"unknown GeLU template {name}")


def clip_range(clip_bound: float, clip_min: float | None, clip_max: float | None) -> tuple[float | None, float | None]:
    if clip_min is not None or clip_max is not None:
        return clip_min, clip_max
    if clip_bound > 0.0:
        return -clip_bound, clip_bound
    return None, None


def silent_gelu(
    x: torch.Tensor,
    coeffs: list[float],
    clip_min: float | None,
    clip_max: float | None,
) -> torch.Tensor:
    if clip_min is not None or clip_max is not None:
        x = torch.clamp(x, min=clip_min, max=clip_max)
    y = torch.zeros_like(x)
    # Horner evaluation, coefficients ordered low to high.
    for coeff in reversed(coeffs):
        y = y * x + coeff
    return y


class SilentGelu(torch.nn.Module):
    def __init__(self, coeffs: list[float], clip_min: float | None, clip_max: float | None):
        super().__init__()
        self.coeffs = list(coeffs)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silent_gelu(x, self.coeffs, self.clip_min, self.clip_max)


def fit_exp_coeffs(domain: float, degree: int) -> np.ndarray:
    xs = np.linspace(-domain, 0.0, 2048)
    ys = np.exp(xs)
    # np.polyfit returns high-to-low; reverse to low-to-high.
    return np.polyfit(xs, ys, degree)[::-1].copy()


def poly_exp(x: torch.Tensor, coeffs: torch.Tensor, domain: float) -> torch.Tensor:
    x = torch.clamp(x, min=-domain, max=0.0)
    y = torch.zeros_like(x)
    for coeff in reversed(coeffs):
        y = y * x + coeff
    return torch.clamp(y, min=0.0)


@contextlib.contextmanager
def patch_softmax(enabled: bool, domain: float, degree: int, device: torch.device):
    if not enabled:
        yield
        return

    coeffs = torch.tensor(fit_exp_coeffs(domain, degree), dtype=torch.float32, device=device)
    original_softmax = F.softmax

    def approx_softmax(input: torch.Tensor, dim: int | None = None, _stacklevel: int = 3, dtype=None):
        if dim is None:
            dim = -1
        x = input if dtype is None else input.to(dtype)
        finite = torch.isfinite(x)
        masked = torch.where(finite, x, torch.full_like(x, -1e9))
        centered = masked - torch.max(masked, dim=dim, keepdim=True).values
        exp_x = poly_exp(centered, coeffs.to(x.device, x.dtype), domain)
        denom = torch.sum(exp_x, dim=dim, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
        return exp_x / denom

    F.softmax = approx_softmax  # type: ignore[assignment]
    try:
        yield
    finally:
        F.softmax = original_softmax  # type: ignore[assignment]


def patch_gelu(
    model: torch.nn.Module,
    enabled: bool,
    coeffs: list[float],
    clip_min: float | None,
    clip_max: float | None,
) -> int:
    if not enabled:
        return 0
    count = 0
    base = getattr(model, "bert", None)
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", [])
    for layer in layers:
        if hasattr(layer, "intermediate"):
            current = getattr(layer.intermediate, "intermediate_act_fn", None)
            if isinstance(current, torch.nn.Module):
                layer.intermediate.intermediate_act_fn = SilentGelu(coeffs, clip_min, clip_max)
            else:
                layer.intermediate.intermediate_act_fn = lambda x: silent_gelu(
                    x, coeffs, clip_min, clip_max
                )
            count += 1
    return count


def batches(examples: list[Example], size: int) -> Iterable[list[Example]]:
    for start in range(0, len(examples), size):
        yield examples[start : start + size]


def predict(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    batch_size: int,
    max_length: int,
    device: torch.device,
    approx_softmax_enabled: bool,
    softmax_domain: float,
    softmax_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    logits_all: list[np.ndarray] = []
    labels_all: list[int] = []
    model.eval()
    with torch.no_grad(), patch_softmax(
        approx_softmax_enabled, softmax_domain, softmax_degree, device
    ):
        for batch in batches(examples, batch_size):
            texts_a = [ex.text_a for ex in batch]
            texts_b = [ex.text_b for ex in batch] if batch[0].text_b is not None else None
            encoded = tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits.detach().cpu().numpy()
            logits_all.append(logits)
            labels_all.extend(ex.label for ex in batch)
    return np.concatenate(logits_all, axis=0), np.asarray(labels_all, dtype=np.int64)


def metrics(labels: np.ndarray, base_logits: np.ndarray, approx_logits: np.ndarray) -> dict[str, float]:
    base_pred = base_logits.argmax(axis=1)
    approx_pred = approx_logits.argmax(axis=1)
    return {
        "n": float(labels.shape[0]),
        "base_accuracy": float(np.mean(base_pred == labels)),
        "approx_accuracy": float(np.mean(approx_pred == labels)),
        "label_agreement": float(np.mean(base_pred == approx_pred)),
        "logit_linf_mean": float(np.mean(np.max(np.abs(base_logits - approx_logits), axis=1))),
        "logit_l2_mean": float(np.mean(np.linalg.norm(base_logits - approx_logits, axis=1))),
        "confidence_drop_mean": confidence_drop(base_logits, approx_logits),
    }


def confidence_drop(base_logits: np.ndarray, approx_logits: np.ndarray) -> float:
    base_prob = softmax_np(base_logits)
    approx_prob = softmax_np(approx_logits)
    base_pred = base_prob.argmax(axis=1)
    rows = np.arange(base_prob.shape[0])
    return float(np.mean(base_prob[rows, base_pred] - approx_prob[rows, base_pred]))


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model_name = args.model or DEFAULT_MODELS[args.task]

    examples = load_examples(args.task, args.split, args.local_glue_dir, args.limit)
    if not examples:
        raise SystemExit("No examples loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    approx_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    coeffs = gelu_coeffs(args.gelu_template, args.gelu_coeffs)
    gelu_clip_min, gelu_clip_max = clip_range(
        args.gelu_clip_bound, args.gelu_clip_min, args.gelu_clip_max
    )
    patched_layers = patch_gelu(
        approx_model, args.approx_gelu, coeffs, gelu_clip_min, gelu_clip_max
    )

    base_logits, labels = predict(
        base_model,
        tokenizer,
        examples,
        args.batch_size,
        args.max_length,
        device,
        False,
        args.softmax_domain,
        args.softmax_degree,
    )
    approx_logits, _ = predict(
        approx_model,
        tokenizer,
        examples,
        args.batch_size,
        args.max_length,
        device,
        args.approx_softmax,
        args.softmax_domain,
        args.softmax_degree,
    )
    row = metrics(labels, base_logits, approx_logits)
    if not args.approx_gelu and not args.approx_softmax:
        paper_usable_reason = "sanity_no_approximation_not_paper_evidence"
    elif row["base_accuracy"] - row["approx_accuracy"] > 0.005 or row["label_agreement"] < 0.99:
        paper_usable_reason = "degraded_task_fidelity_do_not_use_in_paper"
    else:
        paper_usable_reason = "probe_only_verify_against_final_protocol_semantics_before_use"
    row.update(
        {
            "task": args.task,
            "split": args.split,
            "model": model_name,
            "limit": args.limit or len(examples),
            "max_length": args.max_length,
            "approx_gelu": args.approx_gelu,
            "gelu_template": args.gelu_template,
            "gelu_clip_bound": args.gelu_clip_bound,
            "gelu_clip_min": "" if gelu_clip_min is None else gelu_clip_min,
            "gelu_clip_max": "" if gelu_clip_max is None else gelu_clip_max,
            "gelu_coeff_count": len(coeffs),
            "gelu_coeffs_low_to_high": ";".join(f"{coeff:.17g}" for coeff in coeffs),
            "approx_softmax": args.approx_softmax,
            "softmax_domain": args.softmax_domain,
            "softmax_degree": args.softmax_degree,
            "patched_gelu_layers": patched_layers,
            "paper_usable": False,
            "paper_usable_reason": paper_usable_reason,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists()
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as existing:
            header = existing.readline().strip().split(",")
        if header != list(row.keys()):
            raise SystemExit(
                f"Existing output header does not match current fields: {args.output}"
            )
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"wrote {args.output}")
    for key in ["base_accuracy", "approx_accuracy", "label_agreement", "logit_linf_mean", "confidence_drop_mean"]:
        print(f"{key}={row[key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
