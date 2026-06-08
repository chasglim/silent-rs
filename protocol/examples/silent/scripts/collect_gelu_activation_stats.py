#!/usr/bin/env python3
"""Collect BERT GeLU pre-activation statistics for GLUE task probes.

The script hooks the output of each BERT intermediate dense layer, i.e. the
tensor consumed by GeLU. It writes streaming aggregate statistics and
reservoir-sampled quantiles. These rows are diagnostics for choosing public
approximation ranges; they are not private-inference protocol results.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from run_glue_task_fidelity import DEFAULT_MODELS, TASK_COLUMNS, Example, load_examples


@dataclasses.dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    abs_gt_1: int = 0
    abs_gt_2: int = 0
    abs_gt_3: int = 0
    abs_gt_4: int = 0
    abs_gt_6: int = 0
    reservoir: list[np.ndarray] = dataclasses.field(default_factory=list)

    def update(self, values: torch.Tensor, sample_per_batch: int, rng: np.random.Generator) -> None:
        flat = values.detach().float().cpu().reshape(-1).numpy()
        if flat.size == 0:
            return
        self.count += int(flat.size)
        self.total += float(np.sum(flat, dtype=np.float64))
        self.total_sq += float(np.sum(flat.astype(np.float64) ** 2))
        self.min_value = min(self.min_value, float(np.min(flat)))
        self.max_value = max(self.max_value, float(np.max(flat)))
        abs_flat = np.abs(flat)
        self.abs_gt_1 += int(np.sum(abs_flat > 1.0))
        self.abs_gt_2 += int(np.sum(abs_flat > 2.0))
        self.abs_gt_3 += int(np.sum(abs_flat > 3.0))
        self.abs_gt_4 += int(np.sum(abs_flat > 4.0))
        self.abs_gt_6 += int(np.sum(abs_flat > 6.0))
        if sample_per_batch > 0:
            take = min(sample_per_batch, flat.size)
            idx = rng.choice(flat.size, size=take, replace=False)
            self.reservoir.append(flat[idx])

    def row(self, layer: str, task: str, split: str, model: str, examples: int) -> dict[str, Any]:
        sample = np.concatenate(self.reservoir) if self.reservoir else np.array([], dtype=np.float32)
        mean = self.total / self.count if self.count else 0.0
        var = max(self.total_sq / self.count - mean * mean, 0.0) if self.count else 0.0
        quantiles = (
            np.quantile(sample, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
            if sample.size
            else np.zeros(7)
        )
        return {
            "task": task,
            "split": split,
            "model": model,
            "examples": examples,
            "layer": layer,
            "count": self.count,
            "mean": mean,
            "std": var ** 0.5,
            "min": self.min_value,
            "max": self.max_value,
            "q001": float(quantiles[0]),
            "q01": float(quantiles[1]),
            "q05": float(quantiles[2]),
            "q50": float(quantiles[3]),
            "q95": float(quantiles[4]),
            "q99": float(quantiles[5]),
            "q999": float(quantiles[6]),
            "abs_gt_1_rate": self.abs_gt_1 / self.count if self.count else 0.0,
            "abs_gt_2_rate": self.abs_gt_2 / self.count if self.count else 0.0,
            "abs_gt_3_rate": self.abs_gt_3 / self.count if self.count else 0.0,
            "abs_gt_4_rate": self.abs_gt_4 / self.count if self.count else 0.0,
            "abs_gt_6_rate": self.abs_gt_6 / self.count if self.count else 0.0,
            "paper_usable": False,
            "paper_usable_reason": "activation_distribution_diagnostic_only",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_COLUMNS), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--local-glue-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-per-batch-layer", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def batches(examples: list[Example], size: int) -> Iterable[list[Example]]:
    for start in range(0, len(examples), size):
        yield examples[start : start + size]


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    model_name = args.model or DEFAULT_MODELS[args.task]
    examples = load_examples(args.task, args.split, args.local_glue_dir, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    layer_stats: dict[int, RunningStats] = {}
    aggregate = RunningStats()
    handles = []

    base = getattr(model, "bert", None)
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", [])
    for idx, layer in enumerate(layers):
        layer_stats[idx] = RunningStats()

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                layer_stats[layer_idx].update(output, args.sample_per_batch_layer, rng)
                aggregate.update(output, args.sample_per_batch_layer, rng)

            return hook

        handles.append(layer.intermediate.dense.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        for batch in batches(examples, args.batch_size):
            texts_a = [ex.text_a for ex in batch]
            texts_b = [ex.text_b for ex in batch] if batch[0].text_b is not None else None
            encoded = tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            _ = model(**encoded)

    for handle in handles:
        handle.remove()

    rows = [aggregate.row("all", args.task, args.split, model_name, len(examples))]
    rows.extend(
        layer_stats[idx].row(str(idx), args.task, args.split, model_name, len(examples))
        for idx in sorted(layer_stats)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    all_row = rows[0]
    print(f"wrote {args.output}")
    print(
        "all: "
        f"q001={all_row['q001']:.4f} q999={all_row['q999']:.4f} "
        f"abs_gt_3={all_row['abs_gt_3_rate']:.6f} abs_gt_4={all_row['abs_gt_4_rate']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
