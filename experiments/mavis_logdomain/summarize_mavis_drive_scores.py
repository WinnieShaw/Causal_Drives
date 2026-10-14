#!/usr/bin/env python3
"""Summarize canonical MAVIS drive-score outputs without rerunning the model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("QCD", "VCD", "PCD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def percentile_interval(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if args.bins < 2 or args.bootstrap < 1:
        raise ValueError("Require at least two bins and one bootstrap replicate.")

    rows = load_jsonl(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    binned = {metric: {index: [] for index in range(args.bins)} for metric in METRICS}
    normalized = {metric: {index: [] for index in range(args.bins)} for metric in METRICS}

    for row in rows:
        if row.get("schema", {}).get("vcd_intervention") != "target_image_with_null_question":
            raise ValueError(f"VCD schema mismatch at global_index={row.get('global_index')}")
        drive = row.get("drive_metrics") or {}
        steps = [step for step in drive.get("per_step", []) if int(step.get("t", 0)) > 0]
        length = int((row.get("generation") or {}).get("num_generated_tokens", 0))
        if not steps or length <= 0:
            continue

        sample = {
            "global_index": int(row["global_index"]),
            "length": length,
            "token_f1": float((row.get("text_metrics") or {}).get("token_f1", np.nan)),
            "rouge_l_f1": float((row.get("text_metrics") or {}).get("rouge_l_f1", np.nan)),
        }
        for metric in METRICS:
            sample[metric.lower()] = float(drive[f"avg_{metric.lower()}"])
        sample_rows.append(sample)

        per_sample_bins = {metric: {index: [] for index in range(args.bins)} for metric in METRICS}
        per_sample_norm = {metric: {index: [] for index in range(args.bins)} for metric in METRICS}
        for step in steps:
            relative = (int(step["t"]) - 1) / max(length - 1, 1)
            bin_index = min(int(relative * args.bins), args.bins - 1)
            values = {metric: float(step[f"{metric}_t"]) for metric in METRICS}
            denominator = sum(abs(value) for value in values.values())
            for metric, value in values.items():
                per_sample_bins[metric][bin_index].append(value)
                if denominator > 0:
                    per_sample_norm[metric][bin_index].append(abs(value) / denominator)

        for metric in METRICS:
            for bin_index in range(args.bins):
                if per_sample_bins[metric][bin_index]:
                    binned[metric][bin_index].append(float(np.mean(per_sample_bins[metric][bin_index])))
                if per_sample_norm[metric][bin_index]:
                    normalized[metric][bin_index].append(float(np.mean(per_sample_norm[metric][bin_index])))

    if not sample_rows:
        raise ValueError("No valid canonical drive rows found.")

    rng = np.random.default_rng(args.seed)
    curve_rows = []
    for kind, values_by_metric in (("raw_bits", binned), ("absolute_share", normalized)):
        for metric in METRICS:
            for bin_index in range(args.bins):
                values = np.asarray(values_by_metric[metric][bin_index], dtype=float)
                if len(values) == 0:
                    continue
                boots = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(args.bootstrap)]
                low, high = percentile_interval(boots)
                curve_rows.append({
                    "kind": kind,
                    "metric": metric,
                    "bin": bin_index,
                    "relative_position": (bin_index + 0.5) / args.bins,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                })

    sample_path = output / "per_sample_drive_summary.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    curve_path = output / "relative_position_curve_bootstrap_95ci.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    length = np.asarray([row["length"] for row in sample_rows], dtype=float)
    length_rows = []
    metric_rows = []
    for metric_index, metric in enumerate(METRICS):
        values = np.asarray([row[metric.lower()] for row in sample_rows], dtype=float)
        metric_rng = np.random.default_rng(args.seed + 1009 * metric_index)
        boots = []
        for _ in range(args.bootstrap):
            indices = metric_rng.integers(0, len(values), len(values))
            boots.append(spearman(length[indices], values[indices]))
        low, high = percentile_interval(boots)
        length_rows.append({
            "metric": metric,
            "n": len(values),
            "spearman_rho": spearman(length, values),
            "ci95_low": low,
            "ci95_high": high,
        })
        for conventional in ("token_f1", "rouge_l_f1"):
            quality = np.asarray([row[conventional] for row in sample_rows], dtype=float)
            metric_rows.append({
                "metric": metric,
                "quality_metric": conventional,
                "n": int(np.isfinite(values * quality).sum()),
                "spearman_rho": spearman(values, quality),
            })

    for filename, table in (
        ("length_correlations.csv", length_rows),
        ("conventional_metric_correlations.csv", metric_rows),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)

    summary = {
        "input": args.input,
        "num_valid_samples": len(sample_rows),
        "bins": args.bins,
        "bootstrap": args.bootstrap,
        "schema": ["QCD", "VCD", "PCD"],
        "vcd_intervention": "target_image_with_null_question",
        "outputs": [str(sample_path), str(curve_path)],
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
