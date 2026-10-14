#!/usr/bin/env python3
"""Bootstrap full-precision sample-level drive summaries and correlations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[2] / "analysis" / "causal_drive_results" / "cache" / "mavis_aligned"


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
    if np.std(x) == 0 or np.std(y) == 0:
        raise ValueError("Undefined correlation from a constant bootstrap sample")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CACHE / "per_sample_drive_summary.csv")
    parser.add_argument("--output", type=Path, default=CACHE / "sample_level_bootstrap_95ci.csv")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("At least one bootstrap replicate is required")
    with args.input.open(encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle))
    if len(samples) != 500 or sorted(int(row["global_index"]) for row in samples) != list(range(500)):
        raise ValueError("Expected the complete, unique 500-answer aligned sample")
    metrics = ("QCD", "VCD", "PCD")
    outcomes = (("token_f1", "Token F1"), ("rouge_l_f1", "ROUGE-L"))
    drives = {metric: np.array([float(row[metric.lower()]) for row in samples]) for metric in metrics}
    qualities = {key: np.array([float(row[key]) for row in samples]) for key, _ in outcomes}
    if not all(np.all(np.isfinite(values)) for values in (*drives.values(), *qualities.values())):
        raise ValueError("All 500 observations must be finite; do not silently exclude samples")

    # Resample complete answers jointly across all three drives and both
    # outcomes. The same draw IDs are used in every comparable statistic.
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(samples), size=(args.bootstrap, len(samples)))
    result = []
    for metric in metrics:
        values = drives[metric]
        low, high = np.quantile(np.mean(values[indices], axis=1), [0.025, 0.975])
        result.append({
            "section": "mean", "metric": metric, "estimate": float(np.mean(values)),
            "ci_low": float(low), "ci_high": float(high), "position": "",
            "n": len(samples), "bootstrap": args.bootstrap, "seed": args.seed,
        })
    for metric in metrics:
        values = drives[metric]
        for key, display in outcomes:
            quality = qualities[key]
            boots = [spearman(values[draw], quality[draw]) for draw in indices]
            low, high = np.quantile(boots, [0.025, 0.975])
            result.append({
                "section": "quality", "metric": metric, "estimate": spearman(values, quality),
                "ci_low": float(low), "ci_high": float(high), "position": display,
                "n": len(samples), "bootstrap": args.bootstrap, "seed": args.seed,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    metadata = {
        "source": str(args.input), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "n": len(samples), "bootstrap": args.bootstrap, "seed": args.seed,
        "unit": "whole answer", "interval": "percentile 95%",
        "resampling": "same paired draw IDs for all nine statistics", "excluded_samples": 0,
        "output": str(args.output), "statistics": result,
    }
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
