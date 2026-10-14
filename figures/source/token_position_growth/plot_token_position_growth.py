#!/usr/bin/env python3
"""Plot QCD, VCD and PCD across relative answer-token positions."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_WORK = HERE.parents[2]
ANALYSIS = PROJECT_WORK / "data" / "derived"
CACHE = PROJECT_WORK / "data" / "raw_local_only" / "mavis_complete"
OUT = PROJECT_WORK / "figures" / "pdf"
METRICS = ("QCD", "VCD", "PCD")
MODELS = ("8b", "32b")
COLORS = {"QCD": "#4477AA", "VCD": "#228833", "PCD": "#CC6677"}
MARKERS = {"QCD": "o", "VCD": "s", "PCD": "^"}
ROUNDS = 10_000
SEED = 42
PNG_DPI = 300
TIFF_DPI = 600

def load_model(model: str) -> list[dict]:
    rows = [json.loads(line) for path in (CACHE / model).glob("mavis_drive_scores_[0-9]*.jsonl")
            for line in path.open(encoding="utf-8") if line.strip()]
    rows.sort(key=lambda row: row["global_index"])
    if len(rows) != 500 or [row["global_index"] for row in rows] != list(range(500)):
        raise ValueError(f"Expected 500 unique samples for {model}")
    return rows


def summarize(rows: list[dict], metric: str, draw_indices: np.ndarray) -> list[dict]:
    per_answer = np.full((500, 10), np.nan)
    for i, row in enumerate(rows):
        length = int(row["generation"]["num_generated_tokens"])
        steps = row["drive_metrics"]["per_step"]
        t = np.asarray([int(step["t"]) for step in steps])
        values = np.asarray([float(step[f"{metric}_t"]) for step in steps])
        expected_t = np.arange(1, length + 1, 2)
        if not np.array_equal(t, expected_t):
            raise ValueError(f"Unexpected saved positions for sample {row['global_index']}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {metric} for sample {row['global_index']}")
        bins = np.minimum((t / length * 10).astype(int), 9)
        for b in range(10):
            selected = values[bins == b]
            if selected.size:
                per_answer[i, b] = selected.mean()

    if np.isnan(per_answer).all(axis=0).any():
        raise ValueError(f"Empty relative-position bin for {metric}")

    block_sums = np.zeros((25, 10), dtype=float)
    block_counts = np.zeros((25, 10), dtype=float)
    for block in range(25):
        chunk = per_answer[block * 20:(block + 1) * 20]
        valid = np.isfinite(chunk)
        block_sums[block] = np.nansum(chunk, axis=0)
        block_counts[block] = valid.sum(axis=0)
    # Gather whole support blocks directly. This is the ordinary cluster
    # bootstrap (25 blocks sampled with replacement), without a BLAS matmul.
    boot_sums = block_sums[draw_indices].sum(axis=1)
    boot_counts = block_counts[draw_indices].sum(axis=1)
    boot_means = boot_sums / boot_counts

    output = []
    for b in range(10):
        valid = np.isfinite(per_answer[:, b])
        mean = float(per_answer[valid, b].mean())
        low, high = np.quantile(boot_means[:, b], [.025, .975])
        output.append({"metric": metric, "relative_bin": b + 1,
                       "relative_position": (b + .5) / 10,
                       "n_answers": int(valid.sum()), "mean_bits": mean,
                       "ci95_low": float(low), "ci95_high": float(high)})
    return output


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .7,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def main() -> None:
    configure_style()
    # Resample observed K=20 support blocks only; this does not simulate scores.
    bootstrap_rng = random.Random(SEED)
    draw_indices = np.asarray(
        [[bootstrap_rng.randrange(25) for _ in range(25)] for _ in range(ROUNDS)],
        dtype=np.int16,
    )
    # Use exactly the same answer-equal statistics and cluster intervals as
    # the numerical tables. The cap256 generation records remain immutable.
    with (ANALYSIS / "temporal_source_data.csv").open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    data = {}
    for model in MODELS:
        for metric in METRICS:
            data[(model, metric)] = [{"metric": metric, "relative_bin": int(r["relative_bin"]),
                "relative_position": float(r["relative_position"]), "n_answers": int(r["n_answers"]),
                "mean_bits": float(r["mean_bits"]), "ci95_low": float(r["ci95_low"]),
                "ci95_high": float(r["ci95_high"])} for r in source if r["model"] == model and r["metric"] == metric]
            assert len(data[(model, metric)]) == 10
    report = json.loads((ANALYSIS / "complete_temporal.json").read_text(encoding="utf-8"))
    assert all(report[m]["n"] == 500 and report[m]["ended_with_eos"] == 499 for m in MODELS)
    OUT.mkdir(parents=True, exist_ok=True)
    source_path = OUT / "source_data.csv"
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *next(iter(data.values()))[0].keys()])
        writer.writeheader()
        for model in MODELS:
            for metric in METRICS:
                for row in data[(model, metric)]:
                    writer.writerow({"model": model.upper(), **row})

    # Same observed data, common axes, bottom panel captions.
    sys.path.insert(0, str(PROJECT_WORK / "figures/source/corrected_results"))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    def draw_axis(ax, model):
        for metric in METRICS:
            series = data[(model, metric)]
            x = np.asarray([row["relative_position"] for row in series])
            mean = np.asarray([row["mean_bits"] for row in series])
            low = np.asarray([row["ci95_low"] for row in series])
            high = np.asarray([row["ci95_high"] for row in series])
            ax.fill_between(x, low, high, color=COLORS[metric], alpha=.15, linewidth=0, zorder=1)
            ax.plot(x, mean, color=COLORS[metric], linewidth=1.7,
                    marker=MARKERS[metric], markersize=3.7, markeredgewidth=.45,
                    markeredgecolor="white", label=metric, zorder=3)
        ax.set_xlim(0, 1)
        upper = np.ceil(max(r["ci95_high"] for values in data.values() for r in values) / 50) * 50
        ax.set_ylim(0, upper)
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.set_yticks(np.arange(0, upper + 1, 50))
        ax.set_xlabel("Relative answer-token position", labelpad=5)
        ax.set_ylabel("Mean drive score (bits)", labelpad=5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E8EDF2", linewidth=.6)
        ax.tick_params(length=3, width=.7)

    def save(fig, stem):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(
            fig, json_out=str(stem) + ".alignment.json",
            overlay_svg=str(stem) + ".alignment.svg",
            require_panel_labels=False, tolerance_pt=1.5, strict=True)
        for extension, kwargs in [(".pdf", {}), (".svg", {}), (".png", {"dpi":600})]:
            fig.savefig(str(stem)+extension, **kwargs)
        plt.close(fig)

    mpl.rcParams.update({"font.size":8.5, "axes.labelsize":8.5,
                         "xtick.labelsize":8, "ytick.labelsize":8, "legend.fontsize":8})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.15))
    fig.subplots_adjust(left=.085, right=.985, bottom=.29, top=.88, wspace=.34)
    for ax, model, letter in zip(axes, MODELS, "ab"):
        draw_axis(ax, model)
        bbox=ax.get_position()
        fig.text((bbox.x0+bbox.x1)/2, .07, f"({letter}) Qwen3-VL-{model.upper()}",
                 ha="center", va="center", fontsize=9, fontweight="semibold")
    fig.legend(handles=axes[0].lines, labels=METRICS, loc="upper center",
               bbox_to_anchor=(.5,.99), ncol=3, handlelength=1.8, columnspacing=2.2, frameon=False)
    save(fig, OUT / "token_position_trajectories")

    for model, letter in zip(MODELS, "ab"):
        fig, ax = plt.subplots(figsize=(3.5, 3.15))
        fig.subplots_adjust(left=.17, right=.97, bottom=.29, top=.88)
        draw_axis(ax, model)
        fig.legend(handles=ax.lines, labels=METRICS, loc="upper center",
                   bbox_to_anchor=(.54,.99), ncol=3, handlelength=1.2, columnspacing=1.1, frameon=False)
        fig.text(.56,.07,f"({letter}) Qwen3-VL-{model.upper()}",
                 ha="center",va="center",fontsize=9,fontweight="semibold")
        save(fig, OUT / f"token_position_trajectories_{model}")


if __name__ == "__main__":
    main()
