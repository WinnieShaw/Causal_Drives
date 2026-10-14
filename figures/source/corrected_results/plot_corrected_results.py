#!/usr/bin/env python3
"""Render corrected log-domain result figures from a traceable CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from audit_panel_alignment import require_matplotlib_panel_alignment


HERE = Path(__file__).resolve().parent
DATA = HERE / "corrected_results.csv"
ALIGNED_DATA = HERE.parents[2] / "data" / "derived"
TRAJECTORY_DATA = ALIGNED_DATA / "relative_position_curve_bootstrap_95ci.csv"
LENGTH_DATA = ALIGNED_DATA / "length_correlations.csv"
SAMPLE_DATA = ALIGNED_DATA / "sample_level_bootstrap_95ci.csv"
OUT = HERE.parents[2] / "figures" / "pdf"

COLORS = {
    "QCD": "#4477AA",
    "VCD": "#228833",
    "PCD": "#CC6677",
    "PCD-VCD": "#AA3377",
    "first": "#B8C6D9",
    "last": "#355F8A",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def load_rows() -> list[dict[str, str]]:
    with DATA.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_sample_rows() -> list[dict[str, str]]:
    with SAMPLE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9 or any(int(row["n"]) != 500 for row in rows):
        raise ValueError("Expected all nine full-precision summaries of 500 aligned answers")
    for row in rows:
        values = np.array([float(row[key]) for key in ("estimate", "ci_low", "ci_high")])
        if not np.all(np.isfinite(values)) or not values[1] <= values[0] <= values[2]:
            raise ValueError(f"Invalid sample-level interval for {row['metric']}/{row['position']}")
    return rows


def load_trajectory() -> list[dict[str, str]]:
    with TRAJECTORY_DATA.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["kind"] == "raw_bits"]
    # Plot all three raw-bit curves and all ten bins; other kinds describe
    # different estimands and are not mixed into a drive-in-bits figure.
    if len(rows) != 30:
        raise ValueError("Expected all 30 raw-bit metric/bin summaries")
    for metric in ("QCD", "VCD", "PCD"):
        series = sorted((row for row in rows if row["metric"] == metric), key=lambda row: int(row["bin"]))
        if [int(row["bin"]) for row in series] != list(range(10)):
            raise ValueError(f"Missing or duplicate bins for {metric}")
        for row in series:
            values = np.array([float(row[key]) for key in ("relative_position", "mean", "ci95_low", "ci95_high")])
            if not np.all(np.isfinite(values)) or not values[2] <= values[1] <= values[3]:
                raise ValueError(f"Invalid finite mean/interval for {metric}, bin {row['bin']}")
            if not 0 < int(row["n"]) <= 500:
                raise ValueError(f"Invalid contributing-answer count for {metric}, bin {row['bin']}")
    return rows


def load_length() -> list[dict[str, str]]:
    with LENGTH_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["metric"] for row in rows] != ["QCD", "VCD", "PCD"] or any(int(row["n"]) != 500 for row in rows):
        raise ValueError("Expected QCD/VCD/PCD length correlations from all 500 answers")
    for row in rows:
        values = np.array([float(row[key]) for key in ("spearman_rho", "ci95_low", "ci95_high")])
        if not np.all(np.isfinite(values)) or not values[1] <= values[0] <= values[2]:
            raise ValueError(f"Invalid length-correlation interval for {row['metric']}")
    return rows


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-18, 5),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=OUT / f"{stem}.alignment.json",
        overlay_svg=OUT / f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_summary(
    length: list[dict[str, str]], trajectory: list[dict[str, str]]
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.15), constrained_layout=True)

    ax = axes[0]
    y = np.arange(len(length))[::-1]
    estimates = np.array([f(row, "spearman_rho") for row in length])
    lows = np.array([f(row, "ci95_low") for row in length])
    highs = np.array([f(row, "ci95_high") for row in length])
    labels = [row["metric"] for row in length]
    for yi, label, value, low, high in zip(y, labels, estimates, lows, highs):
        ax.errorbar(
            value,
            yi,
            xerr=np.array([[value - low], [high - value]]),
            fmt="o",
            markersize=4.2,
            color=COLORS[label],
            ecolor=COLORS[label],
            capsize=2.0,
            elinewidth=1.0,
            zorder=3,
        )
    ax.axvline(0, color="#777777", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Spearman ρ with answer length")
    ax.set_title("Length dependence (500 samples)", loc="left", pad=4)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    panel_label(ax, "a")

    ax = axes[1]
    markers = {"QCD": "o", "VCD": "^", "PCD": "s"}
    for metric in ("QCD", "VCD", "PCD"):
        series = sorted((row for row in trajectory if row["metric"] == metric), key=lambda row: int(row["bin"]))
        x = np.array([f(row, "relative_position") for row in series])
        if not np.all(np.diff(x) > 0):
            raise ValueError(f"Non-monotone bin positions for {metric}")
        mean = np.array([f(row, "mean") for row in series])
        low = np.array([f(row, "ci95_low") for row in series])
        high = np.array([f(row, "ci95_high") for row in series])
        ax.fill_between(x, low, high, color=COLORS[metric], alpha=0.16, linewidth=0)
        ax.plot(
            x,
            mean,
            color=COLORS[metric],
            marker=markers[metric],
            markersize=2.8,
            markeredgewidth=0,
            label=metric,
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 310)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Relative answer-token position")
    ax.set_ylabel("Mean drive (bits)")
    ax.set_title("Token-level trajectories (500 samples)", loc="left", pad=4)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", ncol=3, handlelength=1.5)
    panel_label(ax, "b")

    save_figure(fig, "corrected_temporal_summary")


def plot_aligned_sample_summary(rows: list[dict[str, str]]) -> None:
    means = [row for row in rows if row["section"] == "mean"]
    quality = [row for row in rows if row["section"] == "quality"]

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.15), constrained_layout=True)

    ax = axes[0]
    y = np.arange(len(means))[::-1]
    estimates = np.array([f(row, "estimate") for row in means])
    lows = np.array([f(row, "ci_low") for row in means])
    highs = np.array([f(row, "ci_high") for row in means])
    labels = [row["metric"] for row in means]
    # Place exact-value labels in one right-side column beyond the intervals
    # and to the right of the next grid tick, not across a plotted grid line.
    value_column = 20 * np.ceil(np.max(highs) / 20) + 3
    for yi, label, value, low, high in zip(y, labels, estimates, lows, highs):
        ax.errorbar(
            value,
            yi,
            xerr=np.array([[value - low], [high - value]]),
            fmt="o",
            markersize=4.2,
            color=COLORS[label],
            ecolor=COLORS[label],
            capsize=2.0,
            elinewidth=1.0,
            zorder=3,
        )
        ax.text(
            value_column,
            yi,
            f"{value:.1f}",
            va="center",
            ha="left",
            color=COLORS[label],
            fontsize=6.5,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, value_column + 17)
    ax.set_xticks(np.arange(0, value_column + 17.1, 20))
    ax.set_xlabel("Mean drive (bits; 95% bootstrap CI)")
    ax.set_title("Aligned sample-level scores (500 samples)", loc="left", pad=4)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    panel_label(ax, "a")

    ax = axes[1]
    metrics = ["QCD", "VCD", "PCD"]
    outcomes = ["Token F1", "ROUGE-L"]
    base_y = np.arange(len(metrics))[::-1]
    offsets = {"Token F1": 0.11, "ROUGE-L": -0.11}
    markers = {"Token F1": "o", "ROUGE-L": "s"}
    for outcome in outcomes:
        for yi, metric in zip(base_y, metrics):
            row = next(
                item
                for item in quality
                if item["metric"] == metric and item["position"] == outcome
            )
            value = f(row, "estimate")
            low = f(row, "ci_low")
            high = f(row, "ci_high")
            ax.errorbar(
                value,
                yi + offsets[outcome],
                xerr=np.array([[value - low], [high - value]]),
                fmt=markers[outcome],
                markersize=4.0,
                color=COLORS[metric],
                ecolor=COLORS[metric],
                markerfacecolor=COLORS[metric] if outcome == "Token F1" else "white",
                markeredgecolor=COLORS[metric],
                capsize=2.0,
                elinewidth=1.0,
                label=outcome if metric == metrics[0] else None,
                zorder=3,
            )
    ax.axvline(0, color="#777777", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_yticks(base_y, metrics)
    ax.set_xlim(-0.11, 0.22)
    ax.set_xlabel("Spearman ρ with answer-level metric")
    ax.set_title("Answer-metric associations", loc="left", pad=4)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    panel_label(ax, "b")

    save_figure(fig, "aligned_sample_summary")


def plot_recovery(rows: list[dict[str, str]]) -> None:
    recovery = [row for row in rows if row["section"] == "recovery"]
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.55), constrained_layout=True)

    settings = {
        "Question": {"xlim": (-5.5, 13.5), "color": COLORS["QCD"], "label": "Q-Obs error − QCD error"},
        "Prefix": {"xlim": (-0.05, 0.52), "color": COLORS["PCD"], "label": "P-Obs error − PCD error"},
    }

    for panel, (ax, row) in enumerate(zip(axes, recovery)):
        name = row["metric"]
        estimate = f(row, "estimate")
        low = f(row, "ci_low")
        high = f(row, "ci_high")
        cfg = settings[name]
        ax.axvline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=1)
        ax.errorbar(
            estimate,
            0,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="o",
            markersize=5.0,
            color=cfg["color"],
            ecolor=cfg["color"],
            capsize=2.5,
            elinewidth=1.2,
            zorder=3,
        )
        ax.set_xlim(*cfg["xlim"])
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel(f"{cfg['label']} (bits; positive favors causal drive)")
        ax.set_title(f"{name} intervention", loc="left", pad=4)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
        panel_label(ax, chr(ord("a") + panel))

    save_figure(fig, "intervention_recovery_corrected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "temporal", "aligned", "recovery"), default="all")
    args = parser.parse_args()
    if args.figure in ("all", "temporal"):
        plot_temporal_summary(load_length(), load_trajectory())
    if args.figure in ("all", "aligned"):
        plot_aligned_sample_summary(load_sample_rows())
    if args.figure in ("all", "recovery"):
        plot_recovery(load_rows())


if __name__ == "__main__":
    main()
