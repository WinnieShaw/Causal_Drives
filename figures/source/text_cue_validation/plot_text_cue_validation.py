#!/usr/bin/env python3
"""Observed text-candidate matching and matched intervention effects.

Figure contract: explicit text candidates measurably change answer behaviour.
Panel a displays every completed example's observed rate (not a random-run
estimate); panel b shows the paired gain with topic-cluster bootstrap CIs.
The seven topics are the resampling units. No row is excluded for this figure.
This is an intervention manipulation check, not hidden-source identification.

Style-only inheritance: established Arial typography and blue/grey palette.
The grouped-bar grammar is adapted to Matplotlib so the same drawing backend
can supply render-time panel alignment and editable vector exports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DEFAULT_DATA = PROJECT / "data/derived/text_cue_summary.json"
DEFAULT_OUT = PROJECT / "figures/pdf"
STEM = "text_cue_behaviour_validation"
BLUE, GREY, INK = "#4477AA", "#999999", "#262626"

# Use the existing backend-neutral checker; do not duplicate shared source.
alignment_file = PROJECT / "figures/source/corrected_results/audit_panel_alignment.py"
spec = importlib.util.spec_from_file_location("audit_panel_alignment", alignment_file)
alignment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = alignment
spec.loader.exec_module(alignment)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def load_data(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for size, expected_n in [("8B", 700), ("32B", 140)]:
        row = data["models"][size]["behaviour"]
        if row["n"] != expected_n or row["n_topics"] != 7:
            raise ValueError("Require the full completed seven-topic cohort")
        domains = row["domains"]
        if len(domains) != 7 or sum(d["n"] for d in domains) != row["n"]:
            raise ValueError("Topic counts do not cover all examples")
        if sum(d["neutral_count"] for d in domains) != row["neutral_count"]:
            raise ValueError("Neutral matching count mismatch")
        if sum(d["cue_count"] for d in domains) != row["cue_count"]:
            raise ValueError("Explicit-cue matching count mismatch")
        neutral = row["neutral_count"] / row["n"] * 100
        cue = row["cue_count"] / row["n"] * 100
        delta = (row["cue_count"] - row["neutral_count"]) / row["n"] * 100
        if not np.allclose([neutral, cue, delta],
                           [row["neutral_rate_pct"], row["cue_rate_pct"], row["paired_delta_pp"]],
                           rtol=0, atol=1e-10):
            raise ValueError("Saved rates do not match exact completed counts")
        low, high = row["paired_delta_domain_ci95_pp"]
        if not low <= delta <= high:
            raise ValueError("Invalid topic-cluster confidence interval")
        rows.append({"model": size, "n": expected_n, "n_topics": 7,
                     "neutral_count": row["neutral_count"], "cue_count": row["cue_count"],
                     "neutral_rate_pct": neutral, "cue_rate_pct": cue,
                     "paired_delta_pp": delta, "ci_low_pp": low, "ci_high_pp": high})
    return rows, data


def panel_label(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-23, 7),
                textcoords="offset points", ha="left", va="bottom", fontsize=9,
                fontweight="bold", clip_on=False)


def draw(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.09, right=0.988, bottom=0.34, top=0.95, wspace=0.34)
    ax = axes[0]
    centers = np.array([0., 1.])
    width = 0.29
    for offset, field, color, label in [(-width / 2, "neutral_rate_pct", GREY, "Neutral prompt"),
                                        (width / 2, "cue_rate_pct", BLUE, "Explicit text cue")]:
        values = [row[field] for row in rows]
        bars = ax.bar(centers + offset, values, width=width, color=color, label=label,
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=INK)
    ax.set_xticks(centers, [f"{row['model']}\n(n = {row['n']})" for row in rows])
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.55, 1.55)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Text-candidate matching (%)")
    ax.set_axisbelow(True)
    # Light reference guides below the data; no guide crosses a value label.
    for guide in [25, 50]:
        ax.axhline(guide, color="#E7E7E7", linewidth=0.6, zorder=0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29), ncol=2,
              handlelength=1.2, handletextpad=0.45, columnspacing=1.2, frameon=False,
              borderaxespad=0)

    ax = axes[1]
    for row, y in zip(rows, [1, 0]):
        value, low, high = [row[key] for key in ["paired_delta_pp", "ci_low_pp", "ci_high_pp"]]
        ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o",
                    color=BLUE, ecolor=BLUE, elinewidth=1.35, markersize=5,
                    markeredgewidth=0.9, capsize=3, capthick=1.1, zorder=3)
        ax.text(value, y + 0.23, f"+{value:.2f} pp  [{low:.2f}, {high:.2f}]",
                ha="center", va="bottom", fontsize=7.5, color=BLUE)
    ax.set_yticks([1, 0], [row["model"] for row in rows])
    ax.set_ylim(-0.40, 1.58)
    ax.set_xlim(-1, 15)
    ax.set_xticks([0, 5, 10, 15])
    ax.axvline(0, color="#8A8A8A", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlabel("Candidate matching increase (pp)", labelpad=7)
    for panel, title, axis in zip("ab", ["Answer behaviour", "Paired effect (95% CI)"], axes):
        bbox = axis.get_position()
        fig.text((bbox.x0+bbox.x1)/2, .055, f"({panel}) {title}",
                 fontsize=9, fontweight="semibold", ha="center", va="center")

    fig.canvas.draw()
    alignment.require_matplotlib_panel_alignment(
        fig, json_out=out / f"{STEM}.alignment.json", overlay_svg=out / f"{STEM}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
    # Do not crop: exact 6.9 × 2.6-inch dimensions are retained in every format.
    fig.savefig(out / f"{STEM}.pdf")
    fig.savefig(out / f"{STEM}.svg")
    fig.savefig(out / f"{STEM}.png", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows, summary = load_data(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "source_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"source_summary": str(args.data.relative_to(PROJECT)),
                  "source_summary_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  "n": 840, "excluded_examples": 0, "model": "Qwen3-VL-Instruct",
                  "dataset": "VLMBias", "bootstrap_unit": summary["behaviour_ci_unit"],
                  "bootstrap": summary["domain_bootstrap"], "ci": "95% percentile interval",
                  "panel_a": "Observed fixed-evaluation-set proportions; no CI plotted",
                  "panel_b": "Matched difference; complete topic clusters resampled",
                  "claim": "Explicit text candidates measurably affect observable answer matching",
                  "not_tested": "Hidden reasoning-source identification or uniform effects in every topic",
                  "dimensions_inches": [6.9, 3.0], "png_dpi": 600}
    (args.out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    draw(rows, args.out)
    print(json.dumps({"rows": rows, "output": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
