#!/usr/bin/env python3
"""Fixed-answer visual input sensitivity, from audited complete output.

Quantitative grid; style-only inheritance of the manuscript Arial/blue/grey
theme. Two complementary roles: native VCD intervention response and a
same-question probability control. Paired decreases are target minus perturbed
image, computed within each of all 500 answers. CI is the supplied 95% paired
answer bootstrap interval, 5,000 draws with seed 42. No outcome selection,
hidden-source label, source fraction, or superiority claim is introduced.

The native cumulative VCD and conditional local probability lift have different
units and interventions. Separate axes preserve their supplied definitions.
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
CACHE = PROJECT / "data/derived/visual_input_sensitivity"
OUT = PROJECT / "figures/pdf"
STEM = "fixed_answer_visual_sensitivity"
BLUE, GREY, INK = "#4477AA", "#777777", "#262626"
CONDITIONS = ["blur_8", "blur_27", "mismatch_42", "mismatch_137", "blank_black", "blank_gray", "blank_white"]
LABELS = ["Blur (radius 8)", "Blur (radius 27)", "Mismatch (seed 42)", "Mismatch (seed 137)",
          "Black blank †", "Gray blank ‡", "White blank"]
METRICS = ["mean_vcd", "mean_local_visual_lift"]

# This existing checker measures final Matplotlib plot areas without redrawing.
alignment_file = PROJECT / "figures/source/corrected_results/audit_panel_alignment.py"
spec = importlib.util.spec_from_file_location("audit_panel_alignment", alignment_file)
alignment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = alignment
spec.loader.exec_module(alignment)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def load_data(cache):
    summary_path, sample_path = cache / "audit_summary.json", cache / "audit_per_sample.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with sample_path.open(encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle))
    if summary["rows"] != 500 or len(samples) != 500 or [int(r["global_index"]) for r in samples] != list(range(500)):
        raise ValueError("Require every completed unique example 0..499")
    if summary["bootstrap"] != {"unit": "answer", "draws": 5000, "seed": 42, "equal_answer_weight": True}:
        raise ValueError("The supplied paired bootstrap protocol changed")
    if not summary["all_conditions_and_trajectories_complete_finite"] or not summary["protocol"]["fixed_answer"]:
        raise ValueError("Require complete finite fixed-answer interventions")
    if summary["protocol"]["source_scores_used_for_inference"]:
        raise ValueError("Unsafe legacy scores must not enter the evidence")
    for row in samples:
        if float(row["blank_black_mean_vcd"]) != 0 or float(row["blank_gray_mean_local_visual_lift"]) != 0:
            raise ValueError("Structural reference identities do not hold")
    rows = []
    draws = np.random.default_rng(42).integers(0, 500, size=(5000, 500))
    for panel, metric in zip(["a", "b"], METRICS):
        for name, label in zip(CONDITIONS, LABELS):
            key = f"target_minus_{name}_{metric}"
            values = np.array([float(r[key]) for r in samples])
            direct = np.array([float(r[f"target_{metric}"]) - float(r[f"{name}_{metric}"]) for r in samples])
            if not np.isfinite(values).all() or not np.allclose(values, direct, rtol=0, atol=1e-10):
                raise ValueError("Incomplete/mismatched paired values")
            saved = summary["paired_effects"][key]
            low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
            if not np.allclose([values.mean(), low, high], [saved["mean"], saved["ci_low"], saved["ci_high"]], rtol=0, atol=1e-10):
                raise ValueError("Recomputed mean/CI does not match audited evidence")
            rows.append({"panel": panel, "condition": name, "display_label": label, "metric": metric,
                         "unit": "bits" if panel == "a" else "bits/token", "n": 500,
                         "mean": saved["mean"], "ci_low": saved["ci_low"], "ci_high": saved["ci_high"]})
    return rows, summary, summary_path, sample_path


def panel_label(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-18, 7),
                textcoords="offset points", ha="left", va="bottom", fontsize=9,
                fontweight="bold", clip_on=False)


def draw(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.45), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=.21, right=.992, bottom=.31, top=.93, wspace=.65)
    for ax, panel, title, limits, ticks, value_x, xlabel in zip(
        axes, ["a", "b"], ["Native VCD response", "Full-question control"],
        [(-1.0, 45), (-.05, 2.2)], [[0, 10, 20, 30, 40], [0, .5, 1, 1.5, 2]], [43.0, 2.04],
        ["VCD decrease (bits)", "Conditional lift decrease (bits/token)"]):
        selected = [row for row in rows if row["panel"] == panel]
        for row, y in zip(selected, range(6, -1, -1)):
            name, value = row["condition"], row["mean"]
            color = GREY if name.startswith("blank_") else BLUE
            marker = "D" if name.startswith("blank_") else "s" if name.startswith("mismatch_") else "o"
            ax.errorbar(value, y, xerr=[[value - row["ci_low"]], [row["ci_high"] - value]],
                        fmt=marker, markersize=4.3, markeredgewidth=.8, markerfacecolor=color,
                        markeredgecolor=color, color=color, ecolor=color,
                        elinewidth=1.15, capsize=2.3, capthick=1.0, zorder=3)
            ax.text(value_x, y, f"{value:.2f}", ha="right", va="center", fontsize=7.5, color=color)
        ax.set_yticks(range(6, -1, -1), LABELS)
        ax.set_ylim(-.6, 6.6)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.axvline(0, color="#999999", linestyle=(0, (3, 2)), linewidth=.75, zorder=0)
        ax.set_xlabel(xlabel, labelpad=7)
        ax.set_axisbelow(True)
        ax.grid(False)
        bbox = ax.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, .12, f"({panel}) {title}",
                 ha="center", va="center", fontsize=9, fontweight="semibold")
    fig.text(.995, .978, "Fixed answers (n = 500)", ha="right", va="center", fontsize=7.5, color=GREY)
    fig.text(.03, .035, "† Black blank: VCD = 0.   ‡ Gray blank: conditional lift = 0 (reference conditions).",
             fontsize=7, ha="left", va="center", color=GREY)
    fig.canvas.draw()
    alignment.require_matplotlib_panel_alignment(
        fig, json_out=out / f"{STEM}.alignment.json", overlay_svg=out / f"{STEM}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
    # Retain the requested exact double-column figure size; no tight crop.
    fig.savefig(out / f"{STEM}.pdf")
    fig.savefig(out / f"{STEM}.svg")
    fig.savefig(out / f"{STEM}.png", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rows, summary, summary_path, sample_path = load_data(args.cache)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "source_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    protocol = summary["protocol"]
    provenance = {"source_summary": str(summary_path.relative_to(PROJECT)),
                  "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                  "source_sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                  "raw_merged_sha256": summary["merged_sha256"], "n": 500, "excluded_answers": 0,
                  "model": "Qwen3-VL-8B-Instruct", "dataset": "MAVIS", "fixed_answer": True,
                  "bootstrap": summary["bootstrap"], "paired_difference": "target minus perturbed image",
                  "token_mask": "non-special generated token positions; equal weight per answer",
                  "native_vcd": "original cumulative log ratio, within-answer mean; null question with black-image denominator",
                  "conditional_lift": "local same-question token log-likelihood lift versus gray image; separately measured probability control",
                  "structural_controls": {"black_blank_native_vcd": "zero by identical numerator/denominator",
                                           "gray_blank_conditional_lift": "zero by identical image reference"},
                  "blur_radii_448px_image": protocol["blur_radii_on_448px_image"],
                  "mismatch_seeds": protocol["mismatch_seeds"], "scoring_backend": protocol["scoring_backend"],
                  "source_scores_used_for_inference": protocol["source_scores_used_for_inference"],
                  "blur_dose_response_native_vcd": summary["paired_effects"]["blur_dose_response_native_vcd"],
                  "not_tested": "answer regeneration, correctness, hidden-source identification, superiority to probability baseline",
                  "dimensions_inches": [6.9, 3.45], "png_dpi": 600}
    (args.out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    draw(rows, args.out)
    print(json.dumps({"n": 500, "paired_effects": rows, "out": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
