"""Plot all four MAVIS domains from complete paired visual measurements."""
from pathlib import Path
import csv
import json
import sys
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/derived/domain_visual_response/summary.json"
OUT = ROOT / "figures/pdf"
sys.path.insert(0, str(ROOT / "figures/source/corrected_results"))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .7, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "savefig.facecolor": "white"})
summary = json.loads(SOURCE.read_text(encoding="utf-8"))
rows = summary["domains"]
domains = ["Science", "Art & Design", "Tech & Engineering", "Others"]
assert sum(next(r["n"] for r in rows if r["domain"] == d) for d in domains) == 500
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.15))
fig.subplots_adjust(left=.20, right=.985, bottom=.29, top=.88, wspace=.23)
for ax, conditions, labels, letter, caption in zip(
        axes, [("blur_8", "blur_27"), ("mismatch_42", "mismatch_137")],
        [("Radius 8", "Radius 27"), ("Seed 42", "Seed 137")], "ab",
        ["Blur-strength response", "Image-mismatch response"]):
    for condition, label, offset, color, marker in zip(conditions, labels, [.12, -.12],
                                                    ["#4477AA", "#CC6677"], ["o", "s"]):
        selection = [next(r for r in rows if r["domain"] == d and r["condition"] == condition)
                     for d in domains]
        means = np.array([r["mean_bits"] for r in selection])
        errors = np.array([[r["mean_bits"] - r["ci_low"] for r in selection],
                           [r["ci_high"] - r["mean_bits"] for r in selection]])
        ax.errorbar(means, np.arange(3, -1, -1) + offset, xerr=errors, fmt=marker,
                    color=color, markersize=4, elinewidth=1.2, capsize=2.5, label=label)
    ax.set_xlim(0, 45)
    ax.set_ylim(-.6, 3.6)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks(range(3, -1, -1))
    if letter == "a":
        ax.set_yticklabels([f"{d}\n(n = {selection[i]['n']})" for i, d in enumerate(domains)])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_xlabel("VCD decrease (bits)", labelpad=6)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False,
              handlelength=1.5, columnspacing=1.3, fontsize=8)
    box = ax.get_position()
    fig.text((box.x0 + box.x1)/2, .065, f"({letter}) {caption}", ha="center", va="center",
             fontsize=8.5, fontweight="semibold")
OUT.mkdir(parents=True, exist_ok=True)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=OUT / "domain_visual_response.alignment.json",
                                   overlay_svg=OUT / "domain_visual_response.alignment.svg",
                                   require_panel_labels=False, strict=True)
fig.savefig(OUT / "domain_visual_response.pdf")
fig.savefig(OUT / "domain_visual_response.svg")
fig.savefig(OUT / "domain_visual_response.png", dpi=600)
plt.close(fig)
with (OUT / "source_data.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(OUT / "domain_visual_response.pdf")
