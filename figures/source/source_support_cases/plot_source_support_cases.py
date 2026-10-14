"""Unaltered images and native profiles expose different source-support patterns.

Image plate + quantitative profile; three purposive examples, not population
estimates. Existing Arial typography and bottom panel captions are retained.
Source means are one fixed answer's saved-position means; no error bars apply.
"""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "figures/pdf"
sys.path.insert(0, str(ROOT / "figures/source/corrected_results"))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({"font.family": "Arial", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "savefig.facecolor": "white"})
case_path = ROOT / "data/raw_local_only/selected_source_support_cases.json"
cases = json.loads(case_path.read_text(encoding="utf-8"))["selected_cases"]
titles = {140: "Image text adopted", 495: "Object function explained", 384: "Name supplied in question"}
OUT.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.25), gridspec_kw={"height_ratios": [1.15, 1]})
fig.subplots_adjust(left=.075, right=.98, bottom=.17, top=.98, wspace=.24, hspace=.32)
provenance = []
colors = ["#577CA0", "#5F9C91", "#AD829D"]
for col, (case, letter) in enumerate(zip(cases, "abc")):
    index = case["global_index"]
    ax, quantitative = axes[:, col]
    path = ROOT / case["image_path"]
    with Image.open(path) as im:
        w, h = im.size
        # A normalized peer canvas retains each image's original aspect ratio.
        ratio = w / h
        view_w, view_h = (1.0, 1.0 / ratio) if ratio >= 1 else (ratio, 1.0)
        ax.imshow(im, extent=((1-view_w)/2, (1+view_w)/2, (1-view_h)/2, (1+view_h)/2))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("auto")
    ax.axis("off")
    # Correct image aspect using physical axes dimensions; never stretch pixels.
    box = ax.get_position()
    ax.set_aspect("equal", adjustable="datalim")
    quantitative.set_ylim(-20, 210)
    metrics = ["QCD", "VCD", "PCD"]
    values = [case["native_mean_bits"][m] for m in metrics]
    quantitative.bar(metrics, values, width=.55, color=colors, linewidth=0)
    quantitative.axhline(0, color="#8D969F", linewidth=.6)
    quantitative.set_yticks([0, 100, 200])
    quantitative.spines[["top", "right"]].set_visible(False)
    quantitative.spines[["left", "bottom"]].set_color("#ABB2B9")
    quantitative.tick_params(axis="both", length=2.5, labelsize=8)
    quantitative.set_ylabel("Support (bits)" if col == 0 else "", fontsize=8.5)
    if col:
        quantitative.set_yticklabels([])
    for j, value in enumerate(values):
        quantitative.text(j, max(value, 0) + 7, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    lower_box = quantitative.get_position()
    fig.text((lower_box.x0+lower_box.x1)/2, .055, f"({letter}) {titles[index]}", ha="center", va="center",
             fontsize=8.5, fontweight="semibold")
    provenance.append({"global_index": index, "image": str(path.relative_to(ROOT)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "transformation": "aspect-preserving display, no content edits",
                       "native_mean_bits": case["native_mean_bits"],
                       "uncertainty": "none: descriptive one-answer means, not repeated-run estimates"})
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=OUT / "source_support_cases.alignment.json",
                                   overlay_svg=OUT / "source_support_cases.alignment.svg",
                                   require_panel_labels=False, strict=True)
fig.savefig(OUT / "source_support_cases.pdf")
fig.savefig(OUT / "source_support_cases.svg")
fig.savefig(OUT / "source_support_cases.png", dpi=600)
plt.close(fig)
(OUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print(OUT / "source_support_cases.pdf")
