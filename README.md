# Causal Drives: experiments and reproduction artifacts

This repository contains the current log-domain experiment code and the
derived outputs used by the accompanying manuscript, together with the initial
public release retained under `legacy/initial_release/` for provenance.

## What is current

- `experiments/mavis_logdomain/`: MAVIS scoring, finite-support and temporal
  analyses, and H200 launch scripts.
- `experiments/source_response/`: source-response/VLMBias experiments and
  their analysis utilities.
- `experiments/long_answer_validation/`: long-answer and reference-fragment
  analysis utilities.
- `analysis/`: CPU-side result analysis.
- `data/derived/`: compact, text-free derived tables and summaries used for
  the reported plots and statistics.
- `data/model_outputs/`: sanitized completed model runs, including generated
  responses, token-level traces, and score records (gzip-compressed JSONL).
- `figures/source/corrected_results/`: source for the corrected temporal,
  aligned-profile, and intervention-recovery figures.
- `figures/pdf/`: PDF figures used in the manuscript source.

The root README and these paths describe the current release. Files in
`legacy/initial_release/` are preserved from the repository's earlier version;
they use the earlier probability-domain estimator and refer to a MiraData
question set that is not part of the current MAVIS/VLMBias experimental
release. Do not use those files to reproduce current results.

## Reproducing derived figures

Install Python 3.10+ with `numpy` and `matplotlib`, then run from the repository
root:

```bash
python figures/source/corrected_results/plot_corrected_results.py --figure all
```

The script writes PDF, SVG, PNG, and TIFF variants to `figures/pdf/`; only the
manuscript PDFs are tracked. Input summary tables are in `data/derived/`.
The related token-position, visual-perturbation, text-cue, and domain-response
scripts are under `figures/source/` and use the compact derived inputs shipped
in `data/derived/`.

The GPU scoring scripts require the original MAVIS data, the specified model
checkpoints, and the runtime dependencies listed in the scripts. Set local
paths explicitly; for example:

```bash
MODEL_PATH=/path/to/model \
MAVIS_DATASET_PATH=/path/to/mavis/test.parquet \
bash experiments/mavis_logdomain/run_mavis_complete_answers_8gpu_front.sh "$PWD"
```

Model weights and original dataset files are not stored in this repository.
See `data/README.md` for upstream sources and download instructions.

## Data and scope

The model-output package preserves generated responses and token-level scores
while omitting upstream images, source questions, reference answers/facts, and
local filesystem paths. Exact model reruns still require obtaining the
upstream data and checkpoints separately. See `data/model_outputs/README.md`
for the package format and sanitization details.

The scripts and derived values are research artifacts, not a claim that every
analysis is an independently validated predictor of answer correctness.
Interpretation should follow the stated sample unit, intervention definition,
and confidence interval in the manuscript.

## Release crosswalk

| Analysis in the manuscript | Code | Public derived inputs/results |
| --- | --- | --- |
| MAVIS log-domain scoring and temporal profiles | `experiments/mavis_logdomain/` | `data/derived/` |
| Finite-support intervention recovery | `experiments/mavis_logdomain/randomized_intervention_recovery_mavis_logdomain.py` | `data/derived/per_sample_recovered.csv`, `figures/pdf/intervention_recovery_corrected.pdf` |
| Fragment-level reference diagnostics | `analysis/reference_diagnostics/analyze_complete_evidence.py` | `data/derived/long_window_stratification.json` |
| VLMBias/source-response runs | `experiments/source_response/` | `data/derived/behavior_summary.json`, score-audit summaries |
| Temporal and controlled-response plots | `figures/source/` | `figures/pdf/` and the summary tables under `data/derived/` |

The qualitative case-panel source is included, but case text and original
benchmark images are intentionally not redistributed. Its PDF is therefore
omitted. Generate that panel locally only after obtaining the corresponding
upstream records and images.

## License and citation

Project code is released under the MIT License (see `LICENSE`). Dataset and
model terms are separate; follow the upstream licenses and cite the source
projects. See `CITATION.cff` for this project citation metadata.
