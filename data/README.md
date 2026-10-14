# Data provenance

The repository does not redistribute upstream images, full questions, or
reference-answer text. Compact analysis tables are under `derived/`; sanitized
model-generated records and scoring traces are under [`model_outputs/`](model_outputs/README.md).

## MAVIS

The experiments use the MAVIS test data. Obtain it from the official project
and follow its dataset-card terms:

- Project and setup: <https://github.com/seokwon99/MAVIS>
- The upstream repository provides `python scripts/prepare_data.py --data_dir
  data` to prepare its evaluation data. The upstream benchmark inputs are not
  included in this repository; the model-generated outputs are included in
  sanitized form under `model_outputs/`.

The experiment scripts accept the parquet/data path as an argument or
environment variable; inspect each script's `--help` output before a run.

## VLMBias

The source-response experiments use the official VLMBias release. Obtain it
from the upstream project and follow its MIT code license and the dataset
terms:

- Project and data instructions: <https://github.com/anvo25/vlms-are-biased>
- Dataset card: <https://huggingface.co/datasets/anvo25/vlms-are-biased>

Images and source prompt/answer text are not copied here. Sanitized model
outputs and scores are included under `model_outputs/`.

## Models

Model checkpoints are not included. Download the exact model revisions from
their official model publishers and comply with the corresponding model
licenses. Record the resolved model revision and runtime versions when
reproducing a run; model names alone do not pin an immutable checkpoint.

## Derived results

`derived/` contains compact sample summaries, confidence intervals, and curve
tables. `model_outputs/` contains generated answers, token-level traces, and
model scores with benchmark source text and images removed. Reproducing model
inference still requires obtaining the upstream datasets and checkpoints.
