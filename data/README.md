# Data provenance

This repository does not redistribute upstream images, full questions,
reference answers, generated answer text, or raw model-scoring traces. The
tracked files under `derived/` contain compact score summaries and plot inputs
with no answer text or image payloads.

## MAVIS

The experiments use the MAVIS test data. Obtain it from the official project
and follow its dataset-card terms:

- Project and setup: <https://github.com/seokwon99/MAVIS>
- The upstream repository provides `python scripts/prepare_data.py --data_dir
  data` to prepare its evaluation data. The 500-row subset and model outputs
  used here are not included in this repository.

The experiment scripts accept the parquet/data path as an argument or
environment variable; inspect each script's `--help` output before a run.

## VLMBias

The source-response experiments use the official VLMBias release. Obtain it
from the upstream project and follow its MIT code license and the dataset
terms:

- Project and data instructions: <https://github.com/anvo25/vlms-are-biased>
- Dataset card: <https://huggingface.co/datasets/anvo25/vlms-are-biased>

Images and prompt/answer records are not copied here.

## Models

Model checkpoints are not included. Download the exact model revisions from
their official model publishers and comply with the corresponding model
licenses. Record the resolved model revision and runtime versions when
reproducing a run; model names alone do not pin an immutable checkpoint.

## Derived results

`derived/` contains non-text sample summaries, confidence intervals, and
curve tables. It is sufficient to regenerate the tracked corrected-result
figures, but it is not a replacement for the original benchmark records or
full teacher-forced traces.
