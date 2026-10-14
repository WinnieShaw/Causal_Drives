# Experiment scripts

These scripts preserve the scoring and analysis workflows used for the current
release. They are organized by protocol rather than by submission venue.

## Requirements and inputs

Inference runs require the matching model checkpoint, the upstream dataset,
Python dependencies used by the corresponding script, and the intended GPU
count. The checkpoints and original data are not bundled. Provide paths through
the script arguments or environment variables; do not rely on paths from the
original compute cluster.

The `*_8gpu_front.sh` files are launchers for eight visible GPUs. They do not
submit to a scheduler. Start with the launcher README/help and use a separate
output directory for each run. The analysis utilities can run on CPU when their
input artifacts are available.

## Contents

- `mavis_logdomain/`: complete-answer scoring, log-domain score computation,
  support stability, and randomized-support recovery.
- `source_response/`: text-cue/source-response manipulations and VLMBias
  response scoring.
- `long_answer_validation/`: long-answer source-response and fragment analyses.

Raw inference JSONL shards are excluded. Those records contain original
benchmark text and token-level model traces; the public derived tables are
under `../data/derived/`.
