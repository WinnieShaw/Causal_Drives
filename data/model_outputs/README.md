# Model-generated result records

This directory contains the completed model-output records used by the released experiments. It includes generated answers/prefixes, token IDs and token traces, per-token and aggregate scores, intervention conditions, and stable sample identifiers. The files are JSON Lines compressed with gzip (`.jsonl.gz`); each line is one record.

To keep the release focused on results rather than redistributing benchmark inputs, the package omits source questions, reference answers/facts, benchmark image payloads, and local filesystem paths. The model-generated outputs and their score arrays are retained. No model weights or benchmark images are included. `package_model_outputs.py` documents the sanitization rule and can repackage JSONL run outputs.

`raw/` preserves separate completed run outputs. Some runs contain both chunk files and a merged file; these are intentionally both retained where available so the original run layout remains auditable. `mavis_scaling_safe/8b/frozen_source_legacy.jsonl.gz` is included as a clearly named historical output, not as a current result; use the corresponding `mavis_complete` or `mavis_scaling_safe` merged result for the current analysis.
