#!/usr/bin/env python3
"""Sanitize and gzip completed model-output records for public release.

Removes benchmark-provided text and image payloads while preserving model
generations, token traces, scores, interventions, and sample identifiers.
This is intended for the authors' private result cache, not for raw datasets.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

DROP_KEYS = {
    "question", "reference_answer", "ground_truth", "cue_answer",
    "all_verifiable_facts", "reference_facts", "manifest_record", "visual_candidate",
    "cue_candidate", "base_prediction", "counterfactual_ground_truth",
    "image", "image_bytes", "image_path", "image_data", "raw_image",
    "model_path", "dataset_path", "data_root",
}


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in DROP_KEYS}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str) and any(x in value for x in ("/chubao/", "/Users/", "/root/")):
        return "[local path omitted]"
    return value


def scrub_record(value, path=()):
    """Also remove encoded reference text in long-answer scoring records."""
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k in DROP_KEYS:
                continue
            # These are benchmark reference inputs, not model generations.
            if k in {"text", "ids"} and path[-1:] == ("prefix_conditions",) and path[-2:-1] == ("reference",):
                continue
            if k in {"ids", "tokens", "context_ids"} and "reference" in path and "prefix_fixed_suffix_scores" in path:
                continue
            result[k] = scrub_record(v, path + (k,))
        return result
    if isinstance(value, list):
        return [scrub_record(v, path) for v in value]
    if isinstance(value, str) and any(x in value for x in ("/chubao/", "/Users/", "/root/")):
        return "[local path omitted]"
    return value


def convert(source: Path, target: Path, *, gzip_input=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_input else open
    count = 0
    with opener(source, "rt", encoding="utf-8") as src, gzip.open(
        target, "wt", encoding="utf-8", compresslevel=9
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(json.dumps(scrub_record(json.loads(line)), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="Directory containing completed run folders")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--pattern", default="*.jsonl")
    args = p.parse_args()
    for src in sorted(args.source.rglob(args.pattern)):
        rel = src.relative_to(args.source)
        if any(part in {"logs", "analysis", "summaries"} for part in rel.parts):
            continue
        target = args.dest / (str(rel) + ".gz")
        n = convert(src, target)
        print(f"{n}\t{src.stat().st_size}\t{target.stat().st_size}\t{target}")


if __name__ == "__main__":
    main()
