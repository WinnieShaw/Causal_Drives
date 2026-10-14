#!/usr/bin/env python3
"""Regenerate both model scales with a longer cap, retaining the original equations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import run_mavis_model_scaling as engine
from mavis_supplement_runtime import eos_ids, loading_lock, sha256, write_state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "launch", "worker", "merge"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_size", choices=("8b", "32b"), required=True)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--gpu_max_memory", default="110GiB")
    parser.add_argument("--max_new_tokens", type=int,
                        default=int(os.environ.get("MAVIS_COMPLETE_MAX_NEW_TOKENS", "256")))
    for key in ("core_script", "summary_script", "safe_scoring_script", "local_data_script"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args(argv)
    if not 128 <= args.max_new_tokens <= 1024:
        parser.error("max_new_tokens must be 128..1024")
    if not 0 <= args.worker_id < 8:
        parser.error("worker_id must be 0..7")
    for key in ("project_root", "model_path", "dataset_path", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if not args.output_dir.is_relative_to(args.project_root / "outputs" / "answer_source_validation"):
        parser.error("Output must be inside outputs/answer_source_validation")
    base = Path(__file__).resolve().parent
    for key, name in (("core_script", "compute_mavis_drive_scores_logdomain.py"),
                      ("summary_script", "summarize_mavis_drive_scores.py"),
                      ("safe_scoring_script", "mavis_safe_scoring.py"),
                      ("local_data_script", "mavis_local_data.py")):
        setattr(args, key, (getattr(args, key) or base / name).resolve())
    args.fixed_source = None  # Both scales generate; never silently freeze old 8B answers.
    args.batch_size = 20
    engine.memory_bytes(args.gpu_max_memory)
    return args


def configure(args):
    engine.MAX_NEW_TOKENS = args.max_new_tokens
    # The coordinator must spawn this entry, not the legacy 64-token entry.
    engine.__file__ = str(Path(__file__).resolve())
    os.environ["MAVIS_COMPLETE_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    endings = eos_ids(args.model_path)
    original_validate = engine.validate_rows

    def validate_rows(rows, indices):
        original_validate(rows, indices)
        for row in rows:
            generation = row["generation"]
            ids = generation["generated_token_ids"]
            ended = ids[-1] in endings
            if not ended and len(ids) < args.max_new_tokens:
                raise ValueError(f"Generation stopped without EOS or cap: {row['global_index']}")
            generation.update({"max_new_tokens": args.max_new_tokens, "ended_with_eos": ended,
                               "hit_length_limit_without_eos": not ended and len(ids) == args.max_new_tokens})
            row["schema"]["supplement_protocol"] = "complete_answer_generation_v1"
    engine.validate_rows = validate_rows


def worker(args):
    from transformers import Qwen3VLForConditionalGeneration
    original = Qwen3VLForConditionalGeneration.from_pretrained

    def serial_load(cls, *positional, **keywords):
        with loading_lock(args.output_dir, args.worker_id):
            model = original(*positional, **keywords)
            write_state(args.output_dir, args.worker_id, "weights_loaded")
            return model
    Qwen3VLForConditionalGeneration.from_pretrained = classmethod(serial_load)
    engine.run_worker(args)
    write_state(args.output_dir, args.worker_id, "complete")


def merge(args):
    engine.merge_and_summarize(args)
    rows = engine.read_rows(args.output_dir / "mavis_drive_scores_k20_0-499_merged.jsonl")
    engine.write_json_new(args.output_dir / "generation_completion_summary.json", {
        "n": len(rows), "model_size": args.model_size, "max_new_tokens": args.max_new_tokens,
        "mean_length": sum(r["generation"]["num_generated_tokens"] for r in rows) / len(rows),
        "ended_with_eos": sum(r["generation"]["ended_with_eos"] for r in rows),
        "hit_length_limit_without_eos": sum(r["generation"]["hit_length_limit_without_eos"] for r in rows),
        "entry_sha256": sha256(__file__), "generation": "greedy; original prompt; both scales regenerate",
        "equations_unchanged": True, "temporal_stride": 2,
    })


def main():
    engine.normalize_allocator_environment()
    args = parse_args()
    configure(args)
    if args.preflight:
        engine.preflight(args, check_gpus=False)
    elif args.worker:
        worker(args)
    elif args.merge:
        merge(args)
    else:
        original_merge = engine.merge_and_summarize
        def finish(namespace):
            # Avoid recursively calling the patched engine merger.
            engine.merge_and_summarize = original_merge
            merge(namespace)
        engine.merge_and_summarize = finish
        engine.launch(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
