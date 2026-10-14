#!/usr/bin/env python3
"""Run MAVIS drives with full-prompt replay and one complete model per GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback


START, END, SUPPORT, MAX_NEW_TOKENS, NUM_WORKERS = 0, 499, 20, 64, 8
METRICS = ("QCD", "VCD", "PCD")
SCORING_BACKEND = "full_prompt_teacher_forcing_no_kv_reuse"
DATA_BACKEND = "direct_parquet_in_memory_no_hf_arrow_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="CPU assets/dependencies only; no model load")
    mode.add_argument("--launch", action="store_true", help="Foreground coordinator for all eight GPU workers")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--core_script", type=Path)
    parser.add_argument("--summary_script", type=Path)
    parser.add_argument("--safe_scoring_script", type=Path)
    parser.add_argument("--local_data_script", type=Path)
    parser.add_argument("--fixed_source", type=Path,
                        help="Keep the original 500 generated answers during the 8B repair run")
    parser.add_argument("--gpu_max_memory", default="110GiB")
    parser.add_argument("--model_size", choices=("8b", "32b"), default="32b")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=SUPPORT)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    experiment_dir = Path(__file__).resolve().parent
    args.core_script = (args.core_script or experiment_dir / "compute_mavis_drive_scores_logdomain.py").resolve()
    args.summary_script = (args.summary_script or experiment_dir / "summarize_mavis_drive_scores.py").resolve()
    args.safe_scoring_script = (args.safe_scoring_script or experiment_dir / "mavis_safe_scoring.py").resolve()
    args.local_data_script = (args.local_data_script or experiment_dir / "mavis_local_data.py").resolve()
    for key in ("model_path", "dataset_path", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.batch_size != SUPPORT:
        parser.error("The matched comparison uses the established support batch_size=20")
    if not 0 <= args.worker_id < NUM_WORKERS:
        parser.error("worker_id must be in 0..7")
    if not args.output_dir.is_relative_to(args.project_root / "outputs" / "mavis_model_scaling"):
        parser.error("output_dir must stay inside project_root/outputs/mavis_model_scaling")
    memory_bytes(args.gpu_max_memory)
    if args.fixed_source is not None:
        args.fixed_source = args.fixed_source.resolve()
    if args.model_size == "8b" and args.fixed_source is None:
        parser.error("The matched 8B repair requires --fixed_source containing the original 500 answers")
    if args.model_size == "32b" and args.fixed_source is not None:
        parser.error("The 32B scaling run generates its own answers; do not use --fixed_source")
    return args


def memory_bytes(text: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(GiB|MiB|GB|MB)", text)
    if not match:
        raise ValueError("gpu_max_memory must have a size suffix, for example 110GiB")
    unit = {"GiB": 1024**3, "MiB": 1024**2, "GB": 1000**3, "MB": 1000**2}[match[2]]
    value = int(float(match[1]) * unit)
    if value <= 0:
        raise ValueError("gpu_max_memory must be positive")
    return value


def normalize_allocator_environment() -> None:
    """Retain the allocator policy under its current, nondeprecated name."""
    legacy = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if not os.environ.get("PYTORCH_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = legacy or "expandable_segments:True"


def write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite {description}: {value!r}")
    return float(value)


def fixed_rows_by_question(path: Path) -> dict[str, dict]:
    rows = read_rows(path)
    rows.sort(key=lambda row: int(row["global_index"]))
    if [row["global_index"] for row in rows] != list(range(START, END + 1)):
        raise ValueError("The fixed-answer source must contain unique indices 0..499 exactly")
    mapping = {}
    for row in rows:
        question = row.get("question")
        if not isinstance(question, str) or not question or question in mapping:
            raise ValueError(f"Fixed-answer questions must be nonempty and unique: index={row['global_index']}")
        generation = row.get("generation", {})
        length = generation.get("num_generated_tokens")
        if not isinstance(length, int) or isinstance(length, bool) or not 0 < length <= MAX_NEW_TOKENS:
            raise ValueError(f"Invalid fixed answer length at index={row['global_index']}")
        if not isinstance(row.get("prediction"), str):
            raise ValueError(f"Missing original answer text at index={row['global_index']}")
        mapping[question] = row
    return mapping


def validate_rows(rows: list[dict], expected_indices: list[int]) -> None:
    rows.sort(key=lambda row: int(row["global_index"]))
    if [row["global_index"] for row in rows] != expected_indices:
        raise ValueError(f"Missing/duplicate indices; expected {expected_indices}, got {[r.get('global_index') for r in rows]}")
    for row in rows:
        index = row["global_index"]
        schema = row.get("schema", {})
        if schema.get("data_backend") != DATA_BACKEND:
            raise ValueError(f"Unexpected Parquet data backend at {index}")
        if schema.get("scoring_backend") != SCORING_BACKEND:
            raise ValueError(f"Non-isolated scoring backend at {index}")
        if schema.get("vcd_intervention") != "target_image_with_null_question":
            raise ValueError(f"VCD intervention mismatch at {index}")
        if schema.get("aggregation") != "weighted_logsumexp_without_probability_floor":
            raise ValueError(f"Log-domain aggregation mismatch at {index}")
        if row.get("empirical_distribution", {}).get("joint_support_size") != SUPPORT:
            raise ValueError(f"Incomplete K=20 empirical support at {index}")
        generation = row.get("generation", {})
        length = generation.get("num_generated_tokens")
        if not isinstance(length, int) or isinstance(length, bool) or not 0 < length <= MAX_NEW_TOKENS:
            raise ValueError(f"Invalid/empty generation at {index}: length={length}")
        if generation.get("error"):
            raise ValueError(f"Generation error at {index}: {generation['error']}")
        for field in ("generated_tokens", "generated_prefixes"):
            if not isinstance(generation.get(field), list) or len(generation[field]) != length:
                raise ValueError(f"Incomplete {field} at {index}")
        if not isinstance(generation.get("generated_token_ids"), list) or len(generation["generated_token_ids"]) != length:
            raise ValueError(f"Missing exact generated token IDs at {index}")
        drive = row.get("drive_metrics", {})
        steps = drive.get("per_step", [])
        # The established core records every other token, not every token.
        # Full trajectory here means all prescribed t=1,3,5,... positions.
        expected_steps = list(range(1, length + 1, 2))
        if [step.get("t") for step in steps] != expected_steps:
            raise ValueError(f"Missing prescribed trajectory positions at {index}")
        for metric in METRICS:
            curve = [finite_number(step.get(f"{metric}_t"), f"{metric}_t at {index}/{step.get('t')}") for step in steps]
            mean = finite_number(drive.get(f"avg_{metric.lower()}"), f"avg_{metric.lower()} at {index}")
            if not math.isclose(mean, sum(curve) / len(curve), rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"Mean/trajectory mismatch for {metric} at {index}")
        for step in steps:
            for field in ("log_P_A_le_t_null", "log_P_A_le_t_do_Q", "log_P_A_le_t_do_V", "log_P_A_t_do_A_prev"):
                finite_number(step.get(field), f"{field} at {index}/{step.get('t')}")
        for field in ("token_f1", "rouge_l_f1"):
            finite_number(row.get("text_metrics", {}).get(field), f"{field} at {index}")


def preflight(args: argparse.Namespace, check_gpus: bool) -> list[str]:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {args.output_dir}")
    for path in (args.core_script, args.summary_script, args.safe_scoring_script,
                 args.local_data_script, args.dataset_path,
                 args.model_path / "config.json", args.model_path / "tokenizer_config.json",
                 args.model_path / "tokenizer.json", args.model_path / "preprocessor_config.json"):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty local asset: {path}")
    with (args.model_path / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_type") != "qwen3_vl" or config.get("quantization_config"):
        raise ValueError("The comparison requires a full, unquantized Qwen3-VL checkpoint")
    weight_index = args.model_path / "model.safetensors.index.json"
    if weight_index.is_file():
        with weight_index.open(encoding="utf-8") as handle:
            shards = sorted(set(json.load(handle)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    if not shards:
        raise ValueError("The checkpoint index has no weight shards")
    for name in shards:
        shard = (args.model_path / name).resolve()
        if not shard.is_relative_to(args.model_path) or not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/invalid checkpoint shard: {name}")
    # Imports, processor creation and actual image decoding do not load weights or
    # launch a CUDA inference worker.
    import torch
    import datasets  # noqa: F401
    import transformers
    from PIL import Image
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        raise ImportError("Installed transformers lacks Qwen3VLForConditionalGeneration")
    data = load_module(args.local_data_script, "mavis_local_data_preflight")
    if data.DATA_BACKEND != DATA_BACKEND:
        raise ValueError("Unexpected direct-Parquet loader backend")
    dataset = data.load_mavis_parquet(args.dataset_path)
    required = {"image", "Q", "verifiable_facts", "pos_document_ids", "meta"}
    if len(dataset) < END + 1:
        raise ValueError(f"MAVIS must contain indices 0..499 and columns {sorted(required)}")
    questions = []
    for index in range(START, END + 1):
        row = dataset[index]
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"Missing required MAVIS fields at index={index}")
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise ValueError(f"MAVIS image is not decoded as PIL at index={index}")
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"Invalid image dimensions at index={index}")
        if row["Q"] is None or not str(row["Q"]).strip():
            raise ValueError(f"Empty MAVIS question at index={index}")
        questions.append(str(row["Q"]))
    processor = transformers.AutoProcessor.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=True, max_pixels=256 * 256,
    )
    if args.fixed_source is not None:
        fixed = fixed_rows_by_question(args.fixed_source)
        for index, question in enumerate(questions):
            if str(question) not in fixed or fixed[str(question)]["global_index"] != index:
                raise ValueError(f"Fixed-source question/index does not match MAVIS at {index}")
        safe = load_module(args.safe_scoring_script, "mavis_safe_scoring_preflight")
        if safe.BACKEND != SCORING_BACKEND:
            raise ValueError("Unexpected safe scoring backend")
        for row in fixed.values():
            ids, _ = safe.recover_ids(processor.tokenizer, row)
            if len(ids) != row["generation"]["num_generated_tokens"]:
                raise ValueError(f"Fixed-source token-count recovery mismatch at {row['global_index']}")
            if processor.decode(ids, skip_special_tokens=True).strip() != row["prediction"]:
                raise ValueError(f"Fixed-source decoded answer text mismatch at {row['global_index']}")
    probe = data.verify_output_writable(args.output_dir.parent, num_bytes=65536)
    devices: list[str] = []
    if check_gpus:
        count = torch.cuda.device_count()
        if count != NUM_WORKERS:
            raise RuntimeError(f"Expected exactly eight scheduler-visible GPUs; found {count}")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = [item.strip() for item in visible.split(",")] if visible else [str(i) for i in range(count)]
        if len(devices) != NUM_WORKERS or len(set(devices)) != NUM_WORKERS:
            raise RuntimeError(f"Cannot map eight distinct CUDA_VISIBLE_DEVICES entries: {visible!r}")
    print(json.dumps({
        "status": "assets_and_dependencies_ready", "dataset_rows": len(dataset),
        "data_backend": DATA_BACKEND, "images_decoded": END + 1,
        "output_write_read_probe_bytes": probe["bytes_written_and_verified"],
        "checkpoint_shards": len(shards), "worker_gpu_check": check_gpus,
        "visible_gpu_count": len(devices) if check_gpus else None,
        "model_loaded": False, "start": START, "end_inclusive": END,
        "support_size": SUPPORT, "max_new_tokens": MAX_NEW_TOKENS,
        "dtype": "float16", "attention": "sdpa", "model_cpu_offload": False,
        "model_size": args.model_size, "immutable_prompt_cache": True,
        "scoring": "full_prompt_replay_without_reused_kv_state",
        "fixed_generated_answers": args.fixed_source is not None,
    }), flush=True)
    return devices


def run_worker(args: argparse.Namespace) -> None:
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one GPU")
    # Keep the core's import-time output creation inside this new experiment.
    os.chdir(args.output_dir)
    core = load_module(args.core_script, "mavis_unchanged_core")
    # The old core sets the deprecated allocator alias at import time. Remove
    # that alias again before any model tensor allocation; do not mute errors.
    normalize_allocator_environment()
    safe = load_module(args.safe_scoring_script, "mavis_safe_scoring")
    data = load_module(args.local_data_script, "mavis_local_data")
    if data.DATA_BACKEND != DATA_BACKEND:
        raise ValueError("Unexpected direct-Parquet loader backend")
    if safe.BACKEND != SCORING_BACKEND:
        raise ValueError("Unexpected safe scoring backend")
    core.MODEL_PATH = str(args.model_path)
    core.DATASET_PATH = str(args.dataset_path)
    core.OUTPUT_DIR = str(args.output_dir)
    core.BATCH_SIZE = SUPPORT
    core.MAX_NEW_TOKENS = MAX_NEW_TOKENS
    core.MODEL_DTYPE = torch.float16
    core.ATTN_IMPL = "sdpa"
    os.environ["SKIP_BERTSCORE"] = "1"

    def load_model_full_gpu(model_path: str):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, attn_implementation="sdpa",
            device_map={"": 0}, max_memory={0: args.gpu_max_memory},
            low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True,
        )
        model.eval()
        if any(value in ("cpu", "disk") for value in getattr(model, "hf_device_map", {}).values()):
            raise RuntimeError("Model CPU/disk offload is prohibited")
        if any(parameter.device.type != "cuda" or parameter.device.index != 0 for parameter in model.parameters()):
            raise RuntimeError("All complete model parameters must reside on this worker's GPU")
        count = sum(parameter.numel() for parameter in model.parameters())
        low, high = {"8b": (6_000_000_000, 12_000_000_000),
                     "32b": (30_000_000_000, 40_000_000_000)}[args.model_size]
        if not low <= count <= high:
            raise ValueError(f"Expected a full {args.model_size}-scale checkpoint; actual parameters={count}")
        allocated = torch.cuda.memory_allocated(0)
        if allocated > memory_bytes(args.gpu_max_memory):
            raise RuntimeError(f"Loaded model exceeds configured GPU budget: {allocated} bytes")
        write_json_new(args.output_dir / f"worker_{args.worker_id}_configuration.json", {
            "worker_id": args.worker_id, "model_path": str(args.model_path),
            "parameters": count, "dtype": "float16", "attention": "sdpa",
            "gpu_max_memory": args.gpu_max_memory, "model_cpu_offload": False,
            "model_size": args.model_size, "immutable_prompt_cache": True,
            "cache_isolation": "full prompt replay; no model KV cache reused between scoring calls",
            "safe_scoring_sha256": hashlib.sha256(args.safe_scoring_script.read_bytes()).hexdigest(),
            "data_backend": DATA_BACKEND,
            "local_data_sha256": hashlib.sha256(args.local_data_script.read_bytes()).hexdigest(),
            "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
            "fixed_generated_answers": args.fixed_source is not None,
            "fixed_source": str(args.fixed_source) if args.fixed_source is not None else None,
            "fixed_source_sha256": hashlib.sha256(args.fixed_source.read_bytes()).hexdigest() if args.fixed_source is not None else None,
            "prompt_inputs_residency": "cpu", "prompt_kv_cache_reuse": False,
            "core_sha256": hashlib.sha256(args.core_script.read_bytes()).hexdigest(),
            "start": START, "end_inclusive": END, "support_size": SUPPORT,
            "max_new_tokens": MAX_NEW_TOKENS, "temporal_stride": 2,
        })
        return model

    core.load_model = load_model_full_gpu
    safe.install(core)
    # The unchanged core can skip failed prompt caches internally. Record every
    # such failure and reject that batch instead of accepting a smaller support.
    original_build_cache = core.build_prompt_cache
    cache_failures = []

    def checked_build_cache(*positional, **keywords):
        try:
            return original_build_cache(*positional, **keywords)
        except Exception as error:
            cache_failures.append(f"{type(error).__name__}: {error}")
            raise

    core.build_prompt_cache = checked_build_cache
    dataset = data.load_mavis_parquet(args.dataset_path)
    if len(dataset) < END + 1:
        raise ValueError("The dataset no longer contains all requested 500 indices")
    model = core.load_model(str(args.model_path))
    processor = core.get_processor(core.BASE_MAX_PIXELS)
    fixed = fixed_rows_by_question(args.fixed_source) if args.fixed_source is not None else None
    source_hash = hashlib.sha256(args.fixed_source.read_bytes()).hexdigest() if args.fixed_source is not None else None
    generation_records = []
    original_generate = core.generate_sequence

    def tracked_generate(model, processor, image, question, max_new_tokens):
        if fixed is None:
            if hasattr(getattr(model, "model", None), "rope_deltas"):
                model.model.rope_deltas = None
            result = original_generate(model=model, processor=processor, image=image,
                                       question=question, max_new_tokens=max_new_tokens)
            ids = result["generated_ids"].reshape(-1).tolist()
            method = "model_generation"
            source_index = None
        else:
            started = time.perf_counter()
            source = fixed[str(question)]
            ids, method = safe.recover_ids(processor.tokenizer, source)
            if not 0 < len(ids) <= max_new_tokens or len(ids) != source["generation"]["num_generated_tokens"]:
                raise ValueError(f"Fixed-answer token count mismatch at {source['global_index']}")
            if processor.decode(ids, skip_special_tokens=True).strip() != source["prediction"]:
                raise ValueError(f"Fixed-answer text changed at {source['global_index']}")
            target_ids = torch.tensor(ids, dtype=torch.long, device="cpu")
            true_prompt = core.build_prompt_cache(model=model, processor=processor, image=image, question=question)
            actual = core.score_fixed_sequence_from_prompt_cache(model=model, prompt_cache=true_prompt,
                                                                 target_generated_ids=target_ids)
            if len(actual["token_probs"]) != len(ids):
                raise ValueError("Incomplete teacher-forced original-answer probabilities")
            result = {
                "generated_ids": target_ids, "generated_text": source["prediction"],
                "step_tokens": list(source["generation"]["generated_tokens"]),
                "prefixes": list(source["generation"]["generated_prefixes"]),
                "step_token_probs": actual["token_probs"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            source_index = source["global_index"]
        generation_records.append({
            "question": str(question), "generated_token_ids": ids,
            "token_recovery_method": method, "source_global_index": source_index,
        })
        return result

    core.generate_sequence = tracked_generate
    starts = list(range(START, END + 1, SUPPORT))[args.worker_id::NUM_WORKERS]
    summaries = []
    for start in starts:
        end = start + SUPPORT - 1
        for name in (f"mavis_drive_scores_{start}-{end}.jsonl", f"mavis_drive_scores_summary_{start}-{end}.json"):
            if (args.output_dir / name).exists():
                raise FileExistsError(f"Refusing to overwrite worker shard: {name}")
        cache_failures.clear()
        generation_records.clear()
        subset = [dataset[index] for index in range(start, end + 1)]
        try:
            # Do not use core.main(): it catches batch exceptions and may
            # otherwise exit successfully after an incomplete run.
            summary = core.process_one_batch(model, processor, subset, start, end)
            if cache_failures or summary.get("skipped_prompt_pairs") or summary.get("skipped_null_image_prompt"):
                raise RuntimeError(f"Incomplete prompt support in batch {start}-{end}: {cache_failures}")
            if summary.get("num_samples") != SUPPORT or summary.get("drive_metrics", {}).get("num_valid") != SUPPORT:
                raise RuntimeError(f"Incomplete valid sample count in batch {start}-{end}")
            shard = args.output_dir / f"mavis_drive_scores_{start}-{end}.jsonl"
            rows = read_rows(shard)
            if len(rows) != SUPPORT or len(generation_records) != SUPPORT:
                raise ValueError(f"Incomplete generation records in batch {start}-{end}")
            rows.sort(key=lambda row: row["global_index"])
            for row, record in zip(rows, generation_records):
                if row["question"] != record["question"]:
                    raise ValueError("Generation record/question order mismatch")
                if fixed is not None and record["source_global_index"] != row["global_index"]:
                    raise ValueError("Frozen-answer source/index order mismatch")
                row["schema"]["scoring_backend"] = safe.BACKEND
                row["schema"]["data_backend"] = DATA_BACKEND
                row["schema"]["immutable_prompt_cache"] = True
                row["generation"]["generated_token_ids"] = record["generated_token_ids"]
                row["generation"]["token_recovery_method"] = record["token_recovery_method"]
                row["generation"]["answers_frozen"] = fixed is not None
                row["generation"]["fixed_source_sha256"] = source_hash
            validate_rows(rows, list(range(start, end + 1)))
            with shard.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            summary["scoring_backend"] = safe.BACKEND
            summary["data_backend"] = DATA_BACKEND
            summary["local_data_sha256"] = hashlib.sha256(args.local_data_script.read_bytes()).hexdigest()
            summary["fixed_generated_answers"] = fixed is not None
            summary["fixed_source_sha256"] = source_hash
            with (args.output_dir / f"mavis_drive_scores_summary_{start}-{end}.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            summaries.append(summary)
            print(f"[validated] worker={args.worker_id} indices={start}..{end} n={SUPPORT}", flush=True)
        finally:
            del subset
            gc.collect()
            torch.cuda.empty_cache()
    write_json_new(args.output_dir / f"worker_{args.worker_id}_completion.json", {
        "worker_id": args.worker_id, "batch_starts": starts,
        "num_samples": len(starts) * SUPPORT, "all_batches_validated": True,
    })


def merge_and_summarize(args: argparse.Namespace) -> None:
    pattern = re.compile(r"mavis_drive_scores_(\d+)-(\d+)\.jsonl")
    files = sorted((int(match[1]), int(match[2]), path)
                   for path in args.output_dir.glob("mavis_drive_scores_*.jsonl")
                   if (match := pattern.fullmatch(path.name)))
    expected = [(start, start + SUPPORT - 1) for start in range(START, END + 1, SUPPORT)]
    if [(start, end) for start, end, _ in files] != expected:
        raise ValueError("Missing/extra empirical-support batch shards")
    for worker_id in range(NUM_WORKERS):
        with (args.output_dir / f"worker_{worker_id}_completion.json").open(encoding="utf-8") as handle:
            completion = json.load(handle)
        starts = list(range(START, END + 1, SUPPORT))[worker_id::NUM_WORKERS]
        if completion.get("batch_starts") != starts or completion.get("all_batches_validated") is not True:
            raise ValueError(f"Worker {worker_id} did not complete its exact assigned batches")
    rows = [row for _, _, path in files for row in read_rows(path)]
    validate_rows(rows, list(range(START, END + 1)))
    merged = args.output_dir / "mavis_drive_scores_k20_0-499_merged.jsonl"
    with merged.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    subprocess.run([sys.executable, str(args.summary_script), "--input", str(merged),
                    "--output_dir", str(args.output_dir / "analysis"), "--bins", "10",
                    "--bootstrap", "5000", "--seed", "42"], check=True)
    with (args.output_dir / "analysis" / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("num_valid_samples") != 500:
        raise ValueError("Summary silently excluded samples; expected exactly 500")
    print(f"Validated and summarized all 500 complete trajectories: {args.output_dir}", flush=True)


def launch(args: argparse.Namespace) -> None:
    devices = preflight(args, check_gpus=True)
    # Atomic final-directory creation rejects previous results and concurrent
    # launch attempts. A failed run is retained for diagnosis, never reused.
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(exist_ok=False)
    (args.output_dir / "logs").mkdir()
    children = []
    threads = []
    log_errors = []

    def stream(process: subprocess.Popen, worker_id: int) -> None:
        try:
            with (args.output_dir / "logs" / f"worker_{worker_id}.log").open("x", encoding="utf-8") as handle:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(f"[worker {worker_id}] {line}", end="", flush=True)
        except Exception as error:
            log_errors.append(f"worker={worker_id}: {type(error).__name__}: {error}")

    def terminate_children() -> None:
        for process in children:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in children:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for worker_id, device in enumerate(devices):
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES": device, "SKIP_BERTSCORE": "1",
                                "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--project_root", str(args.project_root), "--model_path", str(args.model_path),
                       "--dataset_path", str(args.dataset_path), "--output_dir", str(args.output_dir),
                       "--core_script", str(args.core_script), "--summary_script", str(args.summary_script),
                       "--safe_scoring_script", str(args.safe_scoring_script),
                       "--local_data_script", str(args.local_data_script),
                       "--gpu_max_memory", args.gpu_max_memory, "--model_size", args.model_size,
                       "--worker_id", str(worker_id)]
            if args.fixed_source is not None:
                command.extend(["--fixed_source", str(args.fixed_source)])
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="strict", bufsize=1,
                                       env=environment, start_new_session=True)
            children.append(process)
            thread = threading.Thread(target=stream, args=(process, worker_id), daemon=True)
            thread.start()
            threads.append(thread)
        pending = set(range(NUM_WORKERS))
        while pending:
            if log_errors:
                raise RuntimeError(f"Log capture failed: {log_errors}")
            for worker_id in list(pending):
                code = children[worker_id].poll()
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f"Worker {worker_id} failed with exit={code}; run is incomplete (see worker log)")
                pending.remove(worker_id)
            if pending:
                time.sleep(0.25)
        for thread in threads:
            thread.join()
        if log_errors:
            raise RuntimeError(f"Log capture failed: {log_errors}")
        merge_and_summarize(args)
    finally:
        terminate_children()
        for thread in threads:
            thread.join(timeout=10)


def main() -> None:
    normalize_allocator_environment()
    args = parse_args()
    if args.preflight:
        preflight(args, check_gpus=False)
    elif args.launch:
        launch(args)
    elif args.worker:
        run_worker(args)
    else:
        merge_and_summarize(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as error:
        print(f"Interrupted: {error}", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
