#!/usr/bin/env python3
"""Fixed-answer visual interventions; no changes to QCD/VCD/PCD definitions.

Pre-register two blur radii and two same-domain mismatch seeds. Preserve every
token log probability. Conditional visual likelihood lift is a separate
baseline, not a renamed VCD. Results do not establish error-detection validity
without independent reference/human labels.

This job is independent of the full-drive repair job: its source supplies only
the frozen answers and identifiers. Every inferential likelihood is recomputed
locally with the state-independent scoring backend. Source scores are diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from mavis_safe_scoring import BACKEND, install as install_safe, recover_ids
from mavis_local_data import DATA_BACKEND, load_mavis_parquet, verify_output_writable

CONDITIONS = ("target", "blur_8", "blur_27", "mismatch_42", "mismatch_137",
              "blank_black", "blank_gray", "blank_white")
LN2 = math.log(2.0)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: int(row["global_index"]))
    if [int(r["global_index"]) for r in rows] != list(range(500)):
        raise ValueError("Require the complete, unique source cohort 0..499")
    for row in rows:
        if row.get("schema", {}).get("vcd_intervention") != "target_image_with_null_question":
            raise ValueError("Source does not use the aligned VCD definition")
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assets(args):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    del Qwen3VLForConditionalGeneration
    root = Path(args.model_path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl" or config.get("quantization_config"):
        raise ValueError("This protocol requires the Qwen3-VL source checkpoint")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for name in set(index["weight_map"].values()):
        shard = (root / name).resolve()
        if not shard.is_relative_to(root) or not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty checkpoint shard: {name}")
    if not args.output_dir:
        raise ValueError("Preflight requires --output_dir for an actual output write test")
    output_dir = Path(args.output_dir)
    if not output_dir.is_relative_to(Path(args.project_root) / "outputs" / "mavis_visual_evidence_probe"):
        raise ValueError("Output must stay inside project_root/outputs/mavis_visual_evidence_probe")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output_dir}")
    rows = read_rows(args.input)
    dataset = load_mavis_parquet(args.dataset_path)
    if len(dataset) < 500:
        raise ValueError("MAVIS must contain the complete indices 0..499")
    processor = AutoProcessor.from_pretrained(root, local_files_only=True, max_pixels=256 * 256)
    tokenizer = processor.tokenizer
    methods = {}
    for row in rows:
        idx = int(row["global_index"])
        record = dataset[idx]  # Decode and load the real PIL image, not only its metadata.
        if str(record["Q"]) != row["question"]:
            raise ValueError(f"Dataset/source question mismatch: {idx}")
        record["image"].load()
        _, method = recover_ids(tokenizer, row)
        methods[method] = methods.get(method, 0) + 1
        # Domain controls must have two different candidate visual contents.
        domain = row.get("meta", {}).get("domain", "unknown")
        candidates = [r for r in rows if r["global_index"] != idx
                      and r.get("meta", {}).get("domain", "unknown") == domain]
        if len(candidates) < 2:
            raise ValueError(f"Insufficient same-domain controls: {idx}")
    write_check = verify_output_writable(output_dir.parent)
    script_dir = Path(__file__).resolve().parent
    print(json.dumps({"cpu_preflight": "passed", "samples": len(rows),
                      "decoded_images": len(rows), "dataset_rows": len(dataset),
                      "data_backend": DATA_BACKEND, "output_write_check": write_check,
                      "token_recovery": methods, "source_sha256": sha256(args.input),
                      "core_sha256": sha256(Path(__file__).resolve().parent / "compute_mavis_drive_scores_logdomain.py"),
                      "safe_scoring_sha256": sha256(script_dir / "mavis_safe_scoring.py"),
                      "local_data_sha256": sha256(script_dir / "mavis_local_data.py"),
                      "requires_full_drive_repair_output": False,
                      "source_scores_used_for_inference": False,
                      "scoring_backend": BACKEND}, indent=2))


def load_core(project_root):
    path = Path(__file__).resolve().parent / "compute_mavis_drive_scores_logdomain.py"
    spec = importlib.util.spec_from_file_location("mavis_probe_core", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def worker(args):
    import torch
    from PIL import Image, ImageFilter
    from transformers import Qwen3VLForConditionalGeneration
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("Invalid worker assignment")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU to each worker")
    # The imported core creates ./output at import time. Keep that incidental
    # directory inside this job, so concurrent jobs do not share writable state.
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(output.parent)
    core = load_core(args.project_root)
    # The unmodified legacy core sets the deprecated alias during import.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    core.MODEL_PATH = args.model_path
    install_safe(core)
    core.run_numerical_self_test()
    rows = read_rows(args.input)
    dataset = load_mavis_parquet(args.dataset_path)
    source_hash = sha256(args.input)
    script_dir = Path(__file__).resolve().parent
    code_hashes = {"core_sha256": sha256(Path(core.__file__)),
                   "safe_scoring_sha256": sha256(script_dir / "mavis_safe_scoring.py"),
                   "local_data_sha256": sha256(script_dir / "mavis_local_data.py")}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, attn_implementation="sdpa",
        device_map={"": 0}, local_files_only=True, low_cpu_mem_usage=True,
    ).eval()
    processor = core.get_processor(core.BASE_MAX_PIXELS)
    def score(image, question, ids):
        cache = core.build_prompt_cache(model, processor, image, question)
        result = core.score_fixed_sequence_from_prompt_cache(model, cache, ids)
        values = result["token_log_probs"]
        if len(values) != len(ids) or not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Incomplete or nonfinite token likelihood")
        return [float(x) for x in values]
    null_cache = core.build_prompt_cache(model, processor, core.build_null_image(), "")
    output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows[args.worker_id::args.num_workers]:
        idx = int(row["global_index"])
        token_ids, recovery = recover_ids(processor.tokenizer, row)
        ids = torch.tensor(token_ids, dtype=torch.long)
        record = dataset[idx]
        if str(record["Q"]) != row["question"]:
            raise ValueError(f"Dataset/source question mismatch: {idx}")
        target, _ = core.normalize_image(record["image"])
        del record
        domain = row.get("meta", {}).get("domain", "unknown")
        target_pixels_hash = hashlib.sha256(target.tobytes()).hexdigest()
        candidates = [int(r["global_index"]) for r in rows if int(r["global_index"]) != idx
                      and r.get("meta", {}).get("domain", "unknown") == domain]
        donors, used, donor_images = {}, set(), {}
        used_hashes = {target_pixels_hash}
        for seed in (42, 137):
            shuffled = list(candidates)
            random.Random(seed + idx * 1000003).shuffle(shuffled)
            for donor in shuffled:
                image, _ = core.normalize_image(dataset[donor]["image"])
                image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                if donor not in used and image_hash not in used_hashes:
                    donors[f"mismatch_{seed}"] = donor
                    donor_images[f"mismatch_{seed}"] = image
                    used.add(donor)
                    used_hashes.add(image_hash)
                    break
            else:
                raise ValueError(f"Cannot find distinct same-domain image control for {idx}")
        images = {
            "target": target,
            "blur_8": target.filter(ImageFilter.GaussianBlur(radius=8)),
            "blur_27": target.filter(ImageFilter.GaussianBlur(radius=27)),
            **donor_images,
            "blank_black": Image.new("RGB", core.FORCE_IMAGE_SIZE, (0, 0, 0)),
            "blank_gray": Image.new("RGB", core.FORCE_IMAGE_SIZE, (127, 127, 127)),
            "blank_white": Image.new("RGB", core.FORCE_IMAGE_SIZE, (255, 255, 255)),
        }
        null = core.score_fixed_sequence_from_prompt_cache(model, null_cache, ids)["token_log_probs"]
        if len(null) != len(token_ids) or not all(math.isfinite(x) for x in null):
            raise ValueError("Invalid null reference")
        scores = {}
        for name in CONDITIONS:
            scores[name] = {
                "visual_only_token_logprobs": list(null) if name == "blank_black" else score(images[name], "", ids),
                "question_conditioned_token_logprobs": score(images[name], row["question"], ids),
            }
        recovered_vcd = (np.cumsum(scores["target"]["visual_only_token_logprobs"]) - np.cumsum(null)) / LN2
        # Legacy scores never enter inferential calculations. Their difference
        # is retained to audit the old backend, not used as a pass criterion.
        # If a safe source is explicitly supplied, agreement remains a check.
        source_steps = row["drive_metrics"]["per_step"]
        gap = max(abs(float(recovered_vcd[int(step["t"]) - 1]) - float(step["VCD_t"]))
                  for step in source_steps)
        source_backend = row.get("schema", {}).get("scoring_backend", "legacy_shared_kv")
        if source_backend == BACKEND and gap > args.max_alignment_gap_bits:
            raise ValueError(f"Original VCD reconstruction mismatch at {idx}: {gap:.6g} bits")
        result = {
            "global_index": idx, "domain": domain, "question": row["question"],
            "prediction": row["prediction"], "reference_answer": row["reference_answer"],
            "generated_token_ids": token_ids,
            "generated_tokens": row["generation"]["generated_tokens"],
            "generated_prefixes": row["generation"]["generated_prefixes"],
            "semantic_token_mask": [i not in processor.tokenizer.all_special_ids for i in token_ids],
            "safe_target_vcd_per_token": recovered_vcd.tolist(),
            "null_visual_only_token_logprobs": list(null), "conditions": scores,
            "mismatch_donor_indices": donors,
            "source_score_diagnostics": {
                "source_scoring_backend": source_backend,
                "source_vcd_max_absolute_difference_bits": gap,
                "source_scores_used_for_inference": False,
            },
            "protocol": {
                "source_sha256": source_hash, "model_path": args.model_path,
                "token_recovery": recovery, "fixed_answer": True,
                "blur_radii_on_448px_image": [8, 27], "mismatch_seeds": [42, 137],
                "image_size": list(core.FORCE_IMAGE_SIZE), "max_pixels": core.BASE_MAX_PIXELS,
                "dtype": "float16", "attention": "sdpa",
                "null_reference": "black_image_with_original_empty_question_prompt",
                "conditional_visual_lift_reference": "gray_image_with_same_question",
                "probability_aggregation": "log_domain_without_probability_floor",
                "scoring_backend": BACKEND,
                "data_backend": DATA_BACKEND, **code_hashes,
                "requires_full_drive_repair_output": False,
                "source_scores_used_for_inference": False,
            },
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        print(f"worker={args.worker_id} index={idx} tokens={len(token_ids)} source_vcd_gap_diagnostic={gap:.6g}", flush=True)
        del ids, scores, result, images, target
    print(f"Completed shard: {output}", flush=True)


def analyze(args):
    root = Path(args.output_dir)
    rows = []
    for worker_id in range(8):
        with (root / f"shard_{worker_id}.jsonl").open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: row["global_index"])
    if [r["global_index"] for r in rows] != list(range(500)):
        raise ValueError("Expected exactly 500 complete unique probe rows")
    hashes = {r["protocol"]["source_sha256"] for r in rows}
    if hashes != {sha256(args.input)}:
        raise ValueError("Probe/source hash mismatch")
    samples = []
    for row in rows:
        if row["protocol"].get("scoring_backend") != BACKEND or row["protocol"].get("data_backend") != DATA_BACKEND:
            raise ValueError("Mixed or unexpected scoring/data backend")
        mask = np.asarray(row["semantic_token_mask"], dtype=bool)
        n = len(mask)
        if not mask.any() or set(row["conditions"]) != set(CONDITIONS):
            raise ValueError("Empty semantic answer or missing condition")
        null = np.asarray(row["null_visual_only_token_logprobs"])
        gray_q = np.asarray(row["conditions"]["blank_gray"]["question_conditioned_token_logprobs"])
        sample = {"global_index": row["global_index"], "domain": row["domain"],
                  "num_semantic_tokens": int(mask.sum())}
        for name in CONDITIONS:
            visual = np.asarray(row["conditions"][name]["visual_only_token_logprobs"])
            full = np.asarray(row["conditions"][name]["question_conditioned_token_logprobs"])
            if any(len(x) != n or not np.isfinite(x).all() for x in (null, gray_q, visual, full)):
                raise ValueError("Incomplete/nonfinite likelihood arrays")
            sample[f"{name}_mean_vcd"] = float(((np.cumsum(visual) - np.cumsum(null)) / LN2)[mask].mean())
            sample[f"{name}_mean_local_visual_lift"] = float(((full - gray_q) / LN2)[mask].mean())
            sample[f"{name}_mean_surprisal"] = float((-full / LN2)[mask].mean())
        for name in CONDITIONS[1:]:
            for metric in ("mean_vcd", "mean_local_visual_lift", "mean_surprisal"):
                sample[f"target_minus_{name}_{metric}"] = sample[f"target_{metric}"] - sample[f"{name}_{metric}"]
        samples.append(sample)
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    with (analysis_dir / "per_sample_probe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader(); writer.writerows(samples)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(samples), size=(5000, len(samples)))
    summary = {"samples": len(samples), "bootstrap_unit": "answer",
               "scoring_backend": BACKEND,
               "data_backend": DATA_BACKEND,
               "requires_full_drive_repair_output": False,
               "source_scores_used_for_inference": False,
               "bootstrap_draws": 5000, "bootstrap_seed": 42,
               "aggregation": "all nonspecial answer tokens; each answer equally weighted",
               "interpretation": "fixed-answer input sensitivity, not natural-error detection",
               "source_sha256": next(iter(hashes)), "paired_effects": {}}
    for key in samples[0]:
        if not key.startswith("target_minus_"):
            continue
        values = np.asarray([r[key] for r in samples], dtype=float)
        ci = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
        summary["paired_effects"][key] = {
            "mean": float(values.mean()), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "fraction_positive": float((values > 0).mean()),
        }
    with (root / "probe_merged.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (analysis_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def launch(args):
    validate_assets(args)
    import torch
    if torch.cuda.device_count() != 8:
        raise RuntimeError("This foreground launcher requires exactly eight visible GPUs")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(devices) != 8 or len(set(devices)) != 8 or any(not item.strip() for item in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must name eight distinct GPUs")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=False)
    (output_dir / "logs").mkdir()
    children, threads, log_errors = [], [], []

    def stream(process, worker_id):
        try:
            with (output_dir / "logs" / f"worker_{worker_id}.log").open("x", encoding="utf-8") as handle:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(f"[worker {worker_id}] {line}", end="", flush=True)
        except Exception as error:
            log_errors.append(f"worker={worker_id}: {type(error).__name__}: {error}")

    def terminate_children():
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
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for worker_id, device in enumerate(devices):
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES": device.strip(), "PYTHONUNBUFFERED": "1",
                                "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1",
                                "TRANSFORMERS_OFFLINE": "1", "SKIP_BERTSCORE": "1"})
            command = [sys.executable, str(Path(__file__).resolve()), "worker",
                       "--project_root", args.project_root, "--input", args.input,
                       "--dataset_path", args.dataset_path, "--model_path", args.model_path,
                       "--worker_id", str(worker_id), "--num_workers", "8",
                       "--max_alignment_gap_bits", str(args.max_alignment_gap_bits),
                       "--output", str(output_dir / f"shard_{worker_id}.jsonl")]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="strict", bufsize=1,
                                       env=environment, start_new_session=True)
            children.append(process)
            thread = threading.Thread(target=stream, args=(process, worker_id), daemon=True)
            thread.start()
            threads.append(thread)
        pending = set(range(8))
        while pending:
            if log_errors:
                raise RuntimeError(f"Log capture failed: {log_errors}")
            for worker_id in list(pending):
                code = children[worker_id].poll()
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f"Worker {worker_id} failed with exit={code}; retained logs and partial trajectories in {output_dir}")
                pending.remove(worker_id)
            if pending:
                time.sleep(0.25)
        for thread in threads:
            thread.join()
        if log_errors:
            raise RuntimeError(f"Log capture failed: {log_errors}")
        analyze(args)
        print(f"Complete: {output_dir}", flush=True)
    finally:
        terminate_children()
        for thread in threads:
            thread.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "launch", "worker", "analyze"))
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_alignment_gap_bits", type=float, default=0.1)
    args = parser.parse_args()
    for name in ("project_root", "input", "dataset_path", "model_path", "output", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, str(Path(value).resolve()))
    legacy = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if not os.environ.get("PYTORCH_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = legacy or "expandable_segments:True"
    {"preflight": validate_assets, "launch": launch, "worker": worker, "analyze": analyze}[args.mode](args)


if __name__ == "__main__":
    main()
