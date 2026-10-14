#!/usr/bin/env python3
"""Eight-worker long-explanation source probes and second-family MAVIS replication."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
import traceback

from long_source_common import HFAdapter, build_manifest, digest, prefix_conditions, question_conditions, resolve_biased_row
from mavis_local_data import load_mavis_parquet, verify_output_writable
from source_response_runtime import DirectParquetRows, model_loading_lock, resource_snapshot, write_state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight", "launch", "worker", "merge"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--task", choices=("long", "mavis"), required=True)
    for name in ("project_root", "model_path", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--existing_manifest", type=Path)
    parser.add_argument("--family", choices=("qwen3_vl", "internvl"), required=True)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--support_size", type=int, default=20)
    args = parser.parse_args(argv)
    if args.cap != 256 or args.support_size != 20 or not 0 <= args.worker_id < 8:
        parser.error("This registered protocol uses cap=256, K=20 and worker_id=0..7")
    for name in ("project_root", "model_path", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.existing_manifest:
        args.existing_manifest = args.existing_manifest.resolve()
    if not args.output_dir.is_relative_to(args.project_root / "outputs" / "long_answer_source_validation"):
        parser.error("Output must stay inside outputs/long_answer_source_validation")
    args.dataset_path = args.project_root / ("vlms-are-biased/data/main-00000-of-00001.parquet" if args.task == "long"
                                              else "MAVIS/data/test-00000-of-00001.parquet")
    if args.task == "long" and not args.existing_manifest:
        parser.error("Long task needs the original matched 210-sample manifest")
    return args


def checkpoint(args):
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != args.family or config.get("quantization_config"):
        raise ValueError("Checkpoint must be complete and match the selected model family")
    index = json.loads((args.model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    files = sorted(set(index["weight_map"].values()))
    for name in files:
        path = (args.model_path / name).resolve()
        if not path.is_relative_to(args.model_path) or not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(name)
    if args.family == "internvl":
        manifest = json.loads((args.model_path / "download_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete") or manifest.get("repo") != "OpenGVLab/InternVL3-8B-hf":
            raise ValueError("InternVL download and LFS checksum validation must finish first")
        sizes = {r["name"]: r["size"] for r in manifest["files"]}
        if any((args.model_path / name).stat().st_size != sizes[name] for name in files):
            raise ValueError("Downloaded weight file size changed")
    return config, files


def mavis_manifest(dataset):
    from PIL import Image
    import hashlib
    records = []
    # First 200 share IDs with the already completed MAVIS cohort. Original block support retained.
    for index in range(200):
        record = dataset[index]
        metadata = record.get("meta", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = ast.literal_eval(metadata)
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid MAVIS metadata at {index}")
        domain = metadata.get("domain", "unknown")
        image = record["image"].convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
        records.append({"index": index, "question": str(record["Q"]), "domain": domain,
                        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "context_ids": list(range((index // 20) * 20, (index // 20 + 1) * 20))})
    for record in records:
        pool = [r["index"] for r in records if r["index"] != record["index"] and r["domain"] == record["domain"]
                and r["image_sha256"] != record["image_sha256"]]
        if len(pool) < 2:
            raise ValueError("Need two same-domain mismatched images")
        used = []
        for seed in (42, 137):
            candidates = list(pool)
            random.Random(seed + record["index"] * 1_000_003).shuffle(candidates)
            chosen = next(i for i in candidates if i not in used
                          and records[i]["image_sha256"] not in {records[j]["image_sha256"] for j in used})
            used.append(chosen)
            record[f"mismatch_{seed}"] = chosen
    return {"schema": "second_family_mavis_v1", "n": 200, "support_size": 20,
            "generation_cap": 256, "physical_image_size": [448, 448], "null_image": "black",
            "prompt": "Original MAVIS user instructions, with model-native chat template and visual processor.",
            "blur_radii": [8, 27], "mismatch_seeds": [42, 137],
            "selection": "Indices 0..199 chosen before second-family outcomes; ten original K=20 blocks.",
            "records": records}


def prepare(args, adapter):
    if args.task == "long":
        dataset = DirectParquetRows(args.dataset_path)
        manifest = build_manifest(args.existing_manifest, dataset, args.dataset_path, args.support_size)
        # Manifest construction has already decoded every target and selected support image.
        examples = {r["index"]: resolve_biased_row(dataset, r["index"], args.dataset_path)
                    for r in manifest["records"][:3]}
        for record in manifest["records"]:
            prefix_conditions(adapter.tokenizer, record["cue_candidate"], record["visual_candidate"])
        for actual in examples.values():
            adapter.prompt(actual["image"], actual["question"])
    else:
        dataset = load_mavis_parquet(args.dataset_path)
        manifest = mavis_manifest(dataset)
        for record in manifest["records"]:
            dataset[record["index"]]["image"].load()
        for record in manifest["records"][:3]:
            adapter.prompt(dataset[record["index"]]["image"], record["question"])
    from PIL import Image
    adapter.prompt(Image.new("RGB", (448, 448)), "")
    adapter.reset()
    return manifest


def preflight(args, check_gpus=False):
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import torch
    from transformers import AutoModelForImageTextToText
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    config, files = checkpoint(args)
    # Resolving the model class does not instantiate weights or allocate GPU tensors.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    model_config = CONFIG_MAPPING[args.family].from_dict(config)
    cls = AutoModelForImageTextToText._model_mapping[type(model_config)]
    adapter = HFAdapter(args.model_path, args.family, style=args.task)
    manifest = prepare(args, adapter)
    probe = verify_output_writable(args.output_dir.parent)
    devices = []
    if check_gpus:
        if torch.cuda.device_count() != 8:
            raise RuntimeError("Exactly eight queue-allocated GPUs are required")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = [x.strip() for x in visible.split(",")] if visible else [str(i) for i in range(8)]
        if len(devices) != 8 or len(set(devices)) != 8:
            raise ValueError("Eight distinct worker device mappings are required")
    print(json.dumps({"cpu_preflight": "passed", "model_loaded": False, "task": args.task,
                      "family": args.family, "native_model_class": cls.__name__, "n": manifest["n"],
                      "support_size": args.support_size, "cap": args.cap, "checkpoint_shards": len(files),
                      "support_image_integrity_exclusions": manifest.get("support_image_integrity_exclusions", {}),
                      "dtype": "bfloat16", "attention": "sdpa", "output_write_probe": probe,
                      "resources": resource_snapshot()}, ensure_ascii=False), flush=True)
    return devices, manifest


def model_load(args):
    import torch
    from transformers import AutoModelForImageTextToText
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One GPU per worker; CPU fallback is prohibited")
    write_state(args.output_dir / f"worker_{args.worker_id}_state.json", "waiting_for_serial_model_load", family=args.family)
    with model_loading_lock(args.output_dir / "model_load.lock"):
        write_state(args.output_dir / f"worker_{args.worker_id}_state.json", "loading_weights", family=args.family)
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model_path), dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": 0}, local_files_only=True, low_cpu_mem_usage=True,
            trust_remote_code=False).eval()
        if any(p.device.type != "cuda" or p.device.index != 0 for p in model.parameters()):
            raise RuntimeError("Full model must stay on the allocated GPU")
        parameters = sum(p.numel() for p in model.parameters())
        allowed = (6_000_000_000, 12_000_000_000) if args.family == "internvl" else (6_000_000_000, 40_000_000_000)
        if not allowed[0] <= parameters <= allowed[1]:
            raise ValueError(f"Full-model parameter count mismatch: {parameters}")
        if torch.cuda.memory_allocated(0) > 120 * 1024 ** 3:
            raise RuntimeError("Loaded weights exceed the 120 GiB budget")
    return model, parameters


def long_sample(args, adapter, dataset, rec):
    actual = resolve_biased_row(dataset, rec["index"], args.dataset_path)
    contexts = [resolve_biased_row(dataset, i, args.dataset_path) for i in rec["context_ids"]]
    questions = question_conditions(actual["question"], rec["cue_candidate"])
    prefixes = prefix_conditions(adapter.tokenizer, rec["cue_candidate"], rec["visual_candidate"])
    q_generated = {name: adapter.generate(actual["image"], question, cap=args.cap) for name, question in questions.items()}
    fixed_q = q_generated["neutral"]["ids"]
    q_scored = {name: adapter.source_scores(actual["image"], question, fixed_q, contexts) for name, question in questions.items()}
    p_generated = {name: adapter.generate(actual["image"], actual["question"], prefix["ids"], args.cap)
                   for name, prefix in prefixes.items()}
    fixed_suffix = p_generated["neutral"]["ids"]
    p_scored = {name: adapter.source_scores(actual["image"], actual["question"], prefix["ids"] + fixed_suffix,
                                            contexts, offset=len(prefix["ids"])) for name, prefix in prefixes.items()}
    return {"index": rec["index"], "sample_id": rec["sample_id"], "topic": rec["topic"],
            "ground_truth": rec["ground_truth"], "cue_answer": rec["cue_answer"], "question": actual["question"],
            "question_conditions": questions, "prefix_conditions": prefixes,
            "question_generated": q_generated, "question_fixed_answer_scores": q_scored,
            "prefix_generated": p_generated, "prefix_fixed_suffix_scores": p_scored}


def mavis_sample(args, adapter, dataset, rec):
    from PIL import Image, ImageFilter
    actual = dataset[rec["index"]]
    contexts = [{"index": i, "image": dataset[i]["image"], "question": str(dataset[i]["Q"])} for i in rec["context_ids"]]
    image = actual["image"].convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
    generated = adapter.generate(image, rec["question"], cap=args.cap)
    scored = adapter.source_scores(image, rec["question"], generated["ids"], contexts)
    controls = {"target": image, "blur_8": image.filter(ImageFilter.GaussianBlur(8)),
                "blur_27": image.filter(ImageFilter.GaussianBlur(27)),
                **{f"mismatch_{seed}": dataset[rec[f"mismatch_{seed}"]]["image"] for seed in (42, 137)}}
    visual = {name: {"visual_only": adapter.score(photo, "", generated["ids"]),
                     "full": adapter.score(photo, rec["question"], generated["ids"])} for name, photo in controls.items()}
    return {"index": rec["index"], "question": rec["question"], "domain": rec["domain"],
            "reference_facts": actual["verifiable_facts"], "generation": generated,
            "source_scores": scored, "visual_probe_raw_log_probs": visual,
            "mismatch_indices": {str(seed): rec[f"mismatch_{seed}"] for seed in (42, 137)}}


def worker(args):
    import torch
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = DirectParquetRows(args.dataset_path) if args.task == "long" else load_mavis_parquet(args.dataset_path)
    checkpoint(args)
    model, parameters = model_load(args)
    adapter = HFAdapter(args.model_path, args.family, model, args.task)
    metadata = {"task": args.task, "family": args.family, "model_path": str(args.model_path),
                "parameters": parameters, "dtype": "bfloat16", "attention": "sdpa", "offload": False,
                "model_config_sha256": digest(args.model_path / "config.json"),
                "entry_sha256": digest(__file__), "common_sha256": digest(Path(__file__).with_name("long_source_common.py")),
                "manifest_sha256": digest(args.output_dir / "manifest.json"),
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "physical_image_size": [448, 448], "visual_tokenization": "model-native processor",
                "null_image": "black 448x448", "score_backend": "full_prompt_teacher_forcing_no_KV_reuse",
                "all_token_scores": True, "cap": args.cap, "support_size": args.support_size}
    if args.family == "internvl":
        metadata["download_manifest_sha256"] = digest(args.model_path / "download_manifest.json")
    (args.output_dir / f"worker_{args.worker_id}_configuration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assigned = manifest["records"][args.worker_id::8]
    with (args.output_dir / f"shard_{args.worker_id}.jsonl").open("x", encoding="utf-8") as stream:
        for rec in assigned:
            write_state(args.output_dir / f"worker_{args.worker_id}_state.json", "sample", index=rec["index"])
            result = long_sample(args, adapter, dataset, rec) if args.task == "long" else mavis_sample(args, adapter, dataset, rec)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            adapter.reset()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[validated] worker={args.worker_id} index={rec['index']}", flush=True)
    (args.output_dir / f"worker_{args.worker_id}_completion.json").write_text(
        json.dumps({"indices": [r["index"] for r in assigned], "complete": True}), encoding="utf-8")
    write_state(args.output_dir / f"worker_{args.worker_id}_state.json", "complete")


def merge(args):
    from analyze_long_source_validation import analyze
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    rows = []
    for worker_id in range(8):
        completed = json.loads((args.output_dir / f"worker_{worker_id}_completion.json").read_text())
        expected = [r["index"] for r in manifest["records"][worker_id::8]]
        if completed.get("indices") != expected or not completed.get("complete"):
            raise ValueError(f"Worker {worker_id} is incomplete")
        with (args.output_dir / f"shard_{worker_id}.jsonl").open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    rows.sort(key=lambda x: x["index"])
    if [r["index"] for r in rows] != sorted(r["index"] for r in manifest["records"]):
        raise ValueError("Missing or duplicate output rows")
    with (args.output_dir / "merged.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    analyze(rows, args.task, args.output_dir)
    print(f"[complete] validated and analyzed {len(rows)} samples: {args.output_dir}", flush=True)


def launch(args):
    devices, manifest = preflight(args, True)
    args.output_dir.mkdir(exist_ok=False)
    (args.output_dir / "logs").mkdir()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    processes, streams, capture_errors = [], [], []

    def capture(process, index):
        try:
            with (args.output_dir / "logs" / f"worker_{index}.log").open("x", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[worker {index}] {line}", end="", flush=True)
        except Exception as error:
            capture_errors.append(str(error))

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for index, device in enumerate(devices):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--task", args.task,
                       "--project_root", str(args.project_root), "--model_path", str(args.model_path),
                       "--output_dir", str(args.output_dir), "--family", args.family, "--worker_id", str(index),
                       "--support_size", str(args.support_size), "--cap", str(args.cap)]
            if args.existing_manifest:
                command.extend(["--existing_manifest", str(args.existing_manifest)])
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="strict", start_new_session=True)
            processes.append(process)
            thread = threading.Thread(target=capture, args=(process, index), daemon=True)
            thread.start()
            streams.append(thread)
        pending = set(range(8))
        while pending:
            if capture_errors:
                raise RuntimeError(f"Log capture failed: {capture_errors}")
            for index in list(pending):
                code = processes[index].poll()
                if code is None:
                    continue
                write_state(args.output_dir / f"worker_{index}_exit.json", "worker_exit", exit_code=code)
                if code:
                    raise RuntimeError(f"Worker {index} failed: exit={code}; see worker log/state")
                pending.remove(index)
            time.sleep(.25)
        for thread in streams:
            thread.join()
        if capture_errors:
            raise RuntimeError(f"Log capture failed: {capture_errors}")
        merge(args)
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for thread in streams:
            thread.join(timeout=10)


def main():
    args = parse_args()
    if args.preflight:
        preflight(args)
    elif args.worker:
        worker(args)
    elif args.merge:
        merge(args)
    else:
        launch(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
