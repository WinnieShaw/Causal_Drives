#!/usr/bin/env python3
"""Fixed-answer support-size/draw sensitivity with all member likelihoods retained."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import traceback

import run_mavis_model_scaling as engine
import mavis_safe_scoring as safe
from mavis_local_data import DATA_BACKEND, load_mavis_parquet, verify_output_writable
from mavis_supplement_runtime import check_checkpoint, load_core, load_model, sha256, write_state

KS = (5, 10, 20, 40)
SEEDS = (42, 137, 2026)
COHORT_SEED = 314159
LN2 = math.log(2.0)


def cumulative(values):
    result, total = [], 0.0
    for value in values:
        total += float(value)
        result.append(total)
    return result


def logmean(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite or empty mixture")
    largest = max(values)
    return largest + math.log(sum(math.exp(x - largest) for x in values)) - math.log(len(values))


def metrics_from_members(question_logs, joint_logs, visual_logs, null_logs, k):
    length = len(null_logs)
    if not length or k > len(question_logs) or k > len(joint_logs):
        raise ValueError("Incomplete support")
    if any(len(x) != length for x in question_logs[:k] + joint_logs[:k] + [visual_logs]):
        raise ValueError("Token alignment mismatch")
    null_cum, visual_cum = cumulative(null_logs), cumulative(visual_logs)
    question_cum = [cumulative(x) for x in question_logs[:k]]
    rows = []
    for position in range(0, length, 2):
        q = logmean([x[position] for x in question_cum])
        p = logmean([x[position] for x in joint_logs[:k]])
        rows.append({"t": position + 1, "log_P_A_le_t_null": null_cum[position],
                     "log_P_A_le_t_do_Q": q, "log_P_A_le_t_do_V": visual_cum[position],
                     "log_P_A_t_do_A_prev": p,
                     "QCD_t": (q - null_cum[position]) / LN2,
                     "VCD_t": (visual_cum[position] - null_cum[position]) / LN2,
                     "PCD_t": (p - null_cum[position]) / LN2})
    return {"support_size": k, "per_step": rows,
            **{f"avg_{m.lower()}": sum(x[f"{m}_t"] for x in rows) / len(rows)
               for m in ("QCD", "VCD", "PCD")}}


def read_source(args):
    rows = engine.read_rows(args.fixed_source)
    rows.sort(key=lambda x: x["global_index"])
    if [r["global_index"] for r in rows] != list(range(500)):
        raise ValueError("Source must contain exactly indices 0..499")
    if len({r["question"] for r in rows}) != 500:
        raise ValueError("Source questions must be unique")
    for row in rows:
        if row.get("schema", {}).get("scoring_backend") != safe.BACKEND:
            raise ValueError("Use safe-backend answers, not a legacy failed/partial score file")
        if row.get("schema", {}).get("data_backend") != DATA_BACKEND:
            raise ValueError("Unexpected source data protocol")
        ids = row.get("generation", {}).get("generated_token_ids")
        if not ids or len(ids) != row["generation"]["num_generated_tokens"]:
            raise ValueError("Source must retain exact token IDs")
    configuration = args.fixed_source.parent / "worker_0_configuration.json"
    config = json.loads(configuration.read_text(encoding="utf-8"))
    if config.get("model_size") != args.model_size or Path(config["model_path"]).resolve() != args.model_path:
        raise ValueError("Source answers and selected checkpoint do not match")
    return rows


def make_manifest(rows, n):
    # This selection is independent of score, length, correctness and intervention outcome.
    ids = sorted(random.Random(COHORT_SEED).sample(list(range(500)), n))
    records = []
    for index in ids:
        supports = {}
        for seed in SEEDS:
            pool = [i for i in range(500) if i != index]
            random.Random(seed * 1_000_003 + index).shuffle(pool)
            supports[str(seed)] = [index] + pool[:max(KS) - 1]
        records.append({"global_index": index, "question": rows[index]["question"],
                        "support_indices_by_seed": supports})
    return {"schema": "mavis_support_stability_v1", "n": n, "cohort_seed": COHORT_SEED,
            "ks": list(KS), "seeds": list(SEEDS), "pool_indices": [0, 499],
            "target_inclusion": "Target is member zero at every K, matching the original empirical support convention.",
            "draws": "Independent across seeds; nested K within each seed; paired real image-question rows.",
            "vcd_note": "VCD has no empirical-support term; K invariance is structural, not an independent validation.",
            "records": records}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "launch", "worker", "merge"):
        modes.add_argument(f"--{mode}", action="store_true")
    for name in ("project_root", "model_path", "dataset_path", "output_dir", "fixed_source"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model_size", choices=("8b", "32b"), required=True)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--gpu_max_memory", default="110GiB")
    parser.add_argument("--n", type=int, default=int(os.environ.get("MAVIS_STABILITY_N", "100")))
    for name in ("core_script", "summary_script", "safe_scoring_script", "local_data_script"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args(argv)
    if not 8 <= args.n <= 500 or not 0 <= args.worker_id < 8:
        parser.error("Require n=8..500 and worker_id=0..7")
    for name in ("project_root", "model_path", "dataset_path", "output_dir", "fixed_source"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.output_dir.is_relative_to(args.project_root / "outputs" / "answer_source_validation"):
        parser.error("Output must stay inside outputs/answer_source_validation")
    base = Path(__file__).resolve().parent
    for key, filename in (("core_script", "compute_mavis_drive_scores_logdomain.py"),
                          ("summary_script", "summarize_mavis_drive_scores.py"),
                          ("safe_scoring_script", "mavis_safe_scoring.py"),
                          ("local_data_script", "mavis_local_data.py")):
        setattr(args, key, (getattr(args, key) or base / filename).resolve())
    engine.memory_bytes(args.gpu_max_memory)
    return args


def preflight(args, check_gpus):
    import torch
    import numpy  # noqa: F401
    import scipy.stats  # noqa: F401
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    del Qwen3VLForConditionalGeneration
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    for path in (args.core_script, args.summary_script, args.safe_scoring_script, args.local_data_script):
        if not path.is_file():
            raise FileNotFoundError(path)
    shard_count = check_checkpoint(args.model_path)
    rows = read_source(args)
    manifest = make_manifest(rows, args.n)
    dataset = load_mavis_parquet(args.dataset_path)
    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True,
                                              trust_remote_code=True, max_pixels=256 * 256)
    for row in rows:
        index = row["global_index"]
        actual = dataset[index]
        actual["image"].load()
        if str(actual["Q"]) != row["question"]:
            raise ValueError(f"Question mismatch: {index}")
        if processor.decode(row["generation"]["generated_token_ids"], skip_special_tokens=True).strip() != row["prediction"]:
            raise ValueError(f"Source token decode mismatch: {index}")
    probe = verify_output_writable(args.output_dir.parent)
    devices = []
    if check_gpus:
        if torch.cuda.device_count() != 8:
            raise RuntimeError("Require exactly eight scheduler-visible GPUs")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else [str(i) for i in range(8)]
        if len(devices) != 8 or len(set(devices)) != 8:
            raise ValueError("Expected eight distinct device assignments")
    print(json.dumps({"cpu_preflight": "passed", "model_loaded": False,
                      "samples": manifest["n"], "ks": KS, "seeds": SEEDS,
                      "decoded_images": 500, "checkpoint_shards": shard_count,
                      "output_write_check": probe, "source_sha256": sha256(args.fixed_source),
                      "data_backend": DATA_BACKEND, "scoring_backend": safe.BACKEND}, ensure_ascii=False), flush=True)
    return devices


def worker(args):
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one GPU per worker")
    os.chdir(args.output_dir)
    rows = read_source(args)
    manifest = make_manifest(rows, args.n)
    core = load_core(args)
    dataset = load_mavis_parquet(args.dataset_path)
    model = load_model(args)
    core.MODEL_PATH = str(args.model_path)
    processor = core.get_processor(core.BASE_MAX_PIXELS)
    source_hash = sha256(args.fixed_source)
    engine.write_json_new(args.output_dir / f"worker_{args.worker_id}_configuration.json", {
        "model_size": args.model_size, "model_path": str(args.model_path),
        "dtype": "float16", "attention": "sdpa", "model_cpu_offload": False,
        "n": args.n, "ks": KS, "seeds": SEEDS, "cohort_seed": COHORT_SEED,
        "source_sha256": source_hash, "entry_sha256": sha256(__file__),
        "core_sha256": sha256(args.core_script), "safe_scoring_sha256": sha256(args.safe_scoring_script),
        "checkpoint_config_sha256": sha256(args.model_path / "config.json"),
        "support_target_included": True, "temporal_stride": 2})

    def score(image, question, ids):
        prompt = core.build_prompt_cache(model, processor, image, question)
        result = core.score_fixed_sequence_from_prompt_cache(model, prompt, ids)
        values = [float(x) for x in result["token_log_probs"]]
        if len(values) != len(ids) or not all(math.isfinite(x) for x in values):
            raise ValueError("Incomplete or nonfinite fixed-answer scores")
        return values

    assigned = manifest["records"][args.worker_id::8]
    output = args.output_dir / f"support_worker_{args.worker_id}.jsonl"
    with output.open("x", encoding="utf-8") as stream:
        for record in assigned:
            index = record["global_index"]
            write_state(args.output_dir, args.worker_id, "scoring", global_index=index)
            row = rows[index]
            ids = torch.tensor(row["generation"]["generated_token_ids"], dtype=torch.long)
            target_image, _ = core.normalize_image(dataset[index]["image"])
            null_logs = score(core.build_null_image(), core.EMPTY_PROMPT, ids)
            visual_logs = score(target_image, core.EMPTY_PROMPT, ids)
            # The same image/question prompt is reused only as immutable CPU inputs.
            # No KV state is reused. Score results for repeated members are exact memoization.
            cache = {}
            draws = []
            for seed in SEEDS:
                members = record["support_indices_by_seed"][str(seed)]
                q_logs, joint_logs = [], []
                for member in members:
                    for kind in ("question", "joint"):
                        key = (member, "joint" if member == index else kind)
                        if key not in cache:
                            actual = dataset[member]
                            image, _ = core.normalize_image(actual["image"])
                            question = row["question"] if kind == "question" else str(actual["Q"])
                            cache[key] = score(image, question, ids)
                        (q_logs if kind == "question" else joint_logs).append(cache[key])
                draws.append({"seed": seed, "support_indices": members,
                              "question_member_token_log_probs": q_logs,
                              "joint_member_token_log_probs": joint_logs,
                              "metrics_by_k": {str(k): metrics_from_members(q_logs, joint_logs, visual_logs, null_logs, k) for k in KS}})
            result = {"global_index": index, "question": row["question"], "prediction": row["prediction"],
                      "meta": row.get("meta"), "generated_token_ids": ids.tolist(),
                      "source_drive_metrics": row["drive_metrics"], "source_support_size": 20,
                      "null_token_log_probs": null_logs, "visual_only_token_log_probs": visual_logs,
                      "draws": draws, "schema": manifest["schema"],
                      "scoring_backend": safe.BACKEND, "source_sha256": source_hash}
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            del cache, draws, ids
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[validated] worker={args.worker_id} sample={index} K={KS} draws={SEEDS}", flush=True)
    engine.write_json_new(args.output_dir / f"worker_{args.worker_id}_completion.json", {
        "worker_id": args.worker_id, "indices": [r["global_index"] for r in assigned], "all_samples_validated": True})
    write_state(args.output_dir, args.worker_id, "complete")


def merge(args):
    import numpy as np
    from scipy.stats import spearmanr
    source = read_source(args)
    manifest = make_manifest(source, args.n)
    expected = [r["global_index"] for r in manifest["records"]]
    rows = []
    for worker_id in range(8):
        completion = json.loads((args.output_dir / f"worker_{worker_id}_completion.json").read_text())
        if completion.get("indices") != expected[worker_id::8] or completion.get("all_samples_validated") is not True:
            raise ValueError(f"Incomplete worker: {worker_id}")
        rows.extend(engine.read_rows(args.output_dir / f"support_worker_{worker_id}.jsonl"))
    rows.sort(key=lambda x: x["global_index"])
    if [r["global_index"] for r in rows] != expected:
        raise ValueError("Missing or duplicate cohort samples")
    # Reconstruct every metric from raw member likelihoods before summarization.
    for row in rows:
        for draw in row["draws"]:
            for k in KS:
                recalculated = metrics_from_members(draw["question_member_token_log_probs"], draw["joint_member_token_log_probs"],
                                                    row["visual_only_token_log_probs"], row["null_token_log_probs"], k)
                if recalculated != draw["metrics_by_k"][str(k)]:
                    raise ValueError("Stored scores do not match raw-member reconstruction")
    with (args.output_dir / "support_stability_merged.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {"n": args.n, "model_size": args.model_size, "ks": list(KS), "seeds": list(SEEDS),
               "source_sha256": sha256(args.fixed_source), "metrics": {},
               "reference": "K=40 is a finite-support sensitivity reference, not causal ground truth.",
               "vcd_note": manifest["vcd_note"]}
    for metric in ("qcd", "vcd", "pcd"):
        statistics = {}
        for k in KS:
            matrix = np.array([[d["metrics_by_k"][str(k)][f"avg_{metric}"] for d in r["draws"]] for r in rows])
            reference = np.array([[d["metrics_by_k"]["40"][f"avg_{metric}"] for d in r["draws"]] for r in rows])
            base = np.array([r["source_drive_metrics"][f"avg_{metric}"] for r in rows])
            statistics[str(k)] = {
                "mean_score_bits": float(matrix.mean()),
                "mean_between_draw_sd_bits": float(matrix.std(axis=1, ddof=1).mean()),
                "mean_absolute_difference_to_nested_k40_bits": float(np.abs(matrix - reference).mean()),
                "spearman_to_nested_k40_by_draw": [float(spearmanr(matrix[:, i], reference[:, i]).statistic) for i in range(3)],
                "spearman_to_original_k20_by_draw": [float(spearmanr(matrix[:, i], base).statistic) for i in range(3)]}
        summary["metrics"][metric] = statistics
    engine.write_json_new(args.output_dir / "support_stability_summary.json", summary)
    engine.write_json_new(args.output_dir / "support_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    engine.normalize_allocator_environment()
    args = parse_args()
    os.environ["MAVIS_STABILITY_N"] = str(args.n)
    engine.__file__ = str(Path(__file__).resolve())
    if args.preflight:
        preflight(args, False)
    elif args.worker:
        worker(args)
    elif args.merge:
        merge(args)
    else:
        engine.preflight = preflight
        engine.merge_and_summarize = merge
        engine.launch(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
