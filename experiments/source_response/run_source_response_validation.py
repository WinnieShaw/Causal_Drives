#!/usr/bin/env python3
"""Independent visual counterfactuals and forced-prefix mechanism diagnostics.

All original drive definitions are imported unchanged. No GPU inference is run
by the CPU manifest/preflight paths. Fixed candidate paths and free generation
are separate outputs; answer matching is not a hidden-source ground truth.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

from source_response_common import aliases, classify, content_indices, drive_window, normalize, read_jsonl
from source_response_runtime import (DirectParquetRows, checkpoint_dtype_name, load_full_gpu_model,
                                     model_loading_lock, write_state)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_manifest(inputs, output, per_topic, seed, mode):
    rows, hashes = read_jsonl(inputs)
    good = [r for r in rows if r.get("status") == "ok"]
    by_id = {r["index"]: r for r in good}
    if len(by_id) != len(good):
        raise ValueError("Duplicate input sample IDs; use one run/model only.")
    groups = defaultdict(list)
    for r in good:
        groups[r["topic"]].append(r)
    rng = random.Random(seed)
    records = []
    for topic in sorted(groups):
        candidates = sorted(groups[topic], key=lambda r: r["index"])
        rng.shuffle(candidates)
        for r in candidates[:per_topic]:
            visual, cue = aliases(r["ground_truth"]), aliases(r["expected_bias"])
            rec = {"index": r["index"], "sample_id": r["sample_id"], "topic": topic,
                   "question": r["question"], "ground_truth": r["ground_truth"],
                   "cue_answer": r["expected_bias"], "visual_candidate": visual[0] if visual else None,
                   "cue_candidate": cue[0] if cue else None, "base_prediction": r["base"]["prediction"],
                   "eligible_prefix": bool(visual and cue and not set(visual).intersection(cue))}
            compatible = [x for x in groups[topic] if x["index"] != r["index"]
                          and str(x["question"]).strip() == str(r["question"]).strip()
                          and set(aliases(x["expected_bias"])) == set(cue)]
            cf = [x for x in compatible if aliases(x["ground_truth"])
                  and not (set(visual) | set(cue)).intersection(aliases(x["ground_truth"]))]
            sham = [x for x in compatible if set(aliases(x["ground_truth"])) == set(visual)]
            rng.shuffle(cf)
            rng.shuffle(sham)
            rec["counterfactual_index"] = cf[0]["index"] if cf else None
            rec["sham_index"] = sham[0]["index"] if sham else None
            rec["counterfactual_ground_truth"] = cf[0]["ground_truth"] if cf else None
            rec["eligible_visual"] = bool(visual and cf and sham)
            excluded = {r["index"], rec["counterfactual_index"], rec["sham_index"]}
            support = [x["index"] for x in groups[topic] if x["index"] not in excluded]
            rng.shuffle(support)
            if len(support) < 2:
                raise ValueError("Need two fixed non-target support examples per sample.")
            rec["context_ids"] = support[:2]
            records.append(rec)
    manifest = {"schema": "source_response_v1", "seed": seed, "per_topic": per_topic,
                "mode": mode, "num_samples": len(records), "input_hashes": hashes,
                "selection": "Topic-stratified input selection before outcomes; no score-based selection.",
                "support": "K=2 fixed examples excluding target, counterfactual and sham; identical across conditions.",
                "records": records}
    Path(output).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(records), "prefix_eligible": sum(r["eligible_prefix"] for r in records),
                      "visual_eligible": sum(r["eligible_visual"] for r in records), "manifest": output}))


def checkpoint_preflight(model_path, dataset_dir, core_dir):
    root = Path(model_path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    else:
        weight_files = {p.name for p in root.glob("*.safetensors")}
    if not weight_files or any(not (root / f).is_file() or (root / f).stat().st_size == 0 for f in weight_files):
        raise FileNotFoundError("Checkpoint has missing/empty weight shards.")
    for path in [Path(dataset_dir) / "data/main-00000-of-00001.parquet",
                 Path(core_dir) / "run_source_conflict_logdomain.py"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset = DirectParquetRows(Path(dataset_dir) / "data/main-00000-of-00001.parquet")
    print(json.dumps({"checkpoint_type": config.get("model_type"), "weight_shards": len(weight_files),
                      "checkpoint_dtype": checkpoint_dtype_name(model_path), "dataset_rows": len(dataset),
                      "weights_bytes": sum((root / f).stat().st_size for f in weight_files),
                      "core_sha256": digest(Path(core_dir) / "run_source_conflict_logdomain.py"),
                      "check": "CPU file preflight only; not a GPU inference pass."}))


def validate_manifest_dataset(core, ds, records, mode, dataset_dir):
    required = sorted({i for rec in records for i in
                       rec["context_ids"] + [rec["index"]] +
                       ([rec["counterfactual_index"], rec["sham_index"]]
                        if mode in {"all", "visual"} and rec["eligible_visual"] else [])})
    for index in required:
        if not 0 <= index < len(ds):
            raise ValueError(f"Manifest index outside main Parquet: {index}")
    for rec in records:
        row = core.resolve_row(ds[rec["index"]], rec["index"], dataset_dir)
        if (row["sample_id"] != str(rec["sample_id"]) or row["question"] != rec["question"]
                or set(aliases(row["ground_truth"])) != set(aliases(rec["ground_truth"]))):
            raise ValueError(f"Manifest/Parquet row mismatch at {rec['index']}")
    return {"target_rows_verified": len(records), "required_indices": len(required), "dataset_rows": len(ds)}


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def pad_prefix_ids(tokenizer, conditions):
    """Length matching is explicit forced-token whitespace padding, not truncation."""
    ids = {name: tokenizer.encode(text, add_special_tokens=False) for name, text in conditions.items()}
    space = tokenizer.encode(" ", add_special_tokens=False)
    if len(space) != 1:
        raise ValueError("Need a single whitespace token for audited prefix length matching.")
    length = max(map(len, ids.values()))
    result = {}
    for name, value in ids.items():
        padded = [space[0]] * (length - len(value)) + value
        result[name] = {"ids": padded, "original_text": conditions[name],
                        "decoded_text": tokenizer.decode(padded, skip_special_tokens=False),
                        "leading_padding_tokens": length - len(value), "token_count": length}
    return result


def generate_after_prefix(core, processor, model, image, question, instruction, prefix_ids, max_new_tokens):
    import torch
    inputs = core.prepare_prompt_inputs(processor, model, image, question, instruction)
    device = inputs["input_ids"].device
    prefix = torch.tensor(prefix_ids, device=device, dtype=torch.long).unsqueeze(0)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(prefix)], dim=1)
    # Extend optional modality-type inputs when emitted by a newer processor.
    for name in ["token_type_ids", "mm_token_type_ids"]:
        if name in inputs:
            inputs[name] = torch.cat([inputs[name], torch.zeros_like(prefix)], dim=1)
    start = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   use_cache=True, pad_token_id=processor.tokenizer.eos_token_id)
    continuation_ids = generated[0, start:].detach().cpu().tolist()
    result = {"text": processor.decode(continuation_ids, skip_special_tokens=True).strip(),
              "generated_ids": continuation_ids}
    del inputs, generated, prefix
    core.cleanup()
    return result


def fixed_scores(core, processor, model, image, question, instruction, target_ids, records, context_ids,
                 null_image, selected, static=None):
    import torch
    target = torch.tensor(target_ids, dtype=torch.long)
    tokenizer = processor.tokenizer
    tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in target_ids]
    prefixes = [tokenizer.decode(target_ids[:i + 1], skip_special_tokens=True) for i in range(len(target_ids))]
    if static:
        assert static["target_ids"] == target_ids and static["context_ids"] == context_ids and static["question"] == question
    null = static["scores"]["null"] if static else core.score_fixed_answer(processor, model, null_image, "", "", target)
    visual = core.score_fixed_answer(processor, model, image, "", "", target)
    full = core.score_fixed_answer(processor, model, image, question, instruction, target)
    null_question = static["scores"]["null_question"] if static else core.score_fixed_answer(processor, model, null_image, question, instruction, target)
    joint = {tuple(map(int, k.split(":"))): v for k, v in static["scores"]["joint"].items()} if static else {}
    # Target question is separate from the fixed empirical support. This prevents
    # image edits from changing the marginalization distribution accidentally.
    target_q = -1
    if not static:
        for vi in context_ids:
            for qi in context_ids:
                joint[(vi, qi)] = core.score_fixed_answer(processor, model, records[vi]["image"],
                                                          records[qi]["question"], instruction, target)
            joint[(vi, target_q)] = core.score_fixed_answer(processor, model, records[vi]["image"],
                                                           question, instruction, target)
    pv, pqv = core.build_uniform_distributions(context_ids)
    metrics = core.compute_causal_drives_logdomain(
        baseline_score=null, visual_only_score=visual, joint_scores=joint,
        target_visual_id=-1, target_question_id=target_q, p_v=pv, p_q_given_v=pqv,
        prefixes=prefixes, tokens=tokens, step_interval=1)
    window = drive_window(metrics, tokens, selected)
    window["full_content_logprob"] = sum(full["token_log_probs"][i] for i in selected) / len(selected)
    window["alternative_image_llr_content_bits"] = sum(
        full["token_log_probs"][i] - joint[(context_ids[0], target_q)]["token_log_probs"][i]
        for i in selected) / (len(selected) * core.LN2)
    window["conditional_visual_llr_content_bits"] = sum(
        full["token_log_probs"][i] - null_question["token_log_probs"][i]
        for i in selected) / (len(selected) * core.LN2)
    return {"target_ids": target_ids, "tokens": tokens, "selected_content_indices": selected,
            "context_ids": context_ids, "question": question,
            "drive_metrics": metrics, "content_summary": window,
            "scores": {"null": null, "null_question": null_question, "visual_only": visual, "full": full,
                       "joint": {f"{vi}:{qi}": s for (vi, qi), s in joint.items()}}}


def run_worker(a):
    import faulthandler
    faulthandler.enable(all_threads=True)
    state_path = Path(a.output).with_suffix(".state.json")
    write_state(state_path, "imports", worker_id=a.worker_id)
    import torch
    from transformers import AutoProcessor
    sys.path.insert(0, str(Path(a.core_dir).resolve()))
    import run_source_conflict_logdomain as core
    write_state(state_path, "file_preflight", worker_id=a.worker_id)
    checkpoint_preflight(a.model_path, a.dataset_dir, a.core_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU worker requested but CUDA unavailable; no CPU model fallback.")
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    if a.mode != manifest["mode"]:
        raise ValueError("Worker mode differs from the planned manifest.")
    assigned = manifest["records"][a.worker_id::a.num_workers]
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    write_state(state_path, "data_open", worker_id=a.worker_id)
    ds = DirectParquetRows(Path(a.dataset_dir) / "data/main-00000-of-00001.parquet")
    # Validate row identities before allocating model weights, without creating
    # a shared HF Arrow cache or keeping every image in process memory.
    validated = validate_manifest_dataset(core, ds, assigned, a.mode, a.dataset_dir)
    write_state(state_path, "data_verified", worker_id=a.worker_id, **validated)
    processor = AutoProcessor.from_pretrained(a.model_path, trust_remote_code=True, local_files_only=True)
    write_state(state_path, "waiting_for_model_lock", worker_id=a.worker_id)
    with model_loading_lock(a.load_lock or str(out.parent / "model_loading.lock")):
        write_state(state_path, "model_loading", worker_id=a.worker_id,
                    model_path=a.model_path, dtype=checkpoint_dtype_name(a.model_path),
                    gpu_name=torch.cuda.get_device_name(0),
                    gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        model = load_full_gpu_model(a.model_path, a.max_memory_per_gpu)
    null_image = core.make_null_image(448, 127)
    meta = {"schema": manifest["schema"], "arguments": vars(a), "manifest_sha256": digest(a.manifest),
            "core_sha256": digest(Path(a.core_dir) / "run_source_conflict_logdomain.py"),
            "worker_sha256": digest(__file__), "checkpoint_config_sha256": digest(Path(a.model_path) / "config.json"),
            "runtime_sha256": digest(Path(__file__).with_name("source_response_runtime.py")),
            "dtype": str(next(model.parameters()).dtype), "transformers": __import__("transformers").__version__,
            "torch": torch.__version__, "data_backend": "direct_parquet_row_groups_no_hf_cache",
            "model_backend": "complete_single_gpu_local_sdpa_no_offload",
            "scope": "Paired source-response diagnostic, not hidden-source ground truth."}
    write_state(out.with_suffix(".metadata.json"), "model_ready", **meta)
    write_state(state_path, "model_ready", worker_id=a.worker_id,
                gpu_allocated_bytes=torch.cuda.memory_allocated(0))
    errors = 0
    for pos, rec in enumerate(assigned, 1):
        write_state(state_path, "sample", worker_id=a.worker_id, index=rec["index"],
                    position=pos, total=len(assigned))
        core.log(f"worker={a.worker_id} [{pos}/{len(assigned)}] index={rec['index']}")
        item = {"status": "ok", "index": rec["index"], "sample_id": rec["sample_id"],
                "topic": rec["topic"], "manifest_record": rec, "prefix": None, "visual": None}
        try:
            needed = set(rec["context_ids"] + [rec["index"]])
            if a.mode in {"all", "visual"} and rec["eligible_visual"]:
                needed.update([rec["counterfactual_index"], rec["sham_index"]])
            records = {i: core.resolve_row(ds[i], i, a.dataset_dir) for i in needed}
            target = records[rec["index"]]
            tokenizer = processor.tokenizer
            suffix_ids = tokenizer.encode("{" + rec["visual_candidate"] + "}", add_special_tokens=False) if rec["visual_candidate"] else []
            suffix_tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in suffix_ids]
            selected_suffix = content_indices(suffix_tokens)
            if suffix_ids and not selected_suffix:
                raise ValueError("Candidate has no identifiable lexical content.")
            if a.mode in {"all", "prefix"} and rec["eligible_prefix"]:
                prefix_texts = {
                    "visual": f"Initial estimate: {rec['visual_candidate']}. Final answer: ",
                    "cue": f"Initial estimate: {rec['cue_candidate']}. Final answer: ",
                    "neutral": "Initial estimate: unknown. Final answer: ",
                }
                padded = pad_prefix_ids(tokenizer, prefix_texts)
                prefix_result = {}
                for condition, prefix in padded.items():
                    ids = prefix["ids"] + suffix_ids
                    selected = [len(prefix["ids"]) + i for i in selected_suffix]
                    scored = fixed_scores(core, processor, model, target["image"], target["question"],
                                          a.answer_instruction, ids, records, rec["context_ids"], null_image, selected)
                    generated = generate_after_prefix(core, processor, model, target["image"], target["question"],
                                                      a.answer_instruction, prefix["ids"], a.max_new_tokens)
                    observed_tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in generated["generated_ids"]]
                    observed_content = content_indices(observed_tokens)
                    observed = fixed_scores(
                        core, processor, model, target["image"], target["question"], a.answer_instruction,
                        prefix["ids"] + generated["generated_ids"], records, rec["context_ids"], null_image,
                        [len(prefix["ids"]) + i for i in observed_content]) if observed_content else None
                    prefix_result[condition] = {"prefix": prefix, "fixed_candidate": scored,
                                                "observed_generation": observed,
                                                "continuation": generated,
                                                "answer_match": classify(generated["text"], rec["ground_truth"], rec["cue_answer"])}
                item["prefix"] = prefix_result
            if a.mode in {"all", "visual"} and rec["eligible_visual"]:
                visual_result = {}
                for condition, idx in [("base", rec["index"]), ("counterfactual", rec["counterfactual_index"]),
                                       ("sham", rec["sham_index"])]:
                    im = records[idx]["image"]
                    # The source estimand and candidate stay fixed; full logs
                    # retain the real question for independent behavioral checks.
                    scored = fixed_scores(core, processor, model, im, target["question"], a.answer_instruction,
                                          suffix_ids, records, rec["context_ids"], null_image, selected_suffix,
                                          static=visual_result["base"]["fixed_candidate"] if visual_result else None)
                    generated = core.generate_answer(processor, model, im, target["question"],
                                                     a.answer_instruction, a.max_new_tokens)
                    observed_ids = generated["generated_ids"].tolist()
                    observed_content = content_indices(generated["tokens"])
                    observed = fixed_scores(core, processor, model, im, target["question"], a.answer_instruction,
                                            observed_ids, records, rec["context_ids"], null_image,
                                            observed_content) if observed_content else None
                    visual_result[condition] = {"image_index": idx, "fixed_candidate": scored,
                                                 "observed_generation": observed,
                                                 "prediction": generated["text"],
                                                 "generated_ids": observed_ids,
                                                 "answer_match": {
                                                     "visual_answer_match": "original_visual_answer_match",
                                                     "cue_answer_match": "counterfactual_visual_answer_match",
                                                 }.get(classify(generated["text"], rec["ground_truth"], rec["counterfactual_ground_truth"]),
                                                       classify(generated["text"], rec["ground_truth"], rec["counterfactual_ground_truth"]))}
                item["visual"] = visual_result
        except Exception as exc:
            errors += 1
            item["status"] = "error"
            item["error"] = str(exc)
            item["traceback"] = traceback.format_exc()
        append_record(out, item)
        core.cleanup()
    write_state(state_path, "completed", worker_id=a.worker_id, records=len(assigned), errors=errors)
    if errors:
        raise SystemExit(f"{errors} records failed; raw errors were retained, run is not complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+")
    p.add_argument("--manifest", required=True)
    p.add_argument("--prepare_only", action="store_true")
    p.add_argument("--preflight_only", action="store_true")
    p.add_argument("--validate_manifest_only", action="store_true", help="CPU real-row/image check; no model loading")
    p.add_argument("--per_topic", type=int, default=30)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--mode", choices=["all", "prefix", "visual"], default="all")
    p.add_argument("--model_path")
    p.add_argument("--dataset_dir")
    p.add_argument("--core_dir")
    p.add_argument("--worker_id", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output")
    p.add_argument("--max_memory_per_gpu", default="120GiB")
    p.add_argument("--cpu_memory", default="200GiB")
    p.add_argument("--load_lock", help="Run-local lock serializing model weight initialization")
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--answer_instruction", default="Answer briefly using only the visible evidence.")
    a = p.parse_args()
    if a.prepare_only:
        if not a.input:
            p.error("--prepare_only requires --input")
        prepare_manifest(a.input, a.manifest, a.per_topic, a.seed, a.mode)
        return
    if not all([a.model_path, a.dataset_dir, a.core_dir]):
        p.error("model, dataset and core paths are required")
    if a.preflight_only:
        checkpoint_preflight(a.model_path, a.dataset_dir, a.core_dir)
        return
    if a.validate_manifest_only:
        checkpoint_preflight(a.model_path, a.dataset_dir, a.core_dir)
        sys.path.insert(0, str(Path(a.core_dir).resolve()))
        import run_source_conflict_logdomain as core
        manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
        ds = DirectParquetRows(Path(a.dataset_dir) / "data/main-00000-of-00001.parquet")
        print(json.dumps(validate_manifest_dataset(core, ds, manifest["records"], manifest["mode"], a.dataset_dir)))
        return
    if not a.output or not (0 <= a.worker_id < a.num_workers):
        p.error("Valid output/worker partition required")
    try:
        run_worker(a)
    except BaseException as exc:
        state_path = Path(a.output).with_suffix(".state.json")
        prior = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        write_state(state_path, "failed", worker_id=a.worker_id,
                    failed_stage=prior.get("stage"), error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
