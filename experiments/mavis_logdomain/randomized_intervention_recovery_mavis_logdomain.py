#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Held-out randomized intervention recovery for MAVIS, evaluated in log space.

Goal
----
Compare the causal estimators (QCD/PCD) and reference-calibrated
observational counterparts (Q-Obs/P-Obs) against independently randomized
intervention references.

No answer is regenerated. The exact generated token sequence stored in the
merged JSONL is reconstructed and teacher-forced under held-out contexts.

Question reference:
  G_Q(t) = log2 [ E_{V~heldout}[ P(A_<=t | V, q_i) ] / P_ref(A_<=t) ]

Prefix reference:
  G_P(t) = log2 [ E_{(V,Q)~heldout}[ P(A_t | A_<t, V, Q) ] / P_ref(A_<=t) ]

The Prefix denominator intentionally matches the PCD definition.
Mixtures and likelihood ratios are evaluated directly from log probabilities;
no probability-domain floor is applied.

Defaults:
  - targets: 5 randomly selected targets per original K=20 block
             => 50 targets across 10 blocks
  - held-out randomized support M=100
  - held-out pool: MAVIS rows 200..end
  - 8-worker friendly via --worker_id / --num_workers
"""

import argparse
import gc
import io
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageFile
from transformers import AutoProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------
# Defaults: same model/data/prompt conventions as the causal experiment
# ---------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "")
DEFAULT_DATASET_PATH = os.environ.get("MAVIS_DATASET_PATH", "")
DEFAULT_MERGED = os.environ.get("MAVIS_MERGED_SCORES", "")

EPS = 1e-12
EMPTY_PROMPT = ""
NULL_IMAGE_SIZE = (448, 448)
NULL_IMAGE_COLOR = (0, 0, 0)

ATTN_IMPL = "sdpa"
BASE_MAX_PIXELS = 256 * 256
RETRY_MAX_PIXELS = 144 * 144
FORCE_IMAGE_SIZE = (448, 448)
MAX_QUESTION_CHARS = 100000
MODEL_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
try:
    from transformers import Qwen3VLForConditionalGeneration

    def load_model(model_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")

        # With CUDA_VISIBLE_DEVICES=<physical_id>, this worker sees one logical GPU: cuda:0.
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=MODEL_DTYPE,
            attn_implementation=ATTN_IMPL,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        return model

except Exception:
    from transformers import AutoModelForImageTextToText

    def load_model(model_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=MODEL_DTYPE,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        return model


# ---------------------------------------------------------------------
# Input / prompt utilities: kept aligned with the causal experiment
# ---------------------------------------------------------------------
def recursive_to_device(obj: Any, device: str):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(recursive_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [recursive_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    return obj


def get_model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def move_batch_to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    return recursive_to_device(batch, str(device))


def normalize_image(image):
    if not hasattr(image, "size"):
        return image

    target_w, target_h = FORCE_IMAGE_SIZE
    try:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if tuple(image.size) != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.BICUBIC)
    except Exception:
        pass
    return image


def normalize_question(question: str):
    q = str(question)
    if len(q) > MAX_QUESTION_CHARS:
        q = q[:MAX_QUESTION_CHARS]
    return q


def build_null_image():
    return Image.new("RGB", NULL_IMAGE_SIZE, color=NULL_IMAGE_COLOR)


def build_prompt(question: str) -> str:
    question = "" if question is None else str(question).strip()

    if question == "":
        return (
            "Look at the image.\n"
            "Respond in English.\n"
            "Continue generation from the image only.\n"
        )

    return (
        "Answer the question based on the image.\n"
        "Respond in English.\n"
        "Be concise and answer in 1-2 sentences.\n\n"
        f"Question: {question}"
    )


def build_messages(image, question: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(question)},
            ],
        }
    ]


def prepare_single_inputs(processor, image, question: str):
    image = normalize_image(image)
    question = normalize_question(question)

    text = processor.apply_chat_template(
        build_messages(image, question),
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    if "attention_mask" not in inputs and "input_ids" in inputs:
        pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            inputs["attention_mask"] = (inputs["input_ids"] != pad_token_id).long()
        else:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

    return inputs


def get_processor(model_path: str, max_pixels: int):
    return AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        max_pixels=max_pixels,
    )


def prepare_single_inputs_with_retry(processor, model_path, image, question: str):
    try:
        return prepare_single_inputs(processor, image, question), processor
    except RuntimeError as e:
        msg = str(e)
        if "visual tokens too large" in msg or "input_ids too large" in msg:
            small_processor = get_processor(model_path, RETRY_MAX_PIXELS)
            return prepare_single_inputs(small_processor, image, question), small_processor
        raise


# ---------------------------------------------------------------------
# Prompt cache + fixed-sequence scoring
# ---------------------------------------------------------------------
@torch.inference_mode()
def build_prompt_cache(model, processor, model_path, image, question: str):
    inputs, used_processor = prepare_single_inputs_with_retry(
        processor, model_path, image, question
    )

    input_device = get_model_input_device(model)
    inputs = move_batch_to_device(inputs, input_device)

    forward_kwargs = dict(inputs)
    forward_kwargs["use_cache"] = True

    try:
        outputs = model(**forward_kwargs, logits_to_keep=1)
    except TypeError:
        try:
            outputs = model(**forward_kwargs, num_logits_to_keep=1)
        except TypeError:
            outputs = model(**forward_kwargs)

    prompt_last_logits = outputs.logits[0, -1].float().detach().cpu()
    past_key_values = outputs.past_key_values

    cache = {
        "prompt_last_logits": prompt_last_logits,
        "past_key_values": past_key_values,
        "prompt_len": int(inputs["input_ids"].shape[1]),
    }

    del outputs
    del inputs
    return cache, used_processor


@torch.inference_mode()
def score_fixed_sequence_from_prompt_cache(
    model,
    prompt_cache: Dict[str, Any],
    target_generated_ids: torch.Tensor,
):
    if target_generated_ids.numel() == 0:
        return {
            "token_log_probs": [],
            "cum_log_probs": [],
        }

    device = get_model_input_device(model)
    target_ids = target_generated_ids.to(device)
    T = target_ids.shape[0]

    token_log_probs = []
    cum_log_probs = []
    running = 0.0

    # First generated token comes from prompt-last logits.
    first_logits = prompt_cache["prompt_last_logits"]
    token0 = int(target_generated_ids[0].item())
    logp0 = (
        first_logits[token0].item()
        - torch.logsumexp(first_logits, dim=-1).item()
    )
    token_log_probs.append(logp0)
    running += logp0
    cum_log_probs.append(running)

    if T > 1:
        continuation_input_ids = target_ids[:-1].unsqueeze(0)

        try:
            outputs = model(
                input_ids=continuation_input_ids,
                past_key_values=prompt_cache["past_key_values"],
                use_cache=False,
                logits_to_keep=T - 1,
            )
        except TypeError:
            try:
                outputs = model(
                    input_ids=continuation_input_ids,
                    past_key_values=prompt_cache["past_key_values"],
                    use_cache=False,
                    num_logits_to_keep=T - 1,
                )
            except TypeError:
                outputs = model(
                    input_ids=continuation_input_ids,
                    past_key_values=prompt_cache["past_key_values"],
                    use_cache=False,
                )

        continuation_logits = outputs.logits[0].float().detach().cpu()
        usable_steps = min(T - 1, continuation_logits.shape[0])

        for t in range(1, usable_steps + 1):
            step_logits = continuation_logits[t - 1]
            token_id = int(target_generated_ids[t].item())
            logp = (
                step_logits[token_id].item()
                - torch.logsumexp(step_logits, dim=-1).item()
            )
            token_log_probs.append(logp)
            running += logp
            cum_log_probs.append(running)

        del continuation_logits
        del outputs
        del continuation_input_ids

    del target_ids

    return {
        "token_log_probs": token_log_probs,
        "cum_log_probs": cum_log_probs,
    }


def score_condition(model, processor, model_path, image, question, target_ids):
    cache, _ = build_prompt_cache(
        model=model,
        processor=processor,
        model_path=model_path,
        image=image,
        question=question,
    )
    score = score_fixed_sequence_from_prompt_cache(
        model=model,
        prompt_cache=cache,
        target_generated_ids=target_ids,
    )
    del cache
    return score


# ---------------------------------------------------------------------
# Math / reconstruction
# ---------------------------------------------------------------------
def logmeanexp(values: List[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    m = np.max(x)
    return float(m + np.log(np.mean(np.exp(x - m))))


def reconstruct_generated_ids(tokenizer, item):
    """
    Reconstruct the exact generated sequence from the per-token decoded pieces
    already stored by the causal run. Fail loudly if token count changes.
    """
    pieces = item.get("generation", {}).get("generated_tokens", [])
    expected_n = int(item.get("generation", {}).get("num_generated_tokens", len(pieces)))

    if not pieces:
        raise ValueError("generated_tokens missing")

    text_with_special = "".join(str(x) for x in pieces)

    # Prefer token-by-token inversion because each stored piece was decoded
    # from one generated token in the original run.
    piece_ids = []
    piecewise_ok = True
    for p in pieces:
        cur = tokenizer.encode(str(p), add_special_tokens=False)
        if len(cur) != 1:
            piecewise_ok = False
            break
        piece_ids.extend(cur)

    if piecewise_ok and len(piece_ids) == expected_n:
        ids = piece_ids
    else:
        ids = tokenizer.encode(
            text_with_special,
            add_special_tokens=False,
        )
        if len(ids) != expected_n:
            raise RuntimeError(
                f"Cannot exactly reconstruct generated ids: expected {expected_n}, "
                f"full-string tokenization gives {len(ids)} tokens, and "
                "piecewise inversion is not one-token-per-piece. "
                "Do not continue because the randomized reference must score "
                "the exact same generated sequence."
            )

    return torch.tensor(ids, dtype=torch.long), text_with_special


def load_merged(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gi = int(obj["global_index"])
            rows[gi] = obj
    return rows


def choose_targets(
    available_indices,
    block_size=20,
    num_blocks=10,
    targets_per_block=5,
    seed=20260810,
):
    available = set(available_indices)
    rng = random.Random(seed)
    chosen = []

    for b in range(num_blocks):
        candidates = [
            i for i in range(b * block_size, (b + 1) * block_size)
            if i in available
        ]
        if len(candidates) < targets_per_block:
            raise ValueError(
                f"Block {b} has only {len(candidates)} available targets."
            )
        block_targets = rng.sample(candidates, targets_per_block)
        chosen.extend(sorted(block_targets))

    return chosen


def existing_done_indices(path):
    p = Path(path)
    if not p.exists():
        return set()

    done = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(int(obj["global_index"]))
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------
# One target
# ---------------------------------------------------------------------
def evaluate_target(
    item,
    ds,
    model,
    processor,
    model_path,
    heldout_indices,
    m,
    seed,
):
    gi = int(item["global_index"])
    question = str(item["question"])

    target_ids, reconstruction_text = reconstruct_generated_ids(
        processor.tokenizer, item
    )
    T = int(target_ids.numel())

    # Independent held-out randomization support.
    rng = random.Random(seed + gi * 1000003)
    if m > len(heldout_indices):
        raise ValueError(f"M={m} > held-out pool size {len(heldout_indices)}")
    support = rng.sample(heldout_indices, m)

    # Same null-image / empty-prompt denominator as existing causal metrics.
    null_score = score_condition(
        model=model,
        processor=processor,
        model_path=model_path,
        image=build_null_image(),
        question=EMPTY_PROMPT,
        target_ids=target_ids,
    )

    ref_cum = null_score["cum_log_probs"]
    if len(ref_cum) != T:
        raise RuntimeError(
            f"Null reference length mismatch for target {gi}: {len(ref_cum)} vs {T}"
        )

    # Store log probabilities only; 100 x 64 is tiny and avoids underflow.
    q_cum_matrix = []
    p_token_matrix = []

    t0 = time.perf_counter()

    for rank, j in enumerate(support, start=1):
        row = ds[int(j)]
        held_img = row["image"]
        held_q = str(row["Q"])

        # Randomized question intervention reference:
        # pair held-out image V_j with fixed target question q_i.
        q_score = score_condition(
            model=model,
            processor=processor,
            model_path=model_path,
            image=held_img,
            question=question,
            target_ids=target_ids,
        )
        q_cum_matrix.append(q_score["cum_log_probs"])

        # Randomized prefix intervention reference:
        # use held-out observed pair (V_j, Q_j), while teacher-forcing target prefix.
        p_score = score_condition(
            model=model,
            processor=processor,
            model_path=model_path,
            image=held_img,
            question=held_q,
            target_ids=target_ids,
        )
        p_token_matrix.append(p_score["token_log_probs"])

        del q_score
        del p_score

        if rank % 10 == 0:
            print(
                f"    target {gi}: randomized support {rank}/{m}",
                flush=True
            )

    elapsed = time.perf_counter() - t0

    q_valid = min([T] + [len(x) for x in q_cum_matrix])
    p_valid = min([T] + [len(x) for x in p_token_matrix])
    valid_T = min(T, q_valid, p_valid)

    comparison_group = item.get("drive_vs_observational") or item.get("causal_vs_pmi", {})
    original_steps = {
        int(s["t"]): s
        for s in comparison_group.get("per_step", [])
    }

    def score(step, canonical, legacy):
        if canonical in step:
            return float(step[canonical])
        return float(step[legacy])

    ln2 = math.log(2.0)
    per_step = []

    # Explicit t=0 theoretical origin.
    if 0 in original_steps:
        s0 = original_steps[0]
        per_step.append({
            "t": 0,
            "relative_position": 0.0,
            "Q_randomized_ref": 0.0,
            "P_randomized_ref": 0.0,
            "QCD": score(s0, "QCD_t", "QACI_t"),
            "Q_Obs": score(s0, "Q_Obs_t", "PMI_Q_t"),
            "PCD": score(s0, "PCD_t", "PACI_t"),
            "P_Obs": score(s0, "P_Obs_t", "PMI_P_t"),
            "QCD_abs_error": 0.0,
            "Q_Obs_abs_error": 0.0,
            "PCD_abs_error": 0.0,
            "P_Obs_abs_error": 0.0,
        })

    # Match exactly the t values already reported by the causal experiment.
    for t in sorted(k for k in original_steps.keys() if k > 0):
        idx = t - 1
        if idx >= valid_T:
            continue

        q_log_num = logmeanexp([x[idx] for x in q_cum_matrix])
        p_log_num = logmeanexp([x[idx] for x in p_token_matrix])

        # Q reference matches QCD cumulative granularity. P reference matches
        # PCD's token numerator and cumulative null denominator. Both are
        # computed directly in log space without probability-domain clipping.
        q_ref = (q_log_num - ref_cum[idx]) / ln2
        p_ref = (p_log_num - ref_cum[idx]) / ln2

        s = original_steps[t]
        qaci = score(s, "QCD_t", "QACI_t")
        pmi_q = score(s, "Q_Obs_t", "PMI_Q_t")
        paci = score(s, "PCD_t", "PACI_t")
        pmi_p = score(s, "P_Obs_t", "PMI_P_t")

        per_step.append({
            "t": t,
            "relative_position": float(t / max(valid_T, 1)),
            "Q_randomized_ref": float(q_ref),
            "P_randomized_ref": float(p_ref),

            "QCD": qaci,
            "Q_Obs": pmi_q,
            "PCD": paci,
            "P_Obs": pmi_p,

            "QCD_abs_error": float(abs(qaci - q_ref)),
            "Q_Obs_abs_error": float(abs(pmi_q - q_ref)),
            "PCD_abs_error": float(abs(paci - p_ref)),
            "P_Obs_abs_error": float(abs(pmi_p - p_ref)),
        })

    positive = [x for x in per_step if x["t"] > 0]
    if not positive:
        raise RuntimeError(f"No common positive time steps for target {gi}")

    def mean_key(key):
        return float(np.mean([x[key] for x in positive]))

    q_causal = mean_key("QCD_abs_error")
    q_pmi = mean_key("Q_Obs_abs_error")
    p_causal = mean_key("PCD_abs_error")
    p_pmi = mean_key("P_Obs_abs_error")

    return {
        "global_index": gi,
        "block_id": gi // 20,
        "question": question,
        "num_generated_tokens": T,
        "reconstructed_num_tokens": int(target_ids.numel()),
        "reconstruction_text": reconstruction_text,
        "heldout_support_size": m,
        "heldout_support_indices": support,
        "heldout_pool_min": min(heldout_indices),
        "heldout_pool_max": max(heldout_indices),
        "runtime_seconds": elapsed,

        "errors": {
            "QCD_MAE_to_randomized": q_causal,
            "Q_Obs_MAE_to_randomized": q_pmi,
            "PCD_MAE_to_randomized": p_causal,
            "P_Obs_MAE_to_randomized": p_pmi,

            "Q_error_reduction_fraction": (
                (q_pmi - q_causal) / max(q_pmi, EPS)
            ),
            "P_error_reduction_fraction": (
                (p_pmi - p_causal) / max(p_pmi, EPS)
            ),

            "QCD_wins": bool(q_causal < q_pmi),
            "PCD_wins": bool(p_causal < p_pmi),
        },
        "per_step": per_step,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--merged_jsonl", default=DEFAULT_MERGED)

    parser.add_argument("--heldout_start", type=int, default=200)
    parser.add_argument("--heldout_end", type=int, default=-1,
                        help="inclusive; -1 means dataset end")
    parser.add_argument("--m", type=int, default=100)

    parser.add_argument("--block_size", type=int, default=20)
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--targets_per_block", type=int, default=5)

    parser.add_argument("--seed", type=int, default=20260810)

    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    print("Loading merged causal/PMI results...", flush=True)
    merged = load_merged(args.merged_jsonl)

    print("Loading MAVIS parquet...", flush=True)
    ds = load_dataset(
        "parquet",
        data_files={"test": args.dataset_path},
        split="test",
    )
    n_data = len(ds)

    heldout_end = n_data - 1 if args.heldout_end < 0 else min(args.heldout_end, n_data - 1)
    heldout_indices = list(range(args.heldout_start, heldout_end + 1))

    # Crucial: held-out pool must not overlap the original 0..199 causal-eval pool.
    overlap = sorted(set(heldout_indices) & set(merged.keys()))
    if overlap:
        raise ValueError(
            f"Held-out pool overlaps evaluation targets, first overlaps: {overlap[:10]}"
        )

    targets = choose_targets(
        available_indices=sorted(merged.keys()),
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        targets_per_block=args.targets_per_block,
        seed=args.seed,
    )

    worker_targets = targets[args.worker_id::args.num_workers]

    print("=" * 100, flush=True)
    print(f"Dataset size              : {n_data}", flush=True)
    print(f"Held-out pool             : [{args.heldout_start}, {heldout_end}]", flush=True)
    print(f"Held-out pool size        : {len(heldout_indices)}", flush=True)
    print(f"Randomized support M      : {args.m}", flush=True)
    print(f"All selected targets      : {targets}", flush=True)
    print(f"Worker {args.worker_id}/{args.num_workers} targets: {worker_targets}", flush=True)
    print("=" * 100, flush=True)

    print("Loading model...", flush=True)
    model = load_model(args.model_path)
    processor = get_processor(args.model_path, BASE_MAX_PIXELS)

    print(f"Model input device: {get_model_input_device(model)}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = existing_done_indices(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    with out_path.open(mode, encoding="utf-8") as fout:
        for pos, gi in enumerate(worker_targets, start=1):
            if gi in done:
                print(f"[skip done] target {gi}", flush=True)
                continue

            print("=" * 100, flush=True)
            print(
                f"[worker {args.worker_id}] target {pos}/{len(worker_targets)} "
                f"global_index={gi}",
                flush=True
            )

            try:
                result = evaluate_target(
                    item=merged[gi],
                    ds=ds,
                    model=model,
                    processor=processor,
                    model_path=args.model_path,
                    heldout_indices=heldout_indices,
                    m=args.m,
                    seed=args.seed,
                )
            except Exception as e:
                result = {
                    "global_index": gi,
                    "block_id": gi // args.block_size,
                    "error": f"{type(e).__name__}: {e}",
                }
                print(result["error"], flush=True)

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            if "errors" in result:
                er = result["errors"]
                print(
                    f"Q: QCD MAE={er['QCD_MAE_to_randomized']:.4f} | "
                    f"Q-Obs MAE={er['Q_Obs_MAE_to_randomized']:.4f} | "
                    f"reduction={100*er['Q_error_reduction_fraction']:.1f}%",
                    flush=True
                )
                print(
                    f"P: PCD MAE={er['PCD_MAE_to_randomized']:.4f} | "
                    f"P-Obs MAE={er['P_Obs_MAE_to_randomized']:.4f} | "
                    f"reduction={100*er['P_error_reduction_fraction']:.1f}%",
                    flush=True
                )

            gc.collect()
            # Do not empty CUDA cache inside the support loop.
            # Once per target is acceptable and avoids fragmentation accumulation.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
