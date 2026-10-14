#!/usr/bin/env python3
"""State-independent full-prompt teacher forcing for the established protocol.

The empirical distributions, prompt templates, generated answers, precision,
and three equations are untouched. No past_key_values are shared or passed.
Each complete multimodal prompt recomputes its own rotary position indices.
"""
from __future__ import annotations

import math
import time

BACKEND = "full_prompt_teacher_forcing_no_kv_reuse"
_core = None


def install(core):
    global _core
    _core = core
    core.build_prompt_cache = build_prompt_cache
    core.score_fixed_sequence_from_prompt_cache = score_fixed_sequence_from_prompt_cache


def recover_ids(tokenizer, row):
    pieces = row["generation"]["generated_tokens"]
    ids = tokenizer.encode("".join(pieces), add_special_tokens=False)
    method = "whole_sequence_roundtrip"
    decoded = lambda values: [tokenizer.decode([i], skip_special_tokens=False) for i in values]
    if decoded(ids) != pieces:
        special = set(tokenizer.all_special_tokens)
        suffix = []
        for piece in reversed(pieces):
            if piece not in special:
                break
            suffix.insert(0, piece)
        candidate = tokenizer.encode(row["generation"]["generated_prefixes"][-1],
                                     add_special_tokens=False)
        for piece in suffix:
            candidate.extend(tokenizer.encode(piece, add_special_tokens=False))
        if decoded(candidate) == pieces:
            ids, method = candidate, "intact_prefix_and_special_suffix_roundtrip"
    if decoded(ids) != pieces:
        ids = []
        for piece in pieces:
            current = tokenizer.encode(piece, add_special_tokens=False)
            if len(current) != 1 or tokenizer.decode(current, skip_special_tokens=False) != piece:
                raise ValueError(f"Cannot recover saved token boundary: {row['global_index']}, {piece!r}")
            ids.extend(current)
        method = "individual_token_roundtrip"
    prefixes = [tokenizer.decode(ids[:i + 1], skip_special_tokens=True).strip()
                for i in range(len(ids))]
    if not ids or prefixes != row["generation"]["generated_prefixes"]:
        raise ValueError(f"Saved-prefix recovery failed: {row['global_index']}")
    return ids, method


def build_prompt_cache(model, processor, image, question):
    import torch
    del model
    started = time.perf_counter()
    inputs, _, _ = _core.prepare_single_inputs_with_retry(processor, image, question)
    # Store inputs, not a mutable DynamicCache. No GPU resident KV templates.
    stored = {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
              for key, value in inputs.items()}
    return {"prompt_inputs": stored, "prompt_len": int(stored["input_ids"].shape[1]),
            "elapsed_seconds": time.perf_counter() - started, "scoring_backend": BACKEND}


def score_fixed_sequence_from_prompt_cache(model, prompt_cache, target_generated_ids):
    import torch
    ids = target_generated_ids.reshape(-1)
    count = int(ids.numel())
    if count == 0:
        return {"token_probs": [], "token_log_probs": [], "cum_log_probs": [],
                "continuation_elapsed_seconds": 0.0}
    with torch.inference_mode():
        started = time.perf_counter()
        device = _core.get_model_input_device(model)
        inputs = {key: value.to(device) if torch.is_tensor(value) else value
                  for key, value in prompt_cache["prompt_inputs"].items()}
        prompt_length = int(inputs["input_ids"].shape[1])
        ids = ids.to(device)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], ids[:-1].unsqueeze(0)), dim=1)
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat((mask, mask.new_ones((1, count - 1))), dim=1)
        # These optional processor fields must span the appended text tokens.
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in inputs:
                values = inputs[key]
                inputs[key] = torch.cat((values, values.new_zeros((1, count - 1))), dim=1)
        for key in ("past_key_values", "cache_position", "position_ids"):
            inputs.pop(key, None)
        inputs["use_cache"] = False
        # Qwen3-VL computes new image-specific rope_deltas when past is absent.
        # Clear any previous value explicitly so no global cache state leaks.
        if hasattr(getattr(model, "model", None), "rope_deltas"):
            model.model.rope_deltas = None
        try:
            output = model(**inputs, logits_to_keep=count)
        except TypeError as error:
            if "logits_to_keep" not in str(error):
                raise
            output = model(**inputs)
        logits = output.logits[0, -count:].float()
        if logits.shape[0] != count or inputs["input_ids"].shape[1] != prompt_length + count - 1:
            raise ValueError("Teacher-forcing token alignment failed")
        selected = logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        values = (selected - torch.logsumexp(logits, dim=-1)).cpu().tolist()
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Nonfinite full-prompt likelihood")
        cumulative, running = [], 0.0
        for value in values:
            running += float(value)
            cumulative.append(running)
        elapsed = time.perf_counter() - started
        del output, logits, selected, inputs
    return {"token_probs": [math.exp(float(x)) for x in values],
            "token_log_probs": values, "cum_log_probs": cumulative,
            "continuation_elapsed_seconds": elapsed}
