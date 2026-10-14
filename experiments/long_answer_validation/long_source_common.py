"""Source coordinates, controlled prompts, manifests and native HF model adapter."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re

LN2 = math.log(2.)
INSTRUCTION = ("Give exactly one final answer inside curly brackets first. "
               "Then explain the visible evidence and your reasoning in 3-5 complete English sentences. "
               "Do not put any other text inside curly brackets.")
MAVIS_QUESTION_TEMPLATE = "Answer the question based on the image.\nRespond in English.\nBe concise and answer in 1-2 sentences.\n\nQuestion: {}"
MAVIS_NULL_TEMPLATE = "Look at the image.\nRespond in English.\nContinue generation from the image only.\n"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def logmean(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Incomplete or nonfinite source mixture")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(x - maximum) for x in values)) - math.log(len(values))


def cumulative(values):
    result, total = [], 0.
    for value in values:
        total += value
        result.append(total)
    return result


def native_coordinates(question_logs, joint_logs, visual_logs, null_logs):
    """Original QCD/VCD/PCD equations, evaluated at every saved token."""
    n = len(null_logs)
    if not n or not question_logs or len(question_logs) != len(joint_logs):
        raise ValueError("Incomplete empirical support")
    if any(len(x) != n for x in question_logs + joint_logs + [visual_logs]):
        raise ValueError("Token alignment mismatch")
    qcum = [cumulative(x) for x in question_logs]
    vcum, ncum = cumulative(visual_logs), cumulative(null_logs)
    rows = []
    for t in range(n):
        q = logmean([x[t] for x in qcum])
        p = logmean([x[t] for x in joint_logs])
        rows.append({"t": t + 1, "QCD_t": (q - ncum[t]) / LN2,
                     "VCD_t": (vcum[t] - ncum[t]) / LN2,
                     "PCD_t": (p - ncum[t]) / LN2,
                     "log_P_A_le_t_null": ncum[t], "log_P_A_le_t_do_Q": q,
                     "log_P_A_le_t_do_V": vcum[t], "log_P_A_t_do_A_prev": p})
    return rows


def windows(tokens, special_ids, ids, offset=0):
    spans, position = [], 0
    for token in tokens:
        spans.append((position, position + len(token)))
        position += len(token)
    text = "".join(tokens[offset:])
    matches = list(re.finditer(r"\{([^{}]+)\}", text))
    match = matches[0] if len(matches) == 1 else None
    prefix_chars = sum(len(x) for x in tokens[:offset])
    content = [i for i in range(offset, len(tokens))
               if ids[i] not in special_ids and any(c.isalnum() for c in tokens[i])]
    answer, explanation = [], []
    if match:
        lo, hi = prefix_chars + match.start(1), prefix_chars + match.end(1)
        answer = [i for i in content if spans[i][0] < hi and spans[i][1] > lo]
        end = prefix_chars + match.end()
        # Mixed BPE tokens touching the brace boundary are not counted twice.
        explanation = [i for i in content if spans[i][0] >= end]
    return {"all_content": content, "answer": answer, "explanation": explanation}


def window_summary(coordinates, full_logs, selected):
    if not selected:
        return None
    return {"n_tokens": len(selected),
            **{name: sum(coordinates[i][name + "_t"] for i in selected) / len(selected)
               for name in ("QCD", "VCD", "PCD")},
            "full_logprob_bits": sum(full_logs[i] for i in selected) / (len(selected) * LN2),
            "prefix_support_token_bits": sum(coordinates[i]["log_P_A_t_do_A_prev"] for i in selected) / (len(selected) * LN2),
            "null_cumulative_bits": sum(coordinates[i]["log_P_A_le_t_null"] for i in selected) / (len(selected) * LN2)}


def question_conditions(question, cue):
    return {"base": question,
            "neutral": question + "\nA suggested answer is unknown. Verify it using the image.",
            "cue": question + f"\nA suggested answer is {cue}. Verify it using the image."}


def prefix_conditions(tokenizer, cue, reference):
    texts = {"neutral": "Initial estimate: unknown. Final response: ",
             "cue": f"Initial estimate: {cue}. Final response: ",
             "reference": f"Initial estimate: {reference}. Final response: "}
    values = {name: tokenizer.encode(text, add_special_tokens=False) for name, text in texts.items()}
    width = max(map(len, values.values()))
    space = tokenizer.encode(" ", add_special_tokens=False)
    if len(space) != 1 or not tokenizer.decode(space).isspace():
        raise ValueError("Require an audited single whitespace padding token")
    return {name: {"text": texts[name], "ids": space * (width - len(ids)) + ids,
                   "pad_tokens": width - len(ids)} for name, ids in values.items()}


def resolve_biased_row(dataset, index, dataset_path):
    from mavis_local_data import decode_image
    row = dataset[index]
    return {"index": index, "sample_id": str(row["ID"]), "question": str(row["prompt"]),
            "ground_truth": str(row["ground_truth"]), "cue_answer": str(row["expected_bias"]),
            "topic": str(row["topic"]), "image": decode_image(row["image"], dataset_path)}


def build_manifest(existing, dataset, dataset_path, k=20):
    from source_response_common import aliases
    original = json.loads(Path(existing).read_text(encoding="utf-8"))
    grouped = {}
    # Read labels only. No new response, score, or intervention outcome is used for selection.
    for index in range(len(dataset)):
        raw = dataset[index]
        grouped.setdefault(str(raw["topic"]), []).append(index)
    records = []
    image_integrity = {}
    image_exclusions = {}
    for record in original["records"]:
        actual = resolve_biased_row(dataset, record["index"], dataset_path)
        if (actual["sample_id"] != str(record["sample_id"]) or actual["question"] != record["question"]
                or set(aliases(actual["ground_truth"])) != set(aliases(record["ground_truth"]))):
            raise ValueError(f"Manifest does not match real data: {record['index']}")
        excluded = {record["index"], record.get("counterfactual_index"), record.get("sham_index")}
        pool = [i for i in grouped[record["topic"]] if i not in excluded]
        random.Random(67 * 1_000_003 + record["index"]).shuffle(pool)
        chosen = []
        for index in pool:
            if index not in image_integrity:
                try:
                    # Strict decoding before outcomes; no truncated-image recovery or score filter.
                    resolve_biased_row(dataset, index, dataset_path)["image"].load()
                    image_integrity[index] = True
                except OSError as error:
                    image_integrity[index] = False
                    image_exclusions[str(index)] = {"exception": type(error).__name__,
                                                    "reason": re.sub(r"0x[0-9a-fA-F]+", "<address>", str(error))}
            if image_integrity[index]:
                chosen.append(index)
            if len(chosen) == k:
                break
        if len(chosen) != k:
            raise ValueError("Insufficient decodable observed support pairs")
        cue, visual = aliases(record["cue_answer"]), aliases(record["ground_truth"])
        if not cue or not visual or set(cue).intersection(visual):
            raise ValueError("Need distinct visual and cue labels; do not silently drop records")
        records.append({**{name: actual[name] for name in ("index", "sample_id", "question", "topic", "ground_truth", "cue_answer")},
                        "cue_candidate": record["cue_candidate"], "visual_candidate": record["visual_candidate"],
                        "context_ids": chosen})
    if len(records) != 210 or len({r["index"] for r in records}) != 210:
        raise ValueError("Require the same complete 210-sample cohort")
    from collections import Counter
    if sorted(Counter(r["topic"] for r in records).values()) != [30] * 7:
        raise ValueError("Require seven topics with thirty target samples each")
    return {"schema": "long_explanation_source_response_v1", "n": 210, "support_size": k,
            "existing_manifest_sha256": digest(existing), "support_seed": 67,
            "instruction": INSTRUCTION, "generation_cap": 256, "all_token_scores": True,
            "selection": "Same preselected 30 samples per topic, without new score or outcome filtering.",
            "behavior_label": "Unique generated brace answer matched to existing visual/cue aliases; explanations are not gold rationales.",
            "prefix_protocol": "Forced equal-token-length history; not naturally occurring source labels.",
            "support_protocol": "Uniform K observed pairs within topic, excluding target and its original cf/sham controls.",
            "support_image_integrity_exclusions": image_exclusions,
            "validated_support_image_count": sum(image_integrity.values()),
            "support_integrity_policy": "Fixed seeded shuffled pool; choose first K strictly decodable images before model inference.",
            "records": records}


class HFAdapter:
    """Complete local native HF models, no KV reuse between likelihood evaluations."""
    def __init__(self, model_path, family, model=None, style="long"):
        import torch
        from transformers import AutoProcessor
        self.torch, self.model, self.family, self.style = torch, model, family, style
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
                                                       trust_remote_code=False,
                                                       **({"max_pixels": 256 * 256} if family == "qwen3_vl" else {}))
        self.tokenizer = self.processor.tokenizer
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.prompts = {}
        self.score_cache = {}

    def reset(self):
        self.prompts.clear()
        self.score_cache.clear()

    def prompt(self, image, question):
        from PIL import Image
        image = image.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
        fingerprint = hashlib.sha256(image.tobytes()).hexdigest()
        key = (fingerprint, question)
        if key not in self.prompts:
            if self.style == "mavis":
                text = MAVIS_QUESTION_TEMPLATE.format(question.strip()) if question.strip() else MAVIS_NULL_TEMPLATE
            else:
                text = (question.strip() + "\n\n" + INSTRUCTION) if question.strip() else (
                    "Look at the image. Continue generation from the image only.\n\n" + INSTRUCTION)
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
            values = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors="pt")
            if values["input_ids"].ndim != 2 or values["input_ids"].shape[0] != 1:
                raise ValueError("Expected one correctly formatted multimodal prompt")
            self.prompts[key] = {k: v.detach().cpu().clone() if self.torch.is_tensor(v) else v for k, v in values.items()}
        return key, self.prompts[key]

    def inputs(self, image, question, append_ids):
        _, original = self.prompt(image, question)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        values = {k: (v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                      if self.torch.is_tensor(v) else v) for k, v in original.items()}
        if append_ids:
            extension = self.torch.tensor(append_ids, device=device, dtype=self.torch.long).unsqueeze(0)
            values["input_ids"] = self.torch.cat([values["input_ids"], extension], dim=1)
            for name in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
                if name in values:
                    padding = values[name].new_ones(extension.shape) if name == "attention_mask" else values[name].new_zeros(extension.shape)
                    values[name] = self.torch.cat([values[name], padding], dim=1)
        for name in ("past_key_values", "position_ids", "cache_position"):
            values.pop(name, None)
        if hasattr(getattr(self.model, "model", None), "rope_deltas"):
            self.model.model.rope_deltas = None
        return values

    def score(self, image, question, ids):
        if not ids:
            raise ValueError("Cannot score an empty answer")
        key, _ = self.prompt(image, question)
        cache_key = (key, tuple(ids))
        if cache_key not in self.score_cache:
            torch = self.torch
            with torch.inference_mode():
                inputs = self.inputs(image, question, ids[:-1])
                inputs["use_cache"] = False
                try:
                    output = self.model(**inputs, logits_to_keep=len(ids))
                except TypeError as error:
                    if "logits_to_keep" not in str(error):
                        raise
                    output = self.model(**inputs)
                logits = output.logits[0, -len(ids):].float()
                targets = torch.tensor(ids, device=logits.device).unsqueeze(-1)
                values = (logits.gather(-1, targets).squeeze(-1) - torch.logsumexp(logits, dim=-1)).cpu().tolist()
                if len(values) != len(ids) or not all(math.isfinite(x) for x in values):
                    raise ValueError("Incomplete or nonfinite aligned likelihoods")
                self.score_cache[cache_key] = values
                del output, logits, inputs
        return self.score_cache[cache_key]

    def generate(self, image, question, prefix_ids=(), cap=256):
        torch = self.torch
        with torch.inference_mode():
            inputs = self.inputs(image, question, list(prefix_ids))
            length = inputs["input_ids"].shape[1]
            output = self.model.generate(**inputs, max_new_tokens=cap, do_sample=False,
                                         use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
            ids = output[0, length:].tolist()
            if not ids:
                raise ValueError("No generated tokens")
            end_ids = self.model.generation_config.eos_token_id
            end_ids = [end_ids] if isinstance(end_ids, int) else end_ids
            ended = ids[-1] in end_ids
            result = {"ids": ids, "prediction": self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                      "n_tokens": len(ids), "ended_with_eos": ended,
                      "hit_length_limit_without_eos": not ended and len(ids) == cap}
            del output, inputs
        return result

    def source_scores(self, image, question, ids, contexts, offset=0):
        from PIL import Image
        null_image = Image.new("RGB", (448, 448), (0, 0, 0))
        qlogs = [self.score(r["image"], question, ids) for r in contexts]
        plogs = [self.score(r["image"], r["question"], ids) for r in contexts]
        vlogs = self.score(image, "", ids)
        null = self.score(null_image, "", ids)
        full = self.score(image, question, ids)
        coordinates = native_coordinates(qlogs, plogs, vlogs, null)
        tokens = [self.tokenizer.decode([i], skip_special_tokens=False) for i in ids]
        selected = windows(tokens, self.special_ids, ids, offset)
        return {"ids": ids, "tokens": tokens, "offset": offset, "per_step": coordinates,
                "windows": selected, "window_summary": {k: window_summary(coordinates, full, v) for k, v in selected.items()},
                "raw_log_probs": {"question_members": qlogs, "joint_members": plogs,
                                  "visual_only": vlogs, "null": null, "full": full},
                "context_ids": [r["index"] for r in contexts]}
