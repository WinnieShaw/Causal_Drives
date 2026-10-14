import os
import argparse
import re
import json
import ast
import math
import string
import time
import gc
from collections import Counter
from typing import Dict, List, Tuple, Any
from PIL import Image

# 把分母也改成了输入空prompt和空图片，而不是求平均
# 调试 CUDA 异步错误时可以打开；正式跑大规模实验可注释掉以减少额外同步开销
# Production run: keep CUDA asynchronous. Set CUDA_LAUNCH_BLOCKING=1 externally only for debugging.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
# 减少显存碎片问题
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import AutoProcessor

# =========================================================
# 0. Config
# =========================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "")
DATASET_PATH = os.environ.get("MAVIS_DATASET_PATH", "")
CACHE_DIR = "."

# 从第0个样本开始
GLOBAL_START_IDX = 0

# 每批固定处理20条
BATCH_SIZE = 20

# None 表示一直自动循环到测试集结尾
GLOBAL_END_IDX = None

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_NEW_TOKENS = 64
EPS = 1e-12  # retained only for empirical weights and reporting compatibility
EMPTY_PROMPT = ""
NULL_IMAGE_SIZE = (448, 448)
NULL_IMAGE_COLOR = (0, 0, 0)

# 更稳，先不要上 flash_attention_2
ATTN_IMPL = "sdpa"
OFFLOAD_PROMPT_CACHE_TO_CPU = True
VERBOSE = False
SYNC_FOR_TIMING = False

# 初始视觉分辨率限制
BASE_MAX_PIXELS = 256 * 256
# 样本异常大时进一步缩小再重试
RETRY_MAX_PIXELS = 144 * 144

# 硬阈值：超过就不直接送进模型，先触发重试/跳过
MAX_VISUAL_TOKENS = 4096000
MAX_INPUT_IDS = 4096000

# 异常样本规整策略
FORCE_IMAGE_SIZE = (448, 448)
MAX_QUESTION_CHARS = 100000

# 默认用 fp16，通常比 bf16 在复杂多卡环境下更稳
MODEL_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# =========================================================
# 1) Load model (keep multi-GPU)
# =========================================================
try:
    from transformers import Qwen3VLForConditionalGeneration

    def load_model(model_path: str):
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No CUDA device found.")

        # 给每张卡留一点余量，避免被塞满
        max_memory = {i: "20GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "120GiB"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=MODEL_DTYPE,
            attn_implementation=ATTN_IMPL,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        return model
except Exception:
    from transformers import AutoModelForImageTextToText

    def load_model(model_path: str):
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No CUDA device found.")

        max_memory = {i: "20GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "120GiB"

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=MODEL_DTYPE,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        return model


# =========================================================
# 2) Utilities
# =========================================================
def safe_parse(x):
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def white_space_fix(text):
        return " ".join(text.split())

    s = remove_articles(s)
    s = remove_punc(s)
    s = white_space_fix(s)
    return s


def exact_match_score(pred: str, ref: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(ref))


def contains_match_score(pred: str, ref: str) -> int:
    p = normalize_answer(pred)
    r = normalize_answer(ref)
    if not p or not r:
        return 0
    return int((r in p) or (p in r))


def token_f1_score(pred: str, ref: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def rouge_l_f1_score(pred: str, ref: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def maybe_compute_bertscore(predictions, references):
    if os.environ.get("SKIP_BERTSCORE", "0") == "1":
        return {
            "bertscore_f1_mean": None,
            "bertscore_f1_per_sample": None,
            "bertscore_error": "skipped for causal-drive recomputation",
        }
    try:
        import evaluate
        bertscore = evaluate.load("bertscore")
        result = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
        )
        avg_f1 = sum(result["f1"]) / len(result["f1"]) if result["f1"] else None
        return {
            "bertscore_f1_mean": avg_f1,
            "bertscore_f1_per_sample": result["f1"],
        }
    except Exception as e:
        return {
            "bertscore_f1_mean": None,
            "bertscore_f1_per_sample": None,
            "bertscore_error": f"{type(e).__name__}: {e}",
        }


def maybe_cuda_sync():
    # Synchronization is useful for precise timing/debugging but slows this
    # evaluation substantially. It does not affect metric values.
    if SYNC_FOR_TIMING and torch.cuda.is_available():
        torch.cuda.synchronize()


def hard_cleanup(*named_objs):
    for obj in named_objs:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def recursive_detach_clone(obj: Any):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(recursive_detach_clone(x) for x in obj)
    if isinstance(obj, list):
        return [recursive_detach_clone(x) for x in obj]
    if isinstance(obj, dict):
        return {k: recursive_detach_clone(v) for k, v in obj.items()}
    return obj


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
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def keep_inputs_on_cpu(batch: Dict[str, Any]):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.contiguous()
        else:
            out[k] = v
    return out


def normalize_image(image):
    if not hasattr(image, "size"):
        return image, False
    target_w, target_h = FORCE_IMAGE_SIZE
    try:
        size = tuple(image.size)
    except Exception:
        return image, False
    if size == (target_w, target_h):
        return image, False
    try:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        image = image.resize((target_w, target_h), Image.BICUBIC)
        return image, True
    except Exception:
        return image, False


def normalize_question(question: str):
    q = str(question)
    truncated = False
    if len(q) > MAX_QUESTION_CHARS:
        q = q[:MAX_QUESTION_CHARS]
        truncated = True
    return q, truncated


def build_null_image(size=NULL_IMAGE_SIZE, color=NULL_IMAGE_COLOR):
    return Image.new("RGB", size, color=color)


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


def build_chat_text(processor, image, question: str) -> str:
    messages = build_messages(image, question)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def inspect_inputs(inputs: Dict[str, Any], tag: str = "input"):
    if not VERBOSE:
        return
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"    {tag}[{k}] shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")


def prepare_single_inputs(processor, image, question: str):
    image, image_resized = normalize_image(image)
    question, question_truncated = normalize_question(question)

    if VERBOSE and image_resized:
        print(f"[INFO] resized image to {FORCE_IMAGE_SIZE}")
    if VERBOSE and question_truncated:
        print(f"[INFO] truncated question chars to {MAX_QUESTION_CHARS}")

    text = build_chat_text(processor, image, question)
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

    inspect_inputs(inputs, tag="input")

    if "pixel_values" in inputs and torch.is_tensor(inputs["pixel_values"]):
        visual_tokens = int(inputs["pixel_values"].shape[0])
        if visual_tokens > MAX_VISUAL_TOKENS:
            raise RuntimeError(f"visual tokens too large: {visual_tokens} > {MAX_VISUAL_TOKENS}")

    if "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
        text_len = int(inputs["input_ids"].shape[1])
        if text_len > MAX_INPUT_IDS:
            raise RuntimeError(f"input_ids too large: {text_len} > {MAX_INPUT_IDS}")

    return keep_inputs_on_cpu(inputs)


def get_processor(max_pixels: int):
    return AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        max_pixels=max_pixels,
    )


def prepare_single_inputs_with_retry(processor, image, question: str):
    try:
        inputs = prepare_single_inputs(processor, image, question)
        return inputs, processor, False
    except RuntimeError as e:
        msg = str(e)
        if "visual tokens too large" in msg or "input_ids too large" in msg:
            print(f"[WARN] oversized sample detected, retry with smaller max_pixels: {msg}")
            small_processor = get_processor(RETRY_MAX_PIXELS)
            inputs = prepare_single_inputs(small_processor, image, question)
            return inputs, small_processor, True
        raise


# =========================================================
# 3) Generation
# =========================================================
@torch.inference_mode()
def generate_sequence(model, processor, image, question: str, max_new_tokens: int):
    inputs, used_processor, _ = prepare_single_inputs_with_retry(processor, image, question)

    # Move model inputs to the device where the input embedding layer lives.
    input_device = get_model_input_device(model)
    inputs = recursive_to_device(inputs, str(input_device))

    maybe_cuda_sync()
    t0 = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=used_processor.tokenizer.eos_token_id,
    )

    maybe_cuda_sync()
    elapsed = time.perf_counter() - t0

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0][input_len:]
    generated_text = used_processor.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    step_token_probs = []
    step_tokens = []
    prefixes = []

    actual_steps = min(len(outputs.scores), generated_ids.shape[0])

    for step_idx in range(actual_steps):
        score_t = outputs.scores[step_idx]
        step_logits = score_t[0].float().detach().cpu()
        token_id = generated_ids[step_idx].item()

        log_denom = torch.logsumexp(step_logits, dim=-1)
        p = torch.exp(step_logits[token_id] - log_denom).item()
        p = max(p, EPS)

        step_token_probs.append(p)
        step_tokens.append(
            used_processor.decode([token_id], skip_special_tokens=False)
        )
        prefixes.append(
            used_processor.decode(generated_ids[: step_idx + 1], skip_special_tokens=True).strip()
        )

    del outputs
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "generated_ids": generated_ids.detach().cpu(),
        "generated_text": generated_text,
        "step_token_probs": step_token_probs,
        "step_tokens": step_tokens,
        "prefixes": prefixes,
        "elapsed_seconds": elapsed,
    }


# =========================================================
# 4) Prompt KV Cache
# =========================================================
@torch.inference_mode()
def build_prompt_cache(model, processor, image, question: str):
    inputs, _, _ = prepare_single_inputs_with_retry(processor, image, question)

    # Move prompt inputs to the model input device before forward.
    input_device = get_model_input_device(model)
    inputs = recursive_to_device(inputs, str(input_device))

    maybe_cuda_sync()
    t0 = time.perf_counter()

    forward_kwargs = dict(inputs)
    forward_kwargs["use_cache"] = True

    try:
        outputs = model(**forward_kwargs, logits_to_keep=1)
    except TypeError:
        try:
            outputs = model(**forward_kwargs, num_logits_to_keep=1)
        except TypeError:
            outputs = model(**forward_kwargs)

    maybe_cuda_sync()
    elapsed = time.perf_counter() - t0

    prompt_last_logits = outputs.logits[0, -1].float().detach().cpu()
    past_key_values = recursive_detach_clone(outputs.past_key_values)

    if OFFLOAD_PROMPT_CACHE_TO_CPU:
        past_key_values = recursive_to_device(past_key_values, "cpu")

    cache = {
        "prompt_last_logits": prompt_last_logits,
        "past_key_values": past_key_values,
        "prompt_len": int(inputs["input_ids"].shape[1]),
        "elapsed_seconds": elapsed,
    }

    del outputs
    # Do not gc.collect()/empty_cache here: this function is called O(n^2)
    # times and forced allocator flushes dominate runtime. Batch-level cleanup
    # is performed later and metric values are unchanged.
    return cache


# =========================================================
# 5) Score fixed sequence from prompt cache
# =========================================================
@torch.inference_mode()
def score_fixed_sequence_from_prompt_cache(
    model,
    prompt_cache: Dict[str, Any],
    target_generated_ids: torch.Tensor,
):
    if target_generated_ids.numel() == 0:
        return {
            "token_probs": [],
            "token_log_probs": [],
            "cum_log_probs": [],
            "continuation_elapsed_seconds": 0.0,
        }

    target_ids = target_generated_ids.to(get_model_input_device(model))
    T = target_ids.shape[0]

    token_probs = []
    token_log_probs = []
    cum_log_probs = []
    running = 0.0

    first_logits = prompt_cache["prompt_last_logits"]
    token0 = target_generated_ids[0].item()

    log_denom = torch.logsumexp(first_logits, dim=-1).item()
    logp0 = (first_logits[token0].item() - log_denom)
    p0 = math.exp(logp0)

    token_probs.append(p0)
    token_log_probs.append(logp0)
    running += logp0
    cum_log_probs.append(running)

    continuation_elapsed = 0.0
    if T > 1:
        past_key_values = prompt_cache["past_key_values"]
        if OFFLOAD_PROMPT_CACHE_TO_CPU:
            past_key_values = recursive_to_device(past_key_values, str(get_model_input_device(model)))

        continuation_input_ids = target_ids[:-1].unsqueeze(0)

        maybe_cuda_sync()
        t0 = time.perf_counter()

        try:
            outputs = model(
                input_ids=continuation_input_ids,
                past_key_values=past_key_values,
                use_cache=False,
                logits_to_keep=T - 1,
            )
        except TypeError:
            try:
                outputs = model(
                    input_ids=continuation_input_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    num_logits_to_keep=T - 1,
                )
            except TypeError:
                outputs = model(
                    input_ids=continuation_input_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                )

        maybe_cuda_sync()
        continuation_elapsed = time.perf_counter() - t0

        continuation_logits = outputs.logits[0].float().detach().cpu()
        usable_steps = min(T - 1, continuation_logits.shape[0])

        for t in range(1, usable_steps + 1):
            step_logits = continuation_logits[t - 1]
            token_id = target_generated_ids[t].item()

            log_denom = torch.logsumexp(step_logits, dim=-1).item()
            logp = step_logits[token_id].item() - log_denom
            p = math.exp(logp)

            token_probs.append(p)
            token_log_probs.append(logp)
            running += logp
            cum_log_probs.append(running)

        del continuation_logits
        del outputs
        del continuation_input_ids
        del target_ids
        if OFFLOAD_PROMPT_CACHE_TO_CPU:
            del past_key_values

        # Avoid gc.collect()/empty_cache after every support score.
        # These scores are called many times per target and forced allocator
        # flushes make the H200 spend most of its time waiting.

    return {
        "token_probs": token_probs,
        "token_log_probs": token_log_probs,
        "cum_log_probs": cum_log_probs,
        "continuation_elapsed_seconds": continuation_elapsed,
    }


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def log_weighted_mean(log_values, weights):
    """Return log(sum_i w_i exp(x_i) / sum_i w_i) without leaving log space."""
    terms = [math.log(w) + x for x, w in zip(log_values, weights) if w > 0.0]
    positive_weights = [w for w in weights if w > 0.0]
    if not terms or not positive_weights:
        raise ValueError("log_weighted_mean requires at least one positive weight")
    anchor = max(terms)
    log_numerator = anchor + math.log(sum(math.exp(x - anchor) for x in terms))
    return log_numerator - math.log(sum(positive_weights))


def run_numerical_self_test():
    values = [-100.0, -500.0, -1000.0]
    got = log_weighted_mean(values, [1.0, 1.0, 1.0])
    expected = values[0] + math.log(
        sum(math.exp(x - values[0]) for x in values) / len(values)
    )
    if not math.isfinite(got) or abs(got - expected) > 1e-12:
        raise AssertionError(f"log-domain mixture failed: got={got}, expected={expected}")
    ratio_bits = (got - (-1000.0)) / math.log(2.0)
    if not math.isfinite(ratio_bits) or ratio_bits <= 0.0:
        raise AssertionError(f"log-ratio failed: {ratio_bits}")
    print(json.dumps({
        "status": "ok",
        "log_weighted_mean": got,
        "log_ratio_bits": ratio_bits,
        "tested_log_probabilities": values,
    }, indent=2))


# =========================================================
# 6) Process one batch: causal drives + reference-calibrated observational baselines
# =========================================================
def process_one_batch(model, processor, subset, batch_start_global, batch_end_global):
    """
    Empirical-distribution implementation.

    We treat the n observed rows in the current evaluation batch as the
    empirical joint support D = {(v_i, q_i)}_{i=1}^n, each with mass 1/n.

    This changes NO causal estimand/formula. It only replaces the previous
    artificial uniform P(q|v)=1/n over all cross-pairs by the empirical
    distribution induced by the actually observed MAVIS pairings.

    Canonical causal-drive definitions:
      QCD_t = log2 P(A_<=t | do(Q=q_i)) / P_ref(A_<=t)
      VCD_t = log2 P(A_<=t | V=v_i, Q=null) / P_ref(A_<=t)
      PCD_t = log2 P(A_t | do(A_<t=a_<t)) / P_ref(A_<=t)

    Reference-calibrated observational counterparts, aligned to the same
    denominator and target-event granularity:
      Q_Obs_t = log2 sum_v P(v|q_i) P(A_<=t|v,q_i) / P_ref(A_<=t)
      P_Obs_t = log2 sum_vq P(v,q|A_<t) P(A_t|A_<t,v,q) / P_ref(A_<=t)

    Notes:
      * At t=0 all metrics are exactly 0.
      * At the first generated token (t=1), A_<1 is empty, so P_Obs == PCD.
      * We intentionally keep the PCD denominator exactly as in the supplied
        causal code (cumulative null-reference probability), so the comparison
        does not modify the original causal metric.
    """
    n = len(subset)

    output_jsonl = os.path.join(
        OUTPUT_DIR,
        f"mavis_drive_scores_{batch_start_global}-{batch_end_global}.jsonl"
    )
    output_summary = os.path.join(
        OUTPUT_DIR,
        f"mavis_drive_scores_summary_{batch_start_global}-{batch_end_global}.json"
    )

    print("=" * 100)
    print(f"Using samples [{batch_start_global}, {batch_end_global}], total = {n}")

    all_images = {i: subset[i]["image"] for i in range(n)}
    all_questions = {i: str(subset[i]["Q"]) for i in range(n)}

    # ---------------------------------------------------------
    # Empirical distribution from ACTUAL observed pairings.
    # Each row is one observed joint sample (v_i, q_i), mass 1/n.
    # ---------------------------------------------------------
    Pv = {i: 1.0 / n for i in range(n)}

    # Empirical joint support contains only observed pairs (v_i, q_i).
    P_joint = {(i, i): 1.0 / n for i in range(n)}

    # With image instance v_i represented by row i, its observed paired
    # question is q_i. Thus empirical P(q_i | v_i)=1.
    Pq_given_v = {i: {i: 1.0} for i in range(n)}

    # Exact empirical P(v | q): if the same question string occurs several
    # times, distribute mass uniformly over the observed rows carrying it.
    question_to_v_indices: Dict[str, List[int]] = {}
    for i in range(n):
        q_text = all_questions[i]
        question_to_v_indices.setdefault(q_text, []).append(i)

    Pv_given_q: Dict[int, Dict[int, float]] = {}
    for qi in range(n):
        matching_v = question_to_v_indices[all_questions[qi]]
        mass = 1.0 / len(matching_v)
        Pv_given_q[qi] = {vi: mass for vi in matching_v}

    print("Empirical distribution built from observed MAVIS pairings.")
    print(f"  observed joint pairs: {len(P_joint)}")
    print(f"  unique question strings: {len(question_to_v_indices)} / {n}")

    # ---------------------------------------------------------
    # Prompt caches.
    # QCD needs cross-pairs (v_j, q_i), so all n^2 prompt caches are built.
    # PCD / P-Obs use observed pairs. VCD uses the target image with a null question.
    # ---------------------------------------------------------
    print("=" * 100)
    print("Building cross-pair prompt caches required by QCD ...")
    prompt_cache_dict: Dict[Tuple[int, int], Dict[str, Any]] = {}
    visual_only_prompt_cache_dict: Dict[int, Dict[str, Any]] = {}
    null_image_prompt_cache: Dict[str, Any] = {}
    prompt_cache_total_seconds = 0.0
    skipped_prompt_pairs = []
    skipped_null_image_prompt = None

    for vi in range(n):
        for qi in range(n):
            if VERBOSE:
                print(f"Building prompt cache for (v={vi}, q={qi})")
            try:
                cache = build_prompt_cache(
                    model=model,
                    processor=processor,
                    image=all_images[vi],
                    question=all_questions[qi],
                )
                prompt_cache_dict[(vi, qi)] = cache
                prompt_cache_total_seconds += cache["elapsed_seconds"]
            except Exception as e:
                print(f"[SKIP] prompt cache failed at (v={vi}, q={qi}): {type(e).__name__}: {e}")
                skipped_prompt_pairs.append({
                    "v": vi,
                    "q": qi,
                    "error": f"{type(e).__name__}: {e}",
                })

        try:
            cache = build_prompt_cache(
                model=model,
                processor=processor,
                image=all_images[vi],
                question=EMPTY_PROMPT,
            )
            visual_only_prompt_cache_dict[vi] = cache
            prompt_cache_total_seconds += cache["elapsed_seconds"]
        except Exception as e:
            print(f"[SKIP] visual-only prompt cache failed at v={vi}: {type(e).__name__}: {e}")

    print("=" * 100)
    print("Building null-image + empty-prompt denominator cache ...")
    try:
        null_image = build_null_image()
        null_image_prompt_cache = build_prompt_cache(
            model=model,
            processor=processor,
            image=null_image,
            question=EMPTY_PROMPT,
        )
        prompt_cache_total_seconds += null_image_prompt_cache["elapsed_seconds"]
        print("Null-image denominator cache built successfully.")
    except Exception as e:
        print(f"[SKIP] null-image prompt cache failed: {type(e).__name__}: {e}")
        skipped_null_image_prompt = f"{type(e).__name__}: {e}"

    print(f"Prompt cache build done. Total prompt cache time: {prompt_cache_total_seconds:.2f} s")
    print(f"Valid cross-pair prompt caches: {len(prompt_cache_dict)} / {n*n}")
    print(f"Valid visual-only prompt caches: {len(visual_only_prompt_cache_dict)} / {n}")
    print(f"Null-image denominator cache ready: {bool(null_image_prompt_cache)}")

    all_results = []
    all_predictions = []
    all_references = []

    total_generate_seconds = 0.0
    total_score_seconds = 0.0

    for i in range(n):
        sample = subset[i]
        global_index = batch_start_global + i
        target_img_idx = i
        target_q_idx = i

        question = str(sample["Q"])
        image = sample["image"]

        facts = safe_parse(sample["verifiable_facts"])
        pos_document_ids = safe_parse(sample["pos_document_ids"])
        meta_info = safe_parse(sample["meta"])

        reference_answer = str(facts[0]) if isinstance(facts, list) and len(facts) > 0 else ""

        print("=" * 100)
        print(f"[sample {i+1}/{n}] [global index {global_index}]")
        print("Q:", question)

        try:
            target_info = generate_sequence(
                model=model,
                processor=processor,
                image=image,
                question=question,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        except Exception as e:
            print(f"[SKIP] generate failed on sample global={global_index}: {type(e).__name__}: {e}")
            item = {
                "index": i,
                "global_index": global_index,
                "question": question,
                "prediction": "",
                "reference_answer": reference_answer,
                "all_verifiable_facts": facts,
                "pos_document_ids": pos_document_ids,
                "meta": meta_info,
                "generation": {
                    "generated_tokens": [],
                    "generated_prefixes": [],
                    "num_generated_tokens": 0,
                    "generate_seconds": 0.0,
                    "score_seconds": 0.0,
                    "error": f"{type(e).__name__}: {e}",
                },
                "drive_metrics": {
                    "avg_qcd": None,
                    "avg_vcd": None,
                    "avg_pcd": None,
                    "per_step": [],
                },
                "observational_baselines": {
                    "avg_q_obs": None,
                    "avg_p_obs": None,
                    "per_step": [],
                },
                "text_metrics": {
                    "exact_match": 0,
                    "contains_match": 0,
                    "token_f1": 0.0,
                    "rouge_l_f1": 0.0,
                },
            }
            all_results.append(item)
            all_predictions.append("")
            all_references.append(reference_answer)
            continue

        generate_elapsed = target_info["elapsed_seconds"]
        total_generate_seconds += generate_elapsed

        target_generated_ids = target_info["generated_ids"]
        target_text = target_info["generated_text"]
        target_step_tokens = target_info["step_tokens"]
        target_prefixes = target_info["prefixes"]
        T = len(target_generated_ids)

        print(f"  generate done in {generate_elapsed:.2f}s, generated {T} tokens")

        if T == 0:
            print(f"[WARN] sample {i} generated 0 tokens, skip drive computation.")
            item = {
                "index": i,
                "global_index": global_index,
                "question": question,
                "prediction": target_text,
                "reference_answer": reference_answer,
                "all_verifiable_facts": facts,
                "pos_document_ids": pos_document_ids,
                "meta": meta_info,
                "generation": {
                    "generated_tokens": target_step_tokens,
                    "generated_prefixes": target_prefixes,
                    "num_generated_tokens": T,
                    "generate_seconds": generate_elapsed,
                    "score_seconds": 0.0,
                },
                "drive_metrics": {
                    "avg_qcd": None,
                    "avg_vcd": None,
                    "avg_pcd": None,
                    "per_step": [],
                },
                "observational_baselines": {
                    "avg_q_obs": None,
                    "avg_p_obs": None,
                    "per_step": [],
                },
                "text_metrics": {
                    "exact_match": exact_match_score(target_text, reference_answer),
                    "contains_match": contains_match_score(target_text, reference_answer),
                    "token_f1": token_f1_score(target_text, reference_answer),
                    "rouge_l_f1": rouge_l_f1_score(target_text, reference_answer),
                },
            }
            all_results.append(item)
            all_predictions.append(target_text)
            all_references.append(reference_answer)
            hard_cleanup(target_generated_ids, target_info, target_step_tokens, target_prefixes)
            continue

        # -----------------------------------------------------
        # IMPORTANT SPEED FIX:
        # Old code rescored the target sequence under ALL n^2 cross-pairs.
        # For target q_i, QCD only needs {(v_j, q_i)}_j.
        # PCD/P-Obs use observed {(v_j, q_j)}_j; VCD uses (v_i, null question).
        # Therefore only <= 2n-1 prompt conditions are rescored per target.
        # -----------------------------------------------------
        needed_keys = set((vi, target_q_idx) for vi in range(n))
        needed_keys.update((j, j) for j in range(n))

        scored: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
        null_image_with_prefix: Dict[str, List[float]] = {}
        visual_only_with_prefix: Dict[str, List[float]] = {}
        sample_score_seconds = 0.0

        for vi, qi in sorted(needed_keys):
            if (vi, qi) not in prompt_cache_dict:
                continue
            if VERBOSE:
                print(f"  scoring needed cache (v={vi}, q={qi})")

            prompt_cache = prompt_cache_dict[(vi, qi)]
            score_result = score_fixed_sequence_from_prompt_cache(
                model=model,
                prompt_cache=prompt_cache,
                target_generated_ids=target_generated_ids,
            )
            sample_score_seconds += score_result["continuation_elapsed_seconds"]
            scored[(vi, qi)] = {
                "token_probs": score_result["token_probs"],
                "token_log_probs": score_result["token_log_probs"],
                "cum_log_probs": score_result["cum_log_probs"],
            }

        if target_img_idx in visual_only_prompt_cache_dict:
            score_result = score_fixed_sequence_from_prompt_cache(
                model=model,
                prompt_cache=visual_only_prompt_cache_dict[target_img_idx],
                target_generated_ids=target_generated_ids,
            )
            sample_score_seconds += score_result["continuation_elapsed_seconds"]
            visual_only_with_prefix = {
                "token_probs": score_result["token_probs"],
                "token_log_probs": score_result["token_log_probs"],
                "cum_log_probs": score_result["cum_log_probs"],
            }

        if null_image_prompt_cache:
            if VERBOSE:
                print("  scoring null-image denominator")
            score_result = score_fixed_sequence_from_prompt_cache(
                model=model,
                prompt_cache=null_image_prompt_cache,
                target_generated_ids=target_generated_ids,
            )
            sample_score_seconds += score_result["continuation_elapsed_seconds"]
            null_image_with_prefix = {
                "token_probs": score_result["token_probs"],
                "token_log_probs": score_result["token_log_probs"],
                "cum_log_probs": score_result["cum_log_probs"],
            }

        total_score_seconds += sample_score_seconds

        # Required sets under current target.
        qcd_keys = [(vi, target_q_idx) for vi in range(n) if (vi, target_q_idx) in scored]
        observed_keys = [(j, j) for j in range(n) if (j, j) in scored]
        q_obs_keys = [
            (vi, target_q_idx)
            for vi in Pv_given_q[target_q_idx]
            if (vi, target_q_idx) in scored
        ]
        length_candidates = []
        for key in set(qcd_keys + observed_keys + q_obs_keys):
            length_candidates.append(len(scored[key]["token_probs"]))
            length_candidates.append(len(scored[key]["cum_log_probs"]))
        if visual_only_with_prefix:
            length_candidates.append(len(visual_only_with_prefix["token_probs"]))
            length_candidates.append(len(visual_only_with_prefix["cum_log_probs"]))
        if null_image_with_prefix:
            length_candidates.append(len(null_image_with_prefix["token_probs"]))
            length_candidates.append(len(null_image_with_prefix["cum_log_probs"]))

        if (
            qcd_keys
            and observed_keys
            and q_obs_keys
            and visual_only_with_prefix
            and null_image_with_prefix
            and length_candidates
        ):
            valid_steps = min(T, min(length_candidates), len(target_prefixes), len(target_step_tokens))
        else:
            valid_steps = 0

        causal_rows = []
        observational_rows = []
        combined_rows = []

        qcd_curve = []
        vcd_curve = []
        pcd_curve = []
        q_obs_curve = []
        p_obs_curve = []

        # Explicit theoretical t=0 origin.
        # Empty generated sequence has probability 1 under every condition,
        # therefore every log-ratio is exactly zero.
        combined_rows.append({
            "t": 0,
            "prefix": "",
            "token": "",
            "QCD_t": 0.0,
            "VCD_t": 0.0,
            "PCD_t": 0.0,
            "Q_Obs_t": 0.0,
            "P_Obs_t": 0.0,
            "QCD_minus_Q_Obs": 0.0,
            "PCD_minus_P_Obs": 0.0,
        })

        # Keep the original causal code's temporal stride: t = 1,3,5,...
        for t in range(1, valid_steps + 1, 2):
            idx = t - 1

            # -------------------------------------------------
            # SAME denominator as the supplied causal code.
            # -------------------------------------------------
            logp_uncond = null_image_with_prefix["cum_log_probs"][idx]
            p_prefix_uncond = math.exp(logp_uncond)

            # =================================================
            # Q causal: empirical P(v) backdoor adjustment
            # =================================================
            q_do_logs = []
            q_do_weights = []
            for vi, qi in qcd_keys:
                logp = scored[(vi, qi)]["cum_log_probs"][idx]
                w = Pv[vi]
                q_do_logs.append(logp)
                q_do_weights.append(w)
            logp_prefix_do_q = log_weighted_mean(q_do_logs, q_do_weights)
            p_prefix_do_q = math.exp(logp_prefix_do_q)

            # =================================================
            # Q-Obs: empirical P(v | q_i)
            # derived ONLY from actual MAVIS pairings.
            # =================================================
            q_obs_logs = []
            q_obs_weights = []
            for vi, qi in q_obs_keys:
                logp = scored[(vi, qi)]["cum_log_probs"][idx]
                w = Pv_given_q[target_q_idx][vi]
                q_obs_logs.append(logp)
                q_obs_weights.append(w)
            logp_prefix_obs_q = log_weighted_mean(q_obs_logs, q_obs_weights)
            p_prefix_obs_q = math.exp(logp_prefix_obs_q)

            # =================================================
            # VCD: target image with the question replaced by the null prompt.
            # =================================================
            logp_prefix_do_v = visual_only_with_prefix["cum_log_probs"][idx]
            p_prefix_do_v = math.exp(logp_prefix_do_v)

            # =================================================
            # PCD: empirical joint prior P(v,q).
            # IMPORTANT: same formula/denominator style as supplied code.
            # =================================================
            p_do_logs = []
            p_do_weights = []
            for vi, qi in observed_keys:
                logp_token = scored[(vi, qi)]["token_log_probs"][idx]
                w = P_joint[(vi, qi)]
                p_do_logs.append(logp_token)
                p_do_weights.append(w)
            logp_do_prev_state = log_weighted_mean(p_do_logs, p_do_weights)
            p_do_prev_state = math.exp(logp_do_prev_state)

            # =================================================
            # P-Obs:
            #   P(v,q | A_<t) ∝ P(A_<t | v,q) P_D(v,q)
            # where P_D is the actual empirical observed-pair distribution.
            # Use log-domain normalization for numerical stability.
            # =================================================
            posterior_log_weights = []
            for vi, qi in observed_keys:
                prior_w = P_joint[(vi, qi)]
                log_prior = math.log(max(prior_w, EPS))

                if idx == 0:
                    # t=1 => A_<1 is empty, likelihood is exactly 1.
                    log_prefix_likelihood = 0.0
                else:
                    # A_<t corresponds to generated tokens 1..t-1.
                    log_prefix_likelihood = scored[(vi, qi)]["cum_log_probs"][idx - 1]

                posterior_log_weights.append(
                    (vi, qi, log_prior + log_prefix_likelihood)
                )

            posterior_den_anchor = max(x[2] for x in posterior_log_weights)
            log_posterior_den = posterior_den_anchor + math.log(sum(
                math.exp(logw - posterior_den_anchor)
                for _, _, logw in posterior_log_weights
            ))
            posterior_num_terms = [
                logw + scored[(vi, qi)]["token_log_probs"][idx]
                for vi, qi, logw in posterior_log_weights
            ]
            posterior_num_anchor = max(posterior_num_terms)
            log_posterior_num = posterior_num_anchor + math.log(sum(
                math.exp(x - posterior_num_anchor) for x in posterior_num_terms
            ))
            logp_token_obs_prefix = log_posterior_num - log_posterior_den
            p_token_obs_prefix = math.exp(logp_token_obs_prefix)

            # -------------------------------------------------
            # Final log-ratios.
            # -------------------------------------------------
            ln2 = math.log(2.0)
            qcd_t = (logp_prefix_do_q - logp_uncond) / ln2
            vcd_t = (logp_prefix_do_v - logp_uncond) / ln2
            pcd_t = (logp_do_prev_state - logp_uncond) / ln2

            q_obs_t = (logp_prefix_obs_q - logp_uncond) / ln2
            p_obs_t = (logp_token_obs_prefix - logp_uncond) / ln2

            qcd_curve.append(qcd_t)
            vcd_curve.append(vcd_t)
            pcd_curve.append(pcd_t)
            q_obs_curve.append(q_obs_t)
            p_obs_curve.append(p_obs_t)

            causal_rows.append({
                "t": t,
                "prefix": target_prefixes[idx],
                "token": target_step_tokens[idx],
                "P_A_t": p_prefix_uncond,
                "P_A_t_do_Q": p_prefix_do_q,
                "P_A_t_do_V": p_prefix_do_v,
                "P_A_t_do_A_prev": p_do_prev_state,
                "log_P_A_le_t_null": logp_uncond,
                "log_P_A_le_t_do_Q": logp_prefix_do_q,
                "log_P_A_le_t_do_V": logp_prefix_do_v,
                "log_P_A_t_do_A_prev": logp_do_prev_state,
                "QCD_t": qcd_t,
                "VCD_t": vcd_t,
                "PCD_t": pcd_t,
            })

            observational_rows.append({
                "t": t,
                "prefix": target_prefixes[idx],
                "token": target_step_tokens[idx],
                "P_A_t": p_prefix_uncond,
                "P_A_t_given_Q_obs": p_prefix_obs_q,
                "P_A_t_given_prefix_obs": p_token_obs_prefix,
                "log_P_A_le_t_given_Q_obs": logp_prefix_obs_q,
                "log_P_A_t_given_prefix_obs": logp_token_obs_prefix,
                "Q_Obs_t": q_obs_t,
                "P_Obs_t": p_obs_t,
            })

            combined_rows.append({
                "t": t,
                "prefix": target_prefixes[idx],
                "token": target_step_tokens[idx],
                "QCD_t": qcd_t,
                "VCD_t": vcd_t,
                "PCD_t": pcd_t,
                "Q_Obs_t": q_obs_t,
                "P_Obs_t": p_obs_t,
                "QCD_minus_Q_Obs": qcd_t - q_obs_t,
                "PCD_minus_P_Obs": pcd_t - p_obs_t,
            })

        avg_qcd = sum(qcd_curve) / len(qcd_curve) if qcd_curve else None
        avg_vcd = sum(vcd_curve) / len(vcd_curve) if vcd_curve else None
        avg_pcd = sum(pcd_curve) / len(pcd_curve) if pcd_curve else None
        avg_q_obs = sum(q_obs_curve) / len(q_obs_curve) if q_obs_curve else None
        avg_p_obs = sum(p_obs_curve) / len(p_obs_curve) if p_obs_curve else None

        # Sanity check implied by the prefix definitions at the first token.
        first_step_pcd_p_obs_gap = (
            abs(pcd_curve[0] - p_obs_curve[0])
            if pcd_curve and p_obs_curve else None
        )

        em = exact_match_score(target_text, reference_answer)
        cm = contains_match_score(target_text, reference_answer)
        f1 = token_f1_score(target_text, reference_answer)
        rouge_l = rouge_l_f1_score(target_text, reference_answer)

        all_predictions.append(target_text)
        all_references.append(reference_answer)

        item = {
            "index": i,
            "global_index": global_index,
            "question": question,
            "prediction": target_text,
            "reference_answer": reference_answer,
            "all_verifiable_facts": facts,
            "pos_document_ids": pos_document_ids,
            "meta": meta_info,
            "generation": {
                "generated_tokens": target_step_tokens,
                "generated_prefixes": target_prefixes,
                "num_generated_tokens": T,
                "generate_seconds": generate_elapsed,
                "score_seconds": sample_score_seconds,
            },
            "drive_metrics": {
                "avg_qcd": avg_qcd,
                "avg_vcd": avg_vcd,
                "avg_pcd": avg_pcd,
                "per_step": causal_rows,
            },
            "observational_baselines": {
                "avg_q_obs": avg_q_obs,
                "avg_p_obs": avg_p_obs,
                "per_step": observational_rows,
            },
            "drive_vs_observational": {
                "per_step": combined_rows,
                "sanity": {
                    "t0_all_zero": True,
                    "first_generated_step_abs_pcd_minus_p_obs": first_step_pcd_p_obs_gap,
                },
            },
            "schema": {
                "version": 2,
                "canonical_names": ["QCD", "VCD", "PCD", "Q-Obs", "P-Obs"],
                "vcd_intervention": "target_image_with_null_question",
                "aggregation": "weighted_logsumexp_without_probability_floor",
            },
            "empirical_distribution": {
                "joint_support_size": len(P_joint),
                "question_support_size_for_target": len(Pv_given_q[target_q_idx]),
                "target_question_matching_visual_indices": list(Pv_given_q[target_q_idx].keys()),
                "description": (
                    "Empirical distribution over actually observed MAVIS row pairings; "
                    "no artificial cross-pair mass is assigned to P_D(v,q)."
                ),
            },
            "text_metrics": {
                "exact_match": em,
                "contains_match": cm,
                "token_f1": f1,
                "rouge_l_f1": rouge_l,
            },
        }

        all_results.append(item)

        print(f"PRED: {target_text}")
        print(f"REF : {reference_answer}")
        if avg_qcd is not None:
            print(
                f"avg_qcd={avg_qcd:.4f} | avg_vcd={avg_vcd:.4f} | avg_pcd={avg_pcd:.4f} | "
                f"avg_q_obs={avg_q_obs:.4f} | avg_p_obs={avg_p_obs:.4f}"
            )
            print(
                f"sanity: first-step |PCD-P-Obs|={first_step_pcd_p_obs_gap:.6e}"
            )
        else:
            print("causal-drive/observational metrics unavailable")
        print(f"EM={em} | CONTAINS={cm} | F1={f1:.4f} | ROUGE-L={rouge_l:.4f}")

        hard_cleanup(
            scored,
            null_image_with_prefix,
            target_generated_ids,
            target_info,
            target_step_tokens,
            target_prefixes,
            causal_rows,
            observational_rows,
            combined_rows,
            qcd_curve,
            vcd_curve,
            pcd_curve,
            q_obs_curve,
            p_obs_curve,
        )

    bertscore_result = maybe_compute_bertscore(all_predictions, all_references)
    per_sample_bs = bertscore_result.get("bertscore_f1_per_sample")
    if per_sample_bs is not None:
        for item, bs in zip(all_results, per_sample_bs):
            item["text_metrics"]["bertscore_f1"] = bs

    valid_causal_items = [x for x in all_results if x["drive_metrics"]["avg_qcd"] is not None]
    valid_observational_items = [x for x in all_results if x["observational_baselines"]["avg_q_obs"] is not None]
    n_eff = len(all_results)
    n_causal = len(valid_causal_items)
    n_observational = len(valid_observational_items)

    def mean_metric(items, group, key):
        return (
            sum(x[group][key] for x in items) / len(items)
            if items else None
        )

    avg_qcd_all = mean_metric(valid_causal_items, "drive_metrics", "avg_qcd")
    avg_vcd_all = mean_metric(valid_causal_items, "drive_metrics", "avg_vcd")
    avg_pcd_all = mean_metric(valid_causal_items, "drive_metrics", "avg_pcd")
    avg_q_obs_all = mean_metric(valid_observational_items, "observational_baselines", "avg_q_obs")
    avg_p_obs_all = mean_metric(valid_observational_items, "observational_baselines", "avg_p_obs")

    em_acc = sum(x["text_metrics"]["exact_match"] for x in all_results) / max(n_eff, 1)
    contains_acc = sum(x["text_metrics"]["contains_match"] for x in all_results) / max(n_eff, 1)
    avg_f1 = sum(x["text_metrics"]["token_f1"] for x in all_results) / max(n_eff, 1)
    avg_rouge_l = sum(x["text_metrics"]["rouge_l_f1"] for x in all_results) / max(n_eff, 1)

    sanity_p_first = [
        x["drive_vs_observational"]["sanity"]["first_generated_step_abs_pcd_minus_p_obs"]
        for x in all_results
        if "drive_vs_observational" in x
        and x["drive_vs_observational"]["sanity"]["first_generated_step_abs_pcd_minus_p_obs"] is not None
    ]

    summary = {
        "model_path": MODEL_PATH,
        "dataset_path": DATASET_PATH,
        "batch_start_idx": batch_start_global,
        "batch_end_idx": batch_end_global,
        "num_samples": n_eff,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "attn_implementation": ATTN_IMPL,
        "offload_prompt_cache_to_cpu": OFFLOAD_PROMPT_CACHE_TO_CPU,
        "base_max_pixels": BASE_MAX_PIXELS,
        "retry_max_pixels": RETRY_MAX_PIXELS,
        "force_image_size": list(FORCE_IMAGE_SIZE),
        "model_dtype": str(MODEL_DTYPE),
        "empty_prompt": EMPTY_PROMPT,
        "probability_aggregation": "weighted_logsumexp_without_probability_floor",
        "null_image_size": list(NULL_IMAGE_SIZE),
        "null_image_color": list(NULL_IMAGE_COLOR),
        "empirical_distribution": {
            "joint_support": "actual observed MAVIS row pairings only",
            "joint_support_size": len(P_joint),
            "unique_question_strings": len(question_to_v_indices),
            "Pv": "empirical over observed image instances",
            "Pq_given_v": "empirical from actual row pairing",
            "Pv_given_q": "empirical from exact observed question matches",
            "no_artificial_uniform_cross_pair_joint": True,
        },
        "skipped_prompt_pairs": skipped_prompt_pairs,
        "skipped_null_image_prompt": skipped_null_image_prompt,
        "drive_metrics": {
            "num_valid": n_causal,
            "avg_qcd": avg_qcd_all,
            "avg_vcd": avg_vcd_all,
            "avg_pcd": avg_pcd_all,
            "vcd_intervention": "target_image_with_null_question",
        },
        "observational_baselines": {
            "num_valid": n_observational,
            "avg_q_obs": avg_q_obs_all,
            "avg_p_obs": avg_p_obs_all,
        },
        "sanity": {
            "max_over_samples_first_step_abs_pcd_minus_p_obs": max(sanity_p_first) if sanity_p_first else None,
            "expected": {
                "t0_all_metrics_zero": True,
                "p_obs_equals_pcd_at_first_generated_token": True,
            },
        },
        "text_metrics": {
            "exact_match_accuracy": em_acc,
            "contains_match_accuracy": contains_acc,
            "avg_token_f1": avg_f1,
            "avg_rouge_l_f1": avg_rouge_l,
            "avg_bertscore_f1": bertscore_result.get("bertscore_f1_mean"),
        },
        "runtime": {
            "prompt_cache_total_seconds": prompt_cache_total_seconds,
            "generate_total_seconds": total_generate_seconds,
            "score_total_seconds": total_score_seconds,
            "total_seconds": prompt_cache_total_seconds + total_generate_seconds + total_score_seconds,
        },
        "output_jsonl": output_jsonl,
        "output_summary": output_summary,
    }

    if "bertscore_error" in bertscore_result:
        summary["bertscore_error"] = bertscore_result["bertscore_error"]

    save_jsonl(output_jsonl, all_results)
    save_json(output_summary, summary)

    print("\n" + "#" * 100)
    print(f"BATCH SUMMARY [{batch_start_global}, {batch_end_global}]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved per-sample results to: {output_jsonl}")
    print(f"Saved summary to: {output_summary}")

    hard_cleanup(
        prompt_cache_dict,
        visual_only_prompt_cache_dict,
        null_image_prompt_cache,
        all_results,
        all_predictions,
        all_references,
        all_images,
        all_questions,
        subset,
    )

    return summary


# =========================================================
# 7) Main: auto loop by 20 samples
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MAVIS causal-drive evaluation with log-domain mixture aggregation."
    )
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--dataset_path", default=DATASET_PATH)
    parser.add_argument("--start", type=int, default=GLOBAL_START_IDX)
    parser.add_argument("--end", type=int, default=-1,
                        help="Inclusive; -1 means dataset end.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Empirical support size. Keep 20 for the established protocol.")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def main():
    global MODEL_PATH, DATASET_PATH, BATCH_SIZE, OUTPUT_DIR
    args = parse_args()
    if args.self_test:
        run_numerical_self_test()
        return
    if args.num_workers < 1 or not 0 <= args.worker_id < args.num_workers:
        raise ValueError("Require num_workers >= 1 and 0 <= worker_id < num_workers")
    MODEL_PATH = args.model_path
    DATASET_PATH = args.dataset_path
    BATCH_SIZE = args.batch_size
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    ds = load_dataset(
        "parquet",
        data_files={"test": DATASET_PATH},
        split="test",
        cache_dir=CACHE_DIR,
    )
    total_len = len(ds)

    start = max(0, args.start)
    end = total_len - 1 if args.end < 0 else min(args.end, total_len - 1)

    if start > end:
        raise ValueError(
            f"Invalid sample range: start={args.start}, "
            f"end={args.end}, dataset size={total_len}"
        )

    print(f"Dataset size: {total_len}")
    print(f"Auto processing range: [{start}, {end}]")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Worker: {args.worker_id}/{args.num_workers}")

    print("Loading model...")
    model = load_model(MODEL_PATH)
    model.eval()

    processor = get_processor(BASE_MAX_PIXELS)

    print(f"Using dtype: {MODEL_DTYPE}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"BASE_MAX_PIXELS={BASE_MAX_PIXELS}, RETRY_MAX_PIXELS={RETRY_MAX_PIXELS}")
    print(f"MAX_VISUAL_TOKENS={MAX_VISUAL_TOKENS}, MAX_INPUT_IDS={MAX_INPUT_IDS}")
    print(f"FORCE_IMAGE_SIZE={FORCE_IMAGE_SIZE}, MAX_QUESTION_CHARS={MAX_QUESTION_CHARS}")

    all_batch_summaries = []
    batch_id = 0

    all_batch_starts = list(range(start, end + 1, BATCH_SIZE))
    worker_batch_starts = all_batch_starts[args.worker_id::args.num_workers]

    for batch_start in worker_batch_starts:
        batch_end = min(batch_start + BATCH_SIZE - 1, end)

        print("\n" + "=" * 120)
        print(f"Starting batch #{batch_id}: global [{batch_start}, {batch_end}]")

        subset = [ds[i] for i in range(batch_start, batch_end + 1)]

        try:
            summary = process_one_batch(
                model=model,
                processor=processor,
                subset=subset,
                batch_start_global=batch_start,
                batch_end_global=batch_end,
            )
            all_batch_summaries.append(summary)

        except Exception as e:
            print(f"[ERROR] batch #{batch_id} failed: {type(e).__name__}: {e}")

        finally:
            print(f"\nCleaning caches after batch #{batch_id} ...")

            try:
                del subset
            except Exception:
                pass

            gc.collect()

            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

        batch_id += 1

    merged_summary_path = os.path.join(
        OUTPUT_DIR,
        f"mavis_drive_scores_worker{args.worker_id}_summary_{start}-{end}.json"
    )
    save_json(merged_summary_path, {
        "global_start_idx": start,
        "global_end_idx": end,
        "batch_size": BATCH_SIZE,
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "num_batches": len(all_batch_summaries),
        "batch_summaries": all_batch_summaries,
    })

    print("\n" + "=" * 120)
    print("ALL BATCHES FINISHED")
    print(f"Merged summary saved to: {merged_summary_path}")


if __name__ == "__main__":
    main()
