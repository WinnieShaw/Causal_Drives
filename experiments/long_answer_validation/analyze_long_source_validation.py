"""Reconstruct saved scores and analyze paired behavior without selecting effect signs."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from long_source_common import cumulative, native_coordinates
from source_response_common import classify, evaluate


def paired_cluster(values, groups, seed=42, rounds=5000):
    values, groups = np.asarray(values, float), np.asarray(groups)
    unique, codes = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), (rounds, len(unique)))
    counts = np.zeros((rounds, len(unique)), dtype=float)
    np.add.at(counts, (np.repeat(np.arange(rounds), len(unique)), draws.ravel()), 1)
    weights = counts[:, codes]
    boot = (weights * values).sum(axis=1) / weights.sum(axis=1)
    return {"n": len(values), "groups": len(unique), "mean": float(values.mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "bootstrap": "paired topic-cluster; 5000 draws"}


def validate_scores(score):
    raw = score["raw_log_probs"]
    reconstructed = native_coordinates(raw["question_members"], raw["joint_members"], raw["visual_only"], raw["null"])
    if reconstructed != score["per_step"]:
        raise ValueError("Stored trajectory differs from raw-member reconstruction")
    return len(reconstructed)


def long_analysis(rows):
    labels, lengths, scores = [], [], {}
    maximum_error = 0.
    for row in rows:
        out = {"index": row["index"], "topic": row["topic"]}
        for kind in ("question", "prefix"):
            generated = row[kind + "_generated"]
            out[kind] = {name: classify(item["prediction"], row["ground_truth"], row["cue_answer"])
                         for name, item in generated.items()}
            for name, item in generated.items():
                lengths.append({"kind": kind, "condition": name, "n_tokens": item["n_tokens"],
                                "hit_cap": item["hit_length_limit_without_eos"]})
            field = "question_fixed_answer_scores" if kind == "question" else "prefix_fixed_suffix_scores"
            for score in row[field].values():
                validate_scores(score)
            before, after = row[field]["neutral"], row[field]["cue"]
            out[kind + "_score_delta"] = {}
            for window in ("all_content", "answer", "explanation"):
                a, b = before["window_summary"][window], after["window_summary"][window]
                if a is None or b is None:
                    out[kind + "_score_delta"][window] = None
                    continue
                delta = {name: b[name] - a[name] for name in
                         ("QCD", "VCD", "PCD", "full_logprob_bits", "prefix_support_token_bits", "null_cumulative_bits")}
                # PCD = one-token mixture support minus cumulative reference, exactly as defined.
                maximum_error = max(maximum_error, abs(delta["PCD"] -
                                                       (delta["prefix_support_token_bits"] - delta["null_cumulative_bits"])))
                out[kind + "_score_delta"][window] = delta
        labels.append(out)
    groups = [r["topic"] for r in rows]
    effects = {}
    for kind in ("question", "prefix"):
        for target, condition in (("cue_answer_match", "cue"), ("visual_answer_match", "cue")):
            values = [int(r[kind][condition] == target) - int(r[kind]["neutral"] == target) for r in labels]
            effects[f"{kind}_{condition}_{target}_increment"] = paired_cluster(values, groups)
    values = [int(r["prefix"]["reference"] == "visual_answer_match") - int(r["prefix"]["neutral"] == "visual_answer_match") for r in labels]
    effects["prefix_reference_visual_answer_match_increment"] = paired_cluster(values, groups)
    base_values = [int(r["question"]["neutral"] == "cue_answer_match") - int(r["question"]["base"] == "cue_answer_match") for r in labels]
    effects["question_neutral_vs_base_cue_increment"] = paired_cluster(base_values, groups)
    discrimination = {}
    for kind, metric in (("question", "QCD"), ("prefix", "PCD")):
        anchor = [r for r in labels if r[kind]["neutral"] == "visual_answer_match"]
        for window in ("all_content", "answer", "explanation"):
            valid = [r for r in anchor if r[kind + "_score_delta"][window] is not None]
            y = [r[kind]["cue"] == "cue_answer_match" for r in valid]
            score_map = {
                "native_support_loss": [-r[kind + "_score_delta"][window][metric] for r in valid],
                "full_conditional_logprob_loss": [-r[kind + "_score_delta"][window]["full_logprob_bits"] for r in valid]}
            estimate = evaluate(y, score_map, [r["topic"] for r in valid])
            discrimination[kind + "_" + window] = {**estimate, "anchors_before_window_filter": len(anchor),
                                                     "missing_window": len(anchor) - len(valid),
                                                     "score_direction": "neutral minus cue, fixed before observing results",
                                                     "primary": window == "explanation"}
    completion = {}
    for kind in ("question", "prefix"):
        conditions = ("base", "neutral", "cue") if kind == "question" else ("neutral", "cue", "reference")
        for condition in conditions:
            subset = [x for x in lengths if x["kind"] == kind and x["condition"] == condition]
            completion[kind + "_" + condition] = {"n": len(subset), "mean_tokens": float(np.mean([x["n_tokens"] for x in subset])),
                                                   "median_tokens": float(np.median([x["n_tokens"] for x in subset])),
                                                   "answers_at_least_32_tokens": sum(x["n_tokens"] >= 32 for x in subset),
                                                   "hit_cap_without_eos": sum(x["hit_cap"] for x in subset),
                                                   "outcome_counts": dict(Counter(r[kind][condition] for r in labels))}
    return {"n": len(rows), "topic_counts": dict(Counter(groups)), "completion": completion,
            "behavior_effects": effects, "source_behavior_alignment": discrimination,
            "max_pcd_decomposition_error_bits": maximum_error,
            "scope": "Controlled long explanations; existing visual/cue labels apply to the brace answer, not rationale correctness.",
            "multiplicity": "Nominal intervals; explanation window preselected as primary; other windows diagnostic.",
            "no_outcome_filtering": True}, labels


def mavis_analysis(rows):
    details = []
    for row in rows:
        score = row["source_scores"]
        validate_scores(score)
        selected = score["windows"]["all_content"]
        if not selected:
            raise ValueError("Empty content window; retain raw data and do not summarize a reduced cohort")
        probes = row["visual_probe_raw_log_probs"]
        target_v = cumulative(probes["target"]["visual_only"])
        target_full = probes["target"]["full"]
        result = {"index": row["index"], "domain": row["domain"]}
        for name, raw in probes.items():
            v = cumulative(raw["visual_only"])
            result[name] = {"vcd_drop_bits": sum(target_v[i] - v[i] for i in selected) / (len(selected) * math.log(2.)),
                            "full_local_logprob_drop_bits_per_token": sum(target_full[i] - raw["full"][i] for i in selected) / (len(selected) * math.log(2.))}
        details.append(result)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(rows), size=(5000, len(rows)))
    effects = {}
    for name in ("blur_8", "blur_27", "mismatch_42", "mismatch_137"):
        values = np.array([r[name]["vcd_drop_bits"] for r in details])
        effects[name] = {"mean": float(values.mean()), "ci95": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
                         "positive_fraction": float((values > 0).mean())}
    values = np.array([r["blur_27"]["vcd_drop_bits"] - r["blur_8"]["vcd_drop_bits"] for r in details])
    effects["strong_minus_mild"] = {"mean": float(values.mean()), "ci95": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
                                    "positive_fraction": float((values > 0).mean())}
    return {"n": len(rows), "visual_effects": effects,
            "mean_generated_tokens": float(np.mean([r["generation"]["n_tokens"] for r in rows])),
            "hit_cap_without_eos": sum(r["generation"]["hit_length_limit_without_eos"] for r in rows),
            "native_mean_bits": {m: float(np.mean([r["source_scores"]["window_summary"]["all_content"][m] for r in rows]))
                                 for m in ("QCD", "VCD", "PCD")},
            "scope": "200 matched MAVIS IDs; model-specific answers and native visual preprocessing."}, details


def analyze(rows, task, output_dir):
    summary, details = long_analysis(rows) if task == "long" else mavis_analysis(rows)
    root = Path(output_dir)
    with (root / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (root / "per_sample_analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(details, stream, ensure_ascii=False, allow_nan=False)
