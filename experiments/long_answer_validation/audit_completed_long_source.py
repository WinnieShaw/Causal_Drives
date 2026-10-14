#!/usr/bin/env python3
"""CPU re-audit of completed runs; preserve raw scores and registered analysis.

Secondary sensitivities are explicit: consistent repeated brace answers and
exact first-line answers; explanation tokens outside a unique answer brace.
Neither sensitivity redefines QCD/VCD/PCD or replaces registered results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "source-response-validation" / "source"))
import analyze_long_source_validation as registered_analysis
from analyze_long_source_validation import long_analysis, mavis_analysis, paired_cluster
from long_source_common import native_coordinates, window_summary
from source_response_common import classify, evaluate, normalize, aliases


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_error(score):
    raw = score["raw_log_probs"]
    rebuilt = native_coordinates(raw["question_members"], raw["joint_members"], raw["visual_only"], raw["null"])
    keys = list(rebuilt[0])
    stored = np.array([[r[k] for k in keys] for r in score["per_step"]])
    expected = np.array([[r[k] for k in keys] for r in rebuilt])
    assert stored.shape == expected.shape
    error = float(np.max(np.abs(stored - expected)))
    assert error <= 1e-10, error
    return error


def cross_platform_validate(score):
    # Linux and macOS libm differ at the last binary digits of log/exp.
    # Registered metrics remain unchanged; only equality tolerance is relaxed.
    trajectory_error(score)
    return len(score["per_step"])


registered_analysis.validate_scores = cross_platform_validate


def equivalent(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for key in a:
            equivalent(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equivalent(x, y)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert np.isclose(a, b, atol=1e-10, rtol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def conservative_answer(text):
    matches = re.findall(r"\{([^{}]+)\}", text)
    if len(matches) == 1:
        return matches[0], "unique_brace"
    if matches and len({normalize(x) for x in matches}) == 1:
        return matches[0], "consistent_repeated_brace"
    if matches:
        return "", "conflicting_braces"
    line = text.splitlines()[0].strip() if text else ""
    return line, "exact_first_line"


def explanation_outside_unique_brace(score, allow_consistent_repetition=False):
    offset = score["offset"]
    tokens = score["tokens"]
    text = "".join(tokens[offset:])
    matches = list(re.finditer(r"\{([^{}]+)\}", text))
    if not matches or (len(matches) != 1 and not (
            allow_consistent_repetition and len({normalize(m.group(1)) for m in matches}) == 1)):
        return None
    base = sum(map(len, tokens[:offset]))
    excluded = [(base + m.start(), base + m.end()) for m in matches]
    positions, cursor = [], 0
    for token in tokens:
        positions.append((cursor, cursor + len(token)))
        cursor += len(token)
    selected = [i for i in score["windows"]["all_content"]
                if all(positions[i][1] <= lo or positions[i][0] >= hi for lo, hi in excluded)]
    return window_summary(score["per_step"], score["raw_log_probs"]["full"], selected)


def audit_run(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (root / "merged.jsonl").open(encoding="utf-8") if x.strip()]
    expected = sorted(r["index"] for r in manifest["records"])
    assert [r["index"] for r in rows] == expected and len(set(expected)) == len(expected)
    configs = []
    for worker in range(8):
        assert json.loads((root / f"worker_{worker}_exit.json").read_text())["exit_code"] == 0
        completed = json.loads((root / f"worker_{worker}_completion.json").read_text())
        assert completed["complete"]
        assert completed["indices"] == [r["index"] for r in manifest["records"][worker::8]]
        shard = [json.loads(x) for x in (root / f"shard_{worker}.jsonl").open() if x.strip()]
        assert [r["index"] for r in shard] == completed["indices"]
        cfg = json.loads((root / f"worker_{worker}_configuration.json").read_text())
        assert cfg["entry_sha256"] == sha(HERE / "run_long_source_validation.py")
        assert cfg["common_sha256"] == sha(HERE / "long_source_common.py")
        assert cfg["manifest_sha256"] == sha(root / "manifest.json")
        assert cfg["dtype"] == "bfloat16" and cfg["attention"] == "sdpa" and not cfg["offload"]
        configs.append(cfg)
        del shard
    task = configs[0]["task"]
    generated_total, score_total, token_total, cap_hits = 0, 0, 0, 0
    maximum_trajectory_error = 0.
    format_counts, missing, sensitivity_labels = {}, {}, []
    all_score_deltas = []
    for row in rows:
        score_fields = ([row["source_scores"]] if task == "mavis" else
                        list(row["question_fixed_answer_scores"].values()) + list(row["prefix_fixed_suffix_scores"].values()))
        for score in score_fields:
            raw = score["raw_log_probs"]
            n = len(score["ids"])
            assert len(score["tokens"]) == len(score["per_step"]) == n
            assert len(raw["question_members"]) == len(raw["joint_members"]) == 20
            assert all(len(x) == n and np.isfinite(x).all()
                       for x in raw["question_members"] + raw["joint_members"] + [raw["visual_only"], raw["null"], raw["full"]])
            maximum_trajectory_error = max(maximum_trajectory_error, trajectory_error(score))
            for name, indices in score["windows"].items():
                equivalent(window_summary(score["per_step"], raw["full"], indices), score["window_summary"][name])
            score_total += 1
            token_total += n
        generations = ([row["generation"]] if task == "mavis" else
                       list(row["question_generated"].values()) + list(row["prefix_generated"].values()))
        for generation in generations:
            assert generation["n_tokens"] == len(generation["ids"])
            assert generation["ended_with_eos"] or generation["hit_length_limit_without_eos"]
            generated_total += 1
            cap_hits += generation["hit_length_limit_without_eos"]
        if task == "mavis":
            continue
        item = {"index": row["index"], "topic": row["topic"]}
        for kind in ("question", "prefix"):
            item[kind] = {}
            for condition, generation in row[kind + "_generated"].items():
                answer, fmt = conservative_answer(generation["prediction"])
                format_counts.setdefault(kind + "_" + condition, Counter())[fmt] += 1
                item[kind][condition] = classify("{" + answer + "}", row["ground_truth"], row["cue_answer"]) if answer else "other"
            field = "question_fixed_answer_scores" if kind == "question" else "prefix_fixed_suffix_scores"
            before, after = row[field]["neutral"], row[field]["cue"]
            assert before["windows"] == after["windows"]
            if kind == "question":
                assert before["ids"] == after["ids"] == row["question_generated"]["neutral"]["ids"]
            else:
                widths = {len(x["ids"]) for x in row["prefix_conditions"].values()}
                assert len(widths) == 1
                for name, score in row[field].items():
                    assert score["ids"] == row["prefix_conditions"][name]["ids"] + row["prefix_generated"]["neutral"]["ids"]
                    assert score["offset"] == len(row["prefix_conditions"][name]["ids"])
            corrected_a, corrected_b = explanation_outside_unique_brace(before), explanation_outside_unique_brace(after)
            original_missing = before["window_summary"]["explanation"] is None
            missing.setdefault(kind, Counter())["original_missing"] += original_missing
            missing[kind]["recovered_before_brace"] += original_missing and corrected_a is not None
            item[kind + "_outside_explanation_delta"] = None if corrected_a is None or corrected_b is None else {
                name: corrected_b[name] - corrected_a[name] for name in ("QCD", "PCD", "full_logprob_bits")}
            repeat_a = explanation_outside_unique_brace(before, allow_consistent_repetition=True)
            repeat_b = explanation_outside_unique_brace(after, allow_consistent_repetition=True)
            item[kind + "_recovered_explanation_delta"] = None if repeat_a is None or repeat_b is None else {
                name: repeat_b[name] - repeat_a[name] for name in ("QCD", "PCD", "full_logprob_bits")}
            if kind == "question":
                for metric in ("QCD", "VCD", "PCD"):
                    all_score_deltas.append((metric, after["window_summary"]["all_content"][metric] - before["window_summary"]["all_content"][metric]))
        sensitivity_labels.append(item)
    stored = json.loads((root / "summary.json").read_text())
    rebuilt, registered_labels = long_analysis(rows) if task == "long" else mavis_analysis(rows)
    equivalent(rebuilt, stored)
    result = {"run": root.name, "n": len(rows), "merged_sha256": sha(root / "merged.jsonl"),
              "completion_and_configuration": "PASS", "registered_summary_reproduced": True,
              "score_sets_checked": score_total, "score_token_rows_checked": token_total,
              "max_trajectory_reconstruction_error": maximum_trajectory_error,
              "cross_platform_absolute_tolerance": 1e-10,
              "generated_answers": generated_total, "cap_hits": cap_hits,
              "registered_summary": rebuilt}
    if task == "long":
        groups = [r["topic"] for r in rows]
        recovery = {}
        for kind in ("question", "prefix"):
            neutral = Counter(r[kind]["neutral"] for r in sensitivity_labels)
            cue = Counter(r[kind]["cue"] for r in sensitivity_labels)
            increments = [int(r[kind]["cue"] == "cue_answer_match") - int(r[kind]["neutral"] == "cue_answer_match") for r in sensitivity_labels]
            recovery[kind] = {"neutral": dict(neutral), "cue": dict(cue), "paired_cue_increment": paired_cluster(increments, groups)}
            # Keep registered anchors/labels; change only location of explanation.
            selected = [i for i, x in enumerate(registered_labels) if x[kind]["neutral"] == "visual_answer_match"
                        and sensitivity_labels[i][kind + "_outside_explanation_delta"] is not None]
            delta = [sensitivity_labels[i][kind + "_outside_explanation_delta"] for i in selected]
            recovery[kind]["outside_brace_explanation_same_registered_anchors"] = evaluate(
                [registered_labels[i][kind]["cue"] == "cue_answer_match" for i in selected],
                {"native_support_loss": [-x["QCD" if kind == "question" else "PCD"] for x in delta],
                 "full_conditional_logprob_loss": [-x["full_logprob_bits"] for x in delta]}, [rows[i]["topic"] for i in selected])
            recovered_anchors = [i for i, x in enumerate(sensitivity_labels) if x[kind]["neutral"] == "visual_answer_match"]
            selected = [i for i in recovered_anchors if sensitivity_labels[i][kind + "_recovered_explanation_delta"] is not None]
            delta = [sensitivity_labels[i][kind + "_recovered_explanation_delta"] for i in selected]
            estimate = evaluate(
                [sensitivity_labels[i][kind]["cue"] == "cue_answer_match" for i in selected],
                {"native_support_loss": [-x["QCD" if kind == "question" else "PCD"] for x in delta],
                 "full_conditional_logprob_loss": [-x["full_logprob_bits"] for x in delta]}, [rows[i]["topic"] for i in selected])
            recovery[kind]["recovered_labels_and_outside_consistent_braces"] = {
                **estimate, "anchors_before_window_filter": len(recovered_anchors),
                "missing_window": len(recovered_anchors) - len(selected),
                "protocol": "Posthoc CPU sensitivity: identical repeated brace answers accepted; explanation outside every consistent brace span."}
            complete = [i for i in selected if all(not rows[i][kind + "_generated"][c]["hit_length_limit_without_eos"]
                                                   for c in ("neutral", "cue"))]
            delta = [sensitivity_labels[i][kind + "_recovered_explanation_delta"] for i in complete]
            recovery[kind]["recovered_alignment_excluding_cap_hits"] = evaluate(
                [sensitivity_labels[i][kind]["cue"] == "cue_answer_match" for i in complete],
                {"native_support_loss": [-x["QCD" if kind == "question" else "PCD"] for x in delta],
                 "full_conditional_logprob_loss": [-x["full_logprob_bits"] for x in delta]}, [rows[i]["topic"] for i in complete])
        result["format_counts"] = {k: dict(v) for k, v in format_counts.items()}
        result["window_audit"] = {k: dict(v) for k, v in missing.items()}
        result["posthoc_sensitivity_not_primary"] = recovery
        result["fixed_question_score_response"] = {
            m: {"mean_cue_minus_neutral_bits": float(np.mean([x for name, x in all_score_deltas if name == m])),
                "max_abs_change_bits": max(abs(x) for name, x in all_score_deltas if name == m)}
            for m in ("QCD", "VCD", "PCD")}
    del rows
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE.parent / "cache" / "completed_runs")
    parser.add_argument("--output", type=Path, default=HERE.parent / "build" / "completed_analysis" / "audit_summary.json")
    args = parser.parse_args()
    results = []
    for root in sorted(args.cache.iterdir()):
        if (root / "summary.json").exists():
            result = audit_run(root)
            results.append(result)
            print(json.dumps({k: result[k] for k in ("run", "n", "registered_summary_reproduced", "score_sets_checked", "score_token_rows_checked")}), flush=True)
    assert len(results) == 4, "Require all four completed runs"
    long_manifests = [json.loads((args.cache / r["run"] / "manifest.json").read_text(encoding="utf-8"))["records"]
                      for r in results if "long_explanation" in r["run"]]
    assert all(x == long_manifests[0] for x in long_manifests), "Long tasks must share target and K20 support identities"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".next.json")
    temporary.write_text(json.dumps({"audit_date": "2026-10-11", "no_gpu_inference": True,
                                    "original_formulas_unchanged": True, "matched_long_task_manifests": True,
                                    "runs": results}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
