"""CPU analyses of complete answers, target-excluded supports and long windows.

Read immutable experiment records. Keep every model, window and topic result;
secondary analyses retain their own labels and never change the drive equations.
"""
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
LONG = WORK / "experiments/long-answer-source-validation"
sys.path.insert(0, str(LONG / "source"))
sys.path.insert(0, str(WORK / "experiments/source-response-validation/source"))
from source_response_common import classify, evaluate, normalize
from long_source_common import native_coordinates, window_summary
from analyze_long_source_validation import paired_cluster

LN2 = np.log(2.)
METRICS = ("QCD", "VCD", "PCD")
OUT = HERE / "complete_evidence"


def ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=float)
    lo = 0
    while lo < len(x):
        hi = lo + 1
        while hi < len(x) and x[order[hi]] == x[order[lo]]:
            hi += 1
        result[order[lo:hi]] = (lo + hi - 1) / 2
        lo = hi
    return result


def rho(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def lse(x):
    maximum = np.max(x, axis=0)
    return maximum + np.log(np.exp(x - maximum).sum(axis=0))


def cluster_mean(values, blocks, rounds=10000):
    values, blocks = np.asarray(values, float), np.asarray(blocks)
    unique, codes = np.unique(blocks, return_inverse=True)
    counts = np.bincount(codes)
    sums = np.bincount(codes, weights=values)
    draws = np.random.default_rng(42).integers(0, len(unique), (rounds, len(unique)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "groups": len(unique)}


def complete_answers():
    result, curve_records = {}, []
    for model in ("8b", "32b"):
        cache = HERE / "cache/mavis_complete" / model
        rows = sorted([json.loads(line) for p in cache.glob("mavis_drive_scores_[0-9]*.jsonl")
                       for line in p.open(encoding="utf-8") if line.strip()], key=lambda x: x["global_index"])
        assert [r["global_index"] for r in rows] == list(range(500))
        for worker in range(8):
            cfg = json.loads((cache / f"worker_{worker}_configuration.json").read_text(encoding="utf-8"))
            done = json.loads((cache / f"worker_{worker}_completion.json").read_text(encoding="utf-8"))
            assert cfg["max_new_tokens"] == 256 and cfg["support_size"] == 20
            assert not cfg["model_cpu_offload"] and not cfg["prompt_kv_cache_reuse"]
            assert done["all_batches_validated"]
        summary = json.loads((cache / "generation_completion_summary.json").read_text(encoding="utf-8"))
        lengths, ended = [], []
        averages, deltas, slopes = {m: [] for m in METRICS}, {m: [] for m in METRICS}, {m: [] for m in METRICS}
        binned = {m: np.full((500, 10), np.nan) for m in METRICS}
        max_identity_error = 0.
        for i, row in enumerate(rows):
            gen, steps = row["generation"], row["drive_metrics"]["per_step"]
            length = gen["num_generated_tokens"]
            assert length == len(gen["generated_token_ids"]) == len(gen["generated_tokens"])
            assert row["schema"]["scoring_backend"] == "full_prompt_teacher_forcing_no_kv_reuse"
            t = np.array([s["t"] for s in steps])
            assert np.array_equal(t, np.arange(1, length + 1, 2))
            lengths.append(length); ended.append(gen["ended_with_eos"])
            bins = np.minimum((t / length * 10).astype(int), 9)
            early, late = t / length <= .5, t / length > .5
            assert early.any() and late.any()
            for metric, numerator in zip(METRICS, ("log_P_A_le_t_do_Q", "log_P_A_le_t_do_V", "log_P_A_t_do_A_prev")):
                values = np.array([s[metric + "_t"] for s in steps])
                rebuilt = np.array([(s[numerator] - s["log_P_A_le_t_null"]) / LN2 for s in steps])
                assert np.isfinite(values).all()
                max_identity_error = max(max_identity_error, float(abs(values - rebuilt).max()))
                assert max_identity_error < 1e-10
                averages[metric].append(float(values.mean()))
                assert abs(values.mean() - row["drive_metrics"]["avg_" + metric.lower()]) < 1e-10
                deltas[metric].append(float(values[late].mean() - values[early].mean()))
                slopes[metric].append(float(np.polyfit(t, values, 1)[0]))
                for b in range(10):
                    if (bins == b).any():
                        binned[metric][i, b] = values[bins == b].mean()
        lengths, ended, blocks = np.array(lengths), np.array(ended), np.arange(500) // 20
        assert lengths.mean() == summary["mean_length"] and int(ended.sum()) == summary["ended_with_eos"]
        entry = {"n": 500, "mean_length": float(lengths.mean()), "median_length": float(np.median(lengths)),
                 "ended_with_eos": int(ended.sum()), "cap_hits": int((~ended).sum()),
                 "max_formula_error": max_identity_error, "metrics": {}, "length_bands": [],
                 "timing": {key: {"mean_seconds": float(np.mean([r["generation"][key] for r in rows])),
                                   "median_seconds": float(np.median([r["generation"][key] for r in rows]))}
                            for key in ("generate_seconds", "score_seconds")}}
        for metric in METRICS:
            a, d, s = (np.array(x[metric]) for x in (averages, deltas, slopes))
            entry["metrics"][metric] = {"mean_score": cluster_mean(a, blocks), "half_delta": cluster_mean(d, blocks),
                "ols_slope": cluster_mean(s, blocks), "eos_only_mean": cluster_mean(a[ended], blocks[ended]),
                "eos_only_half_delta": cluster_mean(d[ended], blocks[ended]), "length_spearman": rho(lengths, a)}
            for b in range(10):
                selected = np.isfinite(binned[metric][:, b])
                estimate = cluster_mean(binned[metric][selected, b], blocks[selected])
                curve_records.append({"model": model, "metric": metric, "relative_bin": b + 1,
                                      "relative_position": (b + .5) / 10, "n_answers": int(selected.sum()),
                                      "mean_bits": estimate["mean"], "ci95_low": estimate["ci95"][0], "ci95_high": estimate["ci95"][1]})
        for lo, hi in ((1, 24), (25, 48), (49, 80), (81, 256)):
            sel = (lengths >= lo) & (lengths <= hi)
            entry["length_bands"].append({"range": [lo, hi], "n": int(sel.sum()),
                "half_delta": {m: cluster_mean(np.array(deltas[m])[sel], blocks[sel]) for m in METRICS}})
        result[model] = entry
    with (OUT / "temporal_source_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve_records[0])); writer.writeheader(); writer.writerows(curve_records)
    return result


def target_excluded_stability():
    result = {}
    for model in ("8b", "32b"):
        rows = [json.loads(line) for line in (HERE / "cache/support_stability" / model / "support_stability_merged.jsonl").open(encoding="utf-8") if line.strip()]
        assert len(rows) == 100 and len({r["global_index"] for r in rows}) == 100
        values = {m: {k: [] for k in (4, 9, 19, 39)} for m in ("QCD", "PCD")}
        for row in rows:
            n = np.cumsum(row["null_token_log_probs"]); pos = np.arange(0, len(n), 2)
            own = {m: {k: [] for k in (4, 9, 19, 39)} for m in values}
            assert [d["seed"] for d in row["draws"]] == [42, 137, 2026]
            for draw in row["draws"]:
                assert draw["support_indices"][0] == row["global_index"]
                assert row["global_index"] not in draw["support_indices"][1:]
                arrays = {"QCD": np.cumsum(draw["question_member_token_log_probs"], axis=1),
                          "PCD": np.array(draw["joint_member_token_log_probs"])}
                for metric, array in arrays.items():
                    for k in own[metric]:
                        trajectory = (lse(array[1:k+1]) - np.log(k) - n) / LN2
                        own[metric][k].append(float(trajectory[pos].mean()))
            for metric in values:
                for k in values[metric]: values[metric][k].append(own[metric][k])
        result[model] = {"n": len(rows), "metrics": {}}
        for metric in values:
            reference = np.array(values[metric][39])
            result[model]["metrics"][metric] = {}
            for k, scores in values[metric].items():
                a = np.array(scores)
                result[model]["metrics"][metric][str(k)] = {
                    "mae_to_nested_39": float(abs(a-reference).mean()),
                    "mean_draw_sd": float(a.std(axis=1, ddof=1).mean()),
                    "rho_to_39_by_draw": [rho(a[:,j], reference[:,j]) for j in range(3)],
                    "rho_between_draws": [rho(a[:,i], a[:,j]) for i,j in ((0,1),(0,2),(1,2))]}
    return result


def parse_answer(text):
    matches = re.findall(r"\{([^{}]+)\}", text)
    if matches and len({normalize(x) for x in matches}) == 1:
        return matches[0]
    if matches: return ""
    return text.splitlines()[0].strip() if text else ""


def unified_windows(score):
    tokens, offset = score["tokens"], score["offset"]
    text = "".join(tokens[offset:]); base = sum(map(len, tokens[:offset]))
    matches = list(re.finditer(r"\{([^{}]+)\}", text))
    if matches and len({normalize(m.group(1)) for m in matches}) == 1:
        spans = [(base+m.start(), base+m.end()) for m in matches]
    elif not matches and text:
        line = text.splitlines()[0]
        spans = [(base, base + len(line))]
    else:
        return {"all_content": score["windows"]["all_content"], "answer": [], "explanation": []}
    content = score["windows"]["all_content"]
    positions, cursor = [], 0
    for token in tokens:
        positions.append((cursor, cursor + len(token))); cursor += len(token)
    answer = [i for i in content if any(positions[i][0] < hi and positions[i][1] > lo for lo,hi in spans)]
    explanation = [i for i in content if all(positions[i][1] <= lo or positions[i][0] >= hi for lo,hi in spans)]
    assert not set(answer) & set(explanation)
    return {"all_content": content, "answer": answer, "explanation": explanation}


def long_windows():
    results, records = {}, []
    for model, name in (("8b", "qwen3vl8b"), ("32b", "qwen3vl32b"), ("internvl3", "internvl3_8b")):
        path = LONG / "cache/completed_runs" / f"{name}_long_explanation_qp_k20_cap256_210/merged.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        assert len(rows) == 210 and len({r["index"] for r in rows}) == 210
        assert set(Counter(r["topic"] for r in rows).values()) == {30}
        models = {"question": [], "prefix": []}; error = 0.
        for row in rows:
            for kind, field, metric in (("question", "question_fixed_answer_scores", "QCD"), ("prefix", "prefix_fixed_suffix_scores", "PCD")):
                before, after = row[field]["neutral"], row[field]["cue"]
                assert before["windows"] == after["windows"]
                for score in (before,after):
                    raw = score["raw_log_probs"]
                    rebuilt = native_coordinates(raw["question_members"], raw["joint_members"], raw["visual_only"], raw["null"])
                    keys = list(rebuilt[0])
                    error = max(error, float(abs(np.array([[r[k] for k in keys] for r in rebuilt])-np.array([[r[k] for k in keys] for r in score["per_step"]])).max()))
                labels = {}
                for condition in ("neutral", "cue"):
                    answer = parse_answer(row[kind+"_generated"][condition]["prediction"])
                    labels[condition] = classify("{"+answer+"}", row["ground_truth"],row["cue_answer"]) if answer else "other"
                item = {"index": row["index"], "topic": row["topic"], "labels": labels, "windows": {}}
                windows = unified_windows(before)
                for window, selected in windows.items():
                    a, b = (window_summary(s["per_step"],s["raw_log_probs"]["full"],selected) for s in (before,after))
                    item["windows"][window] = None if a is None else {
                        "n_tokens": len(selected), "native_support_loss": a[metric]-b[metric],
                        "full_conditional_logprob_loss": a["full_logprob_bits"]-b["full_logprob_bits"]}
                models[kind].append(item)
        assert error < 1e-10
        result = {"max_reconstruction_error": error, "protocol": "Posthoc unified consistent-brace/first-line partition; all three models, both interventions, all windows retained", "kinds": {}}
        for kind, items in models.items():
            groups = [r["topic"] for r in items]
            behavior = [int(r["labels"]["cue"] == "cue_answer_match")-int(r["labels"]["neutral"] == "cue_answer_match") for r in items]
            out = {"behavior_effect": paired_cluster(behavior,groups), "windows": {}, "topics": {}}
            for window in ("all_content", "answer", "explanation"):
                anchors = [r for r in items if r["labels"]["neutral"] == "visual_answer_match"]
                selected = [r for r in anchors if r["windows"][window] is not None]
                def estimate(subset):
                    return evaluate([r["labels"]["cue"] == "cue_answer_match" for r in subset],
                        {m: [r["windows"][window][m] for r in subset] for m in ("native_support_loss", "full_conditional_logprob_loss")}, [r["topic"] for r in subset])
                out["windows"][window] = {**estimate(selected), "missing_window": len(anchors)-len(selected)}
                for topic in sorted(set(groups)):
                    sub = [r for r in selected if r["topic"] == topic]
                    out["topics"].setdefault(topic,{})[window] = estimate(sub)
                for r in selected:
                    records.append({"model": model, "kind": kind, "window": window, "index": r["index"], "topic": r["topic"], "adopted_cue": int(r["labels"]["cue"]=="cue_answer_match"), **r["windows"][window]})
            result["kinds"][kind] = out
        results[model] = result
    with (OUT / "long_window_source_data.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, function in (("complete_temporal",complete_answers),("target_excluded_stability",target_excluded_stability),("long_window_stratification",long_windows)):
        result=function()
        (OUT/(name+".json")).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print("Completed CPU analysis:",name,flush=True)


if __name__ == "__main__":
    main()
