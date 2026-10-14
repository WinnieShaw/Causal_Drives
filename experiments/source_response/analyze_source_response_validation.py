#!/usr/bin/env python3
"""Analyse new diagnostic outputs with fixed contrasts and original coordinates."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from source_response_common import evaluate, paired_mean, read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    a = p.parse_args()
    rows, hashes = read_jsonl(a.input)
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    keys = [r["index"] for r in rows]
    expected = {r["index"] for r in manifest["records"]}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Shards incomplete or duplicated; cannot report a completed experiment.")
    good = [r for r in rows if r["status"] == "ok"]
    prefix, visual = [], []
    for r in good:
        base = {"index": r["index"], "sample_id": r["sample_id"], "topic": r["topic"]}
        if r["prefix"]:
            x = {**base}
            conditions = r["prefix"]
            for cond in ["neutral", "cue", "visual"]:
                info = conditions[cond]
                x[cond + "_answer_match"] = info["answer_match"]
                for score, value in info["fixed_candidate"]["content_summary"].items():
                    if isinstance(value, (float, int)):
                        x[cond + "_" + score] = value
            for score in ["pcd_level", "prefix_numerator_bits", "null_sequence_bits", "full_content_logprob"]:
                x["cue_delta_" + score] = x["cue_" + score] - x["neutral_" + score]
                x["visual_delta_" + score] = x["visual_" + score] - x["neutral_" + score]
            x["cue_adoption"] = int(x["cue_answer_match"] == "cue_answer_match")
            prefix.append(x)
        if r["visual"]:
            x = {**base}
            for cond in ["base", "counterfactual", "sham"]:
                info = r["visual"][cond]
                x[cond + "_answer_match"] = info["answer_match"]
                for score, value in info["fixed_candidate"]["content_summary"].items():
                    if isinstance(value, (float, int)):
                        x[cond + "_" + score] = value
            for score in ["vcd_level", "vcd_increment", "full_content_logprob", "conditional_visual_llr_content_bits"]:
                x["counterfactual_delta_" + score] = x["counterfactual_" + score] - x["base_" + score]
                x["sham_delta_" + score] = x["sham_" + score] - x["base_" + score]
                x["counterfactual_minus_sham_" + score] = x["counterfactual_" + score] - x["sham_" + score]
            x["counterfactual_following"] = int(x["counterfactual_answer_match"] == "counterfactual_visual_answer_match")
            visual.append(x)

    def analyse(frame, target, score_columns):
        return evaluate([r[target] for r in frame],
                        {name: [sign * r[col] for r in frame] for name, col, sign in score_columns},
                        [r["index"] for r in frame], a.bootstrap)

    prefix_anchor = [r for r in prefix if r["neutral_answer_match"] == "visual_answer_match"]
    visual_anchor = [r for r in visual if r["base_answer_match"] == "original_visual_answer_match"]
    outcomes = {
        "prefix_cue_adoption_from_neutral_visual_answer": analyse(prefix_anchor, "cue_adoption", [
            ("negative_delta_native_PCD", "cue_delta_pcd_level", -1),
            ("negative_delta_prefix_numerator", "cue_delta_prefix_numerator_bits", -1),
            ("negative_delta_full_logprob", "cue_delta_full_content_logprob", -1)]),
        "counterfactual_answer_following_from_visual_anchor": analyse(visual_anchor, "counterfactual_following", [
            ("negative_delta_VCD_content", "counterfactual_delta_vcd_level", -1),
            ("negative_delta_VCD_increment", "counterfactual_delta_vcd_increment", -1),
            ("negative_delta_full_logprob", "counterfactual_delta_full_content_logprob", -1),
            ("negative_delta_conditional_visual_LLR", "counterfactual_delta_conditional_visual_llr_content_bits", -1)])}
    paired = {}
    for name, frame, columns in [
        ("prefix", prefix, ["cue_delta_pcd_level", "cue_delta_prefix_numerator_bits", "cue_delta_null_sequence_bits",
                             "visual_delta_pcd_level", "visual_delta_prefix_numerator_bits"]),
        ("visual", visual, ["counterfactual_delta_vcd_level", "sham_delta_vcd_level", "counterfactual_minus_sham_vcd_level",
                             "counterfactual_minus_sham_vcd_increment", "counterfactual_minus_sham_full_content_logprob"]),
    ]:
        paired[name] = {col: paired_mean([r[col] for r in frame], [r["index"] for r in frame], a.bootstrap) for col in columns}
    report = {"n_expected": len(expected), "n_ok": len(good), "errors": len(rows) - len(good),
              "prefix_eligible": sum(r["eligible_prefix"] for r in manifest["records"]),
              "visual_eligible": sum(r["eligible_visual"] for r in manifest["records"]),
              "prefix_completed": len(prefix), "visual_completed": len(visual), "source_hashes": hashes,
              "observed_generation_content_coverage": {
                  kind: {cond: sum(bool(r[kind] and r[kind][cond].get("observed_generation")) for r in good)
                         for cond in conditions}
                  for kind, conditions in [("prefix", ["neutral", "cue", "visual"]),
                                           ("visual", ["base", "counterfactual", "sham"])]},
              "mode": manifest["mode"], "paired_effects": paired, "behavioral_alignment": outcomes,
              "answer_matches": {
                  "prefix": {c: dict(Counter(r[c + "_answer_match"] for r in prefix)) for c in ["neutral", "cue", "visual"]},
                  "visual": {c: dict(Counter(r[c + "_answer_match"] for r in visual)) for c in ["base", "counterfactual", "sham"]}},
              "interpretation": ["Original QCD/VCD/PCD definitions retained; no additive contribution shares.",
                                 "Prefix edits are forced off-policy history interventions, not natural-history ground truth.",
                                 "Native PCD changes include null-reference history effects; numerator and reference decomposition are reported separately.",
                                 "Paired effects score a shared reference candidate. Actual free generations also retain full scored trajectories for post-hoc case analysis, but are different paths and cannot be conflated with fixed-path effects.",
                                 "Real-image swaps match question, topic and prior-answer metadata; they are not pixel-localized interventions.",
                                 "Other and ambiguous generated answers are retained; source matching is behavioral evidence, not internal mechanism proof.",
                                 "All score directions fixed in code before new execution; nominal intervals without multiplicity correction."]}
    if not report["errors"]:
        if manifest["mode"] in {"all", "prefix"} and len(prefix) != report["prefix_eligible"]:
            raise ValueError("Missing eligible prefix conditions.")
        if manifest["mode"] in {"all", "visual"} and len(visual) != report["visual_eligible"]:
            raise ValueError("Missing eligible visual conditions.")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, frame in [("prefix", prefix), ("visual", visual)]:
        if frame:
            with (out / (name + "_samples.csv")).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(frame[0]))
                writer.writeheader()
                writer.writerows(frame)
    print(json.dumps({k: report[k] for k in ["n_expected", "n_ok", "errors", "prefix_completed", "visual_completed"]}))
    if report["errors"]:
        raise SystemExit("Errors retained; this is a partial analysis, not a completed run.")


if __name__ == "__main__":
    main()
