#!/usr/bin/env python3
"""Reanalyse existing paired outputs; no GPU, score-sign search or label fitting."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from source_response_common import classify, drive_window, evaluate, read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    a = p.parse_args()
    rows, hashes = read_jsonl(a.input)
    valid, skipped = [], Counter()
    seen = set()
    for r in rows:
        if r.get("status") != "ok":
            skipped["input_error"] += 1
            continue
        if r["sample_id"] in seen:
            raise ValueError("Duplicate sample ID; do not mix models or runs.")
        seen.add(r["sample_id"])
        labels = {condition: classify(r[condition]["prediction"], r["ground_truth"], r["expected_bias"])
                  for condition in ["base", "prior_prompt", "neutral_prompt", "title_image"]}
        windows = {condition: drive_window(r[condition]["drive_metrics"], r["fixed_path"]["tokens"])
                   for condition in ["base", "prior_prompt", "neutral_prompt"]}
        if any(w is None for w in windows.values()):
            skipped["no_content_token"] += 1
            continue
        f = {"index": r["index"], "sample_id": r["sample_id"], "topic": r["topic"],
             "prediction": r["base"]["prediction"], "ground_truth": r["ground_truth"],
             "cue_answer": r["expected_bias"], "response_length": r["fixed_path"]["num_tokens"],
             **{k + "_label": v for k, v in labels.items()},
             **{k: v for k, v in windows["base"].items() if k != "content_token_indices"}}
        for score in ["qcd_level", "qcd_increment", "pcd_level", "pcd_increment"]:
            f["cue_minus_neutral_" + score] = windows["prior_prompt"][score] - windows["neutral_prompt"][score]
        f["title_delta_vcd_response_mean"] = r["title_image"]["fixed_path_vcd"]["mean_vcd"] - r["base"]["drive_summary"]["mean_vcd"]
        f["blur_delta_vcd_response_mean"] = r["blur"]["fixed_path_vcd"][1]["mean_vcd"] - r["base"]["drive_summary"]["mean_vcd"]
        f["blur_label"] = classify(r["blur"]["severe_prediction"], r["ground_truth"], r["expected_bias"])
        f["blur_answer_changed"] = int(r["blur"]["severe_outcome"]["prediction_normalized"] != r["base"]["outcome"]["prediction_normalized"])
        for name in ["visual_llr_per_token", "question_llr_per_token"]:
            f[name] = r["simple_baselines"][name]
        f["full_mean_logprob"] = r["simple_baselines"]["full_score"]["mean_token_logprob"]
        valid.append(f)

    def analyse(cohort, target, scores):
        return evaluate([target(r) for r in cohort],
                        {name: [fn(r) for r in cohort] for name, fn in scores.items()},
                        [r["index"] for r in cohort], a.bootstrap)

    base_binary = [r for r in valid if r["base_label"] in {"visual_answer_match", "cue_answer_match"}]
    # Explicitly a behavioural proxy, not an assertion about hidden source use.
    current = analyse(base_binary, lambda r: int(r["base_label"] == "visual_answer_match"), {
        "vcd_content_level": lambda r: r["vcd_level"],
        "vcd_content_increment": lambda r: r["vcd_increment"],
        "conditional_visual_llr": lambda r: r["visual_llr_per_token"],
        "full_input_logprob": lambda r: r["full_mean_logprob"],
        "response_length": lambda r: r["response_length"],
    })
    question_cohort = [r for r in valid if r["base_label"] == "visual_answer_match"
                       and r["neutral_prompt_label"] == "visual_answer_match"]
    question = analyse(question_cohort, lambda r: int(r["prior_prompt_label"] == "cue_answer_match"), {
        "negative_cue_delta_qcd_content": lambda r: -r["cue_minus_neutral_qcd_level"],
        "negative_cue_delta_qcd_increment": lambda r: -r["cue_minus_neutral_qcd_increment"],
        "low_base_visual_increment": lambda r: -r["vcd_increment"],
        "low_conditional_visual_llr": lambda r: -r["visual_llr_per_token"],
        "low_full_confidence": lambda r: -r["full_mean_logprob"],
    })
    visual_cohort = [r for r in valid if r["base_label"] == "visual_answer_match"]
    title = analyse(visual_cohort, lambda r: int(r["title_image_label"] == "cue_answer_match"), {
        "negative_title_delta_vcd_response_mean": lambda r: -r["title_delta_vcd_response_mean"],
        "low_base_visual_increment": lambda r: -r["vcd_increment"],
        "low_conditional_visual_llr": lambda r: -r["visual_llr_per_token"],
        "low_full_confidence": lambda r: -r["full_mean_logprob"],
    })
    blur = analyse(valid, lambda r: r["blur_answer_changed"], {
        "negative_blur_delta_vcd_response_mean": lambda r: -r["blur_delta_vcd_response_mean"],
        "low_base_visual_increment": lambda r: -r["vcd_increment"],
        "low_conditional_visual_llr": lambda r: -r["visual_llr_per_token"],
        "low_full_confidence": lambda r: -r["full_mean_logprob"],
    })
    report = {"n_input": len(rows), "n_valid": len(valid), "skipped": dict(skipped), "source_hashes": hashes,
              "provenance": "Exploratory reuse of fixed saved paths; no independent hidden-source ground truth.",
              "selection": "Content tokens selected by saved token strings; no selection by scores or outcomes.",
              "labels": {c: dict(Counter(r[c + "_label"] for r in valid)) for c in ["base", "prior_prompt", "neutral_prompt", "title_image", "blur"]},
              "explicit_cue_cohort_topic_counts": dict(Counter(r["topic"] for r in question_cohort)),
              "explicit_cue_event_topic_counts": dict(Counter(r["topic"] for r in question_cohort if r["prior_prompt_label"] == "cue_answer_match")),
              "endpoints": {"base_visual_vs_prior_answer_match": current,
                            "explicit_cue_adoption_from_visual_anchor": question,
                            "title_cue_adoption_from_visual_anchor": title,
                            "blur_answer_change": blur},
              "interpretation": ["Answer matches do not uniquely identify internal information sources.",
                                 "Prompt-invariant VCD and constant first-brace null shifts are structural checks, not faithfulness evidence.",
                                 "Title/blur paths retained only response-level aggregates; no content-token reconstruction is claimed.",
                                 "Legacy full confidence/conditional LLR are whole-response aggregates, not content-window matched baselines; no superiority claim is justified.",
                                 "No directions flipped after viewing outcomes; intervals nominal, no multiplicity correction.",
                                 "Cross-coordinate maxima or percentages are not used as source ground truth."]}
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_following_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if valid:
        with (out / "source_following_samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(valid[0]))
            writer.writeheader()
            writer.writerows(valid)
    lines = ["来源跟随行为：已有配对结果的CPU重分析", "",
             "这是一组探索性行为一致性分析，不将视觉答案/先验答案匹配当作内部来源真值。",
             f"输入{len(rows)}，有效{len(valid)}；跳过：{dict(skipped)}。", ""]
    for end, value in report["endpoints"].items():
        lines.append(f"{end}: n={value['n']}, positives={value['positives']}")
        if value.get("low_event_warning"):
            lines.append("  事件数少：下列名义bootstrap区间不作为稳健泛化证据；单正例可出现退化区间。")
        for name, m in value["metrics"].items():
            lines.append(f"  {name}: AUROC={m['auroc']:.4f} [{m['auroc_ci95'][0]:.4f},{m['auroc_ci95'][1]:.4f}]")
    lines += ["", "全部配对差值在JSON中；不得将单项名义差异外推为普遍忠实归因。",
              f"显式线索事件的领域分布：{report['explicit_cue_event_topic_counts']}；不能当成全领域验证。",
              "旧完整输入置信度/条件LLR只有全回答聚合，不能与内容窗口分数构成严格匹配的优越性比较。新作业保存所有条件的token日志用于统一窗口。",
              "旧title/blur汇总只有全回答聚合及首/末值，不能恢复其内容token轨迹。"]
    (out / "source_following_report_zh.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(valid), "output": str(out), "labels": report["labels"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
