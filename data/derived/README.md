# Derived outputs

These compact outputs contain aggregate estimates or score-only sample rows;
they do not contain images, prompts, answers, reference text, or token-probability
traces.

- `per_sample_drive_summary.csv`, `length_correlations.csv`,
  `relative_position_curve_bootstrap_95ci.csv`, and
  `sample_level_bootstrap_95ci.csv`: 500 MAVIS response-level summaries and
  temporal profiles.
- `per_sample_recovered.csv`: finite-support recovery output. Raw candidate
  texts are omitted.
- `complete_temporal.json`, `temporal_source_data.csv`,
  `target_excluded_stability.json`, and `long_window_stratification.json`:
  completed-answer, support-stability, and fragment-analysis summaries.
- `text_cue_summary.json` and `text_cue_source_data.csv`: topic-level summaries
  for the paired text-cue manipulation.
- `visual_input_sensitivity/`: fixed-answer perturbation summary and numeric
  paired outcomes.
- `domain_visual_response/`: domain-level and figure-source summaries.
- `behavior_summary.json` and `score_audit_*.json`: response-level audit
  summaries for the two model-size probes.
- `summary.json`: aligned MAVIS scoring protocol and compact run provenance.

The figure scripts check input counts and reported intervals before rendering.
The full inference traces are not included, so these aggregate summaries cannot
be used to rerun model scoring or independently reconstruct every statistic.
