# Causal Drive Metrics for VLM Generation

This repository provides a simple implementation of three token-level causal drive metrics for vision-language model (VLM) autoregressive generation:

- **QCD**: Question Causal Drive
- **VCD**: Visual Causal Drive
- **PCD**: Prefix Causal Drive

The main metric implementation is in `causal_drive_metrics.py`, and an example computation script is provided in `compute.py`.

## Files

```text
.
├── causal_drive_metrics.py   # Core implementation of QCD, VCD, and PCD
├── compute.py                # Example script for model scoring and metric computation
└── README.md
```

## Metric Function

The core function is:

```python
from causal_drive_metrics import compute_causal_drives
```

It computes QCD, VCD, and PCD for one generated answer using pre-computed teacher-forced scores.

## Required Inputs

The function expects the following score dictionaries:

```python
score = {
    "token_probs": [...],      # token-level probabilities
    "cum_log_probs": [...],   # cumulative log probabilities of generated prefixes
}
```

The main inputs are:

- `baseline_score`: score under null visual input and empty prompt, used as `P(A_<=t)`.
- `visual_only_score`: score under target visual input and empty prompt, used for VCD.
- `joint_scores`: scores under all `(V=v, Q=q)` conditions.
- `p_v`: explicit distribution `P(V=v)`.
- `p_q_given_v`: explicit conditional distribution `P(Q=q | V=v)`.

The metric function does **not** silently assume uniform distributions. If uniform empirical distributions are used, they should be explicitly defined in `compute.py`.

## Example Usage

```python
from causal_drive_metrics import compute_causal_drives

metrics = compute_causal_drives(
    baseline_score=null_image_with_prefix,
    visual_only_score=empty_prompt_with_prefix[target_img_idx],
    joint_scores=cache_with_prefix,
    target_visual_id=target_img_idx,
    target_question_id=target_q_idx,
    p_v=Pv,
    p_q_given_v=Pq_given_v,
    prefixes=target_prefixes,
    tokens=target_step_tokens,
    step_interval=2,
)

causal_rows = metrics["per_step"]

avg_qcd = metrics["avg_qcd"]
avg_vcd = metrics["avg_vcd"]
avg_pcd = metrics["avg_pcd"]

qcd_curve = metrics["qcd_curve"]
vcd_curve = metrics["vcd_curve"]
pcd_curve = metrics["pcd_curve"]
```

For compatibility with older code that uses ACI naming:

```python
avg_qaci = metrics["avg_qcd"]
avg_vaci = metrics["avg_vcd"]
avg_paci = metrics["avg_pcd"]
```

## Notes

- `joint_scores[(vi, qi)]` should contain the teacher-forced probabilities of the same generated answer under visual input `vi` and question `qi`.
- `baseline_score` is usually computed with a null image and an empty prompt.
- `visual_only_score` is usually computed with the target image and an empty prompt.
- `step_interval=1` computes metrics for every generated token; larger values reduce computation and output size.

## Output

The returned dictionary contains:

```python
{
    "avg_qcd": ...,       # average QCD over selected token positions
    "avg_vcd": ...,       # average VCD over selected token positions
    "avg_pcd": ...,       # average PCD over selected token positions
    "per_step": [...],    # token-level metric values
    "meta": {...},        # metadata
}
```
### MiraData Video Question Files

The question files are stored as `.txt` files. They contain the video-question pairs used for open-ended video QA evaluation on the MiraData dataset. Each line corresponds to one video and includes the relative path of the video in MiraData and its associated question.

Each record consists of two fields:

- `rel_path`: the relative path of the video in the MiraData dataset;
- `question`: the open-ended question designed for the corresponding video.


The files are organized as follows:

```txt
question0-99.txt
question100-199.txt
question200-299.txt
question300-399.txt
question400-499.txt
