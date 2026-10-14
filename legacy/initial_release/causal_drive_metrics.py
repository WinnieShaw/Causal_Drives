#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
causal_drive_metrics.py

Open-source utilities for computing the three causal drive metrics in a
VLM autoregressive generation setting:

    QCD: Question Causal Drive
    VCD: Visual Causal Drive
    PCD: Prefix Causal Drive

The implementation follows the formula style in the paper:

    QCD_t = log2( P(A_<=t = a_<=t | do(Q=q)) / P(A_<=t = a_<=t) )

    VCD_t = log2( P(A_<=t = a_<=t | do(V=v)) / P(A_<=t = a_<=t) )

    PCD_t = log2( P(A_<=t = a_<=t | do(A_<=t-1=a_<=t-1)) / P(A_<=t = a_<=t) )

Computational interpretation:
    - P(A_<=t) is read from `baseline_score["cum_log_probs"][t-1]`,
      usually scored under null visual input + empty/null prompt.
    - QCD uses sum_v P(V=v) P(A_<=t | V=v, Q=q).
    - VCD uses P(A_<=t | V=v), supplied as `visual_only_score`,
      usually scored under target visual input + empty/null prompt.
    - PCD uses sum_v sum_q P(V=v) P(Q=q|V=v)
      P(e_t | A_<=t-1, V=v, Q=q). Since A_<=t-1 is fixed by intervention,
      the computable term is the teacher-forced probability of the current
      token e_t under each (V, Q) condition.

Score dictionary format:
    score = {
        "token_probs": [p(e_1 | condition), p(e_2 | e_1, condition), ...],
        "cum_log_probs": [
            log P(e_1 | condition),
            log P(e_1, e_2 | condition),
            ...
        ],
    }

`joint_scores` format:
    joint_scores[(v_id, q_id)] = score
"""

from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

EPS = 1e-12

Score = Mapping[str, Sequence[float]]
JointScores = Mapping[Tuple[Hashable, Hashable], Score]
Distribution = Mapping[Hashable, float]
ConditionalDistribution = Mapping[Hashable, Mapping[Hashable, float]]


def _safe_exp(logp: float, eps: float = EPS) -> float:
    """Convert a log-probability to probability with numerical floor."""
    return max(math.exp(float(logp)), eps)


def _safe_log2_ratio(numerator: float, denominator: float, eps: float = EPS) -> float:
    """Compute log2(numerator / denominator) with numerical floor."""
    return math.log2(max(numerator, eps) / max(denominator, eps))


def _require_score(score: Score, name: str) -> None:
    if "token_probs" not in score:
        raise KeyError(f"{name} must contain key 'token_probs'.")
    if "cum_log_probs" not in score:
        raise KeyError(f"{name} must contain key 'cum_log_probs'.")


def _check_distribution_sums_to_one(
    dist: Mapping[Hashable, float],
    name: str,
    tol: float = 1e-6,
) -> None:
    total = float(sum(dist.values()))
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"{name} must sum to 1.0 under the paper formula, but got {total:.12f}. "
            f"Please pass an explicit valid distribution instead of relying on 1/n."
        )


def validate_causal_drive_inputs(
    *,
    baseline_score: Score,
    visual_only_score: Score,
    joint_scores: JointScores,
    target_visual_id: Hashable,
    target_question_id: Hashable,
    p_v: Distribution,
    p_q_given_v: ConditionalDistribution,
    check_distribution: bool = True,
) -> None:
    """
    Validate inputs for strict formula-based causal drive computation.

    This function intentionally does not create uniform distributions.
    `p_v` and `p_q_given_v` must be supplied explicitly.
    """
    _require_score(baseline_score, "baseline_score")
    _require_score(visual_only_score, "visual_only_score")

    if not joint_scores:
        raise ValueError("joint_scores is empty.")

    if target_visual_id not in p_v:
        raise KeyError(f"target_visual_id={target_visual_id!r} is missing from p_v.")

    for v_id, pv in p_v.items():
        if pv < 0:
            raise ValueError(f"p_v[{v_id!r}] must be non-negative, got {pv}.")
        if v_id not in p_q_given_v:
            raise KeyError(f"p_q_given_v is missing conditional distribution for V={v_id!r}.")
        if check_distribution:
            _check_distribution_sums_to_one(p_q_given_v[v_id], f"p_q_given_v[{v_id!r}]")

        for q_id, pqv in p_q_given_v[v_id].items():
            if pqv < 0:
                raise ValueError(
                    f"p_q_given_v[{v_id!r}][{q_id!r}] must be non-negative, got {pqv}."
                )
            key = (v_id, q_id)
            if key not in joint_scores:
                raise KeyError(
                    f"joint_scores is missing key {key!r}. "
                    "For strict formula computation, all nonzero-probability (V,Q) "
                    "conditions must be scored."
                )
            _require_score(joint_scores[key], f"joint_scores[{key!r}]")

    if check_distribution:
        _check_distribution_sums_to_one(p_v, "p_v")


def get_valid_steps(
    *,
    baseline_score: Score,
    visual_only_score: Score,
    joint_scores: JointScores,
    prefixes: Optional[Sequence[str]] = None,
    tokens: Optional[Sequence[str]] = None,
) -> int:
    """
    Return the maximum number of token steps that can be safely computed.
    """
    lengths: List[int] = [
        len(baseline_score["cum_log_probs"]),
        len(visual_only_score["cum_log_probs"]),
    ]

    for score in joint_scores.values():
        lengths.append(len(score["token_probs"]))
        lengths.append(len(score["cum_log_probs"]))

    if prefixes is not None:
        lengths.append(len(prefixes))
    if tokens is not None:
        lengths.append(len(tokens))

    return min(lengths) if lengths else 0


def compute_qcd_at_step(
    *,
    t: int,
    baseline_score: Score,
    joint_scores: JointScores,
    target_question_id: Hashable,
    p_v: Distribution,
    eps: float = EPS,
) -> Dict[str, float]:
    """
    Compute QCD at token step t, where t is 1-indexed.

    Formula:
        P(A_<=t | do(Q=q))
        = sum_v P(V=v) P(A_<=t | V=v, Q=q)

        QCD_t = log2( P(A_<=t | do(Q=q)) / P(A_<=t) )
    """
    idx = t - 1
    p_uncond = _safe_exp(baseline_score["cum_log_probs"][idx], eps)

    p_do_q = 0.0
    for v_id, pv in p_v.items():
        key = (v_id, target_question_id)
        if key not in joint_scores:
            raise KeyError(f"joint_scores is missing key {key!r} required by QCD.")
        p_prefix = _safe_exp(joint_scores[key]["cum_log_probs"][idx], eps)
        p_do_q += float(pv) * p_prefix

    qcd_t = _safe_log2_ratio(p_do_q, p_uncond, eps)

    return {
        "P_A_le_t": p_uncond,
        "P_A_le_t_do_Q": max(p_do_q, eps),
        "QCD_t": qcd_t,
    }


def compute_vcd_at_step(
    *,
    t: int,
    baseline_score: Score,
    visual_only_score: Score,
    eps: float = EPS,
) -> Dict[str, float]:
    """
    Compute VCD at token step t, where t is 1-indexed.

    Formula:
        VCD_t = log2( P(A_<=t | do(V=v)) / P(A_<=t) )

    Computable form used here:
        P(A_<=t | do(V=v)) = P(e_1 | V=v) prod_{i=2}^t P(e_i | A_<=i-1, V=v)

    Therefore `visual_only_score` should be obtained by teacher-forced scoring
    of the target answer under target visual input and an empty/null question
    prompt, i.e., the V-only condition.
    """
    idx = t - 1
    p_uncond = _safe_exp(baseline_score["cum_log_probs"][idx], eps)
    p_do_v = _safe_exp(visual_only_score["cum_log_probs"][idx], eps)

    vcd_t = _safe_log2_ratio(p_do_v, p_uncond, eps)

    return {
        "P_A_le_t": p_uncond,
        "P_A_le_t_do_V": p_do_v,
        "VCD_t": vcd_t,
    }


def compute_pcd_at_step(
    *,
    t: int,
    baseline_score: Score,
    joint_scores: JointScores,
    p_v: Distribution,
    p_q_given_v: ConditionalDistribution,
    eps: float = EPS,
) -> Dict[str, float]:
    """
    Compute PCD at token step t, where t is 1-indexed.

    Paper formula:
        P(A_<=t | do(A_<=t-1=a_<=t-1))
        = sum_v sum_q P(V=v) P(Q=q|V=v)
          P(A_<=t | V=v, Q=q, A_<=t-1=a_<=t-1)

    Since A_<=t-1 is fixed by intervention, the computable term is:
        P(e_t | A_<=t-1, V=v, Q=q)

    Therefore this implementation uses `token_probs[t-1]`, not
    `cum_log_probs[t-1]`, in the numerator.
    """
    idx = t - 1
    p_uncond = _safe_exp(baseline_score["cum_log_probs"][idx], eps)

    p_do_prefix = 0.0
    for v_id, pv in p_v.items():
        for q_id, pqv in p_q_given_v[v_id].items():
            key = (v_id, q_id)
            if key not in joint_scores:
                raise KeyError(f"joint_scores is missing key {key!r} required by PCD.")
            p_token = max(float(joint_scores[key]["token_probs"][idx]), eps)
            p_do_prefix += float(pv) * float(pqv) * p_token

    pcd_t = _safe_log2_ratio(p_do_prefix, p_uncond, eps)

    return {
        "P_A_le_t": p_uncond,
        "P_A_le_t_do_prefix": max(p_do_prefix, eps),
        "PCD_t": pcd_t,
    }


def compute_causal_drives(
    *,
    baseline_score: Score,
    visual_only_score: Score,
    joint_scores: JointScores,
    target_visual_id: Hashable,
    target_question_id: Hashable,
    p_v: Distribution,
    p_q_given_v: ConditionalDistribution,
    prefixes: Optional[Sequence[str]] = None,
    tokens: Optional[Sequence[str]] = None,
    step_interval: int = 1,
    eps: float = EPS,
    check_distribution: bool = True,
) -> Dict[str, Any]:
    """
    Compute QCD, VCD, and PCD curves for one generated answer.

    Parameters
    ----------
    baseline_score:
        Score under null visual input + empty/null prompt. Provides P(A_<=t).

    visual_only_score:
        Score under target visual input + empty/null prompt. Provides
        P(A_<=t | do(V=v)) in the paper's V-only computable form.

    joint_scores:
        Scores under all (V=v, Q=q) conditions. Keys are `(v_id, q_id)`.
        These are used by QCD and PCD.

    target_visual_id:
        The visual id of the target sample. Kept for output metadata and
        validation. VCD itself is read from `visual_only_score`.

    target_question_id:
        The question id q fixed by do(Q=q) for QCD.

    p_v:
        Explicit distribution P(V=v). This function never assumes 1/n.

    p_q_given_v:
        Explicit conditional distribution P(Q=q | V=v). This function never
        assumes 1/n.

    prefixes, tokens:
        Optional generated prefixes/tokens for readable per-step outputs.

    step_interval:
        Compute every `step_interval` tokens. Use 1 for all tokens.

    check_distribution:
        If True, require p_v and each p_q_given_v[v] to sum to 1.

    Returns
    -------
    A dictionary with:
        avg_qcd, avg_vcd, avg_pcd, per_step
    """
    if step_interval <= 0:
        raise ValueError(f"step_interval must be positive, got {step_interval}.")

    validate_causal_drive_inputs(
        baseline_score=baseline_score,
        visual_only_score=visual_only_score,
        joint_scores=joint_scores,
        target_visual_id=target_visual_id,
        target_question_id=target_question_id,
        p_v=p_v,
        p_q_given_v=p_q_given_v,
        check_distribution=check_distribution,
    )

    valid_steps = get_valid_steps(
        baseline_score=baseline_score,
        visual_only_score=visual_only_score,
        joint_scores=joint_scores,
        prefixes=prefixes,
        tokens=tokens,
    )

    qcd_curve: List[float] = []
    vcd_curve: List[float] = []
    pcd_curve: List[float] = []
    per_step: List[Dict[str, Any]] = []

    for t in range(1, valid_steps + 1, step_interval):
        qcd = compute_qcd_at_step(
            t=t,
            baseline_score=baseline_score,
            joint_scores=joint_scores,
            target_question_id=target_question_id,
            p_v=p_v,
            eps=eps,
        )
        vcd = compute_vcd_at_step(
            t=t,
            baseline_score=baseline_score,
            visual_only_score=visual_only_score,
            eps=eps,
        )
        pcd = compute_pcd_at_step(
            t=t,
            baseline_score=baseline_score,
            joint_scores=joint_scores,
            p_v=p_v,
            p_q_given_v=p_q_given_v,
            eps=eps,
        )

        row: Dict[str, Any] = {
            "t": t,
            "P_A_le_t": qcd["P_A_le_t"],
            "P_A_le_t_do_Q": qcd["P_A_le_t_do_Q"],
            "P_A_le_t_do_V": vcd["P_A_le_t_do_V"],
            "P_A_le_t_do_prefix": pcd["P_A_le_t_do_prefix"],
            "QCD_t": qcd["QCD_t"],
            "VCD_t": vcd["VCD_t"],
            "PCD_t": pcd["PCD_t"],
        }

        if prefixes is not None:
            row["prefix"] = prefixes[t - 1]
        if tokens is not None:
            row["token"] = tokens[t - 1]

        per_step.append(row)
        qcd_curve.append(qcd["QCD_t"])
        vcd_curve.append(vcd["VCD_t"])
        pcd_curve.append(pcd["PCD_t"])

    return {
        "avg_qcd": sum(qcd_curve) / len(qcd_curve) if qcd_curve else None,
        "avg_vcd": sum(vcd_curve) / len(vcd_curve) if vcd_curve else None,
        "avg_pcd": sum(pcd_curve) / len(pcd_curve) if pcd_curve else None,
        "per_step": per_step,
        "meta": {
            "target_visual_id": target_visual_id,
            "target_question_id": target_question_id,
            "valid_steps": valid_steps,
            "step_interval": step_interval,
            "distribution_checked": check_distribution,
        },
    }


def build_uniform_distributions(
    visual_ids: Iterable[Hashable],
    question_ids: Iterable[Hashable],
) -> Tuple[Dict[Hashable, float], Dict[Hashable, Dict[Hashable, float]]]:
    """
    Convenience helper for experiments that intentionally use empirical
    uniform distributions.

    This helper is separate from `compute_causal_drives` so that the metric
    implementation itself never silently replaces P(V) or P(Q|V) by 1/n.
    """
    v_ids = list(visual_ids)
    q_ids = list(question_ids)
    if not v_ids:
        raise ValueError("visual_ids is empty.")
    if not q_ids:
        raise ValueError("question_ids is empty.")

    p_v = {v: 1.0 / len(v_ids) for v in v_ids}
    p_q_given_v = {
        v: {q: 1.0 / len(q_ids) for q in q_ids}
        for v in v_ids
    }
    return p_v, p_q_given_v


# Backward-compatible aliases if old result files used ACI naming.
compute_qaci_at_step = compute_qcd_at_step
compute_vaci_at_step = compute_vcd_at_step
compute_paci_at_step = compute_pcd_at_step
