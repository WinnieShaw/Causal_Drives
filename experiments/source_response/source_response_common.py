"""CPU-only helpers: explicit answer matching, content windows and paired stats."""
import hashlib
import json
import re
import string
from pathlib import Path

import numpy as np

NUMBERS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve".split(), map(str, range(13))))


def normalize(text):
    text = str(text).strip().lower().replace("_", " ")
    for word, digit in NUMBERS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return " ".join("".join(c if c not in string.punctuation else " " for c in text).split())


def aliases(value):
    values = value if isinstance(value, (list, tuple)) else re.split(r"\s*(?:\||;|/|,)\s*", str(value))
    return sorted({normalize(v) for v in values if normalize(v)})


def answer_body(prediction):
    """Strict single-answer extraction, not substring matching or an LLM judge."""
    spans = re.findall(r"\{([^{}]+)\}", str(prediction))
    if len(spans) == 1:
        return normalize(spans[0])
    if len(spans) > 1:
        return ""
    return normalize(prediction)


def classify(prediction, visual_answer, cue_answer):
    a = answer_body(prediction)
    visual = a in aliases(visual_answer) if a else False
    cue = a in aliases(cue_answer) if a else False
    if visual and cue:
        return "ambiguous_match"
    if visual:
        return "visual_answer_match"
    if cue:
        return "cue_answer_match"
    return "other"


def content_indices(tokens):
    """Select alphanumeric tokens inside a unique brace pair; else lexical tokens.

    This is an answer-content window, not an inferred input attribution. Special
    tokens, braces and punctuation-only tokens are excluded. Combined BPE tokens
    touching a boundary are retained if they also contain answer characters.
    """
    spans, start = [], 0
    for token in tokens:
        spans.append((start, start + len(token)))
        start += len(token)
    joined = "".join(tokens)
    brackets = list(re.finditer(r"\{([^{}]+)\}", joined))
    bounds = (brackets[0].start(1), brackets[0].end(1)) if len(brackets) == 1 else None
    selected = []
    for i, (token, (lo, hi)) in enumerate(zip(tokens, spans)):
        if token.startswith("<|") and token.endswith("|>"):
            continue
        if not any(c.isalnum() for c in token):
            continue
        if bounds and not (lo < bounds[1] and hi > bounds[0]):
            continue
        selected.append(i)
    return selected


def drive_window(metrics, tokens, selected=None):
    rows = metrics["per_step"]
    assert len(rows) == len(tokens) and [r["t"] for r in rows] == list(range(1, len(tokens) + 1))
    selected = content_indices(tokens) if selected is None else selected
    if not selected:
        return None
    out = {"content_token_indices": selected, "content_token_count": len(selected)}
    for name in ["QCD", "VCD", "PCD"]:
        values = np.array([r[name + "_t"] for r in rows], dtype=float)
        out[name.lower() + "_level"] = float(values[selected].mean())
        # Original coordinates are retained. PCD differences are diagnostic
        # temporal differences, not a redefined token-probability estimand.
        out[name.lower() + "_increment"] = float(np.diff(np.r_[0., values])[selected].mean())
    out["prefix_numerator_bits"] = float(np.mean([rows[i]["log_P_A_t_do_prefix"] for i in selected]) / np.log(2))
    out["null_sequence_bits"] = float(np.mean([rows[i]["log_P_A_le_t"] for i in selected]) / np.log(2))
    return out


def read_jsonl(paths):
    result, hashes = [], {}
    for path in paths:
        path = Path(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    result.append(json.loads(line))
    return result, hashes


def weighted_metrics(y, score, weights):
    order = np.argsort(score, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
    pos = np.add.reduceat(weights[:, order] * y[order], boundaries, axis=1)
    neg = np.add.reduceat(weights[:, order] * (1 - y[order]), boundaries, axis=1)
    p, n = pos.sum(axis=1), neg.sum(axis=1)
    valid = (p > 0) & (n > 0)
    auc, ap = np.full(len(weights), np.nan), np.full(len(weights), np.nan)
    below = np.cumsum(neg, axis=1) - neg
    auc[valid] = ((pos * (below + neg / 2)).sum(axis=1) / np.maximum(p * n, 1))[valid]
    dp, dn = pos[:, ::-1], neg[:, ::-1]
    cp = np.cumsum(dp, axis=1)
    ca = cp + np.cumsum(dn, axis=1)
    precision = np.divide(cp, ca, out=np.zeros_like(cp), where=ca > 0)
    ap[valid] = ((dp * precision).sum(axis=1) / np.maximum(p, 1))[valid]
    return auc, ap


def evaluate(y, score_map, groups, rounds=5000, seed=42):
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    result = {"n": len(y), "positives": int(y.sum()), "groups": len(np.unique(groups))}
    if len(np.unique(y)) != 2:
        return {**result, "metrics": {}, "note": "single class or empty cohort"}
    _, codes = np.unique(groups, return_inverse=True)
    g = codes.max() + 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, g, (rounds, g))
    counts = np.zeros((rounds, g))
    np.add.at(counts, (np.repeat(np.arange(rounds), g), draws.ravel()), 1)
    weights = counts[:, codes]
    metrics, boot = {}, {}
    for name, score in score_map.items():
        score = np.asarray(score, float)
        assert score.shape == y.shape and np.isfinite(score).all()
        observed_auc, observed_ap = weighted_metrics(y, score, np.ones((1, len(y))))
        auc, ap = weighted_metrics(y, score, weights)
        boot[name] = auc
        metrics[name] = {"auroc": float(observed_auc[0]), "auprc": float(observed_ap[0]),
                         "auroc_ci95": np.nanquantile(auc, [.025, .975]).tolist(),
                         "auprc_ci95": np.nanquantile(ap, [.025, .975]).tolist(),
                         "valid_rounds": int(np.isfinite(auc).sum())}
    differences = {}
    names = list(metrics)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            differences[a + " minus " + b] = {
                "delta_auroc": metrics[a]["auroc"] - metrics[b]["auroc"],
                "ci95": np.nanquantile(boot[a] - boot[b], [.025, .975]).tolist()}
    return {**result, "positive_rate": float(y.mean()), "metrics": metrics, "paired_differences": differences,
            "low_event_warning": bool(min(int(y.sum()), int((1-y).sum())) < 10),
            "ci_note": "Few observed events can yield degenerate bootstrap intervals; they do not establish generalization."}


def paired_mean(values, groups, rounds=5000, seed=42):
    values, groups = np.asarray(values, float), np.asarray(groups)
    if not len(values):
        return {"n": 0}
    unique = np.unique(groups)
    # One paired difference per original sample (or averaged within sample).
    per_group = np.array([values[groups == g].mean() for g in unique])
    rng = np.random.default_rng(seed)
    estimates = per_group[rng.integers(0, len(unique), (rounds, len(unique)))].mean(axis=1)
    return {"n": len(values), "groups": len(unique), "mean": float(per_group.mean()),
            "ci95": np.quantile(estimates, [.025, .975]).tolist()}
