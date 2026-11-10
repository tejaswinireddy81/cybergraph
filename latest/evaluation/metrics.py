"""
evaluation/metrics.py
=====================
Metrics for the model-improvement study. Accuracy is reported but must never drive a decision.

Provided
  * binary_metrics        precision, recall, F1, false-positive rate, accuracy at a threshold
  * roc_auc / pr_auc      ranking quality (PR-AUC is meaningless without prevalence, so it is returned with it)
  * recall_at_fpr / fpr_at_recall / threshold_for_fpr / threshold_for_recall
  * precision_at_prevalence   precision re-weighted to a realistic attack share (UNSW is ~34 % attacks)
  * calibration           Brier score + expected calibration error + reliability bins
  * bootstrap_ci          confidence intervals resampled by GROUP (session / family / device), not by flow
  * per_group_table       per traffic class / attack family table with counts (small classes stay visible)
  * evaluate_scores       everything above in one dict
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

NAN = float("nan")


def _arr(a):
    return np.asarray(a)


def confusion(y_true, y_pred) -> Dict[str, int]:
    y_true, y_pred = _arr(y_true).astype(int), _arr(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _div(a, b):
    return a / b if b else NAN


def binary_metrics(y_true, score, threshold: float) -> Dict[str, float]:
    """Metrics at a fixed threshold (alert when score >= threshold)."""
    y_true, score = _arr(y_true).astype(int), _arr(score).astype(float)
    c = confusion(y_true, score >= threshold)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    f1 = _div(2 * precision * recall, precision + recall) if not (math.isnan(precision) or math.isnan(recall)) else NAN
    return {
        **c,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _div(fp, fp + tn),
        "accuracy": _div(tp + tn, tp + fp + tn + fn),  # reported, never used for decisions
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()) if len(y_true) else NAN,
    }


def roc_auc(y_true, score) -> float:
    y_true = _arr(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return NAN
    return float(roc_auc_score(y_true, score))


def pr_auc(y_true, score) -> float:
    y_true = _arr(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return NAN
    return float(average_precision_score(y_true, score))


def recall_at_fpr(y_true, score, target_fpr: float) -> float:
    """Best recall achievable while keeping the false-positive rate <= target_fpr."""
    y_true = _arr(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return NAN
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def fpr_at_recall(y_true, score, target_recall: float) -> float:
    """Lowest false-positive rate achievable while keeping recall >= target_recall."""
    y_true = _arr(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return NAN
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = tpr >= target_recall
    return float(fpr[ok].min()) if ok.any() else 1.0


def threshold_for_fpr(y_true, score, target_fpr: float) -> float:
    """Smallest threshold whose FPR on THIS data is <= target_fpr. Fit on validation data only."""
    y_true, score = _arr(y_true).astype(int), _arr(score).astype(float)
    neg = np.sort(score[y_true == 0])
    if len(neg) == 0:
        raise ValueError("need benign samples to choose a threshold for a target FPR")
    k = int(math.floor((1.0 - target_fpr) * len(neg)))
    k = min(max(k, 0), len(neg) - 1)
    # alert when score >= thr ; place thr just above the k-th benign score
    thr = float(np.nextafter(neg[k], np.inf)) if target_fpr < 1.0 else float(neg[0])
    return thr


def precision_at_prevalence(y_true, score, threshold: float, prevalence: float) -> float:
    """Precision re-weighted to a different attack prevalence: TPR*pi / (TPR*pi + FPR*(1-pi))."""
    m = binary_metrics(y_true, score, threshold)
    tpr, fpr = m["recall"], m["fpr"]
    if math.isnan(tpr) or math.isnan(fpr):
        return NAN
    den = tpr * prevalence + fpr * (1.0 - prevalence)
    return NAN if den == 0 else float(tpr * prevalence / den)


def calibration(y_true, prob, n_bins: int = 10) -> Dict[str, object]:
    y_true, prob = _arr(y_true).astype(float), np.clip(_arr(prob).astype(float), 0.0, 1.0)
    brier = float(np.mean((prob - y_true) ** 2))
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf, acc = float(prob[m].mean()), float(y_true[m].mean())
        ece += m.mean() * abs(conf - acc)
        bins.append({"bin": b, "n": int(m.sum()), "mean_prob": conf, "frac_positive": acc})
    return {"brier": brier, "ece": float(ece), "bins": bins}


def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], float], y_true, score,
                 groups: Optional[Sequence] = None, n_boot: int = 500, seed: int = 0,
                 alpha: float = 0.05) -> Dict[str, float]:
    """Percentile CI. With `groups`, whole groups (session/family/device) are resampled together,
    which is the honest unit when flows within a group are correlated."""
    rng = np.random.default_rng(seed)
    y_true, score = _arr(y_true), _arr(score)
    n = len(y_true)
    point = metric(y_true, score)
    if n == 0:
        return {"point": point, "lo": NAN, "hi": NAN, "n_boot": 0}
    if groups is None:
        index_sets = None
    else:
        groups = _arr(groups)
        uniq = np.unique(groups)
        index_sets = [np.flatnonzero(groups == g) for g in uniq]
    vals = []
    for _ in range(n_boot):
        if index_sets is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(index_sets), len(index_sets))
            idx = np.concatenate([index_sets[i] for i in pick])
        v = metric(y_true[idx], score[idx])
        if not (v is None or (isinstance(v, float) and math.isnan(v))):
            vals.append(v)
    if not vals:
        return {"point": point, "lo": NAN, "hi": NAN, "n_boot": 0}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": len(vals)}


def per_group_table(y_true, score, threshold: float, groups) -> list:
    """One row per group (traffic class, attack family ...). Includes counts so tiny groups are not hidden.
    For all-benign groups only FPR is meaningful; for all-attack groups only recall."""
    y_true, score, groups = _arr(y_true).astype(int), _arr(score).astype(float), _arr(groups)
    rows = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        mt = binary_metrics(y_true[m], score[m], threshold)
        rows.append({
            "group": g, "n": int(m.sum()), "n_benign": int((y_true[m] == 0).sum()),
            "n_attack": int((y_true[m] == 1).sum()),
            "fpr": mt["fpr"], "recall": mt["recall"], "precision": mt["precision"],
            "alerts": int(((score[m] >= threshold)).sum()),
        })
    return rows


def evaluate_scores(y_true, score, threshold: float, groups=None, with_ci: bool = False,
                    n_boot: int = 300, seed: int = 0) -> Dict[str, object]:
    """Complete report for one score vector at a threshold chosen elsewhere (validation data)."""
    out: Dict[str, object] = {
        "at_threshold": binary_metrics(y_true, score, threshold),
        "roc_auc": roc_auc(y_true, score),
        "pr_auc": pr_auc(y_true, score),
        "recall_at_fpr": {str(f): recall_at_fpr(y_true, score, f) for f in (0.001, 0.005, 0.01)},
        "fpr_at_recall": {str(r): fpr_at_recall(y_true, score, r) for r in (0.90, 0.95)},
        "precision_at_prevalence": {str(p): precision_at_prevalence(y_true, score, threshold, p)
                                    for p in (0.001, 0.01, 0.10)},
    }
    s = _arr(score).astype(float)
    if s.min() >= 0.0 and s.max() <= 1.0:
        out["calibration"] = calibration(y_true, s)
    if groups is not None:
        out["per_group"] = per_group_table(y_true, score, threshold, groups)
    if with_ci:
        out["ci"] = {
            "roc_auc": bootstrap_ci(roc_auc, y_true, score, groups, n_boot, seed),
            "pr_auc": bootstrap_ci(pr_auc, y_true, score, groups, n_boot, seed),
            "recall": bootstrap_ci(lambda y, s_: binary_metrics(y, s_, threshold)["recall"], y_true, score, groups, n_boot, seed),
            "fpr": bootstrap_ci(lambda y, s_: binary_metrics(y, s_, threshold)["fpr"], y_true, score, groups, n_boot, seed),
        }
    return out
