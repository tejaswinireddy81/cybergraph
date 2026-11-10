"""Tests for the evaluation framework: metrics, splits, sealed test sets and feature sets."""
import math

import numpy as np
import pandas as pd
import pytest

from config import feature_sets as fs
from evaluation import metrics as M
from evaluation.sealed import SealedAccessError, SealedTestSet, ensure_not_sealed
from evaluation.splits import (SplitOverlapError, assert_disjoint, chronological_split, fingerprint,
                               group_split, leave_one_family_out)


# ──────────────────────────────────────────────────────────────── metrics
def test_binary_metrics_known_values():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .2, .6, .1, .1, .1, .1, .1])
    m = M.binary_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 5, 1)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(1 / 6) and m["f1"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.8)


def test_auc_perfect_random_and_single_class():
    y = np.array([0, 0, 1, 1])
    assert M.roc_auc(y, [.1, .2, .8, .9]) == 1.0 and M.pr_auc(y, [.1, .2, .8, .9]) == 1.0
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 5000)
    assert abs(M.roc_auc(yy, rng.random(5000)) - 0.5) < 0.03
    assert math.isnan(M.roc_auc(np.zeros(5), np.arange(5)))        # undefined, not a fake number
    assert math.isnan(M.pr_auc(np.ones(5), np.arange(5)))


def test_threshold_for_fpr_respects_target_on_its_own_data():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(2000), np.ones(500)].astype(int)
    s = np.r_[rng.normal(0, 1, 2000), rng.normal(3, 1, 500)]
    for target in (0.001, 0.005, 0.05):
        thr = M.threshold_for_fpr(y, s, target)
        assert M.binary_metrics(y, s, thr)["fpr"] <= target + 1e-12


def test_recall_at_fpr_and_fpr_at_recall_are_consistent():
    y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    s = np.r_[np.linspace(0, 0.6, 1000), np.linspace(0.4, 1.0, 1000)]
    r = M.recall_at_fpr(y, s, 0.01)
    assert 0.0 <= r <= 1.0
    f = M.fpr_at_recall(y, s, r)
    assert f <= 0.01 + 1e-9


def test_precision_at_prevalence_matches_bayes():
    y = np.r_[np.zeros(900), np.ones(100)].astype(int)
    s = np.r_[np.r_[np.zeros(855), np.ones(45)], np.r_[np.zeros(10), np.ones(90)]]
    m = M.binary_metrics(y, s, 0.5)
    tpr, fpr = m["recall"], m["fpr"]                       # 0.9 and 0.05
    for pi in (0.001, 0.01, 0.1):
        assert M.precision_at_prevalence(y, s, 0.5, pi) == pytest.approx(tpr * pi / (tpr * pi + fpr * (1 - pi)))
    # realistic prevalence makes precision far lower than at the dataset's own 10 %
    assert M.precision_at_prevalence(y, s, 0.5, 0.001) < 0.05


def test_calibration_of_a_calibrated_and_a_miscalibrated_score():
    rng = np.random.default_rng(2)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert M.calibration(y, p)["ece"] < 0.03
    assert M.calibration(y, np.clip(p * 0.2, 0, 1))["ece"] > 0.15


def test_bootstrap_resamples_whole_groups_and_brackets_the_point():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(20), 50)
    y = rng.integers(0, 2, 1000)
    s = y * 0.5 + rng.random(1000) * 0.8
    ci = M.bootstrap_ci(M.roc_auc, y, s, groups=groups, n_boot=200, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"] and ci["n_boot"] > 0


def test_per_group_table_keeps_small_groups_visible():
    y = np.array([0] * 50 + [0] * 3 + [1] * 10)
    s = np.array([0.1] * 50 + [0.9, 0.9, 0.1] + [0.9] * 10)
    g = np.array(["dns"] * 50 + ["igmp"] * 3 + ["scan"] * 10)
    rows = {r["group"]: r for r in M.per_group_table(y, s, 0.5, g)}
    assert rows["igmp"]["n"] == 3 and rows["igmp"]["fpr"] == pytest.approx(2 / 3)
    assert rows["scan"]["recall"] == 1.0 and rows["dns"]["fpr"] == 0.0


def test_evaluate_scores_contains_every_requested_metric():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 600)
    s = np.clip(y * 0.4 + rng.random(600) * 0.6, 0, 1)
    r = M.evaluate_scores(y, s, 0.7, groups=rng.choice(["a", "b"], 600), with_ci=True, n_boot=50)
    for key in ("at_threshold", "roc_auc", "pr_auc", "recall_at_fpr", "fpr_at_recall",
                "precision_at_prevalence", "calibration", "per_group", "ci"):
        assert key in r
    for key in ("precision", "recall", "f1", "fpr", "accuracy"):
        assert key in r["at_threshold"]


# ──────────────────────────────────────────────────────────────── splits
def _frame(n=1000):
    rng = np.random.default_rng(5)
    fam = rng.choice(["Benign", "Benign", "Benign", "Exploits", "DoS", "Recon"], n)
    return pd.DataFrame({"t": rng.permutation(n), "family": fam})


def test_chronological_split_is_disjoint_and_ordered():
    df = _frame()
    sp = chronological_split(df, "t", (0.6, 0.2, 0.2))
    assert_disjoint(train=sp.train, cal=sp.cal, test=sp.test)
    assert df.t.iloc[sp.train].max() < df.t.iloc[sp.cal].min() <= df.t.iloc[sp.cal].max() < df.t.iloc[sp.test].min()
    assert len(sp.train) + len(sp.cal) + len(sp.test) == len(df)


def test_group_split_never_straddles_a_group():
    groups = np.repeat([f"s{i}" for i in range(10)], 30)
    sp = group_split(groups, seed=1)
    g = lambda idx: set(groups[idx])  # noqa: E731
    assert not (g(sp.train) & g(sp.cal)) and not (g(sp.train) & g(sp.test)) and not (g(sp.cal) & g(sp.test))
    with pytest.raises(ValueError):
        group_split(["a", "b"], seed=0)


def test_leave_one_family_out_hides_the_held_out_family_from_training_and_calibration():
    df = _frame()
    seen = []
    for sp in leave_one_family_out(df, "family", "Benign", "t"):
        fam = sp.notes["held_out_family"]
        seen.append(fam)
        assert fam not in set(df.family.iloc[sp.train]) | set(df.family.iloc[sp.cal])
        assert fam in set(df.family.iloc[sp.test])
        assert_disjoint(train=sp.train, cal=sp.cal, test=sp.test)
    assert sorted(seen) == ["DoS", "Exploits", "Recon"]


def test_overlap_is_detected_and_fingerprints_are_order_independent():
    with pytest.raises(SplitOverlapError):
        assert_disjoint(a=[1, 2, 3], b=[3, 4])
    assert fingerprint([3, 1, 2]) == fingerprint([1, 2, 3]) != fingerprint([1, 2, 4])


# ──────────────────────────────────────────────────────────────── sealed test sets
def test_sealed_set_hides_data_and_allows_exactly_one_logged_look(tmp_path):
    log = tmp_path / "access.jsonl"
    sealed = SealedTestSet(pd.DataFrame({"a": [1, 2, 3]}), "final", log_path=str(log), max_looks=1)
    for access in (lambda: sealed.a, lambda: sealed[0], lambda: list(iter(sealed))):
        with pytest.raises(SealedAccessError):
            access()
    assert sealed.evaluate(lambda d: int(d.a.sum()), "unit test") == 6
    with pytest.raises(SealedAccessError):
        sealed.evaluate(lambda d: 0, "second look")           # a second look needs an explicit decision
    assert len(sealed.looks) == 1 and log.read_text().count("\n") == 1
    with pytest.raises(SealedAccessError):
        sealed.name = "tampered"


def test_sealed_data_cannot_be_used_for_fitting_or_calibration():
    sealed = SealedTestSet(pd.DataFrame({"a": [1]}), "final")
    with pytest.raises(SealedAccessError):
        ensure_not_sealed(sealed, context="calibration")
    ensure_not_sealed(pd.DataFrame({"a": [1]}))               # ordinary frames pass


def test_run_candidate_fit_functions_reject_the_sealed_test_set():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "experiments", "run_candidate.py")
    spec = importlib.util.spec_from_file_location("run_candidate_mod", path)
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    sealed = SealedTestSet(pd.DataFrame({"x": [1]}), "final")
    with pytest.raises(SealedAccessError):
        rc.fit_base_models(sealed, "F0", 1)
    with pytest.raises(SealedAccessError):
        rc.fit_fusion_and_thresholds(sealed, {"xgb": np.array([0.5]), "ae": np.array([0.0])}, 1)
    with pytest.raises(SystemExit):                            # approval gate: no full run without the flag
        rc.run("F0")


# ──────────────────────────────────────────────────────────────── feature sets
def test_feature_set_sizes_and_nesting():
    assert len(fs.columns("F0")) == 47
    sizes = {k: len(fs.columns(k)) for k in fs.FEATURE_SETS}
    assert sizes == {"F0": 47, "F1": 45, "F2": 38, "F3": 29, "F4": 32, "F5": 34}
    f0, f1, f2, f3 = (set(fs.columns(k)) for k in ("F0", "F1", "F2", "F3"))
    assert f3 < f2 < f1 < f0                                   # each step only removes features


def test_min_packet_length_features_are_gone_from_f1_onwards():
    assert {"MIN_IP_PKT_LEN", "SHORTEST_FLOW_PKT"} <= set(fs.columns("F0"))
    for k in ("F1", "F2", "F3", "F4", "F5"):
        assert not {"MIN_IP_PKT_LEN", "SHORTEST_FLOW_PKT"} & set(fs.columns(k)), k


def test_f4_f5_drop_shortcuts_and_add_derived_features():
    f4 = set(fs.columns("F4"))
    assert not f4 & set(fs.RATE_COLUMNS) and "PROTOCOL" not in f4 and not f4 & set(fs.RAW_VOLUME)
    assert set(fs.DERIVED_F4) <= f4
    assert set(fs.DERIVED_F5_EXTRA) <= set(fs.columns("F5")) and not set(fs.DERIVED_F5_EXTRA) & f4


def test_feature_matrix_is_float32_finite_and_ordered():
    from evaluation.synthetic_traffic import generate
    df = generate(seed=0, n_sessions=1, counts={"dns_udp": 3, "https_tcp": 3, "syn_scan_nmap": 1})
    for k in fs.FEATURE_SETS:
        X = fs.feature_matrix(df, k)
        assert list(X.columns) == fs.columns(k) and X.dtypes.eq(np.float32).all()
        assert np.isfinite(X.to_numpy()).all()
    from train_xgboost_ae import FEATURE_COLS
    assert fs.columns("F0") == list(FEATURE_COLS)             # F0 is exactly the production set
