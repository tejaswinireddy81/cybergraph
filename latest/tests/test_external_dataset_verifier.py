"""The external-dataset verifier must catch HTML shells, missing columns, bad checksums, and accept a compatible file."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import verify_external_dataset as V  # noqa: E402

from train_xgboost_ae import FEATURE_COLS  # noqa: E402


def make_csv(path, drop=(), label="both", n=50, nan_cell=False):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.integers(0, 100, n).astype(float) for c in FEATURE_COLS})
    df["PROTOCOL"] = rng.choice([6, 17], n)
    df["FLOW_START_MILLISECONDS"] = 1_500_000_000_000 + np.arange(n) * 1000
    df["Attack"] = np.where(np.arange(n) % 5 == 0, "DoS", "Benign")
    df["Label"] = (df["Attack"] != "Benign").astype(int)
    if label == "attack_only":
        df = df.drop(columns=["Label"])
    if label == "none":
        df = df.drop(columns=["Label", "Attack"])
    df = df.drop(columns=list(drop))
    if nan_cell:
        df.loc[0, "IN_BYTES"] = np.nan
    df.to_csv(path, index=False)
    return path


def run(path, **kw):
    return V.verify(str(path), **kw)


def test_compatible_csv_is_accepted_with_counts(tmp_path):
    r = run(make_csv(tmp_path / "ok.csv"))
    assert r["ok"] and r["features_present"] == "47/47" and r["label_columns_found"] == ["Label", "Attack"]
    assert r["rows"] == 50 and r["attack"] == 10 and r["benign"] == 40 and r["attack_categories"] == {"Benign": 40, "DoS": 10}
    assert len(r["sha256"]) == 64 and r["non_finite_feature_values"] == 0 and r["has_time_column"]


def test_html_web_app_shell_saved_as_csv_is_rejected(tmp_path):
    f = tmp_path / "NF-CSE-CIC-IDS2018-v3.csv"
    f.write_text("<!doctype html><html><head><script src='/assets/index.js'></script></head><body></body></html>")
    r = run(f)
    assert not r["ok"] and any("HTML page" in p for p in r["problems"]) and "sha256" not in r


def test_missing_feature_columns_are_reported_by_name(tmp_path):
    r = run(make_csv(tmp_path / "miss.csv", drop=("MIN_IP_PKT_LEN", "DNS_TTL_ANSWER")))
    assert not r["ok"] and r["features_present"] == "45/47"
    assert r["missing_model_features"] == ["MIN_IP_PKT_LEN", "DNS_TTL_ANSWER"]


def test_missing_time_or_label_columns_are_problems(tmp_path):
    r = run(make_csv(tmp_path / "nolabel.csv", label="none"))
    assert not r["ok"] and any("no label column" in p for p in r["problems"])
    r2 = run(make_csv(tmp_path / "notime.csv", drop=("FLOW_START_MILLISECONDS",)))
    assert not r2["ok"] and any("time column" in p for p in r2["problems"])


def test_attack_only_label_column_is_enough(tmp_path):
    r = run(make_csv(tmp_path / "a.csv", label="attack_only"))
    assert r["ok"] and r["label_columns_found"] == ["Attack"] and r["attack"] == 10


def test_checksum_mismatch_is_flagged_and_match_is_confirmed(tmp_path):
    f = make_csv(tmp_path / "c.csv")
    good = V.sha256_file(str(f))
    assert run(f, expect_sha256=good)["sha256_matches_expected"] is True
    bad = run(f, expect_sha256="0" * 64)
    assert not bad["ok"] and bad["sha256_matches_expected"] is False


def test_non_finite_values_are_counted(tmp_path):
    assert run(make_csv(tmp_path / "n.csv", nan_cell=True))["non_finite_feature_values"] == 1


def test_missing_file_and_cli_exit_codes(tmp_path, capsys):
    assert not run(tmp_path / "nope.csv")["ok"]
    good = make_csv(tmp_path / "g.csv")
    assert V.main([str(good), "--out-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "external_verification_g.csv.json").read_text())["ok"] is True
    html = tmp_path / "h.csv"
    html.write_text("<html></html>")
    assert V.main([str(html), "--out-dir", str(tmp_path)]) == 1
    assert "NOT COMPATIBLE" in capsys.readouterr().out
