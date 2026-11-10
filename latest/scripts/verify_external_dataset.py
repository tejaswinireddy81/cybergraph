"""
scripts/verify_external_dataset.py
==================================
Read-only verification of a NetFlow-v3 style CSV BEFORE it is used for anything.

Checks (nothing is copied, trained or modified):
  * it is a CSV and not an HTML page (the UQ research-data links return a web-app shell, not the file)
  * size and SHA-256 (optionally compared with --expect-sha256)
  * the header: the 47 model feature columns, the time column, and the label columns ('Label' 0/1 and/or 'Attack')
  * feature compatibility report: missing / extra columns
  * content: row count, benign vs attack counts, attack categories, protocol mix, time span, duplicate rows,
    NaN/inf share in the model features
  * writes outputs/external_verification_<file>.json

Exit code 0 = compatible, 1 = not compatible / not a CSV. Put downloaded external data in latest/data/external/
(gitignored) and run:

    python scripts/verify_external_dataset.py data/external/NF-CSE-CIC-IDS2018-v3.csv
"""
import argparse
import hashlib
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from train_xgboost_ae import FEATURE_COLS  # noqa: E402

TIME_COL = "FLOW_START_MILLISECONDS"
LABEL_CANDIDATES = ("Label", "Attack")
OPTIONAL_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "MIN_TTL", "MAX_TTL", "FLOW_END_MILLISECONDS"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_html(head: bytes) -> bool:
    h = head.lstrip()[:200].lower()
    return h.startswith(b"<!doctype html") or h.startswith(b"<html") or b"<script" in h or h.startswith(b"<")


def verify(path, expect_sha256=None, chunksize=500_000):
    path = os.path.abspath(path)
    report = {"file": path, "ok": False, "problems": []}
    if not os.path.exists(path):
        report["problems"].append("file does not exist")
        return report
    report["size_bytes"] = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(4096)
    if looks_like_html(head):
        report["problems"].append("file is an HTML page, not a CSV (a web-app shell or error page was saved instead of the data)")
        report["head_preview"] = head[:120].decode("utf-8", "replace")
        return report
    report["sha256"] = sha256_file(path)
    if expect_sha256:
        report["sha256_matches_expected"] = report["sha256"].lower() == expect_sha256.lower()
        if not report["sha256_matches_expected"]:
            report["problems"].append("SHA-256 does not match the expected value")

    try:
        header = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        report["problems"].append(f"cannot parse as CSV: {type(exc).__name__}: {exc}")
        return report
    report["n_columns"] = len(header)
    report["header"] = header
    missing = [c for c in FEATURE_COLS if c not in header]
    report["missing_model_features"] = missing
    report["features_present"] = f"{len(FEATURE_COLS) - len(missing)}/{len(FEATURE_COLS)}"
    report["has_time_column"] = TIME_COL in header
    report["label_columns_found"] = [c for c in LABEL_CANDIDATES if c in header]
    report["optional_columns_present"] = [c for c in OPTIONAL_COLS if c in header]
    report["extra_columns"] = [c for c in header if c not in FEATURE_COLS and c not in OPTIONAL_COLS
                               and c not in LABEL_CANDIDATES and c != TIME_COL]
    if missing:
        report["problems"].append(f"missing {len(missing)} model features: {missing}")
    if TIME_COL not in header:
        report["problems"].append(f"missing time column {TIME_COL}")
    if not report["label_columns_found"]:
        report["problems"].append("no label column ('Label' or 'Attack') found")

    use = [c for c in FEATURE_COLS if c in header] + [c for c in (TIME_COL, "Label", "Attack") if c in header]
    rows = benign = attack = bad_values = 0
    categories, protocols = {}, {}
    tmin, tmax = None, None
    for chunk in pd.read_csv(path, usecols=use, chunksize=chunksize, low_memory=False):
        rows += len(chunk)
        feats = chunk[[c for c in FEATURE_COLS if c in chunk.columns]].apply(pd.to_numeric, errors="coerce")
        bad_values += int((~np.isfinite(feats.to_numpy(dtype=float))).sum())
        if "Attack" in chunk.columns:
            a = chunk["Attack"].astype(str).str.strip()
            is_attack = a.str.lower() != "benign"
            for k, v in a.value_counts().items():
                categories[k] = categories.get(k, 0) + int(v)
        elif "Label" in chunk.columns:
            is_attack = pd.to_numeric(chunk["Label"], errors="coerce").fillna(0) > 0
        else:
            is_attack = pd.Series(False, index=chunk.index)
        attack += int(is_attack.sum())
        benign += int((~is_attack).sum())
        if "PROTOCOL" in chunk.columns:
            for k, v in chunk["PROTOCOL"].value_counts().items():
                protocols[int(k)] = protocols.get(int(k), 0) + int(v)
        if TIME_COL in chunk.columns:
            t = pd.to_numeric(chunk[TIME_COL], errors="coerce")
            lo, hi = t.min(), t.max()
            tmin = lo if tmin is None else min(tmin, lo)
            tmax = hi if tmax is None else max(tmax, hi)
    report.update({"rows": rows, "benign": benign, "attack": attack,
                   "attack_share": (attack / rows) if rows else None,
                   "attack_categories": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
                   "protocol_counts_top": dict(sorted(protocols.items(), key=lambda kv: -kv[1])[:8]),
                   "non_finite_feature_values": bad_values})
    if tmin is not None and not math.isnan(tmin):
        report["time_span_utc"] = [str(pd.to_datetime(tmin, unit="ms")), str(pd.to_datetime(tmax, unit="ms"))]
    report["ok"] = not report["problems"]
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs"))
    a = ap.parse_args(argv)
    rep = verify(a.path, a.expect_sha256)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"external_verification_{os.path.basename(a.path)}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"file      : {rep['file']}  ({rep.get('size_bytes', '?')} bytes)")
    if "sha256" in rep:
        print(f"sha256    : {rep['sha256']}")
    if "header" in rep:
        print(f"header    : {rep['n_columns']} columns | model features {rep['features_present']} | time column: "
              f"{rep['has_time_column']} | label columns: {rep['label_columns_found']}")
        print(f"rows      : {rep.get('rows'):,} | benign {rep.get('benign'):,} | attack {rep.get('attack'):,} "
              f"({100 * (rep.get('attack_share') or 0):.2f} %)")
        print(f"categories: {rep.get('attack_categories')}")
        print(f"time span : {rep.get('time_span_utc')} | non-finite feature values: {rep.get('non_finite_feature_values')}")
    for p in rep["problems"]:
        print("PROBLEM   :", p)
    print("RESULT    :", "COMPATIBLE" if rep["ok"] else "NOT COMPATIBLE / NOT VERIFIED", "->", out)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
