"""
config/feature_sets.py
======================
Feature sets F0-F5 for the model-improvement study (see docs/model_improvement_plan.md).

Every set is built from the SAME base columns that both the NetFlow datasets and the live
FlowExtractor provide, so the comparison isolates what is removed or derived.

  F0  production set (47)                         baseline, must reproduce today's behaviour
  F1  F0 - MIN_IP_PKT_LEN, SHORTEST_FLOW_PKT      the direct "with vs without MIN_IP_PKT_LEN" answer
  F2  F1 - packet-size family                     removes the whole size signature
  F3  F2 - identity-like fields                   removes ports / DNS id+ttl / L7 / ICMP type / FTP code
  F4  behaviour-only: F3 - rate columns - PROTOCOL - raw bytes/pkts, + scale-robust derived features
  F5  F4 + coarse port buckets

Derived features are computed by `derive_features(df)` from base columns only.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from train_xgboost_ae import FEATURE_COLS as F0_COLUMNS  # the production 47, single source of truth

SIZE_SIGNATURE = ["MIN_IP_PKT_LEN", "SHORTEST_FLOW_PKT"]
PACKET_SIZE_FAMILY = [
    "MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
]
IDENTITY_LIKE = [
    "L4_SRC_PORT", "L4_DST_PORT", "L7_PROTO", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "FTP_COMMAND_RET_CODE",
]
RATE_COLUMNS = [
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
]
RAW_VOLUME = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS"]

DERIVED_F4 = [
    "log_in_bytes", "log_out_bytes", "log_in_pkts", "log_out_pkts", "log_duration_ms",
    "bytes_per_pkt_in", "bytes_per_pkt_out", "log_out_in_byte_ratio",
    "is_tcp", "is_udp", "is_icmp", "is_other_proto",
]
DERIVED_F5_EXTRA = ["src_port_bucket", "dst_port_bucket"]


def _without(cols: List[str], drop: List[str]) -> List[str]:
    d = set(drop)
    return [c for c in cols if c not in d]


F1_BASE = _without(F0_COLUMNS, SIZE_SIGNATURE)
F2_BASE = _without(F1_BASE, PACKET_SIZE_FAMILY)
F3_BASE = _without(F2_BASE, IDENTITY_LIKE)
F4_BASE = _without(F3_BASE, RATE_COLUMNS + ["PROTOCOL"] + RAW_VOLUME)

FEATURE_SETS: Dict[str, Dict[str, object]] = {
    "F0": {"base": F0_COLUMNS, "derived": []},
    "F1": {"base": F1_BASE, "derived": []},
    "F2": {"base": F2_BASE, "derived": []},
    "F3": {"base": F3_BASE, "derived": []},
    "F4": {"base": F4_BASE, "derived": DERIVED_F4},
    "F5": {"base": F4_BASE, "derived": DERIVED_F4 + DERIVED_F5_EXTRA},
}


def columns(set_id: str) -> List[str]:
    """Final model-input column order for a feature set."""
    fs = FEATURE_SETS[set_id]
    return list(fs["base"]) + list(fs["derived"])


def _popcount(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(np.int64).map(lambda v: bin(int(v)).count("1"))


def _bucket(port: pd.Series) -> pd.Series:
    p = port.fillna(0).astype(np.int64)
    return np.where(p < 1024, 0, np.where(p < 49152, 1, 2))


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-robust features computed from base columns only (no labels, no future information)."""
    f = lambda c: df[c].astype(float).fillna(0.0)  # noqa: E731
    out = pd.DataFrame(index=df.index)
    out["log_in_bytes"] = np.log1p(f("IN_BYTES"))
    out["log_out_bytes"] = np.log1p(f("OUT_BYTES"))
    out["log_in_pkts"] = np.log1p(f("IN_PKTS"))
    out["log_out_pkts"] = np.log1p(f("OUT_PKTS"))
    out["log_duration_ms"] = np.log1p(f("FLOW_DURATION_MILLISECONDS"))
    out["bytes_per_pkt_in"] = f("IN_BYTES") / np.maximum(f("IN_PKTS"), 1.0)
    out["bytes_per_pkt_out"] = f("OUT_BYTES") / np.maximum(f("OUT_PKTS"), 1.0)
    out["log_out_in_byte_ratio"] = np.log((f("OUT_BYTES") + 1.0) / (f("IN_BYTES") + 1.0))
    proto = df["PROTOCOL"].astype(float).fillna(0).astype(int)
    out["is_tcp"] = (proto == 6).astype(float)
    out["is_udp"] = (proto == 17).astype(float)
    out["is_icmp"] = (proto == 1).astype(float)
    out["is_other_proto"] = (~proto.isin([1, 6, 17])).astype(float)
    out["src_port_bucket"] = _bucket(df["L4_SRC_PORT"]).astype(float)
    out["dst_port_bucket"] = _bucket(df["L4_DST_PORT"]).astype(float)
    return out


def feature_matrix(df: pd.DataFrame, set_id: str) -> pd.DataFrame:
    """Model input for `set_id` as float32 in the canonical column order."""
    fs = FEATURE_SETS[set_id]
    base = df[list(fs["base"])].apply(pd.to_numeric, errors="coerce")
    base = base.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if fs["derived"]:
        der = derive_features(df)[list(fs["derived"])]
        base = pd.concat([base, der], axis=1)
    return base[columns(set_id)].astype(np.float32)
