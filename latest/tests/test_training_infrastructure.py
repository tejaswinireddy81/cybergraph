"""Training infrastructure: production-overwrite guard, candidate directories, manifests, floor switch,
synthetic traffic, and a smoke run of the candidate pipeline."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pytest

from config import paths
from modules import trusted_model_loader as tml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_prod(tmp_path, monkeypatch):
    """A stand-in 'production' models/ dir holding a model, so no real file can be touched."""
    prod = tmp_path / "models"
    prod.mkdir()
    (prod / "scaler.pkl").write_bytes(b"production-bytes")
    (prod / "xgboost_model.json").write_text("{}")
    monkeypatch.setattr(paths, "PRODUCTION_MODELS_DIR", str(prod))
    monkeypatch.setattr(paths, "BACKUPS_ROOT", str(tmp_path / "backups"))
    monkeypatch.setattr(paths, "CANDIDATES_ROOT", str(tmp_path / "candidates"))
    monkeypatch.delenv(paths.ALLOW_OVERWRITE_ENV, raising=False)
    monkeypatch.delenv(paths.MODELS_DIR_ENV, raising=False)
    return prod


def _digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ──────────────────────────────────────────────────────────────── guard
def test_resolve_models_dir_priority(fake_prod, monkeypatch, tmp_path):
    assert paths.resolve_models_dir() == str(fake_prod)                       # historical default
    monkeypatch.setenv(paths.MODELS_DIR_ENV, str(tmp_path / "from_env"))
    assert paths.resolve_models_dir() == str(tmp_path / "from_env")
    assert paths.resolve_models_dir(str(tmp_path / "cli")) == str(tmp_path / "cli")   # CLI wins


def test_overwriting_existing_production_models_is_refused(fake_prod):
    before = _digest(fake_prod / "scaler.pkl")
    with pytest.raises(paths.ProductionOverwriteError):
        paths.guard_output_dir(str(fake_prod), targets=["scaler.pkl"])
    with pytest.raises(paths.ProductionOverwriteError):
        paths.guard_output_dir(str(fake_prod))                                # no targets: any model present
    assert _digest(fake_prod / "scaler.pkl") == before


def test_new_files_in_production_are_allowed_but_existing_ones_are_not(fake_prod):
    info = paths.guard_output_dir(str(fake_prod), targets=["gat_model.pt"])   # does not exist yet -> no overwrite
    assert info["backup"] is None
    with pytest.raises(paths.ProductionOverwriteError):
        paths.guard_output_dir(str(fake_prod), targets=["gat_model.pt", "scaler.pkl"])


def test_explicit_production_overwrite_takes_a_backup_first(fake_prod, tmp_path):
    info = paths.guard_output_dir(str(fake_prod), overwrite_production=True, targets=["scaler.pkl"])
    assert info["backup"] and (Path(info["backup"]) / "scaler.pkl").read_bytes() == b"production-bytes"
    assert (fake_prod / "scaler.pkl").read_bytes() == b"production-bytes"      # guard itself never modifies


def test_env_var_permission_also_backs_up(fake_prod, monkeypatch):
    monkeypatch.setenv(paths.ALLOW_OVERWRITE_ENV, "1")
    assert paths.guard_output_dir(str(fake_prod), targets=["scaler.pkl"])["backup"]


def test_candidate_dirs_hold_several_components_but_never_overwrite(fake_prod, tmp_path):
    cand = paths.new_candidate_dir("run-a")
    assert not paths.is_production_dir(cand) and str(tmp_path / "candidates") in cand
    paths.guard_output_dir(cand, targets=["xgboost_model.json", "scaler.pkl"])
    Path(cand, "xgboost_model.json").write_text("{}")
    paths.guard_output_dir(cand, targets=["dynamic_temporal_gnn.pt"])          # a different component: fine
    with pytest.raises(paths.ProductionOverwriteError):
        paths.guard_output_dir(cand, targets=["xgboost_model.json"])           # same file twice: refused
    with pytest.raises(FileExistsError):
        paths.new_candidate_dir("run-a")                                       # run ids are not reusable


def test_every_training_entry_point_refuses_to_overwrite_production(fake_prod, monkeypatch):
    import train_xgboost_ae
    import train_temporal_gnn
    import train_gnn
    with pytest.raises(paths.ProductionOverwriteError):
        train_xgboost_ae.train_pipeline(models_dir=str(fake_prod))
    with pytest.raises(paths.ProductionOverwriteError):
        train_xgboost_ae._main(["--models-dir", str(fake_prod)])
    monkeypatch.setattr(paths, "has_models", paths.has_models)
    (fake_prod / "dynamic_temporal_gnn.pt").write_bytes(b"x")
    with pytest.raises(paths.ProductionOverwriteError):
        train_temporal_gnn.train_gnn(models_dir=str(fake_prod))
    (fake_prod / "gat_model.pt").write_bytes(b"x")
    with pytest.raises(paths.ProductionOverwriteError):
        train_gnn.main(models_dir=str(fake_prod))
    assert (fake_prod / "scaler.pkl").read_bytes() == b"production-bytes"


# ──────────────────────────────────────────────────────────────── manifests / trusted loader
def test_manifest_is_written_next_to_the_models_never_into_production(tmp_path):
    prod_manifest = Path(tml.MANIFEST_PATH)
    before = _digest(prod_manifest) if prod_manifest.exists() else None
    cand = tmp_path / "cand"
    cand.mkdir()
    joblib.dump({"k": 1}, cand / "thing.pkl")
    manifest = tml.regenerate_model_manifest(str(cand))                          # no explicit manifest_path
    assert (cand / "model_manifest.json").exists() and "thing.pkl" in manifest
    after = _digest(prod_manifest) if prod_manifest.exists() else None
    assert before == after                                                       # production manifest untouched


def test_candidate_models_verify_against_their_own_manifest_and_tampering_is_caught(tmp_path):
    cand = tmp_path / "cand"
    cand.mkdir()
    f = cand / "scaler.pkl"
    joblib.dump({"k": 1}, f)
    tml.regenerate_model_manifest(str(cand))
    assert tml.safe_load_joblib(str(f)) == {"k": 1}                              # adjacent manifest is used
    f.write_bytes(f.read_bytes() + b"tamper")
    with pytest.raises(tml.ModelTamperingException):
        tml.safe_load_joblib(str(f))


def test_model_in_a_directory_without_a_manifest_fails_closed(tmp_path):
    f = tmp_path / "orphan.pkl"
    joblib.dump(1, f)
    with pytest.raises(tml.ModelTamperingException):
        tml.safe_load_joblib(str(f))


# ──────────────────────────────────────────────────────────────── TCP floor switch
def _syn_min_len(floor):
    from scapy.all import IP, TCP, Ether
    from flow_extractor import FlowExtractor
    ex = FlowExtractor() if floor is None else FlowExtractor(min_pkt_floor_tcp=floor)
    ex.add_packet(Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=4000, dport=80, flags="S"))
    return [ex._export(f)["MIN_IP_PKT_LEN"] for f in ex._flows.values()][0]


def test_floor_default_is_unchanged_and_can_be_switched_off():
    assert _syn_min_len(None) == 52.0          # historical default preserved
    assert _syn_min_len(52) == 52.0
    assert _syn_min_len(0) == 40.0             # off: the true wire length of a bare SYN


# ──────────────────────────────────────────────────────────────── synthetic traffic
SMALL = {"dns_udp": 4, "https_tcp": 3, "quic_udp": 2, "igmp": 2, "syn_scan_bare": 1, "udp_flood": 1}


def test_synthetic_traffic_is_deterministic_and_labelled():
    from evaluation.synthetic_traffic import generate
    a, b = generate(seed=5, n_sessions=2, counts=SMALL), generate(seed=5, n_sessions=2, counts=SMALL)
    cols = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "MIN_IP_PKT_LEN", "FLOW_DURATION_MILLISECONDS"]
    assert a[cols].equals(b[cols]) and len(a) > 0
    assert set(a.loc[a._label == 1, "_traffic_class"]) <= {"syn_scan_bare", "udp_flood"}
    assert set(a.loc[a._label == 0, "_traffic_class"]) <= {"dns_udp", "https_tcp", "quic_udp", "igmp"}
    assert a._synthetic.all() and generate(seed=6, n_sessions=2, counts=SMALL)[cols].equals(a[cols]) is False


def test_floor_switch_changes_only_tcp_rows_of_the_same_synthetic_traffic():
    from evaluation.synthetic_traffic import generate
    on, off = generate(seed=2, n_sessions=1, counts=SMALL, floor=52), generate(seed=2, n_sessions=1, counts=SMALL, floor=0)
    assert on.shape == off.shape
    differ = on.MIN_IP_PKT_LEN != off.MIN_IP_PKT_LEN
    assert differ.any() and (on.loc[differ, "PROTOCOL"] == 6).all()


def test_synthetic_generation_never_sends_packets(monkeypatch):
    import scapy.sendrecv as sr

    def boom(*a, **k):
        raise AssertionError("synthetic traffic must stay in memory")
    for name in ("send", "sendp", "sr", "sr1", "srp", "srp1"):
        monkeypatch.setattr(sr, name, boom, raising=False)
    from evaluation.synthetic_traffic import generate
    assert len(generate(seed=1, n_sessions=1, counts={"dns_udp": 2, "syn_scan_bare": 1})) > 0


def test_characterization_extractor_creates_orphan_flows_after_the_first_fin():
    """Documents CURRENT extractor behaviour (it exports at the first FIN, so the closing ACKs start new one-packet
    flows). If the extractor is fixed this test should be updated deliberately."""
    from evaluation.synthetic_traffic import generate
    df = generate(seed=3, n_sessions=1, counts={"https_tcp": 6})
    orphans = df[(df.PROTOCOL == 6) & ((df.IN_PKTS + df.OUT_PKTS) <= 1)]
    assert len(orphans) >= 6 and len(df) > 6 * 1.5


def test_all_documented_traffic_classes_exist():
    from evaluation.synthetic_traffic import ATTACK_CLASSES, BENIGN_CLASSES, GENERATORS
    for needed in ("dns_udp", "dns_tcp", "https_tcp", "quic_udp", "http_tcp", "other_tcp", "ntp_udp", "dhcp_udp",
                   "icmp_ping", "igmp", "mdns_udp", "ssdp_udp"):
        assert needed in BENIGN_CLASSES and needed in GENERATORS
    assert set(ATTACK_CLASSES) <= set(GENERATORS) and len(ATTACK_CLASSES) >= 6


# ──────────────────────────────────────────────────────────────── end-to-end smoke run of the candidate pipeline
@pytest.mark.requires_models
def test_candidate_smoke_run_writes_only_to_its_own_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CANDIDATES_ROOT", str(tmp_path / "candidates"))
    prod_dir = Path(paths.PRODUCTION_MODELS_DIR)
    before = {p.name: _digest(p) for p in prod_dir.iterdir() if p.is_file()}
    spec = importlib.util.spec_from_file_location("rc_smoke", ROOT / "scripts" / "experiments" / "run_candidate.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    out_dir, results = rc.run("F1", run_id="pytest-smoke", max_rows=6000, with_synthetic=True)
    out = Path(out_dir)
    assert str(tmp_path) in str(out)
    for name in ("xgboost_model.json", "scaler.pkl", "ae_model.pkl", "fusion_model.pkl", "model_manifest.json",
                 "candidate_artifacts.json", "metrics.json", "run_manifest.json", "sealed_access.jsonl"):
        assert (out / name).exists(), name
    man = json.loads((out / "run_manifest.json").read_text())
    assert [l["set"] for l in man["sealed_looks"]] == ["unsw_chrono_test", "synthetic_modern"]    # one look each
    assert sum(man["split"]["sizes"].values()) == man["dataset"]["rows_used"] == 6000
    art = json.loads((out / "candidate_artifacts.json").read_text())
    assert "MIN_IP_PKT_LEN" not in art["feature_columns"] and art["feature_set"] == "F1"
    for model in ("xgb", "ae", "fusion"):                                      # individual models AND the pipeline
        r = results["unsw_chrono_test"][model]
        assert {"roc_auc", "pr_auc", "at_threshold", "per_group", "ci"} <= set(r)
    assert tml.safe_load_joblib(str(out / "scaler.pkl")) is not None            # loads via its own manifest
    after = {p.name: _digest(p) for p in prod_dir.iterdir() if p.is_file()}
    assert before == after                                                      # production byte-identical
    with pytest.raises(FileExistsError):
        rc.run("F1", run_id="pytest-smoke", max_rows=6000)                      # never overwrite a candidate
