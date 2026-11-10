"""
scripts/preflight_retrain.py
============================
READ-ONLY readiness check to run BEFORE retraining. It never trains, never writes into models/
and never overwrites anything. Optional --backup copies models/ to models_backup/<timestamp>/
(a copy; the originals stay untouched).

Checks
  1. dataset present, size, SHA-256 (compared with datasets/NF-UNSW-NB15-v3.csv.sha256 if present),
     required columns, row count
  2. installed library versions vs requirements-train.txt pins
  3. which library version each saved model was written with, and whether it matches the
     version installed now (mixed versions => retrain candidates)
  4. models/model_manifest.json still matches the files on disk (nothing tampered)
  5. free disk space for a backup
Then it prints the exact commands that WOULD be run after you approve retraining.

Usage
    python scripts/preflight_retrain.py
    python scripts/preflight_retrain.py --backup
    python scripts/preflight_retrain.py --hash        # also compute the dataset SHA-256 (slow, ~1 s/100 MB)
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import re
import shutil
import sys
import time
import warnings
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
MODELS = os.path.join(ROOT, "models")
DATASET = os.path.join(ROOT, "NF-UNSW-NB15-v3.csv")
DATASET_SHA_FILE = os.path.join(ROOT, "datasets", "NF-UNSW-NB15-v3.csv.sha256")
REQ_TRAIN = os.path.join(ROOT, "requirements-train.txt")

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
problems, warnings_found = [], []


def report(level, msg):
    print(f"{level} {msg}")
    if level == FAIL:
        problems.append(msg)
    elif level == WARN:
        warnings_found.append(msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_dataset(do_hash):
    print("\n== 1. Dataset ==")
    if not os.path.exists(DATASET):
        report(FAIL, f"dataset missing: {DATASET} (python scripts/download_dataset.py)")
        return
    report(OK, f"dataset present: {os.path.getsize(DATASET) / 1e6:.1f} MB")
    try:
        import pandas as pd
        from train_xgboost_ae import FEATURE_COLS
        head = pd.read_csv(DATASET, nrows=1)
        need = list(FEATURE_COLS) + ["Attack", "FLOW_START_MILLISECONDS"]
        missing = [c for c in need if c not in head.columns]
        if missing:
            report(FAIL, f"dataset lacks required columns: {missing}")
        else:
            report(OK, f"all {len(FEATURE_COLS)} feature columns + Attack + FLOW_START_MILLISECONDS present")
        with open(DATASET, "rb") as f:
            rows = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b"")) - 1
        report(OK, f"rows: {rows:,}")
    except Exception as exc:  # pragma: no cover - diagnostic path
        report(FAIL, f"could not inspect dataset: {exc}")
    if os.path.exists(DATASET_SHA_FILE):
        expected = open(DATASET_SHA_FILE, encoding="utf-8").read().split()[0].strip().lower()
        actual = sha256(DATASET)
        if actual == expected:
            report(OK, f"dataset SHA-256 matches reference ({actual[:16]}...)")
        else:
            report(FAIL, f"dataset SHA-256 differs from reference: {actual} != {expected}")
    elif do_hash:
        report(WARN, f"no reference hash file; dataset SHA-256 = {sha256(DATASET)}")
    else:
        report(WARN, "no reference hash file (datasets/NF-UNSW-NB15-v3.csv.sha256); pass --hash to compute one")


def parse_pins():
    pins = {}
    if not os.path.exists(REQ_TRAIN):
        return pins
    for line in open(REQ_TRAIN, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)$", line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def check_versions():
    print("\n== 2. Library versions vs requirements-train.txt ==")
    pins = parse_pins()
    if not pins:
        report(WARN, "requirements-train.txt not found or has no == pins")
    for name, want in sorted(pins.items()):
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            report(WARN, f"{name}: not installed (pinned {want})")
            continue
        base = have.split("+")[0]
        if base == want:
            report(OK, f"{name} {have}")
        else:
            report(WARN, f"{name}: installed {have}, pinned {want}")


def saved_with_versions():
    """Return {file: (library, version)} read from the artifacts themselves."""
    import joblib
    out = {}
    for fn in sorted(os.listdir(MODELS)):
        path = os.path.join(MODELS, fn)
        try:
            if fn.endswith(".pkl"):
                # NOTE: never read `_sklearn_version` from a loaded object - __getstate__() stamps the
                # CURRENT version. scikit-learn itself compares the pickled version with the installed
                # one while unpickling and warns on a mismatch; that warning is the reliable signal.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    joblib.load(path)
                saved = set()
                for w in caught:
                    m = re.search(r"from version ([\d.]+) when using version ([\d.]+)", str(w.message))
                    if m:
                        saved.add(m.group(1))
                if saved:
                    out[fn] = ("scikit-learn", "/".join(sorted(saved)))
                else:
                    out[fn] = ("scikit-learn", md.version("scikit-learn").split("+")[0])  # no warning => same version
            elif fn == "xgboost_model.json":
                v = json.load(open(path, encoding="utf-8")).get("version")
                out[fn] = ("xgboost", ".".join(map(str, v)) if v else None)
            elif fn.endswith(".keras"):
                meta = json.loads(zipfile.ZipFile(path).read("metadata.json"))
                out[fn] = ("keras", meta.get("keras_version"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            out[fn] = ("?", f"unreadable: {type(exc).__name__}")
    return out


def check_models():
    print("\n== 3. Version each saved model was written with ==")
    if not os.path.isdir(MODELS):
        report(WARN, "no models/ directory yet (fresh setup: nothing to compare)")
        return
    mixed = set()
    for fn, (lib, ver) in saved_with_versions().items():
        try:
            installed = md.version(lib) if lib != "?" else None
        except md.PackageNotFoundError:
            installed = None
        if ver is None or installed is None:
            report(WARN, f"{fn}: {lib} version unknown/not installed")
        elif ver == installed.split("+")[0]:
            report(OK, f"{fn}: {lib} {ver} (matches installed)")
        else:
            mixed.add(fn)
            report(WARN, f"{fn}: saved with {lib} {ver}, installed {installed}  -> retrain candidate")
    if mixed:
        print(f"   {len(mixed)} artifact(s) differ from the installed library version.")


def check_manifest():
    print("\n== 4. Manifest integrity ==")
    mp = os.path.join(MODELS, "model_manifest.json")
    if not os.path.exists(mp):
        report(WARN, "models/model_manifest.json missing (generated by retraining / setup_pipeline.py)")
        return
    manifest = json.load(open(mp, encoding="utf-8"))
    bad = 0
    for fn, entry in manifest.items():
        p = os.path.join(MODELS, fn)
        if not os.path.exists(p):
            report(FAIL, f"manifest lists {fn} but the file is missing")
            bad += 1
        elif sha256(p) != entry.get("sha256"):
            report(FAIL, f"{fn} does not match the manifest hash (modified?)")
            bad += 1
    if not bad:
        report(OK, f"all {len(manifest)} manifest entries match the files on disk")


def check_disk_and_backup(do_backup):
    print("\n== 5. Backup / disk ==")
    if not os.path.isdir(MODELS):
        return
    size = sum(os.path.getsize(os.path.join(MODELS, f)) for f in os.listdir(MODELS))
    free = shutil.disk_usage(ROOT).free
    report(OK if free > size * 3 else FAIL, f"models/ is {size / 1e6:.1f} MB; free disk {free / 1e9:.1f} GB")
    if do_backup:
        dest = os.path.join(ROOT, "models_backup", time.strftime("%Y%m%d-%H%M%S"))
        shutil.copytree(MODELS, dest)
        report(OK, f"backup written to {dest} (originals untouched)")
    else:
        print("   (no backup made; pass --backup to copy models/ aside before any retraining)")


def print_plan():
    print("\n== Retraining plan (NOT executed; needs your approval) ==")
    print("  0. python scripts/preflight_retrain.py --backup")
    print("  1. python train_xgboost_ae.py      # XGBoost + autoencoder + scaler + fusion (writes models/)")
    print("  2. python train_temporal_gnn.py    # dynamic temporal GNN + gnn_scalers.pkl")
    print("  3. python scripts/setup_pipeline.py  # BARI baseline + regenerates models/model_manifest.json")
    print("  4. python -m pytest tests -q       # full suite, models present")
    print("  NOTE: retraining on the same features/data reproduces the shortcut-learning problem described in")
    print("        docs/false_positive_investigation.md. Decide the feature/evaluation changes first.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backup", action="store_true", help="copy models/ to models_backup/<timestamp>/")
    ap.add_argument("--hash", action="store_true", help="compute the dataset SHA-256 when no reference file exists")
    args = ap.parse_args()
    print(f"Preflight for retraining  |  python {sys.version.split()[0]}  |  root: {ROOT}")
    check_dataset(args.hash)
    check_versions()
    check_models()
    check_manifest()
    check_disk_and_backup(args.backup)
    print_plan()
    print(f"\nSummary: {len(problems)} failure(s), {len(warnings_found)} warning(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
