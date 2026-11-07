"""
scripts/setup_pipeline.py
=========================
One-command setup and model training utility:
1. Verifies/downloads NF-UNSW-NB15-v3.csv dataset
2. Trains XGBoost and Autoencoder baseline models if not present
3. Trains Dynamic Bipartite Temporal GNN model if not present
4. Precomputes BARI normal route intelligence baseline

Usage:
    python scripts/setup_pipeline.py
"""

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LATEST_DIR = os.path.abspath(os.path.join(HERE, ".."))
MODELS_DIR = os.path.join(LATEST_DIR, "models")
DATASET_PATH = os.path.join(LATEST_DIR, "NF-UNSW-NB15-v3.csv")


def run_cmd(cmd, desc):
    print(f"\n[SETUP] {desc} ...")
    res = subprocess.run(cmd, cwd=LATEST_DIR)
    if res.returncode != 0:
        print(f"[ERROR] Failed during: {desc} (Exit code: {res.returncode})")
        sys.exit(res.returncode)
    print(f"[SUCCESS] {desc} completed successfully.")


def main():
    print("=" * 60)
    print("CYBERGRAPH PIPELINE SETUP & MODEL PREPARATION")
    print("=" * 60)

    # 1. Dataset Check / Download
    if not os.path.exists(DATASET_PATH):
        from download_dataset import check_and_download
        ok = check_and_download()
        if not ok:
            print("[ERROR] Dataset setup failed. Aborting.")
            sys.exit(1)
    else:
        size_mb = os.path.getsize(DATASET_PATH) / (1024 * 1024)
        print(f"[OK] Found dataset NF-UNSW-NB15-v3.csv ({size_mb:.2f} MB)")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 2. Check XGBoost & Autoencoder
    xgb_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    ae_path = os.path.join(MODELS_DIR, "autoencoder_model.keras")
    if not (os.path.exists(xgb_path) and os.path.exists(ae_path)):
        run_cmd([sys.executable, "train_xgboost_ae.py"], "Training XGBoost & Autoencoder Models")
    else:
        print("[OK] XGBoost and Autoencoder models already exist in models/.")

    # 3. Check Dynamic Temporal GNN
    gnn_path = os.path.join(MODELS_DIR, "dynamic_temporal_gnn.pt")
    if not os.path.exists(gnn_path):
        run_cmd([sys.executable, "train_temporal_gnn.py"], "Training Dynamic Bipartite Temporal GNN")
    else:
        print("[OK] Dynamic Temporal GNN model already exists in models/.")

    # 4. Check BARI Baseline Cache
    bari_path = os.path.join(MODELS_DIR, "bari_baseline.json")
    if not os.path.exists(bari_path):
        from modules.bari_engine import BARIEngine
        from train_xgboost_ae import load_dataset
        print("\n[SETUP] Precomputing BARI route intelligence baseline ...")
        df_full = load_dataset()
        df_benign = df_full[df_full["label_enc"] == 0][["IPV4_SRC_ADDR", "L4_DST_PORT", "PROTOCOL"]]
        engine = BARIEngine()
        engine.learn_normal_baseline(df_benign)
        engine.save_baseline(bari_path)
        print(f"[SUCCESS] Saved BARI baseline cache to {bari_path}")
    else:
        print("[OK] BARI baseline cache already exists in models/.")

    print("\n" + "=" * 60)
    print("ALL CYBERGRAPH PIPELINE PREREQUISITES READY!")
    print("Start server with: python app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
