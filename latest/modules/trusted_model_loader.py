"""
modules/trusted_model_loader.py
===============================
Cryptographic Model Integrity Verification & Safe Deserialization Gateway.

Security Policy:
1. Every model, weights checkpoint, and scaler must match its SHA-256 signature
   defined in the trusted manifest (models/model_manifest.json).
2. Raises ModelTamperingException if the file is absent, modified, or has an invalid hash.
3. Enforces weights_only=True on PyTorch torch.load() operations.
4. Prevents arbitrary remote code execution via untrusted pickle deserialization.
"""

import os
import json
import hashlib
import logging
from typing import Any, Optional
import joblib
import torch

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "models"))
MANIFEST_PATH = os.path.join(MODELS_DIR, "model_manifest.json")


class ModelTamperingException(Exception):
    """Raised when a model file has been modified, corrupted, or failed SHA-256 integrity checks."""
    pass


def compute_sha256(filepath: str) -> str:
    """Compute hex-encoded SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _adjacent_manifest(filepath: str) -> str:
    """Manifest that sits next to the model file (production models/ -> models/model_manifest.json;
    a candidate directory -> its own manifest). Never falls back to another directory's manifest."""
    return os.path.join(os.path.dirname(os.path.abspath(filepath)), os.path.basename(MANIFEST_PATH))


def verify_model_integrity(filepath: str, manifest_path: Optional[str] = None) -> bool:
    """Verify that a model file matches its trusted manifest SHA-256 digest.
    
    Fail-closed policy:
    1. If the manifest file does not exist, loading is strictly aborted (fail closed).
    2. If the model file is not registered in the manifest, loading is strictly aborted.
    3. If the actual SHA-256 digest does not match the expected digest, loading is strictly aborted.
    4. In production (CYBERGRAPH_ENV=production or FLASK_ENV=production), fail-closed is strictly enforced
       and CYBERGRAPH_ALLOW_UNVERIFIED_MODELS is ignored.
    """
    if manifest_path is None:
        manifest_path = _adjacent_manifest(filepath)

    is_prod = (
        os.environ.get("CYBERGRAPH_ENV", "").lower() in ("prod", "production") or
        os.environ.get("FLASK_ENV", "").lower() == "production"
    )

    if not is_prod:
        # Check if bypass requested for local training/unverified model experiments
        if os.environ.get("CYBERGRAPH_ALLOW_UNVERIFIED_MODELS") == "1":
            logger.warning(
                f"[SECURITY NOTICE] CYBERGRAPH_ALLOW_UNVERIFIED_MODELS is active in non-production. "
                f"Skipping integrity verification for '{filepath}'."
            )
            return True

    if not os.path.exists(manifest_path):
        err_msg = (
            f"[CRITICAL SECURITY FAILURE] Model manifest not found at '{manifest_path}'. "
            "Refusing to load unverified model binaries under fail-closed security policy."
        )
        logger.critical(err_msg)
        raise ModelTamperingException(err_msg)

    if not os.path.exists(filepath):
        err_msg = f"[MODEL NOT FOUND] Target model file does not exist: '{filepath}'"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    filename = os.path.basename(filepath)
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        err_msg = f"[SECURITY INTEGRITY VIOLATION] Unable to parse model manifest at '{manifest_path}': {e}"
        logger.critical(err_msg)
        raise ModelTamperingException(err_msg)

    if filename not in manifest:
        # File is not registered in trusted manifest
        err_msg = f"[SECURITY INTEGRITY VIOLATION] Unregistered model file: '{filename}' not found in trusted manifest."
        logger.error(err_msg)
        raise ModelTamperingException(err_msg)

    expected_sha256 = manifest[filename]["sha256"]
    actual_sha256 = compute_sha256(filepath)

    if actual_sha256 != expected_sha256:
        err_msg = (
            f"[CRITICAL SECURITY ALERT] Hash mismatch on model file '{filename}'! "
            f"Expected: {expected_sha256}, Actual: {actual_sha256}. Possible tampering detected!"
        )
        logger.critical(err_msg)
        raise ModelTamperingException(err_msg)

    return True


def safe_load_joblib(filepath: str, manifest_path: Optional[str] = None) -> Any:
    """Cryptographically verify file integrity before calling joblib.load()."""
    verify_model_integrity(filepath, manifest_path=manifest_path)
    return joblib.load(filepath)


def safe_load_torch(filepath: str, weights_only: bool = True, map_location: str = "cpu", manifest_path: Optional[str] = None) -> Any:
    """Cryptographically verify file integrity and enforce weights_only=True on torch.load()."""
    verify_model_integrity(filepath, manifest_path=manifest_path)
    if not weights_only:
        # Only for artifacts that are not plain tensors (e.g. the temporal graph dataset). The
        # SHA-256 manifest check above is what makes this acceptable; never use it on unverified files.
        logger.warning("Loading %s with weights_only=False (integrity-verified pickle)", os.path.basename(filepath))
    return torch.load(filepath, map_location=map_location, weights_only=weights_only)  # nosec B614 - manifest-verified, default is weights_only=True


def regenerate_model_manifest(models_dir: str = MODELS_DIR, manifest_path: Optional[str] = None) -> dict:
    """Scan all model files, weights, and scalers in models_dir, compute SHA-256 and file sizes,
    and update model_manifest.json.
    """
    if manifest_path is None:
        # Written next to the models it describes, so a candidate run can never touch the production manifest.
        manifest_path = os.path.join(models_dir, os.path.basename(MANIFEST_PATH))
    valid_exts = (".pkl", ".keras", ".pt", ".json", ".joblib")
    manifest = {}
    if os.path.exists(models_dir):
        for fname in sorted(os.listdir(models_dir)):
            if fname == os.path.basename(manifest_path):
                continue
            fpath = os.path.join(models_dir, fname)
            if os.path.isfile(fpath) and any(fname.endswith(ext) for ext in valid_exts):
                manifest[fname] = {
                    "sha256": compute_sha256(fpath),
                    "size": os.path.getsize(fpath),
                }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"[TRUSTED LOADER] Regenerated model manifest at '{manifest_path}' with {len(manifest)} models.")
    return manifest
