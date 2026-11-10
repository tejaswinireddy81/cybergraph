"""
config/paths.py
===============
Where models, candidate models, backups and lab captures live, plus the guard that stops a
training run from overwriting production models.

Rules enforced here
  * Default target stays the production ``models/`` directory (existing behaviour).
  * A different target is chosen with ``--models-dir`` or ``CYBERGRAPH_MODELS_DIR`` (candidate runs
    normally use ``models_candidates/<run_id>/``).
  * Writing into a production directory that already holds models is refused unless the caller passes
    ``overwrite_production=True`` (``--overwrite-production``) or sets
    ``CYBERGRAPH_ALLOW_PRODUCTION_OVERWRITE=1``; even then a backup copy is taken first.
  * A candidate directory that already holds models is never overwritten.
  * Every manifest is written NEXT TO the models it describes (never into another directory).
"""
import os
import shutil
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
LATEST_DIR = os.path.abspath(os.path.join(HERE, ".."))

PRODUCTION_MODELS_DIR = os.path.join(LATEST_DIR, "models")
CANDIDATES_ROOT = os.path.join(LATEST_DIR, "models_candidates")
BACKUPS_ROOT = os.path.join(LATEST_DIR, "models_backup")
LAB_CAPTURES_DIR = os.path.join(LATEST_DIR, "data", "lab_captures")

MODELS_DIR_ENV = "CYBERGRAPH_MODELS_DIR"
ALLOW_OVERWRITE_ENV = "CYBERGRAPH_ALLOW_PRODUCTION_OVERWRITE"
MODEL_EXTS = (".pkl", ".keras", ".pt", ".json", ".joblib")
MANIFEST_NAME = "model_manifest.json"


class ProductionOverwriteError(RuntimeError):
    """Raised when a run would overwrite existing models without explicit permission."""


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def is_production_dir(path: str) -> bool:
    return _norm(path) == _norm(PRODUCTION_MODELS_DIR)


def has_models(path: str) -> bool:
    """True if the directory already contains model artifacts (the manifest alone does not count)."""
    if not os.path.isdir(path):
        return False
    return any(
        f.endswith(MODEL_EXTS) and f != MANIFEST_NAME and os.path.isfile(os.path.join(path, f))
        for f in os.listdir(path)
    )


def resolve_models_dir(cli_value: Optional[str] = None) -> str:
    """CLI value > CYBERGRAPH_MODELS_DIR > production models/ (the historical default)."""
    chosen = cli_value or os.environ.get(MODELS_DIR_ENV) or PRODUCTION_MODELS_DIR
    return os.path.abspath(chosen)


def manifest_path_for(models_dir: str) -> str:
    return os.path.join(os.path.abspath(models_dir), MANIFEST_NAME)


def new_candidate_dir(run_id: Optional[str] = None) -> str:
    """Create (and return) a fresh, empty models_candidates/<run_id>/ directory."""
    run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S")
    path = os.path.join(CANDIDATES_ROOT, run_id)
    if os.path.exists(path) and os.listdir(path):
        raise FileExistsError(f"candidate directory already exists and is not empty: {path}")
    os.makedirs(path, exist_ok=True)
    return path


def backup_models(models_dir: str) -> str:
    """Copy a models directory to models_backup/<timestamp>/ (originals untouched)."""
    dest = os.path.join(BACKUPS_ROOT, time.strftime("%Y%m%d-%H%M%S"))
    n = 1
    while os.path.exists(dest):
        n += 1
        dest = os.path.join(BACKUPS_ROOT, time.strftime("%Y%m%d-%H%M%S") + f"-{n}")
    shutil.copytree(models_dir, dest)
    return dest


def guard_output_dir(models_dir: str, overwrite_production: bool = False, targets=None) -> dict:
    """Validate (and prepare) an output directory BEFORE any file is written. Returns an info dict.

    targets : file names this run is about to write. Only those are checked, so several components
              (XGBoost/AE, GNN, GAT ...) can be written one after another into the same candidate
              directory. If omitted, "any model already present" counts as a collision.
    """
    models_dir = os.path.abspath(models_dir)
    info = {"models_dir": models_dir, "production": is_production_dir(models_dir), "backup": None}
    if targets is None:
        collision = has_models(models_dir)
    else:
        collision = any(os.path.exists(os.path.join(models_dir, f)) for f in targets)
    if info["production"]:
        if collision:
            allowed = overwrite_production or os.environ.get(ALLOW_OVERWRITE_ENV) == "1"
            if not allowed:
                raise ProductionOverwriteError(
                    f"Refusing to overwrite the existing production models in {models_dir}. "
                    f"To train a candidate instead use --models-dir models_candidates/<run_id> "
                    f"(or set {MODELS_DIR_ENV}). "
                    f"To really replace production use --overwrite-production "
                    f"(or {ALLOW_OVERWRITE_ENV}=1); a backup is taken first."
                )
            info["backup"] = backup_models(models_dir)
    elif collision:
        raise ProductionOverwriteError(
            f"Refusing to overwrite existing model files in candidate directory {models_dir}; use a new run id."
        )
    os.makedirs(models_dir, exist_ok=True)
    return info
