"""
evaluation/splits.py
====================
Train / validation-calibration / final-test splits with recorded fingerprints.

Rules
  * three disjoint parts: TRAIN (fit models), CAL (thresholds + fusion calibration + AE normalisation),
    TEST (final, sealed, touched once);
  * leave-one-family-out (LOFO) for "unseen attack family" generalisation;
  * group splits (session / day / device) for lab data, so related flows never straddle two parts;
  * every split is hashed (`fingerprint`) and stored in the run manifest so a result can be reproduced.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd


class SplitOverlapError(AssertionError):
    pass


def fingerprint(index: Sequence) -> str:
    """Order-independent hash of a set of row identifiers."""
    arr = np.sort(np.asarray(index, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def assert_disjoint(**parts: Sequence) -> None:
    names = list(parts)
    sets = {n: set(np.asarray(parts[n]).tolist()) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                raise SplitOverlapError(f"splits '{a}' and '{b}' share {len(inter)} rows")


@dataclass
class SplitSpec:
    name: str
    train: np.ndarray
    cal: np.ndarray
    test: np.ndarray
    notes: Dict[str, object] = field(default_factory=dict)

    def manifest(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "sizes": {"train": len(self.train), "cal": len(self.cal), "test": len(self.test)},
            "fingerprints": {"train": fingerprint(self.train), "cal": fingerprint(self.cal),
                             "test": fingerprint(self.test)},
            "notes": self.notes,
        }


def chronological_split(df: pd.DataFrame, time_col: str, fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2)) -> SplitSpec:
    """Earliest part trains, the middle part calibrates, the latest part is the final test."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    order = np.argsort(df[time_col].to_numpy(), kind="stable")
    n = len(order)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    spec = SplitSpec("chronological", order[:a], order[a:b], order[b:], {"fractions": list(fractions), "time_col": time_col})
    assert_disjoint(train=spec.train, cal=spec.cal, test=spec.test)
    return spec


def group_split(groups: Sequence, fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2), seed: int = 0) -> SplitSpec:
    """Assign whole groups (session / day / device) to train, cal or test."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 3:
        raise ValueError(f"need at least 3 groups for a train/cal/test group split, got {len(uniq)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    a = max(1, int(round(n * fractions[0])))
    b = max(a + 1, min(n - 1, int(round(n * (fractions[0] + fractions[1])))))
    tr_g, ca_g, te_g = set(uniq[:a]), set(uniq[a:b]), set(uniq[b:])
    idx = np.arange(len(groups))
    spec = SplitSpec(
        "group", idx[np.isin(groups, list(tr_g))], idx[np.isin(groups, list(ca_g))], idx[np.isin(groups, list(te_g))],
        {"seed": seed, "train_groups": sorted(map(str, tr_g)), "cal_groups": sorted(map(str, ca_g)),
         "test_groups": sorted(map(str, te_g))},
    )
    assert_disjoint(train=spec.train, cal=spec.cal, test=spec.test)
    return spec


def leave_one_family_out(df: pd.DataFrame, family_col: str, benign_label: str, time_col: str,
                         cal_fraction: float = 0.2) -> Iterator[SplitSpec]:
    """For each attack family F: TEST = all flows of F (+ the latest benign slice),
    TRAIN/CAL = the other families (+ earlier benign), split chronologically.
    The held-out family is never seen in TRAIN or CAL."""
    fam = df[family_col].astype(str).str.strip().to_numpy()
    t = df[time_col].to_numpy()
    benign = fam == benign_label
    order_b = np.flatnonzero(benign)[np.argsort(t[benign], kind="stable")]
    cut = int(len(order_b) * 0.8)
    benign_dev, benign_test = order_b[:cut], order_b[cut:]
    for f in sorted(set(fam[~benign].tolist())):
        held = np.flatnonzero(fam == f)
        rest_attack = np.flatnonzero(~benign & (fam != f))
        dev = np.concatenate([benign_dev, rest_attack])
        dev = dev[np.argsort(t[dev], kind="stable")]
        k = int(len(dev) * (1 - cal_fraction))
        spec = SplitSpec(f"lofo:{f}", dev[:k], dev[k:], np.concatenate([benign_test, held]),
                         {"held_out_family": f, "cal_fraction": cal_fraction})
        assert_disjoint(train=spec.train, cal=spec.cal, test=spec.test)
        yield spec
