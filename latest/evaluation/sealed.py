"""
evaluation/sealed.py
====================
Enforces "never use the final test data to train or calibrate".

A SealedTestSet hides its data. The only way to read it is `evaluate(fn, purpose)`, which
  * counts and logs every look (so the number of times a test set was used is on record),
  * refuses a second look unless the caller explicitly allows it (`max_looks`).
Fitting / calibration helpers call `ensure_not_sealed(...)` and raise if handed a sealed set.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, List, Optional


class SealedAccessError(RuntimeError):
    """Raised when sealed test data is accessed outside `evaluate`, or used for fitting/calibration."""


class SealedTestSet:
    def __init__(self, data: Any, name: str, log_path: Optional[str] = None, max_looks: int = 1):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_log_path", log_path)
        object.__setattr__(self, "_max_looks", int(max_looks))
        object.__setattr__(self, "_looks", [])

    # -- anything that looks like reading the data is refused -------------------------------
    def __getattr__(self, item):
        raise SealedAccessError(f"sealed test set '{self.name}': attribute '{item}' is not readable; use evaluate()")

    def __getitem__(self, item):
        raise SealedAccessError(f"sealed test set '{self.name}' cannot be indexed; use evaluate()")

    def __iter__(self):
        raise SealedAccessError(f"sealed test set '{self.name}' cannot be iterated; use evaluate()")

    def __len__(self):
        return len(self._data)  # size is not sensitive

    def __repr__(self):
        return f"<SealedTestSet {self.name!r} looks={len(self._looks)}/{self._max_looks}>"

    def __setattr__(self, key, value):
        raise SealedAccessError("sealed test sets are immutable")

    @property
    def looks(self) -> List[dict]:
        return list(self._looks)

    def evaluate(self, fn: Callable[[Any], Any], purpose: str) -> Any:
        """Run `fn(data)` exactly as a final evaluation. Every call is recorded."""
        if len(self._looks) >= self._max_looks:
            raise SealedAccessError(
                f"sealed test set '{self.name}' already evaluated {len(self._looks)} time(s) "
                f"(max_looks={self._max_looks}); a new look needs an explicit decision"
            )
        entry = {"set": self.name, "purpose": purpose, "ts": time.time(), "n": len(self._data)}
        self._looks.append(entry)
        if self._log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self._log_path)), exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + chr(10))
        return fn(self._data)


def ensure_not_sealed(*frames, context: str = "fitting/calibration") -> None:
    """Call at the top of any function that trains, tunes or calibrates."""
    for fr in frames:
        if isinstance(fr, SealedTestSet):
            raise SealedAccessError(
                f"sealed test set '{fr.name}' must not be used for {context}"
            )
