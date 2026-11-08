"""Legacy alias for inference.py.

This module re-exports inference from the authoritative implementation in latest/
to preserve backward compatibility while eliminating redundant code.
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from inference import *  # noqa: F401, F403
