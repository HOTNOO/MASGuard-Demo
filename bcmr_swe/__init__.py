"""BCMR-SWE runtime package.

This package hosts the MASGuard recovery pipeline:
provenance recording, failed-state construction, counterfactual actions,
and action selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
LOCAL_DEPS = PROJECT_ROOT / ".pydeps"
DATA_ROOT = PROJECT_ROOT / "data" / "bcmr"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "bcmr"

if LOCAL_DEPS.exists():
    deps_path = str(LOCAL_DEPS)
    if deps_path not in sys.path:
        sys.path.insert(0, deps_path)

DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

__all__ = ["PACKAGE_ROOT", "PROJECT_ROOT", "DATA_ROOT", "OUTPUT_ROOT", "LOCAL_DEPS"]
