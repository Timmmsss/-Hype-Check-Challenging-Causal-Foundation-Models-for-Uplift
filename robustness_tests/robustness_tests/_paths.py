"""Make ``baseline_benchmark`` importable from the robustness package.

The existing runners are executed from inside ``baseline_benchmark/``, so they
import ``baseline_benchmark.data`` directly. The robustness scripts live one
directory over, so they prepend the same directory to ``sys.path`` and then use
identical import paths. Nothing in ``baseline_benchmark`` is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO_ROOT / "baseline_benchmark"


def ensure_baseline_on_path() -> Path:
    """Prepend the baseline benchmark package root to ``sys.path`` once."""
    if not (BASELINE_ROOT / "baseline_benchmark" / "data.py").exists():
        raise RuntimeError(
            f"Expected the baseline benchmark package under {BASELINE_ROOT}; "
            "run the robustness scripts from inside the project checkout."
        )
    path = str(BASELINE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return BASELINE_ROOT


ensure_baseline_on_path()
