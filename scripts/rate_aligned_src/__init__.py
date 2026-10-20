"""Experiment-local source package.

The server run uploads the exact local pruning kernels into this directory.
When this package is used directly from the workspace, the project ``lib``
directory is a fallback so the same runner remains importable without copies.
"""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "lib"))
