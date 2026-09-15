#!/usr/bin/env python3
"""Thin wrapper so `python scripts/run.py paper.pdf` works without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from papersum.cli import main  # noqa: E402

main()
