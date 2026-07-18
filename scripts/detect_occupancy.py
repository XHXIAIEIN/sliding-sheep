"""CLI entry for the visual detection pipeline.

The implementation lives in the ``vision`` package; this thin wrapper keeps
the historical command working:

Run: py scripts/detect_occupancy.py [--image images/_game.png] [--params grid_params.json]
"""
from __future__ import annotations

from vision import main

if __name__ == "__main__":
    main()
