"""Solving domain: board rules, search strategies, planning, and learning.

* ``model``     Board rule model, small-board optimal A*, greedy fallback;
* ``search``    large-board macro beam / weighted A* strategies;
* ``planner``   the single solve policy shared by GUI and CLI;
* ``learning``  silent per-board strategy profiling.
"""
from __future__ import annotations

from .model import DIRS, Board, Move, greedy_solve, solve

__all__ = ["DIRS", "Board", "Move", "greedy_solve", "solve"]
