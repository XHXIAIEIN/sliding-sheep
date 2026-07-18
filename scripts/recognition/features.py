"""Position-independent visual features for grid cells and two-cell pieces."""
from __future__ import annotations

import hashlib
import math
import cv2
import numpy as np


PAIR_FEATURE_SCHEMA = "rect-pair-v1"


PAIR_FEATURE_NAMES = (
    "red", "orange", "yellow", "green", "cyan", "blue", "magenta",
    "white", "dark", "skin", "edge", "saturation", "value",
)


CELL_SIZE = 64


def cell_key(candidate: dict) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(tuple(int(v) for v in cell) for cell in candidate.get("cells", [])))


def _clip(value, low=0.0, high=1.0):
    return float(max(low, min(high, value)))


def _cell_visual_stats(rect: np.ndarray, cell: tuple[int, int], cell_size=CELL_SIZE) -> list[float]:
    """Return compact, position-independent colour/edge evidence for one grid cell."""
    row, col = (int(cell[0]), int(cell[1]))
    y0, y1 = row * cell_size, (row + 1) * cell_size
    x0, x1 = col * cell_size, (col + 1) * cell_size
    patch = rect[y0:y1, x0:x1]
    if patch.size == 0:
        return [0.0] * len(PAIR_FEATURE_NAMES)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    strong = (sat >= 70) & (val >= 55)
    masks = (
        (((hue <= 7) | (hue >= 172)) & strong),
        ((hue >= 8) & (hue < 25) & strong),
        ((hue >= 25) & (hue < 40) & strong),
        ((hue >= 40) & (hue < 76) & strong),
        ((hue >= 76) & (hue < 103) & strong),
        ((hue >= 103) & (hue < 136) & strong),
        ((hue >= 136) & (hue < 172) & strong),
        ((sat <= 60) & (val >= 145)),
        (val <= 80),
        ((hue <= 24) & (sat >= 35) & (sat <= 215) & (val >= 55) & (val <= 245)),
    )
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 160)
    values = [float(mask.mean()) for mask in masks]
    values.extend((float((edge > 0).mean()), float(sat.mean() / 255.0),
                   float(val.mean() / 255.0)))
    return [round(value, 6) for value in values]


def pair_visual_feature(rect: np.ndarray, piece_or_cells, *, cell_size=CELL_SIZE) -> dict | None:
    """Describe an adjacent two-cell visual without using its absolute board position."""
    cells = cell_key(piece_or_cells if isinstance(piece_or_cells, dict)
                     else {"cells": piece_or_cells})
    if len(cells) != 2:
        return None
    if abs(cells[0][0] - cells[1][0]) + abs(cells[0][1] - cells[1][1]) != 1:
        return None
    axis = "H" if cells[0][0] == cells[1][0] else "V"
    ordered = sorted(cells, key=lambda rc: rc[1] if axis == "H" else rc[0])
    low = _cell_visual_stats(rect, ordered[0], cell_size)
    high = _cell_visual_stats(rect, ordered[1], cell_size)
    symmetric = [round((a + b) * 0.5, 6) for a, b in zip(low, high)]
    endpoint = [round(b - a, 6) for a, b in zip(low, high)]
    rows, cols = [cell[0] for cell in ordered], [cell[1] for cell in ordered]
    crop = rect[min(rows) * cell_size:(max(rows) + 1) * cell_size,
                min(cols) * cell_size:(max(cols) + 1) * cell_size]
    patch_hash = hashlib.sha1(crop.tobytes()).hexdigest() if crop.size else None
    return {
        "schema": PAIR_FEATURE_SCHEMA,
        "axis": axis,
        "names": list(PAIR_FEATURE_NAMES),
        "symmetric": symmetric,
        "endpoint": endpoint,
        "patch_hash": patch_hash,
    }


def _feature_distance(first: dict, second: dict, field: str) -> float:
    a, b = first.get(field) or [], second.get(field) or []
    if (first.get("schema") != PAIR_FEATURE_SCHEMA
            or second.get("schema") != PAIR_FEATURE_SCHEMA
            or first.get("axis") != second.get("axis")
            or len(a) != len(b) or not a):
        return math.inf
    values = [(float(x) - float(y)) ** 2 for x, y in zip(a, b)]
    if not all(math.isfinite(value) for value in values):
        return math.inf
    return math.sqrt(sum(values) / len(values))
