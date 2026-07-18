"""Board grid geometry built from grid_params.json.

The detector works in two coordinate systems:
  - source image pixels: the original game screenshot
  - rectified grid pixels: a top-down board where each cell is CELL x CELL

This module owns the perspective transform and exports a concrete grid data file
so visual recognition can be checked independently from sheep detection.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from paths import BOARD_GRID_JSON, GRID_PARAMS_JSON

CELL = 64
CORNER_KEYS = ("TL", "TR", "BR", "BL")


@dataclass(frozen=True)
class BoardGrid:
    rows: int
    cols: int
    corners: dict[str, list[float]]
    cell: int = CELL
    image_size: tuple[int, int] | None = None

    @property
    def rect_size(self) -> tuple[int, int]:
        return self.cols * self.cell, self.rows * self.cell

    @property
    def source_quad(self) -> np.ndarray:
        return np.array([self.corners[k] for k in CORNER_KEYS], np.float32)

    @property
    def rect_quad(self) -> np.ndarray:
        width, height = self.rect_size
        return np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)

    @property
    def matrix(self) -> np.ndarray:
        return cv2.getPerspectiveTransform(self.source_quad, self.rect_quad)

    @property
    def inverse_matrix(self) -> np.ndarray:
        return cv2.getPerspectiveTransform(self.rect_quad, self.source_quad)

    def warp(self, image: np.ndarray) -> np.ndarray:
        width, height = self.rect_size
        return cv2.warpPerspective(image, self.matrix, (width, height))

    def rect_to_source(self, x: float, y: float) -> tuple[float, float]:
        v = self.inverse_matrix @ np.array([x, y, 1.0], np.float64)
        return float(v[0] / v[2]), float(v[1] / v[2])

    def source_to_rect(self, x: float, y: float) -> tuple[float, float]:
        v = self.matrix @ np.array([x, y, 1.0], np.float64)
        return float(v[0] / v[2]), float(v[1] / v[2])

    def cell_rect(self, row: int, col: int, pad: int = 0) -> tuple[int, int, int, int]:
        x0 = col * self.cell + pad
        y0 = row * self.cell + pad
        x1 = (col + 1) * self.cell - pad
        y1 = (row + 1) * self.cell - pad
        return x0, y0, x1, y1

    def cell_center_rect(self, row: int, col: int) -> tuple[float, float]:
        return (col + 0.5) * self.cell, (row + 0.5) * self.cell

    def cell_center_source(self, row: int, col: int) -> tuple[float, float]:
        return self.rect_to_source(*self.cell_center_rect(row, col))

    def cell_polygon_source(self, row: int, col: int) -> list[list[float]]:
        corners = [
            (col * self.cell, row * self.cell),
            ((col + 1) * self.cell, row * self.cell),
            ((col + 1) * self.cell, (row + 1) * self.cell),
            (col * self.cell, (row + 1) * self.cell),
        ]
        return [[round(x, 2), round(y, 2)] for x, y in (self.rect_to_source(*p) for p in corners)]

    def iter_cells(self):
        for row in range(self.rows):
            for col in range(self.cols):
                yield row, col

    def to_json(self) -> dict:
        cells = []
        for row, col in self.iter_cells():
            cx, cy = self.cell_center_source(row, col)
            cells.append({
                "row": row,
                "col": col,
                "rect": list(self.cell_rect(row, col)),
                "center_rect": [round(v, 2) for v in self.cell_center_rect(row, col)],
                "center_source": [round(cx, 2), round(cy, 2)],
                "polygon_source": self.cell_polygon_source(row, col),
            })
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cell": self.cell,
            "image_size": list(self.image_size) if self.image_size else None,
            "corners": self.corners,
            "rect_size": list(self.rect_size),
            "cells": cells,
        }


def _scaled_corners(params: dict, image_shape) -> dict[str, list[float]]:
    corners = params["corners"]
    if image_shape is None:
        return {k: [float(corners[k][0]), float(corners[k][1])] for k in CORNER_KEYS}
    height, width = image_shape[:2]
    old_w, old_h = params.get("imgW"), params.get("imgH")
    if old_w and old_h and (int(old_w), int(old_h)) != (int(width), int(height)):
        sx, sy = width / float(old_w), height / float(old_h)
        return {k: [float(corners[k][0]) * sx, float(corners[k][1]) * sy] for k in CORNER_KEYS}
    return {k: [float(corners[k][0]), float(corners[k][1])] for k in CORNER_KEYS}


def load_params(path: str | Path = GRID_PARAMS_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def from_params(params: dict, image_shape=None, cell: int = CELL) -> BoardGrid:
    image_size = None
    if image_shape is not None:
        height, width = image_shape[:2]
        image_size = (int(width), int(height))
    return BoardGrid(
        rows=int(params["rows"]),
        cols=int(params["cols"]),
        corners=_scaled_corners(params, image_shape),
        cell=int(cell),
        image_size=image_size,
    )


def load_grid(params_path: str | Path = GRID_PARAMS_JSON, image: np.ndarray | None = None) -> BoardGrid:
    return from_params(load_params(params_path), None if image is None else image.shape)


def save_grid_data(grid: BoardGrid, path: str | Path = BOARD_GRID_JSON) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(grid.to_json(), f, ensure_ascii=False, indent=2)
