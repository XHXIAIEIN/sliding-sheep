"""Elephant detector: 2x3 block texture and trunk-side facing."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import CELL, _exclude


def _elephant_facing_from_texture(rect: np.ndarray, row0: int, col0: int,
                                  horizontal: bool):
    """Find the elephant's head end from trunk/face edge detail.

    The sprite's rump is a broad, smooth grey mass while the head end contains
    the eye, tusk and trunk contours.  Board position is not direction
    evidence: two elephants on the same half of the board may face opposite
    ways.
    """
    if horizontal:
        ends = {
            "L": rect[row0 * CELL:(row0 + 2) * CELL,
                      col0 * CELL:(col0 + 1) * CELL],
            "R": rect[row0 * CELL:(row0 + 2) * CELL,
                      (col0 + 2) * CELL:(col0 + 3) * CELL],
        }
    else:
        ends = {
            "U": rect[row0 * CELL:(row0 + 1) * CELL,
                      col0 * CELL:(col0 + 2) * CELL],
            "D": rect[(row0 + 2) * CELL:(row0 + 3) * CELL,
                      col0 * CELL:(col0 + 2) * CELL],
        }

    scores = {}
    details = {}
    for direction, crop in ends.items():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = float(np.hypot(sx, sy).mean())
        edge_ratio = float(np.count_nonzero(cv2.Canny(gray, 45, 120))) / max(1, gray.size)
        score = gradient + 70.0 * edge_ratio
        scores[direction] = score
        details[direction] = {
            "gradient": round(gradient, 3),
            "edge_ratio": round(edge_ratio, 4),
            "score": round(score, 3),
        }
    facing = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) / max(ordered[0], 1e-6)
    confidence = min(0.96, 0.72 + margin * 0.72)
    return facing, confidence, details


def elephant_pieces(rect: np.ndarray, rows: int, cols: int, exclusion_mask=None):
    """Detect the large desaturated 2x3 elephant obstacle/piece."""
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((sat <= 70) & (val >= 35) & (val <= 205)).astype(np.uint8) * 255
    mask = _exclude(mask, exclusion_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    pieces, kept_mask, components = [], np.zeros(mask.shape, np.uint8), []

    def emit(label_id, x, y, width, height, area, cx, cy, part=None):
        if (area < 7000 or area > 26000 or min(width, height) < 90
                or max(width, height) < 145):
            return
        horizontal = width >= height
        if horizontal:
            row0 = int(round(cy / CELL - 1.0))
            col0 = int(round(cx / CELL - 1.5))
            if not (0 <= row0 <= rows - 2 and 0 <= col0 <= cols - 3):
                return
            cells = [(r, c) for r in range(row0, row0 + 2)
                     for c in range(col0, col0 + 3)]
            facing, facing_confidence, direction_detail = (
                _elephant_facing_from_texture(rect, row0, col0, True))
            head_col = col0 if facing == "L" else col0 + 2
            rump_col = col0 + 2 if facing == "L" else col0
            head, rump, axis = (row0, head_col), (row0, rump_col), "H"
        else:
            row0 = int(round(cy / CELL - 1.5))
            col0 = int(round(cx / CELL - 1.0))
            if not (0 <= row0 <= rows - 3 and 0 <= col0 <= cols - 2):
                return
            cells = [(r, c) for r in range(row0, row0 + 3)
                     for c in range(col0, col0 + 2)]
            facing, facing_confidence, direction_detail = (
                _elephant_facing_from_texture(rect, row0, col0, False))
            head_row = row0 if facing == "U" else row0 + 2
            rump_row = row0 + 2 if facing == "U" else row0
            head, rump, axis = (head_row, col0), (rump_row, col0), "V"
        pieces.append({
            "cells": [list(cell) for cell in cells],
            "axis": axis, "rump": list(rump), "head": list(head),
            "facing": facing, "species": "elephant",
            "confidence": {"occupancy": 0.96, "axis": 0.95,
                           "facing": round(facing_confidence, 4), "species": 0.99},
            "detectors": ["elephant-body"],
            "direction_confidence": round(120.0 + facing_confidence * 120.0, 2),
            "direction_votes": {"elephant_body": list(head),
                                "body_box": [x, y, width, height],
                                "head_detail": direction_detail},
            "source_id": f"elephant:{label_id}{'-' + str(part) if part is not None else ''}",
            "quality": float(area),
        })
        components.append({"box": [x, y, width, height], "area": area,
                           "cells": [list(cell) for cell in cells], "facing": facing,
                           "facing_confidence": round(facing_confidence, 4),
                           "head_detail": direction_detail})

    for label_id in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label_id])
        if area < 7000 or area > 36000:
            continue
        cx, cy = (float(v) for v in centers[label_id])
        kept_mask[labels == label_id] = 255
        # Two nearby elephants may touch through trunk/tail pixels. A component
        # spanning roughly two bodies vertically is split at its midpoint.
        if area > 26000 and height >= 230 and width >= 150:
            component = labels == label_id
            middle = y + height // 2
            for part, (lo, hi) in enumerate(((y, middle), (middle, y + height))):
                ys, xs = np.where(component[lo:hi])
                if len(xs) < 7000:
                    continue
                ys = ys + lo
                px0, px1 = int(xs.min()), int(xs.max()) + 1
                py0, py1 = int(ys.min()), int(ys.max()) + 1
                emit(label_id, px0, py0, px1 - px0, py1 - py0, len(xs),
                     float(xs.mean()), float(ys.mean()), part=part)
        else:
            emit(label_id, x, y, width, height, area, cx, cy)
    return pieces, kept_mask, {"components": components} if components else None
