"""Pig detector: skin-tone body and snout evidence, sleeping/awake state."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import CELL, DIRS, _cell_count, _exclude


def pig_candidates(rect: np.ndarray, rows: int, cols: int, exclusion_mask=None):
    """Detect two-cell pink pigs and whether their eyes are open.

    Sleeping and awake pigs share the same salmon body.  A sleeping pig keeps
    its physical facing but cannot be tapped.  Open white eyes are the stable
    current-frame signal for ``awake``; floating Z artwork is decorative and
    can disappear between animation frames.
    """
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((hue <= 8) & (sat >= 40) & (val >= 80)).astype(np.uint8) * 255
    mask = _exclude(mask, exclusion_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates, components = [], []
    for label_id in range(1, count):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if (area < 3200 or area > 7600 or min(width, height) < 48
                or max(width, height) > 126
                or max(width, height) / max(1.0, float(min(width, height))) < 1.12):
            continue

        cx, cy = map(float, centers[label_id])
        axis = "V" if height > width else "H"
        if axis == "H":
            row = int(round(cy / CELL - 0.5))
            boundary = int(round(cx / CELL))
            endpoints = [(row, boundary - 1), (row, boundary)]
        else:
            col = int(round(cx / CELL - 0.5))
            boundary = int(round(cy / CELL))
            endpoints = [(boundary - 1, col), (boundary, col)]
        if any(r < 0 or r >= rows or c < 0 or c >= cols for r, c in endpoints):
            continue

        ys, xs = np.where(labels == label_id)
        hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
        hull_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        inside = hull_mask > 0
        dark_face = (val < 180) & inside
        brown_face = ((hue >= 5) & (hue <= 25) & (sat >= 30) & (val < 200) & inside)
        white_eye = (sat <= 70) & (val >= 150) & inside

        def support(feature, cell):
            row, col = cell
            return int(feature[row * CELL:(row + 1) * CELL,
                               col * CELL:(col + 1) * CELL].sum())

        first, second = endpoints
        metrics = {}
        scores = {}
        for cell in endpoints:
            dark = support(dark_face, cell)
            brown = support(brown_face, cell)
            white = support(white_eye, cell)
            pink = _cell_count(mask, cell)
            metrics[str(cell)] = {
                "pink_body": pink, "dark_face": dark,
                "brown_face": brown, "white_eye": white,
            }
            scores[cell] = dark + brown * 0.25 + white * 1.5
        max_eye = max(metrics[str(first)]["white_eye"], metrics[str(second)]["white_eye"])
        awake = max_eye >= 25
        if awake:
            head = max(endpoints, key=lambda cell: metrics[str(cell)]["white_eye"])
            direction_confidence = abs(
                metrics[str(first)]["white_eye"] - metrics[str(second)]["white_eye"])
        else:
            # A sleeping pig tucks its snout inward; the head endpoint has a
            # consistently smaller salmon silhouette even while the floating
            # Z animation and closed-eye shading change between frames.
            head = min(endpoints, key=lambda cell: metrics[str(cell)]["pink_body"])
            direction_confidence = abs(
                metrics[str(first)]["pink_body"] - metrics[str(second)]["pink_body"])
        rump = second if head == first else first
        dr, dc = head[0] - rump[0], head[1] - rump[1]
        if (dr, dc) not in DIRS:
            continue
        facing = DIRS[(dr, dc)]
        pair_score = metrics[str(first)]["pink_body"] + metrics[str(second)]["pink_body"]
        component = {
            "box": [x, y, width, height], "area": area,
            "center_rect": [round(cx, 2), round(cy, 2)],
            "cells": [list(rump), list(head)], "facing": facing,
            "awake": bool(awake), "metrics": metrics,
        }
        components.append(component)
        candidates.append({
            "source_id": f"pig-body:{label_id}",
            "detector": "pig-body", "species": "pig",
            "cells": [list(rump), list(head)], "axis": axis,
            "rump": list(rump), "head": list(head), "facing": facing,
            "awake": bool(awake),
            "area": area, "center_rect": [round(cx, 2), round(cy, 2)],
            "quality": round(float(14500 + pair_score + direction_confidence), 2),
            "pair_score": round(float(pair_score), 2),
            "direction_confidence": round(float(direction_confidence), 2),
            "direction_votes": {
                "pig_head": list(head), "pig_body_box": [x, y, width, height],
                "pig_awake": bool(awake),
            },
            "head_scores": {str(first): round(float(scores[first]), 2),
                            str(second): round(float(scores[second]), 2)},
            "metrics": metrics,
        })
    return candidates, mask, components
