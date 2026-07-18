"""Goat detector: horn and beard evidence distinguishing goats from wolves."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import CELL, DIRS, _cell_count, _exclude


def goat_candidates(rect: np.ndarray, rows: int, cols: int, exclusion_mask=None):
    """Detect the two-cell beige goats with dark curled horns.

    Their desaturated warm body forms one large, elongated component.  Normal
    cyan sheep only create small face fragments in this mask, so the component
    area is a strong species signal.  The endpoint with more nearby brown horn
    and face pixels is the head.
    """
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((hue < 20) & (sat >= 25) & (sat <= 100)
            & (val >= 80) & (val <= 245)).astype(np.uint8) * 255
    mask = _exclude(mask, exclusion_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates, components = [], []
    for label_id in range(1, count):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        short, long = min(width, height), max(width, height)
        component_pixels = labels == label_id
        median_value = float(np.median(val[component_pixels]))
        if (area < 3000 or area > 6800 or short < 42 or long > 138
                or long / max(1.0, float(short)) < 1.35 or median_value > 214):
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

        component_mask = (labels == label_id).astype(np.uint8) * 255
        near_body = cv2.dilate(component_mask, np.ones((15, 15), np.uint8)) > 0
        brown = ((hue <= 25) & (sat >= 70) & (val >= 35) & (val < 190)
                 & near_body)

        def support(feature, cell):
            row, col = cell
            return int(feature[row * CELL:(row + 1) * CELL,
                               col * CELL:(col + 1) * CELL].sum())

        first, second = endpoints
        metrics = {}
        for cell in endpoints:
            metrics[str(cell)] = {
                "goat_body": _cell_count(component_mask, cell),
                "brown_head": support(brown, cell),
            }
        head = max(endpoints, key=lambda cell: metrics[str(cell)]["brown_head"])
        rump = second if head == first else first
        direction_confidence = abs(
            metrics[str(first)]["brown_head"] - metrics[str(second)]["brown_head"])
        if direction_confidence < 80:
            continue
        dr, dc = head[0] - rump[0], head[1] - rump[1]
        if (dr, dc) not in DIRS:
            continue
        facing = DIRS[(dr, dc)]
        pair_score = sum(metrics[str(cell)]["goat_body"] for cell in endpoints)
        component = {
            "box": [x, y, width, height], "area": area,
            "median_value": round(median_value, 2),
            "center_rect": [round(cx, 2), round(cy, 2)],
            "cells": [list(rump), list(head)], "facing": facing,
            "metrics": metrics,
        }
        components.append(component)
        candidates.append({
            "source_id": f"goat-body:{label_id}",
            "detector": "goat-body", "species": "goat",
            "cells": [list(rump), list(head)], "axis": axis,
            "rump": list(rump), "head": list(head), "facing": facing,
            "area": area, "center_rect": [round(cx, 2), round(cy, 2)],
            "median_value": round(median_value, 2),
            "quality": round(float(15000 + pair_score + direction_confidence), 2),
            "pair_score": round(float(pair_score), 2),
            "direction_confidence": round(float(direction_confidence), 2),
            "direction_votes": {
                "goat_head": list(head), "goat_body_box": [x, y, width, height],
            },
            "head_scores": {
                str(first): float(metrics[str(first)]["brown_head"]),
                str(second): float(metrics[str(second)]["brown_head"]),
            },
            "metrics": metrics,
        })
    return candidates, mask, components
