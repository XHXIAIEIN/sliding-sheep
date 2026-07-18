"""Pink sheep detector: bow/ribbon colour evidence on a sheep body."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import CELL, DIRS, _cell_count, _exclude


def pink_sheep_candidates(rect: np.ndarray, body_mask: np.ndarray,
                          face_mask: np.ndarray, rows: int, cols: int,
                          exclusion_mask=None):
    """Detect pink sheep from the saturated magenta bow around their neck.

    The fleece itself changes hue substantially with board lighting and is a
    poor anchor.  The bow remains saturated magenta, occupies roughly one
    third of a cell, and is perpendicular to the sheep's movement axis.  Its
    centre sits on the boundary between the two occupied cells; face support
    on the two sides then identifies the head.
    """
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bow_mask = ((hue >= 160) & (hue <= 179)
                & (sat >= 70) & (val >= 100)).astype(np.uint8) * 255
    bow_mask = _exclude(bow_mask, exclusion_mask)
    bow_mask = cv2.morphologyEx(
        bow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bow_mask = cv2.morphologyEx(
        bow_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    count, labels, stats, centers = cv2.connectedComponentsWithStats(bow_mask, 8)
    candidates, components = [], []
    for label_id in range(1, count):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 650 or area > 2200 or min(width, height) < 22 or max(width, height) > 72:
            continue
        if max(width, height) / max(1.0, float(min(width, height))) < 1.22:
            continue

        cx, cy = map(float, centers[label_id])
        # A vertical bow belongs to a horizontal sheep and vice versa.
        if height > width:
            axis = "H"
            row = int(round(cy / CELL - 0.5))
            boundary = int(round(cx / CELL))
            endpoints = [(row, boundary - 1), (row, boundary)]
        else:
            axis = "V"
            col = int(round(cx / CELL - 0.5))
            boundary = int(round(cy / CELL))
            endpoints = [(boundary - 1, col), (boundary, col)]
        if any(r < 0 or r >= rows or c < 0 or c >= cols for r, c in endpoints):
            continue

        first, second = endpoints
        face_first, face_second = _cell_count(face_mask, first), _cell_count(face_mask, second)
        body_first, body_second = _cell_count(body_mask, first), _cell_count(body_mask, second)
        head = first if face_first >= face_second else second
        rump = second if head == first else first
        dr, dc = head[0] - rump[0], head[1] - rump[1]
        if (dr, dc) not in DIRS:
            continue
        facing = DIRS[(dr, dc)]
        direction_confidence = abs(face_first - face_second)
        pair_score = area + 0.18 * (face_first + face_second)
        component = {
            "box": [x, y, width, height],
            "area": area,
            "center_rect": [round(cx, 2), round(cy, 2)],
            "axis": axis,
            "cells": [list(first), list(second)],
            "face_support": [face_first, face_second],
        }
        components.append(component)
        candidates.append({
            "source_id": f"pink-bow:{label_id}",
            "detector": "pink-bow",
            "species": "pink_sheep",
            "cells": [list(rump), list(head)],
            "axis": axis,
            "rump": list(rump),
            "head": list(head),
            "facing": facing,
            "area": area,
            "center_rect": [round(cx, 2), round(cy, 2)],
            "quality": round(float(15000 + pair_score + direction_confidence), 2),
            "pair_score": round(float(pair_score), 2),
            "direction_confidence": round(float(direction_confidence), 2),
            "direction_votes": {
                "pink_bow": list(head),
                "pink_bow_box": [x, y, width, height],
                "pink_face_support": {str(first): face_first, str(second): face_second},
            },
            "head_scores": {str(first): float(face_first), str(second): float(face_second)},
            "metrics": {
                str(first): {"body_support": body_first, "face": face_first,
                             "pink_bow": _cell_count(bow_mask, first)},
                str(second): {"body_support": body_second, "face": face_second,
                              "pink_bow": _cell_count(bow_mask, second)},
            },
        })
    return candidates, bow_mask, components
