"""Colour/shape candidate detectors for pink sheep, pigs, and goats."""
from __future__ import annotations

import cv2
import numpy as np
from .masks import CELL, DIRS, _cell_count, _exclude


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
