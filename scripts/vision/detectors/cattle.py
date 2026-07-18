"""Cattle detector: body/face masks and multi-scale cell candidates."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import CELL, DIRS, _cell_count, _cell_of, _exclude
from ..segmentation import _candidate_pairs


def cattle_masks(rect: np.ndarray, exclusion_mask=None):
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    body = ((hue >= 5) & (hue <= 18) & (sat >= 75) & (val >= 80) & (val <= 200)).astype(np.uint8) * 255
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    near_body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0
    warm_face = (hue >= 0) & (hue <= 18) & (sat >= 80) & (val >= 45) & (val <= 150)
    dark_face = val <= 72
    white_face = (sat <= 55) & (val >= 150)
    face = ((warm_face | dark_face | white_face) & near_body).astype(np.uint8) * 255
    face = cv2.morphologyEx(face, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _exclude(body, exclusion_mask), _exclude(face, exclusion_mask)


def _cattle_body_candidates(body_mask, face_mask, rows, cols):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(body_mask, 8)
    candidates = []
    height, width = body_mask.shape[:2]
    for idv in range(1, n):
        area = int(stats[idv, cv2.CC_STAT_AREA])
        if area < 3500 or area > 9000:
            continue
        x, y = int(stats[idv, cv2.CC_STAT_LEFT]), int(stats[idv, cv2.CC_STAT_TOP])
        w, h = int(stats[idv, cv2.CC_STAT_WIDTH]), int(stats[idv, cv2.CC_STAT_HEIGHT])
        if x < 48 or y < 48 or x + w > width - 48 or y + h > height - 48:
            continue
        axis = "V" if h > w * 1.2 else ("H" if w > h * 1.2 else ("V" if h >= w else "H"))
        comp = labels == idv
        ys, xs = np.where(comp)
        hist: dict[tuple[int, int], int] = {}
        for px, py in zip(xs, ys):
            cell = _cell_of(int(px), int(py), rows, cols)
            if cell is not None:
                hist[cell] = hist.get(cell, 0) + 1
        if not hist:
            continue
        cx, cy = float(cents[idv][0]), float(cents[idv][1])
        best = None
        for a, b in _candidate_pairs(hist, axis, rows, cols):
            pair_cx = ((a[1] + 0.5) + (b[1] + 0.5)) * CELL / 2.0
            pair_cy = ((a[0] + 0.5) + (b[0] + 0.5)) * CELL / 2.0
            score = hist.get(a, 0) + hist.get(b, 0) - 0.02 * np.hypot(pair_cx - cx, pair_cy - cy)
            if best is None or score > best[0]:
                best = (score, a, b)
        if best is None:
            continue
        score, a, b = best
        if axis == "H":
            a, b = sorted([a, b], key=lambda rc: rc[1])
        else:
            a, b = sorted([a, b], key=lambda rc: rc[0])
        face_a = _cell_count(face_mask, a)
        face_b = _cell_count(face_mask, b)
        if max(face_a, face_b) < 120:
            continue
        head = a if face_a >= face_b else b
        rump = b if head == a else a
        dr, dc = head[0] - rump[0], head[1] - rump[1]
        if (dr, dc) not in DIRS:
            continue
        facing = DIRS[(dr, dc)]
        body_support = hist.get(a, 0) + hist.get(b, 0)
        candidates.append({
            "source_id": int(9000 + idv),
            "species": "cattle",
            "cells": [list(rump), list(head)],
            "axis": axis,
            "rump": list(rump),
            "head": list(head),
            "facing": facing,
            "area": area,
            "center_rect": [round(cx, 2), round(cy, 2)],
            "quality": round(float(8600 + score * 0.06 + abs(face_a - face_b) * 0.2), 2),
            "pair_score": round(float(body_support), 2),
            "direction_confidence": round(float(abs(face_a - face_b)), 2),
            "direction_votes": {"cattle_body": list(head), "body_box": [x, y, w, h]},
            "head_scores": {str(a): round(float(face_a), 2), str(b): round(float(face_b), 2)},
            "metrics": {
                str(a): {"body_support": hist.get(a, 0), "face": face_a},
                str(b): {"body_support": hist.get(b, 0), "face": face_b},
            },
        })
    return candidates


def _cattle_cell_candidates(body_mask, face_mask, rows, cols):
    body_counts = {}
    face_counts = {}
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            body_counts[cell] = _cell_count(body_mask, cell)
            face_counts[cell] = _cell_count(face_mask, cell)

    candidates = []
    source_id = 11000
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            for dr, dc, axis in ((1, 0, "V"), (0, 1, "H")):
                a = (r, c)
                b = (r + dr, c + dc)
                if b[0] >= rows - 1 or b[1] >= cols - 1:
                    continue
                body_pair = body_counts[a] + body_counts[b]
                face_pair = face_counts[a] + face_counts[b]
                max_face = max(face_counts[a], face_counts[b])
                min_body = min(body_counts[a], body_counts[b])
                if body_pair < 2300 or max_face < 180 or min_body < 250:
                    continue

                head = a if face_counts[a] >= face_counts[b] else b
                rump = b if head == a else a
                drh, dch = head[0] - rump[0], head[1] - rump[1]
                if (drh, dch) not in DIRS:
                    continue
                score = body_pair + max_face * 3.0 - min_body * 0.2
                source_id += 1
                candidates.append({
                    "source_id": source_id,
                    "species": "cattle",
                    "cells": [list(rump), list(head)],
                    "axis": axis,
                    "rump": list(rump),
                    "head": list(head),
                    "facing": DIRS[(drh, dch)],
                    "area": int(body_pair),
                    "center_rect": [
                        round(((a[1] + 0.5) + (b[1] + 0.5)) * CELL / 2.0, 2),
                        round(((a[0] + 0.5) + (b[0] + 0.5)) * CELL / 2.0, 2),
                    ],
                    # Low-priority filler: complete body/face candidates win if
                    # they overlap this pair.
                    "quality": round(float(8050 + score * 0.04), 2),
                    "pair_score": round(float(body_pair), 2),
                    "direction_confidence": round(float(abs(face_counts[a] - face_counts[b])), 2),
                    "direction_votes": {
                        "cattle_cell_stats": list(head),
                        "body_counts": [body_counts[a], body_counts[b]],
                        "face_counts": [face_counts[a], face_counts[b]],
                    },
                    "head_scores": {str(a): round(float(face_counts[a]), 2),
                                    str(b): round(float(face_counts[b]), 2)},
                    "metrics": {
                        str(a): {"body_support": body_counts[a], "face": face_counts[a]},
                        str(b): {"body_support": body_counts[b], "face": face_counts[b]},
                    },
                })
    return candidates


def _cattle_candidates(rect, rows, cols, exclusion_mask=None):
    body_mask, face_mask = cattle_masks(rect, exclusion_mask)
    body_candidates = _cattle_body_candidates(body_mask, face_mask, rows, cols)
    cell_candidates = _cattle_cell_candidates(body_mask, face_mask, rows, cols)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(face_mask, 8)
    candidates = list(body_candidates) + list(cell_candidates)
    seen = {tuple(sorted(tuple(c) for c in cand["cells"])) for cand in candidates}
    height, width = body_mask.shape[:2]
    dirs = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

    for idv in range(1, n):
        area = int(stats[idv, cv2.CC_STAT_AREA])
        if area < 250 or area > 1400:
            continue
        x, y = int(stats[idv, cv2.CC_STAT_LEFT]), int(stats[idv, cv2.CC_STAT_TOP])
        w, h = int(stats[idv, cv2.CC_STAT_WIDTH]), int(stats[idv, cv2.CC_STAT_HEIGHT])
        # Most false positives are fence bits on the board border.
        if x < 48 or y < 48 or x + w > width - 48 or y + h > height - 48:
            continue
        cx, cy = float(cents[idv][0]), float(cents[idv][1])
        base_row = int(cy // CELL)
        base_col = int(cx // CELL)

        best = None
        for head_row in range(base_row - 1, base_row + 2):
            for head_col in range(base_col - 1, base_col + 2):
                if not (0 <= head_row < rows and 0 <= head_col < cols):
                    continue
                head_center = np.array([(head_col + 0.5) * CELL, (head_row + 0.5) * CELL])
                center_penalty = float(np.linalg.norm(np.array([cx, cy]) - head_center)) * 2.2
                for facing, (dr, dc) in dirs.items():
                    dir_axis = "V" if facing in ("U", "D") else "H"
                    if h > w * 1.5 and dir_axis != "V":
                        continue
                    if w > h * 1.5 and dir_axis != "H":
                        continue
                    rump = (head_row - dr, head_col - dc)
                    head = (head_row, head_col)
                    if not (0 <= rump[0] < rows and 0 <= rump[1] < cols):
                        continue
                    body_support = _cell_count(body_mask, rump) + _cell_count(body_mask, head)
                    head_face = _cell_count(face_mask, head)
                    rump_face = _cell_count(face_mask, rump)
                    if body_support < 1800 or head_face < 120:
                        continue
                    axis_bias = 0.0
                    if h > w * 1.12 and dir_axis == "V":
                        axis_bias = 180.0
                    elif w > h * 1.12 and dir_axis == "H":
                        axis_bias = 180.0
                    score = body_support + 2.4 * head_face - 1.3 * rump_face + axis_bias - center_penalty
                    if best is None or score > best[0]:
                        best = (score, facing, rump, head, body_support, head_face, rump_face)
        if best is None:
            continue

        score, facing, rump, head, body_support, head_face, rump_face = best
        key = tuple(sorted((rump, head)))
        if key in seen:
            continue
        seen.add(key)
        axis = "V" if rump[1] == head[1] else "H"
        candidates.append({
            "source_id": int(10000 + idv),
            "species": "cattle",
            "cells": [list(rump), list(head)],
            "axis": axis,
            "rump": list(rump),
            "head": list(head),
            "facing": facing,
            "area": area,
            "center_rect": [round(cx, 2), round(cy, 2)],
            # Keep cattle below arrow sheep in overlap resolution.
            "quality": round(float(8200 + score * 0.06), 2),
            "pair_score": round(float(body_support), 2),
            "direction_confidence": round(float(max(0, head_face - rump_face)), 2),
            "direction_votes": {
                "cattle_face": list(head),
                "face_box": [x, y, w, h],
            },
            "head_scores": {str(rump): round(float(rump_face), 2),
                            str(head): round(float(head_face), 2)},
            "metrics": {
                str(rump): {"body_support": _cell_count(body_mask, rump), "face": rump_face},
                str(head): {"body_support": _cell_count(body_mask, head), "face": head_face},
            },
        })
    return candidates
