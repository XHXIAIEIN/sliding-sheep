"""Watershed body segmentation and region scoring."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import label as ndlabel, maximum_filter
from .masks import CELL, DIRS, _cell_count, _cell_of


def watershed_regions(rect: np.ndarray, body_mask: np.ndarray, dt: np.ndarray):
    peak_radius = 18
    peaks = (dt == maximum_filter(dt, size=2 * peak_radius + 1)) & (dt >= 7.0)
    peaks = cv2.dilate(peaks.astype(np.uint8), np.ones((5, 5), np.uint8))
    seeds, nseed = ndlabel(peaks)

    markers = seeds.astype(np.int32)
    markers[body_mask == 0] = nseed + 1
    cv2.watershed(rect, markers)
    return markers, int(nseed)


def _candidate_pairs(hist: dict[tuple[int, int], int], axis: str, rows: int, cols: int):
    pairs = set()
    for row, col in hist:
        if axis == "H":
            for c0 in (col - 1, col):
                if 0 <= c0 < cols - 1:
                    pairs.add(((row, c0), (row, c0 + 1)))
        else:
            for r0 in (row - 1, row):
                if 0 <= r0 < rows - 1:
                    pairs.add(((r0, col), (r0 + 1, col)))
    return pairs


def _axis_from_pixels(xs: np.ndarray, ys: np.ndarray) -> str:
    vx = float(((xs - xs.mean()) ** 2).mean())
    vy = float(((ys - ys.mean()) ** 2).mean())
    return "V" if vy >= vx else "H"


def _score_region(idv, markers, body_mask, face_mask, dt, rows, cols):
    ys, xs = np.where(markers == idv)
    area = len(xs)
    if area < 450 or area > 9000:
        return None

    hist: dict[tuple[int, int], int] = {}
    for x, y in zip(xs, ys):
        cell = _cell_of(int(x), int(y), rows, cols)
        if cell is not None:
            hist[cell] = hist.get(cell, 0) + 1
    if not hist:
        return None

    axis = _axis_from_pixels(xs, ys)
    region_mask = markers == idv
    support = cv2.dilate(region_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
    face_support = (face_mask > 0) & support
    body_count_cache: dict[tuple[int, int], int] = {}

    def global_body(cell):
        if cell not in body_count_cache:
            body_count_cache[cell] = _cell_count(body_mask, cell)
        return body_count_cache[cell]

    best_pair = None
    best_score = -1.0
    cx, cy = float(xs.mean()), float(ys.mean())
    for a, b in _candidate_pairs(hist, axis, rows, cols):
        ar, ac = a
        br, bc = b
        pair_cx = ((ac + 0.5) + (bc + 0.5)) * CELL / 2.0
        pair_cy = ((ar + 0.5) + (br + 0.5)) * CELL / 2.0
        center_penalty = 0.015 * np.hypot(pair_cx - cx, pair_cy - cy)
        # Watershed only follows the white fleece.  A head cell can contain
        # little assigned region, so include weak whole-cell body-mask support
        # when choosing the adjacent two cells.  Do not include warm face mask
        # here: orange board squares look face-like.
        score = hist.get(a, 0) + hist.get(b, 0) + 0.35 * (global_body(a) + global_body(b)) - center_penalty
        if score > best_score:
            best_score = score
            best_pair = (a, b)
    if best_pair is None:
        return None

    a, b = best_pair
    if axis == "H":
        a, b = sorted([a, b], key=lambda rc: rc[1])
    else:
        a, b = sorted([a, b], key=lambda rc: rc[0])

    if min(hist.get(a, 0) + 0.35 * global_body(a), hist.get(b, 0) + 0.35 * global_body(b)) < 160:
        return None

    def metrics(cell):
        row, col = cell
        y0, y1 = row * CELL, (row + 1) * CELL
        x0, x1 = col * CELL, (col + 1) * CELL
        region_cell = region_mask[y0:y1, x0:x1]
        white = int((region_cell & (body_mask[y0:y1, x0:x1] > 0)).sum())
        face = int(face_support[y0:y1, x0:x1].sum())
        dt_vals = dt[y0:y1, x0:x1][region_cell]
        dt_mean = float(dt_vals.mean()) if dt_vals.size else 0.0
        return {"white": white, "face": face, "dt_mean": dt_mean,
                "hist": hist.get(cell, 0), "body_support": global_body(cell)}

    ma, mb = metrics(a), metrics(b)
    # Head is the pointy end of the water-drop: less white fleece and a lower
    # distance-transform radius.  Warm face/ear pixels are useful, but they are
    # noisy because horns/feet can sit near the rump too, so keep color as a
    # small tie-breaker instead of the main signal.
    head_score_a = ma["face"] * 0.12 - ma["white"] * 0.15 - ma["dt_mean"] * 18.0
    head_score_b = mb["face"] * 0.12 - mb["white"] * 0.15 - mb["dt_mean"] * 18.0
    shape_head = a if head_score_a > head_score_b else b
    pair_cx = ((a[1] + 0.5) + (b[1] + 0.5)) * CELL / 2.0
    pair_cy = ((a[0] + 0.5) + (b[0] + 0.5)) * CELL / 2.0
    if axis == "H":
        centroid_head = a if cx > pair_cx else b
        centroid_offset = cx - pair_cx
    else:
        centroid_head = a if cy > pair_cy else b
        centroid_offset = cy - pair_cy
    if abs(head_score_a - head_score_b) < 8:
        head = a if ma["white"] < mb["white"] else b
    else:
        head = shape_head
    rump = b if head == a else a
    dr, dc = head[0] - rump[0], head[1] - rump[1]
    if (dr, dc) not in DIRS:
        return None

    confidence = float(best_score + abs(head_score_a - head_score_b) + max(ma["face"], mb["face"]) * 0.2)
    return {
        "source_id": int(idv),
        "species": "sheep",
        "cells": [list(rump), list(head)],
        "axis": axis,
        "rump": list(rump),
        "head": list(head),
        "facing": DIRS[(dr, dc)],
        "area": int(area),
        "center_rect": [round(cx, 2), round(cy, 2)],
        "quality": round(confidence, 2),
        "pair_score": round(float(best_score), 2),
        "direction_confidence": round(float(abs(head_score_a - head_score_b)), 2),
        "direction_votes": {
            "shape": list(shape_head),
            "centroid": list(centroid_head),
            "centroid_offset": round(float(centroid_offset), 2),
        },
        "head_scores": {str(a): round(float(head_score_a), 2),
                        str(b): round(float(head_score_b), 2)},
        "metrics": {str(a): ma, str(b): mb},
    }
