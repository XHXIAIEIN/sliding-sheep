"""Rocket, bomb, cattle, and elephant piece detectors."""
from __future__ import annotations

import cv2
import numpy as np
from .masks import CELL, DIRS, _cell_count, _cell_of, _exclude


def rocket_masks(rect: np.ndarray, exclusion_mask=None):
    """Return the distinctive red/white timer artwork and a sheep-face mask.

    Rocket sheep do not carry the normal orange facing arrow and most of their
    turquoise body is hidden by the rocket and countdown placard.  The placard
    is the only large red/white object inside the rectified board, while the
    cattle face mask is useful here because it rejects the orange board cells.
    """
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (((hue <= 7) | (hue >= 172)) & (sat >= 85) & (val >= 70)).astype(np.uint8) * 255
    white = ((sat <= 60) & (val >= 145)).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _cattle_body, face = cattle_masks(rect, exclusion_mask)
    return (_exclude(red, exclusion_mask), _exclude(white, exclusion_mask), face)


def _rocket_candidates(rect: np.ndarray, rows: int, cols: int, exclusion_mask=None):
    """Detect two-cell rocket sheep from their countdown placard.

    Candidate direction is the adjacent endpoint with more sheep-face pixels.
    Several adjacent pairs can contain spill-over from the same placard.  Keep
    them all for the global non-overlap optimizer: pruning by inferred head can
    delete two valid staggered rockets in favour of one horizontal bridge.
    """
    red_mask, white_mask, face_mask = rocket_masks(rect, exclusion_mask)
    candidates = []
    source_id = 20000
    for row in range(rows):
        for col in range(cols):
            for dr, dc, axis in ((0, 1, "H"), (1, 0, "V")):
                other = (row + dr, col + dc)
                if other[0] >= rows or other[1] >= cols:
                    continue
                a, b = (row, col), other
                red_support = _cell_count(red_mask, a) + _cell_count(red_mask, b)
                white_support = _cell_count(white_mask, a) + _cell_count(white_mask, b)
                face_a, face_b = _cell_count(face_mask, a), _cell_count(face_mask, b)
                direction_confidence = abs(face_a - face_b)
                if (red_support < 450 or white_support < 900
                        or max(face_a, face_b) < 100 or direction_confidence < 60):
                    continue
                head = a if face_a >= face_b else b
                rump = b if head == a else a
                facing = DIRS[(head[0] - rump[0], head[1] - rump[1])]
                pair_score = white_support + red_support * 2.0
                rocket_score = pair_score + max(face_a, face_b) * 3.0 + direction_confidence
                source_id += 1
                candidates.append({
                    "source_id": source_id,
                    "detector": "rocket",
                    "species": "rocket",
                    "cells": [list(rump), list(head)],
                    "axis": axis,
                    "rump": list(rump),
                    "head": list(head),
                    "facing": facing,
                    "quality": round(float(12000 + rocket_score), 2),
                    "pair_score": round(float(pair_score), 2),
                    "direction_confidence": round(float(direction_confidence), 2),
                    "direction_votes": {
                        "rocket_face": list(head),
                        "rocket_stats": {
                            "red": red_support,
                            "white": white_support,
                        },
                    },
                    "head_scores": {str(a): float(face_a), str(b): float(face_b)},
                    "metrics": {
                        str(a): {"red": _cell_count(red_mask, a),
                                 "white": _cell_count(white_mask, a), "face": face_a},
                        str(b): {"red": _cell_count(red_mask, b),
                                 "white": _cell_count(white_mask, b), "face": face_b},
                    },
                    "rocket_score": round(float(rocket_score), 2),
                })

    return candidates, red_mask


def classify_bomb_digit(component_mask: np.ndarray):
    """Recognize the blue 1/2/3 glyph inside a bomb counter disc."""
    if not isinstance(component_mask, np.ndarray) or component_mask.size == 0:
        return None, 0.0, {}
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 60:
        return None, 0.0, {}
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    glyph = (component_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    height, width = glyph.shape
    ratio = width / max(1.0, float(height))
    if not (8 <= width <= 25 and 18 <= height <= 34 and 0.32 <= ratio <= 0.92):
        return None, 0.0, {"width": width, "height": height, "ratio": round(ratio, 3)}
    normalized = cv2.resize(glyph, (24, 32), interpolation=cv2.INTER_NEAREST) > 0
    lower = normalized[16:24]
    middle_density = float(lower[:, 8:16].mean())
    right_density = float(lower[:, 16:24].mean())
    if ratio <= 0.56:
        digit = 1
        confidence = min(0.99, 0.72 + max(0.0, 0.56 - ratio) * 2.0)
    elif right_density > middle_density:
        digit = 3
        separation = (right_density - middle_density) / max(0.15, right_density + middle_density)
        confidence = min(0.99, 0.68 + 0.28 * separation)
    else:
        digit = 2
        separation = (middle_density - right_density) / max(0.15, right_density + middle_density)
        confidence = min(0.99, 0.68 + 0.28 * separation)
    return digit, round(float(confidence), 4), {
        "width": width, "height": height, "ratio": round(ratio, 3),
        "lower_middle": round(middle_density, 4),
        "lower_right": round(right_density, 4),
    }


def bomb_markers(rect: np.ndarray, rows: int, cols: int, exclusion_mask=None):
    """Find blue count discs mounted on red dynamite bundles."""
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = ((hue >= 102) & (hue <= 135) & (sat >= 90) & (val >= 90)).astype(np.uint8) * 255
    red = (((hue <= 7) | (hue >= 172)) & (sat >= 90) & (val >= 60)).astype(np.uint8) * 255
    blue, red = _exclude(blue, exclusion_mask), _exclude(red, exclusion_mask)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(blue, 8)
    components, marker_mask = [], np.zeros(blue.shape, np.uint8)
    for label_id in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label_id])
        # Active bombs add flame/smoke around the counter.  Perspective and
        # bloom can widen the blue disc beyond the old 45 px ceiling or split
        # it into a narrow component; red dynamite support remains the strong
        # discriminator against cyan sheep artwork.
        if not (35 <= area <= 1600 and 7 <= width <= 60 and 7 <= height <= 60):
            continue
        if not 0.40 <= width / float(height) <= 2.20:
            continue
        pad = 22
        y0, y1 = max(0, y - pad), min(red.shape[0], y + height + pad)
        x0, x1 = max(0, x - pad), min(red.shape[1], x + width + pad)
        red_support = int((red[y0:y1, x0:x1] > 0).sum())
        if red_support < 70:
            continue
        cx, cy = (float(v) for v in centers[label_id])
        cell = _cell_of(int(cx), int(cy), rows, cols)
        if cell is None:
            continue
        component_mask = (labels[y:y + height, x:x + width] == label_id).astype(np.uint8) * 255
        digit, digit_confidence, digit_features = classify_bomb_digit(component_mask)
        components.append({
            "cell": cell, "box": [x, y, width, height], "area": area,
            "red_support": red_support, "digit": digit,
            "digit_confidence": digit_confidence, "digit_features": digit_features,
        })
        marker_mask[labels == label_id] = 255
    grouped = {}
    for component in components:
        grouped.setdefault(tuple(component["cell"]), []).append(component)
    markers = []
    for cell, items in sorted(grouped.items()):
        base = max(items, key=lambda item: (item["red_support"], item["area"]))
        digit_items = [item for item in items
                       if item.get("digit") in {1, 2, 3}
                       and float(item.get("digit_confidence") or 0.0) >= 0.68]
        best_digit = max(digit_items, key=lambda item: item["digit_confidence"], default=None)
        # Unknown counters are deliberately treated as one remaining hit.  It
        # is safer to forbid a collision than to invent spare bomb capacity.
        hits_remaining = int(best_digit["digit"]) if best_digit else 1
        markers.append({
            "cell": cell,
            "hits_remaining": hits_remaining,
            "hit_limit": 3,
            "counter_confident": bool(best_digit),
            "counter_confidence": (float(best_digit["digit_confidence"])
                                   if best_digit else 0.0),
            "counter_unknown": best_digit is None,
            "counter_box": best_digit.get("box") if best_digit else None,
            "counter_features": best_digit.get("digit_features") if best_digit else None,
            "box": base["box"],
            "red_support": base["red_support"],
            "components": items,
        })
    return markers, marker_mask


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
