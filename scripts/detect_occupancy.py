"""Detect sheep occupancy and facing from a game screenshot.

Pipeline:
  1. Build a concrete board grid from grid_params.json perspective corners.
  2. Rectify the board, threshold a black/white sheep-body mask, and segment
     water-drop shaped sheep bodies with distance-transform seeds.
  3. Score each segmented body against adjacent grid-cell pairs.  The body axis
     gives H/V; the head cell is the end with less white body and more face/ear
     evidence.

Outputs from CLI:
  - board_grid.json: every cell center/polygon in source and rectified space
  - board_layout.json: initialized 2D board layout with per-cell occupancy
  - board.json: solver input
  - sheep_candidates.json: raw/kept/dropped visual candidates for tuning
  - images/_occ_axis_rect.png: rectified board with occupied cells, facing, sheep ids
  - images/_grid_labels.png: rectified board with row/column labels
  - images/_layout.png: synthetic 2D board initialized from detected pieces

Run: py scripts/detect_occupancy.py [--image images/_game.png] [--params grid_params.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label as ndlabel, maximum_filter

import board_grid as G
import level_cache
import safety
import board_io
import recognition
from paths import image_path

CELL = G.CELL
DIRS = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}


def gesture_occlusion(rect: np.ndarray, rows: int, cols: int):
    """Detect the large white tutorial hand drawn over the board.

    The returned expanded mask is excluded from ordinary visual detectors and
    exposed as an execution blocker.  A separate narrow path may still recover
    the uniquely red-outlined target from its surviving orange arrow.
    """
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    white = ((sat <= 45) & (val >= 210)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(
        white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(white, 8)
    # Tutorial gestures point at a red outlined sheep.  The outline is often
    # split into several pieces by the white hand, so retain its connected
    # components and combine only fragments immediately above the hand.
    red = ((((hue <= 12) | (hue >= 168)) & (sat >= 120) & (val >= 105))
           .astype(np.uint8) * 255)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red_count, _red_labels, red_stats, red_centers = cv2.connectedComponentsWithStats(red, 8)
    mask = np.zeros(white.shape, np.uint8)
    components = []
    for label_id in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label_id])
        if area < 1800 or area > 24000:
            continue
        # Perspective warps can expose white wedges at the left/top/right
        # boundary.  A hand clipped only by the bottom edge is different: it
        # is the tutorial pointer aimed at the item buttons below the board.
        if x <= 2 or y <= 2 or x + width >= cols * CELL - 2:
            continue
        if width < 45 or height < 55 or width > CELL * 3 or height > CELL * 3:
            continue
        if area / float(max(1, width * height)) < 0.24:
            continue
        component = (labels == label_id).astype(np.uint8) * 255
        max_white_radius = float(cv2.distanceTransform(
            (component > 0).astype(np.uint8), cv2.DIST_L2, 5).max())
        smoke_like = max_white_radius < 20.0
        bottom_ui_hint = y + height >= rows * CELL - 2
        entry = {
            "box": [x, y, width, height],
            "white_area": area,
            "max_white_radius": round(max_white_radius, 2),
            "kind": ("ui_item_hint" if bottom_ui_hint else
                     "motion_smoke" if smoke_like else "board_occlusion"),
            "blocking": not bottom_ui_hint,
        }
        if not bottom_ui_hint and not smoke_like:
            roi_left = max(0, x - int(CELL * 0.8))
            roi_right = min(cols * CELL, x + width + int(CELL * 0.8))
            roi_top = max(0, y - int(CELL * 1.8))
            roi_bottom = min(rows * CELL, y + int(height * 0.35))
            fragments = []
            for red_id in range(1, red_count):
                rx, ry, rw, rh, red_area = (int(v) for v in red_stats[red_id])
                cx, cy = (float(v) for v in red_centers[red_id])
                if red_area < 24 or red_area > 4000:
                    continue
                if not (roi_left <= cx <= roi_right and roi_top <= cy <= roi_bottom):
                    continue
                fragments.append((rx, ry, rw, rh, red_area))
            if fragments:
                left = min(item[0] for item in fragments)
                top = min(item[1] for item in fragments)
                right = max(item[0] + item[2] for item in fragments)
                bottom = max(item[1] + item[3] for item in fragments)
                total_red = sum(item[4] for item in fragments)
                target_width, target_height = right - left, bottom - top
                if (total_red >= 140 and CELL * 0.65 <= target_width <= CELL * 3.0
                        and CELL * 0.45 <= target_height <= CELL * 2.6):
                    entry.update({
                        "kind": "tutorial_hand",
                        "tutorial_target_rect": [
                            round(left + target_width / 2.0, 2),
                            round(top + target_height / 2.0, 2),
                        ],
                        "tutorial_target_box": [left, top, target_width, target_height],
                        "tutorial_target_red_area": total_red,
                        "target_confidence": 0.92,
                    })
        # A large white in-board component without smoke geometry or a red
        # tutorial target is often a legitimate white/pink game piece.  Do not
        # erase it from every downstream detector and call it a hand.  Real
        # tutorial input is actionable only after the red target is proven.
        if entry["kind"] == "board_occlusion":
            continue
        expanded = cv2.dilate(
            component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        mask = cv2.bitwise_or(mask, expanded)
        components.append(entry)

    if not components:
        return mask, None
    affected = []
    for row in range(rows):
        for col in range(cols):
            coverage = _cell_count(mask, (row, col)) / float(CELL * CELL)
            if coverage >= 0.03:
                affected.append([row, col])
    return mask, {
        "components": components,
        "affected_cells": affected,
        "masked_pixels": int((mask > 0).sum()),
        "blocking": any(item["blocking"] for item in components),
    }


def _exclude(mask: np.ndarray, exclusion_mask: np.ndarray | None):
    if isinstance(exclusion_mask, np.ndarray) and exclusion_mask.size:
        mask = mask.copy()
        mask[exclusion_mask > 0] = 0
    return mask


def _warp(game, corners, rows, cols):
    """Compatibility wrapper used by scripts/app.py and scripts/solve_board.py."""
    grid = G.BoardGrid(rows=int(rows), cols=int(cols), corners=corners, image_size=(game.shape[1], game.shape[0]))
    return grid.warp(game), grid.inverse_matrix, grid.source_quad


def _grid_from_args(game, corners, rows, cols) -> G.BoardGrid:
    return G.BoardGrid(rows=int(rows), cols=int(cols), corners={
        k: [float(corners[k][0]), float(corners[k][1])] for k in G.CORNER_KEYS
    }, image_size=(game.shape[1], game.shape[0]))


def make_masks(rect: np.ndarray, exclusion_mask=None):
    """Return (body_mask, face_mask, dt).

    body_mask is the high-value/low-saturation sheep fleece.  face_mask is only
    used inside a dilated sheep support region, because the board itself also
    contains warm colors.
    """
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    old_wool = (sat < 65) & (val > 155)
    blue_skin = (hue >= 72) & (hue <= 98) & (sat >= 45) & (val >= 95)
    body = ((old_wool | blue_skin).astype(np.uint8)) * 255
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    warm = (hue >= 7) & (hue <= 28) & (sat >= 55) & (val >= 70) & (val <= 220)
    dark = (val < 120) & (sat > 35)
    face = ((warm | dark).astype(np.uint8)) * 255
    face = cv2.morphologyEx(face, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    body = _exclude(body, exclusion_mask)
    face = _exclude(face, exclusion_mask)

    dt = cv2.distanceTransform(body, cv2.DIST_L2, 5)
    return body, face, dt


def arrow_mask(rect: np.ndarray, exclusion_mask=None):
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Newer skins draw a saturated orange facing arrow on each movable sheep.
    # The board's orange cells are less saturated, so this remains compact.
    mask = ((hue >= 5) & (hue <= 25) & (sat >= 190) & (val >= 170)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _exclude(mask, exclusion_mask)


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


def fence_edges(rect: np.ndarray, rows: int, cols: int):
    """Detect wooden fence runs on board boundaries and inside board cells.

    Boundary fences use U/D/L/R.  Internal fences occupy cells and use H/V to
    preserve their visual orientation.  The latter matters on levels such as
    116, where a long wooden rail otherwise looks like several adjacent cows.
    """
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Fence timber in the game uses a stable ochre band. Restricting detection
    # to a narrow boundary strip and requiring a 3-cell run rejects animal fur.
    mask = ((hue >= 11) & (hue <= 18) & (sat >= 35)
            & (val >= 35) & (val <= 210)).astype(np.uint8) * 255
    strip = max(16, min(28, int(round(CELL * 0.375))))
    threshold = int(strip * CELL * 0.18)
    edge_band = CELL
    scores = {}
    spans = {}
    boundary_runs = []
    # A real boundary rail spans almost the full cell.  Large sheep faces at
    # the lower edge can reach exactly three quarters of a cell (level 172,
    # E18), so keep a little margin above that false-positive shape.
    span_threshold = int(round(CELL * 0.82))
    continuity_threshold = int(round(CELL * 1.5))

    def longitudinal_component_span(direction, begin, end):
        """Return the longest timber component parallel to a boundary run.

        Sheep faces at the edge can produce a high timber-colour score in
        several neighbouring cells.  Those blobs stop inside each animal,
        whereas a real boundary fence has a rail/post component that remains
        connected across at least three cells.  Pixel totals and per-cell
        scanlines alone cannot distinguish the two cases.
        """
        if direction == "L":
            roi = mask[begin * CELL:end * CELL, :strip]
            longitudinal_index = cv2.CC_STAT_HEIGHT
        elif direction == "R":
            roi = mask[begin * CELL:end * CELL, -strip:]
            longitudinal_index = cv2.CC_STAT_HEIGHT
        elif direction == "U":
            roi = mask[:strip, begin * CELL:end * CELL]
            longitudinal_index = cv2.CC_STAT_WIDTH
        else:
            roi = mask[-edge_band:, begin * CELL:end * CELL]
            longitudinal_index = cv2.CC_STAT_WIDTH
        if not np.any(roi):
            return 0
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (roi > 0).astype(np.uint8), 8)
        if len(stats) <= 1:
            return 0
        return int(np.max(stats[1:, longitudinal_index]))

    def edge_scores(direction):
        values = []
        transverse_spans = []
        count = rows if direction in {"L", "R"} else cols
        for index in range(count):
            # The bottom artwork sits well inside the rectified last row on
            # this camera angle. Other boundaries align with the narrow edge
            # strip and must stay narrow to avoid animal fur in edge cells.
            band = edge_band if direction == "D" else strip
            if direction == "L":
                roi = mask[index * CELL:(index + 1) * CELL, :band]
            elif direction == "R":
                roi = mask[index * CELL:(index + 1) * CELL, -band:]
            elif direction == "U":
                roi = mask[:band, index * CELL:(index + 1) * CELL]
            else:
                roi = mask[-band:, index * CELL:(index + 1) * CELL]
            values.append(int(np.count_nonzero(roi)))
            # A rail crosses most of the cell parallel to the board edge.
            # Posts and their long shadows can contain just as many timber
            # pixels, but never form a nearly full-width scan line.  Keep the
            # loose pixel score for locating a fence run, then use this span
            # to avoid turning the gaps between rails into solid fences.
            parallel_axis = 1 if direction in {"U", "D"} else 0
            scanline_counts = np.count_nonzero(roi, axis=parallel_axis)
            transverse_spans.append(int(scanline_counts.max(initial=0)))
        scores[direction] = values
        spans[direction] = transverse_spans
        return values

    fences = []
    for direction in ("L", "R", "U", "D"):
        values = edge_scores(direction)
        active = [value >= threshold for value in values]
        solid = [enabled and spans[direction][index] >= span_threshold
                 for index, enabled in enumerate(active)]
        start = None
        runs = []
        for index, enabled in enumerate(active + [False]):
            if enabled and start is None:
                start = index
            elif not enabled and start is not None:
                runs.append((start, index))
                start = None
        for begin, end in runs:
            if direction in {"L", "R"} and end - begin < 3:
                continue
            component_span = longitudinal_component_span(direction, begin, end)
            boundary_runs.append({
                "direction": direction, "begin": begin, "end": end,
                "component_span": component_span,
            })
            if (direction in {"L", "R"}
                    and component_span < continuity_threshold):
                continue
            # Perspective cropping can trim the first post of a long boundary
            # rail. Recover only a weak run endpoint backed by two solid cells;
            # never bridge weak interior post/gap cells (e.g. level 122).
            emitted = set(index for index in range(begin, end) if solid[index])
            if (direction in {"L", "R"} and end - begin >= 3
                    and solid[begin + 1] and solid[begin + 2]):
                emitted.add(begin)
            if (direction in {"L", "R"} and end - begin >= 3
                    and solid[end - 2] and solid[end - 3]):
                emitted.add(end - 1)
            for index in sorted(emitted):
                if direction == "L":
                    cell = [index, 0]
                elif direction == "R":
                    cell = [index, cols - 1]
                elif direction == "U":
                    cell = [0, index]
                else:
                    cell = [rows - 1, index]
                fences.append({
                    "cell": cell, "direction": direction,
                    "confidence": round(min(1.0, values[index] / max(1.0, threshold * 2.0)), 4),
                    "score": values[index],
                })
    # Internal rails have the same brown/cream palette as cattle, but unlike a
    # cow they fill at least three consecutive cells with a very uniform high
    # face-mask ratio.  Real cattle normally span two cells and remain well
    # below this cream-pixel threshold.
    cattle_body, cattle_face = cattle_masks(rect)
    cell_body = np.zeros((rows, cols), dtype=np.int32)
    cell_face = np.zeros((rows, cols), dtype=np.int32)
    internal_active = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            body_count = _cell_count(cattle_body, cell)
            face_count = _cell_count(cattle_face, cell)
            cell_body[r, c] = body_count
            cell_face[r, c] = face_count
            internal_active[r, c] = (
                body_count >= 1800 and face_count >= 850
                and face_count / max(1.0, body_count) >= 0.38)

    internal = []

    def add_internal_runs(axis):
        outer = rows if axis == "H" else cols
        inner = cols if axis == "H" else rows
        for fixed in range(outer):
            active = [bool(internal_active[fixed, i] if axis == "H"
                           else internal_active[i, fixed]) for i in range(inner)]
            start = None
            for index, enabled in enumerate(active + [False]):
                if enabled and start is None:
                    start = index
                elif not enabled and start is not None:
                    if index - start >= 3:
                        for moving in range(start, index):
                            r, c = ((fixed, moving) if axis == "H" else (moving, fixed))
                            face_count = int(cell_face[r, c])
                            body_count = int(cell_body[r, c])
                            item = {
                                "cell": [r, c], "direction": axis,
                                "confidence": round(min(1.0, face_count / 1200.0), 4),
                                "score": face_count,
                            }
                            fences.append(item)
                            internal.append(item)
                    start = None

    add_internal_runs("H")
    add_internal_runs("V")
    return fences, mask, {
        "strip": strip, "edge_band": edge_band, "threshold": threshold, "scores": scores,
        "span_threshold": span_threshold, "spans": spans,
        "continuity_threshold": continuity_threshold, "boundary_runs": boundary_runs,
        "internal": internal,
        "internal_body_scores": cell_body.tolist(),
        "internal_face_scores": cell_face.tolist(),
    }


def wolf_hazards(rect: np.ndarray, rows: int, cols: int, exclusion_mask=None):
    """Detect wolf artwork as dangerous board cells."""
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    _hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hazard_map: dict[tuple[int, int], dict] = {}
    wolf_mask = np.zeros(rect.shape[:2], dtype=bool)
    metas = []

    def add_component(component_mask, meta, min_pixels=120, min_coverage=0.045, include_center=True):
        ys, xs = np.where(component_mask)
        if len(xs) == 0:
            return
        for r in range(rows):
            for c in range(cols):
                roi = component_mask[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
                count = int(roi.sum())
                coverage = count / float(CELL * CELL)
                if count >= min_pixels or coverage >= min_coverage:
                    prev = hazard_map.get((r, c))
                    if prev is None or coverage > prev["coverage"]:
                        hazard_map[(r, c)] = {
                            "row": r,
                            "col": c,
                            "kind": "wolf_body",
                            "coverage": round(float(coverage), 3),
                            "pixels": count,
                        }
        if include_center:
            cy, cx = float(ys.mean()), float(xs.mean())
            cell = _cell_of(int(round(cx)), int(round(cy)), rows, cols)
            if cell is not None:
                r, c = cell
                hazard_map.setdefault((r, c), {
                    "row": r,
                    "col": c,
                    "kind": "wolf_body",
                    "coverage": 0.001,
                    "pixels": 1,
                })
        metas.append(meta)

    gray = (sat <= 70) & (val >= 35) & (val <= 205)
    dark = (val <= 85) & (sat <= 120)
    if isinstance(exclusion_mask, np.ndarray) and exclusion_mask.size:
        allowed = exclusion_mask == 0
        gray &= allowed
        dark &= allowed
    mask = ((gray | dark).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))

    # Border wolves used to promote their entire row/column from one still
    # frame.  A normally posed wolf near the lower edge has the same tall blob
    # shape, which produced false full-column hazards (notably level 121).
    # Keep only its observed body cells here; the app infers an actual patrol
    # lane from consecutive pre-click frames.
    runner_mask = (((sat <= 90) & (val >= 25) & (val <= 190)).astype(np.uint8)) * 255
    runner_mask = _exclude(runner_mask, exclusion_mask)
    runner_mask = cv2.morphologyEx(runner_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    runner_mask = cv2.morphologyEx(
        runner_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    rn, rlabels, rstats, rcenters = cv2.connectedComponentsWithStats(runner_mask, 8)
    for idv in range(1, rn):
        area = int(rstats[idv, cv2.CC_STAT_AREA])
        x = int(rstats[idv, cv2.CC_STAT_LEFT])
        y = int(rstats[idv, cv2.CC_STAT_TOP])
        w = int(rstats[idv, cv2.CC_STAT_WIDTH])
        h = int(rstats[idv, cv2.CC_STAT_HEIGHT])
        cx, cy = float(rcenters[idv][0]), float(rcenters[idv][1])
        if not (2500 <= area <= 9500 and w >= 70 and h >= 35
                and cy >= rows * CELL * 0.80):
            continue
        component = rlabels == idv
        wolf_mask |= component
        add_component(component, {
            "kind": "runner_candidate",
            "box": [x, y, w, h], "area": area,
            "center_rect": [round(cx, 2), round(cy, 2)],
        }, min_pixels=90, min_coverage=0.035, include_center=True)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    board_area = rows * cols * CELL * CELL
    for idv in range(1, n):
        area = int(stats[idv, cv2.CC_STAT_AREA])
        x = int(stats[idv, cv2.CC_STAT_LEFT])
        y = int(stats[idv, cv2.CC_STAT_TOP])
        w = int(stats[idv, cv2.CC_STAT_WIDTH])
        h = int(stats[idv, cv2.CC_STAT_HEIGHT])
        if area < max(12000, board_area * 0.025):
            continue
        if w < CELL * 2 or h < CELL * 2:
            continue
        if 9000 <= area <= 26000 and w >= 105 and h >= 120:
            continue  # elephant, handled as a 2x3 piece
        # Ignore UI bars and border shadows; the wolf is a large interior blob.
        if y > rows * CELL * 0.78 or x + w < CELL or x > (cols - 1) * CELL:
            continue
        score = area - (4000 if x <= 2 or y <= 2 else 0)
        if best is None or score > best[0]:
            best = (score, idv, area, x, y, w, h)
    if best is not None:
        _score, idv, area, x, y, w, h = best
        component = labels == idv
        wolf_mask |= component
        add_component(component, {
            "kind": "large",
            "area": area,
            "box": [x, y, w, h],
            "center_rect": [round(float(cents[idv][0]), 2), round(float(cents[idv][1]), 2)],
        }, min_pixels=260, min_coverage=0.08, include_center=False)

    dark_mask = ((val <= 78) & (sat <= 120)).astype(np.uint8) * 255
    dark_mask = _exclude(dark_mask, exclusion_mask)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    dn, dlabels, dstats, dcents = cv2.connectedComponentsWithStats(dark_mask, 8)
    for idv in range(1, dn):
        area = int(dstats[idv, cv2.CC_STAT_AREA])
        x = int(dstats[idv, cv2.CC_STAT_LEFT])
        y = int(dstats[idv, cv2.CC_STAT_TOP])
        w = int(dstats[idv, cv2.CC_STAT_WIDTH])
        h = int(dstats[idv, cv2.CC_STAT_HEIGHT])
        if area < 700 or area > 8500:
            continue
        if w < 18 or h < 18 or w > CELL * 2.4 or h > CELL * 2.4:
            continue
        if y > rows * CELL - CELL * 1.2:
            continue
        component = dlabels == idv
        wolf_mask |= component
        add_component(component, {
            "kind": "small",
            "area": area,
            "box": [x, y, w, h],
            "center_rect": [round(float(dcents[idv][0]), 2), round(float(dcents[idv][1]), 2)],
        }, min_pixels=90, min_coverage=0.035, include_center=True)

    # The broad lower-edge runner mask and the precise dark-body mask can
    # describe the same wolf.  Keep one component so motion matching does not
    # invent a third animal or a bogus second trajectory.
    deduped_metas = []
    for meta in metas:
        center = meta.get("center_rect") or []
        duplicate = None
        if len(center) == 2:
            for index, kept in enumerate(deduped_metas):
                other = kept.get("center_rect") or []
                if (len(other) == 2
                        and float(np.hypot(float(center[0]) - float(other[0]),
                                           float(center[1]) - float(other[1])))
                        <= CELL * 0.55):
                    duplicate = index
                    break
        if duplicate is None:
            deduped_metas.append(meta)
        elif (deduped_metas[duplicate].get("kind") == "runner_candidate"
              and meta.get("kind") != "runner_candidate"):
            deduped_metas[duplicate] = meta

    if not hazard_map:
        return [], (wolf_mask.astype(np.uint8) * 255), None
    return [hazard_map[k] for k in sorted(hazard_map)], (wolf_mask.astype(np.uint8) * 255), {
        "count": len(deduped_metas),
        "components": deduped_metas,
    }


def watershed_regions(rect: np.ndarray, body_mask: np.ndarray, dt: np.ndarray):
    peak_radius = 18
    peaks = (dt == maximum_filter(dt, size=2 * peak_radius + 1)) & (dt >= 7.0)
    peaks = cv2.dilate(peaks.astype(np.uint8), np.ones((5, 5), np.uint8))
    seeds, nseed = ndlabel(peaks)

    markers = seeds.astype(np.int32)
    markers[body_mask == 0] = nseed + 1
    cv2.watershed(rect, markers)
    return markers, int(nseed)


def _cell_of(x: int, y: int, rows: int, cols: int):
    row, col = int(y // CELL), int(x // CELL)
    if 0 <= row < rows and 0 <= col < cols:
        return row, col
    return None


def _cell_count(mask: np.ndarray, cell: tuple[int, int]) -> int:
    row, col = cell
    y0, y1 = row * CELL, (row + 1) * CELL
    x0, x1 = col * CELL, (col + 1) * CELL
    return int(mask[y0:y1, x0:x1].sum() > 0) if mask.dtype == bool else int((mask[y0:y1, x0:x1] > 0).sum())


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


def _arrow_candidates(rect, body_mask, rows, cols, exclusion_mask=None):
    mask = arrow_mask(rect, exclusion_mask)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for idv in range(1, n):
        area = int(stats[idv, cv2.CC_STAT_AREA])
        # Border sheep can have their arrow clipped by the perspective crop;
        # body support below still rejects board/UI orange fragments.
        if area < 300 or area > 900:
            continue
        x, y = int(stats[idv, cv2.CC_STAT_LEFT]), int(stats[idv, cv2.CC_STAT_TOP])
        w, h = int(stats[idv, cv2.CC_STAT_WIDTH]), int(stats[idv, cv2.CC_STAT_HEIGHT])
        if w < 18 or h < 18 or w > 64 or h > 64:
            continue
        comp = labels[y:y + h, x:x + w] == idv
        cx, cy = float(cents[idv][0]), float(cents[idv][1])
        if w >= h:
            left = int(comp[:, :w // 2].sum())
            right = int(comp[:, w // 2:].sum())
            facing = "R" if right > left else "L"
            direction_confidence = abs(right - left)
            row = int(round(cy / CELL - 0.5))
            c0 = int(round(cx / CELL - 1.0))
            if facing == "R":
                rump, head = (row, c0), (row, c0 + 1)
            else:
                rump, head = (row, c0 + 1), (row, c0)
            axis = "H"
        else:
            top = int(comp[:h // 2, :].sum())
            bottom = int(comp[h // 2:, :].sum())
            facing = "D" if bottom > top else "U"
            direction_confidence = abs(bottom - top)
            r0 = int(round(cy / CELL - 1.0))
            col = int(round(cx / CELL - 0.5))
            if facing == "D":
                rump, head = (r0, col), (r0 + 1, col)
            else:
                rump, head = (r0 + 1, col), (r0, col)
            axis = "V"

        cells = [rump, head]
        if any(r < 0 or r >= rows or c < 0 or c >= cols for r, c in cells):
            continue
        # Keep only arrows that sit on a detected body. This rejects UI arrows
        # and any saturated board decoration that survives the color threshold.
        # During a slide/occlusion frame the orange arrow can remain fully
        # visible while one fleece half is hidden.  A strong, compact arrow is
        # then safer evidence than the fixed body total, so allow a narrower
        # support threshold only when both its area and directional split are
        # convincing.  Weak arrows keep the conservative original threshold.
        body_support = sum(_cell_count(body_mask, cell) for cell in cells)
        partial_support = (
            area >= 460 and direction_confidence >= max(70, int(area * 0.10))
        )
        min_body_support = 380 if partial_support else 550
        if body_support < min_body_support:
            continue
        candidates.append({
            "source_id": int(idv),
            "species": "sheep",
            "cells": [list(rump), list(head)],
            "axis": axis,
            "rump": list(rump),
            "head": list(head),
            "facing": facing,
            "area": area,
            "center_rect": [round(cx, 2), round(cy, 2)],
            "quality": round(float(10000 + area + direction_confidence + body_support * 0.05), 2),
            "pair_score": round(float(body_support), 2),
            "direction_confidence": round(float(direction_confidence), 2),
            "direction_votes": {
                "arrow": list(head),
                "arrow_box": [x, y, w, h],
                "partial_body_support": bool(partial_support),
                "min_body_support": min_body_support,
            },
            "head_scores": {str(rump): 0.0, str(head): 1.0},
            "metrics": {
                str(rump): {"body_support": _cell_count(body_mask, rump)},
                str(head): {"body_support": _cell_count(body_mask, head)},
            },
        })
    return candidates


def _gesture_target_arrow_candidates(rect, body_mask, rows, cols,
                                     gesture_meta, regular_candidates):
    """Recover the sheep explicitly outlined by a tutorial hand.

    The broad hand mask is still authoritative for every ordinary detector.
    This narrow exception re-runs only the saturated orange-arrow detector and
    accepts the unique arrow nearest a high-confidence red tutorial outline.
    """
    components = list((gesture_meta or {}).get("components") or [])
    affected = {tuple(cell) for cell in (gesture_meta or {}).get("affected_cells") or []}
    if not components or not affected:
        return []
    regular_keys = {tuple(sorted(tuple(cell) for cell in item.get("cells", [])))
                    for item in (regular_candidates or [])}
    unmasked = _arrow_candidates(rect, body_mask, rows, cols, exclusion_mask=None)
    recovered = []
    for component in components:
        target = component.get("tutorial_target_rect")
        if (component.get("kind") != "tutorial_hand" or not target
                or float(component.get("target_confidence", 0.0)) < .85):
            continue
        tx, ty = map(float, target)
        nearby = []
        for candidate in unmasked:
            key = tuple(sorted(tuple(cell) for cell in candidate.get("cells", [])))
            if key in regular_keys or not any(tuple(cell) in affected
                                              for cell in candidate.get("cells", [])):
                continue
            cx, cy = map(float, candidate.get("center_rect") or (0, 0))
            distance = float(np.hypot(cx - tx, cy - ty))
            if (distance <= CELL * .90 and float(candidate.get("area", 0)) >= 300
                    and float(candidate.get("pair_score", 0)) >= 700):
                nearby.append((distance, candidate))
        nearby.sort(key=lambda item: item[0])
        if not nearby or (len(nearby) > 1 and nearby[1][0] - nearby[0][0] < CELL * .18):
            continue
        distance, candidate = nearby[0]
        restored = dict(candidate)
        restored["detector"] = "gesture-target-arrow"
        restored["gesture_recovered"] = True
        restored["gesture_target_distance"] = round(distance, 2)
        restored["quality"] = float(restored.get("quality", 0.0)) + 25000.0
        restored.setdefault("direction_votes", {})["gesture_target"] = list(target)
        recovered.append(restored)
        regular_keys.add(tuple(sorted(tuple(cell) for cell in restored.get("cells", []))))
    return recovered


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


def resolve_candidates(candidates: list[dict], *, return_meta=False):
    """Choose a globally optimal non-overlapping candidate set."""
    kept, dropped, optimization = recognition.global_assignment(candidates)
    sheep = []
    for i, cand in enumerate(kept):
        s = {k: cand[k] for k in ("cells", "axis", "rump", "head", "facing")}
        s["species"] = cand.get("species", _candidate_species(cand))
        review = cand.get("review_reason") or _candidate_review(cand)
        if review:
            s["review"] = True
            s["review_reason"] = review
        for k in ("direction_confidence", "direction_votes", "head_scores", "metrics", "confidence",
                  "fusion", "detectors", "review_reasons", "selection_score",
                  "learned_template", "learned_provisional", "learned_support",
                  "learned_sample_ids"):
            if k in cand:
                s[k] = cand[k]
        s["source_id"] = cand["source_id"]
        s["quality"] = cand["quality"]
        for key in ("hit_limit", "hits_remaining", "counter_confident",
                    "counter_confidence", "counter_unknown"):
            if key in cand:
                s[key] = cand[key]
        if "awake" in cand:
            s["awake"] = bool(cand["awake"])
        s["id"] = i
        sheep.append(s)
    if return_meta:
        return sheep, dropped, optimization
    return sheep, dropped


def _candidate_species(cand):
    if cand.get("species"):
        return cand["species"]
    votes = cand.get("direction_votes", {})
    if any(str(k).startswith("cattle") for k in votes):
        return "cattle"
    return "sheep"


def _candidate_review(cand):
    species = cand.get("species", _candidate_species(cand))
    if species != "cattle":
        return None
    votes = cand.get("direction_votes", {})
    confidence = float(cand.get("direction_confidence", 0) or 0)
    if "cattle_cell_stats" in votes and confidence < 180:
        return "cattle_cell_stats"
    if confidence < 180:
        return "low_cattle_direction_confidence"
    return None


def suppress_special_hazard_overlaps(sheep, hazards):
    """Remove dark-artwork hazards caused by rocket/bomb art and bomb smoke."""
    special_cells = {
        tuple(cell)
        for piece in (sheep or [])
        if (piece.get("species") in {"rocket", "bomb", "elephant", "black_sheep", "pink_sheep", "pig", "goat"}
            and not piece.get("learned_template")
            and not piece.get("learned_provisional"))
        for cell in piece.get("cells", [])
    }
    black_cells = {
        tuple(cell) for piece in (sheep or [])
        if (piece.get("species") == "black_sheep"
            and not piece.get("learned_template")
            and not piece.get("learned_provisional"))
        for cell in piece.get("cells", [])
    }
    kept, suppressed = [], []
    for item in hazards or []:
        cell = ((int(item["row"]), int(item["col"]))
                if isinstance(item, dict) else tuple(item))
        near_special = any(
            max(abs(cell[0] - special[0]), abs(cell[1] - special[1])) <= 1
            for special in special_cells
        )
        weak_small_blob = (
            float(item.get("coverage", 1.0)) < 0.25
            and int(item.get("pixels", 10_000)) < 1600
            and item.get("kind") != "wolf_track"
        ) if isinstance(item, dict) else False
        near_black = any(max(abs(cell[0] - black[0]), abs(cell[1] - black[1])) <= 1
                         for black in black_cells)
        if cell in special_cells or near_black or (near_special and weak_small_blob):
            suppressed.append(item)
        else:
            kept.append(item)
    return kept, suppressed


WOLF_FORWARD_MIN_CELLS = 5
WOLF_DIAGONAL_MIN_CELLS = 3


def resolve_goat_wolf_conflicts(pieces, hazards, rows, cols):
    """Resolve candidates that look like both a goat and a wolf.

    A goat is allowed to suppress dark-artwork hazards only when the same
    footprint cannot plausibly be a wolf.  Wolves are placed with enough board
    depth to patrol: at least five cells ahead and a three-cell forward
    diagonal.  This is a board-geometry check rather than an occupancy check;
    moving wolves can cross the traffic formed by the puzzle pieces.
    """
    directions = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    kept, rejected, decisions = [], [], []

    def ray_clearance(origin, delta):
        r, c = origin
        dr, dc = delta
        distance = 0
        while True:
            r, c = r + dr, c + dc
            if not (0 <= r < rows and 0 <= c < cols):
                return distance
            distance += 1

    for piece in pieces or []:
        cells = {tuple(cell) for cell in piece.get("cells", [])}
        overlap = set()
        for item in hazards or []:
            cell = ((int(item["row"]), int(item["col"]))
                    if isinstance(item, dict) else tuple(item))
            weak_nearby = (isinstance(item, dict)
                           and float(item.get("coverage", 1.0)) < 0.25
                           and int(item.get("pixels", 10_000)) < 1600
                           and item.get("kind") != "wolf_track"
                           and any(max(abs(cell[0] - r), abs(cell[1] - c)) <= 1
                                   for r, c in cells))
            if cell in cells or weak_nearby:
                overlap.add(cell)
        eligible = (piece.get("species") == "goat" and overlap
                    and not piece.get("learned_template")
                    and not piece.get("learned_provisional"))
        facing = str(piece.get("facing") or "")
        if not eligible or facing not in directions or not cells:
            kept.append(piece)
            continue

        dr, dc = directions[facing]
        head = max(cells, key=lambda cell: cell[0] * dr + cell[1] * dc)
        perpendicular = (-dc, dr)
        forward = ray_clearance(head, (dr, dc))
        diagonals = [
            ray_clearance(head, (dr + perpendicular[0], dc + perpendicular[1])),
            ray_clearance(head, (dr - perpendicular[0], dc - perpendicular[1])),
        ]
        diagonal = max(diagonals)
        wolf_environment = (
            forward >= WOLF_FORWARD_MIN_CELLS
            and diagonal >= WOLF_DIAGONAL_MIN_CELLS
        )
        evidence = {
            "cells": [list(cell) for cell in sorted(cells)],
            "facing": facing,
            "head": list(head),
            "hazard_overlap": [list(cell) for cell in sorted(overlap)],
            "forward_clearance": int(forward),
            "diagonal_clearance": int(diagonal),
            "diagonal_sides": [int(value) for value in diagonals],
            "required_forward": WOLF_FORWARD_MIN_CELLS,
            "required_diagonal": WOLF_DIAGONAL_MIN_CELLS,
            "decision": "wolf" if wolf_environment else "goat",
        }
        decisions.append(evidence)
        if wolf_environment:
            rejected.append({
                **piece,
                "drop_reason": "wolf_environment_override",
                "wolf_environment": evidence,
            })
        else:
            piece.setdefault("direction_votes", {})["wolf_environment"] = evidence
            kept.append(piece)
    return kept, rejected, decisions


def reject_hazard_piece_overlaps(sheep, hazards):
    """Wolf occupancy wins over animal candidates assembled from wolf artwork."""
    hazard_cells = {
        (int(item["row"]), int(item["col"])) if isinstance(item, dict) else tuple(item)
        for item in (hazards or [])
    }
    kept, rejected = [], []
    for piece in sheep or []:
        overlap = hazard_cells & {tuple(cell) for cell in piece.get("cells", [])}
        if overlap:
            rejected.append({**piece, "drop_reason": "wolf_occupancy_override",
                             "overlap": [list(cell) for cell in sorted(overlap)]})
        else:
            kept.append(piece)
    return kept, rejected


def classify_black_sheep(sheep, hazards):
    """An orange-arrow sheep inside a dark 2x2 blob is a black sheep, not a wolf."""
    hazard_cells = {
        (int(item["row"]), int(item["col"])) if isinstance(item, dict) else tuple(item)
        for item in (hazards or [])
    }
    applied = []
    for piece in sheep or []:
        cells = {tuple(cell) for cell in piece.get("cells", [])}
        arrow_confirmed = "arrow" in set(piece.get("detectors") or [])
        local_dark = {
            (hr, hc) for hr, hc in hazard_cells
            if any(max(abs(r - hr), abs(c - hc)) <= 1 for r, c in cells)
        }
        if (piece.get("species", "sheep") == "sheep" and arrow_confirmed
                and cells & hazard_cells and len(local_dark) >= 3):
            piece["species"] = "black_sheep"
            piece.setdefault("confidence", {})["species"] = 0.98
            piece.setdefault("direction_votes", {})["black_sheep_dark_cluster"] = True
            piece["direction_votes"]["black_sheep_dark_cells"] = [
                list(cell) for cell in sorted(local_dark)]
            applied.append({
                "id": piece.get("id"),
                "cells": [list(cell) for cell in sorted(cells)],
                "dark_cells": [list(cell) for cell in sorted(local_dark)],
            })
    return sheep, applied


def recover_black_sheep_clusters(sheep, wolf_meta, rect, rows, cols,
                                 exclusion_mask=None):
    """Recover black sheep from strongly directional dark body components.

    Some black-sheep layouts hide the orange arrows almost completely.  Their
    sprites still form similarly sized, strongly oriented dark components.  A
    connected pack remains the strongest signal, but levels can also contain
    two separated black sheep.  In that case both components must independently
    show a much stronger face imbalance than a normal wolf before either is
    promoted.

    Sprite bounds, rather than only their centers, anchor the logical footprint:
    left/up sprites overhang the far side while right/down sprites overhang the
    near side.  This keeps both orientations on the same two grid cells.
    """
    components = [
        dict(item) for item in ((wolf_meta or {}).get("components") or [])
        if item.get("kind") == "small" and len(item.get("box") or []) == 4
    ]
    if not components:
        return sheep, []

    # Connected packs can use a lower per-sprite direction margin because their
    # repeated layout is already strong species evidence.
    centers = [tuple(map(float, item.get("center_rect") or (0, 0)))
               for item in components]
    remaining = set(range(len(components)))
    packs = []
    link_distance = CELL * 3.0
    while remaining:
        seed = remaining.pop()
        pack, frontier = {seed}, [seed]
        while frontier:
            current = frontier.pop()
            cx, cy = centers[current]
            linked = {
                index for index in remaining
                if np.hypot(centers[index][0] - cx, centers[index][1] - cy)
                <= link_distance
            }
            remaining -= linked
            pack |= linked
            frontier.extend(linked)
        if len(pack) >= 3:
            packs.append(sorted(pack))

    _body, cattle_face = cattle_masks(rect, exclusion_mask)
    descriptors = {}
    for component_index, component in enumerate(components):
        x, y, w, h = map(int, component["box"])
        long_side, short_side = max(w, h), max(1, min(w, h))
        if long_side / short_side < 1.30:
            continue
        axis = "H" if w >= h else "V"
        roi = cattle_face[max(0, y):min(cattle_face.shape[0], y + h),
                          max(0, x):min(cattle_face.shape[1], x + w)] > 0
        if roi.size == 0:
            continue
        height, width = roi.shape
        if axis == "H":
            first = int(roi[:, :max(1, width // 2)].sum())
            second = int(roi[:, width // 2:].sum())
            facing = "L" if first >= second else "R"
        else:
            first = int(roi[:max(1, height // 2), :].sum())
            second = int(roi[height // 2:, :].sum())
            facing = "U" if first >= second else "D"
        face_total = first + second
        face_margin = abs(first - second) / max(1.0, float(face_total))
        descriptors[component_index] = {
            "box": [x, y, w, h], "axis": axis, "facing": facing,
            "first": first, "second": second, "face_total": face_total,
            "face_margin": face_margin,
        }

    clustered = {
        component_index: pack_id
        for pack_id, pack in enumerate(packs, 1)
        for component_index in pack
        if component_index in descriptors
    }
    # A lone wolf can be somewhat asymmetric in one still frame.  Sparse black
    # sheep recovery therefore requires two independently strong components in
    # the same scene; one qualifying blob remains a hazard for manual review.
    sparse = [
        component_index for component_index, item in descriptors.items()
        if component_index not in clustered
        and item["face_total"] >= 180 and item["face_margin"] >= 0.12
    ]
    sparse = set(sparse if len(sparse) >= 2 else [])
    eligible = sorted(set(clustered) | sparse)
    if not eligible:
        return sheep, []
    eligible_boxes = [components[index]["box"] for index in eligible]

    pieces = list(sheep or [])
    applied = []
    recovered_boxes = []
    for component_index in eligible:
        component = components[component_index]
        item = descriptors[component_index]
        x, y, w, h = item["box"]
        axis, facing = item["axis"], item["facing"]
        first, second = item["first"], item["second"]
        face_total, face_margin = item["face_total"], item["face_margin"]
        min_margin = 0.05 if component_index in clustered else 0.12
        if face_total < 180 or face_margin < min_margin:
            continue
        cx, _cy = centers[component_index]
        if axis == "H":
            if facing == "L":
                row = int(np.floor(y / CELL))
                right_col = int(np.floor(cx / CELL))
                cells = [(row, right_col - 1), (row, right_col)]
            else:
                row = int(np.floor((y + h - 1) / CELL))
                left_col = int(np.floor(cx / CELL))
                cells = [(row, left_col), (row, left_col + 1)]
        else:
            col = int(np.floor(cx / CELL))
            if facing == "U":
                lower_row = int(np.floor(y / CELL))
            else:
                lower_row = int(np.floor((y + h - 1) / CELL))
            cells = [(lower_row - 1, col), (lower_row, col)]
        if any(not (0 <= row < rows and 0 <= col < cols) for row, col in cells):
            continue
        dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[facing]
        head = max(cells, key=lambda cell: cell[0] * dr + cell[1] * dc)
        rump = min(cells, key=lambda cell: cell[0] * dr + cell[1] * dc)
        placement = set(cells)
        exact = next((candidate for candidate in pieces
                      if {tuple(cell) for cell in candidate.get("cells", [])} == placement), None)
        overlaps = [candidate for candidate in pieces if candidate is not exact and
                    placement & {tuple(cell) for cell in candidate.get("cells", [])}]
        # Tan face/feet inside the same dark sprite can produce a displaced
        # one-cell-overlapping cattle candidate.  It is the same animal, not a
        # genuine occupancy conflict; the post-pass below removes it once the
        # black component has been accepted.
        def component_cattle(candidate):
            face_box = (candidate.get("direction_votes") or {}).get("face_box")
            if (candidate.get("species") != "cattle" or not isinstance(face_box, list)
                    or len(face_box) != 4):
                return False
            fx, fy, fw, fh = map(float, face_box)
            face_center = (fx + fw / 2.0, fy + fh / 2.0)
            return any(
                bx - 8 <= face_center[0] <= bx + bw + 8
                and by - 8 <= face_center[1] <= by + bh + 8
                for bx, by, bw, bh in eligible_boxes
            )
        overlaps = [candidate for candidate in overlaps if not component_cattle(candidate)]
        if overlaps:
            continue
        pack_id = clustered.get(component_index)
        detector = "black-pack" if pack_id is not None else "black-sparse-pair"
        evidence = {
            "black_sheep_component": component_index,
            "black_sheep_pack": pack_id,
            "black_sheep_sparse_pair": pack_id is None,
            "black_sheep_component_box": [x, y, w, h],
            "black_sheep_face_halves": [first, second],
        }
        if exact is None:
            exact = {
                "id": None,
                "source_id": f"{detector}:{pack_id or 0}:{component_index}",
                "detector": detector,
                "detectors": [detector],
                "cells": [list(rump), list(head)],
                "axis": axis,
                "rump": list(rump),
                "head": list(head),
                "facing": facing,
                "species": "black_sheep",
                "quality": 19000.0,
                "selection_score": 190.0,
                "confidence": {
                    "occupancy": 0.94 if pack_id is not None else 0.92,
                    "axis": 0.98,
                    "facing": round(min(0.99, 0.88 + face_margin), 4),
                    "species": 0.99, "detector_diversity": 0.3333,
                    "temporal_presence": 1.0, "temporal_facing": 1.0,
                },
                "direction_votes": evidence,
            }
            pieces.append(exact)
        else:
            exact["cells"] = [list(rump), list(head)]
            exact["rump"], exact["head"] = list(rump), list(head)
            exact["axis"], exact["facing"] = axis, facing
            exact["species"] = "black_sheep"
            exact["detectors"] = sorted(set(exact.get("detectors") or []) | {detector})
            exact.setdefault("confidence", {}).update({
                "axis": 0.98,
                "facing": round(min(0.99, 0.88 + face_margin), 4),
                "species": 0.99,
            })
            exact.setdefault("direction_votes", {}).update(evidence)
            exact.pop("review", None)
            exact.pop("review_reason", None)
        recovered_boxes.append([x, y, w, h])
        applied.append({
            "id": exact.get("id"), "cells": [list(rump), list(head)],
            "facing": facing, "component": component_index,
            "recovery": "pack" if pack_id is not None else "sparse_pair",
            "face_margin": round(float(face_margin), 4),
        })
    # Cattle-face detection can fire on the tan face/feet inside the same dark
    # sprite.  Once the component has been recovered as a black sheep, remove
    # any remaining cattle candidate whose visual face box belongs to that
    # component; otherwise one animal is counted twice in adjacent cells.
    filtered = []
    for piece in pieces:
        face_box = (piece.get("direction_votes") or {}).get("face_box")
        duplicate_cattle = False
        if piece.get("species") == "cattle" and isinstance(face_box, list) and len(face_box) == 4:
            fx, fy, fw, fh = map(float, face_box)
            face_center = (fx + fw / 2.0, fy + fh / 2.0)
            duplicate_cattle = any(
                bx - 8 <= face_center[0] <= bx + bw + 8
                and by - 8 <= face_center[1] <= by + bh + 8
                for bx, by, bw, bh in recovered_boxes
            )
        if not duplicate_cattle:
            filtered.append(piece)
    return filtered, applied


def reject_partial_exit_candidates(candidates, body_mask, rows, cols, *, enabled=True):
    """Drop outward-moving pieces whose visible body has crossed the grid edge.

    The perspective warp already excludes source pixels outside the calibrated
    quadrilateral.  During an exit animation, however, the last visible half of
    a sheep can still be paired with two edge cells.  A real edge piece remains
    centered over those two cells; a departing piece's body centroid is pulled
    strongly beyond its outward-facing head.  Wolves are detected separately
    and never pass through this filter.
    """
    if not enabled:
        return list(candidates or []), []
    height, width = body_mask.shape[:2]
    kept, rejected = [], []
    deltas = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    for candidate in candidates or []:
        cells = [tuple(cell) for cell in candidate.get("cells", [])]
        facing = str(candidate.get("facing") or "")
        head = tuple(candidate.get("head") or ())
        outward = (
            (facing == "U" and len(head) == 2 and head[0] == 0)
            or (facing == "D" and len(head) == 2 and head[0] == rows - 1)
            or (facing == "L" and len(head) == 2 and head[1] == 0)
            or (facing == "R" and len(head) == 2 and head[1] == cols - 1)
        )
        if len(cells) != 2 or not outward or facing not in deltas:
            kept.append(candidate)
            continue

        points_x, points_y = [], []
        for row, col in cells:
            y0, y1 = max(0, row * CELL), min(height, (row + 1) * CELL)
            x0, x1 = max(0, col * CELL), min(width, (col + 1) * CELL)
            ys, xs = np.where(body_mask[y0:y1, x0:x1] > 0)
            points_x.extend((xs + x0).tolist())
            points_y.extend((ys + y0).tolist())
        # Insufficient cyan/white body evidence is common for special pieces;
        # leave those to their dedicated detectors rather than guessing.
        if len(points_x) < 500:
            kept.append(candidate)
            continue

        cx, cy = float(np.mean(points_x)), float(np.mean(points_y))
        expected_x = float(np.mean([(col + 0.5) * CELL for _row, col in cells]))
        expected_y = float(np.mean([(row + 0.5) * CELL for row, _col in cells]))
        dr, dc = deltas[facing]
        outward_shift = (cx - expected_x) * dc + (cy - expected_y) * dr
        if outward_shift >= CELL * 0.30:
            rejected.append({
                **candidate,
                "drop_reason": "outside_calibration_region",
                "outside_shift_px": round(float(outward_shift), 2),
            })
        else:
            kept.append(candidate)
    return kept, rejected


def reject_departing_edge_pieces(pieces, body_mask, rows, cols, *, enabled=True):
    """Reject a clipped exit remnant even when its inferred facing points inward."""
    if not enabled:
        return list(pieces or []), []
    kept, rejected = [], []
    for piece in pieces or []:
        cells = [tuple(cell) for cell in piece.get("cells", [])]
        detectors = set(piece.get("detectors") or [])
        if (len(cells) != 2 or piece.get("species", "sheep") != "sheep"
                or detectors != {"body"}):
            kept.append(piece)
            continue
        rows_used = [cell[0] for cell in cells]
        cols_used = [cell[1] for cell in cells]
        edge = ("L" if min(cols_used) == 0 else "R" if max(cols_used) == cols - 1
                else "U" if min(rows_used) == 0 else "D" if max(rows_used) == rows - 1
                else None)
        if edge is None:
            kept.append(piece)
            continue

        points_x, points_y = [], []
        cell_support = []
        for row, col in cells:
            y0, y1 = row * CELL, (row + 1) * CELL
            x0, x1 = col * CELL, (col + 1) * CELL
            ys, xs = np.where(body_mask[y0:y1, x0:x1] > 0)
            cell_support.append(len(xs))
            points_x.extend((xs + x0).tolist())
            points_y.extend((ys + y0).tolist())
        if len(points_x) < 500 or min(cell_support, default=0) <= 0:
            kept.append(piece)
            continue
        cx, cy = float(np.mean(points_x)), float(np.mean(points_y))
        expected_x = float(np.mean([(col + 0.5) * CELL for _row, col in cells]))
        expected_y = float(np.mean([(row + 0.5) * CELL for row, _col in cells]))
        edge_shift = {"L": expected_x - cx, "R": cx - expected_x,
                      "U": expected_y - cy, "D": cy - expected_y}[edge]
        imbalance = max(cell_support) / max(1.0, float(min(cell_support)))
        temporal_presence = float((piece.get("confidence") or {}).get("temporal_presence", 1.0))
        strong_exit_shape = edge_shift >= CELL * 0.25 and imbalance >= 2.65
        new_edge_blob = temporal_presence <= 0.34 and edge_shift >= CELL * 0.20 and imbalance >= 2.0
        if strong_exit_shape or new_edge_blob:
            rejected.append({
                **piece,
                "drop_reason": "departing_edge_artifact",
                "edge": edge,
                "edge_shift_px": round(float(edge_shift), 2),
                "edge_support_ratio": round(float(imbalance), 3),
            })
        else:
            kept.append(piece)
    return kept, rejected


def apply_species_anchors(sheep, pink_candidates, pigs, goats):
    """Apply mutually exclusive current-frame species evidence by specificity."""
    pink_placements = {
        frozenset(tuple(cell) for cell in item.get("cells", []))
        for item in pink_candidates
    }
    pig_by_placement = {
        frozenset(tuple(cell) for cell in item.get("cells", [])): item
        for item in pigs
    }
    goat_placements = {
        frozenset(tuple(cell) for cell in item.get("cells", []))
        for item in goats
    }
    for piece in sheep:
        placement = frozenset(tuple(cell) for cell in piece.get("cells", []))
        if placement in pink_placements:
            # A magenta bow is more specific than the broad salmon pig-body
            # mask; pink sheep frequently satisfy both color detectors.
            piece["species"] = "pink_sheep"
            piece.pop("awake", None)
        elif placement in pig_by_placement:
            piece["species"] = "pig"
            piece["awake"] = bool(pig_by_placement[placement].get("awake"))
        elif placement in goat_placements:
            piece["species"] = "goat"
            piece.pop("awake", None)
    return sheep


def reject_internal_fence_overlaps(pieces, fences):
    """Drop animal candidates occupying cells that are actually wooden rails."""
    fence_cells = {
        tuple(item["cell"]) for item in (fences or [])
        if item.get("direction") in {"H", "V"}
    }
    if not fence_cells:
        return list(pieces), []
    kept, rejected = [], []
    for piece in pieces:
        overlap = fence_cells & {tuple(cell) for cell in piece.get("cells", [])}
        if overlap:
            rejected.append({
                **piece,
                "drop_reason": "internal_fence_occupancy_override",
                "fence_overlap": [list(cell) for cell in sorted(overlap)],
            })
        else:
            kept.append(piece)
    return kept, rejected


def analyze(game: np.ndarray, grid: G.BoardGrid, temporal_history=None,
            *, recover_missing_edges=False, ignore_outside_pieces=True):
    rect = grid.warp(game)
    gesture_mask, gesture_meta = gesture_occlusion(rect, grid.rows, grid.cols)
    body_mask, face_mask, dt = make_masks(rect, gesture_mask)
    markers, nseed = watershed_regions(rect, body_mask, dt)
    hazards, wolf_mask, wolf_meta = wolf_hazards(
        rect, grid.rows, grid.cols, gesture_mask)
    elephants, elephant_mask, elephant_meta = elephant_pieces(
        rect, grid.rows, grid.cols, gesture_mask)
    fences, fence_mask, fence_meta = fence_edges(rect, grid.rows, grid.cols)

    arrow_candidates = _arrow_candidates(
        rect, body_mask, grid.rows, grid.cols, gesture_mask)
    for candidate in arrow_candidates:
        candidate["detector"] = "arrow"
    gesture_target_candidates = _gesture_target_arrow_candidates(
        rect, body_mask, grid.rows, grid.cols, gesture_meta, arrow_candidates)
    body_candidates = []
    for idv in range(1, nseed + 1):
        cand = _score_region(idv, markers, body_mask, face_mask, dt, grid.rows, grid.cols)
        if cand is not None:
            cand["detector"] = "body"
            body_candidates.append(cand)
    rocket_candidates, rocket_mask = _rocket_candidates(
        rect, grid.rows, grid.cols, gesture_mask)
    cattle_candidates = _cattle_candidates(
        rect, grid.rows, grid.cols, gesture_mask)
    pink_candidates, pink_mask, pink_components = pink_sheep_candidates(
        rect, body_mask, face_mask, grid.rows, grid.cols, gesture_mask)
    pigs, pig_mask, pig_components = pig_candidates(
        rect, grid.rows, grid.cols, gesture_mask)
    goats, goat_mask, goat_components = goat_candidates(
        rect, grid.rows, grid.cols, gesture_mask)
    raw_candidates = (arrow_candidates + gesture_target_candidates
                      + body_candidates + rocket_candidates
                      + cattle_candidates + pink_candidates + pigs + goats)
    # Bomb counters are current-frame species evidence.  Attach them before
    # consulting saved templates so a manually corrected bomb footprint can
    # use native bomb candidates as its visual anchor.
    bomb_meta, bomb_mask = bomb_markers(rect, grid.rows, grid.cols, gesture_mask)
    for candidate in raw_candidates:
        cells = {tuple(cell) for cell in candidate.get("cells", [])}
        marker = next((item for item in bomb_meta if tuple(item["cell"]) in cells), None)
        if marker and candidate.get("species", "sheep") in {"sheep", "rocket", "bomb"}:
            candidate["species"] = "bomb"
            candidate["hit_limit"] = marker["hit_limit"]
            candidate["hits_remaining"] = marker["hits_remaining"]
            candidate["counter_confident"] = marker["counter_confident"]
            candidate["counter_confidence"] = marker["counter_confidence"]
            candidate["counter_unknown"] = marker["counter_unknown"]
            candidate.setdefault("direction_votes", {})["bomb_counter"] = list(marker["cell"])
            candidate["direction_votes"]["bomb_counter_value"] = marker["hits_remaining"]
            candidate["direction_votes"]["bomb_counter_confidence"] = marker["counter_confidence"]
    learned_candidates, manual_learning = recognition.manual_candidate_proposals(
        rect, grid.rows, grid.cols, raw_candidates)
    raw_candidates += learned_candidates
    raw_candidates, fence_candidate_rejected = reject_internal_fence_overlaps(
        raw_candidates, fences)
    bomb_learning_rejected = []
    bomb_checked_candidates = []
    for candidate in raw_candidates:
        cells = {tuple(cell) for cell in candidate.get("cells", [])}
        marker = next((item for item in bomb_meta if tuple(item["cell"]) in cells), None)
        if (marker and candidate.get("learned_template")
                and candidate.get("species", "sheep") != "bomb"):
            bomb_learning_rejected.append({
                **candidate, "drop_reason": "strong_bomb_counter_override",
                "bomb_cell": list(marker["cell"]),
            })
            continue
        if marker and candidate.get("species", "sheep") in {"sheep", "rocket", "bomb"}:
            candidate["species"] = "bomb"
            candidate["hit_limit"] = marker["hit_limit"]
            candidate["hits_remaining"] = marker["hits_remaining"]
            candidate["counter_confident"] = marker["counter_confident"]
            candidate["counter_confidence"] = marker["counter_confidence"]
            candidate["counter_unknown"] = marker["counter_unknown"]
            candidate.setdefault("direction_votes", {})["bomb_counter"] = list(marker["cell"])
            candidate["direction_votes"]["bomb_counter_value"] = marker["hits_remaining"]
            candidate["direction_votes"]["bomb_counter_confidence"] = marker["counter_confidence"]
        bomb_checked_candidates.append(candidate)
    raw_candidates = bomb_checked_candidates
    raw_candidates, manual_learning_rejected, manual_rejections = (
        recognition.manual_candidate_rejections(rect, raw_candidates)
    )
    eligible_candidates, outside_rejected = reject_partial_exit_candidates(
        raw_candidates, body_mask, grid.rows, grid.cols,
        enabled=bool(ignore_outside_pieces))
    candidates, fusion_rejected = recognition.fuse_candidates(eligible_candidates)
    sheep, dropped, optimization = resolve_candidates(candidates, return_meta=True)
    elephant_cells = {tuple(cell) for item in elephants for cell in item["cells"]}
    if elephant_cells:
        retained = []
        for piece in sheep:
            overlap = elephant_cells & {tuple(cell) for cell in piece.get("cells", [])}
            if overlap:
                dropped.append({**piece, "drop_reason": "elephant_occupancy_override",
                                "overlap": [list(cell) for cell in sorted(overlap)]})
            else:
                retained.append(piece)
        sheep = retained + elephants
    sheep, hazards, temporal = recognition.apply_temporal(
        sheep, hazards, list(temporal_history or []), grid.rows, grid.cols,
        recover_missing_edges=(recover_missing_edges and not ignore_outside_pieces))
    sheep, fence_temporal_rejected = reject_internal_fence_overlaps(sheep, fences)
    # Current-frame species anchors override older temporal labels.  Resolve
    # overlapping color detectors in one place so a pig mask cannot overwrite
    # the more distinctive pink-sheep bow.
    apply_species_anchors(sheep, pink_candidates, pigs, goats)
    sheep, departing_rejected = reject_departing_edge_pieces(
        sheep, body_mask, grid.rows, grid.cols,
        enabled=bool(ignore_outside_pieces))
    # A stable-history vote must not turn a currently visible bomb back into a
    # rocket.  The counter disc is direct frame evidence and wins here.
    for piece in sheep:
        cells = {tuple(cell) for cell in piece.get("cells", [])}
        marker = next((item for item in bomb_meta if tuple(item["cell"]) in cells), None)
        if marker and piece.get("species") in {"sheep", "rocket", "bomb"}:
            piece["species"] = "bomb"
            piece["hit_limit"] = marker["hit_limit"]
            piece["hits_remaining"] = marker["hits_remaining"]
            piece["counter_confident"] = marker["counter_confident"]
            piece["counter_confidence"] = marker["counter_confidence"]
            piece["counter_unknown"] = marker["counter_unknown"]
            piece.setdefault("direction_votes", {})["bomb_counter"] = list(marker["cell"])
            piece["direction_votes"]["bomb_counter_value"] = marker["hits_remaining"]
            piece["direction_votes"]["bomb_counter_confidence"] = marker["counter_confidence"]
    # Manual direction evidence is authoritative over stale temporal votes.
    # Apply it only after current-frame species anchors (pig/goat/bomb) settle.
    sheep, learned_directions = recognition.apply_direction_learning(sheep)
    sheep, learned_labels = recognition.apply_manual_label_learning(sheep, rect)
    sheep, black_sheep_cluster = recover_black_sheep_clusters(
        sheep, wolf_meta, rect, grid.rows, grid.cols, gesture_mask)
    sheep, goat_wolf_rejected, goat_wolf_environment = resolve_goat_wolf_conflicts(
        sheep, hazards, grid.rows, grid.cols)
    hazards, cluster_suppressed_hazards = suppress_special_hazard_overlaps(sheep, hazards)
    sheep, black_sheep_applied = classify_black_sheep(sheep, hazards)
    hazards, isolated_suppressed_hazards = suppress_special_hazard_overlaps(sheep, hazards)
    suppressed_hazards = cluster_suppressed_hazards + isolated_suppressed_hazards
    sheep, wolf_overlap_rejected = reject_hazard_piece_overlaps(sheep, hazards)
    # Candidate order and temporal restoration order are implementation
    # details.  Stable spatial ids keep equivalent boards on the same revision
    # and make cached move ids reproducible across adjacent captures.
    sheep.sort(key=lambda item: (
        min(tuple(cell) for cell in item.get("cells", [[grid.rows, grid.cols]])),
        tuple(sorted(tuple(cell) for cell in item.get("cells", []))),
        str(item.get("facing") or ""), str(item.get("species") or "sheep"),
    ))
    for piece_id, piece in enumerate(sheep):
        piece["id"] = piece_id
    dropped = (fence_candidate_rejected + fence_temporal_rejected
               + bomb_learning_rejected + manual_learning_rejected
               + outside_rejected + departing_rejected
               + fusion_rejected + dropped + goat_wolf_rejected + wolf_overlap_rejected)
    debug = {
        "grid": grid,
        "rect": rect,
        "body_mask": body_mask,
        "face_mask": face_mask,
        "arrow_mask": arrow_mask(rect, gesture_mask),
        "rocket_mask": rocket_mask,
        "pink_mask": pink_mask,
        "pink_sheep": pink_components,
        "pig_mask": pig_mask,
        "pigs": pig_components,
        "goat_mask": goat_mask,
        "goats": goat_components,
        "goat_wolf_environment": goat_wolf_environment,
        "bomb_mask": bomb_mask,
        "bomb_meta": bomb_meta,
        "gesture_mask": gesture_mask,
        "gesture": gesture_meta,
        "wolf_mask": wolf_mask,
        "wolf_meta": wolf_meta,
        "elephant_mask": elephant_mask,
        "elephant_meta": elephant_meta,
        "fence_mask": fence_mask,
        "fence_meta": fence_meta,
        "fences": fences,
        "hazards": hazards,
        "suppressed_hazards": suppressed_hazards,
        "black_sheep_cluster": black_sheep_cluster,
        "black_sheep_applied": black_sheep_applied,
        "markers": markers,
        "detector": "fusion",
        "candidate_count": len(candidates),
        "raw_candidate_count": len(raw_candidates),
        "outside_rejected_count": len(outside_rejected),
        "departing_rejected_count": len(departing_rejected),
        "ignore_outside_pieces": bool(ignore_outside_pieces),
        "learned_directions": learned_directions,
        "learned_labels": learned_labels,
        "manual_learning": manual_learning + manual_rejections,
        "manual_learning_rejections": manual_rejections,
        "provisional_learning_rejection_count": len({
            tuple(sorted(tuple(cell) for cell in item.get("cells", [])))
            for item in manual_learning_rejected if item.get("learned_provisional")
        }),
        "learned_candidate_count": sum(
            bool(item.get("learned_template")) for item in raw_candidates),
        "learned_rejected": bomb_learning_rejected,
        "dropped": dropped,
        "candidates": candidates,
        "raw_candidates": raw_candidates,
        "optimization": optimization,
        "temporal": temporal,
        "observation": recognition.observation_record(sheep, hazards, grid.rows, grid.cols),
    }
    return sheep, debug


def detect(game, corners, rows, cols):
    """Return sheep list: each has cells/rump/head/facing. cells[0]=rump, cells[1]=head."""
    grid = _grid_from_args(game, corners, rows, cols)
    sheep, _debug = analyze(game, grid)
    return sheep


def _to_px(Minv, c, r):
    v = Minv @ np.array([c * CELL, r * CELL, 1.0])
    return v[0] / v[2], v[1] / v[2]


AXIS_COLORS = {
    "V": (216, 126, 47),   # BGR, blue
    "H": (47, 135, 216),   # BGR, amber
}
SPECIES_COLORS = {
    "sheep": None,          # sheep keep axis colors
    "cattle": (190, 92, 155),
    "wolf": (112, 105, 94),
    "pig": (182, 94, 210),
    "goat": (92, 154, 205),
    "rocket": (44, 66, 214),
    "elephant": (156, 111, 118),
    "bomb": (42, 42, 210),
    "pink_sheep": (180, 82, 224),
}
GRID_COLOR = (246, 248, 250)
INK = (34, 38, 45)
FOCUS_RED = (58, 76, 200)
HAZARD_COLOR = (112, 105, 94)


def _piece_color(piece):
    if piece.get("review"):
        return FOCUS_RED
    species = piece.get("species", "sheep")
    return SPECIES_COLORS.get(species) or AXIS_COLORS[piece["axis"]]


def _piece_label(piece):
    prefix = {"cattle": "C", "wolf": "W",
              "pig": ("P" if piece.get("awake") else "ZP"), "rocket": "R",
              "goat": "G",
              "elephant": "E", "pink_sheep": "L",
              "bomb": f"B{piece.get('hits_remaining', 3)}-"}.get(
                  piece.get("species"), "")
    label = f"{prefix}{piece['id']}" if prefix else str(piece["id"])
    return f"?{label}" if piece.get("review") else label


def _blend_poly(img, pts, color, alpha):
    layer = img.copy()
    cv2.fillPoly(layer, [np.array(pts, np.int32)], color)
    cv2.addWeighted(layer, alpha, img, 1.0 - alpha, 0, img)


def _draw_soft_poly(img, pts, color, alpha=0.14):
    pts = np.array(pts, np.int32)
    _blend_poly(img, pts, color, alpha)
    cv2.polylines(img, [pts], True, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)


def _draw_arrow(img, p0, p1, color, thickness=3):
    p0 = tuple(map(int, p0))
    p1 = tuple(map(int, p1))
    cv2.arrowedLine(img, p0, p1, (255, 255, 255), thickness + 5, cv2.LINE_AA, tipLength=0.32)
    cv2.arrowedLine(img, p0, p1, INK, thickness + 2, cv2.LINE_AA, tipLength=0.32)
    cv2.arrowedLine(img, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.32)
    cv2.circle(img, p1, max(4, thickness + 1), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, p1, max(2, thickness), color, -1, cv2.LINE_AA)


def _piece_arrow_points(piece):
    """Return a centered arrow in rectified-grid pixels.

    Elephant footprints span three cells in their travel direction.  Their
    arrow occupies only the forward part of the body so it stays readable and
    does not cut through the central id badge.
    """
    cells = [tuple(cell) for cell in piece.get("cells") or []]
    facing = str(piece.get("facing") or "")
    delta = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}.get(facing)
    if not cells or delta is None:
        return None
    dr, dc = delta
    projections = [row * dr + col * dc for row, col in cells]
    lo, hi = min(projections), max(projections)
    rump_cells = [cell for cell, value in zip(cells, projections) if value == lo]
    head_cells = [cell for cell, value in zip(cells, projections) if value == hi]

    def center(group):
        return np.array([
            sum((col + 0.5) * CELL for _row, col in group) / len(group),
            sum((row + 0.5) * CELL for row, _col in group) / len(group),
        ], dtype=np.float64)

    start, end = center(rump_cells), center(head_cells)
    if piece.get("species") == "elephant":
        vector = end - start
        start, end = start + vector * 0.46, start + vector * 0.88
    return tuple(start), tuple(end)


def _round_rect(img, p0, p1, radius, fill, border=None, alpha=1.0):
    x0, y0 = map(int, p0)
    x1, y1 = map(int, p1)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1] - 1, x1), min(img.shape[0] - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return
    radius = int(min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
    target = img.copy() if alpha < 1.0 else img

    cv2.rectangle(target, (x0 + radius, y0), (x1 - radius, y1), fill, -1, cv2.LINE_AA)
    cv2.rectangle(target, (x0, y0 + radius), (x1, y1 - radius), fill, -1, cv2.LINE_AA)
    for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x1 - radius, y1 - radius), (x0 + radius, y1 - radius)):
        cv2.circle(target, (cx, cy), radius, fill, -1, cv2.LINE_AA)

    if border is not None:
        cv2.rectangle(target, (x0 + radius, y0), (x1 - radius, y1), border, 2, cv2.LINE_AA)
        cv2.rectangle(target, (x0, y0 + radius), (x1, y1 - radius), border, 2, cv2.LINE_AA)
        for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                       (x1 - radius, y1 - radius), (x0 + radius, y1 - radius)):
            cv2.circle(target, (cx, cy), radius, border, 2, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(target, alpha, img, 1.0 - alpha, 0, img)


def _draw_badge(img, top_left, text, color):
    x, y = tuple(map(int, top_left))
    scale = 0.36 if len(text) < 2 else 0.32
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    width = max(23, tw + 12)
    height = 18
    x = min(max(4, x), img.shape[1] - width - 4)
    y = min(max(4, y), img.shape[0] - height - 4)
    _round_rect(img, (x + 1, y + 2), (x + width + 1, y + height + 2), 9, (0, 0, 0), alpha=0.18)
    _round_rect(img, (x, y), (x + width, y + height), 9, (255, 255, 255), border=color)
    cv2.putText(img, text, (x + width // 2 - tw // 2, y + height // 2 + th // 2 - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, INK, 1, cv2.LINE_AA)


def _draw_text_with_halo(img, text, org, scale, color=INK, thickness=1):
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def render(game, sheep, corners, rows, cols):
    """Overlay occupied cells and head arrows on the source image."""
    rect, Minv, src = _warp(game, corners, rows, cols)
    ov = game.copy()
    for s in sheep:
        col = _piece_color(s)
        for row, cell_col in s["cells"]:
            poly = [
                _to_px(Minv, cell_col, row),
                _to_px(Minv, cell_col + 1, row),
                _to_px(Minv, cell_col + 1, row + 1),
                _to_px(Minv, cell_col, row + 1),
            ]
            _draw_soft_poly(ov, poly, col)
        arrow = _piece_arrow_points(s)
        if arrow:
            p0 = tuple(map(int, _to_px(Minv, arrow[0][0] / CELL, arrow[0][1] / CELL)))
            p1 = tuple(map(int, _to_px(Minv, arrow[1][0] / CELL, arrow[1][1] / CELL)))
            _draw_arrow(ov, p0, p1, col, thickness=2)
    for r in range(int(rows) + 1):
        cv2.line(ov, tuple(map(int, _to_px(Minv, 0, r))), tuple(map(int, _to_px(Minv, cols, r))), GRID_COLOR, 1, cv2.LINE_AA)
    for c in range(int(cols) + 1):
        cv2.line(ov, tuple(map(int, _to_px(Minv, c, 0))), tuple(map(int, _to_px(Minv, c, rows))), GRID_COLOR, 1, cv2.LINE_AA)
    return ov, src


def render_rect_debug(debug, sheep):
    grid: G.BoardGrid = debug["grid"]
    vis = debug["rect"].copy()
    grid_layer = vis.copy()
    for r in range(grid.rows + 1):
        cv2.line(grid_layer, (0, r * CELL), (grid.cols * CELL, r * CELL), GRID_COLOR, 1, cv2.LINE_AA)
    for c in range(grid.cols + 1):
        cv2.line(grid_layer, (c * CELL, 0), (c * CELL, grid.rows * CELL), GRID_COLOR, 1, cv2.LINE_AA)
    cv2.addWeighted(grid_layer, 0.22, vis, 0.78, 0, vis)
    gesture_mask = debug.get("gesture_mask")
    if isinstance(gesture_mask, np.ndarray) and np.any(gesture_mask):
        tint = np.zeros_like(vis)
        tint[:] = FOCUS_RED
        selected = gesture_mask > 0
        vis[selected] = (0.55 * vis[selected] + 0.45 * tint[selected]).astype(np.uint8)
        contours, _ = cv2.findContours(gesture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, FOCUS_RED, 2, cv2.LINE_AA)
    for hz in debug.get("hazards", []):
        row, col = int(hz["row"]), int(hz["col"])
        pts = [
            (col * CELL + 5, row * CELL + 5),
            ((col + 1) * CELL - 5, row * CELL + 5),
            ((col + 1) * CELL - 5, (row + 1) * CELL - 5),
            (col * CELL + 5, (row + 1) * CELL - 5),
        ]
        _draw_soft_poly(vis, pts, HAZARD_COLOR, alpha=0.20)
        _draw_badge(vis, (col * CELL + 12, row * CELL + 12), "W", HAZARD_COLOR)
    for s in sheep:
        color = _piece_color(s)
        for row, col in s["cells"]:
            pts = [
                (col * CELL + 7, row * CELL + 7),
                ((col + 1) * CELL - 7, row * CELL + 7),
                ((col + 1) * CELL - 7, (row + 1) * CELL - 7),
                (col * CELL + 7, (row + 1) * CELL - 7),
            ]
            _draw_soft_poly(vis, pts, color)
        arrow = _piece_arrow_points(s)
        if arrow:
            _draw_arrow(vis, arrow[0], arrow[1], color, thickness=2)
        min_row = min(row for row, _ in s["cells"])
        min_col = min(col for _, col in s["cells"])
        _draw_badge(vis, (min_col * CELL + 7, min_row * CELL + 7), _piece_label(s), color)
    return vis


def render_grid_labels(debug, sheep):
    """Board audit image with spreadsheet-like coordinates and sheep ids."""
    grid: G.BoardGrid = debug["grid"]
    board = render_rect_debug(debug, sheep)
    top, left, pad = 42, 48, 10
    h, w = board.shape[:2]
    canvas = np.full((top + h + pad, left + w + pad, 3), (245, 247, 248), np.uint8)
    canvas[top:top + h, left:left + w] = board

    # Header bands.
    cv2.rectangle(canvas, (left, 0), (left + w, top - 1), (238, 242, 244), -1)
    cv2.rectangle(canvas, (0, top), (left - 1, top + h), (238, 242, 244), -1)
    cv2.line(canvas, (left, top - 1), (left + w, top - 1), (194, 202, 210), 1, cv2.LINE_AA)
    cv2.line(canvas, (left - 1, top), (left - 1, top + h), (194, 202, 210), 1, cv2.LINE_AA)

    for c in range(grid.cols):
        label = chr(ord("A") + c) if c < 26 else f"C{c + 1}"
        cx = left + c * CELL + CELL // 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        _draw_text_with_halo(canvas, label, (cx - tw // 2, 27), 0.58, INK, 1)
    for r in range(grid.rows):
        label = str(r + 1)
        cy = top + r * CELL + CELL // 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        _draw_text_with_halo(canvas, label, (left - 14 - tw, cy + th // 2), 0.48, INK, 1)

    _draw_text_with_halo(canvas, "A1", (12, 27), 0.5, (80, 84, 92), 1)
    return canvas


def render_layout(debug, sheep):
    """Synthetic 2D layout audit, independent of screenshot texture/skin."""
    grid: G.BoardGrid = debug["grid"]
    h, w = grid.rows * CELL, grid.cols * CELL
    board = np.full((h, w, 3), (234, 222, 183), np.uint8)
    for r in range(grid.rows):
        for c in range(grid.cols):
            color = (188, 178, 71) if (r + c) % 2 == 0 else (247, 180, 109)
            cv2.rectangle(board, (c * CELL, r * CELL), ((c + 1) * CELL, (r + 1) * CELL), color, -1)
    grid_layer = board.copy()
    for r in range(grid.rows + 1):
        cv2.line(grid_layer, (0, r * CELL), (w, r * CELL), (255, 255, 255), 1, cv2.LINE_AA)
    for c in range(grid.cols + 1):
        cv2.line(grid_layer, (c * CELL, 0), (c * CELL, h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(grid_layer, 0.28, board, 0.72, 0, board)

    debug2 = {**debug, "rect": board}
    return render_grid_labels(debug2, sheep)


def render_segments(debug):
    markers = debug["markers"]
    rect = debug["rect"]
    out = rect.copy()
    rng = np.random.default_rng(12)
    ids = [int(x) for x in np.unique(markers) if x > 0]
    for idv in ids:
        if idv == max(ids):
            continue
        color = rng.integers(40, 230, size=3, dtype=np.uint8).tolist()
        mask = markers == idv
        out[mask] = (0.55 * out[mask] + 0.45 * np.array(color)).astype(np.uint8)
    return out


def _remove_obsolete_images():
    for name in (
        "_occ_axis.png",
        "_occ_axis_zoom.png",
        "_sheep_face_mask.png",
        "_sheep_mask.png",
        "_sheep_segments.png",
    ):
        path = image_path(name)
        if path.exists():
            path.unlink()


def _hazard_cells(hazards):
    return [[int(h["row"]), int(h["col"])] if isinstance(h, dict) else list(h) for h in (hazards or [])]


def to_board(sheep, rows, cols, model="facing", slide_mode="all", hazards=None, fences=None):
    # Wolves move continuously.  Their observed body cells belong to the live
    # execution guard and sandbox annotation, never to the solver's permanent
    # obstacle set.  Untagged/manual hazards remain real board obstacles.
    static_hazards = [
        item for item in (hazards or [])
        if not (isinstance(item, dict) and item.get("kind") == "wolf_body")
    ]
    return {
        "rows": rows,
        "cols": cols,
        "model": model,
        "slide_mode": slide_mode,
        "hazards": _hazard_cells(static_hazards),
        "fences": [{"cell": list(item["cell"]), "direction": item["direction"]}
                   for item in (fences or [])],
        "pieces": {
            str(i): {"cells": [list(c) for c in s["cells"]],
                     "facing": s["facing"],
                     "species": s.get("species", "sheep"),
                     **({"awake": bool(s.get("awake"))}
                        if s.get("species") == "pig" else {}),
                     **({"hit_limit": s.get("hit_limit", 3),
                         "hits_remaining": s.get("hits_remaining", 3)}
                        if s.get("species") == "bomb" else {}),
                     **({"review": True, "review_reason": s.get("review_reason")}
                        if s.get("review") else {})}
            for i, s in enumerate(sheep)
        },
    }


def to_layout(sheep, rows, cols, dropped=None, hazards=None, fences=None):
    hazard_cells = {tuple(x) for x in _hazard_cells(hazards)}
    occupied: dict[tuple[int, int], list[dict]] = {}
    pieces = []
    for s in sheep:
        pid = int(s["id"])
        cells = [tuple(rc) for rc in s["cells"]]
        piece = {
            "id": pid,
            "species": s.get("species", "sheep"),
            "awake": (bool(s.get("awake")) if s.get("species") == "pig" else None),
            "hit_limit": s.get("hit_limit"),
            "hits_remaining": s.get("hits_remaining"),
            "cells": [list(rc) for rc in cells],
            "rump": list(s["rump"]),
            "head": list(s["head"]),
            "axis": s["axis"],
            "facing": s["facing"],
            "source_id": s.get("source_id", f"manual:{pid}"),
            "quality": s.get("quality", 1.0 if s.get("manual") else 0.0),
            "direction_confidence": s.get("direction_confidence"),
            "confidence": s.get("confidence"),
        }
        pieces.append(piece)
        for rc in cells:
            role = "head" if list(rc) == s["head"] else "rump"
            occupied.setdefault(rc, []).append({"piece_id": pid, "role": role})

    dropped_by_cell: dict[tuple[int, int], list] = {}
    for cand in dropped or []:
        sid = cand.get("source_id", -1)
        for rc in cand.get("cells", []):
            dropped_by_cell.setdefault(tuple(rc), []).append(sid)

    cells = []
    empty = []
    conflicts = []
    for r in range(rows):
        row = []
        for c in range(cols):
            occ = occupied.get((r, c), [])
            dropped_ids = sorted(set(dropped_by_cell.get((r, c), [])), key=str)
            cell = {
                "row": r,
                "col": c,
                "label": f"{chr(ord('A') + c)}{r + 1}" if c < 26 else f"C{c + 1}R{r + 1}",
                "occupied": bool(occ),
                "hazard": (r, c) in hazard_cells,
                "piece_ids": [x["piece_id"] for x in occ],
                "roles": {str(x["piece_id"]): x["role"] for x in occ},
                "dropped_source_ids": dropped_ids,
            }
            if not occ:
                empty.append([r, c])
            if len(occ) > 1:
                conflicts.append({"cell": [r, c], "piece_ids": [x["piece_id"] for x in occ]})
            row.append(cell)
        cells.append(row)

    vertical = sum(1 for s in sheep if s["axis"] == "V")
    facing_counts = {d: sum(1 for s in sheep if s["facing"] == d) for d in "UDLR"}
    return {
        "rows": rows,
        "cols": cols,
        "fences": [{"cell": list(item["cell"]), "direction": item["direction"]}
                   for item in (fences or [])],
        "cell": CELL,
        "piece_count": len(pieces),
        "occupied_count": sum(len(s["cells"]) for s in sheep),
        "empty_count": len(empty),
        "conflicts": conflicts,
        "axis_counts": {"V": vertical, "H": len(pieces) - vertical},
        "facing_counts": facing_counts,
        "pieces": pieces,
        "cells": cells,
        "empty_cells": empty,
        "hazards": _hazard_cells(hazards),
    }


def _conflicts(sheep):
    occ = {}
    for i, s in enumerate(sheep):
        for rc in s["cells"]:
            occ.setdefault(tuple(rc), []).append(i)
    return occ, {k: v for k, v in occ.items() if len(v) > 1}


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(image_path("_game.png")))
    ap.add_argument("--params", default="grid_params.json")
    ap.add_argument("--board", default="board.json")
    args = ap.parse_args(argv)

    game = cv2.imread(args.image)
    if game is None:
        raise SystemExit(f"读不到图片: {args.image}")
    _remove_obsolete_images()
    grid = G.load_grid(args.params, game)
    G.save_grid_data(grid, "board_grid.json")

    sheep, debug = analyze(game, grid)
    occ, conflicts = _conflicts(sheep)
    vertical = sum(1 for s in sheep if s["axis"] == "V")
    facing_n = " ".join(f"{d}{sum(1 for s in sheep if s['facing'] == d)}" for d in "UDLR")

    print(f"网格 {grid.rows}x{grid.cols}  -> board_grid.json")
    print(f"候选 {debug['candidate_count']}  保留 {len(sheep)}  丢弃 {len(debug['dropped'])}")
    print(f"羊 {len(sheep)} 只  竖 {vertical}  横 {len(sheep)-vertical}   头向 {facing_n}")
    print(f"狼危险格 {len(debug.get('hazards', []))}")
    print(f"占用格 {len(occ)}/{grid.rows * grid.cols}  冲突 {len(conflicts)}")
    for rc, ids in sorted(conflicts.items()):
        print(f"  冲突 {rc} <- {ids}")

    board_data = to_board(sheep, grid.rows, grid.cols, hazards=debug.get("hazards"),
                          fences=debug.get("fences"))
    layout_data = to_layout(sheep, grid.rows, grid.cols, debug["dropped"],
                            hazards=debug.get("hazards"), fences=debug.get("fences"))
    try:
        params_data = json.load(open(args.params, encoding="utf-8"))
    except Exception:
        params_data = None
    calibration_blockers, calibration_warnings = safety.validate_calibration(params_data, game.shape)
    report = safety.classify_scene(
        game, debug, sheep, grid.rows, grid.cols,
        calibration_blockers=calibration_blockers, layout=layout_data)
    report["warnings"] = calibration_warnings
    try:
        board_io.validate_board_data(board_data)
    except board_io.BoardValidationError as exc:
        report = safety.add_blockers(report, [
            safety.blocker("board_schema_invalid", "棋盘结构校验失败", detail=exc.errors)
        ])
    _write_json("scene_report.json", report)
    _write_json("sheep_candidates.json", {
        "scene_state": report["scene_state"],
        "scene_reason": report["scene_reason"],
        "execution_blockers": report["execution_blockers"],
        "metrics": report["metrics"],
        "kept": sheep,
        "hazards": debug.get("hazards", []),
        "wolf": debug.get("wolf_meta"),
        "black_sheep_cluster": debug.get("black_sheep_cluster", []),
        "black_sheep_applied": debug.get("black_sheep_applied", []),
        "pink_sheep": debug.get("pink_sheep", []),
        "pigs": debug.get("pigs", []),
        "goats": debug.get("goats", []),
        "fusion": {"detector": debug.get("detector"),
                   "raw_candidate_count": debug.get("raw_candidate_count"),
                   "fused_candidate_count": debug.get("candidate_count"),
                   "optimization": debug.get("optimization")},
        "temporal": debug.get("temporal"),
        "dropped": debug["dropped"],
        "raw": debug.get("raw_candidates", debug["candidates"]),
        "fused": debug["candidates"],
    })

    cv2.imwrite(str(image_path("_occ_axis_rect.png")), render_rect_debug(debug, sheep))
    cv2.imwrite(str(image_path("_grid_labels.png")), render_grid_labels(debug, sheep))
    cv2.imwrite(str(image_path("_layout.png")), render_layout(debug, sheep))
    if report["scene_state"] != "gameplay":
        for stale in (Path(args.board), Path("board_layout.json")):
            if stale.exists():
                stale.unlink()
        print(f"场景 {report['scene_state']}：{report['scene_reason']}")
        print("禁止生成 board.json、求解和执行")
        raise SystemExit(2)

    _write_json(args.board, board_data)
    _write_json("board_layout.json", layout_data)
    cache_meta = level_cache.save_capture(
        board_data,
        source="cli-detect",
        extra={"rows": grid.rows, "cols": grid.cols, "candidate_count": debug["candidate_count"],
               "hazard_count": len(debug.get("hazards", [])),
               "scene_state": report["scene_state"], "executable": report["executable"],
               "execution_blockers": [item["code"] for item in report["execution_blockers"]]},
    )
    print("saved board.json / board_layout.json / sheep_candidates.json / images/_occ_axis_rect.png / images/_grid_labels.png / images/_layout.png")
    print(f"cache {cache_meta['capture_id']} -> cache/levels/{cache_meta['level_key']}")


if __name__ == "__main__":
    main()
