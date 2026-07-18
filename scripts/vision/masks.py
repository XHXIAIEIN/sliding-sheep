"""Board rectification, shared grid constants, colour masks, and gesture occlusion."""
from __future__ import annotations

import cv2
import numpy as np
import board_grid as G


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
