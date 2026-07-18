"""Rocket sheep detector: saturated rocket-body colour mask."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import DIRS, _cell_count, _exclude
from .cattle import cattle_masks


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
