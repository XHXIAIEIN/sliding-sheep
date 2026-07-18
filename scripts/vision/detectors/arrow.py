"""Facing-arrow detector: the primary direction evidence for two-cell sheep."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import CELL, _cell_count, arrow_mask


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
