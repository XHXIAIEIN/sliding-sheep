"""Wolf hazard detector: dark predator bodies that block execution."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import CELL, _cell_of, _exclude


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
