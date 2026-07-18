"""Fence edge and wolf hazard detection."""
from __future__ import annotations

import cv2
import numpy as np
from .masks import CELL, _cell_count, _cell_of, _exclude
from .species_special import cattle_masks


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
