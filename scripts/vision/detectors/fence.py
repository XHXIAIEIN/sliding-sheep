"""Fence detector: board-edge and internal fence segments."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import CELL, _cell_count
from .cattle import cattle_masks


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
