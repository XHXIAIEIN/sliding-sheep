"""Watershed segmentation, arrow candidates, and region scoring."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import label as ndlabel, maximum_filter
from .masks import CELL, DIRS, _cell_count, _cell_of, arrow_mask
from .species_special import cattle_masks


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
