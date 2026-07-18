"""Black sheep detector: reclassify dark bodies and recover clusters that the wolf detector absorbed."""
from __future__ import annotations

import numpy as np
from ..masks import CELL
from .cattle import cattle_masks


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
