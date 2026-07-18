"""Candidate resolution and cross-detector conflict rules."""
from __future__ import annotations

import numpy as np
import recognition
from .masks import CELL
from .species_special import cattle_masks


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
