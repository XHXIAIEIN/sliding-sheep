"""Candidate resolution and cross-detector conflict rules."""
from __future__ import annotations

import numpy as np
import recognition
from .masks import CELL


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
