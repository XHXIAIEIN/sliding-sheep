"""Detector calibration, candidate fusion, and global cell assignment."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from .features import _clip, cell_key


DETECTOR_RELIABILITY = {
    "arrow": 0.98,
    "gesture-target-arrow": 0.96,
    "rocket": 0.96,
    "pink-bow": 0.97,
    "pig-body": 0.96,
    "goat-body": 0.97,
    "learned-template": 0.84,
    "body": 0.78,
    "cattle-body": 0.88,
    "cattle-face": 0.74,
    "cattle-cell": 0.52,
    "unknown": 0.45,
}


def _detector(candidate: dict) -> str:
    explicit = candidate.get("detector")
    if explicit:
        return str(explicit)
    votes = candidate.get("direction_votes") or {}
    if "arrow" in votes:
        return "arrow"
    if "cattle_body" in votes:
        return "cattle-body"
    if "cattle_face" in votes:
        return "cattle-face"
    if "cattle_cell_stats" in votes:
        return "cattle-cell"
    return "body"


def _candidate_confidence(candidate: dict) -> dict:
    detector = _detector(candidate)
    reliability = DETECTOR_RELIABILITY.get(detector, DETECTOR_RELIABILITY["unknown"])
    pair_score = max(0.0, float(candidate.get("pair_score") or 0.0))
    direction_score = max(0.0, float(candidate.get("direction_confidence") or 0.0))
    occupancy_scale = 1600.0 if detector.startswith("cattle") else 1200.0
    direction_scale = 150.0 if detector.startswith("cattle") else 90.0
    occupancy = _clip((1.0 - math.exp(-pair_score / occupancy_scale)) * reliability)
    facing = _clip((1.0 - math.exp(-direction_score / direction_scale)) * reliability)
    # Axis is independently supported by the pair geometry and detector shape.
    axis = _clip(0.55 + 0.42 * reliability)
    species = _clip(
        0.98 if detector in {"rocket", "pink-bow", "pig-body", "goat-body"}
        else 0.88 if detector == "learned-template"
        else 0.96 if detector.startswith("cattle")
        else 0.92
    )
    score = 38.0 * occupancy + 22.0 * axis + 20.0 * facing + 12.0 * species + 8.0 * reliability
    return {
        "occupancy": round(occupancy, 4),
        "axis": round(axis, 4),
        "facing": round(facing, 4),
        "species": round(species, 4),
        "detector": round(reliability, 4),
        "selection_score": round(score, 4),
    }


def fuse_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fuse same-placement candidates while retaining every source score."""
    groups = defaultdict(list)
    for raw in candidates:
        candidate = deepcopy(raw)
        candidate["detector"] = _detector(candidate)
        candidate["confidence"] = _candidate_confidence(candidate)
        key = cell_key(candidate)
        if len(key) != 2:
            candidate["drop_reason"] = "invalid_cell_count"
            groups[(key, "__invalid__")].append(candidate)
            continue
        groups[(key, str(candidate.get("species") or "sheep"))].append(candidate)

    fused, rejected = [], []
    for (key, species), sources in groups.items():
        if species == "__invalid__":
            rejected.extend(sources)
            continue
        facing_votes = defaultdict(float)
        axis_votes = defaultdict(float)
        detector_names = set()
        for source in sources:
            conf = source["confidence"]
            vote_multiplier = 3.2 if source["detector"] == "gesture-target-arrow" else (
                3.0 if source["detector"] == "arrow" else (
                2.8 if source["detector"] in {"rocket", "pink-bow", "pig-body", "goat-body"} else (
                2.35 if source["detector"] == "learned-template" else (
                2.0 if source["detector"] == "cattle-body" else 1.0))))
            weight = (0.35 + conf["occupancy"] + conf["facing"]) * vote_multiplier
            facing_votes[str(source.get("facing"))] += weight
            axis_votes[str(source.get("axis"))] += 0.4 + conf["axis"]
            detector_names.add(source["detector"])
        facing = max(facing_votes, key=facing_votes.get)
        axis = max(axis_votes, key=axis_votes.get)
        agreeing = [source for source in sources if str(source.get("facing")) == facing]
        representative = max(agreeing or sources,
                             key=lambda item: item["confidence"]["selection_score"])

        occupancy = 1.0
        for source in sources:
            occupancy *= 1.0 - source["confidence"]["occupancy"]
        occupancy = 1.0 - occupancy
        vote_total = sum(facing_votes.values()) or 1.0
        facing_conf = facing_votes[facing] / vote_total
        axis_total = sum(axis_votes.values()) or 1.0
        axis_conf = axis_votes[axis] / axis_total
        diversity = min(1.0, len(detector_names) / 3.0)
        species_conf = max(source["confidence"]["species"] for source in sources)
        anchor_bonus = 105.0 if "gesture-target-arrow" in detector_names else (
            100.0 if "arrow" in detector_names else (
            95.0 if ({"rocket", "pink-bow", "pig-body", "goat-body"} & detector_names) else (
            72.0 if "learned-template" in detector_names else (
            55.0 if "cattle-body" in detector_names else
            30.0 if "cattle-face" in detector_names else 0.0))))
        selection_score = (48.0 * occupancy + 18.0 * axis_conf + 18.0 * facing_conf
                           + 10.0 * species_conf + 8.0 * diversity
                           + min(8.0, 2.0 * (len(sources) - 1)) + anchor_bonus)

        result = deepcopy(representative)
        result.update({
            "source_id": "fusion:" + "+".join(str(item.get("source_id")) for item in sources),
            "detector": "fusion",
            "detectors": sorted(detector_names),
            "species": species,
            "axis": axis,
            "facing": facing,
            "quality": round(selection_score * 100.0, 2),
            "selection_score": round(selection_score, 4),
            "confidence": {
                "occupancy": round(occupancy, 4),
                "axis": round(axis_conf, 4),
                "facing": round(facing_conf, 4),
                "species": round(species_conf, 4),
                "detector_diversity": round(diversity, 4),
            },
            "fusion": {
                "source_count": len(sources),
                "sources": [{
                    "source_id": item.get("source_id"),
                    "detector": item["detector"],
                    "facing": item.get("facing"),
                    "axis": item.get("axis"),
                    "confidence": item["confidence"],
                } for item in sources],
                "facing_votes": {key: round(value, 4) for key, value in facing_votes.items()},
                "axis_votes": {key: round(value, 4) for key, value in axis_votes.items()},
            },
        })
        # Recompute ordered rump/head from the fused direction.
        drdc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        dr, dc = drdc[facing]
        head = max(key, key=lambda rc: rc[0] * dr + rc[1] * dc)
        rump = min(key, key=lambda rc: rc[0] * dr + rc[1] * dc)
        result["cells"] = [list(rump), list(head)]
        result["rump"], result["head"] = list(rump), list(head)

        review_reasons = []
        if occupancy < (0.48 if detector_names == {"cattle-cell"} else 0.52):
            review_reasons.append("low_occupancy_confidence")
        if len(facing_votes) > 1 and facing_conf < 0.62:
            review_reasons.append("detector_facing_disagreement")
        if len(axis_votes) > 1 and axis_conf < 0.70:
            review_reasons.append("detector_axis_disagreement")
        if (detector_names == {"cattle-cell"}
                and float(representative.get("direction_confidence") or 0.0) < 180):
            review_reasons.append("weak_cattle_cell_only")
        if review_reasons:
            result["review"] = True
            result["review_reason"] = review_reasons[0]
            result["review_reasons"] = review_reasons
        fused.append(result)
    return fused, rejected


def global_assignment(candidates: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Solve weighted adjacent-cell set packing as a binary MILP."""
    if not candidates:
        return [], [], {"method": "milp", "status": "empty", "objective": 0.0}
    cells = sorted({cell for candidate in candidates for cell in cell_key(candidate)})
    cell_index = {cell: index for index, cell in enumerate(cells)}
    matrix = lil_matrix((len(cells), len(candidates)), dtype=float)
    for col, candidate in enumerate(candidates):
        for cell in cell_key(candidate):
            matrix[cell_index[cell], col] = 1.0
    scores = np.asarray([
        max(0.001, float(candidate.get("selection_score") or
                         (candidate.get("confidence") or {}).get("selection_score") or
                         candidate.get("quality") or 0.001))
        for candidate in candidates
    ], dtype=float)
    try:
        result = milp(
            c=-scores,
            integrality=np.ones(len(candidates), dtype=int),
            bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
            constraints=LinearConstraint(matrix.tocsr(),
                                         np.zeros(len(cells)), np.ones(len(cells))),
            options={"time_limit": 3.0, "mip_rel_gap": 0.0},
        )
        if not result.success or result.x is None:
            raise RuntimeError(str(result.message))
        selected_indices = {i for i, value in enumerate(result.x) if value >= 0.5}
        method, status = "milp", str(result.message)
    except Exception as exc:
        selected_indices, occupied = set(), set()
        for index in sorted(range(len(candidates)), key=lambda i: -scores[i]):
            placement = set(cell_key(candidates[index]))
            if not placement & occupied:
                selected_indices.add(index)
                occupied |= placement
        method, status = "greedy-fallback", str(exc)

    kept = [deepcopy(candidates[i]) for i in sorted(selected_indices)]
    chosen_cells = {cell: candidate for candidate in kept for cell in cell_key(candidate)}
    dropped = []
    for index, candidate in enumerate(candidates):
        if index in selected_indices:
            continue
        conflicts = sorted({str(chosen_cells[cell].get("source_id"))
                            for cell in cell_key(candidate) if cell in chosen_cells})
        item = deepcopy(candidate)
        item["drop_reason"] = "global_occupancy_conflict"
        item["conflicts_with"] = conflicts
        item["optimization_loss"] = round(float(candidate.get("selection_score") or 0.0), 4)
        dropped.append(item)
    kept.sort(key=lambda item: (min(cell[0] for cell in cell_key(item)),
                                min(cell[1] for cell in cell_key(item))))
    return kept, dropped, {
        "method": method,
        "status": status,
        "candidate_count": len(candidates),
        "selected_count": len(kept),
        "objective": round(sum(float(item.get("selection_score") or 0.0) for item in kept), 4),
    }
