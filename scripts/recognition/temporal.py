"""Temporal stabilization of pieces and hazards across recent frames."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from .features import cell_key


def _piece_observation(piece: dict) -> dict:
    return {
        "cells": [list(cell) for cell in cell_key(piece)],
        "axis": piece.get("axis"),
        "facing": piece.get("facing"),
        "species": piece.get("species", "sheep"),
        "confidence": deepcopy(piece.get("confidence") or {}),
        "hit_limit": piece.get("hit_limit"),
        "hits_remaining": piece.get("hits_remaining"),
        "awake": piece.get("awake"),
    }


def observation_record(pieces: list[dict], hazards: list[dict], rows: int, cols: int) -> dict:
    return {
        "rows": int(rows), "cols": int(cols),
        "pieces": [_piece_observation(piece) for piece in pieces],
        "hazards": [[int(item["row"]), int(item["col"])] if isinstance(item, dict)
                    else [int(item[0]), int(item[1])] for item in (hazards or [])],
    }


def apply_temporal(pieces: list[dict], hazards: list[dict], history: list[dict],
                   rows: int, cols: int, *, recover_missing_edges=False
                   ) -> tuple[list[dict], list[dict], dict]:
    """Stabilize current evidence using up to four compatible prior frames."""
    compatible = [frame for frame in (history or [])
                  if int(frame.get("rows", -1)) == int(rows)
                  and int(frame.get("cols", -1)) == int(cols)][-4:]
    result = deepcopy(pieces)
    previous_by_cells = defaultdict(list)
    for frame in compatible:
        for piece in frame.get("pieces") or []:
            previous_by_cells[cell_key(piece)].append(piece)

    corrections = []
    for piece in result:
        prior = previous_by_cells.get(cell_key(piece), [])
        samples = prior + [piece]
        facing_votes = Counter(str(item.get("facing")) for item in samples if item.get("facing"))
        species_votes = Counter(str(item.get("species") or "sheep") for item in samples)
        consensus = max(facing_votes.values(), default=1) / max(1, sum(facing_votes.values()))
        current_conf = piece.setdefault("confidence", {})
        current_conf["temporal_presence"] = round((len(prior) + 1) / (len(compatible) + 1), 4)
        current_conf["temporal_facing"] = round(consensus, 4)
        if len(prior) >= 2 and facing_votes:
            stable_facing, votes = facing_votes.most_common(1)[0]
            if stable_facing != piece.get("facing") and votes >= 3 and consensus >= 0.75:
                old = piece.get("facing")
                piece["facing"] = stable_facing
                dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[stable_facing]
                placement = cell_key(piece)
                head = max(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
                rump = min(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
                piece["cells"], piece["rump"], piece["head"] = [list(rump), list(head)], list(rump), list(head)
                corrections.append({"cells": [list(c) for c in placement], "field": "facing",
                                    "from": old, "to": stable_facing, "votes": dict(facing_votes)})
        if len(prior) >= 2 and species_votes:
            stable_species, votes = species_votes.most_common(1)[0]
            # Black sheep is a visual state, not a persistent identity.  A
            # stale frame can contain a dark hazard/pack false positive; do
            # not let that historical majority relabel a currently ordinary
            # sheep.  The current-frame black-pack / dark-cluster classifiers
            # run after temporal stabilization and remain the authority for a
            # genuine black sheep.
            current_black_evidence = (
                piece.get("species") == "black_sheep"
                or "black-pack" in set(piece.get("detectors") or [])
                or bool((piece.get("direction_votes") or {}).get(
                    "black_sheep_dark_cluster"))
            )
            if (stable_species != piece.get("species") and votes >= 3
                    and (stable_species != "black_sheep" or current_black_evidence)):
                corrections.append({"cells": [list(c) for c in cell_key(piece)], "field": "species",
                                    "from": piece.get("species"), "to": stable_species,
                                    "votes": dict(species_votes)})
                piece["species"] = stable_species

    restored = []
    if recover_missing_edges and compatible:
        occupied = {cell for piece in result for cell in cell_key(piece)}
        next_id = max([int(piece.get("id", -1)) for piece in result] + [-1]) + 1
        for prior_piece in compatible[-1].get("pieces") or []:
            placement = cell_key(prior_piece)
            if len(placement) != 2 or any(cell in occupied for cell in placement):
                continue
            if not any(r in (0, rows - 1) or c in (0, cols - 1) for r, c in placement):
                continue
            confidence = deepcopy(prior_piece.get("confidence") or {})
            if float(confidence.get("occupancy", 0.0)) < 0.70:
                continue
            facing = str(prior_piece.get("facing") or "")
            if facing not in {"U", "D", "L", "R"}:
                continue
            dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[facing]
            head = max(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
            rump = min(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
            axis = "H" if rump[0] == head[0] else "V"
            confidence.update({"temporal_presence": 0.5, "temporal_facing": 1.0})
            recovered = {
                "id": next_id,
                "cells": [list(rump), list(head)],
                "rump": list(rump), "head": list(head),
                "axis": axis, "facing": facing,
                "species": prior_piece.get("species", "sheep"),
                "hit_limit": prior_piece.get("hit_limit"),
                "hits_remaining": prior_piece.get("hits_remaining"),
                "awake": prior_piece.get("awake"),
                "confidence": confidence,
                "source_id": "temporal-edge",
                "quality": 1.0,
                "temporal_restored": True,
            }
            result.append(recovered)
            occupied.update(placement)
            restored.append({"cells": [list(cell) for cell in placement],
                             "facing": facing, "species": recovered["species"]})
            next_id += 1

    current_hazards = {(int(item["row"]), int(item["col"])) if isinstance(item, dict)
                       else (int(item[0]), int(item[1])) for item in (hazards or [])}
    hazard_counts = Counter(current_hazards)
    for frame in compatible:
        hazard_counts.update(tuple(cell) for cell in (frame.get("hazards") or []))
    horizon = len(compatible) + 1
    timeline = []
    uncertain = []
    for cell, count in sorted(hazard_counts.items()):
        current = cell in current_hazards
        ratio = count / horizon
        if current and count >= 2:
            state = "stable"
        elif current:
            state = "emerging"
            uncertain.append(list(cell))
        elif count >= 2:
            state = "fading"
            uncertain.append(list(cell))
        else:
            state = "transient"
        timeline.append({"cell": list(cell), "state": state, "present": current,
                         "observations": int(count), "frames": horizon,
                         "confidence": round(ratio, 4)})
    hazard_lookup = {tuple(item["cell"]): item for item in timeline}
    hazards_out = []
    for item in hazards or []:
        cell = (int(item["row"]), int(item["col"])) if isinstance(item, dict) else tuple(item)
        base = deepcopy(item) if isinstance(item, dict) else {"row": cell[0], "col": cell[1]}
        temporal = hazard_lookup.get(cell, {})
        base["temporal_state"] = temporal.get("state", "emerging")
        base["confidence"] = temporal.get("confidence", 1.0)
        hazards_out.append(base)
    return result, hazards_out, {
        "frames": horizon,
        "history_frames": len(compatible),
        "corrections": corrections,
        "restored_edge_pieces": restored,
        "hazards": timeline,
        "uncertain_hazard_cells": uncertain,
    }
