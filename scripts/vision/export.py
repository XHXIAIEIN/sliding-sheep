"""Convert detected pieces into solver board and layout JSON."""
from __future__ import annotations

import json
from .masks import CELL


def _hazard_cells(hazards):
    return [[int(h["row"]), int(h["col"])] if isinstance(h, dict) else list(h) for h in (hazards or [])]


def to_board(sheep, rows, cols, model="facing", slide_mode="all", hazards=None, fences=None):
    # Wolves move continuously.  Their observed body cells belong to the live
    # execution guard and sandbox annotation, never to the solver's permanent
    # obstacle set.  Untagged/manual hazards remain real board obstacles.
    static_hazards = [
        item for item in (hazards or [])
        if not (isinstance(item, dict) and item.get("kind") == "wolf_body")
    ]
    return {
        "rows": rows,
        "cols": cols,
        "model": model,
        "slide_mode": slide_mode,
        "hazards": _hazard_cells(static_hazards),
        "fences": [{"cell": list(item["cell"]), "direction": item["direction"]}
                   for item in (fences or [])],
        "pieces": {
            str(i): {"cells": [list(c) for c in s["cells"]],
                     "facing": s["facing"],
                     "species": s.get("species", "sheep"),
                     **({"awake": bool(s.get("awake"))}
                        if s.get("species") == "pig" else {}),
                     **({"hit_limit": s.get("hit_limit", 3),
                         "hits_remaining": s.get("hits_remaining", 3)}
                        if s.get("species") == "bomb" else {}),
                     **({"review": True, "review_reason": s.get("review_reason")}
                        if s.get("review") else {})}
            for i, s in enumerate(sheep)
        },
    }


def to_layout(sheep, rows, cols, dropped=None, hazards=None, fences=None):
    hazard_cells = {tuple(x) for x in _hazard_cells(hazards)}
    occupied: dict[tuple[int, int], list[dict]] = {}
    pieces = []
    for s in sheep:
        pid = int(s["id"])
        cells = [tuple(rc) for rc in s["cells"]]
        piece = {
            "id": pid,
            "species": s.get("species", "sheep"),
            "awake": (bool(s.get("awake")) if s.get("species") == "pig" else None),
            "hit_limit": s.get("hit_limit"),
            "hits_remaining": s.get("hits_remaining"),
            "cells": [list(rc) for rc in cells],
            "rump": list(s["rump"]),
            "head": list(s["head"]),
            "axis": s["axis"],
            "facing": s["facing"],
            "source_id": s.get("source_id", f"manual:{pid}"),
            "quality": s.get("quality", 1.0 if s.get("manual") else 0.0),
            "direction_confidence": s.get("direction_confidence"),
            "confidence": s.get("confidence"),
        }
        pieces.append(piece)
        for rc in cells:
            role = "head" if list(rc) == s["head"] else "rump"
            occupied.setdefault(rc, []).append({"piece_id": pid, "role": role})

    dropped_by_cell: dict[tuple[int, int], list] = {}
    for cand in dropped or []:
        sid = cand.get("source_id", -1)
        for rc in cand.get("cells", []):
            dropped_by_cell.setdefault(tuple(rc), []).append(sid)

    cells = []
    empty = []
    conflicts = []
    for r in range(rows):
        row = []
        for c in range(cols):
            occ = occupied.get((r, c), [])
            dropped_ids = sorted(set(dropped_by_cell.get((r, c), [])), key=str)
            cell = {
                "row": r,
                "col": c,
                "label": f"{chr(ord('A') + c)}{r + 1}" if c < 26 else f"C{c + 1}R{r + 1}",
                "occupied": bool(occ),
                "hazard": (r, c) in hazard_cells,
                "piece_ids": [x["piece_id"] for x in occ],
                "roles": {str(x["piece_id"]): x["role"] for x in occ},
                "dropped_source_ids": dropped_ids,
            }
            if not occ:
                empty.append([r, c])
            if len(occ) > 1:
                conflicts.append({"cell": [r, c], "piece_ids": [x["piece_id"] for x in occ]})
            row.append(cell)
        cells.append(row)

    vertical = sum(1 for s in sheep if s["axis"] == "V")
    facing_counts = {d: sum(1 for s in sheep if s["facing"] == d) for d in "UDLR"}
    return {
        "rows": rows,
        "cols": cols,
        "fences": [{"cell": list(item["cell"]), "direction": item["direction"]}
                   for item in (fences or [])],
        "cell": CELL,
        "piece_count": len(pieces),
        "occupied_count": sum(len(s["cells"]) for s in sheep),
        "empty_count": len(empty),
        "conflicts": conflicts,
        "axis_counts": {"V": vertical, "H": len(pieces) - vertical},
        "facing_counts": facing_counts,
        "pieces": pieces,
        "cells": cells,
        "empty_cells": empty,
        "hazards": _hazard_cells(hazards),
    }


def _conflicts(sheep):
    occ = {}
    for i, s in enumerate(sheep):
        for rc in s["cells"]:
            occ.setdefault(tuple(rc), []).append(i)
    return occ, {k: v for k, v in occ.items() if len(v) > 1}


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
