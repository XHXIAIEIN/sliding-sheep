
import os
import json
import time
from copy import deepcopy
import cv2
import numpy as np
from board import io as board_io
import vision as D
from levels import cache as level_cache
import recognition
from solver import DIRS, Move
from paths import image_path
from . import common
from .common import _write_json_atomic


class BoardStateOps:
    """Mixin: board cloning, serialization, and detection artifacts."""

    def _clone_board(self, board):
        return type(board)(
            rows=board.rows,
            cols=board.cols,
            model=board.model,
            slide_mode=board.slide_mode,
            hazards=[list(rc) for rc in getattr(board, "hazards", [])],
            no_stop=[list(rc) for rc in getattr(board, "no_stop", [])],
            fences=[{"cell": [r, c], "direction": direction}
                    for r, c, direction in getattr(board, "fences", [])],
            returning={pid: {"cells": [list(cell) for cell in piece["cells"]],
                             "facing": piece.get("facing"),
                             "species": piece.get("species", "black_sheep")}
                       for pid, piece in getattr(board, "returning", {}).items()},
            pieces={
                pid: {"cells": [list(rc) for rc in sorted(p["cells"])],
                      "facing": p.get("facing"),
                      "species": p.get("species", "sheep"),
                      **({"awake": bool(p.get("awake", True))}
                         if p.get("species") == "pig" else {}),
                      **({"hit_limit": p.get("hit_limit", 3),
                          "hits_remaining": p.get("hits_remaining", 3)}
                         if p.get("species") == "bomb" else {})}
                for pid, p in board.pieces.items()
            },
        )

    def _sync_species(self):
        self._species_by_id = {
            str(s.get("id")): {
                "species": s.get("species", "sheep"),
                "review": bool(s.get("review")),
                "review_reason": s.get("review_reason"),
                "hit_limit": s.get("hit_limit"),
                "hits_remaining": s.get("hits_remaining"),
                "awake": s.get("awake"),
            }
            for s in (self.sheep or [])
        }

    def _board_data(self, board):
        return {
            "rows": board.rows,
            "cols": board.cols,
            "model": board.model,
            "slide_mode": board.slide_mode,
            "hazards": [list(rc) for rc in sorted(getattr(board, "hazards", []))],
            "no_stop": [list(rc) for rc in sorted(getattr(board, "no_stop", []))],
            "fences": [{"cell": [r, c], "direction": direction}
                       for r, c, direction in sorted(getattr(board, "fences", []))],
            "returning": {pid: {"cells": [list(cell) for cell in sorted(piece["cells"])],
                                "facing": piece.get("facing"),
                                "species": piece.get("species", "black_sheep")}
                          for pid, piece in getattr(board, "returning", {}).items()},
            "pieces": {
                str(pid): {"cells": [list(rc) for rc in sorted(p["cells"])],
                           "facing": p.get("facing"),
                           "species": p.get("species", self._species_by_id.get(str(pid), {}).get("species", "sheep")),
                           **({"awake": bool(p.get("awake", True))}
                              if p.get("species") == "pig" else {}),
                           **({"hit_limit": p.get("hit_limit", 3),
                               "hits_remaining": p.get("hits_remaining", 3)}
                              if p.get("species") == "bomb" else {})}
                for pid, p in board.pieces.items()
            },
        }

    def _record_execution_step(self, board, move, *, mode, batch_id=None,
                               batch_index=None):
        self._opening_coarse_pending = False
        pid = str(move.piece_id)
        piece = board.pieces[pid]
        piece_data = {
            "id": pid,
            "cells": [list(cell) for cell in sorted(piece["cells"])],
            "facing": piece.get("facing"),
            "species": piece.get("species", "sheep"),
            **({"awake": bool(piece.get("awake", True))}
               if piece.get("species") == "pig" else {}),
            **({"hit_limit": piece.get("hit_limit", 3),
                "hits_remaining": piece.get("hits_remaining", 3)}
               if piece.get("species") == "bomb" else {}),
        }
        move_data = {
            "piece_id": pid,
            "direction": move.direction,
            "anchor": list(move.anchor),
            "result": move.result,
            "distance": move.distance,
            "description": board_io.describe(move),
        }
        return level_cache.save_execution_step(
            self._board_data(board), piece_data, move_data,
            level_key=self._level_key, capture_meta=self._last_cache,
            mode=mode, batch_id=batch_id, batch_index=batch_index)

    def _state_signature(self, state):
        if not state:
            return ""
        hazards = ";".join(f"{r},{c}" for r, c in sorted(tuple(x) for x in state.get("hazards", [])))
        fences = ";".join(
            f"{item.get('direction')}:{item.get('cell', [None, None])[0]},{item.get('cell', [None, None])[1]}"
            for item in sorted(state.get("fences", []), key=lambda x: (x.get("direction"), x.get("cell"))))
        items = []
        for piece in state.get("pieces", []):
            cells = sorted(tuple(cell) for cell in piece.get("cells", []))
            cell_key = ";".join(f"{r},{c}" for r, c in cells)
            items.append(
                f"{piece.get('species', 'sheep')}:{piece.get('facing') or '?'}:"
                f"{piece.get('awake', '')}:{piece.get('hits_remaining', '')}:{cell_key}")
        return f"R:{state.get('rows')}|C:{state.get('cols')}|H:{hazards}|F:{fences}|P:" + "|".join(sorted(items))

    def _state_diff(self, expected, actual):
        def piece_map(state):
            out = {}
            for piece in (state or {}).get("pieces", []):
                cells = sorted(tuple(cell) for cell in piece.get("cells", []))
                key = (f"{piece.get('species', 'sheep')}:{piece.get('facing') or '?'}:"
                       f"{piece.get('awake', '')}:{piece.get('hits_remaining', '')}:"
                       + ";".join(f"{r},{c}" for r, c in cells))
                out[key] = piece
            return out

        def occ(state):
            cells = set()
            for piece in (state or {}).get("pieces", []):
                cells.update(tuple(cell) for cell in piece.get("cells", []))
            cells.update(tuple(cell) for cell in (state or {}).get("hazards", []))
            return cells

        exp_pieces = piece_map(expected)
        act_pieces = piece_map(actual)
        exp_occ = occ(expected)
        act_occ = occ(actual)
        missing_cells = sorted([list(cell) for cell in exp_occ - act_occ])
        extra_cells = sorted([list(cell) for cell in act_occ - exp_occ])
        suspect_cells = sorted({tuple(cell) for cell in missing_cells + extra_cells})
        return {
            "missing_pieces": sorted(set(exp_pieces) - set(act_pieces)),
            "extra_pieces": sorted(set(act_pieces) - set(exp_pieces)),
            "expected_hazards": sorted([list(cell) for cell in (expected or {}).get("hazards", [])]),
            "actual_hazards": sorted([list(cell) for cell in (actual or {}).get("hazards", [])]),
            "expected_fences": list((expected or {}).get("fences", [])),
            "actual_fences": list((actual or {}).get("fences", [])),
            "missing_cells": missing_cells,
            "extra_cells": extra_cells,
            "suspect_cells": [list(cell) for cell in suspect_cells],
            "expected_count": len(exp_pieces),
            "actual_count": len(act_pieces),
        }

    def _verification_feedback(self, expected_state, actual_state, planned_steps=1):
        if not expected_state:
            return None
        exp_sig = self._state_signature(expected_state)
        act_sig = self._state_signature(actual_state)
        matched = exp_sig == act_sig
        diff = self._state_diff(expected_state, actual_state)
        feedback = {
            "kind": "post-click",
            "planned_steps": int(planned_steps or 1),
            "matched": matched,
            "expected_signature": exp_sig,
            "actual_signature": act_sig,
            "diff": diff,
        }
        if matched:
            feedback["mismatch_type"] = None
        elif diff["expected_hazards"] != diff["actual_hazards"]:
            feedback["mismatch_type"] = "hazard_mismatch"
        elif diff["expected_count"] != diff["actual_count"]:
            feedback["mismatch_type"] = "piece_count_mismatch"
        elif diff["missing_cells"] or diff["extra_cells"]:
            feedback["mismatch_type"] = "occupancy_mismatch"
        elif diff["missing_pieces"] or diff["extra_pieces"]:
            feedback["mismatch_type"] = "facing_or_species_mismatch"
        else:
            feedback["mismatch_type"] = "unknown"
        return feedback

    def _record_direction_correction(self, pid, cells, original_facing, corrected_facing,
                                     *, source):
        """Persist visual evidence for a manual facing correction."""
        if not original_facing or str(original_facing) == str(corrected_facing):
            return None
        placement = {tuple(cell) for cell in cells}
        pools = [self._detected_sheep_data or [], self.sheep or []]
        evidence = None
        for pool in pools:
            matches = [deepcopy(item) for item in pool
                       if {tuple(cell) for cell in item.get("cells", [])} == placement]
            # Detector ids are spatially reassigned after every capture.  The
            # footprint is the durable identity; prefer evidence with actual
            # endpoint metrics instead of an id-equal manual shell.
            evidence = max(matches, key=lambda item: (
                bool(item.get("metrics")), bool(item.get("direction_votes")),
                str(item.get("id")) == str(pid), float(item.get("quality") or 0.0)
            ), default=None)
            if evidence is not None:
                break
        if evidence is None:
            return None
        evidence["facing"] = str(original_facing)
        sample_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"
        learned = recognition.record_direction_correction(
            evidence, str(corrected_facing), source=source,
            sample_id=sample_id, artifact="piece.png")
        if not learned:
            return None

        rect = (self.debug or {}).get("rect")
        if isinstance(rect, np.ndarray) and rect.size:
            rows = [cell[0] for cell in placement]
            cols = [cell[1] for cell in placement]
            pad = 14
            y0 = max(0, min(rows) * D.CELL - pad)
            y1 = min(rect.shape[0], (max(rows) + 1) * D.CELL + pad)
            x0 = max(0, min(cols) * D.CELL - pad)
            x1 = min(rect.shape[1], (max(cols) + 1) * D.CELL + pad)
            folder = recognition.DIRECTION_LEARNING_DIR / "samples" / sample_id
            cv2.imwrite(str(folder / "piece.png"), rect[y0:y1, x0:x1])
        learned["recorded"] = True
        learned["sample_path"] = str(
            recognition.DIRECTION_LEARNING_DIR / "samples" / sample_id)
        # Remove the stale temporal majority that would otherwise immediately
        # vote the just-confirmed direction back to its old value.
        for frame in self._frame_history:
            for item in frame.get("pieces") or []:
                if {tuple(cell) for cell in item.get("cells", [])} == placement:
                    item["facing"] = str(corrected_facing)
        return learned

    def _patch_sheep_direction(self, pid, cells, facing, axis):
        if not self.sheep:
            return
        dr, dc = DIRS[facing]
        head = max(cells, key=lambda rc: rc[0] * dr + rc[1] * dc)
        rump = min(cells, key=lambda rc: rc[0] * dr + rc[1] * dc)
        for s in self.sheep:
            if str(s.get("id")) == str(pid):
                s["axis"] = axis
                s["cells"] = ([list(cell) for cell in cells]
                              if len(cells) > 2 else [list(rump), list(head)])
                s["rump"] = list(rump)
                s["head"] = list(head)
                s["facing"] = facing
                s["manual"] = True
                return

    def _write_current_board(self, *, include_detection=True):
        bd = {
            "rows": self.board.rows,
            "cols": self.board.cols,
            "model": self.board.model,
            "slide_mode": self.board.slide_mode,
            "hazards": [list(rc) for rc in sorted(getattr(self.board, "hazards", []))],
            "fences": [{"cell": [r, c], "direction": direction}
                       for r, c, direction in sorted(getattr(self.board, "fences", []))],
            "returning": {pid: {"cells": [list(cell) for cell in sorted(piece["cells"])],
                                "facing": piece.get("facing"),
                                "species": piece.get("species", "black_sheep")}
                          for pid, piece in getattr(self.board, "returning", {}).items()},
            "pieces": {
                str(pid): {"cells": [list(rc) for rc in sorted(p["cells"])],
                           "facing": p.get("facing"),
                           "species": p.get("species", "sheep"),
                           **({"awake": bool(p.get("awake", True))}
                              if p.get("species") == "pig" else {}),
                           **({"hit_limit": p.get("hit_limit", 3),
                               "hits_remaining": p.get("hits_remaining", 3)}
                              if p.get("species") == "bomb" else {})}
                for pid, p in self.board.pieces.items()
            },
        }
        layout = None
        candidates = None
        if include_detection and self.sheep is not None:
            debug = self.debug or {}
            layout = D.to_layout(self.sheep, self.board.rows, self.board.cols,
                                 debug.get("dropped", []), hazards=debug.get("hazards"),
                                 fences=[{"cell": [r, c], "direction": direction}
                                         for r, c, direction in getattr(self.board, "fences", [])])
            candidates = {"kept": self.sheep,
                          "hazards": debug.get("hazards", []),
                          "fences": [{"cell": [r, c], "direction": direction}
                                     for r, c, direction in getattr(self.board, "fences", [])],
                          "wolf": debug.get("wolf_meta"),
                          "black_sheep_cluster": debug.get("black_sheep_cluster", []),
                          "black_sheep_applied": debug.get("black_sheep_applied", []),
                          "pink_sheep": debug.get("pink_sheep", []),
                          "pigs": debug.get("pigs", []),
                          "goats": debug.get("goats", []),
                          "goat_wolf_environment": debug.get("goat_wolf_environment", []),
                          "fusion": {"detector": debug.get("detector"),
                                     "raw_candidate_count": debug.get("raw_candidate_count"),
                                     "fused_candidate_count": debug.get("candidate_count"),
                                     "optimization": debug.get("optimization")},
                          "temporal": debug.get("temporal"),
                          "dropped": debug.get("dropped", []),
                          "raw": debug.get("raw_candidates", debug.get("candidates", [])),
                          "fused": debug.get("candidates", [])}

        # Do not truncate any existing file until every payload can serialize.
        json.dumps(bd, ensure_ascii=False)
        if layout is not None:
            json.dumps(layout, ensure_ascii=False)
            json.dumps(candidates, ensure_ascii=False)
        _write_json_atomic(common.data_path("board.json"), bd)
        if layout is not None:
            _write_json_atomic(common.data_path("board_layout.json"), layout)
            _write_json_atomic(common.data_path("sheep_candidates.json"), candidates)

    def _rerender_detection_images(self):
        if not self.debug or self.sheep is None:
            return
        cv2.imwrite(str(image_path("_occ_axis_rect.png")), D.render_rect_debug(self.debug, self.sheep))
        cv2.imwrite(str(image_path("_grid_labels.png")), D.render_grid_labels(self.debug, self.sheep))
        cv2.imwrite(str(image_path("_layout.png")), D.render_layout(self.debug, self.sheep))

    def _snapshot(self, board, highlight, Minv=None, live_wolf_annotations=True):
        """把某一时刻的 Board 转成前端可画的形状列表。"""
        pieces = []
        for pid, p in board.pieces.items():
            cells = sorted(p["cells"])
            axis = "V" if len({c for _, c in cells}) == 1 else "H"
            polys = [self._cell_poly(r, c, Minv=Minv) for r, c in cells]
            arrow = None
            facing = p.get("facing")
            if facing and len(cells) >= 2:
                dr, dc = DIRS[facing]
                projections = {cell: cell[0] * dr + cell[1] * dc for cell in cells}
                head_cells = [cell for cell in cells
                              if projections[cell] == max(projections.values())]
                rump_cells = [cell for cell in cells
                              if projections[cell] == min(projections.values())]

                def edge_center(group):
                    points = [self._cell_center(*cell, Minv=Minv) for cell in group]
                    return [sum(point[0] for point in points) / len(points),
                            sum(point[1] for point in points) / len(points)]

                arrow = [edge_center(rump_cells), edge_center(head_cells)]
            cx = sum(self._cell_center(r, c, Minv=Minv)[0] for r, c in cells) / len(cells)
            cy = sum(self._cell_center(r, c, Minv=Minv)[1] for r, c in cells) / len(cells)
            meta = self._species_by_id.get(str(pid), {"species": "sheep"})
            pieces.append({"id": pid, "axis": axis, "facing": facing,
                           "species": meta.get("species", "sheep"),
                           "awake": (p.get("awake", meta.get("awake", True))
                                     if meta.get("species") == "pig" else None),
                           "hit_limit": p.get("hit_limit"),
                           "hits_remaining": p.get("hits_remaining"),
                           "review": bool(meta.get("review")),
                           "review_reason": meta.get("review_reason"),
                           "cells": [list(rc) for rc in cells],
                           "polys": polys, "arrow": arrow, "center": [cx, cy]})
        hazards = [list(rc) for rc in sorted(getattr(board, "hazards", []))]
        no_stop = [list(rc) for rc in sorted(getattr(board, "no_stop", []))]
        hazard_polys = [{"cell": list(rc), "poly": self._cell_poly(*rc, Minv=Minv)}
                        for rc in sorted(getattr(board, "hazards", []))]
        dynamic_hazards = []
        dynamic_hazard_polys = []
        if live_wolf_annotations:
            for item in (self.debug or {}).get("hazards") or []:
                if not isinstance(item, dict) or item.get("kind") != "wolf_body":
                    continue
                cell = (int(item["row"]), int(item["col"]))
                dynamic_hazards.append(list(cell))
                dynamic_hazard_polys.append({"cell": list(cell),
                                             "poly": self._cell_poly(*cell, Minv=Minv)})
        fences = []
        for r, c, direction in sorted(getattr(board, "fences", [])):
            if direction == "L":
                segment = [self._px(c, r, Minv=Minv), self._px(c, r + 1, Minv=Minv)]
            elif direction == "R":
                segment = [self._px(c + 1, r, Minv=Minv), self._px(c + 1, r + 1, Minv=Minv)]
            elif direction == "U":
                segment = [self._px(c, r, Minv=Minv), self._px(c + 1, r, Minv=Minv)]
            elif direction == "D":
                segment = [self._px(c, r + 1, Minv=Minv), self._px(c + 1, r + 1, Minv=Minv)]
            elif direction == "H":
                segment = [self._px(c, r + .5, Minv=Minv),
                           self._px(c + 1, r + .5, Minv=Minv)]
            else:  # V: internal fence centered in its occupied cell
                segment = [self._px(c + .5, r, Minv=Minv),
                           self._px(c + .5, r + 1, Minv=Minv)]
            fences.append({"cell": [r, c], "direction": direction, "segment": segment})
        return {"rows": board.rows, "cols": board.cols, "pieces": pieces,
                "hazards": hazards, "no_stop": no_stop,
                "hazard_polys": hazard_polys,
                "dynamic_hazards": dynamic_hazards,
                "dynamic_hazard_polys": dynamic_hazard_polys,
                "fences": fences, "highlight": highlight}
