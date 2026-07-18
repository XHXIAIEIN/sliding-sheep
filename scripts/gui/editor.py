
import os
import json
import hashlib
import time
from copy import deepcopy
import cv2
import numpy as np
import board_io
import vision as D
import level_cache
import recognition
import safety
from solver import DIRS, Move
from . import common
from .common import _safe_error, _wrap, _write_json_atomic


class EditorOps:
    """Mixin: manual board review, editing, and sample publication."""

    def set_facing(self, piece_id, facing):
        """Manually correct a sheep direction, then return a fresh sandbox state."""
        requested_facing = str(facing).upper()
        def run():
            if self.board is None:
                raise RuntimeError("请先识别")
            if self.runtime.snapshot().get("busy"):
                raise RuntimeError("求解尚未结束，请先暂停后再修改方向")
            pid = str(piece_id)
            if pid not in self.board.pieces:
                raise RuntimeError(f"找不到羊 {piece_id}")

            piece = self.board.pieces[pid]
            cells = sorted(piece["cells"])
            axis = "V" if len({c for _, c in cells}) == 1 else "H"
            target_facing = requested_facing
            if target_facing == "FLIP":
                target_facing = {"U": "D", "D": "U", "L": "R", "R": "L"}.get(piece.get("facing"), piece.get("facing"))
            allowed = {"V": {"U", "D"}, "H": {"L", "R"}}[axis]
            if target_facing not in allowed:
                raise RuntimeError(f"{axis} 向羊只能设为 {'/'.join(sorted(allowed))}")

            learning = self._record_direction_correction(
                pid, cells, piece.get("facing"), target_facing, source="direction-panel")
            piece["facing"] = target_facing
            self._patch_sheep_direction(pid, cells, target_facing, axis)
            self._write_current_board()
            self._rerender_detection_images()
            self.runtime.reset()
            return {"rows": self.board.rows, "cols": self.board.cols,
                    "count": self.board.remaining_count(),
                    "state": self._snapshot(self.board, highlight=pid),
                    "direction_learning": learning}
        return _wrap(run)

    def _editor_board_data(self):
        if self.board is None:
            raise RuntimeError("请先识别棋盘")
        return {
            "rows": int(self.board.rows), "cols": int(self.board.cols),
            "model": self.board.model, "slide_mode": self.board.slide_mode,
            "hazards": [list(cell) for cell in sorted(self.board.hazards)],
            "no_stop": [list(cell) for cell in sorted(getattr(self.board, "no_stop", []))],
            "fences": [{"cell": [r, c], "direction": direction}
                       for r, c, direction in sorted(getattr(self.board, "fences", []))],
            "returning": {pid: {"cells": [list(cell) for cell in sorted(piece["cells"])],
                                "facing": piece.get("facing"),
                                "species": piece.get("species", "black_sheep")}
                          for pid, piece in getattr(self.board, "returning", {}).items()},
            "pieces": {
                str(pid): {
                    "cells": [list(cell) for cell in sorted(piece["cells"])],
                    "facing": piece.get("facing"),
                    "species": piece.get("species", "sheep"),
                    **({"awake": bool(piece.get("awake", True))}
                       if piece.get("species") == "pig" else {}),
                    **({"hit_limit": piece.get("hit_limit", 3),
                        "hits_remaining": piece.get("hits_remaining", 3)}
                       if piece.get("species") == "bomb" else {}),
                } for pid, piece in self.board.pieces.items()
            },
        }

    @staticmethod
    def _editor_cells_for_facing(cells, old_facing, target_facing, rows, cols,
                                 occupied=()):
        """Rotate a two-cell footprint when its facing changes axis."""
        normalized = [tuple(int(value) for value in cell) for cell in cells]
        if len(normalized) != 2 or target_facing not in DIRS:
            return [list(cell) for cell in normalized]
        target_axis = "V" if target_facing in {"U", "D"} else "H"
        current_axis = ("V" if len({col for _row, col in normalized}) == 1
                        else "H" if len({row for row, _col in normalized}) == 1
                        else None)
        if current_axis == target_axis:
            return [list(cell) for cell in normalized]
        if old_facing not in DIRS:
            old_facing = "D" if current_axis == "V" else "R"
        old_dr, old_dc = DIRS[old_facing]
        new_dr, new_dc = DIRS[target_facing]
        ordered = sorted(normalized, key=lambda cell: cell[0] * old_dr + cell[1] * old_dc)
        rump, head = ordered[0], ordered[-1]
        candidates = [
            [rump, (rump[0] + new_dr, rump[1] + new_dc)],
            [(head[0] - new_dr, head[1] - new_dc), head],
            [head, (head[0] + new_dr, head[1] + new_dc)],
            [(rump[0] - new_dr, rump[1] - new_dc), rump],
        ]
        occupied = {tuple(cell) for cell in occupied}
        for candidate in candidates:
            if (len(set(candidate)) == 2
                    and all(0 <= row < int(rows) and 0 <= col < int(cols)
                            for row, col in candidate)
                    and not any(cell in occupied for cell in candidate)):
                return [list(cell) for cell in candidate]
        raise RuntimeError("旋转后的相邻格均被占用或超出棋盘，请先调整周围棋子")

    def _load_editor_board(self, data, *, pending=True, confirmed=False):
        board_io.validate_board_data(data)
        previous_runtime = {
            "board": self.board,
            "sheep": self.sheep,
            "species": deepcopy(self._species_by_id),
            "manual_pending": self._manual_edit_pending,
            "scene_report": deepcopy(self.scene_report),
            "board_revision": self.board_revision,
            "active_plan": self._active_plan,
            "debug_hazards": deepcopy(self.debug.get("hazards")) if self.debug is not None else None,
            "debug_fences": deepcopy(self.debug.get("fences")) if self.debug is not None else None,
        }
        self.board = board_io.Board(
            rows=data["rows"], cols=data["cols"], pieces=data["pieces"],
            model=data.get("model", "facing"), slide_mode=data.get("slide_mode", "all"),
            hazards=data.get("hazards", []),
            fences=data.get("fences", []),
            returning=data.get("returning", {}),
            no_stop=data.get("no_stop", []),
        )
        previous = {str(item.get("id")): item for item in (self.sheep or [])}
        rebuilt = []
        for pid, piece in self.board.pieces.items():
            cells = sorted(piece["cells"])
            facing = piece.get("facing")
            dr, dc = DIRS[facing]
            head = max(cells, key=lambda rc: rc[0] * dr + rc[1] * dc)
            rump = min(cells, key=lambda rc: rc[0] * dr + rc[1] * dc)
            old = previous.get(str(pid), {})
            item = {
                "id": pid, "cells": [list(cell) for cell in cells],
                "rump": list(rump), "head": list(head), "facing": facing,
                "axis": "V" if facing in {"U", "D"} else "H",
                "species": piece.get("species", "sheep"),
                **({"awake": bool(piece.get("awake", True))}
                   if piece.get("species") == "pig" else {}),
                "manual": True,
                "source_id": old.get("source_id", f"manual:{pid}"),
                "quality": old.get("quality", 1.0),
                "direction_confidence": old.get("direction_confidence", 1.0),
                **({"hit_limit": piece.get("hit_limit", 3),
                    "hits_remaining": piece.get("hits_remaining", 3)}
                   if piece.get("species") == "bomb" else {}),
                "confidence": old.get("confidence") or {
                    "occupancy": 1.0, "axis": 1.0, "facing": 1.0, "species": 1.0,
                },
            }
            rebuilt.append(item)
        self.sheep = rebuilt
        self._sync_species()
        if self.debug is not None:
            self.debug["hazards"] = [
                {"row": r, "col": c, "coverage": 1.0, "pixels": D.CELL * D.CELL,
                 "temporal_state": "manual", "confidence": 1.0}
                for r, c in sorted(self.board.hazards)
            ]
            self.debug["fences"] = [
                {"cell": [r, c], "direction": direction,
                 "temporal_state": "manual", "confidence": 1.0}
                for r, c, direction in sorted(self.board.fences)
            ]
        self._manual_edit_pending = bool(pending)
        resolved_blockers = {
            "manual_board_unconfirmed", "manual_review_required",
            "board_schema_invalid", "piece_overlap",
        }
        if confirmed:
            # "确认并使用" is the human authority for this exact board.  A
            # provisional single-sample recognition must remain blocked until
            # this point, but keeping that stale blocker after confirmation
            # makes continuous execution impossible even though every piece
            # has just been reviewed.
            resolved_blockers.add("manual_learning_confirmation_required")
        blockers = [item for item in self.scene_report.get("execution_blockers", [])
                    if item.get("code") not in resolved_blockers]
        advisories = [item for item in self.scene_report.get("advisories", [])
                      if item.get("code") != "manual_review_required"]
        warnings = [item for item in self.scene_report.get("warnings", [])
                    if item.get("code") != "manual_review_required"]
        if pending:
            blockers.append(safety.blocker(
                "manual_board_unconfirmed", "整盘复核尚未完成，只允许继续编辑和沙盘求解"))
        self.scene_report = {
            **self.scene_report,
            "execution_blockers": blockers,
            "advisories": advisories,
            "warnings": warnings,
            "executable": self.scene_report.get("scene_state") == "gameplay" and not blockers,
        }
        self.board_revision = level_cache.board_hash(data)
        self._active_plan = None
        try:
            self._write_current_board()
        except Exception:
            self.board = previous_runtime["board"]
            self.sheep = previous_runtime["sheep"]
            self._species_by_id = previous_runtime["species"]
            self._manual_edit_pending = previous_runtime["manual_pending"]
            self.scene_report = previous_runtime["scene_report"]
            self.board_revision = previous_runtime["board_revision"]
            self._active_plan = previous_runtime["active_plan"]
            if self.debug is not None:
                self.debug["hazards"] = previous_runtime["debug_hazards"]
                self.debug["fences"] = previous_runtime["debug_fences"]
            raise
        try:
            self._rerender_detection_images()
        except Exception as exc:
            # Diagnostic PNGs are best-effort and must not invalidate a board edit.
            _safe_error(exc)
        self.runtime.reset()
        return {
            "rows": self.board.rows, "cols": self.board.cols,
            "count": self.board.remaining_count(), "state": self._snapshot(self.board, highlight=None),
            "board_revision": self.board_revision,
            "execution_blockers": self.scene_report["execution_blockers"],
            "executable": self.scene_report["executable"],
            "scene_state": self.scene_report.get("scene_state", "unknown"),
            "scene_reason": self.scene_report.get("scene_reason", "手工棋盘"),
            "manual_pending": self._manual_edit_pending,
            "can_undo": bool(self._editor_undo),
            "can_redo": bool(self._editor_redo),
        }

    def edit_board(self, command):
        """Apply one validated manual board edit from the sandbox editor."""
        def run():
            if not isinstance(command, dict):
                raise RuntimeError("编辑命令必须是对象")
            if self.runtime.snapshot().get("busy"):
                raise RuntimeError("求解尚未结束，请先暂停后再编辑棋盘")
            action = str(command.get("action") or "")
            if action == "reset":
                if not self._detected_board_data:
                    raise RuntimeError("没有可还原的识别棋盘")
                result = self._load_editor_board(
                    json.loads(json.dumps(self._detected_board_data)), pending=False)
                self._editor_undo.clear()
                self._editor_redo.clear()
                result.update(can_undo=False, can_redo=False)
                return result
            if action in {"undo", "redo"}:
                source = self._editor_undo if action == "undo" else self._editor_redo
                target = self._editor_redo if action == "undo" else self._editor_undo
                if not source:
                    raise RuntimeError("没有可撤销的修改" if action == "undo" else "没有可重做的修改")
                current = self._editor_board_data()
                restored = source[-1]
                pending = not self._detected_board_data or (
                    level_cache.board_hash(restored) != level_cache.board_hash(self._detected_board_data))
                result = self._load_editor_board(restored, pending=pending)
                source.pop()
                target.append(json.loads(json.dumps(current)))
                result.update(can_undo=bool(self._editor_undo), can_redo=bool(self._editor_redo))
                return result
            data = self._editor_board_data()
            before = json.loads(json.dumps(data))
            pieces = data["pieces"]
            edit_detail = None
            if action in {"add_piece", "update_piece"}:
                cells = [[int(cell[0]), int(cell[1])] for cell in command.get("cells", [])]
                species = str(command.get("species") or "sheep")
                expected_cells = 6 if species == "elephant" else 2
                if len(cells) != expected_cells:
                    raise RuntimeError("大象必须占用 2×3 六格" if species == "elephant" else "棋子必须占用两个连续格")
                pid = str(command.get("piece_id")) if action == "update_piece" else None
                if action == "update_piece" and pid not in pieces:
                    raise RuntimeError(f"找不到棋子 {pid}")
                if pid is None:
                    numeric = [int(key) for key in pieces if str(key).isdigit()]
                    pid = str(max(numeric, default=-1) + 1)
                old_piece = deepcopy(pieces.get(pid)) if action == "update_piece" else None
                target_facing = str(command.get("facing") or "").upper()
                if target_facing not in DIRS:
                    raise RuntimeError("棋子朝向必须是上、下、左、右之一")
                if action == "update_piece" and len(cells) == 2:
                    occupied = {
                        tuple(cell)
                        for other_id, other_piece in pieces.items()
                        if str(other_id) != pid
                        for cell in other_piece.get("cells", [])
                    }
                    cells = self._editor_cells_for_facing(
                        cells, old_piece.get("facing"), target_facing,
                        data["rows"], data["cols"], occupied)
                pieces[pid] = {
                    "cells": cells,
                    "facing": target_facing,
                    "species": species,
                    **({"awake": bool(command.get("awake", old_piece.get("awake", False)
                                                   if old_piece else False))}
                       if species == "pig" else {}),
                    **({"hit_limit": max(1, int(command.get("hit_limit") or 3)),
                        "hits_remaining": max(1, int(command.get("hits_remaining") or
                                                     command.get("hit_limit") or 3))}
                       if species == "bomb" else {}),
                }
            elif action == "delete_piece":
                pid = str(command.get("piece_id"))
                if pid not in pieces:
                    raise RuntimeError(f"找不到棋子 {pid}")
                del pieces[pid]
            elif action in {"toggle_hazard", "add_hazard"}:
                cell = [int(value) for value in command.get("cell", [])]
                if len(cell) != 2:
                    raise RuntimeError("危险格坐标无效")
                hazards = {tuple(value) for value in data["hazards"]}
                target = tuple(cell)
                if target in hazards:
                    if action == "toggle_hazard":
                        hazards.remove(target)
                else:
                    occupied = {tuple(cell) for piece in pieces.values()
                                for cell in piece.get("cells", [])}
                    internal_fences = {tuple(item["cell"]) for item in data["fences"]
                                       if item.get("direction") in {"H", "V"}}
                    if target in occupied:
                        raise RuntimeError("已有棋子的格子不能设为狼危险格")
                    if target in internal_fences:
                        raise RuntimeError("已有内部栅栏的格子不能设为狼危险格")
                    hazards.add(target)
                data["hazards"] = [list(value) for value in sorted(hazards)]
            elif action in {"toggle_fence", "add_fence"}:
                cell = [int(value) for value in command.get("cell", [])]
                direction = str(command.get("direction") or "").upper()
                if len(cell) != 2 or direction not in board_io.VALID_FENCE_DIRECTION:
                    raise RuntimeError("栅栏坐标或方向无效")
                fences = {(tuple(item["cell"]), item["direction"])
                          for item in data["fences"]}
                target = (tuple(cell), direction)
                if target in fences:
                    if action == "toggle_fence":
                        fences.remove(target)
                else:
                    if direction in {"H", "V"}:
                        occupied = {tuple(value) for piece in pieces.values()
                                    for value in piece.get("cells", [])}
                        if tuple(cell) in occupied:
                            raise RuntimeError("已有棋子的格子不能放内部栅栏")
                        if tuple(cell) in {tuple(value) for value in data["hazards"]}:
                            raise RuntimeError("狼危险格不能同时放内部栅栏")
                    fences.add(target)
                data["fences"] = [
                    {"cell": list(fence_cell), "direction": fence_direction}
                    for fence_cell, fence_direction in sorted(fences)
                ]
            elif action == "clear_cell":
                cell = [int(value) for value in command.get("cell", [])]
                if len(cell) != 2:
                    raise RuntimeError("清除格坐标无效")
                row, col = cell
                if not (0 <= row < int(data["rows"]) and 0 <= col < int(data["cols"])):
                    raise RuntimeError("清除格超出棋盘")
                target = (row, col)
                removed_piece_ids = [
                    str(pid) for pid, piece in pieces.items()
                    if target in {tuple(value) for value in piece.get("cells", [])}
                ]
                for pid in removed_piece_ids:
                    del pieces[pid]
                old_hazards = {tuple(value) for value in data["hazards"]}
                removed_hazard = target in old_hazards
                old_hazards.discard(target)
                data["hazards"] = [list(value) for value in sorted(old_hazards)]
                removed_fences = [
                    item for item in data["fences"]
                    if tuple(item.get("cell") or ()) == target
                ]
                data["fences"] = [
                    item for item in data["fences"]
                    if tuple(item.get("cell") or ()) != target
                ]
                edit_detail = {
                    "cell": cell,
                    "removed_piece_ids": removed_piece_ids,
                    "removed_hazard": removed_hazard,
                    "removed_fence_directions": [item["direction"] for item in removed_fences],
                }
            else:
                raise RuntimeError(f"未知编辑动作: {action}")
            if data == before:
                return {
                    "rows": self.board.rows, "cols": self.board.cols,
                    "count": self.board.remaining_count(),
                    "state": self._snapshot(self.board, highlight=None),
                    "board_revision": self.board_revision,
                    "execution_blockers": self.scene_report.get("execution_blockers", []),
                    "executable": self.scene_report.get("executable", False),
                    "scene_state": self.scene_report.get("scene_state", "unknown"),
                    "scene_reason": self.scene_report.get("scene_reason", "手工棋盘"),
                    "manual_pending": self._manual_edit_pending,
                    "can_undo": bool(self._editor_undo), "can_redo": bool(self._editor_redo),
                    "changed": False, "edit_detail": edit_detail,
                }
            board_io.validate_board_data(data)
            result = self._load_editor_board(data, pending=True)
            self._editor_undo.append(before)
            self._editor_undo = self._editor_undo[-50:]
            self._editor_redo.clear()
            result.update(can_undo=True, can_redo=False, changed=True,
                          edit_detail=edit_detail)
            return result
        return _wrap(run)

    def confirm_manual_board(self):
        def run():
            data = self._editor_board_data()
            board_io.validate_board_data(data)
            result = self._load_editor_board(data, pending=False, confirmed=True)
            self._editor_undo.clear()
            self._editor_redo.clear()
            result.update(can_undo=False, can_redo=False)
            return result
        return _wrap(run)

    def save_manual_sample(self, note=""):
        def run():
            data = self._editor_board_data()
            board_io.validate_board_data(data)
            detected = json.loads(json.dumps(self._detected_board_data or data))
            corrections = recognition.board_corrections(detected, data)
            corrected_placements = {
                recognition.cell_key(item.get("after") or item.get("before") or {})
                for item in corrections
            }
            manual_by_placement = {
                recognition.cell_key(piece): deepcopy(piece)
                for piece in data.get("pieces", {}).values()
            }
            # Explicitly saving a provisional learned candidate on a new
            # screenshot is the second human confirmation that promotes it.
            for evidence in (self._detected_sheep_data or []):
                placement = recognition.cell_key(evidence)
                target = manual_by_placement.get(placement)
                if (not target or placement in corrected_placements
                        or not (evidence.get("learned_provisional")
                                or evidence.get("learned_direction_provisional"))):
                    continue
                presence = bool(evidence.get("learned_provisional"))
                corrections.append({
                    "kind": "add" if presence else "update",
                    "fields": (["presence", "species", "facing"] if presence else ["facing"]),
                    "before_id": None, "after_id": str(evidence.get("id")),
                    "before": None, "after": target,
                    "confirmation": True,
                    "confirms_samples": list(evidence.get("learned_sample_ids") or [
                        evidence.get("learned_sample_id") or
                        evidence.get("manual_learning_sample_id")
                    ]),
                })
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
            folder = os.path.join(common.HERE, "cache", "manual_samples", stamp)
            os.makedirs(folder, exist_ok=False)
            _write_json_atomic(os.path.join(folder, "board.json"), data)
            _write_json_atomic(os.path.join(folder, "manual_board.json"), data)
            _write_json_atomic(os.path.join(folder, "detected_board.json"), detected)
            _write_json_atomic(os.path.join(folder, "detected_sheep.json"),
                               self._detected_sheep_data or [])

            params_path = os.path.join(common.HERE, "grid_params.json")
            params = json.load(open(params_path, encoding="utf-8")) if os.path.exists(params_path) else {}
            grid_hash = hashlib.sha1(json.dumps(
                params, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
            observation_hash = (hashlib.sha1(self.game.tobytes()).hexdigest()
                                if isinstance(self.game, np.ndarray) and self.game.size else None)
            rect = (self.debug or {}).get("rect")
            candidate_pool = []
            for name in ("raw_candidates", "candidates", "dropped"):
                candidate_pool.extend((self.debug or {}).get(name) or [])

            def compact_candidate(candidate):
                fields = ("source_id", "detector", "detectors", "species", "cells", "facing",
                          "axis", "pair_score", "direction_confidence", "selection_score",
                          "quality", "drop_reason", "metrics", "direction_votes")
                return {key: deepcopy(candidate.get(key)) for key in fields if key in candidate}

            records = []
            evidence_dump = []
            for index, correction in enumerate(corrections):
                piece = correction.get("after") or correction.get("before") or {}
                placement = {tuple(cell) for cell in piece.get("cells", [])}
                overlaps = [compact_candidate(item) for item in candidate_pool
                            if placement and placement & {tuple(cell) for cell in item.get("cells", [])}]
                feature = (recognition.pair_visual_feature(rect, piece)
                           if isinstance(rect, np.ndarray) and rect.size else None)
                sample_id = f"{stamp}-{index + 1:03d}"
                record = {
                    "schema": recognition.MANUAL_LEARNING_SCHEMA,
                    "sample_id": sample_id,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "status": "active", "source": "manual-board-editor",
                    "observation_hash": observation_hash,
                    "grid_hash": grid_hash,
                    "recognition_version": "manual-supervision-v2",
                    "taxonomy_version": 1,
                    "sample_path": os.path.relpath(folder, common.HERE).replace("\\", "/"),
                    "correction": deepcopy(correction),
                    "feature": feature,
                    "evidence": {
                        "overlapping_candidates": overlaps,
                        "patch_hash": (feature or {}).get("patch_hash"),
                    },
                }
                learnable_fields = {"presence", "species", "facing"}
                if (feature is not None
                        and correction.get("kind") in {"add", "update", "delete"}
                        and learnable_fields & set(correction.get("fields") or [])):
                    records.append(record)
                evidence_dump.append({"sample_id": sample_id,
                                      "overlapping_candidates": overlaps})

            _write_json_atomic(os.path.join(folder, "corrections.json"), corrections)
            _write_json_atomic(os.path.join(folder, "recognition_evidence.json"), evidence_dump)
            _write_json_atomic(os.path.join(folder, "grid_params.json"), params)
            if self.game is not None:
                if not cv2.imwrite(os.path.join(folder, "capture.png"), self.game):
                    raise RuntimeError("人工样本原图写入失败")
            if isinstance(rect, np.ndarray) and rect.size:
                if not cv2.imwrite(os.path.join(folder, "rectified.png"), rect):
                    raise RuntimeError("人工样本校正图写入失败")
            # Publish to the active index only after the complete evidence
            # bundle is durable.  A failed bundle can never become a live
            # cross-level template.
            learning = recognition.record_manual_learning(records)
            final_corrected_placements = {
                recognition.cell_key(item.get("after") or item.get("before") or {})
                for item in corrections
            }
            automatic_confirmation_count = sum(
                len(recognition.cell_key(piece)) == 2
                and recognition.cell_key(piece) not in final_corrected_placements
                for piece in data.get("pieces", {}).values()
            )
            metadata = {
                "schema": 2, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "note": str(note or ""), "board_revision": level_cache.board_hash(data),
                "detected_board_revision": level_cache.board_hash(detected),
                "source": "manual-board-editor", "observation_hash": observation_hash,
                "grid_hash": grid_hash, "recognition_version": "manual-supervision-v2",
                "correction_count": len(corrections),
                "automatic_confirmation_count": automatic_confirmation_count,
                "learning": learning,
            }
            _write_json_atomic(os.path.join(folder, "metadata.json"), metadata)
            return {"saved": True, "path": folder, "metadata": metadata,
                    "corrections": len(corrections), "learning": learning}
        return _wrap(run)

    def _review_piece_ids(self):
        return {str(piece_id) for piece_id, meta in self._species_by_id.items()
                if meta.get("review")}
