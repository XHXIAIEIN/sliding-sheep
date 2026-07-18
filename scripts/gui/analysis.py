
import os
import json
import base64
from copy import deepcopy
import cv2
import numpy as np
from board import grid as G
from board import io as board_io
from core import analysis as analysis_engine
import vision as D
from levels import cache as level_cache
from levels import reader as level_reader
from core import safety
from core.capture import find_window, grab, list_windows  # 注：import 时已 SetProcessDPIAware
from paths import image_path
from . import common
from .common import TITLE, _wrap


class AnalysisOps:
    """Mixin: capture, frame analysis, and source-level snapshots."""

    @staticmethod
    def _confirm_stable_runtime_reviews(pieces, history, minimum_prior_frames=2):
        """Keep single-sample learning as a hard review blocker.

        Repeated runtime frames are correlated observations of the same learned
        mutation, so they cannot provide the independent evidence required to
        authorize it.
        """
        return []

    def _annotate_source_comparison(self, comparison):
        """Render direction-review evidence onto the exact current screenshot."""
        if self.game is None or self.Minv is None or not comparison:
            return comparison
        changes = (comparison.get("previous_direction_changes")
                   if comparison.get("previous_direction_changed")
                   else comparison.get("direction_changes")) or []
        if not changes:
            return comparison
        image = self.game.copy()
        tint = image.copy()
        old_color = (150, 135, 110)
        new_color = (0, 165, 255)
        review_color = (141, 77, 255)
        vectors = {"L": (0, -1), "R": (0, 1), "U": (-1, 0), "D": (1, 0)}
        for change in changes:
            cells = [tuple(map(int, cell)) for cell in change.get("cells", [])]
            if not cells:
                continue
            for r, c in cells:
                polygon = np.asarray(self._cell_poly(r, c), dtype=np.int32)
                cv2.fillPoly(tint, [polygon], review_color)
                cv2.polylines(image, [polygon], True, review_color, 4, cv2.LINE_AA)
            center_r = sum(r + .5 for r, _c in cells) / len(cells)
            center_c = sum(c + .5 for _r, c in cells) / len(cells)

            def arrow(facing, color, offset):
                dr, dc = vectors.get(str(facing), (0, 0))
                # Offset the two evidence arrows perpendicular to their axis so
                # opposite directions remain visible instead of covering each other.
                pr, pc = -dc * offset, dr * offset
                start = self._px(center_c + pc - dc * .18,
                                 center_r + pr - dr * .18)
                end = self._px(center_c + pc + dc * .52,
                               center_r + pr + dr * .52)
                cv2.arrowedLine(image, tuple(np.round(start).astype(int)),
                                tuple(np.round(end).astype(int)), color, 5,
                                cv2.LINE_AA, tipLength=.28)

            arrow(change.get("from"), old_color, -.13)
            arrow(change.get("to"), new_color, .13)
            center = np.asarray(self._px(center_c, center_r), dtype=float)
            label = ("-".join(self._cell_label(cell) for cell in cells)
                     + f"  {change.get('from', '?')} > {change.get('to', '?')}")
            (tw, th), _baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, .72, 2)
            x = max(4, min(image.shape[1] - tw - 12, int(center[0] - tw / 2)))
            y = max(th + 8, min(image.shape[0] - 8, int(center[1] - 45)))
            cv2.rectangle(image, (x - 5, y - th - 6), (x + tw + 5, y + 6),
                          (28, 32, 38), -1)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .72,
                        (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.addWeighted(tint, .18, image, .82, 0)
        output = level_cache.source_annotation_path(comparison)
        if not cv2.imwrite(str(output), image):
            raise RuntimeError("源关卡审核标注图保存失败")
        return level_cache.register_source_annotation(comparison)

    def _read_source_level(self, fallback=None, *, allow_missing=False):
        """Prefer the visible title and retain the supplied label only as fallback."""
        reading = level_reader.read_level(self.game)
        label = reading.label if reading else " ".join(str(fallback or "").strip().split())
        if not label:
            if allow_missing:
                return {"level_label": None, "level_auto_read": False,
                        "level_read_method": "unavailable", "level_bbox": None}
            raise RuntimeError("未能从画面识别“第 XXX 关”，也没有可用的源关卡编号")
        if self._source_level_label and label != self._source_level_label:
            self._frame_history.clear()
            self._wolf_observations.clear()
            self._wolf_motion = None
            self._wolf_danger_cells.clear()
            self._wolf_confirmed_cells.clear()
            self._opening_coarse_pending = True
        self._source_level_label = label
        self._level_key = level_cache.source_level_key(label)
        return {
            "level_label": label,
            "level_auto_read": reading is not None,
            "level_read_method": reading.method if reading else "input-fallback",
            "level_bbox": list(reading.bbox) if reading else None,
        }

    def capture(self, source_level_label=None):
        def run():
            self._cancel_event.clear()
            self.hwnd = self._target_window()
            if not self.hwnd:
                raise RuntimeError(f"找不到窗口「{TITLE}」，确认游戏已打开")
            img, rectinfo, mode = grab(self.hwnd)
            self.game = img
            self._input_mode = "operator"
            self.win = rectinfo
            level = self._read_source_level(source_level_label)
            cv2.imwrite(str(image_path("_game.png")), img)   # 与命令行流水线共用
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                raise RuntimeError("截图编码失败")
            x, y, w, h = rectinfo
            return {"win": {"w": w, "h": h}, "mode": mode, **level,
                    "img": base64.b64encode(buf.tobytes()).decode("ascii")}
        return _wrap(run)

    def _analyze_frame(self, *, source, persist=True):
        """Analyze self.game once and apply the single authoritative safety gate."""
        if self.game is None:
            raise RuntimeError("请先截图")
        previous_revision = self.board_revision
        previous_board = self._clone_board(self.board) if self.board is not None else None
        previous_plan = self._active_plan
        previous_Minv = self.Minv.copy() if self.Minv is not None else None
        previous_sheep = deepcopy(self.sheep)
        params_path = common.data_path("grid_params.json")
        bundle = analysis_engine.analyze_image(
            self.game,
            analysis_engine.load_params(params_path),
            temporal_history=list(self._frame_history),
            recover_missing_edges=source == "app-execution-preflight",
            ignore_outside_pieces=self.ignore_outside_pieces,
        )
        if not bundle.calibrated:
            self.board = None
            self.board_revision = None
            self.scene_report = bundle.report
            return {"rows": 0, "cols": 0, "grid": [], "count": 0, "cache": None,
                    "layout": None, "state": None, **self.scene_report}

        grid_model = bundle.grid
        rows, cols = grid_model.rows, grid_model.cols
        self.rows, self.cols = rows, cols
        self.Minv = grid_model.inverse_matrix
        D._remove_obsolete_images()
        G.save_grid_data(grid_model, common.data_path("board_grid.json"))
        self.sheep, debug = bundle.sheep, bundle.debug
        runtime_confirmed = self._confirm_stable_runtime_reviews(
            self.sheep, self._frame_history)
        if runtime_confirmed:
            debug["runtime_confirmed_reviews"] = runtime_confirmed
        self._detected_sheep_data = deepcopy(self.sheep)
        self.debug = debug
        self._remember_wolf_observation(debug)
        self._sync_species()
        bd = deepcopy(bundle.board_data)
        # Keep planning/cache identity independent of the wolf's per-frame
        # body position. Only a confirmed patrol corridor constrains landing;
        # live body cells belong to the execution guard below.
        bd["no_stop"] = [list(cell) for cell in sorted(self._wolf_confirmed_cells)]
        self._detected_board_data = json.loads(json.dumps(bd))
        self._manual_edit_pending = False
        self._editor_undo.clear()
        self._editor_redo.clear()
        layout = bundle.layout
        report = deepcopy(bundle.report)
        try:
            board_io.validate_board_data(bd)
        except board_io.BoardValidationError as exc:
            report = safety.add_blockers(report, [
                safety.blocker("board_schema_invalid", "棋盘结构校验失败", detail=exc.errors)
            ])
        self.scene_report = report

        candidates = analysis_engine.audit_payload(
            bundle, runtime_confirmed_reviews=runtime_confirmed)
        if persist:
            json.dump(candidates, open(common.data_path("sheep_candidates.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            json.dump(report, open(common.data_path("scene_report.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            cv2.imwrite(str(image_path("_occ_axis_rect.png")), D.render_rect_debug(debug, self.sheep))
            cv2.imwrite(str(image_path("_grid_labels.png")), D.render_grid_labels(debug, self.sheep))
            cv2.imwrite(str(image_path("_layout.png")), D.render_layout(debug, self.sheep))

        # A tutorial hand or short-lived smoke may cover a completely unrelated
        # sheep.  When a trusted solved board already exists, retain that board
        # and plan instead of replacing it with the masked recognition result.
        if (self._visual_transient_only(report) and previous_board is not None
                and previous_plan and previous_Minv is not None):
            # Execution preflight must see the tutorial target.  The normal
            # refresh path may reuse the trusted board, but the orchestrator
            # needs the red target so it can click the indicated direct exit.
            if source.startswith(("app-execution", "app-burst")):
                pass
            else:
                restored = self._restore_predicted_after_visual_transient(
                    report, previous_board, previous_Minv, previous_sheep)
                self._active_plan = previous_plan
                if persist:
                    json.dump(restored, open(common.data_path("scene_report.json"), "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
                return {
                    "grid": self._grid_lines(rows, cols), "rows": rows, "cols": cols,
                    "count": self.board.remaining_count(), "cache": self._last_cache,
                    "layout": layout, "board_revision": self.board_revision,
                    **restored,
                }

        if report["scene_state"] == "gameplay":
            self._frame_history.append(debug.get("observation"))

        if report["scene_state"] != "gameplay":
            self.board = None
            self.board_revision = None
            self._active_plan = None
            for stale in ("board.json", "board_layout.json"):
                path = common.data_path(stale)
                if os.path.exists(path):
                    os.remove(path)
            return {"grid": self._grid_lines(rows, cols), "rows": rows, "cols": cols,
                    "count": 0 if report.get("execution_complete") else len(self.sheep),
                    "cache": None, "layout": layout,
                    "state": None, **report}

        if persist:
            json.dump(bd, open(common.data_path("board.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            json.dump(layout, open(common.data_path("board_layout.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        try:
            board_io.validate_board_data(bd)
            self.board = board_io.load(common.data_path("board.json")) if persist else board_io.Board(
                rows=bd["rows"], cols=bd["cols"], pieces=bd["pieces"], model=bd["model"],
                slide_mode=bd["slide_mode"], hazards=bd.get("hazards", []),
                fences=bd.get("fences", []), no_stop=bd.get("no_stop", []))
        except board_io.BoardValidationError:
            self.board = None
        self.board_revision = level_cache.board_hash(bd)
        # A preflight capture of the same board must not invalidate the plan
        # that it is trying to verify.  Only a real board change makes the
        # current plan stale.
        if self.board_revision != previous_revision:
            self._active_plan = None

        cache_meta = None
        source_comparison = None
        if persist:
            if self._level_key is None:
                self._level_key = self.board_revision[:16]
            comparison_sources = {
                "app-detect", "app-auto-retry-failure", "app-auto-retry-in_level",
                "app-next-level-stabilize",
            }
            if self._source_level_label and source in comparison_sources:
                source_comparison = level_cache.record_source_comparison(
                    bd, self._source_level_label, source=source)
                source_comparison = self._annotate_source_comparison(source_comparison)
            cache_meta = level_cache.save_capture(
                bd, level_key=self._level_key, source=source,
                extra={"rows": rows, "cols": cols, "candidate_count": debug["candidate_count"],
                       "hazard_count": len(debug.get("hazards", [])),
                       "fence_count": len(debug.get("fences", [])),
                       "scene_state": report["scene_state"], "executable": report["executable"],
                       "execution_blockers": [b["code"] for b in report["execution_blockers"]],
                       "source_comparison": source_comparison},
            )
            if source_comparison:
                cache_meta["source_comparison"] = source_comparison
            self._last_cache = cache_meta
        state = self._snapshot(self.board, highlight=None) if self.board is not None else None
        return {"grid": self._grid_lines(rows, cols), "rows": rows, "cols": cols,
                "count": len(self.sheep), "cache": cache_meta, "layout": layout,
                "state": state, "board_revision": self.board_revision,
                "source_comparison": source_comparison, **report}

    def rebuild_source_level(self, source_level_label):
        """Replace the selected level's source with the currently recognized full board."""
        def run():
            if self.board is None:
                raise RuntimeError("请先采集并识别当前关卡")
            label = " ".join(str(source_level_label or "").strip().split())
            if not label:
                raise RuntimeError("请先填写关卡编号或名称")
            data = self._board_data(self.board)
            self._source_level_label = label
            self._level_key = level_cache.source_level_key(label)
            result = level_cache.record_source_comparison(
                data, label, rebuild=True, source="app-rebuild-source")
            return {"source_comparison": result}
        return _wrap(run)

    def get_source_snapshot(self, level_key, sample_number=0, variant="current"):
        """Return a source-review snapshot as base64 for the pywebview audit modal."""
        def run():
            path = level_cache.source_snapshot_path(
                level_key, int(sample_number or 0), str(variant or "current"))
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return {"img": encoded, "variant": str(variant),
                    "sample_number": int(sample_number or 0), "name": path.name}
        return _wrap(run)

    def detect(self, ignore_outside_pieces=True):
        def run():
            self.ignore_outside_pieces = bool(ignore_outside_pieces)
            return self._analyze_frame(source="app-detect")
        return _wrap(run)
