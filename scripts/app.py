"""「套住那只羊」pywebview 桌面应用。

The UI submits one user intent through workflow_start().  OperationCoordinator
is the only background owner; analysis, planning, execution and verification
publish one observable job state.  Automatic mode repeats the same verified
single-step primitive as manual execution, so there is no recursive lock path
and no unverified burst click path.

运行：py scripts/app.py（游戏窗口需可见；F12 / Esc 在当前点击结束后暂停）
"""
import os
import sys
import json
import base64
import hashlib
import time
import threading
import ctypes
import traceback
from copy import deepcopy
from collections import deque
from ctypes import wintypes

import cv2
import numpy as np
import webview

import board_grid as G
import board_io
import analysis_engine
import vision as D
import level_cache
import level_reader
import planner
import recognition
import runtime as app_runtime
import safety
import solver_learning
from solver import DIRS, Move
from capture_window import find_window, grab, list_windows  # 注：import 时已 SetProcessDPIAware
from paths import image_path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITLE = "套住那只羊"
DEFAULT_SOLVE_TIMEOUT = 10.0
DESKTOP_WINDOW_SIZE = (1320, 900)
REFERENCE_WINDOW_SIZE = (460, 900)
MIN_WINDOW_SIZE = (390, 640)
LOG_DIR = os.path.join(HERE, "logs")
RETRY_CONTROLS_PATH = os.path.join(HERE, "retry_controls.json")
RUNTIME_SETTINGS_PATH = os.path.join(HERE, "runtime_settings.json")
DEFAULT_RUNTIME_SETTINGS = {
    "solve_timeout_s": 10,
    "timeout_extension_s": 5,
    "timeout_max_s": 30,
    "elastic_timeout": True,
    "settle_ms": 60,
    "max_steps": 200,
    "source_level_label": "",
    "updated_at_ms": 0,
}
_RUNTIME_SETTINGS_LOCK = threading.RLock()
_SINGLETON_HANDLE = None
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
HOTKEYS = {
    1: ("capture", 0x75),    # F6
    2: ("exec", 0x76),       # F7
    3: ("auto", 0x77),       # F8
    4: ("replay", 0x78),     # F9
    5: ("stop", 0x7B),       # F12
    6: ("stop", 0x23),       # End
    7: ("stop", 0x13),       # Pause
    8: ("stop", 0x1B),       # Esc
}


class ExecutionReviewRequired(RuntimeError):
    """A planned click needs a human-facing, precisely located review marker."""

    def __init__(self, piece_id, cells, reason=None):
        cells = [list(cell) for cell in sorted(cells)]
        labels = [f"{chr(65 + int(c)) if int(c) < 26 else int(c) + 1}{int(r) + 1}"
                  for r, c in cells]
        location = "–".join(labels)
        message = (f"下一步需要低置信度棋子 #{piece_id}（{location}）；"
                   "请确认或修正后继续执行")
        super().__init__(message)
        self.payload = {
            "error_code": "manual_review_required",
            "review_required": True,
            "review_message": message,
            "piece_id": str(piece_id),
            "cells": cells,
            "location": location,
            "review_reason": reason,
        }


def _write_json_atomic(path, data):
    """Serialize completely before replacing a JSON artifact."""
    tmp = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        with open(tmp, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _bounded_int(value, fallback, minimum, maximum):
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        value = int(fallback)
    return max(int(minimum), min(int(maximum), value))


def _normalize_runtime_settings(data=None):
    data = data if isinstance(data, dict) else {}
    initial = _bounded_int(data.get("solve_timeout_s"), 10, 1, 60)
    maximum = max(
        initial,
        _bounded_int(data.get("timeout_max_s"), 30, 1, 300),
    )
    elastic = data.get("elastic_timeout", True)
    if not isinstance(elastic, bool):
        elastic = True
    source = data.get("source_level_label", "")
    if not isinstance(source, str):
        source = str(source or "")
    return {
        "solve_timeout_s": initial,
        "timeout_extension_s": _bounded_int(
            data.get("timeout_extension_s"), 5, 1, 60),
        "timeout_max_s": maximum,
        "elastic_timeout": elastic,
        "settle_ms": _bounded_int(data.get("settle_ms"), 60, 20, 3000),
        "max_steps": _bounded_int(data.get("max_steps"), 200, 1, 500),
        "source_level_label": source.strip()[:120],
        "updated_at_ms": max(
            0, _bounded_int(
                data.get("updated_at_ms"), 0, 0,
                int(time.time() * 1000) + 86_400_000,
            )),
    }


def _read_runtime_settings():
    try:
        with open(RUNTIME_SETTINGS_PATH, encoding="utf-8") as stream:
            return _normalize_runtime_settings(json.load(stream))
    except Exception:
        return dict(DEFAULT_RUNTIME_SETTINGS)


def _load_params():
    P = json.load(open(os.path.join(HERE, "grid_params.json"), encoding="utf-8"))
    return (P["corners"], int(P["rows"]), int(P["cols"]),
            int(P.get("imgW", 0)), int(P.get("imgH", 0)))


def _empty_scene_report():
    return {
        "scene_state": "unknown",
        "execution_blockers": [safety.blocker("not_analyzed", "尚未完成场景识别")],
        "executable": False,
    }


class Api:
    """暴露给前端 JS 的桥。每个方法成功返回带 ok=True 的 dict，异常被 _wrap 兜成 {ok:False,error}。"""

    def __init__(self):
        self.game = None          # 当前截图 BGR ndarray
        self.hwnd = None          # 游戏窗口句柄（click 时重新取实时位置）
        self.target_hwnd = None   # 用户显式选择的游戏窗口句柄
        self._ui_window = None     # pywebview 窗口，用于全局快捷键回调前端
        self._window_mode = "operator"
        self._operator_window_rect = None
        self.win = None           # (x, y, w, h) 截图时的窗口屏幕矩形
        self.Minv = None          # 校正网格 -> 原图像素 的逆透视矩阵
        self.rows = self.cols = 0
        self.sheep = None         # detect 得到的羊列表
        self._species_by_id = {}
        self.debug = None         # detect 的中间结果，方向修正后重绘标注图用
        self.board = None         # board_io.Board
        self.runtime = app_runtime.OperationCoordinator()
        self._hotkey_thread = None
        self._level_key = None
        self._source_level_label = None
        self._last_cache = None
        self.scene_report = _empty_scene_report()
        self.board_revision = None
        self._active_plan = None
        self._opening_coarse_pending = True
        self._execution_lock = threading.Lock()
        self._cancel_event = self.runtime.cancel_event
        self._frame_history = deque(maxlen=4)
        self._wolf_observations = deque(maxlen=8)
        self._wolf_motion = None
        self._wolf_danger_cells = set()
        self._wolf_confirmed_cells = set()
        self._manual_edit_pending = False
        self._detected_board_data = None
        self._detected_sheep_data = None
        self._editor_undo = []
        self._editor_redo = []
        self.ignore_outside_pieces = True
        self._input_mode = "operator"

    @staticmethod
    def _confirm_stable_runtime_reviews(pieces, history, minimum_prior_frames=2):
        """Keep single-sample learning as a hard review blocker.

        Repeated runtime frames are correlated observations of the same learned
        mutation, so they cannot provide the independent evidence required to
        authorize it.
        """
        return []

    # ---- 坐标工具：校正网格坐标(列gx,行gy) -> 截图原图像素 [x,y] ----
    def _px(self, gx, gy, Minv=None):
        matrix = self.Minv if Minv is None else Minv
        v = matrix @ np.array([gx * D.CELL, gy * D.CELL, 1.0])
        return [float(v[0] / v[2]), float(v[1] / v[2])]

    def _cell_center(self, r, c, Minv=None):
        return self._px(c + 0.5, r + 0.5, Minv=Minv)

    def _cell_poly(self, r, c, Minv=None):
        return [
            self._px(c, r, Minv=Minv),
            self._px(c + 1, r, Minv=Minv),
            self._px(c + 1, r + 1, Minv=Minv),
            self._px(c, r + 1, Minv=Minv),
        ]

    @staticmethod
    def _cell_label(cell):
        r, c = int(cell[0]), int(cell[1])
        return f"{chr(65 + c) if c < 26 else c + 1}{r + 1}"

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

    def start_hotkeys(self, window):
        # The window is also used by the mode-aware sizing API.  Register it
        # even when global hotkeys are disabled for tests or local debugging.
        self._ui_window = window
        if os.environ.get("SHEEP_DISABLE_HOTKEYS") == "1":
            print("全局快捷键已通过 SHEEP_DISABLE_HOTKEYS=1 禁用")
            return
        if self._hotkey_thread and self._hotkey_thread.is_alive():
            return
        self._hotkey_thread = threading.Thread(target=self._hotkey_loop, daemon=True)
        self._hotkey_thread.start()

    @staticmethod
    def _reference_window_rect():
        """Return a centered phone-like rect that fits the primary work area."""
        target_width, target_height = REFERENCE_WINDOW_SIZE
        target_x = target_y = None
        work_area = wintypes.RECT()
        try:
            if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0):
                dpi = int(user32.GetDpiForSystem()) if hasattr(user32, "GetDpiForSystem") else 96
                scale = max(1.0, dpi / 96.0)
                work_left, work_top = work_area.left / scale, work_area.top / scale
                available_width = (work_area.right - work_area.left) / scale
                available_height = (work_area.bottom - work_area.top) / scale
                target_width = min(target_width, max(MIN_WINDOW_SIZE[0], int(available_width - 32)))
                target_height = min(target_height, max(MIN_WINDOW_SIZE[1], int(available_height - 32)))
                target_x = int(work_left + (available_width - target_width) / 2)
                target_y = int(work_top + (available_height - target_height) / 2)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
        return target_x, target_y, target_width, target_height

    def set_window_mode(self, mode):
        """Resize the host with the operator/reference product switch."""
        def run():
            window = self._ui_window
            if window is None:
                raise RuntimeError("应用窗口尚未就绪")
            next_mode = "reference" if str(mode) == "reference" else "operator"
            current = (
                int(getattr(window, "x", 0) or 0),
                int(getattr(window, "y", 0) or 0),
                int(getattr(window, "width", DESKTOP_WINDOW_SIZE[0]) or DESKTOP_WINDOW_SIZE[0]),
                int(getattr(window, "height", DESKTOP_WINDOW_SIZE[1]) or DESKTOP_WINDOW_SIZE[1]),
            )

            if next_mode == "reference":
                if self._window_mode != "reference":
                    self._operator_window_rect = current
                x, y, width, height = self._reference_window_rect()
                window.restore()
                window.resize(width, height)
                if x is not None and y is not None:
                    window.move(x, y)
            elif self._window_mode == "reference":
                x, y, width, height = self._operator_window_rect or (
                    current[0], current[1], *DESKTOP_WINDOW_SIZE)
                width = max(960, int(width))
                height = max(640, int(height))
                window.restore()
                window.resize(width, height)
                window.move(int(x), int(y))
            else:
                width, height = current[2], current[3]

            self._window_mode = next_mode
            return {"mode": next_mode, "width": int(width), "height": int(height)}

        return _wrap(run)

    def _hotkey_loop(self):
        registered = []
        for hotkey_id, (_action, vk) in HOTKEYS.items():
            if user32.RegisterHotKey(None, hotkey_id, MOD_NOREPEAT, vk):
                registered.append(hotkey_id)
        if not registered:
            print("全局快捷键注册失败：可能被其他程序占用")
            return
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                if msg.message == WM_HOTKEY:
                    action = HOTKEYS.get(int(msg.wParam), (None, None))[0]
                    if action:
                        self._dispatch_hotkey(action)
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            for hotkey_id in registered:
                user32.UnregisterHotKey(None, hotkey_id)

    def _dispatch_hotkey(self, action):
        if not self._ui_window:
            return
        detail = json.dumps(str(action))
        script = f"window.dispatchEvent(new CustomEvent('app-global-hotkey', {{detail: {detail}}}));"
        try:
            self._ui_window.evaluate_js(script)
        except Exception:
            pass

    # ---- 1) 截图 ----
    def list_targets(self):
        def run():
            return {"windows": list_windows(TITLE),
                    "selected": str(self.target_hwnd) if self.target_hwnd else None}
        return _wrap(run)

    def load_runtime_settings(self):
        """Return normalized durable UI settings without changing app state."""
        return _wrap(lambda: {"settings": _read_runtime_settings()})

    def save_runtime_settings(self, settings=None):
        """Atomically keep the newest settings snapshot from the UI."""
        def run():
            incoming = _normalize_runtime_settings(settings)
            with _RUNTIME_SETTINGS_LOCK:
                current = _read_runtime_settings()
                if int(current.get("updated_at_ms") or 0) > incoming["updated_at_ms"]:
                    saved = current
                else:
                    _write_json_atomic(RUNTIME_SETTINGS_PATH, incoming)
                    saved = incoming
            return {"settings": saved}
        return _wrap(run)

    def hard_refresh(self):
        """Reset the live app state and remove only the current capture artifacts.

        Calibration, runtime settings, recognition learning, and archived
        level caches are durable data and intentionally survive this operation.
        The frontend reloads after this bridge call so its JS state is recreated.
        """
        def run():
            self.runtime.cancel()
            self._cancel_event.set()
            if not self.runtime.wait(timeout=4.0):
                raise RuntimeError("旧任务尚未退出，请稍后再强刷")
            if self._execution_lock.locked():
                raise RuntimeError("正在执行点击，请等待当前点击完成后再强刷")

            removed = []
            current_artifacts = tuple(level_cache.ARTIFACTS) + ("images/_solution.png",)
            for relative in current_artifacts:
                path = os.path.join(HERE, relative)
                if os.path.exists(path):
                    os.remove(path)
                    removed.append(relative.replace("\\", "/"))
            self.game = None
            self.hwnd = None
            self.target_hwnd = None
            self.win = None
            self.Minv = None
            self.rows = self.cols = 0
            self.sheep = None
            self._species_by_id = {}
            self.debug = None
            self.board = None
            self._level_key = None
            self._source_level_label = None
            self._last_cache = None
            self.scene_report = _empty_scene_report()
            self.board_revision = None
            self._active_plan = None
            self._opening_coarse_pending = True
            self._frame_history.clear()
            self._wolf_observations.clear()
            self._wolf_motion = None
            self._wolf_danger_cells.clear()
            self._wolf_confirmed_cells.clear()
            self._manual_edit_pending = False
            self._detected_board_data = None
            self._detected_sheep_data = None
            self._editor_undo.clear()
            self._editor_redo.clear()
            self.ignore_outside_pieces = True
            self._input_mode = "operator"
            self._cancel_event.clear()
            self.runtime.reset()
            return {"reset": True, "removed": removed}
        return _wrap(run)

    def select_target(self, hwnd):
        def run():
            requested = int(hwnd)
            candidates = {int(item["hwnd"]): item for item in list_windows(TITLE)}
            if requested not in candidates:
                raise RuntimeError("所选窗口已失效，请刷新列表")
            self.target_hwnd = requested
            self.hwnd = requested
            self._frame_history.clear()
            self._wolf_observations.clear()
            self._wolf_motion = None
            self._wolf_danger_cells.clear()
            self._wolf_confirmed_cells.clear()
            return {"selected": str(requested), "window": candidates[requested]}
        return _wrap(run)

    def _target_window(self):
        if self.target_hwnd and user32.IsWindow(self.target_hwnd):
            return self.target_hwnd
        self.target_hwnd = None
        return find_window(TITLE)

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

    def _grid_lines(self, rows, cols):
        grid = []
        for r in range(rows + 1):
            grid.append([self._px(0, r), self._px(cols, r)])
        for c in range(cols + 1):
            grid.append([self._px(c, 0), self._px(c, rows)])
        return grid

    def _analyze_frame(self, *, source, persist=True):
        """Analyze self.game once and apply the single authoritative safety gate."""
        if self.game is None:
            raise RuntimeError("请先截图")
        previous_revision = self.board_revision
        previous_board = self._clone_board(self.board) if self.board is not None else None
        previous_plan = self._active_plan
        previous_Minv = self.Minv.copy() if self.Minv is not None else None
        previous_sheep = deepcopy(self.sheep)
        params_path = os.path.join(HERE, "grid_params.json")
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
        G.save_grid_data(grid_model, os.path.join(HERE, "board_grid.json"))
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
            json.dump(candidates, open(os.path.join(HERE, "sheep_candidates.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            json.dump(report, open(os.path.join(HERE, "scene_report.json"), "w", encoding="utf-8"),
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
                    json.dump(restored, open(os.path.join(HERE, "scene_report.json"), "w", encoding="utf-8"),
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
                path = os.path.join(HERE, stale)
                if os.path.exists(path):
                    os.remove(path)
            return {"grid": self._grid_lines(rows, cols), "rows": rows, "cols": cols,
                    "count": 0 if report.get("execution_complete") else len(self.sheep),
                    "cache": None, "layout": layout,
                    "state": None, **report}

        if persist:
            json.dump(bd, open(os.path.join(HERE, "board.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            json.dump(layout, open(os.path.join(HERE, "board_layout.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        try:
            board_io.validate_board_data(bd)
            self.board = board_io.load(os.path.join(HERE, "board.json")) if persist else board_io.Board(
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

    # ---- 2) 识别 ----
    def detect(self, ignore_outside_pieces=True):
        def run():
            self.ignore_outside_pieces = bool(ignore_outside_pieces)
            return self._analyze_frame(source="app-detect")
        return _wrap(run)

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
            folder = os.path.join(HERE, "cache", "manual_samples", stamp)
            os.makedirs(folder, exist_ok=False)
            _write_json_atomic(os.path.join(folder, "board.json"), data)
            _write_json_atomic(os.path.join(folder, "manual_board.json"), data)
            _write_json_atomic(os.path.join(folder, "detected_board.json"), detected)
            _write_json_atomic(os.path.join(folder, "detected_sheep.json"),
                               self._detected_sheep_data or [])

            params_path = os.path.join(HERE, "grid_params.json")
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
                    "sample_path": os.path.relpath(folder, HERE).replace("\\", "/"),
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

    # ---- 3) 求解 ----
    def solve(self):
        def run():
            if self.board is None:
                raise RuntimeError("当前场景没有通过棋盘结构校验")
            return self._solve_with_budget(self._clone_board(self.board), DEFAULT_SOLVE_TIMEOUT,
                                           self.Minv.copy())
        return _wrap(run)

    def solve_start(self, timeout_ms=None):
        """Start the sole background job; repeated clicks cannot spawn peers."""
        def run():
            if self.board is None:
                raise RuntimeError("当前场景没有通过棋盘结构校验")
            timeout_s = DEFAULT_SOLVE_TIMEOUT
            if timeout_ms is not None:
                timeout_s = max(1.0, min(60.0, float(timeout_ms) / 1000.0))
            board = self._clone_board(self.board)
            Minv = self.Minv.copy()

            def worker(context):
                context.publish(phase=app_runtime.Phase.SOLVING,
                                detail="正在求解", timeout_ms=int(timeout_s * 1000))
                return self._solve_with_budget(
                    board, timeout_s, Minv, job_id=context.job_id,
                    started=time.monotonic())

            job = self.runtime.start("solve", worker)
            return {"job": self._job_public(job["id"])}
        return _wrap(run)

    def solve_status(self, job_id=None):
        def run():
            return {"job": self._job_public(job_id)}
        return _wrap(run)

    def _workflow_capture(self, context, source_level_label=None):
        context.publish(phase=app_runtime.Phase.CAPTURING, detail="正在采集游戏画面")
        context.checkpoint()
        rectinfo, mode = self._capture_live(require_same_window=False)
        level = self._read_source_level(source_level_label, allow_missing=True)
        context.publish(phase=app_runtime.Phase.ANALYZING, detail="正在识别棋盘")
        result = self._analyze_frame(source="app-detect")
        ok, encoded = cv2.imencode(".png", self.game)
        if not ok:
            raise RuntimeError("截图编码失败")
        return {
            "ok": True,
            **result,
            **level,
            "reference_only": False,
            "capture": {"mode": mode, "win": {"w": rectinfo[2], "h": rectinfo[3]}},
            "img": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

    def _workflow_upload(self, context, encoded_image, file_name=None):
        """Decode one album screenshot and analyze it without window access.

        Uploaded frames are reference-only inputs.  They may be reviewed and
        solved, but never authorize click execution against a desktop window.
        """
        context.publish(phase=app_runtime.Phase.CAPTURING,
                        detail="正在读取相册截图")
        context.checkpoint()
        encoded = str(encoded_image or "")
        if "," in encoded and encoded.lstrip().startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        if not encoded:
            raise RuntimeError("没有收到截图数据")
        if len(encoded) > 48 * 1024 * 1024:
            raise RuntimeError("截图过大，请上传 24 MB 以内的图片")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise RuntimeError("截图数据无法解码") from exc
        if len(raw) > 24 * 1024 * 1024:
            raise RuntimeError("截图过大，请上传 24 MB 以内的图片")
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise RuntimeError("无法读取这张图片，请改用 PNG、JPEG 或 WebP")
        height, width = image.shape[:2]
        if width < 320 or height < 320:
            raise RuntimeError("截图分辨率太低，请上传完整棋盘截图")
        if width * height > 36_000_000:
            raise RuntimeError("截图像素过大，请先缩小后再上传")

        self.game = image
        self.win = (0, 0, width, height)
        self.hwnd = None
        self._input_mode = "reference"
        self._frame_history.clear()
        self._wolf_observations.clear()
        self._wolf_motion = None
        self._wolf_danger_cells.clear()
        self._wolf_confirmed_cells.clear()
        cv2.imwrite(str(image_path("_game.png")), image)
        level = self._read_source_level(allow_missing=True)
        context.publish(phase=app_runtime.Phase.ANALYZING,
                        detail="正在识别上传的棋盘")
        result = self._analyze_frame(source="app-upload")
        ok, normalized = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("截图编码失败")
        return {
            "ok": True,
            **result,
            **level,
            "reference_only": True,
            "uploaded_name": str(file_name or "截图")[:160],
            "capture": {"mode": "album", "win": {"w": width, "h": height}},
            "img": base64.b64encode(normalized.tobytes()).decode("ascii"),
        }

    @staticmethod
    def _panel_solution_payload(solution):
        """Keep the live inspector useful without polling every sandbox state."""
        if not isinstance(solution, dict):
            return None
        keys = (
            "total", "solved", "remaining", "kind", "timeout", "timeout_ms",
            "initial_timeout_ms", "max_timeout_ms", "elapsed_ms",
            "budget", "process_trace",
            "suspicious", "suspicion", "result_type", "failure_reason",
            "execution_ready", "safe_prefix_ready", "execution_scope",
            "execution_total", "execution_blocker", "coarse_total",
            "refine_total", "bomb_count", "bomb_live_control",
            "wolf_risk_total", "board_revision", "quick_exit",
            "exit_layers", "max_exit_layers",
        )
        panel = {key: deepcopy(solution.get(key)) for key in keys if key in solution}
        panel["moves"] = deepcopy(list(solution.get("moves") or []))
        return panel

    @staticmethod
    def _panel_analysis_payload(payload):
        """Extract the fields rendered by the right-side safety panel."""
        if not isinstance(payload, dict):
            return None
        keys = (
            "scene_state", "scene_reason", "execution_blockers", "advisories",
            "warnings", "metrics", "rows", "cols", "count", "board_revision",
            "cache", "executable", "execution_complete", "capture",
        )
        return {key: deepcopy(payload.get(key)) for key in keys if key in payload}

    def _workflow_solve(self, context, timeout_s, plan_completed_base=0,
                        *, elastic_timeout=False, extension_s=5.0,
                        max_timeout_s=None):
        if self.board is None or self.Minv is None:
            raise RuntimeError("当前没有可求解的棋盘，请先分析")
        context.publish(
            phase=app_runtime.Phase.SOLVING,
            detail="正在求解",
            progress={},
            panel_solution=None,
            plan_completed_base=max(0, int(plan_completed_base)),
        )
        solution = self._solve_with_budget(
            self._clone_board(self.board), timeout_s, self.Minv.copy(),
            job_id=context.job_id, started=time.monotonic(),
            elastic_timeout=elastic_timeout, extension_s=extension_s,
            max_timeout_s=max_timeout_s)
        context.publish(
            phase=app_runtime.Phase.SOLVING,
            detail=(f"解法已生成 · {int(solution.get('total') or 0)} 步 · "
                    f"剩 {int(solution.get('remaining') or 0)} 只"),
            progress={
                "completed": int(solution.get("total") or 0),
                "total": int(solution.get("total") or 0),
                "remaining": int(solution.get("remaining") or 0),
            },
            panel_solution=self._panel_solution_payload(solution),
            plan_completed_base=max(0, int(plan_completed_base)),
        )
        return solution

    def _workflow_opening_coarse(self, context):
        """Publish only the immediately executable ordinary-sheep exits."""
        if self.board is None or self.Minv is None:
            raise RuntimeError("当前没有可粗解的棋盘，请先分析")
        context.publish(
            phase=app_runtime.Phase.OPENING_COARSE,
            detail="正在生成开局直出闭包",
            progress={"steps": 0, "remaining": self.board.remaining_count()},
            panel_solution=None,
            plan_completed_base=0,
        )

        def progress(_phase, data):
            context.publish(
                phase=app_runtime.Phase.OPENING_COARSE,
                detail="正在收集可直接离场的普通羊",
                progress=data,
            )

        try:
            planned = planner.coarse_exit_plan(
                self._clone_board(self.board),
                cancel=self._cancel_event.is_set,
                progress=progress,
                exit_priority=self._execution_exit_priority,
            )
        except planner.PlanningCancelled as exc:
            raise RuntimeError("执行已取消") from exc
        if not planned.steps:
            return None
        solution = self._build_solution_payload(
            self._clone_board(self.board), planned.steps, planned.final_board,
            planned.solved, planned.remaining, planned.kind, 0.0, False,
            self.Minv.copy(), cache_info={"hit": False, "opening_coarse": True},
            suspicion=None,
        )
        context.publish(
            phase=app_runtime.Phase.OPENING_COARSE,
            detail=f"开局粗解已生成 · {int(solution.get('total') or 0)} 步",
            progress={
                "completed": 0,
                "total": int(solution.get("total") or 0),
                "remaining": int(solution.get("remaining") or 0),
            },
            panel_solution=self._panel_solution_payload(solution),
            plan_completed_base=0,
        )
        return solution

    def _workflow_quick_exit(self, context, max_layers=3):
        """Build at most three freshly exposed layers of plain EXIT moves."""
        if self.board is None or self.Minv is None:
            raise RuntimeError("当前没有可快速清理的棋盘，请先分析")
        max_layers = max(1, min(3, int(max_layers or 3)))
        context.publish(
            phase="quick-exit",
            detail=f"正在扫描可直接离场的普通羊 · 最多 {max_layers} 层",
            progress={
                "steps": 0, "remaining": self.board.remaining_count(),
                "layer": 0, "max_layers": max_layers,
            },
            panel_solution=None,
            plan_completed_base=0,
        )

        def progress(_phase, data):
            layer = int(data.get("layer") or 0)
            context.publish(
                phase="quick-exit",
                detail=(f"正在收集第 {layer}/{max_layers} 层直出"
                        if layer else "正在扫描第一层直出"),
                progress=data,
            )

        try:
            planned = planner.coarse_exit_plan(
                self._clone_board(self.board),
                cancel=self._cancel_event.is_set,
                progress=progress,
                exit_priority=self._execution_exit_priority,
                max_layers=max_layers,
            )
        except planner.PlanningCancelled as exc:
            raise RuntimeError("快速解法已暂停") from exc

        solution = self._build_solution_payload(
            self._clone_board(self.board), planned.steps, planned.final_board,
            planned.solved, planned.remaining, planned.kind, 0.0, False,
            self.Minv.copy(), cache_info={"hit": False, "quick_exit": True},
            suspicion=None,
        )
        exit_layers = dict(planned.info.get("exit_layers") or {})
        for move in solution.get("moves") or []:
            move["exit_layer"] = int(exit_layers.get(str(move.get("piece"))) or 1)
        solution.update({
            "quick_exit": True,
            "exit_layers": int(planned.info.get("layers") or 0),
            "max_exit_layers": max_layers,
            "result_type": "quick_exit" if planned.steps else "quick_no_exit",
            "failure_reason": None,
        })
        context.publish(
            phase="quick-exit",
            detail=(f"快速解法已生成 · {solution['exit_layers']} 层 · "
                    f"{int(solution.get('total') or 0)} 只可直接离场"),
            progress={
                "completed": 0,
                "total": int(solution.get("total") or 0),
                "remaining": int(solution.get("remaining") or 0),
                "layer": int(solution.get("exit_layers") or 0),
                "max_layers": max_layers,
            },
            panel_solution=self._panel_solution_payload(solution),
            plan_completed_base=0,
        )
        return solution

    def _wolf_requires_live_control(self):
        """Return whether automatic execution must stay on the live-guard path."""
        motion = self._wolf_motion or {}
        return bool(
            (self.debug or {}).get("hazards")
            or motion.get("present")
            or motion.get("tracks")
            or self._wolf_danger_cells
        )

    def _bomb_requires_live_control(self):
        """Keep bomb hit budgets synchronized with the live game after every click."""
        if self.board is None:
            return False
        pieces = list(self.board.pieces.values()) + list(self.board.returning.values())
        return any(piece.get("species") == "bomb" for piece in pieces)

    @staticmethod
    def _report_notices(report):
        return (list((report or {}).get("execution_blockers") or [])
                + list((report or {}).get("advisories") or []))

    def _review_piece_ids(self):
        return {str(piece_id) for piece_id, meta in self._species_by_id.items()
                if meta.get("review")}

    def _execution_exit_priority(self, current, move):
        """Prefer every safe exit before a low-confidence piece."""
        risk = self._move_wolf_risk(current, move, self._wolf_track_cells())
        return (
            1 if str(move.piece_id) in self._review_piece_ids() else 0,
            0 if risk["risky"] else 1,
            move.anchor[0], move.anchor[1], str(move.piece_id),
        )

    def _review_pause_payload(self, piece_id, cells, reason=None, **extra):
        guidance = ExecutionReviewRequired(piece_id, cells, reason)
        report = self.scene_report or {}
        state = (self._snapshot(self.board, highlight=None)
                 if self.board is not None and self.Minv is not None else None)
        return {
            "ok": True,
            **guidance.payload,
            "execution_paused_for_review": True,
            "scene_state": report.get("scene_state", "gameplay"),
            "execution_blockers": list(report.get("execution_blockers") or []),
            "advisories": list(report.get("advisories") or []),
            "executable": bool(report.get("executable", True)),
            "board_revision": self.board_revision,
            **({"state": state} if state is not None else {}),
            **extra,
        }

    @staticmethod
    def _hard_execution_blockers(report):
        """Tutorial overlays may lower capture quality but never block a click."""
        return [item for item in (report or {}).get("execution_blockers") or []
                if item.get("code") != "gesture_occlusion"]

    @staticmethod
    def _execution_allowed(report):
        return bool((report or {}).get("scene_state") == "gameplay"
                    and not Api._hard_execution_blockers(report))

    def _wait_direct_settle(self, seconds):
        """Wait for the current click animation; cancellation stops the next click."""
        return self._cancel_event.wait(max(0.0, float(seconds)))

    def _execute_trusted_plan_direct(self, context, max_steps, settle_ms, hold_ms,
                                     solve_timeout_ms, stage="complete-plan",
                                     progress_offset=0, progress_total=None):
        """Execute one already-proven ordinary plan without per-step recapture.

        The board and the complete move sequence were authorized together by
        the initial analysis/solve.  We advance that model after each click and
        publish the predicted state immediately.  One final capture verifies
        the batch.  Wolf and bomb boards never enter this path.
        """
        if self._wolf_requires_live_control():
            raise RuntimeError("检测到狼，连续执行必须使用点击前动线确认")
        if self._bomb_requires_live_control():
            raise RuntimeError("检测到炸弹羊，连续执行必须逐步核对剩余耐久")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("已有执行任务正在进行")
        try:
            plan = self._active_plan
            opening_coarse = stage == "opening-coarse"
            quick_exit = stage == "quick-exit"
            safe_exit_batch = opening_coarse or quick_exit
            if not self._plan_is_execution_ready(plan, self.board_revision):
                raise RuntimeError(self._incomplete_plan_error("连续执行计划"))
            plan_scope = plan.get("execution_scope")
            if not safe_exit_batch and plan_scope != "complete":
                raise RuntimeError(self._incomplete_plan_error("连续执行计划"))
            planned_moves = list(plan.get("moves") or [])
            reorders = []
            gap_schedule = []
            gap_reasons = []
            if safe_exit_batch:
                cursor = self._clone_board(self.board)
                for planned in planned_moves:
                    live = self._match_planned_move(cursor, planned)
                    if live is None or not self._move_is_plain_exit(cursor, live):
                        raise RuntimeError("快速清理只允许普通羊直接离场")
                    cursor = cursor.apply(live)
                planned_moves, reorders = self._schedule_wait_avoiding_exits(
                    self.board, planned_moves, settle_ms,
                    limit=len(planned_moves), wolf_track=self._wolf_track_cells(),
                    review_ids=self._review_piece_ids())
                cursor = self._clone_board(self.board)
                motions = []
                for planned in planned_moves:
                    live = self._match_planned_move(cursor, planned)
                    if live is None:
                        raise RuntimeError("快速清理重排后动作失效")
                    motions.append(self._move_motion(cursor, live))
                    cursor = cursor.apply(live)
                gap_schedule, gap_reasons = self._burst_gap_schedule(
                    motions, settle_ms)
            requested_limit = min(max(1, int(max_steps)), len(planned_moves))
            progress_offset = max(0, int(progress_offset or 0))
            progress_total = max(
                progress_offset + requested_limit,
                int(progress_total or (progress_offset + requested_limit)),
            )
            review_stop = None
            cursor = self._clone_board(self.board)
            limit = requested_limit
            for index, planned in enumerate(planned_moves[:requested_limit]):
                live = self._match_planned_move(cursor, planned)
                if live is None:
                    break
                review_meta = self._species_by_id.get(str(live.piece_id), {})
                if review_meta.get("review"):
                    review_stop = (
                        live.piece_id,
                        cursor.pieces[str(live.piece_id)]["cells"],
                        review_meta.get("review_reason"),
                    )
                    limit = index
                    break
                cursor = cursor.apply(live)
            planned_suffix = planned_moves[limit:]
            records, clicked_items = [], []

            if review_stop is not None and limit == 0:
                return self._review_pause_payload(
                    *review_stop,
                    direct_execution=True,
                    direct_stage=stage,
                    opening_coarse=opening_coarse,
                    quick_exit=quick_exit,
                    steps_completed=0,
                )

            for index, planned in enumerate(planned_moves[:limit]):
                context.checkpoint()
                move = self._match_planned_move(self.board, planned)
                if move is None:
                    raise RuntimeError(f"连续执行第 {index + 1} 步与可信计划不一致")
                before = self._clone_board(self.board)
                piece = before.pieces[str(move.piece_id)]
                cells = sorted(piece["cells"])
                px = sum(self._cell_center(r, c)[0] for r, c in cells) / len(cells)
                py = sum(self._cell_center(r, c)[1] for r, c in cells) / len(cells)
                step_records = []
                clicked = self._click_image_point(
                    px, py, hold_ms,
                    before_click=lambda: step_records.append(
                        self._record_execution_step(before, move, mode="direct-plan")))
                records.extend(step_records)
                clicked_items.append({
                    "piece_id": str(move.piece_id),
                    "result": move.result,
                    "clicked": clicked,
                })

                # Commit the predicted result immediately after the irreversible
                # click.  Pause may prevent the next click, never this update.
                self.board = before.apply(move)
                data = self._board_data(self.board)
                self._detected_board_data = json.loads(json.dumps(data))
                self.board_revision = level_cache.board_hash(data)
                remaining_moves = planned_moves[index + 1:]
                self._active_plan = ({
                    "revision": self.board_revision,
                    "moves": remaining_moves,
                    "complete": plan_scope == "complete",
                    "execution_scope": plan_scope,
                    "created_at": plan.get("created_at", time.monotonic()),
                } if remaining_moves else None)
                preview_state = self._snapshot(self.board, highlight=None)
                context.publish(
                    phase=app_runtime.Phase.EXECUTING,
                    detail=(f"快速解法直出第 {index + 1}/{limit} 只"
                            if quick_exit else
                            f"开局粗解直出第 {index + 1}/{limit} 只"
                            if opening_coarse else
                            f"直接执行第 {index + 1}/{limit} 步"),
                    progress={
                        "completed": progress_offset + index + 1,
                        "total": progress_total,
                    },
                    preview_state=preview_state,
                    direct_execution=True,
                    opening_coarse=opening_coarse,
                    quick_exit=quick_exit,
                )
                # This path executes an already-authorized complete plan and
                # verifies only once at the end.  Honor the user's requested
                # inter-click gap literally; per-step live-control paths keep
                # their longer animation-aware verification floor below.
                base_gap_ms = max(20, min(3000, int(settle_ms or 60)))
                direct_gap_ms = (gap_schedule[index]
                                 if safe_exit_batch and index < len(gap_schedule)
                                 else base_gap_ms)
                if self._wait_direct_settle(direct_gap_ms / 1000.0):
                    return {
                        "ok": True,
                        "direct_execution": True,
                        "opening_coarse": opening_coarse,
                        "quick_exit": quick_exit,
                        "direct_stage": stage,
                        "steps_completed": index + 1,
                        "paused": True,
                        "state": preview_state,
                    }

            predicted_state = self._snapshot(self.board, highlight=None)
            context.publish(
                phase=app_runtime.Phase.VERIFYING,
                detail="连续计划已执行，正在进行最终核对",
                progress={
                    "completed": progress_offset + limit,
                    "total": progress_total,
                },
                preview_state=predicted_state,
                direct_execution=True,
            )
            rectinfo, mode = self._capture_live(require_same_window=True)
            post = self._analyze_frame(source="app-direct-final-refresh")
            verification = self._verification_feedback(
                predicted_state, post.get("state"), limit)
            if verification:
                verification = level_cache.save_feedback(
                    verification, capture_meta=self._last_cache,
                    level_key=self._level_key)
            continuation = None
            if (planned_suffix and verification and verification.get("matched")
                    and self.board is not None):
                continuation = self._continuation_solution(
                    self._clone_board(self.board), planned_suffix, self.Minv.copy())
            review_extra = (ExecutionReviewRequired(*review_stop).payload
                            if review_stop is not None else {})
            if review_stop is not None:
                review_extra["execution_paused_for_review"] = True
            return self._execution_refresh_payload(
                post, rectinfo, mode, solve_timeout_ms,
                solution_override=({} if quick_exit else continuation),
                clicked=(clicked_items[-1]["clicked"] if clicked_items else None),
                verification=verification,
                execution_records=records,
                direct_execution=True,
                direct_stage=stage,
                opening_coarse=opening_coarse,
                quick_exit=quick_exit,
                opening_coarse_steps=(len(clicked_items) if opening_coarse else 0),
                batch_size=len(clicked_items),
                batch_moves=[item["piece_id"] for item in clicked_items],
                batch_profile={
                    "stage": stage,
                    "interval_ms": max(20, min(3000, int(settle_ms or 60))),
                    "gap_schedule_ms": gap_schedule[:max(0, limit - 1)],
                    "gap_reasons": gap_reasons[:max(0, limit - 1)],
                    "reorders": reorders,
                },
                steps_completed=len(clicked_items),
                **review_extra,
            )
        finally:
            self._execution_lock.release()

    def workflow_start(self, action="analyze", options=None):
        """Start one complete user intent under the single runtime owner."""
        def run():
            opts = dict(options or {})
            requested = str(action or "analyze").strip().lower()
            if requested not in {"analyze", "upload", "solve", "quick", "step", "auto"}:
                raise RuntimeError(f"未知操作：{requested}")
            if requested in {"quick", "step", "auto"} and self._input_mode == "reference":
                raise RuntimeError("相册截图处于移动参考模式，不能执行桌面点击")
            timeout_s = max(1.0, min(60.0, float(opts.get("timeout_ms", 10000)) / 1000.0))
            elastic_timeout = bool(opts.get("elastic_timeout", True))
            extension_s = max(
                1.0, min(60.0, float(opts.get("timeout_extension_ms", 5000)) / 1000.0))
            max_timeout_s = max(
                timeout_s,
                min(300.0, float(opts.get("timeout_max_ms", 30000)) / 1000.0),
            )
            if not elastic_timeout:
                max_timeout_s = timeout_s
            solve_budget = {
                "elastic_timeout": elastic_timeout,
                "extension_s": extension_s,
                "max_timeout_s": max_timeout_s,
            }

            def worker(context):
                analysis = None
                if requested == "upload":
                    return self._workflow_upload(
                        context, opts.get("image_data"), opts.get("file_name"))
                capture_if_missing = bool(opts.get("capture_if_missing", True))
                needs_capture = (
                    requested == "analyze"
                    or (capture_if_missing and (
                        self.board is None
                        or self.Minv is None
                        or (requested == "quick"
                            and self.scene_report.get("scene_state") != "gameplay")))
                )
                if needs_capture:
                    analysis = self._workflow_capture(
                        context, opts.get("source_level_label"))
                if requested == "analyze":
                    return analysis
                if self.board is None or self.scene_report.get("scene_state") != "gameplay":
                    if requested == "quick" and analysis is not None:
                        return {**analysis, "ok": False,
                                "error": "最新采集画面不是可快速清理的棋盘"}
                    return analysis or {"ok": False,
                                        "error": "当前没有可求解棋盘，请先上传并分析截图"}

                if requested == "quick":
                    if not self._execution_allowed(self.scene_report):
                        blockers = self._hard_execution_blockers(self.scene_report)
                        message = (blockers[0].get("message") if blockers else None)
                        return {**(analysis or {}), "ok": False,
                                "error": message or "当前棋盘未通过安全检查"}
                    if (self._wolf_requires_live_control()
                            or self._bomb_requires_live_control()):
                        return {**(analysis or {}), "ok": False,
                                "error": "当前有狼或炸弹，请改用连续执行逐步核验"}
                    solution = self._workflow_quick_exit(context, max_layers=3)
                    if not self._solution_is_execution_ready(solution):
                        return {**(analysis or {}), "ok": True,
                                "solution": solution, "solution_history": True,
                                "quick_complete": True, "steps_completed": 0}
                    total = int(solution.get("execution_total") or 0)
                    context.publish(
                        phase=app_runtime.Phase.EXECUTING,
                        detail=f"快速解法正在清理 {total} 只直出羊",
                        progress={"completed": 0, "total": total},
                    )
                    result = self._execute_trusted_plan_direct(
                        context, total,
                        settle_ms=max(20, int(opts.get("settle_ms", 60))),
                        hold_ms=max(35, int(opts.get("hold_ms", 70))),
                        solve_timeout_ms=int(timeout_s * 1000),
                        stage="quick-exit")
                    result["solution"] = solution
                    result["solution_history"] = True
                    result["quick_complete"] = True
                    result["quick_exit_layers"] = int(solution.get("exit_layers") or 0)
                    return result

                if requested == "solve":
                    solution = self._workflow_solve(
                        context, timeout_s, **solve_budget)
                    return {"ok": True, "solution": solution,
                            "board_revision": self.board_revision}

                max_steps = 1 if requested == "step" else max(
                    1, min(500, int(opts.get("max_steps", 200))))
                completed = 0
                last = analysis
                if requested == "auto":
                    opening_pending = self._opening_coarse_pending
                    self._opening_coarse_pending = False
                    if (opening_pending
                            and self._execution_allowed(self.scene_report)
                            and not self._wolf_requires_live_control()
                            and not self._bomb_requires_live_control()):
                        coarse_solution = self._workflow_opening_coarse(context)
                        if coarse_solution and self._solution_is_execution_ready(coarse_solution):
                            coarse_limit = min(
                                max_steps,
                                int(coarse_solution.get("execution_total") or 0))
                            context.publish(
                                phase=app_runtime.Phase.EXECUTING,
                                detail=f"开局粗解已确认，正在快速直出 {coarse_limit} 只羊",
                                progress={"completed": 0, "total": coarse_limit},
                                opening_coarse=True,
                            )
                            last = self._execute_trusted_plan_direct(
                                context, coarse_limit,
                                settle_ms=max(20, int(opts.get("settle_ms", 60))),
                                hold_ms=max(35, int(opts.get("hold_ms", 70))),
                                solve_timeout_ms=int(timeout_s * 1000),
                                stage="opening-coarse")
                            completed = int(last.get("steps_completed") or 0)
                            last["opening_coarse_steps"] = completed
                            if last.get("review_required"):
                                return last
                            if not last.get("ok"):
                                return last
                            if (last.get("execution_complete")
                                    or last.get("scene_state") == "victory"
                                    or self.board is not None and self.board.is_solved()):
                                last["auto_complete"] = True
                                return last
                            context.checkpoint()
                            if completed >= max_steps:
                                last["limit_reached"] = True
                                return last
                    if not self._plan_is_execution_ready(
                            self._active_plan, self.board_revision):
                        solution = self._workflow_solve(
                            context, timeout_s, plan_completed_base=completed,
                            **solve_budget)
                        if not self._solution_is_execution_ready(solution):
                            return {"ok": False,
                                    "error": solution.get("execution_blocker")
                                             or self._incomplete_plan_error(),
                                    "solution": solution,
                                    "steps_completed": completed,
                                    "opening_coarse_steps": completed}
                    if (self._active_plan.get("execution_scope") == "complete"
                            and self._execution_allowed(self.scene_report)
                            and not self._wolf_requires_live_control()
                            and not self._bomb_requires_live_control()):
                        context.publish(
                            phase=app_runtime.Phase.EXECUTING,
                            detail="完整方案已确认，正在直接连续执行",
                            progress={"completed": completed,
                                      "total": min(
                                          max_steps,
                                          completed + len(self._active_plan["moves"]))})
                        direct = self._execute_trusted_plan_direct(
                            context, max_steps - completed,
                            settle_ms=max(20, int(opts.get("settle_ms", 60))),
                            hold_ms=max(35, int(opts.get("hold_ms", 70))),
                            solve_timeout_ms=int(timeout_s * 1000),
                            progress_offset=completed,
                            progress_total=min(
                                max_steps,
                                completed + len(self._active_plan["moves"])))
                        direct_steps = int(direct.get("steps_completed") or 0)
                        direct["steps_completed"] = completed + direct_steps
                        direct["opening_coarse_steps"] = completed
                        if (direct.get("execution_complete")
                                or direct.get("scene_state") == "victory"
                                or self.board is not None and self.board.is_solved()):
                            direct["auto_complete"] = True
                        elif direct["steps_completed"] >= max_steps:
                            direct["limit_reached"] = True
                        return direct
                for index in range(completed, max_steps):
                    context.checkpoint()
                    if not self._plan_is_execution_ready(self._active_plan, self.board_revision):
                        solution = self._workflow_solve(
                            context, timeout_s, plan_completed_base=index,
                            **solve_budget)
                        if not self._solution_is_execution_ready(solution):
                            return {"ok": False,
                                    "error": solution.get("execution_blocker")
                                             or self._incomplete_plan_error(),
                                    "solution": solution,
                                    "steps_completed": index}
                    context.publish(
                        phase=app_runtime.Phase.EXECUTING,
                        detail=f"正在执行第 {index + 1} 步",
                        progress={"completed": index, "total": max_steps})
                    last = self.execute_step(
                        self.board_revision, "0",
                        settle_ms=max(20, int(opts.get("settle_ms", 60))),
                        hold_ms=max(35, int(opts.get("hold_ms", 70))),
                        solve_timeout_ms=int(timeout_s * 1000))
                    if last.get("review_required"):
                        last["steps_completed"] = index
                        context.publish(
                            phase=app_runtime.Phase.DONE,
                            detail=last.get("review_message") or "请复核低置信度棋子",
                            progress={"completed": index, "total": max_steps},
                            preview_state=deepcopy(last.get("state")),
                        )
                        return last
                    if not last.get("ok"):
                        last["steps_completed"] = index
                        return last
                    context.publish(
                        phase=app_runtime.Phase.VERIFYING,
                        detail="已点击，正在核对新棋盘",
                        progress={"completed": index + 1, "total": max_steps},
                        preview_state=deepcopy(last.get("state")),
                        panel_analysis=self._panel_analysis_payload(last),
                        panel_solution=self._panel_solution_payload(last.get("solution")),
                        plan_completed_base=index + 1)
                    if (last.get("execution_complete")
                            or last.get("scene_state") == "victory"
                            or self.board is not None and self.board.is_solved()):
                        last["steps_completed"] = index + 1
                        last["auto_complete"] = True
                        return last
                    context.checkpoint()  # pause means stop after this click
                    if requested == "step":
                        last["steps_completed"] = 1
                        return last
                last = dict(last or {})
                last.update(ok=True, steps_completed=max_steps, limit_reached=True)
                return last

            job = self.runtime.start(requested, worker)
            return {"job": self._job_public(job["id"])}
        return _wrap(run)

    def workflow_status(self, job_id=None):
        return _wrap(lambda: {"job": self._job_public(job_id)})

    def workflow_cancel(self):
        return self.cancel()

    def _job_public(self, job_id=None):
        state = self.runtime.snapshot(job_id)
        phase = state.get("phase", "idle")
        state["status"] = ("running" if state.get("busy") else
                           "done" if phase == app_runtime.Phase.DONE.value else
                           "cancelled" if phase == app_runtime.Phase.CANCELLED.value else
                           "error" if phase == app_runtime.Phase.ERROR.value else phase)
        return state

    def _set_job(self, job_id, **fields):
        self.runtime.update(str(job_id), **fields)

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

    def _moves_to_records(self, steps):
        records = []
        for mv, phase in steps:
            records.append({
                "piece": str(mv.piece_id),
                "direction": mv.direction,
                "anchor": list(mv.anchor),
                "result": mv.result,
                "distance": int(mv.distance),
                "phase": phase,
            })
        return records

    def _solution_suspicion(self, steps, solved, remaining, refine_info=None,
                            final_board=None):
        if solved or remaining <= 0:
            return None
        info = refine_info or {}
        structural = list(info.get("structural_deadlocks") or [])
        if structural:
            pairs = [" ↔ ".join(map(str, item.get("pieces") or []))
                     for item in structural[:3]]
            suffix = f"（{'、'.join(pair for pair in pairs if pair)}）" if any(pairs) else ""
            return {
                "type": "structural_conflict",
                "message": f"检测到迎头相向的方向冲突{suffix}；请复核这些羊的位置和朝向后重算",
                "structural_deadlocks": structural,
            }
        if final_board is not None and not final_board.legal_moves():
            return {
                "type": "dead_end",
                "message": f"死局：剩余 {remaining} 只羊且没有任何合法移动，请修正位置或朝向后重算",
            }
        if info.get("timeout"):
            budget = info.get("budget") or {}
            if budget.get("elastic") and int(budget.get("extensions") or 0) > 0:
                total_s = max(1, round(int(budget.get("allocated_ms") or 0) / 1000))
                return {
                    "type": "timeout",
                    "message": f"弹性求解已自动续时至 {total_s} 秒，仍剩 {remaining} 只",
                }
            return {
                "type": "timeout",
                "message": f"搜索时间已用完，仍剩 {remaining} 只；可提高求解上限后重试",
            }
        if not steps:
            return {
                "type": "search_stalled",
                "message": "未找到可用步骤，但棋盘仍有合法移动；可增加求解时间后重试",
            }
        if info.get("loop"):
            return {
                "type": "loop",
                "message": "求解进入重复状态，疑似识别或朝向错误",
            }
        return None

    @staticmethod
    def _plan_is_execution_ready(plan, revision=None):
        """Only a complete plan or monotonic direct-exit prefix may click."""
        if not plan or not plan.get("moves"):
            return False
        scope = plan.get("execution_scope")
        if scope is None and plan.get("complete") is True:
            scope = "complete"
        if scope not in {"complete", "safe_exit_prefix"}:
            return False
        return revision is None or str(plan.get("revision")) == str(revision)

    @staticmethod
    def _solution_is_execution_ready(solution):
        return bool(
            solution
            and ((solution.get("execution_scope") == "complete"
                  and solution.get("execution_ready") is True)
                 or (solution.get("execution_scope") == "safe_exit_prefix"
                     and solution.get("safe_prefix_ready") is True))
            and solution.get("moves")
        )

    @staticmethod
    def _move_is_plain_exit(board, move):
        piece = board.pieces.get(str(move.piece_id)) if board is not None else None
        return bool(piece and piece.get("species", "sheep") == "sheep"
                    and move.result == "EXIT")

    @staticmethod
    def _incomplete_plan_error(prefix="当前方案"):
        return f"{prefix}未证明可以清空棋盘，仅供沙盘预览；已禁止自动点击"

    def _records_to_steps(self, board, records):
        steps = []
        cur = self._clone_board(board)
        for rec in records or []:
            pid = str(rec["piece"])
            if pid not in cur.pieces:
                raise RuntimeError(f"缓存动作失效：找不到羊 {pid}")
            mv = Move(pid, rec["direction"], tuple(rec["anchor"]),
                      rec["result"], int(rec.get("distance", 0)))
            legal = cur.legal_moves()
            if not any(str(m.piece_id) == str(mv.piece_id)
                       and m.direction == mv.direction
                       and m.result == mv.result
                       and m.distance == mv.distance
                       for m in legal):
                raise RuntimeError(f"缓存动作失效：{pid} {mv.direction} {mv.result}")
            steps.append((mv, rec.get("phase", "cached")))
            cur = cur.apply(mv)
        return steps, cur

    def _cached_solution_payload(self, board, timeout_s, Minv, job_id=None, started=None):
        board_data = self._board_data(board)
        cached = level_cache.load_solution(board_data, level_key=self._level_key, require_complete=True)
        if not cached:
            return None
        try:
            steps, final_board = self._records_to_steps(board, cached.get("moves", []))
        except Exception as e:
            _safe_error(e)
            return None
        elapsed_ms = int((time.monotonic() - started) * 1000)
        process_trace = [{
            "seq": 1, "phase": "cache-hit", "event": "finish",
            "at_ms": elapsed_ms, "elapsed_ms": elapsed_ms,
            "remaining": 0, "solved": True,
        }]
        if job_id:
            self._set_job(job_id, phase="cache-hit",
                          detail="已复用相同棋盘的完整解法",
                          solve_trace=deepcopy(process_trace),
                          elapsed_ms=elapsed_ms)
        remaining = final_board.remaining_count()
        payload = self._build_solution_payload(
            board, steps, final_board, remaining == 0, remaining,
            cached.get("kind", "cache"), timeout_s, False, Minv,
            cache_info={"hit": True, "path": cached.get("_cache_path"),
                        "capture_id": cached.get("capture_id")}
        )
        payload["kind"] = f"缓存命中 · {payload['kind']}"
        payload["result_type"] = "complete_solution"
        payload["process_trace"] = process_trace
        payload["elapsed_ms"] = elapsed_ms
        payload["budget"] = {
            "initial_ms": int(timeout_s * 1000), "extension_ms": 0,
            "max_ms": int(timeout_s * 1000), "allocated_ms": 0,
            "elapsed_ms": elapsed_ms, "extensions": 0, "elastic": False,
            "cache_hit": True,
        }
        return payload

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
        _write_json_atomic(os.path.join(HERE, "board.json"), bd)
        if layout is not None:
            _write_json_atomic(os.path.join(HERE, "board_layout.json"), layout)
            _write_json_atomic(os.path.join(HERE, "sheep_candidates.json"), candidates)

    def _rerender_detection_images(self):
        if not self.debug or self.sheep is None:
            return
        cv2.imwrite(str(image_path("_occ_axis_rect.png")), D.render_rect_debug(self.debug, self.sheep))
        cv2.imwrite(str(image_path("_grid_labels.png")), D.render_grid_labels(self.debug, self.sheep))
        cv2.imwrite(str(image_path("_layout.png")), D.render_layout(self.debug, self.sheep))

    def _solve_with_budget(self, board, timeout_s, Minv, job_id=None, started=None,
                           *, elastic_timeout=False, extension_s=5.0,
                           max_timeout_s=None):
        started = started or time.monotonic()
        if self._cancel_event.is_set():
            raise RuntimeError("执行已取消")
        cached_payload = (None if self._review_piece_ids() else
                          self._cached_solution_payload(
                              board, timeout_s, Minv,
                              job_id=job_id, started=started))
        if cached_payload:
            return cached_payload
        process_trace = []

        def progress(phase, data):
            details = deepcopy(data or {})
            entry = {
                "seq": len(process_trace) + 1,
                "phase": str(phase),
                "event": str(details.get("event") or "progress"),
                "at_ms": int((time.monotonic() - started) * 1000),
            }
            for key in (
                    "attempt", "remaining", "steps", "solved", "elapsed_ms",
                    "budget_ms", "expanded", "restarts", "depth", "added_ms",
                    "initial_ms", "extension_ms", "max_ms", "allocated_ms",
                    "extensions", "elastic"):
                if key in details:
                    entry[key] = deepcopy(details[key])
            if (process_trace and entry["event"] == "progress"
                    and process_trace[-1].get("event") == "progress"
                    and process_trace[-1].get("phase") == entry["phase"]):
                entry["seq"] = process_trace[-1]["seq"]
                process_trace[-1] = entry
            else:
                process_trace.append(entry)
                if len(process_trace) > 64:
                    process_trace.pop(1 if process_trace[0].get("phase") == "solve-budget" else 0)
            if job_id:
                self._set_job(
                    job_id, phase=phase,
                    detail={
                        "solve-budget": "正在分配求解预算",
                        "budget-extension": "当前预算已用完，正在自动续时",
                        "exit-closure": "正在清理可直出的羊",
                        "exact-a*": "正在验证最短解",
                        "macro-beam": "正在搜索移动顺序",
                        "randomized-macro": "正在尝试多路移动顺序",
                        "online-greedy": "正在生成快速候选",
                        "weighted-a*": "正在搜索解法",
                        "beam": "正在扩大搜索",
                        "greedy": "正在整理可用提示",
                    }.get(phase, "正在求解"),
                    progress=details,
                    solve_trace=deepcopy(process_trace),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )

        strategy_policy = solver_learning.policy_for(board)
        try:
            planned = planner.solve_board(
                board, timeout_s,
                cancel=self._cancel_event.is_set,
                progress=progress,
                exit_priority=self._execution_exit_priority,
                online_dynamic=bool(getattr(board, "no_stop", None)),
                elastic_timeout=elastic_timeout,
                extension_s=extension_s,
                max_timeout_s=max_timeout_s or timeout_s,
                strategy_policy=strategy_policy,
            )
        except planner.PlanningCancelled as exc:
            raise RuntimeError("执行已取消") from exc

        steps = planned.steps
        remaining = planned.remaining
        final_board = planned.final_board
        solved = planned.solved
        timed_out = planned.timed_out
        kind = planned.kind
        suspicion = self._solution_suspicion(
            steps, solved, remaining, planned.info, final_board=final_board)
        payload = self._build_solution_payload(board, steps, final_board, solved, remaining,
                                               kind, timeout_s, timed_out, Minv,
                                               cache_info={"hit": False},
                                               suspicion=suspicion)
        budget_info = deepcopy(planned.info.get("budget") or {})
        payload["process_trace"] = deepcopy(process_trace)
        payload["budget"] = budget_info
        payload["initial_timeout_ms"] = int(
            budget_info.get("initial_ms", timeout_s * 1000))
        payload["max_timeout_ms"] = int(
            budget_info.get("max_ms", payload["initial_timeout_ms"]))
        payload["timeout_ms"] = int(
            budget_info.get("allocated_ms", payload["initial_timeout_ms"]))
        payload["elapsed_ms"] = int(
            budget_info.get("elapsed_ms", (time.monotonic() - started) * 1000))
        result_type = ("complete_solution" if solved else
                       "structural_conflict" if suspicion and suspicion.get("type") == "structural_conflict" else
                       "dead_end" if suspicion and suspicion.get("type") == "dead_end" else
                       "loop" if suspicion and suspicion.get("type") == "loop" else
                       "timeout" if timed_out else
                       "partial_hint" if steps else "proven_unsat")
        payload["result_type"] = result_type
        payload["failure_reason"] = ((suspicion or {}).get("message")
                                     or planned.info.get("reason"))
        payload["search_info"] = {
            key: deepcopy(planned.info[key])
            for key in ("kind", "reason", "expanded", "restarts",
                        "structural_deadlocks", "budget")
            if key in planned.info
        }
        saved = level_cache.save_solution(
            self._board_data(board),
            {
                "kind": kind,
                "solved": bool(solved),
                "remaining": int(remaining),
                "result_type": result_type,
                "timeout": bool(timed_out),
                "timeout_ms": int(payload.get("timeout_ms") or timeout_s * 1000),
                "initial_timeout_ms": int(payload.get("initial_timeout_ms") or timeout_s * 1000),
                "max_timeout_ms": int(payload.get("max_timeout_ms") or timeout_s * 1000),
                "budget": budget_info,
                # Partial hints remain useful for preview/debugging, but must
                # never become a cache-backed execution authority.
                "usable": bool(solved) and not bool(suspicion),
                "suspicious": bool(suspicion),
                "suspicion": suspicion,
                "coarse_total": sum(1 for _mv, phase in steps if phase == "coarse"),
                "refine_total": sum(1 for _mv, phase in steps if phase != "coarse"),
                "moves": self._moves_to_records(steps),
            },
            level_key=self._level_key,
            capture_meta=self._last_cache,
            source="app-solver",
        )
        payload["cache"] = {
            "hit": False,
            "stored": True,
            "path": saved.get("best_path"),
            "revision_id": saved.get("revision_id"),
            "selected_best": saved.get("selected_best"),
            "capture_id": saved.get("capture_id"),
        }
        # The learner owns its daemon writer.  This enqueue is intentionally
        # best-effort and cannot delay or invalidate the solve result.
        solver_learning.record_async(
            board, process_trace, solved=solved, remaining=remaining)
        return payload

    def _build_solution_payload(self, start_board, steps, final_board, solved, remaining,
                                kind, timeout_s, timed_out, Minv, cache_info=None, suspicion=None):
        revision = level_cache.board_hash(self._board_data(start_board))
        steps = list(steps)
        wolf_track = self._wolf_track_cells()
        if wolf_track and steps:
            prefix = []
            prefix_phases = {}
            cursor = self._clone_board(start_board)
            for move, phase in steps:
                live = self._match_planned_move(cursor, move)
                if live is None or not self._move_is_plain_exit(cursor, live):
                    break
                prefix.append(live)
                prefix_phases[live] = phase
                cursor = cursor.apply(live)
            if prefix:
                scheduled, _reorders = self._schedule_wait_avoiding_exits(
                    start_board, prefix, 35, limit=len(prefix), wolf_track=wolf_track,
                    review_ids=self._review_piece_ids())
                steps = ([(move, prefix_phases[move]) for move in scheduled]
                         + steps[len(prefix):])
        cur = start_board
        states, out_moves = [], []
        for i, (mv, phase) in enumerate(steps, 1):
            states.append(self._snapshot(
                cur, highlight=mv.piece_id, Minv=Minv,
                live_wolf_annotations=(i == 1)))
            r, c = mv.anchor
            piece = cur.pieces[str(mv.piece_id)]
            wolf_risk = self._move_wolf_risk(cur, mv, wolf_track)
            cells = sorted(piece["cells"])
            tap_x = sum(self._cell_center(rr, cc, Minv=Minv)[0] for rr, cc in cells) / len(cells)
            tap_y = sum(self._cell_center(rr, cc, Minv=Minv)[1] for rr, cc in cells) / len(cells)
            next_board = cur.apply(mv)
            bomb_changes = []
            for bomb_id, before_bomb in cur.pieces.items():
                if before_bomb.get("species") != "bomb":
                    continue
                after_bomb = next_board.pieces.get(str(bomb_id))
                before_hits = int(before_bomb.get("hits_remaining") or
                                  before_bomb.get("hit_limit") or 3)
                if after_bomb is None:
                    bomb_changes.append({
                        "piece": str(bomb_id), "before": before_hits,
                        "after": None, "event": "exit",
                    })
                    continue
                after_hits = int(after_bomb.get("hits_remaining") or
                                 after_bomb.get("hit_limit") or 3)
                if after_hits != before_hits:
                    bomb_changes.append({
                        "piece": str(bomb_id), "before": before_hits,
                        "after": after_hits, "event": "hit",
                    })
            out_moves.append({"desc": board_io.describe(mv),
                              "anchor": self._cell_center(r, c, Minv=Minv),
                              "cell": [int(r), int(c)],
                              "tap": [tap_x, tap_y],
                              "piece": mv.piece_id,
                              "direction": mv.direction,
                              "result": mv.result,
                              "distance": mv.distance,
                              "phase": phase,
                              "wolf_risk": wolf_risk["risky"],
                              "wolf_overlap": wolf_risk["overlap"],
                              "species": piece.get("species", "sheep"),
                              "bomb_changes": bomb_changes,
                              "move_id": str(i - 1),
                              "step": i})
            cur = next_board
        states.append(self._snapshot(
            final_board, highlight=None, Minv=Minv,
            live_wolf_annotations=False))
        coarse_total = sum(1 for _mv, phase in steps if phase == "coarse")
        safe_exit_prefix = []
        safe_cursor = start_board
        for move, phase in steps:
            if phase != "coarse" or not self._move_is_plain_exit(safe_cursor, move):
                break
            safe_exit_prefix.append(move)
            safe_cursor = safe_cursor.apply(move)
        if bool(solved) and int(remaining) == 0 and steps:
            execution_scope = "complete"
            execution_moves = [move for move, _phase in steps]
        elif safe_exit_prefix and not suspicion:
            execution_scope = "safe_exit_prefix"
            execution_moves = safe_exit_prefix
        else:
            execution_scope = "preview"
            execution_moves = []
        execution_ready = execution_scope == "complete" and bool(execution_moves)
        safe_prefix_ready = execution_scope == "safe_exit_prefix" and bool(execution_moves)
        authorized = execution_ready or safe_prefix_ready
        start_bombs = [piece for piece in start_board.pieces.values()
                       if piece.get("species") == "bomb"]
        payload = {"total": len(steps), "solved": bool(solved), "remaining": int(remaining),
                "kind": kind, "timeout": bool(timed_out), "timeout_ms": int(timeout_s * 1000),
                "suspicious": bool(suspicion), "suspicion": suspicion,
                "execution_ready": execution_ready,
                "safe_prefix_ready": safe_prefix_ready,
                "execution_scope": execution_scope,
                "execution_total": len(execution_moves),
                "execution_blocker": None if authorized else self._incomplete_plan_error(),
                "coarse_total": coarse_total, "refine_total": len(steps) - coarse_total,
                "cache": cache_info or {"hit": False},
                "board_revision": revision,
                "scene_state": self.scene_report.get("scene_state", "unknown"),
                "execution_complete": bool(self.scene_report.get("execution_complete", False)),
                "execution_blockers": list(self.scene_report.get("execution_blockers") or []),
                "executable": bool(self.scene_report.get("executable")),
                "wolf_motion": deepcopy(self._wolf_motion),
                "wolf_track": [list(cell) for cell in sorted(wolf_track)],
                "wolf_zone": [list(cell) for cell in sorted(self._wolf_confirmed_cells)],
                "wolf_risk_total": sum(1 for item in out_moves if item.get("wolf_risk")),
                "bomb_count": len(start_bombs),
                "bomb_min_hits": min((int(piece.get("hits_remaining") or
                                           piece.get("hit_limit") or 3)
                                      for piece in start_bombs), default=None),
                "bomb_live_control": bool(start_bombs),
                "rows": start_board.rows, "cols": start_board.cols,
                "moves": out_moves, "states": states}
        if revision == self.board_revision and authorized:
            self._active_plan = {
                "revision": revision,
                "moves": execution_moves,
                "complete": execution_scope == "complete",
                "execution_scope": execution_scope,
                "created_at": time.monotonic(),
            }
        elif (revision == self.board_revision and self._active_plan
              and str(self._active_plan.get("revision")) == str(revision)):
            self._active_plan = None
        return payload

    @staticmethod
    def _match_planned_move(board, planned):
        """Rebind a plan move after live recognition renumbers piece ids."""
        legal = board.legal_moves()
        exact = next((move for move in legal if move == planned), None)
        if exact is not None:
            return exact
        candidates = [
            move for move in legal
            if tuple(move.anchor) == tuple(planned.anchor)
            and move.direction == planned.direction
            and move.result == planned.result
            and int(move.distance) == int(planned.distance)
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _continuation_solution(self, board, moves, Minv):
        """Rebase an already planned suffix onto a predicted/verified board cheaply."""
        cur = self._clone_board(board)
        steps = []
        planned_moves = list(moves or [])
        continuation_phase = ("coarse" if planned_moves
                              and all(move.result == "EXIT" for move in planned_moves)
                              else "continuation")
        for planned in planned_moves:
            move = self._match_planned_move(cur, planned)
            if move is None:
                break
            steps.append((move, continuation_phase))
            cur = cur.apply(move)
        return self._build_solution_payload(
            board, steps, cur, cur.is_solved(), cur.remaining_count(),
            "沿用已验证计划", 0.0, False, Minv,
            cache_info={"hit": True, "continuation": True}, suspicion=None)

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

    # ---- 4) 真实点击：只接受 board_revision + move_id ----
    def cancel(self):
        def run():
            self._cancel_event.set()
            job = self.runtime.cancel()
            return {"cancelled": True, "job": job,
                    "message": "当前点击完成后暂停"}
        return _wrap(run)

    def click(self, *_args):
        """Legacy raw-pixel API is intentionally disarmed."""
        return {"ok": False, "error": "裸像素点击协议已禁用，请使用 execute_step(board_revision, move_id)"}

    def _capture_live(self, *, require_same_window=True):
        hwnd = self._target_window()
        if not hwnd:
            raise RuntimeError(f"找不到窗口「{TITLE}」")
        if require_same_window and self.hwnd and int(hwnd) != int(self.hwnd):
            raise RuntimeError("目标窗口句柄已变化，已阻止执行；请重新采集确认")
        img, rectinfo, mode = grab(hwnd)
        self.hwnd, self.game, self.win = hwnd, img, rectinfo
        self._input_mode = "operator"
        cv2.imwrite(str(image_path("_game.png")), img)
        return rectinfo, mode

    def _wait_or_cancel(self, seconds):
        if self._cancel_event.wait(max(0.0, float(seconds))):
            raise RuntimeError("执行已取消")

    def _focus_target_window(self, hwnd):
        """Bring the verified game HWND forward, then prove it owns foreground."""
        GA_ROOT = 2
        SWP_NOSIZE, SWP_NOMOVE, SWP_SHOWWINDOW = 0x0001, 0x0002, 0x0040
        target_root = user32.GetAncestor(hwnd, GA_ROOT) or hwnd

        def owns_foreground():
            foreground = user32.GetForegroundWindow()
            foreground_root = user32.GetAncestor(foreground, GA_ROOT) or foreground
            return bool(foreground_root and foreground_root == target_root)

        if owns_foreground():
            return True
        current_tid = int(kernel32.GetCurrentThreadId())
        for attempt in range(3):
            foreground = user32.GetForegroundWindow()
            attached = []
            for window in (foreground, target_root):
                if not window:
                    continue
                tid = int(user32.GetWindowThreadProcessId(window, None))
                if tid and tid != current_tid and tid not in attached:
                    if user32.AttachThreadInput(current_tid, tid, True):
                        attached.append(tid)
            try:
                user32.ShowWindow(target_root, 9)  # SW_RESTORE
                user32.SetWindowPos(
                    target_root, 0, 0, 0, 0, 0,
                    SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW)
                user32.BringWindowToTop(target_root)
                user32.SetForegroundWindow(target_root)
                user32.SetActiveWindow(target_root)
                user32.SetFocus(target_root)
            finally:
                for tid in reversed(attached):
                    user32.AttachThreadInput(current_tid, tid, False)
            self._wait_or_cancel(.08 + attempt * .06)
            if owns_foreground():
                return True
            # Windows may deny the first SetForegroundWindow call when the RPC
            # runs on pywebview's worker thread.  This fallback still targets
            # the already-validated HWND and is followed by the same proof.
            if attempt == 1 and hasattr(user32, "SwitchToThisWindow"):
                user32.SwitchToThisWindow(target_root, True)
        return owns_foreground()

    def _click_image_point(self, px, py, hold_ms=70, before_click=None):
        if self._cancel_event.is_set():
            raise RuntimeError("执行已取消")
        hwnd = self.hwnd
        if not hwnd or not user32.IsWindow(hwnd):
            raise RuntimeError("目标游戏窗口已失效")
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise RuntimeError("无法读取目标窗口位置")
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if self.win and (abs(width - self.win[2]) > 2 or abs(height - self.win[3]) > 2):
            raise RuntimeError("点击前窗口尺寸发生变化，已阻止执行")
        polygon = np.asarray([
            self._px(0, 0), self._px(self.cols, 0),
            self._px(self.cols, self.rows), self._px(0, self.rows),
        ], dtype=np.float32)
        if cv2.pointPolygonTest(polygon, (float(px), float(py)), False) < 0:
            raise RuntimeError("计算出的点击点不在棋盘多边形内")
        sx, sy = int(round(rect.left + float(px))), int(round(rect.top + float(py)))
        if not (rect.left <= sx < rect.right and rect.top <= sy < rect.bottom):
            raise RuntimeError("计算出的点击点不在目标窗口内")

        old = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(old))
        pressed = False
        try:
            if not self._focus_target_window(hwnd):
                raise RuntimeError("目标游戏窗口未处于前台，已阻止执行")
            user32.SetCursorPos(sx, sy)
            self._wait_or_cancel(0.04)
            if before_click is not None:
                before_click()
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            pressed = True
            self._wait_or_cancel(max(0.03, min(0.25, float(hold_ms) / 1000.0)))
        finally:
            if pressed:
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            user32.SetCursorPos(old.x, old.y)
        return {"at": [sx, sy], "hwnd": int(hwnd)}

    def _click_window_ratio(self, rx, ry, hold_ms=70):
        """Click a verified chrome/menu control outside the calibrated board."""
        if self._cancel_event.is_set():
            raise RuntimeError("执行已取消")
        hwnd = self.hwnd or self._target_window()
        if not hwnd or not user32.IsWindow(hwnd):
            raise RuntimeError("目标游戏窗口已失效")
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise RuntimeError("无法读取目标窗口位置")
        width, height = rect.right - rect.left, rect.bottom - rect.top
        rx, ry = float(rx), float(ry)
        if not (0.0 < rx < 1.0 and 0.0 < ry < 1.0):
            raise RuntimeError("重试点击比例超出窗口")
        sx = int(round(rect.left + width * rx))
        sy = int(round(rect.top + height * ry))
        old = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(old))
        pressed = False
        try:
            if not self._focus_target_window(hwnd):
                raise RuntimeError("目标游戏窗口未处于前台，已阻止重试")
            user32.SetCursorPos(sx, sy)
            self._wait_or_cancel(0.04)
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            pressed = True
            self._wait_or_cancel(max(0.03, min(0.25, float(hold_ms) / 1000.0)))
        finally:
            if pressed:
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            user32.SetCursorPos(old.x, old.y)
        return {"at": [sx, sy], "ratio": [rx, ry], "hwnd": int(hwnd)}

    @staticmethod
    def _retry_control_score(img, region, color):
        """Return the saturated red/blue pixel ratio in a normalized ROI."""
        if img is None or getattr(img, "size", 0) == 0:
            return 0.0
        h, w = img.shape[:2]
        x1, y1, x2, y2 = region
        roi = img[int(h * y1):int(h * y2), int(w * x1):int(w * x2)]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if color == "red":
            mask = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
                    & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 120))
        elif color == "green":
            mask = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90)
                    & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 100))
        else:
            mask = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 135)
                    & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 100))
        return float(mask.mean())

    def retry_level(self, mode="failure"):
        """Restart a failed or still-running level, then return a fresh observation."""
        def run():
            if not self._execution_lock.acquire(blocking=False):
                raise RuntimeError("上一执行阶段尚未结束")
            try:
                self._cancel_event.clear()
                mode_name = str(mode or "failure")
                if mode_name not in {"failure", "in_level"}:
                    raise RuntimeError("未知的自动重试类型")
                with open(RETRY_CONTROLS_PATH, encoding="utf-8") as stream:
                    retry_config = json.load(stream)
                controls = retry_config.get("controls") or {}
                timing = retry_config.get("timing_ms") or {}

                def control(name):
                    item = controls.get(name)
                    if not isinstance(item, dict) or not item.get("point"):
                        raise RuntimeError(f"自动重试经验缺少控件：{name}")
                    return item

                def score(name):
                    item = control(name)
                    return self._retry_control_score(
                        self.game, tuple(item.get("region") or (0, 0, 0, 0)),
                        str(item.get("color") or "blue"))

                def click(name):
                    point = control(name)["point"]
                    return self._click_window_ratio(point[0], point[1])

                def minimum(name):
                    return float(control(name).get("minimum_score", .07))

                actions = []
                self._capture_live(require_same_window=False)
                if mode_name == "failure":
                    close_score = score("failure_popup_close")
                    if close_score >= minimum("failure_popup_close"):
                        actions.append({"action": "close_failure_popup",
                                        **click("failure_popup_close")})
                        self._wait_or_cancel(float(timing.get("after_close", 650)) / 1000.0)
                        self._capture_live(require_same_window=False)
                    failure_score = score("failure_restart")
                    settings_score = score("settings_restart")
                    if max(failure_score, settings_score) < min(
                            minimum("failure_restart"), minimum("settings_restart")):
                        raise RuntimeError("未识别到失败页右下角的蓝色重新开始按钮")
                    if settings_score > failure_score * 1.25:
                        actions.append({"action": "restart_after_failure_settings",
                                        **click("settings_restart")})
                    else:
                        actions.append({"action": "restart_after_failure",
                                        **click("failure_restart")})
                else:
                    restart_score = score("settings_restart")
                    if restart_score < minimum("settings_restart"):
                        actions.append({"action": "open_settings",
                                        **click("settings_open")})
                        self._wait_or_cancel(float(timing.get("after_open_settings", 450)) / 1000.0)
                        self._capture_live(require_same_window=False)
                        restart_score = score("settings_restart")
                    if restart_score < minimum("settings_restart"):
                        raise RuntimeError("未识别到设置页右下角的蓝色重新开始按钮")
                    actions.append({"action": "restart_in_level",
                                    **click("settings_restart")})

                self._wait_or_cancel(float(timing.get("after_restart", 1200)) / 1000.0)
                result = None
                rectinfo = None
                capture_mode = None
                stabilize_attempts = max(1, min(20, int(timing.get("stabilize_attempts", 8))))
                stabilize_interval = float(timing.get("stabilize_interval", 350)) / 1000.0
                for index in range(stabilize_attempts):
                    rectinfo, capture_mode = self._capture_live(require_same_window=False)
                    result = self._analyze_frame(source=f"app-auto-retry-{mode_name}")
                    if result.get("scene_state") == "gameplay":
                        break
                    if index + 1 < stabilize_attempts:
                        self._wait_or_cancel(stabilize_interval)
                if not result or result.get("scene_state") != "gameplay":
                    raise RuntimeError("重新开始后棋盘尚未恢复")
                ok, buf = cv2.imencode(".png", self.game)
                if not ok:
                    raise RuntimeError("截图编码失败")
                return {
                    **result,
                    "capture": {"mode": capture_mode, "win": {
                        "w": rectinfo[2], "h": rectinfo[3]}},
                    "img": base64.b64encode(buf.tobytes()).decode("ascii"),
                    "retry_mode": mode_name,
                    "retry_actions": actions,
                }
            finally:
                self._execution_lock.release()
        return _wrap(run)

    def advance_level(self, solve_timeout_ms=10000, next_source_level_label=None):
        """Click the learned green next-level control after a verified clear."""
        def run():
            if not self._execution_lock.acquire(blocking=False):
                raise RuntimeError("上一执行阶段尚未结束")
            try:
                self._cancel_event.clear()
                with open(RETRY_CONTROLS_PATH, encoding="utf-8") as stream:
                    config = json.load(stream)
                control = (config.get("controls") or {}).get("victory_next") or {}
                timing = config.get("timing_ms") or {}
                if not control.get("point") or not control.get("region"):
                    raise RuntimeError("控件经验中缺少下一关绿色按钮")

                self._capture_live(require_same_window=True)
                victory = self._analyze_frame(source="app-victory-next-preflight")
                if (victory.get("scene_state") != "victory"
                        or victory.get("execution_complete") is not True):
                    raise RuntimeError("当前不是已确认的过关页面，拒绝点击下一关")
                score = self._retry_control_score(
                    self.game, tuple(control["region"]),
                    str(control.get("color") or "green"))
                minimum = float(control.get("minimum_score", .35))
                if score < minimum:
                    raise RuntimeError(
                        f"过关页未识别到绿色下一关按钮（{score:.3f} < {minimum:.3f}）")
                point = control["point"]
                clicked = self._click_window_ratio(point[0], point[1])
                self._wait_or_cancel(float(timing.get("after_next", 900)) / 1000.0)
                self._opening_coarse_pending = True
                self._wolf_observations.clear()
                self._frame_history.clear()
                self._wolf_motion = None
                self._wolf_danger_cells.clear()
                self._wolf_confirmed_cells.clear()

                if next_source_level_label is not None:
                    label = " ".join(str(next_source_level_label).strip().split())
                    if not label:
                        raise RuntimeError("下一关编号或名称不能为空")
                    self._source_level_label = label
                    self._level_key = level_cache.source_level_key(label)

                result = None
                rectinfo = None
                capture_mode = None
                attempts = max(1, min(20, int(timing.get("stabilize_attempts", 8))))
                interval = float(timing.get("stabilize_interval", 350)) / 1000.0
                level = None
                for index in range(attempts):
                    rectinfo, capture_mode = self._capture_live(require_same_window=True)
                    level = self._read_source_level(
                        next_source_level_label or self._source_level_label,
                        allow_missing=True)
                    result = self._analyze_frame(source="app-next-level-stabilize")
                    if result.get("scene_state") == "gameplay" and result.get("executable"):
                        break
                    if index + 1 < attempts:
                        self._wait_or_cancel(interval)
                if not result or result.get("scene_state") != "gameplay":
                    raise RuntimeError("点击下一关后，新棋盘尚未稳定")

                solution = None
                if self.board is not None and result.get("executable"):
                    timeout_s = max(1.0, min(60.0, float(solve_timeout_ms) / 1000.0))
                    solution = self._solve_with_budget(
                        self._clone_board(self.board), timeout_s, self.Minv.copy())
                ok, buf = cv2.imencode(".png", self.game)
                if not ok:
                    raise RuntimeError("截图编码失败")
                return {
                    **result,
                    **(level or {}),
                    "capture": {"mode": capture_mode, "win": {
                        "w": rectinfo[2], "h": rectinfo[3]}},
                    "img": base64.b64encode(buf.tobytes()).decode("ascii"),
                    "solution": solution,
                    "next_level_clicked": clicked,
                    "next_level_green_score": round(score, 4),
                }
            finally:
                self._execution_lock.release()
        return _wrap(run)

    @staticmethod
    def _wolf_motion_summary(observations, rows, cols):
        """Infer wolf patrol lanes from explicit tracks and consecutive frames."""
        recent = list(observations or [])[-8:]
        if not recent:
            return None
        track_cells = set()
        current_cells = {
            tuple(cell) for cell in (recent[-1].get("hazards") or [])
        }
        tracks = []
        center_frames = []
        for observation in recent:
            centers = []
            for component in ((observation.get("wolf") or {}).get("components") or []):
                explicit = [tuple(cell) for cell in component.get("track") or []]
                if explicit:
                    track_cells.update(explicit)
                    tracks.append({
                        "kind": component.get("kind", "runner"),
                        "axis": component.get("axis"),
                        "direction": None,
                        "cells": [list(cell) for cell in explicit],
                        "observed": True,
                    })
                center = component.get("center_rect")
                if isinstance(center, (list, tuple)) and len(center) == 2:
                    centers.append({
                        "kind": str(component.get("kind") or "wolf"),
                        "x": float(center[0]), "y": float(center[1]),
                    })
            center_frames.append(centers)

        # Associate detections frame-to-frame, not merely first-to-last.  With
        # two identical dark wolves, matching by component kind alone can swap
        # identities and manufacture a lane between different animals.
        paths = []
        active = []
        max_link = D.CELL * 2.25
        for frame_index, centers in enumerate(center_frames):
            candidates = []
            for path_index in active:
                prior = paths[path_index]["points"][-1]
                for center_index, center in enumerate(centers):
                    if paths[path_index]["kind"] != center["kind"]:
                        continue
                    distance = float(np.hypot(center["x"] - prior["x"],
                                              center["y"] - prior["y"]))
                    if distance <= max_link:
                        candidates.append((distance, path_index, center_index))
            used_paths, used_centers, next_active = set(), set(), []
            for _distance, path_index, center_index in sorted(candidates):
                if path_index in used_paths or center_index in used_centers:
                    continue
                paths[path_index]["points"].append({
                    **centers[center_index], "frame": frame_index,
                })
                used_paths.add(path_index)
                used_centers.add(center_index)
                next_active.append(path_index)
            for center_index, center in enumerate(centers):
                if center_index in used_centers:
                    continue
                paths.append({
                    "kind": center["kind"],
                    "points": [{**center, "frame": frame_index}],
                })
                next_active.append(len(paths) - 1)
            active = next_active

        min_motion = max(14.0, D.CELL * 0.35)
        max_lane_drift = D.CELL * 0.55
        for path in paths:
            points = path["points"]
            if len(points) < 3:
                continue
            xs = [item["x"] for item in points]
            ys = [item["y"] for item in points]
            x_span, y_span = max(xs) - min(xs), max(ys) - min(ys)
            dominant, cross = (x_span, y_span) if x_span >= y_span else (y_span, x_span)
            if (dominant < min_motion or cross > max_lane_drift
                    or dominant < max(1.8 * cross, min_motion)):
                continue
            axis = "H" if x_span >= y_span else "V"
            if axis == "H":
                lane = max(0, min(rows - 1, int(round(
                    float(np.median(ys)) / D.CELL - 0.5))))
                cells = [(lane, col) for col in range(cols)]
                delta = points[-1]["x"] - points[-2]["x"]
                direction = "R" if delta >= 0 else "L"
            else:
                lane = max(0, min(cols - 1, int(round(
                    float(np.median(xs)) / D.CELL - 0.5))))
                cells = [(row, lane) for row in range(rows)]
                delta = points[-1]["y"] - points[-2]["y"]
                direction = "D" if delta >= 0 else "U"
            track_cells.update(cells)
            tracks.append({
                "kind": path["kind"], "axis": axis, "direction": direction,
                "cells": [list(cell) for cell in cells],
                "center_rect": [round(float(points[-1]["x"]), 2),
                                round(float(points[-1]["y"]), 2)],
                "distance_px": round(float(dominant), 2),
                "samples": len(points), "observed": True,
            })
        present = any((item.get("wolf") or {}).get("components") for item in recent)
        return {
            "present": bool(present),
            "observed": bool(tracks),
            "sample_count": len(recent),
            "tracks": tracks,
            "track_cells": [list(cell) for cell in sorted(track_cells)],
            "current_cells": [list(cell) for cell in sorted(current_cells)],
        }

    @staticmethod
    def _wolf_patrol_zone(pieces, motion, rows, cols):
        """Flood each confirmed wolf footprint along its observed movement axis.

        Sheep cells are walls.  Flooding footprint placements instead of bare
        cells prevents a two-cell-wide wolf from leaking through one-cell gaps,
        while unioning every reachable footprint fills the empty middle of the
        patrol corridor.
        """
        if not motion:
            return set()
        current = {tuple(cell) for cell in motion.get("current_cells") or []}
        occupied = {
            tuple(cell) for piece in (pieces or []) for cell in piece.get("cells") or []
        }
        pending = set(current)
        groups = []
        while pending:
            seed = pending.pop()
            group, queue = {seed}, [seed]
            while queue:
                r, c = queue.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        neighbour = (r + dr, c + dc)
                        if neighbour in pending:
                            pending.remove(neighbour)
                            group.add(neighbour)
                            queue.append(neighbour)
            groups.append(group)

        danger = set(current)
        unused = set(range(len(groups)))
        for track in motion.get("tracks") or []:
            if not track.get("observed"):
                continue
            center = track.get("center_rect") or []
            if len(center) != 2:
                danger.update(tuple(cell) for cell in track.get("cells") or [])
                continue
            if not unused:
                # No current body component is available to size and anchor
                # the corridor. Keep the previously confirmed zone unchanged;
                # never fall back to the motion summary's full row/column.
                continue
            target = (float(center[1]) / D.CELL - 0.5,
                      float(center[0]) / D.CELL - 0.5)
            group_index = min(
                unused,
                key=lambda index: min(
                    (r - target[0]) ** 2 + (c - target[1]) ** 2
                    for r, c in groups[index]),
            )
            unused.remove(group_index)
            group = groups[group_index]
            min_r, max_r = min(r for r, _c in group), max(r for r, _c in group)
            min_c, max_c = min(c for _r, c in group), max(c for _r, c in group)
            height, width = max_r - min_r + 1, max_c - min_c + 1
            rectangle_offsets = {
                (dr, dc) for dr in range(height) for dc in range(width)
            }
            exact_offsets = {(r - min_r, c - min_c) for r, c in group}

            def footprint(anchor, offsets):
                ar, ac = anchor
                return {(ar + dr, ac + dc) for dr, dc in offsets}

            def fits(anchor, offsets):
                cells = footprint(anchor, offsets)
                return (all(0 <= r < rows and 0 <= c < cols for r, c in cells)
                        and not cells & occupied)

            offsets = rectangle_offsets
            start = (min_r, min_c)
            if not fits(start, offsets):
                offsets = exact_offsets
            axis = str(track.get("axis") or "")
            deltas = ((0, -1), (0, 1)) if axis == "H" else ((-1, 0), (1, 0))
            anchors, queue = set(), [start]
            while queue:
                anchor = queue.pop()
                if anchor in anchors or not fits(anchor, offsets):
                    continue
                anchors.add(anchor)
                queue.extend((anchor[0] + dr, anchor[1] + dc) for dr, dc in deltas)
            zone = set().union(*(footprint(anchor, offsets) for anchor in anchors)) if anchors else set(group)
            danger.update(zone)
            if axis == "H" and zone:
                lane = max(0, min(rows - 1, int(round(target[0]))))
                path_cells = [(lane, c) for c in range(min(c for _r, c in zone),
                                                       max(c for _r, c in zone) + 1)]
            elif zone:
                lane = max(0, min(cols - 1, int(round(target[1]))))
                path_cells = [(r, lane) for r in range(min(r for r, _c in zone),
                                                       max(r for r, _c in zone) + 1)]
            else:
                path_cells = []
            track["cells"] = [list(cell) for cell in path_cells]
            track["zone_cells"] = [list(cell) for cell in sorted(zone)]
        return danger

    def _remember_wolf_observation(self, debug=None):
        debug = debug or self.debug or {}
        hazards = []
        for item in debug.get("hazards") or []:
            if isinstance(item, dict):
                hazards.append([int(item["row"]), int(item["col"])])
            else:
                hazards.append([int(item[0]), int(item[1])])
        # Only surviving wolf cells count as a live observation.  Species and
        # environment rules may discard a raw dark component as goat artwork,
        # bomb smoke, or another special piece.
        wolf = deepcopy(debug.get("wolf_meta")) if hazards else None
        if wolf or self._wolf_observations:
            self._wolf_observations.append({
                "at": time.monotonic(), "wolf": wolf, "hazards": hazards,
            })
        self._wolf_motion = self._wolf_motion_summary(
            self._wolf_observations, self.rows, self.cols)
        fresh_zone = self._wolf_patrol_zone(
            self.sheep, self._wolf_motion, self.rows, self.cols)
        confirmed_now = set()
        for track in (self._wolf_motion or {}).get("tracks") or []:
            if not track.get("observed"):
                continue
            confirmed_now.update(
                tuple(cell) for cell in (track.get("zone_cells") or track.get("cells") or []))
        self._wolf_confirmed_cells.update(confirmed_now)
        current_cells = {
            tuple(cell) for cell in ((self._wolf_motion or {}).get("current_cells") or [])
        }
        # Current detections are replace-on-refresh.  Only a 3-frame-confirmed
        # patrol zone may survive a later frame; this prevents one bad wolf
        # component from becoming a permanent no-stop residue.
        self._wolf_danger_cells = set(self._wolf_confirmed_cells) | current_cells
        if self._wolf_motion is not None:
            self._wolf_motion["danger_cells"] = [
                list(cell) for cell in sorted(self._wolf_danger_cells)
            ]
            self._wolf_motion["track_cells"] = [
                list(cell) for cell in sorted(self._wolf_danger_cells)
            ]
        return self._wolf_motion

    def _wolf_track_cells(self):
        return set(self._wolf_danger_cells)

    def _wolf_guard_cells(self):
        """Current wolf footprint plus its short-horizon next positions."""
        motion = self._wolf_motion or {}
        guarded = {tuple(cell) for cell in motion.get("current_cells") or []}
        for track in motion.get("tracks") or []:
            direction = track.get("direction")
            if direction not in DIRS:
                continue
            zone = {tuple(cell) for cell in track.get("zone_cells") or []}
            active = guarded & zone
            dr, dc = DIRS[direction]
            for distance in (1, 2):
                guarded.update({
                    (r + dr * distance, c + dc * distance)
                    for r, c in active
                    if (r + dr * distance, c + dc * distance) in zone
                })
        return guarded

    @staticmethod
    def _move_wolf_risk(board, move, track_cells):
        track = {tuple(cell) for cell in (track_cells or [])}
        if not track or move.result != "EXIT":
            return {"risky": False, "overlap": [], "track_cells": len(track)}
        motion = Api._move_motion(board, move)
        overlap = sorted(motion["trail"] & track)
        return {
            "risky": bool(overlap),
            "overlap": [list(cell) for cell in overlap],
            "track_cells": len(track),
        }

    @staticmethod
    def _move_motion(board, move):
        """Model the in-board cells swept by one click animation."""
        cells = set(board.pieces[str(move.piece_id)]["cells"])
        dr, dc = DIRS[move.direction]
        frontier = set(cells)
        trail = set(cells)
        max_steps = max(board.rows, board.cols) + max(2, len(cells))
        steps = move.distance if move.result != "EXIT" else max_steps
        landing = set(cells)
        travel_steps = 0
        for _ in range(max(0, int(steps))):
            frontier = {(r + dr, c + dc) for r, c in frontier}
            travel_steps += 1
            inside = {cell for cell in frontier if board.in_board(*cell)}
            trail.update(inside)
            landing = inside
            if move.result == "EXIT" and not inside:
                landing = set()
                break
        return {
            "piece": str(move.piece_id),
            "direction": move.direction,
            "result": move.result,
            "travel_steps": travel_steps,
            "start": cells,
            "trail": trail,
            "landing": landing,
        }

    @staticmethod
    def _burst_gap_schedule(motions, base_interval_ms):
        """Scale dependent gaps by the earlier sheep's travel distance."""
        base = max(20, int(round(float(base_interval_ms))))
        click_cost_ms = 50
        gaps, reasons = [], []
        for index in range(1, len(motions)):
            current = motions[index]
            gap = base
            transition_reasons = []
            for lookback in range(1, min(3, index) + 1):
                previous = motions[index - lookback]
                overlap = previous["trail"] & current["trail"]
                previous_blocks_path = previous["landing"] & current["trail"]
                if overlap or previous_blocks_path:
                    same_direction = previous.get("direction") == current.get("direction")
                    horizontal = previous.get("direction") in {"L", "R"}
                    same_corridor = bool(overlap and same_direction and (
                        ({r for r, _c in previous["start"]}
                         == {r for r, _c in current["start"]}) if horizontal else
                        ({c for _r, c in previous["start"]}
                         == {c for _r, c in current["start"]})))
                    travel_steps = max(1, int(previous.get("travel_steps") or 1))
                    if previous_blocks_path:
                        required_age = round(max(720, 240 + travel_steps * 90) * 0.40)
                        kind = "landing_on_path"
                    elif same_corridor:
                        required_age = round(max(650, 220 + travel_steps * 80) * 0.40)
                        kind = "same_corridor"
                    else:
                        required_age = round(max(420, 180 + travel_steps * 60) * 0.40)
                        kind = "path_overlap"
                    previous_index = index - lookback
                    elapsed = (sum(gaps[previous_index:index - 1])
                               + click_cost_ms * max(0, lookback - 1))
                    needed_gap = max(base, required_age - elapsed)
                    gap = max(gap, needed_gap)
                    transition_reasons.append({
                        "lookback": lookback,
                        "kind": kind,
                        "travel_steps": travel_steps,
                        "required_age_ms": required_age,
                        "elapsed_before_gap_ms": elapsed,
                        "needed_gap_ms": needed_gap,
                        "cells": [list(cell) for cell in sorted(overlap or previous_blocks_path)[:8]],
                    })
                elif lookback == 1:
                    adjacent = any(
                        abs(ar - br) + abs(ac - bc) == 1
                        for ar, ac in previous["trail"] for br, bc in current["start"])
                    if adjacent:
                        gap = max(gap, 72)
                        transition_reasons.append({
                            "lookback": 1, "kind": "adjacent_path",
                            "required_age_ms": 72, "cells": []})
            gaps.append(gap)
            reasons.append(transition_reasons)
        return gaps, reasons

    @staticmethod
    def _schedule_wait_avoiding_exits(board, planned_moves, base_interval_ms,
                                      *, previous_motion=None, limit=16,
                                      wolf_track=None, review_ids=None):
        """Move an independent direct exit ahead of a corridor-delayed exit.

        Only ordinary sheep EXIT moves are commuted.  Removing such a piece
        cannot change another piece's location; we still prove that the
        deferred move remains legal after the candidate exits.
        """
        base = max(20, int(round(float(base_interval_ms))))
        cursor = board
        remaining = list(planned_moves or [])
        ordered, reorders = [], []
        prior_motion = previous_motion
        review_ids = {str(piece_id) for piece_id in (review_ids or [])}
        while remaining and len(ordered) < max(1, int(limit)):
            legal = cursor.legal_moves()
            first = next((item for item in legal if item == remaining[0]), None)
            if first is None:
                break
            chosen_index = 0
            if str(first.piece_id) in review_ids:
                for index in range(1, min(len(remaining), 9)):
                    candidate = next((item for item in legal if item == remaining[index]), None)
                    if (candidate is None or candidate.result != "EXIT"
                            or str(candidate.piece_id) in review_ids):
                        continue
                    piece = cursor.pieces[str(candidate.piece_id)]
                    if piece.get("species", "sheep") != "sheep":
                        continue
                    after_candidate = cursor.apply(candidate)
                    if not any(item == first for item in after_candidate.legal_moves()):
                        continue
                    chosen_index = index
                    reorders.append({
                        "deferred_piece": str(first.piece_id),
                        "preferred_piece": str(candidate.piece_id),
                        "reason": "low_confidence_avoidance",
                        "avoided_wait_ms": 0,
                    })
                    break
            for index, planned in enumerate(remaining):
                if chosen_index:
                    break
                candidate = next((item for item in legal if item == planned), None)
                if (candidate is not None
                        and not (str(candidate.piece_id) in review_ids
                                 and str(first.piece_id) not in review_ids)
                        and Api._move_wolf_risk(cursor, candidate, wolf_track)["risky"]):
                    chosen_index = index
                    if index:
                        reorders.append({
                            "deferred_piece": str(first.piece_id),
                            "preferred_piece": str(candidate.piece_id),
                            "reason": "wolf_track_priority",
                            "avoided_wait_ms": 0,
                        })
                    break
            first_motion = Api._move_motion(cursor, first)
            first_gap = base
            if prior_motion is not None:
                schedule, _ = Api._burst_gap_schedule([prior_motion, first_motion], base)
                first_gap = schedule[0] if schedule else base
            if chosen_index == 0 and prior_motion is not None and first_gap > base:
                for index in range(1, min(len(remaining), 9)):
                    candidate = next((item for item in legal if item == remaining[index]), None)
                    if (candidate is None or candidate.result != "EXIT"
                            or (str(candidate.piece_id) in review_ids
                                and str(first.piece_id) not in review_ids)):
                        continue
                    piece = cursor.pieces[str(candidate.piece_id)]
                    if piece.get("species", "sheep") != "sheep":
                        continue
                    candidate_motion = Api._move_motion(cursor, candidate)
                    schedule, _ = Api._burst_gap_schedule(
                        [prior_motion, candidate_motion], base)
                    if schedule and schedule[0] > base:
                        continue
                    after_candidate = cursor.apply(candidate)
                    if not any(item == first for item in after_candidate.legal_moves()):
                        continue
                    chosen_index = index
                    reorders.append({
                        "deferred_piece": str(first.piece_id),
                        "preferred_piece": str(candidate.piece_id),
                        "avoided_wait_ms": int(first_gap - base),
                    })
                    break
            chosen = remaining.pop(chosen_index)
            chosen = next(item for item in legal if item == chosen)
            prior_motion = Api._move_motion(cursor, chosen)
            ordered.append(chosen)
            cursor = cursor.apply(chosen)
        return ordered + remaining, reorders

    @staticmethod
    def _single_settle_ms(move, piece, requested_ms=60):
        """Return a lower bound that prevents recapture during one move."""
        base = max(350.0, float(requested_ms or 0))
        result = str(getattr(move, "result", "") or "")
        distance = max(0, int(getattr(move, "distance", 0) or 0))
        species = str((piece or {}).get("species", "sheep"))
        if result in {"MOVE", "STEP"}:
            base = max(base, 900.0 + min(700.0, distance * 110.0))
        else:
            base = max(base, 900.0)
        if species == "goat":
            base = max(base, 1200.0)
        if species == "bomb":
            # Bomb impacts update a visible counter/effect after the movement
            # finishes.  Recapturing too early can retain a stale hit budget.
            base = max(base, 1400.0)
        return base

    @staticmethod
    def _batch_soft_report(report):
        blockers = list((report or {}).get("execution_blockers") or [])
        return ((report or {}).get("scene_state") == "gameplay" and bool(blockers)
                and all(item.get("code") == "manual_review_required"
                        for item in blockers))

    def autonomous_refresh(self, solve_timeout_ms=10000, max_attempts=4, retry_ms=260):
        """Recapture, stabilize review evidence, and replan without user input."""
        def run():
            if not self._execution_lock.acquire(blocking=False):
                raise RuntimeError("上一执行阶段尚未结束")
            try:
                self._cancel_event.clear()
                attempts = max(1, min(8, int(max_attempts or 4)))
                delay_s = max(0.08, min(1.0, float(retry_ms or 260) / 1000.0))
                result = None
                rectinfo = None
                mode = None
                for index in range(attempts):
                    rectinfo, mode = self._capture_live(require_same_window=True)
                    result = self._analyze_frame(source="app-autonomous-refresh")
                    blocker_codes = {
                        item.get("code") for item in result.get("execution_blockers", [])
                    }
                    waiting_for_evidence = bool(blocker_codes & {
                        "manual_learning_confirmation_required", "manual_review_required",
                        "motion_smoke", "scene_not_gameplay",
                    })
                    if (result.get("scene_state") in {"victory"}
                            or (result.get("executable") and not waiting_for_evidence)):
                        break
                    if index + 1 < attempts:
                        self._wait_or_cancel(delay_s)

                timeout_s = max(1.0, min(60.0, float(solve_timeout_ms) / 1000.0))
                solution = None
                if (self.board is not None and result is not None
                        and (result.get("executable") or self._batch_soft_report(result))):
                    solution = self._solve_with_budget(
                        self._clone_board(self.board), timeout_s, self.Minv.copy())
                ok, buf = cv2.imencode(".png", self.game)
                if not ok:
                    raise RuntimeError("截图编码失败")
                return {
                    **(result or {}),
                    "capture": {"mode": mode, "win": {
                        "w": rectinfo[2], "h": rectinfo[3]}},
                    "img": base64.b64encode(buf.tobytes()).decode("ascii"),
                    "solution": solution,
                    "autonomous_attempts": attempts,
                    "runtime_confirmed_reviews": list(
                        (self.debug or {}).get("runtime_confirmed_reviews") or []),
                }
            finally:
                self._execution_lock.release()
        return _wrap(run)

    @staticmethod
    def _visual_transient_only(report):
        """Visual effects may hide pixels, but do not invalidate a trusted plan."""
        notices = Api._report_notices(report)
        visual_codes = {"gesture_occlusion", "motion_smoke"}
        visual_notices = [item for item in notices
                          if item.get("code") in visual_codes]
        hard_blockers = Api._hard_execution_blockers(report)
        return ((report or {}).get("scene_state") == "gameplay"
                and bool(visual_notices)
                and all(item.get("code") in visual_codes for item in hard_blockers))

    @staticmethod
    def _motion_smoke_only(report):
        """Return true when the only execution blocker is transient smoke."""
        blockers = list((report or {}).get("execution_blockers") or [])
        return ((report or {}).get("scene_state") == "gameplay" and bool(blockers)
                and all(item.get("code") == "motion_smoke" for item in blockers))

    def _capture_execution_preflight(self, source, *, max_wait_ms=2600,
                                     retry_interval_ms=180):
        """Capture until a transient sheep-movement effect has cleared.

        Smoke is safe to wait through, but never safe to click through because it
        can hide the piece or change the apparent occupancy.  The old path made
        one capture and immediately failed, which turned a short animation into
        a persistent execution error in the UI.
        """
        rectinfo, mode = self._capture_live(require_same_window=True)
        report = self._analyze_frame(source=source)
        retries = 0
        deadline = time.monotonic() + max(0.0, float(max_wait_ms) / 1000.0)
        while (self._motion_smoke_only(report)
               and time.monotonic() < deadline):
            retries += 1
            remaining = max(0.0, deadline - time.monotonic())
            self._wait_or_cancel(min(float(retry_interval_ms) / 1000.0, remaining))
            rectinfo, mode = self._capture_live(require_same_window=True)
            report = self._analyze_frame(source=source)
        # One still image only tells us where a wolf is now.  On wolf boards,
        # sample two more frames before authorizing any click so the patrol
        # axis/direction can be inferred and exposed to both planning and UI.
        if (self._wolf_requires_live_control()
                and not bool((self._wolf_motion or {}).get("observed"))):
            for _index in range(2):
                self._wait_or_cancel(0.12)
                rectinfo, mode = self._capture_live(require_same_window=True)
                report = self._analyze_frame(source=f"{source}-wolf-motion")
            motion = self._wolf_motion
            if motion:
                report = {**report, "wolf_motion": motion}
        return rectinfo, mode, report, retries

    def _guard_wolf_risk_move(self, board, move, *, max_wait_ms=2600,
                              retry_interval_ms=120):
        """Confirm a track-crossing EXIT against the wolf's latest position."""
        initial = self._move_wolf_risk(board, move, self._wolf_track_cells())
        if not initial["risky"]:
            return board, move, None
        deadline = time.monotonic() + max(0.3, float(max_wait_ms) / 1000.0)
        attempts = 0
        last_report = None
        while time.monotonic() < deadline:
            attempts += 1
            self._wait_or_cancel(max(0.04, float(retry_interval_ms) / 1000.0))
            _rectinfo, mode = self._capture_live(require_same_window=True)
            last_report = self._analyze_frame(source="app-wolf-trajectory-preflight")
            if (not last_report.get("executable")
                    and not self._batch_soft_report(last_report)):
                continue
            if self.board is None:
                continue
            fresh = self._match_planned_move(self.board, move)
            if fresh is None:
                # The wolf currently occupies the departure ray, or the live
                # board changed. Keep observing instead of clicking blindly.
                continue
            risk = self._move_wolf_risk(
                self.board, fresh, self._wolf_track_cells())
            current_risk = self._move_wolf_risk(
                self.board, fresh,
                self._wolf_guard_cells())
            if current_risk["risky"]:
                # A dynamic wolf is not a permanent Board obstacle, so test
                # its latest body cells explicitly before allowing the click.
                continue
            return self._clone_board(self.board), fresh, {
                "required": True,
                "observed": bool((self._wolf_motion or {}).get("observed")),
                "attempts": attempts,
                "capture_mode": mode,
                "risk": risk,
                "current_risk": current_risk,
                "motion": deepcopy(self._wolf_motion),
            }
        message = "狼仍在该羊的离场轨迹上，已观察动线但未出现安全点击窗口"
        blockers = list((last_report or {}).get("execution_blockers") or [])
        if blockers:
            message += "；" + "；".join(item.get("message", "") for item in blockers)
        raise RuntimeError(message)

    def _restore_predicted_after_visual_transient(self, report, predicted_board,
                                                   trusted_Minv, trusted_sheep):
        """Keep the click-result model until the next mandatory preflight."""
        transient = list(report.get("execution_blockers") or [])
        softened = {
            **report,
            "scene_reason": "瞬时视觉特效已忽略；沿用点击前可信棋盘并在下一步预检",
            "execution_blockers": [],
            "executable": True,
            "advisories": list(report.get("advisories") or []) + transient,
            "warnings": list(report.get("warnings") or []) + transient,
            "visual_transient_deferred": True,
        }
        self.board = self._clone_board(predicted_board)
        self.Minv = trusted_Minv.copy()
        active_ids = {str(pid) for pid in self.board.pieces}
        self.sheep = [deepcopy(item) for item in (trusted_sheep or [])
                      if str(item.get("id")) in active_ids]
        self._sync_species()
        data = self._board_data(self.board)
        self._detected_board_data = json.loads(json.dumps(data))
        self.board_revision = level_cache.board_hash(data)
        self._active_plan = None
        softened["board_revision"] = self.board_revision
        softened["count"] = self.board.remaining_count()
        softened["state"] = self._snapshot(self.board, highlight=None, Minv=self.Minv)
        self.scene_report = softened
        return softened

    def _execution_refresh_payload(self, result, rectinfo, mode, solve_timeout_ms,
                                   allow_soft=False, solution_override=None,
                                   background_replan=False, **extra):
        solution = solution_override
        can_solve = self.board is not None and (self._execution_allowed(result)
                                                 or (allow_soft and self._batch_soft_report(result)))
        coordinator_owned = self.runtime.snapshot().get("busy")
        if solution is None and can_solve and (not background_replan or coordinator_owned):
            solution = self._solve_with_budget(
                self._clone_board(self.board),
                max(1.0, min(30.0, float(solve_timeout_ms) / 1000.0)),
                self.Minv.copy(),
                job_id=(self.runtime.snapshot().get("id") if coordinator_owned else None),
                started=time.monotonic())
        replan_job = None
        if background_replan and can_solve and not coordinator_owned:
            started = self.solve_start(solve_timeout_ms)
            if started.get("ok"):
                replan_job = started.get("job")
        ok, buf = cv2.imencode(".png", self.game)
        if not ok:
            raise RuntimeError("截图编码失败")
        return {
            **result,
            "capture": {"mode": mode, "win": {"w": rectinfo[2], "h": rectinfo[3]}},
            "img": base64.b64encode(buf.tobytes()).decode("ascii"),
            "solution": solution,
            "replan_job": replan_job,
            **extra,
        }

    def execute_step(self, board_revision, move_id="0", settle_ms=60, hold_ms=70,
                     solve_timeout_ms=8000):
        """Preflight the live board, execute exactly one authoritative move, then replan."""
        def run():
            if not self._execution_lock.acquire(blocking=False):
                raise RuntimeError("已有执行任务正在进行")
            try:
                # The coordinator owns the token during workflow/auto mode.
                # Clearing it here would lose a pause that arrived between two
                # verified steps.
                if not self.runtime.snapshot().get("busy"):
                    self._cancel_event.clear()
                requested_revision = str(board_revision or "")
                action = str(move_id)
                if action != "0":
                    raise RuntimeError("在线执行只允许当前方案的第 1 步；其余步骤仅供沙盘预览")
                if (not self._execution_allowed(self.scene_report)
                        and not self._batch_soft_report(self.scene_report)
                        and not self._visual_transient_only(self.scene_report)):
                    messages = "；".join(
                        b["message"] for b in self._hard_execution_blockers(self.scene_report))
                    raise RuntimeError(messages or "当前识别结果禁止执行")
                plan = self._active_plan
                planned_move = (plan["moves"][0]
                                if self._plan_is_execution_ready(plan, requested_revision)
                                else None)
                planned_suffix = (list(plan["moves"][1:])
                                  if self._plan_is_execution_ready(plan, requested_revision)
                                  else [])
                if planned_move is None:
                    raise RuntimeError(self._incomplete_plan_error("当前计划"))

                # Mandatory preflight: recapture and prove the plan still starts
                # from a currently recognized board before any irreversible
                # mouse input.  The preflight frame is authoritative: if edge
                # animation changes recognition, replan it in-place instead of
                # bouncing the user through another capture cycle.
                rectinfo, mode, preflight, smoke_retries = (
                    self._capture_execution_preflight("app-execution-preflight"))

                preflight_allowed = self._execution_allowed(preflight)
                if not preflight_allowed and not self._batch_soft_report(preflight):
                    if self._motion_smoke_only(preflight):
                        # A previous click may still be animating.  Do not
                        # click through the smoke, but make the transient
                        # state retryable so auto execution can keep its
                        # trusted plan and try again after a short wait.
                        retry_after_ms = max(
                            220, min(900, 180 + smoke_retries * 120))
                        return {
                            **preflight,
                            "capture": {"mode": mode,
                                        "win": {"w": rectinfo[2],
                                                "h": rectinfo[3]}},
                            "clicked": None,
                            "retryable": True,
                            "retry_after_ms": retry_after_ms,
                            "smoke_retries": smoke_retries,
                        }
                    messages = "；".join(
                        b["message"] for b in self._hard_execution_blockers(preflight))
                    raise RuntimeError(messages or "点击前场景校验未通过")
                wolf_live_replan = bool(
                    self._wolf_requires_live_control()
                    and (self._wolf_motion or {}).get("observed"))
                preflight_replanned = (
                    self.board_revision != requested_revision
                    or planned_move is None
                    or wolf_live_replan)
                if preflight_replanned:
                    if self.board is None or self.Minv is None:
                        raise RuntimeError("点击前棋盘尚未准备完成")
                    fresh_solution = self._solve_with_budget(
                        self._clone_board(self.board),
                        max(1.0, min(30.0, float(solve_timeout_ms) / 1000.0)),
                        self.Minv.copy(),
                    )
                    if not self._solution_is_execution_ready(fresh_solution):
                        raise RuntimeError(
                            fresh_solution.get("execution_blocker")
                            or self._incomplete_plan_error("点击前重求方案"))
                    plan = self._active_plan
                    if not self._plan_is_execution_ready(plan, self.board_revision):
                        raise RuntimeError(self._incomplete_plan_error("点击前当前方案"))
                    planned_move = plan["moves"][0]
                    planned_suffix = list(plan["moves"][1:])
                    requested_revision = self.board_revision
                move = self._match_planned_move(self.board, planned_move)
                if move is None:
                    raise RuntimeError("点击前动作已不合法，旧方案已作废")
                review_ids = {str(item.get("id")) for item in (self.sheep or [])
                              if item.get("review")}
                if str(move.piece_id) in review_ids:
                    evidence = next((item for item in (self.sheep or [])
                                     if str(item.get("id")) == str(move.piece_id)), {})
                    return self._review_pause_payload(
                        move.piece_id, self.board.pieces[str(move.piece_id)]["cells"],
                        evidence.get("review_reason"), steps_completed=0)
                guarded_board, guarded_move, wolf_guard = self._guard_wolf_risk_move(
                    self.board, move)
                if wolf_guard:
                    self.board = guarded_board
                    move = guarded_move
                    planned_suffix = []
                    preflight_replanned = True
                expected_board = self.board.apply(move)
                expected_state = self._snapshot(expected_board, highlight=None)
                piece = self.board.pieces[str(move.piece_id)]
                cells = sorted(piece["cells"])
                px = sum(self._cell_center(r, c)[0] for r, c in cells) / len(cells)
                py = sum(self._cell_center(r, c)[1] for r, c in cells) / len(cells)
                execution_records = []
                clicked = self._click_image_point(
                    px, py, hold_ms,
                    before_click=lambda: execution_records.append(
                        self._record_execution_step(self.board, move, mode="single")))
                effective_settle_ms = self._single_settle_ms(move, piece, settle_ms)
                self._wait_or_cancel(max(0.35, min(3.0, effective_settle_ms / 1000.0)))

                rectinfo, mode = self._capture_live(require_same_window=True)
                post = self._analyze_frame(source="app-click-refresh")
                actual_state = post.get("state")
                verification = self._verification_feedback(expected_state, actual_state, 1)
                if verification:
                    verification = level_cache.save_feedback(
                        verification, capture_meta=self._last_cache, level_key=self._level_key)
                continuation = None
                if verification and verification.get("matched") and self.board is not None:
                    continuation = self._continuation_solution(
                        self._clone_board(self.board), planned_suffix, self.Minv.copy())
                    if (not self.board.is_solved()
                            and not self._solution_is_execution_ready(continuation)):
                        continuation = None
                return self._execution_refresh_payload(
                    post, rectinfo, mode, solve_timeout_ms,
                    solution_override=continuation,
                    clicked=clicked,
                    verification=verification,
                    execution_records=execution_records,
                    preflight_replanned=preflight_replanned,
                    smoke_retries=smoke_retries,
                    wolf_guard=wolf_guard,
                    wolf_live_replan=wolf_live_replan,
                )
            finally:
                self._execution_lock.release()
        return _wrap(run)

    def execute_burst(self, board_revision, max_steps=10, interval_ms=35, settle_ms=60,
                      hold_ms=35, solve_timeout_ms=8000):
        """Compatibility endpoint: every request executes exactly one verified step.

        The previous implementation held the execution lock while recursively
        falling back into execute_step(), which made self-deadlock possible.
        Automatic mode now repeats this safe primitive from the sole runtime
        worker and checks pause after the current click.
        """
        result = self.execute_step(
            board_revision, "0", settle_ms=max(20, int(settle_ms)),
            hold_ms=max(35, int(hold_ms)), solve_timeout_ms=solve_timeout_ms)
        if result.get("ok"):
            clicked = 1 if result.get("clicked") else 0
            result.update({
                "batch_size": clicked,
                "batch_moves": ([str(result.get("piece_id"))] if clicked else []),
                "batch_fallback": "single_verified_step",
                "stage_capture": {
                    "source": "automatic-stage-capture",
                    "stage": "single-step-only",
                    "mode": "post-step-live-refresh",
                },
                "batch_profile": {
                    "requested": max(1, int(max_steps)),
                    "risk_used": clicked,
                    "stage": "single-step-only",
                    "stage_size": clicked,
                    "exits": clicked if result.get("result") == "EXIT" else 0,
                    "moves": clicked if result.get("result") != "EXIT" else 0,
                    "interval_ms": 0,
                    "gap_schedule_ms": [],
                    "settle_ms": max(20, int(settle_ms)),
                    "adaptive": False,
                },
            })
        return result
    def refresh_and_solve(self, solve_timeout_ms=8000, expected_state=None, planned_steps=0):
        """Capture the live board, detect current hazards/pieces, and re-solve without clicking."""
        def run():
            rectinfo, mode = self._capture_live(require_same_window=False)
            result = self._analyze_frame(source="app-manual-refresh")
            actual_state = result.get("state")
            verification = self._verification_feedback(expected_state, actual_state, planned_steps)
            if verification and self._last_cache:
                verification = level_cache.save_feedback(
                    verification, capture_meta=self._last_cache, level_key=self._level_key)
            solution = None
            if self.board is not None and result.get("executable"):
                solution = self._solve_with_budget(
                    self._clone_board(self.board),
                    max(1.0, min(30.0, float(solve_timeout_ms) / 1000.0)),
                    self.Minv.copy())
            ok, buf = cv2.imencode(".png", self.game)
            if not ok:
                raise RuntimeError("截图编码失败")
            return {**result,
                    "capture": {"mode": mode, "win": {"w": rectinfo[2], "h": rectinfo[3]}},
                    "img": base64.b64encode(buf.tobytes()).decode("ascii"),
                    "verification": verification, "solution": solution}

            # Legacy implementation retained below temporarily for easy source
            # comparison; the authoritative safety path above always returns.
            self.hwnd = find_window(TITLE)
            if not self.hwnd:
                raise RuntimeError(f"找不到窗口「{TITLE}」")
            img, rectinfo, mode = grab(self.hwnd)
            self.game = img
            self.win = rectinfo
            cv2.imwrite(str(image_path("_game.png")), img)

            corners, rows, cols, ow, oh = _load_params()
            h, w = self.game.shape[:2]
            if ow and oh and (w, h) != (ow, oh):
                sx, sy = w / ow, h / oh
                corners = {k: [corners[k][0] * sx, corners[k][1] * sy] for k in corners}
            grid_model = D._grid_from_args(self.game, corners, rows, cols)
            self.rows, self.cols = rows, cols
            self.Minv = grid_model.inverse_matrix
            self.sheep, debug = D.analyze(self.game, grid_model)
            self.debug = debug
            self._sync_species()
            G.save_grid_data(grid_model, os.path.join(HERE, "board_grid.json"))
            bd = D.to_board(self.sheep, rows, cols, hazards=debug.get("hazards"),
                            fences=debug.get("fences"))
            layout = D.to_layout(self.sheep, rows, cols, debug["dropped"],
                                 hazards=debug.get("hazards"), fences=debug.get("fences"))
            json.dump(bd, open(os.path.join(HERE, "board.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            json.dump(layout, open(os.path.join(HERE, "board_layout.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            json.dump({"kept": self.sheep,
                       "hazards": debug.get("hazards", []),
                       "fences": debug.get("fences", []),
                       "wolf": debug.get("wolf_meta"),
                       "black_sheep_cluster": debug.get("black_sheep_cluster", []),
                       "black_sheep_applied": debug.get("black_sheep_applied", []),
                       "pink_sheep": debug.get("pink_sheep", []),
                       "pigs": debug.get("pigs", []),
                       "goats": debug.get("goats", []),
                       "goat_wolf_environment": debug.get("goat_wolf_environment", []),
                       "dropped": debug["dropped"],
                       "raw": debug["candidates"]},
                      open(os.path.join(HERE, "sheep_candidates.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            cv2.imwrite(str(image_path("_occ_axis_rect.png")), D.render_rect_debug(debug, self.sheep))
            cv2.imwrite(str(image_path("_grid_labels.png")), D.render_grid_labels(debug, self.sheep))
            cv2.imwrite(str(image_path("_layout.png")), D.render_layout(debug, self.sheep))
            if self._level_key is None:
                self._level_key = level_cache.board_hash(bd)[:16]
            cache_meta = level_cache.save_capture(
                bd,
                level_key=self._level_key,
                source="app-preflight-refresh",
                extra={"rows": rows, "cols": cols, "candidate_count": debug["candidate_count"],
                       "hazard_count": len(debug.get("hazards", []))},
            )
            self._last_cache = cache_meta
            self.board = board_io.load(os.path.join(HERE, "board.json"))
            actual_state = self._snapshot(self.board, highlight=None)
            verification = self._verification_feedback(expected_state, actual_state, planned_steps)
            if verification:
                verification = level_cache.save_feedback(
                    verification,
                    capture_meta=cache_meta,
                    level_key=self._level_key,
                )
            ok, buf = cv2.imencode(".png", self.game)
            if not ok:
                raise RuntimeError("截图编码失败")
            grid = []
            for r in range(rows + 1):
                grid.append([self._px(0, r), self._px(cols, r)])
            for c in range(cols + 1):
                grid.append([self._px(c, 0), self._px(c, rows)])
            result = self._solve_with_budget(self._clone_board(self.board),
                                             max(1.0, min(30.0, float(solve_timeout_ms) / 1000.0)),
                                             self.Minv.copy())
            return {
                "capture": {"mode": mode, "win": {"w": rectinfo[2], "h": rectinfo[3]}},
                "img": base64.b64encode(buf.tobytes()).decode("ascii"),
                "grid": grid,
                "rows": rows,
                "cols": cols,
                "count": len(self.sheep),
                "hazard_count": len(debug.get("hazards", [])),
                "cache": cache_meta,
                "layout": layout,
                "state": actual_state,
                "verification": verification,
                "solution": result,
            }
        return _wrap(run)

    def load_params(self):
        """启动/重置时读当前目录的 grid_params.json（没有则返回 None）。"""
        def run():
            p = os.path.join(HERE, "grid_params.json")
            if not os.path.exists(p):
                return {"params": None}
            return {"params": json.load(open(p, encoding="utf-8"))}
        return _wrap(run)

    def calibration_preview(self, corners, rows, cols):
        """Project calibration lines with the detector's authoritative homography."""
        def run():
            if self.game is None:
                raise RuntimeError("请先截图")
            rows_value, cols_value = int(rows), int(cols)
            if rows_value < 2 or cols_value < 2:
                raise RuntimeError("棋盘行列数不能小于 2")
            grid = G.BoardGrid(
                rows=rows_value,
                cols=cols_value,
                corners={key: [float(corners[key][0]), float(corners[key][1])]
                         for key in G.CORNER_KEYS},
                image_size=(int(self.game.shape[1]), int(self.game.shape[0])),
            )
            lines = []
            width, height = grid.rect_size
            for row in range(rows_value + 1):
                y = row * grid.cell
                lines.append([list(grid.rect_to_source(0, y)),
                              list(grid.rect_to_source(width, y))])
            for col in range(cols_value + 1):
                x = col * grid.cell
                lines.append([list(grid.rect_to_source(x, 0)),
                              list(grid.rect_to_source(x, height))])
            return {"grid": lines}
        return _wrap(run)

    def editor_grid(self):
        """Expose one atomic frame + board snapshot for visual manual editing."""
        def run():
            if self.game is None or self.Minv is None or self.board is None:
                raise RuntimeError("请先采集并分析棋盘")
            ok, encoded = cv2.imencode(".png", self.game)
            if not ok:
                raise RuntimeError("人工校验截图编码失败")
            height, width = self.game.shape[:2]
            cells = []
            for row in range(self.board.rows):
                for col in range(self.board.cols):
                    cells.append({
                        "row": row,
                        "col": col,
                        "poly": self._cell_poly(row, col),
                        "center": self._cell_center(row, col),
                    })
            return {"rows": self.board.rows, "cols": self.board.cols,
                    "img": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    "image_size": [int(width), int(height)],
                    "state": self._snapshot(self.board, highlight=None),
                    "board_revision": self.board_revision,
                    "cells": cells, "grid": self._grid_lines(self.board.rows, self.board.cols),
                    "can_undo": bool(self._editor_undo), "can_redo": bool(self._editor_redo),
                    "manual_pending": bool(self._manual_edit_pending)}
        return _wrap(run)

    def save_params(self, corners, rows, cols, locked=None):
        """调参器保存：四角(截图像素坐标) + 行列 + 标定时分辨率 -> 写 grid_params.json。
        记下 imgW/imgH，换分辨率时可把这份四角按比例缩放成新起点。"""
        def run():
            h, w = self.game.shape[:2] if self.game is not None else (0, 0)
            P = {"corners": {k: [round(float(corners[k][0]), 1), round(float(corners[k][1]), 1)]
                             for k in ("TL", "TR", "BR", "BL")},
                 "rows": int(rows), "cols": int(cols),
                 "imgW": int(w), "imgH": int(h),
                 "nudge": [0, 0],
                 "locked": [key for key in (locked or []) if key in G.CORNER_KEYS],
                 "image": "images/_game.png"}
            blockers, _warnings = safety.validate_calibration(P, self.game.shape if self.game is not None else (h, w, 3))
            if blockers:
                raise RuntimeError("；".join(item["message"] for item in blockers))
            json.dump(P, open(os.path.join(HERE, "grid_params.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            self._frame_history.clear()
            return {"saved": True}
        return _wrap(run)

    def _sheep_centers(self):
        """白羊掩膜 -> 距离变换峰 -> 大半径 NMS：每只羊一个中心(像素)。"""
        g = self.game
        H, W = g.shape[:2]
        hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
        Sc, Vc = hsv[:, :, 1], hsv[:, :, 2]
        white = ((Sc < 70) & (Vc > 150)).astype(np.uint8)
        white[:int(0.13 * H)] = 0; white[int(0.80 * H):] = 0
        white[:, :int(0.16 * W)] = 0; white[:, int(0.84 * W):] = 0
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        dt = cv2.distanceTransform(white, cv2.DIST_L2, 5)
        md = 46
        dil = cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (md, md)))
        ys, xs = np.where((dt == dil) & (dt > 7))
        order = sorted(zip(xs.tolist(), ys.tolist(), dt[ys, xs].tolist()), key=lambda t: -t[2])
        cen = []
        for x, y, _v in order:
            if all((x - a) ** 2 + (y - b) ** 2 > md * md for a, b in cen):
                cen.append((x, y))
        return cen

    def _guess_from_sheep(self):
        """无任何标定时的兜底：羊中心 minAreaRect 粗估四角+行列（已知偏窄，仅作起点）。"""
        cen = self._sheep_centers()
        if len(cen) < 4:
            raise RuntimeError("羊中心太少，无法估计")
        C = np.array(cen, np.float32)
        box = cv2.boxPoints(cv2.minAreaRect(C))
        ssum = box.sum(1); sdif = box[:, 1] - box[:, 0]
        TL = box[np.argmin(ssum)]; BR = box[np.argmax(ssum)]
        TR = box[np.argmin(sdif)]; BL = box[np.argmax(sdif)]
        ctr = box.mean(0)
        def expand(p):
            return [round(float(p[0] + (p[0] - ctr[0]) * 0.10), 1),
                    round(float(p[1] + (p[1] - ctr[1]) * 0.10), 1)]
        corners = {k: expand(p) for k, p in [("TL", TL), ("TR", TR), ("BR", BR), ("BL", BL)]}
        nn = []
        for i in range(len(C)):
            dd = np.hypot(*(C - C[i]).T); dd[i] = 1e9; nn.append(float(dd.min()))
        pitch = float(np.median(nn)) if nn else 65.0
        ex = (TR - TL) / (np.linalg.norm(TR - TL) + 1e-9)
        ey = (BL - TL) / (np.linalg.norm(BL - TL) + 1e-9)
        pc = (C - TL) @ ex; pr = (C - TL) @ ey
        cols = max(2, int(round((pc.max() - pc.min()) / pitch)) + 1)
        rows = max(2, int(round((pr.max() - pr.min()) / pitch)) + 1)
        return {"corners": corners, "rows": rows, "cols": cols}

    def seed_params(self):
        """校准起点：优先把已存标定按分辨率缩放到当前截图；没有标定才退回羊中心粗估。"""
        def run():
            if self.game is None:
                raise RuntimeError("请先截图")
            h, w = self.game.shape[:2]
            centers = [[int(x), int(y)] for x, y in self._sheep_centers()]
            p = os.path.join(HERE, "grid_params.json")
            if os.path.exists(p):
                P = json.load(open(p, encoding="utf-8"))
                cor, ow, oh = P.get("corners"), P.get("imgW"), P.get("imgH")
                if cor and ow and oh:                    # 按比例缩放上次标定
                    sx, sy = w / ow, h / oh
                    sc = {k: [round(cor[k][0] * sx, 1), round(cor[k][1] * sy, 1)]
                          for k in ("TL", "TR", "BR", "BL")}
                    return {"corners": sc, "rows": P.get("rows", 18), "cols": P.get("cols", 12),
                            "locked": P.get("locked", []),
                            "centers": centers, "mode": "scaled",
                            "fromRes": [ow, oh], "toRes": [w, h]}
                if cor:                                  # 有标定但旧版没存分辨率：原样
                    return {"corners": cor, "rows": P.get("rows", 18), "cols": P.get("cols", 12),
                            "locked": P.get("locked", []),
                            "centers": centers, "mode": "asis"}
            g = self._guess_from_sheep()                 # 无标定兜底
            g.update({"centers": centers, "mode": "guess", "locked": []})
            return g
        return _wrap(run)


def _wrap(fn):
    try:
        r = fn()
        r = r or {}
        r["ok"] = True
        return r
    except Exception as e:
        result = {"ok": False, "error": _safe_error(e)}
        payload = getattr(e, "payload", None)
        if isinstance(payload, dict):
            result.update(payload)
        return result


def _safe_error(exc):
    """Return a short error string without letting COM/VARIANT repr recurse."""
    name = type(exc).__name__
    try:
        msg = str(exc)
    except RecursionError:
        msg = "maximum recursion depth exceeded while formatting error"
    except Exception:
        try:
            msg = repr(exc)
        except Exception:
            msg = ""
    msg = (msg or name).replace("\r", " ").replace("\n", " ")
    if len(msg) > 500:
        msg = msg[:500] + "..."
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "app-runtime.log")
        if os.path.exists(log_path) and os.path.getsize(log_path) >= 2 * 1024 * 1024:
            for index in range(2, 0, -1):
                src = f"{log_path}.{index}"
                dst = f"{log_path}.{index + 1}"
                if os.path.exists(src):
                    if index == 2:
                        os.remove(src)
                    else:
                        os.replace(src, dst)
            os.replace(log_path, f"{log_path}.1")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {name}: {msg}\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            f.write("\n")
    except Exception:
        pass
    return msg


def main():
    global _SINGLETON_HANDLE
    _SINGLETON_HANDLE = ctypes.windll.kernel32.CreateMutexW(
        None, False, "Local\\SheepSolverPyWebViewSingleton")
    if ctypes.windll.kernel32.GetLastError() == 183:
        print("求解器已经在运行，本次启动已退出。")
        return 2
    api = Api()
    window = webview.create_window(
        "套住那只羊 · 求解器",
        url=os.path.join(HERE, "app", "index.html"),
        js_api=api,
        width=DESKTOP_WINDOW_SIZE[0],
        height=DESKTOP_WINDOW_SIZE[1],
        min_size=MIN_WINDOW_SIZE,
    )
    webview.start(api.start_hotkeys, (window,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
