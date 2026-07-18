"""「套住那只羊」pywebview 桌面应用。

The UI submits one user intent through workflow_start().  OperationCoordinator
is the only background owner; analysis, planning, execution and verification
publish one observable job state.  Automatic mode repeats the same verified
single-step primitive as manual execution, so there is no recursive lock path
and no unverified burst click path.

Api 的具体实现按职责拆分在 gui/ 包的 mixin 模块中。

运行：py scripts/app.py（游戏窗口需可见；F12 / Esc 在当前点击结束后暂停）
"""
import ctypes
import os
import threading
from collections import deque

import cv2  # noqa: F401  测试通过 app.cv2 打桩截图写盘
import webview

import level_reader  # noqa: F401  测试通过 app.level_reader 打桩关卡读取
import planner  # noqa: F401  测试通过 app.planner 打桩求解
import runtime as app_runtime
import solver_learning  # noqa: F401  测试通过 app.solver_learning 打桩策略画像
from gui import (  # noqa: F401  ExecutionReviewRequired/_wrap 为兼容旧引用保留
    AnalysisOps,
    BoardStateOps,
    CalibrationOps,
    EditorOps,
    ExecutionOps,
    ExecutionReviewRequired,
    GridGeometryOps,
    SettingsOps,
    SolveOps,
    WindowOps,
    WolfOps,
    WorkflowOps,
    _safe_error,
    _wrap,
)
from gui.common import (
    DESKTOP_WINDOW_SIZE,
    HERE,
    MIN_WINDOW_SIZE,
    _empty_scene_report,
)

_SINGLETON_HANDLE = None


class Api(
    GridGeometryOps,
    WindowOps,
    SettingsOps,
    AnalysisOps,
    EditorOps,
    BoardStateOps,
    SolveOps,
    WorkflowOps,
    ExecutionOps,
    WolfOps,
    CalibrationOps,
):
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
