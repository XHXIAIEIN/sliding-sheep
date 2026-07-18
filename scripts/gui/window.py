
import os
import json
import threading
import ctypes
from ctypes import wintypes
from capture_window import find_window, grab, list_windows  # 注：import 时已 SetProcessDPIAware
from .common import DESKTOP_WINDOW_SIZE, HOTKEYS, MIN_WINDOW_SIZE, MOD_NOREPEAT, REFERENCE_WINDOW_SIZE, TITLE, WM_HOTKEY, _wrap, kernel32, user32


class WindowOps:
    """Mixin: window targeting, mode switching, and global hotkeys."""

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

    def list_targets(self):
        def run():
            return {"windows": list_windows(TITLE),
                    "selected": str(self.target_hwnd) if self.target_hwnd else None}
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
