
import os
from levels import cache as level_cache
from . import common
from .common import _RUNTIME_SETTINGS_LOCK, _empty_scene_report, _normalize_runtime_settings, _read_runtime_settings, _wrap, _write_json_atomic


class SettingsOps:
    """Mixin: runtime settings persistence and hard refresh."""

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
                    _write_json_atomic(common.RUNTIME_SETTINGS_PATH, incoming)
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
                path = (common.data_path(relative) if relative in level_cache.DATA_ARTIFACTS
                        else os.path.join(common.HERE, relative))
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
