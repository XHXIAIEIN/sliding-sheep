"""Shared constants, runtime settings, and error plumbing for the GUI API."""

import os
import json
import time
import threading
import ctypes
import traceback
from core import safety
from paths import ROOT


HERE = str(ROOT)
TITLE = "套住那只羊"
DEFAULT_SOLVE_TIMEOUT = 10.0
DESKTOP_WINDOW_SIZE = (1320, 900)
REFERENCE_WINDOW_SIZE = (460, 900)
MIN_WINDOW_SIZE = (390, 640)
LOG_DIR = os.path.join(HERE, "logs")
RETRY_CONTROLS_PATH = os.path.join(HERE, "data", "retry_controls.json")
RUNTIME_SETTINGS_PATH = os.path.join(HERE, "data", "runtime_settings.json")
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


def data_path(name):
    """Path of a runtime artifact under <HERE>/data, honouring HERE redirection."""
    folder = os.path.join(HERE, "data")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


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
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
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
    P = json.load(open(data_path("grid_params.json"), encoding="utf-8"))
    return (P["corners"], int(P["rows"]), int(P["cols"]),
            int(P.get("imgW", 0)), int(P.get("imgH", 0)))


def _empty_scene_report():
    return {
        "scene_state": "unknown",
        "execution_blockers": [safety.blocker("not_analyzed", "尚未完成场景识别")],
        "executable": False,
    }


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
