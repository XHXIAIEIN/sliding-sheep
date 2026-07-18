"""Single-owner background operation coordinator.

The desktop app used to let capture, solve, burst execution and frontend timers
each own a piece of the runtime state.  This module deliberately provides only
one active operation, one cancellation token and one public snapshot.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import threading
import time
import traceback
from typing import Any, Callable


class Phase(str, Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    ANALYZING = "analyzing"
    OPENING_COARSE = "opening-coarse"
    SOLVING = "solving"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    PAUSING = "pausing"
    DONE = "done"
    CANCELLED = "cancelled"
    ERROR = "error"


TERMINAL_PHASES = {Phase.IDLE, Phase.DONE, Phase.CANCELLED, Phase.ERROR}


class OperationBusy(RuntimeError):
    pass


class OperationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class OperationContext:
    job_id: str
    cancel_event: threading.Event
    publish: Callable[..., None]

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def checkpoint(self) -> None:
        if self.cancelled:
            raise OperationCancelled("操作已暂停")

    def wait(self, seconds: float) -> None:
        if self.cancel_event.wait(max(0.0, float(seconds))):
            raise OperationCancelled("操作已暂停")


class OperationCoordinator:
    """Run at most one background job and expose an immutable status copy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._state = self._fresh_state()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel

    @staticmethod
    def _fresh_state() -> dict[str, Any]:
        return {
            "id": None,
            "action": None,
            "phase": Phase.IDLE.value,
            "busy": False,
            "can_cancel": False,
            "detail": "",
            "progress": None,
            "started_at": None,
            "finished_at": None,
            "elapsed_ms": 0,
            "result": None,
            "error": None,
        }

    def _elapsed_ms(self) -> int:
        started = self._state.get("started_at")
        return int((time.monotonic() - started) * 1000) if started else 0

    def snapshot(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if job_id is not None and self._state.get("id") != str(job_id):
                return {"id": str(job_id), "phase": "stale", "busy": False}
            state = deepcopy(self._state)
            if state["busy"]:
                state["elapsed_ms"] = self._elapsed_ms()
            return state

    def update(self, job_id: str, *, phase: Phase | str | None = None,
               detail: str | None = None, progress: Any = None,
               **fields: Any) -> None:
        with self._lock:
            if self._state.get("id") != str(job_id):
                return
            if phase is not None:
                self._state["phase"] = phase.value if isinstance(phase, Phase) else str(phase)
            if detail is not None:
                self._state["detail"] = str(detail)
            if progress is not None:
                self._state["progress"] = deepcopy(progress)
            self._state.update(deepcopy(fields))
            self._state["elapsed_ms"] = self._elapsed_ms()

    def start(self, action: str, worker: Callable[[OperationContext], Any]) -> dict[str, Any]:
        with self._lock:
            if self._state["busy"]:
                raise OperationBusy(
                    f"当前正在{self._state.get('detail') or self._state.get('action')}，请先暂停或等待完成")
            self._seq += 1
            job_id = str(self._seq)
            self._cancel.clear()
            self._state = self._fresh_state()
            self._state.update({
                "id": job_id,
                "action": str(action),
                "phase": Phase.ANALYZING.value if action == "analyze" else Phase.SOLVING.value,
                "busy": True,
                "can_cancel": True,
                "detail": str(action),
                "started_at": time.monotonic(),
            })

        def run() -> None:
            context = OperationContext(
                job_id=job_id,
                cancel_event=self._cancel,
                publish=lambda **fields: self.update(job_id, **fields),
            )
            try:
                result = worker(context)
                context.checkpoint()
            except OperationCancelled:
                self._finish(job_id, Phase.CANCELLED)
            except Exception as exc:
                self._finish(job_id, Phase.ERROR, error=self._format_error(exc))
            else:
                self._finish(job_id, Phase.DONE, result=result)
            finally:
                with self._lock:
                    if self._thread is threading.current_thread():
                        self._thread = None

        thread = threading.Thread(
            target=run, name=f"sheep-{action}-{job_id}", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()
        return self.snapshot(job_id)

    @staticmethod
    def _format_error(exc: Exception) -> dict[str, Any]:
        message = str(exc).replace("\r", " ").replace("\n", " ")[:500]
        result = {
            "type": type(exc).__name__,
            "message": message or type(exc).__name__,
            "trace": "".join(traceback.format_exception(exc))[-4000:],
        }
        payload = getattr(exc, "payload", None)
        if isinstance(payload, dict):
            result.update(deepcopy(payload))
        return result

    def _finish(self, job_id: str, phase: Phase, *, result: Any = None,
                error: Any = None) -> None:
        with self._lock:
            if self._state.get("id") != str(job_id):
                return
            self._state.update({
                "phase": phase.value,
                "busy": False,
                "can_cancel": False,
                "finished_at": time.monotonic(),
                "elapsed_ms": self._elapsed_ms(),
                "result": deepcopy(result),
                "error": deepcopy(error),
            })

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if not self._state["busy"]:
                return deepcopy(self._state)
            self._cancel.set()
            self._state.update({
                "phase": Phase.PAUSING.value,
                "detail": "当前点击完成后暂停",
                "can_cancel": False,
            })
            return deepcopy(self._state)

    def reset(self) -> None:
        with self._lock:
            if self._state["busy"]:
                raise OperationBusy("当前操作尚未结束")
            self._cancel.clear()
            self._state = self._fresh_state()

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
        if not thread:
            return True
        thread.join(timeout)
        return not thread.is_alive()
