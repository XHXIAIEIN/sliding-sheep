
import os
import json
import base64
import time
import ctypes
from copy import deepcopy
from ctypes import wintypes
import cv2
import numpy as np
import board_grid as G
import board_io
import vision as D
import level_cache
import runtime as app_runtime
from capture_window import find_window, grab, list_windows  # 注：import 时已 SetProcessDPIAware
from paths import image_path
from . import common
from .common import ExecutionReviewRequired, MOUSEEVENTF_LEFTDOWN, TITLE, _load_params, _wrap, user32


class ExecutionOps:
    """Mixin: verified click execution, retries, and refresh cycles."""

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
                    and not ExecutionOps._hard_execution_blockers(report))

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
                with open(common.RETRY_CONTROLS_PATH, encoding="utf-8") as stream:
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
                with open(common.RETRY_CONTROLS_PATH, encoding="utf-8") as stream:
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
        notices = ExecutionOps._report_notices(report)
        visual_codes = {"gesture_occlusion", "motion_smoke"}
        visual_notices = [item for item in notices
                          if item.get("code") in visual_codes]
        hard_blockers = ExecutionOps._hard_execution_blockers(report)
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
            G.save_grid_data(grid_model, os.path.join(common.HERE, "board_grid.json"))
            bd = D.to_board(self.sheep, rows, cols, hazards=debug.get("hazards"),
                            fences=debug.get("fences"))
            layout = D.to_layout(self.sheep, rows, cols, debug["dropped"],
                                 hazards=debug.get("hazards"), fences=debug.get("fences"))
            json.dump(bd, open(os.path.join(common.HERE, "board.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            json.dump(layout, open(os.path.join(common.HERE, "board_layout.json"), "w", encoding="utf-8"),
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
                      open(os.path.join(common.HERE, "sheep_candidates.json"), "w", encoding="utf-8"),
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
            self.board = board_io.load(os.path.join(common.HERE, "board.json"))
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
