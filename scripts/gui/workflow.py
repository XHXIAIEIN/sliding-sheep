
import base64
import time
from copy import deepcopy
import cv2
import numpy as np
from solver import planner
from core import runtime as app_runtime
from paths import image_path
from .common import _wrap


class WorkflowOps:
    """Mixin: single-intent workflow jobs (capture, upload, solve, coarse, quick exit)."""

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
