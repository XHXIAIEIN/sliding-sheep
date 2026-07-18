
import time
from copy import deepcopy
from board import io as board_io
from levels import cache as level_cache
from solver import planner
from core import runtime as app_runtime
from solver import learning as solver_learning
from solver import DIRS, Move
from .common import DEFAULT_SOLVE_TIMEOUT, _safe_error, _wrap


class SolveOps:
    """Mixin: solve entry points, budgeted search, and solution payloads."""

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
