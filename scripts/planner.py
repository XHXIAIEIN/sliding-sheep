"""One solver policy shared by the GUI and command line entry points."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from solver import Board, Move, greedy_solve, solve as exact_solve
from solver_search import (beam_solve, randomized_macro_solve,
                           forced_exit_candidates, forced_exit_sort_key,
                           structural_deadlocks, supports_forced_exit_closure,
                           weighted_astar_solve)


class PlanningCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanResult:
    steps: list[tuple[Move, str]]
    final_board: Board
    solved: bool
    remaining: int
    kind: str
    timed_out: bool
    info: dict


@dataclass
class ElasticBudget:
    """A bounded deadline that can grow in explicit, observable increments."""
    initial_s: float
    extension_s: float
    max_s: float
    enabled: bool = False

    def __post_init__(self):
        self.started = time.monotonic()
        self.initial_s = max(.1, float(self.initial_s))
        self.extension_s = max(.1, float(self.extension_s))
        self.max_s = max(self.initial_s, float(self.max_s))
        if not self.enabled:
            self.max_s = self.initial_s
        self.allocated_s = self.initial_s
        self.deadline = self.started + self.allocated_s
        self.extensions = 0

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def extend(self) -> float:
        if not self.enabled or self.allocated_s >= self.max_s - .001:
            return 0.0
        added = min(self.extension_s, self.max_s - self.allocated_s)
        self.allocated_s += added
        self.deadline += added
        self.extensions += 1
        return added

    def info(self) -> dict:
        return {
            "initial_ms": int(self.initial_s * 1000),
            "extension_ms": int(self.extension_s * 1000),
            "max_ms": int(self.max_s * 1000),
            "allocated_ms": int(self.allocated_s * 1000),
            "elapsed_ms": int((time.monotonic() - self.started) * 1000),
            "extensions": int(self.extensions),
            "elastic": bool(self.enabled),
        }


Progress = Callable[[str, dict], None]
Cancelled = Callable[[], bool]
ExitPriority = Callable[[Board, Move], tuple]


def _noop_progress(_phase: str, _data: dict) -> None:
    pass


def _never_cancelled() -> bool:
    return False


def apply_moves(board: Board, moves: list[Move]) -> tuple[int, Board]:
    current = board
    for move in moves:
        current = current.apply(move)
    return current.remaining_count(), current


def _planner_exit_sort_key(board: Board, move: Move,
                           exit_priority: ExitPriority | None):
    policy = exit_priority(board, move) if exit_priority else ()
    if not isinstance(policy, tuple):
        policy = (policy,)
    return (*policy, *forced_exit_sort_key(board, move))


def coarse_exit_plan(board: Board, *, cancel: Cancelled | None = None,
                     progress: Progress | None = None,
                     exit_priority: ExitPriority | None = None,
                     max_layers: int | None = None) -> PlanResult:
    """Build the opening ordinary-sheep EXIT closure without running search.

    Every selected move is legal on the successively reduced board, removes an
    ordinary sheep, and is not currently needed as a staged stopper.  This
    makes the prefix safe to execute as one verified opening batch while the
    expensive search keeps stopper-order choices for the smaller post-batch
    board.
    """
    cancel = cancel or _never_cancelled
    progress = progress or _noop_progress
    current = board
    coarse: list[Move] = []
    exit_layers: dict[str, int] = {}
    layer = 0
    layer_limit = None if max_layers is None else max(0, int(max_layers))
    progress("opening-coarse", {
        "steps": 0, "remaining": current.remaining_count(), "layer": 0,
        "max_layers": layer_limit,
    })
    while not current.is_solved() and (layer_limit is None or layer < layer_limit):
        if cancel():
            raise PlanningCancelled("粗解已暂停")
        frontier = forced_exit_candidates(current, ordinary_only=True)
        if not frontier:
            break
        layer += 1
        frontier_ids = {str(move.piece_id) for move in frontier}
        layer_steps = 0
        while frontier_ids:
            if cancel():
                raise PlanningCancelled("粗解已暂停")
            exits = [move for move in forced_exit_candidates(current, ordinary_only=True)
                     if str(move.piece_id) in frontier_ids]
            if not exits:
                break
            exits.sort(key=lambda move: _planner_exit_sort_key(
                current, move, exit_priority))
            move = exits[0]
            piece_id = str(move.piece_id)
            frontier_ids.discard(piece_id)
            coarse.append(move)
            exit_layers[piece_id] = layer
            layer_steps += 1
            current = current.apply(move)
            progress("opening-coarse", {
                "steps": len(coarse), "remaining": current.remaining_count(),
                "layer": layer, "layer_steps": layer_steps,
                "max_layers": layer_limit,
            })
    remaining = current.remaining_count()
    solved = remaining == 0
    kind = (f"快速直出{layer}层/{len(coarse)}步"
            if layer_limit is not None else f"开局粗解{len(coarse)}")
    return PlanResult(
        steps=[(move, "coarse") for move in coarse],
        final_board=current,
        solved=solved,
        remaining=remaining,
        kind=kind,
        timed_out=False,
        info={
            "solved": solved,
            "remaining": remaining,
            "kind": "opening-coarse",
            "coarse_only": True,
            "layers": layer,
            "max_layers": layer_limit,
            "exit_layers": exit_layers,
        },
    )


def _best_candidate(board: Board, candidates: list[tuple[list[Move], dict]]):
    def score(item):
        moves, info = item
        remaining, candidate = apply_moves(board, moves)
        legal = len(candidate.legal_moves()) if remaining else 0
        return (
            0 if info.get("solved") else 1,
            int(info.get("remaining", remaining)),
            1 if remaining and legal == 0 else 0,
            -legal,
            len(moves),
        )
    return min(candidates, key=score)


def _policy_order(policy: dict | None, family: str, defaults: list[str]) -> list[str]:
    requested = list(((policy or {}).get("orders") or {}).get(family) or [])
    return [name for name in requested if name in defaults] + [
        name for name in defaults if name not in requested]


def _policy_weight(policy: dict | None, phase: str, fallback: float) -> float:
    value = ((policy or {}).get("time_weights") or {}).get(phase, fallback)
    try:
        return max(.18, min(.82, float(value)))
    except (TypeError, ValueError):
        return fallback


def _strategy_result(progress: Progress, phase: str, started: float,
                     start_remaining: int, remaining: int, info: dict,
                     attempt: int, budget_s: float) -> None:
    progress(phase, {
        "event": "finish",
        "attempt": attempt + 1,
        "start_remaining": int(start_remaining),
        "remaining": int(remaining),
        "solved": bool(info.get("solved") or remaining == 0),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "budget_ms": int(max(0.0, budget_s) * 1000),
        **{key: int(info[key]) for key in ("expanded", "restarts", "depth")
           if isinstance(info.get(key), (int, float))},
    })


def _refine(board: Board, deadline: float, cancel: Cancelled,
            progress: Progress, *, policy: dict | None = None,
            attempt: int = 0) -> tuple[list[Move], dict]:
    if cancel():
        raise PlanningCancelled("求解已暂停")
    if board.is_solved():
        return [], {"solved": True, "kind": "coarse-only", "remaining": 0}
    remaining_time = deadline - time.monotonic()
    if remaining_time <= .05:
        return [], {"solved": False, "kind": "精解超时", "timeout": True,
                    "remaining": board.remaining_count()}

    candidates: list[tuple[list[Move], dict]] = []
    greedy_seeded = False
    macro = supports_forced_exit_closure(board)
    start_remaining = board.remaining_count()

    if len(board.pieces) <= 14:
        run_started = time.monotonic()
        progress("exact-a*", {"event": "start", "attempt": attempt + 1,
                              "remaining": start_remaining,
                              "budget_ms": int(max(0.0, remaining_time) * 1000)})
        moves, info = exact_solve(board, max_nodes=400_000, cancel=cancel)
        if info.get("cancelled"):
            raise PlanningCancelled("求解已暂停")
        remaining = start_remaining if moves is None else apply_moves(board, moves)[0]
        exact_info = {**info, "solved": moves is not None and remaining == 0,
                      "kind": "A*最优" if moves is not None else "A*搜索",
                      "remaining": remaining}
        _strategy_result(progress, "exact-a*", run_started, start_remaining,
                         remaining, exact_info, attempt, remaining_time)
        if moves is not None:
            return moves, exact_info
        if info.get("reason") == "无解（搜索穷尽）":
            return [], {**info, "solved": False, "kind": "A*证明无解",
                        "remaining": start_remaining}

    if macro:
        deadlocks = structural_deadlocks(board)
        if deadlocks:
            return [], {
                "solved": False,
                "kind": "结构死锁",
                "remaining": start_remaining,
                "structural_deadlocks": deadlocks,
                "reason": "固定朝向棋子迎头相向，需核对特殊羊规则或识别方向",
            }
        for phase in _policy_order(
                policy, "macro", ["macro-beam", "randomized-macro"]):
            remaining_time = deadline - time.monotonic()
            if remaining_time <= .18:
                break
            default_weight = .32 if phase == "macro-beam" else .68
            weight = _policy_weight(policy, phase, default_weight)
            cap = 5.0 if phase == "macro-beam" else 12.0
            limit = min(cap, max(.1, remaining_time * weight))
            run_started = time.monotonic()
            progress(phase, {"event": "start", "attempt": attempt + 1,
                             "remaining": start_remaining,
                             "budget_ms": int(limit * 1000)})
            if phase == "macro-beam":
                moves, info = beam_solve(
                    board, width=14, max_depth=96, time_limit=limit,
                    seen_cap=120_000, cancel=cancel)
                info = {**info, "kind": f"macro-{info.get('kind', 'beam')}"}
            else:
                moves, info = randomized_macro_solve(
                    board, seed=attempt, time_limit=limit, cancel=cancel)
            if info.get("cancelled"):
                raise PlanningCancelled("求解已暂停")
            remaining, _ = apply_moves(board, moves)
            info = {**info, "remaining": remaining, "solved": remaining == 0}
            _strategy_result(progress, phase, run_started, start_remaining,
                             remaining, info, attempt, limit)
            candidates.append((moves, info))
            if info.get("solved"):
                return moves, info

    if len(board.pieces) >= 35 and not macro:
        run_started = time.monotonic()
        progress("online-greedy", {"event": "start", "attempt": attempt + 1,
                                   "remaining": start_remaining})
        moves, info = greedy_solve(board, max_steps=80, cancel=cancel)
        remaining, _ = apply_moves(board, moves)
        info = {**info, "kind": "online-greedy", "remaining": remaining,
                "partial": not bool(info.get("solved")), "timeout": False}
        _strategy_result(progress, "online-greedy", run_started, start_remaining,
                         remaining, info, attempt, 0.0)
        candidates.append((moves, info))
        greedy_seeded = True
        if info.get("solved"):
            return moves, info

    standard = ["weighted-a*"] if macro else _policy_order(
        policy, "standard", ["weighted-a*", "beam"])
    for index, phase in enumerate(standard):
        remaining_time = deadline - time.monotonic()
        if remaining_time <= .18:
            break
        fallback = .62 if phase == "weighted-a*" else .38
        weight = _policy_weight(policy, phase, fallback)
        limit = max(.1, remaining_time - .1) if index == len(standard) - 1 else max(
            .1, min(remaining_time - .1, remaining_time * weight))
        run_started = time.monotonic()
        progress(phase, {"event": "start", "attempt": attempt + 1,
                         "remaining": start_remaining,
                         "budget_ms": int(limit * 1000)})
        if phase == "weighted-a*":
            node_limit = max(
                90_000,
                min(360_000, int(90_000 * max(1.0, limit / 3.0))),
            )
            moves, info = weighted_astar_solve(
                board, max_nodes=node_limit, time_limit=limit, cancel=cancel)
        else:
            moves, info = beam_solve(
                board, width=3500, max_depth=260,
                time_limit=limit, seen_cap=500_000, cancel=cancel)
        if info.get("cancelled"):
            raise PlanningCancelled("求解已暂停")
        remaining, _ = apply_moves(board, moves)
        info = {**info, "remaining": remaining, "solved": remaining == 0}
        _strategy_result(progress, phase, run_started, start_remaining,
                         remaining, info, attempt, limit)
        candidates.append((moves, info))
        if info.get("solved"):
            return moves, info

    if not greedy_seeded:
        current_best = min((info.get("remaining", start_remaining)
                            for _moves, info in candidates), default=start_remaining)
        run_started = time.monotonic()
        progress("greedy", {"event": "start", "attempt": attempt + 1,
                            "remaining": current_best})
        moves, info = greedy_solve(board, cancel=cancel)
        if info.get("cancelled"):
            raise PlanningCancelled("求解已暂停")
        remaining, _ = apply_moves(board, moves)
        info = {**info, "kind": "greedy", "remaining": remaining,
                "solved": remaining == 0}
        _strategy_result(progress, "greedy", run_started, start_remaining,
                         remaining, info, attempt, 0.0)
        candidates.append((moves, info))
    moves, info = _best_candidate(board, candidates)
    return moves, {**info, "timeout": time.monotonic() >= deadline}


def solve_board(board: Board, timeout_s: float = 10.0, *,
                cancel: Cancelled | None = None,
                progress: Progress | None = None,
                exit_priority: ExitPriority | None = None,
                online_dynamic: bool = False,
                elastic_timeout: bool = False,
                extension_s: float = 5.0,
                max_timeout_s: float | None = None,
                strategy_policy: dict | None = None) -> PlanResult:
    """Solve with a deterministic exit prefix followed by one search policy."""
    cancel = cancel or _never_cancelled
    progress = progress or _noop_progress
    initial_s = max(.1, float(timeout_s))
    budget = ElasticBudget(
        initial_s=initial_s,
        extension_s=max(.1, float(extension_s)),
        max_s=max(initial_s, float(max_timeout_s or initial_s)),
        enabled=bool(elastic_timeout),
    )
    progress("solve-budget", {
        "event": "budget-start",
        "remaining": board.remaining_count(),
        **budget.info(),
    })
    current = board
    coarse: list[Move] = []

    if supports_forced_exit_closure(board):
        progress("exit-closure", {"event": "progress", "steps": 0,
                                  "remaining": current.remaining_count()})
        while not current.is_solved() and time.monotonic() < budget.deadline:
            if cancel():
                raise PlanningCancelled("求解已暂停")
            exits = forced_exit_candidates(current)
            if not exits:
                break
            exits.sort(key=lambda move: _planner_exit_sort_key(
                current, move, exit_priority))
            move = exits[0]
            coarse.append(move)
            current = current.apply(move)
            progress("exit-closure", {
                "event": "progress", "steps": len(coarse),
                "remaining": current.remaining_count()})

    if online_dynamic and coarse and not current.is_solved():
        refine: list[Move] = []
        info = {"solved": False, "kind": "dynamic-online-prefix",
                "remaining": current.remaining_count(), "partial": True,
                "timeout": False}
    else:
        candidates: list[tuple[list[Move], dict]] = []
        attempt = 0
        while True:
            refine, info = _refine(
                current, budget.deadline, cancel, progress,
                policy=strategy_policy, attempt=attempt,
            )
            candidates.append((refine, info))
            if info.get("solved"):
                break
            expired = bool(info.get("timeout")) or budget.remaining() <= .02
            if not expired:
                break
            added = budget.extend()
            if added <= 0:
                break
            attempt += 1
            progress("budget-extension", {
                "event": "extension",
                "attempt": attempt + 1,
                "added_ms": int(added * 1000),
                "remaining": int(info.get("remaining", current.remaining_count())),
                **budget.info(),
            })
        if len(candidates) > 1 and not info.get("solved"):
            refine, info = _best_candidate(current, candidates)

    steps = [(move, "coarse") for move in coarse]
    steps.extend((move, "refine") for move in refine)
    remaining, final_board = apply_moves(board, [move for move, _ in steps])
    solved = remaining == 0
    timed_out = bool(not solved and (info.get("timeout") or budget.remaining() <= .02))
    kind = f"粗解{len(coarse)} + {info.get('kind', '精解')}"
    return PlanResult(
        steps=steps,
        final_board=final_board,
        solved=solved,
        remaining=remaining,
        kind=kind,
        timed_out=timed_out,
        info={**info, "solved": solved, "remaining": remaining,
              "timeout": timed_out, "budget": budget.info()},
    )
