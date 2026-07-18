from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core import analysis as analysis_engine
from solver import planner
from core import runtime
from solver import learning as solver_learning
from solver import Board


def test_coordinator_has_one_owner_and_pause_finishes_current_unit():
    coordinator = runtime.OperationCoordinator()
    entered = threading.Event()
    release = threading.Event()

    def worker(context):
        context.publish(phase=runtime.Phase.EXECUTING, detail="click")
        entered.set()
        release.wait(2.0)  # one already-started click is atomic
        context.checkpoint()
        return {"unexpected": True}

    job = coordinator.start("auto", worker)
    assert entered.wait(1.0)
    with pytest.raises(runtime.OperationBusy):
        coordinator.start("solve", lambda _context: None)
    pausing = coordinator.cancel()
    assert pausing["phase"] == "pausing" and pausing["busy"]
    release.set()
    assert coordinator.wait(2.0)
    done = coordinator.snapshot(job["id"])
    assert done["phase"] == "cancelled" and not done["busy"]


def test_coordinator_snapshot_is_detached_from_internal_state():
    coordinator = runtime.OperationCoordinator()
    state = coordinator.snapshot()
    state["phase"] = "corrupted"
    assert coordinator.snapshot()["phase"] == "idle"


def test_coordinator_preserves_structured_error_guidance():
    coordinator = runtime.OperationCoordinator()

    class GuidedError(RuntimeError):
        payload = {"error_code": "manual_review_required", "piece_id": "7"}

    job = coordinator.start("auto", lambda _context: (_ for _ in ()).throw(
        GuidedError("请复核棋子 #7")))
    assert coordinator.wait(2.0)
    failed = coordinator.snapshot(job["id"])
    assert failed["phase"] == "error", failed
    assert failed["error"]["error_code"] == "manual_review_required", failed
    assert failed["error"]["piece_id"] == "7", failed


def test_planner_uses_exact_policy_for_small_interactive_board():
    board = Board(3, 5, {
        "A": {"cells": [(1, 3), (1, 4)], "facing": "R", "species": "pink_sheep"},
    }, model="facing")
    phases = []
    result = planner.solve_board(
        board, timeout_s=2.0,
        progress=lambda phase, _data: phases.append(phase))
    assert result.solved and result.remaining == 0
    assert result.info["kind"] == "A*最优" and "exact-a*" in phases


def test_elastic_budget_extends_once_and_keeps_process_observable(monkeypatch):
    board = Board(1, 3, {
        "A": {"cells": [(0, 0), (0, 1)], "facing": "R", "species": "pink_sheep"},
    }, model="facing")
    move = next(item for item in board.legal_moves() if item.piece_id == "A")
    attempts = []
    phases = []

    def fake_refine(current, _deadline, _cancel, _progress, *, policy=None, attempt=0):
        attempts.append((attempt, policy))
        if attempt == 0:
            return [], {"solved": False, "remaining": 1, "kind": "first", "timeout": True}
        return [move], {"solved": True, "remaining": 0, "kind": "second", "timeout": False}

    monkeypatch.setattr(planner, "_refine", fake_refine)
    result = planner.solve_board(
        board, timeout_s=.1, elastic_timeout=True, extension_s=.1,
        max_timeout_s=.2, strategy_policy={"samples": 4},
        progress=lambda phase, data: phases.append((phase, data)),
    )

    assert result.solved and result.remaining == 0
    assert [item[0] for item in attempts] == [0, 1]
    assert all(item[1] == {"samples": 4} for item in attempts)
    assert result.info["budget"]["extensions"] == 1
    assert any(phase == "budget-extension" and data["added_ms"] == 100
               for phase, data in phases)


def test_incremental_strategy_profile_changes_order_without_touching_planner_state():
    profile = {
        "solves": 8,
        "strategies": {
            "macro-beam": {"attempts": 4, "solved": 0, "progress_total": .4,
                           "elapsed_ms_total": 20_000},
            "randomized-macro": {"attempts": 4, "solved": 3, "progress_total": 3.5,
                                 "elapsed_ms_total": 12_000},
            "weighted-a*": {"attempts": 4, "solved": 0, "progress_total": .2,
                            "elapsed_ms_total": 24_000},
            "beam": {"attempts": 4, "solved": 2, "progress_total": 3.0,
                     "elapsed_ms_total": 10_000},
        },
    }
    policy = solver_learning.policy_from_profile(profile)
    assert policy["orders"]["macro"][0] == "randomized-macro"
    assert policy["orders"]["standard"][0] == "beam"
    assert all(.2 <= value <= .8 for value in policy["time_weights"].values())


def test_learning_observations_accumulate_per_strategy():
    state = solver_learning._empty_state()
    observation = {
        "profile_key": "test", "features": {"piece_count": 10},
        "created_at": "2026-07-16T00:00:00+0800", "initial_remaining": 10,
        "solved": True, "remaining": 0,
        "trace": [{"phase": "weighted-a*", "event": "finish",
                   "start_remaining": 10, "remaining": 0, "solved": True,
                   "elapsed_ms": 500, "expanded": 120}],
    }
    solver_learning._apply_observation(state, observation)
    solver_learning._apply_observation(state, observation)
    profile = state["profiles"]["test"]
    stats = profile["strategies"]["weighted-a*"]
    assert profile["solves"] == 2 and profile["solved"] == 2
    assert stats["attempts"] == 2 and stats["solved"] == 2
    assert stats["expanded_total"] == 240 and stats["progress_total"] == 2.0


def test_analysis_without_calibration_is_a_normal_blocked_result():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    bundle = analysis_engine.analyze_image(image, None)
    assert not bundle.calibrated and not bundle.gameplay
    assert bundle.report["execution_blockers"]
    assert bundle.report["executable"] is False
