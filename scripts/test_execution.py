"""Regression tests for adaptive batch click timing."""
from __future__ import annotations

import base64
import json
from collections import deque
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import level_cache
import numpy as np
import app as app_module
import recognition
from app import Api, ExecutionReviewRequired
from solver import Board


def test_host_window_tracks_reference_mode_and_restores_operator_rect(monkeypatch):
    class FakeWindow:
        def __init__(self):
            self.x, self.y = 80, 48
            self.width, self.height = 1320, 900
            self.calls = []

        def restore(self):
            self.calls.append(("restore",))

        def resize(self, width, height):
            self.width, self.height = width, height
            self.calls.append(("resize", width, height))

        def move(self, x, y):
            self.x, self.y = x, y
            self.calls.append(("move", x, y))

    api = Api()
    window = FakeWindow()
    monkeypatch.setenv("SHEEP_DISABLE_HOTKEYS", "1")
    api.start_hotkeys(window)
    monkeypatch.setattr(api, "_reference_window_rect", lambda: (220, 16, 460, 820))

    reference = api.set_window_mode("reference")
    assert reference == {"ok": True, "mode": "reference", "width": 460, "height": 820}
    assert (window.width, window.height) == (460, 820)
    assert (window.x, window.y) == (220, 16)
    assert api._operator_window_rect == (80, 48, 1320, 900)

    operator = api.set_window_mode("operator")
    assert operator == {"ok": True, "mode": "operator", "width": 1320, "height": 900}
    assert (window.x, window.y, window.width, window.height) == (80, 48, 1320, 900)
    assert window.calls[-2:] == [("resize", 1320, 900), ("move", 80, 48)]


def test_structural_deadlock_is_reported_as_a_reviewable_direction_conflict():
    api = Api.__new__(Api)
    suspicion = api._solution_suspicion(
        [], False, 2,
        {"kind": "结构死锁", "structural_deadlocks": [{"pieces": ["7", "12"]}]},
    )
    assert suspicion["type"] == "structural_conflict", suspicion
    assert "7 ↔ 12" in suspicion["message"] and "复核" in suspicion["message"], suspicion


def _exit_motions(board, piece_ids):
    current = board
    motions = []
    for piece_id in piece_ids:
        move = next(item for item in current.legal_moves()
                    if item.piece_id == piece_id and item.result == "EXIT")
        motions.append(Api._move_motion(current, move))
        current = current.apply(move)
    return motions


def test_same_corridor_waits_for_leading_sheep():
    board = Board(4, 12, {
        "front": {"cells": [(1, 4), (1, 5)], "facing": "R"},
        "rear": {"cells": [(1, 2), (1, 3)], "facing": "R"},
    }, model="facing")
    gaps, reasons = Api._burst_gap_schedule(
        _exit_motions(board, ["front", "rear"]), 70)
    assert gaps == [344], (gaps, reasons)
    assert reasons[0][0]["kind"] == "same_corridor", reasons
    assert reasons[0][0]["travel_steps"] == 8, reasons


def test_independent_rows_keep_fast_interval():
    board = Board(4, 12, {
        "top": {"cells": [(0, 4), (0, 5)], "facing": "R"},
        "bottom": {"cells": [(3, 4), (3, 5)], "facing": "R"},
    }, model="facing")
    gaps, reasons = Api._burst_gap_schedule(
        _exit_motions(board, ["top", "bottom"]), 70)
    assert gaps == [70], (gaps, reasons)
    assert reasons == [[]], reasons


def test_independent_exit_is_promoted_ahead_of_corridor_wait():
    board = Board(4, 12, {
        "front": {"cells": [(1, 4), (1, 5)], "facing": "R"},
        "rear": {"cells": [(1, 2), (1, 3)], "facing": "R"},
        "other": {"cells": [(3, 4), (3, 5)], "facing": "R"},
    }, model="facing")
    current = board
    planned = []
    for piece_id in ["front", "rear", "other"]:
        move = next(item for item in current.legal_moves()
                    if item.piece_id == piece_id and item.result == "EXIT")
        planned.append(move)
        current = current.apply(move)
    ordered, reorders = Api._schedule_wait_avoiding_exits(board, planned, 35)
    assert [item.piece_id for item in ordered] == ["front", "other", "rear"], ordered
    assert reorders == [{
        "deferred_piece": "rear", "preferred_piece": "other",
        "avoided_wait_ms": 309,
    }], reorders


def test_wolf_track_exit_is_promoted_ahead_of_safe_exit():
    board = Board(4, 8, {
        "safe": {"cells": [(0, 4), (0, 5)], "facing": "R"},
        "risk": {"cells": [(3, 4), (3, 5)], "facing": "R"},
    }, model="facing")
    safe = next(move for move in board.legal_moves() if move.piece_id == "safe")
    after_safe = board.apply(safe)
    risk = next(move for move in after_safe.legal_moves() if move.piece_id == "risk")
    ordered, reorders = Api._schedule_wait_avoiding_exits(
        board, [safe, risk], 35,
        wolf_track={(3, col) for col in range(board.cols)})
    assert [move.piece_id for move in ordered] == ["risk", "safe"], ordered
    assert reorders[0]["reason"] == "wolf_track_priority", reorders


def test_consecutive_wolf_frames_infer_patrol_lane_and_risky_exit():
    observations = [
        {"wolf": {"components": [{"kind": "small", "center_rect": [96, 96]}]},
         "hazards": [[1, 1]]},
        {"wolf": {"components": [{"kind": "small", "center_rect": [118, 97]}]},
         "hazards": [[1, 1], [1, 2]]},
        {"wolf": {"components": [{"kind": "small", "center_rect": [142, 98]}]},
         "hazards": [[1, 2]]},
    ]
    motion = Api._wolf_motion_summary(observations, rows=4, cols=8)
    assert motion["observed"] and motion["tracks"][0]["axis"] == "H", motion
    assert motion["tracks"][0]["direction"] == "R", motion
    assert motion["track_cells"] == [[1, col] for col in range(8)], motion

    board = Board(4, 8, {
        "risk": {"cells": [(1, 4), (1, 5)], "facing": "R"},
    }, model="facing")
    move = next(iter(board.legal_moves()))
    risk = Api._move_wolf_risk(board, move, motion["track_cells"])
    assert risk["risky"] and risk["overlap"], risk


def test_stationary_wolf_history_is_not_drawn_as_a_patrol_lane():
    observations = [
        {"wolf": {"components": [{"kind": "small", "center_rect": [96, 96]}]},
         "hazards": [[1, 1]]},
        {"wolf": {"components": [{"kind": "small", "center_rect": [99, 97]}]},
         "hazards": [[1, 1], [1, 2]]},
        {"wolf": {"components": [{"kind": "small", "center_rect": [97, 95]}]},
         "hazards": [[1, 2]]},
    ]
    motion = Api._wolf_motion_summary(observations, rows=4, cols=8)
    assert motion["present"] and not motion["observed"], motion
    assert motion["track_cells"] == [], motion
    assert motion["current_cells"] == [[1, 2]], motion


def test_two_same_kind_wolves_keep_nearest_frame_to_frame_identity():
    observations = [
        {"wolf": {"components": [
            {"kind": "small", "center_rect": [96, 96]},
            {"kind": "small", "center_rect": [400, 480]},
        ]}, "hazards": []},
        {"wolf": {"components": [
            {"kind": "small", "center_rect": [421, 481]},
            {"kind": "small", "center_rect": [118, 97]},
        ]}, "hazards": []},
        {"wolf": {"components": [
            {"kind": "small", "center_rect": [142, 98]},
            {"kind": "small", "center_rect": [444, 482]},
        ]}, "hazards": []},
    ]
    motion = Api._wolf_motion_summary(observations, rows=9, cols=8)
    assert motion["observed"] and len(motion["tracks"]) == 2, motion
    assert {item["axis"] for item in motion["tracks"]} == {"H"}, motion
    assert {tuple(item["cells"][0]) for item in motion["tracks"]} == {(1, 0), (7, 0)}, motion


def test_wolf_patrol_zone_fills_corridor_between_sheep_walls():
    pieces = [
        {"cells": [[2, 1], [3, 1]]},
        {"cells": [[2, 6], [3, 6]]},
    ]
    motion = {
        "current_cells": [[2, 3], [2, 4], [3, 3], [3, 4]],
        "tracks": [{
            "observed": True, "axis": "H", "direction": "R",
            "center_rect": [4 * 64, 3 * 64], "cells": [],
        }],
    }
    zone = Api._wolf_patrol_zone(pieces, motion, rows=6, cols=8)
    expected = {(r, c) for r in (2, 3) for c in range(2, 6)}
    assert zone == expected, (zone, expected)
    assert motion["tracks"][0]["cells"] == [[2, c] for c in range(2, 6)], motion

    crossing_board = Board(7, 5, {
        "crossing": {"cells": [[0, 3], [1, 3]], "facing": "D"},
    }, model="facing")
    crossing_move = next(iter(crossing_board.legal_moves()))
    crossing_risk = Api._move_wolf_risk(
        crossing_board, crossing_move, {(3, 3), (4, 3)})
    assert crossing_risk["risky"], crossing_risk
    assert crossing_risk["overlap"] == [[3, 3], [4, 3]], crossing_risk


def test_sandbox_payload_marks_wolf_track_and_risk_piece():
    api = Api()
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    api._wolf_motion = {
        "present": True, "observed": True,
        "track_cells": [[1, col] for col in range(6)],
        "current_cells": [[1, 0]], "tracks": [], "sample_count": 3,
    }
    api._wolf_danger_cells = {(1, col) for col in range(6)}
    api.debug = {"hazards": [
        {"row": 1, "col": 0, "kind": "wolf_body"},
        {"row": 1, "col": 1, "kind": "wolf_body"},
    ]}
    board = Board(3, 6, {
        "safe": {"cells": [(0, 3), (0, 4)], "facing": "R"},
        "risk": {"cells": [(1, 3), (1, 4)], "facing": "R"},
    }, model="facing")
    safe = next(move for move in board.legal_moves() if move.piece_id == "safe")
    after_safe = board.apply(safe)
    risk_move = next(move for move in after_safe.legal_moves() if move.piece_id == "risk")
    final = after_safe.apply(risk_move)
    api.board_revision = level_cache.board_hash(api._board_data(board))
    payload = api._build_solution_payload(
        board, [(safe, "coarse"), (risk_move, "coarse")], final, True, 0,
        "test", 1.0, False, api.Minv)
    assert payload["wolf_track"] == [[1, col] for col in range(6)], payload
    assert payload["wolf_risk_total"] == 1, payload
    assert payload["moves"][0]["wolf_risk"], payload["moves"][0]
    assert payload["moves"][0]["piece"] == "risk", payload["moves"]
    assert payload["states"][0]["hazards"] == [], payload["states"][0]
    assert payload["states"][0]["dynamic_hazards"] == [[1, 0], [1, 1]], payload["states"][0]
    assert payload["states"][1]["dynamic_hazards"] == [], payload["states"][1]
    assert payload["states"][-1]["dynamic_hazards"] == [], payload["states"][-1]


def test_unconfirmed_wolf_cells_replace_instead_of_accumulating():
    api = Api()
    api.rows, api.cols = 5, 7
    api.sheep = []
    first = {
        "wolf_meta": {"components": [
            {"kind": "small", "center_rect": [96, 96]},
        ]},
        "hazards": [{"row": 1, "col": 1, "kind": "wolf_body"}],
    }
    api._remember_wolf_observation(first)
    assert api._wolf_danger_cells == {(1, 1)}, api._wolf_danger_cells

    second = {
        "wolf_meta": {"components": [
            {"kind": "small", "center_rect": [99, 97]},
        ]},
        "hazards": [{"row": 1, "col": 2, "kind": "wolf_body"}],
    }
    api._remember_wolf_observation(second)
    assert api._wolf_danger_cells == {(1, 2)}, api._wolf_danger_cells
    assert not api._wolf_confirmed_cells, api._wolf_confirmed_cells


def test_execution_step_persists_piece_and_pre_click_board():
    board = Board(3, 5, {
        "S": {"cells": [(1, 1), (1, 2)], "facing": "R"},
    }, model="facing")
    move = next(item for item in board.legal_moves() if item.piece_id == "S")
    board_data = {
        "rows": board.rows, "cols": board.cols, "model": board.model,
        "slide_mode": board.slide_mode, "hazards": [], "fences": [],
        "returning": {},
        "pieces": {"S": {"cells": [[1, 1], [1, 2]], "facing": "R", "species": "sheep"}},
    }
    piece = {"id": "S", **board_data["pieces"]["S"]}
    move_data = {"piece_id": "S", "direction": move.direction,
                 "result": move.result, "distance": move.distance}
    old_cache_dir, old_root = level_cache.CACHE_DIR, level_cache.ROOT
    try:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            level_cache.ROOT = root
            level_cache.CACHE_DIR = root / "cache" / "levels"
            saved = level_cache.save_execution_step(
                board_data, piece, move_data, level_key="test-level")
            path = root / saved["path"]
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["step"] == 1 and data["piece"]["id"] == "S", data
            assert data["board_before"] == board_data, data
            assert (level_cache.CACHE_DIR / "test-level" / "execution_index.jsonl").exists()
    finally:
        level_cache.CACHE_DIR, level_cache.ROOT = old_cache_dir, old_root


def test_review_error_identifies_piece_and_board_cells():
    error = ExecutionReviewRequired("0", [(0, 0), (0, 1)], "detector_facing_disagreement")
    assert error.payload["piece_id"] == "0", error.payload
    assert error.payload["location"] == "A1–B1", error.payload
    assert error.payload["review_required"], error.payload
    assert "#0" in str(error) and "A1–B1" in str(error), str(error)
    assert "确认或修正" in str(error), str(error)


def test_low_confidence_exit_is_deprioritized_without_blocking_the_board():
    api = Api()
    board = Board(3, 6, {
        "review": {"cells": [(0, 4), (0, 5)], "facing": "R"},
        "safe": {"cells": [(2, 0), (2, 1)], "facing": "L"},
    }, model="facing")
    api._species_by_id = {
        "review": {"species": "sheep", "review": True},
        "safe": {"species": "sheep", "review": False},
    }
    moves = {str(move.piece_id): move for move in board.legal_moves()}
    assert api._execution_exit_priority(board, moves["safe"]) < (
        api._execution_exit_priority(board, moves["review"]))

    ordered, reorders = Api._schedule_wait_avoiding_exits(
        board, [moves["review"], moves["safe"]], 60,
        review_ids={"review"})
    assert [str(move.piece_id) for move in ordered] == ["safe", "review"], ordered
    assert reorders[0]["reason"] == "low_confidence_avoidance", reorders


def test_review_pause_is_a_guided_success_result_not_an_execution_error():
    api = Api()
    api.scene_report = {
        "scene_state": "gameplay", "execution_blockers": [],
        "advisories": [], "executable": True,
    }
    payload = api._review_pause_payload(
        "7", [(2, 3), (2, 4)], "detector_facing_disagreement",
        steps_completed=2)
    assert payload["ok"] and payload["review_required"], payload
    assert payload["execution_paused_for_review"], payload
    assert payload["piece_id"] == "7" and payload["location"] == "D3–E3", payload
    assert payload["steps_completed"] == 2 and not payload["execution_blockers"], payload


def test_direct_execution_pauses_for_review_before_any_mouse_click():
    api = Api()
    board = Board(3, 6, {
        "7": {"cells": [(1, 4), (1, 5)], "facing": "R"},
    }, model="facing")
    move = next(item for item in board.legal_moves() if item.piece_id == "7")
    api.board = board
    api.Minv = np.eye(3, dtype=np.float64)
    api.sheep = [{
        "id": "7", "cells": [[1, 4], [1, 5]], "facing": "R",
        "species": "sheep", "review": True,
        "review_reason": "detector_facing_disagreement",
    }]
    api._sync_species()
    api.scene_report = {
        "scene_state": "gameplay", "execution_blockers": [],
        "advisories": [], "executable": True,
    }
    api.board_revision = level_cache.board_hash(api._board_data(board))
    api._active_plan = {
        "revision": api.board_revision, "moves": [move], "complete": True,
        "execution_scope": "complete",
    }
    result = api._execute_trusted_plan_direct(
        SimpleNamespace(checkpoint=lambda: None), 1, 60, 70, 8000)
    assert result["ok"] and result["review_required"], result
    assert result["piece_id"] == "7" and result["steps_completed"] == 0, result


def test_tutorial_gesture_is_advisory_while_other_safety_blockers_remain_hard():
    report = {
        "scene_state": "gameplay", "executable": False,
        "execution_blockers": [{
            "code": "manual_learning_confirmation_required",
            "message": "单样本学习候选需再次确认",
        }],
        "advisories": [{
            "code": "gesture_occlusion",
            "detail": {"components": [{
                "tutorial_target_rect": [100, 120], "target_confidence": .92,
            }]},
        }],
    }

    assert not Api._execution_allowed(report)
    assert [item["code"] for item in Api._hard_execution_blockers(report)] == [
        "manual_learning_confirmation_required"]


def test_tutorial_hand_remains_transient_with_unrelated_advisories():
    report = {
        "scene_state": "gameplay", "executable": True,
        "execution_blockers": [],
        "advisories": [
            {"code": "gesture_occlusion"},
            {"code": "dynamic_scene", "message": "采集稳定度降低"},
        ],
    }

    assert Api._execution_allowed(report)
    assert Api._visual_transient_only(report)


def test_single_move_wait_cannot_be_shortened_below_animation_time():
    exit_move = SimpleNamespace(result="EXIT", distance=5)
    slide_move = SimpleNamespace(result="MOVE", distance=4)
    goat_move = SimpleNamespace(result="STEP", distance=3)

    assert Api._single_settle_ms(exit_move, {"species": "sheep"}, 320) == 900
    assert Api._single_settle_ms(slide_move, {"species": "sheep"}, 320) >= 1300
    assert Api._single_settle_ms(goat_move, {"species": "goat"}, 320) >= 1200
    assert Api._single_settle_ms(slide_move, {"species": "bomb"}, 320) >= 1400


def test_execution_preflight_waits_for_motion_smoke_to_clear():
    api = Api()
    smoke = {"scene_state": "gameplay", "executable": False,
             "execution_blockers": [{"code": "motion_smoke"}]}
    clean = {"scene_state": "gameplay", "executable": True,
             "execution_blockers": []}
    reports = iter([smoke, clean])
    waits = []
    captures = []

    def capture(*, require_same_window):
        captures.append(require_same_window)
        return (1, 2, 3, 4), "test"

    api._capture_live = capture
    api._analyze_frame = lambda *, source: next(reports)
    api._wait_or_cancel = lambda seconds: waits.append(seconds)

    rectinfo, mode, report, retries = api._capture_execution_preflight(
        "test-preflight", max_wait_ms=1000, retry_interval_ms=10)

    assert rectinfo == (1, 2, 3, 4) and mode == "test", (rectinfo, mode)
    assert report is clean and retries == 1, (report, retries)
    assert captures == [True, True] and len(waits) == 1, (captures, waits)


def test_confirmed_wolf_lane_does_not_add_two_frames_to_every_preflight():
    api = Api()
    api.debug = {"wolf_meta": {"components": [{"kind": "small"}]}}
    api._wolf_motion = {"observed": True, "tracks": [{"axis": "H"}]}
    captures = []
    api._capture_live = lambda *, require_same_window: (
        captures.append(require_same_window) or ((1, 2, 3, 4), "test"))
    api._analyze_frame = lambda *, source: {
        "scene_state": "gameplay", "executable": True, "execution_blockers": [],
    }
    api._wait_or_cancel = lambda _seconds: None

    _rectinfo, _mode, report, retries = api._capture_execution_preflight("test")
    assert report["executable"] and retries == 0, report
    assert captures == [True], captures


def test_retry_control_score_distinguishes_blue_and_red_controls():
    img = np.zeros((1000, 500, 3), dtype=np.uint8)
    img[700:880, 360:470] = (255, 120, 20)  # saturated blue in BGR
    img[160:300, 430:495] = (20, 20, 255)

    blue = Api._retry_control_score(img, (.72, .70, .94, .88), "blue")
    red = Api._retry_control_score(img, (.86, .16, .99, .30), "red")

    assert blue > .9, blue
    assert red > .9, red
    assert Api._retry_control_score(img, (.72, .70, .94, .88), "red") == 0.0


def test_retry_control_score_recognizes_green_next_level_button():
    img = np.zeros((1000, 500, 3), dtype=np.uint8)
    img[830:930, 155:345] = (45, 225, 45)

    score = Api._retry_control_score(img, (.31, .83, .69, .93), "green")

    assert score > .9, score


def test_advance_level_requires_victory_then_waits_for_gameplay():
    api = Api()
    victory_img = np.zeros((1000, 500, 3), dtype=np.uint8)
    victory_img[830:930, 155:345] = (45, 225, 45)
    gameplay_img = np.full((1000, 500, 3), 160, dtype=np.uint8)
    frames = iter([victory_img, gameplay_img])
    reports = iter([
        {"scene_state": "victory", "execution_complete": True,
         "execution_blockers": [], "executable": False},
        {"scene_state": "gameplay", "execution_complete": False,
         "execution_blockers": [], "executable": True, "count": 3,
         "rows": 18, "cols": 12},
    ])
    clicks = []

    api._capture_live = lambda **_kwargs: (
        setattr(api, "game", next(frames)) or (0, 0, 500, 1000), "test")
    api._analyze_frame = lambda **_kwargs: next(reports)
    api._click_window_ratio = lambda x, y, hold_ms=70: (
        clicks.append((x, y)) or {"ratio": [x, y], "hwnd": 1})
    api._wait_or_cancel = lambda _seconds: None

    result = api.advance_level(1000)

    assert result["ok"] and result["scene_state"] == "gameplay", result
    assert clicks == [(0.50, 0.875)], clicks
    assert result["next_level_green_score"] >= .9, result


def test_in_level_retry_uses_gear_then_settings_restart():
    api = Api()
    gameplay = np.zeros((1000, 500, 3), dtype=np.uint8)
    settings = np.zeros((1000, 500, 3), dtype=np.uint8)
    settings[600:780, 360:480] = (255, 120, 20)
    clicks = []
    frames = iter([gameplay, settings, settings])
    api._capture_live = lambda **_kwargs: (
        setattr(api, "game", next(frames)) or (0, 0, 500, 1000), "test")
    api._click_window_ratio = lambda x, y, hold_ms=70: (
        clicks.append((x, y)) or {"at": [0, 0], "ratio": [x, y], "hwnd": 1})
    api._wait_or_cancel = lambda _seconds: None
    api._analyze_frame = lambda **_kwargs: {
        "scene_state": "gameplay", "execution_blockers": [],
        "executable": True, "count": 1, "rows": 18, "cols": 12,
    }

    result = api.retry_level("in_level")

    assert result["ok"] is True, result
    assert clicks == [(.12, .08), (.84, .69)], clicks
    assert [item["action"] for item in result["retry_actions"]] == [
        "open_settings", "restart_in_level"]


def test_failure_retry_handles_bomb_popup_over_settings():
    api = Api()
    bomb = np.zeros((1000, 500, 3), dtype=np.uint8)
    bomb[160:300, 430:495] = (20, 20, 255)
    settings = np.zeros((1000, 500, 3), dtype=np.uint8)
    settings[600:780, 360:480] = (255, 120, 20)
    frames = iter([bomb, settings, settings])
    clicks = []
    api._capture_live = lambda **_kwargs: (
        setattr(api, "game", next(frames)) or (0, 0, 500, 1000), "test")
    api._click_window_ratio = lambda x, y, hold_ms=70: (
        clicks.append((x, y)) or {"at": [0, 0], "ratio": [x, y], "hwnd": 1})
    api._wait_or_cancel = lambda _seconds: None
    api._analyze_frame = lambda **_kwargs: {
        "scene_state": "gameplay", "execution_blockers": [],
        "executable": True, "count": 1, "rows": 18, "cols": 12,
    }

    result = api.retry_level("failure")

    assert result["ok"] is True, result
    assert clicks == [(.92, .22), (.84, .69)], clicks
    assert [item["action"] for item in result["retry_actions"]] == [
        "close_failure_popup", "restart_after_failure_settings"]


def test_execute_step_exposes_motion_smoke_as_retryable():
    api = Api()
    board = Board(3, 6, {
        "A": {"cells": [(0, 4), (0, 5)], "facing": "R"},
    }, model="facing")
    move = next(iter(board.legal_moves()))
    api.board = board
    api.Minv = np.eye(3, dtype=np.float64)
    api.board_revision = level_cache.board_hash(api._board_data(board))
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    api._active_plan = {"revision": api.board_revision, "moves": [move],
                        "complete": True, "execution_scope": "complete"}
    smoke = {"scene_state": "gameplay", "executable": False,
             "execution_blockers": [{"code": "motion_smoke"}]}
    api._capture_execution_preflight = lambda _source: (
        (0, 0, 100, 100), "test", smoke, 4)

    result = api.execute_step(api.board_revision)

    assert result["ok"] and result["retryable"], result
    assert result["clicked"] is None and result["retry_after_ms"] >= 220, result
    assert not api._execution_lock.locked()


def test_partial_solution_is_preview_only_and_clears_active_plan():
    api = Api()
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    board = Board(3, 6, {
        "A": {"cells": [(0, 4), (0, 5)], "facing": "R"},
        "B": {"cells": [(2, 2), (2, 3)], "facing": "L"},
    }, model="facing")
    move = next(item for item in board.legal_moves()
                if item.piece_id == "A" and item.result == "EXIT")
    final_board = board.apply(move)
    api.board_revision = level_cache.board_hash(api._board_data(board))
    api._active_plan = {"revision": api.board_revision, "moves": [move],
                        "complete": True}

    payload = api._build_solution_payload(
        board, [(move, "refine")], final_board, False, 1,
        "partial", 1.0, False, api.Minv)

    assert not payload["execution_ready"], payload
    assert "仅供沙盘预览" in payload["execution_blocker"], payload
    assert api._active_plan is None, api._active_plan


def test_partial_path_that_ends_without_legal_moves_is_reported_as_dead_end():
    api = Api()
    board = Board(4, 5, {
        "S": {"cells": [(1, 0), (1, 1)], "facing": "L", "species": "sheep"},
    }, model="facing", fences=[{"cell": [1, 0], "direction": "L"}])

    suspicion = api._solution_suspicion(
        [(object(), "refine")], False, 1, {}, final_board=board)

    assert suspicion["type"] == "dead_end", suspicion
    assert "死局" in suspicion["message"], suspicion


def test_complete_solution_publishes_execution_ready_plan():
    api = Api()
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    board = Board(3, 6, {
        "A": {"cells": [(0, 4), (0, 5)], "facing": "R"},
    }, model="facing")
    move = next(item for item in board.legal_moves() if item.piece_id == "A")
    final_board = board.apply(move)
    api.board_revision = level_cache.board_hash(api._board_data(board))

    payload = api._build_solution_payload(
        board, [(move, "refine")], final_board, True, 0,
        "complete", 1.0, False, api.Minv)

    assert payload["execution_ready"], payload
    assert Api._plan_is_execution_ready(api._active_plan, api.board_revision)


def test_solution_payload_exposes_bomb_budget_changes():
    api = Api()
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    board = Board(3, 6, {
        "B": {"cells": [(1, 1), (1, 2)], "facing": "R", "species": "bomb",
              "hit_limit": 3, "hits_remaining": 3},
        "S": {"cells": [(1, 4), (1, 5)], "facing": "R", "species": "sheep"},
    }, model="facing")
    move = next(item for item in board.legal_moves()
                if item.piece_id == "B" and item.result == "MOVE")
    final_board = board.apply(move)
    api.board_revision = level_cache.board_hash(api._board_data(board))

    payload = api._build_solution_payload(
        board, [(move, "refine")], final_board, False, 2,
        "bomb-budget", 1.0, False, api.Minv)

    assert payload["bomb_count"] == 1
    assert payload["bomb_min_hits"] == 3
    assert payload["bomb_live_control"]
    assert payload["moves"][0]["bomb_changes"] == [
        {"piece": "B", "before": 3, "after": 2, "event": "hit"},
    ]


def test_burst_advances_special_phase_with_single_step_and_releases_lock():
    api = Api()
    board = Board(3, 6, {
        "G": {"cells": [(1, 4), (1, 5)], "facing": "R", "species": "goat"},
    }, model="facing")
    move = next(iter(board.legal_moves()))
    api.board = board
    api.Minv = np.eye(3, dtype=np.float64)
    api.board_revision = level_cache.board_hash(api._board_data(board))
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    api._active_plan = {"revision": api.board_revision, "moves": [move],
                        "complete": True}
    api.hwnd = 1
    api.sheep = []
    api._capture_live = lambda **_kwargs: ((0, 0, 100, 100), "test")
    api._analyze_frame = lambda **_kwargs: {"executable": True}
    called = {}

    def fallback(revision, move_id, settle_ms, hold_ms, solve_timeout_ms):
        called.update({"revision": revision, "move_id": move_id,
                       "settle_ms": settle_ms, "hold_ms": hold_ms,
                       "solve_timeout_ms": solve_timeout_ms})
        return {"ok": True, "clicked": {"count": 1}, "capture": {},
                "solution": {}}

    api.execute_step = fallback
    result = api.execute_burst(api.board_revision, max_steps=4)

    assert result["ok"] and result["batch_fallback"] == "single_verified_step", result
    assert result["stage_capture"]["source"] == "automatic-stage-capture", result
    assert called["move_id"] == "0", called
    assert not api._execution_lock.locked()


def test_direct_plan_executes_without_per_step_capture_and_verifies_once():
    api = Api()
    board = Board(2, 6, {
        "A": {"cells": [(0, 4), (0, 5)], "facing": "R"},
        "B": {"cells": [(1, 4), (1, 5)], "facing": "R"},
    }, model="facing")
    first = next(move for move in board.legal_moves() if move.piece_id == "A")
    after_first = board.apply(first)
    second = next(move for move in after_first.legal_moves() if move.piece_id == "B")
    api.board = board
    api.Minv = np.eye(3, dtype=np.float64)
    api.game = np.zeros((100, 100, 3), dtype=np.uint8)
    api.board_revision = level_cache.board_hash(api._board_data(board))
    api.scene_report = {
        "scene_state": "gameplay", "execution_blockers": [],
        "executable": True,
        "advisories": [{"code": "gesture_occlusion"}],
    }
    api._active_plan = {
        "revision": api.board_revision, "moves": [first, second],
        "complete": True, "execution_scope": "complete",
    }
    api._species_by_id = {"A": {"species": "sheep"}, "B": {"species": "sheep"}}
    clicks, captures, publishes, waits = [], [], [], []
    api._click_image_point = lambda x, y, hold_ms, before_click=None: (
        before_click() if before_click else None,
        clicks.append((x, y)),
        {"at": [x, y]},
    )[-1]
    api._record_execution_step = lambda *_args, **_kwargs: {"recorded": True}
    api._wait_direct_settle = lambda seconds: waits.append(seconds) or False
    api._capture_live = lambda **_kwargs: (
        captures.append("final") or ((0, 0, 100, 100), "test"))
    api._analyze_frame = lambda **_kwargs: {
        "scene_state": "victory", "execution_complete": True,
        "executable": False, "execution_blockers": [],
        "state": api._snapshot(api.board, highlight=None),
    }
    api._execution_refresh_payload = lambda result, _rect, _mode, _timeout, **extra: {
        "ok": True, **result, **extra,
    }
    context = type("Context", (), {
        "checkpoint": lambda self: None,
        "publish": lambda self, **fields: publishes.append(fields),
    })()

    result = api._execute_trusted_plan_direct(
        context, max_steps=2, settle_ms=60, hold_ms=70,
        solve_timeout_ms=8000, progress_offset=4, progress_total=6)

    assert result["ok"] and result["direct_execution"], result
    assert result["batch_size"] == 2 and len(clicks) == 2, (result, clicks)
    assert waits == [0.06, 0.06], waits
    assert captures == ["final"], captures
    assert api.board.is_solved(), api._board_data(api.board)
    assert [item["progress"]["completed"] for item in publishes[:2]] == [5, 6], publishes
    assert all(item["progress"]["total"] == 6 for item in publishes), publishes
    assert [len(item["preview_state"]["pieces"]) for item in publishes[:2]] == [1, 0], publishes
    assert not api._execution_lock.locked()


def test_opening_coarse_executes_safe_prefix_as_one_adaptive_batch():
    api = Api()
    board = Board(2, 6, {
        "front": {"cells": [(0, 4), (0, 5)], "facing": "R"},
        "rear": {"cells": [(0, 2), (0, 3)], "facing": "R"},
        "G": {"cells": [(1, 4), (1, 5)], "facing": "R", "species": "goat"},
    }, model="facing")
    first = next(move for move in board.legal_moves() if move.piece_id == "front")
    after_first = board.apply(first)
    second = next(move for move in after_first.legal_moves() if move.piece_id == "rear")
    api.board = board
    api.Minv = np.eye(3, dtype=np.float64)
    api.game = np.zeros((100, 100, 3), dtype=np.uint8)
    api.board_revision = level_cache.board_hash(api._board_data(board))
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    api._active_plan = {
        "revision": api.board_revision, "moves": [first, second],
        "complete": False, "execution_scope": "safe_exit_prefix",
    }
    api._species_by_id = {
        "front": {"species": "sheep"}, "rear": {"species": "sheep"},
        "G": {"species": "goat"},
    }
    clicks, captures, waits = [], [], []
    api._click_image_point = lambda x, y, hold_ms, before_click=None: (
        before_click() if before_click else None,
        clicks.append((x, y)),
        {"at": [x, y]},
    )[-1]
    api._record_execution_step = lambda *_args, **_kwargs: {"recorded": True}
    api._wait_direct_settle = lambda seconds: waits.append(seconds) or False
    api._capture_live = lambda **_kwargs: (
        captures.append("final") or ((0, 0, 100, 100), "test"))
    api._analyze_frame = lambda **_kwargs: {
        "scene_state": "gameplay", "execution_complete": False,
        "executable": True, "execution_blockers": [],
        "state": api._snapshot(api.board, highlight=None),
    }
    api._execution_refresh_payload = lambda result, _rect, _mode, _timeout, **extra: {
        "ok": True, **result, **extra,
    }
    context = type("Context", (), {
        "checkpoint": lambda self: None,
        "publish": lambda self, **_fields: None,
    })()

    result = api._execute_trusted_plan_direct(
        context, max_steps=2, settle_ms=60, hold_ms=70,
        solve_timeout_ms=8000, stage="opening-coarse")

    assert result["ok"] and result["opening_coarse"], result
    assert result["batch_size"] == 2 and len(clicks) == 2, (result, clicks)
    assert result["batch_profile"]["gap_schedule_ms"] == [260], result
    assert waits == [.26, .06], waits
    assert captures == ["final"], captures
    assert set(api.board.pieces) == {"G"}, api._board_data(api.board)
    assert not api._execution_lock.locked()


def test_auto_workflow_runs_opening_coarse_before_full_search():
    api = Api()
    board = Board(2, 6, {
        "front": {"cells": [(0, 4), (0, 5)], "facing": "R"},
        "rear": {"cells": [(0, 2), (0, 3)], "facing": "R"},
        "G": {"cells": [(1, 4), (1, 5)], "facing": "R", "species": "goat"},
    }, model="facing")
    api.board = board
    api.Minv = np.eye(3, dtype=np.float64)
    api.board_revision = level_cache.board_hash(api._board_data(board))
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    calls = []

    def direct(_context, max_steps, settle_ms, hold_ms, solve_timeout_ms,
               stage="complete-plan"):
        calls.append({"stage": stage, "max_steps": max_steps})
        return {"ok": True, "scene_state": "victory",
                "execution_complete": True, "steps_completed": max_steps}

    api._execute_trusted_plan_direct = direct
    started = api.workflow_start("auto", {
        "max_steps": 20, "settle_ms": 60, "timeout_ms": 1000,
    })
    assert started["ok"], started
    assert api.runtime.wait(2.0)
    job = api.runtime.snapshot(started["job"]["id"])

    assert job["phase"] == "done" and job["result"]["auto_complete"], job
    assert calls == [{"stage": "opening-coarse", "max_steps": 2}], calls
    assert job["result"]["opening_coarse_steps"] == 2, job
    assert not api._opening_coarse_pending


def test_solve_workflow_publishes_compact_right_panel_plan():
    api = Api()
    board = Board(2, 3, {
        "A": {"cells": [(0, 1), (0, 2)], "facing": "R"},
    }, model="facing")
    api.board = board
    api.Minv = np.eye(3, dtype=np.float64)
    api.board_revision = level_cache.board_hash(api._board_data(board))
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    solution = {
        "total": 1, "solved": True, "remaining": 0, "kind": "test-plan",
        "execution_ready": True, "moves": [{"step": 1, "piece": "A"}],
        "states": [{"large": "initial"}, {"large": "final"}],
    }
    solve_options = {}

    def fake_solve(*_args, **kwargs):
        solve_options.update(kwargs)
        return deepcopy(solution)

    api._solve_with_budget = fake_solve

    started = api.workflow_start("solve", {
        "timeout_ms": 1000, "elastic_timeout": True,
        "timeout_extension_ms": 7000, "timeout_max_ms": 25000,
    })
    assert started["ok"], started
    assert api.runtime.wait(2.0)
    job = api.runtime.snapshot(started["job"]["id"])

    assert job["phase"] == "done", job
    assert job["panel_solution"]["moves"] == solution["moves"], job
    assert "states" not in job["panel_solution"], job["panel_solution"]
    assert job["plan_completed_base"] == 0, job
    assert job["progress"] == {"completed": 1, "total": 1, "remaining": 0}, job
    assert solve_options["elastic_timeout"] is True
    assert solve_options["extension_s"] == 7.0
    assert solve_options["max_timeout_s"] == 25.0


def test_album_upload_has_its_own_reference_only_workflow():
    api = Api()
    received = []

    def upload(_context, encoded_image, file_name=None):
        received.append((encoded_image, file_name))
        api._input_mode = "reference"
        return {"ok": True, "reference_only": True, "scene_state": "gameplay"}

    api._workflow_upload = upload
    started = api.workflow_start("upload", {
        "image_data": "c2NyZWVuc2hvdA==", "file_name": "board.png",
    })
    assert started["ok"], started
    assert api.runtime.wait(2.0)
    job = api.runtime.snapshot(started["job"]["id"])
    assert job["phase"] == "done" and job["result"]["reference_only"], job
    assert received == [("c2NyZWVuc2hvdA==", "board.png")]

    for action in ("quick", "step", "auto"):
        blocked = api.workflow_start(action, {"timeout_ms": 1000})
        assert not blocked["ok"], (action, blocked)
        assert "不能执行桌面点击" in blocked["error"]


def test_album_upload_decodes_pixels_without_touching_window_capture(monkeypatch):
    api = Api()
    image = np.full((420, 360, 3), 127, dtype=np.uint8)
    ok, encoded = app_module.cv2.imencode(".png", image)
    assert ok
    writes = []
    monkeypatch.setattr(app_module.cv2, "imwrite", lambda path, _image: writes.append(str(path)) or True)
    api._read_source_level = lambda *args, **kwargs: {
        "level_label": None, "level_auto_read": False,
        "level_read_method": "unavailable", "level_bbox": None,
    }
    api._analyze_frame = lambda **kwargs: {
        "scene_state": "gameplay", "rows": 18, "cols": 12,
        "count": 24, "state": {"pieces": []}, "source": kwargs["source"],
    }
    published = []
    context = SimpleNamespace(
        publish=lambda **payload: published.append(payload), checkpoint=lambda: None)

    result = api._workflow_upload(
        context, base64.b64encode(encoded.tobytes()).decode("ascii"), "board.png")

    assert result["ok"] and result["reference_only"]
    assert result["capture"] == {"mode": "album", "win": {"w": 360, "h": 420}}
    assert result["source"] == "app-upload"
    assert api._input_mode == "reference" and api.hwnd is None
    assert api.game.shape == (420, 360, 3)
    assert writes and writes[0].endswith("_game.png")
    assert [item["phase"].value for item in published] == ["capturing", "analyzing"]


def test_runtime_settings_are_normalized_persisted_and_newest_write_wins(monkeypatch, tmp_path):
    settings_path = tmp_path / "runtime_settings.json"
    monkeypatch.setattr(app_module, "RUNTIME_SETTINGS_PATH", str(settings_path))
    api = Api()

    defaults = api.load_runtime_settings()
    assert defaults["ok"] and defaults["settings"]["solve_timeout_s"] == 10

    saved = api.save_runtime_settings({
        "solve_timeout_s": 75,
        "timeout_extension_s": 0,
        "timeout_max_s": 20,
        "elastic_timeout": False,
        "settle_ms": 8,
        "max_steps": 999,
        "source_level_label": "  第 188 关  ",
        "updated_at_ms": 200,
    })
    assert saved["ok"], saved
    assert saved["settings"] == {
        "solve_timeout_s": 60,
        "timeout_extension_s": 1,
        "timeout_max_s": 60,
        "elastic_timeout": False,
        "settle_ms": 20,
        "max_steps": 500,
        "source_level_label": "第 188 关",
        "updated_at_ms": 200,
    }
    assert settings_path.exists()

    stale = api.save_runtime_settings({
        "solve_timeout_s": 3, "updated_at_ms": 100,
    })
    assert stale["ok"] and stale["settings"]["solve_timeout_s"] == 60
    loaded = Api().load_runtime_settings()
    assert loaded["settings"] == saved["settings"]


def test_solve_payload_carries_process_trace_budget_and_silent_learning(monkeypatch):
    api = Api()
    board = Board(1, 3, {
        "A": {"cells": [(0, 0), (0, 1)], "facing": "R", "species": "pink_sheep"},
    }, model="facing")
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    api.board_revision = level_cache.board_hash(api._board_data(board))
    learned = []

    def fake_plan(current, _timeout, *, progress, **_kwargs):
        progress("solve-budget", {
            "event": "budget-start", "remaining": 1, "initial_ms": 100,
            "extension_ms": 100, "max_ms": 200, "allocated_ms": 100,
            "extensions": 0, "elastic": True,
        })
        progress("weighted-a*", {
            "event": "start", "attempt": 1, "remaining": 1, "budget_ms": 100,
        })
        progress("weighted-a*", {
            "event": "finish", "attempt": 1, "start_remaining": 1,
            "remaining": 1, "elapsed_ms": 100, "expanded": 40,
            "solved": False,
        })
        progress("budget-extension", {
            "event": "extension", "attempt": 2, "remaining": 1,
            "added_ms": 100, "allocated_ms": 200, "max_ms": 200,
            "extensions": 1, "elastic": True,
        })
        return app_module.planner.PlanResult(
            steps=[], final_board=current, solved=False, remaining=1,
            kind="stub", timed_out=True,
            info={"solved": False, "remaining": 1, "kind": "stub",
                  "timeout": True, "budget": {
                      "initial_ms": 100, "extension_ms": 100, "max_ms": 200,
                      "allocated_ms": 200, "elapsed_ms": 200,
                      "extensions": 1, "elastic": True,
                  }},
        )

    monkeypatch.setattr(level_cache, "load_solution", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(level_cache, "save_solution", lambda *_args, **_kwargs: {
        "best_path": None, "revision_id": "test", "selected_best": False,
        "capture_id": None,
    })
    monkeypatch.setattr(app_module.planner, "solve_board", fake_plan)
    monkeypatch.setattr(app_module.solver_learning, "policy_for", lambda _board: {"samples": 3})
    monkeypatch.setattr(
        app_module.solver_learning, "record_async",
        lambda _board, trace, **result: learned.append((deepcopy(trace), result)),
    )

    payload = api._solve_with_budget(
        board, .1, api.Minv, elastic_timeout=True,
        extension_s=.1, max_timeout_s=.2,
    )

    assert [item["phase"] for item in payload["process_trace"]] == [
        "solve-budget", "weighted-a*", "weighted-a*", "budget-extension",
    ]
    assert payload["budget"]["extensions"] == 1
    assert payload["timeout_ms"] == 200 and payload["max_timeout_ms"] == 200
    assert learned and learned[0][1] == {"solved": False, "remaining": 1}


def test_quick_workflow_reuses_current_board_and_executes_only_three_exit_layers():
    api = Api()
    current = Board(1, 8, {
        "layer4": {"cells": [(0, 0), (0, 1)], "facing": "R"},
        "layer3": {"cells": [(0, 2), (0, 3)], "facing": "R"},
        "layer2": {"cells": [(0, 4), (0, 5)], "facing": "R"},
        "layer1": {"cells": [(0, 6), (0, 7)], "facing": "R"},
    }, model="facing")
    api.board = current
    api.Minv = np.eye(3, dtype=np.float64)
    api.board_revision = level_cache.board_hash(api._board_data(current))
    api.scene_report = {
        "scene_state": "gameplay", "execution_blockers": [],
        "executable": True,
        "advisories": [{"code": "gesture_occlusion"}],
    }
    api._species_by_id = {
        piece_id: {"species": "sheep"} for piece_id in current.pieces
    }
    captures, direct_calls = [], []

    def capture(_context, _source_level_label=None):
        captures.append("unexpected")
        raise AssertionError("已有棋盘时快速解法不应重新采集")

    def direct(_context, max_steps, settle_ms, hold_ms, solve_timeout_ms,
               stage="complete-plan", **_kwargs):
        direct_calls.append({"max_steps": max_steps, "stage": stage})
        return {"ok": True, "state": api._snapshot(api.board, highlight=None),
                "scene_state": "gameplay", "steps_completed": max_steps}

    api._workflow_capture = capture
    api._execute_trusted_plan_direct = direct

    started = api.workflow_start("quick", {"timeout_ms": 1000})
    assert started["ok"], started
    assert api.runtime.wait(2.0)
    job = api.runtime.snapshot(started["job"]["id"])
    assert job["phase"] == "done", job
    solution = job["result"]["solution"]

    assert captures == [], captures
    assert direct_calls == [{"max_steps": 3, "stage": "quick-exit"}], direct_calls
    assert [move["piece"] for move in solution["moves"]] == [
        "layer1", "layer2", "layer3",
    ], solution
    assert [move["exit_layer"] for move in solution["moves"]] == [1, 2, 3], solution
    assert solution["quick_exit"] and solution["remaining"] == 1, solution
    assert job["result"]["solution_history"] is True, job


def test_quick_workflow_captures_only_when_current_board_is_missing():
    api = Api()
    captured = Board(1, 2, {
        "only": {"cells": [(0, 0), (0, 1)], "facing": "R"},
    }, model="facing")
    captures, direct_calls = [], []

    def capture(_context, _source_level_label=None):
        captures.append("fresh")
        api.board = captured
        api.Minv = np.eye(3, dtype=np.float64)
        api.board_revision = level_cache.board_hash(api._board_data(captured))
        api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                            "executable": True}
        api._species_by_id = {"only": {"species": "sheep"}}
        return {"ok": True, **api.scene_report,
                "state": api._snapshot(captured, highlight=None)}

    def direct(_context, max_steps, settle_ms, hold_ms, solve_timeout_ms,
               stage="complete-plan", **_kwargs):
        direct_calls.append({"max_steps": max_steps, "stage": stage})
        return {"ok": True, "state": api._snapshot(api.board, highlight=None),
                "scene_state": "gameplay", "steps_completed": max_steps}

    api._workflow_capture = capture
    api._execute_trusted_plan_direct = direct

    started = api.workflow_start("quick", {"timeout_ms": 1000})
    assert started["ok"], started
    assert api.runtime.wait(2.0)
    job = api.runtime.snapshot(started["job"]["id"])

    assert job["phase"] == "done", job
    assert captures == ["fresh"], captures
    assert direct_calls == [{"max_steps": 1, "stage": "quick-exit"}], direct_calls


def test_wolf_presence_disables_direct_plan_execution():
    api = Api()
    api.debug = {
        "wolf_meta": {"count": 1, "components": [{"kind": "small"}]},
        "hazards": [{"row": 2, "col": 3}],
    }
    assert api._wolf_requires_live_control()


def test_bomb_presence_disables_direct_plan_execution():
    api = Api()
    api.board = Board(3, 6, {
        "B": {"cells": [(1, 4), (1, 5)], "facing": "R", "species": "bomb",
              "hit_limit": 3, "hits_remaining": 1},
    }, model="facing")
    assert api._bomb_requires_live_control()


def test_tutorial_gesture_remains_advisory_without_execution_control():
    api = Api()
    api.scene_report = {
        "scene_state": "gameplay", "executable": True,
        "execution_blockers": [],
        "advisories": [{
            "code": "gesture_occlusion",
            "detail": {"components": [{"tutorial_target_rect": [100, 120]}]},
        }],
    }
    assert api._execution_allowed(api.scene_report)
    assert api._hard_execution_blockers(api.scene_report) == []


def test_confirmed_manual_board_clears_single_sample_execution_blocker():
    api = Api()
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {
        "scene_state": "gameplay", "executable": False,
        "execution_blockers": [{
            "code": "manual_learning_confirmation_required",
            "message": "单样本学习候选需人工确认",
        }],
    }
    api.board = Board(3, 6, {
        "A": {"cells": [(1, 4), (1, 5)], "facing": "R", "species": "sheep"},
    }, model="facing")
    api.sheep = []
    api.debug = {"hazards": [], "fences": []}
    api._write_current_board = lambda: None
    api._rerender_detection_images = lambda: None
    data = api._board_data(api.board)

    result = api._load_editor_board(data, pending=False, confirmed=True)

    assert result["execution_blockers"] == []
    assert result["executable"]


def test_suppressed_dark_component_does_not_enable_wolf_live_control():
    api = Api()
    api.debug = {
        "wolf_meta": {"count": 1, "components": [{"kind": "small"}]},
        "hazards": [],
    }
    assert not api._wolf_requires_live_control()


def test_partial_hint_is_saved_as_non_usable_cache_revision():
    api = Api()
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    board = Board(3, 6, {
        "A": {"cells": [(0, 4), (0, 5)], "facing": "R"},
        "B": {"cells": [(2, 2), (2, 3)], "facing": "L"},
    }, model="facing")
    move = next(item for item in board.legal_moves()
                if item.piece_id == "A" and item.result == "EXIT")
    api.board_revision = level_cache.board_hash(api._board_data(board))
    captured = {}
    old_load, old_save = level_cache.load_solution, level_cache.save_solution
    old_plan = app_module.planner.solve_board
    try:
        level_cache.load_solution = lambda *_args, **_kwargs: None
        final = board.apply(move)
        app_module.planner.solve_board = lambda *_args, **_kwargs: app_module.planner.PlanResult(
            steps=[(move, "refine")], final_board=final, solved=False,
            remaining=1, kind="粗解0 + stub", timed_out=False,
            info={"solved": False, "remaining": 1, "kind": "stub", "timeout": False})

        def fake_save(_board, data, **_kwargs):
            captured.update(data)
            return {"best_path": None, "revision_id": "test", "selected_best": False,
                    "capture_id": None}

        level_cache.save_solution = fake_save
        payload = api._solve_with_budget(board, 1.0, api.Minv)
    finally:
        level_cache.load_solution, level_cache.save_solution = old_load, old_save
        app_module.planner.solve_board = old_plan

    assert payload["result_type"] == "partial_hint", payload
    assert not payload["execution_ready"], payload
    assert captured["usable"] is False, captured
    assert api._active_plan is None, api._active_plan


def test_partial_hint_publishes_only_plain_exit_prefix():
    api = Api()
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    board = Board(3, 8, {
        "A": {"cells": [(0, 6), (0, 7)], "facing": "R", "species": "sheep"},
        "B": {"cells": [(1, 6), (1, 7)], "facing": "R", "species": "sheep"},
        "G": {"cells": [(2, 6), (2, 7)], "facing": "R", "species": "goat"},
    }, model="facing")
    cursor = board
    steps = []
    for piece_id in ("A", "B", "G"):
        move = next(item for item in cursor.legal_moves()
                    if item.piece_id == piece_id and item.result == "EXIT")
        steps.append((move, "coarse"))
        cursor = cursor.apply(move)
    api.board_revision = level_cache.board_hash(api._board_data(board))

    payload = api._build_solution_payload(
        board, steps, cursor, False, 1, "partial", 1.0, False, api.Minv)

    assert not payload["execution_ready"] and payload["safe_prefix_ready"], payload
    assert payload["execution_scope"] == "safe_exit_prefix", payload
    assert payload["execution_total"] == 2, payload
    assert [move.piece_id for move in api._active_plan["moves"]] == ["A", "B"]
    assert not api._active_plan["complete"], api._active_plan
    assert not Api._move_is_plain_exit(board, steps[-1][0])


def test_safe_exit_continuation_keeps_scope_without_replan():
    api = Api()
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    board = Board(3, 8, {
        "A": {"cells": [(0, 6), (0, 7)], "facing": "R"},
        "B": {"cells": [(1, 6), (1, 7)], "facing": "R"},
        "G": {"cells": [(2, 6), (2, 7)], "facing": "R", "species": "goat"},
    }, model="facing")
    first = next(item for item in board.legal_moves() if item.piece_id == "A")
    after = board.apply(first)
    second = next(item for item in after.legal_moves() if item.piece_id == "B")
    rebound = Board(3, 8, {
        "X": {"cells": [(1, 6), (1, 7)], "facing": "R"},
        "G": {"cells": [(2, 6), (2, 7)], "facing": "R", "species": "goat"},
    }, model="facing")
    api.board_revision = level_cache.board_hash(api._board_data(rebound))

    payload = api._continuation_solution(rebound, [second], api.Minv)

    assert payload["execution_scope"] == "safe_exit_prefix", payload
    assert Api._plan_is_execution_ready(api._active_plan, api.board_revision)
    assert api._active_plan["moves"][0].piece_id == "X", api._active_plan


def test_state_signature_includes_special_piece_semantics():
    api = Api()
    base = {"rows": 3, "cols": 4, "hazards": [], "fences": [], "pieces": [{
        "id": "A", "cells": [[1, 1], [1, 2]], "facing": "R", "species": "sheep"}]}
    goat = json.loads(json.dumps(base))
    goat["pieces"][0]["species"] = "goat"
    pig_sleeping = json.loads(json.dumps(base))
    pig_sleeping["pieces"][0].update(species="pig", awake=False)
    pig_awake = json.loads(json.dumps(pig_sleeping))
    pig_awake["pieces"][0]["awake"] = True
    assert api._state_signature(base) != api._state_signature(goat)
    assert api._state_signature(pig_sleeping) != api._state_signature(pig_awake)


def test_archived_level113_app_schedule_finds_complete_plan():
    path = (Path(__file__).resolve().parents[1] / "cache" / "levels" /
            "7614d5dd102dae13" / "executions" / "step-0001.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))["board_before"]
    board = Board(data["rows"], data["cols"], data["pieces"],
                  model=data["model"], slide_mode=data["slide_mode"],
                  hazards=data.get("hazards"), fences=data.get("fences"),
                  returning=data.get("returning"))
    api = Api()
    api.Minv = np.eye(3, dtype=np.float64)
    api.scene_report = {"scene_state": "gameplay", "execution_blockers": [],
                        "executable": True}
    api.board_revision = level_cache.board_hash(api._board_data(board))
    old_load, old_save = level_cache.load_solution, level_cache.save_solution
    try:
        level_cache.load_solution = lambda *_args, **_kwargs: None
        level_cache.save_solution = lambda *_args, **_kwargs: {
            "best_path": None, "revision_id": "test", "selected_best": False,
            "capture_id": None}
        payload = api._solve_with_budget(board, 5.0, api.Minv)
    finally:
        level_cache.load_solution, level_cache.save_solution = old_load, old_save

    assert payload["solved"] and payload["remaining"] == 0, payload
    assert payload["kind"] == "粗解36 + macro-beam", payload["kind"]
    assert payload["total"] == 88 and payload["execution_scope"] == "complete", payload


def test_save_manual_sample_persists_supervision_bundle():
    old_here = app_module.HERE
    old_learning_dir = recognition.MANUAL_LEARNING_DIR
    old_learning_index = recognition.MANUAL_LEARNING_INDEX
    try:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            app_module.HERE = str(root)
            recognition.MANUAL_LEARNING_DIR = root / "cache" / "recognition_learning"
            recognition.MANUAL_LEARNING_INDEX = recognition.MANUAL_LEARNING_DIR / "index.jsonl"
            api = Api()
            api.board = Board(4, 4, {
                "0": {"cells": [(0, 0), (0, 1)], "facing": "R", "species": "sheep"},
                "1": {"cells": [(1, 2), (2, 2)], "facing": "D", "species": "rocket"},
            }, model="facing")
            api._detected_board_data = {
                "rows": 4, "cols": 4, "model": "facing", "slide_mode": "all",
                "hazards": [], "fences": [], "returning": {},
                "pieces": {"0": {"cells": [[0, 0], [0, 1]],
                                   "facing": "R", "species": "sheep"}},
            }
            api._detected_sheep_data = []
            api.game = np.zeros((4 * 64, 4 * 64, 3), dtype=np.uint8)
            rect = api.game.copy()
            rect[64:3 * 64, 2 * 64:3 * 64] = (220, 170, 40)
            api.debug = {"rect": rect, "raw_candidates": [], "candidates": [], "dropped": []}
            result = api.save_manual_sample("round-trip")
            assert result["ok"] and result["learning"]["recorded"] == 1, result
            folder = Path(result["path"])
            expected = {"board.json", "manual_board.json", "detected_board.json",
                        "detected_sheep.json", "corrections.json", "recognition_evidence.json",
                        "grid_params.json", "metadata.json", "capture.png", "rectified.png"}
            assert expected <= {path.name for path in folder.iterdir()}, list(folder.iterdir())
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            corrections = json.loads((folder / "corrections.json").read_text(encoding="utf-8"))
            assert metadata["schema"] == 2 and metadata["correction_count"] == 1, metadata
            assert metadata["automatic_confirmation_count"] == 1, metadata
            assert corrections[0]["kind"] == "add" and corrections[0]["after"]["species"] == "rocket"
            learned = recognition.load_manual_learning()
            assert {item["correction"]["kind"] for item in learned} == {"add", "confirm"}, learned
    finally:
        app_module.HERE = old_here
        recognition.MANUAL_LEARNING_DIR = old_learning_dir
        recognition.MANUAL_LEARNING_INDEX = old_learning_index


def test_unconfirmed_board_edit_does_not_publish_direction_learning():
    api = Api()
    api.board = Board(4, 4, {
        "0": {"cells": [(1, 1), (1, 2)], "facing": "R", "species": "sheep"},
    }, model="facing")
    published = []
    api._record_direction_correction = lambda *args, **kwargs: published.append((args, kwargs))
    api._load_editor_board = lambda data, pending=True: {
        "manual_pending": pending, "can_undo": False, "can_redo": False,
    }
    result = api.edit_board({
        "action": "update_piece", "piece_id": "0",
        "cells": [[1, 1], [1, 2]], "facing": "L", "species": "sheep",
    })
    assert result["ok"] and result["manual_pending"], result
    assert not published, published


def test_update_piece_species_replaces_existing_piece_without_adding_one():
    api = Api()
    api.board = Board(4, 5, {
        "0": {"cells": [(1, 1), (1, 2)], "facing": "R", "species": "sheep"},
        "1": {"cells": [(3, 1), (3, 2)], "facing": "L", "species": "sheep"},
    }, model="facing")
    captured = {}

    def load(data, pending=True):
        captured.update(deepcopy(data))
        return {"manual_pending": pending, "can_undo": False, "can_redo": False}

    api._load_editor_board = load
    result = api.edit_board({
        "action": "update_piece", "piece_id": "0",
        "cells": [[1, 1], [1, 2]], "facing": "R", "species": "rocket",
    })

    assert result["ok"], result
    assert set(captured["pieces"]) == {"0", "1"}, captured["pieces"]
    assert captured["pieces"]["0"]["species"] == "rocket", captured["pieces"]


def test_updating_facing_across_axis_rotates_two_cell_footprint():
    api = Api()
    api.board = Board(5, 5, {
        "0": {"cells": [(1, 1), (2, 1)], "facing": "D", "species": "bomb"},
        "1": {"cells": [(3, 3), (3, 4)], "facing": "R", "species": "sheep"},
    }, model="facing")
    captured = {}

    def load(data, pending=True):
        captured.update(deepcopy(data))
        return {"manual_pending": pending, "can_undo": False, "can_redo": False}

    api._load_editor_board = load
    result = api.edit_board({
        "action": "update_piece", "piece_id": "0",
        "cells": [[1, 1], [2, 1]], "facing": "R", "species": "bomb",
    })

    assert result["ok"], result
    assert captured["pieces"]["0"]["cells"] == [[1, 1], [1, 2]]
    assert captured["pieces"]["0"]["facing"] == "R"


def test_dragging_piece_updates_cells_without_changing_facing():
    api = Api()
    api.board = Board(5, 6, {
        "0": {"cells": [(1, 1), (1, 2)], "facing": "R", "species": "sheep"},
    }, model="facing")
    captured = {}

    def load(data, pending=True):
        captured.update(deepcopy(data))
        return {"manual_pending": pending, "can_undo": False, "can_redo": False}

    api._load_editor_board = load
    result = api.edit_board({
        "action": "update_piece", "piece_id": "0",
        "cells": [[3, 3], [3, 4]], "facing": "R", "species": "sheep",
    })

    assert result["ok"], result
    assert captured["pieces"]["0"]["cells"] == [[3, 3], [3, 4]]
    assert captured["pieces"]["0"]["facing"] == "R"


def test_editor_can_toggle_wolf_cells_and_fences():
    api = Api()
    api.board = Board(4, 5, {
        "0": {"cells": [(1, 1), (1, 2)], "facing": "R", "species": "sheep"},
    }, model="facing")
    captured = {}

    def load(data, pending=True):
        captured.clear()
        captured.update(deepcopy(data))
        api.board = Board(data["rows"], data["cols"], data["pieces"],
                          model=data["model"], slide_mode=data["slide_mode"],
                          hazards=data.get("hazards"), fences=data.get("fences"))
        return {"manual_pending": pending, "can_undo": False, "can_redo": False}

    api._load_editor_board = load
    result = api.edit_board({"action": "toggle_hazard", "cell": [2, 3]})
    assert result["ok"] and captured["hazards"] == [[2, 3]], result
    result = api.edit_board({"action": "toggle_fence", "cell": [0, 3], "direction": "U"})
    assert result["ok"] and captured["fences"] == [{"cell": [0, 3], "direction": "U"}], result
    result = api.edit_board({"action": "toggle_fence", "cell": [0, 3], "direction": "U"})
    assert result["ok"] and captured["fences"] == [], result


def test_editor_obstacle_add_tools_do_not_toggle_existing_marks_off():
    api = Api()
    api.board = Board(4, 5, {}, model="facing")
    captured = {}

    def load(data, pending=True):
        captured.clear()
        captured.update(deepcopy(data))
        api.board = Board(data["rows"], data["cols"], data["pieces"],
                          model=data["model"], slide_mode=data["slide_mode"],
                          hazards=data.get("hazards"), fences=data.get("fences"))
        return {"manual_pending": pending, "can_undo": False, "can_redo": False}

    api._load_editor_board = load
    api._snapshot = lambda board, highlight: {
        "rows": board.rows, "cols": board.cols, "pieces": [],
        "hazards": [list(cell) for cell in sorted(board.hazards)],
        "fences": [{"cell": [r, c], "direction": direction}
                   for r, c, direction in sorted(board.fences)],
    }
    first = api.edit_board({"action": "add_hazard", "cell": [2, 3]})
    second = api.edit_board({"action": "add_hazard", "cell": [2, 3]})
    assert first["ok"] and first["changed"], first
    assert second["ok"] and not second["changed"], second
    assert list(api.board.hazards) == [(2, 3)]

    first = api.edit_board({"action": "add_fence", "cell": [0, 3], "direction": "U"})
    second = api.edit_board({"action": "add_fence", "cell": [0, 3], "direction": "U"})
    assert first["ok"] and first["changed"], first
    assert second["ok"] and not second["changed"], second
    assert list(api.board.fences) == [(0, 3, "U")]


def test_editor_clear_cell_removes_every_visible_mark_in_one_edit():
    api = Api()
    api.board = Board(4, 5, {
        "0": {"cells": [(0, 0), (0, 1)], "facing": "R", "species": "sheep"},
        "1": {"cells": [(2, 1), (2, 2)], "facing": "L", "species": "rocket"},
    }, model="facing", hazards=[(3, 4)], fences=[
        {"cell": [0, 0], "direction": "U"},
        {"cell": [3, 4], "direction": "D"},
    ])
    captured = {}

    def load(data, pending=True):
        captured.clear()
        captured.update(deepcopy(data))
        api.board = Board(data["rows"], data["cols"], data["pieces"],
                          model=data["model"], slide_mode=data["slide_mode"],
                          hazards=data.get("hazards"), fences=data.get("fences"))
        return {"manual_pending": pending, "can_undo": False, "can_redo": False}

    api._load_editor_board = load
    result = api.edit_board({"action": "clear_cell", "cell": [0, 0]})

    assert result["ok"] and result["changed"], result
    assert set(captured["pieces"]) == {"1"}, captured["pieces"]
    assert captured["hazards"] == [[3, 4]], captured["hazards"]
    assert captured["fences"] == [{"cell": [3, 4], "direction": "D"}], captured["fences"]
    assert result["edit_detail"] == {
        "cell": [0, 0],
        "removed_piece_ids": ["0"],
        "removed_hazard": False,
        "removed_fence_directions": ["U"],
    }

    result = api.edit_board({"action": "clear_cell", "cell": [3, 4]})
    assert result["ok"] and result["changed"], result
    assert captured["hazards"] == []
    assert captured["fences"] == []
    assert result["edit_detail"]["removed_hazard"] is True
    assert result["edit_detail"]["removed_fence_directions"] == ["D"]


def test_editor_grid_returns_atomic_frame_and_board_snapshot():
    api = Api()
    api.game = np.zeros((18, 24, 3), dtype=np.uint8)
    api.Minv = np.eye(3, dtype=np.float32)
    api.board = Board(2, 2, {
        "0": {"cells": [(0, 0), (0, 1)], "facing": "R", "species": "sheep"},
    }, model="facing")
    api.board_revision = "same-frame-revision"
    api._cell_poly = lambda row, col: [
        [col, row], [col + 1, row], [col + 1, row + 1], [col, row + 1],
    ]
    api._cell_center = lambda row, col: [col + .5, row + .5]
    api._grid_lines = lambda rows, cols: []
    api._snapshot = lambda board, highlight: {
        "rows": board.rows, "cols": board.cols, "pieces": [{"id": "0"}],
    }

    result = api.editor_grid()

    assert result["ok"], result
    assert result["image_size"] == [24, 18]
    assert result["board_revision"] == "same-frame-revision"
    assert result["state"]["pieces"] == [{"id": "0"}]
    encoded = np.frombuffer(base64.b64decode(result["img"]), dtype=np.uint8)
    decoded = app_module.cv2.imdecode(encoded, app_module.cv2.IMREAD_COLOR)
    assert decoded.shape == api.game.shape


def test_failed_sample_bundle_never_publishes_active_learning():
    old_here = app_module.HERE
    old_learning_dir = recognition.MANUAL_LEARNING_DIR
    old_learning_index = recognition.MANUAL_LEARNING_INDEX
    old_imwrite = app_module.cv2.imwrite
    try:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            app_module.HERE = str(root)
            recognition.MANUAL_LEARNING_DIR = root / "cache" / "recognition_learning"
            recognition.MANUAL_LEARNING_INDEX = recognition.MANUAL_LEARNING_DIR / "index.jsonl"
            api = Api()
            api.board = Board(4, 4, {
                "0": {"cells": [(0, 0), (0, 1)], "facing": "R", "species": "sheep"},
                "1": {"cells": [(1, 2), (2, 2)], "facing": "D", "species": "rocket"},
            }, model="facing")
            api._detected_board_data = {
                "rows": 4, "cols": 4, "model": "facing", "slide_mode": "all",
                "hazards": [], "fences": [], "returning": {},
                "pieces": {"0": {"cells": [[0, 0], [0, 1]],
                                   "facing": "R", "species": "sheep"}},
            }
            api._detected_sheep_data = []
            api.game = np.zeros((4 * 64, 4 * 64, 3), dtype=np.uint8)
            api.debug = {"rect": api.game.copy(), "raw_candidates": [],
                         "candidates": [], "dropped": []}
            app_module.cv2.imwrite = lambda *_args, **_kwargs: False
            result = api.save_manual_sample("must-fail")
            assert not result["ok"], result
            assert not recognition.MANUAL_LEARNING_INDEX.exists(), recognition.MANUAL_LEARNING_INDEX
            assert recognition.load_manual_learning() == []
    finally:
        app_module.HERE = old_here
        recognition.MANUAL_LEARNING_DIR = old_learning_dir
        recognition.MANUAL_LEARNING_INDEX = old_learning_index
        app_module.cv2.imwrite = old_imwrite


def test_saving_provisional_candidate_records_second_confirmation():
    old_here = app_module.HERE
    old_learning_dir = recognition.MANUAL_LEARNING_DIR
    old_learning_index = recognition.MANUAL_LEARNING_INDEX
    try:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            app_module.HERE = str(root)
            recognition.MANUAL_LEARNING_DIR = root / "cache" / "recognition_learning"
            recognition.MANUAL_LEARNING_INDEX = recognition.MANUAL_LEARNING_DIR / "index.jsonl"
            piece = {"cells": [[1, 2], [2, 2]], "facing": "D", "species": "rocket"}
            board_data = {"rows": 4, "cols": 4, "model": "facing", "slide_mode": "all",
                          "hazards": [], "fences": [], "returning": {},
                          "pieces": {"0": piece}}
            api = Api()
            api.board = Board(4, 4, {"0": piece}, model="facing")
            api._detected_board_data = deepcopy(board_data)
            api._detected_sheep_data = [{
                "id": 0, **piece, "learned_template": True,
                "learned_provisional": True, "learned_sample_ids": ["first-observation"],
            }]
            api.game = np.zeros((4 * 64, 4 * 64, 3), dtype=np.uint8)
            api.game[64:3 * 64, 2 * 64:3 * 64] = (220, 170, 40)
            api.debug = {"rect": api.game.copy(), "raw_candidates": [],
                         "candidates": [], "dropped": []}
            result = api.save_manual_sample("confirm learned candidate")
            assert result["ok"] and result["learning"]["recorded"] == 1, result
            corrections = json.loads((Path(result["path"]) / "corrections.json").read_text("utf-8"))
            assert len(corrections) == 1 and corrections[0]["confirmation"], corrections
            assert corrections[0]["fields"] == ["presence", "species", "facing"], corrections
    finally:
        app_module.HERE = old_here
        recognition.MANUAL_LEARNING_DIR = old_learning_dir
        recognition.MANUAL_LEARNING_INDEX = old_learning_index


def test_runtime_frames_do_not_promote_single_sample_learning():
    piece = {
        "id": 8, "cells": [[10, 2], [10, 3]], "facing": "R", "species": "sheep",
        "review": True, "review_reason": "manual_learning_single_observation",
        "learned_template": True, "learned_provisional": True, "learned_support": 1,
        "confidence": {"occupancy": 0.79, "temporal_facing": 1.0},
    }
    observation = {
        "rows": 18, "cols": 12,
        "pieces": [{"cells": [[10, 2], [10, 3]], "facing": "R", "species": "sheep"}],
        "hazards": [],
    }
    promoted = Api._confirm_stable_runtime_reviews(
        [piece], [deepcopy(observation), deepcopy(observation)])
    assert not promoted, promoted
    assert not piece.get("runtime_confirmed") and piece.get("review"), piece
    assert piece["learned_provisional"] and piece["learned_support"] == 1, piece


def test_level_change_clears_recognition_and_wolf_history(monkeypatch):
    api = Api.__new__(Api)
    api.game = np.zeros((20, 20, 3), dtype=np.uint8)
    api._source_level_label = "第121关"
    api._frame_history = deque([{"rows": 18, "cols": 12, "pieces": [{}]}], maxlen=4)
    api._wolf_observations = deque([{"cells": [[1, 1]]}], maxlen=8)
    api._wolf_motion = {"axis": "H"}
    api._wolf_danger_cells = {(1, 1)}
    api._wolf_confirmed_cells = {(1, 2)}
    reading = SimpleNamespace(label="第122关", method="ocr", bbox=(1, 2, 3, 4))
    monkeypatch.setattr(app_module.level_reader, "read_level", lambda _image: reading)

    result = api._read_source_level()

    assert result["level_label"] == "第122关"
    assert api._opening_coarse_pending
    assert not api._frame_history and not api._wolf_observations
    assert api._wolf_motion is None
    assert not api._wolf_danger_cells and not api._wolf_confirmed_cells


def test_autonomous_soft_gate_never_softens_single_sample_learning():
    review_report = {
        "scene_state": "gameplay",
        "execution_blockers": [{"code": "manual_review_required"}],
    }
    learning_report = {
        "scene_state": "gameplay",
        "execution_blockers": [{"code": "manual_learning_confirmation_required"}],
    }
    mixed_report = {
        "scene_state": "gameplay",
        "execution_blockers": [
            {"code": "manual_learning_confirmation_required"},
            {"code": "calibration_missing"},
        ],
    }
    assert Api._batch_soft_report(review_report)
    assert not Api._batch_soft_report(learning_report)
    assert not Api._batch_soft_report(mixed_report)


if __name__ == "__main__":
    tests = [
        test_same_corridor_waits_for_leading_sheep,
        test_independent_rows_keep_fast_interval,
        test_independent_exit_is_promoted_ahead_of_corridor_wait,
        test_wolf_track_exit_is_promoted_ahead_of_safe_exit,
        test_consecutive_wolf_frames_infer_patrol_lane_and_risky_exit,
        test_sandbox_payload_marks_wolf_track_and_risk_piece,
        test_execution_step_persists_piece_and_pre_click_board,
        test_review_error_identifies_piece_and_board_cells,
        test_low_confidence_exit_is_deprioritized_without_blocking_the_board,
        test_review_pause_is_a_guided_success_result_not_an_execution_error,
        test_direct_execution_pauses_for_review_before_any_mouse_click,
        test_tutorial_gesture_is_advisory_while_other_safety_blockers_remain_hard,
        test_tutorial_hand_remains_transient_with_unrelated_advisories,
        test_tutorial_gesture_remains_advisory_without_execution_control,
        test_execution_preflight_waits_for_motion_smoke_to_clear,
        test_retry_control_score_recognizes_green_next_level_button,
        test_advance_level_requires_victory_then_waits_for_gameplay,
        test_execute_step_exposes_motion_smoke_as_retryable,
        test_partial_solution_is_preview_only_and_clears_active_plan,
        test_partial_path_that_ends_without_legal_moves_is_reported_as_dead_end,
        test_complete_solution_publishes_execution_ready_plan,
        test_burst_advances_special_phase_with_single_step_and_releases_lock,
        test_direct_plan_executes_without_per_step_capture_and_verifies_once,
        test_opening_coarse_executes_safe_prefix_as_one_adaptive_batch,
        test_auto_workflow_runs_opening_coarse_before_full_search,
        test_solve_workflow_publishes_compact_right_panel_plan,
        test_quick_workflow_reuses_current_board_and_executes_only_three_exit_layers,
        test_quick_workflow_captures_only_when_current_board_is_missing,
        test_wolf_presence_disables_direct_plan_execution,
        test_suppressed_dark_component_does_not_enable_wolf_live_control,
        test_partial_hint_is_saved_as_non_usable_cache_revision,
        test_partial_hint_publishes_only_plain_exit_prefix,
        test_safe_exit_continuation_keeps_scope_without_replan,
        test_state_signature_includes_special_piece_semantics,
        test_archived_level113_app_schedule_finds_complete_plan,
        test_save_manual_sample_persists_supervision_bundle,
        test_unconfirmed_board_edit_does_not_publish_direction_learning,
        test_update_piece_species_replaces_existing_piece_without_adding_one,
        test_failed_sample_bundle_never_publishes_active_learning,
        test_saving_provisional_candidate_records_second_confirmation,
        test_autonomous_soft_gate_never_softens_single_sample_learning,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("execution timing tests passed")
