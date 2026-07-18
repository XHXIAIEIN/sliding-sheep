"""P0 regression tests for scene gates, board validation, and cache identity."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile

import cv2
import numpy as np

from board import grid as G
from board import io as board_io
import vision as D
from levels import cache as level_cache
from core import safety
from paths import ROOT


def test_scaled_calibration_is_checked_in_current_image_space():
    params = {
        "rows": 18, "cols": 12, "imgW": 1139, "imgH": 2062,
        "corners": {
            "TL": [44.7, 540.8], "TR": [815.3, 390.3],
            "BR": [1136.9, 1580.2], "BL": [251.7, 1770.6],
        },
    }

    blockers, warnings = safety.validate_calibration(params, (1889, 1037, 3))

    assert not any(item["code"] == "calibration_out_of_bounds"
                   for item in blockers), blockers
    assert any(item["code"] == "calibration_scaled" for item in warnings), warnings


def test_window_size_and_aspect_change_do_not_block_complete_board():
    params = {
        "rows": 18, "cols": 12, "imgW": 1000, "imgH": 1000,
        "corners": {
            "TL": [100, 120], "TR": [900, 120],
            "BR": [900, 880], "BL": [100, 880],
        },
    }

    blockers, warnings = safety.validate_calibration(params, (500, 4000, 3))

    assert not blockers, blockers
    assert not any(item["code"] == "calibration_resolution_drift" for item in blockers)
    assert any(item["code"] == "calibration_scaled"
               and item["detail"]["board_complete"] for item in warnings), warnings


def test_small_but_complete_board_is_not_rejected_by_window_area_ratio():
    params = {
        "rows": 8, "cols": 8, "imgW": 1000, "imgH": 1000,
        "corners": {
            "TL": [420, 420], "TR": [580, 420],
            "BR": [580, 580], "BL": [420, 580],
        },
    }

    blockers, _warnings = safety.validate_calibration(params, (1000, 1000, 3))

    assert not blockers, blockers


def test_only_incomplete_board_boundary_requires_recalibration():
    params = {
        "rows": 18, "cols": 12, "imgW": 1000, "imgH": 1000,
        "corners": {
            "TL": [-20, 100], "TR": [900, 100],
            "BR": [900, 900], "BL": [-20, 900],
        },
    }

    blockers, _warnings = safety.validate_calibration(params, (1000, 1000, 3))

    assert [item["code"] for item in blockers] == ["calibration_out_of_bounds"]
    assert "棋盘区域未完整包含" in blockers[0]["message"]


def test_scene_synthetic_gameplay():
    image = np.zeros((600, 360, 3), dtype=np.uint8)
    for r in range(12):
        for c in range(8):
            color = (90, 175, 210) if (r + c) % 2 else (100, 155, 155)
            image[r * 50:(r + 1) * 50, c * 45:(c + 1) * 45] = color
    body = np.zeros((600, 360), dtype=np.uint8)
    body[180:260, 100:200] = 255
    sheep = [{"id": 0, "cells": [[3, 2], [3, 3]], "facing": "R"}]
    report = safety.classify_scene(
        image,
        {"candidate_count": 1, "hazards": [], "body_mask": body},
        sheep, 12, 8,
        layout={"conflicts": []},
    )
    assert report["scene_state"] == "gameplay", report
    assert report["executable"], report
    assert not report["execution_complete"], report


def test_execution_complete_requires_observed_victory_not_a_solver_plan():
    image = np.zeros((600, 360, 3), dtype=np.uint8)
    for r in range(12):
        for c in range(8):
            image[r * 50:(r + 1) * 50, c * 45:(c + 1) * 45] = \
                (90, 175, 210) if (r + c) % 2 else (100, 155, 155)
    report = safety.classify_scene(
        image, {"candidate_count": 0, "hazards": [],
                "body_mask": np.zeros((600, 360), dtype=np.uint8)},
        [], 12, 8, layout={"conflicts": []})
    assert report["scene_state"] == "victory", report
    assert report["execution_complete"], report


def test_real_style_victory_overlay_overrides_popup_false_candidates():
    image = np.full((600, 360, 3), 28, dtype=np.uint8)
    image[55:195, 25:335] = (45, 55, 245)   # broad red clear ribbon
    image[455:570, 75:285] = (35, 225, 45)  # large green next-level button
    false_sheep = [
        {"id": index, "cells": [[index, 0], [index, 1]], "facing": "R"}
        for index in range(3)
    ]
    report = safety.classify_scene(
        image,
        {"candidate_count": 17, "hazards": [[0, 0]],
         "body_mask": np.zeros((600, 360), dtype=np.uint8)},
        false_sheep, 12, 8, layout={"conflicts": []})

    assert report["scene_state"] == "victory", report
    assert report["execution_complete"], report
    assert report["metrics"]["victory_overlay"] is True, report


def test_single_sample_learning_candidate_blocks_whole_board_execution():
    image = np.zeros((600, 360, 3), dtype=np.uint8)
    for r in range(12):
        for c in range(8):
            image[r * 50:(r + 1) * 50, c * 45:(c + 1) * 45] = \
                (90, 175, 210) if (r + c) % 2 else (100, 155, 155)
    body = np.zeros((600, 360), dtype=np.uint8)
    body[180:260, 100:200] = 255
    sheep = [{"id": 0, "cells": [[3, 2], [3, 3]], "facing": "R",
              "species": "rocket", "review": True,
              "learned_template": True, "learned_provisional": True,
              "learned_support": 1}]
    report = safety.classify_scene(
        image, {"candidate_count": 1, "hazards": [], "body_mask": body},
        sheep, 12, 8, layout={"conflicts": []})
    assert not report["executable"], report
    assert any(item["code"] == "manual_learning_confirmation_required"
               for item in report["execution_blockers"]), report

    sheep[0].update(review=False, learned_provisional=False, learned_support=2)
    promoted = safety.classify_scene(
        image, {"candidate_count": 1, "hazards": [], "body_mask": body},
        sheep, 12, 8, layout={"conflicts": []})
    assert promoted["executable"], promoted


def test_single_sample_learned_deletion_also_blocks_execution():
    image = np.zeros((600, 360, 3), dtype=np.uint8)
    for row in range(12):
        for col in range(8):
            image[row * 50:(row + 1) * 50, col * 45:(col + 1) * 45] = \
                (90, 175, 210) if (row + col) % 2 else (100, 155, 155)
    body = np.zeros((600, 360), dtype=np.uint8)
    body[180:260, 100:200] = 255
    sheep = [{"id": 0, "cells": [[3, 2], [3, 3]], "facing": "R",
              "species": "sheep"}]

    report = safety.classify_scene(
        image, {"candidate_count": 1, "hazards": [], "body_mask": body,
                "provisional_learning_rejection_count": 1},
        sheep, 12, 8, layout={"conflicts": []})

    assert report["metrics"]["provisional_learning_count"] == 1, report
    assert not report["executable"], report
    assert any(item["code"] == "manual_learning_confirmation_required"
               for item in report["execution_blockers"]), report


def test_cached_popup_is_blocked():
    popup = (ROOT / "cache" / "levels" / "5ad4e10809c44751" /
             "5ad4e10809c44751-left033-cap0026-db3876fc" / "images" / "_game.png")
    if not popup.exists():
        return
    image = cv2.imread(str(popup))
    grid = G.load_grid(str(ROOT / "grid_params.json"), image)
    sheep, debug = D.analyze(image, grid)
    layout = D.to_layout(sheep, grid.rows, grid.cols, debug["dropped"], hazards=debug.get("hazards"))
    report = safety.classify_scene(image, debug, sheep, grid.rows, grid.cols, layout=layout)
    assert report["scene_state"] == "popup", report
    assert not report["executable"], report
    assert any(item["code"] == "scene_not_gameplay" for item in report["execution_blockers"])


def test_tutorial_hand_is_masked_but_does_not_block_execution():
    rect = np.zeros((12 * G.CELL, 8 * G.CELL, 3), dtype=np.uint8)
    rect[:] = (95, 165, 185)
    # Split red tutorial outline immediately above the hand.
    cv2.rectangle(rect, (135, 168), (245, 225), (20, 20, 235), 8)
    cv2.ellipse(rect, (190, 260), (55, 72), 0, 0, 360, (22, 22, 22), -1)
    cv2.ellipse(rect, (190, 260), (45, 62), 0, 0, 360, (250, 250, 250), -1)
    mask, gesture = D.gesture_occlusion(rect, 12, 8)
    assert gesture and gesture["affected_cells"] and int((mask > 0).sum()) > 1800, gesture
    assert gesture["blocking"], gesture
    targets = [item for item in gesture["components"] if item.get("tutorial_target_rect")]
    assert len(targets) == 1, gesture
    tx, ty = targets[0]["tutorial_target_rect"]
    assert 170 <= tx <= 210 and 180 <= ty <= 215, targets[0]
    image = cv2.resize(rect, (360, 600))
    body = np.zeros((600, 360), dtype=np.uint8)
    body[180:260, 100:200] = 255
    sheep = [{"id": 0, "cells": [[3, 2], [3, 3]], "facing": "R"}]
    report = safety.classify_scene(
        image, {"candidate_count": 1, "hazards": [], "body_mask": body,
                "gesture": gesture}, sheep, 12, 8, layout={"conflicts": []})
    assert report["executable"], report
    assert not any(item["code"] == "gesture_occlusion"
                   for item in report["execution_blockers"]), report
    assert any(item["code"] == "gesture_occlusion"
               for item in report["advisories"]), report


def test_bottom_item_hint_is_masked_without_blocking_board():
    rect = np.zeros((12 * G.CELL, 8 * G.CELL, 3), dtype=np.uint8)
    rect[:] = (95, 165, 185)
    cv2.ellipse(rect, (255, 12 * G.CELL - 30), (48, 68), 0, 0, 360, (22, 22, 22), -1)
    cv2.ellipse(rect, (255, 12 * G.CELL - 30), (39, 59), 0, 0, 360, (250, 250, 250), -1)
    mask, gesture = D.gesture_occlusion(rect, 12, 8)
    assert gesture and not gesture["blocking"] and int((mask > 0).sum()) > 1800, gesture
    assert all(item["kind"] == "ui_item_hint" for item in gesture["components"]), gesture


def test_large_white_game_piece_without_red_target_is_not_a_tutorial_hand():
    rect = np.zeros((12 * G.CELL, 8 * G.CELL, 3), dtype=np.uint8)
    rect[:] = (95, 165, 185)
    cv2.ellipse(rect, (260, 420), (52, 66), 0, 0, 360, (248, 248, 248), -1)
    cv2.circle(rect, (260, 386), 24, (190, 85, 235), -1)

    mask, gesture = D.gesture_occlusion(rect, 12, 8)

    assert gesture is None, gesture
    assert not np.any(mask), int(np.count_nonzero(mask))


def test_level144_pink_sheep_is_not_reported_as_tutorial_hand():
    capture = (ROOT / "cache" / "levels" / "source-7320828c9" /
               "source-7320828c9-left080-cap0001-5ed298f2")
    image_path = capture / "images" / "_game.png"
    grid_path = capture / "board_grid.json"
    if not image_path.exists() or not grid_path.exists():
        return
    image = cv2.imread(str(image_path))
    saved_grid = json.loads(grid_path.read_text(encoding="utf-8"))
    grid = G.BoardGrid(
        rows=int(saved_grid["rows"]), cols=int(saved_grid["cols"]),
        corners=saved_grid["corners"], cell=int(saved_grid["cell"]),
        image_size=tuple(saved_grid["image_size"]),
    )

    mask, gesture = D.gesture_occlusion(grid.warp(image), grid.rows, grid.cols)

    assert gesture is None, gesture
    assert not np.any(mask), int(np.count_nonzero(mask))


def test_motion_smoke_is_not_mislabelled_as_tutorial_hand():
    rect = np.zeros((12 * G.CELL, 8 * G.CELL, 3), dtype=np.uint8)
    rect[:] = (95, 165, 185)
    for index in range(9):
        cv2.circle(rect, (90 + index * 10, 150 + index * 8), 12,
                   (245, 245, 245), -1)
    mask, gesture = D.gesture_occlusion(rect, 12, 8)
    assert gesture and gesture["blocking"] and int((mask > 0).sum()) > 1800, gesture
    assert all(item["kind"] == "motion_smoke" for item in gesture["components"]), gesture
    image = cv2.resize(rect, (360, 600))
    body = np.zeros((600, 360), dtype=np.uint8)
    body[180:260, 100:200] = 255
    sheep = [{"id": 0, "cells": [[3, 2], [3, 3]], "facing": "R"}]
    report = safety.classify_scene(
        image, {"candidate_count": 1, "hazards": [], "body_mask": body,
                "gesture": gesture}, sheep, 12, 8, layout={"conflicts": []})
    assert any(item["code"] == "motion_smoke" for item in report["execution_blockers"]), report
    assert not any(item["code"] == "gesture_occlusion" for item in report["execution_blockers"]), report


def test_review_and_dynamic_hazard_are_advisories_not_click_blockers():
    image = np.zeros((600, 360, 3), dtype=np.uint8)
    for r in range(12):
        for c in range(8):
            color = (90, 175, 210) if (r + c) % 2 else (100, 155, 155)
            image[r * 50:(r + 1) * 50, c * 45:(c + 1) * 45] = color
    body = np.zeros((600, 360), dtype=np.uint8)
    body[180:260, 100:200] = 255
    sheep = [{
        "id": 0, "cells": [[3, 2], [3, 3]], "facing": "R", "review": True,
        "review_reason": "detector_facing_disagreement",
        "confidence": {"occupancy": .8, "axis": 1.0, "facing": .55, "species": .9},
    }]
    report = safety.classify_scene(
        image,
        {"candidate_count": 1, "hazards": [{"row": 2, "col": 2}],
         "body_mask": body,
         "temporal": {"history_frames": 2, "uncertain_hazard_cells": [[2, 2]]}},
        sheep, 12, 8, layout={"conflicts": []},
    )
    codes = {item["code"] for item in report["advisories"]}
    assert report["executable"], report
    assert not report["execution_blockers"], report
    assert codes == {"manual_review_required", "dynamic_hazard_unstable"}, report
    review = next(item for item in report["advisories"]
                  if item["code"] == "manual_review_required")
    assert "#0（C4–D4）" in review["message"], review
    assert review["detail"]["pieces"] == [{
        "id": "0", "cells": [[3, 2], [3, 3]], "location": "C4–D4",
        "reason": "detector_facing_disagreement",
        "confidence": {"occupancy": .8, "axis": 1.0, "facing": .55, "species": .9},
    }], review


def test_board_rejects_hazard_piece_overlap():
    data = {
        "rows": 4, "cols": 4, "model": "facing", "slide_mode": "all",
        "hazards": [[1, 1]],
        "pieces": {"A": {"cells": [[1, 1], [1, 2]], "facing": "R"}},
    }
    try:
        board_io.validate_board_data(data)
    except board_io.BoardValidationError as exc:
        assert any("危险格与棋子" in error for error in exc.errors), exc.errors
    else:
        raise AssertionError("hazard-piece overlap must be rejected")


def test_board_rejects_bad_facing_and_bounds():
    data = {
        "rows": 3, "cols": 3, "model": "facing", "slide_mode": "all",
        "hazards": [],
        "pieces": {"A": {"cells": [[2, 2], [2, 3]], "facing": "D"}},
    }
    try:
        board_io.validate_board_data(data)
    except board_io.BoardValidationError as exc:
        text = "；".join(exc.errors)
        assert "越界" in text and "轴线" in text, text
    else:
        raise AssertionError("invalid board must be rejected")


def test_manual_layout_accepts_missing_detector_provenance():
    manual = [{
        "id": 9, "cells": [[1, 1], [1, 2]], "rump": [1, 1], "head": [1, 2],
        "axis": "H", "facing": "R", "species": "sheep", "manual": True,
        "confidence": {"occupancy": 1.0, "axis": 1.0, "facing": 1.0, "species": 1.0},
    }]
    layout = D.to_layout(manual, 4, 4, dropped=[], hazards=[])
    piece = layout["pieces"][0]
    assert piece["source_id"] == "manual:9", piece
    assert piece["quality"] == 1.0, piece


def test_cache_hash_includes_hazards_species_and_rules():
    base = {
        "rows": 4, "cols": 4, "model": "facing", "slide_mode": "all",
        "hazards": [], "fences": [], "rule_flags": {},
        "pieces": {"A": {"cells": [[1, 1], [1, 2]], "facing": "R", "species": "sheep"}},
    }
    variants = [
        {**base, "hazards": [[0, 0]]},
        {**base, "no_stop": [[2, 2]]},
        {**base, "fences": [{"cell": [1, 0], "direction": "L"}]},
        {**base, "pieces": {"A": {**base["pieces"]["A"], "species": "cattle"}}},
        {**base, "pieces": {"A": {**base["pieces"]["A"], "species": "goat"}}},
        {**base, "pieces": {"A": {**base["pieces"]["A"], "species": "pig",
                                    "awake": False}}},
        {**base, "rule_flags": {"dynamic_hazard": True}},
        {**base, "pieces": {}, "returning": {
            "A": {"cells": [[1, 1], [1, 2]], "facing": "R", "species": "black_sheep"}
        }},
    ]
    original = level_cache.board_hash(base)
    assert all(level_cache.board_hash(item) != original for item in variants)
    sleeping_pig = {**base, "pieces": {"A": {**base["pieces"]["A"],
                                                "species": "pig", "awake": False}}}
    awake_pig = {**base, "pieces": {"A": {**base["pieces"]["A"],
                                             "species": "pig", "awake": True}}}
    assert level_cache.board_hash(sleeping_pig) != level_cache.board_hash(awake_pig)


def test_solution_revisions_never_regress_best():
    board = {
        "rows": 4, "cols": 4, "model": "facing", "slide_mode": "all",
        "hazards": [],
        "pieces": {"A": {"cells": [[1, 1], [1, 2]], "facing": "R", "species": "sheep"}},
    }
    old_cache, old_global = level_cache.CACHE_DIR, level_cache.GLOBAL_SOLUTION_DIR
    with tempfile.TemporaryDirectory() as temp:
        level_cache.CACHE_DIR = Path(temp) / "levels"
        level_cache.GLOBAL_SOLUTION_DIR = level_cache.CACHE_DIR / "_solutions"
        try:
            complete = {"solved": True, "remaining": 0, "usable": True,
                        "moves": [{"piece": "A"}, {"piece": "B"}]}
            partial = {"solved": False, "remaining": 1, "usable": True,
                       "moves": [{"piece": "A"}]}
            level_cache.save_solution(board, complete, level_key="level")
            level_cache.save_solution(board, partial, level_key="level")
            best = level_cache.load_solution(board, level_key="level")
            assert best and best["solved"] and len(best["moves"]) == 2, best
            assert level_cache.load_solution(board, level_key="level", require_complete=True)["solved"]
            revisions = list((level_cache.CACHE_DIR / "level" / "solutions" /
                              level_cache.board_hash(board) / "revisions").glob("*.json"))
            assert len(revisions) == 2, revisions
        finally:
            level_cache.CACHE_DIR, level_cache.GLOBAL_SOLUTION_DIR = old_cache, old_global


def test_capture_publication_is_serialized_between_refresh_threads():
    board = {
        "rows": 4, "cols": 4, "model": "facing", "slide_mode": "all",
        "hazards": [],
        "pieces": {"A": {"cells": [[1, 1], [1, 2]], "facing": "R",
                         "species": "sheep"}},
    }
    old_cache, old_root = level_cache.CACHE_DIR, level_cache.ROOT
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        level_cache.ROOT = root
        level_cache.CACHE_DIR = root / "cache" / "levels"
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                saved = list(pool.map(
                    lambda index: level_cache.save_capture(
                        board, level_key="threaded-level", source=f"thread-{index}"),
                    range(12)))
            assert len({item["capture_id"] for item in saved}) == 12
            published = list((level_cache.CACHE_DIR / "threaded-level").glob("*/meta.json"))
            assert len(published) == 12, published
            assert not list((level_cache.CACHE_DIR / "threaded-level").glob(".staging-*"))
        finally:
            level_cache.CACHE_DIR, level_cache.ROOT = old_cache, old_root


if __name__ == "__main__":
    tests = [
        test_scene_synthetic_gameplay,
        test_execution_complete_requires_observed_victory_not_a_solver_plan,
        test_single_sample_learning_candidate_blocks_whole_board_execution,
        test_single_sample_learned_deletion_also_blocks_execution,
        test_cached_popup_is_blocked,
        test_tutorial_hand_is_masked_but_does_not_block_execution,
        test_bottom_item_hint_is_masked_without_blocking_board,
        test_motion_smoke_is_not_mislabelled_as_tutorial_hand,
        test_review_and_dynamic_hazard_are_advisories_not_click_blockers,
        test_board_rejects_hazard_piece_overlap,
        test_board_rejects_bad_facing_and_bounds,
        test_manual_layout_accepts_missing_detector_provenance,
        test_cache_hash_includes_hazards_species_and_rules,
        test_solution_revisions_never_regress_best,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("P0 safety tests passed")
