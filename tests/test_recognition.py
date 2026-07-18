"""P1 recognition regression tests without game-window dependencies."""
from __future__ import annotations

import cv2
import json
import numpy as np
from copy import deepcopy
from pathlib import Path
import tempfile

from board import grid as G
import recognition
import vision as D


def candidate(source, cells, facing, score, detector="body", species="sheep"):
    axis = "H" if cells[0][0] == cells[1][0] else "V"
    return {
        "source_id": source, "cells": [list(cell) for cell in cells],
        "rump": list(cells[0]), "head": list(cells[1]), "facing": facing,
        "axis": axis, "species": species, "detector": detector,
        "pair_score": score, "direction_confidence": score / 5,
        "quality": score, "center_rect": [0, 0],
    }


def test_pink_bow_wins_when_the_same_piece_also_matches_pig_body():
    cells = [[1, 6], [0, 6]]
    pieces = [{"id": 2, "cells": deepcopy(cells), "facing": "U",
               "species": "pig", "awake": True}]
    pink = [{"cells": deepcopy(cells), "species": "pink_sheep"}]
    pigs = [{"cells": deepcopy(cells), "species": "pig", "awake": True}]

    D.apply_species_anchors(pieces, pink, pigs, [])

    assert pieces[0]["species"] == "pink_sheep", pieces
    assert "awake" not in pieces[0], pieces


def test_tutorial_hand_recovers_only_unique_red_target_arrow(monkeypatch):
    rect = np.zeros((18 * D.CELL, 12 * D.CELL, 3), dtype=np.uint8)
    body = np.zeros(rect.shape[:2], dtype=np.uint8)
    hidden = candidate(71, [(14, 10), (15, 10)], "D", 1895, detector="arrow")
    hidden.update({"area": 377, "center_rect": [692.98, 932.7]})
    monkeypatch.setattr(D.detectors.arrow, "_arrow_candidates", lambda *_args, **_kwargs: [hidden])
    gesture = {
        "affected_cells": [[14, 10], [15, 10], [15, 11]],
        "components": [{
            "kind": "tutorial_hand", "tutorial_target_rect": [660.0, 921.5],
            "target_confidence": .92,
        }],
    }

    recovered = D._gesture_target_arrow_candidates(
        rect, body, 18, 12, gesture, regular_candidates=[])

    assert len(recovered) == 1, recovered
    assert recovered[0]["cells"] == [[14, 10], [15, 10]], recovered
    assert recovered[0]["facing"] == "D", recovered
    assert recovered[0]["detector"] == "gesture-target-arrow", recovered


def level113_manual_fixture():
    root = Path(__file__).resolve().parent.parent
    folder = root / "cache" / "manual_samples" / "20260713-013545-489"
    image_path = folder / "capture.png"
    board_path = folder / "board.json"
    if not image_path.exists() or not board_path.exists():
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    grid = G.load_grid(str(root / "grid_params.json"), image)
    manual = json.loads(board_path.read_text(encoding="utf-8"))
    return image, grid, manual


def test_fusion_keeps_all_detector_evidence():
    raw = [
        candidate("arrow", [(1, 1), (1, 2)], "R", 1800, "arrow"),
        candidate("body", [(1, 2), (1, 1)], "L", 1500, "body"),
        candidate("body2", [(1, 1), (1, 2)], "R", 1200, "body"),
    ]
    fused, rejected = recognition.fuse_candidates(raw)
    assert not rejected and len(fused) == 1, (fused, rejected)
    item = fused[0]
    assert item["facing"] == "R", item
    assert item["fusion"]["source_count"] == 3, item
    assert set(item["detectors"]) == {"arrow", "body"}, item
    assert 0 < item["confidence"]["occupancy"] <= 1, item


def test_rocket_candidate_keeps_species_and_anchor_priority():
    raw = [candidate("timer", [(6, 4), (6, 5)], "R", 3200, "rocket", "rocket")]
    fused, rejected = recognition.fuse_candidates(raw)
    assert not rejected and len(fused) == 1, (fused, rejected)
    item = fused[0]
    assert item["species"] == "rocket" and item["facing"] == "R", item
    assert item["selection_score"] > 150, item


def test_global_assignment_beats_local_greedy():
    # The tempting middle edge (10) blocks both outer edges (6 + 6).
    items = [
        {**candidate("middle", [(0, 1), (0, 2)], "R", 1), "selection_score": 10},
        {**candidate("left", [(0, 0), (0, 1)], "R", 1), "selection_score": 6},
        {**candidate("right", [(0, 2), (0, 3)], "R", 1), "selection_score": 6},
    ]
    kept, dropped, meta = recognition.global_assignment(items)
    assert {item["source_id"] for item in kept} == {"left", "right"}, (kept, dropped, meta)
    assert meta["method"] == "milp" and meta["objective"] == 12, meta
    assert dropped[0]["drop_reason"] == "global_occupancy_conflict", dropped


def test_strong_arrow_survives_partial_body_occlusion():
    rect = np.zeros((128, 128, 3), dtype=np.uint8)
    # Saturated orange arrow: its left shaft is deliberately larger than the
    # right tip so the direction split is unambiguous after morphology.
    cv2.rectangle(rect, (30, 53), (58, 70), (0, 140, 255), -1)
    cv2.fillConvexPoly(rect, np.array([[58, 55], [84, 62], [58, 69]], np.int32),
                       (0, 140, 255))
    body = np.zeros((128, 128), dtype=np.uint8)
    body[20:30, 20:40] = 255
    body[20:30, 80:100] = 255

    candidates = D._arrow_candidates(rect, body, 2, 2)
    assert len(candidates) == 1, candidates
    item = candidates[0]
    assert item["direction_votes"]["partial_body_support"] is True, item
    assert item["direction_votes"]["min_body_support"] == 380, item


def test_temporal_facing_and_dynamic_hazards():
    prior_piece = candidate("prior", [(2, 2), (2, 3)], "R", 1000)
    history = [recognition.observation_record([prior_piece], [{"row": 0, "col": 0}], 5, 5)
               for _ in range(3)]
    current = candidate("current", [(2, 3), (2, 2)], "L", 1000)
    pieces, hazards, temporal = recognition.apply_temporal(
        [current], [{"row": 0, "col": 0}, {"row": 4, "col": 4}], history, 5, 5)
    assert pieces[0]["facing"] == "R", (pieces, temporal)
    states = {tuple(item["cell"]): item["state"] for item in temporal["hazards"]}
    assert states[(0, 0)] == "stable" and states[(4, 4)] == "emerging", states
    assert [4, 4] in temporal["uncertain_hazard_cells"], temporal
    assert any(item["temporal_state"] == "stable" for item in hazards), hazards


def test_temporal_black_sheep_cannot_relabel_current_plain_sheep():
    prior_piece = candidate("prior", [(2, 2), (2, 3)], "R", 1000)
    prior_piece["species"] = "black_sheep"
    history = [recognition.observation_record(
        [prior_piece], [], 5, 5) for _ in range(3)]
    current = candidate("current", [(2, 2), (2, 3)], "R", 1000)
    pieces, _hazards, _temporal = recognition.apply_temporal(
        [current], [], history, 5, 5)
    assert pieces[0]["species"] == "sheep", pieces[0]


def test_preflight_restores_missing_high_confidence_edge_piece():
    edge = candidate("edge", [(0, 1), (0, 2)], "R", 1600, "arrow")
    edge["id"] = 0
    edge["confidence"] = {"occupancy": 0.9, "axis": 1.0, "facing": 0.9, "species": 0.92}
    history = [recognition.observation_record([edge], [], 5, 5)]
    pieces, _hazards, temporal = recognition.apply_temporal(
        [], [], history, 5, 5, recover_missing_edges=True)
    assert len(pieces) == 1 and pieces[0]["cells"] == [[0, 1], [0, 2]], pieces
    assert pieces[0]["temporal_restored"], pieces[0]
    assert temporal["restored_edge_pieces"], temporal


def test_special_piece_artwork_does_not_become_hazard():
    pieces = [
        {"species": "rocket", "cells": [[10, 6], [10, 7]]},
        {"species": "sheep", "cells": [[9, 9], [10, 9]]},
    ]
    hazards = [
        {"row": 9, "col": 7, "kind": "small"},
        {"row": 10, "col": 7, "kind": "small"},
        {"row": 10, "col": 9, "kind": "small"},
    ]
    kept, suppressed = D.suppress_special_hazard_overlaps(pieces, hazards)
    assert {(item["row"], item["col"]) for item in kept} == {(9, 7), (10, 9)}, kept
    assert [(item["row"], item["col"]) for item in suppressed] == [(10, 7)], suppressed


def test_goat_wolf_conflict_keeps_edge_goat_without_patrol_depth():
    goat = {
        "species": "goat", "cells": [[15, 8], [15, 9]], "facing": "R",
        "direction_votes": {},
    }
    hazards = [{"row": 15, "col": 9, "kind": "wolf_body"}]
    kept, rejected, decisions = D.resolve_goat_wolf_conflicts(
        [goat], hazards, rows=18, cols=12)
    assert kept == [goat] and not rejected, (kept, rejected)
    assert decisions == [{
        "cells": [[15, 8], [15, 9]], "facing": "R", "head": [15, 9],
        "hazard_overlap": [[15, 9]], "forward_clearance": 2,
        "diagonal_clearance": 2, "diagonal_sides": [2, 2],
        "required_forward": 5, "required_diagonal": 3, "decision": "goat",
    }], decisions


def test_goat_wolf_conflict_promotes_open_corridor_to_wolf():
    suspicious = {
        "species": "goat", "cells": [[6, 3], [6, 4]], "facing": "R",
    }
    hazards = [{"row": 6, "col": 4, "kind": "wolf_body"}]
    kept, rejected, decisions = D.resolve_goat_wolf_conflicts(
        [suspicious], hazards, rows=18, cols=12)
    assert not kept and len(rejected) == 1, (kept, rejected)
    assert rejected[0]["drop_reason"] == "wolf_environment_override", rejected
    assert decisions[0]["forward_clearance"] == 7, decisions
    assert decisions[0]["diagonal_clearance"] >= 3, decisions
    assert decisions[0]["decision"] == "wolf", decisions


def test_goat_without_wolf_visual_evidence_is_never_reclassified():
    goat = {"species": "goat", "cells": [[6, 3], [6, 4]], "facing": "R"}
    kept, rejected, decisions = D.resolve_goat_wolf_conflicts(
        [goat], [], rows=18, cols=12)
    assert kept == [goat] and not rejected and not decisions, (kept, rejected, decisions)


def test_nearby_weak_wolf_blob_also_uses_goat_environment_check():
    goat = {"species": "goat", "cells": [[15, 8], [15, 9]], "facing": "R"}
    hazards = [{"row": 14, "col": 9, "kind": "wolf_body",
                "coverage": 0.12, "pixels": 420}]
    kept, rejected, decisions = D.resolve_goat_wolf_conflicts(
        [goat], hazards, rows=18, cols=12)
    assert kept == [goat] and not rejected, (kept, rejected)
    assert decisions[0]["decision"] == "goat", decisions
    assert decisions[0]["hazard_overlap"] == [[14, 9]], decisions


def test_provisional_learned_species_never_suppresses_hazard():
    piece = {"species": "rocket", "cells": [[4, 4], [4, 5]],
             "learned_provisional": True}
    hazards = [{"row": 4, "col": 5, "coverage": 0.12, "pixels": 400}]
    kept, suppressed = D.suppress_special_hazard_overlaps([piece], hazards)
    assert kept == hazards and not suppressed, (kept, suppressed)
    promoted = {"species": "rocket", "cells": [[4, 4], [4, 5]],
                "learned_template": True, "learned_support": 2}
    kept, suppressed = D.suppress_special_hazard_overlaps([promoted], hazards)
    assert kept == hazards and not suppressed, (kept, suppressed)


def test_current_bomb_anchor_beats_manual_rocket_template():
    rect = np.zeros((4 * D.CELL, 4 * D.CELL, 3), dtype=np.uint8)
    rect[D.CELL:3 * D.CELL, 2 * D.CELL:3 * D.CELL] = (245, 245, 245)
    piece = candidate("bomb", [(1, 2), (2, 2)], "D", 1600, "rocket", "bomb")
    piece["direction_votes"] = {"bomb_counter": [1, 2]}
    sample = {
        "schema": recognition.MANUAL_LEARNING_SCHEMA,
        "status": "active", "sample_id": "rocket-label",
        "observation_hash": "rocket-observation",
        "correction": {
            "kind": "update", "fields": ["species"],
            "before": {"cells": [[1, 2], [2, 2]], "species": "sheep", "facing": "D"},
            "after": {"cells": [[1, 2], [2, 2]], "species": "rocket", "facing": "D"},
        },
        "feature": recognition.pair_visual_feature(rect, piece),
    }
    corrected, applied = recognition.apply_manual_label_learning([piece], rect, [sample])
    assert corrected[0]["species"] == "bomb" and not applied, (corrected, applied)

    for species, detector in (("rocket", "rocket"), ("cattle", "cattle-body")):
        anchored = candidate(species, [(1, 2), (2, 2)], "D", 1600, detector, species)
        anchored["detectors"] = [detector]
        relabel = deepcopy(sample)
        relabel["sample_id"] = f"relabel-{species}"
        relabel["correction"] = {
            "kind": "update", "fields": ["species"],
            "before": {"cells": [[1, 2], [2, 2]], "species": species, "facing": "D"},
            "after": {"cells": [[1, 2], [2, 2]], "species": "sheep", "facing": "D"},
        }
        corrected, applied = recognition.apply_manual_label_learning([anchored], rect, [relabel])
        assert corrected[0]["species"] == "sheep" and applied, (corrected, applied)
        assert corrected[0]["learned_provisional"] and corrected[0]["review"], corrected


def test_archived_bomb_counter_rejects_learned_rocket_proposal():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "986e2d1f41249e07" /
                  "986e2d1f41249e07-left050-cap0001-e03ea38c" /
                  "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    pieces, debug = D.analyze(image, grid)
    assert not [piece for piece in pieces
                if piece.get("species") == "rocket" and piece.get("learned_template")
                and "bomb_counter" in (piece.get("direction_votes") or {})], pieces
    assert [piece for piece in pieces
            if piece.get("species") == "bomb" and piece.get("learned_template")], pieces
    assert any(item.get("drop_reason") == "strong_bomb_counter_override"
               for item in debug.get("learned_rejected") or []), debug.get("learned_rejected")


def test_dynamic_only_edit_does_not_become_visual_label():
    rect = np.zeros((4 * D.CELL, 4 * D.CELL, 3), dtype=np.uint8)
    piece = candidate("pig", [(1, 1), (1, 2)], "R", 1200, "pig-body", "pig")
    piece["awake"] = False
    sample = {
        "schema": recognition.MANUAL_LEARNING_SCHEMA,
        "status": "active", "sample_id": "awake-only", "observation_hash": "awake-only",
        "correction": {
            "kind": "update", "fields": ["awake"],
            "before": {"cells": [[1, 1], [1, 2]], "species": "pig",
                       "facing": "R", "awake": False},
            "after": {"cells": [[1, 1], [1, 2]], "species": "pig",
                      "facing": "R", "awake": True},
        },
        "feature": recognition.pair_visual_feature(rect, piece),
    }
    corrected, applied = recognition.apply_manual_label_learning([piece], rect, [sample])
    assert corrected[0]["awake"] is False and not applied, (corrected, applied)


def test_small_bomb_smoke_next_to_piece_is_not_a_wolf_hazard():
    pieces = [{"species": "bomb", "cells": [[10, 6], [10, 7]]}]
    hazards = [
        {"row": 9, "col": 7, "coverage": 0.12, "pixels": 493},
        {"row": 8, "col": 7, "coverage": 0.12, "pixels": 493},
        {"row": 9, "col": 8, "coverage": 0.55, "pixels": 2252},
    ]
    kept, suppressed = D.suppress_special_hazard_overlaps(pieces, hazards)
    assert {(item["row"], item["col"]) for item in suppressed} == {(9, 7)}
    assert {(item["row"], item["col"]) for item in kept} == {(8, 7), (9, 8)}


def test_partial_exit_filter_keeps_edge_piece_but_drops_departing_body():
    piece = candidate("edge", [(1, 3), (0, 3)], "U", 1200, "body")
    normal = np.zeros((6 * D.CELL, 8 * D.CELL), dtype=np.uint8)
    normal[12:116, 3 * D.CELL + 12:4 * D.CELL - 12] = 255
    kept, dropped = D.reject_partial_exit_candidates([piece], normal, 6, 8)
    assert len(kept) == 1 and not dropped, (kept, dropped)

    departing = np.zeros_like(normal)
    departing[0:52, 3 * D.CELL + 12:4 * D.CELL - 12] = 255
    kept, dropped = D.reject_partial_exit_candidates([piece], departing, 6, 8)
    assert not kept and dropped[0]["drop_reason"] == "outside_calibration_region", (kept, dropped)

    kept, dropped = D.reject_partial_exit_candidates([piece], departing, 6, 8, enabled=False)
    assert len(kept) == 1 and not dropped, (kept, dropped)


def test_manual_direction_learning_requires_two_independent_samples():
    piece = candidate("learn", [(2, 2), (2, 3)], "R", 1200, "body")
    piece["metrics"] = {
        "(2, 2)": {"white": 900, "face": 80, "dt_mean": 11, "hist": 920, "body_support": 1550},
        "(2, 3)": {"white": 520, "face": 210, "dt_mean": 7, "hist": 610, "body_support": 1200},
    }
    feature = recognition.direction_feature(piece)
    sample = {
        "sample_id": "manual-1", "corrected_facing": "L", "feature": feature,
    }
    target = deepcopy(piece)
    target["source_id"] = "next-frame"
    corrected, applied = recognition.apply_direction_learning([target], [sample])
    assert corrected[0]["facing"] == "R" and not applied, (corrected, applied)

    second = deepcopy(sample)
    second["sample_id"] = "manual-2"
    corrected, applied = recognition.apply_direction_learning([target], [sample, second])
    assert corrected[0]["facing"] == "L", corrected
    assert corrected[0]["head"] == [2, 2] and corrected[0]["learned_direction"], corrected
    assert applied and applied[0]["sample_id"] == "manual-1", applied


def test_direction_learning_does_not_count_duplicate_sample_twice():
    piece = candidate("learn-duplicate", [(2, 2), (2, 3)], "R", 1200, "body")
    piece["metrics"] = {
        "(2, 2)": {"white": 900, "face": 80, "dt_mean": 11, "hist": 920,
                   "body_support": 1550},
        "(2, 3)": {"white": 520, "face": 210, "dt_mean": 7, "hist": 610,
                   "body_support": 1200},
    }
    sample = {"sample_id": "same", "corrected_facing": "L",
              "feature": recognition.direction_feature(piece)}
    corrected, applied = recognition.apply_direction_learning(
        [deepcopy(piece)], [sample, deepcopy(sample)])
    assert corrected[0]["facing"] == "R" and not applied, (corrected, applied)


def test_direction_learning_rejects_zero_and_conflicting_samples():
    piece = candidate("direction-target", [(2, 2), (2, 3)], "R", 1200, "body")
    piece["metrics"] = {
        "(2, 2)": {"white": 900, "face": 80, "dt_mean": 11,
                    "hist": 920, "body_support": 1550},
        "(2, 3)": {"white": 520, "face": 210, "dt_mean": 7,
                    "hist": 610, "body_support": 1200},
    }
    feature = recognition.direction_feature(piece)
    zero = {
        "sample_id": "zero", "corrected_facing": "L",
        "feature": {**feature, "vector": [0.0] * len(feature["vector"])},
    }
    corrected, applied = recognition.apply_direction_learning([deepcopy(piece)], [zero])
    assert corrected[0]["facing"] == "R" and not applied, (corrected, applied)

    conflict = [
        {"sample_id": "left", "corrected_facing": "L", "feature": feature},
        {"sample_id": "right", "corrected_facing": "R", "feature": feature},
    ]
    corrected, applied = recognition.apply_direction_learning([deepcopy(piece)], conflict)
    assert corrected[0]["facing"] == "R" and not applied, (corrected, applied)


def test_board_corrections_detects_added_level113_rocket():
    fixture = level113_manual_fixture()
    if fixture is None:
        return
    _image, _grid, manual = fixture
    detected = deepcopy(manual)
    target_cells = {(9, 6), (10, 6)}
    target_id = next(
        pid for pid, piece in detected["pieces"].items()
        if piece.get("species") == "rocket"
        and {tuple(cell) for cell in piece.get("cells", [])} == target_cells
    )
    del detected["pieces"][target_id]

    corrections = recognition.board_corrections(detected, manual)
    additions = [item for item in corrections if item.get("kind") == "add"]
    assert len(additions) == 1, corrections
    addition = additions[0]
    assert set(addition["fields"]) == {"presence", "species", "facing"}, addition
    assert addition["before"] is None and addition["after_id"] == str(target_id), addition
    assert addition["after"]["species"] == "rocket", addition
    assert addition["after"]["facing"] == "D", addition
    assert {tuple(cell) for cell in addition["after"]["cells"]} == target_cells, addition


def test_manual_learning_index_round_trip_and_deduplicates():
    old_dir, old_index = recognition.manual_learning.MANUAL_LEARNING_DIR, recognition.manual_learning.MANUAL_LEARNING_INDEX
    try:
        with tempfile.TemporaryDirectory() as folder:
            recognition.manual_learning.MANUAL_LEARNING_DIR = Path(folder)
            recognition.manual_learning.MANUAL_LEARNING_INDEX = Path(folder) / "index.jsonl"
            record = {
                "schema": recognition.MANUAL_LEARNING_SCHEMA,
                "sample_id": "round-trip", "status": "active",
                "observation_hash": "observation-a",
                "correction": {
                    "kind": "add", "fields": ["presence", "species", "facing"],
                    "before": None,
                    "after": {"cells": [[1, 1], [2, 1]],
                              "species": "rocket", "facing": "D"},
                },
                "feature": {
                    "schema": recognition.PAIR_FEATURE_SCHEMA, "axis": "V",
                    "names": list(recognition.PAIR_FEATURE_NAMES),
                    "symmetric": [0.1] * len(recognition.PAIR_FEATURE_NAMES),
                    "endpoint": [0.02] * len(recognition.PAIR_FEATURE_NAMES),
                },
            }
            first = recognition.record_manual_learning([record])
            second = recognition.record_manual_learning([record])
            malformed = deepcopy(record)
            malformed["sample_id"] = "not-serializable"
            malformed["observation_hash"] = "observation-b"
            malformed["evidence"] = {"bad": {1, 2, 3}}
            quarantined = recognition.record_manual_learning([malformed])
            with open(recognition.manual_learning.MANUAL_LEARNING_INDEX, "a", encoding="utf-8") as stream:
                stream.write('{"schema":"bad"}\n[]\nnot-json\n')
            loaded = recognition.load_manual_learning()
            assert first["recorded"] == 1 and second["duplicates"] == 1, (first, second)
            assert quarantined["recorded"] == 0 and quarantined["quarantined"] == 1, quarantined
            assert len(loaded) == 1 and loaded[0]["sample_id"] == "round-trip", loaded
    finally:
        recognition.manual_learning.MANUAL_LEARNING_DIR = old_dir
        recognition.manual_learning.MANUAL_LEARNING_INDEX = old_index


def test_manual_template_proposes_missing_level113_rocket_for_review():
    fixture = level113_manual_fixture()
    if fixture is None:
        return
    image, grid, _manual = fixture
    rect = grid.warp(image)
    pieces, debug = D.analyze(image, grid)
    target_cells = ((9, 6), (10, 6))
    target_key = tuple(sorted(target_cells))
    feature = recognition.pair_visual_feature(rect, target_cells)
    assert feature and feature["schema"] == recognition.PAIR_FEATURE_SCHEMA, feature

    raw = [
        deepcopy(item) for item in debug.get("raw_candidates", [])
        if not (recognition.cell_key(item) == target_key
                and item.get("species", "sheep") == "rocket")
    ]
    assert any(item.get("species") == "rocket"
               and set(recognition.cell_key(item)) & set(target_key) for item in raw), raw
    sample = {
        "schema": recognition.MANUAL_LEARNING_SCHEMA,
        "status": "active",
        "sample_id": "level113-right-rocket",
        "observation_hash": "level113-manual-observation",
        "correction": {
            "kind": "add", "fields": ["presence", "species", "facing"],
            "before": None,
            "after": {"cells": [list(cell) for cell in target_cells],
                      "species": "rocket", "facing": "D"},
        },
        "feature": feature,
    }
    proposals, diagnostics = recognition.manual_candidate_proposals(
        rect, grid.rows, grid.cols, raw, samples=[sample])
    learned = next((item for item in proposals
                    if recognition.cell_key(item) == target_key
                    and item.get("species") == "rocket"), None)
    assert learned is not None, (proposals, diagnostics)
    assert learned["facing"] == "D" and learned["detector"] == "learned-template", learned
    assert learned["review"] and learned["learned_provisional"], learned
    assert learned["learned_support"] == 1, learned
    evidence = next((item for item in diagnostics
                     if tuple(sorted(tuple(cell) for cell in item["cells"])) == target_key), None)
    assert evidence and evidence["accepted"], diagnostics
    assert evidence["presence_distance"] == 0.0
    assert evidence["endpoint_distance"] == 0.0


def test_manual_template_matches_same_visual_at_new_board_position():
    fixture = level113_manual_fixture()
    if fixture is None:
        return
    image, grid, _manual = fixture
    source_rect = grid.warp(image)
    source_cells = ((9, 6), (10, 6))
    feature = recognition.pair_visual_feature(source_rect, source_cells)
    translated = np.zeros_like(source_rect)
    target_cells = ((2, 2), (3, 2))
    for source, target in zip(source_cells, target_cells):
        sy, sx = source[0] * D.CELL, source[1] * D.CELL
        ty, tx = target[0] * D.CELL, target[1] * D.CELL
        translated[ty:ty + D.CELL, tx:tx + D.CELL] = \
            source_rect[sy:sy + D.CELL, sx:sx + D.CELL]
    sample = {
        "schema": recognition.MANUAL_LEARNING_SCHEMA,
        "status": "active", "sample_id": "translated-rocket",
        "observation_hash": "source-observation",
        "correction": {
            "kind": "add", "fields": ["presence", "species", "facing"],
            "before": None,
            "after": {"cells": [list(cell) for cell in source_cells],
                      "species": "rocket", "facing": "D"},
        },
        "feature": feature,
    }
    wrong_bridge = candidate("weak-rocket", [(2, 2), (2, 3)], "R", 900,
                             "rocket", "rocket")
    proposals, diagnostics = recognition.manual_candidate_proposals(
        translated, grid.rows, grid.cols, [wrong_bridge], samples=[sample])
    learned = next((item for item in proposals
                    if recognition.cell_key(item) == tuple(sorted(target_cells))), None)
    assert learned and learned["facing"] == "D", (proposals, diagnostics)
    assert learned["learned_sample_ids"] == ["translated-rocket"], learned


def test_manual_presence_template_conflict_abstains():
    rect = np.zeros((4 * D.CELL, 4 * D.CELL, 3), dtype=np.uint8)
    rect[D.CELL:3 * D.CELL, D.CELL:2 * D.CELL] = (220, 170, 40)
    cells = ((1, 1), (2, 1))
    feature = recognition.pair_visual_feature(rect, cells)
    samples = []
    for facing, observation in (("U", "obs-u"), ("D", "obs-d")):
        samples.append({
            "schema": recognition.MANUAL_LEARNING_SCHEMA,
            "status": "active", "sample_id": observation,
            "observation_hash": observation,
            "correction": {"kind": "add", "fields": ["presence", "species", "facing"],
                           "before": None,
                           "after": {"cells": [list(cell) for cell in cells],
                                     "species": "rocket", "facing": facing}},
            "feature": feature,
        })
    wrong_bridge = candidate("weak-rocket", [(1, 1), (1, 2)], "R", 900,
                             "rocket", "rocket")
    proposals, diagnostics = recognition.manual_candidate_proposals(
        rect, 4, 4, [wrong_bridge], samples=samples)
    assert not proposals, (proposals, diagnostics)
    assert any(item.get("reason") == "learning_conflict" for item in diagnostics), diagnostics


def test_one_saved_label_updates_only_one_matching_piece():
    rect = np.zeros((4 * D.CELL, 4 * D.CELL, 3), dtype=np.uint8)
    pieces = [
        candidate("first", [(1, 1), (2, 1)], "U", 1200),
        candidate("second", [(1, 2), (2, 2)], "U", 1200),
    ]
    feature = recognition.pair_visual_feature(rect, pieces[0])
    sample = {
        "schema": recognition.MANUAL_LEARNING_SCHEMA,
        "status": "active", "sample_id": "one-label",
        "observation_hash": "one-label-observation",
        "correction": {
            "kind": "update", "fields": ["facing"],
            "before": {"cells": [[1, 1], [2, 1]], "species": "sheep", "facing": "U"},
            "after": {"cells": [[1, 1], [2, 1]], "species": "sheep", "facing": "D"},
        },
        "feature": feature,
    }

    corrected, applied = recognition.apply_manual_label_learning(pieces, rect, [sample])

    assert [piece["facing"] for piece in corrected].count("D") == 1, corrected
    assert len(applied) == 1, applied


def test_exact_saved_deletion_rejects_one_footprint_provisionally():
    rect = np.zeros((4 * D.CELL, 4 * D.CELL, 3), dtype=np.uint8)
    rect[D.CELL:3 * D.CELL, D.CELL:2 * D.CELL] = (220, 170, 40)
    target = candidate("false", [(1, 1), (2, 1)], "D", 1200)
    other = candidate("other", [(1, 2), (2, 2)], "D", 1200)
    sample = {
        "schema": recognition.MANUAL_LEARNING_SCHEMA,
        "status": "active", "sample_id": "negative",
        "observation_hash": "negative-observation",
        "correction": {
            "kind": "delete", "fields": ["presence"],
            "before": {"cells": [[1, 1], [2, 1]], "species": "sheep", "facing": "D"},
            "after": None,
        },
        "feature": recognition.pair_visual_feature(rect, target),
    }

    kept, rejected, diagnostics = recognition.manual_candidate_rejections(
        rect, [target, other], [sample])

    assert kept == [other], (kept, rejected)
    assert rejected[0]["learned_provisional"] and rejected[0]["learned_support"] == 1
    assert diagnostics[0]["accepted"] and diagnostics[0]["exact_patch"], diagnostics


def test_saved_bomb_template_can_recover_a_corrected_footprint():
    rect = np.zeros((4 * D.CELL, 4 * D.CELL, 3), dtype=np.uint8)
    cells = ((1, 1), (2, 1))
    rect[D.CELL:3 * D.CELL, D.CELL:2 * D.CELL] = (220, 170, 40)
    sample = {
        "schema": recognition.MANUAL_LEARNING_SCHEMA,
        "status": "active", "sample_id": "bomb-add",
        "observation_hash": "bomb-observation",
        "correction": {
            "kind": "add", "fields": ["presence", "species", "facing"],
            "before": None,
            "after": {"cells": [list(cell) for cell in cells],
                      "species": "bomb", "facing": "D"},
        },
        "feature": recognition.pair_visual_feature(rect, cells),
    }
    wrong = candidate("weak-bomb", [(1, 1), (1, 2)], "R", 900, "rocket", "bomb")

    proposals, diagnostics = recognition.manual_candidate_proposals(
        rect, 4, 4, [wrong], [sample])

    learned = next((piece for piece in proposals
                    if recognition.cell_key(piece) == tuple(sorted(cells))), None)
    assert learned and learned["species"] == "bomb", (proposals, diagnostics)
    assert learned["learned_provisional"] and learned["learned_support"] == 1


def test_saved_manual_bundle_supplies_unchanged_confirmations_without_index():
    old_dir, old_index = recognition.manual_learning.MANUAL_LEARNING_DIR, recognition.manual_learning.MANUAL_LEARNING_INDEX
    try:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recognition.manual_learning.MANUAL_LEARNING_DIR = root / "cache" / "recognition_learning"
            recognition.manual_learning.MANUAL_LEARNING_INDEX = recognition.manual_learning.MANUAL_LEARNING_DIR / "index.jsonl"
            folder = root / "cache" / "manual_samples" / "20260716-173000-000"
            folder.mkdir(parents=True)
            piece = {"cells": [[1, 1], [2, 1]], "species": "bomb", "facing": "D",
                     "hit_limit": 3, "hits_remaining": 3}
            board = {"rows": 4, "cols": 4, "pieces": {"7": piece}}
            (folder / "metadata.json").write_text(json.dumps({
                "schema": 2, "created_at": "2026-07-16T17:30:00+0800",
                "observation_hash": "saved-board-observation", "grid_hash": "grid",
            }), encoding="utf-8")
            (folder / "manual_board.json").write_text(json.dumps(board), encoding="utf-8")
            (folder / "corrections.json").write_text("[]", encoding="utf-8")
            rect = np.zeros((4 * D.CELL, 4 * D.CELL, 3), dtype=np.uint8)
            rect[D.CELL:3 * D.CELL, D.CELL:2 * D.CELL] = (220, 170, 40)
            assert cv2.imwrite(str(folder / "rectified.png"), rect)

            samples = recognition.load_manual_learning()

            assert len(samples) == 1, samples
            correction = samples[0]["correction"]
            assert correction["kind"] == "confirm" and correction["after"]["species"] == "bomb"
    finally:
        recognition.manual_learning.MANUAL_LEARNING_DIR = old_dir
        recognition.manual_learning.MANUAL_LEARNING_INDEX = old_index


def test_level173_saved_board_is_reproduced_by_automatic_learning():
    root = Path(__file__).resolve().parent.parent
    folder = root / "cache" / "manual_samples" / "20260716-124545-463"
    if not (folder / "capture.png").exists():
        return
    image = cv2.imread(str(folder / "capture.png"))
    grid = G.load_grid(str(folder / "grid_params.json"), image)
    pieces, _debug = D.analyze(image, grid)
    manual = json.loads((folder / "manual_board.json").read_text(encoding="utf-8"))
    expected = {
        (recognition.cell_key(piece), piece.get("facing"), piece.get("species", "sheep"))
        for piece in manual["pieces"].values()
    }
    actual = {
        (recognition.cell_key(piece), piece.get("facing"), piece.get("species", "sheep"))
        for piece in pieces
    }
    assert len(actual) == 88 and actual == expected, (expected - actual, actual - expected)


def test_level113_manual_sample_recovers_two_rockets():
    fixture = level113_manual_fixture()
    if fixture is None:
        return
    image, grid, manual = fixture
    pieces, _debug = D.analyze(image, grid)
    manual_rocket_cells = {
        tuple(sorted(tuple(cell) for cell in piece.get("cells", [])))
        for piece in manual["pieces"].values() if piece.get("species") == "rocket"
    }
    rockets = {
        (tuple(sorted(tuple(cell) for cell in piece.get("cells", []))), piece["facing"])
        for piece in pieces if piece.get("species") == "rocket"
    }
    expected = {
        (((8, 5), (9, 5)), "U"),
        (((9, 6), (10, 6)), "D"),
    }
    assert len(manual["pieces"]) == 59, len(manual["pieces"])
    assert manual_rocket_cells == {placement for placement, _facing in expected}, manual_rocket_cells
    assert len(pieces) == 60, len(pieces)
    recovered = [piece for piece in pieces
                 if "gesture-target-arrow" in (piece.get("detectors") or [])]
    assert [(piece["cells"], piece["facing"]) for piece in recovered] == [
        ([[14, 10], [15, 10]], "D")
    ], recovered
    assert rockets == expected, rockets
    learned = next(piece for piece in pieces
                   if {tuple(cell) for cell in piece.get("cells", [])} == {(9, 6), (10, 6)})
    if int(learned.get("learned_support") or 0) < 2:
        assert learned.get("review") and learned.get("learned_provisional"), learned
    else:
        assert not learned.get("review") and not learned.get("learned_provisional"), learned


def test_learned_direction_survives_stale_temporal_majority():
    fixture = level113_manual_fixture()
    if fixture is None:
        return
    image, grid, _manual = fixture
    pieces, _debug = D.analyze(image, grid, temporal_history=[])
    observation = recognition.observation_record(pieces, [], grid.rows, grid.cols)
    target_cells = {(8, 5), (9, 5)}
    for piece in observation["pieces"]:
        if {tuple(cell) for cell in piece.get("cells", [])} == target_cells:
            piece["facing"] = "D"
    pieces, debug = D.analyze(
        image, grid, temporal_history=[deepcopy(observation) for _ in range(4)])
    target = next(piece for piece in pieces
                  if {tuple(cell) for cell in piece.get("cells", [])} == target_cells)
    assert target["facing"] == "U" and target.get("learned_direction"), target
    assert any(item.get("id") == target.get("id") and item.get("to") == "U"
               for item in debug["learned_directions"]), debug["learned_directions"]


def test_inward_facing_body_only_exit_remnant_is_dropped():
    remnant = candidate("exit", [(13, 0), (13, 1)], "R", 900, "body")
    remnant["detectors"] = ["body"]
    remnant["confidence"] = {"temporal_presence": 0.2}
    mask = np.zeros((18 * D.CELL, 12 * D.CELL), dtype=np.uint8)
    mask[13 * D.CELL + 7:14 * D.CELL - 7, 0:60] = 255
    mask[13 * D.CELL + 7:14 * D.CELL - 7, D.CELL:D.CELL + 19] = 255
    kept, dropped = D.reject_departing_edge_pieces([remnant], mask, 18, 12)
    assert not kept and dropped[0]["drop_reason"] == "departing_edge_artifact", (kept, dropped)
    assert dropped[0]["edge"] == "L" and dropped[0]["edge_support_ratio"] > 2.6, dropped

    arrow_confirmed = deepcopy(remnant)
    arrow_confirmed["detectors"] = ["arrow", "body"]
    kept, dropped = D.reject_departing_edge_pieces([arrow_confirmed], mask, 18, 12)
    assert len(kept) == 1 and not dropped, (kept, dropped)


def test_level99_bomb_counters_read_one_two_three():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "9e17a593563a32c9" /
                  "9e17a593563a32c9-left051-cap0004-915da05c" / "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    markers, _mask = D.bomb_markers(grid.warp(image), grid.rows, grid.cols)
    values = {tuple(item["cell"]): item["hits_remaining"] for item in markers}
    assert values[(3, 7)] == 1, markers
    assert values[(10, 7)] == 2, markers
    assert values[(9, 10)] == 3, markers
    assert all(item["counter_confidence"] >= 0.8 for item in markers), markers


def test_level100_elephants_are_2x3_and_boundary_fences_are_detected():
    root = Path(__file__).resolve().parent.parent
    image_path = root / "cache" / "manual_samples" / "20260712-134551-093" / "capture.png"
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    rect = grid.warp(image)
    elephants, _mask, _meta = D.elephant_pieces(rect, grid.rows, grid.cols)
    footprints = {(tuple(map(tuple, item["cells"])), item["facing"]) for item in elephants}
    expected = {
        (tuple((r, c) for r in range(2, 4) for c in range(3, 6)), "L"),
        (tuple((r, c) for r in range(4, 6) for c in range(1, 4)), "L"),
        (tuple((r, c) for r in range(13, 15) for c in range(8, 11)), "R"),
        (tuple((r, c) for r in range(15, 17) for c in range(6, 9)), "R"),
    }
    assert footprints == expected, footprints
    fences, _fence_mask, _fence_meta = D.fence_edges(rect, grid.rows, grid.cols)
    fence_keys = {(tuple(item["cell"]), item["direction"]) for item in fences}
    assert fence_keys == ({((r, 0), "L") for r in range(1, 6)} |
                          {((r, 11), "R") for r in range(12, 17)}), fence_keys


def test_level144_edge_sheep_faces_are_not_boundary_fences():
    root = Path(__file__).resolve().parent.parent
    capture = (root / "cache" / "levels" / "source-7320828c9" /
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

    fences, _mask, meta = D.fence_edges(
        grid.warp(image), grid.rows, grid.cols)

    assert fences == [], fences
    rejected = [run for run in meta["boundary_runs"]
                if run["direction"] == "L" and run["begin"] == 11]
    assert rejected and rejected[0]["component_span"] < meta["continuity_threshold"], meta


def test_level172_bottom_sheep_face_is_not_a_boundary_fence():
    root = Path(__file__).resolve().parent.parent
    capture = (root / "cache" / "levels" / "source-c1aa04bf4" /
               "source-c1aa04bf4-left089-cap0014-03ab11ec")
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

    fences, _mask, meta = D.fence_edges(grid.warp(image), grid.rows, grid.cols)

    assert fences == [], fences
    bottom_e = [run for run in meta["boundary_runs"]
                if run["direction"] == "D" and run["begin"] == 4]
    assert bottom_e and meta["spans"]["D"][4] < meta["span_threshold"], meta


def test_separate_vertical_timber_blobs_do_not_form_a_boundary_fence():
    rows, cols = 5, 4
    rect = np.zeros((rows * D.CELL, cols * D.CELL, 3), dtype=np.uint8)
    timber = cv2.cvtColor(
        np.uint8([[[14, 105, 155]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    for row in range(rows):
        cv2.rectangle(rect, (0, row * D.CELL + 6),
                      (17, (row + 1) * D.CELL - 7), timber, -1)

    fences, _mask, meta = D.fence_edges(rect, rows, cols)

    assert not [item for item in fences if item["direction"] == "L"], fences
    left_run = next(run for run in meta["boundary_runs"]
                    if run["direction"] == "L")
    assert left_run["component_span"] < meta["continuity_threshold"], meta


def test_current_vertical_elephants_use_head_texture_instead_of_board_half():
    root = Path(__file__).resolve().parent.parent
    capture = (root / "cache" / "levels" / "source-7320828c9" /
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
    elephants, _mask, meta = D.elephant_pieces(
        grid.warp(image), grid.rows, grid.cols)

    vertical = [piece for piece in elephants if piece["axis"] == "V"]
    assert len(vertical) == 2, vertical
    assert {piece["facing"] for piece in vertical} == {"D"}, vertical
    assert all(component["head_detail"]["D"]["score"] >
               component["head_detail"]["U"]["score"]
               for component in meta["components"]), meta


def test_boundary_fence_posts_do_not_fill_the_gaps_between_rails():
    rows, cols = 3, 6
    rect = np.zeros((rows * D.CELL, cols * D.CELL, 3), dtype=np.uint8)
    timber = cv2.cvtColor(np.uint8([[[15, 140, 150]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()

    # One loose four-cell timber run: actual rails in C/E, decorative posts in
    # D/F.  The post rectangles exceed the old total-pixel threshold but do
    # not cross their cells parallel to the top edge.
    for col in (2, 4):
        cv2.rectangle(rect, (col * D.CELL, 12),
                      ((col + 1) * D.CELL - 1, 20), timber, -1)
    for col in (3, 5):
        cv2.rectangle(rect, (col * D.CELL + 22, 0),
                      (col * D.CELL + 41, 23), timber, -1)
    # Bottom rails can sit well inside the last rectified row and need not be
    # adjacent. A post-only neighbor must still stay empty.
    for col in (1, 4):
        cv2.rectangle(rect, (col * D.CELL, (rows - 1) * D.CELL + 18),
                      ((col + 1) * D.CELL - 1, (rows - 1) * D.CELL + 26), timber, -1)
    cv2.rectangle(rect, (2 * D.CELL + 22, (rows - 1) * D.CELL),
                  (2 * D.CELL + 41, rows * D.CELL - 1), timber, -1)

    fences, _mask, meta = D.fence_edges(rect, rows, cols)
    top = {(tuple(item["cell"]), item["direction"]) for item in fences}
    assert top == {
        ((0, 2), "U"), ((0, 4), "U"),
        ((rows - 1, 1), "D"), ((rows - 1, 4), "D"),
    }, (top, meta["scores"], meta["spans"])


def test_level116_internal_fence_replaces_false_cattle_candidates():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "a3e073704957f7c7" /
                  "a3e073704957f7c7-left084-cap0001-9ef1802d" /
                  "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    pieces, debug = D.analyze(image, grid)
    internal = {(tuple(item["cell"]), item["direction"])
                for item in debug["fences"] if item["direction"] in {"H", "V"}}
    assert internal == {((3, c), "H") for c in range(4, 8)}, internal
    occupied = {tuple(cell) for piece in pieces for cell in piece["cells"]}
    assert not ({(3, c) for c in range(4, 8)} & occupied), occupied
    cattle = [(tuple(map(tuple, piece["cells"])), piece["facing"])
              for piece in pieces if piece.get("species") == "cattle"]
    assert cattle == [(((3, 1), (3, 2)), "R")], cattle
    rejected = [item for item in debug["dropped"]
                if item.get("drop_reason") == "internal_fence_occupancy_override"]
    assert rejected, debug["dropped"]


def test_dark_arrow_animal_is_black_sheep_not_wolf():
    piece = candidate("black", [(6, 4), (6, 5)], "R", 1600, "arrow")
    piece["detectors"] = ["arrow", "body"]
    hazards = [
        {"row": 6, "col": 5, "coverage": 0.65, "pixels": 2600},
        {"row": 6, "col": 6, "coverage": 0.62, "pixels": 2500},
        {"row": 7, "col": 5, "coverage": 0.64, "pixels": 2550},
        {"row": 7, "col": 6, "coverage": 0.61, "pixels": 2450},
    ]
    pieces, applied = D.classify_black_sheep([piece], hazards)
    assert pieces[0]["species"] == "black_sheep" and applied, (pieces, applied)
    kept_hazards, suppressed = D.suppress_special_hazard_overlaps(pieces, hazards)
    kept_pieces, rejected = D.reject_hazard_piece_overlaps(pieces, kept_hazards)
    assert not kept_hazards and len(suppressed) == 4, (kept_hazards, suppressed)
    assert len(kept_pieces) == 1 and not rejected, (kept_pieces, rejected)


def test_remote_wolf_cells_do_not_turn_an_arrow_sheep_black():
    piece = candidate("plain", [(2, 2), (2, 3)], "R", 1600, "arrow")
    piece["detectors"] = ["arrow"]
    hazards = [
        {"row": 2, "col": 3}, {"row": 3, "col": 3},
        {"row": 10, "col": 9}, {"row": 10, "col": 10}, {"row": 11, "col": 10},
    ]
    pieces, applied = D.classify_black_sheep([piece], hazards)
    assert pieces[0]["species"] == "sheep" and not applied, (pieces, applied)


def test_archived_level102_black_sheep_is_not_restored_as_wolf_hazards():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "bb0c823443f8882e" /
                  "bb0c823443f8882e-left089-cap0001-4afe5106" /
                  "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    pieces, debug = D.analyze(image, grid)
    black = [item for item in pieces if item.get("species") == "black_sheep"]
    assert len(black) == 1, black
    assert black[0]["cells"] == [[6, 4], [6, 5]] and black[0]["facing"] == "R", black[0]
    assert debug["hazards"] == [], debug["hazards"]
    assert debug["black_sheep_applied"], debug["black_sheep_applied"]


def test_archived_level102_real_wolf_stays_a_hazard():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "5fa27cb140252cfb" /
                  "5fa27cb140252cfb-left075-cap0001-c5ff81e6" /
                  "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    pieces, debug = D.analyze(image, grid)
    assert not [item for item in pieces if item.get("species") == "black_sheep"], pieces
    assert {(item["row"], item["col"]) for item in debug["hazards"]} == {(3, 6), (4, 6)}


def test_manual_sample_142317_recovers_sparse_black_sheep_and_bomb_direction():
    root = Path(__file__).resolve().parent.parent
    folder = root / "cache" / "manual_samples" / "20260716-142317-333"
    image_path = folder / "capture.png"
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(folder / "grid_params.json"), image)
    pieces, debug = D.analyze(image, grid)

    black = {
        (tuple(map(tuple, item["cells"])), item["facing"])
        for item in pieces if item.get("species") == "black_sheep"
    }
    assert black == {
        (((7, 7), (8, 7)), "D"),
        (((10, 4), (10, 5)), "R"),
    }, black
    bombs = {
        frozenset(map(tuple, item["cells"])): item
        for item in pieces if item.get("species") == "bomb"
    }
    confirmed = bombs[frozenset({(8, 3), (8, 4)})]
    assert confirmed["facing"] == "R" and not confirmed.get("review"), confirmed
    upward = bombs[frozenset({(10, 8), (11, 8)})]
    assert upward["facing"] == "U" and upward.get("learned_direction"), upward
    assert not upward.get("review"), upward
    ordinary = next(item for item in pieces
                    if {tuple(cell) for cell in item["cells"]} == {(6, 3), (7, 3)})
    assert ordinary["species"] == "sheep", ordinary
    assert len(pieces) == 81 and not debug["hazards"], (len(pieces), debug["hazards"])


def test_level121_border_wolf_does_not_create_full_column_track():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "a9b084d5e5ce4850" /
                  "a9b084d5e5ce4850-left072-cap0001-753b44c5" /
                  "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    _pieces, debug = D.analyze(image, grid)
    hazards = list(debug["hazards"])
    assert not [item for item in hazards if item.get("kind") == "wolf_track"], hazards
    assert all(item.get("kind") == "wolf_body" for item in hazards), hazards
    assert debug["wolf_meta"]["count"] == 2, debug["wolf_meta"]
    left_lane_rows = {item["row"] for item in hazards if item["col"] == 1}
    assert left_lane_rows <= {14, 15}, left_lane_rows
    board_data = D.to_board(_pieces, grid.rows, grid.cols, hazards=hazards)
    assert board_data["hazards"] == [], board_data["hazards"]


def test_to_board_keeps_manual_hazard_but_excludes_dynamic_wolf_body():
    board_data = D.to_board([], 4, 4, hazards=[
        {"row": 1, "col": 1, "kind": "wolf_body"},
        {"row": 2, "col": 2, "kind": "manual"},
        {"row": 3, "col": 3},
    ])
    assert board_data["hazards"] == [[2, 2], [3, 3]], board_data


def test_level103_recovers_exactly_four_black_sheep():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "e1db519666c7617b" /
                  "e1db519666c7617b-left078-cap0001-7dfaaf4b" /
                  "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    pieces, debug = D.analyze(image, grid)
    black = {
        (tuple(map(tuple, item["cells"])), item["facing"])
        for item in pieces if item.get("species") == "black_sheep"
    }
    assert black == {
        (((11, 9), (11, 8)), "L"),
        (((12, 10), (11, 10)), "U"),
        (((13, 8), (12, 8)), "U"),
        (((13, 10), (13, 9)), "L"),
    }, black
    assert debug["hazards"] == [], debug["hazards"]
    assert len(debug["black_sheep_cluster"]) == 4, debug["black_sheep_cluster"]
    assert len(pieces) == 80, len(pieces)
    assert not [item for item in pieces if item.get("species") == "cattle"], pieces


def test_level109_recovers_exactly_two_pink_sheep():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "cd85aa3e613763e2" /
                  "cd85aa3e613763e2-left083-cap0001-7af2db3c" /
                  "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    pieces, debug = D.analyze(image, grid)
    pink = {
        (tuple(map(tuple, item["cells"])), item["facing"])
        for item in pieces if item.get("species") == "pink_sheep"
    }
    assert pink == {
        (((13, 1), (13, 0)), "L"),
        (((16, 6), (17, 6)), "D"),
    }, pink
    assert len(debug["pink_sheep"]) == 2, debug["pink_sheep"]
    assert len(pieces) == 85, len(pieces)


def test_level112_recovers_three_sleeping_and_one_awake_pig():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "119a3123a6e83687" /
                  "119a3123a6e83687-left045-cap0001-f2c4601b" /
                  "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    pieces, debug = D.analyze(image, grid)
    pigs = {
        (tuple(map(tuple, item["cells"])), item["facing"]): bool(item.get("awake"))
        for item in pieces if item.get("species") == "pig"
    }
    assert pigs == {
        (((4, 6), (3, 6)), "U"): False,
        (((7, 3), (7, 4)), "R"): False,
        (((10, 1), (10, 2)), "R"): False,
        (((14, 2), (14, 3)), "R"): True,
    }, pigs
    assert len(debug["pigs"]) == 4, debug["pigs"]
    assert len(pieces) == 49, len(pieces)


def test_level113_recovers_top_and_bottom_goats():
    root = Path(__file__).resolve().parent.parent
    image_path = (root / "cache" / "levels" / "dc3f8a1bc0af6b8a" /
                  "dc3f8a1bc0af6b8a-left070-cap0001-c9845c3f" /
                  "images" / "_game.png")
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    grid = G.load_grid(str(root / "grid_params.json"), image)
    pieces, debug = D.analyze(image, grid)
    goats = {
        (tuple(map(tuple, item["cells"])), item["facing"])
        for item in pieces if item.get("species") == "goat"
    }
    assert goats == {
        (((0, 3), (0, 4)), "R"),
        (((15, 5), (15, 6)), "R"),
    }, goats
    assert len(debug["goats"]) == 2, debug["goats"]
    assert len(pieces) == 73, len(pieces)
    recovered = [piece for piece in pieces
                 if "gesture-target-arrow" in (piece.get("detectors") or [])]
    assert [(piece["cells"], piece["facing"]) for piece in recovered] == [
        ([[14, 10], [15, 10]], "D")
    ], recovered

if __name__ == "__main__":
    tests = [
        test_pink_bow_wins_when_the_same_piece_also_matches_pig_body,
        test_fusion_keeps_all_detector_evidence,
        test_rocket_candidate_keeps_species_and_anchor_priority,
        test_global_assignment_beats_local_greedy,
        test_strong_arrow_survives_partial_body_occlusion,
        test_temporal_facing_and_dynamic_hazards,
        test_temporal_black_sheep_cannot_relabel_current_plain_sheep,
        test_preflight_restores_missing_high_confidence_edge_piece,
        test_special_piece_artwork_does_not_become_hazard,
        test_goat_wolf_conflict_keeps_edge_goat_without_patrol_depth,
        test_goat_wolf_conflict_promotes_open_corridor_to_wolf,
        test_goat_without_wolf_visual_evidence_is_never_reclassified,
        test_nearby_weak_wolf_blob_also_uses_goat_environment_check,
        test_provisional_learned_species_never_suppresses_hazard,
        test_current_bomb_anchor_beats_manual_rocket_template,
        test_archived_bomb_counter_rejects_learned_rocket_proposal,
        test_dynamic_only_edit_does_not_become_visual_label,
        test_small_bomb_smoke_next_to_piece_is_not_a_wolf_hazard,
        test_partial_exit_filter_keeps_edge_piece_but_drops_departing_body,
        test_manual_direction_learning_requires_two_independent_samples,
        test_direction_learning_does_not_count_duplicate_sample_twice,
        test_direction_learning_rejects_zero_and_conflicting_samples,
        test_board_corrections_detects_added_level113_rocket,
        test_manual_learning_index_round_trip_and_deduplicates,
        test_manual_template_proposes_missing_level113_rocket_for_review,
        test_manual_template_matches_same_visual_at_new_board_position,
        test_manual_presence_template_conflict_abstains,
        test_one_saved_label_updates_only_one_matching_piece,
        test_exact_saved_deletion_rejects_one_footprint_provisionally,
        test_saved_bomb_template_can_recover_a_corrected_footprint,
        test_saved_manual_bundle_supplies_unchanged_confirmations_without_index,
        test_level173_saved_board_is_reproduced_by_automatic_learning,
        test_level113_manual_sample_recovers_two_rockets,
        test_learned_direction_survives_stale_temporal_majority,
        test_inward_facing_body_only_exit_remnant_is_dropped,
        test_level99_bomb_counters_read_one_two_three,
        test_level100_elephants_are_2x3_and_boundary_fences_are_detected,
        test_level172_bottom_sheep_face_is_not_a_boundary_fence,
        test_level116_internal_fence_replaces_false_cattle_candidates,
        test_dark_arrow_animal_is_black_sheep_not_wolf,
        test_remote_wolf_cells_do_not_turn_an_arrow_sheep_black,
        test_archived_level102_black_sheep_is_not_restored_as_wolf_hazards,
        test_archived_level102_real_wolf_stays_a_hazard,
        test_manual_sample_142317_recovers_sparse_black_sheep_and_bomb_direction,
        test_level121_border_wolf_does_not_create_full_column_track,
        test_level103_recovers_exactly_four_black_sheep,
        test_level109_recovers_exactly_two_pink_sheep,
        test_level112_recovers_three_sleeping_and_one_awake_pig,
        test_level113_recovers_top_and_bottom_goats,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("P1 recognition tests passed")
