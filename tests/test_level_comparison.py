"""Regression tests for first-capture source-level comparisons."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

import app as app_module
import level_cache


def board(*pieces):
    return {
        "rows": 6, "cols": 6, "model": "facing", "slide_mode": "all",
        "hazards": [], "pieces": {
            str(index): {"cells": cells, "facing": facing, "species": species}
            for index, (cells, facing, species) in enumerate(pieces)
        },
    }


def test_compare_ignores_detector_ids_and_counts_changes():
    source = board(([[1, 1], [1, 2]], "R", "sheep"),
                   ([[3, 2], [4, 2]], "D", "sheep"))
    current = board(([[3, 2], [4, 2]], "U", "sheep"),
                    ([[1, 1], [1, 2]], "R", "sheep"))
    result = level_cache.compare_boards(source, current)
    assert result["unchanged"] == 1, result
    assert result["changed"] == 1, result
    assert result["direction_changed"] == 1, result
    assert result["direction_changes"] == [{
        "cells": [[3, 2], [4, 2]], "species": "sheep", "from": "D", "to": "U",
    }], result


def test_first_capture_is_frozen_and_later_capture_compares():
    original_dir, original_root = level_cache.SOURCE_LEVEL_DIR, level_cache.ROOT
    with tempfile.TemporaryDirectory() as temp:
        level_cache.ROOT = Path(temp)
        level_cache.SOURCE_LEVEL_DIR = Path(temp) / "cache" / "source_levels"
        try:
            (level_cache.ROOT / "images").mkdir(parents=True)
            capture = np.full((512, 512, 3), 45, dtype=np.uint8)
            assert cv2.imwrite(str(level_cache.ROOT / "images" / "_game.png"), capture)
            source = board(([[1, 1], [1, 2]], "R", "sheep"))
            changed = board(([[2, 1], [2, 2]], "R", "sheep"))
            first = level_cache.record_source_comparison(source, "第 12 关")
            second = level_cache.record_source_comparison(changed, "第 12 关")
            assert first["baseline_created"] and first["sample_number"] == 1, first
            assert not second["baseline_created"] and second["changed"] == 1, second
            assert second["sample_number"] == 2, second
            facing_flip = board(([[2, 1], [2, 2]], "L", "sheep"))
            third = level_cache.record_source_comparison(facing_flip, "第 12 关")
            assert third["previous_changed"] == 1, third
            assert third["previous_direction_changed"] == 1, third
            assert third["variable_direction_count"] == 1, third
            saved = level_cache._read_json(
                level_cache.SOURCE_LEVEL_DIR / first["level_key"] / "source.json")
            assert saved["board"] == source, saved
            assert level_cache.source_snapshot_path(
                first["level_key"], first["sample_number"], "current").exists()
        finally:
            level_cache.SOURCE_LEVEL_DIR, level_cache.ROOT = original_dir, original_root


def test_direction_difference_writes_annotated_review_snapshot():
    original_dir, original_root = level_cache.SOURCE_LEVEL_DIR, level_cache.ROOT
    with tempfile.TemporaryDirectory() as temp:
        level_cache.ROOT = Path(temp)
        level_cache.SOURCE_LEVEL_DIR = Path(temp) / "cache" / "source_levels"
        try:
            (level_cache.ROOT / "images").mkdir(parents=True)
            capture = np.full((512, 512, 3), 52, dtype=np.uint8)
            assert cv2.imwrite(str(level_cache.ROOT / "images" / "_game.png"), capture)
            source = board(([[1, 1], [1, 2]], "R", "sheep"))
            flipped = board(([[1, 1], [1, 2]], "L", "sheep"))
            level_cache.record_source_comparison(source, "审核关")
            comparison = level_cache.record_source_comparison(flipped, "审核关")
            api = app_module.Api()
            api.game = capture
            api.Minv = np.eye(3, dtype=float)
            published = api._annotate_source_comparison(comparison)
            annotated = level_cache.source_snapshot_path(
                published["level_key"], published["sample_number"], "annotated")
            rendered = cv2.imread(str(annotated))
            assert published["has_annotated_snapshot"], published
            assert rendered is not None and np.any(rendered != capture), annotated
        finally:
            level_cache.SOURCE_LEVEL_DIR, level_cache.ROOT = original_dir, original_root


if __name__ == "__main__":
    test_compare_ignores_detector_ids_and_counts_changes()
    test_first_capture_is_frozen_and_later_capture_compares()
    test_direction_difference_writes_annotated_review_snapshot()
    print("source-level comparison tests passed")
