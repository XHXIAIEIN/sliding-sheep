"""Transactional recognition facade.

Detection remains intentionally rich, but callers now receive one complete
bundle instead of mutating GUI state while recognition is still running.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from board import grid as board_grid
from board import io as board_io
import vision as detector
from core import safety


@dataclass(frozen=True)
class AnalysisBundle:
    grid: board_grid.BoardGrid | None
    sheep: list[dict]
    debug: dict
    board_data: dict | None
    layout: dict | None
    report: dict
    calibration_warnings: list

    @property
    def calibrated(self) -> bool:
        return self.grid is not None

    @property
    def gameplay(self) -> bool:
        return self.report.get("scene_state") == "gameplay"


def load_params(path: str | Path) -> dict | None:
    target = Path(path)
    if not target.exists():
        return None
    import json
    return json.loads(target.read_text(encoding="utf-8"))


def analyze_image(image: np.ndarray, params: dict | None, *,
                  temporal_history: list[dict] | None = None,
                  recover_missing_edges: bool = False,
                  ignore_outside_pieces: bool = True) -> AnalysisBundle:
    if image is None:
        raise ValueError("缺少待分析截图")
    blockers, warnings = safety.validate_calibration(params, image.shape)
    if not params:
        report = {
            "scene_state": "unknown",
            "scene_reason": "尚未标定",
            "metrics": {},
            "warnings": warnings,
            "execution_blockers": blockers,
            "executable": False,
        }
        return AnalysisBundle(None, [], {}, None, None, report, warnings)

    grid = board_grid.from_params(params, image.shape)
    sheep, debug = detector.analyze(
        image,
        grid,
        temporal_history=list(temporal_history or []),
        recover_missing_edges=bool(recover_missing_edges),
        ignore_outside_pieces=bool(ignore_outside_pieces),
    )
    board_data = detector.to_board(
        sheep, grid.rows, grid.cols, hazards=debug.get("hazards"),
        fences=debug.get("fences"))
    layout = detector.to_layout(
        sheep, grid.rows, grid.cols, debug.get("dropped", []),
        hazards=debug.get("hazards"), fences=debug.get("fences"))
    report = safety.classify_scene(
        image, debug, sheep, grid.rows, grid.cols,
        calibration_blockers=blockers, layout=layout)
    report["warnings"] = warnings + list(report.get("advisories") or [])
    try:
        board_io.validate_board_data(board_data)
    except board_io.BoardValidationError as exc:
        report = safety.add_blockers(report, [
            safety.blocker("board_schema_invalid", "棋盘结构校验失败", detail=exc.errors)
        ])
    return AnalysisBundle(
        grid=grid,
        sheep=sheep,
        debug=debug,
        board_data=board_data,
        layout=layout,
        report=report,
        calibration_warnings=warnings,
    )


def audit_payload(bundle: AnalysisBundle, *,
                  runtime_confirmed_reviews: list | None = None) -> dict[str, Any]:
    debug = bundle.debug
    report = bundle.report
    return {
        "scene_state": report.get("scene_state"),
        "scene_reason": report.get("scene_reason"),
        "execution_blockers": report.get("execution_blockers", []),
        "metrics": report.get("metrics", {}),
        "kept": bundle.sheep,
        "hazards": debug.get("hazards", []),
        "fences": debug.get("fences", []),
        "wolf": debug.get("wolf_meta"),
        "black_sheep_cluster": debug.get("black_sheep_cluster", []),
        "black_sheep_applied": debug.get("black_sheep_applied", []),
        "pink_sheep": debug.get("pink_sheep", []),
        "pigs": debug.get("pigs", []),
        "goats": debug.get("goats", []),
        "goat_wolf_environment": debug.get("goat_wolf_environment", []),
        "fusion": {
            "detector": debug.get("detector"),
            "raw_candidate_count": debug.get("raw_candidate_count"),
            "fused_candidate_count": debug.get("candidate_count"),
            "optimization": debug.get("optimization"),
        },
        "temporal": debug.get("temporal"),
        "manual_learning": debug.get("manual_learning", []),
        "runtime_confirmed_reviews": list(runtime_confirmed_reviews or []),
        "learned_candidate_count": debug.get("learned_candidate_count", 0),
        "learned_labels": debug.get("learned_labels", []),
        "dropped": debug.get("dropped", []),
        "raw": debug.get("raw_candidates", debug.get("candidates", [])),
        "fused": debug.get("candidates", []),
    }
