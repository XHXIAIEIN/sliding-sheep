"""Safety gates shared by recognition, solving, and real execution.

This module is intentionally independent from pywebview/WinAPI so scene and
board safety decisions can be regression-tested without launching the GUI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


SCENE_STATES = {"gameplay", "transition", "victory", "popup", "loading", "unknown"}


def blocker(code: str, message: str, *, detail=None) -> dict:
    item = {"code": code, "message": message}
    if detail is not None:
        item["detail"] = detail
    return item


def _cell_label(cell) -> str:
    row, col = int(cell[0]), int(cell[1])
    column = chr(65 + col) if 0 <= col < 26 else str(col + 1)
    return f"{column}{row + 1}"


def _review_piece_detail(piece: dict) -> dict:
    cells = [list(cell) for cell in sorted(piece.get("cells") or [])]
    return {
        "id": str(piece.get("id")),
        "cells": cells,
        "location": "–".join(_cell_label(cell) for cell in cells),
        "reason": piece.get("review_reason"),
        "confidence": piece.get("confidence") or {},
    }


def _image_metrics(game: np.ndarray) -> dict:
    if game is None or game.size == 0:
        return {
            "mean_value": 0.0,
            "value_std": 0.0,
            "dark_ratio": 1.0,
            "center_edge_delta": 0.0,
            "sharpness": 0.0,
        }
    hsv = cv2.cvtColor(game, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    height, width = value.shape
    # Ignore the small white desktop title bar at the top of the mini-program.
    game_value = value[int(height * 0.04):]
    center = value[int(height * 0.18):int(height * 0.72),
                   int(width * 0.10):int(width * 0.90)]
    edge_parts = [
        value[int(height * 0.08):int(height * 0.82), :max(1, int(width * 0.08))].ravel(),
        value[int(height * 0.08):int(height * 0.82), int(width * 0.92):].ravel(),
    ]
    edge = np.concatenate([part for part in edge_parts if part.size])
    gray = cv2.cvtColor(game, cv2.COLOR_BGR2GRAY)
    return {
        "mean_value": round(float(game_value.mean()), 2),
        "value_std": round(float(game_value.std()), 2),
        "dark_ratio": round(float((game_value < 70).mean()), 4),
        "center_edge_delta": round(float(center.mean() - edge.mean()), 2),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


def validate_calibration(params: dict | None, image_shape) -> tuple[list[dict], list[dict]]:
    """Return hard blockers and non-blocking warnings for current calibration."""
    blockers, warnings = [], []
    if not params:
        return [blocker("calibration_missing", "尚未标定棋盘，禁止执行")], warnings
    try:
        rows, cols = int(params["rows"]), int(params["cols"])
        corners = [params["corners"][name] for name in ("TL", "TR", "BR", "BL")]
        points = np.asarray(corners, dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return [blocker("calibration_invalid", "标定参数字段不完整")], warnings
    if rows < 1 or cols < 1 or rows > 100 or cols > 100:
        blockers.append(blocker("calibration_grid_invalid", "标定行列数不合法"))
    if points.shape != (4, 2) or not np.isfinite(points).all():
        blockers.append(blocker("calibration_points_invalid", "标定四角不是有效坐标"))
        return blockers, warnings

    height, width = image_shape[:2]
    old_w, old_h = int(params.get("imgW") or 0), int(params.get("imgH") or 0)
    check_points = points.copy()
    if old_w and old_h and (old_w, old_h) != (width, height):
        check_points[:, 0] *= width / old_w
        check_points[:, 1] *= height / old_h
    area = abs(float(cv2.contourArea(check_points)))
    edge_lengths = np.linalg.norm(check_points - np.roll(check_points, 1, axis=0), axis=1)
    if (not cv2.isContourConvex(check_points.astype(np.int32))
            or area < 4.0 or float(edge_lengths.min()) < 2.0):
        blockers.append(blocker("calibration_geometry_invalid", "标定四角顺序、凸性或几何形状异常"))

    # Window dimensions and the board-to-window area ratio are not calibration
    # gates.  The only image-boundary requirement is that the complete board
    # quadrilateral remains present in the current capture.  A tiny tolerance
    # absorbs sub-pixel rounding when calibration is scaled to a new capture.
    boundary_tolerance = max(2.0, min(width, height) * 0.002)
    board_complete = not (
        check_points[:, 0].min() < -boundary_tolerance
        or check_points[:, 0].max() > (width - 1) + boundary_tolerance
        or check_points[:, 1].min() < -boundary_tolerance
        or check_points[:, 1].max() > (height - 1) + boundary_tolerance
    )
    if not board_complete:
        blockers.append(blocker(
            "calibration_out_of_bounds",
            "棋盘区域未完整包含在当前截图内，请重新确认标定",
            detail={"image": [width, height], "corners": check_points.round(2).tolist()},
        ))

    if old_w and old_h and (old_w, old_h) != (width, height):
        warnings.append(blocker(
            "calibration_scaled",
            "标定已按当前截图坐标自动适配",
            detail={
                "from": [old_w, old_h],
                "to": [width, height],
                "board_complete": board_complete,
            },
        ))
    return blockers, warnings


def classify_scene(game: np.ndarray, debug: dict, sheep: Iterable[dict], rows: int, cols: int,
                   *, calibration_blockers=None, layout=None) -> dict:
    """Classify a capture conservatively and assemble execution blockers.

    The first hard rule targets a real cached failure mode: modal/popup captures
    have a bright central card over a strongly dimmed background.  Unknown or
    empty scenes are deliberately not promoted to gameplay.
    """
    sheep = list(sheep or [])
    debug = debug or {}
    metrics = _image_metrics(game)
    candidate_count = int(debug.get("candidate_count") or 0)
    hazard_count = len(debug.get("hazards") or [])
    gesture = debug.get("gesture") or None
    review_pieces = [piece for piece in sheep if piece.get("review")]
    review_count = len(review_pieces)
    provisional_learning_count = sum(
        1 for piece in sheep if not piece.get("runtime_confirmed") and (
            piece.get("learned_provisional")
            or piece.get("learned_direction_provisional") or (
                piece.get("learned_template") and int(piece.get("learned_support") or 0) < 2)
        )
    ) + int(debug.get("provisional_learning_rejection_count") or 0)
    confidences = [piece.get("confidence") or {} for piece in sheep]
    confidence_fields = {
        name: round(float(np.mean([float(item.get(name, 0.0)) for item in confidences])), 4)
        if confidences else 0.0
        for name in ("occupancy", "axis", "facing", "species")
    }
    total_cells = max(1, int(rows) * int(cols))
    body_mask = debug.get("body_mask")
    body_ratio = float((body_mask > 0).mean()) if isinstance(body_mask, np.ndarray) and body_mask.size else 0.0
    height, width = game.shape[:2]
    hsv = cv2.cvtColor(game, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
           & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 120))
    green = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90)
             & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 100))
    victory_red_ratio = float(red[
        int(height * .08):int(height * .34),
        int(width * .06):int(width * .94),
    ].mean())
    victory_green_ratio = float(green[
        int(height * .72):int(height * .96),
        int(width * .20):int(width * .80),
    ].mean())
    # The real clear screen has a broad saturated red congratulation ribbon
    # plus a large green next-level button.  Requiring both separates it from
    # the beige failure card, which also dims the board and used to be folded
    # into the generic popup branch.
    victory_overlay = victory_red_ratio >= .14 and victory_green_ratio >= .08
    metrics.update({
        "candidate_count": candidate_count,
        "piece_count": len(sheep),
        "hazard_count": hazard_count,
        "gesture_occlusion_count": len((gesture or {}).get("components") or []),
        "review_count": review_count,
        "provisional_learning_count": provisional_learning_count,
        "body_ratio": round(body_ratio, 4),
        "victory_red_ratio": round(victory_red_ratio, 4),
        "victory_green_ratio": round(victory_green_ratio, 4),
        "victory_overlay": victory_overlay,
        "confidence": confidence_fields,
    })

    if victory_overlay:
        state = "victory"
        reason = "检测到过关红色横幅与下一关绿色按钮"
    elif metrics["dark_ratio"] >= 0.25 and metrics["center_edge_delta"] >= 75:
        state = "popup"
        reason = "检测到中央亮色弹窗与大面积背景变暗"
    elif metrics["mean_value"] < 38 or metrics["value_std"] < 12 or metrics["sharpness"] < 4:
        state = "loading"
        reason = "画面过暗、过于单一或缺少稳定纹理"
    elif not sheep:
        state = "victory" if candidate_count == 0 and metrics["mean_value"] >= 120 else "unknown"
        reason = "没有检测到可确认的棋子"
    elif candidate_count >= 4 and len(sheep) / max(1, candidate_count) < 0.35:
        state = "transition"
        reason = "候选与保留棋子差异过大，疑似转场或特效"
    elif body_ratio < 0.001 or body_ratio > 0.62:
        state = "unknown"
        reason = "棋盘主体掩膜比例异常"
    else:
        state = "gameplay"
        reason = "棋盘与棋子证据满足执行前提"

    blockers = list(calibration_blockers or [])
    advisories = []
    if state != "gameplay":
        blockers.append(blocker("scene_not_gameplay", f"当前场景为 {state}，禁止识别求解和点击"))
    if review_count:
        review_details = [_review_piece_detail(piece) for piece in review_pieces]
        locations = "、".join(
            f"#{piece['id']}（{piece['location']}）" for piece in review_details)
        advisories.append(blocker(
            "manual_review_required",
            f"低置信度棋子：{locations}；请确认或修正，自动执行会先避让它",
            detail={"pieces": review_details},
        ))
    if provisional_learning_count:
        blockers.append(blocker(
            "manual_learning_confirmation_required",
            f"有 {provisional_learning_count} 个单样本学习候选，需在另一张截图再次确认后才能自动执行",
        ))
    tutorial_hand_count = 0
    if gesture and gesture.get("blocking", True):
        affected = list(gesture.get("affected_cells") or [])
        components = gesture.get("components") or []
        smoke = [item for item in components
                 if item.get("blocking") and item.get("kind") == "motion_smoke"]
        hands = [item for item in components
                 if item.get("blocking") and item.get("kind") != "motion_smoke"]
        if hands:
            tutorial_hand_count = len(hands)
            advisories.append(blocker(
                "gesture_occlusion",
                "检测到教程手势；仅作为画面提示，不阻止或改变执行计划",
                detail={"affected_cells": affected, "components": hands},
            ))
        if smoke:
            blockers.append(blocker(
                "motion_smoke",
                "检测到羊移动烟尘仍覆盖棋盘，当前帧不用于继续执行",
                detail={"affected_cells": affected, "components": smoke},
            ))
    conflicts = list((layout or {}).get("conflicts") or [])
    if conflicts:
        blockers.append(blocker("piece_overlap", "检测到棋子占格冲突", detail=conflicts))
    if hazard_count > max(8, int(total_cells * 0.10)):
        blockers.append(blocker("hazard_detection_overflow", "危险格数量异常，疑似场景误识别"))
    temporal = debug.get("temporal") or {}
    uncertain_hazards = list(temporal.get("uncertain_hazard_cells") or [])
    if temporal.get("history_frames", 0) and uncertain_hazards:
        advisories.append(blocker(
            "dynamic_hazard_unstable",
            f"有 {len(uncertain_hazards)} 个危险格在多帧间变化，自动模式将降为单步预检",
            detail=uncertain_hazards,
        ))

    confidence_mean = float(np.mean(list(confidence_fields.values()))) if confidences else 0.0
    health_score = 100.0 * confidence_mean
    health_score -= min(35.0, review_count * 8.0)
    health_score -= min(25.0, len(conflicts) * 12.0)
    health_score -= min(20.0, len(uncertain_hazards) * 4.0)
    # The hand can reduce recognition coverage, but it is a tutorial overlay,
    # not a reason to reject a click.  Keep the confidence signal without
    # putting it into execution_blockers.
    health_score -= min(12.0, tutorial_hand_count * 6.0)
    if state != "gameplay":
        health_score = min(health_score, 30.0)
    metrics["health_score"] = round(max(0.0, min(100.0, health_score)), 2)

    return {
        "scene_state": state,
        "scene_reason": reason,
        "metrics": metrics,
        "advisories": advisories,
        "execution_blockers": blockers,
        "executable": state == "gameplay" and not blockers,
        "execution_complete": state == "victory" and (
            victory_overlay or (not sheep and candidate_count == 0)),
    }


def add_blockers(report: dict, items: Iterable[dict]) -> dict:
    report = dict(report)
    merged = list(report.get("execution_blockers") or [])
    seen = {item.get("code") for item in merged}
    for item in items:
        if item.get("code") not in seen:
            merged.append(item)
            seen.add(item.get("code"))
    report["execution_blockers"] = merged
    report["executable"] = report.get("scene_state") == "gameplay" and not merged
    return report
