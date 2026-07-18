"""Full-frame analysis pipeline and CLI entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from board import grid as G
from levels import cache as level_cache
from core import safety
from board import io as board_io
import recognition
from paths import (BOARD_GRID_JSON, BOARD_JSON, BOARD_LAYOUT_JSON,
                   GRID_PARAMS_JSON, SCENE_REPORT_JSON,
                   SHEEP_CANDIDATES_JSON, image_path)
from .conflicts import (apply_species_anchors, reject_departing_edge_pieces,
                        reject_hazard_piece_overlaps, reject_internal_fence_overlaps,
                        reject_partial_exit_candidates, resolve_candidates,
                        resolve_goat_wolf_conflicts, suppress_special_hazard_overlaps)
from .detectors import (_arrow_candidates, _cattle_candidates,
                        _gesture_target_arrow_candidates, _rocket_candidates,
                        bomb_markers, classify_black_sheep, elephant_pieces,
                        fence_edges, goat_candidates, pig_candidates,
                        pink_sheep_candidates, recover_black_sheep_clusters,
                        wolf_hazards)
from .export import _conflicts, _write_json, to_board, to_layout
from .masks import _grid_from_args, arrow_mask, gesture_occlusion, make_masks
from .render import _remove_obsolete_images, render_grid_labels, render_layout, render_rect_debug
from .segmentation import _score_region, watershed_regions


def analyze(game: np.ndarray, grid: G.BoardGrid, temporal_history=None,
            *, recover_missing_edges=False, ignore_outside_pieces=True):
    rect = grid.warp(game)
    gesture_mask, gesture_meta = gesture_occlusion(rect, grid.rows, grid.cols)
    body_mask, face_mask, dt = make_masks(rect, gesture_mask)
    markers, nseed = watershed_regions(rect, body_mask, dt)
    hazards, wolf_mask, wolf_meta = wolf_hazards(
        rect, grid.rows, grid.cols, gesture_mask)
    elephants, elephant_mask, elephant_meta = elephant_pieces(
        rect, grid.rows, grid.cols, gesture_mask)
    fences, fence_mask, fence_meta = fence_edges(rect, grid.rows, grid.cols)

    arrow_candidates = _arrow_candidates(
        rect, body_mask, grid.rows, grid.cols, gesture_mask)
    for candidate in arrow_candidates:
        candidate["detector"] = "arrow"
    gesture_target_candidates = _gesture_target_arrow_candidates(
        rect, body_mask, grid.rows, grid.cols, gesture_meta, arrow_candidates)
    body_candidates = []
    for idv in range(1, nseed + 1):
        cand = _score_region(idv, markers, body_mask, face_mask, dt, grid.rows, grid.cols)
        if cand is not None:
            cand["detector"] = "body"
            body_candidates.append(cand)
    rocket_candidates, rocket_mask = _rocket_candidates(
        rect, grid.rows, grid.cols, gesture_mask)
    cattle_candidates = _cattle_candidates(
        rect, grid.rows, grid.cols, gesture_mask)
    pink_candidates, pink_mask, pink_components = pink_sheep_candidates(
        rect, body_mask, face_mask, grid.rows, grid.cols, gesture_mask)
    pigs, pig_mask, pig_components = pig_candidates(
        rect, grid.rows, grid.cols, gesture_mask)
    goats, goat_mask, goat_components = goat_candidates(
        rect, grid.rows, grid.cols, gesture_mask)
    raw_candidates = (arrow_candidates + gesture_target_candidates
                      + body_candidates + rocket_candidates
                      + cattle_candidates + pink_candidates + pigs + goats)
    # Bomb counters are current-frame species evidence.  Attach them before
    # consulting saved templates so a manually corrected bomb footprint can
    # use native bomb candidates as its visual anchor.
    bomb_meta, bomb_mask = bomb_markers(rect, grid.rows, grid.cols, gesture_mask)
    for candidate in raw_candidates:
        cells = {tuple(cell) for cell in candidate.get("cells", [])}
        marker = next((item for item in bomb_meta if tuple(item["cell"]) in cells), None)
        if marker and candidate.get("species", "sheep") in {"sheep", "rocket", "bomb"}:
            candidate["species"] = "bomb"
            candidate["hit_limit"] = marker["hit_limit"]
            candidate["hits_remaining"] = marker["hits_remaining"]
            candidate["counter_confident"] = marker["counter_confident"]
            candidate["counter_confidence"] = marker["counter_confidence"]
            candidate["counter_unknown"] = marker["counter_unknown"]
            candidate.setdefault("direction_votes", {})["bomb_counter"] = list(marker["cell"])
            candidate["direction_votes"]["bomb_counter_value"] = marker["hits_remaining"]
            candidate["direction_votes"]["bomb_counter_confidence"] = marker["counter_confidence"]
    learned_candidates, manual_learning = recognition.manual_candidate_proposals(
        rect, grid.rows, grid.cols, raw_candidates)
    raw_candidates += learned_candidates
    raw_candidates, fence_candidate_rejected = reject_internal_fence_overlaps(
        raw_candidates, fences)
    bomb_learning_rejected = []
    bomb_checked_candidates = []
    for candidate in raw_candidates:
        cells = {tuple(cell) for cell in candidate.get("cells", [])}
        marker = next((item for item in bomb_meta if tuple(item["cell"]) in cells), None)
        if (marker and candidate.get("learned_template")
                and candidate.get("species", "sheep") != "bomb"):
            bomb_learning_rejected.append({
                **candidate, "drop_reason": "strong_bomb_counter_override",
                "bomb_cell": list(marker["cell"]),
            })
            continue
        if marker and candidate.get("species", "sheep") in {"sheep", "rocket", "bomb"}:
            candidate["species"] = "bomb"
            candidate["hit_limit"] = marker["hit_limit"]
            candidate["hits_remaining"] = marker["hits_remaining"]
            candidate["counter_confident"] = marker["counter_confident"]
            candidate["counter_confidence"] = marker["counter_confidence"]
            candidate["counter_unknown"] = marker["counter_unknown"]
            candidate.setdefault("direction_votes", {})["bomb_counter"] = list(marker["cell"])
            candidate["direction_votes"]["bomb_counter_value"] = marker["hits_remaining"]
            candidate["direction_votes"]["bomb_counter_confidence"] = marker["counter_confidence"]
        bomb_checked_candidates.append(candidate)
    raw_candidates = bomb_checked_candidates
    raw_candidates, manual_learning_rejected, manual_rejections = (
        recognition.manual_candidate_rejections(rect, raw_candidates)
    )
    eligible_candidates, outside_rejected = reject_partial_exit_candidates(
        raw_candidates, body_mask, grid.rows, grid.cols,
        enabled=bool(ignore_outside_pieces))
    candidates, fusion_rejected = recognition.fuse_candidates(eligible_candidates)
    sheep, dropped, optimization = resolve_candidates(candidates, return_meta=True)
    elephant_cells = {tuple(cell) for item in elephants for cell in item["cells"]}
    if elephant_cells:
        retained = []
        for piece in sheep:
            overlap = elephant_cells & {tuple(cell) for cell in piece.get("cells", [])}
            if overlap:
                dropped.append({**piece, "drop_reason": "elephant_occupancy_override",
                                "overlap": [list(cell) for cell in sorted(overlap)]})
            else:
                retained.append(piece)
        sheep = retained + elephants
    sheep, hazards, temporal = recognition.apply_temporal(
        sheep, hazards, list(temporal_history or []), grid.rows, grid.cols,
        recover_missing_edges=(recover_missing_edges and not ignore_outside_pieces))
    sheep, fence_temporal_rejected = reject_internal_fence_overlaps(sheep, fences)
    # Current-frame species anchors override older temporal labels.  Resolve
    # overlapping color detectors in one place so a pig mask cannot overwrite
    # the more distinctive pink-sheep bow.
    apply_species_anchors(sheep, pink_candidates, pigs, goats)
    sheep, departing_rejected = reject_departing_edge_pieces(
        sheep, body_mask, grid.rows, grid.cols,
        enabled=bool(ignore_outside_pieces))
    # A stable-history vote must not turn a currently visible bomb back into a
    # rocket.  The counter disc is direct frame evidence and wins here.
    for piece in sheep:
        cells = {tuple(cell) for cell in piece.get("cells", [])}
        marker = next((item for item in bomb_meta if tuple(item["cell"]) in cells), None)
        if marker and piece.get("species") in {"sheep", "rocket", "bomb"}:
            piece["species"] = "bomb"
            piece["hit_limit"] = marker["hit_limit"]
            piece["hits_remaining"] = marker["hits_remaining"]
            piece["counter_confident"] = marker["counter_confident"]
            piece["counter_confidence"] = marker["counter_confidence"]
            piece["counter_unknown"] = marker["counter_unknown"]
            piece.setdefault("direction_votes", {})["bomb_counter"] = list(marker["cell"])
            piece["direction_votes"]["bomb_counter_value"] = marker["hits_remaining"]
            piece["direction_votes"]["bomb_counter_confidence"] = marker["counter_confidence"]
    # Manual direction evidence is authoritative over stale temporal votes.
    # Apply it only after current-frame species anchors (pig/goat/bomb) settle.
    sheep, learned_directions = recognition.apply_direction_learning(sheep)
    sheep, learned_labels = recognition.apply_manual_label_learning(sheep, rect)
    sheep, black_sheep_cluster = recover_black_sheep_clusters(
        sheep, wolf_meta, rect, grid.rows, grid.cols, gesture_mask)
    sheep, goat_wolf_rejected, goat_wolf_environment = resolve_goat_wolf_conflicts(
        sheep, hazards, grid.rows, grid.cols)
    hazards, cluster_suppressed_hazards = suppress_special_hazard_overlaps(sheep, hazards)
    sheep, black_sheep_applied = classify_black_sheep(sheep, hazards)
    hazards, isolated_suppressed_hazards = suppress_special_hazard_overlaps(sheep, hazards)
    suppressed_hazards = cluster_suppressed_hazards + isolated_suppressed_hazards
    sheep, wolf_overlap_rejected = reject_hazard_piece_overlaps(sheep, hazards)
    # Candidate order and temporal restoration order are implementation
    # details.  Stable spatial ids keep equivalent boards on the same revision
    # and make cached move ids reproducible across adjacent captures.
    sheep.sort(key=lambda item: (
        min(tuple(cell) for cell in item.get("cells", [[grid.rows, grid.cols]])),
        tuple(sorted(tuple(cell) for cell in item.get("cells", []))),
        str(item.get("facing") or ""), str(item.get("species") or "sheep"),
    ))
    for piece_id, piece in enumerate(sheep):
        piece["id"] = piece_id
    dropped = (fence_candidate_rejected + fence_temporal_rejected
               + bomb_learning_rejected + manual_learning_rejected
               + outside_rejected + departing_rejected
               + fusion_rejected + dropped + goat_wolf_rejected + wolf_overlap_rejected)
    debug = {
        "grid": grid,
        "rect": rect,
        "body_mask": body_mask,
        "face_mask": face_mask,
        "arrow_mask": arrow_mask(rect, gesture_mask),
        "rocket_mask": rocket_mask,
        "pink_mask": pink_mask,
        "pink_sheep": pink_components,
        "pig_mask": pig_mask,
        "pigs": pig_components,
        "goat_mask": goat_mask,
        "goats": goat_components,
        "goat_wolf_environment": goat_wolf_environment,
        "bomb_mask": bomb_mask,
        "bomb_meta": bomb_meta,
        "gesture_mask": gesture_mask,
        "gesture": gesture_meta,
        "wolf_mask": wolf_mask,
        "wolf_meta": wolf_meta,
        "elephant_mask": elephant_mask,
        "elephant_meta": elephant_meta,
        "fence_mask": fence_mask,
        "fence_meta": fence_meta,
        "fences": fences,
        "hazards": hazards,
        "suppressed_hazards": suppressed_hazards,
        "black_sheep_cluster": black_sheep_cluster,
        "black_sheep_applied": black_sheep_applied,
        "markers": markers,
        "detector": "fusion",
        "candidate_count": len(candidates),
        "raw_candidate_count": len(raw_candidates),
        "outside_rejected_count": len(outside_rejected),
        "departing_rejected_count": len(departing_rejected),
        "ignore_outside_pieces": bool(ignore_outside_pieces),
        "learned_directions": learned_directions,
        "learned_labels": learned_labels,
        "manual_learning": manual_learning + manual_rejections,
        "manual_learning_rejections": manual_rejections,
        "provisional_learning_rejection_count": len({
            tuple(sorted(tuple(cell) for cell in item.get("cells", [])))
            for item in manual_learning_rejected if item.get("learned_provisional")
        }),
        "learned_candidate_count": sum(
            bool(item.get("learned_template")) for item in raw_candidates),
        "learned_rejected": bomb_learning_rejected,
        "dropped": dropped,
        "candidates": candidates,
        "raw_candidates": raw_candidates,
        "optimization": optimization,
        "temporal": temporal,
        "observation": recognition.observation_record(sheep, hazards, grid.rows, grid.cols),
    }
    return sheep, debug


def detect(game, corners, rows, cols):
    """Return sheep list: each has cells/rump/head/facing. cells[0]=rump, cells[1]=head."""
    grid = _grid_from_args(game, corners, rows, cols)
    sheep, _debug = analyze(game, grid)
    return sheep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(image_path("_game.png")))
    ap.add_argument("--params", default=str(GRID_PARAMS_JSON))
    ap.add_argument("--board", default=str(BOARD_JSON))
    args = ap.parse_args(argv)

    game = cv2.imread(args.image)
    if game is None:
        raise SystemExit(f"读不到图片: {args.image}")
    _remove_obsolete_images()
    grid = G.load_grid(args.params, game)
    G.save_grid_data(grid, BOARD_GRID_JSON)

    sheep, debug = analyze(game, grid)
    occ, conflicts = _conflicts(sheep)
    vertical = sum(1 for s in sheep if s["axis"] == "V")
    facing_n = " ".join(f"{d}{sum(1 for s in sheep if s['facing'] == d)}" for d in "UDLR")

    print(f"网格 {grid.rows}x{grid.cols}  -> board_grid.json")
    print(f"候选 {debug['candidate_count']}  保留 {len(sheep)}  丢弃 {len(debug['dropped'])}")
    print(f"羊 {len(sheep)} 只  竖 {vertical}  横 {len(sheep)-vertical}   头向 {facing_n}")
    print(f"狼危险格 {len(debug.get('hazards', []))}")
    print(f"占用格 {len(occ)}/{grid.rows * grid.cols}  冲突 {len(conflicts)}")
    for rc, ids in sorted(conflicts.items()):
        print(f"  冲突 {rc} <- {ids}")

    board_data = to_board(sheep, grid.rows, grid.cols, hazards=debug.get("hazards"),
                          fences=debug.get("fences"))
    layout_data = to_layout(sheep, grid.rows, grid.cols, debug["dropped"],
                            hazards=debug.get("hazards"), fences=debug.get("fences"))
    try:
        params_data = json.load(open(args.params, encoding="utf-8"))
    except Exception:
        params_data = None
    calibration_blockers, calibration_warnings = safety.validate_calibration(params_data, game.shape)
    report = safety.classify_scene(
        game, debug, sheep, grid.rows, grid.cols,
        calibration_blockers=calibration_blockers, layout=layout_data)
    report["warnings"] = calibration_warnings
    try:
        board_io.validate_board_data(board_data)
    except board_io.BoardValidationError as exc:
        report = safety.add_blockers(report, [
            safety.blocker("board_schema_invalid", "棋盘结构校验失败", detail=exc.errors)
        ])
    _write_json(SCENE_REPORT_JSON, report)
    _write_json(SHEEP_CANDIDATES_JSON, {
        "scene_state": report["scene_state"],
        "scene_reason": report["scene_reason"],
        "execution_blockers": report["execution_blockers"],
        "metrics": report["metrics"],
        "kept": sheep,
        "hazards": debug.get("hazards", []),
        "wolf": debug.get("wolf_meta"),
        "black_sheep_cluster": debug.get("black_sheep_cluster", []),
        "black_sheep_applied": debug.get("black_sheep_applied", []),
        "pink_sheep": debug.get("pink_sheep", []),
        "pigs": debug.get("pigs", []),
        "goats": debug.get("goats", []),
        "fusion": {"detector": debug.get("detector"),
                   "raw_candidate_count": debug.get("raw_candidate_count"),
                   "fused_candidate_count": debug.get("candidate_count"),
                   "optimization": debug.get("optimization")},
        "temporal": debug.get("temporal"),
        "dropped": debug["dropped"],
        "raw": debug.get("raw_candidates", debug["candidates"]),
        "fused": debug["candidates"],
    })

    cv2.imwrite(str(image_path("_occ_axis_rect.png")), render_rect_debug(debug, sheep))
    cv2.imwrite(str(image_path("_grid_labels.png")), render_grid_labels(debug, sheep))
    cv2.imwrite(str(image_path("_layout.png")), render_layout(debug, sheep))
    if report["scene_state"] != "gameplay":
        for stale in (Path(args.board), BOARD_LAYOUT_JSON):
            if stale.exists():
                stale.unlink()
        print(f"场景 {report['scene_state']}：{report['scene_reason']}")
        print("禁止生成 board.json、求解和执行")
        raise SystemExit(2)

    _write_json(args.board, board_data)
    _write_json(BOARD_LAYOUT_JSON, layout_data)
    cache_meta = level_cache.save_capture(
        board_data,
        source="cli-detect",
        extra={"rows": grid.rows, "cols": grid.cols, "candidate_count": debug["candidate_count"],
               "hazard_count": len(debug.get("hazards", [])),
               "scene_state": report["scene_state"], "executable": report["executable"],
               "execution_blockers": [item["code"] for item in report["execution_blockers"]]},
    )
    print("saved data/board.json data/board_layout.json data/sheep_candidates.json images/_occ_axis_rect.png images/_grid_labels.png images/_layout.png")
    print(f"cache {cache_meta['capture_id']} -> cache/levels/{cache_meta['level_key']}")
