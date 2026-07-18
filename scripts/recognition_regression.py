"""Run detector regression against cached, reviewable board snapshots.

The cached board is treated as a comparison baseline, not unquestionable ground
truth.  The JSON report keeps per-sample deltas so changed detections can be
opened beside their cached overlays and approved manually.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import cv2

import board_grid as G
import detect_occupancy as D
import safety
from paths import ROOT


def _signature_from_board(data):
    return {
        (tuple(sorted(tuple(cell) for cell in piece.get("cells", []))),
         piece.get("facing"), piece.get("species", "sheep"))
        for piece in (data.get("pieces") or {}).values()
    }


def _signature_from_detection(pieces):
    return {
        (tuple(sorted(tuple(cell) for cell in piece.get("cells", []))),
         piece.get("facing"), piece.get("species", "sheep"))
        for piece in pieces
    }


def _placement(signature):
    return {item[0] for item in signature}


def _run_one(args):
    image_path, params_path = map(Path, args)
    capture_dir = image_path.parent.parent
    board_path = capture_dir / "board.json"
    if not board_path.exists():
        return {"sample": str(capture_dir.relative_to(ROOT)), "status": "missing_board"}
    image = cv2.imread(str(image_path))
    if image is None:
        return {"sample": str(capture_dir.relative_to(ROOT)), "status": "bad_image"}
    baseline_data = json.loads(board_path.read_text(encoding="utf-8"))
    # A cache is a snapshot of one concrete capture.  Reusing the current
    # root calibration for every historical image silently turns calibration
    # drift (and resolution changes) into detector errors.  detect_occupancy
    # already writes the exact grid used for that capture, so prefer it for
    # regression and keep the root params as a compatibility fallback for old
    # cache entries.
    cached_grid_path = capture_dir / "board_grid.json"
    if cached_grid_path.exists():
        cached_grid = json.loads(cached_grid_path.read_text(encoding="utf-8"))
        corners = cached_grid.get("corners")
        rows = cached_grid.get("rows")
        cols = cached_grid.get("cols")
        if corners and rows and cols:
            grid = G.BoardGrid(
                rows=int(rows), cols=int(cols),
                corners={key: [float(value[0]), float(value[1])]
                         for key, value in corners.items()},
                image_size=(image.shape[1], image.shape[0]),
            )
        else:
            grid = G.load_grid(str(params_path), image)
    else:
        grid = G.load_grid(str(params_path), image)
    pieces, debug = D.analyze(image, grid)
    layout = D.to_layout(pieces, grid.rows, grid.cols, debug.get("dropped"),
                         hazards=debug.get("hazards"))
    report = safety.classify_scene(image, debug, pieces, grid.rows, grid.cols, layout=layout)

    baseline = _signature_from_board(baseline_data)
    current = _signature_from_detection(pieces)
    baseline_cells, current_cells = _placement(baseline), _placement(current)
    placement_common = baseline_cells & current_cells
    facing_common = baseline & current
    recall = len(placement_common) / max(1, len(baseline_cells))
    precision = len(placement_common) / max(1, len(current_cells))
    facing_accuracy = len(facing_common) / max(1, len(placement_common))
    return {
        "sample": str(capture_dir.relative_to(ROOT)),
        "status": "ok",
        "scene_state": report["scene_state"],
        "baseline_count": len(baseline), "current_count": len(current),
        "placement_recall": round(recall, 4),
        "placement_precision": round(precision, 4),
        "facing_accuracy": round(facing_accuracy, 4),
        "exact": baseline == current,
        "health_score": report["metrics"].get("health_score"),
        "review_count": report["metrics"].get("review_count"),
        "learned_candidate_count": int(debug.get("learned_candidate_count") or 0),
        "learned_label_count": len(debug.get("learned_labels") or []),
        "provisional_learning_count": int(
            report["metrics"].get("provisional_learning_count") or 0),
        "missing": [[list(cell) for cell in placement]
                    for placement in sorted(baseline_cells - current_cells)],
        "extra": [[list(cell) for cell in placement]
                  for placement in sorted(current_cells - baseline_cells)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(ROOT / "cache" / "levels"))
    parser.add_argument("--params", default=str(ROOT / "grid_params.json"))
    parser.add_argument("--output", default=str(ROOT / "cache" / "recognition_regression.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    ns = parser.parse_args(argv)
    images = sorted(Path(ns.cache).glob("*/*/images/_game.png"))
    if ns.limit > 0:
        images = images[:ns.limit]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, ns.workers)) as pool:
        futures = {pool.submit(_run_one, (str(path), ns.params)): path for path in images}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                path = futures[future]
                results.append({"sample": str(path), "status": "error", "error": str(exc)})
            if index % 10 == 0 or index == len(futures):
                print(f"processed {index}/{len(futures)}")
    results.sort(key=lambda item: item.get("sample", ""))
    ok = [item for item in results if item.get("status") == "ok"]
    summary = {
        "schema": 1,
        "samples": len(results), "ok": len(ok),
        "exact": sum(bool(item.get("exact")) for item in ok),
        "gameplay": sum(item.get("scene_state") == "gameplay" for item in ok),
        "mean_placement_recall": round(sum(item["placement_recall"] for item in ok) / max(1, len(ok)), 4),
        "mean_placement_precision": round(sum(item["placement_precision"] for item in ok) / max(1, len(ok)), 4),
        "mean_facing_accuracy": round(sum(item["facing_accuracy"] for item in ok) / max(1, len(ok)), 4),
        "learned_candidate_samples": sum(item.get("learned_candidate_count", 0) > 0 for item in ok),
        "learned_label_samples": sum(item.get("learned_label_count", 0) > 0 for item in ok),
        "provisional_learning_samples": sum(
            item.get("provisional_learning_count", 0) > 0 for item in ok),
    }
    output = Path(ns.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "samples": results}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
