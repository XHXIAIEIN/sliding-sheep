"""Persistent capture cache for solver analysis and replay."""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from paths import ROOT

CACHE_DIR = ROOT / "cache" / "levels"
GLOBAL_SOLUTION_DIR = CACHE_DIR / "_solutions"
SOURCE_LEVEL_DIR = ROOT / "cache" / "source_levels"
CACHE_SCHEMA = 2
SOLUTION_SCHEMA = 2
SOLVER_VERSION = "2026.07-wolf-no-stop-v3"
EXECUTION_SCHEMA = 1
# 缓存条目内的相对文件名；历史缓存依赖这套扁平命名，不能随根目录布局变化。
ARTIFACTS = [
    "board.json",
    "board_layout.json",
    "sheep_candidates.json",
    "scene_report.json",
    "board_grid.json",
    "images/_game.png",
    "images/_occ_axis_rect.png",
    "images/_grid_labels.png",
    "images/_layout.png",
]
# 这些产物当前在项目根的 data/ 下生成。
DATA_ARTIFACTS = frozenset(name for name in ARTIFACTS if name.endswith(".json"))


def artifact_source(rel: str):
    """Current on-disk location of one artifact, keyed by its cache-relative name."""
    return (ROOT / "data" / rel) if rel in DATA_ARTIFACTS else (ROOT / rel)

_CAPTURE_LOCK = threading.RLock()


def source_level_key(label: str) -> str:
    """Return a stable, filesystem-safe key for a user-visible level label."""
    normalized = " ".join(str(label or "").strip().split())
    if not normalized:
        raise ValueError("关卡编号或名称不能为空")
    digest = hashlib.sha1(normalized.casefold().encode("utf-8")).hexdigest()[:12]
    return f"source-{digest}"


def _piece_records(board_data: dict) -> list[dict]:
    records = []
    for collection in (board_data.get("pieces", {}), board_data.get("returning", {})):
        for piece in collection.values():
            records.append({
                "cells": tuple(sorted(tuple(cell) for cell in piece.get("cells", []))),
                "species": str(piece.get("species", "sheep")),
                "facing": str(piece.get("facing", "")),
            })
    return records


def compare_boards(source_board: dict, current_board: dict) -> dict:
    """Compare two initial boards without relying on unstable detector ids."""
    source = _piece_records(source_board)
    current = _piece_records(current_board)
    used_current = set()
    unchanged = 0
    unmatched_source = []
    for original in source:
        match = next((index for index, item in enumerate(current)
                      if index not in used_current and item == original), None)
        if match is None:
            unmatched_source.append(original)
        else:
            used_current.add(match)
            unchanged += 1
    unmatched_current = [item for index, item in enumerate(current)
                         if index not in used_current]

    # Same cells and species but another direction are useful to distinguish
    # from a sheep that moved to a genuinely different random slot.
    direction_changes = []
    remaining_current = list(unmatched_current)
    remaining_source = []
    for original in unmatched_source:
        match = next((index for index, item in enumerate(remaining_current)
                      if item["cells"] == original["cells"]
                      and item["species"] == original["species"]), None)
        if match is None:
            remaining_source.append(original)
        else:
            changed_to = remaining_current[match]
            direction_changes.append({
                "cells": [list(cell) for cell in original["cells"]],
                "species": original["species"],
                "from": original["facing"],
                "to": changed_to["facing"],
            })
            remaining_current.pop(match)

    source_species = {}
    current_species = {}
    for item in remaining_source:
        source_species[item["species"]] = source_species.get(item["species"], 0) + 1
    for item in remaining_current:
        current_species[item["species"]] = current_species.get(item["species"], 0) + 1
    moved_or_replaced = sum(min(count, current_species.get(species, 0))
                            for species, count in source_species.items())
    source_count, current_count = len(source), len(current)
    changed = max(len(unmatched_source), len(unmatched_current))
    return {
        "source_count": source_count,
        "current_count": current_count,
        "unchanged": unchanged,
        "changed": changed,
        "direction_changed": len(direction_changes),
        "direction_changes": direction_changes,
        "moved_or_replaced": moved_or_replaced,
        "added": max(0, current_count - source_count),
        "removed": max(0, source_count - current_count),
        "same": changed == 0,
    }


def record_source_comparison(board_data: dict, label: str, *, rebuild: bool = False,
                             source: str = "detect") -> dict:
    """Create an immutable first-capture source, or compare a later restart to it."""
    normalized = " ".join(str(label or "").strip().split())
    key = source_level_key(normalized)
    folder = SOURCE_LEVEL_DIR / key
    source_path = folder / "source.json"
    previous_path = folder / "previous.json"
    direction_stats_path = folder / "direction_stats.json"
    with _CAPTURE_LOCK:
        folder.mkdir(parents=True, exist_ok=True)
        existing = _read_json(source_path) if source_path.exists() and not rebuild else None
        previous = _read_json(previous_path) if previous_path.exists() and not rebuild else None
        created = existing is None
        if created:
            payload = {
                "schema": 1,
                "level_label": normalized,
                "level_key": key,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "board_hash": board_hash(board_data),
                "board": board_data,
            }
            _atomic_json(source_path, payload)
            game_path = ROOT / "images" / "_game.png"
            if game_path.exists():
                shutil.copy2(game_path, folder / "source.png")
            comparison = compare_boards(board_data, board_data)
        else:
            comparison = compare_boards(existing.get("board", {}), board_data)
        previous_comparison = compare_boards(
            (previous or {}).get("board", board_data), board_data)
        stats_document = {} if rebuild else (_read_json(direction_stats_path) or {})
        stats = stats_document.get("positions", {})
        for piece in _piece_records(board_data):
            position_key = json.dumps(
                [piece["species"], [list(cell) for cell in piece["cells"]]],
                ensure_ascii=False, separators=(",", ":"))
            entry = stats.setdefault(position_key, {
                "cells": [list(cell) for cell in piece["cells"]],
                "species": piece["species"], "directions": [], "observations": 0,
            })
            if piece["facing"] and piece["facing"] not in entry["directions"]:
                entry["directions"].append(piece["facing"])
            entry["observations"] = int(entry.get("observations", 0)) + 1
        direction_variants = sorted(
            (entry for entry in stats.values() if len(entry.get("directions", [])) > 1),
            key=lambda item: item.get("cells", []))
        result = {
            "schema": 1,
            "level_label": normalized,
            "level_key": key,
            "baseline_created": created,
            "baseline_rebuilt": bool(rebuild and created),
            "sample_number": 1,
            "previous_sample_number": int((previous or {}).get("sample_number") or 0),
            "source": source,
            "compared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "previous_direction_changed": previous_comparison["direction_changed"],
            "previous_direction_changes": previous_comparison["direction_changes"],
            "previous_changed": previous_comparison["changed"],
            "variable_direction_count": len(direction_variants),
            "direction_variants": direction_variants,
            **comparison,
        }
        history_path = folder / "comparisons.jsonl"
        if history_path.exists():
            result["sample_number"] = sum(1 for line in history_path.read_text(
                encoding="utf-8").splitlines() if line.strip()) + 1
        current_review = bool(result["previous_direction_changed"] or result["direction_changed"])
        if current_review:
            result["review_sample_number"] = result["sample_number"]
            result["review_previous_sample_number"] = result["previous_sample_number"]
            result["review_basis"] = ("previous" if result["previous_direction_changed"]
                                      else "source")
        else:
            result["review_sample_number"] = int(
                stats_document.get("latest_review_sample") or 0)
            result["review_previous_sample_number"] = int(
                stats_document.get("latest_review_previous_sample") or 0)
            result["review_basis"] = stats_document.get("latest_review_basis")
        snapshots = folder / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        game_path = ROOT / "images" / "_game.png"
        raw_snapshot = snapshots / f"sample-{result['sample_number']:04d}.png"
        if game_path.exists():
            shutil.copy2(game_path, raw_snapshot)
        result["has_snapshot"] = raw_snapshot.exists()
        result["has_annotated_snapshot"] = False
        if result["review_sample_number"] and not current_review:
            historical = snapshots / f"sample-{result['review_sample_number']:04d}-annotated.png"
            result["has_annotated_snapshot"] = historical.exists()
        with open(history_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
        _atomic_json(folder / "latest.json", result)
        _atomic_json(snapshots / f"sample-{result['sample_number']:04d}.json", result)
        _atomic_json(previous_path, {
            "schema": 1, "level_label": normalized,
            "recorded_at": result["compared_at"], "board": board_data,
            "sample_number": result["sample_number"],
        })
        _atomic_json(direction_stats_path, {
            "schema": 1, "level_label": normalized, "positions": stats,
            "latest_review_sample": result["review_sample_number"],
            "latest_review_previous_sample": result["review_previous_sample_number"],
            "latest_review_basis": result["review_basis"],
        })
        return result


def source_snapshot_path(level_key: str, sample_number: int = 0,
                         variant: str = "current") -> Path:
    """Resolve one source-review image without allowing arbitrary filesystem access."""
    key = str(level_key or "")
    if not re.fullmatch(r"source-[0-9a-f]{12}", key):
        raise ValueError("源关卡标识无效")
    number = max(0, int(sample_number or 0))
    folder = SOURCE_LEVEL_DIR / key
    if variant == "source":
        path = folder / "source.png"
    elif variant == "annotated":
        path = folder / "snapshots" / f"sample-{number:04d}-annotated.png"
    elif variant == "current":
        path = folder / "snapshots" / f"sample-{number:04d}.png"
    else:
        raise ValueError("未知的快照类型")
    if not path.exists():
        raise FileNotFoundError("快照不存在")
    return path


def source_annotation_path(comparison: dict) -> Path:
    key = str((comparison or {}).get("level_key") or "")
    number = int((comparison or {}).get("sample_number") or 0)
    if not re.fullmatch(r"source-[0-9a-f]{12}", key) or number <= 0:
        raise ValueError("源关卡审核记录无效")
    folder = SOURCE_LEVEL_DIR / key / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"sample-{number:04d}-annotated.png"


def register_source_annotation(comparison: dict) -> dict:
    """Publish annotation availability after the image has been written."""
    result = dict(comparison or {})
    path = source_annotation_path(result)
    result["has_annotated_snapshot"] = path.exists()
    folder = SOURCE_LEVEL_DIR / result["level_key"]
    _atomic_json(folder / "latest.json", result)
    _atomic_json(folder / "snapshots" / f"sample-{int(result['sample_number']):04d}.json", result)
    return result


def _canonical_board(board_data: dict) -> dict:
    pieces = []
    for pid, piece in sorted(board_data.get("pieces", {}).items(), key=lambda item: str(item[0])):
        pieces.append({
            "id": str(pid),
            "facing": piece.get("facing"),
            "species": piece.get("species", "sheep"),
            "cells": sorted([list(cell) for cell in piece.get("cells", [])]),
            **({"awake": bool(piece.get("awake", True))}
               if piece.get("species") == "pig" else {}),
            **({"hit_limit": piece.get("hit_limit", 3),
                "hits_remaining": piece.get("hits_remaining", 3)}
               if piece.get("species") == "bomb" else {}),
        })
    returning = []
    for pid, piece in sorted(board_data.get("returning", {}).items(), key=lambda item: str(item[0])):
        returning.append({
            "id": str(pid),
            "facing": piece.get("facing"),
            "species": piece.get("species", "black_sheep"),
            "cells": sorted([list(cell) for cell in piece.get("cells", [])]),
        })
    return {
        "hash_schema": CACHE_SCHEMA,
        "rows": board_data.get("rows"),
        "cols": board_data.get("cols"),
        "model": board_data.get("model"),
        "slide_mode": board_data.get("slide_mode"),
        "hazards": sorted([list(cell) for cell in board_data.get("hazards", [])]),
        "no_stop": sorted([list(cell) for cell in board_data.get("no_stop", [])]),
        "fences": sorted([
            {"cell": list(item.get("cell", [])), "direction": item.get("direction")}
            for item in board_data.get("fences", [])
        ], key=lambda item: (item["direction"], item["cell"])),
        "rule_flags": board_data.get("rule_flags", {}),
        "solver_version": SOLVER_VERSION,
        "returning": returning,
        "pieces": pieces,
    }


def board_hash(board_data: dict) -> str:
    payload = json.dumps(_canonical_board(board_data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def next_capture_count(level_key: str) -> int:
    level_dir = CACHE_DIR / level_key
    if not level_dir.exists():
        return 1
    count = 0
    for child in level_dir.iterdir():
        if child.is_dir() and (child / "meta.json").exists():
            count += 1
    return count + 1


def _save_capture_unlocked(board_data: dict, *, level_key: str | None = None,
                           source: str = "detect", extra: dict | None = None) -> dict:
    """Archive the current generated board artifacts and return cache metadata."""
    current_hash = board_hash(board_data)
    level = (level_key or current_hash)[:16]
    remaining = (len(board_data.get("pieces", {}))
                 + len(board_data.get("returning", {})))
    capture_count = next_capture_count(level)
    salt = secrets.token_hex(4)
    capture_id = f"{level}-left{remaining:03d}-cap{capture_count:04d}-{salt}"
    level_dir = CACHE_DIR / level
    level_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = level_dir / capture_id
    staging_dir = level_dir / f".staging-{capture_id}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    copied = []
    for rel in ARTIFACTS:
        src = artifact_source(rel)
        if not src.exists():
            continue
        dst = staging_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel.replace("\\", "/"))

    meta = {
        "schema": CACHE_SCHEMA,
        "capture_id": capture_id,
        "level_id": level,
        "level_key": level,
        "board_hash": current_hash,
        "state_hash": current_hash,
        "observation_hash": _file_hash(ROOT / "images" / "_game.png"),
        "solver_version": SOLVER_VERSION,
        "remaining": remaining,
        "capture_count": capture_count,
        "salt": salt,
        "source": source,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifacts": copied,
        "extra": extra or {},
    }
    with open(staging_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    replace_error = None
    for attempt in range(6):
        try:
            staging_dir.replace(capture_dir)
            replace_error = None
            break
        except PermissionError as exc:
            replace_error = exc
            if attempt == 5:
                break
            time.sleep(.04 * (attempt + 1))
    if replace_error is not None:
        raise replace_error

    index_path = level_dir / "index.jsonl"
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + "\n")
    latest_path = CACHE_DIR / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def save_capture(board_data: dict, *, level_key: str | None = None, source: str = "detect",
                 extra: dict | None = None) -> dict:
    """Archive one observation without racing foreground and background refreshes."""
    with _CAPTURE_LOCK:
        return _save_capture_unlocked(
            board_data, level_key=level_key, source=source, extra=extra)


def capture_path(meta: dict) -> Path | None:
    if not meta:
        return None
    level_key = meta.get("level_key")
    capture_id = meta.get("capture_id")
    if not level_key or not capture_id:
        return None
    return CACHE_DIR / str(level_key) / str(capture_id)


def _solution_paths(board_data: dict, level_key: str | None = None) -> list[Path]:
    h = board_hash(board_data)
    paths = []
    if level_key:
        paths.append(CACHE_DIR / str(level_key) / "solutions" / h / "best.json")
        # Schema-2 transitional path from the first P0 implementation.
        paths.append(CACHE_DIR / str(level_key) / "solutions" / f"{h}.json")
    paths.append(GLOBAL_SOLUTION_DIR / h / "best.json")
    paths.append(GLOBAL_SOLUTION_DIR / f"{h}.json")
    if CACHE_DIR.exists():
        for path in CACHE_DIR.glob(f"*/solutions/{h}/best.json"):
            if path not in paths:
                paths.append(path)
        for path in CACHE_DIR.glob(f"*/solutions/{h}.json"):
            if path not in paths:
                paths.append(path)
    return paths


def _is_usable_solution(data: dict) -> bool:
    if data.get("usable") is False:
        return False
    if data.get("suspicious") or data.get("suspicious_dead_end"):
        return False
    moves = data.get("moves") or []
    solved = bool(data.get("solved"))
    remaining = int(data.get("remaining") or 0)
    # A zero-step solved board is valid; a zero-step board with pieces left is
    # usually a recognition or direction error and must not be replayed.
    if not moves and not solved and remaining > 0:
        return False
    return True


def _solution_quality(data: dict) -> tuple:
    """Lower is better; complete solutions always outrank partial plans."""
    moves = data.get("moves") or []
    solved = bool(data.get("solved")) and int(data.get("remaining") or 0) == 0
    suspicious = bool(data.get("suspicious") or data.get("suspicious_dead_end")
                      or data.get("usable") is False)
    if suspicious:
        return (2, int(data.get("remaining") or 1 << 20), len(moves))
    if solved:
        return (0, len(moves), 0)
    return (1, int(data.get("remaining") or 1 << 20), len(moves))


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp.replace(path)


def load_solution(board_data: dict, *, level_key: str | None = None, require_usable: bool = True,
                  require_complete: bool = False) -> dict | None:
    h = board_hash(board_data)
    for path in _solution_paths(board_data, level_key=level_key):
        if not path.exists():
            continue
        data = _read_json(path)
        if data is None:
            continue
        if data.get("schema") != SOLUTION_SCHEMA:
            continue
        if data.get("board_hash") != h:
            continue
        if require_usable and not _is_usable_solution(data):
            continue
        if require_complete and not (data.get("solved") and int(data.get("remaining") or 0) == 0):
            continue
        try:
            data["_cache_path"] = str(path.relative_to(ROOT))
        except ValueError:
            data["_cache_path"] = str(path)
        return data
    return None


def invalidate_solution(board_data: dict, cached: dict, reason: str) -> int:
    """Mark every copy of a replay-invalid best solution as unusable."""
    h = board_hash(board_data)
    revision_id = cached.get("revision_id")
    cached_moves = cached.get("moves") or []
    invalidated = 0
    for path in _solution_paths(board_data, level_key=cached.get("level_key")):
        data = _read_json(path)
        if data is None or data.get("board_hash") != h:
            continue
        same_revision = bool(revision_id and data.get("revision_id") == revision_id)
        if not same_revision and (data.get("moves") or []) != cached_moves:
            continue
        data.update({
            "usable": False,
            "suspicious": True,
            "suspicious_dead_end": True,
            "validation_error": str(reason),
            "invalidated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        _atomic_json(path, data)
        invalidated += 1
    return invalidated


def save_solution(board_data: dict, solution_data: dict, *, level_key: str | None = None,
                  capture_meta: dict | None = None, source: str = "solver") -> dict:
    """Persist a solver plan for the exact board hash, plus a copy in the capture folder."""
    h = board_hash(board_data)
    level = (level_key or (capture_meta or {}).get("level_key") or h)[:16]
    data = {
        "schema": SOLUTION_SCHEMA,
        "board_hash": h,
        "solver_version": SOLVER_VERSION,
        "level_key": level,
        "capture_id": (capture_meta or {}).get("capture_id"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
        **solution_data,
    }
    revision_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(5)}"
    data["revision_id"] = revision_id

    roots = [
        GLOBAL_SOLUTION_DIR / h,
        CACHE_DIR / level / "solutions" / h,
    ]
    cap_dir = capture_path(capture_meta or {})
    if cap_dir:
        roots.append(cap_dir / "solutions")
    selected_best = False
    best_paths = []
    for root in roots:
        revision_path = root / "revisions" / f"{revision_id}.json"
        _atomic_json(revision_path, data)
        best_path = root / "best.json"
        current = _read_json(best_path)
        if current is None or _solution_quality(data) < _solution_quality(current):
            _atomic_json(best_path, data)
            selected_best = True
        best_paths.append(best_path)
    data["selected_best"] = selected_best
    try:
        data["best_path"] = str(best_paths[1].relative_to(ROOT))
    except ValueError:
        data["best_path"] = str(best_paths[1])

    index_path = CACHE_DIR / level / "solution_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    return data


def save_feedback(feedback: dict, *, capture_meta: dict | None = None,
                  level_key: str | None = None) -> dict:
    """Persist post-click expected-vs-actual feedback for calibration review."""
    level = (level_key or (capture_meta or {}).get("level_key") or "unknown")[:16]
    data = {
        "schema": 1,
        "level_key": level,
        "capture_id": (capture_meta or {}).get("capture_id"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **feedback,
    }

    cap_dir = capture_path(capture_meta or {})
    if cap_dir:
        cap_dir.mkdir(parents=True, exist_ok=True)
        with open(cap_dir / "feedback.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    level_dir = CACHE_DIR / level
    level_dir.mkdir(parents=True, exist_ok=True)
    with open(level_dir / "feedback_index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary_path = level_dir / "learning_summary.json"
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        summary = {"schema": 2, "level_key": level, "total": 0, "matched": 0, "mismatch": 0,
                   "mismatch_types": {}, "suspect_cells": {}, "last_feedback": None}
    summary["total"] = int(summary.get("total", 0)) + 1
    if data.get("matched"):
        summary["matched"] = int(summary.get("matched", 0)) + 1
    else:
        summary["mismatch"] = int(summary.get("mismatch", 0)) + 1
        mismatch_type = data.get("mismatch_type") or "unknown"
        mismatch_types = summary.setdefault("mismatch_types", {})
        mismatch_types[mismatch_type] = int(mismatch_types.get(mismatch_type, 0)) + 1
        suspect = summary.setdefault("suspect_cells", {})
        for cell in data.get("diff", {}).get("suspect_cells", []):
            key = f"{cell[0]},{cell[1]}"
            suspect[key] = int(suspect.get(key, 0)) + 1
    summary["last_feedback"] = data
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return data


def save_execution_step(board_before: dict, piece: dict, move: dict, *,
                        level_key: str | None = None,
                        capture_meta: dict | None = None,
                        mode: str = "auto", batch_id: str | None = None,
                        batch_index: int | None = None) -> dict:
    """Persist one click intent together with the complete pre-click board.

    Execution is serialized by the app, so the number of existing step files is
    a stable, human-readable sequence number even across app restarts.
    """
    current_hash = board_hash(board_before)
    level = (level_key or (capture_meta or {}).get("level_key") or current_hash)[:16]
    execution_dir = CACHE_DIR / level / "executions"
    execution_dir.mkdir(parents=True, exist_ok=True)
    existing_steps = []
    for existing in execution_dir.glob("step-*.json"):
        try:
            existing_steps.append(int(existing.stem.split("-")[-1]))
        except (TypeError, ValueError):
            continue
    step = max(existing_steps, default=0) + 1
    execution_id = f"step-{step:04d}"
    data = {
        "schema": EXECUTION_SCHEMA,
        "execution_id": execution_id,
        "step": step,
        "level_key": level,
        "capture_id": (capture_meta or {}).get("capture_id"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": str(mode),
        "batch_id": batch_id,
        "batch_index": batch_index,
        "board_revision": current_hash,
        "piece": piece,
        "move": move,
        "board_before": board_before,
    }
    path = execution_dir / f"{execution_id}.json"
    _atomic_json(path, data)
    data["path"] = str(path.relative_to(ROOT))
    index_path = CACHE_DIR / level / "execution_index.jsonl"
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    return data
