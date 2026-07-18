"""Detector fusion, global cell assignment, and temporal board evidence.

The visual detectors deliberately remain in ``detect_occupancy.py``.  This
module owns the model-level decisions so they can be tested with synthetic
candidates instead of screenshots:

* calibrate and fuse independent detector candidates;
* choose the maximum-weight, non-overlapping set globally;
* stabilize facing/species confidence over recent frames;
* expose a dynamic hazard timeline without silently inventing safe cells.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


DETECTOR_RELIABILITY = {
    "arrow": 0.98,
    "gesture-target-arrow": 0.96,
    "rocket": 0.96,
    "pink-bow": 0.97,
    "pig-body": 0.96,
    "goat-body": 0.97,
    "learned-template": 0.84,
    "body": 0.78,
    "cattle-body": 0.88,
    "cattle-face": 0.74,
    "cattle-cell": 0.52,
    "unknown": 0.45,
}

DIRECTION_LEARNING_DIR = Path(__file__).resolve().parent.parent / "cache" / "direction_learning"
DIRECTION_LEARNING_INDEX = DIRECTION_LEARNING_DIR / "index.jsonl"
MANUAL_LEARNING_DIR = Path(__file__).resolve().parent.parent / "cache" / "recognition_learning"
MANUAL_LEARNING_INDEX = MANUAL_LEARNING_DIR / "index.jsonl"
MANUAL_LEARNING_SCHEMA = 2
MAX_ACTIVE_CONFIRMATIONS = 128
PAIR_FEATURE_SCHEMA = "rect-pair-v1"
MANUAL_PRESENCE_SPECIES = frozenset({
    "sheep", "rocket", "bomb", "pink_sheep", "pig", "goat", "black_sheep", "cattle",
})
MANUAL_LABEL_SPECIES = MANUAL_PRESENCE_SPECIES
PAIR_FEATURE_NAMES = (
    "red", "orange", "yellow", "green", "cyan", "blue", "magenta",
    "white", "dark", "skin", "edge", "saturation", "value",
)
CELL_SIZE = 64
_MANUAL_LEARNING_LOCK = threading.RLock()
_MANUAL_BUNDLE_CACHE_KEY = None
_MANUAL_BUNDLE_CACHE_ITEMS = []


@contextmanager
def _manual_learning_file_lock():
    """Serialize index publication across desktop/helper processes on Windows."""
    MANUAL_LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = MANUAL_LEARNING_DIR / ".index.lock"
    stream = open(lock_path, "a+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            locked = True
        except (ImportError, OSError):
            locked = False
        try:
            yield
        finally:
            if locked:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        stream.close()


def cell_key(candidate: dict) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(tuple(int(v) for v in cell) for cell in candidate.get("cells", [])))


def _clip(value, low=0.0, high=1.0):
    return float(max(low, min(high, value)))


def _cell_visual_stats(rect: np.ndarray, cell: tuple[int, int], cell_size=CELL_SIZE) -> list[float]:
    """Return compact, position-independent colour/edge evidence for one grid cell."""
    row, col = (int(cell[0]), int(cell[1]))
    y0, y1 = row * cell_size, (row + 1) * cell_size
    x0, x1 = col * cell_size, (col + 1) * cell_size
    patch = rect[y0:y1, x0:x1]
    if patch.size == 0:
        return [0.0] * len(PAIR_FEATURE_NAMES)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    strong = (sat >= 70) & (val >= 55)
    masks = (
        (((hue <= 7) | (hue >= 172)) & strong),
        ((hue >= 8) & (hue < 25) & strong),
        ((hue >= 25) & (hue < 40) & strong),
        ((hue >= 40) & (hue < 76) & strong),
        ((hue >= 76) & (hue < 103) & strong),
        ((hue >= 103) & (hue < 136) & strong),
        ((hue >= 136) & (hue < 172) & strong),
        ((sat <= 60) & (val >= 145)),
        (val <= 80),
        ((hue <= 24) & (sat >= 35) & (sat <= 215) & (val >= 55) & (val <= 245)),
    )
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 160)
    values = [float(mask.mean()) for mask in masks]
    values.extend((float((edge > 0).mean()), float(sat.mean() / 255.0),
                   float(val.mean() / 255.0)))
    return [round(value, 6) for value in values]


def pair_visual_feature(rect: np.ndarray, piece_or_cells, *, cell_size=CELL_SIZE) -> dict | None:
    """Describe an adjacent two-cell visual without using its absolute board position."""
    cells = cell_key(piece_or_cells if isinstance(piece_or_cells, dict)
                     else {"cells": piece_or_cells})
    if len(cells) != 2:
        return None
    if abs(cells[0][0] - cells[1][0]) + abs(cells[0][1] - cells[1][1]) != 1:
        return None
    axis = "H" if cells[0][0] == cells[1][0] else "V"
    ordered = sorted(cells, key=lambda rc: rc[1] if axis == "H" else rc[0])
    low = _cell_visual_stats(rect, ordered[0], cell_size)
    high = _cell_visual_stats(rect, ordered[1], cell_size)
    symmetric = [round((a + b) * 0.5, 6) for a, b in zip(low, high)]
    endpoint = [round(b - a, 6) for a, b in zip(low, high)]
    rows, cols = [cell[0] for cell in ordered], [cell[1] for cell in ordered]
    crop = rect[min(rows) * cell_size:(max(rows) + 1) * cell_size,
                min(cols) * cell_size:(max(cols) + 1) * cell_size]
    patch_hash = hashlib.sha1(crop.tobytes()).hexdigest() if crop.size else None
    return {
        "schema": PAIR_FEATURE_SCHEMA,
        "axis": axis,
        "names": list(PAIR_FEATURE_NAMES),
        "symmetric": symmetric,
        "endpoint": endpoint,
        "patch_hash": patch_hash,
    }


def _feature_distance(first: dict, second: dict, field: str) -> float:
    a, b = first.get(field) or [], second.get(field) or []
    if (first.get("schema") != PAIR_FEATURE_SCHEMA
            or second.get("schema") != PAIR_FEATURE_SCHEMA
            or first.get("axis") != second.get("axis")
            or len(a) != len(b) or not a):
        return math.inf
    values = [(float(x) - float(y)) ** 2 for x, y in zip(a, b)]
    if not all(math.isfinite(value) for value in values):
        return math.inf
    return math.sqrt(sum(values) / len(values))


def board_corrections(detected: dict, manual: dict) -> list[dict]:
    """Diff detector output and a confirmed editor board by spatial footprint."""
    before_pieces = detected.get("pieces") or {}
    after_pieces = manual.get("pieces") or {}
    before_by_cells = {cell_key(piece): (str(pid), deepcopy(piece))
                       for pid, piece in before_pieces.items()}
    after_by_cells = {cell_key(piece): (str(pid), deepcopy(piece))
                      for pid, piece in after_pieces.items()}
    corrections = []
    for placement in sorted(set(before_by_cells) | set(after_by_cells)):
        before_item, after_item = before_by_cells.get(placement), after_by_cells.get(placement)
        if before_item is None:
            pid, after = after_item
            corrections.append({"kind": "add", "fields": ["presence", "species", "facing"],
                                "before_id": None, "after_id": pid,
                                "before": None, "after": after})
            continue
        if after_item is None:
            pid, before = before_item
            corrections.append({"kind": "delete", "fields": ["presence"],
                                "before_id": pid, "after_id": None,
                                "before": before, "after": None})
            continue
        before_id, before = before_item
        after_id, after = after_item
        fields = [name for name in ("species", "facing", "awake", "hit_limit", "hits_remaining")
                  if before.get(name) != after.get(name)]
        if fields:
            corrections.append({"kind": "update", "fields": fields,
                                "before_id": before_id, "after_id": after_id,
                                "before": before, "after": after})
    before_hazards = {tuple(cell) for cell in detected.get("hazards") or []}
    after_hazards = {tuple(cell) for cell in manual.get("hazards") or []}
    for cell in sorted(after_hazards - before_hazards):
        corrections.append({"kind": "add_hazard", "fields": ["hazard"],
                            "before": None, "after": {"cell": list(cell)}})
    for cell in sorted(before_hazards - after_hazards):
        corrections.append({"kind": "delete_hazard", "fields": ["hazard"],
                            "before": {"cell": list(cell)}, "after": None})
    return corrections


def _valid_manual_learning_record(item: dict) -> bool:
    try:
        if (not isinstance(item, dict)
                or int(item.get("schema") or 0) != MANUAL_LEARNING_SCHEMA
                or item.get("status", "active") != "active"
                or item.get("recognition_version") != "manual-supervision-v2"
                or int(item.get("taxonomy_version") or 0) != 1
                or not str(item.get("observation_hash") or "")):
            return False
        correction = item.get("correction")
        feature = item.get("feature")
        if not isinstance(correction, dict) or not isinstance(feature, dict):
            return False
        if correction.get("kind") not in {"add", "update", "delete", "confirm"}:
            return False
        fields = correction.get("fields")
        if not isinstance(fields, list) or not fields:
            return False
        target = correction.get("after") or correction.get("before") or {}
        placement = cell_key(target)
        if (len(placement) != 2
                or abs(placement[0][0] - placement[1][0])
                   + abs(placement[0][1] - placement[1][1]) != 1):
            return False
        if (feature.get("schema") != PAIR_FEATURE_SCHEMA
                or feature.get("axis") not in {"H", "V"}
                or feature.get("names") != list(PAIR_FEATURE_NAMES)):
            return False
        symmetric, endpoint = feature.get("symmetric"), feature.get("endpoint")
        if (not isinstance(symmetric, list) or not isinstance(endpoint, list)
                or len(symmetric) != len(PAIR_FEATURE_NAMES)
                or len(endpoint) != len(PAIR_FEATURE_NAMES)):
            return False
        if not all(math.isfinite(float(value)) for value in symmetric + endpoint):
            return False
        facing = str(target.get("facing") or "")
        if facing not in ({"L", "R"} if feature["axis"] == "H" else {"U", "D"}):
            return False
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return True


def _load_saved_sample_confirmations() -> list[dict]:
    """Read complete, human-saved boards as positive recognition examples.

    Older sample bundles published only edited differences to ``index.jsonl``.
    The saved manual board is stronger evidence: every unchanged two-cell piece
    was also visible and accepted by the user.  Load those confirmations lazily
    from durable bundles so existing samples immediately become useful without
    a destructive migration or re-save.
    """
    global _MANUAL_BUNDLE_CACHE_KEY, _MANUAL_BUNDLE_CACHE_ITEMS
    root = MANUAL_LEARNING_DIR.parent / "manual_samples"
    if not root.exists():
        return []
    folders = []
    try:
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            required = [folder / "metadata.json", folder / "manual_board.json",
                        folder / "corrections.json", folder / "rectified.png"]
            if not all(path.exists() for path in required):
                continue
            folders.append((folder, tuple(path.stat().st_mtime_ns for path in required)))
    except OSError:
        return []
    cache_key = (str(root.resolve()), tuple((folder.name, stamps)
                                            for folder, stamps in folders))
    if cache_key == _MANUAL_BUNDLE_CACHE_KEY:
        return list(_MANUAL_BUNDLE_CACHE_ITEMS)

    items = []
    seen_visuals = set()
    for folder, _stamps in reversed(folders):
        try:
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            board = json.loads((folder / "manual_board.json").read_text(encoding="utf-8"))
            corrections = json.loads((folder / "corrections.json").read_text(encoding="utf-8"))
            rect = cv2.imread(str(folder / "rectified.png"), cv2.IMREAD_COLOR)
        except (OSError, TypeError, ValueError):
            continue
        if (not isinstance(metadata, dict) or not isinstance(board, dict)
                or not isinstance(corrections, list) or rect is None or not rect.size):
            continue
        observation_hash = str(metadata.get("observation_hash") or "")
        if not observation_hash:
            continue
        corrected_placements = {
            cell_key(item.get("after") or item.get("before") or {})
            for item in corrections if isinstance(item, dict)
        }
        for piece_id, piece in (board.get("pieces") or {}).items():
            placement = cell_key(piece)
            if len(placement) != 2 or placement in corrected_placements:
                continue
            feature = pair_visual_feature(rect, piece)
            if feature is None:
                continue
            visual_key = (observation_hash, feature.get("patch_hash"))
            if visual_key in seen_visuals:
                continue
            seen_visuals.add(visual_key)
            sample_id = f"{folder.name}-confirm-{piece_id}"
            correction = {
                "kind": "confirm", "fields": ["presence", "species", "facing"],
                "before_id": str(piece_id), "after_id": str(piece_id),
                "before": deepcopy(piece), "after": deepcopy(piece),
            }
            record = {
                "schema": MANUAL_LEARNING_SCHEMA,
                "sample_id": sample_id,
                "created_at": metadata.get("created_at"),
                "status": "active", "source": "saved-manual-sample",
                "observation_hash": observation_hash,
                "grid_hash": metadata.get("grid_hash"),
                "recognition_version": "manual-supervision-v2",
                "taxonomy_version": 1,
                "sample_path": str(folder.relative_to(MANUAL_LEARNING_DIR.parent.parent))
                    .replace("\\", "/"),
                "correction": correction, "feature": feature,
                "evidence": {"patch_hash": feature.get("patch_hash")},
            }
            identity = json.dumps({
                "observation_hash": observation_hash,
                "correction": correction,
                "feature_schema": feature.get("schema"),
            }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            record["record_id"] = hashlib.sha1(identity.encode("utf-8")).hexdigest()
            if _valid_manual_learning_record(record):
                items.append(record)
        if len(items) >= MAX_ACTIVE_CONFIRMATIONS:
            break
    # The general loader keeps the newest tail.  Restore chronological order
    # after walking bundles newest-first so the current saved boards win.
    items.reverse()
    _MANUAL_BUNDLE_CACHE_KEY = cache_key
    _MANUAL_BUNDLE_CACHE_ITEMS = items
    return list(items)


def load_manual_learning(limit=1000) -> list[dict]:
    items = []
    if MANUAL_LEARNING_INDEX.exists():
        try:
            with _MANUAL_LEARNING_LOCK:
                with open(MANUAL_LEARNING_INDEX, "r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            item = json.loads(line)
                        except (TypeError, ValueError):
                            continue
                        if _valid_manual_learning_record(item):
                            items.append(item)
        except OSError:
            pass
    items.extend(_load_saved_sample_confirmations())
    # A later correction of the same visual observation supersedes the older
    # label instead of creating an unresolvable self-conflict.  Different
    # screenshots remain independent support votes.
    collapsed = {}
    for item in items:
        feature = item.get("feature") or {}
        correction = item.get("correction") or {}
        target = correction.get("after") or correction.get("before") or {}
        visual_key = feature.get("patch_hash") or json.dumps(
            {"cells": target.get("cells"), "schema": feature.get("schema")},
            sort_keys=True, separators=(",", ":"))
        key = (str(item.get("observation_hash") or item.get("sample_id") or
                   item.get("record_id")), str(visual_key))
        collapsed[key] = item
    try:
        safe_limit = max(1, int(limit))
    except (TypeError, ValueError, OverflowError):
        safe_limit = 1000
    active = list(collapsed.values())
    edited = [item for item in active
              if (item.get("correction") or {}).get("kind") != "confirm"]
    confirmations = [item for item in active
                     if (item.get("correction") or {}).get("kind") == "confirm"]
    # Explicit edits are sparse and must never be pushed out of the active
    # window by hundreds of whole-board confirmations.  Keep only the newest
    # confirmation reservoir; it supplies broad positive examples without
    # turning each recognition pass into an unbounded historical scan.
    edited = edited[-safe_limit:]
    remaining = max(0, safe_limit - len(edited))
    confirmation_limit = min(MAX_ACTIVE_CONFIRMATIONS, remaining)
    return edited + confirmations[-confirmation_limit:]


def record_manual_learning(records: list[dict]) -> dict:
    """Validate then atomically append a batch of manual supervision records."""
    records = [deepcopy(item) for item in (records or [])]
    if not records:
        return {"recorded": 0, "duplicates": 0, "quarantined": 0,
                "index": str(MANUAL_LEARNING_INDEX)}
    prepared, quarantined = [], 0
    for item in records:
        item.setdefault("schema", MANUAL_LEARNING_SCHEMA)
        item.setdefault("status", "active")
        item.setdefault("recognition_version", "manual-supervision-v2")
        item.setdefault("taxonomy_version", 1)
        if not _valid_manual_learning_record(item):
            quarantined += 1
            continue
        identity = {
            "observation_hash": item.get("observation_hash"),
            "correction": item.get("correction"),
            "feature_schema": (item.get("feature") or {}).get("schema"),
        }
        try:
            identity_json = json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":"), allow_nan=False)
            item["record_id"] = item.get("record_id") or hashlib.sha1(
                identity_json.encode("utf-8")).hexdigest()
            payload = json.dumps(item, ensure_ascii=False, separators=(",", ":"),
                                 allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            quarantined += 1
            continue
        prepared.append((item, payload))

    written, duplicates = 0, 0
    with _MANUAL_LEARNING_LOCK:
        with _manual_learning_file_lock():
            existing = {str(item.get("record_id")) for item in load_manual_learning(limit=100000)
                        if item.get("record_id")}
            pending = []
            for item, payload in prepared:
                record_id = str(item["record_id"])
                if record_id in existing:
                    duplicates += 1
                    continue
                pending.append(payload)
                existing.add(record_id)
            if pending:
                with open(MANUAL_LEARNING_INDEX, "a", encoding="utf-8") as stream:
                    stream.write("\n".join(pending) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                written = len(pending)
    return {"recorded": written, "duplicates": duplicates,
            "quarantined": quarantined, "index": str(MANUAL_LEARNING_INDEX)}


def manual_candidate_proposals(rect: np.ndarray, rows: int, cols: int,
                               raw_candidates: list[dict], samples=None,
                               *, max_presence_distance=0.115,
                               max_endpoint_distance=0.10,
                               min_margin=0.012) -> tuple[list[dict], list[dict]]:
    """Propose a missing/relabelled two-cell candidate from persisted visual supervision.

    A template never invents a piece from an empty board: a current-frame candidate
    of the same species must overlap at least one proposed cell.  One independent
    human observation remains review-only; two matching observations may auto-apply.
    """
    samples = list(load_manual_learning() if samples is None else samples)
    usable = []
    for sample in samples:
        correction = sample.get("correction") or {}
        target = correction.get("after") or {}
        feature = sample.get("feature") or {}
        fields = set(correction.get("fields") or [])
        if (correction.get("kind") not in {"add", "confirm"} or "presence" not in fields
                or len(cell_key(target)) != 2
                or feature.get("schema") != PAIR_FEATURE_SCHEMA
                or target.get("species") not in MANUAL_PRESENCE_SPECIES):
            continue
        usable.append((sample, correction, target, feature))
    if not usable:
        return [], []

    feature_cache = {}
    native_exact = {(cell_key(item), str(item.get("species") or "sheep"))
                    for item in raw_candidates
                    if item.get("detector") != "learned-template"}
    matches = defaultdict(list)
    diagnostics = []
    for sample, correction, target, learned in usable:
        species = str(target.get("species") or "sheep")
        facing = str(target.get("facing") or "")
        axis = learned.get("axis")
        if facing not in ({"L", "R"} if axis == "H" else {"U", "D"}):
            continue
        supporting = [item for item in raw_candidates
                      if str(item.get("species") or "sheep") == species]
        if not supporting:
            continue
        placements = []
        for row in range(int(rows)):
            for col in range(int(cols)):
                other = ((row, col + 1) if axis == "H" else (row + 1, col))
                if other[0] >= rows or other[1] >= cols:
                    continue
                placement = tuple(sorted(((row, col), other)))
                if (placement, species) in native_exact:
                    continue
                if not any(set(placement) & set(cell_key(item)) for item in supporting):
                    continue
                current = feature_cache.get(placement)
                if current is None:
                    current = pair_visual_feature(rect, placement)
                    feature_cache[placement] = current
                presence = _feature_distance(learned, current or {}, "symmetric")
                endpoint = _feature_distance(learned, current or {}, "endpoint")
                placements.append((presence, endpoint, placement))
        placements.sort(key=lambda item: (item[0], item[1], item[2]))
        if not placements:
            continue
        best = placements[0]
        margin = (placements[1][0] - best[0]) if len(placements) > 1 else math.inf
        best_feature = feature_cache.get(best[2]) or {}
        exact = bool(learned.get("patch_hash")
                     and learned.get("patch_hash") == best_feature.get("patch_hash"))
        accepted = (best[0] <= max_presence_distance
                    and best[1] <= max_endpoint_distance
                    and (margin >= min_margin or best[0] <= 0.025))
        diagnostics.append({
            "sample_id": sample.get("sample_id"),
            "record_id": sample.get("record_id"),
            "species": species, "facing": facing,
            "cells": [list(cell) for cell in best[2]],
            "presence_distance": round(float(best[0]), 5),
            "endpoint_distance": round(float(best[1]), 5),
            "margin": None if not math.isfinite(margin) else round(float(margin), 5),
            "exact_patch": exact,
            "accepted": bool(accepted),
        })
        if accepted:
            key = (best[2], species, facing)
            matches[key].append((sample, best[0], best[1], exact))

    proposals = []
    label_groups = defaultdict(list)
    for (placement, species, facing), evidence in matches.items():
        label_groups[(placement, species)].append((facing, evidence))
    for index, ((placement, species), labels) in enumerate(sorted(label_groups.items())):
        by_label = {}
        for facing, evidence in labels:
            observations = {}
            for sample, presence, endpoint, exact in evidence:
                observation = str(sample.get("observation_hash") or sample.get("sample_id") or
                                  sample.get("record_id"))
                previous = observations.get(observation)
                if previous is None or (exact and not previous[3]):
                    observations[observation] = (sample, presence, endpoint, exact)
            by_label[facing] = observations
        total_support = sum(len(observations) for observations in by_label.values())
        facing, observations = max(by_label.items(), key=lambda item: len(item[1]))
        support = len(observations)
        if total_support and support / total_support < (2.0 / 3.0):
            diagnostics.append({
                "species": species, "cells": [list(cell) for cell in placement],
                "accepted": False, "reason": "learning_conflict",
                "votes": {label: len(items) for label, items in by_label.items()},
            })
            continue
        exact_support = any(value[3] for value in observations.values())
        explicit_add = any(
            (value[0].get("correction") or {}).get("kind") == "add"
            for value in observations.values()
        )
        if support < 2 and not exact_support and not explicit_add:
            diagnostics.append({
                "species": species, "cells": [list(cell) for cell in placement],
                "accepted": False, "reason": "confirmation_needs_second_observation",
                "support": support,
            })
            continue
        dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[facing]
        head = max(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
        rump = min(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
        overlapping = [item for item in raw_candidates
                       if str(item.get("species") or "sheep") == species
                       and set(placement) & set(cell_key(item))]
        base_score = max([float(item.get("pair_score") or 0.0) for item in overlapping] + [900.0])
        sample_ids = [value[0].get("sample_id") for value in observations.values()]
        proposal = {
            "source_id": f"learned:{index}:" + "+".join(str(value) for value in sample_ids),
            "detector": "learned-template", "species": species,
            "cells": [list(rump), list(head)], "rump": list(rump), "head": list(head),
            "axis": "H" if rump[0] == head[0] else "V", "facing": facing,
            "quality": round(11000.0 + base_score, 2),
            "pair_score": round(max(900.0, base_score * 0.96), 2),
            "direction_confidence": 120.0,
            "direction_votes": {"manual_learning": sample_ids},
            "head_scores": {}, "metrics": {},
            "learned_template": True,
            "learned_provisional": support < 2,
            "learned_support": support,
            "learned_sample_ids": sample_ids,
            "review": support < 2,
            "review_reason": "manual_learning_single_observation" if support < 2 else None,
        }
        proposals.append(proposal)
    return proposals, diagnostics


def manual_candidate_rejections(rect: np.ndarray, raw_candidates: list[dict], samples=None,
                                *, max_presence_distance=0.085,
                                max_endpoint_distance=0.075,
                                min_margin=0.012):
    """Remove a false footprint learned from a saved manual board.

    The editor stores deletions as visual negative examples.  Match every saved
    example to at most one current footprint, then reject all detector variants
    for that footprint.  A single independent observation is still reported as
    provisional so the safety layer can block automatic execution.
    """
    samples = list(load_manual_learning() if samples is None else samples)
    usable = []
    for sample in samples:
        correction = sample.get("correction") or {}
        target = correction.get("before") or {}
        feature = sample.get("feature") or {}
        if (correction.get("kind") != "delete"
                or "presence" not in set(correction.get("fields") or [])
                or len(cell_key(target)) != 2
                or feature.get("schema") != PAIR_FEATURE_SCHEMA):
            continue
        usable.append((sample, target, feature))
    if not usable or not raw_candidates:
        return list(raw_candidates), [], []

    placements = {}
    for candidate in raw_candidates:
        placement = cell_key(candidate)
        if len(placement) == 2:
            placements.setdefault(placement, []).append(candidate)
    feature_cache = {placement: pair_visual_feature(rect, placement)
                     for placement in placements}
    matches = defaultdict(list)
    diagnostics = []
    for sample, target, learned in usable:
        target_species = str(target.get("species") or "sheep")
        ranked = []
        for placement, candidates in placements.items():
            if not any(str(item.get("species") or "sheep") == target_species
                       for item in candidates):
                continue
            current = feature_cache.get(placement) or {}
            presence = _feature_distance(learned, current, "symmetric")
            endpoint = _feature_distance(learned, current, "endpoint")
            exact = bool(learned.get("patch_hash")
                         and learned.get("patch_hash") == current.get("patch_hash"))
            ranked.append((not exact, presence + endpoint * 0.7, presence,
                           endpoint, placement, exact))
        ranked.sort(key=lambda item: item[:5])
        if not ranked:
            continue
        best = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else math.inf
        margin = runner_score - best[1]
        accepted = bool(best[5] or (
            best[2] <= max_presence_distance
            and best[3] <= max_endpoint_distance
            and margin >= min_margin
        ))
        diagnostics.append({
            "sample_id": sample.get("sample_id"),
            "record_id": sample.get("record_id"),
            "kind": "delete", "species": target_species,
            "cells": [list(cell) for cell in best[4]],
            "presence_distance": round(float(best[2]), 5),
            "endpoint_distance": round(float(best[3]), 5),
            "margin": None if not math.isfinite(margin) else round(float(margin), 5),
            "exact_patch": bool(best[5]), "accepted": accepted,
        })
        if accepted:
            matches[best[4]].append((sample, bool(best[5])))

    rejected_placements = {}
    for placement, evidence in matches.items():
        observations = {}
        for sample, exact in evidence:
            observation = str(sample.get("observation_hash") or sample.get("sample_id")
                              or sample.get("record_id"))
            previous = observations.get(observation)
            if previous is None or (exact and not previous[1]):
                observations[observation] = (sample, exact)
        support = len(observations)
        # A single saved image may suppress only the identical patch.  Visual
        # generalization of a negative example requires two independent human
        # observations because a false rejection removes a piece entirely.
        if support < 2 and not any(exact for _sample, exact in observations.values()):
            continue
        rejected_placements[placement] = {
            "support": support,
            "sample_ids": [sample.get("sample_id")
                           for sample, _exact in observations.values()],
            "provisional": support < 2,
        }

    kept, rejected = [], []
    for candidate in raw_candidates:
        learning = rejected_placements.get(cell_key(candidate))
        if learning is None:
            kept.append(candidate)
            continue
        rejected.append({
            **candidate,
            "drop_reason": "manual_learning_rejected_footprint",
            "learned_rejection": True,
            "learned_provisional": learning["provisional"],
            "learned_support": learning["support"],
            "learned_sample_ids": learning["sample_ids"],
        })
    return kept, rejected, diagnostics


def apply_manual_label_learning(pieces: list[dict], rect: np.ndarray, samples=None,
                                *, max_presence_distance=0.07,
                                max_endpoint_distance=0.065,
                                min_margin=0.012):
    """Apply saved labels with a one-sample/one-footprint assignment.

    Earlier code independently matched every board piece, allowing one generic
    sheep example to flip many visually similar sheep in the same frame.  A
    saved observation now selects only its best unambiguous footprint; multiple
    independent observations can still vote for the same label.
    """
    samples = list(load_manual_learning() if samples is None else samples)
    result, applied = deepcopy(pieces), []
    features = [(index, pair_visual_feature(rect, piece))
                for index, piece in enumerate(result)]
    assignments = defaultdict(list)
    for sample in samples:
        correction = sample.get("correction") or {}
        target = correction.get("after") or {}
        learned = sample.get("feature") or {}
        fields = set(correction.get("fields") or [])
        target_species = str(target.get("species") or "sheep")
        apply_species = bool(fields & {"presence", "species"})
        apply_facing = bool(fields & {"presence", "facing"})
        if (correction.get("kind") not in {"add", "update", "confirm"}
                or len(cell_key(target)) != 2
                or learned.get("schema") != PAIR_FEATURE_SCHEMA
                or (apply_species and target_species not in MANUAL_LABEL_SPECIES)
                or not (apply_species or apply_facing)):
            continue

        ranked = []
        before_species = str((correction.get("before") or {}).get("species") or "")
        for piece_index, current in features:
            if not current or learned.get("axis") != current.get("axis"):
                continue
            piece = result[piece_index]
            current_species = str(piece.get("species") or "sheep")
            strong_detectors = {str(value) for value in (piece.get("detectors") or [])}
            strong_species_anchor = (
                current_species in {"rocket", "bomb", "pink_sheep", "pig", "goat",
                                    "black_sheep", "elephant", "cattle"}
                or bool(strong_detectors & {"arrow", "rocket", "pink-bow", "pig-body",
                                            "goat-body", "cattle-body"})
                or "bomb_counter" in (piece.get("direction_votes") or {})
            )
            exact = bool(learned.get("patch_hash")
                         and learned.get("patch_hash") == current.get("patch_hash"))
            explicit_anchor_correction = bool(
                correction.get("kind") == "update"
                and before_species == current_species
                and exact
            )
            if (strong_species_anchor and apply_species
                    and target_species != current_species
                    and not explicit_anchor_correction):
                continue
            presence = _feature_distance(learned, current, "symmetric")
            endpoint = _feature_distance(learned, current, "endpoint")
            if presence > max_presence_distance or endpoint > max_endpoint_distance:
                continue
            effective_target = deepcopy(target)
            effective_target["species"] = target_species if apply_species else current_species
            effective_target["facing"] = (str(target.get("facing") or "")
                                           if apply_facing else str(piece.get("facing") or ""))
            ranked.append((not exact, presence + endpoint * 0.7, piece_index,
                           sample, effective_target, presence, endpoint, exact))
        ranked.sort(key=lambda item: item[:3])
        if not ranked:
            continue
        best = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else math.inf
        margin = runner_score - best[1]
        # An addition with no exact current footprint belongs in the presence
        # proposal path, not on an arbitrary similar sheep elsewhere.
        if correction.get("kind") == "add" and not best[7]:
            continue
        if not best[7] and margin < min_margin:
            continue
        assignments[best[2]].append(
            (best[1], best[3], best[4], best[5], best[6], best[7]))

    for piece_index, matches in assignments.items():
        piece = result[piece_index]
        current = features[piece_index][1]
        if any(item[5] for item in matches):
            matches = [item for item in matches if item[5]]
        unique_observations = {}
        for item in matches:
            sample = item[1]
            key = str(sample.get("observation_hash") or sample.get("sample_id") or
                      sample.get("record_id"))
            if key not in unique_observations or item[0] < unique_observations[key][0]:
                unique_observations[key] = item
        neighborhood = list(unique_observations.values())
        labels = Counter((str(target.get("species") or "sheep"),
                          str(target.get("facing") or ""))
                         for _score, _sample, target, _presence, _endpoint, _exact
                         in neighborhood)
        (species, facing), support = labels.most_common(1)[0]
        if support / len(neighborhood) < (2.0 / 3.0):
            continue
        agreeing = [item for item in neighborhood
                    if (str(item[2].get("species") or "sheep"),
                        str(item[2].get("facing") or "")) == (species, facing)]
        confirmed_samples = {
            str(reference)
            for item in agreeing
            for reference in ((item[1].get("correction") or {}).get("confirms_samples") or [])
            if reference
        }
        effective_support = support + len(confirmed_samples)
        if (support < 2
                and all((item[1].get("correction") or {}).get("kind") == "confirm"
                        for item in agreeing)
                and not any(item[5] for item in agreeing)):
            continue
        _score, sample, _target, presence, endpoint, _exact = agreeing[0]
        axis = current["axis"]
        if facing not in ({"L", "R"} if axis == "H" else {"U", "D"}):
            continue
        original_species, original_facing = piece.get("species", "sheep"), piece.get("facing")
        # A one-observation species relabel remains review-only.  Facing on an
        # already present footprint is equivalent to the existing direction
        # panel correction and may apply immediately.
        if species != original_species:
            piece["species"] = species
            if effective_support < 2:
                piece["review"] = True
                piece["review_reason"] = "manual_learning_single_observation"
                piece["learned_provisional"] = True
        placement = cell_key(piece)
        dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[facing]
        head = max(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
        rump = min(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
        piece["facing"] = facing
        piece["cells"], piece["rump"], piece["head"] = [list(rump), list(head)], list(rump), list(head)
        piece["manual_learning"] = True
        piece["manual_learning_sample_id"] = sample.get("sample_id")
        piece["manual_learning_support"] = effective_support
        if effective_support < 2 and facing != original_facing:
            piece["review"] = True
            piece["review_reason"] = "manual_direction_single_observation"
            piece["learned_direction_provisional"] = True
        elif (effective_support >= 2
              and piece.get("review_reason") in {
                  "manual_learning_single_observation",
                  "manual_direction_single_observation",
              }):
            piece.pop("review", None)
            piece.pop("review_reason", None)
            piece.pop("learned_provisional", None)
            piece.pop("learned_direction_provisional", None)
        if species != original_species or facing != original_facing:
            applied.append({
                "id": piece.get("id"), "sample_id": sample.get("sample_id"),
                "species_from": original_species, "species_to": species,
                "facing_from": original_facing, "facing_to": facing,
                "presence_distance": round(float(presence), 5),
                "endpoint_distance": round(float(endpoint), 5),
                "support": int(effective_support),
            })
    return result, applied


def direction_feature(piece: dict) -> dict | None:
    """Build a position-independent endpoint feature for manual direction learning."""
    placement = cell_key(piece)
    if len(placement) != 2:
        return None
    axis = "H" if placement[0][0] == placement[1][0] else "V"
    ordered = sorted(placement, key=lambda rc: rc[1] if axis == "H" else rc[0])
    metrics = piece.get("metrics") or {}

    def endpoint(cell):
        item = metrics.get(str(tuple(cell))) or metrics.get(str(list(cell))) or {}
        return [float(item.get(name, 0.0) or 0.0)
                for name in ("white", "face", "dt_mean", "hist", "body_support")]

    low, high = endpoint(ordered[0]), endpoint(ordered[1])
    scales = (2000.0, 800.0, 18.0, 2200.0, 3500.0)
    vector = [round((high[i] - low[i]) / max(scales[i], abs(high[i]) + abs(low[i]), 1.0), 6)
              for i in range(len(scales))]
    centroid_offset = float((piece.get("direction_votes") or {}).get("centroid_offset", 0.0) or 0.0)
    vector.append(round(centroid_offset / 64.0, 6))
    detectors = sorted(str(item) for item in (piece.get("detectors") or []))
    if not detectors:
        detectors = [_detector(piece)]
    return {
        "axis": axis,
        "species": str(piece.get("species") or "sheep"),
        "detectors": detectors,
        "vector": vector,
    }


def record_direction_correction(piece: dict, corrected_facing: str, *, source: str,
                                sample_id: str | None = None, artifact: str | None = None) -> dict | None:
    feature = direction_feature(piece)
    original = str(piece.get("facing") or "")
    corrected = str(corrected_facing or "").upper()
    allowed = {"H": {"L", "R"}, "V": {"U", "D"}}
    vector = (feature or {}).get("vector") or []
    if (not feature or not vector or sum(abs(float(value)) for value in vector) <= 1e-6
            or corrected not in allowed[feature["axis"]] or corrected == original):
        return None
    sample_id = sample_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"
    data = {
        "schema": 1,
        "sample_id": sample_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(source),
        "original_facing": original,
        "corrected_facing": corrected,
        "cells": [list(cell) for cell in cell_key(piece)],
        "feature": feature,
        "direction_votes": deepcopy(piece.get("direction_votes") or {}),
        "head_scores": deepcopy(piece.get("head_scores") or {}),
        "metrics": deepcopy(piece.get("metrics") or {}),
        "artifact": artifact,
    }
    DIRECTION_LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    sample_dir = DIRECTION_LEARNING_DIR / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "sample.json", "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    with open(DIRECTION_LEARNING_INDEX, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    return data


def load_direction_corrections(limit=500) -> list[dict]:
    items = []
    if DIRECTION_LEARNING_INDEX.exists():
        try:
            with open(DIRECTION_LEARNING_INDEX, "r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        items.append(json.loads(line))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            pass
    # Board-editor supervision and the older direction panel used to publish
    # into separate indexes.  A correction saved from the board therefore did
    # not join an otherwise identical direction-panel vote, leaving a genuine
    # two-observation consensus marked as provisional forever.  Rebuild the
    # direction feature from the raw candidate evidence stored with each
    # manual board correction and let the existing consensus rules combine the
    # two sources.  Duplicate/replayed board captures are still deduplicated by
    # observation hash when ``apply_direction_learning`` builds its consensus.
    items.extend(manual_direction_corrections())
    return items[-max(1, int(limit)):]


def manual_direction_corrections(samples=None) -> list[dict]:
    """Derive direction-learning votes from board-editor correction evidence.

    Only an actual facing change with a matching raw two-cell candidate is
    eligible.  The saved colour-pair feature is not substituted here: direction
    learning deliberately uses detector endpoint metrics, so bomb backpacks,
    feet and faces remain comparable with older direction-panel samples.
    """
    samples = list(load_manual_learning() if samples is None else samples)
    derived = []
    seen_records = set()
    allowed = {"H": {"L", "R"}, "V": {"U", "D"}}
    for sample in reversed(samples):
        correction = sample.get("correction") or {}
        fields = set(correction.get("fields") or [])
        before, after = correction.get("before") or {}, correction.get("after") or {}
        original = str(before.get("facing") or "").upper()
        corrected = str(after.get("facing") or "").upper()
        target_species = str(after.get("species") or before.get("species") or "sheep")
        placement = cell_key(after or before)
        if (correction.get("kind") != "update" or "facing" not in fields
                or target_species != "bomb"
                or len(placement) != 2 or original == corrected):
            continue
        axis = "H" if placement[0][0] == placement[1][0] else "V"
        if original not in allowed[axis] or corrected not in allowed[axis]:
            continue
        observation = str(sample.get("observation_hash") or sample.get("sample_id") or
                          sample.get("record_id") or "")
        record_identity = (observation, placement, original, corrected)
        if not observation or record_identity in seen_records:
            continue

        candidates = []
        evidence = sample.get("evidence") or {}
        for candidate in evidence.get("overlapping_candidates") or []:
            if cell_key(candidate) != placement or str(candidate.get("facing") or "") != original:
                continue
            feature = direction_feature(candidate)
            vector = (feature or {}).get("vector") or []
            if not feature or not vector or sum(abs(float(value)) for value in vector) <= 1e-6:
                continue
            detectors = set(candidate.get("detectors") or [])
            detector = str(candidate.get("detector") or "")
            rank = (
                detector == "fusion",
                len(detectors),
                float(candidate.get("direction_confidence") or 0.0),
            )
            candidates.append((rank, feature, candidate))
        if not candidates:
            continue
        _rank, feature, candidate = max(candidates, key=lambda item: item[0])
        seen_records.add(record_identity)
        derived.append({
            "schema": 1,
            "sample_id": f"manual:{sample.get('sample_id') or sample.get('record_id')}",
            "observation_hash": observation,
            "created_at": sample.get("created_at"),
            "source": "manual-board-evidence",
            "original_facing": original,
            "corrected_facing": corrected,
            "cells": [list(cell) for cell in placement],
            "feature": feature,
            "direction_votes": deepcopy(candidate.get("direction_votes") or {}),
            "head_scores": deepcopy(candidate.get("head_scores") or {}),
            "metrics": deepcopy(candidate.get("metrics") or {}),
        })
    derived.reverse()
    return derived


def apply_direction_learning(pieces: list[dict], samples=None, *, max_distance=0.16):
    """Apply a non-degenerate consensus backed by independent corrections.

    A single saved correction is only a candidate for human review.  Reusing
    that candidate in several captured frames must not turn it into several
    independent votes and gradually poison later boards.
    """
    samples = list(load_direction_corrections() if samples is None else samples)
    applied = []
    for piece in pieces:
        feature = direction_feature(piece)
        vector = (feature or {}).get("vector") or []
        if not feature or not vector or sum(abs(float(value)) for value in vector) <= 1e-6:
            continue
        candidates = []
        for sample in samples:
            learned = sample.get("feature") or {}
            if (learned.get("axis") != feature["axis"]
                    or learned.get("species") != feature["species"]):
                continue
            a, b = learned.get("vector") or [], feature["vector"]
            if (len(a) != len(b) or not a
                    or sum(abs(float(value)) for value in a) <= 1e-6):
                continue
            detector_overlap = set(learned.get("detectors") or []) & set(feature.get("detectors") or [])
            if not detector_overlap:
                continue
            distance = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) / len(b))
            candidates.append((distance, sample))
        if not candidates:
            continue
        candidates = [item for item in candidates if item[0] <= float(max_distance)]
        if not candidates:
            continue
        # An index can contain the same correction more than once (for example
        # after retrying a save).  Count it once so support means independent
        # manual observations, not duplicate lines in the learning file.
        unique = {}
        for distance, sample in sorted(candidates, key=lambda item: item[0]):
            identity = str(sample.get("observation_hash") or sample.get("sample_id") or json.dumps({
                "corrected_facing": sample.get("corrected_facing"),
                "feature": sample.get("feature"),
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            unique.setdefault(identity, (distance, sample))
        candidates = list(unique.values())
        best_distance = min(item[0] for item in candidates)
        # An exact board-editor correction describes the current visual patch
        # itself.  Let it join older direction-panel examples across the normal
        # detector distance budget; restricting it to ``best + .025`` would
        # isolate the exact sample at distance zero and make two independent
        # observations impossible to combine.  Without a current manual anchor
        # retain the tighter neighborhood used for unattended transfer.
        has_manual_anchor = any(
            str(sample.get("source") or "") == "manual-board-evidence"
            for _distance, sample in candidates
        )
        neighborhood_limit = float(max_distance) if has_manual_anchor else best_distance + 0.025
        neighborhood = [item for item in candidates if item[0] <= neighborhood_limit]
        votes = Counter(str(sample.get("corrected_facing") or "")
                        for _distance, sample in neighborhood)
        corrected, count = votes.most_common(1)[0]
        # Similar visual evidence with contradictory labels is quarantined.  A
        # stale or accidental edit must not silently poison every later level.
        if count < 2 or count / len(neighborhood) < (2.0 / 3.0):
            continue
        agreeing = [(distance, sample) for distance, sample in neighborhood
                    if str(sample.get("corrected_facing") or "") == corrected]
        distance, sample = min(agreeing, key=lambda item: item[0])
        if corrected == piece.get("facing"):
            continue
        dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[corrected]
        placement = cell_key(piece)
        head = max(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
        rump = min(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
        original = piece.get("facing")
        piece["facing"] = corrected
        piece["cells"], piece["rump"], piece["head"] = [list(rump), list(head)], list(rump), list(head)
        piece["learned_direction"] = True
        piece["learned_sample_id"] = sample.get("sample_id")
        applied.append({"id": piece.get("id"), "from": original, "to": corrected,
                        "distance": round(float(distance), 4),
                        "sample_id": sample.get("sample_id"),
                        "support": int(count), "candidates": len(neighborhood)})
    return pieces, applied


def _detector(candidate: dict) -> str:
    explicit = candidate.get("detector")
    if explicit:
        return str(explicit)
    votes = candidate.get("direction_votes") or {}
    if "arrow" in votes:
        return "arrow"
    if "cattle_body" in votes:
        return "cattle-body"
    if "cattle_face" in votes:
        return "cattle-face"
    if "cattle_cell_stats" in votes:
        return "cattle-cell"
    return "body"


def _candidate_confidence(candidate: dict) -> dict:
    detector = _detector(candidate)
    reliability = DETECTOR_RELIABILITY.get(detector, DETECTOR_RELIABILITY["unknown"])
    pair_score = max(0.0, float(candidate.get("pair_score") or 0.0))
    direction_score = max(0.0, float(candidate.get("direction_confidence") or 0.0))
    occupancy_scale = 1600.0 if detector.startswith("cattle") else 1200.0
    direction_scale = 150.0 if detector.startswith("cattle") else 90.0
    occupancy = _clip((1.0 - math.exp(-pair_score / occupancy_scale)) * reliability)
    facing = _clip((1.0 - math.exp(-direction_score / direction_scale)) * reliability)
    # Axis is independently supported by the pair geometry and detector shape.
    axis = _clip(0.55 + 0.42 * reliability)
    species = _clip(
        0.98 if detector in {"rocket", "pink-bow", "pig-body", "goat-body"}
        else 0.88 if detector == "learned-template"
        else 0.96 if detector.startswith("cattle")
        else 0.92
    )
    score = 38.0 * occupancy + 22.0 * axis + 20.0 * facing + 12.0 * species + 8.0 * reliability
    return {
        "occupancy": round(occupancy, 4),
        "axis": round(axis, 4),
        "facing": round(facing, 4),
        "species": round(species, 4),
        "detector": round(reliability, 4),
        "selection_score": round(score, 4),
    }


def fuse_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fuse same-placement candidates while retaining every source score."""
    groups = defaultdict(list)
    for raw in candidates:
        candidate = deepcopy(raw)
        candidate["detector"] = _detector(candidate)
        candidate["confidence"] = _candidate_confidence(candidate)
        key = cell_key(candidate)
        if len(key) != 2:
            candidate["drop_reason"] = "invalid_cell_count"
            groups[(key, "__invalid__")].append(candidate)
            continue
        groups[(key, str(candidate.get("species") or "sheep"))].append(candidate)

    fused, rejected = [], []
    for (key, species), sources in groups.items():
        if species == "__invalid__":
            rejected.extend(sources)
            continue
        facing_votes = defaultdict(float)
        axis_votes = defaultdict(float)
        detector_names = set()
        for source in sources:
            conf = source["confidence"]
            vote_multiplier = 3.2 if source["detector"] == "gesture-target-arrow" else (
                3.0 if source["detector"] == "arrow" else (
                2.8 if source["detector"] in {"rocket", "pink-bow", "pig-body", "goat-body"} else (
                2.35 if source["detector"] == "learned-template" else (
                2.0 if source["detector"] == "cattle-body" else 1.0))))
            weight = (0.35 + conf["occupancy"] + conf["facing"]) * vote_multiplier
            facing_votes[str(source.get("facing"))] += weight
            axis_votes[str(source.get("axis"))] += 0.4 + conf["axis"]
            detector_names.add(source["detector"])
        facing = max(facing_votes, key=facing_votes.get)
        axis = max(axis_votes, key=axis_votes.get)
        agreeing = [source for source in sources if str(source.get("facing")) == facing]
        representative = max(agreeing or sources,
                             key=lambda item: item["confidence"]["selection_score"])

        occupancy = 1.0
        for source in sources:
            occupancy *= 1.0 - source["confidence"]["occupancy"]
        occupancy = 1.0 - occupancy
        vote_total = sum(facing_votes.values()) or 1.0
        facing_conf = facing_votes[facing] / vote_total
        axis_total = sum(axis_votes.values()) or 1.0
        axis_conf = axis_votes[axis] / axis_total
        diversity = min(1.0, len(detector_names) / 3.0)
        species_conf = max(source["confidence"]["species"] for source in sources)
        anchor_bonus = 105.0 if "gesture-target-arrow" in detector_names else (
            100.0 if "arrow" in detector_names else (
            95.0 if ({"rocket", "pink-bow", "pig-body", "goat-body"} & detector_names) else (
            72.0 if "learned-template" in detector_names else (
            55.0 if "cattle-body" in detector_names else
            30.0 if "cattle-face" in detector_names else 0.0))))
        selection_score = (48.0 * occupancy + 18.0 * axis_conf + 18.0 * facing_conf
                           + 10.0 * species_conf + 8.0 * diversity
                           + min(8.0, 2.0 * (len(sources) - 1)) + anchor_bonus)

        result = deepcopy(representative)
        result.update({
            "source_id": "fusion:" + "+".join(str(item.get("source_id")) for item in sources),
            "detector": "fusion",
            "detectors": sorted(detector_names),
            "species": species,
            "axis": axis,
            "facing": facing,
            "quality": round(selection_score * 100.0, 2),
            "selection_score": round(selection_score, 4),
            "confidence": {
                "occupancy": round(occupancy, 4),
                "axis": round(axis_conf, 4),
                "facing": round(facing_conf, 4),
                "species": round(species_conf, 4),
                "detector_diversity": round(diversity, 4),
            },
            "fusion": {
                "source_count": len(sources),
                "sources": [{
                    "source_id": item.get("source_id"),
                    "detector": item["detector"],
                    "facing": item.get("facing"),
                    "axis": item.get("axis"),
                    "confidence": item["confidence"],
                } for item in sources],
                "facing_votes": {key: round(value, 4) for key, value in facing_votes.items()},
                "axis_votes": {key: round(value, 4) for key, value in axis_votes.items()},
            },
        })
        # Recompute ordered rump/head from the fused direction.
        drdc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        dr, dc = drdc[facing]
        head = max(key, key=lambda rc: rc[0] * dr + rc[1] * dc)
        rump = min(key, key=lambda rc: rc[0] * dr + rc[1] * dc)
        result["cells"] = [list(rump), list(head)]
        result["rump"], result["head"] = list(rump), list(head)

        review_reasons = []
        if occupancy < (0.48 if detector_names == {"cattle-cell"} else 0.52):
            review_reasons.append("low_occupancy_confidence")
        if len(facing_votes) > 1 and facing_conf < 0.62:
            review_reasons.append("detector_facing_disagreement")
        if len(axis_votes) > 1 and axis_conf < 0.70:
            review_reasons.append("detector_axis_disagreement")
        if (detector_names == {"cattle-cell"}
                and float(representative.get("direction_confidence") or 0.0) < 180):
            review_reasons.append("weak_cattle_cell_only")
        if review_reasons:
            result["review"] = True
            result["review_reason"] = review_reasons[0]
            result["review_reasons"] = review_reasons
        fused.append(result)
    return fused, rejected


def global_assignment(candidates: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Solve weighted adjacent-cell set packing as a binary MILP."""
    if not candidates:
        return [], [], {"method": "milp", "status": "empty", "objective": 0.0}
    cells = sorted({cell for candidate in candidates for cell in cell_key(candidate)})
    cell_index = {cell: index for index, cell in enumerate(cells)}
    matrix = lil_matrix((len(cells), len(candidates)), dtype=float)
    for col, candidate in enumerate(candidates):
        for cell in cell_key(candidate):
            matrix[cell_index[cell], col] = 1.0
    scores = np.asarray([
        max(0.001, float(candidate.get("selection_score") or
                         (candidate.get("confidence") or {}).get("selection_score") or
                         candidate.get("quality") or 0.001))
        for candidate in candidates
    ], dtype=float)
    try:
        result = milp(
            c=-scores,
            integrality=np.ones(len(candidates), dtype=int),
            bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
            constraints=LinearConstraint(matrix.tocsr(),
                                         np.zeros(len(cells)), np.ones(len(cells))),
            options={"time_limit": 3.0, "mip_rel_gap": 0.0},
        )
        if not result.success or result.x is None:
            raise RuntimeError(str(result.message))
        selected_indices = {i for i, value in enumerate(result.x) if value >= 0.5}
        method, status = "milp", str(result.message)
    except Exception as exc:
        selected_indices, occupied = set(), set()
        for index in sorted(range(len(candidates)), key=lambda i: -scores[i]):
            placement = set(cell_key(candidates[index]))
            if not placement & occupied:
                selected_indices.add(index)
                occupied |= placement
        method, status = "greedy-fallback", str(exc)

    kept = [deepcopy(candidates[i]) for i in sorted(selected_indices)]
    chosen_cells = {cell: candidate for candidate in kept for cell in cell_key(candidate)}
    dropped = []
    for index, candidate in enumerate(candidates):
        if index in selected_indices:
            continue
        conflicts = sorted({str(chosen_cells[cell].get("source_id"))
                            for cell in cell_key(candidate) if cell in chosen_cells})
        item = deepcopy(candidate)
        item["drop_reason"] = "global_occupancy_conflict"
        item["conflicts_with"] = conflicts
        item["optimization_loss"] = round(float(candidate.get("selection_score") or 0.0), 4)
        dropped.append(item)
    kept.sort(key=lambda item: (min(cell[0] for cell in cell_key(item)),
                                min(cell[1] for cell in cell_key(item))))
    return kept, dropped, {
        "method": method,
        "status": status,
        "candidate_count": len(candidates),
        "selected_count": len(kept),
        "objective": round(sum(float(item.get("selection_score") or 0.0) for item in kept), 4),
    }


def _piece_observation(piece: dict) -> dict:
    return {
        "cells": [list(cell) for cell in cell_key(piece)],
        "axis": piece.get("axis"),
        "facing": piece.get("facing"),
        "species": piece.get("species", "sheep"),
        "confidence": deepcopy(piece.get("confidence") or {}),
        "hit_limit": piece.get("hit_limit"),
        "hits_remaining": piece.get("hits_remaining"),
        "awake": piece.get("awake"),
    }


def observation_record(pieces: list[dict], hazards: list[dict], rows: int, cols: int) -> dict:
    return {
        "rows": int(rows), "cols": int(cols),
        "pieces": [_piece_observation(piece) for piece in pieces],
        "hazards": [[int(item["row"]), int(item["col"])] if isinstance(item, dict)
                    else [int(item[0]), int(item[1])] for item in (hazards or [])],
    }


def apply_temporal(pieces: list[dict], hazards: list[dict], history: list[dict],
                   rows: int, cols: int, *, recover_missing_edges=False
                   ) -> tuple[list[dict], list[dict], dict]:
    """Stabilize current evidence using up to four compatible prior frames."""
    compatible = [frame for frame in (history or [])
                  if int(frame.get("rows", -1)) == int(rows)
                  and int(frame.get("cols", -1)) == int(cols)][-4:]
    result = deepcopy(pieces)
    previous_by_cells = defaultdict(list)
    for frame in compatible:
        for piece in frame.get("pieces") or []:
            previous_by_cells[cell_key(piece)].append(piece)

    corrections = []
    for piece in result:
        prior = previous_by_cells.get(cell_key(piece), [])
        samples = prior + [piece]
        facing_votes = Counter(str(item.get("facing")) for item in samples if item.get("facing"))
        species_votes = Counter(str(item.get("species") or "sheep") for item in samples)
        consensus = max(facing_votes.values(), default=1) / max(1, sum(facing_votes.values()))
        current_conf = piece.setdefault("confidence", {})
        current_conf["temporal_presence"] = round((len(prior) + 1) / (len(compatible) + 1), 4)
        current_conf["temporal_facing"] = round(consensus, 4)
        if len(prior) >= 2 and facing_votes:
            stable_facing, votes = facing_votes.most_common(1)[0]
            if stable_facing != piece.get("facing") and votes >= 3 and consensus >= 0.75:
                old = piece.get("facing")
                piece["facing"] = stable_facing
                dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[stable_facing]
                placement = cell_key(piece)
                head = max(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
                rump = min(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
                piece["cells"], piece["rump"], piece["head"] = [list(rump), list(head)], list(rump), list(head)
                corrections.append({"cells": [list(c) for c in placement], "field": "facing",
                                    "from": old, "to": stable_facing, "votes": dict(facing_votes)})
        if len(prior) >= 2 and species_votes:
            stable_species, votes = species_votes.most_common(1)[0]
            # Black sheep is a visual state, not a persistent identity.  A
            # stale frame can contain a dark hazard/pack false positive; do
            # not let that historical majority relabel a currently ordinary
            # sheep.  The current-frame black-pack / dark-cluster classifiers
            # run after temporal stabilization and remain the authority for a
            # genuine black sheep.
            current_black_evidence = (
                piece.get("species") == "black_sheep"
                or "black-pack" in set(piece.get("detectors") or [])
                or bool((piece.get("direction_votes") or {}).get(
                    "black_sheep_dark_cluster"))
            )
            if (stable_species != piece.get("species") and votes >= 3
                    and (stable_species != "black_sheep" or current_black_evidence)):
                corrections.append({"cells": [list(c) for c in cell_key(piece)], "field": "species",
                                    "from": piece.get("species"), "to": stable_species,
                                    "votes": dict(species_votes)})
                piece["species"] = stable_species

    restored = []
    if recover_missing_edges and compatible:
        occupied = {cell for piece in result for cell in cell_key(piece)}
        next_id = max([int(piece.get("id", -1)) for piece in result] + [-1]) + 1
        for prior_piece in compatible[-1].get("pieces") or []:
            placement = cell_key(prior_piece)
            if len(placement) != 2 or any(cell in occupied for cell in placement):
                continue
            if not any(r in (0, rows - 1) or c in (0, cols - 1) for r, c in placement):
                continue
            confidence = deepcopy(prior_piece.get("confidence") or {})
            if float(confidence.get("occupancy", 0.0)) < 0.70:
                continue
            facing = str(prior_piece.get("facing") or "")
            if facing not in {"U", "D", "L", "R"}:
                continue
            dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[facing]
            head = max(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
            rump = min(placement, key=lambda rc: rc[0] * dr + rc[1] * dc)
            axis = "H" if rump[0] == head[0] else "V"
            confidence.update({"temporal_presence": 0.5, "temporal_facing": 1.0})
            recovered = {
                "id": next_id,
                "cells": [list(rump), list(head)],
                "rump": list(rump), "head": list(head),
                "axis": axis, "facing": facing,
                "species": prior_piece.get("species", "sheep"),
                "hit_limit": prior_piece.get("hit_limit"),
                "hits_remaining": prior_piece.get("hits_remaining"),
                "awake": prior_piece.get("awake"),
                "confidence": confidence,
                "source_id": "temporal-edge",
                "quality": 1.0,
                "temporal_restored": True,
            }
            result.append(recovered)
            occupied.update(placement)
            restored.append({"cells": [list(cell) for cell in placement],
                             "facing": facing, "species": recovered["species"]})
            next_id += 1

    current_hazards = {(int(item["row"]), int(item["col"])) if isinstance(item, dict)
                       else (int(item[0]), int(item[1])) for item in (hazards or [])}
    hazard_counts = Counter(current_hazards)
    for frame in compatible:
        hazard_counts.update(tuple(cell) for cell in (frame.get("hazards") or []))
    horizon = len(compatible) + 1
    timeline = []
    uncertain = []
    for cell, count in sorted(hazard_counts.items()):
        current = cell in current_hazards
        ratio = count / horizon
        if current and count >= 2:
            state = "stable"
        elif current:
            state = "emerging"
            uncertain.append(list(cell))
        elif count >= 2:
            state = "fading"
            uncertain.append(list(cell))
        else:
            state = "transient"
        timeline.append({"cell": list(cell), "state": state, "present": current,
                         "observations": int(count), "frames": horizon,
                         "confidence": round(ratio, 4)})
    hazard_lookup = {tuple(item["cell"]): item for item in timeline}
    hazards_out = []
    for item in hazards or []:
        cell = (int(item["row"]), int(item["col"])) if isinstance(item, dict) else tuple(item)
        base = deepcopy(item) if isinstance(item, dict) else {"row": cell[0], "col": cell[1]}
        temporal = hazard_lookup.get(cell, {})
        base["temporal_state"] = temporal.get("state", "emerging")
        base["confidence"] = temporal.get("confidence", 1.0)
        hazards_out.append(base)
    return result, hazards_out, {
        "frames": horizon,
        "history_frames": len(compatible),
        "corrections": corrections,
        "restored_edge_pieces": restored,
        "hazards": timeline,
        "uncertain_hazard_cells": uncertain,
    }
