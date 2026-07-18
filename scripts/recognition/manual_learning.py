"""Manual sample learning: persistence, proposals, rejections, and label application."""
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

import cv2
import numpy as np

from paths import ROOT

from .features import (PAIR_FEATURE_NAMES, PAIR_FEATURE_SCHEMA,
                       _feature_distance, cell_key, pair_visual_feature)


MANUAL_LEARNING_DIR = ROOT / "cache" / "recognition_learning"
MANUAL_LEARNING_INDEX = MANUAL_LEARNING_DIR / "index.jsonl"
MANUAL_LEARNING_SCHEMA = 2
MAX_ACTIVE_CONFIRMATIONS = 128
MANUAL_PRESENCE_SPECIES = frozenset({
    "sheep", "rocket", "bomb", "pink_sheep", "pig", "goat", "black_sheep", "cattle",
})
MANUAL_LABEL_SPECIES = MANUAL_PRESENCE_SPECIES
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
