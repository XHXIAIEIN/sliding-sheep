"""Facing-direction correction learning and application."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from paths import ROOT

from .features import cell_key
from .fusion import _detector
from .manual_learning import load_manual_learning


DIRECTION_LEARNING_DIR = ROOT / "cache" / "direction_learning"
DIRECTION_LEARNING_INDEX = DIRECTION_LEARNING_DIR / "index.jsonl"


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
