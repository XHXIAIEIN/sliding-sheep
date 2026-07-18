"""Silent, non-blocking incremental learning for solver strategy selection.

The learner deliberately stays outside the planner.  The planner accepts a
plain policy dictionary and remains deterministic when no policy is supplied;
this module observes completed strategy attempts, updates a small in-memory
profile, and persists it on a daemon thread.  Learning failures never affect a
solve or become a user-facing blocker.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import queue
import secrets
import threading
import time
from pathlib import Path

from paths import ROOT


SCHEMA = 1
LEARNING_PATH = ROOT / "cache" / "solver_strategy_learning.json"
STRATEGIES = (
    "macro-beam", "randomized-macro", "weighted-a*", "beam",
    "exact-a*", "online-greedy", "greedy",
)
DEFAULT_ORDERS = {
    "macro": ["macro-beam", "randomized-macro"],
    "standard": ["weighted-a*", "beam"],
}
DEFAULT_WEIGHTS = {
    "macro-beam": .32,
    "randomized-macro": .68,
    "weighted-a*": .62,
    "beam": .38,
}

_lock = threading.RLock()
_state: dict | None = None
_writes: queue.SimpleQueue = queue.SimpleQueue()
_writer_started = False
_warm_started = False


def _empty_state() -> dict:
    return {"schema": SCHEMA, "updated_at": None, "profiles": {}}


def _load_state() -> dict:
    global _state
    with _lock:
        if _state is not None:
            return _state
    try:
        data = json.loads(LEARNING_PATH.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA or not isinstance(data.get("profiles"), dict):
            data = _empty_state()
    except Exception:
        data = _empty_state()
    with _lock:
        if _state is None:
            _state = data
        return _state


def board_features(board) -> dict:
    pieces = list(getattr(board, "pieces", {}).values())
    species: dict[str, int] = {}
    for piece in pieces:
        name = str(piece.get("species", "sheep"))
        species[name] = species.get(name, 0) + 1
    count = len(pieces)
    supported_macro = bool(
        getattr(board, "model", None) == "facing"
        and getattr(board, "slide_mode", None) == "all"
        and not getattr(board, "returning", {})
        and all(name in {"sheep", "goat", "rocket", "bomb"} for name in species)
    )
    return {
        "model": str(getattr(board, "model", "unknown")),
        "slide_mode": str(getattr(board, "slide_mode", "unknown")),
        "rows": int(getattr(board, "rows", 0)),
        "cols": int(getattr(board, "cols", 0)),
        "piece_bucket": min(120, (count // 10) * 10),
        "piece_count": count,
        "special_species": sorted(name for name in species if name != "sheep"),
        "has_hazards": bool(getattr(board, "hazards", [])),
        "has_fences": bool(getattr(board, "fences", [])),
        "has_no_stop": bool(getattr(board, "no_stop", [])),
        "macro": supported_macro,
    }


def profile_key(features: dict) -> str:
    specials = ",".join(features.get("special_species") or []) or "plain"
    flags = "".join((
        "h" if features.get("has_hazards") else "-",
        "f" if features.get("has_fences") else "-",
        "n" if features.get("has_no_stop") else "-",
        "m" if features.get("macro") else "-",
    ))
    return (f"{features.get('model')}:{features.get('slide_mode')}|"
            f"{features.get('rows')}x{features.get('cols')}|"
            f"p{features.get('piece_bucket')}|{specials}|{flags}")


def _strategy_value(stats: dict, total_attempts: int) -> float:
    attempts = max(0, int(stats.get("attempts", 0)))
    if attempts <= 0:
        return .55
    solved_rate = float(stats.get("solved", 0)) / attempts
    progress_rate = float(stats.get("progress_total", 0.0)) / attempts
    average_ms = float(stats.get("elapsed_ms_total", 0.0)) / attempts
    exploration = math.sqrt(math.log(max(2, total_attempts + 1)) / attempts)
    return solved_rate * 4.0 + progress_rate * 1.8 + exploration * .35 - min(.7, average_ms / 90_000.0)


def policy_from_profile(profile: dict | None) -> dict:
    profile = profile or {}
    strategies = profile.get("strategies") or {}
    samples = int(profile.get("solves", 0))
    result = {
        "samples": samples,
        "orders": deepcopy(DEFAULT_ORDERS),
        "time_weights": deepcopy(DEFAULT_WEIGHTS),
    }
    # Preserve the hand-tuned portfolio during cold start.  After a few solves,
    # rank by smoothed success, progress, speed and a small exploration bonus.
    if samples < 3:
        return result
    total_attempts = sum(int(item.get("attempts", 0)) for item in strategies.values())
    for family, defaults in DEFAULT_ORDERS.items():
        ranked = sorted(
            defaults,
            key=lambda name: (
                -_strategy_value(strategies.get(name, {}), total_attempts),
                defaults.index(name),
            ),
        )
        result["orders"][family] = ranked
        values = {name: max(.15, _strategy_value(strategies.get(name, {}), total_attempts))
                  for name in defaults}
        total = sum(values.values()) or 1.0
        for name, value in values.items():
            # Keep every portfolio member alive so learning remains incremental
            # instead of permanently locking onto an early lucky result.
            result["time_weights"][name] = max(.2, min(.8, value / total))
    return result


def policy_for(board) -> dict:
    """Return immediately from memory while persistence warms in background."""
    features = board_features(board)
    key = profile_key(features)
    warm_async()
    with _lock:
        profile = deepcopy((((_state or {}).get("profiles") or {}).get(key)))
    policy = policy_from_profile(profile)
    policy["profile_key"] = key
    return policy


def warm_async() -> None:
    """Start one best-effort background load without delaying a solve."""
    global _warm_started
    with _lock:
        if _warm_started or _state is not None:
            return
        _warm_started = True

    def load() -> None:
        try:
            _load_state()
        except Exception:
            return

    threading.Thread(
        target=load, name="sheep-solver-learning-warm", daemon=True,
    ).start()


def _apply_observation(state: dict, observation: dict) -> None:
    profiles = state.setdefault("profiles", {})
    key = observation["profile_key"]
    profile = profiles.setdefault(key, {
        "features": observation["features"],
        "solves": 0,
        "solved": 0,
        "best_remaining": observation.get("initial_remaining", 0),
        "strategies": {},
    })
    profile["solves"] = int(profile.get("solves", 0)) + 1
    profile["solved"] = int(profile.get("solved", 0)) + int(bool(observation.get("solved")))
    profile["best_remaining"] = min(
        int(profile.get("best_remaining", observation.get("remaining", 0))),
        int(observation.get("remaining", 0)),
    )
    profile["last_seen_at"] = observation.get("created_at")
    for event in observation.get("trace") or []:
        if event.get("event") != "finish" or event.get("phase") not in STRATEGIES:
            continue
        name = event["phase"]
        stats = profile.setdefault("strategies", {}).setdefault(name, {
            "attempts": 0, "solved": 0, "elapsed_ms_total": 0,
            "expanded_total": 0, "progress_total": 0.0,
        })
        stats["attempts"] = int(stats.get("attempts", 0)) + 1
        stats["solved"] = int(stats.get("solved", 0)) + int(bool(event.get("solved")))
        stats["elapsed_ms_total"] = int(stats.get("elapsed_ms_total", 0)) + max(
            0, int(event.get("elapsed_ms", 0)))
        stats["expanded_total"] = int(stats.get("expanded_total", 0)) + max(
            0, int(event.get("expanded", 0)))
        before = max(1, int(event.get("start_remaining") or observation.get("initial_remaining") or 1))
        after = max(0, int(event.get("remaining", before)))
        stats["progress_total"] = float(stats.get("progress_total", 0.0)) + max(
            0.0, min(1.0, (before - after) / before))
        stats["best_remaining"] = min(int(stats.get("best_remaining", after)), after)
        stats["last_elapsed_ms"] = max(0, int(event.get("elapsed_ms", 0)))
    state["updated_at"] = observation.get("created_at")


def _persist(state: dict) -> None:
    LEARNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = LEARNING_PATH.with_name(f".{LEARNING_PATH.name}.{secrets.token_hex(4)}.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(LEARNING_PATH)


def _writer_loop() -> None:
    while True:
        observation = _writes.get()
        try:
            with _lock:
                state = _load_state()
                _apply_observation(state, observation)
                snapshot = deepcopy(state)
            _persist(snapshot)
        except Exception:
            # Learning is intentionally best-effort and must never surface as
            # a solve failure or block execution.
            continue


def _ensure_writer() -> None:
    global _writer_started
    with _lock:
        if _writer_started:
            return
        threading.Thread(
            target=_writer_loop, name="sheep-solver-learning", daemon=True,
        ).start()
        _writer_started = True


def record_async(board, trace: list[dict], *, solved: bool, remaining: int) -> None:
    """Queue one completed solve observation and return immediately."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        features = board_features(board)
        observation = {
            "profile_key": profile_key(features),
            "features": features,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "initial_remaining": int(features.get("piece_count", 0)),
            "solved": bool(solved),
            "remaining": max(0, int(remaining)),
            "trace": deepcopy(list(trace or [])),
        }
        _ensure_writer()
        _writes.put(observation)
    except Exception:
        return
