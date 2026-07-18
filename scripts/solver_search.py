"""Heuristic search solvers for larger facing-model boards.

The plain greedy solver only follows one local choice and often paints itself
into a corner.  This module keeps multiple candidate states, using Rush Hour
style blocker features to rank them.
"""
from __future__ import annotations

import heapq
import random
import time

from solver import Board, DIRS, Move, greedy_solve, solve as exact_solve


def _shift(cells, direction):
    dr, dc = DIRS[direction]
    return frozenset((r + dr, c + dc) for r, c in cells)


def _cell_owners(board: Board):
    owners = {}
    for pid, piece in board.pieces.items():
        for cell in piece["cells"]:
            owners[cell] = pid
    return owners


def analyze(board: Board):
    """Return cheap blocker/deadlock features for ranking states."""
    pieces = board.pieces
    n = board.remaining_count()
    if not pieces:
        return {
            "remaining": n,
            "blockers": 0,
            "hazard_blockers": 0,
            "stuck": 0,
            "can_exit": 0,
            "movable": 0,
            "deadlocks": 0,
            "terminal_deadlock": 0,
        }

    owners = _cell_owners(board)
    immediate_blocker = {}
    blockers_total = hazard_blockers = stuck = can_exit = movable = 0

    def directions(piece):
        if board.model == "facing":
            return [piece["facing"]]
        rows = {r for r, _ in piece["cells"]}
        cols = {c for _, c in piece["cells"]}
        if len(rows) == 1 and len(cols) > 1:
            return ["L", "R"]
        if len(cols) == 1 and len(rows) > 1:
            return ["U", "D"]
        return ["U", "D", "L", "R"]

    for pid, piece in pieces.items():
        cells = piece["cells"]
        if piece.get("species") == "pig" and not piece.get("awake", True):
            # Sleeping pigs are blockers until another animal collides with
            # them; do not reward the state as if the pig could already exit.
            stuck += 1
            continue
        options = []
        for direction in directions(piece):
            first = _shift(cells, direction)
            first_piece_hits = {owners[cell] for cell in first if cell in owners and cell not in cells}
            first_hazard_hits = set(first) & set(board.hazards)
            if piece.get("species", "sheep") != "cattle":
                first_hazard_hits.update(board.fence_cell_hits(cells, direction))
            if (any(not board.in_board(r, c) for r, c in first)
                    and piece.get("species", "sheep") != "cattle"):
                first_hazard_hits.update(board.fence_crossings(cells, direction))
            blockers, hazard_hits = set(), set()
            frontier = cells
            while True:
                nxt = _shift(frontier, direction)
                if any(not board.in_board(r, c) for r, c in nxt):
                    if piece.get("species", "sheep") != "cattle":
                        hazard_hits.update(board.fence_crossings(frontier, direction))
                    break
                internal_hits = board.fence_cell_hits(frontier, direction)
                if internal_hits and piece.get("species", "sheep") == "cattle":
                    break
                blockers.update(owners[cell] for cell in nxt if cell in owners and cell not in cells)
                hazard_hits.update(set(nxt) & set(board.hazards))
                if piece.get("species", "sheep") != "cattle":
                    hazard_hits.update(internal_hits)
                frontier = nxt
            options.append((first_piece_hits, first_hazard_hits, blockers, hazard_hits))
        if any(not first_p and not first_h for first_p, first_h, _b, _h in options):
            movable += 1
        else:
            stuck += 1
            piece_hits = [hits for hits, hazards, _b, _h in options if hits and not hazards]
            if len(piece_hits) == 1 and len(piece_hits[0]) == 1:
                immediate_blocker[pid] = next(iter(piece_hits[0]))
        best = min(options, key=lambda item: (len(item[2]) + len(item[3]), len(item[3])))
        blockers_total += len(best[2])
        hazard_blockers += len(best[3])
        if not best[2] and not best[3]:
            can_exit += 1

    deadlocks = 0
    seen = set()
    for a, b in immediate_blocker.items():
        if immediate_blocker.get(b) == a:
            pair = frozenset((a, b))
            if pair not in seen:
                seen.add(pair)
                deadlocks += 1

    # A non-empty state without an exit or even a collision move can never
    # recover.  Treat it as a terminal deadlock instead of rewarding its low
    # remaining-piece count; otherwise large-board searches spend most of
    # their budget descending into attractive five-or-six-piece cul-de-sacs.
    no_progress = n > 0 and can_exit == 0 and movable == 0
    simple_species = {"sheep", "goat", "rocket"}
    only_simple_pieces = all(
        piece.get("species", "sheep") in simple_species for piece in pieces.values()
    )
    # Special pieces can have legal BOUNCE/collision actions that the cheap
    # feature pass deliberately does not model.  Confirm those cases through
    # the authoritative move generator before declaring the state terminal.
    terminal_deadlock = int(
        no_progress and (only_simple_pieces or not board.legal_moves())
    )
    if not terminal_deadlock and n <= 12:
        legal = board.legal_moves()
        # Near the endgame, cheaply look through the last apparent escape.
        # A state whose every legal action immediately freezes a non-empty
        # board is just as unsalvageable as an already frozen state.
        terminal_deadlock = int(
            bool(legal)
            and all(
                not (next_board := board.apply(move)).is_solved()
                and not next_board.legal_moves()
                for move in legal
            )
        )

    return {
        "remaining": n,
        "blockers": blockers_total,
        "hazard_blockers": hazard_blockers,
        "stuck": stuck,
        "can_exit": can_exit,
        "movable": movable,
        "deadlocks": deadlocks,
        "terminal_deadlock": terminal_deadlock,
    }


def _heuristic(board: Board):
    f = analyze(board)
    return (
        f["terminal_deadlock"] * 1_000_000
        + f["remaining"] * 120
        + f["blockers"] * 10
        + f["hazard_blockers"] * 18
        + f["stuck"] * 6
        + f["deadlocks"] * 90
        - f["can_exit"] * 22
        - f["movable"] * 2
    )


def _rank(board: Board, depth: int):
    f = analyze(board)
    return (
        f["terminal_deadlock"],
        f["deadlocks"],
        f["remaining"],
        f["blockers"],
        f["hazard_blockers"],
        f["stuck"],
        -f["can_exit"],
        -f["movable"],
        depth,
    )


def _move_rank(mv: Move):
    return (0 if mv.result == "EXIT" else 1, -mv.distance, mv.piece_id)


def _ordered_moves(board: Board):
    return sorted(board.legal_moves(), key=_move_rank)


def structural_deadlocks(board: Board):
    """Find collinear pieces whose fixed facings point toward each other."""
    if board.model != "facing":
        return []
    rows, cols = {}, {}
    for pid, piece in board.pieces.items():
        row_set = {r for r, _ in piece["cells"]}
        col_set = {c for _, c in piece["cells"]}
        if len(row_set) == 1:
            rows.setdefault(next(iter(row_set)), []).append((str(pid), piece))
        if len(col_set) == 1:
            cols.setdefault(next(iter(col_set)), []).append((str(pid), piece))

    pairs = []
    seen = set()
    for axis, groups in (("H", rows), ("V", cols)):
        for lane, pieces in groups.items():
            for index, first in enumerate(pieces):
                for second in pieces[index + 1:]:
                    (a_id, a), (b_id, b) = first, second
                    coordinate = ((lambda p: min(c for _, c in p["cells"])) if axis == "H"
                                  else (lambda p: min(r for r, _ in p["cells"])))
                    if coordinate(a) > coordinate(b):
                        a_id, b_id, a, b = b_id, a_id, b, a
                    opposing = ((a.get("facing"), b.get("facing")) == ("R", "L")
                                if axis == "H" else
                                (a.get("facing"), b.get("facing")) == ("D", "U"))
                    key = tuple(sorted((a_id, b_id)))
                    if not opposing or key in seen:
                        continue
                    seen.add(key)
                    pairs.append({
                        "axis": axis, "lane": int(lane), "pieces": [a_id, b_id],
                        "facings": [a.get("facing"), b.get("facing")],
                        "species": [a.get("species", "sheep"), b.get("species", "sheep")],
                        "cells": [[list(cell) for cell in sorted(a["cells"])],
                                  [list(cell) for cell in sorted(b["cells"])]],
                    })
    return pairs


EXIT_CLOSURE_SPECIES = frozenset({"sheep", "goat", "rocket", "bomb"})


def supports_forced_exit_closure(board: Board):
    """Whether deterministic exit draining is proven monotonic for this board."""
    return bool(
        board.model == "facing"
        and board.slide_mode == "all"
        and not getattr(board, "returning", {})
        and all(piece.get("species", "sheep") in EXIT_CLOSURE_SPECIES
                for piece in board.pieces.values())
    )


def _protected_exit_stoppers(board: Board, legal_moves=None):
    """Return direct exits whose eager removal would close a blocker cycle.

    With ``slide_mode=all`` a click cannot choose an intermediate landing cell.
    A piece that can currently move therefore depends on the pieces farther
    along its facing lane to stop it at useful positions.  Some of those
    stoppers may themselves be direct exits.  Removing an ordinary isolated
    stopper is still profitable; protect only a run whose removal makes the
    mover hit a non-exiting blocker that already depends on the mover.  That
    is the level-172 ``55 -> 54 -> 64 -> 59 -> 70 -> 80 -> 81`` trap.

    Search is then allowed to interleave the staged move with those exits,
    while ordinary acyclic exit lanes keep the short paths and small search
    space of the original macro closure.
    """
    legal_moves = list(legal_moves if legal_moves is not None else board.legal_moves())
    exit_ids = {move.piece_id for move in legal_moves if move.result == "EXIT"}
    if not exit_ids:
        return set()

    owners = {
        cell: pid for pid, piece in board.pieces.items() for cell in piece["cells"]
    }

    def forward_owners(piece_id, direction):
        dr, dc = DIRS[direction]
        frontier = board.pieces[piece_id]["cells"]
        result = []
        while True:
            frontier = frozenset((r + dr, c + dc) for r, c in frontier)
            if any(not board.in_board(r, c) for r, c in frontier):
                break
            found = sorted({
                owners[cell] for cell in frontier
                if cell in owners and owners[cell] != piece_id
            }, key=str)
            for owner in found:
                if owner not in result:
                    result.append(owner)
        return result

    # Collapse every currently direct exit out of the dependency graph.  The
    # remaining edge is where each piece would eventually stop if the eager
    # closure drained all exits first.
    collapsed_blocker = {}
    for pid, piece in board.pieces.items():
        direction = piece.get("facing")
        if not direction:
            continue
        blocker = next((owner for owner in forward_owners(pid, direction)
                        if owner not in exit_ids), None)
        if blocker is not None:
            collapsed_blocker[pid] = blocker

    protected = set()
    for move in legal_moves:
        if move.result == "EXIT" or move.distance <= 0:
            continue
        staged_exits = []
        final_blocker = None
        for owner in forward_owners(move.piece_id, move.direction):
            if owner in exit_ids:
                staged_exits.append(owner)
            else:
                final_blocker = owner
                break
        if not staged_exits or final_blocker is None:
            continue

        current = final_blocker
        seen = set()
        while current not in seen:
            if current == move.piece_id:
                protected.update(staged_exits)
                break
            seen.add(current)
            current = collapsed_blocker.get(current)
            if current is None:
                break
    return protected


def _exit_unlock_count(board: Board, move: Move):
    """Count pieces that are immediately blocked by this exiting piece."""
    if board.model != "facing":
        return 0
    target = board.pieces[move.piece_id]["cells"]
    count = 0
    for pid, piece in board.pieces.items():
        if pid == move.piece_id or not piece.get("facing"):
            continue
        dr, dc = DIRS[piece["facing"]]
        first = {(r + dr, c + dc) for r, c in piece["cells"]}
        if first & target:
            count += 1
    return count


def forced_exit_sort_key(board: Board, move: Move):
    """Prefer exits that expose movement before farther stoppers are removed."""
    return (-_exit_unlock_count(board, move),
            move.anchor[0], move.anchor[1], str(move.piece_id))


def forced_exit_candidates(board: Board, *, ordinary_only=False):
    """Return exits safe for deterministic draining in the current state."""
    legal_moves = board.legal_moves()
    protected = _protected_exit_stoppers(board, legal_moves)
    candidates = []
    for move in legal_moves:
        if move.result != "EXIT" or move.piece_id in protected:
            continue
        if (ordinary_only
                and board.pieces.get(move.piece_id, {}).get("species", "sheep") != "sheep"):
            continue
        candidates.append(move)
    return candidates


def forced_exit_closure(board: Board):
    """Apply only stopper-safe direct exits and return the normalized board.

    Most direct exits only remove blockers and are profitably collapsed into a
    macro step.  Exits used as staged stoppers are deliberately left for search
    because draining them can make ``slide_mode=all`` boards self-lock.
    """
    if not supports_forced_exit_closure(board):
        return [], board
    moves = []
    cur = board
    while True:
        exits = forced_exit_candidates(cur)
        if not exits:
            return moves, cur
        move = min(exits, key=lambda item: forced_exit_sort_key(cur, item))
        moves.append(move)
        cur = cur.apply(move)


def _reconstruct(nodes, idx):
    groups = []
    while idx:
        idx, edge = nodes[idx]
        groups.append(edge if isinstance(edge, (tuple, list)) else (edge,))
    return [move for group in reversed(groups) for move in group]


def randomized_macro_solve(board: Board, seed=0, time_limit=4.0, cancel=None):
    """Diversified restarts over collision moves with direct exits collapsed.

    Deterministic best-first searches tend to repeat the same attractive but
    doomed low-remainder endgames on dense boards.  Facing-model movement is
    monotonic, so a short randomized walk over only the collision decisions is
    an inexpensive complementary search.  A fixed seed keeps CLI/cache output
    reproducible.
    """
    if not supports_forced_exit_closure(board):
        return [], {"solved": False, "kind": "randomized-macro(unsupported)",
                    "remaining": board.remaining_count(), "expanded": 0}

    rng = random.Random(seed)
    start = time.monotonic()
    deadline = start + time_limit
    expanded = restarts = 0
    best_path = []
    best_remaining = board.remaining_count()

    while time.monotonic() < deadline and not (cancel and cancel()):
        cur = board
        path = []
        seen = set()
        restarts += 1
        while time.monotonic() < deadline and not (cancel and cancel()):
            forced, cur = forced_exit_closure(cur)
            path.extend(forced)
            remaining = cur.remaining_count()
            if remaining < best_remaining:
                best_remaining = remaining
                best_path = path[:]
            if cur.is_solved():
                return path, {
                    "solved": True,
                    "kind": "randomized-macro",
                    "remaining": 0,
                    "expanded": expanded,
                    "restarts": restarts,
                    "seed": seed,
                }
            key = cur.key()
            if key in seen:
                break
            seen.add(key)
            expanded += 1

            candidates = []
            for move in cur.legal_moves():
                forced, next_board = forced_exit_closure(cur.apply(move))
                rank = _rank(next_board, len(path) + 1 + len(forced))
                if rank[0]:
                    continue
                # rank fields: terminal, cycle, remaining, blockers, hazards,
                # stuck, -exits, -movable, depth.  Remaining dominates while
                # the last terms retain enough diversity to escape plateaus.
                score = (rank[2] * 100 + rank[3] * 6 + rank[4] * 10
                         + rank[5] * 2 + rank[6] * 12 + rank[7] * 2)
                candidates.append((score, move, forced, next_board))
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])
            pool_size = min(len(candidates), max(2, int(len(candidates) ** 0.5) + 1))
            pool = candidates[:pool_size]
            if rng.random() < 0.3:
                _score, move, forced, cur = rng.choice(candidates)
            else:
                weights = [1 / (1 + index) for index in range(pool_size)]
                _score, move, forced, cur = rng.choices(pool, weights=weights, k=1)[0]
            path.append(move)
            path.extend(forced)

    return best_path, {
        "solved": False,
        "kind": "randomized-macro(best)",
        "remaining": best_remaining,
        "expanded": expanded,
        "restarts": restarts,
        "seed": seed,
        "cancelled": bool(cancel and cancel()),
        "timeout": time.monotonic() >= deadline,
    }


def weighted_astar_solve(board: Board, weight=1.35, max_nodes=120_000, time_limit=8.0,
                         cancel=None):
    """Weighted A*: not optimal, but usually much stronger than greedy."""
    start = time.monotonic()
    prefix, board = forced_exit_closure(board)
    if board.is_solved():
        return prefix, {"solved": True, "kind": "exit-closure", "expanded": 0,
                        "remaining": 0}
    nodes = [(0, None)]
    counter = 0
    heap = [(weight * _heuristic(board), 0, counter, board, 0)]
    best_g = {board.key(): 0}
    best_idx = 0
    best_rank = _rank(board, 0)
    expanded = 0
    endgame_unsat = set()

    while (heap and expanded < max_nodes and time.monotonic() - start < time_limit
           and not (cancel and cancel())):
        _prio, g, _order, cur, node_idx = heapq.heappop(heap)
        if g != best_g.get(cur.key()):
            continue
        if cur.is_solved():
            return prefix + _reconstruct(nodes, node_idx), {
                "solved": True,
                "kind": "weighted-a*",
                "expanded": expanded,
            }
        expanded += 1
        for mv in _ordered_moves(cur):
            forced, nb = forced_exit_closure(cur.apply(mv))
            edge = (mv, *forced)
            nk = nb.key()
            ng = g + len(edge)
            if ng >= best_g.get(nk, 1 << 30):
                continue
            if nk in endgame_unsat:
                continue
            if nb.remaining_count() <= 8:
                tail, tail_info = exact_solve(nb, max_nodes=20_000, cancel=cancel)
                if tail is not None:
                    return prefix + _reconstruct(nodes, node_idx) + list(edge) + tail, {
                        "solved": True,
                        "kind": "weighted-a*+exact-endgame",
                        "expanded": expanded + int(tail_info.get("expanded", 0)),
                        "remaining": 0,
                    }
                if tail_info.get("reason") == "无解（搜索穷尽）":
                    endgame_unsat.add(nk)
                    continue
            rank = _rank(nb, ng)
            if rank[0]:
                continue
            best_g[nk] = ng
            nodes.append((node_idx, edge))
            child_idx = len(nodes) - 1
            if rank < best_rank:
                best_rank = rank
                best_idx = child_idx
            counter += 1
            heapq.heappush(heap, (ng + weight * _heuristic(nb), ng, counter, nb, child_idx))

    path = prefix + _reconstruct(nodes, best_idx)
    solved = best_rank[2] == 0
    return path, {
        "solved": solved,
        "kind": "weighted-a*" if solved else "weighted-a*(best)",
        "remaining": best_rank[2],
        "expanded": expanded,
        "cancelled": bool(cancel and cancel()),
    }


def beam_solve(board: Board, width=5000, max_depth=220, time_limit=12.0, seen_cap=450_000,
               cancel=None):
    """Layered beam search.  Keeps diverse promising partial solutions."""
    start = time.monotonic()
    prefix, board = forced_exit_closure(board)
    if board.is_solved():
        return prefix, {"solved": True, "kind": "exit-closure", "expanded": 0,
                        "remaining": 0, "depth": 0}
    nodes = [(0, None)]
    beam = [(_rank(board, 0), board, 0)]
    seen = {board.key(): 0}
    best_idx = 0
    best_rank = beam[0][0]
    expanded = 0

    for depth in range(1, max_depth + 1):
        if time.monotonic() - start >= time_limit or (cancel and cancel()):
            break
        nxt = []
        for _rank_key, cur, node_idx in beam:
            if time.monotonic() - start >= time_limit or (cancel and cancel()):
                break
            if cur.is_solved():
                return prefix + _reconstruct(nodes, node_idx), {
                    "solved": True,
                    "kind": "beam",
                    "expanded": expanded,
                    "depth": depth - 1,
                }
            expanded += 1
            for mv in _ordered_moves(cur):
                if time.monotonic() - start >= time_limit or (cancel and cancel()):
                    break
                forced, nb = forced_exit_closure(cur.apply(mv))
                edge = (mv, *forced)
                nk = nb.key()
                old_depth = seen.get(nk)
                if old_depth is not None and old_depth <= depth:
                    continue
                rank = _rank(nb, depth)
                if rank[0]:
                    continue
                if len(seen) < seen_cap:
                    seen[nk] = depth
                nodes.append((node_idx, edge))
                child_idx = len(nodes) - 1
                if nb.is_solved():
                    return prefix + _reconstruct(nodes, child_idx), {
                        "solved": True,
                        "kind": "beam",
                        "expanded": expanded,
                        "depth": depth,
                    }
                if rank < best_rank:
                    best_rank = rank
                    best_idx = child_idx
                nxt.append((rank, nb, child_idx))
        if not nxt:
            break
        nxt.sort(key=lambda item: item[0])
        beam = nxt[:width]

    path = prefix + _reconstruct(nodes, best_idx)
    solved = best_rank[2] == 0
    return path, {
        "solved": solved,
        "kind": "beam" if solved else "beam(best)",
        "remaining": best_rank[2],
        "expanded": expanded,
        "cancelled": bool(cancel and cancel()),
    }


def search_solve(board: Board):
    """Try stronger search first, then keep the best result against greedy."""
    candidates = []

    macro_search = supports_forced_exit_closure(board)
    if macro_search:
        prefix, normalized = forced_exit_closure(board)
        deadlocks = structural_deadlocks(normalized)
        if deadlocks:
            return prefix, {
                "solved": False, "kind": "structural-deadlock",
                "remaining": normalized.remaining_count(), "expanded": 0,
                "structural_deadlocks": deadlocks,
                "reason": "固定朝向棋子在同一直线上迎头相向，当前规则模型不可解",
            }
    if macro_search:
        moves, info = randomized_macro_solve(board)
        candidates.append((moves, info))
        if info.get("solved"):
            return moves, info

    if macro_search:
        moves, info = beam_solve(
            board, width=14, max_depth=96, time_limit=3.0, seen_cap=120_000)
        info = {**info, "kind": f"macro-{info.get('kind', 'beam')}"}
        candidates.append((moves, info))
        if info.get("solved"):
            return moves, info

    moves, info = weighted_astar_solve(board)
    candidates.append((moves, info))
    if info.get("solved"):
        return moves, info

    if not macro_search:
        moves, info = beam_solve(board)
        candidates.append((moves, info))
        if info.get("solved"):
            return moves, info

    gmoves, ginfo = greedy_solve(board)
    ginfo = {**ginfo, "kind": "greedy"}
    candidates.append((gmoves, ginfo))

    def result_key(item):
        moves, info = item
        cur = board
        for move in moves:
            cur = cur.apply(move)
        remaining = cur.remaining_count()
        legal_count = len(cur.legal_moves()) if remaining else 0
        return (
            0 if info.get("solved") else 1,
            info.get("remaining", remaining),
            1 if remaining and legal_count == 0 else 0,
            -legal_count,
            len(moves),
        )

    return min(candidates, key=result_key)
