"""自测：构造棋盘 -> 求解 -> 回放校验棋盘清空。运行: py scripts/test_solver.py"""
import json
from pathlib import Path

from solver import Board, solve
from board import io as board_io
from solver import planner
from board.io import render, describe
from solver.search import (analyze as search_analyze, beam_solve,
                           forced_exit_closure, search_solve,
                           randomized_macro_solve,
                           structural_deadlocks, supports_forced_exit_closure,
                           weighted_astar_solve)


def replay(board: Board, moves):
    """逐步回放，校验每一步都是当时的合法动作，最终清空。"""
    cur = board
    for mv in moves:
        legal = cur.legal_moves()
        assert any(m == mv for m in legal), f"非法步: {mv}\n当前:\n{render(cur)}"
        cur = cur.apply(mv)
    return cur


def test_basic_exit():
    # 4x4，三只水平羊错位排列，需先挪开挡路的才能依次赶出
    b = Board(4, 4, {
        "A": {"cells": [[1, 1], [1, 2]]},
        "B": {"cells": [[1, 3]]},          # 单格，挡住 A 向右出口
        "C": {"cells": [[0, 0], [0, 1]]},
    }, model="axis_both", slide_mode="all")
    moves, info = solve(b)
    assert moves is not None, f"应有解: {info}"
    final = replay(b, moves)
    assert final.is_solved(), "回放后棋盘未清空"
    return b, moves, info


def test_blocked_chain():
    # 竖直羊堵住水平羊，必须先动竖直的
    b = Board(5, 5, {
        "A": {"cells": [[2, 1], [2, 2]]},
        "V": {"cells": [[0, 3], [1, 3], [2, 3]]},  # 挡住 A 向右
    }, model="axis_both", slide_mode="all")
    moves, info = solve(b)
    assert moves is not None, f"应有解: {info}"
    assert replay(b, moves).is_solved()
    return b, moves, info


def test_facing_model():
    # facing 模型：每只羊只能朝头的方向走
    b = Board(4, 4, {
        "A": {"cells": [[1, 0], [1, 1]], "facing": "R"},   # 朝右
        "B": {"cells": [[1, 2], [1, 3]], "facing": "R"},   # 朝右，在 A 前方
    }, model="facing", slide_mode="all")
    moves, info = solve(b)
    assert moves is not None, f"应有解: {info}"
    assert replay(b, moves).is_solved()
    # B 必须先出（在 A 前方挡路）
    assert moves[0].piece_id == "B"
    return b, moves, info


def test_opening_coarse_only_drains_direct_ordinary_sheep():
    board = Board(2, 6, {
        "front": {"cells": [[0, 4], [0, 5]], "facing": "R", "species": "sheep"},
        "rear": {"cells": [[0, 2], [0, 3]], "facing": "R", "species": "sheep"},
        "goat": {"cells": [[1, 4], [1, 5]], "facing": "R", "species": "goat"},
    }, model="facing", slide_mode="all")

    result = planner.coarse_exit_plan(board)
    moves = [move for move, phase in result.steps if phase == "coarse"]

    assert [move.piece_id for move in moves] == ["front", "rear"], moves
    assert not result.solved and result.remaining == 1, result
    assert set(result.final_board.pieces) == {"goat"}, result.final_board.pieces
    assert result.kind == "开局粗解2", result.kind
    return board, moves, {"expanded": 0}


def test_quick_exit_plan_stops_after_three_exposure_layers():
    board = Board(1, 8, {
        "layer4": {"cells": [[0, 0], [0, 1]], "facing": "R"},
        "layer3": {"cells": [[0, 2], [0, 3]], "facing": "R"},
        "layer2": {"cells": [[0, 4], [0, 5]], "facing": "R"},
        "layer1": {"cells": [[0, 6], [0, 7]], "facing": "R"},
    }, model="facing", slide_mode="all")

    result = planner.coarse_exit_plan(board, max_layers=3)
    moves = [move for move, phase in result.steps if phase == "coarse"]

    assert [move.piece_id for move in moves] == ["layer1", "layer2", "layer3"], moves
    assert result.remaining == 1 and not result.solved, result
    assert result.info["layers"] == 3, result.info
    assert result.info["exit_layers"] == {
        "layer1": 1, "layer2": 2, "layer3": 3,
    }, result.info
    return board, moves, {"expanded": 0}


def test_exit_closure_preserves_level172_staged_stoppers():
    """Direct exits on a mover's lane must remain available to search."""
    board = Board(8, 6, {
        "54": {"cells": [[0, 1], [1, 1]], "facing": "D"},
        "55": {"cells": [[0, 2], [0, 3]], "facing": "L"},
        "56": {"cells": [[0, 4], [0, 5]], "facing": "R"},
        "59": {"cells": [[1, 2], [2, 2]], "facing": "D"},
        "60": {"cells": [[1, 3], [1, 4]], "facing": "R"},
        "61": {"cells": [[1, 5], [2, 5]], "facing": "D"},
        "64": {"cells": [[2, 0], [2, 1]], "facing": "R"},
        "65": {"cells": [[2, 3], [2, 4]], "facing": "R"},
        "69": {"cells": [[3, 1], [4, 1]], "facing": "D"},
        "70": {"cells": [[3, 2], [4, 2]], "facing": "D"},
        "71": {"cells": [[3, 3], [3, 4]], "facing": "R"},
        "72": {"cells": [[3, 5], [4, 5]], "facing": "D"},
        "77": {"cells": [[4, 3], [4, 4]], "facing": "R"},
        "80": {"cells": [[5, 1], [5, 2]], "facing": "R"},
        "81": {"cells": [[5, 3], [6, 3]], "facing": "U"},
        "82": {"cells": [[5, 4], [6, 4]], "facing": "D"},
        "85": {"cells": [[6, 1], [6, 2]], "facing": "R"},
        "88": {"cells": [[7, 2], [7, 3]], "facing": "R"},
    }, model="facing", slide_mode="all")

    prefix, normalized = forced_exit_closure(board)
    remaining = set(normalized.pieces)
    assert {"60", "65", "71"} <= remaining, (prefix, remaining)
    staged = [move for move in normalized.legal_moves() if move.piece_id == "81"]
    assert len(staged) == 1 and staged[0].result == "MOVE" and staged[0].distance == 1, staged

    moves, info = beam_solve(
        board, width=14, max_depth=32, time_limit=1.0, seen_cap=20_000)
    assert info.get("solved"), info
    assert replay(board, moves).is_solved()
    return board, moves, {"expanded": info.get("expanded", 0)}


def test_key_keeps_facing():
    a = Board(3, 4, {
        "A": {"cells": [[1, 0], [1, 1]], "facing": "R"},
        "B": {"cells": [[1, 2], [1, 3]], "facing": "L"},
    }, model="facing", slide_mode="all")
    b = Board(3, 4, {
        "A": {"cells": [[1, 0], [1, 1]], "facing": "L"},
        "B": {"cells": [[1, 2], [1, 3]], "facing": "R"},
    }, model="facing", slide_mode="all")
    assert a.key() != b.key(), "状态 key 必须区分朝向"
    return a, [], {"expanded": 0}


def test_search_one_hop_boundary():
    b = Board(3, 4, {
        "A": {"cells": [[1, 1], [1, 2]], "facing": "R"},
    }, model="facing", slide_mode="all")
    wmoves, winfo = weighted_astar_solve(b, max_nodes=1, time_limit=1.0)
    assert winfo.get("solved") and winfo.get("remaining") == 0, winfo
    assert replay(b, wmoves).is_solved()
    bmoves, binfo = beam_solve(b, width=4, max_depth=1, time_limit=1.0)
    assert binfo.get("solved") and binfo.get("remaining") == 0, binfo
    assert replay(b, bmoves).is_solved()
    return b, wmoves, {"expanded": winfo.get("expanded", 0)}


def test_opposing_facing_pair_is_reported_as_structural_deadlock():
    board = Board(3, 8, {
        "A": {"cells": [[1, 1], [1, 2]], "facing": "R", "species": "sheep"},
        "B": {"cells": [[1, 5], [1, 6]], "facing": "L", "species": "rocket"},
    }, model="facing", slide_mode="all")
    pairs = structural_deadlocks(board)
    moves, info = search_solve(board)
    assert len(pairs) == 1 and pairs[0]["pieces"] == ["A", "B"], pairs
    assert not moves and info.get("kind") == "structural-deadlock", info
    return board, [], {"expanded": 0}


def test_axis_both_search_and_hazard_features():
    b = Board(4, 6, {
        "A": {"cells": [[1, 1], [1, 2]]},
        "B": {"cells": [[2, 3], [3, 3]]},
    }, model="axis_both", slide_mode="all")
    moves, info = weighted_astar_solve(b, max_nodes=100, time_limit=1.0)
    assert moves is not None, info
    hazard = Board(3, 4, {
        "H": {"cells": [[1, 0], [1, 1]], "facing": "R"},
    }, model="facing", slide_mode="all", hazards=[[1, 3]])
    features = search_analyze(hazard)
    assert features["hazard_blockers"] > 0, features
    return b, moves, {"expanded": info.get("expanded", 0)}


def test_wolf_no_stop_zone_allows_exit_crossing_but_rejects_landing():
    crossing = Board(3, 7, {
        "A": {"cells": [[1, 0], [1, 1]], "facing": "R"},
    }, model="facing", slide_mode="all", no_stop=[[1, 3], [1, 4]])
    crossing_moves = crossing.legal_moves()
    assert len(crossing_moves) == 1 and crossing_moves[0].result == "EXIT", crossing_moves

    landing = Board(3, 7, {
        "A": {"cells": [[1, 0], [1, 1]], "facing": "R"},
        "B": {"cells": [[1, 6]], "facing": "L"},
    }, model="facing", slide_mode="all", no_stop=[[1, 4], [1, 5]])
    moves = [move for move in landing.legal_moves() if move.piece_id == "A"]
    assert moves == [], moves
    assert landing.no_stop == frozenset({(1, 4), (1, 5)})


def test_facing_elephant_2x3_can_move_and_exit():
    data = {"rows": 5, "cols": 5, "model": "facing", "slide_mode": "all",
            "hazards": [],
            "pieces": {"E": {"cells": [[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]],
                              "facing": "U", "species": "elephant"}}}
    board_io.validate_board_data(data)
    board = Board(data["rows"], data["cols"], data["pieces"], model="facing")
    moves = board.legal_moves()
    assert len(moves) == 1 and moves[0].result == "EXIT" and moves[0].direction == "U", moves
    return board, moves, {"expanded": 0}


def test_bomb_collision_budget_blocks_explosion():
    pieces = {
        "A": {"cells": [[1, 0], [1, 1]], "facing": "R", "species": "sheep"},
        "B": {"cells": [[1, 3], [1, 4]], "facing": "R", "species": "bomb",
              "hit_limit": 3, "hits_remaining": 2},
    }
    board = Board(3, 6, pieces, model="facing", slide_mode="all")
    bump = next(move for move in board.legal_moves() if move.piece_id == "A")
    after = board.apply(bump)
    assert after.pieces["B"]["hits_remaining"] == 1, after.pieces["B"]
    dangerous = Board(3, 6, {**pieces, "B": {**pieces["B"], "hits_remaining": 1}},
                      model="facing", slide_mode="all")
    assert all(move.piece_id != "A" for move in dangerous.legal_moves()), dangerous.legal_moves()
    return board, [bump], {"expanded": 0}


def test_moving_bomb_with_one_hit_cannot_collide():
    pieces = {
        "B": {"cells": [[0, 2], [1, 2]], "facing": "D", "species": "bomb",
              "hit_limit": 3, "hits_remaining": 1},
        "S": {"cells": [[3, 2], [4, 2]], "facing": "D", "species": "sheep"},
    }
    dangerous = Board(6, 5, pieces, model="facing", slide_mode="all")
    assert all(move.piece_id != "B" for move in dangerous.legal_moves()), dangerous.legal_moves()
    safe = Board(6, 5, {**pieces, "B": {**pieces["B"], "hits_remaining": 2}},
                 model="facing", slide_mode="all")
    move = next(move for move in safe.legal_moves() if move.piece_id == "B")
    after = safe.apply(move)
    assert after.pieces["B"]["hits_remaining"] == 1, after.pieces
    return safe, [move], {"expanded": 0}


def test_fence_blocks_animals_but_cattle_breaks_it():
    fence = [{"cell": [1, 0], "direction": "L"}]
    sheep = Board(4, 5, {
        "S": {"cells": [[1, 0], [1, 1]], "facing": "L", "species": "sheep"},
    }, model="facing", fences=fence)
    assert not sheep.legal_moves(), sheep.legal_moves()

    elephant = Board(4, 5, {
        "E": {"cells": [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
              "facing": "L", "species": "elephant"},
    }, model="facing", fences=fence)
    assert not elephant.legal_moves(), elephant.legal_moves()

    cattle = Board(4, 5, {
        "C": {"cells": [[1, 0], [1, 1]], "facing": "L", "species": "cattle"},
    }, model="facing", fences=fence)
    move = next(item for item in cattle.legal_moves() if item.result == "EXIT")
    after = cattle.apply(move)
    assert not after.pieces and not after.fences, (after.pieces, after.fences)
    return cattle, [move], {"expanded": 0}


def test_internal_fence_cells_block_animals_and_are_destroyed_by_cattle():
    fence = [{"cell": [1, c], "direction": "H"} for c in range(3, 6)]
    sheep = Board(3, 8, {
        "S": {"cells": [[1, 0], [1, 1]], "facing": "R", "species": "sheep"},
    }, model="facing", fences=fence)
    move = next(item for item in sheep.legal_moves() if item.piece_id == "S")
    assert move.result == "MOVE" and move.distance == 1, move
    stopped = sheep.apply(move)
    assert stopped.pieces["S"]["cells"] == frozenset({(1, 1), (1, 2)})
    assert stopped.fences == sheep.fences

    cattle = Board(3, 8, {
        "C": {"cells": [[1, 0], [1, 1]], "facing": "R", "species": "cattle"},
        # This animal is behind the rail.  It must not make the charging cow
        # stop on the board after the cow has reached the cattle-only exit.
        "S": {"cells": [[1, 6], [1, 7]], "facing": "R", "species": "sheep"},
    }, model="facing", fences=fence)
    charge = next(item for item in cattle.legal_moves() if item.piece_id == "C")
    assert charge.result == "EXIT", charge
    after = cattle.apply(charge)
    assert set(after.pieces) == {"S"} and not after.fences, (after.pieces, after.fences)
    return cattle, [charge], {"expanded": 0}


def test_level116_cattle_exits_when_smashing_internal_fence():
    path = (Path(__file__).resolve().parents[1] / "cache" / "levels" /
            "7223be2b0c3d08d9" /
            "7223be2b0c3d08d9-left082-cap0001-86198ddd" / "board.json")
    if not path.exists():
        return Board(1, 1, {}, model="facing"), [], {"expanded": 0}
    data = json.loads(path.read_text(encoding="utf-8"))
    cattle_id = next(pid for pid, piece in data["pieces"].items()
                     if piece.get("species") == "cattle")
    # Preserve the level-116 cattle/fence geometry and an animal behind the
    # rail, but remove unrelated animals that must leave earlier in the plan.
    pieces = {pid: piece for pid, piece in data["pieces"].items()
              if pid == cattle_id or pid == "10"}
    board = Board(data["rows"], data["cols"], pieces,
                  model=data["model"], slide_mode=data["slide_mode"],
                  hazards=data.get("hazards"), fences=data.get("fences"))
    move = next(item for item in board.legal_moves() if item.piece_id == cattle_id)
    assert move.result == "EXIT", move
    after = board.apply(move)
    assert cattle_id not in after.pieces and not after.fences, (after.pieces, after.fences)
    return board, [move], {"expanded": 0}


def test_black_sheep_stops_at_animal_and_changes_layout():
    board = Board(3, 8, {
        "K": {"cells": [[1, 0], [1, 1]], "facing": "R", "species": "black_sheep"},
        "S": {"cells": [[1, 5], [1, 6]], "facing": "R", "species": "sheep"},
    }, model="facing", slide_mode="all")
    move = next(item for item in board.legal_moves() if item.piece_id == "K")
    assert move.result == "MOVE" and move.distance == 3, move
    after = board.apply(move)
    assert after.pieces["K"]["cells"] == frozenset({(1, 3), (1, 4)}), after.pieces["K"]
    assert not after.returning, after.returning
    return board, [move], {"expanded": 0}


def test_black_sheep_fixed_collision_is_temporary_borrow():
    board = Board(3, 8, {
        "K": {"cells": [[1, 0], [1, 1]], "facing": "R", "species": "black_sheep"},
        "E": {"cells": [[0, 6], [0, 7]], "facing": "U", "species": "sheep"},
    }, model="facing", slide_mode="all", hazards=[[1, 5]])
    bounce = next(item for item in board.legal_moves() if item.piece_id == "K")
    assert bounce.result == "BOUNCE" and bounce.distance == 3, bounce
    borrowed = board.apply(bounce)
    assert "K" not in borrowed.pieces and "K" in borrowed.returning, borrowed.returning
    assert borrowed.remaining_count() == 2 and not borrowed.is_solved(), borrowed.remaining_count()
    exit_move = next(item for item in borrowed.legal_moves() if item.piece_id == "E")
    restored = borrowed.apply(exit_move)
    assert "K" in restored.pieces and not restored.returning, restored.pieces
    assert restored.pieces["K"]["cells"] == frozenset({(1, 0), (1, 1)})
    return board, [bounce, exit_move], {"expanded": 0}


def test_pink_sheep_exit_carries_touching_neighbors():
    board = Board(6, 7, {
        "P": {"cells": [[2, 1], [2, 0]], "facing": "L", "species": "pink_sheep"},
        "edge": {"cells": [[1, 1], [1, 2]], "facing": "R", "species": "sheep"},
        "corner": {"cells": [[3, 2], [4, 2]], "facing": "D", "species": "sheep"},
        "far": {"cells": [[4, 5], [4, 6]], "facing": "R", "species": "sheep"},
    }, model="facing", slide_mode="all")
    move = next(item for item in board.legal_moves() if item.piece_id == "P")
    assert move.result == "EXIT", move
    after = board.apply(move)
    assert set(after.pieces) == {"far"}, after.pieces
    return board, [move], {"expanded": 0}


def test_pink_sheep_move_does_not_carry_neighbors():
    board = Board(5, 8, {
        "P": {"cells": [[2, 1], [2, 2]], "facing": "R", "species": "pink_sheep"},
        "touching": {"cells": [[1, 1], [1, 2]], "facing": "L", "species": "sheep"},
        "blocker": {"cells": [[2, 6], [2, 7]], "facing": "R", "species": "sheep"},
    }, model="facing", slide_mode="all")
    move = next(item for item in board.legal_moves() if item.piece_id == "P")
    assert move.result == "MOVE", move
    after = board.apply(move)
    assert set(after.pieces) == {"P", "touching", "blocker"}, after.pieces
    return board, [move], {"expanded": 0}


def test_sleeping_pig_wakes_after_collision_then_becomes_clickable():
    board = Board(3, 8, {
        "S": {"cells": [[1, 0], [1, 1]], "facing": "R", "species": "sheep"},
        "P": {"cells": [[1, 5], [1, 6]], "facing": "R", "species": "pig",
              "awake": False},
    }, model="facing", slide_mode="all")
    assert all(move.piece_id != "P" for move in board.legal_moves()), board.legal_moves()
    collision = next(move for move in board.legal_moves() if move.piece_id == "S")
    assert collision.result == "MOVE" and collision.distance == 3, collision
    after = board.apply(collision)
    assert after.pieces["P"]["awake"], after.pieces["P"]
    pig_move = next(move for move in after.legal_moves() if move.piece_id == "P")
    assert pig_move.result == "EXIT" and pig_move.direction == "R", pig_move
    return board, [collision, pig_move], {"expanded": 0}


def test_pig_awake_state_changes_board_key():
    pieces = {"P": {"cells": [[1, 1], [1, 2]], "facing": "R", "species": "pig"}}
    sleeping = Board(3, 5, {"P": {**pieces["P"], "awake": False}}, model="facing")
    awake = Board(3, 5, {"P": {**pieces["P"], "awake": True}}, model="facing")
    assert sleeping.key() != awake.key(), (sleeping.key(), awake.key())
    return sleeping, [], {"expanded": 0}


def test_goat_advances_exactly_three_cells_per_click_until_exit():
    board = Board(3, 10, {
        "G": {"cells": [[1, 0], [1, 1]], "facing": "R", "species": "goat"},
    }, model="facing", slide_mode="all")
    first = next(move for move in board.legal_moves() if move.piece_id == "G")
    assert first.result == "STEP" and first.distance == 3, first
    after_first = board.apply(first)
    assert after_first.pieces["G"]["cells"] == frozenset({(1, 3), (1, 4)})
    second = next(move for move in after_first.legal_moves() if move.piece_id == "G")
    assert second.result == "STEP" and second.distance == 3, second
    after_second = after_first.apply(second)
    assert after_second.pieces["G"]["cells"] == frozenset({(1, 6), (1, 7)})
    third = next(move for move in after_second.legal_moves() if move.piece_id == "G")
    assert third.result == "EXIT", third
    assert after_second.apply(third).is_solved()

    blocked = Board(3, 8, {
        "G": {"cells": [[1, 0], [1, 1]], "facing": "R", "species": "goat"},
        "B": {"cells": [[1, 4], [1, 5]], "facing": "R", "species": "sheep"},
    }, model="facing", slide_mode="all")
    collision = next(move for move in blocked.legal_moves() if move.piece_id == "G")
    assert collision.result == "MOVE" and collision.distance == 2, collision
    return board, [first, second, third], {"expanded": 0}


def test_exit_closure_is_disabled_for_pink_sheep():
    board = Board(4, 6, {
        "P": {"cells": [[1, 0], [1, 1]], "facing": "L", "species": "pink_sheep"},
        "S": {"cells": [[3, 4], [3, 5]], "facing": "R", "species": "sheep"},
    }, model="facing", slide_mode="all")
    moves, normalized = forced_exit_closure(board)
    assert not supports_forced_exit_closure(board)
    assert not moves and normalized.key() == board.key(), (moves, normalized.key())
    return board, [], {"expanded": 0}


def test_archived_level113_macro_beam_solves():
    path = (Path(__file__).resolve().parents[1] / "cache" / "levels" /
            "7614d5dd102dae13" / "executions" / "step-0001.json")
    if not path.exists():
        return Board(1, 1, {}, model="facing"), [], {"expanded": 0}
    data = json.loads(path.read_text(encoding="utf-8"))["board_before"]
    board = Board(data["rows"], data["cols"], data["pieces"],
                  model=data["model"], slide_mode=data["slide_mode"],
                  hazards=data.get("hazards"), fences=data.get("fences"),
                  returning=data.get("returning"))
    moves, info = beam_solve(
        board, width=14, max_depth=96, time_limit=5.0, seen_cap=120_000)
    assert info.get("solved"), info
    final = replay(board, moves)
    assert final.is_solved() and len(moves) <= 88, (len(moves), info)
    return Board(1, 1, {}, model="facing"), [], {"expanded": info.get("expanded", 0)}


def test_terminal_deadlock_is_ranked_behind_live_state():
    """A smaller frozen remainder must not beat a larger recoverable state."""
    from solver.search import _heuristic, _rank, analyze

    dead = Board(2, 4, {
        "left": {"cells": [[0, 0], [0, 1]], "facing": "R"},
        "right": {"cells": [[0, 2], [0, 3]], "facing": "L"},
    }, model="facing", slide_mode="all")
    live = Board(2, 5, {
        "a": {"cells": [[0, 0], [0, 1]], "facing": "R"},
        "b": {"cells": [[1, 0], [1, 1]], "facing": "R"},
        "c": {"cells": [[0, 3], [0, 4]], "facing": "R"},
    }, model="facing", slide_mode="all")
    assert analyze(dead)["terminal_deadlock"] == 1, analyze(dead)
    assert analyze(live)["terminal_deadlock"] == 0, analyze(live)
    assert _rank(live, 0) < _rank(dead, 0), (_rank(live, 0), _rank(dead, 0))
    assert _heuristic(live) < _heuristic(dead), (_heuristic(live), _heuristic(dead))
    return live, [], {"expanded": 0}


def test_archived_level119_randomized_macro_solves():
    path = (Path(__file__).resolve().parents[1] / "cache" / "manual_samples" /
            "20260714-213828-505" / "board.json")
    if not path.exists():
        return Board(1, 1, {}, model="facing"), [], {"expanded": 0}
    board = board_io.load(path)
    moves, info = randomized_macro_solve(board, seed=0, time_limit=4.0)
    assert info.get("solved"), info
    final = replay(board, moves)
    assert final.is_solved() and len(moves) <= 115, (len(moves), info)
    return Board(1, 1, {}, model="facing"), [], {"expanded": info.get("expanded", 0)}


def test_archived_level119_shared_planner_uses_search_portfolio():
    """The GUI/CLI planner must actually use the strategy that solves level 119."""
    path = (Path(__file__).resolve().parents[1] / "cache" / "manual_samples" /
            "20260714-213828-505" / "manual_board.json")
    if not path.exists():
        return Board(1, 1, {}, model="facing"), [], {"expanded": 0}
    board = board_io.load(path)
    result = planner.solve_board(board, timeout_s=6.0)
    moves = [move for move, _phase in result.steps]
    assert result.solved and result.remaining == 0, result.info
    assert "randomized-macro" in result.kind, result.kind
    assert replay(board, moves).is_solved()
    return Board(1, 1, {}, model="facing"), [], {
        "expanded": result.info.get("expanded", 0),
    }


if __name__ == "__main__":
    for name, fn in [("基础出口", test_basic_exit),
                     ("竖直堵塞", test_blocked_chain),
                     ("固定朝向", test_facing_model),
                      ("开局粗解普通羊直出", test_opening_coarse_only_drains_direct_ordinary_sheep),
                      ("快速解法最多三层", test_quick_exit_plan_stops_after_three_exposure_layers),
                      ("第172关分段止挡", test_exit_closure_preserves_level172_staged_stoppers),
                      ("状态键朝向", test_key_keeps_facing),
                     ("搜索清盘边界", test_search_one_hop_boundary),
                     ("固定朝向迎头结构死锁", test_opposing_facing_pair_is_reported_as_structural_deadlock),
                     ("双轴与危险格", test_axis_both_search_and_hazard_features),
                     ("2x3 大象", test_facing_elephant_2x3_can_move_and_exit),
                     ("炸弹碰撞预算", test_bomb_collision_budget_blocks_explosion),
                     ("移动炸弹末次碰撞保护", test_moving_bomb_with_one_hit_cannot_collide),
                     ("栅栏与牛破坏", test_fence_blocks_animals_but_cattle_breaks_it),
                     ("内部栅栏与牛冲破", test_internal_fence_cells_block_animals_and_are_destroyed_by_cattle),
                     ("第116关牛撞栏立即离场", test_level116_cattle_exits_when_smashing_internal_fence),
                     ("黑羊碰棋子后改变布局", test_black_sheep_stops_at_animal_and_changes_layout),
                     ("黑羊固定障碍临时借位", test_black_sheep_fixed_collision_is_temporary_borrow),
                     ("粉色羊带走相邻羊", test_pink_sheep_exit_carries_touching_neighbors),
                     ("粉色羊移动不带走邻居", test_pink_sheep_move_does_not_carry_neighbors),
                     ("睡猪碰撞后醒来", test_sleeping_pig_wakes_after_collision_then_becomes_clickable),
                     ("猪醒状态进入棋盘键", test_pig_awake_state_changes_board_key),
                     ("山羊每次固定前进三格", test_goat_advances_exactly_three_cells_per_click_until_exit),
                     ("粉羊禁用强制离场闭包", test_exit_closure_is_disabled_for_pink_sheep),
                     ("终局死锁排序", test_terminal_deadlock_is_ranked_behind_live_state),
                     ("第119关随机宏动作搜索", test_archived_level119_randomized_macro_solves),
                     ("第119关统一规划组合搜索", test_archived_level119_shared_planner_uses_search_portfolio),
                     ("第113关宏动作窄束搜索", test_archived_level113_macro_beam_solves)]:
        b, moves, info = fn()
        print(f"\n=== {name} ===")
        print(render(b))
        print(f"最优 {len(moves)} 步 (展开 {info['expanded']} 节点):")
        for i, mv in enumerate(moves, 1):
            print(f"  {i}. {describe(mv)}")
    print("\n全部测试通过 ✅")
