"""读 board.json -> 求解 -> 打印点击序列 + 基于 images/_occ_axis_rect.png 渲染 images/_solution.png。
小盘(<=14 只)用 A* 求最优；大盘用 weighted A*/beam 搜索，贪心兜底。
运行: py scripts/solve_board.py [board.json]
"""
import sys
import cv2, numpy as np
import board_io
import level_cache
import planner
from solver import Move
import vision as D
from paths import image_path


def _draw_step_badge(img, center, text):
    x, y = map(int, center)
    radius = 13 if len(text) < 2 else 15
    color = (58, 76, 200)
    layer = img.copy()
    cv2.circle(layer, (x + 1, y + 2), radius + 4, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.18, img, 0.82, 0, img)
    layer = img.copy()
    cv2.circle(layer, (x, y), radius + 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(layer, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.88, img, 0.12, 0, img)
    scale = 0.42 if len(text) < 2 else 0.36
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(img, text, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)


def _board_data(board):
    return {
        "rows": board.rows,
        "cols": board.cols,
        "model": board.model,
        "slide_mode": board.slide_mode,
        "hazards": [list(rc) for rc in sorted(board.hazards)],
        "fences": [{"cell": [r, c], "direction": direction}
                   for r, c, direction in sorted(getattr(board, "fences", []))],
        "returning": {
            str(pid): {"cells": [list(rc) for rc in sorted(piece["cells"])],
                       "facing": piece.get("facing"),
                       "species": piece.get("species", "black_sheep")}
            for pid, piece in getattr(board, "returning", {}).items()
        },
        "pieces": {
            str(pid): {"cells": [list(rc) for rc in sorted(p["cells"])],
                       "facing": p.get("facing"),
                       "species": p.get("species", "sheep"),
                       **({"awake": bool(p.get("awake", True))}
                          if p.get("species") == "pig" else {})}
            for pid, p in board.pieces.items()
        },
    }


def _moves_to_records(moves):
    return [{
        "piece": str(mv.piece_id),
        "direction": mv.direction,
        "anchor": list(mv.anchor),
        "result": mv.result,
        "distance": int(mv.distance),
        "phase": "cli",
    } for mv in moves]


def _records_to_moves(board, records):
    moves = []
    cur = board
    for rec in records or []:
        mv = Move(str(rec["piece"]), rec["direction"], tuple(rec["anchor"]),
                  rec["result"], int(rec.get("distance", 0)))
        if not any(str(m.piece_id) == str(mv.piece_id)
                   and m.direction == mv.direction
                   and m.result == mv.result
                   and m.distance == mv.distance
                   for m in cur.legal_moves()):
            raise RuntimeError(f"缓存动作失效: {mv.piece_id} {mv.direction} {mv.result}")
        moves.append(mv)
        cur = cur.apply(mv)
    return moves, cur


def main(path=None):
    path = path or (sys.argv[1] if len(sys.argv) > 1 else "board.json")
    board = board_io.load(path)
    board_data = _board_data(board)
    n = board.remaining_count()
    print(f"棋盘 {board.rows}x{board.cols}  羊 {n} 只  模型 {board.model}")

    cached = level_cache.load_solution(board_data, require_complete=True)
    if cached:
        try:
            moves, cached_final = _records_to_moves(board, cached.get("moves", []))
            if cached.get("solved") and not cached_final.is_solved():
                raise RuntimeError(
                    f"缓存标记已完成，但回放后仍剩 {cached_final.remaining_count()} 只"
                )
            info = {"solved": cached.get("solved"), "remaining": cached.get("remaining", 0)}
            kind = f"缓存命中 · {cached.get('kind', 'cache')}"
            print(f"cache hit {cached.get('_cache_path')}")
        except Exception as e:
            print(f"cache invalid，重新求解: {e}")
            invalidated = level_cache.invalidate_solution(board_data, cached, str(e))
            print(f"cache invalidated {invalidated} copies")
            cached = None
    if not cached:
        planned = planner.solve_board(board, timeout_s=12.0)
        moves = [move for move, _phase in planned.steps]
        info = planned.info
        kind = planned.kind
    else:
        pass
    if moves is None:
        print(f"[{kind}] 无解: {info}")
        return 3
    status = "全部清空 ✅" if info.get("solved", True) else f"⚠️ 卡住，剩 {info['remaining']} 只未清"
    print(f"[{kind}] {len(moves)} 步  {status}")
    for i, mv in enumerate(moves, 1):
        print(f"  {i:3d}. {board_io.describe(mv)}")

    if not cached:
        remaining = int(info.get("remaining", 0 if info.get("solved", True)
                                 else board.remaining_count()))
        suspicious = (not bool(info.get("solved", True))) and remaining > 0 and len(moves) == 0
        level_cache.save_solution(
            board_data,
            {
                "kind": kind,
                "solved": bool(info.get("solved", True)),
                "remaining": remaining,
                "result_type": "complete_solution" if info.get("solved", True) else
                               "timeout" if info.get("timeout") else
                               "partial_hint" if moves else "proven_unsat",
                "timeout": bool(info.get("timeout", False)),
                "timeout_ms": 0,
                "usable": not suspicious,
                "suspicious": suspicious,
                "suspicion": {
                    "type": "dead_end",
                    "message": "剩余羊但没有任何可执行步骤，疑似识别或朝向错误",
                } if suspicious else None,
                "coarse_total": 0,
                "refine_total": len(moves),
                "moves": _moves_to_records(moves),
            },
            source="cli-solver",
        )
        print("cache stored solution")

    # 渲染点击顺序到正视 2D 棋盘图。
    try:
        ov = cv2.imread(str(image_path("_occ_axis_rect.png")))
        if ov is None:
            raise RuntimeError("缺 images/_occ_axis_rect.png，请先运行 scripts/detect_occupancy.py")

        def to_px(r, c):
            return int((c + 0.5) * D.CELL), int((r + 0.5) * D.CELL)

        for i, mv in enumerate(moves, 1):
            r, c = mv.anchor
            _draw_step_badge(ov, to_px(r, c), str(i))
        cv2.imwrite(str(image_path("_solution.png")), ov)
        print("saved images/_solution.png (2D棋盘数字=点击顺序)")
    except Exception as e:
        print(f"(渲染 images/_solution.png 跳过: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
