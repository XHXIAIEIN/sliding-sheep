"""棋盘 JSON 读写 + 文字可视化。"""
from __future__ import annotations

import json

from solver import Board, Move

_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
VALID_MODELS = {"axis_both", "facing"}
VALID_SLIDE_MODES = {"all", "any"}
VALID_FACING = {"U", "D", "L", "R"}
VALID_FENCE_DIRECTION = VALID_FACING | {"H", "V"}


class BoardValidationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("；".join(self.errors))


def _cell(value, label, errors):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        errors.append(f"{label} 不是 [row, col]")
        return None
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        errors.append(f"{label} 坐标必须是整数")
        return None
    return tuple(value)


def validate_board_data(data: dict) -> dict:
    """Validate every board before it reaches Board/search/cache execution."""
    errors = []
    if not isinstance(data, dict):
        raise BoardValidationError(["棋盘根节点必须是对象"])
    rows, cols = data.get("rows"), data.get("cols")
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 100:
        errors.append("rows 必须是 1..100 的整数")
    if not isinstance(cols, int) or isinstance(cols, bool) or not 1 <= cols <= 100:
        errors.append("cols 必须是 1..100 的整数")
    model = data.get("model", "axis_both")
    slide_mode = data.get("slide_mode", "all")
    if model not in VALID_MODELS:
        errors.append(f"未知 model: {model}")
    if slide_mode not in VALID_SLIDE_MODES:
        errors.append(f"未知 slide_mode: {slide_mode}")

    hazards = set()
    for index, raw in enumerate(data.get("hazards", [])):
        cell = _cell(raw, f"hazards[{index}]", errors)
        if cell is None:
            continue
        if isinstance(rows, int) and isinstance(cols, int) and not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
            errors.append(f"危险格越界: {cell}")
        if cell in hazards:
            errors.append(f"危险格重复: {cell}")
        hazards.add(cell)

    no_stop = set()
    for index, raw in enumerate(data.get("no_stop", [])):
        cell = _cell(raw, f"no_stop[{index}]", errors)
        if cell is None:
            continue
        if isinstance(rows, int) and isinstance(cols, int) and not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
            errors.append(f"禁停格越界: {cell}")
        if cell in no_stop:
            errors.append(f"禁停格重复: {cell}")
        no_stop.add(cell)

    fences = set()
    for index, raw in enumerate(data.get("fences", [])):
        if not isinstance(raw, dict):
            errors.append(f"fences[{index}] 必须是对象")
            continue
        cell = _cell(raw.get("cell"), f"fences[{index}].cell", errors)
        direction = str(raw.get("direction") or "")
        if cell is None or direction not in VALID_FENCE_DIRECTION:
            if direction not in VALID_FENCE_DIRECTION:
                errors.append(f"fences[{index}] 方向非法: {direction}")
            continue
        if isinstance(rows, int) and isinstance(cols, int):
            r, c = cell
            in_bounds = 0 <= r < rows and 0 <= c < cols
            if not in_bounds:
                errors.append(f"栅栏格越界: {cell}")
            internal = direction in {"H", "V"} and 0 <= r < rows and 0 <= c < cols
            outward = ((direction == "L" and c == 0) or
                       (direction == "R" and c == cols - 1) or
                       (direction == "U" and r == 0) or
                       (direction == "D" and r == rows - 1))
            if in_bounds and not internal and not outward:
                errors.append(f"栅栏必须位于对应棋盘边界: {cell} {direction}")
        key = (*cell, direction)
        if key in fences:
            errors.append(f"栅栏重复: {cell} {direction}")
        fences.add(key)

    pieces = data.get("pieces")
    if not isinstance(pieces, dict):
        errors.append("pieces 必须是对象")
        pieces = {}
    occupied = {}
    for raw_pid, piece in pieces.items():
        pid = str(raw_pid)
        if not isinstance(piece, dict):
            errors.append(f"棋子 {pid} 必须是对象")
            continue
        raw_cells = piece.get("cells")
        if not isinstance(raw_cells, list) or not raw_cells:
            errors.append(f"棋子 {pid} 缺少 cells")
            continue
        cells = []
        for index, raw in enumerate(raw_cells):
            cell = _cell(raw, f"棋子 {pid}.cells[{index}]", errors)
            if cell is None:
                continue
            if isinstance(rows, int) and isinstance(cols, int) and not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
                errors.append(f"棋子 {pid} 越界: {cell}")
            cells.append(cell)
            if cell in occupied:
                errors.append(f"棋子重叠: {occupied[cell]} 与 {pid} 占用 {cell}")
            occupied[cell] = pid
        if len(set(cells)) != len(cells):
            errors.append(f"棋子 {pid} 内部有重复格")
        if cells:
            row_set = {r for r, _ in cells}
            col_set = {c for _, c in cells}
            species = piece.get("species", "sheep")
            elephant_2x3 = (species == "elephant" and len(cells) == 6
                            and {len(row_set), len(col_set)} == {2, 3}
                            and set(cells) == {(r, c) for r in row_set for c in col_set})
            straight = len(row_set) == 1 or len(col_set) == 1
            if species == "elephant" and not elephant_2x3:
                errors.append(f"大象 {pid} 必须占用连续 2×3 六格")
            elif not straight and not elephant_2x3:
                errors.append(f"棋子 {pid} 不是直线")
            elif straight and len(cells) > 1:
                values = sorted(c for _, c in cells) if len(row_set) == 1 else sorted(r for r, _ in cells)
                if values != list(range(values[0], values[-1] + 1)):
                    errors.append(f"棋子 {pid} 的格子不连续")
        facing = piece.get("facing")
        if facing is not None and facing not in VALID_FACING:
            errors.append(f"棋子 {pid} 朝向非法: {facing}")
        if model == "facing" and facing not in VALID_FACING:
            errors.append(f"facing 模式下棋子 {pid} 必须有合法朝向")
        hit_limit = piece.get("hit_limit")
        hits_remaining = piece.get("hits_remaining")
        if piece.get("species") == "pig" and not isinstance(piece.get("awake"), bool):
            errors.append(f"猪 {pid} 缺少布尔 awake 状态")
        if piece.get("species") == "bomb":
            if not isinstance(hit_limit, int) or hit_limit < 1:
                errors.append(f"炸弹羊 {pid} 缺少合法 hit_limit")
            if (not isinstance(hits_remaining, int) or not isinstance(hit_limit, int)
                    or not 1 <= hits_remaining <= hit_limit):
                errors.append(f"炸弹羊 {pid} 的 hits_remaining 非法")
        if cells and facing in VALID_FACING and piece.get("species") == "elephant":
            row_count = len({r for r, _ in cells})
            col_count = len({c for _, c in cells})
            if ((facing in {"L", "R"} and col_count != 3)
                    or (facing in {"U", "D"} and row_count != 3)):
                errors.append(f"大象 {pid} 的 3 格长边必须与朝向一致")
        elif cells and facing in VALID_FACING and len(cells) > 1:
            horizontal = len({r for r, _ in cells}) == 1
            if (horizontal and facing not in {"L", "R"}) or (not horizontal and facing not in {"U", "D"}):
                errors.append(f"棋子 {pid} 朝向与轴线不一致")

    for cell in sorted(hazards & set(occupied)):
        errors.append(f"危险格与棋子 {occupied[cell]} 重叠: {cell}")
    if errors:
        raise BoardValidationError(errors)
    return data


def load(path) -> Board:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_board_data(data)
    return Board(
        rows=data["rows"],
        cols=data["cols"],
        pieces=data["pieces"],
        model=data.get("model", "axis_both"),
        slide_mode=data.get("slide_mode", "all"),
        hazards=data.get("hazards", []),
        fences=data.get("fences", []),
        returning=data.get("returning", {}),
        no_stop=data.get("no_stop", []),
    )


def dump(board: Board, path):
    data = {
        "rows": board.rows,
        "cols": board.cols,
        "model": board.model,
        "slide_mode": board.slide_mode,
        "hazards": [list(c) for c in sorted(getattr(board, "hazards", []))],
        "no_stop": [list(c) for c in sorted(getattr(board, "no_stop", []))],
        "fences": [{"cell": [r, c], "direction": direction}
                   for r, c, direction in sorted(getattr(board, "fences", []))],
        "returning": {pid: {"cells": [list(c) for c in sorted(p["cells"])],
                            "facing": p.get("facing"),
                            "species": p.get("species", "black_sheep"),
                            **({"awake": bool(p.get("awake", True))}
                               if p.get("species") == "pig" else {})}
                      for pid, p in getattr(board, "returning", {}).items()},
        "pieces": {pid: {"cells": [list(c) for c in sorted(p["cells"])],
                         **({"facing": p["facing"]} if p["facing"] else {}),
                         "species": p.get("species", "sheep"),
                         **({"awake": bool(p.get("awake", True))}
                            if p.get("species") == "pig" else {}),
                         **({"hit_limit": p.get("hit_limit", 3),
                             "hits_remaining": p.get("hits_remaining", 3)}
                            if p.get("species") == "bomb" else {})}
                   for pid, p in board.pieces.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def render(board: Board) -> str:
    """每只羊用一个字符填充其格子，'.' 为空格。"""
    grid = [["." for _ in range(board.cols)] for _ in range(board.rows)]
    for r, c, direction in getattr(board, "fences", []):
        if direction in {"H", "V"}:
            grid[r][c] = "=" if direction == "H" else "|"
    for i, (pid, p) in enumerate(sorted(board.pieces.items())):
        ch = _CHARS[i % len(_CHARS)]
        for r, c in p["cells"]:
            grid[r][c] = ch
    lines = [" " + " ".join(f"{c}" for c in range(board.cols))]
    for r in range(board.rows):
        lines.append(f"{r} " + " ".join(grid[r]))
    return "\n".join(lines)


def describe(mv: Move) -> str:
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}[mv.direction]
    if mv.result == "EXIT":
        return f"点击羊[{mv.piece_id}]@{tuple(mv.anchor)} {arrow} 赶出去"
    if mv.result == "BOUNCE":
        return f"点击黑羊[{mv.piece_id}]@{tuple(mv.anchor)} {arrow} 临时借位后回弹"
    return f"点击羊[{mv.piece_id}]@{tuple(mv.anchor)} {arrow} 滑动{mv.distance}格"
