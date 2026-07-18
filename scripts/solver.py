"""羊停车场（Rush Hour / 华容道式）求解器内核。

棋盘：rows x cols 网格，左上为 (0,0)，r 向下、c 向右。
每只羊是一块占若干连续格子的"棋子"，沿其轴线滑动；滑到棋盘边缘即被赶出（EXIT），
清空所有羊即过关。

移动模型（board['model']）：
  - "axis_both": 沿轴线两个方向都能推（水平棋子可左右、竖直棋子可上下）。华容道经典。
  - "facing":    每只羊有固定朝向 facing（'U'/'D'/'L'/'R'），只能朝头的方向推。

一次"点击/tap"= 把这只羊朝某方向滑动：
  - 若该方向到边缘的通道全空 -> 羊被赶出棋盘（EXIT）。
  - 否则滑到紧贴前方阻挡的羊为止（slide_mode='all'），或可停在任意中途格（slide_mode='any'）。

本模块零第三方依赖，纯 stdlib，便于独立验证；识别/点击在别的模块里接。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


@dataclass(frozen=True)
class Move:
    """一步解：点击某只羊（用其参考格定位）并朝 direction 推。"""
    piece_id: str
    direction: str          # 'U'/'D'/'L'/'R'
    anchor: tuple           # 被点击羊在动作前的一个参考格 (r,c)，供识别/点击定位
    result: str             # 'EXIT'、'MOVE'(撞停) 或 'STEP'(定距前进)
    distance: int           # MOVE/STEP 时移动格数；EXIT 时为 0


def _axis(cells):
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1 and len(cols) >= 1 and len(cols) >= len(rows):
        return "H" if len(cols) > 1 else "S"  # 单格记为 S
    if len(cols) == 1 and len(rows) > 1:
        return "V"
    if len(rows) == 1 and len(cols) == 1:
        return "S"
    raise ValueError(f"棋子不是直线: {cells}")


def _allowed_dirs(cells, model, facing, species="sheep", awake=True):
    if species == "pig" and not awake:
        return []
    if model == "facing":
        if not facing:
            raise ValueError("facing 模型要求每只羊带 facing 字段")
        return [facing]
    ax = _axis(cells)
    if ax == "H":
        return ["L", "R"]
    if ax == "V":
        return ["U", "D"]
    return ["U", "D", "L", "R"]  # 单格


class Board:
    def __init__(self, rows, cols, pieces, model="axis_both", slide_mode="all", hazards=None,
                 fences=None, returning=None, no_stop=None):
        self.rows = rows
        self.cols = cols
        self.model = model
        self.slide_mode = slide_mode
        self.hazards = frozenset(tuple(x) for x in (hazards or []))
        # Dynamic danger zones (for example a wolf patrol corridor) may be
        # crossed under a live preflight, but a non-exit move may never finish
        # on them.  They are deliberately separate from solid hazards.
        self.no_stop = frozenset(tuple(x) for x in (no_stop or []))
        self.fences = frozenset(
            (int(item["cell"][0]), int(item["cell"][1]), str(item["direction"]))
            if isinstance(item, dict) else (int(item[0]), int(item[1]), str(item[2]))
            for item in (fences or []))
        # pieces: dict id -> {'cells': frozenset((r,c)), 'facing': dir|None}
        self.pieces = {}
        for pid, p in pieces.items():
            cells = frozenset(tuple(x) for x in p["cells"])
            self.pieces[pid] = {"cells": cells, "facing": p.get("facing"),
                                "species": p.get("species", "sheep"),
                                "awake": bool(p.get("awake", True)),
                                "hit_limit": p.get("hit_limit"),
                                "hits_remaining": p.get("hits_remaining")}
        self.returning = {
            str(pid): {"cells": frozenset(tuple(x) for x in p["cells"]),
                       "facing": p.get("facing"), "species": p.get("species", "black_sheep"),
                       "hit_limit": p.get("hit_limit"), "hits_remaining": p.get("hits_remaining")}
            for pid, p in (returning or {}).items()
        }

    # ---- 基础查询 ----
    def occupied(self, exclude=None):
        occ = set(self.hazards)
        for pid, p in self.pieces.items():
            if pid == exclude:
                continue
            occ |= p["cells"]
        return occ

    def in_board(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_solved(self):
        return not self.pieces and not self.returning

    def remaining_count(self):
        return len(self.pieces) + len(self.returning)

    def key(self):
        """规范化状态键：忽略 id 标签，但保留形状和朝向。

        面向出口的 puzzle 里，不同朝向的羊交换位置并不等价；只看占格会把
        后续可行动作不同的状态合并掉，搜索会误剪枝。
        """
        returning_key = frozenset((p["facing"], p["cells"], p.get("species", "black_sheep"))
                                  for p in self.returning.values())
        return (self.hazards, self.no_stop, self.fences, returning_key,
                frozenset((p["facing"], p["cells"], p.get("species", "sheep"),
                                         p.get("awake", True), p.get("hits_remaining"))
                          for p in self.pieces.values()))

    def fence_crossings(self, frontier, direction):
        """Boundary fence segments crossed while leaving the board."""
        dr, dc = DIRS[direction]
        return frozenset(
            (r, c, direction) for r, c in frontier
            if not self.in_board(r + dr, c + dc) and (r, c, direction) in self.fences)

    def fence_cell_hits(self, frontier, direction):
        """Internal H/V fence cells entered by the next one-cell advance."""
        dr, dc = DIRS[direction]
        internal = {(r, c) for r, c, kind in self.fences if kind in {"H", "V"}}
        return frozenset(
            (r, c, kind) for r, c in ((r + dr, c + dc) for r, c in frontier)
            for kind in ("H", "V") if (r, c) in internal and (r, c, kind) in self.fences)

    def internal_fence_run(self, hits):
        """Return the complete connected H/V rail touched by a cattle charge."""
        pending = list(hits)
        connected = set(hits)
        while pending:
            r, c, kind = pending.pop()
            neighbours = (((r, c - 1, kind), (r, c + 1, kind)) if kind == "H"
                          else ((r - 1, c, kind), (r + 1, c, kind)))
            for item in neighbours:
                if item in self.fences and item not in connected:
                    connected.add(item)
                    pending.append(item)
        return frozenset(connected)

    def _collision_is_safe(self, moving_pid, frontier, direction):
        dr, dc = DIRS[direction]
        nxt = {(r + dr, c + dc) for r, c in frontier}
        owners = {cell: pid for pid, piece in self.pieces.items()
                  if pid != moving_pid for cell in piece["cells"]}
        affected = {owners[cell] for cell in nxt if cell in owners}
        if self.pieces[moving_pid].get("species") == "bomb":
            affected.add(moving_pid)
        return all(self.pieces[pid].get("species") != "bomb"
                   or int(self.pieces[pid].get("hits_remaining") or 3) > 1
                   for pid in affected)

    def _touching_piece_ids(self, moving_pid, cells):
        """Pieces touching the mover by an edge or corner in its current position."""
        touching = set()
        for pid, piece in self.pieces.items():
            if pid == moving_pid:
                continue
            if any(max(abs(r - other_r), abs(c - other_c)) <= 1
                   for r, c in cells for other_r, other_c in piece["cells"]):
                touching.add(pid)
        return touching

    # ---- 动作生成 ----
    def legal_moves(self):
        moves = []
        for pid, p in self.pieces.items():
            cells = p["cells"]
            others = self.occupied(exclude=pid)
            piece_occupied = {cell for other_pid, piece in self.pieces.items()
                              if other_pid != pid for cell in piece["cells"]}
            for d in _allowed_dirs(cells, self.model, p["facing"],
                                   p.get("species", "sheep"), p.get("awake", True)):
                dr, dc = DIRS[d]
                # 沿 d 推进：整块一起走，只需看"前缘"路径
                steps = 0
                exited = False
                blocked_by_piece = False
                blocked_by_fixed = False
                capped_step = False
                step_limit = 3 if p.get("species") == "goat" else None
                frontier = cells
                while True:
                    nxt = frozenset((r + dr, c + dc) for r, c in frontier)
                    out = [(r, c) for r, c in nxt if not self.in_board(r, c)]
                    if out:
                        # 栅栏位于边界外沿；只有牛能撞破对应栅栏并离场。
                        if (self.fence_crossings(frontier, d)
                                and p.get("species", "sheep") != "cattle"):
                            blocked_by_fixed = True
                            break
                        exited = True
                        break
                    internal_fence_hits = self.fence_cell_hits(frontier, d)
                    if internal_fence_hits:
                        if p.get("species", "sheep") == "cattle":
                            # An internal rail is a cattle-only exit: the cow
                            # smashes the rail and leaves in the same action.
                            exited = True
                        else:
                            blocked_by_fixed = True
                        break
                    if nxt & others:
                        blocked_by_piece = bool(nxt & piece_occupied)
                        blocked_by_fixed = bool(nxt & self.hazards)
                        break
                    steps += 1
                    frontier = nxt
                    if step_limit is not None and steps >= step_limit:
                        capped_step = True
                        break
                anchor = min(cells)  # 稳定参考格
                if exited:
                    moves.append(Move(pid, d, anchor, "EXIT", 0))
                elif steps > 0:
                    if capped_step:
                        if frontier & self.no_stop:
                            continue
                        moves.append(Move(pid, d, anchor, "STEP", steps))
                        continue
                    # Black sheep rebound to their origin after any collision;
                    # animal collisions are the exception: they stop normally
                    # and permanently change the layout.
                    if p.get("species") == "black_sheep":
                        if blocked_by_fixed and not blocked_by_piece and not self.returning:
                            moves.append(Move(pid, d, anchor, "BOUNCE", steps))
                            continue
                    if not self._collision_is_safe(pid, frontier, d):
                        continue
                    if self.slide_mode == "all":
                        if frontier & self.no_stop:
                            continue
                        moves.append(Move(pid, d, anchor, "MOVE", steps))
                    else:
                        for k in range(1, steps + 1):
                            landing = frozenset((r + dr * k, c + dc * k) for r, c in cells)
                            if not landing & self.no_stop:
                                moves.append(Move(pid, d, anchor, "MOVE", k))
        if self.returning:
            valid = []
            for move in moves:
                try:
                    self.apply(move)
                    valid.append(move)
                except ValueError:
                    pass
            moves = valid
        return moves

    def apply(self, mv: Move):
        """返回应用 mv 后的新 Board（不改原对象）。"""
        new_pieces = {pid: {"cells": p["cells"], "facing": p["facing"],
                            "species": p.get("species", "sheep"),
                            "awake": p.get("awake", True),
                            "hit_limit": p.get("hit_limit"),
                            "hits_remaining": p.get("hits_remaining")}
                      for pid, p in self.pieces.items()}
        new_returning = {pid: dict(piece) for pid, piece in self.returning.items()}
        new_fences = set(self.fences)
        moving = new_pieces.get(mv.piece_id)
        if moving and moving.get("species") == "cattle" and mv.result == "EXIT":
            frontier = moving["cells"]
            dr, dc = DIRS[mv.direction]
            while True:
                internal_hits = self.fence_cell_hits(frontier, mv.direction)
                if internal_hits:
                    new_fences.difference_update(self.internal_fence_run(internal_hits))
                    break
                nxt = frozenset((r + dr, c + dc) for r, c in frontier)
                if any(not self.in_board(r, c) for r, c in nxt):
                    new_fences.difference_update(self.fence_crossings(frontier, mv.direction))
                    break
                frontier = nxt
        if mv.result == "BOUNCE":
            new_returning[mv.piece_id] = dict(new_pieces[mv.piece_id])
            del new_pieces[mv.piece_id]
        elif mv.result == "EXIT":
            moving = new_pieces[mv.piece_id]
            carried = (self._touching_piece_ids(mv.piece_id, moving["cells"])
                       if moving.get("species") == "pink_sheep" else set())
            del new_pieces[mv.piece_id]
            for pid in carried:
                new_pieces.pop(pid, None)
        else:
            dr, dc = DIRS[mv.direction]
            k = mv.distance
            old = new_pieces[mv.piece_id]["cells"]
            moved = frozenset((r + dr * k, c + dc * k) for r, c in old)
            new_pieces[mv.piece_id]["cells"] = moved
            if mv.result == "MOVE":
                next_cells = {(r + dr, c + dc) for r, c in moved}
                owners = {cell: pid for pid, piece in new_pieces.items()
                          if pid != mv.piece_id for cell in piece["cells"]}
                affected = {owners[cell] for cell in next_cells if cell in owners}
                if new_pieces[mv.piece_id].get("species") == "bomb":
                    affected.add(mv.piece_id)
                for pid in affected:
                    if (new_pieces[pid].get("species") == "pig"
                            and not new_pieces[pid].get("awake", True)):
                        new_pieces[pid]["awake"] = True
                    if new_pieces[pid].get("species") == "bomb":
                        new_pieces[pid]["hits_remaining"] = int(
                            new_pieces[pid].get("hits_remaining") or 3) - 1
        if mv.result != "BOUNCE" and new_returning:
            occupied = set(self.hazards)
            occupied.update(cell for piece in new_pieces.values() for cell in piece["cells"])
            for pid, piece in new_returning.items():
                if piece["cells"] & occupied:
                    raise ValueError("临时借位后黑羊无法返回原位")
                new_pieces[pid] = piece
                occupied.update(piece["cells"])
            new_returning = {}
        b = Board(self.rows, self.cols, {}, self.model, self.slide_mode,
                  hazards=self.hazards, fences=new_fences, returning=new_returning,
                  no_stop=self.no_stop)
        b.pieces = new_pieces
        b.returning = new_returning
        return b


def greedy_solve(board: Board, lookahead=1, max_steps=10_000, cancel=None):
    """大盘(几十只)best-effort 规划：A* 状态空间爆炸时用它。
    策略：能直接赶出去的羊优先点；都不能时，选一步滑动使"之后能出栏的羊数"最大(贪心解锁)。
    返回 (moves, info)；info['solved'] 标记是否清空，未清空给 remaining。
    注意：贪心不保证最优、也不保证一定能解开(死锁会停)；仅作流水线兜底。"""
    def features(b: Board):
        owners = {}
        for opid, piece in b.pieces.items():
            for cell in piece["cells"]:
                owners[cell] = opid
        exits = movable = blockers = stuck = 0
        for opid, piece in b.pieces.items():
            cells = piece["cells"]
            facing = piece["facing"]
            if piece.get("species") == "pig" and not piece.get("awake", True):
                stuck += 1
                continue
            dr, dc = DIRS[facing]
            first = frozenset((r + dr, c + dc) for r, c in cells)
            first_fence = (piece.get("species", "sheep") != "cattle"
                           and b.fence_cell_hits(cells, facing))
            if (any(cell in owners and owners[cell] != opid for cell in first)
                    or first_fence):
                stuck += 1
            else:
                movable += 1
            seen_blockers = set()
            frontier = cells
            while True:
                nxt = frozenset((r + dr, c + dc) for r, c in frontier)
                if any(not b.in_board(r, c) for r, c in nxt):
                    break
                internal_hits = b.fence_cell_hits(frontier, facing)
                if internal_hits and piece.get("species", "sheep") == "cattle":
                    break
                seen_blockers.update(owners[cell] for cell in nxt if cell in owners and owners[cell] != opid)
                if piece.get("species", "sheep") != "cattle":
                    seen_blockers.update(internal_hits)
                frontier = nxt
            if seen_blockers:
                blockers += len(seen_blockers)
            else:
                exits += 1
        return exits, movable, blockers, stuck

    cur = board
    seq = []
    seen = {cur.key()}
    while not cur.is_solved() and len(seq) < max_steps:
        if cancel and cancel():
            return seq, {"solved": False, "remaining": cur.remaining_count(),
                         "moves": len(seq), "cancelled": True}
        moves = cur.legal_moves()
        exits = [m for m in moves if m.result == "EXIT"]
        if exits:
            mv = exits[0]
        else:
            best, bscore = None, None
            for m in moves:
                if cancel and cancel():
                    return seq, {"solved": False, "remaining": cur.remaining_count(),
                                 "moves": len(seq), "cancelled": True}
                nb = cur.apply(m)
                exits_n, movable_n, blockers_n, stuck_n = features(nb)
                repeated = nb.key() in seen
                score = (
                    exits_n * 120
                    + movable_n * 3
                    - blockers_n * 9
                    - stuck_n * 7
                    - (80 if repeated else 0)
                    + (2 if m.result in {"MOVE", "STEP"} else 0)
                )
                if bscore is None or score > bscore:
                    bscore, best = score, m
            if best is None:
                return seq, {"solved": False, "remaining": cur.remaining_count(), "moves": len(seq)}
            mv = best
        seq.append(mv)
        cur = cur.apply(mv)
        key = cur.key()
        if key in seen and not exits:
            return seq, {"solved": False, "remaining": cur.remaining_count(), "moves": len(seq), "loop": True}
        seen.add(key)
    return seq, {"solved": cur.is_solved(), "remaining": cur.remaining_count(), "moves": len(seq)}


def solve(board: Board, max_nodes=400_000, cancel=None):
    """A* 搜索最少点击解。返回 (moves列表, 统计信息) 或 (None, 信息)。

    启发式 h = 剩余羊数（每只至少需 1 次点击才能出去，且一步最多送走 1 只）-> 可采纳。
    """
    start_key = board.key()
    counter = 0
    h0 = board.remaining_count()
    pq = [(h0, 0, counter, board, [])]
    best_g = {start_key: 0}
    expanded = 0
    while pq:
        if cancel and cancel():
            return None, {"reason": "已取消", "expanded": expanded, "cancelled": True}
        f, g, _, cur, path = heapq.heappop(pq)
        if cur.is_solved():
            return path, {"moves": len(path), "expanded": expanded}
        if g > best_g.get(cur.key(), g):
            continue
        expanded += 1
        if expanded > max_nodes:
            return None, {"reason": "超出搜索节点上限", "expanded": expanded}
        for mv in cur.legal_moves():
            nb = cur.apply(mv)
            nk = nb.key()
            ng = g + 1
            if ng < best_g.get(nk, 1 << 30):
                best_g[nk] = ng
                counter += 1
                nh = nb.remaining_count()
                heapq.heappush(pq, (ng + nh, ng, counter, nb, path + [mv]))
    return None, {"reason": "无解（搜索穷尽）", "expanded": expanded}
