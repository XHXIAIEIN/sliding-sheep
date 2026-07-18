/** 不可变棋盘模型：规则、动作生成与应用。
 *
 * 与 scripts/solver/model.py 语义等价；实现按 JS 引擎特性重新设计：
 *  - 格子是整数编码，占用查询走一次性构建的 owner 映射；
 *  - Board 不可变，apply 做结构共享（未变化的 piece 对象直接复用）；
 *  - 状态键惰性计算并缓存（Python 版每次调用都重建）。
 */
import { DELTA, cell, colOf, rowOf } from "./types.ts";
import type {
  BoardData, Cell, Direction, Move, MoveKind, MoveModel, Piece, PieceData,
  SlideMode, Species,
} from "./types.ts";

/** owners() 映射中危险格的占位所有者。 */
export const HAZARD = "#hazard";
type Owner = string | typeof HAZARD;

function normalizePiece(data: PieceData, defaultSpecies: Species = "sheep"): Piece {
  const cells = data.cells
    .map(([r, c]) => cell(r, c))
    .sort((a, b) => a - b);
  return {
    cells,
    facing: data.facing ?? null,
    species: data.species ?? defaultSpecies,
    awake: data.awake ?? true,
    hitsRemaining: data.hits_remaining ?? null,
  };
}

function signature(p: Piece): string {
  return `${p.facing ?? ""}|${p.species}|${p.awake ? 1 : 0}|${p.hitsRemaining ?? ""}|${p.cells.join(".")}`;
}

const allowedDirs = (piece: Piece, model: MoveModel): readonly Direction[] => {
  if (piece.species === "pig" && !piece.awake) return [];
  if (model === "facing") {
    if (!piece.facing) throw new Error("facing 模型要求每只羊带 facing 字段");
    return [piece.facing];
  }
  const rows = new Set(piece.cells.map(rowOf));
  const cols = new Set(piece.cells.map(colOf));
  if (rows.size === 1 && cols.size > 1) return ["L", "R"];
  if (cols.size === 1 && rows.size > 1) return ["U", "D"];
  if (rows.size === 1 && cols.size === 1) return ["U", "D", "L", "R"];
  throw new Error("棋子不是直线");
};

export class Board {
  readonly rows: number;
  readonly cols: number;
  readonly model: MoveModel;
  readonly slideMode: SlideMode;
  readonly hazards: ReadonlySet<Cell>;
  /** 动态危险区：可穿越，但非离场动作不得停在其上。 */
  readonly noStop: ReadonlySet<Cell>;
  /** 栅栏段 "r,c,K"；K 为边界方向或内部走向 H/V。 */
  readonly fences: ReadonlySet<string>;
  readonly pieces: ReadonlyMap<string, Piece>;
  readonly returning: ReadonlyMap<string, Piece>;

  #key: string | null = null;
  #staticKey: string | null = null;
  #owners: Map<Cell, Owner> | null = null;

  constructor(init: {
    rows: number; cols: number; model: MoveModel; slideMode: SlideMode;
    hazards: ReadonlySet<Cell>; noStop: ReadonlySet<Cell>;
    fences: ReadonlySet<string>;
    pieces: ReadonlyMap<string, Piece>; returning: ReadonlyMap<string, Piece>;
  }) {
    this.rows = init.rows;
    this.cols = init.cols;
    this.model = init.model;
    this.slideMode = init.slideMode;
    this.hazards = init.hazards;
    this.noStop = init.noStop;
    this.fences = init.fences;
    this.pieces = init.pieces;
    this.returning = init.returning;
  }

  static from(data: BoardData, extra: {
    noStop?: readonly (readonly [number, number])[];
  } = {}): Board {
    const pieces = new Map<string, Piece>();
    for (const [pid, piece] of Object.entries(data.pieces ?? {})) {
      pieces.set(pid, normalizePiece(piece));
    }
    const returning = new Map<string, Piece>();
    for (const [pid, piece] of Object.entries(data.returning ?? {})) {
      returning.set(pid, normalizePiece(piece, "black_sheep"));
    }
    return new Board({
      rows: data.rows,
      cols: data.cols,
      model: data.model ?? "axis_both",
      slideMode: data.slide_mode ?? "all",
      hazards: new Set((data.hazards ?? []).map(([r, c]) => cell(r, c))),
      noStop: new Set((extra.noStop ?? []).map(([r, c]) => cell(r, c))),
      fences: new Set((data.fences ?? []).map(
        (f) => `${f.cell[0]},${f.cell[1]},${f.direction}`)),
      pieces,
      returning,
    });
  }

  inBoard(c: Cell): boolean {
    return c >= 0 && (c >> 7) < this.rows && (c & 127) < this.cols;
  }

  isSolved(): boolean {
    return this.pieces.size === 0 && this.returning.size === 0;
  }

  remaining(): number {
    return this.pieces.size + this.returning.size;
  }

  /** 每格的占用者（棋子 id 或危险格标记）；不可变故只建一次。 */
  owners(): ReadonlyMap<Cell, Owner> {
    if (!this.#owners) {
      const owners = new Map<Cell, Owner>();
      for (const c of this.hazards) owners.set(c, HAZARD);
      for (const [pid, piece] of this.pieces) {
        for (const c of piece.cells) owners.set(c, pid);
      }
      this.#owners = owners;
    }
    return this.#owners;
  }

  /** 规范化状态键：忽略 id，保留形状/朝向/物种/醒睡/炸弹计数。 */
  key(): string {
    if (this.#key === null) {
      this.#staticKey ??= [
        [...this.hazards].sort((a, b) => a - b).join("."),
        [...this.noStop].sort((a, b) => a - b).join("."),
        [...this.fences].sort().join(";"),
      ].join("#");
      const pieceSigs = [...this.pieces.values()].map(signature).sort();
      const returningSigs = [...this.returning.values()].map(signature).sort();
      this.#key = `${this.#staticKey}#${returningSigs.join(";")}#${pieceSigs.join(";")}`;
    }
    return this.#key;
  }

  /** 离场瞬间穿过的边界栅栏段。 */
  fenceCrossings(frontier: readonly Cell[], direction: Direction): Set<string> {
    const delta = DELTA[direction];
    const hits = new Set<string>();
    for (const c of frontier) {
      if (!this.inBoard((c + delta) as Cell)) {
        const key = `${rowOf(c)},${colOf(c)},${direction}`;
        if (this.fences.has(key)) hits.add(key);
      }
    }
    return hits;
  }

  /** 下一格推进将进入的内部 H/V 栅栏格。 */
  fenceCellHits(frontier: readonly Cell[], direction: Direction): Set<string> {
    const delta = DELTA[direction];
    const hits = new Set<string>();
    for (const c of frontier) {
      const next = (c + delta) as Cell;
      const r = rowOf(next), col = colOf(next);
      for (const kind of ["H", "V"]) {
        const key = `${r},${col},${kind}`;
        if (this.fences.has(key)) hits.add(key);
      }
    }
    return hits;
  }

  /** 牛冲撞触及的整条连通 H/V 栅栏。 */
  internalFenceRun(hits: ReadonlySet<string>): Set<string> {
    const connected = new Set(hits);
    const pending = [...hits];
    while (pending.length) {
      const [r, c, kind] = pending.pop()!.split(",");
      const ri = Number(r), ci = Number(c);
      const neighbours = kind === "H"
        ? [`${ri},${ci - 1},H`, `${ri},${ci + 1},H`]
        : [`${ri - 1},${ci},V`, `${ri + 1},${ci},V`];
      for (const n of neighbours) {
        if (this.fences.has(n) && !connected.has(n)) {
          connected.add(n);
          pending.push(n);
        }
      }
    }
    return connected;
  }

  /** 碰撞是否会引爆计数耗尽的炸弹羊（含移动者自身）。 */
  #collisionIsSafe(movingPid: string, frontier: readonly Cell[],
                   direction: Direction): boolean {
    const delta = DELTA[direction];
    const owners = this.owners();
    const affected = new Set<string>();
    for (const c of frontier) {
      const owner = owners.get((c + delta) as Cell);
      if (owner !== undefined && owner !== HAZARD && owner !== movingPid) {
        affected.add(owner);
      }
    }
    if (this.pieces.get(movingPid)!.species === "bomb") affected.add(movingPid);
    for (const pid of affected) {
      const piece = this.pieces.get(pid)!;
      if (piece.species === "bomb" && (piece.hitsRemaining || 3) <= 1) return false;
    }
    return true;
  }

  /** 与指定棋子边/角相邻的其它棋子（粉羊离场携带判定）。 */
  #touching(movingPid: string, cells: readonly Cell[]): Set<string> {
    const touching = new Set<string>();
    outer: for (const [pid, piece] of this.pieces) {
      if (pid === movingPid) continue;
      for (const a of cells) {
        const ar = rowOf(a), ac = colOf(a);
        for (const b of piece.cells) {
          if (Math.max(Math.abs(ar - rowOf(b)), Math.abs(ac - colOf(b))) <= 1) {
            touching.add(pid);
            continue outer;
          }
        }
      }
    }
    return touching;
  }

  legalMoves(): Move[] {
    const owners = this.owners();
    let moves: Move[] = [];
    for (const [pid, piece] of this.pieces) {
      for (const d of allowedDirs(piece, this.model)) {
        const move = this.#slide(pid, piece, d, owners);
        if (move) {
          if (Array.isArray(move)) moves.push(...move);
          else moves.push(move);
        }
      }
    }
    if (this.returning.size) {
      // 黑羊借位中：只保留能让它成功归位的动作。
      moves = moves.filter((move) => {
        try {
          this.apply(move);
          return true;
        } catch {
          return false;
        }
      });
    }
    return moves;
  }

  /** 沿 d 推进一只棋子，产出 0/1/多个动作（slide_mode=any 时多个）。 */
  #slide(pid: string, piece: Piece, d: Direction,
         owners: ReadonlyMap<Cell, Owner>): Move | Move[] | null {
    const delta = DELTA[d];
    const cells = piece.cells;
    let steps = 0;
    let exited = false, blockedByPiece = false, blockedByFixed = false, capped = false;
    const stepLimit = piece.species === "goat" ? 3 : Infinity;
    let frontier = cells;

    for (;;) {
      const next = frontier.map((c) => (c + delta) as Cell);
      if (next.some((c) => !this.inBoard(c))) {
        // 栅栏位于边界外沿；只有牛能撞破对应栅栏并离场。
        if (this.fenceCrossings(frontier, d).size && piece.species !== "cattle") {
          blockedByFixed = true;
        } else {
          exited = true;
        }
        break;
      }
      if (this.fenceCellHits(frontier, d).size) {
        if (piece.species === "cattle") exited = true;  // 内部栅栏是牛的专属出口
        else blockedByFixed = true;
        break;
      }
      let hitPiece = false, hitHazard = false;
      for (const c of next) {
        const owner = owners.get(c);
        if (owner === undefined || owner === pid) continue;
        if (owner === HAZARD) hitHazard = true;
        else hitPiece = true;
      }
      if (hitPiece || hitHazard) {
        blockedByPiece = hitPiece;
        blockedByFixed = hitHazard;
        break;
      }
      steps += 1;
      frontier = next;
      if (steps >= stepLimit) {
        capped = true;
        break;
      }
    }

    const first = cells[0]!;
    const anchor = [rowOf(first), colOf(first)] as const;
    const mk = (result: MoveKind, distance: number): Move =>
      ({ pieceId: pid, direction: d, anchor, result, distance });

    if (exited) return mk("EXIT", 0);
    if (steps === 0) return null;
    if (capped) {
      return frontier.some((c) => this.noStop.has(c)) ? null : mk("STEP", steps);
    }
    // 黑羊撞固定障碍会弹回原位；撞动物则正常停下并永久改变布局。
    if (piece.species === "black_sheep" && blockedByFixed && !blockedByPiece
        && this.returning.size === 0) {
      return mk("BOUNCE", steps);
    }
    if (!this.#collisionIsSafe(pid, frontier, d)) return null;
    if (this.slideMode === "all") {
      return frontier.some((c) => this.noStop.has(c)) ? null : mk("MOVE", steps);
    }
    const out: Move[] = [];
    for (let k = 1; k <= steps; k++) {
      if (!cells.some((c) => this.noStop.has((c + delta * k) as Cell))) {
        out.push(mk("MOVE", k));
      }
    }
    return out;
  }

  /** 应用一步，返回新 Board（结构共享，不改原对象）。 */
  apply(mv: Move): Board {
    const pieces = new Map(this.pieces);
    let returning: Map<string, Piece> = new Map(this.returning);
    let fences = this.fences;

    const moving = pieces.get(mv.pieceId);
    if (moving?.species === "cattle" && mv.result === "EXIT") {
      // 牛离场沿途撞毁触及的栅栏（边界段或整条内部围栏）。
      const broken = new Set(this.fences);
      let frontier = moving.cells;
      const delta = DELTA[mv.direction];
      for (;;) {
        const internalHits = this.fenceCellHits(frontier, mv.direction);
        if (internalHits.size) {
          for (const seg of this.internalFenceRun(internalHits)) broken.delete(seg);
          break;
        }
        const next = frontier.map((c) => (c + delta) as Cell);
        if (next.some((c) => !this.inBoard(c))) {
          for (const seg of this.fenceCrossings(frontier, mv.direction)) broken.delete(seg);
          break;
        }
        frontier = next;
      }
      fences = broken;
    }

    switch (mv.result) {
      case "BOUNCE": {
        returning.set(mv.pieceId, pieces.get(mv.pieceId)!);
        pieces.delete(mv.pieceId);
        break;
      }
      case "EXIT": {
        const piece = pieces.get(mv.pieceId)!;
        pieces.delete(mv.pieceId);
        if (piece.species === "pink_sheep") {
          for (const pid of this.#touching(mv.pieceId, piece.cells)) pieces.delete(pid);
        }
        break;
      }
      default: {
        const delta = DELTA[mv.direction] * mv.distance;
        const piece = pieces.get(mv.pieceId)!;
        const moved = piece.cells.map((c) => (c + delta) as Cell);
        pieces.set(mv.pieceId, { ...piece, cells: moved });
        if (mv.result === "MOVE") {
          // 撞击唤醒睡猪、消耗炸弹计数（含移动者自身是炸弹的情况）。
          const step = DELTA[mv.direction];
          const owners = new Map<Cell, string>();
          for (const [pid, p] of pieces) {
            if (pid === mv.pieceId) continue;
            for (const c of p.cells) owners.set(c, pid);
          }
          const affected = new Set<string>();
          for (const c of moved) {
            const owner = owners.get((c + step) as Cell);
            if (owner !== undefined) affected.add(owner);
          }
          if (pieces.get(mv.pieceId)!.species === "bomb") affected.add(mv.pieceId);
          for (const pid of affected) {
            const p = pieces.get(pid)!;
            const woken = p.species === "pig" && !p.awake ? { awake: true } : null;
            const hit = p.species === "bomb"
              ? { hitsRemaining: (p.hitsRemaining || 3) - 1 } : null;
            if (woken || hit) pieces.set(pid, { ...p, ...woken, ...hit });
          }
        }
      }
    }

    if (mv.result !== "BOUNCE" && returning.size) {
      // 非弹回动作结束后，借位中的黑羊必须能全部归位，否则该动作不合法。
      const occupied = new Set<Cell>(this.hazards);
      for (const p of pieces.values()) for (const c of p.cells) occupied.add(c);
      for (const [pid, p] of returning) {
        if (p.cells.some((c) => occupied.has(c))) {
          throw new Error("临时借位后黑羊无法返回原位");
        }
        pieces.set(pid, p);
        for (const c of p.cells) occupied.add(c);
      }
      returning = new Map();
    }

    return new Board({
      rows: this.rows, cols: this.cols, model: this.model, slideMode: this.slideMode,
      hazards: this.hazards, noStop: this.noStop, fences,
      pieces, returning,
    });
  }
}
