/** 搜索策略组合：小盘最优 A* 与大盘 beam / weighted A* / 随机宏动作。 */
import { DELTA, lexCompare } from "./types.ts";
import type { Cell, Move } from "./types.ts";
import { Board, HAZARD } from "./board.ts";
import { forcedExitClosure, supportsForcedExitClosure } from "./closure.ts";
import { heuristic, rank, rankCompare } from "./heuristics.ts";
import type { Rank } from "./heuristics.ts";

export const now = (): number =>
  (globalThis.performance?.now() ?? Date.now()) / 1000;

export interface StrategyInfo {
  solved: boolean;
  kind: string;
  remaining: number;
  expanded: number;
  [extra: string]: unknown;
}

export type StrategyResult = [Move[], StrategyInfo];

export interface SearchOptions {
  readonly signal?: AbortSignal | undefined;
}

/** 二叉最小堆；优先级为定长数字数组，字典序比较。 */
class Heap<T> {
  readonly #prio: (readonly number[])[] = [];
  readonly #items: T[] = [];

  get size(): number { return this.#items.length; }

  push(priority: readonly number[], item: T): void {
    const prio = this.#prio, items = this.#items;
    prio.push(priority);
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (lexCompare(prio[i]!, prio[parent]!) < 0) {
        [prio[i], prio[parent]] = [prio[parent]!, prio[i]!];
        [items[i], items[parent]] = [items[parent]!, items[i]!];
        i = parent;
      } else break;
    }
  }

  pop(): T {
    const prio = this.#prio, items = this.#items;
    const top = items[0]!;
    const lastPrio = prio.pop()!, lastItem = items.pop()!;
    if (items.length) {
      prio[0] = lastPrio;
      items[0] = lastItem;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = 2 * i + 2;
        let m = i;
        if (l < items.length && lexCompare(prio[l]!, prio[m]!) < 0) m = l;
        if (r < items.length && lexCompare(prio[r]!, prio[m]!) < 0) m = r;
        if (m === i) break;
        [prio[i], prio[m]] = [prio[m]!, prio[i]!];
        [items[i], items[m]] = [items[m]!, items[i]!];
        i = m;
      }
    }
    return top;
  }
}

const moveRank = (mv: Move): (number | string)[] =>
  [mv.result === "EXIT" ? 0 : 1, -mv.distance, mv.pieceId];

const orderedMoves = (board: Board): Move[] =>
  board.legalMoves().sort((a, b) => lexCompare(moveRank(a), moveRank(b)));

/** A* 最少点击解；启发式 h = 剩余羊数（可采纳）。无解时返回 moves=null。 */
export function exactSolve(board: Board, {
  maxNodes = 400_000, signal,
}: SearchOptions & { maxNodes?: number } = {}):
  [Move[] | null, { reason?: string; expanded: number; cancelled?: boolean; moves?: number }] {
  let counter = 0;
  const heap = new Heap<[Board, Move[]]>();
  heap.push([board.remaining(), 0, counter], [board, []]);
  const bestG = new Map([[board.key(), 0]]);
  let expanded = 0;
  while (heap.size) {
    if (signal?.aborted) return [null, { reason: "已取消", expanded, cancelled: true }];
    const [cur, path] = heap.pop();
    const g = path.length;
    if (cur.isSolved()) return [path, { moves: path.length, expanded }];
    if (g > (bestG.get(cur.key()) ?? g)) continue;
    expanded += 1;
    if (expanded > maxNodes) return [null, { reason: "超出搜索节点上限", expanded }];
    for (const mv of cur.legalMoves()) {
      const next = cur.apply(mv);
      const key = next.key();
      const ng = g + 1;
      if (ng < (bestG.get(key) ?? Infinity)) {
        bestG.set(key, ng);
        counter += 1;
        heap.push([ng + next.remaining(), ng, counter], [next, [...path, mv]]);
      }
    }
  }
  return [null, { reason: "无解（搜索穷尽）", expanded }];
}

/** 大盘 best-effort 贪心：直出优先，否则选"之后能出栏数"最大的一步。 */
export function greedySolve(board: Board, {
  maxSteps = 10_000, signal,
}: SearchOptions & { maxSteps?: number } = {}): StrategyResult {
  const features = (b: Board): [number, number, number, number] => {
    const owners = b.owners();
    let exits = 0, movable = 0, blockers = 0, stuck = 0;
    for (const [pid, piece] of b.pieces) {
      if (piece.species === "pig" && !piece.awake) { stuck += 1; continue; }
      const facing = piece.facing!;
      const delta = DELTA[facing];
      const firstBlocked = piece.cells.some((c) => {
        const owner = owners.get((c + delta) as Cell);
        return owner !== undefined && owner !== pid && owner !== HAZARD;
      });
      const firstFence = piece.species !== "cattle"
        && b.fenceCellHits(piece.cells, facing).size > 0;
      if (firstBlocked || firstFence) stuck += 1;
      else movable += 1;
      const seen = new Set<string>();
      let frontier = piece.cells;
      for (;;) {
        const next = frontier.map((c) => (c + delta) as Cell);
        if (next.some((c) => !b.inBoard(c))) break;
        const internalHits = b.fenceCellHits(frontier, facing);
        if (internalHits.size && piece.species === "cattle") break;
        for (const c of next) {
          const owner = owners.get(c);
          if (owner !== undefined && owner !== pid && owner !== HAZARD) seen.add(owner);
        }
        if (piece.species !== "cattle") for (const hit of internalHits) seen.add(hit);
        frontier = next;
      }
      if (seen.size) blockers += seen.size;
      else exits += 1;
    }
    return [exits, movable, blockers, stuck];
  };

  let cur = board;
  const seq: Move[] = [];
  const seen = new Set([cur.key()]);
  while (!cur.isSolved() && seq.length < maxSteps) {
    if (signal?.aborted) {
      return [seq, { solved: false, kind: "greedy", remaining: cur.remaining(),
                     expanded: seq.length, cancelled: true }];
    }
    const moves = cur.legalMoves();
    const exits = moves.filter((m) => m.result === "EXIT");
    let mv: Move;
    if (exits.length) {
      mv = exits[0]!;
    } else {
      let best: Move | null = null, bestScore = -Infinity;
      for (const m of moves) {
        const next = cur.apply(m);
        const [e, mo, bl, st] = features(next);
        const score = e * 120 + mo * 3 - bl * 9 - st * 7
          - (seen.has(next.key()) ? 80 : 0)
          + (m.result === "MOVE" || m.result === "STEP" ? 2 : 0);
        if (score > bestScore) { bestScore = score; best = m; }
      }
      if (!best) {
        return [seq, { solved: false, kind: "greedy", remaining: cur.remaining(),
                       expanded: seq.length }];
      }
      mv = best;
    }
    seq.push(mv);
    cur = cur.apply(mv);
    const key = cur.key();
    if (seen.has(key) && !exits.length) {
      return [seq, { solved: false, kind: "greedy", remaining: cur.remaining(),
                     expanded: seq.length, loop: true }];
    }
    seen.add(key);
  }
  return [seq, { solved: cur.isSolved(), kind: "greedy",
                 remaining: cur.remaining(), expanded: seq.length }];
}

type EdgeNode = readonly [parent: number, edge: readonly Move[] | null];

function reconstruct(nodes: readonly EdgeNode[], idx: number): Move[] {
  const groups: (readonly Move[])[] = [];
  while (idx) {
    const [parent, edge] = nodes[idx]!;
    idx = parent;
    if (edge) groups.push(edge);
  }
  return groups.reverse().flat();
}

export function weightedAstarSolve(board: Board, {
  weight = 1.35, maxNodes = 120_000, timeLimit = 8.0, signal,
}: SearchOptions & { weight?: number; maxNodes?: number; timeLimit?: number } = {}):
  StrategyResult {
  const start = now();
  const [prefix, closed] = forcedExitClosure(board);
  if (closed.isSolved()) {
    return [prefix, { solved: true, kind: "exit-closure", expanded: 0, remaining: 0 }];
  }
  const nodes: EdgeNode[] = [[0, null]];
  let counter = 0;
  const heap = new Heap<[Board, number, number]>();  // [board, g, nodeIdx]
  heap.push([weight * heuristic(closed), 0, counter], [closed, 0, 0]);
  const bestG = new Map([[closed.key(), 0]]);
  let bestIdx = 0;
  let bestRank: Rank = rank(closed, 0);
  let expanded = 0;
  const endgameUnsat = new Set<string>();

  while (heap.size && expanded < maxNodes && now() - start < timeLimit
         && !signal?.aborted) {
    const [cur, g, nodeIdx] = heap.pop();
    if (g !== bestG.get(cur.key())) continue;
    if (cur.isSolved()) {
      return [[...prefix, ...reconstruct(nodes, nodeIdx)],
              { solved: true, kind: "weighted-a*", expanded, remaining: 0 }];
    }
    expanded += 1;
    for (const mv of orderedMoves(cur)) {
      const [forced, next] = forcedExitClosure(cur.apply(mv));
      const edge = [mv, ...forced];
      const key = next.key();
      const ng = g + edge.length;
      if (ng >= (bestG.get(key) ?? Infinity) || endgameUnsat.has(key)) continue;
      if (next.remaining() <= 8) {
        // 残局交给最优 A* 收尾；证明无解的残局标记后不再进入。
        const [tail, tailInfo] = exactSolve(next, { maxNodes: 20_000, signal });
        if (tail !== null) {
          return [[...prefix, ...reconstruct(nodes, nodeIdx), ...edge, ...tail],
                  { solved: true, kind: "weighted-a*+exact-endgame",
                    expanded: expanded + tailInfo.expanded, remaining: 0 }];
        }
        if (tailInfo.reason === "无解（搜索穷尽）") {
          endgameUnsat.add(key);
          continue;
        }
      }
      const r = rank(next, ng);
      if (r[0]) continue;
      bestG.set(key, ng);
      nodes.push([nodeIdx, edge]);
      const childIdx = nodes.length - 1;
      if (rankCompare(r, bestRank) < 0) {
        bestRank = r;
        bestIdx = childIdx;
      }
      counter += 1;
      heap.push([ng + weight * heuristic(next), ng, counter], [next, ng, childIdx]);
    }
  }

  const path = [...prefix, ...reconstruct(nodes, bestIdx)];
  const solved = bestRank[2] === 0;
  return [path, { solved, kind: solved ? "weighted-a*" : "weighted-a*(best)",
                  remaining: bestRank[2]!, expanded,
                  cancelled: !!signal?.aborted }];
}

export function beamSolve(board: Board, {
  width = 5000, maxDepth = 220, timeLimit = 12.0, seenCap = 450_000, signal,
}: SearchOptions & { width?: number; maxDepth?: number; timeLimit?: number;
                     seenCap?: number } = {}): StrategyResult {
  const start = now();
  const [prefix, closed] = forcedExitClosure(board);
  if (closed.isSolved()) {
    return [prefix, { solved: true, kind: "exit-closure", expanded: 0,
                      remaining: 0, depth: 0 }];
  }
  const nodes: EdgeNode[] = [[0, null]];
  let beam: [Rank, Board, number][] = [[rank(closed, 0), closed, 0]];
  const seen = new Map([[closed.key(), 0]]);
  let bestIdx = 0;
  let bestRank = beam[0]![0];
  let expanded = 0;

  const expired = () => now() - start >= timeLimit || !!signal?.aborted;

  for (let depth = 1; depth <= maxDepth && !expired(); depth++) {
    const layer: [Rank, Board, number][] = [];
    for (const [, cur, nodeIdx] of beam) {
      if (expired()) break;
      if (cur.isSolved()) {
        return [[...prefix, ...reconstruct(nodes, nodeIdx)],
                { solved: true, kind: "beam", expanded, remaining: 0, depth: depth - 1 }];
      }
      expanded += 1;
      for (const mv of orderedMoves(cur)) {
        if (expired()) break;
        const [forced, next] = forcedExitClosure(cur.apply(mv));
        const key = next.key();
        const oldDepth = seen.get(key);
        if (oldDepth !== undefined && oldDepth <= depth) continue;
        const r = rank(next, depth);
        if (r[0]) continue;
        if (seen.size < seenCap) seen.set(key, depth);
        nodes.push([nodeIdx, [mv, ...forced]]);
        const childIdx = nodes.length - 1;
        if (next.isSolved()) {
          return [[...prefix, ...reconstruct(nodes, childIdx)],
                  { solved: true, kind: "beam", expanded, remaining: 0, depth }];
        }
        if (rankCompare(r, bestRank) < 0) {
          bestRank = r;
          bestIdx = childIdx;
        }
        layer.push([r, next, childIdx]);
      }
    }
    if (!layer.length) break;
    layer.sort((a, b) => rankCompare(a[0], b[0]));
    beam = layer.slice(0, width);
  }

  const path = [...prefix, ...reconstruct(nodes, bestIdx)];
  const solved = bestRank[2] === 0;
  return [path, { solved, kind: solved ? "beam" : "beam(best)",
                  remaining: bestRank[2]!, expanded,
                  cancelled: !!signal?.aborted }];
}

/** 可复现的 32 位 PRNG。 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 折叠直出后的碰撞决策上做多样化重启的随机走。 */
export function randomizedMacroSolve(board: Board, {
  seed = 0, timeLimit = 4.0, signal,
}: SearchOptions & { seed?: number; timeLimit?: number } = {}): StrategyResult {
  if (!supportsForcedExitClosure(board)) {
    return [[], { solved: false, kind: "randomized-macro(unsupported)",
                  remaining: board.remaining(), expanded: 0 }];
  }

  const rng = mulberry32(seed ^ 0x9E3779B9);
  const start = now();
  const deadline = start + timeLimit;
  let expanded = 0, restarts = 0;
  let bestPath: Move[] = [];
  let bestRemaining = board.remaining();

  while (now() < deadline && !signal?.aborted) {
    let cur = board;
    const path: Move[] = [];
    const seen = new Set<string>();
    restarts += 1;
    while (now() < deadline && !signal?.aborted) {
      const [forced, closed] = forcedExitClosure(cur);
      cur = closed;
      path.push(...forced);
      if (cur.remaining() < bestRemaining) {
        bestRemaining = cur.remaining();
        bestPath = [...path];
      }
      if (cur.isSolved()) {
        return [path, { solved: true, kind: "randomized-macro", remaining: 0,
                        expanded, restarts, seed }];
      }
      const key = cur.key();
      if (seen.has(key)) break;
      seen.add(key);
      expanded += 1;

      type Candidate = [number, Move, Move[], Board];
      const candidates: Candidate[] = [];
      for (const move of cur.legalMoves()) {
        const [forcedNext, next] = forcedExitClosure(cur.apply(move));
        const r = rank(next, path.length + 1 + forcedNext.length);
        if (r[0]) continue;
        // rank 字段：terminal, cycle, remaining, blockers, hazards,
        // stuck, -exits, -movable, depth。remaining 主导，尾项保留多样性。
        const score = r[2]! * 100 + r[3]! * 6 + r[4]! * 10 + r[5]! * 2
          + r[6]! * 12 + r[7]! * 2;
        candidates.push([score, move, forcedNext, next]);
      }
      if (!candidates.length) break;
      candidates.sort((a, b) => a[0] - b[0]);
      let picked: Candidate;
      if (rng() < 0.3) {
        picked = candidates[Math.floor(rng() * candidates.length)]!;
      } else {
        const poolSize = Math.min(candidates.length,
          Math.max(2, Math.floor(Math.sqrt(candidates.length)) + 1));
        // 头部指数衰减的加权抽样。
        let total = 0;
        const weights = Array.from({ length: poolSize }, (_v, i) => {
          const w = 1 / (1 + i);
          total += w;
          return w;
        });
        let target = rng() * total;
        let index = 0;
        while (index < poolSize - 1 && (target -= weights[index]!) > 0) index += 1;
        picked = candidates[index]!;
      }
      const [, move, forcedNext, next] = picked;
      cur = next;
      path.push(move, ...forcedNext);
    }
  }

  return [bestPath, { solved: false, kind: "randomized-macro(best)",
                      remaining: bestRemaining, expanded, restarts, seed,
                      cancelled: !!signal?.aborted, timeout: now() >= deadline }];
}
