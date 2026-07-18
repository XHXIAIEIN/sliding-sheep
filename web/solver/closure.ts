/** 确定性直出闭包：可证明单调的棋盘上安全地批量清出直达出口的羊。 */
import { DELTA, lexCompare } from "./types.ts";
import type { Cell, Direction, Move } from "./types.ts";
import { Board, HAZARD } from "./board.ts";

const CLOSURE_SPECIES = new Set(["sheep", "goat", "rocket", "bomb"]);

/** 确定性直出只对已证明单调的棋盘启用。 */
export const supportsForcedExitClosure = (board: Board): boolean =>
  board.model === "facing" && board.slideMode === "all"
  && board.returning.size === 0
  && [...board.pieces.values()].every((p) => CLOSURE_SPECIES.has(p.species));

/** 沿朝向依次遇到的其它棋子（去重保序）。 */
function forwardOwners(board: Board, pieceId: string, direction: Direction): string[] {
  const delta = DELTA[direction];
  const owners = board.owners();
  let frontier = board.pieces.get(pieceId)!.cells;
  const result: string[] = [];
  for (;;) {
    const next = frontier.map((c) => (c + delta) as Cell);
    if (next.some((c) => !board.inBoard(c))) break;
    frontier = next;
    const found: string[] = [];
    for (const c of frontier) {
      const owner = owners.get(c);
      if (owner !== undefined && owner !== HAZARD && owner !== pieceId
          && !found.includes(owner)) {
        found.push(owner);
      }
    }
    found.sort();
    for (const owner of found) {
      if (!result.includes(owner)) result.push(owner);
    }
  }
  return result;
}

/** 找出"急切清除会闭合阻挡环"的直出棋子（level-172 陷阱），留给搜索处理。 */
function protectedExitStoppers(board: Board, legalMoves: readonly Move[]): Set<string> {
  const exitIds = new Set(
    legalMoves.filter((m) => m.result === "EXIT").map((m) => m.pieceId));
  if (!exitIds.size) return new Set();

  // 把当前可直出的棋子从依赖图中折叠掉，剩下的边就是急切闭包
  // 清空所有出口后每只棋子最终会停靠的阻挡者。
  const collapsedBlocker = new Map<string, string>();
  for (const [pid, piece] of board.pieces) {
    if (!piece.facing) continue;
    const blocker = forwardOwners(board, pid, piece.facing)
      .find((owner) => !exitIds.has(owner));
    if (blocker !== undefined) collapsedBlocker.set(pid, blocker);
  }

  const protectedIds = new Set<string>();
  for (const move of legalMoves) {
    if (move.result === "EXIT" || move.distance <= 0) continue;
    const stagedExits: string[] = [];
    let finalBlocker: string | null = null;
    for (const owner of forwardOwners(board, move.pieceId, move.direction)) {
      if (exitIds.has(owner)) stagedExits.push(owner);
      else { finalBlocker = owner; break; }
    }
    if (!stagedExits.length || finalBlocker === null) continue;

    let current: string | undefined = finalBlocker;
    const seen = new Set<string>();
    while (current !== undefined && !seen.has(current)) {
      if (current === move.pieceId) {
        for (const pid of stagedExits) protectedIds.add(pid);
        break;
      }
      seen.add(current);
      current = collapsedBlocker.get(current);
    }
  }
  return protectedIds;
}

/** 有多少棋子被这只将离场的棋子直接挡住（越多越优先放行）。 */
function exitUnlockCount(board: Board, move: Move): number {
  if (board.model !== "facing") return 0;
  const target = new Set(board.pieces.get(move.pieceId)!.cells);
  let count = 0;
  for (const [pid, piece] of board.pieces) {
    if (pid === move.pieceId || !piece.facing) continue;
    const delta = DELTA[piece.facing];
    if (piece.cells.some((c) => target.has((c + delta) as Cell))) count += 1;
  }
  return count;
}

export const forcedExitSortKey = (board: Board, move: Move): (number | string)[] =>
  [-exitUnlockCount(board, move), move.anchor[0], move.anchor[1], move.pieceId];

/** 当前状态下可安全确定性清除的直出动作。 */
export function forcedExitCandidates(board: Board,
                                     { ordinaryOnly = false } = {}): Move[] {
  const legalMoves = board.legalMoves();
  const protectedIds = protectedExitStoppers(board, legalMoves);
  return legalMoves.filter((move) =>
    move.result === "EXIT"
    && !protectedIds.has(move.pieceId)
    && (!ordinaryOnly || board.pieces.get(move.pieceId)?.species === "sheep"));
}

/** 只清除"止挡安全"的直出并返回规范化棋盘。 */
export function forcedExitClosure(board: Board): [Move[], Board] {
  if (!supportsForcedExitClosure(board)) return [[], board];
  const moves: Move[] = [];
  let cur = board;
  for (;;) {
    const exits = forcedExitCandidates(cur);
    if (!exits.length) return [moves, cur];
    const move = exits.reduce((best, item) =>
      lexCompare(forcedExitSortKey(cur, item), forcedExitSortKey(cur, best)) < 0
        ? item : best);
    moves.push(move);
    cur = cur.apply(move);
  }
}
