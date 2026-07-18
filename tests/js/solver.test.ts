/** TS 求解器与 Python 版的行为等价测试（对照 tests/test_solver.py）。 */
import { strict as assert } from "node:assert";
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { Board, analyze, beamSolve, exactSolve, forcedExitClosure, heuristic,
         moveEquals, rank, rankCompare, randomizedMacroSolve, solveBoard,
         structuralDeadlocks, supportsForcedExitClosure,
         weightedAstarSolve } from "../../web/solver/index.ts";
import type { BoardData, Move } from "../../web/solver/index.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

const board = (data: Partial<BoardData> & { rows: number; cols: number },
               noStop: (readonly [number, number])[] = []) =>
  Board.from({ model: "facing", slide_mode: "all", ...data } as BoardData,
             { noStop });

/** 逐步回放：校验每一步都是当时的合法动作。 */
function replay(start: Board, moves: readonly Move[]): Board {
  let cur = start;
  for (const mv of moves) {
    assert.ok(cur.legalMoves().some((m) => moveEquals(m, mv)),
              `非法步: ${JSON.stringify(mv)}`);
    cur = cur.apply(mv);
  }
  return cur;
}

test("基础出口：axis_both 三羊错位 A* 有解", () => {
  const b = board({ rows: 4, cols: 4, model: "axis_both", pieces: {
    A: { cells: [[1, 1], [1, 2]] },
    B: { cells: [[1, 3]] },
    C: { cells: [[0, 0], [0, 1]] },
  } });
  const [moves] = exactSolve(b);
  assert.ok(moves, "应有解");
  assert.ok(replay(b, moves).isSolved());
});

test("竖直堵塞：必须先动竖直羊", () => {
  const b = board({ rows: 5, cols: 5, model: "axis_both", pieces: {
    A: { cells: [[2, 1], [2, 2]] },
    V: { cells: [[0, 3], [1, 3], [2, 3]] },
  } });
  const [moves] = exactSolve(b);
  assert.ok(moves);
  assert.ok(replay(b, moves).isSolved());
});

test("facing 模型：前方羊必须先出", () => {
  const b = board({ rows: 4, cols: 4, pieces: {
    A: { cells: [[1, 0], [1, 1]], facing: "R" },
    B: { cells: [[1, 2], [1, 3]], facing: "R" },
  } });
  const [moves] = exactSolve(b);
  assert.ok(moves);
  assert.ok(replay(b, moves).isSolved());
  assert.equal(moves[0]!.pieceId, "B");
});

test("level172 分段止挡：直出闭包保留环上的止挡出口", () => {
  const b = board({ rows: 8, cols: 6, pieces: {
    "54": { cells: [[0, 1], [1, 1]], facing: "D" },
    "55": { cells: [[0, 2], [0, 3]], facing: "L" },
    "56": { cells: [[0, 4], [0, 5]], facing: "R" },
    "59": { cells: [[1, 2], [2, 2]], facing: "D" },
    "60": { cells: [[1, 3], [1, 4]], facing: "R" },
    "61": { cells: [[1, 5], [2, 5]], facing: "D" },
    "64": { cells: [[2, 0], [2, 1]], facing: "R" },
    "65": { cells: [[2, 3], [2, 4]], facing: "R" },
    "69": { cells: [[3, 1], [4, 1]], facing: "D" },
    "70": { cells: [[3, 2], [4, 2]], facing: "D" },
    "71": { cells: [[3, 3], [3, 4]], facing: "R" },
    "72": { cells: [[3, 5], [4, 5]], facing: "D" },
    "77": { cells: [[4, 3], [4, 4]], facing: "R" },
    "80": { cells: [[5, 1], [5, 2]], facing: "R" },
    "81": { cells: [[5, 3], [6, 3]], facing: "U" },
    "82": { cells: [[5, 4], [6, 4]], facing: "D" },
    "85": { cells: [[6, 1], [6, 2]], facing: "R" },
    "88": { cells: [[7, 2], [7, 3]], facing: "R" },
  } });
  const [, normalized] = forcedExitClosure(b);
  const remaining = new Set(normalized.pieces.keys());
  for (const pid of ["60", "65", "71"]) assert.ok(remaining.has(pid), pid);
  const staged = normalized.legalMoves().filter((m) => m.pieceId === "81");
  assert.equal(staged.length, 1);
  assert.equal(staged[0]!.result, "MOVE");
  assert.equal(staged[0]!.distance, 1);

  const [moves, info] = beamSolve(b, { width: 14, maxDepth: 32,
                                       timeLimit: 1.0, seenCap: 20_000 });
  assert.ok(info.solved, JSON.stringify(info));
  assert.ok(replay(b, moves).isSolved());
});

test("状态键必须区分朝向", () => {
  const a = board({ rows: 3, cols: 4, pieces: {
    A: { cells: [[1, 0], [1, 1]], facing: "R" },
    B: { cells: [[1, 2], [1, 3]], facing: "L" },
  } });
  const b = board({ rows: 3, cols: 4, pieces: {
    A: { cells: [[1, 0], [1, 1]], facing: "L" },
    B: { cells: [[1, 2], [1, 3]], facing: "R" },
  } });
  assert.notEqual(a.key(), b.key());
});

test("搜索一步清盘边界", () => {
  const b = board({ rows: 3, cols: 4, pieces: {
    A: { cells: [[1, 1], [1, 2]], facing: "R" },
  } });
  const [wmoves, winfo] = weightedAstarSolve(b, { maxNodes: 1, timeLimit: 1.0 });
  assert.ok(winfo.solved && winfo.remaining === 0, JSON.stringify(winfo));
  assert.ok(replay(b, wmoves).isSolved());
  const [bmoves, binfo] = beamSolve(b, { width: 4, maxDepth: 1, timeLimit: 1.0 });
  assert.ok(binfo.solved && binfo.remaining === 0, JSON.stringify(binfo));
  assert.ok(replay(b, bmoves).isSolved());
});

test("固定朝向迎头相向是结构死锁", () => {
  const b = board({ rows: 3, cols: 8, pieces: {
    A: { cells: [[1, 1], [1, 2]], facing: "R", species: "sheep" },
    B: { cells: [[1, 5], [1, 6]], facing: "L", species: "rocket" },
  } });
  const pairs = structuralDeadlocks(b);
  assert.equal(pairs.length, 1);
  assert.deepEqual(pairs[0]!.pieces, ["A", "B"]);
  const result = solveBoard(b, { timeoutS: 2 });
  assert.ok(!result.solved);
  // 小盘先被最优 A* 穷尽证明无解；大盘才会落到结构死锁报告。
  assert.match(result.kind, /A\*证明无解|结构死锁/);
});

test("axis_both 搜索与危险格特征", () => {
  const b = board({ rows: 4, cols: 6, model: "axis_both", pieces: {
    A: { cells: [[1, 1], [1, 2]] },
    B: { cells: [[2, 3], [3, 3]] },
  } });
  const [moves] = weightedAstarSolve(b, { maxNodes: 100, timeLimit: 1.0 });
  assert.ok(moves.length);
  const hazard = board({ rows: 3, cols: 4, hazards: [[1, 3]], pieces: {
    H: { cells: [[1, 0], [1, 1]], facing: "R" },
  } });
  assert.ok(analyze(hazard).hazardBlockers > 0);
});

test("狼 no_stop 区：允许穿越离场，禁止停留", () => {
  const crossing = board({ rows: 3, cols: 7, pieces: {
    A: { cells: [[1, 0], [1, 1]], facing: "R" },
  } }, [[1, 3], [1, 4]]);
  const crossingMoves = crossing.legalMoves();
  assert.equal(crossingMoves.length, 1);
  assert.equal(crossingMoves[0]!.result, "EXIT");

  const landing = board({ rows: 3, cols: 7, pieces: {
    A: { cells: [[1, 0], [1, 1]], facing: "R" },
    B: { cells: [[1, 6]], facing: "L" },
  } }, [[1, 4], [1, 5]]);
  assert.deepEqual(landing.legalMoves().filter((m) => m.pieceId === "A"), []);
});

test("2x3 大象可整体移动并离场", () => {
  const b = board({ rows: 5, cols: 5, pieces: {
    E: { cells: [[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]],
         facing: "U", species: "elephant" },
  } });
  const moves = b.legalMoves();
  assert.equal(moves.length, 1);
  assert.equal(moves[0]!.result, "EXIT");
  assert.equal(moves[0]!.direction, "U");
});

test("炸弹碰撞预算：计数耗尽前可撞，耗尽即禁", () => {
  const pieces = {
    A: { cells: [[1, 0], [1, 1]], facing: "R", species: "sheep" },
    B: { cells: [[1, 3], [1, 4]], facing: "R", species: "bomb",
         hit_limit: 3, hits_remaining: 2 },
  } as const;
  const b = board({ rows: 3, cols: 6, pieces });
  const bump = b.legalMoves().find((m) => m.pieceId === "A")!;
  const after = b.apply(bump);
  assert.equal(after.pieces.get("B")!.hitsRemaining, 1);
  const dangerous = board({ rows: 3, cols: 6, pieces: {
    ...pieces, B: { ...pieces.B, hits_remaining: 1 },
  } });
  assert.ok(dangerous.legalMoves().every((m) => m.pieceId !== "A"));
});

test("移动中的炸弹最后一次碰撞被禁止", () => {
  const pieces = {
    B: { cells: [[0, 2], [1, 2]], facing: "D", species: "bomb",
         hit_limit: 3, hits_remaining: 1 },
    S: { cells: [[3, 2], [4, 2]], facing: "D", species: "sheep" },
  } as const;
  const dangerous = board({ rows: 6, cols: 5, pieces });
  assert.ok(dangerous.legalMoves().every((m) => m.pieceId !== "B"));
  const safe = board({ rows: 6, cols: 5, pieces: {
    ...pieces, B: { ...pieces.B, hits_remaining: 2 },
  } });
  const move = safe.legalMoves().find((m) => m.pieceId === "B")!;
  assert.equal(safe.apply(move).pieces.get("B")!.hitsRemaining, 1);
});

test("边界栅栏挡住动物，牛撞破后离场", () => {
  const fences = [{ cell: [1, 0] as const, direction: "L" }];
  const sheep = board({ rows: 4, cols: 5, fences, pieces: {
    S: { cells: [[1, 0], [1, 1]], facing: "L", species: "sheep" },
  } });
  assert.equal(sheep.legalMoves().length, 0);

  const cattle = board({ rows: 4, cols: 5, fences, pieces: {
    C: { cells: [[1, 0], [1, 1]], facing: "L", species: "cattle" },
  } });
  const move = cattle.legalMoves().find((m) => m.result === "EXIT")!;
  const after = cattle.apply(move);
  assert.equal(after.pieces.size, 0);
  assert.equal(after.fences.size, 0);
});

test("内部栅栏挡住动物，牛冲破整条围栏直接离场", () => {
  const fences = [3, 4, 5].map((c) => ({ cell: [1, c] as const, direction: "H" }));
  const sheep = board({ rows: 3, cols: 8, fences, pieces: {
    S: { cells: [[1, 0], [1, 1]], facing: "R", species: "sheep" },
  } });
  const move = sheep.legalMoves().find((m) => m.pieceId === "S")!;
  assert.equal(move.result, "MOVE");
  assert.equal(move.distance, 1);
  const stopped = sheep.apply(move);
  assert.deepEqual([...stopped.pieces.get("S")!.cells].sort((a, b) => a - b),
                   [(1 << 7) | 1, (1 << 7) | 2]);
  assert.equal(stopped.fences.size, sheep.fences.size);

  const cattle = board({ rows: 3, cols: 8, fences, pieces: {
    C: { cells: [[1, 0], [1, 1]], facing: "R", species: "cattle" },
    S: { cells: [[1, 6], [1, 7]], facing: "R", species: "sheep" },
  } });
  const charge = cattle.legalMoves().find((m) => m.pieceId === "C")!;
  assert.equal(charge.result, "EXIT");
  const after = cattle.apply(charge);
  assert.deepEqual([...after.pieces.keys()], ["S"]);
  assert.equal(after.fences.size, 0);
});

test("黑羊撞动物后停下并永久改变布局", () => {
  const b = board({ rows: 3, cols: 8, pieces: {
    K: { cells: [[1, 0], [1, 1]], facing: "R", species: "black_sheep" },
    S: { cells: [[1, 5], [1, 6]], facing: "R", species: "sheep" },
  } });
  const move = b.legalMoves().find((m) => m.pieceId === "K")!;
  assert.equal(move.result, "MOVE");
  assert.equal(move.distance, 3);
  const after = b.apply(move);
  assert.deepEqual([...after.pieces.get("K")!.cells].sort((a, b) => a - b),
                   [(1 << 7) | 3, (1 << 7) | 4]);
  assert.equal(after.returning.size, 0);
});

test("黑羊撞固定障碍是临时借位，动物离场后归位", () => {
  const b = board({ rows: 3, cols: 8, hazards: [[1, 5]], pieces: {
    K: { cells: [[1, 0], [1, 1]], facing: "R", species: "black_sheep" },
    E: { cells: [[0, 6], [0, 7]], facing: "U", species: "sheep" },
  } });
  const bounce = b.legalMoves().find((m) => m.pieceId === "K")!;
  assert.equal(bounce.result, "BOUNCE");
  assert.equal(bounce.distance, 3);
  const borrowed = b.apply(bounce);
  assert.ok(!borrowed.pieces.has("K") && borrowed.returning.has("K"));
  assert.equal(borrowed.remaining(), 2);
  const exitMove = borrowed.legalMoves().find((m) => m.pieceId === "E")!;
  const restored = borrowed.apply(exitMove);
  assert.ok(restored.pieces.has("K"));
  assert.equal(restored.returning.size, 0);
  assert.deepEqual([...restored.pieces.get("K")!.cells].sort((a, b) => a - b),
                   [(1 << 7) | 0, (1 << 7) | 1]);
});

test("粉羊离场携带相邻羊，移动不携带", () => {
  const exitBoard = board({ rows: 6, cols: 7, pieces: {
    P: { cells: [[2, 1], [2, 0]], facing: "L", species: "pink_sheep" },
    edge: { cells: [[1, 1], [1, 2]], facing: "R", species: "sheep" },
    corner: { cells: [[3, 2], [4, 2]], facing: "D", species: "sheep" },
    far: { cells: [[4, 5], [4, 6]], facing: "R", species: "sheep" },
  } });
  const exitMove = exitBoard.legalMoves().find((m) => m.pieceId === "P")!;
  assert.equal(exitMove.result, "EXIT");
  assert.deepEqual([...exitBoard.apply(exitMove).pieces.keys()], ["far"]);

  const moveBoard = board({ rows: 5, cols: 8, pieces: {
    P: { cells: [[2, 1], [2, 2]], facing: "R", species: "pink_sheep" },
    touching: { cells: [[1, 1], [1, 2]], facing: "L", species: "sheep" },
    blocker: { cells: [[2, 6], [2, 7]], facing: "R", species: "sheep" },
  } });
  const slide = moveBoard.legalMoves().find((m) => m.pieceId === "P")!;
  assert.equal(slide.result, "MOVE");
  assert.equal(moveBoard.apply(slide).pieces.size, 3);
});

test("睡猪被撞醒后才可点击", () => {
  const b = board({ rows: 3, cols: 8, pieces: {
    S: { cells: [[1, 0], [1, 1]], facing: "R", species: "sheep" },
    P: { cells: [[1, 5], [1, 6]], facing: "R", species: "pig", awake: false },
  } });
  assert.ok(b.legalMoves().every((m) => m.pieceId !== "P"));
  const collision = b.legalMoves().find((m) => m.pieceId === "S")!;
  assert.equal(collision.result, "MOVE");
  assert.equal(collision.distance, 3);
  const after = b.apply(collision);
  assert.ok(after.pieces.get("P")!.awake);
  const pigMove = after.legalMoves().find((m) => m.pieceId === "P")!;
  assert.equal(pigMove.result, "EXIT");
  assert.equal(pigMove.direction, "R");
});

test("猪的醒睡状态进入棋盘键", () => {
  const cells = [[1, 1], [1, 2]] as const;
  const sleeping = board({ rows: 3, cols: 5, pieces: {
    P: { cells, facing: "R", species: "pig", awake: false } } });
  const awake = board({ rows: 3, cols: 5, pieces: {
    P: { cells, facing: "R", species: "pig", awake: true } } });
  assert.notEqual(sleeping.key(), awake.key());
});

test("山羊每次固定前进三格直到离场", () => {
  const b = board({ rows: 3, cols: 10, pieces: {
    G: { cells: [[1, 0], [1, 1]], facing: "R", species: "goat" },
  } });
  const first = b.legalMoves().find((m) => m.pieceId === "G")!;
  assert.equal(first.result, "STEP");
  assert.equal(first.distance, 3);
  const afterFirst = b.apply(first);
  const second = afterFirst.legalMoves().find((m) => m.pieceId === "G")!;
  assert.equal(second.result, "STEP");
  const afterSecond = afterFirst.apply(second);
  const third = afterSecond.legalMoves().find((m) => m.pieceId === "G")!;
  assert.equal(third.result, "EXIT");
  assert.ok(afterSecond.apply(third).isSolved());

  const blocked = board({ rows: 3, cols: 8, pieces: {
    G: { cells: [[1, 0], [1, 1]], facing: "R", species: "goat" },
    B: { cells: [[1, 4], [1, 5]], facing: "R", species: "sheep" },
  } });
  const collision = blocked.legalMoves().find((m) => m.pieceId === "G")!;
  assert.equal(collision.result, "MOVE");
  assert.equal(collision.distance, 2);
});

test("粉羊在场时禁用强制离场闭包", () => {
  const b = board({ rows: 4, cols: 6, pieces: {
    P: { cells: [[1, 0], [1, 1]], facing: "L", species: "pink_sheep" },
    S: { cells: [[3, 4], [3, 5]], facing: "R", species: "sheep" },
  } });
  assert.ok(!supportsForcedExitClosure(b));
  const [moves, normalized] = forcedExitClosure(b);
  assert.equal(moves.length, 0);
  assert.equal(normalized.key(), b.key());
});

test("终局死锁排在可恢复状态之后", () => {
  const dead = board({ rows: 2, cols: 4, pieces: {
    left: { cells: [[0, 0], [0, 1]], facing: "R" },
    right: { cells: [[0, 2], [0, 3]], facing: "L" },
  } });
  const live = board({ rows: 2, cols: 5, pieces: {
    a: { cells: [[0, 0], [0, 1]], facing: "R" },
    b: { cells: [[1, 0], [1, 1]], facing: "R" },
    c: { cells: [[0, 3], [0, 4]], facing: "R" },
  } });
  assert.equal(analyze(dead).terminalDeadlock, 1);
  assert.equal(analyze(live).terminalDeadlock, 0);
  assert.ok(rankCompare(rank(live, 0), rank(dead, 0)) < 0);
  assert.ok(heuristic(live) < heuristic(dead));
});

// ---- 真实归档关卡（依赖本机 cache 数据；缺失时跳过） ----

const level113Path = join(ROOT, "cache", "levels", "7614d5dd102dae13",
                          "executions", "step-0001.json");
test("归档 level113：宏 beam 解出大盘", { skip: !existsSync(level113Path) }, () => {
  const data = JSON.parse(readFileSync(level113Path, "utf8")).board_before;
  const b = Board.from(data);
  const [moves, info] = beamSolve(b, { width: 14, maxDepth: 96,
                                       timeLimit: 5.0, seenCap: 120_000 });
  assert.ok(info.solved, JSON.stringify(info));
  assert.ok(replay(b, moves).isSolved());
  assert.ok(moves.length <= 88, `${moves.length}`);
});

const level119Path = join(ROOT, "cache", "manual_samples",
                          "20260714-213828-505", "board.json");
test("归档 level119：随机宏动作搜索解出", { skip: !existsSync(level119Path) }, () => {
  const b = Board.from(JSON.parse(readFileSync(level119Path, "utf8")));
  const [moves, info] = randomizedMacroSolve(b, { seed: 0, timeLimit: 4.0 });
  assert.ok(info.solved, JSON.stringify(info));
  assert.ok(replay(b, moves).isSolved());
});

test("归档 level119：统一规划入口解出", { skip: !existsSync(level119Path) }, () => {
  const manualPath = join(ROOT, "cache", "manual_samples",
                          "20260714-213828-505", "manual_board.json");
  const b = Board.from(JSON.parse(readFileSync(manualPath, "utf8")));
  const result = solveBoard(b, { timeoutS: 6 });
  assert.ok(result.solved && result.remaining === 0, JSON.stringify(result.info));
  assert.ok(replay(b, result.steps.map(([move]) => move)).isSolved());
});
