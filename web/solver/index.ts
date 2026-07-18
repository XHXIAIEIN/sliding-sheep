/** 求解器公共出口（测试与页面共用）。 */
export { Board, HAZARD } from "./board.ts";
export { forcedExitCandidates, forcedExitClosure, forcedExitSortKey,
         supportsForcedExitClosure } from "./closure.ts";
export { analyze, heuristic, rank, rankCompare,
         structuralDeadlocks } from "./heuristics.ts";
export { applyMoves, solveBoard } from "./planner.ts";
export type { PlanResult, SolveOptions } from "./planner.ts";
export { beamSolve, exactSolve, greedySolve, mulberry32,
         randomizedMacroSolve, weightedAstarSolve } from "./strategies.ts";
export { DIRS, cell, colOf, lexCompare, moveEquals, rowOf } from "./types.ts";
export type { BoardData, Cell, Direction, Move, Piece, Species } from "./types.ts";
