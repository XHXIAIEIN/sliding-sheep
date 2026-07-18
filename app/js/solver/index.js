/** 求解器公共出口（测试与页面共用）。 */
export { Board, HAZARD } from "./board.js";
export { forcedExitCandidates, forcedExitClosure, forcedExitSortKey, supportsForcedExitClosure } from "./closure.js";
export { analyze, heuristic, rank, rankCompare, structuralDeadlocks } from "./heuristics.js";
export { applyMoves, solveBoard } from "./planner.js";
export { beamSolve, exactSolve, greedySolve, mulberry32, randomizedMacroSolve, weightedAstarSolve } from "./strategies.js";
export { DIRS, cell, colOf, lexCompare, moveEquals, rowOf } from "./types.js";
