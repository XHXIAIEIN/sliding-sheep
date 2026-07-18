/** 求解调度：确定性直出前缀 + 弹性时限内的策略组合。 */
import { lexCompare } from "./types.js";
import { Board } from "./board.js";
import { forcedExitCandidates, forcedExitSortKey, supportsForcedExitClosure } from "./closure.js";
import { structuralDeadlocks } from "./heuristics.js";
import { beamSolve, exactSolve, greedySolve, now, randomizedMacroSolve, weightedAstarSolve } from "./strategies.js";
/** 有上限、可按显式增量延长的时限。 */
class ElasticBudget {
    #started = now();
    initialS;
    extensionS;
    maxS;
    enabled;
    allocatedS;
    deadline;
    extensions = 0;
    constructor(initialS, extensionS, maxS, enabled) {
        this.initialS = Math.max(0.1, initialS);
        this.extensionS = Math.max(0.1, extensionS);
        this.enabled = enabled;
        this.maxS = enabled ? Math.max(this.initialS, maxS) : this.initialS;
        this.allocatedS = this.initialS;
        this.deadline = this.#started + this.allocatedS;
    }
    remaining() { return this.deadline - now(); }
    extend() {
        if (!this.enabled || this.allocatedS >= this.maxS - 0.001)
            return 0;
        const added = Math.min(this.extensionS, this.maxS - this.allocatedS);
        this.allocatedS += added;
        this.deadline += added;
        this.extensions += 1;
        return added;
    }
    info() {
        return {
            initial_ms: Math.floor(this.initialS * 1000),
            extension_ms: Math.floor(this.extensionS * 1000),
            max_ms: Math.floor(this.maxS * 1000),
            allocated_ms: Math.floor(this.allocatedS * 1000),
            elapsed_ms: Math.floor((now() - this.#started) * 1000),
            extensions: this.extensions,
            elastic: this.enabled,
        };
    }
}
export function applyMoves(board, moves) {
    let current = board;
    for (const move of moves)
        current = current.apply(move);
    return [current.remaining(), current];
}
const drainExits = (board, deadline, signal, onStep) => {
    let current = board;
    const coarse = [];
    while (!current.isSolved() && now() < deadline && !signal?.aborted) {
        const exits = forcedExitCandidates(current);
        if (!exits.length)
            break;
        const cur = current;
        const move = exits.reduce((best, item) => lexCompare(forcedExitSortKey(cur, item), forcedExitSortKey(cur, best)) < 0
            ? item : best);
        coarse.push(move);
        current = current.apply(move);
        onStep(coarse.length, current.remaining());
    }
    return [coarse, current];
};
function bestCandidate(board, candidates) {
    let best = null;
    let bestScore = null;
    for (const item of candidates) {
        const [moves, info] = item;
        const [remaining, final] = applyMoves(board, moves);
        const legal = remaining ? final.legalMoves().length : 0;
        const score = [info.solved ? 0 : 1, info.remaining ?? remaining,
            remaining && legal === 0 ? 1 : 0, -legal, moves.length];
        if (bestScore === null || lexCompare(score, bestScore) < 0) {
            bestScore = score;
            best = item;
        }
    }
    return best;
}
function reportFinish(onProgress, phase, started, startRemaining, remaining, info, attempt, budgetS) {
    onProgress(phase, {
        event: "finish", attempt: attempt + 1,
        start_remaining: startRemaining, remaining,
        solved: info.solved || remaining === 0,
        elapsed_ms: Math.floor((now() - started) * 1000),
        budget_ms: Math.floor(Math.max(0, budgetS) * 1000),
        ...Object.fromEntries(["expanded", "restarts", "depth"]
            .filter((key) => typeof info[key] === "number")
            .map((key) => [key, info[key]])),
    });
}
/** 一轮策略组合：小盘最优 A* → 宏搜索 → weighted A* → 贪心兜底。 */
function refine(board, deadline, signal, onProgress, attempt) {
    if (board.isSolved()) {
        return [[], { solved: true, kind: "coarse-only", remaining: 0, expanded: 0 }];
    }
    let remainingTime = deadline - now();
    if (remainingTime <= 0.05) {
        return [[], { solved: false, kind: "精解超时", timeout: true,
                remaining: board.remaining(), expanded: 0 }];
    }
    const candidates = [];
    const macro = supportsForcedExitClosure(board);
    const startRemaining = board.remaining();
    if (board.pieces.size <= 14) {
        const started = now();
        onProgress("exact-a*", { event: "start", attempt: attempt + 1,
            remaining: startRemaining,
            budget_ms: Math.floor(remainingTime * 1000) });
        const [moves, rawInfo] = exactSolve(board, { maxNodes: 400_000, signal });
        const remaining = moves === null ? startRemaining : applyMoves(board, moves)[0];
        const info = {
            ...rawInfo, solved: moves !== null && remaining === 0,
            kind: moves !== null ? "A*最优" : "A*搜索", remaining,
            expanded: rawInfo.expanded,
        };
        reportFinish(onProgress, "exact-a*", started, startRemaining, remaining, info, attempt, remainingTime);
        if (moves !== null)
            return [moves, info];
        if (rawInfo.reason === "无解（搜索穷尽）") {
            return [[], { ...info, solved: false, kind: "A*证明无解",
                    remaining: startRemaining }];
        }
    }
    if (macro) {
        const deadlocks = structuralDeadlocks(board);
        if (deadlocks.length) {
            return [[], { solved: false, kind: "结构死锁", remaining: startRemaining,
                    expanded: 0, structural_deadlocks: deadlocks,
                    reason: "固定朝向棋子迎头相向，需核对特殊羊规则或识别方向" }];
        }
        for (const phase of ["macro-beam", "randomized-macro"]) {
            remainingTime = deadline - now();
            if (remainingTime <= 0.18)
                break;
            const share = phase === "macro-beam" ? 0.32 : 0.68;
            const cap = phase === "macro-beam" ? 5.0 : 12.0;
            const limit = Math.min(cap, Math.max(0.1, remainingTime * share));
            const started = now();
            onProgress(phase, { event: "start", attempt: attempt + 1,
                remaining: startRemaining,
                budget_ms: Math.floor(limit * 1000) });
            let result = phase === "macro-beam"
                ? beamSolve(board, { width: 14, maxDepth: 96, timeLimit: limit,
                    seenCap: 120_000, signal })
                : randomizedMacroSolve(board, { seed: attempt, timeLimit: limit, signal });
            if (phase === "macro-beam") {
                result = [result[0], { ...result[1], kind: `macro-${result[1].kind}` }];
            }
            const [moves, rawInfo] = result;
            const [remaining] = applyMoves(board, moves);
            const info = { ...rawInfo, remaining, solved: remaining === 0 };
            reportFinish(onProgress, phase, started, startRemaining, remaining, info, attempt, limit);
            candidates.push([moves, info]);
            if (info.solved)
                return [moves, info];
        }
    }
    let greedySeeded = false;
    if (board.pieces.size >= 35 && !macro) {
        const started = now();
        onProgress("online-greedy", { event: "start", attempt: attempt + 1,
            remaining: startRemaining });
        const [moves, rawInfo] = greedySolve(board, { maxSteps: 80, signal });
        const [remaining] = applyMoves(board, moves);
        const info = { ...rawInfo, kind: "online-greedy", remaining,
            solved: remaining === 0 };
        reportFinish(onProgress, "online-greedy", started, startRemaining, remaining, info, attempt, 0);
        candidates.push([moves, info]);
        greedySeeded = true;
        if (info.solved)
            return [moves, info];
    }
    const standard = macro ? ["weighted-a*"] : ["weighted-a*", "beam"];
    for (let index = 0; index < standard.length; index++) {
        const phase = standard[index];
        remainingTime = deadline - now();
        if (remainingTime <= 0.18)
            break;
        const share = phase === "weighted-a*" ? 0.62 : 0.38;
        const limit = index === standard.length - 1
            ? Math.max(0.1, remainingTime - 0.1)
            : Math.max(0.1, Math.min(remainingTime - 0.1, remainingTime * share));
        const started = now();
        onProgress(phase, { event: "start", attempt: attempt + 1,
            remaining: startRemaining,
            budget_ms: Math.floor(limit * 1000) });
        const [moves, rawInfo] = phase === "weighted-a*"
            ? weightedAstarSolve(board, {
                maxNodes: Math.max(90_000, Math.min(360_000, Math.floor(90_000 * Math.max(1, limit / 3)))),
                timeLimit: limit, signal
            })
            : beamSolve(board, { width: 3500, maxDepth: 260, timeLimit: limit,
                seenCap: 500_000, signal });
        const [remaining] = applyMoves(board, moves);
        const info = { ...rawInfo, remaining, solved: remaining === 0 };
        reportFinish(onProgress, phase, started, startRemaining, remaining, info, attempt, limit);
        candidates.push([moves, info]);
        if (info.solved)
            return [moves, info];
    }
    if (!greedySeeded) {
        const started = now();
        onProgress("greedy", { event: "start", attempt: attempt + 1,
            remaining: Math.min(startRemaining, ...candidates.map(([, i]) => i.remaining)) });
        const [moves, rawInfo] = greedySolve(board, { signal });
        const [remaining] = applyMoves(board, moves);
        const info = { ...rawInfo, remaining, solved: remaining === 0 };
        reportFinish(onProgress, "greedy", started, startRemaining, remaining, info, attempt, 0);
        candidates.push([moves, info]);
    }
    const [moves, info] = bestCandidate(board, candidates);
    return [moves, { ...info, timeout: now() >= deadline }];
}
/** GUI/网页共用的唯一求解入口。 */
export function solveBoard(board, options = {}) {
    const { signal, onProgress = () => { } } = options;
    const initialS = Math.max(0.1, options.timeoutS ?? 10);
    const budget = new ElasticBudget(initialS, Math.max(0.1, options.extensionS ?? 5), options.maxTimeoutS ?? initialS, options.elasticTimeout ?? false);
    onProgress("solve-budget", { event: "budget-start",
        remaining: board.remaining(), ...budget.info() });
    let coarse = [];
    let current = board;
    if (supportsForcedExitClosure(board)) {
        onProgress("exit-closure", { event: "progress", steps: 0,
            remaining: board.remaining() });
        [coarse, current] = drainExits(board, budget.deadline, signal, (steps, remaining) => onProgress("exit-closure", { event: "progress", steps, remaining }));
    }
    const candidates = [];
    let attempt = 0;
    let result;
    for (;;) {
        result = refine(current, budget.deadline, signal, onProgress, attempt);
        candidates.push(result);
        if (result[1].solved)
            break;
        const expired = !!result[1].timeout || budget.remaining() <= 0.02;
        if (!expired)
            break;
        const added = budget.extend();
        if (added <= 0)
            break;
        attempt += 1;
        onProgress("budget-extension", {
            event: "extension", attempt: attempt + 1,
            added_ms: Math.floor(added * 1000),
            remaining: result[1].remaining, ...budget.info(),
        });
    }
    if (candidates.length > 1 && !result[1].solved) {
        result = bestCandidate(current, candidates);
    }
    const [refineMoves, info] = result;
    const steps = [
        ...coarse.map((move) => [move, "coarse"]),
        ...refineMoves.map((move) => [move, "refine"]),
    ];
    const [remaining, finalBoard] = applyMoves(board, steps.map(([move]) => move));
    const solved = remaining === 0;
    const timedOut = !solved && (!!info.timeout || budget.remaining() <= 0.02);
    return {
        steps, finalBoard, solved, remaining,
        kind: `粗解${coarse.length} + ${info.kind}`,
        timedOut,
        info: { ...info, solved, remaining, timeout: timedOut, budget: budget.info() },
    };
}
