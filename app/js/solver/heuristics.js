/** 阻挡/死锁特征与状态排序：搜索策略共用的评估层。 */
import { DELTA, colOf, lexCompare, rowOf } from "./types.js";
import { Board, HAZARD } from "./board.js";
const directionsOf = (board, piece) => {
    if (board.model === "facing")
        return [piece.facing];
    const rows = new Set(piece.cells.map(rowOf));
    const cols = new Set(piece.cells.map(colOf));
    if (rows.size === 1 && cols.size > 1)
        return ["L", "R"];
    if (cols.size === 1 && rows.size > 1)
        return ["U", "D"];
    return ["U", "D", "L", "R"];
};
/** 便宜的阻挡/死锁特征，用于状态排序（不生成动作）。 */
export function analyze(board) {
    const n = board.remaining();
    if (!board.pieces.size) {
        return { remaining: n, blockers: 0, hazardBlockers: 0, stuck: 0,
            canExit: 0, movable: 0, deadlocks: 0, terminalDeadlock: 0 };
    }
    const owners = board.owners();
    const immediateBlocker = new Map();
    let blockersTotal = 0, hazardBlockers = 0, stuck = 0, canExit = 0, movable = 0;
    for (const [pid, piece] of board.pieces) {
        if (piece.species === "pig" && !piece.awake) {
            // 睡猪在被撞醒前是阻挡物，不能按可离场计分。
            stuck += 1;
            continue;
        }
        const cellsSet = new Set(piece.cells);
        const options = [];
        for (const direction of directionsOf(board, piece)) {
            const delta = DELTA[direction];
            const firstPieces = new Set();
            const firstHazards = new Set();
            let firstOut = false;
            for (const c of piece.cells) {
                const next = (c + delta);
                if (!board.inBoard(next))
                    firstOut = true;
                const owner = owners.get(next);
                if (owner !== undefined && !cellsSet.has(next)) {
                    if (owner === HAZARD)
                        firstHazards.add(next);
                    else
                        firstPieces.add(owner);
                }
            }
            if (piece.species !== "cattle") {
                for (const hit of board.fenceCellHits(piece.cells, direction))
                    firstHazards.add(hit);
                if (firstOut) {
                    for (const hit of board.fenceCrossings(piece.cells, direction))
                        firstHazards.add(hit);
                }
            }
            const blockers = new Set();
            const hazards = new Set();
            let frontier = piece.cells;
            for (;;) {
                const next = frontier.map((c) => (c + delta));
                if (next.some((c) => !board.inBoard(c))) {
                    if (piece.species !== "cattle") {
                        for (const hit of board.fenceCrossings(frontier, direction))
                            hazards.add(hit);
                    }
                    break;
                }
                const internalHits = board.fenceCellHits(frontier, direction);
                if (internalHits.size && piece.species === "cattle")
                    break;
                for (const c of next) {
                    const owner = owners.get(c);
                    if (owner === undefined || cellsSet.has(c))
                        continue;
                    if (owner === HAZARD)
                        hazards.add(c);
                    else
                        blockers.add(owner);
                }
                if (piece.species !== "cattle") {
                    for (const hit of internalHits)
                        hazards.add(hit);
                }
                frontier = next;
            }
            options.push({ firstPieces, firstHazards, blockers, hazards });
        }
        if (options.some((o) => !o.firstPieces.size && !o.firstHazards.size)) {
            movable += 1;
        }
        else {
            stuck += 1;
            const pieceHits = options
                .filter((o) => o.firstPieces.size && !o.firstHazards.size)
                .map((o) => o.firstPieces);
            if (pieceHits.length === 1 && pieceHits[0].size === 1) {
                immediateBlocker.set(pid, pieceHits[0].values().next().value);
            }
        }
        let best = options[0];
        for (const o of options.slice(1)) {
            const cur = o.blockers.size + o.hazards.size;
            const prev = best.blockers.size + best.hazards.size;
            if (cur < prev || (cur === prev && o.hazards.size < best.hazards.size))
                best = o;
        }
        blockersTotal += best.blockers.size;
        hazardBlockers += best.hazards.size;
        if (!best.blockers.size && !best.hazards.size)
            canExit += 1;
    }
    let deadlocks = 0;
    const seenPairs = new Set();
    for (const [a, b] of immediateBlocker) {
        if (immediateBlocker.get(b) === a) {
            const pair = [a, b].sort().join("|");
            if (!seenPairs.has(pair)) {
                seenPairs.add(pair);
                deadlocks += 1;
            }
        }
    }
    // 非空且既无出口也无碰撞动作的状态不可恢复：按终局死锁处理，
    // 否则大盘搜索会把预算耗在诱人的少子死胡同里。
    const noProgress = n > 0 && canExit === 0 && movable === 0;
    const simple = new Set(["sheep", "goat", "rocket"]);
    const onlySimple = [...board.pieces.values()].every((p) => simple.has(p.species));
    // 特殊棋子可能存在便宜特征刻意不建模的 BOUNCE/碰撞动作，
    // 终局判定前用权威动作生成器确认。
    let terminalDeadlock = noProgress && (onlySimple || !board.legalMoves().length) ? 1 : 0;
    if (!terminalDeadlock && n <= 12) {
        const legal = board.legalMoves();
        // 残局便宜地看穿最后一个假出口：每个合法动作都立刻冻结非空棋盘
        // 的状态，与已冻结状态一样不可挽救。
        terminalDeadlock = legal.length > 0 && legal.every((move) => {
            const next = board.apply(move);
            return !next.isSolved() && !next.legalMoves().length;
        }) ? 1 : 0;
    }
    return { remaining: n, blockers: blockersTotal, hazardBlockers, stuck,
        canExit, movable, deadlocks, terminalDeadlock };
}
export function heuristic(board) {
    const f = analyze(board);
    return f.terminalDeadlock * 1_000_000 + f.remaining * 120 + f.blockers * 10
        + f.hazardBlockers * 18 + f.stuck * 6 + f.deadlocks * 90
        - f.canExit * 22 - f.movable * 2;
}
export function rank(board, depth) {
    const f = analyze(board);
    return [f.terminalDeadlock, f.deadlocks, f.remaining, f.blockers,
        f.hazardBlockers, f.stuck, -f.canExit, -f.movable, depth];
}
export const rankCompare = (a, b) => lexCompare(a, b);
/** 同一直线上固定朝向迎头相向的棋子对：当前规则模型下不可解。 */
export function structuralDeadlocks(board) {
    if (board.model !== "facing")
        return [];
    const lanes = { H: new Map(),
        V: new Map() };
    for (const [pid, piece] of board.pieces) {
        const rows = new Set(piece.cells.map(rowOf));
        const cols = new Set(piece.cells.map(colOf));
        if (rows.size === 1) {
            const lane = rows.values().next().value;
            lanes.H.set(lane, [...(lanes.H.get(lane) ?? []), [pid, piece]]);
        }
        if (cols.size === 1) {
            const lane = cols.values().next().value;
            lanes.V.set(lane, [...(lanes.V.get(lane) ?? []), [pid, piece]]);
        }
    }
    const pairs = [];
    const seen = new Set();
    for (const axis of ["H", "V"]) {
        for (const [lane, group] of lanes[axis]) {
            for (let i = 0; i < group.length; i++) {
                for (const second of group.slice(i + 1)) {
                    let [aId, a] = group[i];
                    let [bId, b] = second;
                    const coord = axis === "H"
                        ? (p) => Math.min(...p.cells.map(colOf))
                        : (p) => Math.min(...p.cells.map(rowOf));
                    if (coord(a) > coord(b))
                        [aId, bId, a, b] = [bId, aId, b, a];
                    const opposing = axis === "H"
                        ? a.facing === "R" && b.facing === "L"
                        : a.facing === "D" && b.facing === "U";
                    const key = [aId, bId].sort().join("|");
                    if (!opposing || seen.has(key))
                        continue;
                    seen.add(key);
                    const cellsOf = (p) => p.cells.map((c) => [rowOf(c), colOf(c)]);
                    pairs.push({ axis, lane, pieces: [aId, bId],
                        facings: [a.facing, b.facing],
                        species: [a.species, b.species],
                        cells: [cellsOf(a), cellsOf(b)] });
                }
            }
        }
    }
    return pairs;
}
