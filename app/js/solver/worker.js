/** Web Worker 求解入口：board.json 数据进，进度与步骤列表出。
 *
 * 协议：
 *   -> { type: "solve", board: BoardData, options?: {
 *          timeoutS, elasticTimeout, extensionS, maxTimeoutS } }
 *   <- { type: "progress", phase, data }
 *   <- { type: "result", ok: true, solved, remaining, kind, timedOut,
 *        steps: [{ pieceId, direction, anchor, result, distance, phase }], info }
 *   <- { type: "result", ok: false, error }
 *
 * 取消：求解是同步循环，主线程直接 worker.terminate() 后重建。
 */
import { Board } from "./board.js";
import { solveBoard } from "./planner.js";
self.onmessage = (event) => {
    const { type, board, options = {} } = event.data ?? {};
    if (type !== "solve")
        return;
    try {
        const result = solveBoard(Board.from(board), {
            timeoutS: options.timeoutS ?? 10,
            elasticTimeout: options.elasticTimeout ?? false,
            extensionS: options.extensionS ?? 5,
            maxTimeoutS: options.maxTimeoutS ?? null,
            onProgress: (phase, data) => self.postMessage({ type: "progress", phase, data }),
        });
        self.postMessage({
            type: "result",
            ok: true,
            solved: result.solved,
            remaining: result.remaining,
            kind: result.kind,
            timedOut: result.timedOut,
            steps: result.steps.map(([move, phase]) => ({ ...move, phase })),
            info: result.info,
        });
    }
    catch (error) {
        self.postMessage({
            type: "result", ok: false,
            error: error instanceof Error ? error.message : String(error),
        });
    }
};
