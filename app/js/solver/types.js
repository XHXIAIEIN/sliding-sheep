/** 求解器领域类型：与桌面端 board.json 契约对齐。 */
export const cell = (r, c) => ((r << 7) | c);
export const rowOf = (c) => c >> 7;
export const colOf = (c) => c & 127;
export const DIRS = {
    U: [-1, 0], D: [1, 0], L: [0, -1], R: [0, 1],
};
/** 方向对应的编码增量。 */
export const DELTA = {
    U: -128, D: 128, L: -1, R: 1,
};
export const moveEquals = (a, b) => a.pieceId === b.pieceId && a.direction === b.direction
    && a.result === b.result && a.distance === b.distance
    && a.anchor[0] === b.anchor[0] && a.anchor[1] === b.anchor[1];
/** Python 元组语义的字典序比较（同位置元素同型）。 */
export function lexCompare(a, b) {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
        const x = a[i], y = b[i];
        if (x < y)
            return -1;
        if (x > y)
            return 1;
    }
    return a.length - b.length;
}
