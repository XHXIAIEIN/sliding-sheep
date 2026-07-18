/** 求解器领域类型：与桌面端 board.json 契约对齐。 */

export type Direction = "U" | "D" | "L" | "R";

export type Species =
  | "sheep" | "goat" | "rocket" | "bomb" | "pink_sheep"
  | "black_sheep" | "pig" | "cattle" | "elephant";

export type SlideMode = "all" | "any";
export type MoveModel = "axis_both" | "facing";

/** 格子编码 (r << 7) | c —— 单个整数在热路径上远快于 [r, c] 元组。 */
export type Cell = number & { readonly __brand: "cell" };

export const cell = (r: number, c: number): Cell => (((r << 7) | c) as Cell);
export const rowOf = (c: Cell): number => c >> 7;
export const colOf = (c: Cell): number => c & 127;

export const DIRS = {
  U: [-1, 0], D: [1, 0], L: [0, -1], R: [0, 1],
} as const satisfies Record<Direction, readonly [number, number]>;

/** 方向对应的编码增量。 */
export const DELTA = {
  U: -128, D: 128, L: -1, R: 1,
} as const satisfies Record<Direction, number>;

export type MoveKind = "EXIT" | "MOVE" | "STEP" | "BOUNCE";

export interface Move {
  readonly pieceId: string;
  readonly direction: Direction;
  /** 动作前棋子的稳定参考格 (r, c)，供点击定位与回放。 */
  readonly anchor: readonly [number, number];
  readonly result: MoveKind;
  readonly distance: number;
}

export interface Piece {
  /** 升序排列的占用格。 */
  readonly cells: readonly Cell[];
  readonly facing: Direction | null;
  readonly species: Species;
  readonly awake: boolean;
  readonly hitsRemaining: number | null;
}

/** board.json 的输入契约（桌面识别端产出）。 */
export interface BoardData {
  readonly rows: number;
  readonly cols: number;
  readonly model?: MoveModel;
  readonly slide_mode?: SlideMode;
  readonly hazards?: readonly (readonly [number, number])[];
  readonly fences?: readonly { cell: readonly [number, number]; direction: string }[];
  readonly pieces?: Readonly<Record<string, PieceData>>;
  readonly returning?: Readonly<Record<string, PieceData>>;
}

export interface PieceData {
  readonly cells: readonly (readonly [number, number])[];
  readonly facing?: Direction | null;
  readonly species?: Species;
  readonly awake?: boolean;
  readonly hit_limit?: number | null;
  readonly hits_remaining?: number | null;
}

export const moveEquals = (a: Move, b: Move): boolean =>
  a.pieceId === b.pieceId && a.direction === b.direction
  && a.result === b.result && a.distance === b.distance
  && a.anchor[0] === b.anchor[0] && a.anchor[1] === b.anchor[1];

/** Python 元组语义的字典序比较（同位置元素同型）。 */
export function lexCompare(a: readonly (number | string)[],
                           b: readonly (number | string)[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a[i]!, y = b[i]!;
    if (x < y) return -1;
    if (x > y) return 1;
  }
  return a.length - b.length;
}
