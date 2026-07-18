"""Debug rendering of rectified boards, grid labels, and layouts."""
from __future__ import annotations

import cv2
import numpy as np
from board import grid as G
from paths import image_path
from .masks import CELL, _warp


def _to_px(Minv, c, r):
    v = Minv @ np.array([c * CELL, r * CELL, 1.0])
    return v[0] / v[2], v[1] / v[2]


AXIS_COLORS = {
    "V": (216, 126, 47),   # BGR, blue
    "H": (47, 135, 216),   # BGR, amber
}


SPECIES_COLORS = {
    "sheep": None,          # sheep keep axis colors
    "cattle": (190, 92, 155),
    "wolf": (112, 105, 94),
    "pig": (182, 94, 210),
    "goat": (92, 154, 205),
    "rocket": (44, 66, 214),
    "elephant": (156, 111, 118),
    "bomb": (42, 42, 210),
    "pink_sheep": (180, 82, 224),
}


GRID_COLOR = (246, 248, 250)


INK = (34, 38, 45)


FOCUS_RED = (58, 76, 200)


HAZARD_COLOR = (112, 105, 94)


def _piece_color(piece):
    if piece.get("review"):
        return FOCUS_RED
    species = piece.get("species", "sheep")
    return SPECIES_COLORS.get(species) or AXIS_COLORS[piece["axis"]]


def _piece_label(piece):
    prefix = {"cattle": "C", "wolf": "W",
              "pig": ("P" if piece.get("awake") else "ZP"), "rocket": "R",
              "goat": "G",
              "elephant": "E", "pink_sheep": "L",
              "bomb": f"B{piece.get('hits_remaining', 3)}-"}.get(
                  piece.get("species"), "")
    label = f"{prefix}{piece['id']}" if prefix else str(piece["id"])
    return f"?{label}" if piece.get("review") else label


def _blend_poly(img, pts, color, alpha):
    layer = img.copy()
    cv2.fillPoly(layer, [np.array(pts, np.int32)], color)
    cv2.addWeighted(layer, alpha, img, 1.0 - alpha, 0, img)


def _draw_soft_poly(img, pts, color, alpha=0.14):
    pts = np.array(pts, np.int32)
    _blend_poly(img, pts, color, alpha)
    cv2.polylines(img, [pts], True, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)


def _draw_arrow(img, p0, p1, color, thickness=3):
    p0 = tuple(map(int, p0))
    p1 = tuple(map(int, p1))
    cv2.arrowedLine(img, p0, p1, (255, 255, 255), thickness + 5, cv2.LINE_AA, tipLength=0.32)
    cv2.arrowedLine(img, p0, p1, INK, thickness + 2, cv2.LINE_AA, tipLength=0.32)
    cv2.arrowedLine(img, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.32)
    cv2.circle(img, p1, max(4, thickness + 1), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, p1, max(2, thickness), color, -1, cv2.LINE_AA)


def _piece_arrow_points(piece):
    """Return a centered arrow in rectified-grid pixels.

    Elephant footprints span three cells in their travel direction.  Their
    arrow occupies only the forward part of the body so it stays readable and
    does not cut through the central id badge.
    """
    cells = [tuple(cell) for cell in piece.get("cells") or []]
    facing = str(piece.get("facing") or "")
    delta = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}.get(facing)
    if not cells or delta is None:
        return None
    dr, dc = delta
    projections = [row * dr + col * dc for row, col in cells]
    lo, hi = min(projections), max(projections)
    rump_cells = [cell for cell, value in zip(cells, projections) if value == lo]
    head_cells = [cell for cell, value in zip(cells, projections) if value == hi]

    def center(group):
        return np.array([
            sum((col + 0.5) * CELL for _row, col in group) / len(group),
            sum((row + 0.5) * CELL for row, _col in group) / len(group),
        ], dtype=np.float64)

    start, end = center(rump_cells), center(head_cells)
    if piece.get("species") == "elephant":
        vector = end - start
        start, end = start + vector * 0.46, start + vector * 0.88
    return tuple(start), tuple(end)


def _round_rect(img, p0, p1, radius, fill, border=None, alpha=1.0):
    x0, y0 = map(int, p0)
    x1, y1 = map(int, p1)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1] - 1, x1), min(img.shape[0] - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return
    radius = int(min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
    target = img.copy() if alpha < 1.0 else img

    cv2.rectangle(target, (x0 + radius, y0), (x1 - radius, y1), fill, -1, cv2.LINE_AA)
    cv2.rectangle(target, (x0, y0 + radius), (x1, y1 - radius), fill, -1, cv2.LINE_AA)
    for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x1 - radius, y1 - radius), (x0 + radius, y1 - radius)):
        cv2.circle(target, (cx, cy), radius, fill, -1, cv2.LINE_AA)

    if border is not None:
        cv2.rectangle(target, (x0 + radius, y0), (x1 - radius, y1), border, 2, cv2.LINE_AA)
        cv2.rectangle(target, (x0, y0 + radius), (x1, y1 - radius), border, 2, cv2.LINE_AA)
        for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                       (x1 - radius, y1 - radius), (x0 + radius, y1 - radius)):
            cv2.circle(target, (cx, cy), radius, border, 2, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(target, alpha, img, 1.0 - alpha, 0, img)


def _draw_badge(img, top_left, text, color):
    x, y = tuple(map(int, top_left))
    scale = 0.36 if len(text) < 2 else 0.32
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    width = max(23, tw + 12)
    height = 18
    x = min(max(4, x), img.shape[1] - width - 4)
    y = min(max(4, y), img.shape[0] - height - 4)
    _round_rect(img, (x + 1, y + 2), (x + width + 1, y + height + 2), 9, (0, 0, 0), alpha=0.18)
    _round_rect(img, (x, y), (x + width, y + height), 9, (255, 255, 255), border=color)
    cv2.putText(img, text, (x + width // 2 - tw // 2, y + height // 2 + th // 2 - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, INK, 1, cv2.LINE_AA)


def _draw_text_with_halo(img, text, org, scale, color=INK, thickness=1):
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def render(game, sheep, corners, rows, cols):
    """Overlay occupied cells and head arrows on the source image."""
    rect, Minv, src = _warp(game, corners, rows, cols)
    ov = game.copy()
    for s in sheep:
        col = _piece_color(s)
        for row, cell_col in s["cells"]:
            poly = [
                _to_px(Minv, cell_col, row),
                _to_px(Minv, cell_col + 1, row),
                _to_px(Minv, cell_col + 1, row + 1),
                _to_px(Minv, cell_col, row + 1),
            ]
            _draw_soft_poly(ov, poly, col)
        arrow = _piece_arrow_points(s)
        if arrow:
            p0 = tuple(map(int, _to_px(Minv, arrow[0][0] / CELL, arrow[0][1] / CELL)))
            p1 = tuple(map(int, _to_px(Minv, arrow[1][0] / CELL, arrow[1][1] / CELL)))
            _draw_arrow(ov, p0, p1, col, thickness=2)
    for r in range(int(rows) + 1):
        cv2.line(ov, tuple(map(int, _to_px(Minv, 0, r))), tuple(map(int, _to_px(Minv, cols, r))), GRID_COLOR, 1, cv2.LINE_AA)
    for c in range(int(cols) + 1):
        cv2.line(ov, tuple(map(int, _to_px(Minv, c, 0))), tuple(map(int, _to_px(Minv, c, rows))), GRID_COLOR, 1, cv2.LINE_AA)
    return ov, src


def render_rect_debug(debug, sheep):
    grid: G.BoardGrid = debug["grid"]
    vis = debug["rect"].copy()
    grid_layer = vis.copy()
    for r in range(grid.rows + 1):
        cv2.line(grid_layer, (0, r * CELL), (grid.cols * CELL, r * CELL), GRID_COLOR, 1, cv2.LINE_AA)
    for c in range(grid.cols + 1):
        cv2.line(grid_layer, (c * CELL, 0), (c * CELL, grid.rows * CELL), GRID_COLOR, 1, cv2.LINE_AA)
    cv2.addWeighted(grid_layer, 0.22, vis, 0.78, 0, vis)
    gesture_mask = debug.get("gesture_mask")
    if isinstance(gesture_mask, np.ndarray) and np.any(gesture_mask):
        tint = np.zeros_like(vis)
        tint[:] = FOCUS_RED
        selected = gesture_mask > 0
        vis[selected] = (0.55 * vis[selected] + 0.45 * tint[selected]).astype(np.uint8)
        contours, _ = cv2.findContours(gesture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, FOCUS_RED, 2, cv2.LINE_AA)
    for hz in debug.get("hazards", []):
        row, col = int(hz["row"]), int(hz["col"])
        pts = [
            (col * CELL + 5, row * CELL + 5),
            ((col + 1) * CELL - 5, row * CELL + 5),
            ((col + 1) * CELL - 5, (row + 1) * CELL - 5),
            (col * CELL + 5, (row + 1) * CELL - 5),
        ]
        _draw_soft_poly(vis, pts, HAZARD_COLOR, alpha=0.20)
        _draw_badge(vis, (col * CELL + 12, row * CELL + 12), "W", HAZARD_COLOR)
    for s in sheep:
        color = _piece_color(s)
        for row, col in s["cells"]:
            pts = [
                (col * CELL + 7, row * CELL + 7),
                ((col + 1) * CELL - 7, row * CELL + 7),
                ((col + 1) * CELL - 7, (row + 1) * CELL - 7),
                (col * CELL + 7, (row + 1) * CELL - 7),
            ]
            _draw_soft_poly(vis, pts, color)
        arrow = _piece_arrow_points(s)
        if arrow:
            _draw_arrow(vis, arrow[0], arrow[1], color, thickness=2)
        min_row = min(row for row, _ in s["cells"])
        min_col = min(col for _, col in s["cells"])
        _draw_badge(vis, (min_col * CELL + 7, min_row * CELL + 7), _piece_label(s), color)
    return vis


def render_grid_labels(debug, sheep):
    """Board audit image with spreadsheet-like coordinates and sheep ids."""
    grid: G.BoardGrid = debug["grid"]
    board = render_rect_debug(debug, sheep)
    top, left, pad = 42, 48, 10
    h, w = board.shape[:2]
    canvas = np.full((top + h + pad, left + w + pad, 3), (245, 247, 248), np.uint8)
    canvas[top:top + h, left:left + w] = board

    # Header bands.
    cv2.rectangle(canvas, (left, 0), (left + w, top - 1), (238, 242, 244), -1)
    cv2.rectangle(canvas, (0, top), (left - 1, top + h), (238, 242, 244), -1)
    cv2.line(canvas, (left, top - 1), (left + w, top - 1), (194, 202, 210), 1, cv2.LINE_AA)
    cv2.line(canvas, (left - 1, top), (left - 1, top + h), (194, 202, 210), 1, cv2.LINE_AA)

    for c in range(grid.cols):
        label = chr(ord("A") + c) if c < 26 else f"C{c + 1}"
        cx = left + c * CELL + CELL // 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        _draw_text_with_halo(canvas, label, (cx - tw // 2, 27), 0.58, INK, 1)
    for r in range(grid.rows):
        label = str(r + 1)
        cy = top + r * CELL + CELL // 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        _draw_text_with_halo(canvas, label, (left - 14 - tw, cy + th // 2), 0.48, INK, 1)

    _draw_text_with_halo(canvas, "A1", (12, 27), 0.5, (80, 84, 92), 1)
    return canvas


def render_layout(debug, sheep):
    """Synthetic 2D layout audit, independent of screenshot texture/skin."""
    grid: G.BoardGrid = debug["grid"]
    h, w = grid.rows * CELL, grid.cols * CELL
    board = np.full((h, w, 3), (234, 222, 183), np.uint8)
    for r in range(grid.rows):
        for c in range(grid.cols):
            color = (188, 178, 71) if (r + c) % 2 == 0 else (247, 180, 109)
            cv2.rectangle(board, (c * CELL, r * CELL), ((c + 1) * CELL, (r + 1) * CELL), color, -1)
    grid_layer = board.copy()
    for r in range(grid.rows + 1):
        cv2.line(grid_layer, (0, r * CELL), (w, r * CELL), (255, 255, 255), 1, cv2.LINE_AA)
    for c in range(grid.cols + 1):
        cv2.line(grid_layer, (c * CELL, 0), (c * CELL, h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(grid_layer, 0.28, board, 0.72, 0, board)

    debug2 = {**debug, "rect": board}
    return render_grid_labels(debug2, sheep)


def render_segments(debug):
    markers = debug["markers"]
    rect = debug["rect"]
    out = rect.copy()
    rng = np.random.default_rng(12)
    ids = [int(x) for x in np.unique(markers) if x > 0]
    for idv in ids:
        if idv == max(ids):
            continue
        color = rng.integers(40, 230, size=3, dtype=np.uint8).tolist()
        mask = markers == idv
        out[mask] = (0.55 * out[mask] + 0.45 * np.array(color)).astype(np.uint8)
    return out


def _remove_obsolete_images():
    for name in (
        "_occ_axis.png",
        "_occ_axis_zoom.png",
        "_sheep_face_mask.png",
        "_sheep_mask.png",
        "_sheep_segments.png",
    ):
        path = image_path(name)
        if path.exists():
            path.unlink()
