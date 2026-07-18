
import os
import json
import base64
import cv2
import numpy as np
from board import grid as G
from core import safety
from . import common
from .common import _wrap


class CalibrationOps:
    """Mixin: grid calibration load/save/preview and seeding."""

    def load_params(self):
        """启动/重置时读当前目录的 grid_params.json（没有则返回 None）。"""
        def run():
            p = os.path.join(common.HERE, "grid_params.json")
            if not os.path.exists(p):
                return {"params": None}
            return {"params": json.load(open(p, encoding="utf-8"))}
        return _wrap(run)

    def calibration_preview(self, corners, rows, cols):
        """Project calibration lines with the detector's authoritative homography."""
        def run():
            if self.game is None:
                raise RuntimeError("请先截图")
            rows_value, cols_value = int(rows), int(cols)
            if rows_value < 2 or cols_value < 2:
                raise RuntimeError("棋盘行列数不能小于 2")
            grid = G.BoardGrid(
                rows=rows_value,
                cols=cols_value,
                corners={key: [float(corners[key][0]), float(corners[key][1])]
                         for key in G.CORNER_KEYS},
                image_size=(int(self.game.shape[1]), int(self.game.shape[0])),
            )
            lines = []
            width, height = grid.rect_size
            for row in range(rows_value + 1):
                y = row * grid.cell
                lines.append([list(grid.rect_to_source(0, y)),
                              list(grid.rect_to_source(width, y))])
            for col in range(cols_value + 1):
                x = col * grid.cell
                lines.append([list(grid.rect_to_source(x, 0)),
                              list(grid.rect_to_source(x, height))])
            return {"grid": lines}
        return _wrap(run)

    def editor_grid(self):
        """Expose one atomic frame + board snapshot for visual manual editing."""
        def run():
            if self.game is None or self.Minv is None or self.board is None:
                raise RuntimeError("请先采集并分析棋盘")
            ok, encoded = cv2.imencode(".png", self.game)
            if not ok:
                raise RuntimeError("人工校验截图编码失败")
            height, width = self.game.shape[:2]
            cells = []
            for row in range(self.board.rows):
                for col in range(self.board.cols):
                    cells.append({
                        "row": row,
                        "col": col,
                        "poly": self._cell_poly(row, col),
                        "center": self._cell_center(row, col),
                    })
            return {"rows": self.board.rows, "cols": self.board.cols,
                    "img": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    "image_size": [int(width), int(height)],
                    "state": self._snapshot(self.board, highlight=None),
                    "board_revision": self.board_revision,
                    "cells": cells, "grid": self._grid_lines(self.board.rows, self.board.cols),
                    "can_undo": bool(self._editor_undo), "can_redo": bool(self._editor_redo),
                    "manual_pending": bool(self._manual_edit_pending)}
        return _wrap(run)

    def save_params(self, corners, rows, cols, locked=None):
        """调参器保存：四角(截图像素坐标) + 行列 + 标定时分辨率 -> 写 grid_params.json。
        记下 imgW/imgH，换分辨率时可把这份四角按比例缩放成新起点。"""
        def run():
            h, w = self.game.shape[:2] if self.game is not None else (0, 0)
            P = {"corners": {k: [round(float(corners[k][0]), 1), round(float(corners[k][1]), 1)]
                             for k in ("TL", "TR", "BR", "BL")},
                 "rows": int(rows), "cols": int(cols),
                 "imgW": int(w), "imgH": int(h),
                 "nudge": [0, 0],
                 "locked": [key for key in (locked or []) if key in G.CORNER_KEYS],
                 "image": "images/_game.png"}
            blockers, _warnings = safety.validate_calibration(P, self.game.shape if self.game is not None else (h, w, 3))
            if blockers:
                raise RuntimeError("；".join(item["message"] for item in blockers))
            json.dump(P, open(os.path.join(common.HERE, "grid_params.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            self._frame_history.clear()
            return {"saved": True}
        return _wrap(run)

    def _sheep_centers(self):
        """白羊掩膜 -> 距离变换峰 -> 大半径 NMS：每只羊一个中心(像素)。"""
        g = self.game
        H, W = g.shape[:2]
        hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
        Sc, Vc = hsv[:, :, 1], hsv[:, :, 2]
        white = ((Sc < 70) & (Vc > 150)).astype(np.uint8)
        white[:int(0.13 * H)] = 0; white[int(0.80 * H):] = 0
        white[:, :int(0.16 * W)] = 0; white[:, int(0.84 * W):] = 0
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        dt = cv2.distanceTransform(white, cv2.DIST_L2, 5)
        md = 46
        dil = cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (md, md)))
        ys, xs = np.where((dt == dil) & (dt > 7))
        order = sorted(zip(xs.tolist(), ys.tolist(), dt[ys, xs].tolist()), key=lambda t: -t[2])
        cen = []
        for x, y, _v in order:
            if all((x - a) ** 2 + (y - b) ** 2 > md * md for a, b in cen):
                cen.append((x, y))
        return cen

    def _guess_from_sheep(self):
        """无任何标定时的兜底：羊中心 minAreaRect 粗估四角+行列（已知偏窄，仅作起点）。"""
        cen = self._sheep_centers()
        if len(cen) < 4:
            raise RuntimeError("羊中心太少，无法估计")
        C = np.array(cen, np.float32)
        box = cv2.boxPoints(cv2.minAreaRect(C))
        ssum = box.sum(1); sdif = box[:, 1] - box[:, 0]
        TL = box[np.argmin(ssum)]; BR = box[np.argmax(ssum)]
        TR = box[np.argmin(sdif)]; BL = box[np.argmax(sdif)]
        ctr = box.mean(0)
        def expand(p):
            return [round(float(p[0] + (p[0] - ctr[0]) * 0.10), 1),
                    round(float(p[1] + (p[1] - ctr[1]) * 0.10), 1)]
        corners = {k: expand(p) for k, p in [("TL", TL), ("TR", TR), ("BR", BR), ("BL", BL)]}
        nn = []
        for i in range(len(C)):
            dd = np.hypot(*(C - C[i]).T); dd[i] = 1e9; nn.append(float(dd.min()))
        pitch = float(np.median(nn)) if nn else 65.0
        ex = (TR - TL) / (np.linalg.norm(TR - TL) + 1e-9)
        ey = (BL - TL) / (np.linalg.norm(BL - TL) + 1e-9)
        pc = (C - TL) @ ex; pr = (C - TL) @ ey
        cols = max(2, int(round((pc.max() - pc.min()) / pitch)) + 1)
        rows = max(2, int(round((pr.max() - pr.min()) / pitch)) + 1)
        return {"corners": corners, "rows": rows, "cols": cols}

    def seed_params(self):
        """校准起点：优先把已存标定按分辨率缩放到当前截图；没有标定才退回羊中心粗估。"""
        def run():
            if self.game is None:
                raise RuntimeError("请先截图")
            h, w = self.game.shape[:2]
            centers = [[int(x), int(y)] for x, y in self._sheep_centers()]
            p = os.path.join(common.HERE, "grid_params.json")
            if os.path.exists(p):
                P = json.load(open(p, encoding="utf-8"))
                cor, ow, oh = P.get("corners"), P.get("imgW"), P.get("imgH")
                if cor and ow and oh:                    # 按比例缩放上次标定
                    sx, sy = w / ow, h / oh
                    sc = {k: [round(cor[k][0] * sx, 1), round(cor[k][1] * sy, 1)]
                          for k in ("TL", "TR", "BR", "BL")}
                    return {"corners": sc, "rows": P.get("rows", 18), "cols": P.get("cols", 12),
                            "locked": P.get("locked", []),
                            "centers": centers, "mode": "scaled",
                            "fromRes": [ow, oh], "toRes": [w, h]}
                if cor:                                  # 有标定但旧版没存分辨率：原样
                    return {"corners": cor, "rows": P.get("rows", 18), "cols": P.get("cols", 12),
                            "locked": P.get("locked", []),
                            "centers": centers, "mode": "asis"}
            g = self._guess_from_sheep()                 # 无标定兜底
            g.update({"centers": centers, "mode": "guess", "locked": []})
            return g
        return _wrap(run)
