
import numpy as np
import vision as D


class GridGeometryOps:
    """Mixin: grid-to-pixel geometry helpers."""

    def _px(self, gx, gy, Minv=None):
        matrix = self.Minv if Minv is None else Minv
        v = matrix @ np.array([gx * D.CELL, gy * D.CELL, 1.0])
        return [float(v[0] / v[2]), float(v[1] / v[2])]

    def _cell_center(self, r, c, Minv=None):
        return self._px(c + 0.5, r + 0.5, Minv=Minv)

    def _cell_poly(self, r, c, Minv=None):
        return [
            self._px(c, r, Minv=Minv),
            self._px(c + 1, r, Minv=Minv),
            self._px(c + 1, r + 1, Minv=Minv),
            self._px(c, r + 1, Minv=Minv),
        ]

    @staticmethod
    def _cell_label(cell):
        r, c = int(cell[0]), int(cell[1])
        return f"{chr(65 + c) if c < 26 else c + 1}{r + 1}"

    def _grid_lines(self, rows, cols):
        grid = []
        for r in range(rows + 1):
            grid.append([self._px(0, r), self._px(cols, r)])
        for c in range(cols + 1):
            grid.append([self._px(c, 0), self._px(c, rows)])
        return grid
