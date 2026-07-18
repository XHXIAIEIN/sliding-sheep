"""Shared project paths for scripts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images"
DATA_DIR = ROOT / "data"

# 运行产物（强刷会清理）
BOARD_JSON = DATA_DIR / "board.json"
BOARD_GRID_JSON = DATA_DIR / "board_grid.json"
BOARD_LAYOUT_JSON = DATA_DIR / "board_layout.json"
SCENE_REPORT_JSON = DATA_DIR / "scene_report.json"
SHEEP_CANDIDATES_JSON = DATA_DIR / "sheep_candidates.json"

# 持久配置（强刷保留）
GRID_PARAMS_JSON = DATA_DIR / "grid_params.json"
RETRY_CONTROLS_JSON = DATA_DIR / "retry_controls.json"
RUNTIME_SETTINGS_JSON = DATA_DIR / "runtime_settings.json"


def image_path(name: str) -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGE_DIR / name


def data_path(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name
