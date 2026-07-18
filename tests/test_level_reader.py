import os
from pathlib import Path

import cv2

from levels import reader as level_reader


ROOT = Path(__file__).resolve().parents[1]


def test_tesseract_process_is_silent_on_windows():
    options = level_reader._silent_process_options()
    if os.name != "nt":
        assert options == {}
        return
    assert options["creationflags"] & level_reader.subprocess.CREATE_NO_WINDOW
    assert options["startupinfo"].dwFlags & level_reader.subprocess.STARTF_USESHOWWINDOW
    assert options["startupinfo"].wShowWindow == level_reader.subprocess.SW_HIDE


def _assert_level(path: Path, expected: str):
    if not path.exists() or level_reader._tesseract_command() is None:
        return
    reading = level_reader.read_level(cv2.imread(str(path)))
    assert reading is not None
    assert reading.label == expected, reading


def test_reads_archived_level_117_title():
    # images/_game.png is a mutable runtime artifact and may be any level.
    # Regression fixtures must come from the immutable source archive.
    _assert_level(ROOT / "cache" / "source_levels" / "source-d0e2dbb0bac1" /
                  "source.png", "117")


def test_reads_archived_level_116_title():
    _assert_level(ROOT / "cache" / "source_levels" / "source-683e725c03a8" /
                  "source.png", "116")
