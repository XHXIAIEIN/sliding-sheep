"""Read the numeric ``第 XXX 关`` title from a captured game frame."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LevelReading:
    label: str
    bbox: tuple[int, int, int, int]
    raw_text: str
    method: str = "screen-title-tesseract"


def _tesseract_command() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", ""), "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        r"D:\Program Files\bin\Tesseract-OCR\tesseract.exe",
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def _silent_process_options() -> dict:
    """Keep console-based OCR tools invisible when the GUI runs on Windows."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }


def _digit_components(image: np.ndarray):
    """Return white-fill components belonging to the outlined title digits."""
    height, width = image.shape[:2]
    y0, y1 = int(height * 0.045), int(height * 0.16)
    x0, x1 = int(width * 0.28), int(width * 0.72)
    roi = image[y0:y1, x0:x1]
    mask = cv2.inRange(
        roi, np.array((205, 205, 205), dtype=np.uint8),
        np.array((255, 255, 255), dtype=np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)

    pieces = []
    min_h, max_h = height * 0.016, height * 0.025
    min_w, max_w = width * 0.007, width * 0.036
    for component in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[component])
        if not (min_h <= h <= max_h and min_w <= w <= max_w):
            continue
        if area < max(45, int(height * width * 0.00007)):
            continue
        if not (height * 0.057 <= y0 + y <= height * 0.13):
            continue
        pieces.append((x, y, w, h, area, component))

    if not pieces:
        return labels, []
    pieces.sort(key=lambda item: item[0])
    runs = []
    current = [pieces[0]]
    for piece in pieces[1:]:
        previous = current[-1]
        gap = piece[0] - (previous[0] + previous[2])
        aligned = abs((piece[1] + piece[3] / 2) -
                      (previous[1] + previous[3] / 2)) <= height * 0.009
        if aligned and -width * 0.004 <= gap <= width * 0.025:
            current.append(piece)
        else:
            runs.append(current)
            current = [piece]
    runs.append(current)
    run = max(runs, key=lambda items: (min(len(items), 4), sum(item[4] for item in items)))
    return labels, run if 1 <= len(run) <= 4 else []


def read_level(image: np.ndarray) -> LevelReading | None:
    """Recognize the visible numeric level, returning ``None`` when uncertain."""
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 300:
        return None
    executable = _tesseract_command()
    if not executable:
        return None

    labels, pieces = _digit_components(image)
    if not pieces:
        return None

    glyphs = []
    for x, y, w, h, _area, component in pieces:
        glyph = (labels[y:y + h, x:x + w] == component).astype(np.uint8) * 255
        glyph = cv2.resize(glyph, (max(18, round(w * 96 / h)), 96),
                           interpolation=cv2.INTER_NEAREST)
        glyphs.append(cv2.copyMakeBorder(glyph, 18, 18, 16, 16,
                                         cv2.BORDER_CONSTANT, value=0))
    strip = np.concatenate(glyphs, axis=1)
    ok, encoded = cv2.imencode(".png", strip)
    if not ok:
        return None
    try:
        result = subprocess.run(
            [executable, "stdin", "stdout", "--psm", "7", "-l", "eng",
             "-c", "tessedit_char_whitelist=0123456789"],
            input=encoded.tobytes(), capture_output=True, timeout=4, check=False,
            **_silent_process_options())
    except (OSError, subprocess.SubprocessError):
        return None
    raw = result.stdout.decode("utf-8", errors="ignore").strip()
    match = re.fullmatch(r"\s*(\d{1,4})\s*", raw)
    if result.returncode or not match:
        return None

    x0 = min(item[0] for item in pieces)
    y0 = min(item[1] for item in pieces)
    x1 = max(item[0] + item[2] for item in pieces)
    y1 = max(item[1] + item[3] for item in pieces)
    image_h, image_w = image.shape[:2]
    roi_x, roi_y = int(image_w * 0.28), int(image_h * 0.045)
    return LevelReading(match.group(1), (roi_x + x0, roi_y + y0, x1 - x0, y1 - y0), raw)
