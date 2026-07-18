"""按窗口标题截图（不受其他窗口遮挡），专为微信小程序「套住那只羊」。
微信小程序是 Chromium(Chrome_WidgetWin_0) GPU 渲染窗口，普通 PrintWindow 会黑屏，
必须用 PW_RENDERFULLCONTENT=2；若仍黑则回退到屏幕坐标 BitBlt（需窗口前台不被遮挡）。

用法:
    py scripts/capture_window.py                      # 默认抓「套住那只羊」-> images/_game.png
    py scripts/capture_window.py --title 套住那只羊 --out images/_game.png
返回的 images/_game.png 可直接喂给识别脚本。
"""
import argparse
import ctypes
import os
from ctypes import wintypes
from pathlib import Path
import numpy as np
import cv2
from paths import image_path

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
user32.SetProcessDPIAware()                      # 高 DPI 下拿真实像素，避免坐标缩水

PW_RENDERFULLCONTENT = 2
SRCCOPY = 0x00CC0020


def find_window(title):
    """精确标题优先，找不到再做子串匹配。"""
    current_pid = os.getpid()
    def acceptable(hwnd):
        if not hwnd or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return False
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if int(pid.value) == current_pid:
            return False
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False
        return rect.right - rect.left >= 320 and rect.bottom - rect.top >= 480

    hwnd = user32.FindWindowW(None, title)
    if acceptable(hwnd):
        return hwnd
    found = []
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(h, _):
        if acceptable(h):
            n = user32.GetWindowTextLengthW(h)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(h, buf, n + 1)
                if title in buf.value:
                    # Prefer the closest title, then the largest viable window.
                    rect = wintypes.RECT()
                    user32.GetWindowRect(h, ctypes.byref(rect))
                    found.append((0 if buf.value == title else 1, len(buf.value),
                                  -(rect.right - rect.left) * (rect.bottom - rect.top), h))
        return True
    user32.EnumWindows(cb, 0)
    return sorted(found)[0][-1] if found else None


def list_windows(title=""):
    """Return viable external windows for explicit UI target selection."""
    current_pid = os.getpid()
    found = []
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if not length:
            return True
        text = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, text, length + 1)
        if title and title not in text.value:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if int(pid.value) == current_pid:
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width < 320 or height < 480:
            return True
        class_name = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_name, len(class_name))
        found.append({"hwnd": str(int(hwnd)), "title": text.value, "pid": int(pid.value),
                      "class_name": class_name.value, "width": width, "height": height})
        return True
    user32.EnumWindows(cb, 0)
    return sorted(found, key=lambda item: (0 if item["title"] == title else 1,
                                            len(item["title"]), -item["width"] * item["height"]))


def grab(hwnd):
    """优先 PrintWindow(PW_RENDERFULLCONTENT)，黑屏则回退 BitBlt 屏幕。返回 BGR ndarray。"""
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        raise RuntimeError("窗口尺寸异常，可能已最小化")

    hdc = user32.GetWindowDC(hwnd)
    memdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(memdc, bmp)

    ok = user32.PrintWindow(hwnd, memdc, PW_RENDERFULLCONTENT)
    img = _bmp_to_array(memdc, bmp, w, h)

    # 黑屏检测：PrintWindow 对部分 GPU 窗口仍失败，回退到屏幕 BitBlt
    if not ok or img is None or img.mean() < 3:
        scr = user32.GetDC(0)
        gdi32.BitBlt(memdc, 0, 0, w, h, scr, rect.left, rect.top, SRCCOPY)
        img = _bmp_to_array(memdc, bmp, w, h)
        user32.ReleaseDC(0, scr)
        mode = "BitBlt(屏幕回退)"
    else:
        mode = "PrintWindow(PW_RENDERFULLCONTENT)"

    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(memdc)
    user32.ReleaseDC(hwnd, hdc)
    return img, (rect.left, rect.top, w, h), mode


def _bmp_to_array(memdc, bmp, w, h):
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]
    bi = BITMAPINFOHEADER()
    bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bi.biWidth, bi.biHeight = w, -h          # 负高 = 自上而下
    bi.biPlanes, bi.biBitCount = 1, 32
    buf = ctypes.create_string_buffer(w * h * 4)
    if not gdi32.GetDIBits(memdc, bmp, 0, h, buf, ctypes.byref(bi), 0):
        return None
    arr = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
    return arr[:, :, :3].copy()              # BGRA -> BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="套住那只羊")
    ap.add_argument("--out", default=str(image_path("_game.png")))
    args = ap.parse_args()

    hwnd = find_window(args.title)
    if not hwnd:
        raise SystemExit(f"找不到窗口: {args.title}")
    img, (x, y, w, h), mode = grab(hwnd)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"hwnd=0x{hwnd:X}  位置=({x},{y})  尺寸={w}x{h}  方式={mode}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
