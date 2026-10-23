"""
Helpers to capture the Graphwar client area via Win32 (no hardcoded screen coords).
"""

import ctypes
import json
from pathlib import Path

import cv2
import mss
import numpy as np
import win32gui

DEFAULT_GAME_WINDOW_NAME = "Graphwar"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "capture_config.json"

DEFAULT_MARGINS = {
    "margin_left": 15,
    "margin_top": 15,
    "margin_right": 15,
    "margin_bottom": 135,
}

_CROP_CACHE = {}


def _set_process_dpi_awareness():
    """Keep Win32 window coordinates in the same pixel space as mss."""
    try:
        # Match mss: use per-monitor DPI awareness on modern Windows.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        # Fallback for older Windows versions without shcore.dll.
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


# mss normally sets this when its first capture object is created. Set it before
# the first GetWindowRect/ClientToScreen call so the first screenshot is aligned.
_set_process_dpi_awareness()


def load_capture_margins(path=CONFIG_PATH):
    if not path.exists():
        return DEFAULT_MARGINS.copy()
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    margins = data.get("margins", DEFAULT_MARGINS)
    return {key: int(margins.get(key, DEFAULT_MARGINS[key])) for key in DEFAULT_MARGINS}


def save_capture_margins(margins, path=CONFIG_PATH):
    payload = {"margins": {key: int(margins[key]) for key in DEFAULT_MARGINS}}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def find_game_window(title=DEFAULT_GAME_WINDOW_NAME):
    hwnd = win32gui.FindWindow(None, title)
    return hwnd if hwnd else None


def get_client_screen_rect(hwnd):
    """
    Client area of the window in absolute screen coordinates.

    Returns:
        (left, top, width, height)
    """
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    _, _, right, bottom = win32gui.GetClientRect(hwnd)
    return left, top, right, bottom


def get_capture_field(hwnd, margins=None):
    """
    Build an mss-compatible region dict from the window client area.

    margins keys (pixels inside the client, not screen):
        margin_left, margin_top, margin_right, margin_bottom
    """
    cx, cy, cw, ch = get_client_screen_rect(hwnd)
    margins = margins or {}

    ml = int(margins.get("margin_left", 0))
    mt = int(margins.get("margin_top", 0))
    mr = int(margins.get("margin_right", 0))
    mb = int(margins.get("margin_bottom", 0))

    width = cw - ml - mr
    height = ch - mt - mb

    if width < 1 or height < 1:
        raise ValueError(
            f"Invalid capture region: client {cw}x{ch}, margins L{ml} T{mt} R{mr} B{mb}"
        )

    return {
        "left": cx + ml,
        "top": cy + mt,
        "width": width,
        "height": height,
    }


def auto_crop_white_field(hwnd, fallback_field, use_cache=True):
    """
    Lock onto the interior white battle canvas bounded by the 1px black border,
    completely excluding the outer greenish-gray Java UI panel (g - r > 12 and g - b > 15).
    """
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        pt = win32gui.ClientToScreen(hwnd, (0, 0))
        cw, ch = right - left, bottom - top
        if cw <= 200 or ch <= 200:
            return fallback_field

        cache_key = (int(hwnd), cw, ch)
        if use_cache and cache_key in _CROP_CACHE:
            rel_x, rel_y, w, h = _CROP_CACHE[cache_key]
            return {
                "left": pt[0] + rel_x,
                "top": pt[1] + rel_y,
                "width": w,
                "height": h,
            }

        client_rect = {"left": pt[0], "top": pt[1], "width": cw, "height": ch}
        with mss.mss() as sct:
            img = np.array(sct.grab(client_rect))

        b = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        r = img[:, :, 2].astype(np.int16)

        # Outer Graphwar UI panel is greenish-gray
        is_green_panel = (g - r > 12) & (g - b > 15) & (g > 100) & (g < 230)
        not_panel = (~is_green_panel).astype(np.uint8)
        row_canvas_score = np.mean(not_panel, axis=1)

        y1, y2 = None, None
        for y in range(int(ch * 0.90)):
            if row_canvas_score[y] > 0.85:
                if y1 is None:
                    y1 = y
                y2 = y
            elif y1 is not None and (y - y1) > int(ch * 0.45):
                break

        if y1 is not None and y2 is not None and (y2 - y1) > int(ch * 0.50):
            col_canvas_score = np.mean(not_panel[y1:y2, :], axis=0)
            valid_x = np.where(col_canvas_score > 0.85)[0]
            if len(valid_x) > int(cw * 0.80):
                x1, x2 = int(valid_x[0]), int(valid_x[-1])
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                # Step inward past the 1px black border line without shaving white canvas pixels
                for _ in range(2):
                    if x1 < x2 and np.mean(gray[y1 : y2 + 1, x1]) < 120:
                        x1 += 1
                    if x2 > x1 and np.mean(gray[y1 : y2 + 1, x2]) < 120:
                        x2 -= 1
                    if y1 < y2 and np.mean(gray[y1, x1 : x2 + 1]) < 120:
                        y1 += 1
                    if y2 > y1 and np.mean(gray[y2, x1 : x2 + 1]) < 120:
                        y2 -= 1

                w = x2 - x1 + 1
                h = y2 - y1 + 1
                _CROP_CACHE[cache_key] = (x1, y1, w, h)
                return {
                    "left": pt[0] + x1,
                    "top": pt[1] + y1,
                    "width": w,
                    "height": h,
                }
    except Exception:
        pass

    safe_fallback = dict(fallback_field)
    if safe_fallback.get("height", 0) > 500:
        safe_fallback["height"] = int(safe_fallback["height"] * 0.90)
    return safe_fallback
