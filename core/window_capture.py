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
    Scales default 800x600 margins proportionally if the window is DPI-scaled.

    margins keys (pixels inside the client, not screen):
        margin_left, margin_top, margin_right, margin_bottom
    """
    cx, cy, cw, ch = get_client_screen_rect(hwnd)
    margins = margins or {}

    scale_x = cw / 800.0 if cw > 200 else 1.0
    scale_y = ch / 600.0 if ch > 200 else 1.0

    ml = int(round(int(margins.get("margin_left", 0)) * scale_x))
    mt = int(round(int(margins.get("margin_top", 0)) * scale_y))
    mr = int(round(int(margins.get("margin_right", 0)) * scale_x))
    mb = int(round(int(margins.get("margin_bottom", 0)) * scale_y))

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


def auto_crop_white_field(hwnd, fallback_field, sct=None):
    """
    Lock onto the interior white battle canvas bounded by the 1px black border,
    completely excluding the outer greenish-gray Java UI panel (g - r > 12 and g - b > 15).
    If the window is temporarily covered (e.g. by the terminal during setup),
    falls back to the exact DPI-scaled GraphPlane rectangle (16..784, 16..464 in 800x600).
    """
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        pt = win32gui.ClientToScreen(hwnd, (0, 0))
        cw, ch = right - left, bottom - top
        if cw <= 200 or ch <= 200:
            return fallback_field

        # Exact proportional GraphPlane interior fallback (never includes bottom UI even at 125%/150% DPI)
        prop_x1 = int(round(16.0 * cw / 800.0))
        prop_y1 = int(round(16.0 * ch / 600.0))
        prop_w = int(round(768.0 * cw / 800.0))
        prop_h = int(round(448.0 * ch / 600.0))
        proportional_fallback = {
            "left": pt[0] + prop_x1,
            "top": pt[1] + prop_y1,
            "width": prop_w,
            "height": prop_h,
        }

        client_rect = {"left": pt[0], "top": pt[1], "width": cw, "height": ch}
        if sct is not None:
            img = np.array(sct.grab(client_rect))
        else:
            with mss.mss() as temp_sct:
                img = np.array(temp_sct.grab(client_rect))

        b = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        r = img[:, :, 2].astype(np.int16)

        # Outer Graphwar UI panel is greenish-gray
        is_green_panel = (g - r > 12) & (g - b > 15) & (g > 100) & (g < 230)

        # Verify that the Graphwar green UI border is actually visible (not covered by terminal/IDE)
        top_green_ratio = float(np.mean(is_green_panel[2 : max(4, int(ch * 0.02)), :]))
        mid_bottom_y = int(ch * 0.782)
        bot_green_ratio = float(
            np.mean(is_green_panel[mid_bottom_y : min(ch, mid_bottom_y + max(4, int(ch * 0.015))), :])
        )
        if top_green_ratio < 0.45 or bot_green_ratio < 0.45:
            return proportional_fallback

        not_panel = (~is_green_panel).astype(np.uint8)
        row_canvas_score = np.mean(not_panel, axis=1)

        y1, y2 = None, None
        # Only scan up to 80% of client height (battle canvas ends at 465/600 = 77.5%)
        for y in range(int(ch * 0.80)):
            if row_canvas_score[y] > 0.85:
                if y1 is None:
                    y1 = y
                y2 = y
            elif y1 is not None and (y - y1) > int(ch * 0.45):
                break

        if (
            y1 is not None
            and y2 is not None
            and int(ch * 0.01) <= y1 <= int(ch * 0.045)
            and int(ch * 0.72) <= (y2 - y1 + 1) <= int(ch * 0.775)
        ):
            col_canvas_score = np.mean(not_panel[y1:y2, :], axis=0)
            valid_x = np.where(col_canvas_score > 0.85)[0]
            if len(valid_x) > int(cw * 0.90):
                x1, x2 = int(valid_x[0]), int(valid_x[-1])
                if int(cw * 0.01) <= x1 <= int(cw * 0.04) and int(cw * 0.93) <= (x2 - x1 + 1) <= int(cw * 0.985):
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

                    return {
                        "left": pt[0] + x1,
                        "top": pt[1] + y1,
                        "width": x2 - x1 + 1,
                        "height": y2 - y1 + 1,
                    }
        return proportional_fallback
    except Exception:
        pass

    return fallback_field
