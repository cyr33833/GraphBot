"""
Coordinate transforms for the Graphwar field.

Graphwar (Java AWT, ``catabriga/graphwar``) facts this module relies on:
  - The window content pane is exactly 800 x 600 logical px and not resizable.
  - The battlefield ``GraphPlane`` sits at (15, 15) inside it, size 770 x 450.
  - The coordinate plane is ISOTROPIC: x spans 50 units over 770 px, and y uses
    the same px-per-unit factor (``PLANE_LENGTH / PLANE_GAME_LENGTH``).
  - Origin (0, 0) is the plane centre, so visible y is about +/-14.61 units.

Pixel convention: ``pixel_to_game`` / ``game_to_pixel`` use *continuous*
coordinates where pixel index ``i`` spans [i, i + 1). Pass ``i + 0.5`` when you
mean the centre of pixel ``i``.
"""

import math

# Java constants (logical pixels at 100% scale)
CLIENT_WIDTH_PX = 800
CLIENT_HEIGHT_PX = 600
PLANE_OFFSET_PX = 15
PLANE_LENGTH_PX = 770
PLANE_HEIGHT_PX = 450
PLANE_GAME_LENGTH = 50.0
SOLDIER_RADIUS_PX = 7

DEFAULT_CANVAS_WIDTH = float(PLANE_LENGTH_PX)
DEFAULT_CANVAS_HEIGHT = float(PLANE_HEIGHT_PX)
X_MIN = -25.0
X_MAX = 25.0
Y_HALF_DEFAULT = 25.0 * DEFAULT_CANVAS_HEIGHT / DEFAULT_CANVAS_WIDTH  # 14.61038961038961
Y_MIN = -Y_HALF_DEFAULT
Y_MAX = Y_HALF_DEFAULT

# Soldier hit radius in game units (Function.java: distSquared < SOLDIER_RADIUS^2)
SOLDIER_RADIUS_GAME = SOLDIER_RADIUS_PX * PLANE_GAME_LENGTH / PLANE_LENGTH_PX  # 0.4545...


def pixel_to_game(px, py, field_width, field_height):
    """Convert continuous field coordinates to Graphwar coordinates (isotropic)."""
    if field_width <= 0 or field_height <= 0:
        raise ValueError("Field dimensions must be positive")

    unit_per_px = PLANE_GAME_LENGTH / float(field_width)
    game_x = (float(px) - float(field_width) / 2.0) * unit_per_px
    game_y = (float(field_height) / 2.0 - float(py)) * unit_per_px
    return game_x, game_y


def game_to_pixel(gx, gy, field_width, field_height):
    """Convert Graphwar coordinates to continuous field coordinates (isotropic)."""
    if field_width <= 0 or field_height <= 0:
        raise ValueError("Field dimensions must be positive")

    px_per_unit = float(field_width) / PLANE_GAME_LENGTH
    px = float(field_width) / 2.0 + float(gx) * px_per_unit
    py = float(field_height) / 2.0 - float(gy) * px_per_unit
    return px, py


def get_y_bounds(field_width, field_height):
    """Return (y_min, y_max) for the visible battlefield height."""
    unit_per_px = PLANE_GAME_LENGTH / float(field_width)
    y_half = (float(field_height) / 2.0) * unit_per_px
    return -y_half, y_half


def pixel_radius_to_game(radius_px, field_width, field_height=None):
    """Convert circular pixel radius to game units using the isotropic scale."""
    if field_width <= 0:
        raise ValueError("Field width must be positive")
    return float(radius_px) * (PLANE_GAME_LENGTH / float(field_width))


def pixel_scales(field_width, field_height=None):
    """Return game units represented by one image pixel (identical for x and y)."""
    if field_width <= 0:
        raise ValueError("Field width must be positive")
    scale = PLANE_GAME_LENGTH / float(field_width)
    return scale, scale


def soldier_radius_px(field_width):
    """Soldier radius in captured-image pixels (scales with DPI)."""
    return SOLDIER_RADIUS_PX * float(field_width) / PLANE_LENGTH_PX


def plane_rect_in_client(client_width, client_height, tolerance=0.03):
    """
    Exact battlefield rectangle inside the Graphwar client area.

    Returns (left, top, width, height) in client pixels, or None when the client
    size doesn't look like a (DPI-scaled) 800x600 Graphwar window.
    """
    if client_width <= 0 or client_height <= 0:
        return None
    sx = client_width / float(CLIENT_WIDTH_PX)
    sy = client_height / float(CLIENT_HEIGHT_PX)
    if abs(sx - sy) > tolerance * max(sx, sy):
        return None
    s = sx
    return (
        int(round(PLANE_OFFSET_PX * s)),
        int(round(PLANE_OFFSET_PX * s)),
        int(round(PLANE_LENGTH_PX * s)),
        int(round(PLANE_HEIGHT_PX * s)),
    )


def format_number(value, precision=5):
    """
    Format a float for a Graphwar formula: fixed-point, never scientific
    notation (``1e-05`` is not parseable by Graphwar), no ``-0``.
    """
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"Cannot format non-finite number: {value!r}")
    text = f"{v:.{precision}f}".rstrip("0").rstrip(".")
    if text in ("", "-0", "-"):
        text = "0"
    return text
