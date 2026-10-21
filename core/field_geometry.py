"""
Coordinate transforms for the Graphwar field.

In Graphwar (Java AWT), the coordinate plane is strictly ISOTROPIC:
  - Domain length on X is 50.0 (x in [-25.0, 25.0]).
  - 1 game unit on Y represents the exact same pixel distance as 1 unit on X:
      px_per_unit = field_width / 50.0
      unit_per_px = 50.0 / field_width
  - Origin (0.0, 0.0) is at the exact center (field_width / 2.0, field_height / 2.0).
  - Standard battlefield canvas is 770 x 450 pixels:
      y_half = (450 / 2) * (50 / 770) = 14.61038961038961 game units.
      Visible Y range is [-y_half, y_half] (approx [-14.61, 14.61]).
"""

DEFAULT_CANVAS_WIDTH = 770.0
DEFAULT_CANVAS_HEIGHT = 450.0
X_MIN = -25.0
X_MAX = 25.0
Y_HALF_DEFAULT = 25.0 * DEFAULT_CANVAS_HEIGHT / DEFAULT_CANVAS_WIDTH  # 14.61038961038961
Y_MIN = -Y_HALF_DEFAULT
Y_MAX = Y_HALF_DEFAULT


def pixel_to_game(px, py, field_width, field_height):
    """
    Convert a field-image pixel coordinate to Graphwar coordinates.
    Matches Graphwar Java AWT coordinate projection exactly (isotropic).
    """
    if field_width <= 0 or field_height <= 0:
        raise ValueError("Field dimensions must be positive")

    unit_per_px = 50.0 / float(field_width)
    game_x = (float(px) - float(field_width) / 2.0) * unit_per_px
    game_y = (float(field_height) / 2.0 - float(py)) * unit_per_px
    return game_x, game_y


def game_to_pixel(gx, gy, field_width, field_height):
    """
    Convert Graphwar coordinates to a field-image pixel coordinate.
    Matches Graphwar Java AWT coordinate projection exactly (isotropic).
    """
    if field_width <= 0 or field_height <= 0:
        raise ValueError("Field dimensions must be positive")

    px_per_unit = float(field_width) / 50.0
    px = float(field_width) / 2.0 + float(gx) * px_per_unit
    py = float(field_height) / 2.0 - float(gy) * px_per_unit
    return px, py


def get_y_bounds(field_width, field_height):
    """Return (y_min, y_max) for the visible battlefield height."""
    unit_per_px = 50.0 / float(field_width)
    y_half = (float(field_height) / 2.0) * unit_per_px
    return -y_half, y_half


def pixel_radius_to_game(radius_px, field_width, field_height=None):
    """Convert circular pixel radius to game units using the isotropic scale."""
    if field_width <= 0:
        raise ValueError("Field width must be positive")
    return float(radius_px) * (50.0 / float(field_width))


def pixel_scales(field_width, field_height=None):
    """Return game units represented by one image pixel (identical for x and y)."""
    if field_width <= 0:
        raise ValueError("Field width must be positive")
    scale = 50.0 / float(field_width)
    return scale, scale
