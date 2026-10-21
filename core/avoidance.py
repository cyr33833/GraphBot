"""
Lightweight obstacle avoidance for Graphwar auto paths.

Pixel-exact obstacle representation using Euclidean Distance Field (SDF)
and backward-compatible circle-list fallback.
"""

import math
import cv2
import numpy as np

from .field_geometry import pixel_radius_to_game, pixel_to_game, game_to_pixel

GAME_PRECISION = 5
VERTICAL_MAX_COEFF = 999
VERTICAL_MIN_EPS = 0.001
DEFAULT_CLEARANCE = 0.35


class ObstacleMap:
    """
    Pixel-exact obstacle representation using a Euclidean Distance Field (SDF).
    Replaces the legacy list of (cx, cy, r) circles while remaining backward-compatible.
    """
    def __init__(self, clean_black_mask: np.ndarray, field_w: int, field_h: int):
        self.mask = clean_black_mask.astype(np.uint8)  # uint8, 255 = obstacle, 0 = free space
        self.w = int(field_w)
        self.h = int(field_h)

        # Extract exact contours (including bullet craters & small arcs) for visualization
        contours, _ = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        self.contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 15]

        # Compute Euclidean distance (in pixels) from every free-space pixel to nearest black pixel
        free_space = cv2.bitwise_not(self.mask)
        self.dist_px = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)

        # Convert pixel distance to game coordinate units (50.0 / w horizontally, 30.0 / h vertically)
        scale_x = 50.0 / max(1, self.w)
        scale_y = 30.0 / max(1, self.h)
        self.unit_per_px = min(scale_x, scale_y)
        self.dist_game = self.dist_px * self.unit_per_px

        # Also build a dense boundary circle list [(gx, gy, r_game), ...] sampled every 3px
        # along all obstacle contours so any legacy code iterating over `dangers` still works!
        self.boundary_circles = self._build_boundary_circles(step_px=3)

    def _build_boundary_circles(self, step_px=3):
        circles = []
        r_tiny = round(1.5 * self.unit_per_px, 5)
        for cnt in self.contours:
            pts = cnt[:, 0, :]
            for i in range(0, len(pts), step_px):
                px, py = int(pts[i][0]), int(pts[i][1])
                gx, gy = pixel_to_game(px, py, self.w, self.h)
                circles.append((round(gx, 5), round(gy, 5), r_tiny))
        return circles

    def __iter__(self):
        """Backward-compatibility if a planner iterates over `dangers` as (x, y, r) tuples."""
        return iter(self.boundary_circles)

    def __len__(self):
        return len(self.contours)

    def __getitem__(self, index):
        return self.boundary_circles[index]

    def clearance_at_game(self, gx: float, gy: float) -> float:
        """Return distance in game units to the nearest black obstacle pixel (0.0 if inside obstacle)."""
        px, py = game_to_pixel(gx, gy, self.w, self.h)
        ix, iy = int(round(px)), int(round(py))
        if ix < 0 or ix >= self.w or iy < 0 or iy >= self.h:
            return -1.0  # Out of bounds
        return float(self.dist_game[iy, ix])

    def is_point_safe(self, gx: float, gy: float, clearance: float = 0.25) -> bool:
        return self.clearance_at_game(gx, gy) > clearance

    def is_segment_safe(self, p1, p2, clearance: float = 0.25) -> bool:
        """Exact pixel-marched line-of-sight collision check between two game points."""
        x1, y1 = game_to_pixel(p1[0], p1[1], self.w, self.h)
        x2, y2 = game_to_pixel(p2[0], p2[1], self.w, self.h)
        length_px = float(np.hypot(x2 - x1, y2 - y1))
        steps = max(2, int(np.ceil(length_px / 1.5)))
        xs = np.clip(np.round(np.linspace(x1, x2, steps)).astype(int), 0, self.w - 1)
        ys = np.clip(np.round(np.linspace(y1, y2, steps)).astype(int), 0, self.h - 1)
        min_dist = float(np.min(self.dist_game[ys, xs]))
        return min_dist > clearance


def extract_obstacle_map(screenshot_bgr: np.ndarray) -> ObstacleMap:
    """
    Extract exact binary mask of black obstacles (removing 1px axes lines and text noise)
    while preserving 100% of bullet craters, notches, and small protruding arcs.
    """
    if len(screenshot_bgr.shape) == 3:
        gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = screenshot_bgr
    h, w = gray.shape

    # Black obstacles have intensity < 40
    black_mask = (gray < 40).astype(np.uint8) * 255

    # 3x3 morphological opening removes the 1px coordinate axes and thin text characters
    # without distorting obstacle boundaries or bullet craters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

    # Filter out tiny isolated specks (e.g., bold letters or stray dots < 25px area)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    filtered_mask = np.zeros_like(clean_mask)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 25:
            filtered_mask[labels == lbl] = 255

    return ObstacleMap(filtered_mask, w, h)


def fmt_game(value):
    return round(float(value), GAME_PRECISION)


def vertical_eps(y_from, y_to, max_coeff=VERTICAL_MAX_COEFF):
    dy = abs(y_to - y_from)
    if dy < 1e-9:
        return VERTICAL_MIN_EPS
    return max(VERTICAL_MIN_EPS, dy / (2 * max_coeff))


def field_obstacles_to_game(obstacles_field, field_width, field_height=None):
    """Convert field-pixel obstacles to game coords. Returns ObstacleMap if already an ObstacleMap."""
    if isinstance(obstacles_field, ObstacleMap):
        return obstacles_field
    field_height = field_height or field_width
    game = []
    for cx, cy, r in obstacles_field:
        gx, gy = pixel_to_game(cx, cy, field_width, field_height)
        gr = pixel_radius_to_game(r, field_width, field_height)
        game.append((fmt_game(gx), fmt_game(gy), fmt_game(gr)))
    return game


def segment_intersects_circle(p1, p2, circle, margin=0):
    cx, cy, r = circle
    r = r + margin
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return math.hypot(x1 - cx, y1 - cy) <= r

    fx = x1 - cx
    fy = y1 - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False

    sqrt_disc = math.sqrt(disc)
    for t in ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)):
        if -1e-9 <= t <= 1 + 1e-9:
            return True
    return False


def _segment_hits_any(p1, p2, obstacles, margin):
    if isinstance(obstacles, ObstacleMap):
        return not obstacles.is_segment_safe(p1, p2, clearance=margin)
    return any(segment_intersects_circle(p1, p2, o, margin) for o in obstacles)


def _path_length(points):
    total = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _append_vertical(points, x_col, y_target):
    """Add a near-vertical step at column x_col (Graphwar needs distinct x)."""
    x_col = fmt_game(x_col)
    y_target = fmt_game(y_target)
    if not points:
        points.append([x_col, y_target])
        return

    last_x, last_y = points[-1]
    if abs(last_y - y_target) < 1e-6:
        return

    eps = fmt_game(vertical_eps(last_y, y_target))
    end_x = fmt_game(x_col + eps)
    if abs(last_x - end_x) > 1e-6 or abs(last_y - y_target) > 1e-6:
        points.append([end_x, y_target])


def _detour_step(p1, circle, obstacles, margin, clearance, y_bounds):
    """One bypass step: vertical to clear height, then past the circle's right edge."""
    cx, cy, r = circle
    R = r + margin + clearance
    y_min, y_max = y_bounds
    x_out = fmt_game(cx + R)

    candidates = []
    for y_clear in (fmt_game(min(y_max, cy + R)), fmt_game(max(y_min, cy - R))):
        if not (y_min - 1e-6 <= y_clear <= y_max + 1e-6):
            continue

        route = [list(p1)]
        _append_vertical(route, p1[0], y_clear)
        if x_out > route[-1][0] + 1e-6:
            route.append([x_out, y_clear])
        if len(route) < 2:
            continue

        ok = True
        for i in range(len(route) - 1):
            a = tuple(route[i])
            b = tuple(route[i + 1])
            if b[0] + 1e-6 < a[0]:
                ok = False
                break
            if _segment_hits_any(a, b, obstacles, margin):
                ok = False
                break
        if ok:
            candidates.append(route)

    if not candidates:
        return None
    return min(candidates, key=_path_length)[1:]


def resolve_segment(p1, p2, obstacles, margin=0, clearance=DEFAULT_CLEARANCE, y_bounds=(-15, 15)):
    """
    Return intermediate waypoints (excluding p1, including p2) to reach p2 from p1
    without crossing obstacles. None if no detour found within iteration budget.
    """
    p1 = (fmt_game(p1[0]), fmt_game(p1[1]))
    p2 = (fmt_game(p2[0]), fmt_game(p2[1]))
    hit_margin = margin + clearance

    if p2[0] + 1e-6 < p1[0]:
        return None

    if not _segment_hits_any(p1, p2, obstacles, hit_margin):
        return [list(p2)]

    current = p1
    mids = []
    for _ in range(24):
        if not _segment_hits_any(current, p2, obstacles, hit_margin):
            mids.append(list(p2))
            return mids

        blocking = [
            o for o in obstacles if segment_intersects_circle(current, p2, o, hit_margin)
        ]
        if not blocking:
            mids.append(list(p2))
            return mids

        obstacle = min(blocking, key=lambda o: o[0])
        step = _detour_step(current, obstacle, obstacles, hit_margin, clearance, y_bounds)
        if step is None:
            return None

        for pt in step:
            mids.append(list(pt))
        current = tuple(mids[-1])

    return None


def build_enemy_chain(
    enemies,
    obstacles,
    margin=0,
    clearance=DEFAULT_CLEARANCE,
    y_bounds=(-15, 15),
):
    """
    Build formula waypoints enemy-to-enemy. First enemy = formula anchor
    (where the graph must already be). Active player is NOT included.
    """
    if not enemies:
        return [], [], []

    first = [fmt_game(enemies[0][0]), fmt_game(enemies[0][1])]
    path = [first]
    hit = [first]
    skipped = []

    for enemy in enemies[1:]:
        enemy_pt = [fmt_game(enemy[0]), fmt_game(enemy[1])]
        segment = resolve_segment(
            tuple(path[-1]),
            tuple(enemy_pt),
            obstacles,
            margin=margin,
            clearance=clearance,
            y_bounds=y_bounds,
        )
        if segment is None:
            skipped.append(enemy_pt)
            continue
        path.extend(segment)
        hit.append(enemy_pt)

    return path, hit, skipped


def build_greedy_enemy_path(
    active,
    enemies,
    obstacles,
    margin=0,
    clearance=DEFAULT_CLEARANCE,
    y_bounds=(-15, 15),
):
    """Legacy planner from active — use build_enemy_chain for formula output."""
    if not enemies:
        return [list(active)], [], list(enemies)

    path, hit, skipped = build_enemy_chain(enemies, obstacles, margin, clearance, y_bounds)
    if not path:
        return [list(active)], [], list(enemies)

    if resolve_segment(
        tuple(active), tuple(path[0]), obstacles, margin, clearance, y_bounds
    ) is None:
        skipped = [path[0]] + skipped

    return path, hit, skipped
