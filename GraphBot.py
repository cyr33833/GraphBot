

r"""
   ____                     _      ____          _   
  / ___| _ __  __ _  _ __  | |__  | __ )   ___  | |_ 
 | |  _ | '__|/ _` || '_ \ | '_ \ |  _ \  / _ \ | __|
 | |_| || |  | (_| || |_) || | | || |_) || (_) || |_ 
  \____||_|   \__,_|| .__/ |_| |_||____/  \___/  \__|
                    |_|
                    
=========================================================
 GraphBot.py (Pixel-Exact Boundary & SDF Obstacle Avoidance)
=========================================================
"""

import sys
import ctypes
import cv2
import mss
import time
import random
import re
import numpy as np
import win32api, win32con, win32gui
import pyperclip

from core.window_capture import auto_crop_white_field, find_game_window, get_capture_field, load_capture_margins
from core.detection import (
    detect_all_players_robust,
    load_active_params,
    load_obstacles_params,
    load_players_params,
)
from core.avoidance import (
    DEFAULT_CLEARANCE,
    GAME_PRECISION,
    VERTICAL_MAX_COEFF,
    VERTICAL_MIN_EPS,
    ObstacleMap,
    extract_obstacle_map,
    field_obstacles_to_game,
    fmt_game,
    vertical_eps,
)
from core.field_geometry import format_number, pixel_radius_to_game, pixel_to_game
from core.pathfinding import astar_game, build_enemy_chain_astar, draw_path_on_field
from core.polynomial_planner import (
    draw_polynomial_curve_on_field,
    draw_polynomial_stats_overlay,
    search_best_polynomial,
)
from core.symbolic_ga_planner import (
    draw_symbolic_curve_on_field,
    draw_symbolic_stats_overlay,
    search_best_symbolic_ga,
)


# =========================================================
# Basic Helpers
# =========================================================
def is_key_pressed(key):
    return ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000 != 0


def safe_exit(exit_code):
    print(exit_codes[exit_code])
    sys.exit()


def move_window(window_title, target_x, target_y):
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        print(f"[window] not found: '{window_title}'")
        return None, False

    title = win32gui.GetWindowText(hwnd)
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    print(
        f"[window] found: '{title}' hwnd={hwnd} "
        f"rect=({left}, {top}, {right}, {bottom})"
    )

    if left == target_x and top == target_y:
        print(f"[window] not moved: already at ({target_x}, {target_y})")
        return hwnd, False

    try:
        win32gui.MoveWindow(hwnd, target_x, target_y, width, height, True)
    except Exception as exc:
        print(f"[window] move failed: {exc}")
        return hwnd, False

    new_left, new_top, new_right, new_bottom = win32gui.GetWindowRect(hwnd)
    print(
        "[window] moved: "
        f"({left}, {top}) -> ({new_left}, {new_top}), "
        f"size={new_right - new_left}x{new_bottom - new_top}"
    )
    return hwnd, True


def sleep_key(sec):
    start_time = time.time()
    while True:
        if is_key_pressed(exit_key):
            safe_exit(0)
        remaining = sec - (time.time() - start_time)
        if remaining <= 0:
            break
        time.sleep(min(0.015, remaining))


def detect_obstacles_clean(screenshot_bgr):
    """
    提取像素级绝对精确的黑障碍物地图与欧氏距离场 (SDF)，
    保留 100% 弹坑、隧道与微小凸起，彻底抛弃圆拟合。
    """
    return extract_obstacle_map(screenshot_bgr)


# =========================================================
# Visualization & Math Helpers
# =========================================================
def draw_players_roles_overlay(screenshot_bgr, allies, enemies, active):
    out = screenshot_bgr

    # Enemies (right side): red-orange
    for cx, cy, radius in enemies:
        cv2.circle(out, (int(cx), int(cy)), int(radius), (0, 70, 255), 2)
        cv2.circle(out, (int(cx), int(cy)), 2, (0, 70, 255), -1)

    # Allies (left side, except active): cyan
    for cx, cy, radius in allies:
        cv2.circle(out, (int(cx), int(cy)), int(radius), (255, 200, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 2, (255, 200, 0), -1)

    # Active player: bright green with extra ring
    if active is not None:
        cx, cy, radius = active
        cv2.circle(out, (int(cx), int(cy)), int(radius) + 3, (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 2, (0, 255, 0), -1)

    return out


def draw_obstacle_contours(screenshot_bgr, obstacle_map):
    """用洋红色单一清晰轮廓绘制像素级真实障碍物边界（包含弹坑与隧道）。"""
    if obstacle_map is not None and hasattr(obstacle_map, "contours"):
        cv2.drawContours(screenshot_bgr, obstacle_map.contours, -1, (255, 0, 255), 2, cv2.LINE_AA)
    elif obstacle_map is not None:
        for cx, cy, r in obstacle_map:
            cv2.circle(screenshot_bgr, (int(cx), int(cy)), int(r), (255, 0, 255), 2)
    return screenshot_bgr

draw_obstacle_circles = draw_obstacle_contours


def field_to_game(field_x, field_y):
    game_x, game_y = pixel_to_game(field_x, field_y, field["width"], field["height"])
    return fmt_game(game_x), fmt_game(game_y)


def to_game_cords(cord_list):
    return [list(field_to_game(i[0], i[1])) for i in cord_list]


def field_radius_to_game(radius_px):
    return fmt_game(pixel_radius_to_game(radius_px, field["width"], field["height"]))


def field_circle_to_game(circle):
    cx, cy, radius = circle
    gx, gy = field_to_game(cx, cy)
    gr = field_radius_to_game(radius)
    return (gx, gy, gr)


def collect_symbolic_scene(screenshot_bgr, players_params=None, active_params=None, obstacles_params=None):
    obstacles_field = extract_obstacle_map(screenshot_bgr)
    ours_field, allies_field, enemies_field, active_circle = detect_all_players_robust(
        screenshot_bgr, obstacles_field
    )

    active_norm = None
    if active_circle is not None:
        active_norm = field_to_game(active_circle[0], active_circle[1])

    enemy_circles_game = [field_circle_to_game(circle) for circle in enemies_field]
    allies_game = [field_circle_to_game(circle) for circle in allies_field]
    obstacles_game = field_obstacles_to_game(obstacles_field, field["width"], field["height"])

    total_players_count = len(enemies_field) + len(allies_field) + (1 if active_circle else 0)

    return {
        "total_players_count": total_players_count,
        "enemies_field": enemies_field,
        "allies_field": allies_field,
        "active_circle": active_circle,
        "obstacles_field": obstacles_field,
        "enemy_circles_game": enemy_circles_game,
        "allies_game": allies_game,
        "obstacles_game": obstacles_game,
        "active_norm": active_norm,
        "active_method": "red_glow" if active_circle is not None else "none",
    }


def normalize_formula(text):
    s = str(text)
    while True:
        prev = s
        s = re.sub(r"-\s*-", "+", s)
        s = re.sub(r"\+\s*\+", "+", s)
        s = re.sub(r"\+\s*-", "-", s)
        s = re.sub(r"-\s*\+", "-", s)
        if s == prev:
            break
    return s


def direct_line(p1, p2):
    x1, y1 = fmt_game(p1[0]), fmt_game(p1[1])
    x2, y2 = fmt_game(p2[0]), fmt_game(p2[1])
    dx = x2 - x1
    if abs(dx) < 1e-12:
        dx = fmt_game(vertical_eps(y1, y2)) if y1 != y2 else VERTICAL_MIN_EPS
        x2 = fmt_game(x1 + dx)
    dist = round(-((y1 - y2) / 2.0) / dx, 6)
    dist_str = format_number(dist, 6)
    x1_str = format_number(x1)
    x2_str = format_number(x2)
    return f"{dist_str}*(abs(x - {x1_str}) - abs(x - {x2_str}))"


def waypoints_to_formula(waypoints):
    parts = []
    for i in range(len(waypoints) - 1):
        parts.append(direct_line(tuple(waypoints[i]), tuple(waypoints[i + 1])))
    return normalize_formula(" + ".join(parts))


def is_click_in_field(screen_x, screen_y):
    return (
        field["left"] <= screen_x < field["left"] + field["width"]
        and field["top"] <= screen_y < field["top"] + field["height"]
    )


def collect_clicks():
    print("Click on the game field (outside clicks ignored). F3 = start, F4 = done.")
    if not refresh_field():
        print("Graphwar window not found — cannot collect clicks.")
        return []

    clicks = []
    while not is_key_pressed(clicks_start):
        sleep_key(0.015)

    was_mouse_down = is_key_pressed(left_mouse_key)
    while not is_key_pressed(clicks_end):
        is_down = is_key_pressed(left_mouse_key)
        if is_down and not was_mouse_down:
            screen_x, screen_y = win32gui.GetCursorPos()
            if not is_click_in_field(screen_x, screen_y):
                print(f"Ignored click outside field: screen ({screen_x}, {screen_y})")
            else:
                field_x = screen_x - field["left"]
                field_y = screen_y - field["top"]
                print((field_x, field_y))
                clicks.append((field_x, field_y))
        was_mouse_down = is_down
        sleep_key(0.015)

    return clicks


def warn_no_players(context=""):
    prefix = f"{context}: " if context else ""
    print(f"{prefix}No players detected. Make sure a Graphwar round is visible.")


def safe_copy(text, previous_text):
    if text != previous_text:
        pyperclip.copy(text)
        print("Safely copied!")


def refresh_field(sct=None):
    global field

    hwnd = find_game_window(game_window_name)
    if hwnd is None:
        return False

    raw_field = get_capture_field(hwnd, capture_margins)
    field = auto_crop_white_field(hwnd, raw_field, sct=sct)
    return True


def setup():
    global field

    hwnd, _ = move_window(game_window_name, window_start_cords[0], window_start_cords[1])
    if hwnd is None:
        safe_exit(1)

    raw_field = get_capture_field(hwnd, capture_margins)
    field = auto_crop_white_field(hwnd, raw_field)
    print(f"Capture region (exact white canvas): {field}")

    while not(is_key_pressed(exit_key)):
        print("Select the mode\n0 - automatic\n1 - clicks")
        mode = input().strip()
        if len(mode) == 1 and (mode == '0' or mode == '1'):
            mode = int(mode)
            break
        print("Incorrect input!\n")

    auto_planner = 0
    if mode == 0:
        while not(is_key_pressed(exit_key)):
            print(
                "Select auto planner\n"
                "0 - A* chain (current)\n"
                "1 - polynomial search (experimental)\n"
                "2 - symbolic GA (graphwar-like)"
            )
            planner = input().strip()
            if planner in ("0", "1", "2"):
                auto_planner = int(planner)
                break
            print("Incorrect input!\n")

    return mode, auto_planner


def main():
    prev_text = ""
    prev_summary = ""
    poly_best_coeffs = None
    poly_rng = random.Random()
    poly_scene = None
    sym_best_genome = None
    sym_rng = random.Random()
    players_params = load_players_params()
    active_params = load_active_params()
    obstacles_params = load_obstacles_params()
    mss_ = mss.mss()

    if auto_planner == 0:
        print("Auto mode: F2 = quit. Planner: A*. Updates every ~1 s.")
    elif auto_planner == 1:
        print("Auto mode: F2 = quit. Planner: Polynomial Search. Updates every ~1 s.")
    else:
        print("Auto mode: F2 = quit. Planner: Symbolic GA. Updates every ~1 s.")

    while not(is_key_pressed(exit_key)):
        # ---------------------------------------------------------
        # Mode 0, Planner 2: Symbolic GA
        # ---------------------------------------------------------
        if not mode and auto_planner == 2:
            if not refresh_field(sct=mss_):
                print("Graphwar window not found. Waiting...")
                sleep_key(0.5)
                continue

            screenshot = np.array(mss_.grab(field))
            # 关键修正：mss.grab 返回的是 BGRA，必须用 COLOR_BGRA2BGR，绝不能用 COLOR_RGB2BGR！
            screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            scene = collect_symbolic_scene(
                screenshot_bgr,
                players_params=players_params,
                active_params=active_params,
                obstacles_params=obstacles_params,
            )
            screenshot_vis = cv2.cvtColor(screenshot_r, cv2.COLOR_GRAY2BGR)
            screenshot_vis = draw_players_roles_overlay(
                screenshot_vis,
                scene["allies_field"],
                scene["enemies_field"],
                scene["active_circle"],
            )
            screenshot_vis = draw_obstacle_contours(screenshot_vis, scene["obstacles_field"])

            if scene["total_players_count"] == 0:
                warn_no_players("Auto mode")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            if scene["active_norm"] is None:
                print("Auto mode: no active player detected.")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            if not scene["enemy_circles_game"]:
                print("No enemies on the right side. Waiting...")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            sym_result = search_best_symbolic_ga(
                x0=scene["active_norm"][0],
                y0=scene["active_norm"][1],
                enemies=scene["enemy_circles_game"],
                dangers=scene["obstacles_game"],
                allies=scene["allies_game"],
                prev_best=sym_best_genome,
                rng=sym_rng,
                step=0.05,
                budget_ms=320,
                max_evals=900,
                population_size=50,
            )
            sym_best_genome = sym_result["best_genome"]
            screenshot_vis = draw_symbolic_curve_on_field(
                screenshot_vis,
                sym_result["best_points"],
                field["width"],
                field["height"],
                color=(0, 255, 0),
                thickness=2,
            )

            stats = sym_result["stats"]
            hits = len(sym_result["hit_enemy_ids"])
            stats_lines = [
                f"planner=SYM-GA  score={int(sym_result['best_score'])}  hits={hits}/{len(scene['enemy_circles_game'])}",
                f"evals={stats['evals']}  gen={stats['generations']}  time={stats['time_ms']}ms  ally_hits(best)={stats['ally_hits_best']}",
                f"reject danger={stats['rejects_danger']}  bounds={stats['rejects_bounds']}  invalid={stats['rejects_invalid']}",
            ]
            screenshot_vis = draw_symbolic_stats_overlay(screenshot_vis, stats_lines)

            summary = (
                f"players={scene['total_players_count']}  obstacles={len(scene['obstacles_field'])}  "
                f"enemies={hits}/{len(scene['enemy_circles_game'])}  "
                f"planner=SYM-GA  score={int(sym_result['best_score'])}  "
                f"active={scene['active_method']}"
            )
            if summary != prev_summary:
                print(summary)
                prev_summary = summary

            formula = sym_result["formula"]
            if formula and formula != prev_text:
                print(formula)
                safe_copy(formula, prev_text)
                prev_text = formula

            cv2.imshow("GraphBot", screenshot_vis)
            cv2.waitKey(1)
            sleep_key(0.0)
            continue

        # ---------------------------------------------------------
        # Mode 0, Planner 1: Polynomial Search
        # ---------------------------------------------------------
        if not mode and auto_planner == 1:
            if not refresh_field(sct=mss_):
                print("Graphwar window not found. Waiting...")
                sleep_key(0.5)
                continue

            screenshot = np.array(mss_.grab(field))
            screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            scene = collect_symbolic_scene(
                screenshot_bgr,
                players_params=players_params,
                active_params=active_params,
                obstacles_params=obstacles_params,
            )

            screenshot_vis = cv2.cvtColor(screenshot_r, cv2.COLOR_GRAY2BGR)
            screenshot_vis = draw_players_roles_overlay(
                screenshot_vis,
                scene["allies_field"],
                scene["enemies_field"],
                scene["active_circle"],
            )
            screenshot_vis = draw_obstacle_contours(screenshot_vis, scene["obstacles_field"])

            if scene["active_norm"] is None:
                warn_no_players("Auto mode")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            if not scene["enemy_circles_game"]:
                print("No enemies on the right side. Waiting...")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            active_norm = scene["active_norm"]
            enemy_circles_game = scene["enemy_circles_game"]
            allies_game = scene["allies_game"]
            obstacles_game = scene["obstacles_game"]

            if (
                poly_scene is None
                or np.hypot(active_norm[0] - poly_scene["active_norm"][0], active_norm[1] - poly_scene["active_norm"][1]) > 0.5
                or len(enemy_circles_game) != len(poly_scene["enemy_circles_game"])
            ):
                poly_best_coeffs = None
                poly_scene = {
                    "active_norm": active_norm,
                    "enemy_circles_game": enemy_circles_game,
                }

            poly_result = search_best_polynomial(
                x0=active_norm[0],
                y0=active_norm[1],
                enemies=enemy_circles_game,
                dangers=obstacles_game,
                allies=allies_game,
                prev_best=poly_best_coeffs,
                rng=poly_rng,
                step=0.05,
                budget_ms=320,
                max_evals=900,
            )
            poly_best_coeffs = poly_result["best_coeffs"]

            if poly_result["current_points"]:
                screenshot_vis = draw_polynomial_curve_on_field(
                    screenshot_vis,
                    poly_result["current_points"],
                    field["width"],
                    field["height"],
                    color=(0, 220, 255),
                    thickness=1,
                )
            screenshot_vis = draw_polynomial_curve_on_field(
                screenshot_vis,
                poly_result["best_points"],
                field["width"],
                field["height"],
                color=(0, 255, 0),
                thickness=2,
            )

            stats = poly_result["stats"]
            hits = len(poly_result["hit_enemy_ids"])
            danger_text = "9999+" if stats["rejects_danger"] >= 9999 else str(stats["rejects_danger"])
            bounds_text = "9999+" if stats["rejects_bounds"] >= 9999 else str(stats["rejects_bounds"])
            stats_lines = [
                f"planner=POLY  score={int(poly_result['best_score'])}  hits={hits}/{len(enemy_circles_game)}",
                f"evals={stats['evals']}  time={stats['time_ms']}ms  ally_hits(best)={stats['ally_hits_best']}  near={stats['proximity_bonus_best']}",
                f"reject danger={danger_text}  bounds={bounds_text}  ally_penalized={stats['ally_penalty_hits']}",
            ]
            screenshot_vis = draw_polynomial_stats_overlay(screenshot_vis, stats_lines)

            summary = (
                f"players={scene['total_players_count']}  obstacles={len(scene['obstacles_field'])}  "
                f"enemies={hits}/{len(enemy_circles_game)}  "
                f"planner=POLY  score={int(poly_result['best_score'])}"
            )
            if summary != prev_summary:
                print(summary)
                prev_summary = summary

            formula = poly_result["formula"]
            if formula and formula != prev_text:
                print(formula)
                safe_copy(formula, prev_text)
                prev_text = formula

            cv2.imshow("GraphBot", screenshot_vis)
            cv2.waitKey(1)
            sleep_key(0.0)
            continue

        # ---------------------------------------------------------
        # Mode 0, Planner 0 (A*) OR Mode 1 (Clicks)
        # ---------------------------------------------------------
        if not refresh_field(sct=mss_):
            print("Graphwar window not found. Waiting...")
            sleep_key(0.5)
            continue

        screenshot = np.array(mss_.grab(field))
        screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        scene = collect_symbolic_scene(
            screenshot_bgr,
            players_params=players_params,
            active_params=active_params,
            obstacles_params=obstacles_params,
        )

        if not mode:
            screenshot_vis = cv2.cvtColor(screenshot_r, cv2.COLOR_GRAY2BGR)
            screenshot_vis = draw_players_roles_overlay(
                screenshot_vis,
                scene["allies_field"],
                scene["enemies_field"],
                scene["active_circle"],
            )
            screenshot_vis = draw_obstacle_contours(screenshot_vis, scene["obstacles_field"])

            if scene["active_norm"] is None:
                warn_no_players("Auto mode")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            if not scene["enemies_field"]:
                print("No enemies on the right side. Waiting...")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            active_norm = scene["active_norm"]
            enemy_centers_norm = sorted(
                to_game_cords([(cx, cy) for cx, cy, _ in scene["enemies_field"]]),
                key=lambda point: point[0],
            )
            obstacles_game = scene["obstacles_game"]

            chain_targets = [list(active_norm)] + enemy_centers_norm
            path_waypoints, hit_enemies, skipped_enemies = build_enemy_chain_astar(
                chain_targets,
                obstacles_game,
                clearance=DEFAULT_CLEARANCE,
                allies=scene["allies_game"],
            )

            if len(path_waypoints) < 2 and enemy_centers_norm:
                path_waypoints, hit_enemies, skipped_enemies = build_enemy_chain_astar(
                    enemy_centers_norm,
                    obstacles_game,
                    clearance=DEFAULT_CLEARANCE,
                    allies=scene["allies_game"],
                )

            if len(path_waypoints) < 2 and enemy_centers_norm:
                path_waypoints = [list(active_norm)] + [list(point) for point in enemy_centers_norm]

            screenshot_vis = draw_path_on_field(
                screenshot_vis, path_waypoints, field["width"], field["height"]
            )

            summary = (
                f"players={scene['total_players_count']}  obstacles={len(scene['obstacles_field'])}  "
                f"enemies={len(scene['enemies_field'])}  "
                f"waypoints={len(path_waypoints)}  planner=A*"
            )
            if summary != prev_summary:
                print(summary)
                if skipped_enemies:
                    print("  skipped (no detour):", skipped_enemies)
                prev_summary = summary

            formula = waypoints_to_formula(path_waypoints)
            if not formula:
                print("  no formula (need at least 2 waypoints)")
            elif formula != prev_text:
                print(formula)
                safe_copy(formula, prev_text)
                prev_text = formula

            cv2.imshow("GraphBot", screenshot_vis)
            cv2.waitKey(1)
            sleep_key(0.2)
        else:
            if scene["total_players_count"] == 0:
                warn_no_players("Click mode")
                cv2.imshow("GraphBot", screenshot_r)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            clicks = collect_clicks()
            if not clicks:
                print("No clicks recorded. Press F3 to start, F4 when done.")
                continue

            if not refresh_field():
                print("Graphwar window not found — cannot build formula.")
                continue

            screenshot = np.array(mss_.grab(field))
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            scene = collect_symbolic_scene(
                screenshot_bgr,
                players_params=players_params,
                active_params=active_params,
                obstacles_params=obstacles_params,
            )
            active_norm = scene["active_norm"]
            if active_norm is None:
                print("Click mode: active player not detected.")
                continue

            clicks_norm = sorted(
                [list(field_to_game(field_x, field_y)) for field_x, field_y in clicks],
                key=lambda point: point[0],
            )
            waypoints = [list(active_norm)] + clicks_norm

            print("Clicks (field px):", clicks)
            print("Active (game):", active_norm)
            print("Targets sorted by x:", clicks_norm)
            print("Waypoints:", waypoints)
            formula = waypoints_to_formula(waypoints)
            if not formula:
                print("No formula (need at least one target click).")
                continue
            print()
            print(formula)
            safe_copy(formula, prev_text)
            prev_text = formula

            safe_exit(0)


# =========================================================
# Main Program
# =========================================================
if __name__ == '__main__':
    left_mouse_key = 0x01
    start_key = 0x70 # f1
    exit_key = 0x71 # f2
    clicks_start = 0x72 # f3
    clicks_end = 0x73 # f4

    window_start_cords = (-7, 0)
    game_window_name = 'Graphwar'
    capture_margins = load_capture_margins()
    field = None
    exit_codes = {
        0: "Program has successfuly finished!",
        1: "No window with the game name has found :(\nMake sure Graphwar is running"
    }

    mode, auto_planner = setup()

    while not(is_key_pressed(start_key)):
        sleep_key(0.02)

    main()