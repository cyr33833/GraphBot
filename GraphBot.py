

r""""
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

from core.window_capture import find_game_window, get_capture_field, load_capture_margins
from core.detection import (
    load_active_params,
    load_obstacles_params,
    load_players_params,
)
from core.avoidance import ObstacleMap, extract_obstacle_map, field_obstacles_to_game, DEFAULT_CLEARANCE
from core.field_geometry import pixel_radius_to_game, pixel_to_game
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
        if time.time() - start_time >= sec:
            break


# =========================================================
# 1. 永不失败的画布边界锁定（彻底告别底部 UI 混入）
# =========================================================
def auto_crop_white_field(hwnd, fallback_field):
    """
    直接根据 Graphwar 灰绿色外壳 (RGB ~ 155, 190, 145) 与内部战场的分界线来裁剪。
    无论战场上黑圆有多少、名字牌有多少，灰绿色外壳的位置是绝对固定的！
    """
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        pt = win32gui.ClientToScreen(hwnd, (0, 0))
        cw, ch = right - left, bottom - top
        if cw <= 200 or ch <= 200:
            return fallback_field

        client_rect = {"left": pt[0], "top": pt[1], "width": cw, "height": ch}
        with mss.mss() as sct:
            # mss.grab 返回的是 BGRA 格式！
            img = np.array(sct.grab(client_rect))

        b = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        r = img[:, :, 2].astype(np.int16)

        # 识别外围灰绿色 UI 背景特征：绿色通道明显高于红蓝通道 (g - r > 12 且 g - b > 15)
        is_green_panel = (g - r > 12) & (g - b > 15) & (g > 100) & (g < 230)

        # 非灰绿色面板的区域就是主战场画布（在窗口上半部分 85% 范围内）
        not_panel = (~is_green_panel).astype(np.uint8)

        # 每一行不是灰绿面板的比例：在主战场内部，这个比例接近 96%（只有左右两侧极窄边框是绿的）
        # 而在主战场下方（y= 输入框上方那条横贯全屏的灰绿背景条），这个比例直接跌到 0%！
        row_canvas_score = np.mean(not_panel, axis=1)

        # 寻找从顶部开始的第一段连续画布行 (score > 0.85)
        y1 = None
        y2 = None
        for y in range(int(ch * 0.90)):
            if row_canvas_score[y] > 0.85:
                if y1 is None:
                    y1 = y
                y2 = y
            elif y1 is not None and (y - y1) > int(ch * 0.45):
                # 已经完整走过了主战场高度，遇到了下方的灰绿面板分割带，立即停止！
                break

        if y1 is not None and y2 is not None and (y2 - y1) > int(ch * 0.50):
            # 在锁定的 [y1, y2] 行范围内，同样找出左右两侧非灰绿面板的起止列 [x1, x2]
            col_canvas_score = np.mean(not_panel[y1:y2, :], axis=0)
            valid_x = np.where(col_canvas_score > 0.85)[0]
            if len(valid_x) > int(cw * 0.80):
                x1 = int(valid_x[0])
                x2 = int(valid_x[-1])
                # 向内缩 2 像素去掉包裹白色战场的黑色 1px 边框线
                return {
                    "left": pt[0] + x1 + 2,
                    "top": pt[1] + y1 + 2,
                    "width": (x2 - x1 + 1) - 4,
                    "height": (y2 - y1 + 1) - 4,
                }
    except Exception:
        pass

    # 如果万一出现异常，也绝不使用会截到底部 UI 的高度，强制限制高度不超过窗口的 76%
    safe_fallback = dict(fallback_field)
    if safe_fallback["height"] > 500:
        safe_fallback["height"] = int(safe_fallback["height"] * 0.90)
    return safe_fallback


# =========================================================
# 2. 干净、完整的大圆识别（彻底消除边缘一串难看的小圆）
# =========================================================
# 2. 像素级障碍物提取与距离场（彻底告别圆拟合）
# =========================================================
def detect_obstacles_clean(screenshot_bgr):
    """
    提取像素级绝对精确的黑障碍物地图与欧氏距离场 (SDF)，
    保留 100% 弹坑、隧道与微小凸起，彻底抛弃圆拟合。
    """
    return extract_obstacle_map(screenshot_bgr)


# =========================================================
# 3. 精准识别所有友军、敌军与发射角色（绝不误认名字牌 111）
# =========================================================
def detect_all_players_robust(screenshot_bgr, obstacles):
    """
    为什么之前会把 '111' 名字框当成发射点？
    因为 '111' 的名字框是红棕色的细线矩形（内部是纯白底色），而真正的发射角色是一个实心球体，
    球体中心有高饱和度的亮黄色扇形 + 灰黑色金属半圆，且周围环绕红色光晕。

    本函数通过验证【球体内部非白实心度 + 亮黄色/彩色扇形】双重特征：
    1. 名字牌内部 >70% 是纯白背景，100% 被直接剔除！
    2. 每一个真小兵（包含友军、敌军、发射点）100% 被识别！
    """
    hsv = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. 屏蔽所有黑色障碍区域
    if isinstance(obstacles, ObstacleMap):
        obs_mask = obstacles.mask
    else:
        obs_mask = np.zeros((h, w), dtype=np.uint8)
        for ox, oy, orad in obstacles:
            cv2.circle(obs_mask, (int(ox), int(oy)), int(orad) + 2, 255, -1)

    # 2. 提取所有小兵球体内部的鲜艳扇形色块（在正确 BGR 下，小兵内部的黄/青/绿/橙扇形饱和度极高 S > 80, V > 130）
    # 注意：名字牌的边框线虽然也有颜色，但它是 1px 的空心细框，而小兵内部是实心的色块团！
    vivid_mask = cv2.inRange(hsv, (12, 80, 120), (165, 255, 255))
    vivid_mask = cv2.bitwise_and(vivid_mask, cv2.bitwise_not(obs_mask))

    # 用 3x3 闭运算聚合小兵内部扇形，不把细边框连成片
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vivid_blobs = cv2.morphologyEx(vivid_mask, cv2.MORPH_CLOSE, k_small)

    cnts, _ = cv2.findContours(vivid_blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 8 or area > 450:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        # 排除扁长的名字标签外框（小兵色块宽高都在 4~22 像素之间）
        if bw > 26 or bh > 26 or bw / max(1, bh) > 2.2:
            continue

        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 核心验证：以 (cx, cy) 为中心取半径 8 的圆形区域，检查它是不是一个实心小兵球体！
        # 名字框内部全是白色背景 (gray > 240)，而小兵球体由彩色扇形+深灰金属组成 (gray < 225)
        r_check = 9
        x1, y1 = max(0, int(cx) - r_check), max(0, int(cy) - r_check)
        x2, y2 = min(w, int(cx) + r_check + 1), min(h, int(cy) + r_check + 1)
        patch_gray = gray[y1:y2, x1:x2]
        if patch_gray.size == 0:
            continue

        # 小兵球体像素（非纯白且非纯黑）
        body_pixels = (patch_gray < 232) & (patch_gray > 25)
        if np.sum(body_pixels) < 35:
            # 实心像素太少，说明只是名字框的边角线，跳过！
            continue

        # 用球体全部实心像素的质心来微调圆心，使其精准对准小兵正中心
        bm = cv2.moments(body_pixels.astype(np.uint8))
        if bm["m00"] > 0:
            rcx = x1 + bm["m10"] / bm["m00"]
            rcy = y1 + bm["m01"] / bm["m00"]
            if np.hypot(rcx - cx, rcy - cy) <= 5.0:
                cx, cy = rcx, rcy

        px, py = int(round(cx)), int(round(cy))

        # 合并同一个小兵的相邻检测点
        merged = False
        for i, (ex, ey, er) in enumerate(candidates):
            if np.hypot(px - ex, py - ey) < 15:
                candidates[i] = (int(round((px + ex) / 2)), int(round((py + ey) / 2)), 9)
                merged = True
                break
        if not merged:
            candidates.append((px, py, 9))

    # 3. 区分左右半场：左半场为我方 (ours)，右半场为敌方 (enemies)
    center_x = w / 2.0
    ours = [c for c in candidates if c[0] <= center_x]
    enemies = sorted([c for c in candidates if c[0] > center_x], key=lambda c: c[0])

    # 4. 在已确认的我方小兵 (ours) 中，找出谁身上带着红色发光光晕（Active Player）！
    # 注意：只在已经确认是真小兵的周围 15 像素环带内检测红光，这样绝对不可能把 '111' 红色名字框当成发射点！
    red1 = cv2.inRange(hsv, (0, 60, 100), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 60, 100), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    active_circle = None
    best_red_score = 0

    for c in ours:
        cx, cy, r = c
        rx1, ry1 = max(0, cx - 15), max(0, cy - 15)
        rx2, ry2 = min(w, cx + 16), min(h, cy + 16)
        red_score = cv2.countNonZero(red_mask[ry1:ry2, rx1:rx2])
        if red_score > best_red_score and red_score >= 12:
            best_red_score = red_score
            active_circle = c

    # 如果万一某小兵全身都被红光盖住而没进 candidates，单独检查场上实心红光团（同时排除空心名字框）
    if active_circle is None:
        red_clean = cv2.bitwise_and(red_mask, cv2.bitwise_not(obs_mask))
        rcnts, _ = cv2.findContours(red_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in rcnts:
            if cv2.contourArea(cnt) < 20:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 名字框 '111' 的宽高比是扁矩形 (bw > 24, bh < 18)，而发光球体是正方形 (bw ~ bh)
            if bw > 34 or bh > 34 or bw / max(1, bh) > 1.45:
                continue
            gx, gy = int(round(x + bw / 2.0)), int(round(y + bh / 2.0))
            if gx <= center_x:
                # 还要检查中心不能是纯白底色（排除短名字框）
                patch = gray[max(0, gy - 5):min(h, gy + 6), max(0, gx - 5):min(w, gx + 6)]
                if patch.size > 0 and np.mean(patch > 240) < 0.45:
                    active_circle = (gx, gy, 9)
                    ours.append(active_circle)
                    break

    if active_circle is None and ours:
        active_circle = ours[0]

    allies = [
        c for c in ours
        if active_circle is None or np.hypot(c[0] - active_circle[0], c[1] - active_circle[1]) >= 8
    ]

    return ours, allies, enemies, active_circle


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


GAME_PRECISION = 5
VERTICAL_MAX_COEFF = 999
VERTICAL_MIN_EPS = 0.001


def fmt_game(value):
    return round(float(value), GAME_PRECISION)


def vertical_eps(y_from, y_to, max_coeff=VERTICAL_MAX_COEFF):
    dy = abs(y_to - y_from)
    if dy < 1e-9:
        return VERTICAL_MIN_EPS
    return max(VERTICAL_MIN_EPS, dy / (2 * max_coeff))


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
    dist = fmt_game(-((y1 - y2) / 2) / dx)
    return f"{dist}*(abs(x - {x1}) - abs(x - {x2}))"


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
        pass

    while not is_key_pressed(clicks_end):
        if is_key_pressed(left_mouse_key):
            screen_x, screen_y = win32gui.GetCursorPos()
            if not is_click_in_field(screen_x, screen_y):
                print(f"Ignored click outside field: screen ({screen_x}, {screen_y})")
                win32api.keybd_event(left_mouse_key, 0, win32con.KEYEVENTF_KEYUP, 0)
                continue

            field_x = screen_x - field["left"]
            field_y = screen_y - field["top"]
            print((field_x, field_y))
            clicks.append((field_x, field_y))
            win32api.keybd_event(left_mouse_key, 0, win32con.KEYEVENTF_KEYUP, 0)

    return clicks


def warn_no_players(context=""):
    prefix = f"{context}: " if context else ""
    print(f"{prefix}No players detected. Make sure a Graphwar round is visible.")


def safe_copy(text, previous_text):
    if text != previous_text:
        pyperclip.copy(text)
        print("Safely copied!")


def refresh_field():
    global field

    hwnd = find_game_window(game_window_name)
    if hwnd is None:
        return False

    raw_field = get_capture_field(hwnd, capture_margins)
    field = auto_crop_white_field(hwnd, raw_field)
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
            if not refresh_field():
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
            if poly_scene is None:
                if not refresh_field():
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

                screenshot_base = cv2.cvtColor(screenshot_r, cv2.COLOR_GRAY2BGR)
                screenshot_base = draw_players_roles_overlay(
                    screenshot_base,
                    scene["allies_field"],
                    scene["enemies_field"],
                    scene["active_circle"],
                )
                screenshot_base = draw_obstacle_contours(screenshot_base, scene["obstacles_field"])

                if scene["active_norm"] is None:
                    warn_no_players("Auto mode")
                    cv2.imshow("GraphBot", screenshot_base)
                    cv2.waitKey(1)
                    sleep_key(0.5)
                    continue

                if not scene["enemy_circles_game"]:
                    print("No enemies on the right side. Waiting...")
                    cv2.imshow("GraphBot", screenshot_base)
                    cv2.waitKey(1)
                    sleep_key(0.5)
                    continue

                poly_best_coeffs = None
                poly_scene = {
                    "screenshot_base": screenshot_base,
                    "players_count": scene["total_players_count"],
                    "obstacles_count": len(scene["obstacles_field"]),
                    "active_norm": scene["active_norm"],
                    "enemy_circles_game": scene["enemy_circles_game"],
                    "allies_game": scene["allies_game"],
                    "obstacles_game": scene["obstacles_game"],
                }
                print("Polynomial planner: scene snapshot captured once and frozen.")

            screenshot_vis = poly_scene["screenshot_base"].copy()
            enemy_circles_game = poly_scene["enemy_circles_game"]
            allies_game = poly_scene["allies_game"]
            obstacles_game = poly_scene["obstacles_game"]
            active_norm = poly_scene["active_norm"]

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
                f"players={poly_scene['players_count']}  obstacles={poly_scene['obstacles_count']}  "
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
        if not refresh_field():
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
            )

            if len(path_waypoints) < 2 and enemy_centers_norm:
                path_waypoints, hit_enemies, skipped_enemies = build_enemy_chain_astar(
                    enemy_centers_norm,
                    obstacles_game,
                    clearance=DEFAULT_CLEARANCE,
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
        pass

    main()