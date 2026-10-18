

"""
   ____                     _      ____          _   
  / ___| _ __  __ _  _ __  | |__  | __ )   ___  | |_ 
 | |  _ | '__|/ _` || '_ \ | '_ \ |  _ \  / _ \ | __|
 | |_| || |  | (_| || |_) || | | || |_) || (_) || |_ 
  \____||_|   \__,_|| .__/ |_| |_||____/  \___/  \__|
                    |_|
                    
=========================================================
 GraphBot.py (Ultimate Fix: Canvas Crop, Edge Circles & Player Color Detection)
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

# 注意：不要开启 SetProcessDpiAwareness，因为 Graphwar (Java) 运行在系统缩放兼容模式下，
# 保持 Python 与 Java 处于同一套 Windows 逻辑坐标系，才能保证窗口矩形与截图严丝合缝。

from core.window_capture import find_game_window, get_capture_field, load_capture_margins
from core.detection import (
    load_active_params,
    load_obstacles_params,
    load_players_params,
)
from core.avoidance import field_obstacles_to_game, DEFAULT_CLEARANCE
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
# 1. 精准画布裁剪（彻底消除底部 UI 越界与弹道缩放偏移）
# =========================================================
def auto_crop_white_field(hwnd, fallback_field):
    """
    通过在窗口客户区中寻找纯白画布与 1px 黑色内边框的精确交界，
    彻底切掉周围灰绿色 UI 面板和底部的 y= 输入框，确保 [-25,25]x[-15,15] 比例 100% 精准。
    """
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        pt = win32gui.ClientToScreen(hwnd, (0, 0))
        cw, ch = right - left, bottom - top
        if cw <= 200 or ch <= 200:
            return fallback_field

        client_rect = {"left": pt[0], "top": pt[1], "width": cw, "height": ch}
        with mss.mss() as sct:
            img = np.array(sct.grab(client_rect))

        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Graphwar 的外围面板是灰绿色的 (Hue 约 40~85, Sat > 18, Val < 225)
        # 而游戏主战场内部只有纯白背景 (Val > 240, Sat < 10) 和纯黑障碍圆 (Val < 40)
        is_white_or_black = ((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 15)) | (hsv[:, :, 2] < 45)
        canvas_mask = (is_white_or_black.astype(np.uint8)) * 255

        # 统计每一行、每一列属于画布像素的比例
        # 主战场占据窗口上半部分约 75%~85% 的高度、94% 以上的宽度
        row_ratio = np.mean(canvas_mask > 0, axis=1)
        col_ratio = np.mean(canvas_mask[: int(ch * 0.85), :] > 0, axis=0)

        valid_rows = np.where(row_ratio > 0.88)[0]
        valid_cols = np.where(col_ratio > 0.80)[0]

        if len(valid_rows) > 200 and len(valid_cols) > 300:
            # 找到第一段连续的高占比行（即为上方主战场，跳过底部的白色输入框）
            y1 = int(valid_rows[0])
            y2 = y1
            for r in valid_rows[1:]:
                if r - y2 <= 5:  # 允许坐标轴线等微小间断
                    y2 = int(r)
                else:
                    break

            x1 = int(valid_cols[0])
            x2 = int(valid_cols[-1])

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # 收缩 1 像素以去掉包裹画布的 1px 黑色矩形框线
            if w > cw * 0.85 and h > ch * 0.55:
                return {
                    "left": pt[0] + x1 + 1,
                    "top": pt[1] + y1 + 1,
                    "width": w - 2,
                    "height": h - 2,
                }
    except Exception:
        pass

    return fallback_field


# =========================================================
# 2. 完美识别黑色障碍圆（解决贴边圆变小、重叠圆漏检）
# =========================================================
def detect_obstacles_accurate(screenshot_bgr):
    """
    通过边界外延 + 迭代式距离变换峰值提取，精准识别所有独立黑圆、粘连黑圆以及贴边半圆。
    """
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 提取纯黑像素（障碍圆内部灰度全部为 0）
    black_mask = (gray < 35).astype(np.uint8) * 255

    # 抹除 1px 宽的十字坐标轴线，防止坐标轴切断或干扰距离计算
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_black = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_open)

    # 关键修复：对于贴近画面上下左右边缘的圆（如右下角的圆），如果直接算距离变换，
    # 图像边界外的 0 会把峰值往内推、半径算小。
    # 因此我们在四周复制延拓 PAD 像素，让贴边黑圆自然延伸出去！
    pad = 80
    padded = cv2.copyMakeBorder(clean_black, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # 先用全局距离变换一次性找出所有明显的局部极大值圆心
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    local_max = cv2.dilate(dist, dilate_k)

    # 找出所有半径 >= 7 的局部峰值，按半径从大到小排序
    peaks_mask = (dist == local_max) & (dist >= 7.0)
    ys, xs = np.where(peaks_mask)
    radii = dist[ys, xs]
    order = np.argsort(-radii)

    # 用残差掩码验证：只保留真正覆盖了未解释黑色像素的圆，杜绝左侧大黑块内部产生多余的同心伪圆
    remaining = clean_black.copy()
    circles = []

    for idx in order:
        py, px = int(ys[idx]), int(xs[idx])
        r_est = float(radii[idx])

        cx = px - pad
        cy = py - pad

        # 圆心必须在画布合理范围内
        if cx < -r_est or cx >= w + r_est or cy < -r_est or cy >= h + r_est:
            continue

        r_int = int(round(r_est)) + 1  # +1 补偿刚才开运算腐蚀掉的半个像素边缘

        # 检查这个候选圆在当前 remaining 掩码里能覆盖多少尚未被大圆解释的黑色像素
        check_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(check_mask, (cx, cy), r_int, 255, -1)

        new_black_pixels = cv2.countNonZero(cv2.bitwise_and(remaining, check_mask))
        circle_area = max(1, cv2.countNonZero(check_mask))

        # 如果该圆内部绝大部分黑色区域已经被更大的圆覆盖了，说明是粘连过渡区的伪峰值，跳过
        if new_black_pixels < max(25, circle_area * 0.12):
            continue

        # 确认是真实障碍圆：加入列表，并从 remaining 中擦除它覆盖的区域
        circles.append((cx, cy, r_int))
        cv2.circle(remaining, (cx, cy), int(r_int * 0.92), 0, -1)

    # 补充扫描：万一还有残余的黑色块没被峰值抓到，用迭代最大值兜底
    for _ in range(15):
        if cv2.countNonZero(remaining) < 60:
            break
        pad_rem = cv2.copyMakeBorder(remaining, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        d_rem = cv2.distanceTransform(pad_rem, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(d_rem)
        if max_val < 7.5:
            break
        cx = max_loc[0] - pad
        cy = max_loc[1] - pad
        r_int = int(round(max_val)) + 1
        circles.append((cx, cy, r_int))
        cv2.circle(remaining, (cx, cy), int(r_int * 0.95), 0, -1)

    return circles


# =========================================================
# 3. 专属玩家与敌人识别（100% 识别小兵，100% 忽略名字标牌）
# =========================================================
def detect_all_players_robust(screenshot_bgr, obstacles):
    """
    直接根据 Graphwar 角色球体内部的鲜艳扇形色块 + 灰金属球体特征进行识别。
    名字标牌（如 Multivac、Wolfram、11）内部是白底黑字（饱和度为0），天然被彻底排除！
    同时检测当前回合自带红色发光光晕（Active Glow）的控制角色。
    """
    hsv = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 构造障碍物掩码，向外扩 3 像素，防止黑球边缘干扰
    obs_mask = np.zeros((h, w), dtype=np.uint8)
    for ox, oy, orad in obstacles:
        cv2.circle(obs_mask, (int(ox), int(oy)), int(orad) + 3, 255, -1)

    # 1. 首先定位当前激活角色（Active Player）：身体周围有一圈明显的红色发光光晕
    # 红色在 HSV 中位于 H in [0, 10] 或 [168, 180]
    red1 = cv2.inRange(hsv, (0, 70, 120), (10, 255, 255))
    red2 = cv2.inRange(hsv, (168, 70, 120), (180, 255, 255))
    red_glow = cv2.bitwise_or(red1, red2)
    red_glow = cv2.bitwise_and(red_glow, cv2.bitwise_not(obs_mask))

    active_circle = None
    glow_cnts, _ = cv2.findContours(red_glow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if glow_cnts:
        best_glow = max(glow_cnts, key=cv2.contourArea)
        if cv2.contourArea(best_glow) >= 18:
            (gx, gy), _ = cv2.minEnclosingCircle(best_glow)
            active_circle = (int(round(gx)), int(round(gy)), 9)

    # 2. 定位场上所有小兵（我方+敌方）：
    # 每个小兵球体内部都有一块高饱和度的亮色扇形（如黄、青、绿、紫、橙、蓝等），
    # 而背景是纯白(S=0)、障碍是纯黑(V=0)、名字牌是白底黑字灰框(S<25)。
    # 因此提取 S > 65 且 V > 90 的彩色像素即可精准抓出所有小兵！
    color_mask = cv2.inRange(hsv, (0, 65, 90), (180, 255, 255))
    color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(obs_mask))

    # 把刚才的红色光晕外圈稍微缩一点，避免光晕把单个小兵拆成碎片，用膨胀把同一个小兵内部的色块聚合成团
    kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blobs = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_merge)

    cnts, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_players = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        # 单个小兵色块面积通常在 12 到 500 像素之间（过滤掉 1~2px 的杂色噪点）
        if area < 10 or area > 700:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        # 名字框如果带彩色边框会是扁长条（宽高比极大），而小兵球体接近正方形
        if bw > 36 or bh > 36 or bw / max(1, bh) > 2.5:
            continue

        # 计算色块质心
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + bw / 2.0, y + bh / 2.0

        # 微调球心：小兵的黄色扇形位于球体的左下半边，球体右上半边是灰色金属。
        # 我们在 (cx, cy) 附近 14x14 的窗口内，结合整个非白球体（灰度 < 235）的质心来精修圆心！
        r_win = 11
        x1, y1 = max(0, int(cx) - r_win), max(0, int(cy) - r_win)
        x2, y2 = min(w, int(cx) + r_win + 1), min(h, int(cy) + r_win + 1)
        patch_gray = gray[y1:y2, x1:x2]
        # 球体像素（无论是黄色扇形还是灰色金属）灰度都明显低于纯白背景(< 238)
        body_mask = (patch_gray < 235).astype(np.uint8)
        bm = cv2.moments(body_mask)
        if bm["m00"] > 20:
            refined_cx = x1 + bm["m10"] / bm["m00"]
            refined_cy = y1 + bm["m01"] / bm["m00"]
            # 如果精修中心偏移在合理范围内（<= 4px），采用精修球心
            if np.hypot(refined_cx - cx, refined_cy - cy) <= 4.5:
                cx, cy = refined_cx, refined_cy

        px, py = int(round(cx)), int(round(cy))

        # 去重：如果两个色块属于同一个小兵（距离 < 14px），合并
        merged = False
        for idx, (ex, ey, er) in enumerate(detected_players):
            if np.hypot(px - ex, py - ey) < 14:
                detected_players[idx] = (int(round((px + ex) / 2)), int(round((py + ey) / 2)), 9)
                merged = True
                break
        if not merged:
            detected_players.append((px, py, 9))

    # 按画布中线 (x = w / 2) 区分左半场（我方）和右半场（敌方）
    center_x = w / 2.0
    ours = []
    enemies = []

    for c in detected_players:
        if c[0] > center_x:
            enemies.append(c)
        else:
            ours.append(c)

    enemies.sort(key=lambda c: c[0])

    # 将红光激活角色与我方列表对齐
    if active_circle is not None:
        matched = None
        for c in ours:
            if np.hypot(c[0] - active_circle[0], c[1] - active_circle[1]) < 18:
                matched = c
                break
        if matched is not None:
            active_circle = matched
        else:
            ours.append(active_circle)
    elif ours:
        active_circle = ours[0]

    allies = [c for c in ours if active_circle is None or np.hypot(c[0] - active_circle[0], c[1] - active_circle[1]) >= 8]

    return ours, allies, enemies, active_circle


# =========================================================
# Visualization & Math Helpers
# =========================================================
def draw_players_roles_overlay(screenshot_bgr, allies, enemies, active):
    """Draw allies/enemies/active with distinct colors."""
    out = screenshot_bgr

    # Enemies (right side): red/orange
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


def draw_obstacle_circles(screenshot_bgr, obstacles):
    for cx, cy, r in obstacles:
        cv2.circle(screenshot_bgr, (int(cx), int(cy)), int(r), (255, 0, 255), 2)
    return screenshot_bgr


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
    """
    全新统一场景感知管线：使用精准距离变换检测黑圆 + 彩色连通域检测全部玩家与敌人。
    """
    obstacles_field = detect_obstacles_accurate(screenshot_bgr)
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
    """
    注意：在 Graphwar 中，发射出的函数曲线 g(x) 会被系统自动平移，使得起点对齐角色中心 (x0, y0)，
    即实际轨迹为 y(x) = y0 + g(x) - g(x0)。
    只要画布宽高比与 [-25, 25] x [-15, 15] 完全一致，分段绝对值线性插值就会严丝合缝地穿过每个航点！
    """
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
            screenshot_vis = draw_obstacle_circles(screenshot_vis, scene["obstacles_field"])

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
                screenshot_base = draw_obstacle_circles(screenshot_base, scene["obstacles_field"])

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
            screenshot_vis = draw_obstacle_circles(screenshot_vis, scene["obstacles_field"])

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

            # 将起点设为 active_norm 并串联所有敌人
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