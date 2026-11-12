import os
import asyncio
import json
import traceback
import re
import io
import base64
import math
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# 嘗試載入 google-genai
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# 簡易 .env 讀取（免額外套件依賴）
if os.path.exists(".env"):
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception as e:
        print(f"[Warn] 讀取 .env 失敗: {e}")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

# 尋找中文字型
FONT_CANDIDATES = [
    "NotoSansTC-VariableFont_wght.ttf",
    os.path.join("claudeindex", "NotoSansTC-VariableFont_wght.ttf"),
    "C:/Windows/Fonts/msjh.ttc",       # 微軟正黑體
    "C:/Windows/Fonts/simsun.ttc",     # 新宋體
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansTC-Regular.otf"
]

RESOLVED_FONT_PATH = None
for p in FONT_CANDIDATES:
    if os.path.exists(p):
        RESOLVED_FONT_PATH = p
        break

app = FastAPI(
    title="校園無人機群飛 AI 編排教室 - 3D 奔放幾何導演引擎",
    version="2.0.0",
    description="結合 Google Gemini AI 導演與 NumPy 3D 幾何工匠的無人機群飛編排後端"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- 資料模型定義 -----------------
class PrimitiveElement(BaseModel):
    shape_type: str = Field(description="圖元類型: ground_grid, text, heart, ring, spiral, sphere, star, cone, wave, cylinder, cube")
    ratio: float = Field(default=1.0, description="佔用無人機總數的比例權重 (例如 0.6 代表佔 60% 架數)")
    text: Optional[str] = Field(default="", description="若為 text 類型時的文字內容，如 '富邦' 或 'SLHS'")
    center: Optional[List[float]] = Field(default=[0.0, 4.0, 3.8], description="三維中心座標 [x, y, z]")
    scale: Optional[List[float]] = Field(default=[1.0, 1.0, 1.0], description="三維尺寸縮放係數 [sx, sy, sz]")
    rotate_deg: Optional[List[float]] = Field(default=[0.0, 0.0, 0.0], description="三維旋轉角度 [rx, ry, rz]，單位為度")
    colors: Optional[List[str]] = Field(default=["#00F0FF"], description="色標陣列 (支援多色漸層)，例如 ['#FFFFFF', '#A020F0']")
    gradient_direction: Optional[str] = Field(default="none", description="漸層方向: none, z_axis_up, z_axis_down, x_axis, y_axis, radial")

class SceneDirective(BaseModel):
    scene_name: str = Field(description="幕名，例如 '第一幕：地面起飛' 或 '第三幕：百步蛇立體光環'")
    elements: List[PrimitiveElement] = Field(description="構成該幕的一組或多組幾何圖元")

class DirectorScript(BaseModel):
    scenes: List[SceneDirective]

class ShowRequest(BaseModel):
    student_name: str
    drone_count: int
    prompt: str
    api_key: Optional[str] = None
    model_name: Optional[str] = None

class ImageSceneRequest(BaseModel):
    image_base64: str
    drone_count: int
    mode: Optional[str] = "pixel"
    scene_name: Optional[str] = "自訂圖片造型"

request_lock = asyncio.Lock()

# ----------------- 色彩工具 -----------------
def hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        return (0, 240, 255)
    try:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (0, 240, 255)

def rgb_to_hex(r, g, b):
    return f"#{max(0, min(255, int(r))):02X}{max(0, min(255, int(g))):02X}{max(0, min(255, int(b))):02X}"

def interpolate_colors(color_hex_list: List[str], ratio: float):
    if not color_hex_list:
        return "#00F0FF"
    if len(color_hex_list) == 1:
        return color_hex_list[0]
    
    clamped_ratio = max(0.0, min(1.0, ratio))
    scaled = clamped_ratio * (len(color_hex_list) - 1)
    idx = int(scaled)
    t = scaled - idx

    if idx >= len(color_hex_list) - 1:
        return color_hex_list[-1]

    c1 = hex_to_rgb(color_hex_list[idx])
    c2 = hex_to_rgb(color_hex_list[idx + 1])
    r = c1[0] + (c2[0] - c1[0]) * t
    g = c1[1] + (c2[1] - c1[1]) * t
    b = c1[2] + (c2[2] - c1[2]) * t
    return rgb_to_hex(r, g, b)

# ----------------- 3D 幾何變換 -----------------
def apply_transforms(coords: np.ndarray, center: List[float], scale: List[float], rot_deg: List[float]):
    """套用 3D 縮放、歐拉角旋轉矩陣與空間平移"""
    if len(coords) == 0:
        return coords

    # 縮放
    s = np.array(scale if scale and len(scale) == 3 else [1.0, 1.0, 1.0], dtype=float)
    scaled = coords * s

    # 旋轉
    r = rot_deg if rot_deg and len(rot_deg) == 3 else [0.0, 0.0, 0.0]
    rx, ry, rz = np.radians(r)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    rotated = scaled @ R.T

    # 平移
    c = np.array(center if center and len(center) == 3 else [0.0, 4.0, 3.8], dtype=float)
    transformed = rotated + c
    return transformed

# ----------------- 圖元生成器 -----------------
def generate_primitive_coords(shape: str, n: int, text_content: str = "") -> np.ndarray:
    shape = (shape or "").lower().strip()
    if n <= 0:
        return np.zeros((0, 3))

    if shape == "ground_grid":
        cols = math.ceil(math.sqrt(n * 1.5))
        rows = math.ceil(n / cols)
        pts = []
        for i in range(n):
            r, c = i // cols, i % cols
            x = -5.0 + (c / max(cols - 1, 1)) * 10.0
            y = 2.8 + (r / max(rows - 1, 1)) * 2.4
            z = 0.0
            pts.append([x, y, z])
        return np.array(pts)

    elif shape == "heart":
        t = np.linspace(0, 2 * math.pi, n)
        x = 16 * np.sin(t)**3 / 16.0 * 3.5
        z = (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)) / 16.0 * 3.5
        y = np.random.uniform(-0.35, 0.35, n)
        return np.column_stack([x, y, z])

    elif shape == "ring":
        theta = np.linspace(0, 2 * math.pi, n)
        r = np.random.uniform(3.0, 3.4, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(n)
        return np.column_stack([x, y, z])

    elif shape == "spiral":
        t = np.linspace(0, 1, n)
        z = (t - 0.5) * 4.0
        r = 0.6 + t * 2.8
        theta = t * 6 * math.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y, z])

    elif shape == "sphere":
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = math.pi * (1 + 5**0.5) * indices
        r = 3.0
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        return np.column_stack([x, y, z])

    elif shape == "star":
        angles = np.linspace(0, 4 * math.pi, n)
        r = 3.2 * (0.5 + 0.5 * (np.sin(5 * angles / 2)**2))
        x = r * np.cos(angles)
        z = r * np.sin(angles)
        y = np.random.uniform(-0.2, 0.2, n)
        return np.column_stack([x, y, z])

    elif shape == "cone":
        t = np.linspace(0, 1, n)
        z = (t - 0.5) * 4.0
        r = (1.0 - t) * 3.0
        theta = t * 10 * math.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y, z])

    elif shape == "wave":
        u = np.linspace(-3.5, 3.5, int(math.sqrt(n * 1.4)))
        v = np.linspace(-2.0, 2.0, max(1, n // len(u)))
        U, V = np.meshgrid(u, v)
        x_flat = U.flatten()[:n]
        y_flat = V.flatten()[:n]
        z_flat = 1.2 * np.sin(x_flat * 0.8) * np.cos(y_flat * 0.8)
        # 若數量不足則補齊
        if len(x_flat) < n:
            pad = n - len(x_flat)
            x_flat = np.append(x_flat, np.random.uniform(-3.5, 3.5, pad))
            y_flat = np.append(y_flat, np.random.uniform(-2.0, 2.0, pad))
            z_flat = np.append(z_flat, 1.2 * np.sin(x_flat[-pad:] * 0.8))
        return np.column_stack([x_flat[:n], y_flat[:n], z_flat[:n]])

    elif shape == "cylinder":
        theta = np.linspace(0, 2 * math.pi, n)
        z = np.linspace(-2.0, 2.0, n)
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y, z])

    elif shape == "cube":
        # 3D 立方體表面點分佈
        pts = []
        side = 3.0
        for _ in range(n):
            face = np.random.randint(0, 6)
            u, v = np.random.uniform(-side/2, side/2, 2)
            if face == 0: pts.append([ side/2, u, v])
            elif face == 1: pts.append([-side/2, u, v])
            elif face == 2: pts.append([u,  side/2, v])
            elif face == 3: pts.append([u, -side/2, v])
            elif face == 4: pts.append([u, v,  side/2])
            else:           pts.append([u, v, -side/2])
        return np.array(pts)

    elif shape == "text" or text_content:
        chars = list(text_content) if text_content else ["?"]
        c_count = len(chars)
        pts_per_c = n // c_count
        char_coords_list = []
        
        for c_idx, ch in enumerate(chars):
            dim = 140
            img = Image.new("L", (dim, dim), color=0)
            draw = ImageDraw.Draw(img)
            
            font = None
            if RESOLVED_FONT_PATH:
                try:
                    font = ImageFont.truetype(RESOLVED_FONT_PATH, 100)
                except Exception:
                    font = None
            if font is None:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), ch, font=font)
            tw, th = max(bbox[2] - bbox[0], 1), max(bbox[3] - bbox[1], 1)
            draw.text(((dim - tw)//2, (dim - th)//2), ch, font=font, fill=255)
            arr = np.array(img)
            c_coords = np.argwhere(arr > 80)
            
            alloc_n = pts_per_c if c_idx < c_count - 1 else (n - len(char_coords_list))
            if len(c_coords) == 0:
                c_coords = np.zeros((alloc_n, 2))

            if len(c_coords) >= alloc_n:
                sampled = c_coords[np.linspace(0, len(c_coords)-1, alloc_n, dtype=int)]
            else:
                rep = (alloc_n // len(c_coords)) + 1
                sampled = np.tile(c_coords, (rep, 1))[:alloc_n]

            slot_offset_x = (-4.0 + (c_idx + 0.5) * (8.0 / c_count)) if c_count > 1 else 0.0
            min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
            min_c, max_c = np.min(sampled[:, 1]), np.max(sampled[:, 1])
            norm_x = (sampled[:, 1] - min_c) / max(max_c - min_c, 1) * (6.5 / c_count) - (3.25 / c_count) + slot_offset_x
            norm_z = (max_r - sampled[:, 0]) / max(max_r - min_r, 1) * 3.5 - 1.75
            norm_y = np.random.uniform(-0.15, 0.15, alloc_n)
            char_coords_list.extend(np.column_stack([norm_x, norm_y, norm_z]))

        return np.array(char_coords_list)

    else:
        # 預設環形
        t = np.linspace(0, 2 * math.pi, n)
        x = 3.2 * np.cos(t)
        z = 3.2 * np.sin(t)
        y = np.zeros(n)
        return np.column_stack([x, y, z])

# ----------------- 幾何渲染核心 -----------------
def render_directive_to_points(directive: SceneDirective, total_drones: int):
    elements = directive.elements
    if not elements:
        elements = [PrimitiveElement(shape_type="text", text=directive.scene_name, ratio=1.0)]

    total_ratio = sum(max(0.05, el.ratio) for el in elements)
    allocated_counts = []
    accum = 0
    for idx, el in enumerate(elements):
        if idx == len(elements) - 1:
            cnt = total_drones - accum
        else:
            cnt = int(total_drones * (max(0.05, el.ratio) / total_ratio))
            accum += cnt
        allocated_counts.append(cnt)

    scene_points = []
    for el, count in zip(elements, allocated_counts):
        if count <= 0:
            continue
        base_coords = generate_primitive_coords(el.shape_type, count, el.text or "")
        transformed = apply_transforms(base_coords, el.center, el.scale, el.rotate_deg)

        colors = el.colors or ["#00F0FF"]
        grad_dir = el.gradient_direction or "none"
        
        if len(transformed) > 0:
            min_x, max_x = float(np.min(transformed[:, 0])), float(np.max(transformed[:, 0]))
            min_y, max_y = float(np.min(transformed[:, 1])), float(np.max(transformed[:, 1]))
            min_z, max_z = float(np.min(transformed[:, 2])), float(np.max(transformed[:, 2]))
        else:
            min_x, max_x, min_y, max_y, min_z, max_z = -1, 1, -1, 1, -1, 1

        for pt in transformed:
            px, py, pz = round(float(pt[0]), 2), round(float(pt[1]), 2), round(float(pt[2]), 2)
            
            if grad_dir == "z_axis_up":
                ratio = (pz - min_z) / max(max_z - min_z, 0.01)
            elif grad_dir == "z_axis_down":
                ratio = (max_z - pz) / max(max_z - min_z, 0.01)
            elif grad_dir == "x_axis":
                ratio = (px - min_x) / max(max_x - min_x, 0.01)
            elif grad_dir == "y_axis":
                ratio = (py - min_y) / max(max_y - min_y, 0.01)
            elif grad_dir == "radial":
                c_x = el.center[0] if el.center else 0.0
                c_z = el.center[2] if el.center else 3.8
                dist = math.sqrt((px - c_x)**2 + (pz - c_z)**2)
                ratio = min(1.0, dist / 4.0)
            else:
                ratio = 0.0

            color_hex = interpolate_colors(colors, ratio)
            scene_points.append({"x": px, "y": py, "z": pz, "color": color_hex})

    return scene_points

# ----------------- 圖片轉點陣演算法 -----------------
def zhang_suen_thinning(binary_image: np.ndarray) -> np.ndarray:
    """張-孫細化骨架演算法"""
    img = binary_image.copy()
    prev = np.zeros_like(img)
    while True:
        p2 = np.roll(img, -1, axis=0)
        p3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
        p4 = np.roll(img, 1, axis=1)
        p5 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
        p6 = np.roll(img, 1, axis=0)
        p7 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
        p8 = np.roll(img, -1, axis=1)
        p9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(int) +
            ((p3 == 0) & (p4 == 1)).astype(int) +
            ((p4 == 0) & (p5 == 1)).astype(int) +
            ((p5 == 0) & (p6 == 1)).astype(int) +
            ((p6 == 0) & (p7 == 1)).astype(int) +
            ((p7 == 0) & (p8 == 1)).astype(int) +
            ((p8 == 0) & (p9 == 1)).astype(int) +
            ((p9 == 0) & (p2 == 1)).astype(int)
        )
        c1 = (img == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
        c2 = (p2 * p4 * p6 == 0)
        c3 = (p4 * p6 * p8 == 0)
        img[c1 & c2 & c3] = 0

        p2 = np.roll(img, -1, axis=0)
        p3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
        p4 = np.roll(img, 1, axis=1)
        p5 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
        p6 = np.roll(img, 1, axis=0)
        p7 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
        p8 = np.roll(img, -1, axis=1)
        p9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(int) +
            ((p3 == 0) & (p4 == 1)).astype(int) +
            ((p4 == 0) & (p5 == 1)).astype(int) +
            ((p5 == 0) & (p6 == 1)).astype(int) +
            ((p6 == 0) & (p7 == 1)).astype(int) +
            ((p7 == 0) & (p8 == 1)).astype(int) +
            ((p8 == 0) & (p9 == 1)).astype(int) +
            ((p9 == 0) & (p2 == 1)).astype(int)
        )
        c1 = (img == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
        c2 = (p2 * p4 * p8 == 0)
        c3 = (p2 * p6 * p8 == 0)
        img[c1 & c2 & c3] = 0

        if np.array_equal(img, prev):
            break
        prev = img.copy()
    return img

def image_to_drone_points(image_bytes: bytes, total_n: int, mode: str = "pixel"):
    pil_img = Image.open(io.BytesIO(image_bytes))
    rgba_img = pil_img.convert("RGBA")
    grid_dim = 96
    w, h = rgba_img.size
    scale = grid_dim / max(w, h)
    new_w, new_h = max(10, int(w * scale)), max(10, int(h * scale))
    resized = rgba_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_arr = np.array(resized)

    r_chan, g_chan, b_chan, alpha = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2], img_arr[:, :, 3]
    is_bg = (alpha < 80) | ((r_chan > 235) & (g_chan > 235) & (b_chan > 235))
    subject_coords = np.argwhere(~is_bg)
    if len(subject_coords) < 50:
        subject_coords = np.argwhere(alpha > 0)

    if len(subject_coords) == 0:
        # 回退至置中文字
        return [{"x": 0.0, "y": 4.0, "z": 3.8, "color": "#00F0FF"} for _ in range(total_n)]

    if mode == "pixel":
        if len(subject_coords) >= total_n:
            sampled = subject_coords[np.linspace(0, len(subject_coords) - 1, total_n, dtype=int)]
        else:
            sampled = np.tile(subject_coords, ((total_n // len(subject_coords)) + 1, 1))[:total_n]

        min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
        min_c, max_c = np.min(sampled[:, 1]), np.max(sampled[:, 1])
        range_r, range_c = max(max_r - min_r, 1), max(max_c - min_c, 1)

        pts = []
        for r, c in sampled:
            norm_x, norm_z = (c - min_c) / range_c, (max_r - r) / range_r
            px = round(-4.5 + norm_x * 9.0, 2)
            py = 4.0
            pz = round(1.8 + norm_z * 4.5, 2)
            r_val, g_val, b_val, _ = img_arr[int(r), int(c)]
            hex_color = "#1E293B" if (r_val < 30 and g_val < 30 and b_val < 30) else f"#{r_val:02X}{g_val:02X}{b_val:02X}"
            pts.append({"x": px, "y": py, "z": pz, "color": hex_color})
        return pts
    else:
        binary = (~is_bg).astype(np.uint8)
        skeleton = zhang_suen_thinning(binary)
        coords = np.argwhere(skeleton == 1)
        if len(coords) < 30: 
            coords = subject_coords
        if len(coords) >= total_n:
            sampled = coords[np.linspace(0, len(coords) - 1, total_n, dtype=int)]
        else:
            sampled = np.tile(coords, ((total_n // len(coords)) + 1, 1))[:total_n]

        min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
        min_c, max_c = np.min(sampled[:, 1]), np.max(sampled[:, 1])
        range_r, range_c = max(max_r - min_r, 1), max(max_c - min_c, 1)

        pts = []
        for r, c in sampled:
            px = round(-4.5 + ((c - min_c) / range_c) * 9.0, 2)
            py = 4.0
            pz = round(1.8 + ((max_r - r) / range_r) * 4.5, 2)
            pts.append({"x": px, "y": py, "z": pz, "color": "#00F0FF"})
        return pts

# ----------------- AI 導演提示詞 -----------------
CREATIVE_DIRECTOR_PROMPT = """你是一位完全解鎖想像力的 3D 無人機大秀編導兼圖學專家。
閱讀使用者的演出劇本，將每一幕拆解為多個三維幾何圖元 (elements) 及其變形與色彩，絕不輸出具體點座標。

可用圖元 (shape_type):
- ground_grid: 地面停機坪 (僅用於起飛/第一幕)
- text: 中英文字/字母 (填寫 text 欄位，如 '富邦'、'SLHS')
- heart: 3D 心形
- ring: 圓環、光環、土星環
- spiral: 螺旋、龍捲風
- sphere: 立體球體、星體
- star: 立體五角星
- cone: 圓錐、金字塔錐體
- wave: 波動海浪曲面
- cylinder: 圓柱體
- cube: 立方體

自由變形與色彩屬性:
1. ratio: 該圖元佔用無人機總數的比重 (例如主要文字占 0.6，周圍環繞光環占 0.4)
2. center: 三維中心點 [x, y, z]。X 範圍 [-5.0, 5.0], Y 深度 [2.0, 6.0] (2.4最前, 4.0中間, 5.2後排), Z 高度 [1.5, 6.5]
3. scale: [sx, sy, sz] 尺寸縮放
4. rotate_deg: [rx, ry, rz] 任意旋轉傾斜角 (例如傾斜 30 度: [30, 0, 0])
5. colors: 任意色彩陣列 (支援多色漸層，如 ['#FFFFFF', '#A020F0'] 或虹彩)
6. gradient_direction: 漸層方向 (z_axis_up, z_axis_down, x_axis, y_axis, radial, none)

範例情境處理:
- 「地面起飛」: elements=[{shape_type: "ground_grid", ratio: 1.0, colors: ["#00F0FF"]}]
- 「富邦，一前一後，富為藍色，邦為紫色由上往下漸層」:
  富: text="富", center=[-1.8, 2.5, 3.8], colors=['#008CFF']
  邦: text="邦", center=[1.8, 5.2, 3.8], colors=['#FFFFFF', '#A020F0'], gradient_direction='z_axis_down'
- 「斜躺的粉紅心形，外圍被金色光環圍繞」:
  心: shape_type="heart", ratio=0.6, colors=['#FF69B4'], rotate_deg=[35, 0, 0]
  環: shape_type="ring", ratio=0.4, colors=['#FFD700'], rotate_deg=[35, 0, 0]
"""

# ----------------- 路由端點 -----------------
@app.get("/", response_class=HTMLResponse)
@app.head("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>校園無人機群飛 AI 編排教室運作中！</h1><p>請確認 index.html 存在同目錄。</p>")

@app.get("/api/status")
def api_status():
    return {
        "status": "online",
        "has_api_key": bool(GEMINI_API_KEY),
        "font_path": RESOLVED_FONT_PATH,
        "font_exists": bool(RESOLVED_FONT_PATH and os.path.exists(RESOLVED_FONT_PATH)),
        "available_primitives": ["ground_grid", "text", "heart", "ring", "spiral", "sphere", "star", "cone", "wave", "cylinder", "cube"],
        "message": "無人機群飛 3D 奔放幾何導演引擎運作中！"
    }

@app.post("/api/convert-image-to-drone")
async def convert_image_endpoint(req: ImageSceneRequest):
    try:
        header, encoded = req.image_base64.split(",", 1) if "," in req.image_base64 else ("", req.image_base64)
        image_data = base64.b64decode(encoded)
        points = image_to_drone_points(image_data, req.drone_count, mode=req.mode or "pixel")
        if not points:
            raise HTTPException(status_code=400, detail="無法提取點陣造型")
        return {
            "status": "success",
            "scene": {
                "name": req.scene_name or "圖片造型",
                "points": points
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"圖片解析失敗: {str(e)}")

@app.post("/api/generate-show")
async def generate_show(req: ShowRequest):
    active_key = (req.api_key or "").strip() or GEMINI_API_KEY
    chosen_model = (req.model_name or "").strip() or DEFAULT_MODEL

    async with request_lock:
        print(f"[{req.student_name}] 正在呼叫奔放導演思考中 (Model: {chosen_model})...")
        user_content = f"演出架數: {req.drone_count}\n劇本需求:\n{req.prompt}"

        directives = []
        ai_success = False

        if active_key and HAS_GENAI:
            # 支援備援模型切換
            candidate_models = [chosen_model, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            unique_models = []
            for m in candidate_models:
                if m not in unique_models:
                    unique_models.append(m)

            for try_model in unique_models:
                try:
                    client = genai.Client(api_key=active_key)
                    def call_director():
                        return client.models.generate_content(
                            model=try_model,
                            contents=user_content,
                            config=types.GenerateContentConfig(
                                system_instruction=CREATIVE_DIRECTOR_PROMPT,
                                response_mime_type="application/json",
                                response_schema=DirectorScript,
                                temperature=0.2
                            )
                        )

                    ai_resp = await asyncio.wait_for(asyncio.to_thread(call_director), timeout=35.0)
                    director_data = json.loads(ai_resp.text.strip())
                    directives = director_data.get("scenes", [])
                    print(f"[{req.student_name}] AI 導演成功建構 {len(directives)} 組複合幾何劇幕！(使用模型: {try_model})")
                    ai_success = True
                    break
                except Exception as ex:
                    print(f"[Model {try_model}] 呼叫失敗: {ex}")
                    continue

        # 幾何備援機制（無 API Key、配額超額或解析失敗時自動啟動）
        if not ai_success or not directives:
            print(f"[{req.student_name}] 啟動智慧幾何備援引擎分析劇本...")
            lines = [l.strip() for l in req.prompt.split("\n") if l.strip()]
            directives = []
            for idx, line in enumerate(lines):
                clean_text = re.sub(r"^第[一二三四五六七八九十\d]+幕[：:]\s*", "", line)
                if idx == 0 or "起飛" in line or "停機" in line:
                    directives.append(SceneDirective(
                        scene_name=line,
                        elements=[PrimitiveElement(shape_type="ground_grid", ratio=1.0, colors=["#00F0FF"])]
                    ))
                elif "心" in line or "愛心" in line:
                    directives.append(SceneDirective(
                        scene_name=line,
                        elements=[PrimitiveElement(shape_type="heart", ratio=1.0, colors=["#FF2A85", "#FFAAE0"], gradient_direction="radial")]
                    ))
                elif "環" in line or "太陽" in line or "日" in line or "輪" in line:
                    directives.append(SceneDirective(
                        scene_name=line,
                        elements=[PrimitiveElement(shape_type="ring", ratio=1.0, colors=["#FFB703", "#FB8500"], rotate_deg=[25, 0, 0])]
                    ))
                elif "星" in line or "芒" in line:
                    directives.append(SceneDirective(
                        scene_name=line,
                        elements=[PrimitiveElement(shape_type="star", ratio=1.0, colors=["#FFD700", "#FFF4B8"])]
                    ))
                elif "螺旋" in line or "蛇" in line or "風" in line or "龍" in line:
                    directives.append(SceneDirective(
                        scene_name=line,
                        elements=[PrimitiveElement(shape_type="spiral", ratio=1.0, colors=["#00F5D4", "#7B2CBF"], gradient_direction="z_axis_up")]
                    ))
                elif "球" in line or "地球" in line or "眼" in line:
                    directives.append(SceneDirective(
                        scene_name=line,
                        elements=[PrimitiveElement(shape_type="sphere", ratio=1.0, colors=["#4361EE", "#4CC9F0"], gradient_direction="radial")]
                    ))
                elif "浪" in line or "海" in line or "波" in line:
                    directives.append(SceneDirective(
                        scene_name=line,
                        elements=[PrimitiveElement(shape_type="wave", ratio=1.0, colors=["#0077B6", "#90E0EF"])]
                    ))
                else:
                    # 預設為高品質文字骨架
                    directives.append(SceneDirective(
                        scene_name=line,
                        elements=[PrimitiveElement(shape_type="text", text=clean_text or line, ratio=1.0, colors=["#00F0FF", "#A020F0"], gradient_direction="z_axis_down")]
                    ))

        # 幾何工匠：秒算滿編無人機空間點位
        rendered_scenes = []
        for d in directives:
            directive_obj = SceneDirective(**d) if isinstance(d, dict) else d
            pts = render_directive_to_points(directive_obj, req.drone_count)
            rendered_scenes.append({"name": directive_obj.scene_name, "points": pts})

        print(f"[{req.student_name}] 全場幾何渲染完成，共 {len(rendered_scenes)} 幕！")
        return {"status": "success", "data": json.dumps({"scenes": rendered_scenes})}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 無人機群飛導演伺服器啟動中: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
