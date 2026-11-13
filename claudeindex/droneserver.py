import os
import sys

# 確保 Windows 主控台支援 UTF-8 輸出
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import math
import time
import base64
import json
import re
import random
import asyncio
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Google GenAI SDK (支援 Gemini 3.8 / 3.0 / 2.5 系列)
from google import genai
from google.genai import types

app = FastAPI(title="校園無人機群飛 AI 編排教室 2.0 (全自由幾何架構)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
FONT_PATH = "NotoSansTC-VariableFont_wght.ttf"
SHAPE_NAMES = {
    "ground_grid": "地面停機坪陣列",
    "sphere": "立體球體/軀幹",
    "spiral": "三維螺旋/捲曲長鼻",
    "ring": "立體光環/圓環/耳朵",
    "star": "立體星芒/菱形",
    "box": "3D 立方體/城垛",
    "cylinder": "3D 圓柱體/四肢立柱",
    "cone": "3D 圓錐體/尖塔",
    "wave": "3D 波浪曲面/羽翼",
    "heart": "3D 雙層愛心",
    "text": "高解析漢字點陣"
}

request_lock = asyncio.Lock()

@app.get("/")
@app.head("/")
def read_root():
    return {
        "status": "online",
        "has_api_key": bool(GEMINI_API_KEY),
        "font_exists": os.path.exists(FONT_PATH),
        "engine": "AI Python 幾何程式設計師 (Tier 1) + 複合幾何圖元 (Tier 2)",
        "message": "無人機群飛 3D 自由編排引擎運作中！"
    }

@app.get("/api/test-ai")
def test_ai():
    if not GEMINI_API_KEY:
        return {"status": "error", "message": "GEMINI_API_KEY is not set"}
    
    results = {}
    http_opts = types.HttpOptions(timeout=15000)
    client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts)
    
    available_models = []
    try:
        for m in client.models.list():
            available_models.append(m.name)
    except Exception as e:
        available_models = [f"list_error: {str(e)[:150]}"]
    
    test_candidates = [
        "models/gemini-3.5-flash",
        "models/gemini-3.5-flash-lite",
        "models/gemini-3.1-flash-lite",
        "models/gemini-3.6-flash",
        "models/gemini-omni-flash-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite"
    ]
    for am in available_models:
        if isinstance(am, str) and not am.startswith("list_error") and ("3." in am or "gemma" in am):
            if am not in test_candidates:
                test_candidates.append(am)
            
    if not test_candidates:
        test_candidates = [
            "models/gemini-3.8-flash", "gemini-3.8-flash",
            "models/gemini-2.5-flash", "gemini-2.5-flash",
            "models/gemini-1.5-flash", "gemini-1.5-flash"
        ]
    
    for m in test_candidates[:4]:
        try:
            resp = client.models.generate_content(
                model=m,
                contents="ping"
            )
            results[m] = {"status": "success", "text": resp.text.strip()}
            return {"status": "ok", "working_model": m, "available_models": available_models[:15], "results": results}
        except Exception as e:
            results[m] = {"status": "error", "msg": str(e)[:150]}
            
    return {"status": "all_failed", "available_models": available_models[:15], "results": results}

# ----------------- 資料模型定義 -----------------
class PrimitiveElement(BaseModel):
    shape_type: str = Field(description="圖元類型: ground_grid, text, heart, ring, spiral, sphere, star, box, cylinder, cone, wave")
    ratio: float = Field(default=1.0, description="佔用無人機總數的比例權重 (例如 0.6 代表佔 60% 架數)")
    text: Optional[str] = Field(default="", description="若為 text 類型時的文字內容，如 '富邦' 或 'SLHS'")
    center: Optional[List[float]] = Field(default=[0.0, 4.0, 3.8], description="三維中心座標 [x, y, z]")
    scale: Optional[List[float]] = Field(default=[1.0, 1.0, 1.0], description="三維尺寸縮放係數 [sx, sy, sz]")
    rotate_deg: Optional[List[float]] = Field(default=[0.0, 0.0, 0.0], description="三維旋轉角度 [rx, ry, rz]")
    colors: Optional[List[str]] = Field(default=["#00F0FF"], description="色標陣列 (支援多色漸層)")
    gradient_direction: Optional[str] = Field(default="none", description="漸層方向: none, z_axis_up, z_axis_down, x_axis, y_axis, radial")

class SceneDirective(BaseModel):
    scene_name: str = Field(description="幕名，例如 '第一幕：地面停機坪排列' 或 '第二幕：富邦立體漸層光環'")
    elements: List[PrimitiveElement] = Field(description="構成該幕的一組或多組幾何圖元")

class DirectorScript(BaseModel):
    scenes: List[SceneDirective]

class ShowRequest(BaseModel):
    student_name: str
    drone_count: int = 3000
    prompt: str

class ImageSceneRequest(BaseModel):
    image_base64: str
    drone_count: int = 3000
    scene_name: Optional[str] = "圖片造型"
    mode: Optional[str] = "pixel"

# ----------------- 色彩工具函數 -----------------
def hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c*2 for c in hex_str])
    if len(hex_str) < 6:
        hex_str = hex_str.ljust(6, '0')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return f"#{int(max(0, min(255, r))):02X}{int(max(0, min(255, g))):02X}{int(max(0, min(255, b))):02X}"

def interpolate_colors(color_hex_list: List[str], ratio: float) -> str:
    if not color_hex_list:
        return "#00F0FF"
    if len(color_hex_list) == 1 or ratio <= 0.0:
        return color_hex_list[0]
    if ratio >= 1.0:
        return color_hex_list[-1]

    n_segments = len(color_hex_list) - 1
    segment = ratio * n_segments
    idx = int(segment)
    t = segment - idx

    if idx >= len(color_hex_list) - 1:
        return color_hex_list[-1]

    c1 = hex_to_rgb(color_hex_list[idx])
    c2 = hex_to_rgb(color_hex_list[idx + 1])
    r = c1[0] + (c2[0] - c1[0]) * t
    g = c1[1] + (c2[1] - c1[1]) * t
    b = c1[2] + (c2[2] - c1[2]) * t
    return rgb_to_hex(r, g, b)

def apply_transforms(coords: np.ndarray, center: List[float], scale: List[float], rot_deg: List[float]):
    """套用 3D 旋轉矩陣、縮放與空間平移"""
    scaled = coords * np.array(scale)
    rx, ry, rz = np.radians(rot_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    rotated = scaled @ R.T
    transformed = rotated + np.array(center)
    return transformed

# ----------------- 幾何圖元生成器 (擴充圖元庫) -----------------
def generate_primitive_coords(shape: str, n: int, text_content: str = ""):
    if shape == "ground_grid":
        cols = math.ceil(math.sqrt(n * 2.2))
        rows = math.ceil(n / cols)
        pts = []
        for i in range(n):
            r, c = i // cols, i % cols
            x = -5.5 + (c / max(cols - 1, 1)) * 11.0
            y = 2.4 + (r / max(rows - 1, 1)) * 3.2
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
        y = np.zeros(n)
        return np.column_stack([x, y, z])

    elif shape == "box":
        # 3D 立方體/城垛/箱體表面均勻點陣
        pts = []
        face_counts = [n // 6] * 6
        face_counts[-1] = n - sum(face_counts[:-1])
        for f, cnt in enumerate(face_counts):
            if cnt <= 0: continue
            u = np.random.uniform(-1.5, 1.5, cnt)
            v = np.random.uniform(-1.5, 1.5, cnt)
            w = np.random.uniform(-0.5, 0.5, cnt)
            if f == 0: pts.append(np.column_stack([np.full(cnt, -1.5), w, v]))
            elif f == 1: pts.append(np.column_stack([np.full(cnt, 1.5), w, v]))
            elif f == 2: pts.append(np.column_stack([u, w, np.full(cnt, -1.5)]))
            elif f == 3: pts.append(np.column_stack([u, w, np.full(cnt, 1.5)]))
            elif f == 4: pts.append(np.column_stack([u, np.full(cnt, -0.5), v]))
            elif f == 5: pts.append(np.column_stack([u, np.full(cnt, 0.5), v]))
        return np.vstack(pts) if pts else np.zeros((n, 3))

    elif shape == "cylinder":
        # 3D 圓柱體/火箭主體/光柱
        theta = np.linspace(0, 2 * math.pi, n)
        z = np.linspace(-2.0, 2.0, n)
        r = 1.3
        x = r * np.cos(theta)
        y = np.random.uniform(-0.2, 0.2, n)
        return np.column_stack([x, y, z])

    elif shape == "cone":
        # 3D 圓錐體/尖塔/山峰
        z = np.linspace(-2.0, 2.0, n)
        t = (2.0 - z) / 4.0
        theta = np.linspace(0, 16 * math.pi, n)
        r = t * 2.2
        x = r * np.cos(theta)
        y = np.random.uniform(-0.15, 0.15, n)
        return np.column_stack([x, y, z])

    elif shape == "wave":
        # 3D 波浪/飛鳥羽翼/裙擺
        x = np.linspace(-4.0, 4.0, n)
        y = np.random.uniform(-0.5, 0.5, n)
        z = np.sin(x * 1.2) * 1.5
        return np.column_stack([x, y, z])

    elif shape == "text" or text_content:
        # 文字骨架點陣 (支援多字排版、高解析防裁切、等比例縮放與筆劃加粗)
        chars = list(text_content) if text_content else ["?"]
        c_count = len(chars)
        pts_per_c = n // c_count
        char_coords_list = []
        
        span_x = min(8.6, max(2.5, c_count * 2.1))
        slot_w = span_x / c_count
        max_h = min(2.8, slot_w * 1.25)

        for c_idx, ch in enumerate(chars):
            dim = 200
            img = Image.new("L", (dim, dim), color=0)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(FONT_PATH, 110)
            except Exception:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw = max(bbox[2] - bbox[0], 1)
            th = max(bbox[3] - bbox[1], 1)
            offset_x = (dim - tw) // 2 - bbox[0]
            offset_y = (dim - th) // 2 - bbox[1]
            draw.text((offset_x, offset_y), ch, font=font, fill=255, stroke_width=2)
            
            arr = np.array(img)
            c_coords = np.argwhere(arr > 90)
            
            alloc_n = pts_per_c if c_idx < c_count - 1 else (n - len(char_coords_list))
            if len(c_coords) == 0:
                c_coords = np.zeros((alloc_n, 2))

            if len(c_coords) >= alloc_n:
                sampled = c_coords[np.linspace(0, len(c_coords)-1, alloc_n, dtype=int)]
            else:
                rep = (alloc_n // len(c_coords)) + 1
                sampled = np.tile(c_coords, (rep, 1))[:alloc_n]

            min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
            min_c, max_c = np.min(sampled[:, 1]), np.max(sampled[:, 1])
            h = max(max_r - min_r, 1)
            w = max(max_c - min_c, 1)

            scale = min((slot_w * 0.88) / w, max_h / h)
            slot_center_x = -span_x / 2.0 + (c_idx + 0.5) * slot_w

            norm_x = (sampled[:, 1] - (min_c + max_c) / 2.0) * scale + slot_center_x
            norm_z = ((min_r + max_r) / 2.0 - sampled[:, 0]) * scale
            norm_y = np.random.uniform(-0.1, 0.1, alloc_n)
            char_coords_list.extend(np.column_stack([norm_x, norm_y, norm_z]))

        return np.array(char_coords_list)

    else:
        # 預設圓形環
        t = np.linspace(0, 2 * math.pi, n)
        x = 3.2 * np.cos(t)
        z = 3.2 * np.sin(t)
        y = np.zeros(n)
        return np.column_stack([x, y, z])

# ----------------- 圖元渲染整合核心 (Tier 2 備援) -----------------
def render_directive_to_points(directive: SceneDirective, total_drones: int):
    elements = directive.elements
    if not elements:
        elements = [PrimitiveElement(shape_type="text", text=directive.scene_name, ratio=1.0)]

    total_ratio = sum(max(0.1, el.ratio) for el in elements)
    allocated_counts = []
    accum = 0
    for idx, el in enumerate(elements):
        if idx == len(elements) - 1:
            cnt = total_drones - accum
        else:
            cnt = int(total_drones * (el.ratio / total_ratio))
            accum += cnt
        allocated_counts.append(cnt)

    scene_points = []
    for el, count in zip(elements, allocated_counts):
        if count <= 0:
            continue
        base_coords = generate_primitive_coords(el.shape_type, count, el.text)
        if el.shape_type == "ground_grid":
            transformed = base_coords.copy()
        else:
            transformed = apply_transforms(base_coords, el.center, el.scale, el.rotate_deg)

        colors = el.colors or (["#00E676", "#00F0FF"] if el.shape_type == "ground_grid" else ["#00F0FF"])
        grad_dir = el.gradient_direction or ("y_axis" if el.shape_type == "ground_grid" else "none")
        
        min_x, max_x = np.min(transformed[:, 0]), np.max(transformed[:, 0])
        min_y, max_y = np.min(transformed[:, 1]), np.max(transformed[:, 1])
        min_z, max_z = np.min(transformed[:, 2]), np.max(transformed[:, 2])

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
                dist = math.sqrt((px - el.center[0])**2 + (pz - el.center[2])**2)
                ratio = dist / 4.0
            else:
                ratio = 0.0

            color_hex = interpolate_colors(colors, ratio)
            scene_points.append({"x": px, "y": py, "z": pz, "color": color_hex})

    return scene_points

# ----------------- 圖片轉點陣 (方案 2 工具) -----------------
def zhang_suen_thinning(binary_image: np.ndarray) -> np.ndarray:
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
        mask1 = (img == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
        img[mask1] = 0

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
        mask2 = (img == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
        img[mask2] = 0

        if np.array_equal(img, prev):
            break
        prev = img.copy()
    return img

def image_to_drone_points(image_bytes: bytes, total_n: int, mode: str = "pixel"):
    from io import BytesIO
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    max_dim = 250
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    img_arr = np.array(img)
    alpha = img_arr[:, :, 3]
    gray = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
    is_bg = (alpha < 40) | (gray > 240)
    subject_coords = np.argwhere(~is_bg)
    if len(subject_coords) < 10:
        return []

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
        if len(coords) < 30: coords = subject_coords
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

# ----------------- PROMPTS 定義 -----------------
# Tier 1: AI Python 幾何程式設計師 PROMPT
PYTHON_DIRECTOR_PROMPT = """你是一位擁有頂尖 3D 運算圖學與數學建模能力的「無人機大秀總導演兼 Python 幾何架構師」。
面對學生的天馬行空自由創作劇本（如「富邦LOGO」、「皮卡丘放電」、「台北101煙火」、「星空鯨魚」、「太空梭」等）：
請為每一幕（除第一幕地面起飛外）編寫一段純 Python 幾何點陣生成函數，用數學曲線、三維幾何曲面或 PIL 向量繪圖，精準算出滿編 3000 架無人機的空間 3D 坐標與色彩！

【坐標與空域安全規範】：
- 橫向 X: [-6.0, 6.0] (公尺)
- 深度 Y: [2.2, 5.8] (中心在 4.0，可前後立體錯開 [2.5, 5.5])
- 高度 Z: [1.5, 7.0] (空中表演高度)

【Python 函數規範】：
- 函數名稱固定為 `def generate_scene(n):`
- 參數 n 為無人機總架數
- 返回格式：`(coords, colors)`
  - coords: np.ndarray，形狀 (n, 3)，坐標 [x, y, z]
  - colors: list[str]，長度為 n 的十六進位色碼列表 (如 '#0088FF', '#FFFFFF')

【創作程式庫支援】：
環境中已預先注入 `np` (NumPy), `math`, `random`, `Image`, `ImageDraw`, `ImageFont`, `FONT_PATH`。
1. 品牌商標/文字/圖騰 (如 '富邦LOGO')：
   可使用 PIL (Image.new, ImageDraw, ImageFont.truetype(FONT_PATH, 95)) 繪製文字與向量造型，取 np.argwhere(arr > 90) 映射至 3D，並在外圍拼接旋轉 3D 光環 (ring)！
2. 建築/機械/物體 (如 '台北101', '城堡', '太空梭')：
   可用柱體、圓錐、方形城垛、放射粒子等疊加組合！
3. 動物/角色/自然 (如 '大象', '鯨魚', '雄鷹', '浪花')：
   可用複合三維方程式（如大象：橢球軀幹 + 圓柱四肢 + 圓環大耳 + 捲曲螺旋鼻；鯨魚：正弦波羽翼 + 紡錘體），或使用 PIL 繪製動物剪影取點陣映射至 3D 空間前後錯開，立體鮮明！
4. 若第 1 幕劇本同時提及「地面起飛」與特定造型（如「第一幕：地面起飛，排列成大象」）：
   請拆為 2 幕輸出，第 1 幕為地面起飛待命（z=0），第 2 幕為空中立體造型！

回傳必須為 JSON 格式，Schema 如下：
{
  "scenes": [
    {
      "scene_name": "幕名",
      "code": "def generate_scene(n):\n    import numpy as np\n    # 計算 coords (n, 3) 與 colors (n)\n    return coords, colors"
    }
  ]
}
"""

# Tier 2: AI 複合幾何圖元庫 PROMPT (備援)
CREATIVE_DIRECTOR_PROMPT = """你是一位完全解鎖想像力的 3D 無人機大秀編導兼圖學專家。
閱讀使用者的演出劇本，將每一幕拆解為多個三維幾何圖元 (elements) 及其變形與色彩，絕不輸出具體點座標。

【起飛升空鐵則】：
無人機燈光秀必須以地面停機坪排列作為第 1 幕（shape_type 為 'ground_grid'，center=[0.0, 4.0, 0.0]，Z 高度固定為 0 貼地待命）。

【品牌 LOGO 與城堡造型規範】：
1. 若描述為企業、品牌或校園 LOGO (如 '富邦LOGO', 'SLHS 校徽', '台積電標誌')：
   拆解為【複合品牌徽章圖元】：
   - 圖元 1 (主體核心文字): text 類型，text 填寫品牌名稱（如 '富邦'），ratio 0.6，colors 指定配色。
   - 圖元 2 (外圍科技光環圓徽): ring 類型，ratio 0.4，rotate_deg [15, 0, 0]，形成立體光環圍繞企業文字！
2. 若描述為城堡、宮殿或建築：
   - 圖元 1 (城堡基底城垛): sphere 或 box (ratio=0.4, center=[0.0, 4.0, 2.8])
   - 圖元 2 (中央高聳尖塔): spiral 或 cone (ratio=0.3, center=[0.0, 4.0, 4.2], scale=[0.7, 0.7, 1.2])
   - 圖元 3 (頂端菱形光芒冠冕): star (ratio=0.3, center=[0.0, 4.0, 5.2])
3. 若描述為動物、生物或角色 (如 '大象', '鯨魚', '雄鷹')：
   請發揮立體雕塑思維，使用 2~4 個圖元組合出形象（絕不只用單一球體）！
   例如「大象」：
   - 圖元 1 (身體軀幹): sphere (ratio=0.5, center=[0.0, 4.0, 3.8], scale=[1.4, 1.0, 1.0], colors=['#94A3B8', '#CBD5E1'])
   - 圖元 2 (捲曲長象鼻): spiral (ratio=0.25, center=[-1.4, 4.0, 3.2], scale=[0.6, 0.6, 0.9], rotate_deg=[0, 45, 0], colors=['#94A3B8'])
   - 圖元 3 (扇形大耳朵/四肢): ring 或 cylinder (ratio=0.25, center=[0.8, 4.0, 4.2], scale=[0.8, 0.5, 0.8], colors=['#94A3B8'])
4. 若第 1 幕劇本同時提及「地面起飛」與特定造型（如「第一幕：地面起飛，排列成一隻大象」）：
   請務必拆為 2 幕輸出！第 1 幕純地面起飛 (ground_grid)，第 2 幕再展示空中立體造型！

可用圖元 (shape_type):
- ground_grid: 地面停機坪 (Z=0 待命)
- text: 中英文字/字母
- heart: 3D 心形
- ring: 圓環、光環、土星環、商標外框
- spiral: 螺旋、龍捲風、尖塔
- sphere: 立體球體、星體、建築底座
- star: 立體五角星、菱形光芒
- box: 3D 立方體、城垛、建築物
- cylinder: 3D 圓柱體、光柱、火箭主體
- cone: 3D 圓錐體、山峰、塔尖
- wave: 3D 波浪、飛鳥羽翼

回傳必須為 JSON 格式，Schema 為 DirectorScript。
"""

# ----------------- 執行沙盒 -----------------
def execute_generated_code(code_str: str, drone_count: int):
    scope = {
        "np": np,
        "numpy": np,
        "math": math,
        "random": random,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageFont": ImageFont,
        "FONT_PATH": FONT_PATH
    }
    exec(code_str, scope)
    fn = scope.get("generate_scene") or scope.get("generate_points") or scope.get("main")
    if not fn:
        for k, v in scope.items():
            if callable(v) and not k.startswith("__") and k not in ["np", "numpy", "math", "random", "Image", "ImageDraw", "ImageFont"]:
                fn = v
                break
    if not fn:
        raise ValueError("Code does not define generate_scene function")

    res = fn(drone_count)
    if isinstance(res, tuple) and len(res) >= 2:
        coords, colors = res[0], res[1]
    elif isinstance(res, np.ndarray):
        coords = res
        colors = ["#00F0FF"] * len(coords)
    elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        return res[:drone_count]
    else:
        raise ValueError("Invalid return type from generated function")

    coords = np.array(coords, dtype=float)
    if len(coords) < drone_count:
        rep = (drone_count // len(coords)) + 1
        coords = np.tile(coords, (rep, 1))[:drone_count]
    else:
        coords = coords[:drone_count]

    points = []
    for i in range(drone_count):
        px = round(float(coords[i, 0]), 2)
        py = round(float(coords[i, 1]), 2)
        pz = round(float(coords[i, 2]), 2)
        c = colors[i] if i < len(colors) else "#00F0FF"
        points.append({"x": px, "y": py, "z": pz, "color": c})
    return points

# ----------------- 本地離線智慧備援規則 (Tier 3) -----------------
def run_semantic_fallback(lines: List[str], drone_count: int):
    directives = []
    for idx, line in enumerate(lines):
        clean_l = re.sub(r"^第[一二三四五六七八九十\d]+幕[：:]\s*", "", line)
        colors = []
        if "藍" in clean_l or "青" in clean_l: colors.append("#0088FF")
        if "白" in clean_l: colors.append("#FFFFFF")
        if "紅" in clean_l or "赤" in clean_l: colors.append("#FF0055")
        if "金" in clean_l or "黃" in clean_l: colors.append("#FFD700")
        if "綠" in clean_l: colors.append("#00E676")
        if "紫" in clean_l: colors.append("#A020F0")
        if "粉" in clean_l: colors.append("#FF69B4")
        if not colors: colors = ["#00F0FF"]

        quote_match = re.search(r'[「『"\']([^」』"\']+)[」』"\']', clean_l)

        if "起飛" in clean_l or "停機" in clean_l or "地面" in clean_l:
            directives.append(SceneDirective(
                scene_name=line,
                elements=[PrimitiveElement(shape_type="ground_grid", ratio=1.0, colors=["#00E676", "#00F0FF"], gradient_direction="y_axis")]
            ))
        elif any(k in clean_l.lower() for k in ["logo", "商標", "標誌", "校徽", "圖騰", "標章"]):
            brand_name = "富邦"
            if quote_match:
                brand_name = quote_match.group(1)
            else:
                m_brand = re.search(r'([\u4e00-\u9fa5A-Za-z0-9]{2,6})\s*(?:的)?(?:LOGO|logo|商標|標誌|校徽)', clean_l)
                if m_brand:
                    brand_name = m_brand.group(1)
                elif "富邦" in clean_l:
                    brand_name = "富邦"
                elif "士商" in clean_l or "士林高商" in clean_l:
                    brand_name = "士商"
            
            logo_colors = colors if len(colors) >= 2 else (colors + ["#FFFFFF"])
            directives.append(SceneDirective(
                scene_name=line,
                elements=[
                    PrimitiveElement(shape_type="text", text=brand_name, ratio=0.6, colors=logo_colors, center=[0.0, 4.0, 3.8]),
                    PrimitiveElement(shape_type="ring", ratio=0.4, colors=logo_colors, center=[0.0, 4.0, 3.8], rotate_deg=[15, 0, 0])
                ]
            ))
        elif "城堡" in clean_l or "宮殿" in clean_l or "塔" in clean_l:
            castle_colors = colors if len(colors) >= 2 else ["#FF0055", "#FFFFFF"]
            directives.append(SceneDirective(
                scene_name=line,
                elements=[
                    PrimitiveElement(shape_type="sphere", ratio=0.4, center=[0.0, 4.0, 2.7], scale=[1.4, 1.0, 0.7], colors=castle_colors),
                    PrimitiveElement(shape_type="spiral", ratio=0.3, center=[0.0, 4.0, 4.1], scale=[0.7, 0.7, 1.2], colors=castle_colors, gradient_direction="z_axis_up"),
                    PrimitiveElement(shape_type="star", ratio=0.3, center=[0.0, 4.0, 5.2], colors=["#FFFFFF", castle_colors[0]])
                ]
            ))
        elif "心" in clean_l or "愛心" in clean_l:
            directives.append(SceneDirective(
                scene_name=line,
                elements=[PrimitiveElement(shape_type="heart", ratio=1.0, colors=colors or ["#FF1493", "#FF69B4"])]
            ))
        elif "環" in clean_l or "太陽" in clean_l or "土星" in clean_l or "日" in clean_l:
            directives.append(SceneDirective(
                scene_name=line,
                elements=[PrimitiveElement(shape_type="ring", ratio=1.0, colors=colors or ["#FFD700", "#FFA500"], rotate_deg=[25, 0, 0])]
            ))
        elif "星" in clean_l:
            directives.append(SceneDirective(
                scene_name=line,
                elements=[PrimitiveElement(shape_type="star", ratio=1.0, colors=colors or ["#FFD700", "#FFFFFF"])]
            ))
        elif "螺旋" in clean_l or "龍捲風" in clean_l or "蛇" in clean_l:
            directives.append(SceneDirective(
                scene_name=line,
                elements=[PrimitiveElement(shape_type="spiral", ratio=1.0, colors=colors or ["#00F0FF", "#8A2BE2"], gradient_direction="z_axis_up")]
            ))
        elif "球" in clean_l or "眼" in clean_l or "地球" in clean_l:
            directives.append(SceneDirective(
                scene_name=line,
                elements=[PrimitiveElement(shape_type="sphere", ratio=1.0, colors=colors or ["#38BDF8", "#1E40AF"], gradient_direction="radial")]
            ))
        elif "波" in clean_l or "海" in clean_l or "浪" in clean_l:
            directives.append(SceneDirective(
                scene_name=line,
                elements=[PrimitiveElement(shape_type="wave", ratio=1.0, colors=colors or ["#0077B6", "#00B4D8"])]
            ))
        else:
            if quote_match:
                pure_text = quote_match.group(1)
            else:
                cleaned = re.sub(r'^(天空用|在空中|用|以)?[\u4e00-\u9fa5\w色\s與和跟]+(排成|組成|顯示|呈現|排出|展示|變成)', '', clean_l)
                cleaned = re.sub(r'(大字|字樣|圖形|形狀|圖案|樣式|造型).*$', '', cleaned)
                pure_text = cleaned.strip()[:6] if cleaned.strip() else clean_l[:4]
            directives.append(SceneDirective(
                scene_name=line,
                elements=[PrimitiveElement(shape_type="text", text=pure_text, ratio=1.0, colors=colors)]
            ))

    # 確保第一幕為地面起飛
    ensure_ground_takeoff_directives(directives)
    rendered = []
    for d in directives:
        pts = render_directive_to_points(d, drone_count)
        rendered.append({"name": d.scene_name, "points": pts})
    return rendered

def ensure_ground_takeoff_directives(directives):
    def is_ground_scene(d):
        name = d.get("scene_name", "") if isinstance(d, dict) else getattr(d, "scene_name", "")
        elems = d.get("elements", []) if isinstance(d, dict) else getattr(d, "elements", [])
        if any(k in name for k in ["起飛", "地面", "停機", "跑道", "機坪", "待命", "升空"]):
            return True
        for el in elems:
            st = el.get("shape_type") if isinstance(el, dict) else getattr(el, "shape_type", "")
            if st == "ground_grid":
                return True
        return False

    if len(directives) == 0 or not is_ground_scene(directives[0]):
        ground_scene = SceneDirective(
            scene_name="第一幕：地面停機坪排列（起飛準備）",
            elements=[PrimitiveElement(
                shape_type="ground_grid",
                ratio=1.0,
                colors=["#00E676", "#00F0FF"],
                gradient_direction="y_axis"
            )]
        )
        directives.insert(0, ground_scene)
    else:
        first_d = directives[0]
        first_elems = first_d.get("elements", []) if isinstance(first_d, dict) else getattr(first_d, "elements", [])
        has_grid = any((el.get("shape_type") if isinstance(el, dict) else getattr(el, "shape_type", "")) == "ground_grid" for el in first_elems)
        if not has_grid:
            grid_el = PrimitiveElement(shape_type="ground_grid", ratio=1.0, colors=["#00E676", "#00F0FF"], gradient_direction="y_axis")
            if isinstance(first_d, dict):
                first_d["elements"] = [grid_el.model_dump()]
            else:
                first_d.elements = [grid_el]

def ensure_ground_takeoff_scenes(rendered_scenes, drone_count):
    if not rendered_scenes or not any(k in rendered_scenes[0]["name"] for k in ["地面", "起飛", "停機", "跑道"]):
        pts = generate_primitive_coords("ground_grid", drone_count)
        ground_pts = [{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2), "z": 0.0, "color": "#00E676" if i < drone_count//2 else "#00F0FF"} for i, p in enumerate(pts)]
        rendered_scenes.insert(0, {"name": "第一幕：地面停機坪排列（起飛準備）", "points": ground_pts})

# ----------------- ENDPOINTS -----------------
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
    async with request_lock:
        t_start = time.time()
        print(f"[{req.student_name}] 正在呼叫 AI 3D 總導演思考中...")
        user_content = f"演出架數: {req.drone_count}\n劇本需求:\n{req.prompt}"

        candidate_models = [
            "models/gemini-3.5-flash",
            "models/gemini-3.5-flash-lite",
            "models/gemini-3.1-flash-lite",
            "models/gemini-3.6-flash",
            "models/gemini-omni-flash-preview"
        ]
        rendered_scenes = []
        ai_mode = "none"
        ai_error_msg = None

        tier1_info = {"status": "skipped", "model": None, "error": None}
        tier2_info = {"status": "skipped", "model": None, "error": None}
        tier3_info = {"status": "skipped", "rule": None}
        scenes_breakdown = []

        # =================================================================
        # 第一首選主力 (Tier 1): 方案 3 - AI Python 幾何程式設計師 (動態編程)
        # =================================================================
        if GEMINI_API_KEY:
            try:
                http_opts = types.HttpOptions(timeout=20000)
                client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts)
                code_resp = None
                for m in candidate_models:
                    try:
                        print(f"[{req.student_name}] Tier 1 嘗試 Python Code 引擎模型: {m}")
                        code_resp = client.models.generate_content(
                            model=m,
                            contents=user_content,
                            config=types.GenerateContentConfig(
                                system_instruction=PYTHON_DIRECTOR_PROMPT,
                                response_mime_type="application/json",
                                temperature=0.2
                            )
                        )
                        if code_resp and code_resp.text:
                            tier1_info["model"] = m
                            break
                    except Exception as ex_m:
                        print(f"[{req.student_name}] 模型 {m} 代碼生成異常: {ex_m}")
                        continue

                if code_resp and code_resp.text:
                    parsed_json = json.loads(code_resp.text.strip())
                    code_scenes = parsed_json.get("scenes", [])
                    if code_scenes:
                        for s in code_scenes:
                            s_name = s.get("scene_name", "AI 幾何劇幕")
                            code_str = s.get("code", "")
                            if code_str:
                                pts = execute_generated_code(code_str, req.drone_count)
                                rendered_scenes.append({"name": s_name, "points": pts})
                                scenes_breakdown.append({
                                    "scene_name": s_name,
                                    "drone_count": len(pts),
                                    "generator": "Tier 1: AI 幾何程式設計師 (Python 數學曲面與向量運算)",
                                    "code_preview": (code_str.strip()[:240] + "...") if len(code_str) > 240 else code_str.strip(),
                                    "explanation": "由 Gemini AI 即時動態編寫 Python 幾何函數生成 3D 空間點陣與漸層色彩"
                                })

                        if len(rendered_scenes) > 0:
                            ai_mode = "python_code"
                            tier1_info["status"] = "success"
                            print(f"[{req.student_name}] 🎉 Tier 1 (AI 幾何程式設計師) 成功動態運算 {len(rendered_scenes)} 幕！")
            except Exception as e_code:
                ai_error_msg = f"Tier 1 (Code Engine) 異常: {str(e_code)[:150]}"
                tier1_info["status"] = "error"
                tier1_info["error"] = str(e_code)[:200]
                scenes_breakdown.clear()
                rendered_scenes.clear()
                print(f"[{req.student_name}] {ai_error_msg}，切換至 Tier 2 複合幾何圖元庫備援...")

        # =================================================================
        # 第二層防護 (Tier 2): 方案 1 - AI 複合幾何圖元庫備援 (積木拼接)
        # =================================================================
        if ai_mode == "none" and GEMINI_API_KEY:
            try:
                http_opts = types.HttpOptions(timeout=20000)
                client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts)
                prim_resp = None
                for m in candidate_models:
                    try:
                        print(f"[{req.student_name}] Tier 2 嘗試複合圖元模型: {m}")
                        prim_resp = client.models.generate_content(
                            model=m,
                            contents=user_content,
                            config=types.GenerateContentConfig(
                                system_instruction=CREATIVE_DIRECTOR_PROMPT,
                                response_mime_type="application/json",
                                response_schema=DirectorScript,
                                temperature=0.2
                            )
                        )
                        if prim_resp and prim_resp.text:
                            tier2_info["model"] = m
                            break
                    except Exception as ex_m:
                        print(f"[{req.student_name}] 模型 {m} 圖元解析異常: {ex_m}")
                        continue

                if prim_resp and prim_resp.text:
                    director_data = json.loads(prim_resp.text.strip())
                    directives = director_data.get("scenes", [])
                    if directives:
                        for d in directives:
                            d_obj = SceneDirective(**d) if isinstance(d, dict) else d
                            pts = render_directive_to_points(d_obj, req.drone_count)
                            rendered_scenes.append({"name": d_obj.scene_name, "points": pts})

                            # 解析各圖元詳情供前端除錯面板展示
                            elem_descs = []
                            for el in d_obj.elements:
                                s_type = el.shape_type
                                zh_name = SHAPE_NAMES.get(s_type, s_type)
                                text_info = f" (文字: '{el.text}')" if el.text else ""
                                elem_descs.append({
                                    "shape_type": s_type,
                                    "desc": f"{zh_name}{text_info}",
                                    "ratio": el.ratio,
                                    "colors": el.colors or ["#00F0FF"]
                                })

                            elem_summary = " + ".join([f"{ed['desc']}({int(round(ed['ratio']*100))}%)" for ed in elem_descs])
                            scenes_breakdown.append({
                                "scene_name": d_obj.scene_name,
                                "drone_count": len(pts),
                                "generator": "Tier 2: AI 複合幾何圖元庫 (3D 積木組合拼裝)",
                                "elements": elem_descs,
                                "explanation": f"AI 拆解構圖：{elem_summary}"
                            })

                        if len(rendered_scenes) > 0:
                            ai_mode = "primitive_composer"
                            tier2_info["status"] = "success"
                            print(f"[{req.student_name}] 🧱 Tier 2 (AI 複合幾何圖元庫) 成功搭建 {len(rendered_scenes)} 幕！")
            except Exception as e_prim:
                ai_error_msg = (ai_error_msg or "") + f" | Tier 2 異常: {str(e_prim)[:150]}"
                tier2_info["status"] = "error"
                tier2_info["error"] = str(e_prim)[:200]
                scenes_breakdown.clear()
                rendered_scenes.clear()
                print(f"[{req.student_name}] 切換至 Tier 3 本地智慧語意備援...")

        # =================================================================
        # 第三層保底 (Tier 3): 本地離線智慧語意備援 (保證 100% 成功)
        # =================================================================
        if ai_mode == "none":
            ai_mode = "semantic_fallback"
            tier3_info["status"] = "active"
            tier3_info["rule"] = "本地智慧語意引擎 (商標徽章/立體城堡/漢字居中)"
            lines = [l.strip() for l in req.prompt.split("\n") if l.strip()]
            rendered_scenes = run_semantic_fallback(lines, req.drone_count)
            for s in rendered_scenes:
                scenes_breakdown.append({
                    "scene_name": s["name"],
                    "drone_count": len(s["points"]),
                    "generator": "Tier 3: 本地離線智慧語意引擎 (保底無碰撞航線)",
                    "explanation": "依據劇本關鍵字自動匹配專屬 3D 複合圖元與多字排版"
                })
            print(f"[{req.student_name}] ⚡ Tier 3 (本地智慧語意備援引擎) 順利生成 {len(rendered_scenes)} 幕！")

        # ---------------- 鐵則：確保開場永遠為地面停機坪排列 (Z=0) ----------------
        ensure_ground_takeoff_scenes(rendered_scenes, req.drone_count)
        if len(scenes_breakdown) < len(rendered_scenes):
            scenes_breakdown.insert(0, {
                "scene_name": rendered_scenes[0]["name"],
                "drone_count": len(rendered_scenes[0]["points"]),
                "generator": "安全鐵則起飛引擎 (Ground Launchpad)",
                "explanation": "系統防護鐵則：所有群飛開場強制於地面 Z=0.0 機坪點亮並準備升空"
            })

        elapsed_sec = round(time.time() - t_start, 2)
        debug_info = {
            "model_used": tier1_info.get("model") or tier2_info.get("model") or "本地離線引擎",
            "elapsed_sec": f"{elapsed_sec} 秒",
            "active_tier": "Tier 1: AI 幾何程式設計師 (Python Code)" if ai_mode == "python_code" else ("Tier 2: AI 複合幾何圖元庫 (Primitive Composer)" if ai_mode == "primitive_composer" else "Tier 3: 本地離線智慧語意備援"),
            "tier1": tier1_info,
            "tier2": tier2_info,
            "tier3": tier3_info,
            "scenes_breakdown": scenes_breakdown
        }

        return {
            "status": "success",
            "ai_mode": ai_mode,
            "ai_error": ai_error_msg,
            "debug_info": debug_info,
            "data": json.dumps({"scenes": rendered_scenes})
        }

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    print(f'>> 無人機伺服器啟動: http://localhost:{port}')
    uvicorn.run(app, host='0.0.0.0', port=port)
