import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import os
import asyncio
import json
import traceback
import re
import io
import base64
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types

app = FastAPI(title="Drone Show AI Proxy Server - Infinite Creative Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
FONT_PATH = "NotoSansTC-VariableFont_wght.ttf"

@app.get("/")
@app.head("/")
def read_root():
    return {
        "status": "online",
        "has_api_key": bool(GEMINI_API_KEY),
        "font_exists": os.path.exists(FONT_PATH),
        "message": "無人機群飛 3D 奔放幾何導演引擎運作中！"
    }

@app.get("/api/test-ai")
def test_ai():
    if not GEMINI_API_KEY:
        return {"status": "error", "message": "GEMINI_API_KEY is not set"}
    
    results = {}
    client = genai.Client(api_key=GEMINI_API_KEY)
    test_models = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    
    for m in test_models:
        try:
            resp = client.models.generate_content(
                model=m,
                contents="ping"
            )
            results[m] = {"status": "success", "text": resp.text.strip()}
            return {"status": "ok", "working_model": m, "results": results}
        except Exception as e:
            results[m] = {"status": "error", "msg": str(e)[:150]}
            
    return {"status": "all_failed", "results": results}

# ----------------- 資料模型定義 -----------------
class PrimitiveElement(BaseModel):
    shape_type: str = Field(description="圖元類型: text, heart, ring, spiral, sphere, star, cone, wave, ground_grid")
    ratio: float = Field(default=1.0, description="佔用無人機總數的比例權重 (例如 0.6 代表佔 60% 架數)")
    text: Optional[str] = Field(default="", description="若為 text 類型時的文字內容，如 '富邦' 或 'SLHS'")
    center: Optional[List[float]] = Field(default=[0.0, 4.0, 3.8], description="三維中心座標 [x, y, z]")
    scale: Optional[List[float]] = Field(default=[1.0, 1.0, 1.0], description="三維尺寸縮放係數 [sx, sy, sz]")
    rotate_deg: Optional[List[float]] = Field(default=[0.0, 0.0, 0.0], description="三維旋轉角度 [rx, ry, rz]，例如斜躺或傾斜")
    colors: Optional[List[str]] = Field(default=["#00F0FF"], description="色標陣列 (支援多色漸層)，例如 ['#FFFFFF', '#A020F0']")
    gradient_direction: Optional[str] = Field(default="none", description="漸層方向: none, z_axis_up, z_axis_down, x_axis, y_axis, radial")

class SceneDirective(BaseModel):
    scene_name: str = Field(description="幕名，例如 '第一幕：地面起飛' 或 '第三幕：富邦立體漸層光環'")
    elements: List[PrimitiveElement] = Field(description="構成該幕的一組或多組幾何圖元")

class DirectorScript(BaseModel):
    scenes: List[SceneDirective]

class ShowRequest(BaseModel):
    student_name: str
    drone_count: int
    prompt: str

class ImageSceneRequest(BaseModel):
    image_base64: str
    drone_count: int
    mode: Optional[str] = "pixel"
    scene_name: Optional[str] = "自訂圖片造型"

request_lock = asyncio.Lock()

# ----------------- 3D 數學與多重多色漸層工具 -----------------
def hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        return (0, 240, 255)
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

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

def apply_transforms(coords: np.ndarray, center: List[float], scale: List[float], rot_deg: List[float]):
    """套用 3D 旋轉矩陣、縮放與空間平移"""
    scaled = coords * np.array(scale)

    rx, ry, rz = np.radians(rot_deg)
    # 旋轉矩陣
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    rotated = scaled @ R.T
    transformed = rotated + np.array(center)
    return transformed

# ----------------- 幾何圖元生成器 -----------------
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
        # 3D 心形曲線 (立體加厚)
        t = np.linspace(0, 2 * math.pi, n)
        x = 16 * np.sin(t)**3 / 16.0 * 3.5
        z = (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)) / 16.0 * 3.5
        y = np.random.uniform(-0.35, 0.35, n)
        return np.column_stack([x, y, z])

    elif shape == "ring":
        # 環狀 / 行星環
        theta = np.linspace(0, 2 * math.pi, n)
        r = np.random.uniform(3.0, 3.4, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(n)
        return np.column_stack([x, y, z])

    elif shape == "spiral":
        # 螺旋 / 龍捲風
        t = np.linspace(0, 1, n)
        z = (t - 0.5) * 4.0
        r = 0.6 + t * 2.8
        theta = t * 6 * math.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y, z])

    elif shape == "sphere":
        # 費氏球體 (Fibonacci Sphere) 均勻分佈
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = math.pi * (1 + 5**0.5) * indices
        r = 3.0
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        return np.column_stack([x, y, z])

    elif shape == "star":
        # 五角星
        angles = np.linspace(0, 4 * math.pi, n)
        r = 3.2 * (0.5 + 0.5 * (np.sin(5 * angles / 2)**2))
        x = r * np.cos(angles)
        z = r * np.sin(angles)
        y = np.zeros(n)
        return np.column_stack([x, y, z])

    elif shape == "text" or text_content:
        # 文字骨架點陣 (支援多字排版、高解析防裁切、等比例縮放與筆劃加粗)
        chars = list(text_content) if text_content else ["?"]
        c_count = len(chars)
        pts_per_c = n // c_count
        char_coords_list = []
        
        # 動態計算文字排版橫向總寬度與每字分配槽位
        span_x = min(8.6, max(2.5, c_count * 2.1))
        slot_w = span_x / c_count
        max_h = min(2.8, slot_w * 1.25)

        for c_idx, ch in enumerate(chars):
            dim = 200  # 擴大畫布尺寸至 200x200，徹底防止邊緣筆劃被切斷
            img = Image.new("L", (dim, dim), color=0)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(FONT_PATH, 110)
            except Exception:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw = max(bbox[2] - bbox[0], 1)
            th = max(bbox[3] - bbox[1], 1)
            # 精確補償字型 Bounding Box 偏移量，置中繪製且加粗筆畫以利無人機密集排列
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

            # 嚴格保持文字 1:1 長寬比，避免字體被縱向或橫向過度拉伸
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

# ----------------- 圖元渲染整合核心 -----------------
def render_directive_to_points(directive: SceneDirective, total_drones: int):
    elements = directive.elements
    if not elements:
        elements = [PrimitiveElement(shape_type="text", text=directive.scene_name, ratio=1.0)]

    # 計算各圖元分配架數
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
            # 地面停機坪：所有無人機嚴格貼緊地面 Z=0，依矩陣整齊排列於起飛跑道
            transformed = base_coords.copy()
        else:
            transformed = apply_transforms(base_coords, el.center, el.scale, el.rotate_deg)

        # 計算色彩漸層
        colors = el.colors or (["#00E676", "#00F0FF"] if el.shape_type == "ground_grid" else ["#00F0FF"])
        grad_dir = el.gradient_direction or ("y_axis" if el.shape_type == "ground_grid" else "none")
        
        # 取得維度極值做比例計算
        min_x, max_x = np.min(transformed[:, 0]), np.max(transformed[:, 0])
        min_y, max_y = np.min(transformed[:, 1]), np.max(transformed[:, 1])
        min_z, max_z = np.min(transformed[:, 2]), np.max(transformed[:, 2])

        for pt in transformed:
            px, py, pz = round(float(pt[0]), 2), round(float(pt[1]), 2), round(float(pt[2]), 2)
            
            # 計算該點在漸層向量中的比例 (0.0 ~ 1.0)
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

# ----------------- 圖片轉點陣 -----------------
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
    grid_dim = 90
    w, h = rgba_img.size
    scale = grid_dim / max(w, h)
    new_w, new_h = max(10, int(w * scale)), max(10, int(h * scale))
    resized = rgba_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_arr = np.array(resized)

    r_chan, g_chan, b_chan, alpha = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2], img_arr[:, :, 3]
    is_bg = (alpha < 80) | ((r_chan > 232) & (g_chan > 232) & (b_chan > 232))
    subject_coords = np.argwhere(~is_bg)
    if len(subject_coords) < 50:
        subject_coords = np.argwhere(alpha > 0)

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

# ----------------- 自由創作總導演 SYSTEM PROMPT -----------------
CREATIVE_DIRECTOR_PROMPT = """你是一位完全解鎖想像力的 3D 無人機大秀編導兼圖學專家。
閱讀使用者的演出劇本，將每一幕拆解為多個三維幾何圖元 (elements) 及其變形與色彩，絕不輸出具體點座標。

【起飛升空鐵則】：
無人機燈光秀必須以地面停機坪排列作為第 1 幕（shape_type 為 'ground_grid'，center=[0.0, 4.0, 0.0]，Z 高度固定為 0 貼地待命）。
若使用者劇本第一幕已是地面起飛/停機坪，將其圖元指定為 ground_grid；若劇本未寫起飛，必須在最前面新增第 1 幕地面停機坪排列 (ground_grid)，再接續空中的精彩造型！

可用圖元 (shape_type):
- ground_grid: 地面停機坪 (僅用於起飛/第一幕)
- text: 中英文字/字母 (填寫 text 欄位，如 '富邦'、'SLHS')
- heart: 3D 心形
- ring: 圓環、光環、土星環
- spiral: 螺旋、龍捲風
- sphere: 立體球體、星體
- star: 立體五角星

自由變形與色彩屬性:
1. ratio: 該圖元佔用無人機總數的比重 (例如主要文字占 0.6，周圍環繞光環占 0.4)
2. center: 三維中心點 [x, y, z]。X 範圍 [-5.0, 5.0], Y 深度 [2.0, 6.0] (2.4最前, 4.0中間, 5.2後排), Z 高度 [1.5, 6.5]
3. scale: [sx, sy, sz] 尺寸縮放
4. rotate_deg: [rx, ry, rz] 任意旋轉傾斜角 (例如傾斜 30 度: [30, 0, 0])
5. colors: 任意色彩陣列 (支援多色漸層，如 ['#FFFFFF', '#A020F0'] 或虹彩)
6. gradient_direction: 漸層方向 (z_axis_up, z_axis_down, x_axis, y_axis, radial, none)

範例情境處理能力:
- 若描述：「富邦，一前一後，富為藍色，邦為紫色由上往下由淺入深」：
  拆為兩個 text 圖元，富在 center=[ -1.8, 2.5, 3.8 ], colors=['#008CFF']；邦在 center=[ 1.8, 5.2, 3.8 ], colors=['#FFFFFF', '#A020F0'], gradient_direction='z_axis_down'。
- 若描述：「斜躺的粉紅心形，外圍被金色光環圍繞」：
  圖元1 heart (ratio 0.6, colors=['#FF69B4'], rotate_deg=[35, 0, 0])；圖元2 ring (ratio 0.4, colors=['#FFD700'], rotate_deg=[35, 0, 0])。
"""

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
        print(f"[{req.student_name}] 正在呼叫奔放導演思考中...")
        user_content = f"演出架數: {req.drone_count}\n劇本需求:\n{req.prompt}"

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            def call_creative_director():
                candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.0-flash", "gemini-2.5-flash"]
                all_errors = []
                for m in candidate_models:
                    try:
                        print(f"[{req.student_name}] 嘗試使用 AI 模型: {m}")
                        return client.models.generate_content(
                            model=m,
                            contents=user_content,
                            config=types.GenerateContentConfig(
                                system_instruction=CREATIVE_DIRECTOR_PROMPT,
                                response_mime_type="application/json",
                                response_schema=DirectorScript,
                                temperature=0.2
                            )
                        )
                    except Exception as ex:
                        err_str = f"{m} -> {ex}"
                        print(f"[{req.student_name}] 模型異常: {err_str}")
                        all_errors.append(err_str)
                        continue
                raise Exception(" \n ".join(all_errors))

            ai_resp = await asyncio.wait_for(asyncio.to_thread(call_creative_director), timeout=35.0)
            director_data = json.loads(ai_resp.text.strip())
            directives = director_data.get("scenes", [])
            print(f"[{req.student_name}] AI 導演成功建構 {len(directives)} 組複合幾何劇幕！")

        except Exception as e:
            ai_error_msg = str(e)
            print(f"AI 導演解析異常: {ai_error_msg}，啟動智慧語義幾何備援機制...")
            lines = [l.strip() for l in req.prompt.split("\n") if l.strip()]
            directives = []
            for idx, line in enumerate(lines):
                clean_l = re.sub(r"^第[一二三四五六七八九十\d]+幕[：:]\s*", "", line)
                
                # 色彩萃取
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
                    # 智慧 LOGO 徽章複合圖元 (文字核心 + 外圍光環)
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
                    # 宏偉 3D 城堡複合幾何 (城垛底座 + 中央尖塔 + 頂端光芒)
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
                    # 純文字排版：過濾提示詞雜訊，只排核心名詞
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

        # ---------------- 鐵則：確保開場永遠為地面停機坪起飛排列 ----------------
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

        # 幾何工匠：0.05 秒秒算滿編無人機空間點位
        rendered_scenes = []
        for d in directives:
            directive_obj = SceneDirective(**d) if isinstance(d, dict) else d
            pts = render_directive_to_points(directive_obj, req.drone_count)
            rendered_scenes.append({"name": directive_obj.scene_name, "points": pts})

        print(f"[{req.student_name}] 全場幾何渲染完成，共 {len(rendered_scenes)} 幕！")
        ai_mode = "gemini" if "ai_resp" in locals() and ai_resp else "fallback"
        return {"status": "success", "ai_mode": ai_mode, "ai_error": locals().get("ai_error_msg"), "data": json.dumps({"scenes": rendered_scenes})}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    print(f'>> 無人機伺服器啟動: http://localhost:{port}')
    uvicorn.run(app, host='0.0.0.0', port=port)
