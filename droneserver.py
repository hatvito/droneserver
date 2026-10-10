import os
import asyncio
import json
import traceback
import re
import io
import base64
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types

app = FastAPI(title="Drone Show AI Proxy Server")

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
        "message": "無人機群飛 AI 伺服器運作中！"
    }

class ShowRequest(BaseModel):
    student_name: str
    drone_count: int
    prompt: str

class ImageSceneRequest(BaseModel):
    image_base64: str
    drone_count: int
    mode: Optional[str] = "pixel"
    scene_name: Optional[str] = "自訂圖片造型"

class Point3D(BaseModel):
    x: float
    y: float
    z: float
    color: str

class Scene(BaseModel):
    name: str
    points: List[Point3D]

class DroneShowOutput(BaseModel):
    scenes: List[Scene]

request_lock = asyncio.Lock()

def generate_ground_takeoff_points(total_n: int):
    cols = math.ceil(math.sqrt(total_n * 1.5))
    rows = math.ceil(total_n / cols)
    spacing_x = 10.0 / max(cols - 1, 1)
    spacing_y = 2.5 / max(rows - 1, 1)
    start_x = -5.0
    start_y = 2.8

    pts = []
    for i in range(total_n):
        r = i // cols
        c = i % cols
        px = round(start_x + c * spacing_x, 2)
        py = round(start_y + r * spacing_y, 2)
        pts.append({"x": px, "y": py, "z": 0.0, "color": "#FFFFFF"})
    return pts

def render_text_to_points(text: str, total_n: int, color: str = "#00F0FF"):
    if not text.strip():
        return None

    canvas_w = 160
    canvas_h = 80
    img = Image.new("L", (canvas_w, canvas_h), color=0)
    draw = ImageDraw.Draw(img)

    text_len = len(text)
    font_size = 56 if text_len <= 1 else (40 if text_len <= 4 else 26)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw_x = max(0, (canvas_w - text_w) // 2)
    draw_y = max(0, (canvas_h - text_h) // 2)
    draw.text((draw_x, draw_y), text, font=font, fill=255)

    img_arr = np.array(img)
    stroke_coords = np.argwhere(img_arr > 100)

    if len(stroke_coords) == 0:
        return None

    if len(stroke_coords) >= total_n:
        indices = np.linspace(0, len(stroke_coords) - 1, total_n, dtype=int)
        sampled = stroke_coords[indices]
    else:
        repeat_factor = (total_n // len(stroke_coords)) + 1
        extended = np.tile(stroke_coords, (repeat_factor, 1))
        noise = np.random.uniform(-0.4, 0.4, size=extended.shape)
        sampled = (extended + noise)[:total_n]

    pts = []
    rainbow = ["#FF3366", "#FF9900", "#FFD700", "#33CC33", "#00F0FF", "#9933FF"]
    for idx, (r, c) in enumerate(sampled):
        px = round(-5.0 + (c / canvas_w) * 10.0, 2)
        py = 4.0
        pz = round(1.8 + ((canvas_h - r) / canvas_h) * 4.2, 2)
        if "HAPPY" in text.upper():
            c_idx = int((c / canvas_w) * len(rainbow))
            pt_color = rainbow[min(c_idx, len(rainbow) - 1)]
        else:
            pt_color = color
        pts.append({"x": px, "y": py, "z": pz, "color": pt_color})
    return pts

def generate_tornado_points(total_n: int):
    pts = []
    for i in range(total_n):
        t = i / total_n
        z = 1.8 + t * 4.4
        radius = 0.5 + t * 3.5
        theta = t * 6 * math.pi
        px = round(radius * math.cos(theta), 2)
        py = round(4.0 + radius * math.sin(theta) * 0.4, 2)
        pts.append({"x": px, "y": py, "z": round(z, 2), "color": "#00FFFF" if i % 2 == 0 else "#FFD700"})
    return pts

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

# ----------------- 智慧實體點陣提取引擎 (100% 聚焦主體) -----------------
def image_to_drone_points(image_bytes: bytes, total_n: int, mode: str = "pixel"):
    pil_img = Image.open(io.BytesIO(image_bytes))
    rgba_img = pil_img.convert("RGBA")
    
    # 稍微放大工作畫布保持五官細節 (例如 80x80)
    grid_dim = 90
    w, h = rgba_img.size
    scale = grid_dim / max(w, h)
    new_w, new_h = max(10, int(w * scale)), max(10, int(h * scale))
    resized = rgba_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_arr = np.array(resized)

    # 1. 智慧背景檢測（透明度 < 80 或是 純白/近白底 R>235, G>235, B>235）
    r_chan = img_arr[:, :, 0]
    g_chan = img_arr[:, :, 1]
    b_chan = img_arr[:, :, 2]
    alpha = img_arr[:, :, 3]

    is_bg = (alpha < 80) | ((r_chan > 232) & (g_chan > 232) & (b_chan > 232))
    
    # 抓出「非背景」的實體像素點（柴犬身體、五官、項圈）
    subject_coords = np.argwhere(~is_bg)

    # 若圖片整張都很暗或沒有白底，退回全圖
    if len(subject_coords) < 50:
        subject_coords = np.argwhere(alpha > 0)

    # ======= 模式 1：實心彩色主體像素 (700 架全部分配給柴犬) =======
    if mode == "pixel":
        # 均勻抽樣剛好 total_n 架
        if len(subject_coords) >= total_n:
            indices = np.linspace(0, len(subject_coords) - 1, total_n, dtype=int)
            sampled = subject_coords[indices]
        else:
            rep = (total_n // len(subject_coords)) + 1
            extended = np.tile(subject_coords, (rep, 1))
            noise = np.random.uniform(-0.35, 0.35, size=extended.shape)
            sampled = (extended + noise)[:total_n]

        # 計算柴犬主體邊界，居中放大到全舞台
        min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
        min_c, max_c = np.min(sampled[:, 1]), np.max(sampled[:, 1])
        range_r = max(max_r - min_r, 1)
        range_c = max(max_c - min_c, 1)

        pts = []
        for r, c in sampled:
            norm_x = (c - min_c) / range_c
            norm_z = (max_r - r) / range_r

            px = round(-4.5 + norm_x * 9.0, 2)
            py = 4.0
            pz = round(1.8 + norm_z * 4.5, 2)

            # 抓取原圖真實顏色
            r_idx = min(max(int(r), 0), new_h - 1)
            c_idx = min(max(int(c), 0), new_w - 1)
            r_val, g_val, b_val, _ = img_arr[r_idx, c_idx]
            
            # 若為黑色眼睛或輪廓線，使用深黑/海軍藍維持夜空對比度
            if r_val < 30 and g_val < 30 and b_val < 30:
                hex_color = "#1E293B" # 科技深邃藍黑
            else:
                hex_color = f"#{r_val:02X}{g_val:02X}{b_val:02X}"

            pts.append({"x": px, "y": py, "z": pz, "color": hex_color})
        return pts

    # ======= 模式 2：骨架線條模式 =======
    else:
        binary = (~is_bg).astype(np.uint8)
        skeleton = zhang_suen_thinning(binary)
        coords = np.argwhere(skeleton == 1)
        if len(coords) < 30:
            coords = subject_coords

        if len(coords) >= total_n:
            indices = np.linspace(0, len(coords) - 1, total_n, dtype=int)
            sampled = coords[indices]
        else:
            rep = (total_n // len(coords)) + 1
            sampled = np.tile(coords, (rep, 1))[:total_n]

        min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
