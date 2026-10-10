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
    mode: Optional[str] = "pixel"  # "skeleton" (線條骨架) 或 "pixel" (實心像素)
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

# ----------------- 地面停機坪網格 -----------------
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

# ----------------- 字型點陣轉換 -----------------
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

# ----------------- 骨架細化演算法 (Zhang-Suen) -----------------
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

# ----------------- 圖片轉點陣雙核心引擎 -----------------
def image_to_drone_points(image_bytes: bytes, total_n: int, mode: str = "pixel"):
    pil_img = Image.open(io.BytesIO(image_bytes))
    rgba_img = pil_img.convert("RGBA")
    w, h = rgba_img.size
    aspect_ratio = w / h

    # ======= 模式 1：實心像素點陣螢幕模式 (Pixel Display) =======
    if mode == "pixel":
        # 計算最符合原圖比例的網格寬高，讓 cols * rows 近似於 total_n
        cols = max(2, int(math.sqrt(total_n * aspect_ratio)))
        rows = max(2, int(total_n / cols))
        
        # 縮放原圖至該低解析度像素矩陣
        pixel_img = rgba_img.resize((cols, rows), Image.Resampling.LANCZOS)
        img_arr = np.array(pixel_img)
        
        # 判斷是否為去背透明圖 (Alpha channel)
        alpha = img_arr[:, :, 3]
        has_transparency = np.any(alpha < 100)
        
        pts = []
        if has_transparency:
            # 優先取不透明區域
            valid_coords = np.argwhere(alpha > 80)
            if len(valid_coords) < 10:
                valid_coords = np.argwhere(alpha >= 0)
            
            # 均勻抽樣至 total_n
            if len(valid_coords) >= total_n:
                indices = np.linspace(0, len(valid_coords) - 1, total_n, dtype=int)
                sampled = valid_coords[indices]
            else:
                rep = (total_n // len(valid_coords)) + 1
                sampled = np.tile(valid_coords, (rep, 1))[:total_n]
                
            for r, c in sampled:
                px = round(-5.0 + (c / cols) * 10.0, 2)
                py = 4.0
                pz = round(1.8 + ((rows - r) / rows) * 4.4, 2)
                r_c, g_c, b_c, _ = img_arr[r, c]
                hex_color = f"#{r_c:02X}{g_c:02X}{b_c:02X}"
                pts.append({"x": px, "y": py, "z": pz, "color": hex_color})
        else:
            # 一般彩色實心照片：整面矩形鋪滿
            count = 0
            for r in range(rows):
                for c in range(cols):
                    if count >= total_n:
                        break
                    px = round(-5.0 + (c / cols) * 10.0, 2)
                    py = 4.0
                    pz = round(1.8 + ((rows - r) / rows) * 4.4, 2)
                    r_c, g_c, b_c, _ = img_arr[r, c]
                    hex_color = f"#{r_c:02X}{g_c:02X}{b_c:02X}"
                    pts.append({"x": px, "y": py, "z": pz, "color": hex_color})
                    count += 1
            
            # 若尚有剩餘架數補齊最後一行
            while len(pts) < total_n:
                pts.append(pts[-1])
        return pts

    # ======= 模式 2：骨架線條模式 (Skeleton Outline) =======
    else:
        target_dim = 160
        scale = target_dim / max(w, h)
        new_w, new_h = max(20, int(w * scale)), max(20, int(h * scale))
        resized = rgba_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        alpha = np.array(resized.split()[-1])
        if np.any(alpha < 180):
            binary = (alpha > 120).astype(np.uint8)
        else:
            gray = resized.convert("L")
            arr = np.array(gray)
            binary = (arr < 160).astype(np.uint8)
            if np.sum(binary) < 50:
                binary = (arr > 120).astype(np.uint8)

        skeleton = zhang_suen_thinning(binary)
        coords = np.argwhere(skeleton == 1)
        if len(coords) < 30:
            coords = np.argwhere(binary == 1)

        if len(coords) == 0:
            return []

        repeat_factor = (total_n // len(coords)) + 1
        extended = np.tile(coords, (repeat_factor, 1))
        noise = np.random.uniform(-0.3, 0.3, size=extended.shape)
        sampled = (extended + noise)[:total_n]

        min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
        min_c, max_c = np.min(sampled[:, 1]), np.max(sampled[:, 1])
        range_r = max(max_r - min_r, 1)
        range_c = max(max_c - min_c, 1)

        pts = []
        for r, c in sampled:
            norm_x = (c - min_c) / range_c
            norm_z = (max_r - r) / range_r
            px = round(-4.8 + norm_x * 9.6, 2)
            py = 4.0
            pz = round(1.8 + norm_z * 4.2, 2)
            pts.append({"x": px, "y": py, "z": pz, "color": "#00F0FF"})
        return pts

def extract_target_text(line_text: str):
    quote_match = re.search(r'[「『"“\']([^「『"”\']+)[\」』"”\']', line_text)
    if quote_match:
        return quote_match.group(1).strip()
    eng_match = re.search(r'\b([A-Z]{2,8})\b', line_text.upper())
    if eng_match:
        token = eng_match.group(1).strip()
        if token not in ["GROUND", "TAKEOFF", "START"]:
            return token
    word_match = re.search(r'(?:排成|文字|排字|字樣)\D*?([\u4e00-\u9fa5]{1,4})', line_text)
    if word_match:
        candidate = word_match.group(1)
        if candidate not in ["地面", "起飛", "隊形", "陣列", "幾何"]:
            return candidate
    return None

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
        prompt_lines = [l.strip() for l in req.prompt.split("\n") if l.strip()]
        scenes = []
        for idx, line in enumerate(prompt_lines):
            scene_name = line
            if idx == 0 or "起飛" in line or "地面" in line:
                pts = generate_ground_takeoff_points(req.drone_count)
                scenes.append({"name": "第 1 幕：地面起飛停機坪", "points": pts})
                continue

            if "龍捲風" in line or "螺旋" in line:
                pts = generate_tornado_points(req.drone_count)
                scenes.append({"name": scene_name, "points": pts})
                continue

            target_text = extract_target_text(line)
            if target_text:
                color = "#FFD700" if any("\u4e00" <= c <= "\u9fa5" for c in target_text) else "#00F0FF"
                pts = render_text_to_points(target_text, req.drone_count, color=color)
                if pts:
                    scenes.append({"name": scene_name, "points": pts})
                    continue

            default_pts = []
            for i in range(req.drone_count):
                angle = (i / req.drone_count) * 2 * math.pi
                px = round(3.5 * math.cos(angle), 2)
                pz = round(3.8 + 2.0 * math.sin(angle), 2)
                default_pts.append({"x": px, "y": 4.0, "z": pz, "color": "#38bdf8"})
            scenes.append({"name": scene_name, "points": default_pts})

        return {"status": "success", "data": json.dumps({"scenes": scenes})}
