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
from pydantic import BaseModel
from typing import List, Optional

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

request_lock = asyncio.Lock()

# ----------------- 顏色庫 -----------------
COLOR_MAP = {
    "藍": (0, 140, 255),
    "深藍": (0, 60, 180),
    "淺藍": (135, 206, 250),
    "紫": (160, 32, 240),
    "深紫": (80, 0, 130),
    "淺紫": (221, 160, 221),
    "紅": (255, 40, 40),
    "粉": (255, 105, 180),
    "黃": (255, 215, 0),
    "金": (255, 215, 0),
    "綠": (50, 205, 50),
    "橘": (255, 140, 0),
    "白": (255, 255, 255),
    "青": (0, 240, 255)
}

DEPTH_KEYWORDS = {
    "最前": 2.2,
    "前": 3.0,
    "中": 4.0,
    "後": 5.0,
    "最後": 5.8
}

def hex_from_rgb(rgb_tuple):
    r, g, b = [max(0, min(255, int(v))) for v in rgb_tuple]
    return f"#{r:02X}{g:02X}{b:02X}"

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

# ----------------- 語意位置與色彩解析引擎 -----------------
def parse_char_rules(chars: list, prompt_line: str):
    """
    針對每個字元解析專屬的深度 (Y 軸) 與色彩/漸層設定
    支援語意如：四個字母為紫色漸層，且位置為最前、前、中、後
    """
    rules = []
    num_chars = len(chars)

    # 1. 深度解析 (最前、前、中、後)
    depth_match = re.search(r'(?:位置[為是：:\s]*|分佈[為是：:\s]*)([^，,。]+)', prompt_line)
    assigned_depths = [4.0] * num_chars

    if depth_match:
        depth_str = depth_match.group(1)
        # 尋找所有深度詞彙
        found_depths = []
        # 按長度降序匹配，防止「最前」被切成「最」+「前」
        sorted_keys = sorted(DEPTH_KEYWORDS.keys(), key=lambda k: -len(k))
        tokens = re.split(r'[,，、\s]+', depth_str)
        for tok in tokens:
            for k in sorted_keys:
                if k in tok:
                    found_depths.append(DEPTH_KEYWORDS[k])
                    break
        
        if len(found_depths) >= num_chars:
            assigned_depths = found_depths[:num_chars]
        elif found_depths:
            # 等距補齊
            assigned_depths = [found_depths[i % len(found_depths)] for i in range(num_chars)]

    # 2. 色彩與漸層解析
    has_global_gradient = "漸層" in prompt_line
    # 搜尋主要顏色關鍵字
    base_color_rgb = (160, 32, 240) if "紫" in prompt_line else (0, 140, 255) # 預設紫或藍
    for color_name, rgb in COLOR_MAP.items():
        if color_name in prompt_line:
            base_color_rgb = rgb
            break

    for idx, ch in enumerate(chars):
        char_rule = {
            "y": assigned_depths[idx],
            "base_rgb": base_color_rgb,
            "has_gradient": has_global_gradient,
            "char_ratio": idx / max(num_chars - 1, 1) # 橫向/序號比率 (0.0 ~ 1.0)
        }

        # 檢查該字是否有個別顏色覆蓋 (例如「富字為藍色」)
        char_specific = re.search(rf"{ch}[字為是]*([^，,。]+)", prompt_line)
        if char_specific:
            spec_text = char_specific.group(1)
            for c_name, c_rgb in COLOR_MAP.items():
                if c_name in spec_text:
                    char_rule["base_rgb"] = c_rgb
                    break
            if "漸層" in spec_text:
                char_rule["has_gradient"] = True

        rules.append(char_rule)

    return rules

def render_advanced_text_to_points(text: str, total_n: int, prompt_line: str):
    if not text.strip():
        return None

    chars = list(text)
    num_chars = len(chars)
    drones_per_char = total_n // num_chars

    char_rules = parse_char_rules(chars, prompt_line)
    all_points = []
    
    total_span_x = 10.0
    slot_width = total_span_x / num_chars

    for c_idx, char in enumerate(chars):
        rule = char_rules[c_idx]
        base_rgb = rule["base_rgb"]
        target_y = rule["y"]
        has_gradient = rule["has_gradient"]

        canvas_dim = 120
        img = Image.new("L", (canvas_dim, canvas_dim), color=0)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(FONT_PATH, 92)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), char, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw_x = max(0, (canvas_dim - text_w) // 2)
        draw_y = max(0, (canvas_dim - text_h) // 2)
        draw.text((draw_x, draw_y), char, font=font, fill=255)

        img_arr = np.array(img)
        stroke_coords = np.argwhere(img_arr > 100)

        if len(stroke_coords) == 0:
            continue

        target_char_drones = drones_per_char if c_idx < num_chars - 1 else (total_n - len(all_points))

        if len(stroke_coords) >= target_char_drones:
            indices = np.linspace(0, len(stroke_coords) - 1, target_char_drones, dtype=int)
            sampled = stroke_coords[indices]
        else:
            rep = (target_char_drones // len(stroke_coords)) + 1
            extended = np.tile(stroke_coords, (rep, 1))
            noise = np.random.uniform(-0.35, 0.35, size=extended.shape)
            sampled = (extended + noise)[:target_char_drones]

        char_start_x = -5.0 + c_idx * slot_width
        min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
        range_r = max(max_r - min_r, 1)

        for r, c in sampled:
            norm_c = c / canvas_dim
            px = round(char_start_x + norm_c * slot_width, 2)
            py = round(target_y, 2)  # 使用解析到的深度 (最前、前、中、後)
            
            norm_h = (canvas_dim - r) / canvas_dim
            pz = round(1.8 + norm_h * 4.2, 2)

            # 色彩處理
            if has_gradient:
                # 若說明中提到「由上往下」或預設：垂直由淺入深
                # vertical_ratio: 0.0 (最頂部) -> 1.0 (最底部)
                vertical_ratio = (r - min_r) / range_r
                tint = (1.0 - vertical_ratio) * 0.70
                cur_r = int(base_rgb[0] + (255 - base_rgb[0]) * tint)
                cur_g = int(base_rgb[1] + (255 - base_rgb[1]) * tint)
                cur_b = int(base_rgb[2] + (255 - base_rgb[2]) * tint)
                pt_color = hex_from_rgb((cur_r, cur_g, cur_b))
            else:
                pt_color = hex_from_rgb(base_rgb)

            all_points.append({"x": px, "y": py, "z": pz, "color": pt_color})

    return all_points

# ----------------- 圖片轉點陣雙核心引擎 -----------------
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

    r_chan = img_arr[:, :, 0]
    g_chan = img_arr[:, :, 1]
    b_chan = img_arr[:, :, 2]
    alpha = img_arr[:, :, 3]

    is_bg = (alpha < 80) | ((r_chan > 232) & (g_chan > 232) & (b_chan > 232))
    subject_coords = np.argwhere(~is_bg)

    if len(subject_coords) < 50:
        subject_coords = np.argwhere(alpha > 0)

    if mode == "pixel":
        if len(subject_coords) >= total_n:
            indices = np.linspace(0, len(subject_coords) - 1, total_n, dtype=int)
            sampled = subject_coords[indices]
        else:
            rep = (total_n // len(subject_coords)) + 1
            extended = np.tile(subject_coords, (rep, 1))
            noise = np.random.uniform(-0.35, 0.35, size=extended.shape)
            sampled = (extended + noise)[:total_n]

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

            r_idx = min(max(int(r), 0), new_h - 1)
            c_idx = min(max(int(c), 0), new_w - 1)
            r_val, g_val, b_val, _ = img_arr[r_idx, c_idx]

            if r_val < 30 and g_val < 30 and b_val < 30:
                hex_color = "#1E293B"
            else:
                hex_color = f"#{r_val:02X}{g_val:02X}{b_val:02X}"

            pts.append({"x": px, "y": py, "z": pz, "color": hex_color})
        return pts
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

# ----------------- API 端點 -----------------
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

            target_text = extract_target_text(line)
            if target_text:
                # 傳入整行 prompt_line 進行深度 (Y) 與顏色漸層解析
                pts = render_advanced_text_to_points(target_text, req.drone_count, prompt_line=line)
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
