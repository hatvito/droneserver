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

COLOR_MAP = {
    "藍": (0, 140, 255),
    "深藍": (0, 60, 180),
    "淺藍": (135, 206, 250),
    "紫": (170, 40, 250),
    "深紫": (90, 0, 140),
    "淺紫": (225, 160, 230),
    "紅": (255, 40, 40),
    "粉": (255, 105, 180),
    "黃": (255, 215, 0),
    "金": (255, 215, 0),
    "綠": (50, 205, 50),
    "橘": (255, 140, 0),
    "白": (255, 255, 255),
    "青": (0, 240, 255)
}

def hex_from_rgb(rgb_tuple):
    r, g, b = [max(0, min(255, int(v))) for v in rgb_tuple]
    return f"#{r:02X}{g:02X}{b:02X}"

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

# ----------------- 深度與漸層語意解析核心 -----------------
def parse_multi_char_layout(chars: list, prompt_line: str):
    num_chars = len(chars)
    rules = []

    # 1. 深度解析
    assigned_depths = [4.0] * num_chars
    
    # 支援自然語言：「一前一後」、「前後」
    if "一前一後" in prompt_line or "前、後" in prompt_line or "前後" in prompt_line:
        if num_chars == 2:
            assigned_depths = [2.6, 5.2] # 前者在 Y=2.6, 後者在 Y=5.2
        else:
            assigned_depths = np.linspace(2.5, 5.5, num_chars).tolist()
    elif "最前" in prompt_line or "中" in prompt_line:
        depth_order = []
        for word, val in [("最前", 2.2), ("前", 3.0), ("中", 4.0), ("後", 5.0), ("最後", 5.8)]:
            if word in prompt_line:
                depth_order.append(val)
        if len(depth_order) >= num_chars:
            assigned_depths = depth_order[:num_chars]
        else:
            assigned_depths = np.linspace(2.5, 5.5, num_chars).tolist()

    # 2. 顏色與漸層解析
    has_global_gradient = "漸層" in prompt_line
    default_base_rgb = (0, 240, 255) # 預設科技藍

    for c_idx, ch in enumerate(chars):
        char_rgb = default_base_rgb
        char_grad = has_global_gradient

        # 針對個別文字尋找顏色子句 (例如：富字為藍色、邦字紫色有漸層)
        char_match = re.search(rf"{ch}[字為是色\s]*([^，,。]+)", prompt_line)
        clause = char_match.group(1) if char_match else prompt_line

        for c_name, rgb in COLOR_MAP.items():
            if c_name in clause:
                char_rgb = rgb
                break

        if "漸層" in clause or ("淺" in clause and "深" in clause):
            char_grad = True

        rules.append({
            "y": assigned_depths[c_idx],
            "base_rgb": char_rgb,
            "has_gradient": char_grad
        })

    return rules

def render_advanced_text_to_points(text: str, total_n: int, prompt_line: str):
    if not text.strip():
        return None

    chars = list(text)
    num_chars = len(chars)
    drones_per_char = total_n // num_chars
    rules = parse_multi_char_layout(chars, prompt_line)

    all_points = []
    
    # 決定字在舞台橫向的跨度 (X 軸)
    # 如果是一前一後，字可以稍微重疊錯開或左右擺放
    is_front_back = any(abs(rules[i]["y"] - rules[0]["y"]) > 1.0 for i in range(1, num_chars))
    
    total_w = 9.6
    slot_w = total_w / num_chars if not is_front_back else (total_w * 0.7)

    for c_idx, char in enumerate(chars):
        rule = rules[c_idx]
        target_y = rule["y"]
        base_rgb = rule["base_rgb"]
        has_gradient = rule["has_gradient"]

        # 每個字單獨建立高解析單字畫布
        canvas_w = 120
        canvas_h = 120
        img = Image.new("L", (canvas_w, canvas_h), color=0)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(FONT_PATH, 96)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), char, font=font)
        tw = max(bbox[2] - bbox[0], 1)
        th = max(bbox[3] - bbox[1], 1)
        draw.text(((canvas_w - tw) // 2, (canvas_h - th) // 2), char, font=font, fill=255)

        img_arr = np.array(img)
        coords = np.argwhere(img_arr > 90) # [row, col] -> [y, x]

        if len(coords) == 0:
            continue

        target_char_count = drones_per_char if c_idx < num_chars - 1 else (total_n - len(all_points))

        # 抽樣無人機
        if len(coords) >= target_char_count:
            indices = np.linspace(0, len(coords) - 1, target_char_count, dtype=int)
            sampled = coords[indices]
        else:
            rep = (target_char_count // len(coords)) + 1
            extended = np.tile(coords, (rep, 1))
            noise = np.random.uniform(-0.35, 0.35, size=extended.shape)
            sampled = (extended + noise)[:target_char_count]

        # 計算字的橫向 X 基準點
        if is_front_back:
            # 一前一後：前面偏左一點點(-1.5)，後面偏右一點點(+1.5)，具備最佳 3D 透視感
            char_center_x = -1.8 if c_idx == 0 else 1.8
            char_width_span = 4.2
        else:
            # 正常並排排開
            char_center_x = -4.5 + (c_idx + 0.5) * (9.0 / num_chars)
            char_width_span = (8.0 / num_chars) * 0.85

        min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
        min_c, max_c = np.min(sampled[:, 1]), np.max(sampled[:, 1])
        range_r = max(max_r - min_r, 1)
        range_c = max(max_c - min_c, 1)

        for r, c in sampled:
            # 確保 X 有充裕寬度，絕對不會被壓成一條直線！
            norm_x = (c - min_c) / range_c
            px = round((char_center_x - char_width_span / 2) + norm_x * char_width_span, 2)
            
            py = round(target_y, 2)
            
            # 高度 Z 正常展開 (2.0 ~ 5.8 米)
            norm_z = (max_r - r) / range_r
            pz = round(1.8 + norm_z * 4.2, 2)

            # 顏色與漸層
            if has_gradient:
                # 由上往下由淺入深：上方 (r 小) 較淺，下方 (r 大) 較深
                ratio = (r - min_r) / range_r
                tint = (1.0 - ratio) * 0.75 # 頂部白光偏置
                cur_r = int(base_rgb[0] + (255 - base_rgb[0]) * tint)
                cur_g = int(base_rgb[1] + (255 - base_rgb[1]) * tint)
                cur_b = int(base_rgb[2] + (255 - base_rgb[2]) * tint)
                pt_color = hex_from_rgb((cur_r, cur_g, cur_b))
            else:
                pt_color = hex_from_rgb(base_rgb)

            all_points.append({"x": px, "y": py, "z": pz, "color": pt_color})

    # 若為英文 HAPPY，套用五彩光
    if "HAPPY" in text.upper():
        rainbow = ["#FF3366", "#FF9900", "#FFD700", "#33CC33", "#00F0FF", "#9933FF"]
        for p in all_points:
            norm_x = (p["x"] + 5.0) / 10.0
            c_idx = int(norm_x * len(rainbow))
            p["color"] = rainbow[min(c_idx, len(rainbow) - 1)]

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
