import os
import asyncio
import json
import traceback
import re
import io
import base64
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
    scene_name: Optional[str] = "校徽圖形"

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

# ----------------- 快速形態學骨架細化 (Zhang-Suen 演算法) -----------------
def zhang_suen_thinning(binary_image: np.ndarray) -> np.ndarray:
    """將厚實筆劃壓成 1 像素寬的中心骨架線，防止雙重邊緣雜點"""
    img = binary_image.copy()
    prev = np.zeros_like(img)

    while True:
        # Step 1
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

        # Step 2
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

def image_to_drone_points(image_bytes: bytes, total_n: int):
    pil_img = Image.open(io.BytesIO(image_bytes))
    rgba_img = pil_img.convert("RGBA")

    # 1. 調整至合適尺寸進行骨架萃取 (140x140)
    target_dim = 140
    w, h = rgba_img.size
    scale = target_dim / max(w, h)
    new_w, new_h = max(20, int(w * scale)), max(20, int(h * scale))
    resized = rgba_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 2. 轉為乾淨的二值化矩陣 (1 代表線條筆劃, 0 代表背景)
    alpha = np.array(resized.split()[-1])
    is_png_transparent = np.any(alpha < 180)

    if is_png_transparent:
        binary = (alpha > 120).astype(np.uint8)
    else:
        gray = resized.convert("L")
        arr = np.array(gray)
        # 校徽通常是白底黑字：黑色筆劃 < 160
        binary = (arr < 160).astype(np.uint8)
        if np.sum(binary) < 50:
            binary = (arr > 120).astype(np.uint8)

    # 3. 執行骨架細化：把所有粗筆劃提煉為清晰單像素中心線條
    skeleton = zhang_suen_thinning(binary)
    coords = np.argwhere(skeleton == 1)  # [row, col] -> [y, x]

    if len(coords) < 30:
        # 備援：若骨架點過少則取原筆劃
        coords = np.argwhere(binary == 1)

    if len(coords) == 0:
        return []

    # 4. 沿著筆劃骨架平滑排序
    ordered = [coords[0]]
    remaining = set(range(1, len(coords)))
    curr_idx = 0

    if len(coords) > 1200:
        step = len(coords) // 1200
        coords = coords[::step]
        remaining = set(range(1, len(coords)))

    while remaining and len(ordered) < total_n * 2:
        curr_pt = coords[curr_idx]
        rem_list = list(remaining)
        diffs = coords[rem_list] - curr_pt
        dists = diffs[:, 0]**2 + diffs[:, 1]**2
        best_match_idx = rem_list[np.argmin(dists)]
        ordered.append(coords[best_match_idx])
        remaining.remove(best_match_idx)
        curr_idx = best_match_idx

    ordered_coords = np.array(ordered)

    # 5. 等弧長均勻取樣剛好 total_n 架
    if len(ordered_coords) >= total_n:
        indices = np.linspace(0, len(ordered_coords) - 1, total_n, dtype=int)
        sampled = ordered_coords[indices]
    else:
        repeat_factor = (total_n // len(ordered_coords)) + 1
        extended = np.tile(ordered_coords, (repeat_factor, 1))
        sampled = extended[:total_n]

    # 6. 置中放大並映射至 3D 空間
    min_r, max_r = np.min(sampled[:, 0]), np.max(sampled[:, 0])
    min_c, max_c = np.min(sampled[:, 1]), np.max(sampled[:, 1])
    range_r = max(max_r - min_r, 1)
    range_c = max(max_c - min_c, 1)

    points = []
    # X 範圍 [-5.0, 5.0], 高度 Z [1.8, 6.0]
    for r, c in sampled:
        norm_x = (c - min_c) / range_c
        norm_z = (max_r - r) / range_r

        px = round(-4.8 + norm_x * 9.6, 2)
        py = 4.0
        pz = round(1.8 + norm_z * 4.2, 2)

        # 預設科技藍光，凸顯校徽乾淨輪廓
        points.append({"x": px, "y": py, "z": pz, "color": "#00F0FF"})

    return points

# ----------------- 中文字型點陣引擎 -----------------
def render_text_to_points(text: str, total_n: int, color: str = "#00F0FF"):
    if not text.strip():
        return None

    canvas_w = 120
    canvas_h = 60
    img = Image.new("L", (canvas_w, canvas_h), color=0)
    draw = ImageDraw.Draw(img)

    text_len = len(text)
    font_size = 46 if text_len <= 1 else (32 if text_len <= 4 else 20)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw_x = max(0, (canvas_w - text_w) // 2)
    draw_y = max(0, (canvas_h - text_h) // 2)
    draw.text((draw_x, draw_y), text, font=font, fill=255)

    img_arr = np.array(img)
    stroke_coords = np.argwhere(img_arr > 120)

    if len(stroke_coords) == 0:
        return None

    if len(stroke_coords) >= total_n:
        indices = np.linspace(0, len(stroke_coords) - 1, total_n, dtype=int)
        sampled = stroke_coords[indices]
    else:
        repeat_factor = (total_n // len(stroke_coords)) + 1
        extended = np.tile(stroke_coords, (repeat_factor, 1))
        sampled = extended[:total_n]

    points = []
    for r, c in sampled:
        px = round(-5.0 + (c / canvas_w) * 10.0, 2)
        py = 4.0
        pz = round(1.8 + ((canvas_h - r) / canvas_h) * 3.7, 2)
        points.append({"x": px, "y": py, "z": pz, "color": color})

    return points

SYSTEM_PROMPT = """你是一位專業的無人機群飛幾何工程師。
使用者會提供總架數 N 與演出劇本。請為每一幕計算長度剛好為 N 的 3D 空間點陣。
規則：
1. 坐標範圍：X [-6.0, 6.0], Y [3.5, 4.5], Z [1.8, 6.0]。地面起飛幕 Z=0。
2. 幾何圖形每幕長度剛好為 N。
3. 嚴格輸出符合提供的 JSON Schema 結構。
"""

def extract_target_text(line_text: str, is_first_scene: bool = False):
    if is_first_scene or "起飛" in line_text or "地面" in line_text:
        return None

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
        points = image_to_drone_points(image_data, req.drone_count)
        if not points:
            raise HTTPException(status_code=400, detail="無法從圖片中提取出明顯輪廓，請使用對比度更高的圖片！")
        return {
            "status": "success",
            "scene": {
                "name": req.scene_name or "校徽造型",
                "points": points
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"圖片解析失敗: {str(e)}")

@app.post("/api/generate-show")
async def generate_show(req: ShowRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server GEMINI_API_KEY is missing")

    async with request_lock:
        print(f"[{req.student_name}] 開始處理請求，架數: {req.drone_count}")
        user_content = f"無人機總架數：{req.drone_count}\n劇本需求：\n{req.prompt}"
        model_name = "gemini-3.5-flash-lite"
        prompt_lines = [line.strip() for line in req.prompt.split("\n") if line.strip()]

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Client init error: {str(e)}")

        for attempt in range(1, 4):
            try:
                def call_gemini():
                    return client.models.generate_content(
                        model=model_name,
                        contents=user_content,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            response_mime_type="application/json",
                            response_schema=DroneShowOutput,
                            temperature=0.1
                        )
                    )

                response = await asyncio.wait_for(
                    asyncio.to_thread(call_gemini),
                    timeout=90.0
                )

                resp_obj = json.loads(response.text.strip())
                scenes = resp_obj.get("scenes", [])

                for idx, scene in enumerate(scenes):
                    line_ref = prompt_lines[idx] if idx < len(prompt_lines) else scene.get("name", "")
                    target_text = extract_target_text(line_ref, is_first_scene=(idx == 0))

                    if not target_text and idx != 0:
                        target_text = extract_target_text(scene.get("name", ""), is_first_scene=False)

                    if target_text:
                        print(f"[{req.student_name}] 第 {idx+1} 幕精確匹配文字: [{target_text}] -> 啟動 Pillow 繪製！")
                        color = "#FFD700" if any("\u4e00" <= c <= "\u9fa5" for c in target_text) else "#00F0FF"
                        fixed_pts = render_text_to_points(target_text, req.drone_count, color=color)
                        if fixed_pts:
                            scene["points"] = fixed_pts

                print(f"[{req.student_name}] 成功生成 {req.drone_count} 架無人機演出！")
                await asyncio.sleep(1.0)
                return {"status": "success", "data": json.dumps(resp_obj)}

            except Exception as e:
                err_msg = str(e)
                print(f"嘗試 {attempt} 失敗: {err_msg}")
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    await asyncio.sleep(3.0)
                    continue
                if attempt == 3:
                    raise HTTPException(status_code=500, detail=f"AI 生成失敗: {err_msg}")

        raise HTTPException(status_code=500, detail="伺服器忙碌，請稍候重試")
