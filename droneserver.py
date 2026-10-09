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

# ----------------- 圖片轉點陣核心引擎 -----------------
def image_to_drone_points(image_bytes: bytes, total_n: int):
    """
    將使用者上傳的圖片轉為剛好 N 個三維空間座標點
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    
    # 統一轉成 RGBA
    rgba_img = pil_img.convert("RGBA")
    w, h = rgba_img.size
    
    # 等比例縮放至合理解析度加速計算
    max_dim = 150
    scale = max_dim / max(w, h)
    new_w = max(10, int(w * scale))
    new_h = max(10, int(h * scale))
    resized = rgba_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 提取有效線條點：
    # 1. 若有透明背景 (PNG)，以 Alpha > 60 為線條
    # 2. 若為白色背景圖檔，以灰階亮線/邊緣濾鏡提取線條
    alpha_channel = np.array(resized.split()[-1])
    is_transparent_png = np.any(alpha_channel < 200)

    if is_transparent_png:
        # 有透明度遮罩，抓取非透明圖案邊界
        gray = resized.convert("L")
        edge = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edge)
        active_coords = np.argwhere((alpha_channel > 80) & (edge_arr > 30))
        if len(active_coords) < total_n // 2:
            # 若邊緣太少，直接取整個實體輪廓
            active_coords = np.argwhere(alpha_channel > 100)
    else:
        # 一般 JPG 白底黑線圖（黑線為筆劃）
        gray = resized.convert("L")
        arr = np.array(gray)
        # 黑色/深色筆劃當作線條
        active_coords = np.argwhere(arr < 180)
        if len(active_coords) < 10:
            # 若圖像是黑底白線，反向抓亮色
            active_coords = np.argwhere(arr > 120)

    if len(active_coords) == 0:
        return []

    # 均勻抽樣至 N 顆點
    if len(active_coords) >= total_n:
        indices = np.linspace(0, len(active_coords) - 1, total_n, dtype=int)
        sampled = active_coords[indices]
    else:
        repeat_factor = (total_n // len(active_coords)) + 1
        extended = np.tile(active_coords, (repeat_factor, 1))
        sampled = extended[:total_n]

    points = []
    # 橫向範圍 X: [-5.0, 5.0], Y=4.0, 高度 Z: [1.8, 5.8]
    for r, c in sampled:
        px = round(-5.0 + (c / new_w) * 10.0, 2)
        py = 4.0
        pz = round(1.8 + ((new_h - r) / new_h) * 4.0, 2)
        
        # 提取原圖像素顏色
        r_val, g_val, b_val, _ = resized.getpixel((int(c), int(r)))
        # 若原圖接近黑色，轉為顯眼的科技亮青色，否則保留原色彩
        if r_val < 40 and g_val < 40 and b_val < 40:
            hex_color = "#00FFFF"
        else:
            hex_color = f"#{r_val:02X}{g_val:02X}{b_val:02X}"
            
        points.append({"x": px, "y": py, "z": pz, "color": hex_color})

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

# 上傳圖片轉無人機點陣的獨立接口
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
                "name": req.scene_name or "自訂圖片造型",
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
