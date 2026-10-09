import os
import asyncio
import json
import traceback
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
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

# 對齊 GitHub 倉庫中的實際檔名
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

def render_text_to_points(text: str, total_n: int, color: str = "#00F0FF"):
    if not text.strip():
        return None

    canvas_w = 120
    canvas_h = 60
    img = Image.new("L", (canvas_w, canvas_h), color=0)
    draw = ImageDraw.Draw(img)

    text_len = len(text)
    if text_len <= 1:
        font_size = 46
    elif text_len <= 4:
        font_size = 32
    else:
        font_size = 20

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        print(f"[WARN] 載入 {FONT_PATH} 失敗: {e}")
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

    # 1. 抓取引號中文字（例如「士」）
    quote_match = re.search(r'[「『"“\']([^「『"”\']+)[\」』"”\']', line_text)
    if quote_match:
        return quote_match.group(1).strip()

    # 2. 抓取常見英文縮寫（SLHS, HAPPY）
    eng_match = re.search(r'\b([A-Z]{2,8})\b', line_text.upper())
    if eng_match:
        token = eng_match.group(1).strip()
        if token not in ["GROUND", "TAKEOFF", "START"]:
            return token

    # 3. 抓取排字後方的中文字詞
    word_match = re.search(r'(?:排成|文字|排字|字樣)\D*?([\u4e00-\u9fa5]{1,4})', line_text)
    if word_match:
        candidate = word_match.group(1)
        if candidate not in ["地面", "起飛", "隊形", "陣列", "幾何"]:
            return candidate

    return None

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
