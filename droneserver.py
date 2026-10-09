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

@app.get("/")
@app.head("/")
def read_root():
    return {
        "status": "online",
        "has_api_key": bool(GEMINI_API_KEY),
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

# ----------------- Pillow 文字點陣引擎 -----------------
FONT_PATH = "NotoSansTC-Regular.ttf"

def render_text_to_points(text: str, total_n: int, color: str = "#00F0FF"):
    """
    使用 Pillow 讀取 NotoSans 字型，將任意中英文字串轉成剛好 N 個三維空間座標點
    """
    if not text.strip():
        return None

    # 1. 建立高解析度黑白畫布
    canvas_w = 120
    canvas_h = 60
    img = Image.new("L", (canvas_w, canvas_h), color=0)
    draw = ImageDraw.Draw(img)

    # 2. 載入字型（依照字數自動調整字體大小）
    font_size = 40 if len(text) <= 2 else (30 if len(text) <= 4 else 22)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        print(f"[WARN] 無法載入 {FONT_PATH}，使用預設字型: {e}")
        font = ImageFont.load_default()

    # 3. 取得文字邊界並置中繪製
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw_x = max(0, (canvas_w - text_w) // 2)
    draw_y = max(0, (canvas_h - text_h) // 2)
    draw.text((draw_x, draw_y), text, font=font, fill=255)

    # 4. 提取白色像素座標
    img_arr = np.array(img)
    stroke_coords = np.argwhere(img_arr > 120)  # [[row, col], ...]

    if len(stroke_coords) == 0:
        return None

    # 5. 均勻抽樣剛好 N 顆點
    if len(stroke_coords) >= total_n:
        indices = np.linspace(0, len(stroke_coords) - 1, total_n, dtype=int)
        sampled = stroke_coords[indices]
    else:
        # 若筆劃點不足 N 架，現有點全用，其餘補熄燈隱身點
        sampled = stroke_coords

    points = []
    # 座標映射：橫向寬度 [-5.5, 5.5], Y=4.0, 高度 Z [2.0, 5.6]
    for r, c in sampled:
        px = round(-5.5 + (c / canvas_w) * 11.0, 2)
        py = 4.0
        pz = round(2.0 + ((canvas_h - r) / canvas_h) * 3.6, 2)
        points.append({"x": px, "y": py, "z": pz, "color": color})

    # 多餘架數安排在上方熄燈隱身待命
    while len(points) < total_n:
        points.append({"x": 0.0, "y": 6.0, "z": 6.5, "color": "rgba(0,0,0,0)"})

    return points

SYSTEM_PROMPT = """你是一位專業的無人機群飛幾何工程師。
使用者會提供總架數 N 與演出劇本。請為每一幕計算長度剛好為 N 的 3D 空間點陣。

規則：
1. 坐標範圍：X [-6.0, 6.0], Y [3.5, 4.5], Z [1.8, 6.0]。地面起飛幕 Z=0。
2. 幾何形狀（如圓形、龍捲風螺旋、愛心等）請均勻分佈。
3. 若某幕為文字排字（如 SLHS、中文名字），請在劇幕名稱中明確寫出「文字：XXX」（例如「文字：SLHS」或「排字：士」）。
4. 必須嚴格輸出 JSON Schema 結構。
"""

@app.post("/api/generate-show")
async def generate_show(req: ShowRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server GEMINI_API_KEY is missing")

    async with request_lock:
        print(f"[{req.student_name}] 開始處理請求，架數: {req.drone_count}")
        user_content = f"無人機總架數：{req.drone_count}\n劇本需求：\n{req.prompt}"
        model_name = "gemini-3.5-flash-lite"

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

                # ----------------- 智慧字模攔截與修復 -----------------
                # 遍歷每一幕，若發現包含中英文文字排排字需求，自動套用 Pillow 精準點陣
                for scene in resp_obj.get("scenes", []):
                    scene_name = scene.get("name", "")
                    target_text = None

                    # 匹配規則：劇本中或劇幕名稱中常見的文字關鍵詞
                    if "SLHS" in scene_name.upper() or "SLHS" in req.prompt.upper() and ("校徽" in scene_name or "縮寫" in scene_name):
                        target_text = "SLHS"
                    elif "HAPPY" in scene_name.upper():
                        target_text = "HAPPY"
                    else:
                        # 檢查是否有其他中文字指定（例如「排成『士』」或「文字：林」）
                        match = re.search(r'(?:排成|文字[：:]|排字[：:]|字樣[：:]|寫出)\s*[「『"“]?([\u4e00-\u9fa5A-Za-z0-9]+)[」』"”]?|([「『][\u4e00-\u9fa5A-Za-z0-9]+[」』])', scene_name)
                        if match:
                            target_text = (match.group(1) or match.group(2) or "").strip("「」『』\"'")

                    if target_text:
                        print(f"[{req.student_name}] 偵測到文字排布，啟動 Pillow 向量點陣: {target_text}")
                        fixed_pts = render_text_to_points(target_text, req.drone_count, color="#00F0FF")
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
