import os
import asyncio
import json
import traceback
import re
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

# 內建 5x7 點陣標準英文字型（保證字體完美工整）
FONT_5X7 = {
    'S': [
        " ### ",
        "#   #",
        "#    ",
        " ### ",
        "    #",
        "#   #",
        " ### "
    ],
    'L': [
        "#    ",
        "#    ",
        "#    ",
        "#    ",
        "#    ",
        "#    ",
        "#####"
    ],
    'H': [
        "#   #",
        "#   #",
        "#   #",
        "#####",
        "#   #",
        "#   #",
        "#   #"
    ],
    'A': [
        " ### ",
        "#   #",
        "#   #",
        "#####",
        "#   #",
        "#   #",
        "#   #"
    ],
    'P': [
        "#### ",
        "#   #",
        "#   #",
        "#### ",
        "#    ",
        "#    ",
        "#    "
    ],
    'Y': [
        "#   #",
        "#   #",
        " # # ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  "
    ]
}

def generate_text_points(text: str, total_n: int, color="#00F0FF"):
    text = text.upper()
    valid_chars = [c for c in text if c in FONT_5X7]
    if not valid_chars:
        return None
    
    char_w = 5
    char_h = 7
    spacing = 2
    total_w = len(valid_chars) * char_w + (len(valid_chars) - 1) * spacing
    
    scale_x = 8.0 / max(total_w, 1)
    scale_z = 3.2 / char_h
    start_x = - (total_w * scale_x) / 2.0
    base_z = 2.2
    
    active_points = []
    for char_idx, char in enumerate(valid_chars):
        grid = FONT_5X7[char]
        offset_x = start_x + char_idx * (char_w + spacing) * scale_x
        for r in range(char_h):
            for c in range(char_w):
                if grid[r][c] == '#':
                    px = round(offset_x + c * scale_x, 2)
                    pz = round(base_z + (char_h - 1 - r) * scale_z, 2)
                    active_points.append({"x": px, "y": 4.0, "z": pz, "color": color})
                    
    # 如果點數超過 N，均勻抽樣；不足 N，多餘機身於高空熄燈隱身
    if len(active_points) > total_n:
        step = len(active_points) / total_n
        final_points = [active_points[int(i * step)] for i in range(total_n)]
    else:
        final_points = active_points[:]
        while len(final_points) < total_n:
            final_points.append({"x": 0.0, "y": 6.0, "z": 6.5, "color": "rgba(0,0,0,0)"})
    return final_points

SYSTEM_PROMPT = """你是一位專業的無人機群飛幾何工程師。
使用者會提供總架數 N 與演出劇本。請為每一幕計算長度剛好為 N 的 3D 空間點陣。

規則：
1. 坐標範圍：X [-6.0, 6.0], Y [3.5, 4.5], Z [1.8, 6.0]。地面起飛幕 Z=0。
2. 圖形請以幾何對稱分布（圓形、龍捲風螺旋、愛心等），每幕長度必須剛好等於 N。
3. 若某幕為文字排字（如 SLHS），請依據字形骨架排布，若不足架數請熄燈隱身 (color: "rgba(0,0,0,0)")。
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
                
                # 自動校準檢查：若劇幕名稱或劇本中包含 SLHS 或 HAPPY，自動注入完美字模
                for scene in resp_obj.get("scenes", []):
                    scene_name = scene.get("name", "").upper()
                    # 偵測是否為排字幕
                    matched_text = None
                    if "SLHS" in scene_name or ("SLHS" in req.prompt.upper() and ("校徽" in scene_name or "SLHS" in scene_name or "縮寫" in scene_name)):
                        matched_text = "SLHS"
                    elif "HAPPY" in scene_name or ("HAPPY" in req.prompt.upper() and "HAPPY" in scene_name):
                        matched_text = "HAPPY"
                        
                    if matched_text:
                        print(f"[{req.student_name}] 啟動字模引擎優化劇幕: {scene.get('name')} -> {matched_text}")
                        fixed_points = generate_text_points(matched_text, req.drone_count, color="#00F0FF")
                        if fixed_points:
                            scene["points"] = fixed_points

                print(f"[{req.student_name}] 成功完成 100 架無人機演出生成！")
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
