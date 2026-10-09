import os
import asyncio
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

# 定義嚴格的 Pydantic 輸出結構，強制 Gemini 遵守
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

SYSTEM_PROMPT = """你是一位專業的無人機燈光秀幾何工程師。
使用者會提供無人機總架數 N 與演出劇本。請為每一幕計算長度剛好為 N 的 3D 空間點陣。
限制：
1. 坐標範圍：X [-6.0, 6.0], Y [3.5, 4.5], Z [1.8, 6.0]。地面起飛幕 Z=0。
2. 若某圖形點陣不足 N 架，多餘無人機安排於 Z=6.5, Y=6.0 待命，顏色設定為 "rgba(0,0,0,0)" (熄燈隱身)。
3. 請嚴格遵守提供的 JSON Schema 結構輸出。
"""

@app.post("/api/generate-show")
async def generate_show(req: ShowRequest):
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY 未設定！")
        raise HTTPException(status_code=500, detail="Server GEMINI_API_KEY is missing")

    async with request_lock:
        print(f"[{req.student_name}] 開始處理請求，架數: {req.drone_count}")
        user_content = f"無人機總架數：{req.drone_count}\n劇本需求：\n{req.prompt}"
        # 專注使用額度充裕的 gemini-3.5-flash-lite
        model_name = "gemini-3.5-flash-lite"
        
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Client init error: {str(e)}")

        for attempt in range(1, 4):
            try:
                print(f"[{model_name}] 正在呼叫 Schema 強制輸出 (嘗試 {attempt}/3)...")

                def call_gemini():
                    return client.models.generate_content(
                        model=model_name,
                        contents=user_content,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            response_mime_type="application/json",
                            response_schema=DroneShowOutput,  # 強制符合 Schema，杜絕語法斷裂
                            temperature=0.1
                        )
                    )

                response = await asyncio.wait_for(
                    asyncio.to_thread(call_gemini),
                    timeout=90.0
                )

                resp_text = response.text.strip()
                # 驗證輸出
                _ = json.loads(resp_text)
                print(f"[{req.student_name}] 成功完成 100 架無人機座標生成！")
                await asyncio.sleep(1.0)
                return {"status": "success", "data": resp_text}

            except Exception as e:
                err_msg = str(e)
                print(f"嘗試 {attempt} 失敗: {err_msg}")
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    await asyncio.sleep(3.0)
                    continue
                if attempt == 3:
                    raise HTTPException(status_code=500, detail=f"AI 生成失敗: {err_msg}")

        raise HTTPException(status_code=500, detail="伺服器忙碌，請稍候重試")
