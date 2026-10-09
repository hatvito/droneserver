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

# 跨來源存取設定 (CORS)
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

# 定義嚴格的 Pydantic 輸出結構，強制 Gemini 遵守格式輸出
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

# 強化幾何與視覺對稱性的系統提示詞
SYSTEM_PROMPT = """你是一位世界頂級的無人機群飛幾何工程師兼視覺總監。
使用者會提供無人機總架數 N 與演出劇本。請為每一幕計算長度剛好為 N 的 3D 空間點陣。

【幾何排布核心原則（確保圖形清晰美觀）】：
1. 輪廓對稱與均勻分佈：
   - 若為幾何形狀（如圓形、愛心、五角星、雙螺旋、同心圓），請嚴格依照幾何對稱性均勻取樣空間點，點與點之間距離需平滑等距，不得隨機散亂。
   - 文字或圖標排布（如英文字母）：請將點陣嚴格排列在筆劃骨架上（等距直線或平滑圓弧），維持字形工整清晰。
2. 空間坐標範圍限制：
   - X 軸範圍：[-6.0, 6.0]（橫向展開）
   - Y 軸範圍：[3.5, 4.5]（縱深景深，若為正視圖平面圖形可固定 Y=4.0）
   - Z 軸範圍：[1.8, 6.0]（飛行高度，第 0 幕地面待命或起飛前固定 Z=0.0）
3. 架數剛好等於 N：
   - 每一幕的 points 陣列長度必須剛好等於 N。
   - 若主圖形只需 M 架 (M < N)，剩餘的 (N - M) 架無人機可作為「外圍背景星光」均勻環繞在周圍，或安排在 (x=0.0, y=6.0, z=6.5) 熄燈隱藏 (color 設為 "rgba(0,0,0,0)")。
4. 顏色美學：請使用鮮明對比的 Hex 色碼（例如：科技藍 #00F0FF、烈焰紅 #FF3366、璀璨金 #FFD700、純白 #FFFFFF）。
5. 輸出規範：請嚴格遵守提供的 JSON Schema 結構輸出。
"""

@app.post("/api/generate-show")
async def generate_show(req: ShowRequest):
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY 未設定！")
        raise HTTPException(status_code=500, detail="Server GEMINI_API_KEY is missing")

    async with request_lock:
        print(f"[{req.student_name}] 開始處理請求，架數: {req.drone_count}")
        user_content = f"無人機總架數：{req.drone_count}\n劇本需求：\n{req.prompt}"
        model_name = "gemini-3.5-flash-lite"

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"[ERROR] Client 初始化失敗: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Client init error: {str(e)}")

        for attempt in range(1, 4):
            try:
                print(f"[{model_name}] 正在呼叫幾何生成 (嘗試 {attempt}/3)...")

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

                resp_text = response.text.strip()
                _ = json.loads(resp_text)
                print(f"[{req.student_name}] 成功完成 {req.drone_count} 架無人機幾何陣列生成！")
                await asyncio.sleep(1.0)
                return {"status": "success", "data": resp_text}

            except Exception as e:
                err_msg = str(e)
                print(f"嘗試 {attempt} 失敗: {err_msg}")
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    await asyncio.sleep(3.0)
                    continue
                if attempt == 3:
                    print(f"所有嘗試失敗: {traceback.format_exc()}")
                    raise HTTPException(status_code=500, detail=f"AI 生成失敗: {err_msg}")

        raise HTTPException(status_code=500, detail="伺服器忙碌，請稍候重試")
