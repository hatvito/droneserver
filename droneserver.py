import os
import asyncio
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Drone Show AI Proxy Server")

# 允許跨來源存取 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

@app.get("/")
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

request_lock = asyncio.Lock()

SYSTEM_PROMPT = """你是一位專業的無人機燈光秀幾何工程師。
使用者會提供無人機總架數 N 與演出劇本。請為每一幕計算長度剛好為 N 的 3D 空間點陣。
限制：
1. 坐標範圍：X [-6.0, 6.0], Y [3.5, 4.5], Z [1.8, 6.0]。地面起飛幕 Z=0。
2. 若某圖形點陣不足 N 架，多餘無人機安排於 Z=6.5, Y=6.0 待命，顏色設定為 "rgba(0,0,0,0)" (熄燈隱身)。
3. 嚴格輸出純 JSON，格式如下：
{
  "scenes": [
    {
      "name": "劇幕名稱",
      "points": [
        {"x": 0.0, "y": 4.0, "z": 3.0, "color": "#FF4B4B"}
      ]
    }
  ]
}
"""

@app.post("/api/generate-show")
async def generate_show(req: ShowRequest):
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY 未設定！")
        raise HTTPException(status_code=500, detail="Server GEMINI_API_KEY is missing")

    async with request_lock:
        client = genai.Client(api_key=GEMINI_API_KEY)
        user_content = f"無人機總架數：{req.drone_count}\n劇本需求：\n{req.prompt}"
        candidate_models = ["gemini-3.8-flash"]
        last_error = None

        print(f"[{req.student_name}] 正在生成 {req.drone_count} 架燈光秀...")

        for model_name in candidate_models:
            for attempt in range(2):
                try:
                    def call_gemini():
                        return client.models.generate_content(
                            model=model_name,
                            contents=user_content,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                response_mime_type="application/json",
                                temperature=0.2
                            )
                        )

                    response = await asyncio.to_thread(call_gemini)
                    
                    resp_text = response.text.strip()
                    if resp_text.startswith("```json"):
                        resp_text = resp_text[7:]
                    if resp_text.startswith("```"):
                        resp_text = resp_text[3:]
                    if resp_text.endswith("```"):
                        resp_text = resp_text[:-3]
                    resp_text = resp_text.strip()

                    _ = json.loads(resp_text)
                    print(f"[{req.student_name}] 成功使用 {model_name} 生成！")
                    await asyncio.sleep(2.0)
                    return {"status": "success", "data": resp_text}

                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    print(f"[{model_name}] 嘗試失敗 ({attempt+1}/2): {err_str}")
                    if "503" in err_str or "UNAVAILABLE" in err_str:
                        await asyncio.sleep(1.5)
                        continue
                    break

        print(f"所有模型嘗試均失敗：{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AI 生成失敗: {str(last_error)}")
