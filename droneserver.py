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

request_lock = asyncio.Lock()

SYSTEM_PROMPT = """你是一位專業的無人機燈光秀幾何工程師。
使用者會提供無人機總架數 N 與演出劇本。請為每一幕計算長度剛好為 N 的 3D 空間點陣。
限制：
1. 坐標範圍：X [-6.0, 6.0], Y [3.5, 4.5], Z [1.8, 6.0]。地面起飛幕 Z=0。
2. 若某圖形點陣不足 N 架，多餘無人機安排於 Z=6.5, Y=6.0 待命，顏色設定為 "rgba(0,0,0,0)" (熄燈隱身)。
3. 請直接輸出純 JSON 字串，不要有任何前導或後續 Markdown 說明文字。
格式規範：
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
        print(f"[{req.student_name}] 開始處理請求，架數: {req.drone_count}")
        user_content = f"無人機總架數：{req.drone_count}\n劇本需求：\n{req.prompt}"
        candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
        last_error_str = "No error recorded"

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"[ERROR] Client 初始化失敗: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Client init error: {str(e)}")

        for model_name in candidate_models:
            for attempt in range(1, 4):
                try:
                    print(f"[{model_name}] 正在呼叫 (嘗試 {attempt}/3)...")

                    def call_gemini():
                        return client.models.generate_content(
                            model=model_name,
                            contents=user_content,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                response_mime_type="application/json",
                                temperature=0.1
                            )
                        )

                    # 100 架無人機運算量較大，設定 90 秒逾時
                    response = await asyncio.wait_for(
                        asyncio.to_thread(call_gemini),
                        timeout=120.0
                    )

                    resp_text = response.text.strip()

                    # 擷取最外層的大括號，自動剔除任何多餘字元或標記
                    start_idx = resp_text.find("{")
                    end_idx = resp_text.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        resp_text = resp_text[start_idx:end_idx + 1]

                    # 驗證 JSON 有效性
                    _ = json.loads(resp_text)
                    print(f"[{req.student_name}] 成功使用 {model_name} 完成生成！")
                    await asyncio.sleep(1.0)
                    return {"status": "success", "data": resp_text}

                except asyncio.TimeoutError:
                    print(f"[{model_name}] 呼叫逾時 (超過 90 秒)")
                    last_error_str = f"{model_name} timed out"
                    break
                except Exception as e:
                    last_error_str = str(e)
                    print(f"[{model_name}] 呼叫報錯: {last_error_str}")
                    if "503" in last_error_str or "UNAVAILABLE" in last_error_str:
                        await asyncio.sleep(3.0)
                        continue
                    break

        print(f"所有模型嘗試均失敗，最後錯誤: {last_error_str}")
        raise HTTPException(status_code=500, detail=f"AI 生成失敗: {last_error_str}")
