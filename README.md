# 校園無人機群飛 AI 編排教室 2.0 (Drone Show AI Workstation)

本專案是一個結合 **Google Gemini AI 導演** 與 **NumPy 3D 幾何工匠演算法** 的 3D 無人機燈光秀群飛編排系統。

---

## 🌟 核心架構特色

1. **AI 導演與幾何工匠分離設計**
   * **Gemini 擔任 3D 總導演**：理解自然語言劇本，將每一幕拆解為結構化的幾何圖元參數（`ground_grid`, `text`, `heart`, `ring`, `spiral`, `sphere`, `star`, `cone`, `wave`, `cylinder`, `cube` 等），並規劃空間縮放、3 軸旋轉與色彩漸層。
   * **NumPy + Pillow 幾何工匠**：後端在 0.05 秒內精確計算出數百至上千架無人機的空間 $(x, y, z)$ 點位與顏色代碼，避免大語言模型輸出大量座標時造成的 Token 浪費或格式錯誤。

2. **立體防撞航線與平滑過渡**
   * **最近鄰配對優化 (`optimizePairings`)**：幕與幕切換時，自動計算無人機空間距離最小化配對，避免空中大範圍交叉碰撞。
   * **立體弧形避障軌跡**：過渡時動態計算深度弧線繞行，並呈現立體航跡。

3. **圖片轉點陣雙模式**
   * 🌟 **像素彩圖模式 (`pixel`)**：採樣圖片非背景色彩點，保留圖片真實色彩。
   * ✏️ **線條骨架模式 (`skeleton`)**：透過 **Zhang-Suen（張-孫）細化演算法** 提取線條骨架，適合簡筆畫與民族圖騰。

4. **安全備援機制 (Offline Fallback)**
   * 若無設定 API Key 或遇網路/配額限制，系統自動啟動智慧幾何備援引擎，依文字關鍵字生成對應幾何造型或中英文字型點陣，確保教學展示不中斷。

---

## 📂 目錄結構

```
droneserver/
├── droneserver.py                  # FastAPI 後端服務與 3D 幾何計算核心
├── index.html                      # 3D 視覺化前端單頁應用 (Plotly.js WebGL)
├── NotoSansTC-VariableFont_wght.ttf # 開源思源黑體字型（中文字型點陣繪製）
├── requirements.txt                # Python 依賴清單
├── .env.example                    # 環境變數範本檔
└── README.md                       # 專案說明文件
```

---

## 🚀 快速開始

### 1. 安裝環境相依套件

建議使用 Python 3.10 以上環境：

```bash
pip install -r requirements.txt
```

### 2. 設定環境變數（可選）

複製 `.env.example` 為 `.env` 並填入您的 Gemini API Key：

```bash
# Windows PowerShell
copy .env.example .env
```

在 `.env` 中填入：
```ini
GEMINI_API_KEY=AIzaSy...您的Gemini金鑰
GEMINI_MODEL=gemini-2.5-flash
PORT=8000
```
> 註：前端頁面亦提供直接輸入 API Key 的輸入欄位，亦可免設定環境變數直接使用。

### 3. 啟動伺服器

```bash
python droneserver.py
```

開啟瀏覽器進入：
👉 **http://localhost:8000**

---

## 📡 API 端點規格

| 方法 | 路徑 | 說明 |
|---|---|---|
| `GET` | `/` | 載入 3D 群飛劇本工作台網頁 (`index.html`) |
| `GET` | `/api/status` | 檢查伺服器連線狀態、API Key 與中文字型就緒狀況 |
| `POST` | `/api/generate-show` | 送出劇本，由 AI 導演解析並生成全場各幕無人機 3D 點位 |
| `POST` | `/api/convert-image-to-drone` | 上傳 Base64 圖檔，提取 3D 彩色或骨架無人機點陣 |

---

## ☁️ 雲端部署 (Render / Railway / Docker)

- **Start Command**: `uvicorn droneserver:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `GEMINI_API_KEY`: 您的 Google Gemini API Key
