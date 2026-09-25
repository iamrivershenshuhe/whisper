# whisper-flow

類 [Wispr Flow](https://wisprflow.ai/) 的 AI 語音輸入後端：**語音辨識（ASR）＋ LLM 即時潤稿**，把口語直接變成可以貼上的通順文字。API 供應商可抽換，第一版支援 OpenAI（以及任何 OpenAI 相容 API）。

目前只有後端（Python 函式庫、本機 HTTP API、CLI），桌面端（全域快捷鍵、貼到游標位置）之後再做。

## 功能

| 功能 | 狀態 |
|---|---|
| 語音轉文字（`gpt-transcribe`，中英夾雜語言提示） | ✅ |
| 智慧潤稿：去除贅字、套用自我修正（「五點…不對，六點」→「六點」）、標點、分段 | ✅ |
| 情境風格：依 client 傳來的 app（LINE → 口語、Notion/Gmail → 正式） | ✅ |
| 語音快捷片語（Snippets）：整句說出觸發詞 → 直接輸出預設內容 | ✅ |
| 個人字典：同時提示 ASR（`keywords`）和潤稿模型 | ✅ |
| 本機歷史紀錄：音檔＋文字存在 SQLite，失敗可以重試 | ✅ |
| 多供應商：用設定切換，OpenAI 相容 API 不必寫程式碼 | ✅（目前實作 OpenAI） |
| 串流／即時辨識（邊講邊轉，放開按鍵後 1 秒內出字） | 規劃中 |
| 桌面 client（快捷鍵、偵測前景 app、貼到游標） | 規劃中 |

## 快速開始

需要 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync                      # 安裝相依套件（要用麥克風錄音請加 --extra mic）
cp .env.example .env         # 填入 OPENAI_API_KEY
cp config.example.toml config.toml   # 選用，不建立就全部使用預設值
```

```bash
uv run whisper-flow dictate clip.wav --app LINE.exe   # 處理音檔
uv run whisper-flow record --app Notion               # 麥克風錄音，按 Enter 結束
uv run whisper-flow serve                             # 啟動 HTTP API（http://127.0.0.1:8765）
uv run whisper-flow history                           # 查看歷史紀錄
uv run whisper-flow retry 12                          # 用存下的音檔重跑第 12 筆
uv run whisper-flow config                            # 顯示使用中的設定檔和資料目錄
```

## HTTP API

啟動 `serve` 之後，互動式文件在 `http://127.0.0.1:8765/docs`。

| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/v1/dictate` | multipart：`audio`（wav/mp3/m4a/webm…）、`app`（選填）→ 潤稿結果 |
| `GET` | `/v1/history?limit=&offset=&q=` | 歷史紀錄，`q` 會搜尋文字內容 |
| `GET` | `/v1/history/{id}` | 單筆紀錄 |
| `GET` | `/v1/history/{id}/audio` | 原始音檔 |
| `POST` | `/v1/history/{id}/retry` | 用存下的音檔重新處理 |
| `GET` | `/health` | 狀態與目前使用的模型 |

```bash
curl -F audio=@clip.wav -F app=LINE.exe http://127.0.0.1:8765/v1/dictate
```

回傳範例：

```json
{
  "id": 12,
  "text": "明天下午六點開會。",
  "raw_text": "嗯 明天下午五點 不對 六點開會",
  "style": "casual",
  "snippet": null,
  "languages": ["zh"],
  "asr_model": "gpt-transcribe",
  "llm_model": "gpt-6-luna",
  "timings": {"asr_ms": 640, "llm_ms": 410, "total_ms": 1050},
  "warnings": []
}
```

API 沒有驗證機制，預設只綁 `127.0.0.1`。

## 設定

完整說明見 [config.example.toml](config.example.toml)。設定檔的尋找順序：`--config` → `$WHISPER_FLOW_CONFIG` → `./config.toml` → 使用者設定目錄 → 內建預設值。

### 換供應商

供應商在 `[providers.<名稱>]` 定義，`[asr]` 和 `[llm]` 可以各自指定不同的供應商。只要是 OpenAI 相容的 API 都能直接用 `type = "openai"`：

```toml
[providers.groq]
type = "openai"
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"

[asr]
provider = "groq"
model = "whisper-large-v3"
send_keywords = false
```

要接不相容的 API（例如 Anthropic、Deepgram），在 `src/whisper_flow/providers/` 新增一個實作 `Transcriber` 或 `ChatModel` 介面的模組，再註冊到 `providers/__init__.py` 的 `_TYPES`。

## 架構

```
audio ──► Transcriber (ASR) ──► 符合快捷片語？ ──是──► 片語內容
                                  │否
                                  ▼
                   ChatModel 潤稿（風格依 app 決定）──► text
```

- `providers/`：與供應商無關的介面（`base.py`）、OpenAI 實作、依 `type` 查找的註冊表
- `text.py`：快捷片語比對、app → 風格、提示詞組裝（純函式，容易測試）
- `pipeline.py`：串起整個流程；呼叫任何 API **之前**先存音檔，失敗也不會遺失
- `history.py`：SQLite ＋ 音檔
- `server.py` / `cli.py`：FastAPI 與命令列入口

## 開發

```bash
uv run pytest          # 測試（不會連網，OpenAI 請求用 mock transport 驗證）
uv run ruff check      # lint
uv run ruff format     # 格式化
```
