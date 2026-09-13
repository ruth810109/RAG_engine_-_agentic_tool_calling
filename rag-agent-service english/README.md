# RAG Agent Service

延續你原本的 RAG 專案（Streamlit + ChromaDB → FastAPI + Gemini），擴充成一個
「知識庫問答 + 多步驟 Agent 工作流」的完整服務，涵蓋台灣 AI應用工程師 / AI Agent 工程師
職缺最常提到的 5 大技術關鍵字：

| 關鍵字 | 對應到專案裡的哪裡 |
|---|---|
| 1. Python 後端開發 | 整個專案（FastAPI、pydantic、LangChain 全部是 Python） |
| 2. LangChain / LangGraph | `app/core/agent_graph.py`：多步驟工作流、條件分支、Tool Calling |
| 3. RAG + 向量資料庫 | `app/core/rag_engine.py`：Chroma 向量庫 + Cross-Encoder Reranking |
| 4. FastAPI（API 服務） | `app/main.py`、`app/api/routes.py`：`/ingest`、`/chat` 端點 |
| 5. Prompt Engineering / Multi-LLM | `app/llm/llm_factory.py`：抽換 LLM 供應商只改一行設定 |

全部使用**免費**方案，不需要訂閱任何服務：
- **Gemini API**：Google AI Studio 提供免費額度（[申請連結](https://aistudio.google.com/app/apikey)）
- **Chroma**：本地端向量資料庫，完全免費
- **Cross-Encoder Reranker**：本地端執行的開源模型（`sentence-transformers`），不吃 API 額度
- **DuckDuckGo 搜尋**：免費網路搜尋工具，不需要任何 API Key

---

## 架構說明

```
使用者提問
    │
    ▼
[FastAPI /chat]
    │
    ▼
[LangGraph Agent]
    │
    ├─ 判斷需要查內部文件 ──► search_knowledge_base 工具 ──► RAG 兩階段檢索
    │                                                        （向量粗篩 → Reranking 精篩）
    ├─ 判斷需要查網路 ──► search_web 工具 ──► DuckDuckGo 免費搜尋
    │
    ▼
[guardrail 節點] 檢查回答是否有根據（簡化版 Human-in-the-Loop 概念）
    │
    ▼
回傳最終答案 + 是否需要人工複核
```

---

## 在 MacBook Air（PyCharm）上執行的步驟

### 1. 建立虛擬環境並安裝套件
在 PyCharm 打開這個資料夾後，開啟內建終端機：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 設定 API Key
```bash
cp .env.example .env
```
打開 `.env`，把 `GOOGLE_API_KEY` 換成你在 [Google AI Studio](https://aistudio.google.com/app/apikey) 免費申請的金鑰。

### 3. 先用 CLI 腳本快速驗證整個流程（不用開伺服器）
```bash
python demo_cli.py
```
你會看到：
1. 範例員工手冊被切片、寫入向量資料庫
2. 問「特休天數」「遠端工作天數」時，Agent 會呼叫 `search_knowledge_base`，並附上頁碼來源
3. 問「2026 年颱風假規定」時，知識庫查不到，Agent 會自動改呼叫 `search_web`

### 4. 啟動 FastAPI 服務
```bash
uvicorn app.main:app --reload
```
打開瀏覽器到 `http://127.0.0.1:8000/docs`，會看到自動產生的 Swagger API 文件，
可以直接在網頁上上傳檔案（`/ingest`）、發問（`/chat`）測試。

### 5. 執行測試
```bash
pytest
```

### 6.（選用）用 Docker 打包
等你學到 Docker 那週時：
```bash
docker build -t rag-agent-service .
docker run -p 8000:8000 --env-file .env rag-agent-service
```

---

## 履歷可以怎麼寫

> 開發一個結合 RAG 知識庫問答與 LangGraph 多步驟 Agent 的後端服務：
> 以 FastAPI 提供 API，Agent 能自主判斷查詢內部文件（向量檢索 + Cross-Encoder Reranking）
> 或呼叫網路搜尋工具，並附上原文頁碼佐證；LLM 供應商採抽象層設計，可於 Gemini / OpenAI /
> Claude 間切換而不需更動核心邏輯。

---

## 下一步擴充方向（對應你的 8 週計畫）

1. **CI/CD**：加一個 GitHub Actions，push 時自動跑 `pytest`
2. **Docker Compose**：把 FastAPI 服務跟未來可能加的資料庫（如 Postgres 存對話紀錄）一起編排
3. **Multi-Agent**：把 `agent_graph.py` 拆成「搜尋 Agent」「整理摘要 Agent」「報告產出 Agent」三個節點協作
4. **雲端部署**：部署到 Render / Railway 等有免費額度的平台，取得一個公開 Demo 連結
