from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="RAG Agent Service",
    description="結合 RAG 知識庫查詢 + LangGraph 多步驟 Agent 的作品集專案",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "RAG Agent Service 運作中，請參考 /docs 查看 API 文件"}
