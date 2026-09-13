"""
API routing layer (skill keyword 4: FastAPI service development).
"""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from app.core.agent_graph import run_agent
from app.core.rag_engine import rag_engine
from app.schemas.chat import ChatRequest, ChatResponse, IngestResponse

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: list[UploadFile] = File(...)):
    """Upload PDF or txt/md files and write them into the vector database."""
    saved_paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for f in files:
            dest = Path(tmp_dir) / f.filename
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved_paths.append(str(dest))

        chunks_written = rag_engine.ingest_files(saved_paths)

    return IngestResponse(
        filenames=[f.filename for f in files],
        chunks_written=chunks_written,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Pass the question to the agentic workflow and return the final answer."""
    result = run_agent(request.query)
    return ChatResponse(**result)
