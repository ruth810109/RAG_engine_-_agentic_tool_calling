from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    needs_human_review: bool
    step_count: int


class IngestResponse(BaseModel):
    filenames: list[str]
    chunks_written: int
