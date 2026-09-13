"""
Project settings.
Following the approach from your original RAG project: use Pydantic BaseSettings
to centrally manage all tunable parameters, instead of scattering standalone
constants throughout the codebase.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- LLM settings (skill keyword 5: Prompt Engineering / Multi-LLM integration) ----
    LLM_PROVIDER: str = "gemini"          # Just change this line to switch to openai / claude later
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    CHAT_MODEL: str = "gemini-2.0-flash"          # Chat model available within Gemini's free tier
    EMBEDDING_MODEL: str = "models/text-embedding-004"  # Gemini's free embedding model

    # ---- RAG / vector database settings (skill keyword 3) ----
    CHROMA_PATH: str = "./chroma_db"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    RETRIEVE_TOP_K: int = 8          # Number of candidates from the initial coarse retrieval
    RERANK_TOP_N: int = 3            # Number of top picks kept after reranking

    # ---- Reranker (local model, completely free, doesn't consume API quota) ----
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MIN_RERANK_SCORE: float = 0.15

    # ---- Agent settings (skill keywords 2 / 4: LangGraph workflow, Tool Calling) ----
    MAX_AGENT_STEPS: int = 6
    ENABLE_WEB_SEARCH_TOOL: bool = True   # Uses free DuckDuckGo search, no API key needed

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
