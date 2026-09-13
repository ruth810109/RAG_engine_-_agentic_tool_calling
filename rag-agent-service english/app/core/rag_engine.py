"""
Core RAG engine.

Building on your original architecture (WebBaseLoader / RecursiveCharacterTextSplitter /
Chroma / GoogleGenerativeAIEmbeddings), this extends it to:
1. Support batch ingestion of PDF and plain-text files (skill keyword 3: RAG + vector database)
2. Add a proper reranking step, wiring the cross-encoder used for offline evaluation into the
   real retrieval pipeline
3. Keep each chunk's source filename and page number, so answers can include citations
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.llm.llm_factory import get_llm_provider


@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: int | None
    score: float


class RAGEngine:
    def __init__(self):
        self._embedding_model = get_llm_provider().get_embedding_model()
        self._vectorstore = Chroma(
            collection_name="knowledge_base",
            embedding_function=self._embedding_model,
            persist_directory=settings.CHROMA_PATH,
        )
        # Local cross-encoder, completely free and doesn't consume LLM API quota
        self._reranker = CrossEncoder(settings.RERANKER_MODEL)

    # ---------------- Document ingestion ----------------
    def ingest_files(self, file_paths: list[str]) -> int:
        """Read PDF / txt / md files, split them, write them into the vector store,
        and return the number of chunks written."""
        raw_docs: list[Document] = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                raw_docs.extend(PyPDFLoader(path).load())
            else:
                raw_docs.extend(TextLoader(path, encoding="utf-8").load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(raw_docs)
        if not chunks:
            return 0

        self._vectorstore.add_documents(chunks)
        return len(chunks)

    # ---------------- Retrieval + reranking ----------------
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Two-stage retrieval:
        1. Vector search first pulls the top-k candidates (coarse filtering, fast)
        2. The cross-encoder reranker rescores the candidates (fine filtering, more accurate)
        This is a simplified implementation of the "hybrid retrieval + reranking" pattern
        often mentioned in job postings.
        """
        candidates = self._vectorstore.similarity_search(query, k=settings.RETRIEVE_TOP_K)
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self._reranker.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievedChunk] = []
        for doc, score in scored[: settings.RERANK_TOP_N]:
            if score < settings.MIN_RERANK_SCORE:
                continue
            results.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=os.path.basename(doc.metadata.get("source", "unknown")),
                    page=doc.metadata.get("page"),
                    score=float(score),
                )
            )
        return results


# Singleton, so the API layer and the Agent layer share the same vector database connection
rag_engine = RAGEngine()
