from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "rag-hybrid-eval"
    langsmith_api_key: str = ""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "nimbus_docs"

    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    dense_dim: int = 384

    hybrid_candidates: int = 20
    rerank_top_k: int = 5


settings = Settings()
