# app/config.py

from pydantic_settings import BaseSettings
from pathlib import Path

ENV_FILE_PATH = Path(__file__).parent.parent / ".env"

class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # LangSmith
    langchain_tracing_v2: str = "true"
    langchain_api_key: str = ""
    langchain_project: str = "multimodal-doc-intelligence"

    # Tavily
    tavily_api_key: str = ""

    # Storage
    chroma_path: str = "chroma_db"

    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 400

    model_config = {
        "env_file": str(ENV_FILE_PATH),
        "extra": "ignore",
        "case_sensitive": False,
    }

settings = Settings()