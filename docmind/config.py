"""Settings loaded from environment variables / .env (pydantic-settings)."""
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env into os.environ so provider SDKs (Groq/OpenAI) can find their API keys.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.2

    embeddings_provider: str = "huggingface"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    ollama_base_url: str = "http://localhost:11434"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retriever_k: int = 4
    vectorstore_dir: str = "./data/faiss_index"
    upload_dir: str = "./data/uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
