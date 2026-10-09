"""Model factory: one place that knows how to build chat models and embeddings.

Everything else receives a model as an argument, so providers can be swapped
from .env and tests can inject fake models without touching the network.
"""
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from docmind.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    provider = s.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.llm_model, temperature=s.llm_temperature)
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=s.llm_model, temperature=s.llm_temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=s.llm_model, temperature=s.llm_temperature, base_url=s.ollama_base_url)
    raise ValueError(f"Unknown LLM_PROVIDER '{s.llm_provider}' (use openai | groq | ollama)")


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    provider = s.embeddings_provider.lower()

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=s.embeddings_model)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=s.embeddings_model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=s.embeddings_model, base_url=s.ollama_base_url)
    raise ValueError(
        f"Unknown EMBEDDINGS_PROVIDER '{s.embeddings_provider}' (use huggingface | openai | ollama)"
    )
