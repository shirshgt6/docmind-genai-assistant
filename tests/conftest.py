"""Offline fakes: tests never call a real LLM or download an embedding model."""
import hashlib
import math
import re
from collections.abc import Iterator

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from docmind.config import Settings


class BagOfWordsEmbeddings(Embeddings):
    """Tiny deterministic embedding: hashed word counts. Similar words -> similar vectors."""

    def __init__(self, size: int = 256):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeToolChatModel(BaseChatModel):
    """Returns scripted AIMessages (incl. tool calls) and accepts bind_tools, so agents can be tested offline."""

    messages: Iterator[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "fake-tool-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=next(self.messages))])

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def embeddings():
    return BagOfWordsEmbeddings()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        vectorstore_dir=str(tmp_path / "index"),
        upload_dir=str(tmp_path / "uploads"),
        chunk_size=200,
        chunk_overlap=20,
        retriever_k=2,
    )


@pytest.fixture
def sample_files(tmp_path):
    policy = tmp_path / "leave_policy.txt"
    policy.write_text(
        "Leave Policy 2026.\n\n"
        "Employees get 24 days of paid annual leave per year.\n\n"
        "Sick leave is 12 days per year and needs a medical certificate after 2 days.\n\n"
        "Unused annual leave up to 10 days can be carried forward to the next year.",
        encoding="utf-8",
    )
    laptop = tmp_path / "laptop_policy.md"
    laptop.write_text(
        "# Laptop Policy\n\nEvery engineer receives a laptop worth 85000 rupees, refreshed every 3 years.",
        encoding="utf-8",
    )
    return {"policy": policy, "laptop": laptop}


def fake_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))
