"""FAISS vector store with save/load to disk, and an MMR retriever on top."""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


class DocumentStore:
    def __init__(self, embeddings: Embeddings, persist_dir: str | None = None):
        self.embeddings = embeddings
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._store: FAISS | None = None
        self._load()

    def _load(self) -> None:
        if self.persist_dir and (self.persist_dir / "index.faiss").exists():
            # Only ever loads an index this app wrote itself (pickle-based docstore).
            self._store = FAISS.load_local(
                str(self.persist_dir), self.embeddings, allow_dangerous_deserialization=True
            )

    def save(self) -> None:
        if self._store is not None and self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self.persist_dir))

    @property
    def is_empty(self) -> bool:
        return self._store is None

    def add_documents(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0
        if self._store is None:
            self._store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self._store.add_documents(chunks)
        self.save()
        return len(chunks)

    def sources(self) -> list[str]:
        if self._store is None:
            return []
        return sorted({d.metadata.get("source", "unknown") for d in self._store.docstore._dict.values()})

    def get_chunks(self, source: str | None = None) -> list[Document]:
        """All chunks (optionally of one source) in their original order."""
        if self._store is None:
            return []
        docs = [d for d in self._store.docstore._dict.values() if source is None or d.metadata.get("source") == source]
        return sorted(docs, key=lambda d: (d.metadata.get("source", ""), _chunk_index(d)))

    def as_retriever(self, k: int = 4) -> VectorStoreRetriever:
        if self._store is None:
            raise ValueError("No documents ingested yet")
        # MMR = relevant AND diverse chunks, so the LLM doesn't get 4 copies of the same paragraph.
        return self._store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 5})


def _chunk_index(doc: Document) -> int:
    chunk_id = doc.metadata.get("chunk_id", "#0")
    return int(chunk_id.rsplit("#", 1)[-1])
