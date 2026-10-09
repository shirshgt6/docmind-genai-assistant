"""DocMindService: the application layer the API and UI talk to.

It owns the vector store and chat memory and wires the chains together.
Models are injected, so tests run fully offline with fake models.
"""
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from docmind.agent import build_agent
from docmind.chains.extract import DocumentInsights, build_extract_chain
from docmind.chains.rag import build_rag_chain, sources_from_docs
from docmind.chains.router import build_router_chain
from docmind.chains.summarize import build_summarize_chain
from docmind.config import Settings, get_settings
from docmind.ingest import load_file, load_url, split_documents
from docmind.memory import SessionMemory
from docmind.vectorstore import DocumentStore

EXTRACT_MAX_CHARS = 8000


class NoDocumentsError(Exception):
    pass


class DocMindService:
    def __init__(self, llm: BaseChatModel, embeddings: Embeddings, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.llm = llm
        self.store = DocumentStore(embeddings, self.settings.vectorstore_dir)
        self.memory = SessionMemory()
        self.summarize_chain = build_summarize_chain(llm)
        self.extract_chain = build_extract_chain(llm)
        self.router_chain = build_router_chain(llm)

    # ---------- ingestion ----------
    def ingest_file(self, path: str | Path) -> dict:
        return self._ingest(load_file(path))

    def ingest_url(self, url: str) -> dict:
        return self._ingest(load_url(url))

    def _ingest(self, docs) -> dict:
        chunks = split_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        added = self.store.add_documents(chunks)
        return {"sources": sorted({c.metadata["source"] for c in chunks}), "chunks_added": added}

    def list_sources(self) -> list[str]:
        return self.store.sources()

    # ---------- RAG Q&A with memory ----------
    def ask(self, question: str, session_id: str = "default") -> dict:
        chain = build_rag_chain(self.llm, self._retriever())
        history = self.memory.get(session_id)
        result = chain.invoke({"question": question, "chat_history": history.messages})
        history.add_messages([HumanMessage(question), AIMessage(result["answer"])])
        return {
            "answer": result["answer"],
            "standalone_question": result["standalone_question"],
            "sources": sources_from_docs(result["docs"]),
        }

    # ---------- summarization / extraction ----------
    def summarize(self, source: str | None = None, style: str = "concise bullet-point") -> dict:
        docs = self._chunks(source)
        return {"source": source or "all", "summary": self.summarize_chain.invoke({"docs": docs, "style": style})}

    def extract(self, source: str | None = None) -> DocumentInsights:
        text = "\n\n".join(d.page_content for d in self._chunks(source))[:EXTRACT_MAX_CHARS]
        return self.extract_chain.invoke({"text": text})

    # ---------- smart router ----------
    def assistant(self, question: str, session_id: str = "default", source: str | None = None) -> dict:
        intent = self.router_chain.invoke({"question": question}).intent
        if intent == "summarize":
            result = self.summarize(source)
        elif intent == "extract":
            result = self.extract(source).model_dump()
        else:
            result = self.ask(question, session_id)
        return {"intent": intent, "result": result}

    # ---------- agent ----------
    def agent(self, question: str, session_id: str = "default") -> dict:
        retriever = None if self.store.is_empty else self._retriever()
        history = self.memory.get(f"agent:{session_id}")
        result = build_agent(self.llm, retriever).invoke({"input": question, "chat_history": history.messages})
        history.add_messages([HumanMessage(question), AIMessage(result["output"])])
        steps = [
            {"tool": action.tool, "input": action.tool_input, "output": str(observation)[:500]}
            for action, observation in result.get("intermediate_steps", [])
        ]
        return {"answer": result["output"], "steps": steps}

    def clear_session(self, session_id: str) -> None:
        self.memory.clear(session_id)
        self.memory.clear(f"agent:{session_id}")

    # ---------- helpers ----------
    def _retriever(self):
        if self.store.is_empty:
            raise NoDocumentsError("Upload a document first")
        return self.store.as_retriever(k=self.settings.retriever_k)

    def _chunks(self, source: str | None):
        docs = self.store.get_chunks(source)
        if not docs:
            raise NoDocumentsError(f"No documents found{f' for source {source!r}' if source else ''}")
        return docs
