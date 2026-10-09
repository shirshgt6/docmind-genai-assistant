"""Document loaders + text splitter: raw files/URLs -> metadata-tagged chunks."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()  # one Document per page, metadata has "page"
    elif ext in {".txt", ".md"}:
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    for doc in docs:
        doc.metadata["source"] = path.name
    return docs


def load_url(url: str) -> list[Document]:
    docs = WebBaseLoader(url).load()
    for doc in docs:
        doc.metadata["source"] = url
    return docs


def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """Recursive splitting keeps paragraphs/sentences together where it can.

    Each chunk gets a stable chunk_id so answers can cite exactly where they came from.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}#{i}"
    return chunks
