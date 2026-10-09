import pytest

from docmind.ingest import load_file, split_documents
from docmind.vectorstore import DocumentStore


def test_load_and_split_tags_source_and_chunk_ids(sample_files):
    docs = load_file(sample_files["policy"])
    chunks = split_documents(docs, chunk_size=120, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "leave_policy.txt" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == [f"leave_policy.txt#{i}" for i in range(len(chunks))]
    assert all(len(c.page_content) <= 120 for c in chunks)


def test_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "virus.exe"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported"):
        load_file(bad)


def test_store_retrieves_relevant_chunk_and_persists(sample_files, embeddings, tmp_path):
    store = DocumentStore(embeddings, str(tmp_path / "idx"))
    assert store.is_empty
    store.add_documents(split_documents(load_file(sample_files["policy"]), 120, 10))
    store.add_documents(split_documents(load_file(sample_files["laptop"]), 120, 10))

    top = store.as_retriever(k=1).invoke("sick leave medical certificate")
    assert "Sick leave is 12 days" in top[0].page_content

    reloaded = DocumentStore(embeddings, str(tmp_path / "idx"))
    assert reloaded.sources() == ["laptop_policy.md", "leave_policy.txt"]
    chunk_ids = [d.metadata["chunk_id"] for d in reloaded.get_chunks("leave_policy.txt")]
    assert chunk_ids == sorted(chunk_ids, key=lambda c: int(c.split("#")[1]))
