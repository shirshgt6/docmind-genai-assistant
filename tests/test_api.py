import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from docmind.api import app, get_service
from docmind.service import DocMindService
from tests.conftest import FakeToolChatModel, fake_llm


@pytest.fixture
def make_client(embeddings, settings):
    def _make(llm):
        svc = DocMindService(llm, embeddings, settings)
        app.dependency_overrides[get_service] = lambda: svc
        return TestClient(app), svc

    yield _make
    app.dependency_overrides.clear()


def _upload(client, path):
    with open(path, "rb") as f:
        return client.post("/documents/upload", files={"file": (path.name, f, "text/plain")})


def test_upload_then_ask_returns_answer_with_sources(make_client, sample_files):
    client, _ = make_client(fake_llm("Employees get 12 sick days [leave_policy.txt#2]."))

    res = _upload(client, sample_files["policy"])
    assert res.status_code == 201
    assert res.json()["sources"] == ["leave_policy.txt"]
    assert client.get("/documents").json() == {"sources": ["leave_policy.txt"]}

    res = client.post("/ask", json={"question": "How many sick leave days?"})
    body = res.json()
    assert res.status_code == 200
    assert "12 sick days" in body["answer"]
    assert body["sources"] and body["sources"][0]["source"] == "leave_policy.txt"


def test_follow_up_question_uses_session_memory(make_client, sample_files):
    llm = fake_llm("24 days.", "How much sick leave do employees get?", "12 days.")
    client, svc = make_client(llm)
    _upload(client, sample_files["policy"])

    client.post("/ask", json={"question": "How much annual leave?", "session_id": "s1"})
    body = client.post("/ask", json={"question": "and sick leave?", "session_id": "s1"}).json()

    assert body["standalone_question"] == "How much sick leave do employees get?"
    assert len(svc.memory.get("s1").messages) == 4  # 2 human + 2 ai

    assert client.delete("/sessions/s1").status_code == 204
    assert svc.memory.get("s1").messages == []


def test_ask_before_upload_is_404(make_client):
    client, _ = make_client(fake_llm("unused"))
    res = client.post("/ask", json={"question": "anything?"})
    assert res.status_code == 404
    assert "Upload a document first" in res.json()["detail"]


def test_upload_rejects_bad_type_and_path_tricks(make_client, settings, tmp_path):
    client, _ = make_client(fake_llm("unused"))
    res = client.post("/documents/upload", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert res.status_code == 400

    res = client.post("/documents/upload", files={"file": ("../../escape.txt", b"hello world", "text/plain")})
    assert res.status_code == 201
    assert (tmp_path / "uploads" / "escape.txt").exists()
    assert not (tmp_path.parent / "escape.txt").exists()


def test_extract_and_summarize_endpoints(make_client, sample_files):
    json_reply = (
        '{"title": "Leave Policy 2026", "document_type": "policy", "key_points": ["24 days annual leave"],'
        ' "entities": [], "sentiment": "neutral"}'
    )
    client, _ = make_client(fake_llm(json_reply, "- 24 days annual leave"))
    _upload(client, sample_files["policy"])

    res = client.post("/extract", json={"source": "leave_policy.txt"})
    assert res.status_code == 200 and res.json()["document_type"] == "policy"

    res = client.post("/summarize", json={"source": "leave_policy.txt"})
    assert res.json()["summary"] == "- 24 days annual leave"

    assert client.post("/summarize", json={"source": "missing.pdf"}).status_code == 404


def test_assistant_routes_by_intent(make_client, sample_files):
    client, _ = make_client(fake_llm('{"intent": "summarize"}', "the summary"))
    _upload(client, sample_files["policy"])

    body = client.post("/assistant", json={"question": "give me a tl;dr"}).json()
    assert body == {"intent": "summarize", "result": {"source": "all", "summary": "the summary"}}


def test_agent_endpoint_uses_document_search(make_client, sample_files):
    scripted = iter(
        [
            AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "laptop"}, "id": "t1"}]),
            AIMessage(content="The laptop is worth 85000 rupees."),
        ]
    )
    client, _ = make_client(FakeToolChatModel(messages=scripted))
    _upload(client, sample_files["laptop"])

    body = client.post("/agent", json={"question": "What is my laptop worth?"}).json()

    assert body["answer"] == "The laptop is worth 85000 rupees."
    assert body["steps"][0]["tool"] == "search_documents"
    assert "85000" in body["steps"][0]["output"]
