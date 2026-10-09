from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever

from docmind.chains.extract import DocumentInsights, build_extract_chain
from docmind.chains.rag import build_rag_chain, format_docs
from docmind.chains.router import build_router_chain
from docmind.chains.summarize import STUFF_LIMIT_CHARS, build_summarize_chain
from tests.conftest import fake_llm


class RecordingRetriever(BaseRetriever):
    queries: list[str] = []

    def _get_relevant_documents(self, query, *, run_manager=None):
        self.queries.append(query)
        return [Document(page_content="Sick leave is 12 days.", metadata={"chunk_id": "policy.txt#1"})]


def test_rag_without_history_skips_rewrite():
    retriever = RecordingRetriever(queries=[])
    chain = build_rag_chain(fake_llm("12 days [policy.txt#1]"), retriever)

    out = chain.invoke({"question": "How much sick leave?"})

    assert out["answer"] == "12 days [policy.txt#1]"
    assert out["standalone_question"] == "How much sick leave?"
    assert retriever.queries == ["How much sick leave?"]


def test_rag_with_history_rewrites_question_before_retrieval():
    retriever = RecordingRetriever(queries=[])
    llm = fake_llm("How many days of sick leave do employees get?", "12 days")
    chain = build_rag_chain(llm, retriever)

    out = chain.invoke(
        {
            "question": "and sick leave?",
            "chat_history": [HumanMessage("How much annual leave?"), AIMessage("24 days")],
        }
    )

    assert retriever.queries == ["How many days of sick leave do employees get?"]
    assert out["answer"] == "12 days"


def test_format_docs_tags_each_chunk():
    text = format_docs([Document(page_content="hello", metadata={"chunk_id": "a.txt#0"})])
    assert text == "[a.txt#0]\nhello"


def test_summarize_uses_stuff_for_short_text():
    chain = build_summarize_chain(fake_llm("short summary"))
    assert chain.invoke({"docs": [Document(page_content="tiny doc")], "style": "brief"}) == "short summary"


def test_summarize_uses_map_reduce_for_long_text():
    docs = [Document(page_content="x" * (STUFF_LIMIT_CHARS // 2 + 1)) for _ in range(3)]
    # 3 map calls + 1 reduce call; FakeListChatModel returns responses in order.
    chain = build_summarize_chain(fake_llm("part 1", "part 2", "part 3", "final summary"))
    assert chain.invoke({"docs": docs, "style": "brief"}) == "final summary"


def test_extract_falls_back_to_pydantic_parser_and_validates():
    json_reply = (
        '{"title": "Leave Policy", "document_type": "policy", "key_points": ["24 days annual leave"], '
        '"entities": ["HR"], "sentiment": "neutral"}'
    )
    insights = build_extract_chain(fake_llm(json_reply)).invoke({"text": "..."})
    assert isinstance(insights, DocumentInsights)
    assert insights.document_type == "policy"


def test_router_classifies_intent():
    assert build_router_chain(fake_llm('{"intent": "summarize"}')).invoke({"question": "tl;dr"}).intent == "summarize"
