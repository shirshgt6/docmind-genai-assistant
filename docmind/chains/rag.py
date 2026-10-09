"""Conversational RAG chain built with LCEL runnables.

    question + chat_history
      -> (branch) rewrite into standalone question if there is history
      -> retriever (MMR over FAISS)
      -> prompt with numbered, source-tagged context
      -> LLM -> StrOutputParser
    output: {"answer", "docs", "standalone_question", ...}
"""
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from docmind.prompts import CONTEXTUALIZE_PROMPT, RAG_PROMPT


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('chunk_id', d.metadata.get('source', '?'))}]\n{d.page_content}" for d in docs)


def build_rag_chain(llm: BaseChatModel, retriever: BaseRetriever) -> Runnable:
    rewrite_question = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    # Conditional chain: only spend an LLM call on rewriting when there is history.
    standalone_question = RunnableBranch(
        (lambda x: bool(x.get("chat_history")), rewrite_question),
        RunnableLambda(itemgetter("question")),
    )

    answer = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(chat_history=lambda x: x.get("chat_history", []))
        | RunnablePassthrough.assign(standalone_question=standalone_question)
        | RunnablePassthrough.assign(docs=itemgetter("standalone_question") | retriever)
        | RunnablePassthrough.assign(answer=answer)
    )


def sources_from_docs(docs: list[Document]) -> list[dict]:
    return [
        {
            "chunk_id": d.metadata.get("chunk_id"),
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:200],
        }
        for d in docs
    ]
