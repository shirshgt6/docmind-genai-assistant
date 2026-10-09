"""Summarization with a conditional chain: 'stuff' for short text, map-reduce for long text."""
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda

from docmind.prompts import SUMMARY_MAP_PROMPT, SUMMARY_REDUCE_PROMPT, SUMMARY_STUFF_PROMPT

STUFF_LIMIT_CHARS = 6000


def build_summarize_chain(llm: BaseChatModel) -> Runnable:
    """Input: {"docs": list[Document], "style": str}. Output: summary string."""
    parser = StrOutputParser()
    stuff = (
        RunnableLambda(lambda x: {"text": _join(x["docs"]), "style": x["style"]})
        | SUMMARY_STUFF_PROMPT
        | llm
        | parser
    )

    map_one = SUMMARY_MAP_PROMPT | llm | parser
    map_reduce = (
        RunnableLambda(
            lambda x: {
                # .map() runs the per-chunk chain over every chunk in parallel (batch under the hood).
                "text": "\n\n".join(map_one.map().invoke([{"text": d.page_content} for d in x["docs"]])),
                "style": x["style"],
            }
        )
        | SUMMARY_REDUCE_PROMPT
        | llm
        | parser
    )

    return RunnableBranch(
        (lambda x: len(_join(x["docs"])) <= STUFF_LIMIT_CHARS, stuff),
        map_reduce,
    )


def _join(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)
